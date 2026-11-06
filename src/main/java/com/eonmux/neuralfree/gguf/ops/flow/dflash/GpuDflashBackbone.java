package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.head.GpuRowTopK;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.ops.attn.GpuCacheAttention;
import com.eonmux.neuralfree.gguf.ops.attn.GpuQuantizedKvRotation;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.RopeResolver;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The DFlash decoder stack of {@link DflashBackbone} on the target's GPU (llama.cpp runs the draft context on the
 * target's backend with {@code --spec-draft-backend-sampling}): the same graph (dflash.cpp:806-1106) enqueued on the
 * target's queue over device weights and the {@link GpuDflashKvCache}. Injection fuses the target features on the
 * device too ({@code rms(fc . features) * enc.output_norm}, dflash.cpp:856-858) before writing the K/V rows. A noise
 * block reads back only what its readout consumes: the greedy id of every row ({@link #forwardGreedy},
 * {@code LLM_Argmax}), the top candidates, softmax denominators and {@code t_embd} rows of every row
 * ({@link #forwardTopK}, {@code LLM_Row_TopK} — the p_min and LiLiCorr readouts finish on the host over them), the
 * DSpark chain of biased argmax ids ({@link #forwardDspark}, the Markov bias recomputed on the device per row), or
 * the DFlash2 selector's chosen ids ({@link #forwardSelector}: the block runs through the dynamic convolutions
 * ({@code LLM_DflashConv}) and the selector scores, picks and p_min stop happen on the device ({@code LLM_Dflash2*}),
 * so only the drafted ids — and the probabilities when p_min needs them — are read back). Supported drafts: every
 * DFlash, DFlash2 and DSpark backbone but the DSV4 one, over a cache of
 * any {@link KvDType} ({@link #refusal}): a quantized cache rotates Q, K and V in Hadamard blocks before its store
 * and the attention output back ({@link GpuQuantizedKvRotation}, llama.cpp {@code attn_rot}) and needs a head size
 * that is a multiple of 32, which the store and flash kernels address in 32-value blocks. The matrices are the
 * draft's device weights (its {@code output}, else the target's); the norm vectors, sinks and RoPE factors are
 * uploaded from the draft's host weights; token embeddings are gathered on the host. Not thread-safe.
 */
final class GpuDflashBackbone implements DflashStack, AutoCloseable {

    private static final class Layer {
        GpuWeight wq;
        GpuWeight wk;
        GpuWeight wv;
        GpuWeight wo;
        GpuWeight gate;
        GpuWeight up;
        GpuWeight down;
        GpuWeight attnNorm;
        GpuWeight qNorm;
        GpuWeight kNorm;
        GpuWeight postAttnNorm;
        GpuWeight postFfnNorm;
        GpuWeight ffnNorm;
        GpuWeight attnConvProj;
        GpuWeight ffnConvProj;
        GpuBuffer attnConvBase;
        GpuBuffer ffnConvBase;
        float outScale;
        boolean hasOutScale;
        RopeSpec rope;
        GpuBuffer ropeFactors;
        float[] sinks;
        GpuCacheAttention attention;
        /** Hadamard rotation of a quantized cache: the injected rows' K/V, then the block's Q/K/V and output. */
        GpuQuantizedKvRotation storeRotation;
        GpuQuantizedKvRotation blockRotation;
        int nHead;
        int nHeadKv;
        int nFf;
    }

    /** Per-layer vectors uploaded from the host weights. */
    private static final String[] LAYER_VECTORS = {
            TensorNames.ATTN_NORM, TensorNames.ATTN_Q_NORM, TensorNames.ATTN_K_NORM, TensorNames.ATTN_POST_NORM,
            TensorNames.FFN_POST_NORM, TensorNames.FFN_NORM, TensorNames.ATTN_SINKS};

    /** Position sections per token of the widest RoPE type ({@link RopeMath#positionsPerToken}). */
    private static final int MAX_POSITION_SECTIONS = 4;

    private final DflashSpec spec;
    private final GpuOps ops;
    private final ComputeQueue queue;
    private final ComputeBackend backend;
    private final GpuDflashKvCache cache;
    private final DflashMatrix tokEmbd;
    private final GpuWeight output;
    private final GpuWeight fc;
    private final GpuWeight encNorm;
    private final GpuWeight outputNorm;
    private final Layer[] layers;
    private final int nEmbd;
    private final int headDim;
    private final int maxRows;
    private final int blockRows;
    private final int reduceLocal;
    private final List<AutoCloseable> owned = new ArrayList<>();
    private final Map<String, GpuBuffer> factorBuffers = new HashMap<>();
    private final GpuBuffer feats;
    private final GpuBuffer x;
    private final GpuBuffer h;
    private final GpuBuffer q;
    private final GpuBuffer k;
    private final GpuBuffer v;
    private final GpuBuffer attn;
    private final GpuBuffer cur;
    private final GpuBuffer gate;
    private final GpuBuffer up;
    private final GpuBuffer tEmbd;
    private final GpuBuffer logits;
    private final GpuBuffer ids;
    private final GpuBuffer ropePos;
    private final DflashConv conv;
    private final GpuBuffer convDynamic;
    private final GpuBuffer convHidden;
    private final GpuDflashSelector selector;
    /** The DSpark Markov table rows (the host weights' {@code markov_w1}); null without a Markov head. */
    private final DflashMatrix markovW1;
    /** The device Markov bias matrix ({@code markov_w2}, F32); null without a Markov head. */
    private final GpuWeight markovW2;
    private final int markovRank;
    /** Host row of {@code markov_w1} the next bias reads. */
    private final float[] markovRowHost;
    private final GpuBuffer markovRow;
    private final GpuBuffer markovBias;
    /** Device row the DSpark argmax reads: the Markov-biased logits, or the copied raw row. */
    private final GpuBuffer rowBuf;
    private final GpuRowTopK topK;
    private final int topKK;
    private boolean closed;

    /**
     * Why the device stack cannot run a draft of {@code spec} over a {@code kvType} cache, or null when it can.
     * A quantized cache drafts on the device when its head size is a multiple of 32; another head size names it.
     */
    static String refusal(DflashSpec spec, KvDType kvType) {
        if (spec.dsv4()) {
            return "the DeepSeek-V4 DSpark backbone runs on the host";
        }
        if (kvType.quantized() && spec.headDim() % 32 != 0) {
            return "a " + kvType + " draft KV cache needs a head size that is a multiple of 32 for the device"
                    + " flash kernels, got " + spec.headDim() + " (draft the cache as F16 with -ctkd)";
        }
        if (kvType.perTokenHead()) {
            return "a " + kvType.name().toLowerCase(java.util.Locale.ROOT) + " draft KV cache runs on the host;"
                    + " the device draft kernels keep no per-(token, head) scales (draft the cache as F16 with -ctkd)";
        }
        return null;
    }

    /** Candidates per row of a device top-k readout of {@code spec}: the lattice width, else the sampler's 10. */
    static int topKWidth(DflashSpec spec) {
        int k = spec.lilicorr() ? spec.lilicorrTopK() : spec.dflash2() ? spec.selectorTopK()
                : DflashSpec.SAMPLER_TOP_K;
        return spec.dflash2() ? k : Math.min(k, GemmKernels.TopKPartialPlan.MAX_K);
    }

    /**
     * @param device   the draft's matrices on the target's device
     * @param host     the draft's host weights, for the norm vectors, sinks and RoPE factors
     * @param tokEmbd  the host token embeddings (the draft's, or the target's)
     * @param output   the device output head (the draft's, or the target's)
     * @param fcCols   feature width of the encoder ({@code nTargets * target n_embd})
     * @param topKK    candidates per row of the device top-k readout ({@link #readoutTopK}); 0 binds none, for a
     *                 draft whose device readouts are the greedy ids and the DSpark loop
     * @throws UnsupportedOperationException when a matrix is not resident or of a type the matmul kernels do not
     *                                       take, the draft is refused by {@link #refusal}, or a top-k readout
     *                                       is asked of a backend without the {@code LLM_Row_TopK} kernel
     */
    GpuDflashBackbone(DflashSpec spec, ModelHparams hp, GpuForwardContext ctx, GpuWeights device, CpuWeights host,
                      DflashMatrix tokEmbd, GpuWeight output, GpuDflashKvCache cache, int fcCols, int nCtx,
                      int topKK) {
        this.spec = Objects.requireNonNull(spec, "spec");
        this.ops = ctx.ops();
        this.queue = ctx.queue();
        this.backend = ctx.backend();
        this.cache = Objects.requireNonNull(cache, "cache");
        this.tokEmbd = Objects.requireNonNull(tokEmbd, "tokEmbd");
        this.output = matrix(output);
        this.nEmbd = spec.nEmbd();
        this.headDim = spec.headDim();
        this.maxRows = Math.max(DflashDrafter.INJECT_CHUNK, spec.blockSize());
        this.blockRows = spec.blockSize();
        this.reduceLocal = Math.min(ops.local(), LlmKernelConstants.LLM_DIMX);
        this.topKK = spec.dflash2() ? 0 : topKK;
        this.conv = spec.conv() ? new DflashConv(nEmbd, spec.convKernel(), spec.convGroup()) : null;
        try {
            this.convDynamic = conv == null ? null : floats((long) blockRows * conv.dynamicWidth());
            this.convHidden = conv == null ? null : floats((long) blockRows * nEmbd);
            this.selector = spec.dflash2() ? new GpuDflashSelector(spec, ops, backend, host) : null;
            if (selector != null) owned.add(selector);
            this.fc = matrix(require(device, TensorNames.weight(TensorNames.FC)));
            this.encNorm = vectorWeight(host, TensorNames.weight(TensorNames.ENC_OUTPUT_NORM), nEmbd);
            this.outputNorm = vectorWeight(host, TensorNames.weight(TensorNames.OUTPUT_NORM), nEmbd);
            if (spec.dspark()) {
                this.markovW1 = DflashMatrix.require(host, TensorNames.weight(TensorNames.DSPARK_MARKOV_W1))
                        .expect(spec.nVocab(), DflashMatrix.require(host,
                                TensorNames.weight(TensorNames.DSPARK_MARKOV_W2)).cols());
                this.markovRank = markovW1.cols();
                this.markovRowHost = new float[markovRank];
                this.markovRow = floats(markovRank);
                this.markovBias = floats((int) output.rows());
                this.rowBuf = floats((int) output.rows());
                this.markovW2 = f32Matrix(host, TensorNames.weight(TensorNames.DSPARK_MARKOV_W2),
                        spec.nVocabDraft(), markovRank);
            } else {
                this.markovW1 = null;
                this.markovW2 = null;
                this.markovRank = 0;
                this.markovRowHost = null;
                this.markovRow = null;
                this.markovBias = null;
                this.rowBuf = null;
            }
            RopeResolver ropes = new RopeResolver(hp, host.catalog(), nCtx);
            this.layers = new Layer[spec.nLayer()];
            int qWidth = 0;
            int kvWidth = 0;
            int nFf = 0;
            for (int il = 0; il < layers.length; il++) {
                layers[il] = loadLayer(il, device, host, ropes);
                qWidth = Math.max(qWidth, layers[il].nHead * headDim);
                kvWidth = Math.max(kvWidth, layers[il].nHeadKv * headDim);
                nFf = Math.max(nFf, layers[il].nFf);
            }
            this.feats = floats((long) maxRows * fcCols);
            this.x = floats((long) maxRows * nEmbd);
            this.h = floats((long) blockRows * nEmbd);
            this.q = floats((long) blockRows * qWidth);
            this.k = floats((long) maxRows * kvWidth);
            this.v = floats((long) maxRows * kvWidth);
            this.attn = floats((long) blockRows * qWidth);
            this.cur = floats((long) blockRows * nEmbd);
            this.gate = floats((long) blockRows * nFf);
            this.up = floats((long) blockRows * nFf);
            this.tEmbd = floats((long) blockRows * nEmbd);
            this.logits = floats((long) blockRows * this.output.rows());
            this.ids = floats(blockRows);
            this.ropePos = floats((long) maxRows * MAX_POSITION_SECTIONS);
            this.topK = this.topKK > 0 ? GpuRowTopK.bind(ops, backend, logits, blockRows, (int) output.rows(), this.topKK)
                    : null;
            if (this.topKK > 0 && topK == null) {
                throw new UnsupportedOperationException("dflash device drafter: " + GemmKernels.ROW_TOPK
                        + " is not available on " + backend.backendType() + ", so the block readout reads its rows"
                        + " back");
            }
            if (topK != null) {
                owned.add(topK);
            }
            for (int il = 0; il < layers.length; il++) {
                bindAttention(ctx, il);
            }
        } catch (RuntimeException e) {
            RuntimeException f = Closeables.closeAll(owned, null);
            if (f != null) {
                e.addSuppressed(f);
            }
            throw e;
        }
    }

    private Layer loadLayer(int il, GpuWeights device, CpuWeights host, RopeResolver ropes) {
        Layer l = new Layer();
        l.nHead = spec.nHead(il);
        l.nHeadKv = spec.nHeadKv(il);
        l.wq = layerMatrix(device, il, TensorNames.ATTN_Q, l.nHead * headDim, nEmbd);
        l.wk = layerMatrix(device, il, TensorNames.ATTN_K, l.nHeadKv * headDim, nEmbd);
        GpuWeight wv = device.layerOrNull(il, TensorNames.weight(TensorNames.ATTN_V));
        l.wv = wv == null ? null : expect(matrix(wv), l.nHeadKv * headDim, nEmbd);
        l.wo = layerMatrix(device, il, TensorNames.ATTN_OUT, nEmbd, l.nHead * headDim);
        l.gate = matrix(requireLayer(device, il, TensorNames.FFN_GATE));
        l.nFf = (int) l.gate.rows();
        l.up = layerMatrix(device, il, TensorNames.FFN_UP, l.nFf, nEmbd);
        l.down = layerMatrix(device, il, TensorNames.FFN_DOWN, nEmbd, l.nFf);
        l.attnNorm = layerVector(host, il, TensorNames.ATTN_NORM, nEmbd, true);
        l.qNorm = layerVector(host, il, TensorNames.ATTN_Q_NORM, headDim, true);
        l.kNorm = layerVector(host, il, TensorNames.ATTN_K_NORM, headDim, true);
        l.postAttnNorm = layerVector(host, il, TensorNames.ATTN_POST_NORM, nEmbd, false);
        l.postFfnNorm = layerVector(host, il, TensorNames.FFN_POST_NORM, nEmbd, false);
        l.ffnNorm = layerVector(host, il, TensorNames.FFN_NORM, nEmbd, true);
        if (conv != null) {
            l.attnConvProj = layerMatrix(device, il, TensorNames.DFLASH_ATTN_CONV_PROJ, conv.dynamicWidth(), nEmbd);
            l.ffnConvProj = layerMatrix(device, il, TensorNames.DFLASH_FFN_CONV_PROJ, conv.dynamicWidth(), nEmbd);
            l.attnConvBase = upload(DflashBackbone.vector(host.layerFloats(il, TensorNames.DFLASH_ATTN_CONV_BASE),
                    conv.baseLength(), TensorNames.DFLASH_ATTN_CONV_BASE));
            l.ffnConvBase = upload(DflashBackbone.vector(host.layerFloats(il, TensorNames.DFLASH_FFN_CONV_BASE),
                    conv.baseLength(), TensorNames.DFLASH_FFN_CONV_BASE));
        }
        float[] scale = host.layerFloatsOrNull(il, TensorNames.weight(TensorNames.LAYER_OUT_SCALE));
        if (scale != null) {
            l.outScale = DflashBackbone.vector(scale, 1, "blk." + il + "." + TensorNames.LAYER_OUT_SCALE)[0];
            l.hasOutScale = true;
        }
        float[] sinks = host.layerFloatsOrNull(il, TensorNames.weight(TensorNames.ATTN_SINKS));
        l.sinks = sinks == null ? null
                : DflashBackbone.vector(sinks, l.nHead, "blk." + il + "." + TensorNames.ATTN_SINKS);
        l.rope = ropes.forLayer(il);
        if (l.rope != null && !RopeMath.multiSection(l.rope.type())) {
            // ggml_rope_multi takes no frequency factors; ggml_rope_ext takes layers[0].rope_freqs (dflash.cpp:832-840)
            l.ropeFactors = ropeFactors(host, l.rope.freqFactors());
        }
        return l;
    }

    /** The device copy of the frequency factors {@code ref}, uploaded once and shared by the layers reading it. */
    private GpuBuffer ropeFactors(CpuWeights host, TensorRef ref) {
        if (ref == null) {
            return null;
        }
        GpuBuffer b = factorBuffers.get(ref.name());
        if (b == null) {
            float[] factors = host.floats(ref);
            if (factors == null) {
                return null;
            }
            b = upload(factors);
            factorBuffers.put(ref.name(), b);
        }
        return b;
    }

    /** The layer's flash attention over its cache rows: the block's causal flag, its window and sinks. */
    private void bindAttention(GpuForwardContext ctx, int il) {
        Layer l = layers[il];
        boolean windowed = cache.window(il).windowed();
        AttentionSpec a = AttentionSpec.builder()
                .nHead(l.nHead).nHeadKv(l.nHeadKv).headDimK(headDim).headDimV(headDim)
                .noRope(l.rope == null).nRot(l.rope == null ? 0 : Math.min(headDim, l.rope.nDims()))
                .causal(spec.causal())
                .swaLayer(windowed).swaType(windowed ? SwaType.STANDARD : SwaType.NONE)
                .nSwa(windowed ? spec.nSwa() : 0)
                .kqScale(spec.kqScale())
                .sinks(l.sinks != null)
                .kvOwnerLayer(il)
                .build();
        GpuBuffer sinkBuf = l.sinks == null ? null : upload(l.sinks);
        l.attention = GpuCacheAttention.bind(ctx, a, cache.layer(il), cache.kvPos(il), q, attn, sinkBuf, blockRows);
        owned.add(l.attention);
        RotationPlan rotation = cache.layer(il).rotation();
        int qWidth = l.nHead * headDim;
        int kvWidth = l.nHeadKv * headDim;
        l.storeRotation = GpuQuantizedKvRotation.forStore(ops, rotation, k, kvWidth, v, kvWidth, maxRows);
        if (l.storeRotation != null) {
            owned.add(l.storeRotation);
        }
        l.blockRotation = GpuQuantizedKvRotation.of(ops, rotation, q, qWidth, k, kvWidth, v, kvWidth, attn, qWidth,
                blockRows);
        if (l.blockRotation != null) {
            owned.add(l.blockRotation);
        }
    }

    // ------------------------------------------------------------------ injection

    /**
     * Writes the cache rows of {@code nRows} target feature rows ({@code [nRows][fcCols]}) at positions
     * {@code pos0 ..}: the device encoder {@code rms(fc . features) * enc.output_norm}, then the per-layer K/V of
     * {@link #inject}, in chunks of {@link DflashDrafter#INJECT_CHUNK} rows.
     *
     * @param fused receives the fused rows {@code [nRows][nEmbd]} when not null (LiLiCorr keeps them as anchors)
     */
    void injectFeatures(float[] features, int width, int nRows, int pos0, float[] fused) {
        ensureOpen();
        for (int off = 0; off < nRows; off += maxRows) {
            int n = Math.min(maxRows, nRows - off);
            write(feats, features, off * width, n * width);
            ops.matmul(fc, feats, 0, x, 0, n);
            ops.norm(NormKind.RMS, encNorm, null, spec.rmsEps(), x, 0, x, 0, n, nEmbd);
            if (fused != null) {
                read(x, fused, off * nEmbd, n * nEmbd);
            }
            injectRows(n, pos0 + off);
        }
    }

    /**
     * {@link #injectFeatures(float[], int, int, int, float[])} over feature rows already on the device, laid out at
     * the encoder's width: no upload, the encoder reads them where they are (the device hidden feed).
     *
     * @param fused receives the fused rows {@code [nRows][nEmbd]} when not null (LiLiCorr keeps them as anchors)
     */
    void injectFeatures(GpuBuffer features, int nRows, int pos0, float[] fused) {
        ensureOpen();
        int width = (int) fc.cols();
        for (int off = 0; off < nRows; off += maxRows) {
            int n = Math.min(maxRows, nRows - off);
            ops.matmul(fc, features, off * width, x, 0, n);
            ops.norm(NormKind.RMS, encNorm, null, spec.rmsEps(), x, 0, x, 0, n, nEmbd);
            if (fused != null) {
                read(x, fused, off * nEmbd, n * nEmbd);
            }
            injectRows(n, pos0 + off);
        }
    }

    @Override
    public void inject(float[] inpG, int nRows, int pos0) {
        ensureOpen();
        for (int off = 0; off < nRows; off += maxRows) {
            int n = Math.min(maxRows, nRows - off);
            write(x, inpG, off * nEmbd, n * nEmbd);
            injectRows(n, pos0 + off);
        }
    }

    /**
     * Per layer {@code K = wk . g} with per-head {@code rms * attn_k_norm} and RoPE, {@code V = wv . g} or the
     * weightless per-head RMS of the raw K, rotated in Hadamard blocks when the cache is quantized, stored at
     * {@code pos0 ..} (dflash.cpp:856-906).
     */
    private void injectRows(int n, int pos0) {
        cache.place(pos0, n);
        uploadRopePositions(pos0, n);
        for (int il = 0; il < layers.length; il++) {
            Layer l = layers[il];
            int kvWidth = l.nHeadKv * headDim;
            ops.matmul(l.wk, x, 0, k, 0, n);
            values(l, x, n);
            ops.norm(NormKind.RMS, l.kNorm, null, spec.rmsEps(), k, 0, k, 0, n * l.nHeadKv, headDim);
            rope(l, k, l.nHeadKv, n);
            if (l.storeRotation != null) {
                l.storeRotation.beforeStore(n);
            }
            cache.store(il, k, 0, v, 0);
        }
    }

    // ------------------------------------------------------------------ noise block

    /** The block's logits and {@code t_embd} rows, read back once the queue finished. */
    @Override
    public Block forward(int[] tokens, int pos0) {
        int n = run(tokens, pos0);
        int vocab = (int) output.rows();
        float[] l = new float[n * vocab];
        float[] t = new float[n * nEmbd];
        read(logits, l, 0, l.length);
        read(tEmbd, t, 0, t.length);
        return new Block(n, l, vocab, t, nEmbd);
    }

    /**
     * The greedy id of every block row in draft-head rows ({@code LLM_Argmax}: the first maximum, NaN never
     * selected), the only values read back.
     */
    int[] forwardGreedy(int[] tokens, int pos0) {
        int n = run(tokens, pos0);
        ops.enqueue(GpuLogitsHead.ARGMAX, new long[]{(long) n * reduceLocal}, new long[]{reduceLocal},
                (int) output.rows(), logits, ids, n);
        byte[] raw = new byte[n * Integer.BYTES];
        GpuEvent e = queue.enqueueReadBuffer(ids, 0, raw.length, raw);
        queue.finish();
        close(e);
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = bb.getInt(i * Integer.BYTES);
        }
        return out;
    }

    /** The readback of one device top-k block: the candidates, softmax denominators and {@code t_embd} rows. */
    static final class TopKRows {
        final int rows;
        final int boundK;
        final int k;
        final float[] vals;
        final int[] idx;
        final float[] sumExp;
        final float[] tEmbd;

        TopKRows(int rows, int boundK, int k, float[] vals, int[] idx, float[] sumExp, float[] tEmbd) {
            this.rows = rows;
            this.boundK = boundK;
            this.k = k;
            this.vals = vals;
            this.idx = idx;
            this.sumExp = sumExp;
            this.tEmbd = tEmbd;
        }

        /** Candidate {@code j} of row {@code row} in draft-head rows. */
        int id(int row, int j) {
            return idx[row * boundK + j];
        }

        /** Logit of candidate {@code j} of row {@code row}. */
        float val(int row, int j) {
            return vals[row * boundK + j];
        }
    }

    /** The DSpark readback: the biased argmax head row of every read row and the block's {@code t_embd} rows. */
    static final class DsparkRows {
        final int iBeg;
        final int[] ids;
        final float[] tEmbd;
        /** The raw logit of draft-head row 0, for the non-anchor slot's confidence. */
        final float row0Logit;

        DsparkRows(int iBeg, int[] ids, float[] tEmbd, float row0Logit) {
            this.iBeg = iBeg;
            this.ids = ids;
            this.tEmbd = tEmbd;
            this.row0Logit = row0Logit;
        }
    }

    /**
     * The top {@code k} candidates of every block row in draft-head rows, in rank order (larger logit first, the
     * lower draft row on ties), with each row's softmax denominator {@code sum exp(l - max)} and the block's
     * {@code t_embd} rows; the only values read back.
     *
     * @param k candidates per row, at most the bound {@code topKK}
     */
    TopKRows forwardTopK(int[] tokens, int pos0, int k) {
        if (topK == null) {
            throw new IllegalStateException("dflash device stack has no top-k readout bound");
        }
        if (k < 1 || k > topKK) {
            throw new IllegalArgumentException("device top-k of " + k + " candidates outside [1, " + topKK + "]");
        }
        int n = run(tokens, pos0);
        topK.run(n);
        float[] vals = new float[n * topKK];
        byte[] rawIdx = new byte[n * topKK * Integer.BYTES];
        float[] sumExp = new float[n];
        float[] t = new float[n * nEmbd];
        GpuEvent e1 = queue.enqueueReadBuffer(topK.topVal(), 0, vals.length * Float.BYTES, vals);
        GpuEvent e2 = queue.enqueueReadBuffer(topK.topIdx(), 0, rawIdx.length, rawIdx);
        GpuEvent e3 = queue.enqueueReadBuffer(topK.sumExp(), 0, sumExp.length * Float.BYTES, sumExp);
        GpuEvent e4 = queue.enqueueReadBuffer(tEmbd, 0, t.length * Float.BYTES, t);
        queue.finish();
        close(e1);
        close(e2);
        close(e3);
        close(e4);
        ByteBuffer bb = ByteBuffer.wrap(rawIdx).order(ByteOrder.LITTLE_ENDIAN);
        int[] idx = new int[n * topKK];
        for (int i = 0; i < n * topKK; i++) {
            idx[i] = bb.getInt(i * Integer.BYTES);
        }
        return new TopKRows(n, topKK, k, vals, idx, sumExp, t);
    }

    /**
     * The DSpark readout chain on the device: per read row the argmax of its Markov-biased logits (the bias
     * {@code markov_w2 . markov_w1[prev]} recomputed per row from the previous row's winner, exactly
     * {@code DflashMarkovHead.apply}), or of the raw row without a Markov block. Reads back one id per row, the
     * {@code t_embd} rows and, with {@code iBeg == 1}, draft-head row 0's raw logit.
     *
     * @param anchor      the token the first row's bias reads ({@code markov_w1} rows, the target vocabulary)
     * @param d2t         target id of each draft-head row, or null without a reduced vocabulary
     * @param iBeg        first read row ({@code 0} anchor-first, else 1)
     * @param markovBlock whether the rows are within {@code dflash.block_size}, so the Markov bias applies
     */
    DsparkRows forwardDspark(int[] tokens, int pos0, int anchor, int[] d2t, int iBeg, boolean markovBlock) {
        int vocab = (int) output.rows();
        int n = run(tokens, pos0);
        int[] headIds = new int[Math.max(0, n - iBeg)];
        byte[] one = new byte[Integer.BYTES];
        for (int i = iBeg, made = 0; i < n; i++, made++) {
            if (markovBlock) {
                markovW1.row(anchor, markovRowHost, 0);
                write(markovRow, markovRowHost, 0, markovRank);
                ops.matmul(markovW2, markovRow, 0, markovBias, 0, 1);
                ops.add(markovBias, 0, logits, i * vocab, rowBuf, 0, vocab);
            } else {
                ops.copy(logits, i * vocab, rowBuf, 0, vocab);
            }
            ops.enqueue(GpuLogitsHead.ARGMAX, new long[]{reduceLocal}, new long[]{reduceLocal}, vocab, rowBuf, ids,
                    1);
            GpuEvent e = queue.enqueueReadBuffer(ids, 0, Integer.BYTES, one);
            queue.finish();
            close(e);
            headIds[made] = ((one[0] & 0xff)) | ((one[1] & 0xff) << 8) | ((one[2] & 0xff) << 16)
                    | ((one[3] & 0xff) << 24);
            anchor = d2t == null ? headIds[made] : d2t[headIds[made]];
        }
        float row0 = 0.0f;
        if (iBeg == 1 && n > 0) {
            float[] first = new float[1];
            read(logits, first, 0, 1);
            row0 = first[0];
        }
        float[] t = new float[n * nEmbd];
        read(tEmbd, t, 0, t.length);
        return new DsparkRows(iBeg, headIds, t, row0);
    }

    /**
     * Enqueues the noise block {@code tokens} at {@code pos0 ..}: {@code x = embd(tokens) * embedding_scale}, the
     * layers, then {@code t_embd = rms(x) * output_norm} and the logits (dflash.cpp:947-1063).
     *
     * @return the block length
     */
    private int run(int[] tokens, int pos0) {
        ensureOpen();
        int n = tokens.length;
        if (n < 1 || n > blockRows) {
            throw new IllegalArgumentException("dflash noise block of " + n + " tokens outside [1, " + blockRows
                    + "]");
        }
        float[] rows = new float[n * nEmbd];
        for (int t = 0; t < n; t++) {
            tokEmbd.row(tokens[t], rows, t * nEmbd);
        }
        if (spec.embeddingScale() != 0.0f) {
            for (int i = 0; i < rows.length; i++) {
                rows[i] *= spec.embeddingScale();
            }
        }
        write(x, rows, 0, rows.length);
        cache.place(pos0, n);
        uploadRopePositions(pos0, n);
        for (int il = 0; il < layers.length; il++) {
            layer(il, n, pos0);
        }
        ops.norm(NormKind.RMS, outputNorm, null, spec.rmsEps(), x, 0, tEmbd, 0, n, nEmbd);
        ops.matmul(output, tEmbd, 0, logits, 0, n);
        return n;
    }

    /** One block layer over the residual {@code x}; the layer output replaces it ({@code DflashBackbone.layer}). */
    private void layer(int il, int n, int pos0) {
        Layer l = layers[il];
        float eps = spec.rmsEps();
        ops.norm(NormKind.RMS, l.attnNorm, null, eps, x, 0, h, 0, n, nEmbd);
        if (conv != null) {
            ops.matmul(l.attnConvProj, h, 0, convDynamic, 0, n);
            convolve(h, l.attnConvBase, 0, n);
        }
        ops.matmul(l.wq, h, 0, q, 0, n);
        ops.matmul(l.wk, h, 0, k, 0, n);
        values(l, h, n);
        ops.norm(NormKind.RMS, l.qNorm, null, eps, q, 0, q, 0, n * l.nHead, headDim);
        ops.norm(NormKind.RMS, l.kNorm, null, eps, k, 0, k, 0, n * l.nHeadKv, headDim);
        rope(l, q, l.nHead, n);
        rope(l, k, l.nHeadKv, n);
        if (l.blockRotation != null) {
            l.blockRotation.beforeStore(n);
        }
        cache.store(il, k, 0, v, 0);
        l.attention.attend(pos0, n);
        if (l.blockRotation != null) {
            l.blockRotation.afterAttention(n);
        }
        ops.matmul(l.wo, attn, 0, cur, 0, n);
        if (spec.attnValueScale() != 0.0f) {
            ops.scale(cur, 0, n * nEmbd, spec.attnValueScale());
        }
        if (conv != null) convolve(cur, l.attnConvBase, 1, n);
        if (l.postAttnNorm != null) {
            ops.norm(NormKind.RMS, l.postAttnNorm, null, eps, cur, 0, cur, 0, n, nEmbd);
        }
        ops.add(cur, 0, x, 0, cur, 0, n * nEmbd);
        ops.norm(NormKind.RMS, l.ffnNorm, null, eps, cur, 0, h, 0, n, nEmbd);
        if (conv != null) {
            ops.matmul(l.ffnConvProj, h, 0, convDynamic, 0, n);
            convolve(h, l.ffnConvBase, 0, n);
        }
        ops.matmul(l.gate, h, 0, gate, 0, n);
        ops.matmul(l.up, h, 0, up, 0, n);
        ops.glu(spec.gelu() ? GluOp.GEGLU_TANH : GluOp.SWIGLU, null, gate, 0, l.nFf, up, 0, l.nFf, gate, 0, l.nFf,
                n, l.nFf);
        ops.matmul(l.down, gate, 0, x, 0, n);
        if (conv != null) convolve(x, l.ffnConvBase, 1, n);
        if (l.postFfnNorm != null) {
            ops.norm(NormKind.RMS, l.postFfnNorm, null, eps, x, 0, x, 0, n, nEmbd);
        }
        ops.add(x, 0, cur, 0, x, 0, n * nEmbd);
        if (l.hasOutScale) {
            ops.scale(x, 0, n * nEmbd, l.outScale);
        }
    }

    /** Convolves through a disjoint scratch block; the dynamic projection stays intact for the output side. */
    private void convolve(GpuBuffer rows, GpuBuffer base, int side, int n) {
        ops.copy(rows, 0, convHidden, 0, n * nEmbd);
        int local = ops.local();
        ops.enqueue("LLM_DflashConv", new long[]{roundUp((long) n * nEmbd, local)}, new long[]{local},
                nEmbd, spec.convKernel(), spec.convGroup(), n, side, convHidden, convDynamic, base, rows);
    }

    /** The DFlash2 selector's selected target ids, with probabilities read only when p_min needs them. */
    int[] forwardSelector(int[] tokens, int pos0, int anchor, float pMin) {
        if (selector == null) throw new IllegalStateException("dflash device stack has no DFlash2 selector");
        int n = run(tokens, pos0);
        return selector.select(logits, tEmbd, n, anchor, pMin);
    }

    /** V rows: {@code wv . in}, or the weightless per-head RMS of the raw K projection when V is shared. */
    private void values(Layer l, GpuBuffer in, int n) {
        if (l.wv != null) {
            ops.matmul(l.wv, in, 0, v, 0, n);
        } else {
            ops.norm(NormKind.RMS, null, null, spec.rmsEps(), k, 0, v, 0, n * l.nHeadKv, headDim);
        }
    }

    private void rope(Layer l, GpuBuffer buf, int nHeads, int n) {
        if (l.rope == null) {
            return;
        }
        boolean hasFf = l.ropeFactors != null;
        GpuBuffer ff = hasFf ? l.ropeFactors : ops.placeholders().readOnlyFloats();
        Object[] args = RopeKernels.ropeExtArgs(l.rope, n, nHeads, headDim, nHeads * headDim, l.rope.nOffs(), n, 0,
                hasFf, ropePos, ff, buf);
        long items = RopeKernels.ropeExtWorkItems(l.rope.type(), n, nHeads, l.rope.nDims());
        int local = ops.local();
        ops.enqueue(RopeKernels.ROPE_EXT, new long[]{roundUp(items, local)}, new long[]{local}, args);
    }

    /** The RoPE positions of {@code n} text rows from {@code pos0} ({@link RopePositions#text}), shared by the layers. */
    private void uploadRopePositions(int pos0, int n) {
        RopeSpec rope = firstRope();
        if (rope == null) {
            return;
        }
        int[] pos = RopePositions.text(rope, pos0, n);
        ByteBuffer bb = ByteBuffer.allocate(pos.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int p : pos) {
            bb.putInt(p);
        }
        close(queue.enqueueWriteBuffer(ropePos, 0, bb.capacity(), bb.array()));
    }

    private RopeSpec firstRope() {
        for (Layer l : layers) {
            if (l.rope != null) {
                return l.rope;
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ device memory

    /**
     * Bytes of the buffers a stack of {@code spec} allocates beside its cache and its device matrices, for the context
     * fit: the feature, residual, projection, FFN, logits, id and RoPE position rows of one injection chunk or block,
     * the F32 copies of the host vectors ({@link #hostVectorFloats}), the DSpark Markov tables and rows, the DFlash2
     * convolutions' dynamic and hidden rows and per-layer bases, and — for a DFlash2 draft — its selector tables and
     * readout, else the top-k readout's candidate buffers of {@code topKK} candidates per row (0 for none).
     */
    static long workBytes(DflashSpec spec, TensorCatalog catalog, int fcCols, int qWidth, int kvWidth, int nFf,
                          long vocabRows, int topKK) {
        long maxRows = Math.max(DflashDrafter.INJECT_CHUNK, spec.blockSize());
        long block = spec.blockSize();
        long floats = maxRows * (fcCols + spec.nEmbd() + 2L * kvWidth + MAX_POSITION_SECTIONS)
                + block * (3L * spec.nEmbd() + 2L * qWidth + 2L * nFf + vocabRows + 1)
                + hostVectorFloats(spec, catalog);
        if (spec.dspark()) {
            long rank = catalog.require(TensorNames.weight(TensorNames.DSPARK_MARKOV_W2)).cols();
            floats += spec.nVocab() * rank + spec.nVocabDraft() * rank + rank + 2L * vocabRows;
        }
        if (spec.conv()) {
            long dynamicWidth = 2L * spec.convKernel() * (spec.nEmbd() / spec.convGroup());
            floats += block * (dynamicWidth + spec.nEmbd())
                    + 4L * spec.nLayer() * spec.convKernel() * spec.nEmbd();
        }
        if (topKK > 0 && !spec.dflash2()) {
            floats += (long) maxRows * (3L * topKK + 1);
        }
        return floats * Float.BYTES + (spec.dflash2() ? GpuDflashSelector.bytes(spec) : 0);
    }

    /**
     * Values of the host vectors the stack uploads as F32: the encoder and output norms, every layer's norms and
     * sinks, and each distinct RoPE frequency-factor tensor once.
     */
    static long hostVectorFloats(DflashSpec spec, TensorCatalog catalog) {
        long n = elements(catalog.get(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM)))
                + elements(catalog.get(TensorNames.weight(TensorNames.OUTPUT_NORM)))
                + elements(catalog.get(TensorNames.weight(TensorNames.ROPE_FREQS)));
        for (int il = 0; il < spec.nLayer(); il++) {
            for (String base : LAYER_VECTORS) {
                n += elements(catalog.layer(il, TensorNames.weight(base)));
            }
        }
        return n;
    }

    private static long elements(TensorRef ref) {
        return ref == null ? 0L : ref.numElements();
    }

    private GpuBuffer floats(long count) {
        GpuBuffer b = backend.allocateBuffer(Math.max(1L, count) * Float.BYTES);
        owned.add(b);
        return b;
    }

    private GpuBuffer upload(float[] values) {
        GpuBuffer b = floats(values.length);
        write(b, values, 0, values.length);
        return b;
    }

    private GpuWeight vectorWeight(CpuWeights host, String name, int len) {
        float[] values = DflashBackbone.vector(host.floats(name), len, name);
        return GpuWeight.of(name, GGMLType.F32, new long[]{len, 1}, upload(values));
    }

    /** The draft matrix {@code name} uploaded as F32 rows (the DSpark Markov bias, {rows} x {cols}). */
    private GpuWeight f32Matrix(CpuWeights host, String name, long rows, long cols) {
        float[] values = host.floats(name);
        if (values.length != rows * cols) {
            throw new IllegalArgumentException("dflash drafter: tensor '" + name + "' holds " + values.length
                    + " values, expected " + rows + " x " + cols);
        }
        return GpuWeight.of(name, GGMLType.F32, new long[]{rows, cols}, upload(values));
    }

    private GpuWeight layerVector(CpuWeights host, int il, String base, int len, boolean required) {
        float[] values = host.layerFloatsOrNull(il, TensorNames.weight(base));
        String name = "blk." + il + "." + TensorNames.weight(base);
        if (values == null) {
            if (required) {
                throw new IllegalArgumentException("dflash drafter: model has no tensor '" + name + "'");
            }
            return null;
        }
        DflashBackbone.vector(values, len, name);
        return GpuWeight.of(name, GGMLType.F32, new long[]{len, 1}, upload(values));
    }

    private void write(GpuBuffer dst, float[] src, int srcOff, int count) {
        float[] part = srcOff == 0 && count == src.length ? src : Arrays.copyOfRange(src, srcOff, srcOff + count);
        close(queue.enqueueWriteBuffer(dst, 0, (long) count * Float.BYTES, part));
    }

    /** Reads {@code count} floats of {@code src} into {@code dst} from {@code dstOff}, after the queue finished. */
    private void read(GpuBuffer src, float[] dst, int dstOff, int count) {
        float[] part = dstOff == 0 && count == dst.length ? dst : new float[count];
        GpuEvent e = queue.enqueueReadBuffer(src, 0, (long) count * Float.BYTES, part);
        queue.finish();
        close(e);
        if (part != dst) {
            System.arraycopy(part, 0, dst, dstOff, count);
        }
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    private static GpuWeight require(GpuWeights w, String name) {
        GpuWeight m = w.getOrNull(name);
        if (m == null) {
            throw new UnsupportedOperationException("dflash device drafter: '" + name + "' is not on the device");
        }
        return m;
    }

    private static GpuWeight requireLayer(GpuWeights w, int il, String base) {
        GpuWeight m = w.layerOrNull(il, TensorNames.weight(base));
        if (m == null) {
            throw new UnsupportedOperationException("dflash device drafter: 'blk." + il + "."
                    + TensorNames.weight(base) + "' is not on the device");
        }
        return m;
    }

    private static GpuWeight layerMatrix(GpuWeights w, int il, String base, long rows, long cols) {
        return expect(matrix(requireLayer(w, il, base)), rows, cols);
    }

    /** {@code w} when the matmul kernels take its device type. */
    private static GpuWeight matrix(GpuWeight w) {
        Objects.requireNonNull(w, "weight");
        if (!GpuOps.supportsWeightType(w.type())) {
            throw new UnsupportedOperationException("dflash device drafter: " + w.name() + " is " + w.type()
                    + ", which the GPU matmul kernels do not take");
        }
        return w;
    }

    private static GpuWeight expect(GpuWeight w, long rows, long cols) {
        if (w.rows() != rows || w.cols() != cols) {
            throw new IllegalArgumentException("dflash drafter: " + w.name() + " is " + w.rows() + "x" + w.cols()
                    + ", expected " + rows + "x" + cols);
        }
        return w;
    }

    private static long roundUp(long n, int local) {
        return (n + local - 1) / local * local;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("dflash device stack is closed");
        }
    }

    /** Waits for the queue, then releases the attention launches and every buffer, in reverse order. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = null;
        try {
            queue.finish();
        } catch (RuntimeException e) {
            first = e;
        }
        for (int i = owned.size() - 1; i >= 0; i--) {
            first = Closeables.close(owned.get(i), first);
        }
        owned.clear();
        if (first != null) {
            throw first;
        }
    }
}
