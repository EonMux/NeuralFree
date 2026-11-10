package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Fusion;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.cpu.KvInt8Rows;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.encdec.T5RelativeBias;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.DualChunkRope;
import com.eonmux.neuralfree.gguf.ops.rope.FopeWeights;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU twin of {@link GpuAttentionMixer}: the same order of operations (llama.cpp {@code build_qkv} +
 * {@code build_attn}, src/llama-graph.cpp:1807-1912, 2946-3100, 3115-3290) on {@link CpuScratch} arrays with the
 * kernel-mirroring CPU math ({@link NormMath}, {@link CpuRope}, {@link AttnExtMath}, {@link CpuHadamard},
 * {@link CpuAttentionCore}), the shared Q/K norm geometry ({@link QkNormGeometry}) and the MoVA value projection
 * ({@link CpuValueExperts}).
 *
 * <p>Positions come from the {@link CpuScratch#POS} int slot, which the embedding stage fills per step
 * ({@code pos[s * n + t]}, section 0 = the token positions). With a cache, K and V are stored at
 * {@link KvLayout#rowOf} of each position after the executor's {@link CpuMemory#beginStep}, and attention reads the
 * cache rows with their stored positions ({@link CpuMemory#positionOfRow}); a shared-KV layer reads its owner's
 * {@link KvAccessor}. Without a cache, attention reads the batch's K and V with {@code kv_pos = q_pos}.</p>
 *
 * <p>Every projection adds the {@link CpuProjectionDeltas} registered on it (LoRA) right after its base product and
 * before its bias, as {@link GpuAttentionMixer} does.</p>
 *
 * <p>Over paged memory a batched step may hold one segment per sequence (a cross-sequence ubatch): every segment
 * stores its rows through the step's cells and attends through its sequence's page-table slice only, the unified
 * multi-sequence ubatch of llama.cpp (src/llama-kv-cache.cpp:761).</p>
 *
 * <p>Working arrays are the {@code Q}, {@code K}, {@code V} and {@code ATTN_OUT} slots plus {@code attn.*}
 * temporaries reserved at bind for the ubatch; a step allocates no per-token arrays. Not thread-safe.</p>
 */
public final class CpuAttentionMixer implements CpuMixer {

    private static final String T_QKV = "attn.qkv";
    private static final String T_QLAT = "attn.qlat";
    private static final String T_QGATE = "attn.qgate";
    private static final String T_GATE = "attn.gate";
    private static final String T_SHIFT = "attn.shift";
    private static final String T_ADAPTER = "attn.adapter_rank";
    private static final String T_ADAPTER_OUT = "attn.adapter_out";
    /** Elements per parallel task of the element-wise loops. */
    private static final int ELEMENT_GRAIN = 1 << 15;

    private CpuForwardContext ctx;
    private CpuScratch scratch;
    private CpuMatVec matVec;
    private LayerSpec layer;
    private AttentionSpec spec;
    private QkvProjection proj;
    private int ubatch;
    private boolean bound;
    private boolean closed;

    private CpuTensorRef wq;
    private CpuTensorRef wqUp;
    private CpuTensorRef wk;
    private CpuTensorRef wv;
    private CpuTensorRef wqkv;
    private CpuTensorRef wgate;
    private CpuTensorRef wo;
    /** Per-use Q/K/V adapters of a shared attention block (Zamba2); null without them. */
    private CpuTensorRef wqAdapterA;
    private CpuTensorRef wqAdapterB;
    private CpuTensorRef wkAdapterA;
    private CpuTensorRef wkAdapterB;
    private CpuTensorRef wvAdapterA;
    private CpuTensorRef wvAdapterB;
    private float[] qkvBias;
    private float[] qBias;
    private float[] kBias;
    private float[] vBias;
    private float[] gateBias;
    private float[] outBias;
    private ActParams clamp;

    private CpuProjectionDeltas.Site qSite;
    private CpuProjectionDeltas.Site kSite;
    private CpuProjectionDeltas.Site vSite;
    private CpuProjectionDeltas.Site qkvSite;
    private CpuProjectionDeltas.Site gateSite;
    private CpuProjectionDeltas.Site outSite;
    private CpuValueExperts valueExperts;

    private NormStep qNorm;
    private NormStep qLatentNorm;
    private NormStep kNorm;
    private NormStep vNorm;
    private NormStep subNorm;
    private NormStep outHeadNorm;
    private NormStep outNormStep;
    private RopeFreqTable ropeTable;
    private RopeFreqTable shiftTable;
    private int gateWidth;
    private boolean singlePosition;

    private KvAccessor kv;
    private IntUnaryOperator rowPos;
    /** Raw int8 view of the layer's cache when the fused int8 attention serves it; null otherwise. */
    private KvInt8Rows int8Kv;
    private HostKvAttention hostKvAttention;
    /** The gated mixed attention of an IQuestLoopCoder layer ({@code spec.dual()}), or null. */
    private CpuLoopAttention loopAttention;
    /** The paged-step store and attention plumbing, over the step the memory holds. */
    private CpuPagedAttention pagedAttention;
    private CpuQuantizedKvRotation rotation;
    /** The K/V half of {@link #rotation}, for the cache-write-only step. */
    private CpuQuantizedKvRotation storeRotation;
    private int shiftRotK;
    private CpuAttentionCore.Params params;
    private CpuAttentionCore.Params nonCausalParams;

    /** Dual-chunk attention of the layer (Qwen2.5-1M), or null. */
    private DualChunkSpec dualChunk;
    /** Five query-rotation position arrays for one step, {@code [5][ubatch]}. */
    private int[] dcPositions;
    private float[] dcScratchOut;
    private float[] dcScratchLse;
    private float[] dcMergeLse;
    private PagedStep pagedStep;
    /** limite's value embeddings and exclusive self attention, or null without either. */
    private CpuValueEmbeddings valueEmbeddings;
    private int nEmbd;
    /** Per-KV-head FP8 cache scales of the layer's cache, null without an FP8 cache ({@link #bindKvScales}). */
    private float[] kScale;
    private float[] vScale;
    /** Normalized static sink keys and values of the openPangu sink attention; null without sink tokens. */
    private float[] sinkK;
    private float[] sinkV;
    /** Inkling's query-side relative projection and the short convolutions of the K/V streams, or null. */
    private CpuInklingStages inklingStages;

    /** An unbound mixer; {@link #bind} reads the layer's attention spec. */
    public CpuAttentionMixer() {
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException      when the layer has no attention spec or a tensor has the wrong shape
     * @throws IllegalStateException         when the layer uses a cache but the context has no memory
     * @throws UnsupportedOperationException for a latent cache or value experts on a fused projection
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (bound) {
            throw new IllegalStateException("attention mixer of layer " + this.layer.il() + " is already bound");
        }
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.scratch = ctx.scratch();
        this.matVec = ctx.matVec();
        this.ubatch = scratch.ubatch();
        this.nEmbd = ctx.spec().hparams().nEmbd();
        CpuWeights w = ctx.weights();
        this.proj = QkvProjection.of(layer, s -> w.layerOrNull(layer, s) != null, ctx.spec().arch());
        this.spec = layer.attention();
        if (proj.source() == QkvProjection.Source.NONE) {
            wo = w.layer(layer, proj.outWeight());
            outSite = serveDeltas(proj.outWeight(), wo);
        } else {
            bindAttention(w);
        }
        bound = true;
    }

    private void bindAttention(CpuWeights w) {
        requireSlot(CpuScratch.Q, (long) ubatch * spec.qRopeDim());
        requireSlot(CpuScratch.ATTN_OUT, (long) ubatch * spec.outDim());
        if (proj.projectsKv()) {
            requireSlot(CpuScratch.K, (long) ubatch * spec.kDim());
            requireSlot(CpuScratch.V, (long) ubatch * spec.vDim());
        }
        if ((long) scratch.ints(CpuScratch.POS).length < 4L * ubatch) {
            throw new IllegalStateException("scratch slot " + CpuScratch.POS + " holds fewer than 4 x " + ubatch
                    + " positions");
        }
        singlePosition = layer.rope() == null || layer.rope().positionsPerToken() == 1;
        bindProjections(w);
        bindNorms(w);
        bindRope(w);
        if (spec.valueEmbeddings() || spec.xsaEps() > 0f) {
            valueEmbeddings = new CpuValueEmbeddings(this, w);
        }
        if (spec.usesMemory()) {
            if (!ctx.hasMemory()) {
                throw new IllegalStateException("layer " + layer.il() + " attends through a KV cache but the model "
                        + ctx.spec().arch().ggufName() + " has no memory");
            }
            CpuMemory memory = ctx.memory();
            kv = memory.kv(layer.il());
            if (kv == null) {
                throw new IllegalStateException("layer " + layer.il() + " has no KV cache in the memory plan");
            }
            checkCache(kv.layout());
            final int group = kv.group();
            rowPos = r -> memory.positionOfRow(group, r);
            RotationPlan rot = memory.plan().layer(kv.layer()).rotation();
            rotation = CpuQuantizedKvRotation.of(rot, spec.qDim(), proj.projectsKv() ? spec.kDim() : 0,
                    proj.projectsKv() ? spec.vDim() : 0, spec.outDim());
            storeRotation = proj.projectsKv()
                    ? CpuQuantizedKvRotation.forStore(rot, spec.kDim(), spec.vDim()) : null;
             shiftRotK = kv.layout().dtypeK().quantized() && rot.rotateK() ? rot.nRotK() : 0;
        }
        if (kv != null) {
            bindKvScales(w, kv.layout());
            int8Kv = CpuInt8Fusion.enabled() ? kv.int8Rows() : null;
            hostKvAttention = new HostKvAttention(kv, rowPos, spec.qDim(), spec.kDim(), spec.vDim(), spec.outDim(),
                    spec.headDimK(), spec.headDimV(), kScale, vScale, int8Kv);
        }
        bindStaticSinks(w);
        if (layer.mixerKind() == MixerKind.INKLING) {
            inklingStages = new CpuInklingStages(this, w);
        }
        params = CpuAttentionCore.Params.of(spec, sinks(w), relBias(w), kScale, vScale, sinkK, sinkV);
        nonCausalParams = params.nonCausal();
        bindDualChunk();
        if (spec.dual()) {
            loopAttention = new CpuLoopAttention(this);
            loopAttention.bind(ctx, layer, w);
        }
        pagedAttention = new CpuPagedAttention(this);
        bindOutput(w);
    }

    /**
     * Allocates the dual-chunk workforce of the layer: the five query planes live in the widened {@code Q} slot, and
     * a chunk never holds more than {@code min(ubatch, chunkLength)} rows, which bounds the per-stage scratch.
     */
    private void bindDualChunk() {
        dualChunk = spec.dualChunk();
        if (dualChunk == null) {
            return;
        }
        if (kv == null) {
            throw new IllegalStateException("layer " + layer.il() + " runs dual-chunk attention without a KV cache");
        }
        int rows = Math.min(ubatch, dualChunk.chunkLength());
        dcPositions = new int[DualChunkRope.VARIANTS * ubatch];
        dcScratchOut = new float[rows * spec.outDim()];
        dcScratchLse = new float[rows * spec.nHead()];
        dcMergeLse = new float[rows * spec.nHead()];
    }

    /**
     * Loads the per-KV-head FP8 cache scales of the layer's cache from {@code blk.N.attn_k.kv_scale} /
     * {@code blk.N.attn_v.kv_scale} (a {@code [1]} scale repeated), or null when the cache dtype is not
     * checkpoint-scaled (an FP8 type or NVFP4): vLLM consumes the checkpoint scales of an FP8 cache only and
     * enforces 1.0 for every other (vllm/model_executor/layers/quantization/kv_cache.py
     * process_weights_after_loading). No tensor with such a cache scales by 1.0, as vLLM does; NVFP4 follows the
     * same store/attention rule (vllm csrc/nvfp4_kv_cache_kernels.cu takes the checkpoint scale as its global
     * scale).
     *
     * @throws IllegalArgumentException when a scale tensor holds other than 1 or {@code n_head_kv} values or a
     *                                 non-finite or non-positive value
     */
    private void bindKvScales(CpuWeights w, KvLayout layout) {
        this.kScale = kvScaleValues(w, TensorNames.ATTN_K + "." + TensorClaims.KV_SCALE, layout.dtypeK());
        this.vScale = kvScaleValues(w, TensorNames.ATTN_V + "." + TensorClaims.KV_SCALE, layout.dtypeV());
    }

    /** The scale values of {@code suffix}, or null without a checkpoint-scaled cache or without the tensor. */
    private float[] kvScaleValues(CpuWeights w, String suffix, KvDType type) {
        if (!type.checkpointScaled()) {
            return null;
        }
        float[] v = w.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            float[] ones = new float[spec.nHeadKv()];
            Arrays.fill(ones, 1.0f);
            return ones;
        }
        if (v.length != 1 && v.length != spec.nHeadKv()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": " + suffix + " holds " + v.length
                    + " values; a KV-cache scale holds 1 or " + spec.nHeadKv() + " (one per KV head)");
        }
        for (float value : v) {
            if (!Float.isFinite(value) || value <= 0f) {
                throw new IllegalArgumentException("layer " + layer.il() + ": " + suffix + " holds " + value
                        + "; KV-cache scales must be finite and positive");
            }
        }
        if (v.length == spec.nHeadKv()) {
            return v;
        }
        float[] out = new float[spec.nHeadKv()];
        Arrays.fill(out, v[0]);
        return out;
    }

    void checkCache(KvLayout layout) {
        if (layout.vAliasesK()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " has a latent (MLA) cache; the softmax"
                    + " attention mixer reads separate K and V rows");
        }
        if (layout.widthK() != spec.kDim() || layout.widthV() != spec.vDim()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": cache rows of K " + layout.widthK() + " / V "
                    + layout.widthV() + " do not hold " + spec.nHeadKv() + " heads of (" + spec.headDimK() + ", "
                    + spec.headDimV() + ")");
        }
    }

    private void bindProjections(CpuWeights w) {
        int qw = proj.qProjWidth();
        if (proj.source() == QkvProjection.Source.FUSED) {
            wqkv = w.layer(layer, proj.qkvWeight());
            QkvProjection.checkShape(wqkv.toString(), wqkv.cols(), wqkv.rows(), wqkv.cols(), proj.fusedWidth());
            qkvSite = serveDeltas(proj.qkvWeight(), wqkv);
            qSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_Q), wqkv);
            kSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_K), wqkv);
            vSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_V), wqkv);
            scratch.temp(T_QKV, ubatch * proj.fusedWidth());
            qkvBias = vector(w, proj.qkvBias(), proj.fusedWidth());
        } else {
            wq = w.layer(layer, proj.qWeight());
            QkvProjection.checkShape(proj.qWeight(), wq.cols(), wq.rows(), wq.cols(), qw);
            qSite = serveDeltas(proj.qWeight(), wq);
            if (proj.qLatentDim() > 0) {
                wqUp = w.layer(layer, proj.qUpWeight());
                QkvProjection.checkShape(proj.qUpWeight(), wqUp.cols(), wqUp.rows(), proj.qLatentDim(), spec.qDim());
                float[] qn = w.layerFloatsOrNull(layer, proj.qNormWeight());
                if (qn == null || qn.length != proj.qLatentDim()) {
                    throw new IllegalArgumentException("layer " + layer.il() + ": " + proj.qNormWeight()
                            + " must hold " + proj.qLatentDim() + " values, got " + (qn == null ? 0 : qn.length));
                }
                qLatentNorm = new NormStep(NormKind.RMS, rmsEps(), 1, proj.qLatentDim(), qn, null, 1);
                scratch.temp(T_QLAT, ubatch * proj.qLatentDim());
            }
            if (proj.projectsKv()) {
                wk = w.layer(layer, proj.kWeight());
                QkvProjection.checkShape(proj.kWeight(), wk.cols(), wk.rows(), wq.cols(), spec.kDim());
                kSite = serveDeltas(proj.kWeight(), wk);
                if (proj.valueExperts()) {
                    float[] bias = vector(w, proj.vGateBias(), spec.valueExperts().nExpert());
                    valueExperts = new CpuValueExperts(ctx, layer, proj, spec.valueExperts(), bias);
                    if (valueExperts.inputWidth() != wq.cols()) {
                        throw new IllegalArgumentException("layer " + layer.il() + ": value router "
                                + proj.vGateWeight() + " reads " + valueExperts.inputWidth()
                                + " values, the Q projection " + wq.cols());
                    }
                } else if (!proj.vFromK()) {
                    wv = w.layer(layer, proj.vWeight());
                    QkvProjection.checkShape(proj.vWeight(), wv.cols(), wv.rows(), wq.cols(), spec.vDim());
                    vSite = serveDeltas(proj.vWeight(), wv);
                }
            }
        }
        qBias = vector(w, proj.qBias(), qw);
        if (proj.projectsKv()) {
            kBias = vector(w, proj.kBias(), spec.kDim());
            vBias = vector(w, proj.vBias(), spec.vDim());
        }
        if (spec.clampKqv() > 0f) {
            clamp = ActParams.builder().clamp(-spec.clampKqv(), spec.clampKqv()).build();
        }
        if (proj.interleavedGate()) {
            scratch.temp(T_QGATE, ubatch * qw);
            gateWidth = spec.qDim();
            scratch.temp(T_GATE, ubatch * gateWidth);
        } else if (proj.gateWeight() != null) {
            wgate = w.layer(layer, proj.gateWeight());
            gateWidth = AttnExtMath.gateWidth(spec.gate(), spec.nHead(), spec.headDimV());
            QkvProjection.checkShape(proj.gateWeight(), wgate.cols(), wgate.rows(), wgate.cols(), gateWidth);
            gateSite = serveDeltas(proj.gateWeight(), wgate);
            scratch.temp(T_GATE, ubatch * gateWidth);
        }
        bindAdapters(w);
    }

    /**
     * The per-use Q/K/V adapters of a shared attention block: {@code blk.<qkvAdapterLayer>.attn_{q,k,v}_adapter_a}
     * maps the block input to the adapter rank and {@code _b} back to the projection width; their product is added
     * to the base projections before RoPE (vLLM {@code Zamba2LoRA}).
     *
     * @throws IllegalArgumentException when an adapter has the wrong shape
     */
    private void bindAdapters(CpuWeights w) {
        int rank = spec.qkvAdapterRank();
        if (rank <= 0) {
            return;
        }
        int il = spec.qkvAdapterLayer();
        int in = (int) wq.cols();
        wqAdapterA = adapter(w, il, "attn_q_adapter_a", in, rank);
        wqAdapterB = adapter(w, il, "attn_q_adapter_b", rank, spec.qDim());
        wkAdapterA = adapter(w, il, "attn_k_adapter_a", in, rank);
        wkAdapterB = adapter(w, il, "attn_k_adapter_b", rank, spec.kDim());
        wvAdapterA = adapter(w, il, "attn_v_adapter_a", in, rank);
        wvAdapterB = adapter(w, il, "attn_v_adapter_b", rank, spec.vDim());
        scratch.temp(T_ADAPTER, ubatch * rank);
    }

    private CpuTensorRef adapter(CpuWeights w, int il, String base, long cols, long rows) {
        CpuTensorRef a = w.layer(il, TensorNames.weight(base));
        QkvProjection.checkShape(a.toString(), a.cols(), a.rows(), cols, rows);
        return a;
    }

    /** Adds the per-use adapter products to the projected Q, K and V rows of the step. */
    private void applyAdapters(int n, float[] in, float[] q, float[] k, float[] v) {
        if (wqAdapterA == null) {
            return;
        }
        if (q != null) {
            applyAdapter(wqAdapterA, wqAdapterB, n, in, q, spec.qDim());
        }
        if (k != null) {
            applyAdapter(wkAdapterA, wkAdapterB, n, in, k, spec.kDim());
            applyAdapter(wvAdapterA, wvAdapterB, n, in, v, spec.vDim());
        }
    }

    private void applyAdapter(CpuTensorRef a, CpuTensorRef b, int n, float[] in, float[] target, int width) {
        int rank = spec.qkvAdapterRank();
        float[] rankRows = scratch.temp(T_ADAPTER, ubatch * rank);
        matVec.matmul(a, in, 0, rankRows, 0, n);
        float[] delta = scratch.temp(T_ADAPTER_OUT, ubatch * width);
        matVec.matmul(b, rankRows, 0, delta, 0, n);
        CpuOps.add(target, 0, delta, 0, target, 0, n * width);
    }

    private void bindNorms(CpuWeights w) {
        QkNormKind kind = spec.qkNorm();
        if (kind != QkNormKind.NONE) {
            boolean hasQ = w.layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
            boolean hasK = w.layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
            if (QkNormGeometry.applies(kind, ctx.spec().arch(), false, hasQ, hasK)) {
                qNorm = qkNorm(w, geometry(kind, spec.nHead()), TensorNames.ATTN_Q_NORM);
            }
            if (proj.projectsKv() && QkNormGeometry.applies(kind, ctx.spec().arch(), true, hasQ, hasK)) {
                kNorm = qkNorm(w, geometry(kind, spec.nHeadKv()), TensorNames.ATTN_K_NORM);
            }
        }
        if (spec.vNorm() && proj.projectsKv()) {
            vNorm = new NormStep(NormKind.RMS, rmsEps(), spec.nHeadKv(), spec.headDimV(), null, null, 1);
        }
    }

    private QkNormGeometry geometry(QkNormKind kind, int nHeads) {
        return QkNormGeometry.of(kind, spec.headDimK(), nHeads, hparams().normEps(), rmsEps(), layer.il());
    }

    /**
     * Q or K norm in geometry {@code g}, loading the tensors as {@code GpuAttentionMixer.addQkNorm} does: a weight of
     * one scalar per head expands to {@code n_heads} rows, the bias is loaded only when the graph applies it.
     */
    private NormStep qkNorm(CpuWeights w, QkNormGeometry g, String base) {
        int rowLen = g.rowLen();
        CpuTensorRef weightRef = w.layerOrNull(layer, TensorNames.weight(base));
        CpuTensorRef biasRef = spec.qkNormBias() ? w.layerOrNull(layer, TensorNames.bias(base)) : null;
        float[] weight = weightRef != null ? w.layerFloatsOrNull(layer, TensorNames.weight(base)) : null;
        float[] bias = biasRef != null ? w.layerFloatsOrNull(layer, TensorNames.bias(base)) : null;
        int wRows = 1;
        boolean perHeadScalar = weightRef != null && g.perHeadScalar(weightRef.cols(), weightRef.rows());
        if (perHeadScalar) {
            wRows = g.nHeads();
            weight = broadcastPerHead(weight, rowLen);
            if (bias != null && g.perHeadScalar(biasRef.cols(), biasRef.rows())) {
                bias = broadcastPerHead(bias, rowLen);
            }
        } else if (weight != null) {
            wRows = g.weightRows(TensorNames.weight(base), weight.length);
        }
        if (bias != null) {
            g.checkBias(TensorNames.bias(base), bias.length, wRows);
        }
        return new NormStep(g.normKind(), g.eps(), g.rowsPerToken(), rowLen, weight, bias, wRows);
    }

    /** {@code [scalars.length][rowLen]} with row {@code h} filled with {@code scalars[h]} (ggml broadcast multiply). */
    private static float[] broadcastPerHead(float[] scalars, int rowLen) {
        float[] rows = new float[scalars.length * rowLen];
        for (int h = 0; h < scalars.length; h++) {
            Arrays.fill(rows, h * rowLen, (h + 1) * rowLen, scalars[h]);
        }
        return rows;
    }

    private void bindRope(CpuWeights w) {
        RopeSpec rope = layer.rope();
        if (rope == null || spec.noRope() || rope.type() == RopeType.NONE) {
            return;
        }
        if (spec.hasDualChunk()) {
            ropeTable = RopeFreqTable.forDualChunk(rope);
            return;
        }
        float[] ff = rope.freqFactors() != null ? w.floats(rope.freqFactors()) : null;
        FopeWeights fope = null;
        if (rope.fope() != null) {
            fope = FopeWeights.of(rope.fope(), w.floats(rope.fope().cosCoef()), w.floats(rope.fope().sinCoef()));
        }
        ropeTable = RopeFreqTable.of(rope, ff, fope);
        if (spec.usesMemory() && !spec.sharesKv(layer.il()) && GpuContextShift.refusal(ctx.spec(), layer) == null) {
            shiftTable = RopeFreqTable.forShift(rope, ff);
        }
    }

    private void bindOutput(CpuWeights w) {
        if (proj.outHeadNorm()) {
            outHeadNorm = new NormStep(NormKind.RMS, rmsEps(), spec.nHead(), spec.headDimV(), null, null, 1);
        }
        if (spec.gate() != AttnGateKind.NONE) {
            gateBias = vector(w, proj.gateBias(), spec.nHead());
        }
        if (spec.outNormKind() != null) {
            outNormStep = new NormStep(spec.outNormKind(), spec.outNormEps(), 1, spec.outDim(), null, null, 1);
        }
        if (proj.subNormWeight() != null) {
            NormKind kind = layer.norm().kind();
            float eps = kind == NormKind.LAYER ? hparams().normEps() : rmsEps();
            float[] sw = vector(w, proj.subNormWeight(), spec.outDim());
            float[] sb = vector(w, proj.subNormBias(), spec.outDim());
            subNorm = new NormStep(kind, eps, 1, spec.outDim(), sw, sb, 1);
        }
        wo = w.layer(layer, proj.outWeight());
        QkvProjection.checkShape(proj.outWeight(), wo.cols(), wo.rows(), spec.outDim(), wo.rows());
        outSite = serveDeltas(proj.outWeight(), wo);
        outBias = vector(w, proj.outBias(), wo.rows());
    }

    private float[] sinks(CpuWeights w) {
        if (!spec.sinks()) {
            return null;
        }
        return vector(w, TensorNames.weight(TensorNames.ATTN_SINKS), spec.nHead());
    }

    /**
     * Loads the learned static sink tokens of the openPangu sink attention: the {@code [n_sink * n_head_kv,
     * head_dim]} keys, normalized per KV head by the layer's K norm ({@code k_layernorm}) as vLLM's
     * {@code OpenPanguSinkAttention.post_weight_load} does and left un-rotated, and the matching values (zeros when
     * the checkpoint omits {@code attn_sink_v}, vLLM's non-persistent zero buffer).
     *
     * @throws MissingTensorException   when the sink keys are absent
     * @throws IllegalArgumentException when a sink tensor has the wrong size or the K norm weight is not a head row
     */
    private void bindStaticSinks(CpuWeights w) {
        int n = spec.sinkTokens();
        if (n == 0) {
            return;
        }
        int heads = spec.nHeadKv();
        int dK = spec.headDimK();
        int dV = spec.headDimV();
        float[] rawKeys = w.layerFloatsOrNull(layer, TensorNames.weight(TensorNames.ATTN_SINK_K));
        if (rawKeys == null) {
            throw new MissingTensorException(ctx.spec().arch().ggufName(), layer.il(),
                    TensorNames.weight(TensorNames.ATTN_SINK_K));
        }
        QkvProjection.checkLength(TensorNames.weight(TensorNames.ATTN_SINK_K), rawKeys.length,
                (long) n * heads * dK);
        float[] weight = w.layerFloatsOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM));
        if (weight != null) {
            QkvProjection.checkLength(TensorNames.weight(TensorNames.ATTN_K_NORM), weight.length, dK);
        }
        sinkK = new float[n * heads * dK];
        NormMath.rowNorm(NormKind.RMS, rmsEps(), n * heads, dK, dK, 1, rawKeys, 0, weight, null, sinkK, 0);
        float[] rawValues = w.layerFloatsOrNull(layer, TensorNames.weight(TensorNames.ATTN_SINK_V));
        if (rawValues == null) {
            sinkV = new float[n * heads * dV];
        } else {
            QkvProjection.checkLength(TensorNames.weight(TensorNames.ATTN_SINK_V), rawValues.length,
                    (long) n * heads * dV);
            sinkV = rawValues;
        }
    }

    /**
     * The T5 relative-position bias table of the stack's bias layer ({@link T5RelativeBias#host}: F32
     * {@code [n_buckets][n_head]}), or null without a T5 bias; the layer's relative-logit table for the Inkling bias
     * mode, which the Inkling stages fill every step.
     */
    private float[] relBias(CpuWeights w) {
        if (spec.biasMode() == AttentionSpec.BiasMode.INKLING) {
            return inklingStages.relTable();
        }
        return T5RelativeBias.uses(spec) ? T5RelativeBias.host(w, layer, spec) : null;
    }

    /** A layer vector ({@code suffix}, null for none) decoded to F32 and checked to hold {@code n} values. */
    float[] vector(CpuWeights w, String suffix, long n) {
        if (suffix == null) {
            return null;
        }
        float[] v = w.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            throw new IllegalStateException("layer " + layer.il() + ": tensor " + suffix + " is missing");
        }
        QkvProjection.checkLength(suffix, v.length, n);
        return v;
    }

    private void requireSlot(String name, long n) {
        if (scratch.slot(name).length < n) {
            throw new IllegalStateException("scratch slot " + name + " holds " + scratch.slot(name).length
                    + " floats, " + n + " are needed");
        }
    }

    // ------------------------------------------------------------------ stage accessors

    CpuForwardContext ctx() {
        return ctx;
    }

    AttentionSpec spec() {
        return spec;
    }

    CpuScratch scratch() {
        return scratch;
    }

    CpuMatVec matVec() {
        return matVec;
    }

    int ubatch() {
        return ubatch;
    }

    int nEmbd() {
        return nEmbd;
    }

    KvAccessor kv() {
        return kv;
    }

    IntUnaryOperator rowPos() {
        return rowPos;
    }

    KvInt8Rows int8Kv() {
        return int8Kv;
    }

    float[] kScale() {
        return kScale;
    }

    float[] vScale() {
        return vScale;
    }

    PagedStep pagedStep() {
        return pagedStep;
    }

    boolean singlePosition() {
        return singlePosition;
    }

    CpuAttentionCore.Params params() {
        return params;
    }

    CpuAttentionCore.Params nonCausalParams() {
        return nonCausalParams;
    }

    private ModelHparams hparams() {
        return ctx.spec().hparams();
    }

    private float rmsEps() {
        return hparams().normRmsEps();
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        run(pos0, n, in, out);
    }

    /**
     * The cache writes of a step: the K and V projections (every part of a fused product, so the K/V rows are the
     * ones the whole step writes), their biases, clamps, norms and RoPE, the store rotation and the store. The Q
     * projection, its norms and RoPE, the attention, the output and the output projection do not run.
     */
    @Override
    public void prefillKV(int pos0, int n, float[] in) {
        ensureBound();
        Objects.requireNonNull(in, "in");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
        if (proj.source() == QkvProjection.Source.NONE || kv == null || !proj.projectsKv()) {
            return;
        }
        pagedStep = ctx.hasMemory() ? ctx.memory().currentPagedStep() : null;
        float[] k = scratch.slot(CpuScratch.K);
        float[] v = scratch.slot(CpuScratch.V);
        int[] pos = scratch.ints(CpuScratch.POS);
        if (pagedStep == null && singlePosition && pos[0] != pos0) {
            throw new IllegalStateException("layer " + layer.il() + " runs position " + pos0 + " but the "
                    + CpuScratch.POS + " slot starts at " + pos[0]);
        }
        projectKV(n, in, k, v);
        if (valueEmbeddings != null && valueEmbeddings.hasValues()) {
            valueEmbeddings.apply(n, in, v);
        }
        if (inklingStages != null) {
            inklingStages.convKv(n, k, v);
        }
        if (!spec.qkNormAfterRope()) {
            normalizeK(n, k);
        }
        if (vNorm != null) {
            vNorm.apply(n, v);
        }
        if (ropeTable != null) {
            RopeSpec rope = layer.rope();
            if (dualChunk != null) {
                DualChunkRope.applyKey(ropeTable, dualChunk, k, n, spec.nHeadKv(), spec.headDimK(), spec.kDim(),
                        rope.nOffs(), pos, 0, dcPositions);
            } else {
                CpuRope.apply(ropeTable, k, 0, n, spec.nHeadKv(), spec.headDimK(), spec.kDim(), rope.nOffs(), pos, 0,
                        n);
            }
        }
        if (spec.qkNormAfterRope()) {
            normalizeK(n, k);
        }
        if (dualChunk != null) {
            scaleDualChunk(n, pos0, pos, null, k);
        }
        if (storeRotation != null) {
            storeRotation.beforeStoreKV(n, k, v);
        }
        if (pagedStep != null) {
            pagedAttention.store(pagedStep, n, pos, k, v);
        } else {
            hostKvAttention.store(n, pos, k, v);
        }
    }

    @Override
    public String kvRefusal() {
        ensureBound();
        return null;
    }

    /**
     * One step over the rows of {@code s}: the contiguous step at {@code s.pos0(0)} when the step is contiguous —
     * decode for one token with memory, prefill otherwise — the paged step through the memory's
     * {@code CpuMemory.currentPagedStep} otherwise (the memory carries the page tables, whose {@link #run} reads;
     * the step's own layout only picks the branch). An un-paged step over several sequences is refused: the
     * contiguous key span cannot address more than one sequence. A paged step with several segments runs its store
     * and attend per segment, each against its sequence's page table; the token-wise phases (projections, norms,
     * RoPE) run over the whole step once, since no phase mixes rows.
     */
    @Override
    public void batch(com.eonmux.neuralfree.gguf.memory.StepLayout s, float[] in, float[] out) {
        boolean paged = ctx.hasMemory() && ctx.memory().currentPagedStep() != null;
        if (!paged && !s.contiguousSingle()) {
            throw new UnsupportedOperationException(batchRefusal());
        }
        if (s.nTokens() == 1 && ctx.hasMemory() && !paged) {
            decode(s.pos0(0), in, out);
        } else {
            run(s.pos0(0), s.nTokens(), in, out);
        }
    }

    /** The mixer mixes a multi-sequence step only over paged memory (a paged step the memory holds). */
    @Override
    public String batchRefusal() {
        return "attention mixes several sequences per step only over paged memory (a currentPagedStep)";
    }

    private void run(int pos0, int n, float[] in, float[] out) {
        ensureBound();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
        // the backend flags a media chunk's step non-causal (llama_set_causal_attn(false)); the window type and
        // span stay
        CpuAttentionCore.Params stepParams = ctx.stepNonCausal() ? nonCausalParams : params;
        if (dualChunk != null && ctx.stepNonCausal()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " runs dual-chunk attention, which"
                    + " needs a causal step");
        }
        // a tree verify's step: the rows rope at their depth positions, store at their row slots and see only
        // their ancestors within the fresh rows
        final CpuAttentionCore.TreeMask tree = treeMask(n);
        if (tree != null) {
            if (dualChunk != null) {
                throw new UnsupportedOperationException("layer " + layer.il() + " runs dual-chunk attention, which"
                        + " takes no tree mask");
            }
            if (spec.dual()) {
                throw new UnsupportedOperationException("layer " + layer.il() + " runs dual-attention layers, which"
                        + " take no tree mask");
            }
            com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode mode = spec.biasMode();
            if (mode == com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode.T5_CAUSAL
                    || mode == com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode.T5_BIDIR
                    || mode == com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode.INKLING) {
                throw new UnsupportedOperationException("layer " + layer.il() + " runs " + mode + " position bias,"
                        + " which reads a relative table over the position rules a tree mask replaces");
            }
        }
        int[] stepStore = ctx.stepStorePositions();
        final int[] storeAt = stepStore != null ? java.util.Arrays.copyOf(stepStore, n) : null;
        pagedStep = ctx.hasMemory() ? ctx.memory().currentPagedStep() : null;
        if (proj.source() == QkvProjection.Source.NONE) {
            matVec.matmul(wo, in, 0, out, 0, n);
            addDeltas(outSite, n, in, (int) wo.cols(), out, 0, (int) wo.rows());
            return;
        }
        float[] q = scratch.slot(CpuScratch.Q);
        float[] k = proj.projectsKv() ? scratch.slot(CpuScratch.K) : null;
        float[] v = proj.projectsKv() ? scratch.slot(CpuScratch.V) : null;
        float[] attn = scratch.slot(CpuScratch.ATTN_OUT);
        float[] gate = gateWidth > 0 ? scratch.temp(T_GATE, ubatch * gateWidth) : null;
        int[] pos = scratch.ints(CpuScratch.POS);
        if (pagedStep == null && singlePosition && pos[0] != pos0) {
            throw new IllegalStateException("layer " + layer.il() + " runs position " + pos0 + " but the "
                    + CpuScratch.POS + " slot starts at " + pos[0]);
        }
        project(n, in, q, k, v, gate);
        if (valueEmbeddings != null && valueEmbeddings.hasValues()) {
            valueEmbeddings.apply(n, in, v);
        }
        if (inklingStages != null) {
            inklingStages.projectRel(n, in);
            inklingStages.convKv(n, k, v);
        }
        if (!spec.qkNormAfterRope()) {
            normalizeQk(n, q, k);
        }
        if (vNorm != null) {
            vNorm.apply(n, v);
        }
        if (ropeTable != null) {
            RopeSpec rope = layer.rope();
            if (dualChunk != null) {
                DualChunkRope.applyQuery(ropeTable, dualChunk, q, ubatch * spec.qDim(), n, spec.nHead(),
                        spec.headDimK(), spec.qDim(), rope.nOffs(), pos, 0, dcPositions);
                if (k != null) {
                    DualChunkRope.applyKey(ropeTable, dualChunk, k, n, spec.nHeadKv(), spec.headDimK(), spec.kDim(),
                            rope.nOffs(), pos, 0, dcPositions);
                }
            } else {
                CpuRope.apply(ropeTable, q, 0, n, spec.nHead(), spec.headDimK(), spec.qDim(), rope.nOffs(), pos, 0, n);
                if (k != null) {
                    CpuRope.apply(ropeTable, k, 0, n, spec.nHeadKv(), spec.headDimK(), spec.kDim(), rope.nOffs(), pos,
                            0, n);
                }
            }
        }
        if (spec.qkNormAfterRope()) {
            normalizeQk(n, q, k);
        }
        if (spec.temperature().enabled()) {
            AttnExtMath.applyTemperature(spec.temperature(), n, spec.nHead(), spec.headDimK(), pos, 0, q, 0,
                    spec.qDim());
        }
        if (inklingStages != null) {
            // vLLM scales the normed queries and the relative logits of a full layer by the token's log factor.
            inklingStages.scaleQueries(n, q, pos);
            inklingStages.relLogits(n, pos);
        }
        if (dualChunk != null) {
            scaleDualChunk(n, pos0, pos, q, k);
        }
        if (rotation != null) {
            if (dualChunk != null) {
                rotation.beforeStorePlanes(n, q, DualChunkRope.VARIANTS, ubatch * spec.qDim(), k, v);
            } else {
                rotation.beforeStore(n, q, k, v);
            }
        }
        if (kv != null) {
            if (dualChunk != null) {
                if (pagedStep != null) {
                    if (k != null) {
                        pagedAttention.store(pagedStep, n, pos, k, v);
                    }
                    attendDualChunkPaged(stepParams, q, pos, attn);
                } else {
                    hostKvAttention.store(n, storeAt != null ? storeAt : pos, k, v);
                    attendDualChunkContiguous(stepParams, pos0, n, q, pos, attn);
                }
            } else if (spec.dual()) {
                if (pagedStep != null) {
                    throw new UnsupportedOperationException(loopAttention.pagedRefusal());
                }
                hostKvAttention.store(n, storeAt != null ? storeAt : pos, k, v);
                loopAttention.attend(stepParams, pos0, n, q, pos, attn);
            } else if (pagedStep != null) {
                if (k != null) {
                    pagedAttention.store(pagedStep, n, pos, k, v);
                }
                pagedAttention.attend(stepParams, q, pos, attn, kv.group(), tree);
            } else {
                hostKvAttention.run(stepParams, pos0, n, pos, storeAt != null ? storeAt : pos, q, k, v, attn, tree);
            }
        } else {
            if (tree != null) {
                throw new UnsupportedOperationException("layer " + layer.il() + " attends a cacheless dense batch,"
                        + " which takes no tree mask");
            }
            CpuAttentionCore.DenseKv batch = CpuAttentionCore.DenseKv.of(k, 0, spec.kDim(), spec.headDimK(), v, 0,
                    spec.vDim(), spec.headDimV(), spec.nHeadKv(), spec.headDimK(), spec.headDimV(), pos, 0);
            CpuAttentionCore.attendRows(stepParams, n, q, 0, spec.qDim(), pos, 0, batch, 0, n, attn, 0, spec.outDim());
        }
        if (rotation != null) {
            rotation.afterAttention(n, attn);
        }
        if (valueEmbeddings != null && valueEmbeddings.hasXsa()) {
            valueEmbeddings.applyXsa(n, v, attn);
        }
        if (outHeadNorm != null) {
            outHeadNorm.apply(n, attn);
        }
        if (gate != null) {
            AttnExtMath.applyOutGate(spec.gate(), n, spec.nHead(), spec.headDimV(), attn, 0, spec.outDim(), gate, 0,
                    gateWidth, gateBias, 0);
        }
        if (outNormStep != null) {
            outNormStep.apply(n, attn);
        }
        if (subNorm != null) {
            subNorm.apply(n, attn);
        }
        matVec.matmul(wo, attn, 0, out, 0, n);
        addDeltas(outSite, n, attn, spec.outDim(), out, 0, (int) wo.rows());
        if (outBias != null) {
            CpuOps.addRowBias(out, 0, n, (int) wo.rows(), outBias, 0);
        }
        if (spec.valueScale() != 0f) {
            CpuOps.scale(out, 0, n * (int) wo.rows(), spec.valueScale());
        }
    }

    private void project(int n, float[] in, float[] q, float[] k, float[] v, float[] gate) {
        int qw = proj.qProjWidth();
        float[] qProj = proj.interleavedGate() ? scratch.temp(T_QGATE, ubatch * qw) : q;
        float[] qLatent = wqUp != null ? scratch.temp(T_QLAT, ubatch * qw) : qProj;
        if (wqkv != null) {
            int fw = proj.fusedWidth();
            float[] qkv = scratch.temp(T_QKV, ubatch * fw);
            int cols = (int) wqkv.cols();
            matVec.matmul(wqkv, in, 0, qkv, 0, n);
            addDeltas(qkvSite, n, in, cols, qkv, 0, fw);
            addDeltas(qSite, n, in, cols, qkv, proj.qOffset(), fw);
            addDeltas(kSite, n, in, cols, qkv, proj.kOffset(), fw);
            addDeltas(vSite, n, in, cols, qkv, proj.vOffset(), fw);
            if (qkvBias != null) {
                CpuOps.addRowBias(qkv, 0, n, fw, qkvBias, 0);
            }
            for (int t = 0; t < n; t++) {
                int row = t * fw;
                System.arraycopy(qkv, row + proj.qOffset(), qProj, t * qw, qw);
                if (k != null) {
                    System.arraycopy(qkv, row + proj.kOffset(), k, t * spec.kDim(), spec.kDim());
                    System.arraycopy(qkv, row + proj.vOffset(), v, t * spec.vDim(), spec.vDim());
                }
            }
        } else {
            int cols = (int) wq.cols();
            if (wk != null && wv != null) {
                matVec.matmulGroup(new CpuTensorRef[] {wq, wk, wv}, in, 0, new float[][] {qLatent, k, v}, n);
            } else if (wk != null) {
                matVec.matmulGroup(new CpuTensorRef[] {wq, wk}, in, 0, new float[][] {qLatent, k}, n);
            } else {
                matVec.matmul(wq, in, 0, qLatent, 0, n);
            }
            addDeltas(qSite, n, in, cols, qLatent, 0, qw);
            if (wk != null) {
                addDeltas(kSite, n, in, cols, k, 0, spec.kDim());
                if (wv != null) {
                    addDeltas(vSite, n, in, cols, v, 0, spec.vDim());
                } else if (valueExperts != null) {
                    valueExperts.run(n, in, v);
                } else {
                    System.arraycopy(k, 0, v, 0, n * spec.kDim());
                }
            }
        }
        applyAdapters(n, in, q, k, v);
        if (qBias != null) {
            CpuOps.addRowBias(qProj, 0, n, qw, qBias, 0);
        }
        if (wqUp != null) {
            qLatentNorm.apply(n, qLatent);
            matVec.matmul(wqUp, qLatent, 0, q, 0, n);
        }
        if (kBias != null) {
            CpuOps.addRowBias(k, 0, n, spec.kDim(), kBias, 0);
        }
        if (vBias != null) {
            CpuOps.addRowBias(v, 0, n, spec.vDim(), vBias, 0);
        }
        if (clamp != null) {
            CpuOps.unary(ActOp.CLAMP, clamp, qProj, 0, qProj, 0, n * qw);
            if (k != null) {
                CpuOps.unary(ActOp.CLAMP, clamp, k, 0, k, 0, n * spec.kDim());
                CpuOps.unary(ActOp.CLAMP, clamp, v, 0, v, 0, n * spec.vDim());
            }
        }
        if (proj.interleavedGate()) {
            int hd = spec.headDimK();
            for (int r = 0; r < n * spec.nHead(); r++) {
                System.arraycopy(qProj, r * 2 * hd, q, r * hd, hd);
                System.arraycopy(qProj, r * 2 * hd + hd, gate, r * hd, hd);
            }
        } else if (wgate != null) {
            matVec.matmul(wgate, in, 0, gate, 0, n);
            addDeltas(gateSite, n, in, (int) wgate.cols(), gate, 0, gateWidth);
        }
    }

    private static void addDeltas(CpuProjectionDeltas.Site site, int n, float[] x, int xStride, float[] y, int yOff,
                                  int yStride) {
        if (site != null && !site.isEmpty()) {
            site.apply(n, x, 0, xStride, y, yOff, yStride);
        }
    }

    /**
     * The delta site of projection {@code weight}; a matvec with a non-identity input transform (PRISM Hadamard, or
     * an ACT_QUANT/PERMUTE input transform) refuses deltas.
     */
    CpuProjectionDeltas.Site serveDeltas(String weight, CpuTensorRef w) {
        CpuMatVec.Transforms t = matVec.transformsOf(w);
        String refusal = !t.transform().isIdentity() ? weight + " of layer " + layer.il()
                + " transforms its input (PRISM Hadamard or input transform) inside the matvec; a LoRA delta would"
                + " need the transformed rows" : null;
        return ctx.projectionDeltas().serve(layer, weight, refusal);
    }

    /**
     * The base products the cache writes need: every part of a fused projection (its K and V rows are read from the
     * fused rows, so the product runs whole), otherwise K and V alone. The Q product, the gate and their deltas do
     * not run.
     */
    private void projectKV(int n, float[] in, float[] k, float[] v) {
        if (wqkv != null) {
            int fw = proj.fusedWidth();
            float[] qkv = scratch.temp(T_QKV, ubatch * fw);
            int cols = (int) wqkv.cols();
            matVec.matmul(wqkv, in, 0, qkv, 0, n);
            addDeltas(qkvSite, n, in, cols, qkv, 0, fw);
            addDeltas(kSite, n, in, cols, qkv, proj.kOffset(), fw);
            addDeltas(vSite, n, in, cols, qkv, proj.vOffset(), fw);
            if (qkvBias != null) {
                CpuOps.addRowBias(qkv, 0, n, fw, qkvBias, 0);
            }
            for (int t = 0; t < n; t++) {
                int row = t * fw;
                System.arraycopy(qkv, row + proj.kOffset(), k, t * spec.kDim(), spec.kDim());
                System.arraycopy(qkv, row + proj.vOffset(), v, t * spec.vDim(), spec.vDim());
            }
        } else {
            int cols = (int) wq.cols();
            if (wk != null) {
                matVec.matmul(wk, in, 0, k, 0, n);
                addDeltas(kSite, n, in, cols, k, 0, spec.kDim());
                if (wv != null) {
                    matVec.matmul(wv, in, 0, v, 0, n);
                    addDeltas(vSite, n, in, cols, v, 0, spec.vDim());
                } else if (valueExperts != null) {
                    valueExperts.run(n, in, v);
                } else {
                    System.arraycopy(k, 0, v, 0, n * spec.kDim());
                }
            }
        }
        applyAdapters(n, in, null, k, v);
        if (kBias != null) {
            CpuOps.addRowBias(k, 0, n, spec.kDim(), kBias, 0);
        }
        if (vBias != null) {
            CpuOps.addRowBias(v, 0, n, spec.vDim(), vBias, 0);
        }
        if (clamp != null) {
            CpuOps.unary(ActOp.CLAMP, clamp, k, 0, k, 0, n * spec.kDim());
            CpuOps.unary(ActOp.CLAMP, clamp, v, 0, v, 0, n * spec.vDim());
        }
    }

    private void normalizeK(int n, float[] k) {
        if (kNorm != null) {
            kNorm.apply(n, k);
        }
    }

    private void normalizeQk(int n, float[] q, float[] k) {
        if (qNorm != null) {
            qNorm.apply(n, q);
        }
        if (kNorm != null) {
            kNorm.apply(n, k);
        }
    }

    // ================================================================== dual-chunk attention

    /**
     * The original-context logit scaling of a step (vLLM scales the new K rows before caching and, on a decode, the
     * query too). A contiguous step scales its one sequence; a paged step scales each segment by its sequence's slot
     * count.
     */
    private void scaleDualChunk(int n, int pos0, int[] pos, float[] q, float[] k) {
        if (pagedStep != null) {
            int[] slots = pagedStep.seqSlots(kv.group());
            for (int seg = 0; seg < pagedStep.nSeg(); seg++) {
                int seq = pagedStep.segSeq(seg);
                int start = pagedStep.segStart(seg);
                int len = pagedStep.segLen(seg);
                scaleDualChunkRows(dualChunk.scalingFactor(slots[seq]), q, k, start, len);
            }
            return;
        }
        int seqLen = n == 1 ? pos0 + 1 : pos0 + n;
        scaleDualChunkRows(dualChunk.scalingFactor(seqLen), q, k, 0, n);
    }

    /** Scales rows {@code [start, start + len)} of K and of the three used query planes by {@code factor}. */
    private void scaleDualChunkRows(float factor, float[] q, float[] k, int start, int len) {
        if (factor == 1.0f || len <= 0) {
            return;
        }
        if (k != null) {
            CpuOps.scale(k, start * spec.kDim(), len * spec.kDim(), factor);
        }
        if (q != null) {
            int planeStride = ubatch * spec.qDim();
            for (int plane = 0; plane < 3; plane++) {
                CpuOps.scale(q, plane * planeStride + start * spec.qDim(), len * spec.qDim(), factor);
            }
        }
    }

    /** Dual-chunk attention of a contiguous step over the layer's cache rows {@code [0, pos0 + n)}. */
    private void attendDualChunkContiguous(CpuAttentionCore.Params causal, int pos0, int n, float[] q, int[] pos,
                                           float[] attn) {
        final int keys = Math.min(kv.layout().rows(), pos0 + n);
        CpuDualChunkAttention.KeySpan span = new CpuDualChunkAttention.KeySpan() {
            @Override public int keys() { return keys; }

            @Override
            public void attend(CpuAttentionCore.Params p, int rows, float[] qa, int qOff, int qRowStride,
                               int[] qPos, int qPosOff, int keyStart, int keyEnd, float[] out, int outOff,
                               float[] lse, int lseOff) {
                CpuAttentionCore.attendRowsLse(p, rows, qa, qOff, qRowStride, qPos, qPosOff, kv, rowPos, keyStart,
                        keyEnd, out, outOff, spec.outDim(), lse, lseOff, spec.nHead());
            }
        };
        CpuDualChunkAttention.attend(dualChunk, spec, causal, nonCausalParams, span, n, q, ubatch * spec.qDim(),
                spec.qDim(), 0, pos, 0, attn, 0, spec.outDim(), dcScratchOut, dcScratchLse, dcMergeLse);
    }

    /** Dual-chunk attention of a paged step, segment by segment through its sequence's page table. */
    private void attendDualChunkPaged(CpuAttentionCore.Params causal, float[] q, int[] pos, float[] attn) {
        int shift = CpuPagedAttention.pageShift(pagedStep);
        int group = kv.group();
        for (int seg = 0; seg < pagedStep.nSeg(); seg++) {
            final int seq = pagedStep.segSeq(seg);
            final int start = pagedStep.segStart(seg);
            final int len = pagedStep.segLen(seg);
            final int slots = CpuPagedAttention.slots(pagedStep, group, seq);
            final int[] table = pagedAttention.pageTable(pagedStep, group, seq);
            CpuDualChunkAttention.KeySpan span = new CpuDualChunkAttention.KeySpan() {
                @Override public int keys() { return slots; }

                @Override
                public void attend(CpuAttentionCore.Params p, int rows, float[] qa, int qOff, int qRowStride,
                                   int[] qPos, int qPosOff, int keyStart, int keyEnd, float[] out, int outOff,
                                   float[] lse, int lseOff) {
                    CpuAttentionCore.attendRowsLse(p, rows, qa, qOff, qRowStride, qPos, qPosOff, kv, rowPos, shift,
                            table, keyStart, keyEnd, out, outOff, spec.outDim(), lse, lseOff, spec.nHead());
                }
            };
            CpuDualChunkAttention.attend(dualChunk, spec, causal, nonCausalParams, span, len, q,
                    ubatch * spec.qDim(), spec.qDim(), start, pos, start, attn, start * spec.outDim(),
                    spec.outDim(), dcScratchOut, dcScratchLse, dcMergeLse);
        }
    }

    /**
     * The running step's tree mask over {@code n} rows, or null: the backend set one per ubatch row and the fresh
     * rows are the ubatch's {@code n} rows (a batched tree verify step's masked pieces and plain rows alike).
     */
    private CpuAttentionCore.TreeMask treeMask(int n) {
        long[] bits = ctx.stepTreeMask();
        int fresh = ctx.stepTreeFresh();
        if (bits == null) {
            return null;
        }
        if (n != fresh) {
            throw new IllegalStateException("layer " + layer.il() + " runs " + n + " rows but the step's tree mask"
                    + " covers " + fresh + " fresh rows");
        }
        return new CpuAttentionCore.TreeMask(java.util.Arrays.copyOf(bits, n), fresh);
    }

    // ================================================================== context shift

    /**
     * Applies the RoPE delta of a context shift to this layer's own K rows: the executor has already moved the rows
     * with {@link HeapModelMemory#shiftPositions}; every row is read, rotated by RoPE at its delta (inside the
     * Hadamard rotation of a rotated block-type cache, as {@code LLM_KV_RopeShift_<KV>} does) and written back
     * (llama.cpp {@code build_rope_shift}, src/llama-kv-cache.cpp:1945-2070). Rows with delta 0 are rotated too, as
     * llama.cpp shifts every cell with its per-cell shift (llama-kv-cache.cpp:1545-1556). Only the shifted dims of
     * each head are stored again: the rotated dims, widened for a block type to the kernel's work window
     * ({@link RopeKernels#shiftWindowStart}), so the other blocks keep their stored bits as in the GPU kernel and in
     * llama.cpp, which requantizes only the {@code n_rot} view of each head (llama-kv-cache.cpp:2058-2066).
     *
     * @throws UnsupportedOperationException for a layer without a cache, a model llama.cpp cannot shift
     *                                       ({@link GpuContextShift#refusal}), or a memory without shift deltas
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureBound();
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep + "/"
                    + nDiscard);
        }
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        if (proj.source() == QkvProjection.Source.NONE || shiftTable == null || !GpuContextShift.shifts(layer)) {
            return;
        }
        int[] delta = ((HeapModelMemory) ctx.memory()).shiftDelta()[kv.group()];
        shiftRows(delta);
    }

    /** {@link CpuMixer#applyShiftDeltas}: the paged shift's row moves already happened, rotate by the deltas. */
    @Override
    public void applyShiftDeltas() {
        ensureBound();
        if (kv == null || proj.source() == QkvProjection.Source.NONE || shiftTable == null
                || !GpuContextShift.shifts(layer)) {
            return;
        }
        shiftRows(((HeapModelMemory) ctx.memory()).shiftDelta()[kv.group()]);
    }

    /**
     * Null when {@link #shiftContext} can run: a layer without its own projections, or a cached layer of a model
     * llama.cpp can shift over a memory that records shift deltas.
     */
    @Override
    public String shiftRefusal() {
        ensureBound();
        if (proj.source() == QkvProjection.Source.NONE) {
            return null;
        }
        if (!spec.usesMemory()) {
            return "layer " + layer.il() + " attends only to the current batch; it has no context to shift";
        }
        String refusal = GpuContextShift.refusal(ctx.spec(), layer);
        if (refusal != null) {
            return refusal;
        }
        CpuMemory memory = ctx.memory();
        if (shiftTable != null && GpuContextShift.shifts(layer) && !(memory instanceof HeapModelMemory)) {
            return "context shift needs HeapModelMemory deltas, the memory is " + memory.getClass().getSimpleName();
        }
        return null;
    }

    private void shiftRows(int[] delta) {
        RopeSpec rope = layer.rope();
        int hd = spec.headDimK();
        int width = spec.kDim();
        int rows = Math.min(delta.length, kv.layout().rows());
        float[] row = scratch.temp(T_SHIFT, width);
        int[] d = scratch.intTemp(T_SHIFT, 1);
        HadamardSpec h = shiftRotK > 1 ? HadamardSpec.rotation(shiftRotK) : null;
        boolean blocks = kv.layout().dtypeK().quantized();
        int w0 = blocks ? RopeKernels.shiftWindowStart(rope.nOffs(), shiftRotK) : rope.nOffs();
        int w1 = blocks ? RopeKernels.shiftWindowEnd(rope.nDims(), rope.nOffs(), shiftRotK)
                : rope.nOffs() + rope.nDims();
        for (int r = 0; r < rows; r++) {
            for (int head = 0; head < spec.nHeadKv(); head++) {
                kv.readK(r, head, row, head * hd);
            }
            if (h != null) {
                windowHadamard(h, row, hd, w0, w1);
            }
            d[0] = delta[r];
            CpuRope.apply(shiftTable, row, 0, 1, spec.nHeadKv(), hd, width, rope.nOffs(), d, 0, 1);
            if (h != null) {
                windowHadamard(h, row, hd, w0, w1);
            }
            for (int head = 0; head < spec.nHeadKv(); head++) {
                kv.writeKSlice(r, head * hd + w0, w1 - w0, row, head * hd + w0);
            }
        }
    }

    /** The self-inverse Hadamard rotation of the {@code [w0, w1)} window of every head, in {@code rotK} chunks. */
    private void windowHadamard(HadamardSpec h, float[] row, int hd, int w0, int w1) {
        for (int head = 0; head < spec.nHeadKv(); head++) {
            CpuHadamard.applyInPlace(h, row, head * hd + w0, 1, w1 - w0);
        }
    }

    // ================================================================== lifecycle

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("attention mixer is closed");
        }
        if (!bound) {
            throw new IllegalStateException("attention mixer is not bound");
        }
    }

    /** The layer this mixer is bound to; null before {@link #bind}. */
    public LayerSpec layer() {
        return layer;
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuAttentionMixer{" + (layer == null ? "unbound" : "layer " + layer.il() + ", " + proj) + '}';
    }

    /** Row norm over the dense per-token rows of one projection ({@link NormMath#rowNorm}, in place). */
    private static final class NormStep {
        private final NormKind kind;
        private final float eps;
        private final int rowsPerToken;
        private final int rowLen;
        private final float[] w;
        private final float[] b;
        private final int wRows;

        private NormStep(NormKind kind, float eps, int rowsPerToken, int rowLen, float[] w, float[] b, int wRows) {
            this.kind = kind;
            this.eps = eps;
            this.rowsPerToken = rowsPerToken;
            this.rowLen = rowLen;
            this.w = w;
            this.b = b;
            this.wRows = wRows;
        }

        private void apply(int n, float[] x) {
            NormMath.rowNorm(kind, eps, n * rowsPerToken, rowLen, rowLen, wRows, x, 0, w, b, x, 0);
        }
    }
}
