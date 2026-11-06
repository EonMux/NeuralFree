package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuMatmulTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuIndexerCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.DsaLayout;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.IndexerConfig;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * DeepSeek sparse-attention MLA mixer on the GPU ({@link com.eonmux.neuralfree.gguf.arch.MixerKind#DSA_MLA}) for
 * deepseek32, glm-dsa, dots3note and hy_v4 (llama.cpp src/models/deepseek32.cpp:200-396 and the per-arch differences
 * listed in {@link DsaLayout}). Per step, on the normalized layer input {@code x}:
 * <ol>
 *   <li>{@code qr = RMS(W_q_a . x) * attn_q_a_norm};</li>
 *   <li>full layers: the lightning indexer ({@link GpuDsaIndexer}) writes the step's top-k cache rows to the shared
 *       {@value #TEMP_TOPK} buffer; shared layers (glm-dsa and hy_v4 with {@code DsaSpec.full() == false}) reuse
 *       the selection their source layer wrote earlier in the same step (glm-dsa.cpp:350-355,
 *       hy-v4.cpp:439-441);</li>
 *   <li>{@code q = W_q_b . qr} split per head into {@code [nope | pe]}; {@code [kv_cmpr | k_pe] = W_kv_a . x};
 *       dots3note RMS-normalizes {@code k_pe} with {@code attn_k_norm} first; RoPE on {@code q_pe} and
 *       {@code k_pe}; {@code kv_cmpr = RMS(kv_cmpr) * attn_kv_a_norm};</li>
 *   <li>the latent row {@code [kv_cmpr | k_pe]} is stored at {@code cell_idx} of the layer's latent cache (its
 *       {@code KvDType} follows {@code neuralfree.kv.type}); like llama.cpp's DSA {@code build_attn}
 *       (llama-graph.cpp:3402-3478), no Hadamard rotation is applied to the latent rows;</li>
 *   <li>{@code q_abs[h] = W_k_b[h] . q_nope[h]} ({@code {T}_MatVecId} with one slot per (token, head)), then
 *       {@code Q = [q_abs | q_pe]} per head;</li>
 *   <li>{@code LLM_MLA_SparseDecode/Prefill_<KV>} attends over the selected rows with {@code kq_scale}, the sinks
 *       when present, and writes {@code out_latent [t][h][kv_lora]};</li>
 *   <li>{@code attn[h] = W_v_b[h] . out_latent[h]}; the optional sigmoid gate; then
 *       {@code out = W_o . attn} with its {@code attn_output.scale} when present.</li>
 * </ol>
 *
 * <p>{@code n_kv} of a step is {@code min(cache rows, pos0 + n_tokens)}: rows beyond hold no position visible to the
 * step (a full cache stores position {@code p} in row {@code p}, a ring has every row in use once it wraps), and
 * {@code k = n_sel = min(n_kv, topK)} (deepseek32.cpp:313-315).</p>
 *
 * <p>mistral-large4 projects Q directly with {@code attn_q}, scales the assembled {@code qcur} by the SSMax per-head
 * temperature {@code attn_ssmax[h] * log(pos + 1) + attn_ssmax.bias[h]} ({@code LLM_AttnSsmaxScale}), reads the
 * learned {@code rope_freqs} factors in the MLA and indexer rotations, and its NoPE layers zero-pad the rope dims of Q
 * and of the latent row (mistral-large4.cpp:181-197, 345-421).</p>
 *
 * <p>A paged multi-sequence step ({@link #batch}) runs the token-wise stages over every row of the step and stores
 * the latent rows and indexer keys at the step's cells; every segment then selects among its own sequence's cells:
 * the indexer scores all cells against the sequence's {@code kv_pos} ({@code GpuStepInputs.sequenceKvPos}, every
 * other cell -1), as llama.cpp's KV mask hides other sequences' cells from the scores, so the top-k names the
 * sequence's cells directly and the sparse attention reads them unchanged. The selection width is
 * {@code k = min(cache rows, topK)} for every segment; a sequence with fewer visible rows pads its selection with -1,
 * which the attention skips, so it selects the rows a sequence-local {@code min(slots, topK)} selects.</p>
 *
 * <p>Working buffers are {@code GpuScratch} temporaries reserved at bind; the top-k buffer {@value #TEMP_TOPK} is
 * reserved by every DSA layer with the same size ({@code ubatch * topK}, model constants), so all of a model's DSA
 * layers share one buffer and a shared layer reads what its source layer wrote. Per step only scalars, global sizes
 * and chunk buffers are re-set; nothing is allocated and nothing waits on the host.</p>
 */
public final class GpuDsaMlaMixer implements GpuMixer {

    public static final String TEMP_QR = "dsa.qr";
    public static final String TEMP_Q = "dsa.q";
    public static final String TEMP_KV = "dsa.kvpe";
    public static final String TEMP_QCUR = "dsa.qcur";
    public static final String TEMP_LATENT = "dsa.latent";
    public static final String TEMP_GATE = "dsa.gate";
    public static final String TEMP_GATE_WIDE = "dsa.gate.wide";
    public static final String TEMP_HEAD_SLOTS = "dsa.heads.";
    public static final String TEMP_TOPK = "dsa.topk";
    public static final String TEMP_KV_PROJ = "dsa.kvproj";

    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuForwardContext ctx;
    private GpuOps ops;
    private DsaLayout layout;
    private int ubatch;
    private int local;
    private int idLocal;
    private int idChunkTokens;
    private int sparseLocal;
    private int kvRows;
    private KvDType latentType;
    private GpuStepInputs steps;

    private GpuWeight wq;
    private GpuWeight wqA;
    private GpuWeight wqB;
    private GpuWeight wkvA;
    private GpuWeight wo;
    private GpuWeight gateW;
    /** The per-head {@code attn_k_b} and {@code attn_v_b} the id-matvec kernels multiply. */
    private GpuWeight wkB;
    private GpuWeight wvB;
    /**
     * LoRA deltas of the dense projections, after the product and its scale ({@code build_lora_mm},
     * llama-graph.cpp:1629-1667); the per-head absorbed {@code attn_k_b} / {@code attn_v_b} take none.
     */
    private GpuProjectionDeltas.Site qDeltas;
    private GpuProjectionDeltas.Site qaDeltas;
    private GpuProjectionDeltas.Site qbDeltas;
    private GpuProjectionDeltas.Site kvaDeltas;
    private GpuProjectionDeltas.Site outDeltas;
    private GpuProjectionDeltas.Site gateDeltas;

    private ScratchSlot qr;
    private ScratchSlot q;
    private ScratchSlot kv;
    private ScratchSlot kvProj;
    private ScratchSlot qcur;
    private ScratchSlot latent;
    private ScratchSlot gate;
    private ScratchSlot gateWide;

    private GpuDsaIndexer indexer;
    /** SSMax per-head scale and bias ({@code attn_ssmax}); null when the query is not scaled. */
    private GpuBuffer ssmaxScale;
    private GpuBuffer ssmaxBias;
    /** Per-pair divisors of the layer RoPE ({@code rope_freqs}) and whether the rope dispatches read them. */
    private GpuBuffer ropeFactors;
    private boolean hasRopeFactors;
    /** The latent cache's KV group and the indexer key rows a paged step selects through. */
    private int group;
    private GpuBuffer indexerKeys;
    /** The shared top-k buffer {@value #TEMP_TOPK}. */
    private GpuBuffer selection;

    private PreparedDispatch qANorm;
    private PreparedDispatch kPeNorm;
    private PreparedDispatch kvANorm;
    private PreparedDispatch qRope;
    private PreparedDispatch kRope;
    private PreparedDispatch querySsmax;
    private PreparedDispatch latentStore;
    private PreparedDispatch absorb;
    private PreparedDispatch sparseDecode;
    private PreparedDispatch sparsePrefill;
    private PreparedDispatch decompress;
    private PreparedDispatch gateGlu;
    private boolean closed;

    /** Use {@link SparseBlocks#gpuMixers()}. */
    GpuDsaMlaMixer() {
    }

    /**
     * Binds the mixer: validates the layer's tensors against its spec ({@link DsaLayout}), reserves the working
     * buffers, binds the indexer of a full layer and prepares every dispatch.
     *
     * @throws IllegalStateException         when already bound or closed, the context has no memory, or the
     *                                       tensors or caches contradict the spec
     * @throws UnsupportedOperationException when a tensor type has no kernel or a per-head tensor cannot be bound
     */
    @Override
    public void bind(GpuForwardContext context, LayerSpec layer) {
        if (closed || ctx != null) {
            throw new IllegalStateException("DSA mixer is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(layer, "layer");
        ctx = context;
        try {
            bindLayer(layer);
        } catch (RuntimeException e) {
            releaseAll(e);
            ctx = null;
            throw e;
        }
    }

    private void bindLayer(LayerSpec layer) {
        ops = ctx.ops();
        ubatch = ctx.scratch().ubatch();
        local = ops.local();
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        layout = DsaLayout.of(layer, ctx.spec(), suffix -> shape(weights.layerOrNull(layer, suffix)));
        DsaLayout l = layout;
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(l.where + ": the DSA mixer needs the model's KV memory");
        }
        GpuMemory memory = ctx.memory();
        steps = memory.stepInputs();
        GpuKvLayer cache = latentCache(memory, layer.il());
        kvRows = cache.layout().rows();
        group = cache.group();
        steps.reserveSequenceAddressing(group);
        if (l.kvShared) {
            GpuKvLayer src = memory.kv(l.kvSourceLayer);
            if (src == null || src.group() != group || src.layout().rows() != kvRows) {
                throw new IllegalStateException(l.where + ": reads the latent cache of layer " + l.kvSourceLayer
                        + ", which must use the layer's KV group and rows (group " + group + ", " + kvRows
                        + " rows)");
            }
        }

        wkvA = l.kvShared ? null : projection(weights.layer(layer, DsaLayout.WKV_A));
        wo = projection(outputWeight(layer, weights));
        wkB = perHead(weights.layer(layer, DsaLayout.WK_B));
        wvB = perHead(weights.layer(layer, DsaLayout.WV_B));
        checkWorkGroups(wkB, wvB);
        if (l.qLora > 0) {
            wqA = projection(weights.layer(layer, DsaLayout.WQ_A));
            wqB = projection(weights.layer(layer, DsaLayout.WQ_B));
        } else {
            wq = projection(weights.layer(layer, DsaLayout.WQ));
        }

        if (l.qLora > 0) {
            qr = temp(TEMP_QR, l.qLora);
        }
        q = temp(TEMP_Q, l.qBufferWidth());
        kv = temp(TEMP_KV, l.dK);
        if (!l.ropeApplied) {
            kvProj = temp(TEMP_KV_PROJ, l.kvLora);
        }
        qcur = temp(TEMP_QCUR, (long) l.nHead * l.dK);
        latent = temp(TEMP_LATENT, (long) l.nHead * l.kvLora);
        ScratchSlot slots = headSlots();
        ScratchSlot topK = temp(TEMP_TOPK, l.topK);
        selection = topK.buffer();

        prepareProjections(layer, weights, cache);
        prepareHeadProducts(wkB, wvB, slots);
        prepareSparse(layer, weights, cache, topK);
        prepareSsmax(layer, weights);
        bindGate(layer, weights);
        bindDeltas(layer);
        if (l.full) {
            bindIndexer(layer, weights, memory, cache, topK);
        } else {
            checkSource(memory, cache);
        }
    }

    /** Work-group sizes of the per-head products and the sparse kernels, checked against their kernels. */
    private void checkWorkGroups(GpuWeight wkB, GpuWeight wvB) {
        idLocal = Math.min(local, MoeKernels.MAX_LOCAL);
        MoeKernels.checkIdLocal(wkB.type(), idLocal);
        MoeKernels.checkIdLocal(wvB.type(), idLocal);
        idChunkTokens = idChunkTokens(ctx.kernels().caps().getMaxWorkGroupCount()[1]);
        sparseLocal = Math.min(SparseKernels.SPARSE_MLA_LOCAL, local);
        SparseKernels.checkSparseMlaLocal(sparseLocal);
        NormKernels.checkReduceLocal(local);
    }

    /** The q/kv low-rank norms, the q and k RoPE and the latent-cache store. */
    private void prepareProjections(LayerSpec layer, GpuWeights weights, GpuKvLayer cache) {
        DsaLayout l = layout;
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        if (l.qLora > 0) {
            qANorm = prepareNorm(vector(weights.layer(layer, DsaLayout.Q_A_NORM), l.qLora), qr.buffer(), l.qLora,
                    l.qLora, 0);
        }
        if (l.kPeNorm) {
            kPeNorm = prepareNorm(vector(weights.layer(layer, DsaLayout.K_PE_NORM), l.rope), kv.buffer(), l.rope,
                    l.dK, l.kvLora);
        }
        if (!l.kvShared) {
            kvANorm = prepareNorm(vector(weights.layer(layer, DsaLayout.KV_A_NORM), l.kvLora), kv.buffer(), l.kvLora,
                    l.dK, 0);
        }
        RopeSpec r = l.ropeSpec;
        ropeFactors = bindRopeFactors(weights, r, layer.il());
        hasRopeFactors = ropeFactors != null;
        GpuBuffer ff = hasRopeFactors ? ropeFactors : ph;
        GpuBuffer ropePos = steps.ropePos();
        if (r != null) {
            qRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(r, ubatch, l.nHead), new long[]{local},
                    RopeKernels.ropeExtArgs(r, ubatch, l.nHead, l.headK, l.nHead * l.headK, l.nope, ubatch, 0,
                            hasRopeFactors, ropePos, ff, q.buffer()));
        }
        if (!l.kvShared) {
            if (r != null) {
                kRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(r, ubatch, 1), new long[]{local},
                        RopeKernels.ropeExtArgs(r, ubatch, 1, l.dK, l.dK, l.kvLora, ubatch, 0, hasRopeFactors, ropePos,
                                ff, kv.buffer()));
            }
            KvDType dtype = cache.layout().dtypeK();
            latentType = dtype;
            latentStore = prepare(RopeKernels.kvStore(dtype),
                    elementGlobal(RopeKernels.kvStoreWorkItems(dtype, ubatch, l.dK)), new long[]{local},
                    RopeKernels.kvStoreArgs(dtype, ubatch, l.dK, l.dK, 0, cache.layout().widthK(), l.dK, kv.buffer(),
                            steps.cellIdx(cache.group()), ctx.placeholders().kvScalesOnes(), cache.k()));
        }
    }

    /**
     * The layer's frequency factors as one device buffer, or null when its RoPE has none.
     *
     * @throws MissingTensorException when the spec names factors that are not resident
     * @throws IllegalStateException  when they are not one F32 buffer of at least {@code nDims / 2} values
     */
    private GpuBuffer bindRopeFactors(GpuWeights weights, RopeSpec rope, int il) {
        if (rope == null || rope.freqFactors() == null) {
            return null;
        }
        GpuWeight w = weights.getOrNull(rope.freqFactors().name());
        if (w == null) {
            throw new MissingTensorException(weights.archName(), il, rope.freqFactors().name());
        }
        int pairs = RopeMath.pairCount(rope.type(), rope.nDims());
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() * w.cols() < pairs) {
            throw new IllegalStateException(layout.where + ": " + w.name() + " must be one F32 buffer of at least "
                    + pairs + " frequency factors, got " + w);
        }
        return w.shard(0);
    }

    /** The SSMax per-head query dispatch over the assembled {@code qcur} (mistral-large4.cpp:181-190, 381). */
    private void prepareSsmax(LayerSpec layer, GpuWeights weights) {
        DsaLayout l = layout;
        if (!l.ssmax) {
            return;
        }
        ssmaxScale = vector(weights.layer(layer, DsaLayout.SSMAX), l.nHead).shard(0);
        ssmaxBias = vector(weights.layer(layer, DsaLayout.SSMAX_B), l.nHead).shard(0);
        querySsmax = prepare(AttnExtKernels.SSMAX_SCALE,
                AttnExtKernels.ssmaxGlobal(ubatch, l.nHead, l.dK, local), new long[]{local},
                AttnExtKernels.ssmaxArgs(ubatch, l.nHead, l.dK, l.dK, l.nHead * l.dK, 0, ssmaxScale, ssmaxBias,
                        steps.ropePos(), qcur.buffer()));
    }

    /** The per-head absorption {@code q_nope . W_k_b[h]} and decompression {@code W_v_b[h] . latent}. */
    private void prepareHeadProducts(GpuWeight wkB, GpuWeight wvB, ScratchSlot slots) {
        DsaLayout l = layout;
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        int pairs = ubatch * l.nHead;
        absorb = prepare(MoeKernels.matVecId(wkB.type()), MoeKernels.idMatVecGlobal(l.kvLora, pairs, idLocal),
                new long[]{idLocal, 1}, MoeKernels.matVecIdArgs(l.kvLora, l.nope, pairs, l.headK, 0, 0, false, 0,
                        l.nHead, slots.buffer(), wkB.shard(0), q.buffer(), ph, latent.buffer()));
        decompress = prepare(MoeKernels.matVecId(wvB.type()), MoeKernels.idMatVecGlobal(l.headV, pairs, idLocal),
                new long[]{idLocal, 1}, MoeKernels.matVecIdArgs(l.headV, l.kvLora, pairs, l.kvLora, 0, 0, false, 0,
                        l.nHead, slots.buffer(), wvB.shard(0), latent.buffer(), ph, q.buffer()));
    }

    /** The sparse MLA attention over the selected latent rows, for decode and for prefill. */
    private void prepareSparse(LayerSpec layer, GpuWeights weights, GpuKvLayer cache, ScratchSlot topK) {
        DsaLayout l = layout;
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        GpuBuffer sinks = l.sinks ? vector(weights.layer(layer, DsaLayout.SINKS), l.nHead).shard(0) : ph;
        int hasSinks = l.sinks ? 1 : 0;
        KvDType dtype = cache.layout().dtypeK();
        sparseDecode = prepare(SparseKernels.sparseDecode(dtype),
                SparseKernels.sparseDecodeGlobal(l.nHead, sparseLocal), SparseKernels.local(sparseLocal),
                new Object[]{l.nHead, l.kvLora, l.rope, l.topK, l.kqScale, l.softcap, hasSinks, 0, qcur.buffer(),
                        cache.k(), topK.buffer(), sinks, latent.buffer()});
        sparsePrefill = prepare(SparseKernels.sparsePrefill(dtype),
                SparseKernels.sparsePrefillGlobal(ubatch, l.nHead, sparseLocal), SparseKernels.local(sparseLocal),
                new Object[]{ubatch, l.nHead, l.kvLora, l.rope, l.topK, l.kqScale, l.softcap, hasSinks,
                        l.nHead * l.dK, qcur.buffer(), cache.k(), topK.buffer(), sinks, latent.buffer()});
    }

    /** The LoRA delta sites of the dense projections. */
    private void bindDeltas(LayerSpec layer) {
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        if (layout.qLora > 0) {
            qaDeltas = deltas.serve(layer, DsaLayout.WQ_A, wqA);
            qbDeltas = deltas.serve(layer, DsaLayout.WQ_B, wqB);
        } else {
            qDeltas = deltas.serve(layer, DsaLayout.WQ, wq);
        }
        kvaDeltas = layout.kvShared ? null : deltas.serve(layer, DsaLayout.WKV_A, wkvA);
        outDeltas = deltas.serve(layer, DsaLayout.WO, wo);
        gateDeltas = gateW != null ? deltas.serve(layer, DsaLayout.GATE, gateW) : null;
    }

    /** The latent cache of the layer, checked against the layout and the sparse kernels. */
    private GpuKvLayer latentCache(GpuMemory memory, int il) {
        DsaLayout l = layout;
        GpuKvLayer cache = memory.kv(il);
        if (cache == null) {
            throw new IllegalStateException(l.where + ": no KV cache for the DSA layer");
        }
        KvLayout kvl = cache.layout();
        if (!kvl.vAliasesK() || kvl.widthK() != l.dK) {
            throw new IllegalStateException(l.where + ": the DSA layer needs a latent cache of " + l.dK
                    + "-value rows with V aliasing K, got " + kvl);
        }
        SparseKernels.checkLatentCache(kvl.dtypeK(), l.kvLora, l.rope);
        return cache;
    }

    /**
     * {@code attn_output}, carrying its {@code attn_output.scale} as the projection's output scale when the layer
     * has one: the scalar is read back once here (llama-graph.cpp:1643-1645).
     */
    private GpuWeight outputWeight(LayerSpec layer, GpuWeights weights) {
        GpuWeight w = weights.layer(layer, DsaLayout.WO);
        if (!layout.woScale || w.hasScale()) {
            return w;
        }
        GpuWeight s = vector(weights.layer(layer, DsaLayout.WO_SCALE), 1);
        LayerWeightSource source = ctx.weightSource();
        source.prefetch(layer.weightLayer());
        source.await(layer.weightLayer());
        float[] host = new float[1];
        ComputeQueue queue = ctx.queue();
        closeEvent(queue.enqueueReadBuffer(s.shard(0), 0, Float.BYTES, host));
        queue.finish();
        if (!Float.isFinite(host[0])) {
            throw new IllegalStateException(layout.where + ": " + s.name() + " = " + host[0] + " is not finite");
        }
        return w.withScale(host[0]);
    }

    private void bindGate(LayerSpec layer, GpuWeights weights) {
        DsaLayout l = layout;
        if (l.gate == DsaLayout.Gate.NONE) {
            return;
        }
        gateW = projection(weights.layer(layer, DsaLayout.GATE));
        gate = temp(TEMP_GATE, l.gateWidth());
        GpuBuffer attn = q.buffer();
        if (l.gate == DsaLayout.Gate.PER_HEAD) {
            gateWide = temp(TEMP_GATE_WIDE, (long) l.nHead * l.headV);
            int rows = ubatch * l.nHead;
            gateGlu = prepare(ActKernels.GLU, ActKernels.global1d((long) rows * l.headV, local), new long[]{local},
                    ActKernels.gluArgs(GluOp.SIGMUL, ActParams.NONE, rows, l.headV, gateWide.buffer(), 0, l.headV,
                            attn, 0, l.headV, attn, 0, l.headV));
        } else {
            int width = l.gateWidth();
            gateGlu = prepare(ActKernels.GLU, ActKernels.global1d((long) ubatch * width, local), new long[]{local},
                    ActKernels.gluArgs(GluOp.SIGMUL, ActParams.NONE, ubatch, width, gate.buffer(), 0, width,
                            attn, 0, width, attn, 0, width));
        }
    }

    private void bindIndexer(LayerSpec layer, GpuWeights weights, GpuMemory memory, GpuKvLayer cache,
                             ScratchSlot topK) {
        DsaLayout l = layout;
        GpuIndexerCache idx = memory.indexer(layer.il());
        if (idx == null) {
            throw new IllegalStateException(l.where + ": no indexer cache for the full DSA layer");
        }
        if (idx.group() != cache.group() || idx.plan().rows() != kvRows || idx.plan().width() != l.idxDim) {
            throw new IllegalStateException(l.where + ": the indexer cache " + idx.plan() + " (group " + idx.group()
                    + ") must hold one " + l.idxDim + "-value row per latent row of group " + cache.group()
                    + ", whose rows the top-k indices address");
        }
        IndexerConfig cfg = IndexerConfig.forLayer(layer, idx.plan().rotation(), cache.layout().window(), l.normEps);
        indexer = new GpuDsaIndexer(ctx, l.where, cfg);
        RopeSpec rope = cfg.rope();
        GpuBuffer factors = bindRopeFactors(weights, rope, layer.il());
        indexer.bindQueries(weights.layer(layer, DsaLayout.IDX_Q_B), weights.layer(layer, DsaLayout.IDX_PROJ),
                steps.ropePos(), factors);
        if (l.indexerQueryNorm) {
            indexer.bindQueryNorm(weights.layer(layer, DsaLayout.IDX_Q_NORM),
                    weights.layer(layer, DsaLayout.IDX_Q_NORM_B));
        }
        if (l.indexerSsmax) {
            indexer.bindSsmax(vector(weights.layer(layer, DsaLayout.IDX_SSMAX), l.idxHeads),
                    vector(weights.layer(layer, DsaLayout.IDX_SSMAX_B), l.idxHeads), steps.ropePos());
        }
        indexer.bindKeys(weights.layer(layer, DsaLayout.IDX_K),
                l.idxKeyNorm ? weights.layer(layer, DsaLayout.IDX_K_NORM) : null,
                l.idxKeyBias ? weights.layer(layer, DsaLayout.IDX_K_NORM_B) : null, steps.ropePos(),
                steps.cellIdx(idx.group()), idx.keys(), idx.plan().dtype(), kvRows);
        indexer.bindSelection(idx.keys(), idx.plan().dtype(), kvRows, steps.kvPos(idx.group()), steps.qPos(), topK);
        indexer.bindDeltas(layer);
        indexerKeys = idx.keys();
        if (steps.pageShift() >= 0) {
            indexer.enableSegments();
        }
    }

    /** A shared layer's selection addresses the source layer's rows: both caches must use the same rows. */
    private void checkSource(GpuMemory memory, GpuKvLayer cache) {
        DsaLayout l = layout;
        GpuKvLayer src = memory.kv(l.sourceLayer);
        if (src == null || src.group() != cache.group() || src.layout().rows() != kvRows) {
            throw new IllegalStateException(l.where + ": the shared DSA layer reuses the top-k of layer "
                    + l.sourceLayer + ", whose cache must use the same KV group and rows (group " + cache.group()
                    + ", " + kvRows + " rows), got " + (src == null ? "none" : "group " + src.group() + ", "
                    + src.layout().rows() + " rows"));
        }
    }

    // ------------------------------------------------------------------ steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        forward(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        forward(pos0, nTokens, in, out);
    }

    /**
     * One step over the rows of {@code s}: the contiguous step when no paged step is current, else the paged step
     * through the memory's current {@code PagedStep} (see the class description). A non-causal step
     * ({@code GpuStepInputs.stepNonCausal}) selects without the causal mask.
     *
     * @throws UnsupportedOperationException when the step is not contiguous and the memory holds no paged step
     */
    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        Objects.requireNonNull(s, "s");
        PagedStep step = steps.currentPagedStep();
        if (step == null) {
            GpuMixer.super.batch(s, in, out);
            return;
        }
        forwardPaged(s, in, out);
    }

    @Override
    public String batchRefusal() {
        return "the DSA mixer mixes several sequences per step only over paged memory (a currentPagedStep)";
    }

    private void forward(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        DsaLayout l = layout;
        checkStep(pos0, n, in, out);
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        int nKv = Math.min(kvRows, pos0 + n);

        // The query, latent and gate projections and the indexer of a single-token step all read this row,
        // so its Q8_1 blocks are quantized once.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(xOff, l.nEmbd)
                ? ops.shareQ8Activations(x, xOff, l.nEmbd) : null;
        try {
            if (l.qLora > 0) {
                projectLowRankQuery(n, x, xOff);
            }
            int k;
            if (indexer != null) {
                indexer.queries(n, qr != null ? qr.buffer() : x, 0, x, xOff);
                indexer.storeKeys(n, x, xOff);
                k = indexer.select(n, nKv);
            } else {
                k = SparseKernels.selectedCount(nKv, l.topK);
            }
            attendAndProject(n, k, x, xOff, out);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    /**
     * The paged step: the token-wise stages and stores over the step's rows (positions and cells from the paged step
     * inputs), then every segment's selection among its sequence's cells into its rows of the top-k buffer.
     */
    private void forwardPaged(StepLayout s, ScratchSlot in, ScratchSlot out) {
        DsaLayout l = layout;
        int n = s.nTokens();
        if (n < 1 || n > ubatch || steps.nTokens() != n) {
            throw new IllegalArgumentException(l.where + ": paged DSA step of " + n + " tokens, the step inputs hold "
                    + steps.nTokens() + " within [1, " + ubatch + "]");
        }
        long width = (long) n * l.nEmbd;
        Objects.requireNonNull(in, "in").require(width);
        Objects.requireNonNull(out, "out").require(width);
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        // The query, latent and gate projections and the indexer of a single-token step all read this row,
        // so its Q8_1 blocks are quantized once.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(xOff, l.nEmbd)
                ? ops.shareQ8Activations(x, xOff, l.nEmbd) : null;
        try {
            if (l.qLora > 0) {
                projectLowRankQuery(n, x, xOff);
            }
            int k = SparseKernels.selectedCount(kvRows, l.topK);
            if (indexer != null) {
                indexer.queries(n, qr != null ? qr.buffer() : x, 0, x, xOff);
                indexer.storeKeys(n, x, xOff);
                boolean causal = indexer.config().mask().causal() && !steps.stepNonCausal();
                for (int seg = 0; seg < s.nSeg(); seg++) {
                    int t0 = s.tokStart(seg);
                    indexer.selectRows(t0, s.len(seg), kvRows, indexerKeys, steps.sequenceKvPos(group, s.seq(seg)),
                            causal, selection, (long) t0 * k);
                }
            }
            attendAndProject(n, k, x, xOff, out);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    /** {@code qr = RMS(W_q_a . x) * attn_q_a_norm}. */
    private void projectLowRankQuery(int n, GpuBuffer x, int xOff) {
        ops.matmul(wqA, x, xOff, qr.buffer(), 0, n);
        qaDeltas.applyRows(n, x, xOff, qr.buffer(), 0);
        rowNorm(qANorm, n);
    }

    /**
     * Q and the latent rows (stored at the step's cells), the absorbed queries, the sparse attention over the
     * {@code k} selected rows of every token, the decompression, the gate and {@code W_o}.
     */
    private void attendAndProject(int n, int k, GpuBuffer x, int xOff, ScratchSlot out) {
        DsaLayout l = layout;
        if (l.qLora > 0) {
            ops.matmul(wqB, qr.buffer(), 0, q.buffer(), 0, n);
            qbDeltas.applyRows(n, qr.buffer(), 0, q.buffer(), 0);
        } else {
            ops.matmul(wq, x, xOff, q.buffer(), 0, n);
            qDeltas.applyRows(n, x, xOff, q.buffer(), 0);
        }
        if (!l.kvShared) {
            // a shared-KV layer reads the source layer's rows and must not overwrite them (mistral-large4.cpp has_kv)
            if (l.ropeApplied) {
                ops.matmul(wkvA, x, xOff, kv.buffer(), 0, n);
                kvaDeltas.applyRows(n, x, xOff, kv.buffer(), 0);
                if (kPeNorm != null) {
                    rowNorm(kPeNorm, n);
                }
                rope(qRope, n, l.nHead);
                rope(kRope, n, 1);
            } else {
                // a NoPE layer projects the latent only; its roped cache columns stay zero (mistral-large4.cpp:415-416)
                ops.matmul(wkvA, x, xOff, kvProj.buffer(), 0, n);
                kvaDeltas.applyRows(n, x, xOff, kvProj.buffer(), 0);
                ops.fill(kv.buffer(), 0, (long) n * l.dK * Float.BYTES, 0);
                ops.copy2d(kvProj.buffer(), 0, l.kvLora, kv.buffer(), 0, l.dK, n, l.kvLora);
            }
            rowNorm(kvANorm, n);
            latentStore.setInt(RopeKernels.ARG_N_TOKENS, n);
            latentStore.setGlobal(elementGlobal(RopeKernels.kvStoreWorkItems(latentType, n, l.dK)));
            latentStore.dispatch();
        } else if (qRope != null) {
            rope(qRope, n, l.nHead);
        }

        int pairs = n * l.nHead;
        GpuMatmulTap.heads(ops.queue(), wkB, l.nHead, q.buffer(), 0, l.headK, l.nope, n);
        idMatVec(absorb, n, l.kvLora, l.headK);
        if (l.ropeApplied) {
            ops.copy2d(latent.buffer(), 0, l.kvLora, qcur.buffer(), 0, l.dK, pairs, l.kvLora);
            ops.copy2d(q.buffer(), l.nope, l.headK, qcur.buffer(), l.kvLora, l.dK, pairs, l.rope);
        } else {
            ops.fill(qcur.buffer(), 0, (long) pairs * l.dK * Float.BYTES, 0);
            ops.copy2d(latent.buffer(), 0, l.kvLora, qcur.buffer(), 0, l.dK, pairs, l.kvLora);
        }
        if (querySsmax != null) {
            querySsmax.setInt(AttnExtKernels.ARG_SSMAX_N_TOKENS, n);
            querySsmax.setGlobal(AttnExtKernels.ssmaxGlobal(n, l.nHead, l.dK, local));
            querySsmax.dispatch();
        }

        if (n == 1) {
            sparseDecode.setInt(SparseKernels.ARG_DECODE_N_SEL, k);
            sparseDecode.dispatch();
        } else {
            sparsePrefill.setInt(SparseKernels.ARG_PREFILL_N_TOKENS, n);
            sparsePrefill.setInt(SparseKernels.ARG_PREFILL_N_SEL, k);
            sparsePrefill.setGlobal(SparseKernels.sparsePrefillGlobal(n, l.nHead, sparseLocal));
            sparsePrefill.dispatch();
        }
        GpuMatmulTap.heads(ops.queue(), wvB, l.nHead, latent.buffer(), 0, l.kvLora, l.kvLora, n);
        idMatVec(decompress, n, l.headV, l.kvLora);

        if (gateGlu != null) {
            applyGate(n, x, xOff);
        }
        ops.matmul(wo, q.buffer(), 0, out.buffer(), out.offset(), n);
        outDeltas.applyRows(n, q.buffer(), 0, out.buffer(), out.offset());
    }

    /**
     * The sigmoid output gate on {@code attn [t][h][head_v]}: element-wise (hy-v4.cpp:486-488) or one value per head
     * broadcast over the head (dots3note.cpp:386-397). The head gate is broadcast by placing it in column 0 of
     * {@value #TEMP_GATE_WIDE} and doubling the filled columns with strided copies.
     */
    private void applyGate(int n, GpuBuffer x, int xOff) {
        DsaLayout l = layout;
        ops.matmul(gateW, x, xOff, gate.buffer(), 0, n);
        gateDeltas.applyRows(n, x, xOff, gate.buffer(), 0);
        int rows;
        int cols;
        if (l.gate == DsaLayout.Gate.PER_HEAD) {
            rows = n * l.nHead;
            cols = l.headV;
            GpuBuffer wide = gateWide.buffer();
            ops.copy2d(gate.buffer(), 0, 1, wide, 0, cols, rows, 1);
            for (int filled = 1; filled < cols; filled *= 2) {
                ops.copy2d(wide, 0, cols, wide, filled, cols, rows, Math.min(filled, cols - filled));
            }
        } else {
            rows = n;
            cols = l.gateWidth();
        }
        gateGlu.setInt(ActKernels.ARG_GLU_N_ROWS, rows);
        gateGlu.setGlobal(ActKernels.global1d((long) rows * cols, local));
        gateGlu.dispatch();
    }

    /**
     * A per-head {@code {T}_MatVecId} over the {@code n * n_head} (token, head) pairs of the step, in calls of at
     * most {@link #idChunkTokens} tokens so the pair dimension of the grid stays within the device's work-group
     * count (CUDA and Vulkan: 65535). Every call starts on a token boundary, so pair {@code s} of a call reads head
     * {@code s % n_head} of the constant slot table; the input and output offsets select the call's pairs.
     */
    private void idMatVec(PreparedDispatch d, int n, int outDim, int xStride) {
        int nHead = layout.nHead;
        for (int t0 = 0; t0 < n; t0 += idChunkTokens) {
            int sel = Math.min(idChunkTokens, n - t0) * nHead;
            int pair0 = t0 * nHead;
            d.setInt(MoeKernels.ARG_ID_N_SEL, sel);
            d.setInt(MoeKernels.ARG_ID_IN_OFF, pair0 * xStride);
            d.setInt(MoeKernels.ARG_ID_OUT_OFF, pair0 * outDim);
            d.setGlobal(MoeKernels.idMatVecGlobal(outDim, sel, idLocal));
            d.dispatch();
        }
    }

    /**
     * Whole tokens per {@code {T}_MatVecId} call whose {@code n_head} pairs each fit a grid of at most
     * {@code maxGroupsY} work-groups in its second dimension.
     *
     * @throws IllegalStateException when even one token's heads exceed the limit
     */
    private int idChunkTokens(long maxGroupsY) {
        long tokens = Math.min(Integer.MAX_VALUE, maxGroupsY) / layout.nHead;
        if (tokens < 1) {
            throw new IllegalStateException(layout.where + ": " + layout.nHead + " MLA heads exceed the device's "
                    + maxGroupsY + " work-groups in the second grid dimension");
        }
        return (int) Math.min(tokens, ubatch);
    }

    private void rowNorm(PreparedDispatch d, int n) {
        d.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, n);
        d.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(n), local));
        d.dispatch();
    }

    private void rope(PreparedDispatch d, int n, int heads) {
        d.setInt(RopeKernels.ARG_N_TOKENS, n);
        d.setInt(RopeKernels.ARG_POS_STRIDE, n);
        d.setGlobal(ropeGlobal(layout.ropeSpec, n, heads));
        d.dispatch();
    }

    private void checkStep(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        if (n < 1 || n > ubatch || pos0 < 0) {
            throw new IllegalArgumentException(layout.where + ": DSA step of " + n + " tokens at " + pos0
                    + " is outside [1, " + ubatch + "] tokens at a non-negative position");
        }
        if (steps.pos0() != pos0 || steps.nTokens() != n) {
            throw new IllegalStateException(layout.where + ": the step inputs hold " + steps.nTokens()
                    + " tokens at " + steps.pos0() + ", the DSA step is " + n + " tokens at " + pos0);
        }
        long width = (long) n * layout.nEmbd;
        Objects.requireNonNull(in, "in").require(width);
        Objects.requireNonNull(out, "out").require(width);
    }

    // ------------------------------------------------------------------ bind helpers

    private PreparedDispatch prepareNorm(GpuWeight w, GpuBuffer x, int rowLen, int rowStride, int off) {
        return prepare(NormKernels.ROW_NORM, NormKernels.reduceGlobal(NormKernels.rowNormGroups(ubatch), local),
                new long[]{local}, NormKernels.rowNormArgs(NormKind.RMS, layout.rmsEps, ubatch, rowLen, rowStride,
                        x, off, x, off, w.shard(0), null, 1, ctx.placeholders().readOnlyFloats()));
    }

    /**
     * {@value #TEMP_HEAD_SLOTS}{@code <n_head>}: {@code slots[s] = s % n_head} for {@code s < ubatch * n_head}, the
     * per-head weight each (token, head) pair of the id-matvecs reads. Written once here; every holder of the name
     * writes the same values.
     */
    private ScratchSlot headSlots() {
        int nHead = layout.nHead;
        int n = ubatch * nHead;
        ScratchSlot s = temp(TEMP_HEAD_SLOTS + nHead, nHead);
        ByteBuffer bytes = ByteBuffer.allocate(n * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            bytes.putInt(i % nHead);
        }
        closeEvent(ctx.queue().enqueueWriteBuffer(s.buffer(), 0, (long) n * Integer.BYTES, bytes.array()));
        return s;
    }

    private GpuWeight projection(GpuWeight w) {
        if (!GpuOps.supportsWeightType(w.type())) {
            throw new GpuModelUnsupportedException(layout.where + ": no GPU matvec kernels for " + w.name()
                    + " of type " + w.type());
        }
        return w;
    }

    /**
     * A per-head tensor ({@code attn_k_b}, {@code attn_v_b}) for the id-matvec: one buffer, a type with id-matvec
     * kernels, and no PRISM transform, input transform or output scale, which llama.cpp's plain {@code ggml_mul_mat}
     * of these tensors would not apply either (deepseek32.cpp:363, llama-graph.cpp build_attn_mha {@code v_mla}).
     */
    private GpuWeight perHead(GpuWeight w) {
        if (!MoeKernels.supports(w.type())) {
            throw new UnsupportedOperationException(layout.where + ": no id-matvec kernels for " + w.name()
                    + " of type " + w.type());
        }
        if (!w.isSingleShard()) {
            throw new UnsupportedOperationException(layout.where + ": per-head tensor " + w.name() + " is split into "
                    + w.shardCount() + " row shards; the id-matvec reads every head from one buffer");
        }
        if (w.hadamard() != null || w.hasScale() || (w.inputTransform() != null && !w.inputTransform().isIdentity())) {
            throw new UnsupportedOperationException(layout.where + ": per-head tensor " + w.name()
                    + " carries an input transform or output scale, which the absorbed MLA products do not apply");
        }
        return w;
    }

    private GpuWeight vector(GpuWeight v, int n) {
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || v.cols() * v.rows() != n) {
            throw new IllegalStateException(layout.where + ": " + v.name() + " must be a single-buffer F32 vector of "
                    + n + " values, got " + v);
        }
        return v;
    }

    /** A temporary of {@code ubatch} rows of {@code width} floats, starting at element 0 of its buffer. */
    private ScratchSlot temp(String name, long width) {
        ScratchSlot s = ctx.scratch().temp(name, (long) ubatch * width);
        if (s.offset() != 0) {
            throw new IllegalStateException(layout.where + ": DSA buffer " + s + " must start at element 0");
        }
        return s;
    }

    private long[] ropeGlobal(RopeSpec r, int n, int heads) {
        return elementGlobal(RopeKernels.ropeExtWorkItems(r.type(), n, heads, r.nDims()));
    }

    private long[] elementGlobal(long items) {
        return new long[]{RopeKernels.roundUp(Math.max(1L, items), local)};
    }

    private PreparedDispatch prepare(String kernel, long[] global, long[] localSize, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, localSize, args);
        prepared.add(d);
        return d;
    }

    private static long[] shape(GpuWeight w) {
        return w == null ? null : new long[]{w.cols(), w.rows()};
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("DSA mixer is closed");
        }
        if (ctx == null) {
            throw new IllegalStateException("DSA mixer is not bound");
        }
    }

    // ------------------------------------------------------------------ close

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        releaseAll(null);
    }

    private void releaseAll(RuntimeException cause) {
        RuntimeException failure = null;
        if (indexer != null) {
            try {
                indexer.close();
            } catch (RuntimeException e) {
                failure = e;
            }
            indexer = null;
        }
        failure = Closeables.closeAll(prepared, failure);
        prepared.clear();
        if (failure != null) {
            if (cause != null) {
                cause.addSuppressed(failure);
            } else {
                throw failure;
            }
        }
    }

    @Override
    public String toString() {
        return "GpuDsaMlaMixer{" + (layout != null ? layout.toString() : "unbound") + '}';
    }
}
