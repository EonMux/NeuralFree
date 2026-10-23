package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
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
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.DsaLayout;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.IndexerConfig;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
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

    private GpuWeight wqA;
    private GpuWeight wqB;
    private GpuWeight wkvA;
    private GpuWeight wo;
    private GpuWeight gateW;
    /**
     * LoRA deltas of the dense projections, after the product and its scale ({@code build_lora_mm},
     * llama-graph.cpp:1629-1667); the per-head absorbed {@code attn_k_b} / {@code attn_v_b} take none.
     */
    private GpuProjectionDeltas.Site qaDeltas;
    private GpuProjectionDeltas.Site qbDeltas;
    private GpuProjectionDeltas.Site kvaDeltas;
    private GpuProjectionDeltas.Site outDeltas;
    private GpuProjectionDeltas.Site gateDeltas;

    private ScratchSlot qr;
    private ScratchSlot q;
    private ScratchSlot kv;
    private ScratchSlot qcur;
    private ScratchSlot latent;
    private ScratchSlot gate;
    private ScratchSlot gateWide;

    private GpuDsaIndexer indexer;

    private PreparedDispatch qANorm;
    private PreparedDispatch kPeNorm;
    private PreparedDispatch kvANorm;
    private PreparedDispatch qRope;
    private PreparedDispatch kRope;
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

        wqA = projection(weights.layer(layer, DsaLayout.WQ_A));
        wqB = projection(weights.layer(layer, DsaLayout.WQ_B));
        wkvA = projection(weights.layer(layer, DsaLayout.WKV_A));
        wo = projection(outputWeight(layer, weights));
        GpuWeight wkB = perHead(weights.layer(layer, DsaLayout.WK_B));
        GpuWeight wvB = perHead(weights.layer(layer, DsaLayout.WV_B));
        checkWorkGroups(wkB, wvB);

        qr = temp(TEMP_QR, l.qLora);
        q = temp(TEMP_Q, l.qBufferWidth());
        kv = temp(TEMP_KV, l.dK);
        qcur = temp(TEMP_QCUR, (long) l.nHead * l.dK);
        latent = temp(TEMP_LATENT, (long) l.nHead * l.kvLora);
        ScratchSlot slots = headSlots();
        ScratchSlot topK = temp(TEMP_TOPK, l.topK);

        prepareProjections(layer, weights, cache);
        prepareHeadProducts(wkB, wvB, slots);
        prepareSparse(layer, weights, cache, topK);
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
        qANorm = prepareNorm(vector(weights.layer(layer, DsaLayout.Q_A_NORM), l.qLora), qr.buffer(), l.qLora,
                l.qLora, 0);
        if (l.kPeNorm) {
            kPeNorm = prepareNorm(vector(weights.layer(layer, DsaLayout.K_PE_NORM), l.rope), kv.buffer(), l.rope,
                    l.dK, l.kvLora);
        }
        kvANorm = prepareNorm(vector(weights.layer(layer, DsaLayout.KV_A_NORM), l.kvLora), kv.buffer(), l.kvLora,
                l.dK, 0);
        RopeSpec r = l.ropeSpec;
        GpuBuffer ropePos = steps.ropePos();
        qRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(r, ubatch, l.nHead), new long[]{local},
                RopeKernels.ropeExtArgs(r, ubatch, l.nHead, l.headK, l.nHead * l.headK, l.nope, ubatch, 0, false,
                        ropePos, ph, q.buffer()));
        kRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(r, ubatch, 1), new long[]{local},
                RopeKernels.ropeExtArgs(r, ubatch, 1, l.dK, l.dK, l.kvLora, ubatch, 0, false, ropePos, ph,
                        kv.buffer()));
        KvDType dtype = cache.layout().dtypeK();
        latentType = dtype;
        latentStore = prepare(RopeKernels.kvStore(dtype),
                elementGlobal(RopeKernels.kvStoreWorkItems(dtype, ubatch, l.dK)), new long[]{local},
                RopeKernels.kvStoreArgs(dtype, ubatch, l.dK, l.dK, 0, cache.layout().widthK(), kv.buffer(),
                        steps.cellIdx(cache.group()), cache.k()));
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
        qaDeltas = deltas.serve(layer, DsaLayout.WQ_A, wqA);
        qbDeltas = deltas.serve(layer, DsaLayout.WQ_B, wqB);
        kvaDeltas = deltas.serve(layer, DsaLayout.WKV_A, wkvA);
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
        indexer.bindQueries(weights.layer(layer, DsaLayout.IDX_Q_B), weights.layer(layer, DsaLayout.IDX_PROJ),
                steps.ropePos());
        indexer.bindKeys(weights.layer(layer, DsaLayout.IDX_K),
                l.idxKeyNorm ? weights.layer(layer, DsaLayout.IDX_K_NORM) : null,
                l.idxKeyBias ? weights.layer(layer, DsaLayout.IDX_K_NORM_B) : null, steps.ropePos(),
                steps.cellIdx(idx.group()), idx.keys(), idx.plan().dtype(), kvRows);
        indexer.bindSelection(idx.keys(), idx.plan().dtype(), kvRows, steps.kvPos(idx.group()), steps.qPos(), topK);
        indexer.bindDeltas(layer);
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

    private void forward(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        DsaLayout l = layout;
        checkStep(pos0, n, in, out);
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        int nKv = Math.min(kvRows, pos0 + n);

        ops.matmul(wqA, x, xOff, qr.buffer(), 0, n);
        qaDeltas.applyRows(n, x, xOff, qr.buffer(), 0);
        rowNorm(qANorm, n);

        int k;
        if (indexer != null) {
            indexer.queries(n, qr.buffer(), 0, x, xOff);
            indexer.storeKeys(n, x, xOff);
            k = indexer.select(n, nKv);
        } else {
            k = SparseKernels.selectedCount(nKv, l.topK);
        }

        ops.matmul(wqB, qr.buffer(), 0, q.buffer(), 0, n);
        qbDeltas.applyRows(n, qr.buffer(), 0, q.buffer(), 0);
        ops.matmul(wkvA, x, xOff, kv.buffer(), 0, n);
        kvaDeltas.applyRows(n, x, xOff, kv.buffer(), 0);
        if (kPeNorm != null) {
            rowNorm(kPeNorm, n);
        }
        rope(qRope, n, l.nHead);
        rope(kRope, n, 1);
        rowNorm(kvANorm, n);
        latentStore.setInt(RopeKernels.ARG_N_TOKENS, n);
        latentStore.setGlobal(elementGlobal(RopeKernels.kvStoreWorkItems(latentType, n, l.dK)));
        latentStore.dispatch();

        int pairs = n * l.nHead;
        idMatVec(absorb, n, l.kvLora, l.headK);
        ops.copy2d(latent.buffer(), 0, l.kvLora, qcur.buffer(), 0, l.dK, pairs, l.kvLora);
        ops.copy2d(q.buffer(), l.nope, l.headK, qcur.buffer(), l.kvLora, l.dK, pairs, l.rope);

        if (n == 1) {
            sparseDecode.setInt(SparseKernels.ARG_DECODE_N_SEL, k);
            sparseDecode.dispatch();
        } else {
            sparsePrefill.setInt(SparseKernels.ARG_PREFILL_N_TOKENS, n);
            sparsePrefill.setInt(SparseKernels.ARG_PREFILL_N_SEL, k);
            sparsePrefill.setGlobal(SparseKernels.sparsePrefillGlobal(n, l.nHead, sparseLocal));
            sparsePrefill.dispatch();
        }
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
            throw new UnsupportedOperationException(layout.where + ": no GPU matvec kernels for " + w.name()
                    + " of type " + w.type());
        }
        return w;
    }

    /**
     * A per-head tensor ({@code attn_k_b}, {@code attn_v_b}) for the id-matvec: one buffer, a type with id-matvec
     * kernels, and no PRISM transform or output scale, which llama.cpp's plain {@code ggml_mul_mat} of these tensors
     * would not apply either (deepseek32.cpp:363, llama-graph.cpp build_attn_mha {@code v_mla}).
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
        if (w.hadamard() != null || w.hasScale()) {
            throw new UnsupportedOperationException(layout.where + ": per-head tensor " + w.name()
                    + " carries a PRISM transform or output scale, which the absorbed MLA products do not apply");
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
