package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.mla.CpuMlaHeadProducts;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexer.RowReader;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.DsaLayout;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.IndexerConfig;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Arrays;
import java.util.Objects;

/**
 * CPU twin of {@link GpuDsaMlaMixer}: the same stages in the same order with the kernels' CPU math. Projections use
 * {@link CpuMatVec}, the per-head {@code attn_k_b} / {@code attn_v_b} products {@link CpuMlaHeadProducts} over every
 * (token, head) pair of a step in one parallel pass (the twin of {@code {T}_MatVecId}); norms {@link NormMath#rowNorm} ({@code LLM_RowNorm});
 * RoPE {@link CpuRope#apply} ({@code LLM_RoPE_Ext}); the latent store {@link KvAccessor#writeK} at
 * {@code layout.rowOf(pos)} ({@code LLM_KV_Store_<KV>}); the sparse attention {@link CpuSparseMla}
 * ({@code LLM_MLA_Sparse*}); the gate {@link CpuOps#glu} with {@code SIGMUL} ({@code LLM_GLU_F32}); the indexer
 * {@link CpuDsaIndexerStage}. Q comes from {@code wq_b . qr} or directly from {@code attn_q . x}; mistral-large4
 * additionally scales {@code qcur} by the per-head SSMax temperature and reads the learned {@code rope_freqs} factors,
 * and its NoPE layers zero-pad the rope dims of Q and of the latent row.
 *
 * <p>The top-k selection lives in the shared int temporary {@value GpuDsaMlaMixer#TEMP_TOPK} of
 * {@code ubatch * topK} entries, which a shared layer reads after its source layer wrote it in the same step.
 * Working arrays are {@link CpuScratch} temporaries taken at bind; a step allocates no per-token arrays. Not
 * thread-safe.</p>
 */
public final class CpuDsaMlaMixer implements CpuMixer {

    public static final String TEMP_POSITIONS = "dsa.pos";
    public static final String TEMP_SLOT_ROWS = "dsa.slot.rows";
    public static final String TEMP_KV_PROJ = "dsa.kvproj";

    private CpuForwardContext ctx;
    private CpuMatVec matVec;
    private DsaLayout layout;
    private int ubatch;
    private int kvRows;

    private CpuTensorRef wq;
    private CpuTensorRef wqA;
    private CpuTensorRef wqB;
    private CpuTensorRef wkvA;
    private CpuTensorRef wkB;
    private CpuTensorRef wvB;
    private CpuTensorRef wo;
    private CpuTensorRef gateW;
    private float woScale = 1.0f;
    /**
     * LoRA deltas of the dense projections, after the product and its scale ({@code build_lora_mm},
     * llama-graph.cpp:1629-1667); the per-head absorbed {@code attn_k_b} / {@code attn_v_b} take none.
     */
    private CpuProjectionDeltas.Site qDeltas;
    private CpuProjectionDeltas.Site qaDeltas;
    private CpuProjectionDeltas.Site qbDeltas;
    private CpuProjectionDeltas.Site kvaDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private CpuProjectionDeltas.Site gateDeltas;

    private float[] qANormW;
    private float[] kvANormW;
    private float[] kPeNormW;
    private float[] sinks;
    /** SSMax per-head scale and bias ({@code attn_ssmax}); null when the query is not scaled. */
    private float[] ssmaxW;
    private float[] ssmaxB;

    private KvAccessor cache;
    private RowReader latentRows;
    private RopeFreqTable ropeTable;

    private float[] qr;
    private float[] q;
    private float[] kv;
    private float[] kvProj;
    private float[] qcur;
    private float[] latent;
    private float[] gate;
    private float[] gateWide;
    private int[] topK;
    private int[] ropePos;
    /** The cells of one sequence's slots in a paged step ({@code PagedStep.seqCells}), one per cache row. */
    private int[] slotRows;

    private CpuDsaIndexerStage indexer;
    private boolean closed;

    /** Use {@link SparseBlocks#cpuMixers()}. */
    CpuDsaMlaMixer() {
    }

    /**
     * Binds the mixer: validates the layer's tensors against its spec ({@link DsaLayout}), takes the working arrays,
     * binds the indexer of a full layer.
     *
     * @throws IllegalStateException         when already bound or closed, the context has no memory, or the
     *                                       tensors or caches contradict the spec
     * @throws UnsupportedOperationException when a per-head tensor carries a transform the absorbed products do not
     *                                       apply
     */
    @Override
    public void bind(CpuForwardContext context, LayerSpec layer) {
        if (closed || ctx != null) {
            throw new IllegalStateException("DSA mixer is " + (closed ? "closed" : "already bound"));
        }
        ctx = Objects.requireNonNull(context, "context");
        try {
            bindLayer(Objects.requireNonNull(layer, "layer"));
        } catch (RuntimeException e) {
            ctx = null;
            throw e;
        }
    }

    private void bindLayer(LayerSpec layer) {
        matVec = ctx.matVec();
        ubatch = ctx.scratch().ubatch();
        CpuWeights weights = ctx.weights();
        layout = DsaLayout.of(layer, ctx.spec(), suffix -> shape(weights.layerOrNull(layer, suffix)));
        DsaLayout l = layout;
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(l.where + ": the DSA mixer needs the model's KV memory");
        }
        CpuMemory memory = ctx.memory();
        cache = latentCache(memory, layer.il());
        kvRows = cache.layout().rows();
        latentRows = RowReader.ofKeys(cache);
        if (l.kvShared) {
            KvAccessor src = memory.kv(l.kvSourceLayer);
            if (src == null || src.group() != cache.group() || src.layout().rows() != kvRows) {
                throw new IllegalStateException(l.where + ": reads the latent cache of layer " + l.kvSourceLayer
                        + ", which must use the layer's KV group and rows (group " + cache.group() + ", " + kvRows
                        + " rows)");
            }
        }

        wkvA = l.kvShared ? null : weights.layer(layer, DsaLayout.WKV_A);
        wkB = perHead(weights.layer(layer, DsaLayout.WK_B));
        wvB = perHead(weights.layer(layer, DsaLayout.WV_B));
        wo = weights.layer(layer, DsaLayout.WO);
        if (l.woScale && !matVec.transformsOf(wo).hasScale()) {
            woScale = vector(weights.layerFloatsOrNull(layer, DsaLayout.WO_SCALE), 1, DsaLayout.WO_SCALE)[0];
            if (!Float.isFinite(woScale)) {
                throw new IllegalStateException(l.where + ": " + DsaLayout.WO_SCALE + " = " + woScale
                        + " is not finite");
            }
        }
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        if (l.qLora > 0) {
            wqA = weights.layer(layer, DsaLayout.WQ_A);
            wqB = weights.layer(layer, DsaLayout.WQ_B);
            qaDeltas = deltas.serve(layer, DsaLayout.WQ_A, wqA, matVec);
            qbDeltas = deltas.serve(layer, DsaLayout.WQ_B, wqB, matVec);
            qANormW = vector(weights.layerFloatsOrNull(layer, DsaLayout.Q_A_NORM), l.qLora, DsaLayout.Q_A_NORM);
        } else {
            wq = weights.layer(layer, DsaLayout.WQ);
            qDeltas = deltas.serve(layer, DsaLayout.WQ, wq, matVec);
        }
        if (!l.kvShared) {
            kvaDeltas = deltas.serve(layer, DsaLayout.WKV_A, wkvA, matVec);
            kvANormW = vector(weights.layerFloatsOrNull(layer, DsaLayout.KV_A_NORM), l.kvLora, DsaLayout.KV_A_NORM);
        }
        outDeltas = deltas.serve(layer, DsaLayout.WO, wo, matVec);
        if (l.kPeNorm) {
            kPeNormW = vector(weights.layerFloatsOrNull(layer, DsaLayout.K_PE_NORM), l.rope, DsaLayout.K_PE_NORM);
        }
        if (l.sinks) {
            sinks = vector(weights.layerFloatsOrNull(layer, DsaLayout.SINKS), l.nHead, DsaLayout.SINKS);
        }
        if (l.ssmax) {
            ssmaxW = vector(weights.layerFloatsOrNull(layer, DsaLayout.SSMAX), l.nHead, DsaLayout.SSMAX);
            ssmaxB = vector(weights.layerFloatsOrNull(layer, DsaLayout.SSMAX_B), l.nHead, DsaLayout.SSMAX_B);
        }
        RopeSpec r = l.ropeSpec;
        if (r != null) {
            float[] factors = null;
            if (r.freqFactors() != null) {
                factors = weights.floats(r.freqFactors());
                if (factors == null) {
                    throw new MissingTensorException(weights.archName(), layer.il(), r.freqFactors().name());
                }
            }
            ropeTable = RopeFreqTable.of(r, factors);
        }

        CpuScratch s = ctx.scratch();
        if (l.qLora > 0) {
            qr = s.temp(GpuDsaMlaMixer.TEMP_QR, ubatch * l.qLora);
        }
        q = s.temp(GpuDsaMlaMixer.TEMP_Q, ubatch * l.qBufferWidth());
        kv = s.temp(GpuDsaMlaMixer.TEMP_KV, ubatch * l.dK);
        if (!l.ropeApplied) {
            kvProj = s.temp(TEMP_KV_PROJ, ubatch * l.kvLora);
        }
        qcur = s.temp(GpuDsaMlaMixer.TEMP_QCUR, ubatch * l.nHead * l.dK);
        latent = s.temp(GpuDsaMlaMixer.TEMP_LATENT, ubatch * l.nHead * l.kvLora);
        topK = s.intTemp(GpuDsaMlaMixer.TEMP_TOPK, ubatch * l.topK);
        if (r != null) {
            ropePos = s.intTemp(TEMP_POSITIONS, r.positionsPerToken() * ubatch);
        }
        slotRows = s.intTemp(TEMP_SLOT_ROWS, kvRows);
        if (l.gate != DsaLayout.Gate.NONE) {
            gateW = weights.layer(layer, DsaLayout.GATE);
            gateDeltas = ctx.projectionDeltas().serve(layer, DsaLayout.GATE, gateW, matVec);
            gate = s.temp(GpuDsaMlaMixer.TEMP_GATE, ubatch * l.gateWidth());
            if (l.gate == DsaLayout.Gate.PER_HEAD) {
                gateWide = s.temp(GpuDsaMlaMixer.TEMP_GATE_WIDE, ubatch * l.nHead * l.headV);
            }
        }

        if (l.full) {
            bindIndexer(layer, weights, memory);
        } else {
            KvAccessor src = memory.kv(l.sourceLayer);
            if (src == null || src.group() != cache.group() || src.layout().rows() != kvRows) {
                throw new IllegalStateException(l.where + ": the shared DSA layer reuses the top-k of layer "
                        + l.sourceLayer + ", whose cache must use the same KV group and rows (group "
                        + cache.group() + ", " + kvRows + " rows)");
            }
        }
    }

    private KvAccessor latentCache(CpuMemory memory, int il) {
        DsaLayout l = layout;
        KvAccessor c = memory.kv(il);
        if (c == null) {
            throw new IllegalStateException(l.where + ": no KV cache for the DSA layer");
        }
        KvLayout kvl = c.layout();
        if (!kvl.vAliasesK() || kvl.widthK() != l.dK || c.nHeadKv() != 1 || c.headDimK() != l.dK) {
            throw new IllegalStateException(l.where + ": the DSA layer needs a single-head latent cache of " + l.dK
                    + "-value rows with V aliasing K, got " + kvl);
        }
        SparseKernels.checkLatentCache(kvl.dtypeK(), l.kvLora, l.rope);
        return c;
    }

    private void bindIndexer(LayerSpec layer, CpuWeights weights, CpuMemory memory) {
        DsaLayout l = layout;
        KvAccessor idx = memory.indexer(layer.il());
        if (idx == null) {
            throw new IllegalStateException(l.where + ": no indexer cache for the full DSA layer");
        }
        if (idx.group() != cache.group() || idx.layout().rows() != kvRows) {
            throw new IllegalStateException(l.where + ": the indexer cache (group " + idx.group() + ", "
                    + idx.layout().rows() + " rows) must hold one row per latent row of group " + cache.group()
                    + " (" + kvRows + " rows)");
        }
        IndexerConfig cfg = IndexerConfig.forLayer(layer, memory.plan().layer(layer.il()).indexer().rotation(),
                cache.layout().window(), l.normEps);
        indexer = new CpuDsaIndexerStage(ctx, l.where, cfg);
        RopeSpec rope = cfg.rope();
        float[] factors = null;
        if (rope != null && rope.freqFactors() != null) {
            factors = weights.floats(rope.freqFactors());
            if (factors == null) {
                throw new MissingTensorException(weights.archName(), layer.il(), rope.freqFactors().name());
            }
        }
        indexer.bindQueries(weights.layer(layer, DsaLayout.IDX_Q_B), weights.layer(layer, DsaLayout.IDX_PROJ), factors);
        if (l.indexerQueryNorm) {
            indexer.bindQueryNorm(vector(weights.layerFloatsOrNull(layer, DsaLayout.IDX_Q_NORM), l.nEmbd,
                    DsaLayout.IDX_Q_NORM), vector(weights.layerFloatsOrNull(layer, DsaLayout.IDX_Q_NORM_B), l.nEmbd,
                    DsaLayout.IDX_Q_NORM_B));
        }
        if (l.indexerSsmax) {
            indexer.bindSsmax(vector(weights.layerFloatsOrNull(layer, DsaLayout.IDX_SSMAX), l.idxHeads,
                    DsaLayout.IDX_SSMAX), vector(weights.layerFloatsOrNull(layer, DsaLayout.IDX_SSMAX_B), l.idxHeads,
                    DsaLayout.IDX_SSMAX_B));
        }
        indexer.bindKeys(weights.layer(layer, DsaLayout.IDX_K),
                l.idxKeyNorm ? vector(weights.layerFloatsOrNull(layer, DsaLayout.IDX_K_NORM), l.idxDim,
                        DsaLayout.IDX_K_NORM) : null,
                l.idxKeyBias ? vector(weights.layerFloatsOrNull(layer, DsaLayout.IDX_K_NORM_B), l.idxDim,
                        DsaLayout.IDX_K_NORM_B) : null, idx);
        int group = idx.group();
        indexer.bindSelection(RowReader.ofKeys(idx), kvRows, row -> memory.positionOfRow(group, row));
        indexer.bindDeltas(layer);
    }

    // ------------------------------------------------------------------ steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        forward(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        forward(pos0, n, in, out);
    }

    /**
     * One step over the rows of {@code s}: the contiguous step when the memory runs no paged step, else the paged
     * step through the memory's {@code CpuMemory.currentPagedStep}. The token-wise stages run over the whole step;
     * the indexer keys and the latent rows go to the step's cells; every segment selects among its own sequence's
     * slots only (through the sequence's page table, {@link CpuDsaIndexerStage#selectSlots}) and attends the
     * selected cells, so a segment computes the outputs its rows would compute run one sequence at a time. A
     * non-causal step ({@code CpuForwardContext.stepNonCausal}) selects without the causal mask.
     */
    @Override
    public void batch(StepLayout s, float[] in, float[] out) {
        ensureBound();
        PagedStep step = ctx.memory().currentPagedStep();
        if (step == null) {
            CpuMixer.super.batch(s, in, out);
            return;
        }
        forwardPaged(s, step, in, out);
    }

    @Override
    public String batchRefusal() {
        return "the DSA mixer mixes several sequences per step only over paged memory (a currentPagedStep)";
    }

    private void forward(int pos0, int n, float[] x, float[] out) {
        ensureBound();
        DsaLayout l = layout;
        checkStep(pos0, n, x, out);
        int nKv = Math.min(kvRows, pos0 + n);

        if (l.qLora > 0) {
            projectLowRankQuery(n, x);
        }
        int k;
        if (indexer != null) {
            indexer.queries(pos0, n, qr, 0, x, 0);
            indexer.storeKeys(pos0, n, x, 0);
            k = indexer.select(pos0, n, nKv, topK, 0);
        } else {
            k = SparseKernels.selectedCount(nKv, l.topK);
        }
        RopeSpec r = l.ropeSpec;
        if (r != null) {
            RopePositions.fill(r.type(), r.textPositions(), pos0, n, ropePos);
        }
        projectQuery(n, x);
        applyQueryRope(n);
        if (!l.kvShared) {
            projectLatent(n, x);
            KvLayout kvl = cache.layout();
            for (int t = 0; t < n; t++) {
                cache.writeK(kvl.rowOf(pos0 + t), kv, t * l.dK);
            }
        }
        absorbQueries(n);
        applyQuerySsmax(0, n, null, pos0);
        attend(0, n, k);
        output(n, x, out);
    }

    /**
     * The paged step: the rows' positions come from the {@link CpuScratch#POS} slot (section 0), their indexer keys
     * and latent rows are stored at the step's cells, and every segment of {@code s} selects and attends over its
     * sequence's slots.
     */
    private void forwardPaged(StepLayout s, PagedStep step, float[] x, float[] out) {
        DsaLayout l = layout;
        int n = s.nTokens();
        checkStep(0, n, x, out);
        int[] pos = ctx.scratch().ints(CpuScratch.POS);
        int group = cache.group();
        int[] cells = step.cellIdx(group);

        if (l.qLora > 0) {
            projectLowRankQuery(n, x);
        }
        if (indexer != null) {
            indexer.queries(pos, n, qr, 0, x, 0);
            indexer.storeKeys(pos, cells, n, x, 0);
        }
        RopeSpec r = l.ropeSpec;
        if (r != null) {
            RopePositions.fill(r.type(), r.textPositions(), pos, 0, n, ropePos);
        }
        projectQuery(n, x);
        applyQueryRope(n);
        if (!l.kvShared) {
            projectLatent(n, x);
            for (int t = 0; t < n; t++) {
                cache.writeK(cells[t], kv, t * l.dK);
            }
        }
        absorbQueries(n);
        applyQuerySsmax(0, n, pos, 0);
        boolean causal = indexer == null || (indexer.config().mask().causal() && !ctx.stepNonCausal());
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int t0 = s.tokStart(seg);
            int len = s.len(seg);
            int slots = step.seqCells(group, s.seq(seg), slotRows);
            int k = indexer != null
                    ? indexer.selectSlots(t0, len, pos, slots, slotRows, causal, topK, t0 * l.topK)
                    : SparseKernels.selectedCount(slots, l.topK);
            attend(t0, len, k);
        }
        output(n, x, out);
    }

    /** {@code qr = rms_norm(wq_a @ x)}, the low-rank query the indexer and {@code wq_b} read. */
    private void projectLowRankQuery(int n, float[] x) {
        DsaLayout l = layout;
        matVec.matmul(wqA, x, 0, qr, 0, n);
        qaDeltas.applyRows(n, x, 0, qr, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.qLora, l.qLora, 1, qr, 0, qANormW, null, qr, 0);
    }

    /** Q of every token, from {@code wq_b . qr} or directly from {@code attn_q . x}. */
    private void projectQuery(int n, float[] x) {
        DsaLayout l = layout;
        if (l.qLora > 0) {
            matVec.matmul(wqB, qr, 0, q, 0, n);
            qbDeltas.applyRows(n, qr, 0, q, 0);
        } else {
            matVec.matmul(wq, x, 0, q, 0, n);
            qDeltas.applyRows(n, x, 0, q, 0);
        }
    }

    /**
     * The latent row of every token: {@code [kv_cmpr | k_pe]} for a roped layer, {@code [kv_cmpr | 0]} for a NoPE
     * layer (mistral-large4.cpp:192-197, 388-421); the latent prefix is RMS-normed and the rotary key rotated.
     */
    private void projectLatent(int n, float[] x) {
        DsaLayout l = layout;
        if (l.ropeApplied) {
            matVec.matmul(wkvA, x, 0, kv, 0, n);
            kvaDeltas.applyRows(n, x, 0, kv, 0);
            if (kPeNormW != null) {
                NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.rope, l.dK, 1, kv, l.kvLora, kPeNormW, null, kv,
                        l.kvLora);
            }
            CpuRope.apply(ropeTable, kv, 0, n, 1, l.dK, l.dK, l.kvLora, ropePos, 0, n);
        } else {
            matVec.matmul(wkvA, x, 0, kvProj, 0, n);
            kvaDeltas.applyRows(n, x, 0, kvProj, 0);
            for (int t = 0; t < n; t++) {
                System.arraycopy(kvProj, t * l.kvLora, kv, t * l.dK, l.kvLora);
                Arrays.fill(kv, t * l.dK + l.kvLora, (t + 1) * l.dK, 0f);
            }
        }
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.kvLora, l.dK, 1, kv, 0, kvANormW, null, kv, 0);
    }

    /** The rotary part of Q of every token (on a NoPE layer Q has none). */
    private void applyQueryRope(int n) {
        DsaLayout l = layout;
        if (l.ropeApplied) {
            CpuRope.apply(ropeTable, q, 0, n, l.nHead, l.headK, l.nHead * l.headK, l.nope, ropePos, 0, n);
        }
    }

    /**
     * SSMax: multiplies {@code qcur[t][h]} by {@code attn_ssmax.weight[h] * log(pos + 1) + bias[h]}
     * (mistral-large4.cpp:181-190, 381). A null {@code tokenPos} uses {@code pos0 + t}.
     */
    private void applyQuerySsmax(int t0, int n, int[] tokenPos, int pos0) {
        if (ssmaxW == null) {
            return;
        }
        DsaLayout l = layout;
        for (int t = 0; t < n; t++) {
            int pos = tokenPos != null ? tokenPos[t0 + t] : pos0 + t;
            double basis = Math.log(pos + 1.0);
            for (int h = 0; h < l.nHead; h++) {
                float f = (float) (ssmaxW[h] * basis) + ssmaxB[h];
                int base = ((t0 + t) * l.nHead + h) * l.dK;
                for (int d = 0; d < l.dK; d++) {
                    qcur[base + d] *= f;
                }
            }
        }
    }

    /** The absorbed queries {@code [wk_b . q_nope | q_pe]} of every (token, head) pair into {@link #qcur}. */
    private void absorbQueries(int n) {
        DsaLayout l = layout;
        CpuMlaHeadProducts.run(wkB, l.nHead, l.kvLora, n, q, 0, l.nHead * l.headK, l.headK, latent, 0,
                l.nHead * l.kvLora, l.kvLora);
        for (int pair = 0; pair < n * l.nHead; pair++) {
            System.arraycopy(latent, pair * l.kvLora, qcur, pair * l.dK, l.kvLora);
            if (l.ropeApplied) {
                System.arraycopy(q, pair * l.headK + l.nope, qcur, pair * l.dK + l.kvLora, l.rope);
            } else {
                Arrays.fill(qcur, pair * l.dK + l.kvLora, (pair + 1) * l.dK, 0f);
            }
        }
    }

    /**
     * Sparse attention of the tokens {@code [t0, t0 + len)} over their {@code k} selected latent rows
     * {@code topK[t0*topK + t*k ..]}, into {@link #latent} at the tokens' rows.
     */
    private void attend(int t0, int len, int k) {
        DsaLayout l = layout;
        int qOff = t0 * l.nHead * l.dK;
        int selOff = t0 * l.topK;
        int outOff = t0 * l.nHead * l.kvLora;
        if (len == 1) {
            CpuSparseMla.decode(l.nHead, l.kvLora, l.rope, k, l.kqScale, l.softcap, sinks != null, qcur, qOff,
                    latentRows, topK, selOff, sinks, 0, latent, outOff);
        } else {
            CpuSparseMla.prefill(len, l.nHead, l.kvLora, l.rope, k, l.kqScale, l.softcap, sinks != null, qcur, qOff,
                    l.nHead * l.dK, latentRows, topK, selOff, sinks, 0, latent, outOff);
        }
    }

    /** {@code wv_b} expansion, the optional gate and {@code wo} of every token. */
    private void output(int n, float[] x, float[] out) {
        DsaLayout l = layout;
        CpuMlaHeadProducts.run(wvB, l.nHead, l.headV, n, latent, 0, l.nHead * l.kvLora, l.kvLora, q, 0,
                l.nHead * l.headV, l.headV);
        if (gateW != null) {
            applyGate(n, x);
        }
        matVec.matmul(wo, q, 0, out, 0, n);
        if (woScale != 1.0f) {
            CpuOps.scale(out, 0, n * l.nEmbd, woScale);
        }
        outDeltas.applyRows(n, q, 0, out, 0);
    }

    /** Twin of {@code GpuDsaMlaMixer.applyGate}: {@code attn = sigmoid(gate) * attn}, per head or per element. */
    private void applyGate(int n, float[] x) {
        DsaLayout l = layout;
        matVec.matmul(gateW, x, 0, gate, 0, n);
        gateDeltas.applyRows(n, x, 0, gate, 0);
        int size = n * l.nHead * l.headV;
        float[] g = gate;
        if (l.gate == DsaLayout.Gate.PER_HEAD) {
            for (int pair = 0; pair < n * l.nHead; pair++) {
                Arrays.fill(gateWide, pair * l.headV, (pair + 1) * l.headV, gate[pair]);
            }
            g = gateWide;
        }
        CpuOps.glu(GluOp.SIGMUL, ActParams.NONE, g, 0, q, 0, q, 0, size);
    }

    private void checkStep(int pos0, int n, float[] in, float[] out) {
        if (n < 1 || n > ubatch || pos0 < 0) {
            throw new IllegalArgumentException(layout.where + ": DSA step of " + n + " tokens at " + pos0
                    + " is outside [1, " + ubatch + "] tokens at a non-negative position");
        }
        long width = (long) n * layout.nEmbd;
        if (Objects.requireNonNull(in, "in").length < width || Objects.requireNonNull(out, "out").length < width) {
            throw new IllegalArgumentException(layout.where + ": DSA step arrays hold " + in.length + " / "
                    + out.length + " floats, " + width + " are needed");
        }
    }

    // ------------------------------------------------------------------ bind helpers

    /**
     * A per-head tensor without an input transform (PRISM Hadamard, ACT_QUANT, PERMUTE) or output scale (see
     * {@code GpuDsaMlaMixer.perHead}), of a type the CPU vec-dot reads.
     */
    private CpuTensorRef perHead(CpuTensorRef w) {
        CpuMatVec.Transforms t = matVec.transformsOf(w);
        if (!t.transform().isIdentity() || t.hasScale()) {
            throw new UnsupportedOperationException(layout.where + ": per-head tensor " + w + " carries an input"
                    + " transform or output scale, which the absorbed MLA products do not apply");
        }
        CpuMatVec.vecDot(w.type());
        return w;
    }

    private float[] vector(float[] v, int n, String name) {
        if (v == null || v.length != n) {
            throw new IllegalStateException(layout.where + ": " + name + " must hold " + n + " values, got "
                    + (v == null ? "none" : String.valueOf(v.length)));
        }
        return v;
    }

    private static long[] shape(CpuTensorRef w) {
        return w == null ? null : new long[]{w.cols(), w.rows()};
    }

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("DSA mixer is closed");
        }
        if (ctx == null) {
            throw new IllegalStateException("DSA mixer is not bound");
        }
    }

    @Override
    public void close() {
        closed = true;
        indexer = null;
    }

    @Override
    public String toString() {
        return "CpuDsaMlaMixer{" + (layout != null ? layout.toString() : "unbound") + '}';
    }
}
