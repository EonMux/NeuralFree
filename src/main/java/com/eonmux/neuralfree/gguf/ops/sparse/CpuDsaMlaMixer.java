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
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
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
 * {@link CpuDsaIndexerStage}.
 *
 * <p>The top-k selection lives in the shared int temporary {@value GpuDsaMlaMixer#TEMP_TOPK} of
 * {@code ubatch * topK} entries, which a shared layer reads after its source layer wrote it in the same step.
 * Working arrays are {@link CpuScratch} temporaries taken at bind; a step allocates no per-token arrays. Not
 * thread-safe.</p>
 */
public final class CpuDsaMlaMixer implements CpuMixer {

    public static final String TEMP_POSITIONS = "dsa.pos";

    private CpuForwardContext ctx;
    private CpuMatVec matVec;
    private DsaLayout layout;
    private int ubatch;
    private int kvRows;

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
    private CpuProjectionDeltas.Site qaDeltas;
    private CpuProjectionDeltas.Site qbDeltas;
    private CpuProjectionDeltas.Site kvaDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private CpuProjectionDeltas.Site gateDeltas;

    private float[] qANormW;
    private float[] kvANormW;
    private float[] kPeNormW;
    private float[] sinks;

    private KvAccessor cache;
    private RowReader latentRows;
    private RopeFreqTable ropeTable;

    private float[] qr;
    private float[] q;
    private float[] kv;
    private float[] qcur;
    private float[] latent;
    private float[] gate;
    private float[] gateWide;
    private int[] topK;
    private int[] ropePos;

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

        wqA = weights.layer(layer, DsaLayout.WQ_A);
        wqB = weights.layer(layer, DsaLayout.WQ_B);
        wkvA = weights.layer(layer, DsaLayout.WKV_A);
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
        qaDeltas = deltas.serve(layer, DsaLayout.WQ_A, wqA, matVec);
        qbDeltas = deltas.serve(layer, DsaLayout.WQ_B, wqB, matVec);
        kvaDeltas = deltas.serve(layer, DsaLayout.WKV_A, wkvA, matVec);
        outDeltas = deltas.serve(layer, DsaLayout.WO, wo, matVec);
        qANormW = vector(weights.layerFloatsOrNull(layer, DsaLayout.Q_A_NORM), l.qLora, DsaLayout.Q_A_NORM);
        kvANormW = vector(weights.layerFloatsOrNull(layer, DsaLayout.KV_A_NORM), l.kvLora, DsaLayout.KV_A_NORM);
        if (l.kPeNorm) {
            kPeNormW = vector(weights.layerFloatsOrNull(layer, DsaLayout.K_PE_NORM), l.rope, DsaLayout.K_PE_NORM);
        }
        if (l.sinks) {
            sinks = vector(weights.layerFloatsOrNull(layer, DsaLayout.SINKS), l.nHead, DsaLayout.SINKS);
        }
        RopeSpec r = l.ropeSpec;
        ropeTable = RopeFreqTable.of(r, null);

        CpuScratch s = ctx.scratch();
        qr = s.temp(GpuDsaMlaMixer.TEMP_QR, ubatch * l.qLora);
        q = s.temp(GpuDsaMlaMixer.TEMP_Q, ubatch * l.qBufferWidth());
        kv = s.temp(GpuDsaMlaMixer.TEMP_KV, ubatch * l.dK);
        qcur = s.temp(GpuDsaMlaMixer.TEMP_QCUR, ubatch * l.nHead * l.dK);
        latent = s.temp(GpuDsaMlaMixer.TEMP_LATENT, ubatch * l.nHead * l.kvLora);
        topK = s.intTemp(GpuDsaMlaMixer.TEMP_TOPK, ubatch * l.topK);
        ropePos = s.intTemp(TEMP_POSITIONS, r.positionsPerToken() * ubatch);
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
        indexer.bindQueries(weights.layer(layer, DsaLayout.IDX_Q_B), weights.layer(layer, DsaLayout.IDX_PROJ));
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

    private void forward(int pos0, int n, float[] x, float[] out) {
        ensureBound();
        DsaLayout l = layout;
        checkStep(pos0, n, x, out);
        int nKv = Math.min(kvRows, pos0 + n);

        matVec.matmul(wqA, x, 0, qr, 0, n);
        qaDeltas.applyRows(n, x, 0, qr, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.qLora, l.qLora, 1, qr, 0, qANormW, null, qr, 0);

        int k;
        if (indexer != null) {
            indexer.queries(pos0, n, qr, 0, x, 0);
            indexer.storeKeys(pos0, n, x, 0);
            k = indexer.select(pos0, n, nKv, topK, 0);
        } else {
            k = SparseKernels.selectedCount(nKv, l.topK);
        }

        matVec.matmul(wqB, qr, 0, q, 0, n);
        qbDeltas.applyRows(n, qr, 0, q, 0);
        matVec.matmul(wkvA, x, 0, kv, 0, n);
        kvaDeltas.applyRows(n, x, 0, kv, 0);
        if (kPeNormW != null) {
            NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.rope, l.dK, 1, kv, l.kvLora, kPeNormW, null, kv, l.kvLora);
        }
        RopeSpec r = l.ropeSpec;
        RopePositions.fill(r.type(), r.textPositions(), pos0, n, ropePos);
        CpuRope.apply(ropeTable, q, 0, n, l.nHead, l.headK, l.nHead * l.headK, l.nope, ropePos, 0, n);
        CpuRope.apply(ropeTable, kv, 0, n, 1, l.dK, l.dK, l.kvLora, ropePos, 0, n);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.kvLora, l.dK, 1, kv, 0, kvANormW, null, kv, 0);
        KvLayout kvl = cache.layout();
        for (int t = 0; t < n; t++) {
            cache.writeK(kvl.rowOf(pos0 + t), kv, t * l.dK);
        }

        CpuMlaHeadProducts.run(wkB, l.nHead, l.kvLora, n, q, 0, l.nHead * l.headK, l.headK, latent, 0,
                l.nHead * l.kvLora, l.kvLora);
        for (int pair = 0; pair < n * l.nHead; pair++) {
            System.arraycopy(latent, pair * l.kvLora, qcur, pair * l.dK, l.kvLora);
            System.arraycopy(q, pair * l.headK + l.nope, qcur, pair * l.dK + l.kvLora, l.rope);
        }

        if (n == 1) {
            CpuSparseMla.decode(l.nHead, l.kvLora, l.rope, k, l.kqScale, l.softcap, sinks != null, qcur, 0,
                    latentRows, topK, 0, sinks, 0, latent, 0);
        } else {
            CpuSparseMla.prefill(n, l.nHead, l.kvLora, l.rope, k, l.kqScale, l.softcap, sinks != null, qcur, 0,
                    l.nHead * l.dK, latentRows, topK, 0, sinks, 0, latent, 0);
        }
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
     * A per-head tensor without a PRISM transform or output scale (see {@code GpuDsaMlaMixer.perHead}), of a type the
     * CPU vec-dot reads.
     */
    private CpuTensorRef perHead(CpuTensorRef w) {
        CpuMatVec.Transforms t = matVec.transformsOf(w);
        if (t.input() != null || t.hasScale()) {
            throw new UnsupportedOperationException(layout.where + ": per-head tensor " + w + " carries a PRISM"
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
