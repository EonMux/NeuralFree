package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Blocks.Dsv4Layout;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels.CompMode;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexer.RowReader;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexerStage;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Objects;

/**
 * CPU twin of {@link GpuDsv4Mixer}: the same stages in the same order with the kernels' CPU math. Projections use
 * {@link CpuMatVec} (the grouped {@code attn_output_a} through {@link CpuMatVec#matvecExpert} with one "expert" per
 * group, the twin of {@code {T}_MatVecId}); norms {@link NormMath#rowNorm} ({@code LLM_RowNorm}); RoPE
 * {@link CpuRope#apply} ({@code LLM_RoPE_Ext}); the compressors {@link CpuDsv4Compressor#compress},
 * {@link CpuDsv4Compressor#lidUpdate}, {@link CpuDsv4Compressor#stateSnapshot} (when the memory keeps rollback
 * snapshots) and {@link CpuDsv4Compressor#stateUpdate} ({@code LLM_DSV4_Compress}, {@code _LidUpdate},
 * {@code _StateSnapshot}, {@code _StateUpdate}); the raw store {@link CpuDsv4Compressor#cacheStore}
 * ({@code LLM_DSV4_CacheStore}); the attention {@link CpuDsv4Attention#attend} ({@code LLM_DSV4_CompressedAttn});
 * the lightning indexer {@link CpuDsaIndexerStage}.
 *
 * <p>Raw row positions come from {@link CpuMemory#positionOfRow}, which the executor marks for the step's rows
 * before the mixers run ({@link CpuMemory#beginStep}); indexer row {@code r} stands for position
 * {@code (r + 1) * 4 - 1}. Working arrays are {@link CpuScratch} temporaries taken at bind; a step allocates no
 * per-token arrays. Not thread-safe.</p>
 */
public final class CpuDsv4Mixer implements CpuMixer {

    public static final String TEMP_ROPE_POS = "dsv4.rope.pos";
    public static final String TEMP_Q_POS = "dsv4.qpos";
    public static final String TEMP_CELLS = "dsv4.cells";
    public static final String TEMP_RAW_POS = "dsv4.raw.pos";

    private CpuForwardContext ctx;
    private CpuMemory memory;
    private CpuMatVec matVec;
    private Dsv4Layout layout;
    private int ubatch;

    private CpuTensorRef wqA;
    private CpuTensorRef wqB;
    private CpuTensorRef wkv;
    private CpuTensorRef woA;
    private CpuTensorRef woB;
    /**
     * LoRA deltas of the projections llama.cpp runs through {@code build_lora_mm} (deepseek4.cpp:918-1039, 1223);
     * the grouped {@code attn_output_a} is a plain {@code ggml_mul_mat} there (:1218) and takes none.
     */
    private CpuProjectionDeltas.Site qaDeltas;
    private CpuProjectionDeltas.Site qbDeltas;
    private CpuProjectionDeltas.Site kvDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private float[] qANormW;
    private float[] kvNormW;

    private KvAccessor raw;
    private int rawRows;
    private int rawRot;
    private RopeFreqTable ropeTable;
    private CpuDsv4Attention.Layer attnLayer;
    private KvAccessor compCache;

    private float[] qr;
    private float[] q;
    private float[] kv;
    private float[] attn;
    private float[] oa;
    private float[] curKv;
    private float[] curScore;
    private int[] topK;
    private int[] ropePos;
    private int[] qPos;
    private int[] cells;
    private int[] rawPos;

    private Compressor compressor;
    private Compressor lidCompressor;
    private CpuDsaIndexerStage indexer;
    private boolean closed;

    /** Use {@link Dsv4Blocks#cpuMixers()}. */
    CpuDsv4Mixer() {
    }

    /**
     * Binds the mixer: validates the layer's tensors against its spec ({@link Dsv4Layout}) and its caches against
     * the kernel contracts, takes the working arrays and binds the indexer of a CSA layer.
     *
     * @throws IllegalStateException         when already bound or closed, the context has no memory, or the
     *                                       tensors or caches contradict the spec
     * @throws UnsupportedOperationException when the grouped output tensor carries a transform the grouped product
     *                                       does not apply
     */
    @Override
    public void bind(CpuForwardContext context, LayerSpec layer) {
        if (closed || ctx != null) {
            throw new IllegalStateException("DeepSeek-V4 mixer is " + (closed ? "closed" : "already bound"));
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
        layout = Dsv4Layout.of(layer, ctx.spec(), suffix -> shape(weights.layerOrNull(layer, suffix)));
        Dsv4Layout l = layout;
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(l.where + ": the DeepSeek-V4 mixer needs the model's KV memory");
        }
        memory = ctx.memory();
        raw = memory.kv(l.il);
        if (raw == null) {
            throw new IllegalStateException(l.where + ": no raw window cache for the DeepSeek-V4 layer");
        }
        if (raw.nHeadKv() != 1 || raw.headDimK() != l.headDim) {
            throw new IllegalStateException(l.where + ": the DeepSeek-V4 raw cache must hold one K head of "
                    + l.headDim + " values per row, got " + raw.nHeadKv() + " x " + raw.headDimK());
        }
        KvLayout kvl = raw.layout();
        l.checkRawLayout(kvl);
        rawRows = kvl.rows();
        KvDType rawType = kvl.dtypeK();
        RotationPlan rot = memory.plan().layer(l.il).rotation();
        rawRot = rot.rotateK() ? rot.nRotK() : 0;
        Dsv4Kernels.checkWidth(l.headDim, rawType);
        Dsv4Kernels.checkRotBlock(rawRot, l.headDim);

        wqA = weights.layer(layer, Dsv4Layout.WQ_A);
        wqB = weights.layer(layer, Dsv4Layout.WQ_B);
        wkv = weights.layer(layer, Dsv4Layout.WKV);
        woA = perGroup(weights.layer(layer, Dsv4Layout.WO_A));
        woB = weights.layer(layer, Dsv4Layout.WO_B);
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        qaDeltas = deltas.serve(layer, Dsv4Layout.WQ_A, wqA, matVec);
        qbDeltas = deltas.serve(layer, Dsv4Layout.WQ_B, wqB, matVec);
        kvDeltas = deltas.serve(layer, Dsv4Layout.WKV, wkv, matVec);
        outDeltas = deltas.serve(layer, Dsv4Layout.WO_B, woB, matVec);
        qANormW = vector(weights.layerFloatsOrNull(layer, Dsv4Layout.Q_A_NORM), l.qLora, Dsv4Layout.Q_A_NORM);
        kvNormW = vector(weights.layerFloatsOrNull(layer, Dsv4Layout.KV_NORM), l.headDim, Dsv4Layout.KV_NORM);
        float[] sinks = l.sinks ? vector(weights.layerFloatsOrNull(layer, Dsv4Layout.SINKS), l.nHead,
                Dsv4Layout.SINKS) : null;
        RopeSpec r = l.attnRope;
        ropeTable = RopeFreqTable.of(r, null);
        attnLayer = CpuDsv4Attention.Layer.of(l.nHead, l.headDim, l.kqScale, l.window, l.ratio, rawRot, l.attnParams,
                sinks);

        CpuScratch s = ctx.scratch();
        int qWidth = l.nHead * l.headDim;
        qr = s.temp(GpuDsv4Mixer.TEMP_QR, ubatch * l.qLora);
        q = s.temp(GpuDsv4Mixer.TEMP_Q, ubatch * qWidth);
        kv = s.temp(GpuDsv4Mixer.TEMP_KV, ubatch * l.headDim);
        attn = s.temp(GpuDsv4Mixer.TEMP_ATTN, ubatch * qWidth);
        oa = s.temp(GpuDsv4Mixer.TEMP_OA, ubatch * l.nGroups * l.oLora);
        ropePos = s.intTemp(TEMP_ROPE_POS, r.positionsPerToken() * ubatch);
        qPos = s.intTemp(TEMP_Q_POS, ubatch);
        cells = s.intTemp(TEMP_CELLS, ubatch);
        rawPos = s.intTemp(TEMP_RAW_POS, rawRows);

        if (l.mode != CompMode.NONE) {
            int curWidth = Math.max(l.stateWidth, l.idxStateWidth);
            curKv = s.temp(GpuDsv4Mixer.TEMP_CUR_KV, ubatch * curWidth);
            curScore = s.temp(GpuDsv4Mixer.TEMP_CUR_SCORE, ubatch * curWidth);
            compressor = bindCompressor(layer, weights, l.compressedKind(), rawType, rawRot);
            compCache = compressor.cache;
            if (l.mode == CompMode.SELECTED) {
                int lidRot = Dsv4Layout.lidRotationBlock(l.idxDim);
                lidCompressor = bindCompressor(layer, weights, CompressedKvPlan.Kind.INDEXER, rawType, lidRot);
                topK = s.intTemp(GpuDsv4Mixer.TEMP_TOPK, ubatch * l.topK);
                indexer = new CpuDsaIndexerStage(ctx, l.where, l.indexerConfig(lidRot));
                indexer.bindQueries(weights.layer(layer, Dsv4Layout.IDX_Q_B),
                        weights.layer(layer, Dsv4Layout.IDX_PROJ));
                KvAccessor lid = lidCompressor.cache;
                indexer.bindSelection(RowReader.ofKeys(lid), lid.layout().rows(),
                        row -> Dsv4Layout.compressedRowPosition(row, Dsv4Spec.RATIO_CSA));
                indexer.bindDeltas(layer);
            }
        }
    }

    /**
     * The CSA / HCA compressor ({@code attn_compressor_*}) or the indexer compressor ({@code indexer_compressor_*})
     * of a compressed cache kind, with its cache and planes checked against the layout.
     */
    private Compressor bindCompressor(LayerSpec layer, CpuWeights weights, CompressedKvPlan.Kind kind,
                                      KvDType rawType, int rotBlock) {
        Dsv4Layout l = layout;
        boolean lid = kind == CompressedKvPlan.Kind.INDEXER;
        CompressedKvPlan plan = null;
        for (CompressedKvPlan p : memory.plan().layer(l.il).compressed()) {
            if (p.kind() == kind) {
                plan = p;
            }
        }
        KvAccessor cache = memory.compressed(l.il, kind);
        RecurrentStateAccessor planes = memory.compressorState(l.il, kind);
        if (plan == null || cache == null || planes == null) {
            throw new IllegalStateException(l.where + ": no " + kind + " compressed cache for the DeepSeek-V4 layer");
        }
        l.checkCompressedPlan(plan, rawType);
        int ratio = plan.ratio();
        int width = plan.width();
        int planeSize = plan.planes() * plan.planeFloats();
        if (cache.nHeadKv() != 1 || cache.headDimK() != width || cache.layout().rows() != plan.rows()) {
            throw new IllegalStateException(l.where + ": the " + kind + " cache must hold " + plan.rows()
                    + " single-head rows of " + width + " values, got " + cache.layout());
        }
        float[] stateKv = plane(planes, CpuMemory.COMPRESSOR_KV, planeSize, kind);
        float[] stateScore = plane(planes, CpuMemory.COMPRESSOR_SCORE, planeSize, kind);
        String apeName = lid ? Dsv4Layout.IDX_COMP_APE : Dsv4Layout.COMP_APE;
        String normName = lid ? Dsv4Layout.IDX_COMP_NORM : Dsv4Layout.COMP_NORM;
        float[] ape = vector(weights.layerFloatsOrNull(layer, apeName), ratio * plan.stateWidth(), apeName);
        float[] norm = vector(weights.layerFloatsOrNull(layer, normName), width, normName);
        Dsv4Kernels.checkWidth(width, plan.dtype());
        Dsv4Kernels.checkRotBlock(rotBlock, width);
        Dsv4Kernels.checkRope(l.compressParams, width);
        if (lid) {
            Dsv4Kernels.checkLidCache(plan.dtype(), rotBlock, width);
        }
        CpuDsv4Compressor.Config cfg = CpuDsv4Compressor.Config.of(ratio, width, l.rmsEps, l.compressParams, rotBlock);
        String kvName = lid ? Dsv4Layout.IDX_COMP_WKV : Dsv4Layout.COMP_WKV;
        String gateName = lid ? Dsv4Layout.IDX_COMP_WGATE : Dsv4Layout.COMP_WGATE;
        CpuTensorRef wKv = weights.layer(layer, kvName);
        CpuTensorRef wGate = weights.layer(layer, gateName);
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        CpuProjectionDeltas.Site[] sites = {deltas.serve(layer, kvName, wKv, matVec),
                deltas.serve(layer, gateName, wGate, matVec)};
        return new Compressor(kind, cfg, plan.rows(), plan.snapshots(), wKv, wGate, sites, ape, norm, stateKv,
                stateScore, cache, lid);
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

    /** Twin of {@link GpuDsv4Mixer#shiftRefusal()}. */
    @Override
    public String shiftRefusal() {
        return "context shift is not supported for DeepSeek-V4 compressed attention (llama-kv-cache-dsv4.cpp:1452)";
    }

    private void forward(int pos0, int n, float[] x, float[] out) {
        ensureBound();
        Dsv4Layout l = layout;
        checkStep(pos0, n, x, out);
        int last = pos0 + n - 1;
        int qWidth = l.nHead * l.headDim;

        matVec.matmul(wqA, x, 0, qr, 0, n);
        qaDeltas.applyRows(n, x, 0, qr, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.qLora, l.qLora, 1, qr, 0, qANormW, null, qr, 0);
        matVec.matmul(wqB, qr, 0, q, 0, n);
        qbDeltas.applyRows(n, qr, 0, q, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n * l.nHead, l.headDim, l.headDim, 1, q, 0, null, null, q, 0);
        RopeSpec r = l.attnRope;
        RopePositions.fill(r.type(), r.textPositions(), pos0, n, ropePos);
        CpuRope.apply(ropeTable, q, 0, n, l.nHead, l.headDim, qWidth, l.nope, ropePos, 0, n);
        matVec.matmul(wkv, x, 0, kv, 0, n);
        kvDeltas.applyRows(n, x, 0, kv, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.headDim, l.headDim, 1, kv, 0, kvNormW, null, kv, 0);
        CpuRope.apply(ropeTable, kv, 0, n, 1, l.headDim, l.headDim, l.nope, ropePos, 0, n);

        CpuDsv4Attention.Compressed comp = CpuDsv4Attention.Compressed.none();
        if (compressor != null) {
            compressor.run(pos0, n, x);
        }
        if (l.mode == CompMode.SELECTED) {
            lidCompressor.run(pos0, n, x);
            int nLid = Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_CSA);
            int k = 0;
            if (nLid > 0) {
                indexer.queries(pos0, n, qr, 0, x, 0);
                k = indexer.select(pos0, n, nLid, topK, 0);
            }
            comp = CpuDsv4Attention.Compressed.selected(compCache, k, topK, 0, k);
        } else if (l.mode == CompMode.DENSE) {
            comp = CpuDsv4Attention.Compressed.dense(compCache, Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_HCA));
        }

        KvLayout kvl = raw.layout();
        for (int t = 0; t < n; t++) {
            cells[t] = kvl.rowOf(pos0 + t);
            qPos[t] = pos0 + t;
        }
        CpuDsv4Compressor.cacheStore(n, l.headDim, rawRot, kv, 0, l.headDim, cells, raw);

        int nRaw = Math.min(rawRows, pos0 + n);
        int group = raw.group();
        for (int row = 0; row < nRaw; row++) {
            rawPos[row] = memory.positionOfRow(group, row);
        }
        CpuDsv4Attention.attend(attnLayer, n, q, 0, qPos, raw, nRaw, rawPos, comp, attn, 0);

        for (int pair = 0; pair < n * l.nGroups; pair++) {
            matVec.matvecExpert(woA, pair % l.nGroups, l.nGroups, attn, pair * l.oGroupDim, oa, pair * l.oLora);
        }
        matVec.matmul(woB, oa, 0, out, 0, n);
        outDeltas.applyRows(n, oa, 0, out, 0);
    }

    private void checkStep(int pos0, int n, float[] in, float[] out) {
        Dsv4Layout l = layout;
        if (n < 1 || n > ubatch || pos0 < 0) {
            throw new IllegalArgumentException(l.where + ": DeepSeek-V4 step of " + n + " tokens at " + pos0
                    + " is outside [1, " + ubatch + "] tokens at a non-negative position");
        }
        if (compressor != null) {
            l.checkCapacity(pos0, n, compressor.cfg.ratio(), compressor.rows, compressor.kind.toString());
        }
        if (lidCompressor != null) {
            l.checkCapacity(pos0, n, lidCompressor.cfg.ratio(), lidCompressor.rows, lidCompressor.kind.toString());
        }
        long width = (long) n * l.nEmbd;
        if (Objects.requireNonNull(in, "in").length < width || Objects.requireNonNull(out, "out").length < width) {
            throw new IllegalArgumentException(l.where + ": DeepSeek-V4 step arrays hold " + in.length + " / "
                    + out.length + " floats, " + width + " are needed");
        }
    }

    /** One compressor of the layer: its weights, planes and cache. */
    private final class Compressor {
        final CompressedKvPlan.Kind kind;
        final CpuDsv4Compressor.Config cfg;
        final int rows;
        /** Rollback snapshot planes after the live plane ({@code n_rs_seq}). */
        final int snapshots;
        final CpuTensorRef wKv;
        final CpuTensorRef wGate;
        /** LoRA deltas of {@code wKv} and {@code wGate} (deepseek4.cpp:952-970, 1036-1039). */
        final CpuProjectionDeltas.Site kvDeltas;
        final CpuProjectionDeltas.Site gateDeltas;
        final float[] ape;
        final float[] norm;
        final float[] stateKv;
        final float[] stateScore;
        final KvAccessor cache;
        final boolean lid;

        Compressor(CompressedKvPlan.Kind kind, CpuDsv4Compressor.Config cfg, int rows, int snapshots,
                   CpuTensorRef wKv, CpuTensorRef wGate, CpuProjectionDeltas.Site[] deltas, float[] ape, float[] norm,
                   float[] stateKv, float[] stateScore, KvAccessor cache, boolean lid) {
            this.kind = kind;
            this.cfg = cfg;
            this.rows = rows;
            this.snapshots = snapshots;
            this.wKv = wKv;
            this.wGate = wGate;
            this.kvDeltas = deltas[0];
            this.gateDeltas = deltas[1];
            this.ape = ape;
            this.norm = norm;
            this.stateKv = stateKv;
            this.stateScore = stateScore;
            this.cache = cache;
            this.lid = lid;
        }

        /**
         * Twin of {@code GpuDsv4Mixer.Compressor.run}: project, compress the completed blocks, write the rollback
         * snapshots, persist the planes.
         */
        void run(int pos0, int n, float[] x) {
            matVec.matmul(wKv, x, 0, curKv, 0, n);
            kvDeltas.applyRows(n, x, 0, curKv, 0);
            matVec.matmul(wGate, x, 0, curScore, 0, n);
            gateDeltas.applyRows(n, x, 0, curScore, 0);
            CpuDsv4Compressor.Step step = new CpuDsv4Compressor.Step(pos0, n, curKv, 0, curScore, 0, ape);
            if (Dsv4Kernels.blockCount(pos0, n, cfg.ratio()) > 0) {
                if (lid) {
                    CpuDsv4Compressor.lidUpdate(cfg, step, stateKv, stateScore, norm, cache);
                } else {
                    CpuDsv4Compressor.compress(cfg, step, stateKv, stateScore, norm, cache);
                }
            }
            if (snapshots > 0) {
                CpuDsv4Compressor.stateSnapshot(cfg, step, stateKv, stateScore, snapshots);
            }
            CpuDsv4Compressor.stateUpdate(cfg, step, stateKv, stateScore);
        }
    }

    // ------------------------------------------------------------------ bind helpers

    /** Twin of {@code GpuDsv4Mixer.perGroup}: the grouped {@code attn_output_a} without PRISM transform or scale. */
    private CpuTensorRef perGroup(CpuTensorRef w) {
        CpuMatVec.Transforms t = matVec.transformsOf(w);
        if (t.input() != null || t.hasScale()) {
            throw new UnsupportedOperationException(layout.where + ": grouped tensor " + w + " carries a PRISM"
                    + " transform or output scale, which the grouped output product does not apply");
        }
        return w;
    }

    private float[] plane(RecurrentStateAccessor planes, String name, int size, CompressedKvPlan.Kind kind) {
        float[] p = planes.tensor(name);
        if (p == null || p.length < size) {
            throw new IllegalStateException(layout.where + ": the " + kind + " compressor plane '" + name
                    + "' must hold " + size + " floats, got " + (p == null ? "none" : String.valueOf(p.length)));
        }
        return p;
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
            throw new IllegalStateException("DeepSeek-V4 mixer is closed");
        }
        if (ctx == null) {
            throw new IllegalStateException("DeepSeek-V4 mixer is not bound");
        }
    }

    @Override
    public void close() {
        closed = true;
        indexer = null;
        compressor = null;
        lidCompressor = null;
    }

    @Override
    public String toString() {
        return "CpuDsv4Mixer{" + (layout != null ? layout.toString() : "unbound") + '}';
    }
}
