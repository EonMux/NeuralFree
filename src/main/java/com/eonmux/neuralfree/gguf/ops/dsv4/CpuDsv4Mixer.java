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
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Blocks.Dsv4Layout;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels.CompMode;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexer.RowReader;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexerStage;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseKernels;
import com.eonmux.neuralfree.gguf.quant.TapProduct;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntUnaryOperator;

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
 *
 * <p>A paged multi-sequence step ({@link #batch}) runs the token-wise projections over every row of the step, stores
 * the raw rows at the step's cells, then compresses, selects and attends sequence by sequence as llama.cpp runs one
 * DSV4 stream per sequence (llama-kv-cache-dsv4.cpp:51-60, 200-208): each sequence's compressors read and write its
 * own compressed stream and planes ({@link CpuMemory#compressed(int, CompressedKvPlan.Kind, int)}), its indexer
 * selects among its own compressed rows, and its raw window is its own cells in slot order. The rows of a segment
 * must sit at consecutive positions (the compressors fold consecutive blocks), and the attention is causal, so a
 * non-causal step is refused by name.</p>
 */
public final class CpuDsv4Mixer implements CpuMixer {

    public static final String TEMP_ROPE_POS = "dsv4.rope.pos";
    public static final String TEMP_Q_POS = "dsv4.qpos";
    public static final String TEMP_CELLS = "dsv4.cells";
    public static final String TEMP_RAW_POS = "dsv4.raw.pos";
    public static final String TEMP_RAW_CELLS = "dsv4.raw.cells";
    public static final String TEMP_SEG_POS = "dsv4.seg.pos";
    public static final String TEMP_CAND = "dsv4.cand.blocks";
    public static final String TEMP_COMP_LATENT = "dsv4.comp.latent";
    public static final String TEMP_COMP_INDEX = "dsv4.comp.index";
    public static final String TEMP_COMP_ROWS = "dsv4.comp.rows";

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
    private int[] rawCells;
    private int[] segPos;

    private Compressor compressor;
    private Compressor lidCompressor;
    private CpuDsaIndexerStage indexer;
    private final IntUnaryOperator lidRowPosition =
            row -> Dsv4Layout.compressedRowPosition(row, Dsv4Spec.RATIO_CSA);
    private boolean closed;

    private boolean v41;
    private int compReadLayer;
    /** The KV source layer's compressed caches and readers of one stream, or null for a window-only layer. */
    private KvAccessor[] compCaches;
    private RowReader[] compReaders;
    /** V4.1 index-key readers over the compressed rows (offset {@code headDim}), or null for V4.0. */
    private RowReader[] idxKeyReaders;
    private int compRows;
    private CpuTensorRef idxK;
    private CpuProjectionDeltas.Site idxKDeltas;
    private float[] idxKNormW;
    private float[] compLatent;
    private float[] compIndex;
    private float[] compRowsBuf;
    private float[] zeroApe;
    private int[] candBlocks;
    private boolean picksCandidate;
    private boolean usesCandidate;
    private int candTopKBlocks;
    private int candBlockSize;
    private IntUnaryOperator compRowPosition;

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
        v41 = l.v41 != null;
        attnLayer = v41 && l.ratio != 0
                ? CpuDsv4Attention.Layer.ofSelected(l.nHead, l.headDim, l.kqScale, l.window, l.ratio, rawRot,
                        l.attnParams, sinks)
                : CpuDsv4Attention.Layer.of(l.nHead, l.headDim, l.kqScale, l.window, l.ratio, rawRot, l.attnParams,
                        sinks);
        compRowPosition = l.ratio == 0 ? null : row -> Dsv4Layout.compressedRowPosition(row, l.ratio);

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
        segPos = s.intTemp(TEMP_SEG_POS, ubatch);
        PagedSequences paged = memory.pagedSequences();
        int rawKeys = paged == null ? rawRows : Math.max(rawRows, paged.maxPages() * paged.pageSize());
        rawPos = s.intTemp(TEMP_RAW_POS, rawKeys);
        rawCells = s.intTemp(TEMP_RAW_CELLS, rawKeys);

        if (l.ratio != 0) {
            int curWidth = Math.max(l.stateWidth, l.idxStateWidth);
            curKv = s.temp(GpuDsv4Mixer.TEMP_CUR_KV, ubatch * curWidth);
            curScore = s.temp(GpuDsv4Mixer.TEMP_CUR_SCORE, ubatch * curWidth);
            if (v41) {
                bindV41(layer, weights, s, rawType);
            } else {
                compressor = bindCompressor(layer, weights, l.compressedKind(), rawType, rawRot);
                if (l.mode == CompMode.SELECTED) {
                    int lidRot = Dsv4Layout.lidRotationBlock(l.idxDim);
                    lidCompressor = bindCompressor(layer, weights, CompressedKvPlan.Kind.INDEXER, rawType, lidRot);
                    topK = s.intTemp(GpuDsv4Mixer.TEMP_TOPK, ubatch * l.topK);
                    indexer = new CpuDsaIndexerStage(ctx, l.where, l.indexerConfig(lidRot));
                    indexer.bindQueries(weights.layer(layer, Dsv4Layout.IDX_Q_B),
                            weights.layer(layer, Dsv4Layout.IDX_PROJ), null);
                    KvAccessor lid = lidCompressor.caches[0];
                    indexer.bindSelection(lidCompressor.readers[0], lid.layout().rows(), lidRowPosition);
                    indexer.bindDeltas(layer);
                }
            }
        }
    }

    /**
     * Binds the DeepSeek-V4.1 compressed attention of a layer: the compressed caches and index keys of its KV source
     * layer, its own compressor when it is that source, its indexer when it computes the top-k, and the shared
     * candidate-block selection (models/deepseek41.cpp:456-631).
     */
    private void bindV41(LayerSpec layer, CpuWeights weights, CpuScratch s, KvDType rawType) {
        Dsv4Layout l = layout;
        CompressedKvPlan.Kind kind = l.compressedKind();
        compReadLayer = l.v41.kvSourceLayer();
        CompressedKvPlan plan = planOf(compReadLayer, kind);
        l.checkCompressedPlan(plan, rawType);
        compRows = plan.rows();
        compCaches = new KvAccessor[plan.streams()];
        compReaders = new RowReader[plan.streams()];
        idxKeyReaders = new RowReader[plan.streams()];
        float[][] stateKv = new float[plan.streams()][];
        float[][] stateScore = new float[plan.streams()][];
        int planeSize = plan.planes() * plan.planeFloats();
        for (int seq = 0; seq < plan.streams(); seq++) {
            KvAccessor cache = memory.compressed(compReadLayer, kind, seq);
            if (cache == null || cache.nHeadKv() != 1 || cache.headDimK() != plan.width()
                    || cache.layout().rows() != plan.rows()) {
                throw new IllegalStateException(l.where + ": the " + kind + " cache of KV source layer " + compReadLayer
                        + " must hold " + plan.rows() + " single-head rows of " + plan.width() + " values, got "
                        + cache);
            }
            compCaches[seq] = cache;
            compReaders[seq] = (row, dst, off) -> cache.readKSlice(row, 0, l.headDim, dst, off);
            idxKeyReaders[seq] = (row, dst, off) -> cache.readKSlice(row, l.headDim, l.idxDim, dst, off);
            if (l.ownsCompressed) {
                RecurrentStateAccessor planes = memory.compressorState(compReadLayer, kind, seq);
                if (planes == null) {
                    throw new IllegalStateException(l.where + ": no " + kind + " compressor state for KV source layer "
                            + compReadLayer + " stream " + seq);
                }
                stateKv[seq] = plane(planes, CpuMemory.COMPRESSOR_KV, planeSize, kind);
                stateScore[seq] = plane(planes, CpuMemory.COMPRESSOR_SCORE, planeSize, kind);
            }
        }
        if (l.ownsCompressed) {
            zeroApe = new float[l.ratio * l.headDim];
            compressor = bindV41Compressor(layer, weights, plan, stateKv, stateScore);
            idxK = weights.layer(layer, Dsv4Layout.IDX_K);
            if (idxK.cols() != l.headDim || idxK.rows() != l.idxDim) {
                throw new IllegalStateException(l.where + ": indexer.attn_k has shape [" + idxK.cols() + ", "
                        + idxK.rows() + "], expected [" + l.headDim + ", " + l.idxDim + "]");
            }
            idxKNormW = vector(weights.layerFloatsOrNull(layer, Dsv4Layout.IDX_K_NORM), l.idxDim,
                    Dsv4Layout.IDX_K_NORM);
            idxKDeltas = ctx.projectionDeltas().serve(layer, Dsv4Layout.IDX_K, idxK, matVec);
            compLatent = s.temp(TEMP_COMP_LATENT, ubatch * l.headDim);
            compIndex = s.temp(TEMP_COMP_INDEX, ubatch * l.idxDim);
            compRowsBuf = s.temp(TEMP_COMP_ROWS, ubatch * l.compWidth);
        }
        // one top-k plane per indexer source: consumers read their source's selection of this step
        topK = s.intTemp(topKName(l.v41.topkSourceLayer()), ubatch * l.topK);
        if (l.computesTopk) {
            indexer = new CpuDsaIndexerStage(ctx, l.where, l.indexerConfig(0));
            indexer.bindQueries(weights.layer(layer, Dsv4Layout.IDX_Q_B),
                    weights.layer(layer, Dsv4Layout.IDX_PROJ), null);
            indexer.bindSelection(idxKeyReaders[0], compRows, compRowPosition);
            indexer.bindDeltas(layer);
        }
        int cand = l.v41.candidateSourceLayer();
        if (cand >= 0) {
            candTopKBlocks = l.v41.candidateTopKBlocks();
            candBlockSize = l.v41.candidateBlockSize();
            picksCandidate = l.computesTopk && l.v41.picksCandidateBlocks();
            usesCandidate = l.computesTopk && l.v41.usesCandidateBlocks();
            if (picksCandidate || usesCandidate) {
                // level one keeps the candidate blocks for every consumer of the same source
                candBlocks = s.intTemp(candName(cand), ubatch * candTopKBlocks);
            }
        }
    }

    /** Scratch name of the top-k selection plane of indexer source layer {@code il}. */
    private static String topKName(int il) {
        return GpuDsv4Mixer.TEMP_TOPK + ".src." + il;
    }

    /** Scratch name of the candidate-block plane of candidate source layer {@code il}. */
    private static String candName(int il) {
        return TEMP_CAND + ".src." + il;
    }

    /** The compressed plan of {@code kind} at {@code il}. */
    private CompressedKvPlan planOf(int il, CompressedKvPlan.Kind kind) {
        for (CompressedKvPlan p : memory.plan().layer(il).compressed()) {
            if (p.kind() == kind) {
                return p;
            }
        }
        throw new IllegalStateException(layout.where + ": KV source layer " + il + " has no " + kind
                + " compressed cache");
    }

    /**
     * The DeepSeek-V4.1 compressor of a KV source layer: pooled rows of {@code headDim} values with no APE, the
     * index-key projection of each compressed latent, and (above ratio 1) the score gate. The combined rows
     * {@code [compressed KV | index key]} are written to the source layer's cache.
     */
    private Compressor bindV41Compressor(LayerSpec layer, CpuWeights weights, CompressedKvPlan plan,
                                         float[][] stateKv, float[][] stateScore) {
        Dsv4Layout l = layout;
        int ratio = plan.ratio();
        float[] norm = vector(weights.layerFloatsOrNull(layer, Dsv4Layout.COMP_NORM), l.headDim, Dsv4Layout.COMP_NORM);
        Dsv4Kernels.checkWidth(l.headDim, plan.dtype());
        Dsv4Kernels.checkRope(l.compressParams, l.headDim);
        CpuDsv4Compressor.Config cfg = CpuDsv4Compressor.Config.of(ratio, l.headDim, l.rmsEps, l.compressParams, 0);
        CpuTensorRef wKv = weights.layer(layer, Dsv4Layout.COMP_WKV);
        CpuTensorRef wGate = ratio > 1 ? weights.layer(layer, Dsv4Layout.COMP_WGATE) : null;
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        CpuProjectionDeltas.Site[] sites = {deltas.serve(layer, Dsv4Layout.COMP_WKV, wKv, matVec),
                wGate != null ? deltas.serve(layer, Dsv4Layout.COMP_WGATE, wGate, matVec) : null};
        Streams streams = new Streams(compCaches, compReaders, stateKv, stateScore);
        return new Compressor(plan.kind(), cfg, plan.rows(), plan.snapshots(), wKv, wGate, sites, zeroApe, norm,
                streams, false, true, wGate != null);
    }

    /**
     * The CSA / HCA compressor ({@code attn_compressor_*}) or the indexer compressor ({@code indexer_compressor_*})
     * of a compressed cache kind, with the cache and planes of every stream checked against the layout.
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
        if (plan == null) {
            throw new IllegalStateException(l.where + ": no " + kind + " compressed cache for the DeepSeek-V4 layer");
        }
        l.checkCompressedPlan(plan, rawType);
        int ratio = plan.ratio();
        int width = plan.width();
        Streams streams = bindStreams(plan, kind);
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
        return new Compressor(kind, cfg, plan.rows(), plan.snapshots(), wKv, wGate, sites, ape, norm, streams, lid,
                false, true);
    }

    /** The cache and planes of every stream of a compressed kind, checked against its plan. */
    private Streams bindStreams(CompressedKvPlan plan, CompressedKvPlan.Kind kind) {
        Dsv4Layout l = layout;
        int planeSize = plan.planes() * plan.planeFloats();
        Streams st = new Streams(plan.streams());
        for (int seq = 0; seq < plan.streams(); seq++) {
            KvAccessor cache = memory.compressed(l.il, kind, seq);
            RecurrentStateAccessor planes = memory.compressorState(l.il, kind, seq);
            if (cache == null || planes == null) {
                throw new IllegalStateException(l.where + ": no " + kind + " compressed stream " + seq
                        + " for the DeepSeek-V4 layer");
            }
            if (cache.nHeadKv() != 1 || cache.headDimK() != plan.width() || cache.layout().rows() != plan.rows()) {
                throw new IllegalStateException(l.where + ": the " + kind + " cache must hold " + plan.rows()
                        + " single-head rows of " + plan.width() + " values, got " + cache.layout());
            }
            st.caches[seq] = cache;
            st.readers[seq] = RowReader.ofKeys(cache);
            st.stateKv[seq] = plane(planes, CpuMemory.COMPRESSOR_KV, planeSize, kind);
            st.stateScore[seq] = plane(planes, CpuMemory.COMPRESSOR_SCORE, planeSize, kind);
        }
        return st;
    }

    /** The per-sequence caches, key readers and planes of one compressed kind. */
    private static final class Streams {
        final KvAccessor[] caches;
        final RowReader[] readers;
        final float[][] stateKv;
        final float[][] stateScore;

        Streams(int n) {
            caches = new KvAccessor[n];
            readers = new RowReader[n];
            stateKv = new float[n][];
            stateScore = new float[n][];
        }

        Streams(KvAccessor[] caches, RowReader[] readers, float[][] stateKv, float[][] stateScore) {
            this.caches = caches;
            this.readers = readers;
            this.stateKv = stateKv;
            this.stateScore = stateScore;
        }
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
     * The cache writes of a step: the raw KV projection with its norm and RoPE, the compressors' folds and the raw
     * store. The query projections, the indexer's selection, the attention and the output projection do not run.
     */
    @Override
    public void prefillKV(int pos0, int n, float[] in) {
        ensureBound();
        Dsv4Layout l = layout;
        if (n < 1 || n > ubatch || pos0 < 0) {
            throw new IllegalArgumentException(l.where + ": DeepSeek-V4 step of " + n + " tokens at " + pos0
                    + " is outside [1, " + ubatch + "] tokens at a non-negative position");
        }
        checkCapacity(pos0, n);
        if (Objects.requireNonNull(in, "in").length < (long) n * l.nEmbd) {
            throw new IllegalArgumentException(l.where + ": DeepSeek-V4 step array holds " + in.length + " floats, "
                    + (long) n * l.nEmbd + " are needed");
        }
        KvLayout kvl = raw.layout();
        for (int t = 0; t < n; t++) {
            cells[t] = kvl.rowOf(pos0 + t);
            qPos[t] = pos0 + t;
        }
        projectKV(n, in);
        CpuDsv4Compressor.cacheStore(n, l.headDim, rawRot, kv, 0, l.headDim, cells, raw);
        if (compressor != null) {
            compressor.run(0, pos0, n, in, 0);
        }
        if (lidCompressor != null) {
            lidCompressor.run(0, pos0, n, in, 0);
        }
    }

    @Override
    public String kvRefusal() {
        ensureBound();
        return null;
    }

    /** The raw KV row alone: its projection with its delta, RMS norm and RoPE. */
    private void projectKV(int n, float[] x) {
        Dsv4Layout l = layout;
        RopeSpec r = l.attnRope;
        RopePositions.fill(r.type(), r.textPositions(), qPos, 0, n, ropePos);
        matVec.matmul(wkv, x, 0, kv, 0, n);
        kvDeltas.applyRows(n, x, 0, kv, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.headDim, l.headDim, 1, kv, 0, kvNormW, null, kv, 0);
        CpuRope.apply(ropeTable, kv, 0, n, 1, l.headDim, l.headDim, l.nope, ropePos, 0, n);
    }

    /** Twin of {@link GpuDsv4Mixer#shiftRefusal()}. */
    @Override
    public String shiftRefusal() {
        return "context shift is not supported for DeepSeek-V4 compressed attention (llama-kv-cache-dsv4.cpp:1452)";
    }

    /**
     * A paged multi-sequence step: every segment's rows run through their own sequence's compressed streams and raw
     * cells (see the class description); without a current paged step the contiguous single-segment form runs.
     *
     * @throws UnsupportedOperationException for a non-causal step or a segment whose rows are not at consecutive
     *                                       positions
     */
    @Override
    public void batch(StepLayout s, float[] in, float[] out) {
        ensureBound();
        PagedStep step = memory.currentPagedStep();
        if (step == null) {
            CpuMixer.super.batch(s, in, out);
            return;
        }
        forwardPaged(s, step, in, out);
    }

    @Override
    public String batchRefusal() {
        return "the DeepSeek-V4 mixer mixes several sequences per step only over paged memory (a currentPagedStep)";
    }

    private void forward(int pos0, int n, float[] x, float[] out) {
        ensureBound();
        checkStep(pos0, n, x, out);
        KvLayout kvl = raw.layout();
        for (int t = 0; t < n; t++) {
            cells[t] = kvl.rowOf(pos0 + t);
            qPos[t] = pos0 + t;
        }
        project(n, x);
        CpuDsv4Compressor.cacheStore(n, layout.headDim, rawRot, kv, 0, layout.headDim, cells, raw);
        int nRaw = Math.min(rawRows, pos0 + n);
        int group = raw.group();
        for (int row = 0; row < nRaw; row++) {
            rawPos[row] = memory.positionOfRow(group, row);
        }
        attendSequence(0, 0, n, null, nRaw, x);
        output(n, out);
    }

    private void forwardPaged(StepLayout s, PagedStep step, float[] x, float[] out) {
        Dsv4Layout l = layout;
        int n = s.nTokens();
        if (ctx.stepNonCausal()) {
            throw new UnsupportedOperationException(l.where + ": DeepSeek-V4 attention and its compressors are"
                    + " causal; a non-causal step is not supported");
        }
        int[] pos = ctx.scratch().ints(CpuScratch.POS);
        for (int seg = 0; seg < s.nSeg(); seg++) {
            checkSegment(s, seg, pos);
        }
        checkStep(s.pos0(0), n, x, out);
        System.arraycopy(pos, 0, qPos, 0, n);
        project(n, x);
        int group = raw.group();
        CpuDsv4Compressor.cacheStore(n, l.headDim, rawRot, kv, 0, l.headDim, step.cellIdx(group), raw);
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int seq = s.seq(seg);
            int nRaw = step.seqCells(group, seq, rawCells);
            for (int i = 0; i < nRaw; i++) {
                rawPos[i] = memory.positionOfRow(group, rawCells[i]);
            }
            attendSequence(seq, s.tokStart(seg), s.len(seg), rawCells, nRaw, x);
        }
        output(n, out);
    }

    /**
     * The token-wise stages over the step's {@code n} rows at {@code qPos}: the low-rank query with its norm, the
     * per-head query norm and RoPE, the raw K/V row with its norm and RoPE, and the indexer queries of a CSA layer.
     */
    private void project(int n, float[] x) {
        Dsv4Layout l = layout;
        int qWidth = l.nHead * l.headDim;
        matVec.matmul(wqA, x, 0, qr, 0, n);
        qaDeltas.applyRows(n, x, 0, qr, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.qLora, l.qLora, 1, qr, 0, qANormW, null, qr, 0);
        matVec.matmul(wqB, qr, 0, q, 0, n);
        qbDeltas.applyRows(n, qr, 0, q, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n * l.nHead, l.headDim, l.headDim, 1, q, 0, null, null, q, 0);
        RopeSpec r = l.attnRope;
        RopePositions.fill(r.type(), r.textPositions(), qPos, 0, n, ropePos);
        CpuRope.apply(ropeTable, q, 0, n, l.nHead, l.headDim, qWidth, l.nope, ropePos, 0, n);
        matVec.matmul(wkv, x, 0, kv, 0, n);
        kvDeltas.applyRows(n, x, 0, kv, 0);
        NormMath.rowNorm(NormKind.RMS, l.rmsEps, n, l.headDim, l.headDim, 1, kv, 0, kvNormW, null, kv, 0);
        CpuRope.apply(ropeTable, kv, 0, n, 1, l.headDim, l.headDim, l.nope, ropePos, 0, n);
        if (l.computesTopk) {
            indexer.queries(qPos, n, qr, 0, x, 0);
        }
    }

    /**
     * One sequence's rows {@code [t0, t0 + len)}: its compressors fold the rows into its streams, a CSA layer's
     * indexer selects among its compressed rows, and the rows attend its raw keys ({@code rawList[0, nRaw)}, or rows
     * {@code [0, nRaw)} when null, at {@code rawPos}) and its compressed keys.
     */
    private void attendSequence(int seq, int t0, int len, int[] rawList, int nRaw, float[] x) {
        Dsv4Layout l = layout;
        int pos0 = qPos[t0];
        int last = pos0 + len - 1;
        CpuDsv4Attention.Compressed comp = CpuDsv4Attention.Compressed.none();
        if (compressor != null) {
            compressor.run(seq, pos0, len, x, t0);
        }
        if (v41) {
            comp = attendV41(seq, t0, len, last);
        } else if (l.mode == CompMode.SELECTED) {
            lidCompressor.run(seq, pos0, len, x, t0);
            int nLid = Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_CSA);
            int k = 0;
            if (nLid > 0) {
                k = indexer.select(t0, len, qPos, nLid, lidCompressor.readers[seq], lidRowPosition,
                        indexer.config().mask().causal(), topK, 0);
            }
            comp = CpuDsv4Attention.Compressed.selected(compressor.caches[seq], k, topK, 0, k);
        } else if (l.mode == CompMode.DENSE) {
            comp = CpuDsv4Attention.Compressed.dense(compressor.caches[seq],
                    Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_HCA));
        }
        System.arraycopy(qPos, t0, segPos, 0, len);
        int qWidth = l.nHead * l.headDim;
        CpuDsv4Attention.attend(attnLayer, len, q, t0 * qWidth, segPos, raw, rawList, nRaw, rawPos, comp, attn,
                t0 * qWidth);
    }

    /**
     * The DeepSeek-V4.1 compressed keys of one sequence (models/deepseek41.cpp:520-744): the KV source's visible
     * compressed rows, of which the attention keeps the indexer's top-k. The indexer source of the layer computes
     * that selection, inside the candidate blocks of the two-level selection when the layer is a candidate consumer;
     * every other compressed layer reuses its source's selection. Both ratios select their rows, so the result is a
     * {@code SELECTED} key set.
     */
    private CpuDsv4Attention.Compressed attendV41(int seq, int t0, int len, int last) {
        Dsv4Layout l = layout;
        if (l.ratio == 0) {
            return CpuDsv4Attention.Compressed.none();
        }
        int nVis = Dsv4Kernels.visibleRows(last, l.ratio);
        if (nVis == 0) {
            return CpuDsv4Attention.Compressed.none();
        }
        int k = SparseKernels.selectedCount(nVis, l.topK);
        if (l.computesTopk) {
            int off = t0 * k;
            if (usesCandidate && candBlocks != null) {
                int nBlk = (nVis + candBlockSize - 1) / candBlockSize;
                int stride = Math.min(candTopKBlocks, nBlk);
                indexer.selectInsideBlocks(t0, len, qPos, nVis, idxKeyReaders[seq], compRowPosition,
                        indexer.config().mask().causal(), candBlockSize, candBlocks, t0 * stride, stride, topK, off);
            } else {
                indexer.select(t0, len, qPos, nVis, idxKeyReaders[seq], compRowPosition,
                        indexer.config().mask().causal(), topK, off);
            }
            if (picksCandidate && candBlocks != null) {
                int nBlk = (nVis + candBlockSize - 1) / candBlockSize;
                int stride = Math.min(candTopKBlocks, nBlk);
                indexer.selectBlocks(t0, len, qPos, nVis, idxKeyReaders[seq], compRowPosition,
                        indexer.config().mask().causal(), l.ratio, candBlockSize, candTopKBlocks, candBlocks,
                        t0 * stride);
            }
            sortSelection(topK, off, len, k);
        }
        return CpuDsv4Attention.Compressed.selected(compReaders[seq], compRows, k, topK, t0 * k, k);
    }

    /**
     * Sorts the valid rows of each token's selection ascending and moves the {@code -1} padding behind them, the row
     * order the DeepSeek-V4.1 gather path attends in (deepseek41.cpp:659-663); the set is unchanged.
     */
    private static void sortSelection(int[] selection, int off, int rows, int k) {
        for (int t = 0; t < rows; t++) {
            int o = off + t * k;
            Arrays.sort(selection, o, o + k);
            int negative = 0;
            while (negative < k && selection[o + negative] < 0) {
                negative++;
            }
            if (negative == 0) {
                continue;
            }
            System.arraycopy(selection, o + negative, selection, o, k - negative);
            Arrays.fill(selection, o + k - negative, o + k, -1);
        }
    }

    /** The grouped {@code attn_output_a} and {@code attn_output_b} over the step's {@code n} attention rows. */
    private void output(int n, float[] out) {
        Dsv4Layout l = layout;
        try (TapProduct groups = TapProduct.dense(n)) {
            for (int pair = 0; pair < n * l.nGroups; pair++) {
                matVec.matvecExpert(woA, pair % l.nGroups, l.nGroups, attn, pair * l.oGroupDim, oa,
                        pair * l.oLora);
            }
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
        checkCapacity(pos0, n);
        long width = (long) n * l.nEmbd;
        if (Objects.requireNonNull(in, "in").length < width || Objects.requireNonNull(out, "out").length < width) {
            throw new IllegalArgumentException(l.where + ": DeepSeek-V4 step arrays hold " + in.length + " / "
                    + out.length + " floats, " + width + " are needed");
        }
    }

    private void checkCapacity(int pos0, int n) {
        if (compressor != null) {
            layout.checkCapacity(pos0, n, compressor.cfg.ratio(), compressor.rows, compressor.kind.toString());
        }
        if (lidCompressor != null) {
            layout.checkCapacity(pos0, n, lidCompressor.cfg.ratio(), lidCompressor.rows,
                    lidCompressor.kind.toString());
        }
    }

    /**
     * Checks one segment of a paged step: its sequence has a compressed stream, its rows sit at consecutive
     * positions from its {@code pos0}, and its compressed rows fit the streams.
     */
    private void checkSegment(StepLayout s, int seg, int[] pos) {
        Dsv4Layout l = layout;
        int seq = s.seq(seg);
        Compressor c = compressor != null ? compressor : lidCompressor;
        int streams = c != null ? c.caches.length : compCaches != null ? compCaches.length : 0;
        if (streams > 0 && seq >= streams) {
            throw new IllegalStateException(l.where + ": sequence " + seq + " has no compressed stream (the memory"
                    + " keeps " + streams + ")");
        }
        int t0 = s.tokStart(seg);
        for (int t = 0; t < s.len(seg); t++) {
            if (pos[t0 + t] != s.pos0(seg) + t) {
                throw new UnsupportedOperationException(l.where + ": row " + (t0 + t) + " of sequence " + seq
                        + " is at position " + pos[t0 + t] + ", not " + (s.pos0(seg) + t) + "; the DeepSeek-V4"
                        + " compressors fold consecutive positions (a media chunk sharing positions is refused)");
            }
        }
        checkCapacity(s.pos0(seg), s.len(seg));
    }

    /** One compressor of the layer: its weights and the cache and planes of every stream. */
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
        /** Per stream (one per paged sequence): the compressed cache, its key reader and the two planes. */
        final KvAccessor[] caches;
        final RowReader[] readers;
        final float[][] stateKv;
        final float[][] stateScore;
        final boolean lid;
        /** DeepSeek-V4.1: combined {@code [compressed KV | index key]} rows, no Hadamard on the compressed row. */
        final boolean v41;
        /** True when the compressor projects its own scores ({@code attn_compressor_gate}). */
        final boolean gated;

        Compressor(CompressedKvPlan.Kind kind, CpuDsv4Compressor.Config cfg, int rows, int snapshots,
                   CpuTensorRef wKv, CpuTensorRef wGate, CpuProjectionDeltas.Site[] deltas, float[] ape, float[] norm,
                   Streams streams, boolean lid, boolean v41, boolean gated) {
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
            this.caches = streams.caches;
            this.readers = streams.readers;
            this.stateKv = streams.stateKv;
            this.stateScore = streams.stateScore;
            this.lid = lid;
            this.v41 = v41;
            this.gated = gated;
        }

        /**
         * Twin of {@code GpuDsv4Mixer.Compressor.run} over stream {@code seq}: project the step rows
         * {@code [row0, row0 + n)} at positions {@code [pos0, pos0 + n)}, compress the completed blocks, write the
         * rollback snapshots, persist the planes. A DeepSeek-V4.1 source additionally derives the index key of every
         * completed block and stores the combined row.
         */
        void run(int seq, int pos0, int n, float[] x, int row0) {
            int xOff = row0 * layout.nEmbd;
            matVec.matmul(wKv, x, xOff, curKv, 0, n);
            kvDeltas.applyRows(n, x, xOff, curKv, 0);
            if (gated) {
                matVec.matmul(wGate, x, xOff, curScore, 0, n);
                gateDeltas.applyRows(n, x, xOff, curScore, 0);
            } else {
                Arrays.fill(curScore, 0, n * cfg.stateWidth(), 0f);
            }
            CpuDsv4Compressor.Step step = new CpuDsv4Compressor.Step(pos0, n, curKv, 0, curScore, 0, ape);
            float[] kvPlane = stateKv[seq];
            float[] scorePlane = stateScore[seq];
            if (Dsv4Kernels.blockCount(pos0, n, cfg.ratio()) > 0) {
                if (v41) {
                    compressV41(seq, step, kvPlane, scorePlane);
                } else if (lid) {
                    CpuDsv4Compressor.lidUpdate(cfg, step, kvPlane, scorePlane, norm, caches[seq]);
                } else {
                    CpuDsv4Compressor.compress(cfg, step, kvPlane, scorePlane, norm, caches[seq]);
                }
            }
            if (snapshots > 0) {
                CpuDsv4Compressor.stateSnapshot(cfg, step, kvPlane, scorePlane, snapshots);
            }
            CpuDsv4Compressor.stateUpdate(cfg, step, kvPlane, scorePlane);
        }

        /**
         * The DeepSeek-V4.1 compressed rows of one stream (models/deepseek41.cpp:480-501): the pooled, normed latent
         * without RoPE, its index key through {@code indexer.attn_k}, its norm and the index-key RoPE, then the
         * latent RoPE and the combined row {@code [latent | index key]} into the source cache. Both rotations use the
         * block's first position {@code b * ratio} and the compress-base frequency.
         */
        private void compressV41(int seq, CpuDsv4Compressor.Step step, float[] kvPlane, float[] scorePlane) {
            Dsv4Layout l = layout;
            int nBlocks = Dsv4Kernels.blockCount(step.pos0(), step.nTokens(), cfg.ratio());
            int b0 = Dsv4Kernels.firstBlock(step.pos0(), cfg.ratio());
            CpuDsv4Compressor.compressUnroped(cfg, step, kvPlane, scorePlane, norm, compLatent);
            for (int g = 0; g < nBlocks; g++) {
                int b = b0 + g;
                int pos = b * cfg.ratio();
                int lOff = g * l.headDim;
                matVec.matmul(idxK, compLatent, lOff, compIndex, 0, 1);
                idxKDeltas.applyRows(1, compLatent, lOff, compIndex, 0);
                NormMath.rowNorm(NormKind.RMS, l.rmsEps, 1, l.idxDim, l.idxDim, 1, compIndex, 0, idxKNormW, null,
                        compIndex, 0);
                CpuDsv4Compressor.rope(compIndex, 0, l.idxDim, l.indexerParams, pos, false);
                CpuDsv4Compressor.rope(compLatent, lOff, l.headDim, l.compressParams, pos, false);
                if (rawRot > 0) {
                    // the raw window is stored rotated for a quantized cache; the compressed rows join that space
                    CpuDsv4Compressor.hadamard(compLatent, lOff, l.headDim, rawRot);
                }
                int rOff = g * l.compWidth;
                System.arraycopy(compLatent, lOff, compRowsBuf, rOff, l.headDim);
                System.arraycopy(compIndex, 0, compRowsBuf, rOff + l.headDim, l.idxDim);
                caches[seq].writeK(b, compRowsBuf, rOff);
            }
        }
    }

    // ------------------------------------------------------------------ bind helpers

    /** Twin of {@code GpuDsv4Mixer.perGroup}: the grouped {@code attn_output_a} without input transform or scale. */
    private CpuTensorRef perGroup(CpuTensorRef w) {
        CpuMatVec.Transforms t = matVec.transformsOf(w);
        if (!t.transform().isIdentity() || t.hasScale()) {
            throw new UnsupportedOperationException(layout.where + ": grouped tensor " + w + " carries an input"
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
