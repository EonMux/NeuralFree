package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexer.RowReader;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.IndexerConfig;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU twin of {@link GpuDsaIndexer}: the same three stages with the kernels' CPU math. Queries and weights use
 * {@link CpuMatVec}, {@link CpuRope#apply} ({@code LLM_RoPE_Ext}), {@link CpuHadamard} ({@code LLM_HadamardRotate})
 * and {@link CpuOps#scale}; keys use the {@link NormMath#rowNorm} layer norm ({@code LLM_RowNorm}) and
 * {@link KvAccessor#writeK} into the indexer cache in its element type ({@code LLM_KV_Store_<KV>}); selection uses
 * {@link CpuDsaIndexer#scores} ({@code LLM_DSA_IndexerScore_<KV>}) and {@link CpuTopK#topKRows}
 * ({@code LLM_TopK_Rows}), scored in chunks of {@link #chunkTokens()} tokens exactly as on the GPU.
 *
 * <p>Working arrays are {@link CpuScratch} temporaries taken at bind time; a step allocates nothing. Queries may come
 * from the {@code q_lora} row or, with {@link #bindQueryNorm}, from the layer-normed input; {@link #bindSsmax} applies
 * mistral-large4's per-head SSMax scale and both rotations read {@code IndexerConfig.rope()}'s frequency factors.
 * Not thread-safe.</p>
 */
public final class CpuDsaIndexerStage {

    public static final String TEMP_QUERIES = GpuDsaIndexer.TEMP_QUERIES;
    public static final String TEMP_WEIGHTS = GpuDsaIndexer.TEMP_WEIGHTS;
    public static final String TEMP_KEYS = GpuDsaIndexer.TEMP_KEYS;
    public static final String TEMP_SCORES = GpuDsaIndexer.TEMP_SCORES;
    public static final String TEMP_POSITIONS = "dsa.indexer.pos";
    public static final String TEMP_Q_POS = "dsa.indexer.qpos";
    public static final String TEMP_KV_POS = "dsa.indexer.kvpos";
    public static final String TEMP_BLOCK_SCORES = "dsa.indexer.blocks";
    public static final String TEMP_Q_INPUT = "dsa.indexer.qin";

    private final CpuForwardContext ctx;
    private final IndexerConfig cfg;
    private final String where;
    private final int ubatch;
    private final CpuMatVec matVec;
    private final HadamardSpec rotation;
    private final float[] queries;
    private final float[] weights;
    private final int[] ropePos;

    private RopeFreqTable ropeTable;
    private CpuTensorRef qB;
    private CpuTensorRef proj;
    /** {@code indexer.q_norm} weight and bias over the layer input; null when queries come from the q_lora row. */
    private float[] queryNormW;
    private float[] queryNormB;
    private float[] queryNormInput;
    /** Per-head SSMax scale and bias ({@code indexer.ssmax}); null when the query is not scaled. */
    private float[] ssmaxW;
    private float[] ssmaxB;

    private CpuTensorRef kProj;
    /**
     * LoRA deltas of the three indexer projections ({@code build_lora_mm}, deepseek4.cpp:609, 622); served by
     * {@link #bindDeltas}, which the steps require.
     */
    private CpuProjectionDeltas.Site qBDeltas;
    private CpuProjectionDeltas.Site projDeltas;
    private CpuProjectionDeltas.Site kProjDeltas;
    private float[] keyNormW;
    private float[] keyNormB;
    private KvAccessor keyCache;
    private float[] keyRowsTemp;

    private RowReader keyReader;
    private IntUnaryOperator rowPosition;
    private int keyRows;
    private int chunk;
    private float[] scores;
    private float[] blockScores;
    private int[] qPos;
    private int[] kvPos;
    /** The cache rows of the slots {@link #selectSlots} scores, read through {@link #slotReader} meanwhile. */
    private int[] columnRows;
    private RowReader slotReader;

    /**
     * An unbound stage holding its query and weight arrays for the context's {@code ubatch}.
     *
     * @param where context of error messages (architecture and layer)
     */
    public CpuDsaIndexerStage(CpuForwardContext ctx, String where, IndexerConfig cfg) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.where = Objects.requireNonNull(where, "where");
        this.cfg = Objects.requireNonNull(cfg, "cfg");
        this.ubatch = ctx.scratch().ubatch();
        this.matVec = ctx.matVec();
        this.rotation = cfg.rotation();
        CpuScratch s = ctx.scratch();
        this.queries = s.temp(TEMP_QUERIES, ubatch * cfg.queryWidth());
        this.weights = s.temp(TEMP_WEIGHTS, ubatch * cfg.nHeads());
        RopeSpec r = cfg.rope();
        this.ropePos = r != null ? s.intTemp(TEMP_POSITIONS, r.positionsPerToken() * ubatch) : null;
    }

    public IndexerConfig config() {
        return cfg;
    }

    /** Tokens scored per chunk ({@link GpuDsaIndexer#SCORE_BUDGET_FLOATS}); 0 before {@link #bindSelection}. */
    public int chunkTokens() {
        return chunk;
    }

    /** The rotated, roped queries {@code [ubatch][n_heads][head_dim]} written by {@link #queries}. */
    public float[] queryArray() {
        return queries;
    }

    /** The scaled head weights {@code [ubatch][n_heads]} written by {@link #queries}. */
    public float[] weightArray() {
        return weights;
    }

    // ------------------------------------------------------------------ bind

    /**
     * Binds the query stage ({@code indexer.attn_q_b}, {@code indexer.proj}).
     *
     * @param qB          {@code indexer.attn_q_b}: {@code n_heads * head_dim} rows over the {@code q_lora} row or,
     *                    with {@link #bindQueryNorm}, over the layer-normed input
     * @param ropeFactors per-pair divisors of the indexer RoPE ({@code rope_freqs}) or null
     * @throws IllegalStateException when already bound or a tensor has the wrong row count
     */
    public void bindQueries(CpuTensorRef qB, CpuTensorRef proj, float[] ropeFactors) {
        if (this.qB != null) {
            throw new IllegalStateException(where + ": indexer query stage already bound");
        }
        rows(qB, cfg.queryWidth(), "indexer.attn_q_b");
        rows(proj, cfg.nHeads(), "indexer.proj");
        if (queryNormW != null && qB.cols() != queryNormW.length) {
            throw new IllegalStateException(where + ": indexer.attn_q_b reads " + qB.cols() + " values, the layernormed"
                    + " input holds " + queryNormW.length);
        }
        RopeSpec r = cfg.rope();
        if (r != null && r.freqFactors() != null && ropeFactors == null) {
            throw new IllegalStateException(where + ": the indexer RoPE reads the frequency factors "
                    + r.freqFactors().name() + ", none were bound");
        }
        this.ropeTable = r != null ? RopeFreqTable.of(r, ropeFactors) : null;
        this.qB = qB;
        this.proj = proj;
    }

    /**
     * Binds the layer norm that produces the indexer queries from the layer input ({@code indexer.q_norm}): the
     * queries are then {@code indexer.attn_q_b . LayerNorm(x)} instead of {@code indexer.attn_q_b . q_lora}
     * (mistral-large4.cpp:232-235).
     *
     * @param normW {@code indexer.q_norm.weight}, one value per layer-input dim
     * @param normB {@code indexer.q_norm.bias}, one value per layer-input dim
     * @throws IllegalStateException when already bound, the vectors differ in length, or the query projection does
     *                               not read that width
     */
    public void bindQueryNorm(float[] normW, float[] normB) {
        if (queryNormW != null) {
            throw new IllegalStateException(where + ": indexer query norm already bound");
        }
        Objects.requireNonNull(normW, "indexer.q_norm.weight");
        Objects.requireNonNull(normB, "indexer.q_norm.bias");
        if (normW.length != normB.length || normW.length < 1) {
            throw new IllegalStateException(where + ": indexer.q_norm holds " + normW.length + " / " + normB.length
                    + " weight / bias values");
        }
        if (qB != null && qB.cols() != normW.length) {
            throw new IllegalStateException(where + ": indexer.attn_q_b reads " + qB.cols() + " values, the layernormed"
                    + " input holds " + normW.length);
        }
        this.queryNormW = normW;
        this.queryNormB = normB;
        this.queryNormInput = ctx.scratch().temp(TEMP_Q_INPUT, ubatch * normW.length);
    }

    /**
     * Scales the indexer queries by {@code weight[h] * log(pos + 1) + bias[h]} per head (SSMax,
     * mistral-large4.cpp:240).
     *
     * @param weight {@code indexer.ssmax.weight}, one value per indexer head
     * @param bias   {@code indexer.ssmax.bias}, one value per indexer head
     * @throws IllegalStateException when already bound or the vectors do not hold one value per head
     */
    public void bindSsmax(float[] weight, float[] bias) {
        if (ssmaxW != null) {
            throw new IllegalStateException(where + ": indexer SSMax already bound");
        }
        values(Objects.requireNonNull(weight, "indexer.ssmax.weight"), cfg.nHeads(), "indexer.ssmax.weight");
        values(Objects.requireNonNull(bias, "indexer.ssmax.bias"), cfg.nHeads(), "indexer.ssmax.bias");
        this.ssmaxW = weight;
        this.ssmaxB = bias;
    }

    /**
     * Binds the key stage over a single-head indexer cache with rows of {@code head_dim} values of its layout's K
     * type (whole 32-value blocks for a block type, {@link SparseKernels#checkIndexerCache}).
     *
     * @param normW {@code indexer.k_norm.weight} values, or null (glm-dsa loads it as optional)
     * @param normB {@code indexer.k_norm.bias} values, or null
     * @throws IllegalStateException when already bound or a tensor or the cache does not fit the configuration
     */
    public void bindKeys(CpuTensorRef kProj, float[] normW, float[] normB, KvAccessor cache) {
        if (this.kProj != null) {
            throw new IllegalStateException(where + ": indexer key stage already bound");
        }
        int dim = cfg.headDim();
        rows(kProj, dim, "indexer.attn_k");
        if (normW != null) {
            values(normW, dim, "indexer.k_norm.weight");
        }
        if (normB != null) {
            values(normB, dim, "indexer.k_norm.bias");
        }
        checkCache(cache);
        this.keyNormW = normW;
        this.keyNormB = normB;
        this.keyCache = cache;
        this.keyRowsTemp = ctx.scratch().temp(TEMP_KEYS, ubatch * dim);
        this.kProj = kProj;
    }

    /**
     * Serves the LoRA deltas of the bound projections ({@code indexer.attn_q_b}, {@code indexer.proj},
     * {@code indexer.attn_k}) of {@code layer} in the forward pass's {@link CpuProjectionDeltas}.
     *
     * @throws IllegalStateException when no projection is bound yet
     */
    public void bindDeltas(LayerSpec layer) {
        if (qB == null && kProj == null) {
            throw new IllegalStateException(where + ": bind the indexer projections before their LoRA deltas");
        }
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        if (qB != null) {
            qBDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.INDEXER_ATTN_Q_B), qB, matVec);
            projDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.INDEXER_PROJ), proj, matVec);
        }
        if (kProj != null) {
            kProjDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.INDEXER_ATTN_K), kProj, matVec);
        }
    }

    /**
     * Binds the selection stage.
     *
     * @param keys        reader of the key rows of {@code head_dim}
     * @param keyRows     rows the reader serves
     * @param rowPosition position held by a row, -1 when empty ({@code CpuMemory.positionOfRow} of the group)
     * @throws IllegalStateException when already bound or {@code keyRows} is not positive
     */
    public void bindSelection(RowReader keys, int keyRows, IntUnaryOperator rowPosition) {
        if (keyReader != null) {
            throw new IllegalStateException(where + ": indexer selection stage already bound");
        }
        if (keyRows < 1) {
            throw new IllegalStateException(where + ": the indexer cache needs at least one row, got " + keyRows);
        }
        this.keyReader = Objects.requireNonNull(keys, "keys");
        this.slotReader = (slot, dst, off) -> keys.read(columnRows[slot], dst, off);
        this.rowPosition = Objects.requireNonNull(rowPosition, "rowPosition");
        this.keyRows = keyRows;
        this.chunk = (int) Math.max(1L, Math.min(ubatch, GpuDsaIndexer.SCORE_BUDGET_FLOATS / keyRows));
        CpuScratch s = ctx.scratch();
        this.scores = s.temp(TEMP_SCORES, chunk * keyRows);
        this.blockScores = s.temp(TEMP_BLOCK_SCORES, chunk * keyRows);
        this.qPos = s.intTemp(TEMP_Q_POS, ubatch);
        this.kvPos = s.intTemp(TEMP_KV_POS, keyRows);
    }

    // ------------------------------------------------------------------ steps

    /** Projects the queries and head weights of {@code n} tokens at positions {@code pos0 ..}. */
    public void queries(int pos0, int n, float[] qr, int qrOff, float[] x, int xOff) {
        ensureBound(qB != null && qBDeltas != null, "query");
        checkTokens(n);
        if (ropeTable != null) {
            fillPositions(pos0, n);
        }
        projectQueries(n, qr, qrOff, x, xOff, pos0, null);
    }

    /**
     * Projects the queries and head weights of {@code n} tokens at the explicit positions {@code tokenPos[0, n)} (the
     * rows of a multi-sequence step, each at its own sequence's position).
     */
    public void queries(int[] tokenPos, int n, float[] qr, int qrOff, float[] x, int xOff) {
        ensureBound(qB != null && qBDeltas != null, "query");
        checkTokens(n);
        if (ropeTable != null) {
            fillPositions(tokenPos, n);
        }
        projectQueries(n, qr, qrOff, x, xOff, 0, tokenPos);
    }

    private void projectQueries(int n, float[] qr, int qrOff, float[] x, int xOff, int pos0, int[] tokenPos) {
        int heads = cfg.nHeads();
        int dim = cfg.headDim();
        if (queryNormW != null) {
            NormMath.rowNorm(NormKind.LAYER, cfg.keyNormEps(), n, queryNormW.length, queryNormW.length, 1, x, xOff,
                    queryNormW, queryNormB, queryNormInput, 0);
            matVec.matmul(qB, queryNormInput, 0, queries, 0, n);
            qBDeltas.applyRows(n, queryNormInput, 0, queries, 0);
        } else {
            matVec.matmul(qB, qr, qrOff, queries, 0, n);
            qBDeltas.applyRows(n, qr, qrOff, queries, 0);
        }
        if (ssmaxW != null) {
            applySsmax(n, heads, dim, pos0, tokenPos);
        }
        if (ropeTable != null) {
            CpuRope.apply(ropeTable, queries, 0, n, heads, dim, heads * dim, cfg.ropeOffset(), ropePos, 0, n);
        }
        if (rotation != null) {
            CpuHadamard.applyInPlace(rotation, queries, 0, n * heads, dim);
        }
        matVec.matmul(proj, x, xOff, weights, 0, n);
        projDeltas.applyRows(n, x, xOff, weights, 0);
        if (cfg.weightScale() != 1.0f) {
            CpuOps.scale(weights, 0, n * heads, cfg.weightScale());
        }
    }

    /** {@code queries[t][h][d] *= weight[h] * log(pos + 1) + bias[h]} (mistral-large4.cpp:240). */
    private void applySsmax(int n, int heads, int dim, int pos0, int[] tokenPos) {
        for (int t = 0; t < n; t++) {
            int pos = tokenPos != null ? tokenPos[t] : pos0 + t;
            double basis = Math.log(pos + 1.0);
            for (int h = 0; h < heads; h++) {
                float f = (float) (ssmaxW[h] * basis) + ssmaxB[h];
                int base = (t * heads + h) * dim;
                for (int d = 0; d < dim; d++) {
                    queries[base + d] *= f;
                }
            }
        }
    }

    /** Projects, normalizes, ropes, rotates and stores the keys of {@code n} tokens in their cache rows. */
    public void storeKeys(int pos0, int n, float[] x, int xOff) {
        ensureBound(kProj != null && kProjDeltas != null, "key");
        checkTokens(n);
        if (ropeTable != null) {
            fillPositions(pos0, n);
        }
        projectKeys(n, x, xOff);
        KvLayout layout = keyCache.layout();
        int dim = cfg.headDim();
        for (int t = 0; t < n; t++) {
            keyCache.writeK(layout.rowOf(pos0 + t), keyRowsTemp, t * dim);
        }
    }

    /**
     * Projects, normalizes, ropes, rotates and stores the keys of {@code n} tokens at the explicit positions
     * {@code tokenPos[0, n)} in the cache rows {@code rows[0, n)} (the step's cells of a paged memory).
     *
     * @throws IllegalArgumentException when {@code rows} holds fewer than {@code n} rows
     */
    public void storeKeys(int[] tokenPos, int[] rows, int n, float[] x, int xOff) {
        ensureBound(kProj != null && kProjDeltas != null, "key");
        checkTokens(n);
        if (Objects.requireNonNull(rows, "rows").length < n) {
            throw new IllegalArgumentException(where + ": " + rows.length + " key rows for " + n + " tokens");
        }
        if (ropeTable != null) {
            fillPositions(tokenPos, n);
        }
        projectKeys(n, x, xOff);
        int dim = cfg.headDim();
        for (int t = 0; t < n; t++) {
            keyCache.writeK(rows[t], keyRowsTemp, t * dim);
        }
    }

    private void projectKeys(int n, float[] x, int xOff) {
        int dim = cfg.headDim();
        float[] k = keyRowsTemp;
        matVec.matmul(kProj, x, xOff, k, 0, n);
        kProjDeltas.applyRows(n, x, xOff, k, 0);
        NormMath.rowNorm(NormKind.LAYER, cfg.keyNormEps(), n, dim, dim, 1, k, 0, keyNormW, keyNormB, k, 0);
        if (ropeTable != null) {
            CpuRope.apply(ropeTable, k, 0, n, 1, dim, dim, cfg.ropeOffset(), ropePos, 0, n);
        }
        if (rotation != null) {
            CpuHadamard.applyInPlace(rotation, k, 0, n, dim);
        }
    }

    /**
     * Scores the {@code n} queries at positions {@code pos0 ..} against rows {@code [0, nKv)} and writes token
     * {@code t}'s selection to {@code topK[topKOff + t*k ..]}.
     *
     * @return {@code k = min(nKv, topK)}
     * @throws IllegalArgumentException when {@code n} or {@code nKv} is outside its bound range
     */
    public int select(int pos0, int n, int nKv, int[] topK, int topKOff) {
        ensureBound(keyReader != null, "selection");
        checkTokens(n);
        checkColumns(nKv);
        for (int t = 0; t < n; t++) {
            qPos[t] = pos0 + t;
        }
        for (int row = 0; row < nKv; row++) {
            kvPos[row] = rowPosition.applyAsInt(row);
        }
        return selectColumns(0, n, nKv, keyReader, cfg.mask().causal(), topK, topKOff);
    }

    /**
     * Scores the queries of the step's tokens {@code [t0, t0 + n)} at positions {@code tokenPos[t0 ..]} against one
     * sequence's slots {@code [0, nSlots)}, slot {@code j} held by the bound cache's row {@code slotRows[j]} (a paged
     * memory's cells through the sequence's page table), and writes token {@code t}'s selected rows (not slots) to
     * {@code topK[topKOff + t*k ..]}. Ties go to the lowest slot, the order a contiguous sequence's rows have.
     *
     * @param causal whether a row after the query's position is masked (false for a non-causal step)
     * @return {@code k = min(nSlots, topK)}
     * @throws IllegalArgumentException when the tokens or slots are outside the bound ranges
     */
    public int selectSlots(int t0, int n, int[] tokenPos, int nSlots, int[] slotRows, boolean causal, int[] topK,
                           int topKOff) {
        ensureBound(keyReader != null, "selection");
        checkTokens(t0, n, tokenPos);
        checkColumns(nSlots);
        if (Objects.requireNonNull(slotRows, "slotRows").length < nSlots) {
            throw new IllegalArgumentException(where + ": " + slotRows.length + " slot rows for " + nSlots + " slots");
        }
        System.arraycopy(tokenPos, t0, qPos, 0, n);
        for (int j = 0; j < nSlots; j++) {
            kvPos[j] = rowPosition.applyAsInt(slotRows[j]);
        }
        columnRows = slotRows;
        try {
            int k = selectColumns(t0, n, nSlots, slotReader, causal, topK, topKOff);
            for (int i = topKOff, end = topKOff + n * k; i < end; i++) {
                if (topK[i] >= 0) {
                    topK[i] = slotRows[topK[i]];
                }
            }
            return k;
        } finally {
            columnRows = null;
        }
    }

    /**
     * Scores the queries of the step's tokens {@code [t0, t0 + n)} at positions {@code tokenPos[t0 ..]} against the
     * rows {@code [0, nKv)} of an explicit reader (one sequence's own key rows, as the DSV4 compressed indexer keys
     * of a paged memory are), row {@code r} at position {@code rowPosition(r)}, and writes token {@code t}'s
     * selection to {@code topK[topKOff + t*k ..]}.
     *
     * @param causal whether a row after the query's position is masked (false for a non-causal step)
     * @return {@code k = min(nKv, topK)}
     * @throws IllegalArgumentException when the tokens or rows are outside the bound ranges
     */
    public int select(int t0, int n, int[] tokenPos, int nKv, RowReader keys, IntUnaryOperator rowPosition,
                      boolean causal, int[] topK, int topKOff) {
        ensureBound(keyReader != null, "selection");
        checkTokens(t0, n, tokenPos);
        checkColumns(nKv);
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(rowPosition, "rowPosition");
        System.arraycopy(tokenPos, t0, qPos, 0, n);
        for (int row = 0; row < nKv; row++) {
            kvPos[row] = rowPosition.applyAsInt(row);
        }
        return selectColumns(t0, n, nKv, keys, causal, topK, topKOff);
    }

    /**
     * Scores and selects in chunks: the queries of tokens {@code [t0, t0 + n)} at {@code qPos[0, n)} against the
     * columns {@code [0, nCols)} of {@code reader} at {@code kvPos[0, nCols)}.
     */
    private int selectColumns(int t0, int n, int nCols, RowReader reader, boolean causal, int[] topK, int topKOff) {
        Objects.requireNonNull(topK, "topK");
        int k = SparseKernels.selectedCount(nCols, cfg.topK());
        KvWindow mask = cfg.mask();
        int heads = cfg.nHeads();
        for (int c0 = 0; c0 < n; c0 += chunk) {
            int c = Math.min(chunk, n - c0);
            int tok = t0 + c0;
            CpuDsaIndexer.scores(c, nCols, heads, cfg.headDim(), causal, mask.type(), mask.nSwa(),
                    queries, tok * cfg.queryWidth(), weights, tok * heads, reader, qPos, c0, kvPos, 0, scores, 0);
            CpuTopK.topKRows(scores, 0, c, nCols, k, topK, topKOff + c0 * k);
        }
        return k;
    }

    /**
     * Scores the queries of tokens {@code [t0, t0 + n)} against rows {@code [0, nKv)} and selects the top
     * {@code topKBlocks} blocks of {@code blockSize} consecutive rows, writing token {@code t}'s blocks to
     * {@code blocks[blocksOff + t*k ..]}. A block scores its best row's indexer score; the block holding the newest
     * compressed row visible to the query gets {@code +1e30}, so a partly filled block is always kept (deepseek41.cpp:
     * 588-597).
     *
     * @param ratio       compress ratio of the rows, the unit of {@link com.eonmux.neuralfree.gguf.ops.dsv4
     *                    .Dsv4Kernels#visibleRows}
     * @return {@code k = min(topKBlocks, ceil(nKv / blockSize))}
     * @throws IllegalArgumentException when the tokens, rows or block shape are outside their ranges
     */
    public int selectBlocks(int t0, int n, int[] tokenPos, int nKv, RowReader keys, IntUnaryOperator rowPosition,
                            boolean causal, int ratio, int blockSize, int topKBlocks, int[] blocks, int blocksOff) {
        ensureBound(keyReader != null, "selection");
        checkTokens(t0, n, tokenPos);
        checkColumns(nKv);
        if (ratio < 1 || blockSize < 1 || topKBlocks < 1) {
            throw new IllegalArgumentException(where + ": block selection needs a positive ratio, block size and block"
                    + " count, got " + ratio + "/" + blockSize + "/" + topKBlocks);
        }
        if (topKBlocks > SparseKernels.TOPK_MAX) {
            throw new IllegalArgumentException(where + ": " + topKBlocks + " candidate blocks exceed the top-k limit "
                    + SparseKernels.TOPK_MAX);
        }
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(rowPosition, "rowPosition");
        Objects.requireNonNull(blocks, "blocks");
        int nBlk = (nKv + blockSize - 1) / blockSize;
        int k = Math.min(topKBlocks, nBlk);
        if (blocksOff < 0 || (long) n * k > blocks.length - blocksOff) {
            throw new IllegalArgumentException(where + ": " + blocks.length + " block slots cannot hold " + n
                    + " rows of " + k + " at offset " + blocksOff);
        }
        System.arraycopy(tokenPos, t0, qPos, 0, n);
        for (int row = 0; row < nKv; row++) {
            kvPos[row] = rowPosition.applyAsInt(row);
        }
        for (int c0 = 0; c0 < n; c0 += chunk) {
            int c = Math.min(chunk, n - c0);
            int tok = t0 + c0;
            CpuDsaIndexer.scores(c, nKv, cfg.nHeads(), cfg.headDim(), causal, cfg.mask().type(), cfg.mask().nSwa(),
                    queries, tok * cfg.queryWidth(), weights, tok * cfg.nHeads(), keys, qPos, c0, kvPos, 0, scores, 0);
            for (int i = 0; i < c; i++) {
                int nVis = com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels.visibleRows(qPos[c0 + i], ratio);
                Dsv41Selection.blockTopK(scores, i * nKv, nKv, blockScores, blockSize, topKBlocks, nVis, blocks,
                        blocksOff + (c0 + i) * k);
            }
        }
        return k;
    }

    /**
     * Scores the queries of tokens {@code [t0, t0 + n)} against rows {@code [0, nKv)}, masks every row whose block is
     * not one of the query's candidate blocks and writes the top {@code cfg.topK()} rows to
     * {@code topK[topKOff + t*k ..]} (deepseek41.cpp:598-611).
     *
     * @param blocks       candidate block indices of the step's tokens
     * @param blocksStride slots per token in {@code blocks}
     * @return {@code k = min(nKv, topK)}
     * @throws IllegalArgumentException when the tokens, rows or block shape are outside their ranges
     */
    public int selectInsideBlocks(int t0, int n, int[] tokenPos, int nKv, RowReader keys,
                                  IntUnaryOperator rowPosition, boolean causal, int blockSize, int[] blocks,
                                  int blocksOff, int blocksStride, int[] topK, int topKOff) {
        ensureBound(keyReader != null, "selection");
        checkTokens(t0, n, tokenPos);
        checkColumns(nKv);
        if (blockSize < 1 || blocksStride < 0) {
            throw new IllegalArgumentException(where + ": block selection needs a positive block size and a"
                    + " non-negative stride, got " + blockSize + "/" + blocksStride);
        }
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(rowPosition, "rowPosition");
        Objects.requireNonNull(blocks, "blocks");
        Objects.requireNonNull(topK, "topK");
        int nBlk = (nKv + blockSize - 1) / blockSize;
        if (n > 0 && (blocksOff < 0 || (long) (n - 1) * blocksStride + blocksStride > blocks.length - blocksOff)) {
            throw new IllegalArgumentException(where + ": " + blocks.length + " block slots cannot hold " + n
                    + " rows of stride " + blocksStride + " at offset " + blocksOff);
        }
        int k = SparseKernels.selectedCount(nKv, cfg.topK());
        if (topKOff < 0 || (long) n * k > topK.length - topKOff) {
            throw new IllegalArgumentException(where + ": " + topK.length + " top-k slots cannot hold " + n
                    + " rows of " + k + " at offset " + topKOff);
        }
        System.arraycopy(tokenPos, t0, qPos, 0, n);
        for (int row = 0; row < nKv; row++) {
            kvPos[row] = rowPosition.applyAsInt(row);
        }
        int[] blockMask = new int[nBlk];
        for (int c0 = 0; c0 < n; c0 += chunk) {
            int c = Math.min(chunk, n - c0);
            int tok = t0 + c0;
            CpuDsaIndexer.scores(c, nKv, cfg.nHeads(), cfg.headDim(), causal, cfg.mask().type(), cfg.mask().nSwa(),
                    queries, tok * cfg.queryWidth(), weights, tok * cfg.nHeads(), keys, qPos, c0, kvPos, 0, scores, 0);
            for (int i = 0; i < c; i++) {
                Dsv41Selection.restrictToBlocks(scores, i * nKv, nKv, blockSize, blocks,
                        blocksOff + (c0 + i) * blocksStride, blocksStride, blockMask);
                CpuTopK.topKRow(scores, i * nKv, nKv, k, topK, topKOff + (c0 + i) * k);
            }
        }
        return k;
    }

    // ------------------------------------------------------------------ helpers

    private void fillPositions(int pos0, int n) {
        RopeSpec r = cfg.rope();
        RopePositions.fill(r.type(), r.textPositions(), pos0, n, ropePos);
    }

    private void fillPositions(int[] tokenPos, int n) {
        RopeSpec r = cfg.rope();
        RopePositions.fill(r.type(), r.textPositions(), tokenPos, 0, n, ropePos);
    }

    private void checkColumns(int nKv) {
        if (nKv < 1 || nKv > keyRows) {
            throw new IllegalArgumentException(where + ": indexer n_kv " + nKv + " is outside [1, " + keyRows + "]");
        }
    }

    private void checkTokens(int t0, int n, int[] tokenPos) {
        checkTokens(n);
        if (t0 < 0 || t0 + n > ubatch || Objects.requireNonNull(tokenPos, "tokenPos").length < t0 + n) {
            throw new IllegalArgumentException(where + ": indexer tokens [" + t0 + ", " + (t0 + n)
                    + ") are outside the step's " + ubatch + " rows or their positions");
        }
    }

    private void rows(CpuTensorRef w, int rows, String what) {
        Objects.requireNonNull(w, what);
        if (w.rows() != rows) {
            throw new IllegalStateException(where + ": " + what + " " + w + " has " + w.rows() + " rows, the indexer"
                    + " needs " + rows);
        }
    }

    private void values(float[] v, int n, String what) {
        if (v.length != n) {
            throw new IllegalStateException(where + ": " + what + " holds " + v.length + " values, the indexer needs "
                    + n);
        }
    }

    private void checkCache(KvAccessor cache) {
        Objects.requireNonNull(cache, "cache");
        if (cache.nHeadKv() != 1 || cache.headDimK() != cfg.headDim()) {
            throw new IllegalStateException(where + ": the indexer cache must hold one key of " + cfg.headDim()
                    + " values per row, got " + cache.nHeadKv() + " x " + cache.headDimK());
        }
        try {
            SparseKernels.checkIndexerCache(cache.layout().dtypeK(), cfg.headDim());
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException(where + ": " + e.getMessage(), e);
        }
    }

    private void checkTokens(int n) {
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException(where + ": indexer step of " + n + " tokens is outside [1, " + ubatch
                    + "]");
        }
    }

    private void ensureBound(boolean bound, String stage) {
        if (!bound) {
            throw new IllegalStateException(where + ": indexer " + stage + " stage or its LoRA deltas are not"
                    + " bound");
        }
    }

    @Override
    public String toString() {
        return "CpuDsaIndexerStage{" + where + ", " + cfg + ", chunk=" + chunk + '}';
    }
}
