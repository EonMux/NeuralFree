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
 * <p>Working arrays are {@link CpuScratch} temporaries taken at bind time; a step allocates nothing. Not
 * thread-safe.</p>
 */
public final class CpuDsaIndexerStage {

    public static final String TEMP_QUERIES = GpuDsaIndexer.TEMP_QUERIES;
    public static final String TEMP_WEIGHTS = GpuDsaIndexer.TEMP_WEIGHTS;
    public static final String TEMP_KEYS = GpuDsaIndexer.TEMP_KEYS;
    public static final String TEMP_SCORES = GpuDsaIndexer.TEMP_SCORES;
    public static final String TEMP_POSITIONS = "dsa.indexer.pos";
    public static final String TEMP_Q_POS = "dsa.indexer.qpos";
    public static final String TEMP_KV_POS = "dsa.indexer.kvpos";

    private final CpuForwardContext ctx;
    private final IndexerConfig cfg;
    private final String where;
    private final int ubatch;
    private final CpuMatVec matVec;
    private final HadamardSpec rotation;
    private final RopeFreqTable ropeTable;
    private final float[] queries;
    private final float[] weights;
    private final int[] ropePos;

    private CpuTensorRef qB;
    private CpuTensorRef proj;

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
    private int[] qPos;
    private int[] kvPos;

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
        this.ropeTable = r != null ? RopeFreqTable.of(r, null) : null;
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
     * @throws IllegalStateException when already bound or a tensor has the wrong row count
     */
    public void bindQueries(CpuTensorRef qB, CpuTensorRef proj) {
        if (this.qB != null) {
            throw new IllegalStateException(where + ": indexer query stage already bound");
        }
        rows(qB, cfg.queryWidth(), "indexer.attn_q_b");
        rows(proj, cfg.nHeads(), "indexer.proj");
        this.qB = qB;
        this.proj = proj;
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
        this.rowPosition = Objects.requireNonNull(rowPosition, "rowPosition");
        this.keyRows = keyRows;
        this.chunk = (int) Math.max(1L, Math.min(ubatch, GpuDsaIndexer.SCORE_BUDGET_FLOATS / keyRows));
        CpuScratch s = ctx.scratch();
        this.scores = s.temp(TEMP_SCORES, chunk * keyRows);
        this.qPos = s.intTemp(TEMP_Q_POS, ubatch);
        this.kvPos = s.intTemp(TEMP_KV_POS, keyRows);
    }

    // ------------------------------------------------------------------ steps

    /** Projects the queries and head weights of {@code n} tokens at positions {@code pos0 ..}. */
    public void queries(int pos0, int n, float[] qr, int qrOff, float[] x, int xOff) {
        ensureBound(qB != null && qBDeltas != null, "query");
        checkTokens(n);
        matVec.matmul(qB, qr, qrOff, queries, 0, n);
        qBDeltas.applyRows(n, qr, qrOff, queries, 0);
        int heads = cfg.nHeads();
        int dim = cfg.headDim();
        if (ropeTable != null) {
            fillPositions(pos0, n);
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

    /** Projects, normalizes, ropes, rotates and stores the keys of {@code n} tokens in their cache rows. */
    public void storeKeys(int pos0, int n, float[] x, int xOff) {
        ensureBound(kProj != null && kProjDeltas != null, "key");
        checkTokens(n);
        int dim = cfg.headDim();
        float[] k = keyRowsTemp;
        matVec.matmul(kProj, x, xOff, k, 0, n);
        kProjDeltas.applyRows(n, x, xOff, k, 0);
        NormMath.rowNorm(NormKind.LAYER, cfg.keyNormEps(), n, dim, dim, 1, k, 0, keyNormW, keyNormB, k, 0);
        if (ropeTable != null) {
            fillPositions(pos0, n);
            CpuRope.apply(ropeTable, k, 0, n, 1, dim, dim, cfg.ropeOffset(), ropePos, 0, n);
        }
        if (rotation != null) {
            CpuHadamard.applyInPlace(rotation, k, 0, n, dim);
        }
        KvLayout layout = keyCache.layout();
        for (int t = 0; t < n; t++) {
            keyCache.writeK(layout.rowOf(pos0 + t), k, t * dim);
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
        if (nKv < 1 || nKv > keyRows) {
            throw new IllegalArgumentException(where + ": indexer n_kv " + nKv + " is outside [1, " + keyRows + "]");
        }
        Objects.requireNonNull(topK, "topK");
        int k = SparseKernels.selectedCount(nKv, cfg.topK());
        for (int t = 0; t < n; t++) {
            qPos[t] = pos0 + t;
        }
        for (int row = 0; row < nKv; row++) {
            kvPos[row] = rowPosition.applyAsInt(row);
        }
        KvWindow mask = cfg.mask();
        int heads = cfg.nHeads();
        for (int t0 = 0; t0 < n; t0 += chunk) {
            int c = Math.min(chunk, n - t0);
            CpuDsaIndexer.scores(c, nKv, heads, cfg.headDim(), mask.causal(), mask.type(), mask.nSwa(),
                    queries, t0 * cfg.queryWidth(), weights, t0 * heads, keyReader, qPos, t0, kvPos, 0, scores, 0);
            CpuTopK.topKRows(scores, 0, c, nKv, k, topK, topKOff + t0 * k);
        }
        return k;
    }

    // ------------------------------------------------------------------ helpers

    private void fillPositions(int pos0, int n) {
        RopeSpec r = cfg.rope();
        RopePositions.fill(r.type(), r.textPositions(), pos0, n, ropePos);
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
