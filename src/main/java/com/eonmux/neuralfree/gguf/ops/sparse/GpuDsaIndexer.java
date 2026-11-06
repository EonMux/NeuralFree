package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.IndexerConfig;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The DeepSeek lightning indexer on the GPU (llama.cpp src/models/deepseek32.cpp:207-316): per step it projects the
 * indexer queries and head weights, optionally projects, normalizes, rotates and stores the indexer keys, then scores
 * every cached row and selects the top-k rows of each token.
 *
 * <p>Stages, each bound once and then run per step:</p>
 * <ul>
 *   <li><b>Queries</b> ({@link #bindQueries}, {@link #queries}): {@code iq = W_q_b . qr} as
 *       {@code [n_tokens][n_heads][head_dim]}, or {@code W_q_b . LayerNorm(x)} when {@code indexer.q_norm} is bound
 *       ({@link #bindQueryNorm}, mistral-large4.cpp:232-235); the optional SSMax per-head scale
 *       {@code s[h] * log(pos + 1) + b[h]} ({@link #bindSsmax}, mistral-large4.cpp:240), RoPE on
 *       {@code IndexerConfig.rope()} dims at {@code ropeOffset()} with the layer's frequency factors
 *       ({@code LLM_RoPE_Ext}), the Hadamard rotation of {@code rotationBlock()} ({@code LLM_HadamardRotate}), and
 *       {@code w = (W_proj . x) * weightScale} as {@code [n_tokens][n_heads]} (deepseek32.cpp:269 folds the scale into
 *       the weights).</li>
 *   <li><b>Keys</b> ({@link #bindKeys}, {@link #storeKeys}): {@code ik = LayerNorm(W_k . x)} with the
 *       {@code indexer.k_norm} weight and bias when present, the same RoPE and rotation as the queries, then
 *       {@code LLM_KV_Store_<KV>} into the indexer cache at {@code cell_idx} (deepseek32.cpp:229-254). Callers
 *       whose keys come from elsewhere (DeepSeek-V4 compressed keys) skip this stage.</li>
 *   <li><b>Selection</b> ({@link #bindSelection}, {@link #select}): {@code LLM_DSA_IndexerScore_<KV>} over cache rows
 *       {@code [0, n_kv)} with the visibility of {@code IndexerConfig.mask()} from {@code kv_pos} and
 *       {@code q_pos}, then {@code LLM_TopK_Rows} with {@code k = SparseKernels.selectedCount(n_kv, topK)} into
 *       {@code topK[t*k + j]} (deepseek32.cpp:313-315). Masked rows score -INF and selections shorter than
 *       {@code k} are padded with -1.</li>
 * </ul>
 *
 * <p>The indexer cache holds rows of {@code head_dim} values in its plan's {@link KvDType} (the context's K type,
 * llama.cpp src/llama-kv-cache-dsa.cpp:51, llama-kv-cache-dsv4.cpp:1308-1312); the store and score kernels are the
 * variants of that type. A block-type cache needs {@code head_dim} to be whole 32-value blocks
 * ({@link SparseKernels#checkIndexerCache}).</p>
 *
 * <p>Scores need {@code n_tokens * n_kv} floats. They are bounded by scoring at most {@link #chunkTokens()} tokens at
 * a time ({@link #SCORE_BUDGET_FLOATS} per chunk); since the score and top-k kernels address their buffers from
 * element 0, a chunk after the first stages its queries, weights and positions and copies its selection back.
 * Scoring is per token, so chunking does not change the result.</p>
 *
 * <p>Every working buffer is a {@code GpuScratch} temporary reserved at bind time; per step only token counts, row
 * counts, global sizes and the chunk buffers are re-set. Not thread-safe.</p>
 */
public final class GpuDsaIndexer implements AutoCloseable {

    public static final String TEMP_QUERIES = "dsa.indexer.q";
    public static final String TEMP_WEIGHTS = "dsa.indexer.w";
    public static final String TEMP_KEYS = "dsa.indexer.k";
    public static final String TEMP_SCORES = "dsa.indexer.scores";
    public static final String TEMP_STAGE_QUERIES = "dsa.indexer.stage.q";
    public static final String TEMP_STAGE_WEIGHTS = "dsa.indexer.stage.w";
    public static final String TEMP_STAGE_POSITIONS = "dsa.indexer.stage.pos";
    public static final String TEMP_STAGE_TOPK = "dsa.indexer.stage.topk";
    /** Layer-normed input the indexer queries read when {@code indexer.q_norm} is bound. */
    public static final String TEMP_Q_INPUT = "dsa.indexer.qin";
    public static final String TEMP_BLOCK_SCORES = "dsa.indexer.block.scores";
    public static final String TEMP_BLOCK_STAGE = "dsa.indexer.block.stage";

    /** Largest score buffer of one chunk: 16 Mi floats (64 MiB). */
    public static final long SCORE_BUDGET_FLOATS = 16L << 20;

    private static final int ARG_SCORE_IQ = SparseKernels.ARG_SCORE_IQ;
    private static final int ARG_SCORE_W = SparseKernels.ARG_SCORE_W;
    private static final int ARG_SCORE_Q_POS = SparseKernels.ARG_SCORE_Q_POS;
    private static final int ARG_TOPK_OUT = SparseKernels.ARG_TOPK_OUT;

    private final GpuForwardContext ctx;
    private final GpuOps ops;
    private final IndexerConfig cfg;
    private final String where;
    private final int ubatch;
    private final int local;
    private final HadamardSpec rotation;
    private final ScratchSlot queries;
    private final ScratchSlot weights;
    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuWeight qB;
    private GpuWeight proj;
    private PreparedDispatch queryRope;
    /** {@code indexer.q_norm} weight and bias over the layer input, and the norm that applies them. */
    private GpuWeight queryNormW;
    private GpuWeight queryNormB;
    private ScratchSlot queryNormInput;
    private int queryNormWidth;
    private PreparedDispatch queryNorm;
    /** {@code indexer.ssmax} per-head scale and bias and the dispatch that applies them to the queries. */
    private GpuBuffer ssmaxW;
    private GpuBuffer ssmaxB;
    private GpuBuffer ssmaxPos;
    private PreparedDispatch querySsmax;

    private GpuWeight kProj;
    /**
     * LoRA deltas of the three indexer projections ({@code build_lora_mm}, deepseek4.cpp:609, 622); served by
     * {@link #bindDeltas}, which the steps require.
     */
    private GpuProjectionDeltas.Site qBDeltas;
    private GpuProjectionDeltas.Site projDeltas;
    private GpuProjectionDeltas.Site kProjDeltas;
    private ScratchSlot keyRowsTemp;
    private PreparedDispatch keyNorm;
    private PreparedDispatch keyRope;
    private PreparedDispatch keyStore;
    private KvDType storeType;

    private int keyRows;
    private int chunk;
    /** Per-pair divisors of the indexer RoPE ({@code rope_freqs}), or null. */
    private GpuBuffer ropeFactors;
    private int scoreLocal;
    private int topKLocal;
    private GpuBuffer qPos;
    private ScratchSlot topK;
    private ScratchSlot stageQueries;
    private ScratchSlot stageWeights;
    private ScratchSlot stagePositions;
    private ScratchSlot stageTopK;
    private PreparedDispatch score;
    private PreparedDispatch select;
    private PreparedDispatch blockScore;
    private PreparedDispatch restrict;
    private ScratchSlot scoreBuffer;
    private ScratchSlot blockScores;
    private ScratchSlot blockStage;
    private int blockLocal;
    private int keyStride;
    private int keyOff;
    /** The key rows and {@code kv_pos} {@link #bindSelection} bound, restored after a {@link #selectRows} call. */
    private GpuBuffer boundKeys;
    private GpuBuffer boundKvPos;
    private boolean closed;

    /**
     * An unbound indexer reserving its query and weight buffers for the context's {@code ubatch}.
     *
     * @param where context of error messages (architecture and layer)
     * @throws IllegalStateException when a scratch temporary does not start at element 0
     */
    public GpuDsaIndexer(GpuForwardContext ctx, String where, IndexerConfig cfg) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.where = Objects.requireNonNull(where, "where");
        this.cfg = Objects.requireNonNull(cfg, "cfg");
        this.ops = ctx.ops();
        this.ubatch = ctx.scratch().ubatch();
        this.local = ops.local();
        this.rotation = cfg.rotation();
        this.queries = temp(TEMP_QUERIES, (long) ubatch * cfg.queryWidth());
        this.weights = temp(TEMP_WEIGHTS, (long) ubatch * cfg.nHeads());
    }

    public IndexerConfig config() {
        return cfg;
    }

    /**
     * Tokens scored per chunk; {@code ubatch} when every step fits the score budget. 0 before
     * {@link #bindSelection}.
     */
    public int chunkTokens() {
        return chunk;
    }

    /** The rotated, roped queries {@code [ubatch][n_heads][head_dim]} written by {@link #queries}. */
    public ScratchSlot queryBuffer() {
        return queries;
    }

    /** The scaled head weights {@code [ubatch][n_heads]} written by {@link #queries}. */
    public ScratchSlot weightBuffer() {
        return weights;
    }

    // ------------------------------------------------------------------ bind

    /**
     * Binds the query stage.
     *
     * @param qB           {@code indexer.attn_q_b}: {@code n_heads * head_dim} rows over the {@code q_lora} row or,
     *                     with {@link #bindQueryNorm}, over the layer-normed input
     * @param proj         {@code indexer.proj}: {@code n_heads} rows over the layer input
     * @param ropePos      the step's RoPE positions ({@code GpuStepInputs.ropePos()})
     * @param ropeFactors  per-pair divisors of the indexer RoPE ({@code rope_freqs}), or null
     * @throws IllegalStateException when already bound or a tensor has the wrong row count or an unsupported type
     */
    public void bindQueries(GpuWeight qB, GpuWeight proj, GpuBuffer ropePos, GpuBuffer ropeFactors) {
        ensureOpen();
        if (this.qB != null) {
            throw new IllegalStateException(where + ": indexer query stage already bound");
        }
        projection(qB, cfg.queryWidth(), "indexer.attn_q_b");
        projection(proj, cfg.nHeads(), "indexer.proj");
        RopeSpec boundRope = cfg.rope();
        if (boundRope != null && boundRope.freqFactors() != null && ropeFactors == null) {
            throw new IllegalStateException(where + ": the indexer RoPE reads the frequency factors "
                    + boundRope.freqFactors().name() + ", none were bound");
        }
        this.ropeFactors = ropeFactors;
        if (cfg.rope() != null) {
            queryRope = prepareRope(cfg.nHeads(), queries.buffer(), Objects.requireNonNull(ropePos, "ropePos"));
        }
        this.qB = qB;
        this.proj = proj;
    }

    /**
     * Binds the layer norm that produces the indexer queries from the layer input ({@code indexer.q_norm}): the
     * queries are then {@code indexer.attn_q_b . LayerNorm(x)} instead of {@code indexer.attn_q_b . q_lora}
     * (mistral-large4.cpp:232-235).
     *
     * @param normW {@code indexer.q_norm.weight}, F32 one value per layer-input dim
     * @param normB {@code indexer.q_norm.bias}, F32 one value per layer-input dim
     * @throws IllegalStateException when already bound, the vectors differ in length, or the query projection does
     *                               not read that width
     */
    public void bindQueryNorm(GpuWeight normW, GpuWeight normB) {
        ensureOpen();
        if (queryNormW != null) {
            throw new IllegalStateException(where + ": indexer query norm already bound");
        }
        int width = (int) normW.cols();
        if (width < 1 || normW.rows() != 1 || normB.rows() != 1 || normB.cols() != width) {
            throw new IllegalStateException(where + ": indexer.q_norm holds " + normW.cols() + "x" + normW.rows()
                    + " / " + normB.cols() + "x" + normB.rows() + " weight / bias values");
        }
        vector(normW, width, "indexer.q_norm.weight");
        vector(normB, width, "indexer.q_norm.bias");
        if (qB != null && qB.cols() != width) {
            throw new IllegalStateException(where + ": indexer.attn_q_b reads " + qB.cols() + " values, the layernormed"
                    + " input holds " + width);
        }
        this.queryNormW = normW;
        this.queryNormB = normB;
        this.queryNormWidth = width;
        this.queryNormInput = temp(TEMP_Q_INPUT, (long) ubatch * width);
        NormKernels.checkReduceLocal(local);
        this.queryNorm = prepare(NormKernels.ROW_NORM,
                NormKernels.reduceGlobal(NormKernels.rowNormGroups(ubatch), local), new long[]{local},
                NormKernels.rowNormArgs(NormKind.LAYER, cfg.keyNormEps(), ubatch, width, width, queryNormInput.buffer(),
                        0, queryNormInput.buffer(), 0, normW.shard(0), normB.shard(0), 1,
                        ctx.placeholders().readOnlyFloats()));
    }

    /**
     * Applies the per-head SSMax scale {@code weight[h] * log(pos + 1) + bias[h]} to the indexer queries
     * (mistral-large4.cpp:240).
     *
     * @param weight {@code indexer.ssmax.weight}, F32 one value per indexer head
     * @param bias   {@code indexer.ssmax.bias}, F32 one value per indexer head
     * @param pos    the step's positions ({@code GpuStepInputs.ropePos()})
     * @throws IllegalStateException when already bound or the vectors do not hold one value per head
     */
    public void bindSsmax(GpuWeight weight, GpuWeight bias, GpuBuffer pos) {
        ensureOpen();
        if (ssmaxW != null) {
            throw new IllegalStateException(where + ": indexer SSMax already bound");
        }
        vector(weight, cfg.nHeads(), "indexer.ssmax.weight");
        vector(bias, cfg.nHeads(), "indexer.ssmax.bias");
        this.ssmaxW = weight.shard(0);
        this.ssmaxB = bias.shard(0);
        this.ssmaxPos = Objects.requireNonNull(pos, "pos");
        int dim = cfg.headDim();
        int heads = cfg.nHeads();
        this.querySsmax = prepare(AttnExtKernels.SSMAX_SCALE,
                AttnExtKernels.ssmaxGlobal(ubatch, heads, dim, local), new long[]{local},
                AttnExtKernels.ssmaxArgs(ubatch, heads, dim, dim, heads * dim, 0, ssmaxW, ssmaxB, ssmaxPos,
                        queries.buffer()));
    }

    /**
     * Binds the key stage over an indexer cache of {@code keyRows} rows of {@code head_dim} values of {@code keyType}.
     *
     * @param kProj   {@code indexer.attn_k}: {@code head_dim} rows over the layer input
     * @param normW   {@code indexer.k_norm.weight}, F32 {@code head_dim}, or null (glm-dsa loads it as optional)
     * @param normB   {@code indexer.k_norm.bias}, F32 {@code head_dim}, or null
     * @param ropePos the step's RoPE positions
     * @param cellIdx the step's cache rows of the indexer cache's KV group ({@code GpuStepInputs.cellIdx})
     * @param keys    the indexer cache rows
     * @param keyType the element type of the indexer cache rows
     * @throws IllegalStateException when already bound or a tensor or the cache does not fit the configuration
     */
    public void bindKeys(GpuWeight kProj, GpuWeight normW, GpuWeight normB, GpuBuffer ropePos, GpuBuffer cellIdx,
                         GpuBuffer keys, KvDType keyType, int keyRows) {
        ensureOpen();
        if (this.kProj != null) {
            throw new IllegalStateException(where + ": indexer key stage already bound");
        }
        int dim = cfg.headDim();
        projection(kProj, dim, "indexer.attn_k");
        if (normW != null) {
            vector(normW, dim, "indexer.k_norm.weight");
        }
        if (normB != null) {
            vector(normB, dim, "indexer.k_norm.bias");
        }
        checkCache(keys, keyType, keyRows, cfg.headDim(), 0);
        keyRowsTemp = temp(TEMP_KEYS, (long) ubatch * dim);
        GpuBuffer k = keyRowsTemp.buffer();
        NormKernels.checkReduceLocal(local);
        keyNorm = prepare(NormKernels.ROW_NORM, NormKernels.reduceGlobal(NormKernels.rowNormGroups(ubatch), local),
                new long[]{local}, NormKernels.rowNormArgs(NormKind.LAYER, cfg.keyNormEps(), ubatch, dim, dim, k, 0, k,
                        0, normW != null ? normW.shard(0) : null, normB != null ? normB.shard(0) : null, 1,
                        ctx.placeholders().readOnlyFloats()));
        if (cfg.rope() != null) {
            keyRope = prepareRope(1, k, Objects.requireNonNull(ropePos, "ropePos"));
        }
        keyStore = prepare(RopeKernels.kvStore(keyType),
                elementGlobal(RopeKernels.kvStoreWorkItems(keyType, ubatch, dim)), new long[]{local},
                RopeKernels.kvStoreArgs(keyType, ubatch, dim, dim, 0, dim, dim, k,
                        Objects.requireNonNull(cellIdx, "cellIdx"), ctx.placeholders().kvScalesOnes(), keys));
        this.storeType = keyType;
        this.kProj = kProj;
    }

    /**
     * Serves the LoRA deltas of the bound projections ({@code indexer.attn_q_b}, {@code indexer.proj},
     * {@code indexer.attn_k}) of {@code layer} in the forward pass's {@link GpuProjectionDeltas}.
     *
     * @throws IllegalStateException when no projection is bound yet
     */
    public void bindDeltas(LayerSpec layer) {
        if (qB == null && kProj == null) {
            throw new IllegalStateException(where + ": bind the indexer projections before their LoRA deltas");
        }
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        if (qB != null) {
            qBDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.INDEXER_ATTN_Q_B), qB);
            projDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.INDEXER_PROJ), proj);
        }
        if (kProj != null) {
            kProjDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.INDEXER_ATTN_K), kProj);
        }
    }

    /**
     * Binds the selection stage.
     *
     * @param keys    key rows of {@code head_dim} values of {@code keyType}, {@code keyRows} of them
     * @param keyType the element type of the indexer cache rows
     * @param kvPos   {@code int[keyRows]}: the position held by each row, -1 when empty
     * @param qPos    {@code int[n_tokens]}: the step's query positions
     * @param topK    output of {@code ubatch * topK} ints at element 0 of its buffer, written {@code [t][k]}
     * @throws IllegalStateException when already bound, the cache is too small or {@code topK} does not fit
     */
    public void bindSelection(GpuBuffer keys, KvDType keyType, int keyRows, GpuBuffer kvPos, GpuBuffer qPos,
                              ScratchSlot topK) {
        bindSelection(keys, keyType, keyRows, cfg.headDim(), 0, kvPos, qPos, topK);
    }

    /**
     * Binds the selection stage with the layout of a key row inside the cache.
     *
     * @param keys      cache rows of {@code keyStride} values of {@code keyType}, the key of {@code headDim} values
     *                  starting at {@code keyOff} (a DeepSeek-V4.1 compressed row stores the index key beside its
     *                  latent)
     * @param keyStride values per cache row
     * @param keyOff    offset of the key inside a row
     * @param kvPos     {@code int[keyRows]}: the position held by each row, -1 when empty
     * @param qPos      {@code int[n_tokens]}: the step's query positions
     * @param topK      output of {@code ubatch * topK} ints at element 0 of its buffer, written {@code [t][k]}
     * @throws IllegalStateException when already bound, the cache is too small or {@code topK} does not fit
     */
    public void bindSelection(GpuBuffer keys, KvDType keyType, int keyRows, int keyStride, int keyOff,
                              GpuBuffer kvPos, GpuBuffer qPos, ScratchSlot topK) {
        ensureOpen();
        if (score != null) {
            throw new IllegalStateException(where + ": indexer selection stage already bound");
        }
        checkCache(keys, keyType, keyRows, keyStride, keyOff);
        if (storeType != null && storeType != keyType) {
            throw new IllegalStateException(where + ": the indexer keys were bound as " + storeType
                    + " rows, the selection reads " + keyType);
        }
        Objects.requireNonNull(kvPos, "kvPos");
        if (kvPos.sizeInBytes() < (long) keyRows * Integer.BYTES) {
            throw new IllegalStateException(where + ": kv_pos holds " + kvPos.sizeInBytes() / Integer.BYTES
                    + " rows, the indexer cache " + keyRows);
        }
        this.qPos = Objects.requireNonNull(qPos, "qPos");
        this.topK = zeroBased(Objects.requireNonNull(topK, "topK"), "top-k output")
                .require((long) ubatch * cfg.topK());
        this.keyRows = keyRows;
        this.keyStride = keyStride;
        this.keyOff = keyOff;
        this.chunk = (int) Math.max(1L, Math.min(ubatch, SCORE_BUDGET_FLOATS / keyRows));
        ScratchSlot scores = temp(TEMP_SCORES, (long) chunk * keyRows);
        this.scoreBuffer = scores;
        if (chunk < ubatch) {
            reserveStages();
        }
        this.boundKeys = keys;
        this.boundKvPos = kvPos;
        scoreLocal = Math.min(SparseKernels.INDEXER_LOCAL, local);
        topKLocal = Math.min(SparseKernels.TOPK_LOCAL, local);
        SparseKernels.checkIndexerLocal(scoreLocal);
        SparseKernels.checkTopKLocal(topKLocal);
        KvWindow mask = cfg.mask();
        score = prepare(SparseKernels.indexerScore(keyType), SparseKernels.indexerGlobal(chunk, keyRows, scoreLocal),
                SparseKernels.local(scoreLocal),
                new Object[]{chunk, keyRows, cfg.nHeads(), cfg.headDim(), mask.causal() ? 1 : 0, mask.type().code(),
                        mask.nSwa(), keyStride, keyOff, queries.buffer(), weights.buffer(), keys, qPos, kvPos,
                        scores.buffer()});
        select = prepare(SparseKernels.TOPK_ROWS, SparseKernels.topKGlobal(chunk, topKLocal),
                SparseKernels.local(topKLocal),
                new Object[]{chunk, keyRows, SparseKernels.selectedCount(keyRows, cfg.topK()), scores.buffer(),
                        topK.buffer()});
    }

    /**
     * Binds the DeepSeek-V4.1 candidate-block stages: the block scores of the queries' row scores and the row
     * restriction to an earlier source's blocks.
     */
    public void bindBlocks() {
        ensureBound(score != null, "selection");
        if (blockScore != null) {
            return;
        }
        reserveStages();
        this.blockLocal = Math.max(1, Math.min(64, local));
        this.blockScores = temp(TEMP_BLOCK_SCORES, (long) chunk * keyRows);
        this.blockStage = temp(TEMP_BLOCK_STAGE, (long) chunk * SparseKernels.TOPK_MAX);
        blockScore = prepare(SparseKernels.BLOCK_SCORE,
                SparseKernels.blockScoreGlobal(chunk, keyRows, 1, blockLocal), SparseKernels.local(blockLocal),
                new Object[]{chunk, keyRows, 1, 1, qPos, scoreBuffer.buffer(), blockScores.buffer()});
        restrict = prepare(SparseKernels.RESTRICT_BLOCKS,
                SparseKernels.restrictGlobal(chunk, keyRows, blockLocal), SparseKernels.local(blockLocal),
                new Object[]{chunk, keyRows, 1, 0, 1, ctx.placeholders().readOnlyInts(), scoreBuffer.buffer()});
    }

    /**
     * Reserves the staging buffers {@link #selectRows} scores a segment's tokens through. A mixer bound to a paged
     * memory calls it after {@link #bindSelection}; a step whose tokens exceed the score budget reserves them too.
     */
    public void enableSegments() {
        ensureBound(score != null, "selection");
        reserveStages();
    }

    private void reserveStages() {
        if (stageQueries != null) {
            return;
        }
        stageQueries = temp(TEMP_STAGE_QUERIES, (long) chunk * cfg.queryWidth());
        stageWeights = temp(TEMP_STAGE_WEIGHTS, (long) chunk * cfg.nHeads());
        stagePositions = temp(TEMP_STAGE_POSITIONS, chunk);
        stageTopK = temp(TEMP_STAGE_TOPK, (long) chunk * cfg.topK());
    }

    // ------------------------------------------------------------------ steps

    /**
     * Projects the queries and head weights of {@code nTokens} tokens.
     *
     * @param qr the normalized {@code q_lora} rows ({@code nTokens} rows of the {@code indexer.attn_q_b} width)
     * @param x  the layer input rows
     */
    public void queries(int nTokens, GpuBuffer qr, int qrOff, GpuBuffer x, int xOff) {
        ensureBound(qB != null && qBDeltas != null, "query");
        checkTokens(nTokens);
        int heads = cfg.nHeads();
        if (queryNorm != null) {
            ops.copy2d(x, xOff, queryNormWidth, queryNormInput.buffer(), 0, queryNormWidth, nTokens, queryNormWidth);
            queryNorm.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, nTokens);
            queryNorm.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(nTokens), local));
            queryNorm.dispatch();
            ops.matmul(qB, queryNormInput.buffer(), 0, queries.buffer(), 0, nTokens);
            qBDeltas.applyRows(nTokens, queryNormInput.buffer(), 0, queries.buffer(), 0);
        } else {
            ops.matmul(qB, qr, qrOff, queries.buffer(), 0, nTokens);
            qBDeltas.applyRows(nTokens, qr, qrOff, queries.buffer(), 0);
        }
        if (querySsmax != null) {
            querySsmax.setInt(AttnExtKernels.ARG_SSMAX_N_TOKENS, nTokens);
            querySsmax.setGlobal(AttnExtKernels.ssmaxGlobal(nTokens, heads, cfg.headDim(), local));
            querySsmax.dispatch();
        }
        if (queryRope != null) {
            runRope(queryRope, nTokens, heads);
        }
        if (rotation != null) {
            ops.hadamard(rotation, queries.buffer(), 0, queries.buffer(), 0, nTokens * heads, cfg.headDim());
        }
        ops.matmul(proj, x, xOff, weights.buffer(), 0, nTokens);
        projDeltas.applyRows(nTokens, x, xOff, weights.buffer(), 0);
        if (cfg.weightScale() != 1.0f) {
            ops.scale(weights.buffer(), 0, nTokens * heads, cfg.weightScale());
        }
    }

    /** Projects, normalizes, ropes, rotates and stores the keys of {@code nTokens} tokens at the step's cells. */
    public void storeKeys(int nTokens, GpuBuffer x, int xOff) {
        ensureBound(kProj != null && kProjDeltas != null, "key");
        checkTokens(nTokens);
        GpuBuffer k = keyRowsTemp.buffer();
        ops.matmul(kProj, x, xOff, k, 0, nTokens);
        kProjDeltas.applyRows(nTokens, x, xOff, k, 0);
        keyNorm.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, nTokens);
        keyNorm.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(nTokens), local));
        keyNorm.dispatch();
        if (keyRope != null) {
            runRope(keyRope, nTokens, 1);
        }
        if (rotation != null) {
            ops.hadamard(rotation, k, 0, k, 0, nTokens, cfg.headDim());
        }
        keyStore.setInt(RopeKernels.ARG_N_TOKENS, nTokens);
        keyStore.setGlobal(elementGlobal(RopeKernels.kvStoreWorkItems(storeType, nTokens, cfg.headDim())));
        keyStore.dispatch();
    }

    /**
     * Scores the step's {@code nTokens} queries against cache rows {@code [0, nKv)} and writes the selected rows of
     * token {@code t} to {@code topK[t*k .. t*k + k)}.
     *
     * @return {@code k = min(nKv, topK)}, the row stride of the selection and {@code n_sel} of the sparse attention
     * @throws IllegalArgumentException when {@code nTokens} or {@code nKv} is outside its bound range
     */
    public int select(int nTokens, int nKv) {
        ensureBound(score != null, "selection");
        checkTokens(nTokens);
        if (nKv < 1 || nKv > keyRows) {
            throw new IllegalArgumentException(where + ": indexer n_kv " + nKv + " is outside [1, " + keyRows + "]");
        }
        int k = SparseKernels.selectedCount(nKv, cfg.topK());
        if (nTokens <= chunk) {
            scoreChunk(nTokens, nKv, k, queries.buffer(), weights.buffer(), qPos, topK.buffer());
            return k;
        }
        int width = cfg.queryWidth();
        int heads = cfg.nHeads();
        for (int t0 = 0; t0 < nTokens; t0 += chunk) {
            int c = Math.min(chunk, nTokens - t0);
            ops.copy(queries.buffer(), t0 * width, stageQueries.buffer(), 0, c * width);
            ops.copy(weights.buffer(), t0 * heads, stageWeights.buffer(), 0, c * heads);
            ctx.queue().enqueueCopyBuffer(qPos, (long) t0 * Integer.BYTES, stagePositions.buffer(), 0,
                    (long) c * Integer.BYTES);
            scoreChunk(c, nKv, k, stageQueries.buffer(), stageWeights.buffer(), stagePositions.buffer(),
                    stageTopK.buffer());
            ctx.queue().enqueueCopyBuffer(stageTopK.buffer(), 0, topK.buffer(), (long) t0 * k * Integer.BYTES,
                    (long) c * k * Integer.BYTES);
        }
        return k;
    }

    /**
     * Scores the step's tokens {@code [t0, t0 + n)} (queries, weights and {@code q_pos} rows of the step) against rows
     * {@code [0, nKv)} of {@code keys}, row {@code r} at position {@code kvPos[r]} (-1 hides it), and writes token
     * {@code t0 + i}'s selection to {@code out[outOff + i*k ..]}: one paged sequence's selection, whose
     * {@code kvPos} masks every row outside the sequence ({@code GpuStepInputs.sequenceKvPos}) or whose keys are
     * its own stream. The tokens run through the staging buffers ({@link #enableSegments}) in chunks of
     * {@link #chunkTokens()}; the bound keys and mask come back afterwards.
     *
     * @param causal whether a row after the token's position is masked (false for a non-causal step)
     * @return {@code k = min(nKv, topK)}, the row stride of the written selection
     * @throws IllegalArgumentException when the tokens or rows are outside the bound ranges
     * @throws IllegalStateException    when the staging buffers were not reserved
     */
    public int selectRows(int t0, int n, int nKv, GpuBuffer keys, GpuBuffer kvPos, boolean causal, GpuBuffer out,
                          long outOff) {
        ensureBound(score != null, "selection");
        if (stageQueries == null) {
            throw new IllegalStateException(where + ": segment selection needs the staging buffers (enableSegments)");
        }
        checkTokens(n);
        if (t0 < 0 || t0 + n > ubatch) {
            throw new IllegalArgumentException(where + ": indexer tokens [" + t0 + ", " + (t0 + n)
                    + ") are outside the step's " + ubatch + " rows");
        }
        if (nKv < 1 || nKv > keyRows) {
            throw new IllegalArgumentException(where + ": indexer n_kv " + nKv + " is outside [1, " + keyRows + "]");
        }
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(kvPos, "kvPos");
        Objects.requireNonNull(out, "out");
        int k = SparseKernels.selectedCount(nKv, cfg.topK());
        int width = cfg.queryWidth();
        int heads = cfg.nHeads();
        score.setBuffer(SparseKernels.ARG_SCORE_KEYS, keys);
        score.setBuffer(SparseKernels.ARG_SCORE_KV_POS, kvPos);
        score.setInt(SparseKernels.ARG_SCORE_CAUSAL, causal ? 1 : 0);
        try {
            for (int c0 = 0; c0 < n; c0 += chunk) {
                int c = Math.min(chunk, n - c0);
                int tok = t0 + c0;
                ops.copy(queries.buffer(), tok * width, stageQueries.buffer(), 0, c * width);
                ops.copy(weights.buffer(), tok * heads, stageWeights.buffer(), 0, c * heads);
                ctx.queue().enqueueCopyBuffer(qPos, (long) tok * Integer.BYTES, stagePositions.buffer(), 0,
                        (long) c * Integer.BYTES);
                scoreChunk(c, nKv, k, stageQueries.buffer(), stageWeights.buffer(), stagePositions.buffer(),
                        stageTopK.buffer());
                ctx.queue().enqueueCopyBuffer(stageTopK.buffer(), 0, out, (outOff + (long) c0 * k) * Integer.BYTES,
                        (long) c * k * Integer.BYTES);
            }
        } finally {
            score.setBuffer(SparseKernels.ARG_SCORE_KEYS, boundKeys);
            score.setBuffer(SparseKernels.ARG_SCORE_KV_POS, boundKvPos);
            score.setInt(SparseKernels.ARG_SCORE_CAUSAL, cfg.mask().causal() ? 1 : 0);
        }
        return k;
    }

    /**
     * Scores the step's tokens {@code [t0, t0 + n)} against rows {@code [0, nKv)} and selects the top
     * {@code topKBlocks} blocks of {@code blockSize} consecutive rows, writing token {@code t0 + i}'s blocks to
     * {@code blocks[blocksOff + i*k ..]} with {@code k = min(topKBlocks, ceil(nKv / blockSize))}. A block scores its
     * best row's indexer score; the block holding the newest visible row visible to the token gets {@code +1e30}, so a
     * partly filled block is always kept (models/deepseek41.cpp:588-597). The keys, positions and causal mask are the
     * ones bound by {@link #bindSelection}.
     *
     * @param ratio compress ratio of the rows, the unit of the visibility
     * @throws IllegalArgumentException when the tokens, rows or block shape are outside their ranges
     * @throws IllegalStateException    when the block stages were not bound ({@link #bindBlocks}) or the staging
     *                                  buffers were not reserved
     */
    public int selectBlocks(int t0, int n, int nKv, int ratio, int blockSize, int topKBlocks, GpuBuffer blocks,
                            int blocksOff) {
        return selectBlocks(t0, n, nKv, ratio, blockSize, topKBlocks, blocks, blocksOff, null, null, true);
    }

    /**
     * {@link #selectBlocks} against explicit key rows and positions instead of the ones bound by
     * {@link #bindSelection} (a DeepSeek-V4.1 paged step scores each sequence's own compressed stream). The bound
     * rows, positions and causal flag come back afterwards.
     *
     * @param keys  cache rows of the bound stride and element type, or null for the bound rows
     * @param kvPos positions of those rows, or null for the bound positions
     * @param causal whether a row after the token's position is masked
     */
    public int selectBlocks(int t0, int n, int nKv, int ratio, int blockSize, int topKBlocks, GpuBuffer blocks,
                            int blocksOff, GpuBuffer keys, GpuBuffer kvPos, boolean causal) {
        ensureBlocks();
        checkTokens(n);
        checkBlockRange(t0, n, nKv, blocks, blocksOff, topKBlocks);
        GpuBuffer kRows = keys != null ? keys : boundKeys;
        GpuBuffer kPos = keys != null ? kvPos : boundKvPos;
        boolean kCausal = keys != null ? causal : cfg.mask().causal();
        Objects.requireNonNull(kRows, "keys");
        Objects.requireNonNull(kPos, "kvPos");
        int nBlk = SparseKernels.blockCount(nKv, blockSize);
        int k = Math.min(topKBlocks, nBlk);
        int width = cfg.queryWidth();
        int heads = cfg.nHeads();
        bindScoreSource(kRows, kPos, kCausal);
        try {
            for (int c0 = 0; c0 < n; c0 += chunk) {
                int c = Math.min(chunk, n - c0);
                int tok = t0 + c0;
                stage(tok, c, width, heads);
                scoreChunkRaw(c, nKv);
                blockScore.setInt(SparseKernels.ARG_BLOCK_SCORE_N_TOKENS, c);
                blockScore.setInt(SparseKernels.ARG_BLOCK_SCORE_N_KV, nKv);
                blockScore.setInt(2, blockSize);
                blockScore.setInt(3, ratio);
                blockScore.setBuffer(SparseKernels.ARG_BLOCK_SCORE_Q_POS, stagePositions.buffer());
                blockScore.setGlobal(SparseKernels.blockScoreGlobal(c, nKv, blockSize, blockLocal));
                blockScore.dispatch();
                select.setInt(SparseKernels.ARG_TOPK_N_ROWS, c);
                select.setInt(SparseKernels.ARG_TOPK_N_COLS, nBlk);
                select.setInt(SparseKernels.ARG_TOPK_K, k);
                select.setBuffer(SparseKernels.ARG_TOPK_SCORES, blockScores.buffer());
                select.setBuffer(ARG_TOPK_OUT, blockStage.buffer());
                select.setGlobal(SparseKernels.topKGlobal(c, topKLocal));
                select.dispatch();
                ctx.queue().enqueueCopyBuffer(blockStage.buffer(), 0, blocks,
                        (blocksOff + (long) c0 * k) * Integer.BYTES, (long) c * k * Integer.BYTES);
            }
        } finally {
            restoreScoreSource();
        }
        return k;
    }

    /** The key rows, positions and causal flag of the score dispatch when {@link #bindSelection} bound them. */
    private void bindScoreSource(GpuBuffer keys, GpuBuffer kvPos, boolean causal) {
        score.setBuffer(SparseKernels.ARG_SCORE_KEYS, keys);
        score.setBuffer(SparseKernels.ARG_SCORE_KV_POS, kvPos);
        score.setInt(SparseKernels.ARG_SCORE_CAUSAL, causal ? 1 : 0);
    }

    private void restoreScoreSource() {
        bindScoreSource(boundKeys, boundKvPos, cfg.mask().causal());
    }

    /**
     * Scores the step's tokens {@code [t0, t0 + n)} against rows {@code [0, nKv)}, masks every row whose block is not
     * one of the token's candidate blocks {@code blocks[blocksOff + i*blocksStride ..]} and writes the top
     * {@code cfg.topK()} rows to {@code topK[topKOff + (t0 + i)*k ..]} (models/deepseek41.cpp:598-611).
     *
     * @return {@code k = min(nKv, topK)}
     * @throws IllegalArgumentException when the tokens, rows or block shape are outside their ranges
     * @throws IllegalStateException    when the block stages were not bound ({@link #bindBlocks}) or the staging
     *                                  buffers were not reserved
     */
    public int selectInsideBlocks(int t0, int n, int nKv, int blockSize, GpuBuffer blocks, int blocksOff,
                                  int blocksStride, ScratchSlot topK, int topKOff) {
        return selectInsideBlocks(t0, n, nKv, blockSize, blocks, blocksOff, blocksStride, topK, topKOff, null, null,
                true);
    }

    /**
     * {@link #selectInsideBlocks} against explicit key rows and positions (a DeepSeek-V4.1 paged step scores each
     * sequence's own compressed stream); the bound rows, positions and causal flag come back afterwards.
     *
     * @param keys  cache rows of the bound stride and element type, or null for the bound rows
     * @param kvPos positions of those rows, or null for the bound positions
     * @param causal whether a row after the token's position is masked
     */
    public int selectInsideBlocks(int t0, int n, int nKv, int blockSize, GpuBuffer blocks, int blocksOff,
                                  int blocksStride, ScratchSlot topK, int topKOff, GpuBuffer keys, GpuBuffer kvPos,
                                  boolean causal) {
        ensureBlocks();
        checkTokens(n);
        if (blockSize < 1 || blocksStride < 0) {
            throw new IllegalArgumentException(where + ": block selection needs a positive block size and a"
                    + " non-negative stride, got " + blockSize + "/" + blocksStride);
        }
        if (nKv < 1 || nKv > keyRows) {
            throw new IllegalArgumentException(where + ": indexer n_kv " + nKv + " is outside [1, " + keyRows + "]");
        }
        Objects.requireNonNull(blocks, "blocks");
        Objects.requireNonNull(topK, "topK");
        GpuBuffer kRows = keys != null ? keys : boundKeys;
        GpuBuffer kPos = keys != null ? kvPos : boundKvPos;
        boolean kCausal = keys != null ? causal : cfg.mask().causal();
        Objects.requireNonNull(kRows, "keys");
        Objects.requireNonNull(kPos, "kvPos");
        int k = SparseKernels.selectedCount(nKv, cfg.topK());
        if (topKOff < 0 || (long) topKOff + (long) n * k > (long) ubatch * cfg.topK()) {
            throw new IllegalArgumentException(where + ": " + topK + " cannot hold " + (t0 + n) + " rows of " + k);
        }
        int width = cfg.queryWidth();
        int heads = cfg.nHeads();
        bindScoreSource(kRows, kPos, kCausal);
        try {
            for (int c0 = 0; c0 < n; c0 += chunk) {
                int c = Math.min(chunk, n - c0);
                int tok = t0 + c0;
                stage(tok, c, width, heads);
                scoreChunkRaw(c, nKv);
                restrict.setInt(SparseKernels.ARG_RESTRICT_N_TOKENS, c);
                restrict.setInt(SparseKernels.ARG_RESTRICT_N_KV, nKv);
                restrict.setInt(SparseKernels.ARG_RESTRICT_BLOCK_SIZE, blockSize);
                restrict.setInt(SparseKernels.ARG_RESTRICT_BLOCKS_OFF, blocksOff + c0 * blocksStride);
                restrict.setInt(SparseKernels.ARG_RESTRICT_BLOCKS_STRIDE, blocksStride);
                restrict.setBuffer(SparseKernels.ARG_RESTRICT_BLOCKS, blocks);
                restrict.setGlobal(SparseKernels.restrictGlobal(c, nKv, blockLocal));
                restrict.dispatch();
                select.setInt(SparseKernels.ARG_TOPK_N_ROWS, c);
                select.setInt(SparseKernels.ARG_TOPK_N_COLS, nKv);
                select.setInt(SparseKernels.ARG_TOPK_K, k);
                select.setBuffer(SparseKernels.ARG_TOPK_SCORES, scoreBuffer.buffer());
                select.setBuffer(ARG_TOPK_OUT, stageTopK.buffer());
                select.setGlobal(SparseKernels.topKGlobal(c, topKLocal));
                select.dispatch();
                ctx.queue().enqueueCopyBuffer(stageTopK.buffer(), 0, topK.buffer(),
                        (topKOff + (long) c0 * k) * Integer.BYTES, (long) c * k * Integer.BYTES);
            }
        } finally {
            restoreScoreSource();
        }
        return k;
    }

    /** The staging copies of tokens {@code [tok, tok + c)}: queries, weights and positions. */
    private void stage(int tok, int c, int width, int heads) {
        ops.copy(queries.buffer(), tok * width, stageQueries.buffer(), 0, c * width);
        ops.copy(weights.buffer(), tok * heads, stageWeights.buffer(), 0, c * heads);
        ctx.queue().enqueueCopyBuffer(qPos, (long) tok * Integer.BYTES, stagePositions.buffer(), 0,
                (long) c * Integer.BYTES);
    }

    /** Scores {@code c} staged queries against rows {@code [0, nKv)} into the score buffer. */
    private void scoreChunkRaw(int c, int nKv) {
        score.setInt(SparseKernels.ARG_SCORE_N_TOKENS, c);
        score.setInt(SparseKernels.ARG_SCORE_N_KV, nKv);
        score.setBuffer(ARG_SCORE_IQ, stageQueries.buffer());
        score.setBuffer(ARG_SCORE_W, stageWeights.buffer());
        score.setBuffer(ARG_SCORE_Q_POS, stagePositions.buffer());
        score.setGlobal(SparseKernels.indexerGlobal(c, nKv, scoreLocal));
        score.dispatch();
    }

    /** Checks the ranges shared by the block stages and enables them ({@link #bindBlocks}). */
    private void checkBlockRange(int t0, int n, int nKv, GpuBuffer blocks, int blocksOff, int topKBlocks) {
        Objects.requireNonNull(blocks, "blocks");
        if (topKBlocks < 1 || topKBlocks > SparseKernels.TOPK_MAX) {
            throw new IllegalArgumentException(where + ": " + topKBlocks + " candidate blocks are outside [1, "
                    + SparseKernels.TOPK_MAX + "]");
        }
        if (nKv < 1 || nKv > keyRows) {
            throw new IllegalArgumentException(where + ": indexer n_kv " + nKv + " is outside [1, " + keyRows + "]");
        }
        if (blocksOff < 0) {
            throw new IllegalArgumentException(where + ": block slots offset " + blocksOff + " is negative");
        }
        if (stageQueries == null) {
            throw new IllegalStateException(where + ": block selection needs the staging buffers (enableSegments)");
        }
    }

    private void ensureBlocks() {
        ensureBound(score != null, "selection");
        if (blockScore == null) {
            throw new IllegalStateException(where + ": the candidate-block stage is not bound (bindBlocks)");
        }
        if (stageQueries == null) {
            throw new IllegalStateException(where + ": block selection needs the staging buffers (enableSegments)");
        }
    }

    private void scoreChunk(int c, int nKv, int k, GpuBuffer iq, GpuBuffer w, GpuBuffer pos, GpuBuffer out) {
        score.setInt(SparseKernels.ARG_SCORE_N_TOKENS, c);
        score.setInt(SparseKernels.ARG_SCORE_N_KV, nKv);
        score.setBuffer(ARG_SCORE_IQ, iq);
        score.setBuffer(ARG_SCORE_W, w);
        score.setBuffer(ARG_SCORE_Q_POS, pos);
        score.setGlobal(SparseKernels.indexerGlobal(c, nKv, scoreLocal));
        score.dispatch();
        select.setInt(SparseKernels.ARG_TOPK_N_ROWS, c);
        select.setInt(SparseKernels.ARG_TOPK_N_COLS, nKv);
        select.setInt(SparseKernels.ARG_TOPK_K, k);
        select.setBuffer(SparseKernels.ARG_TOPK_SCORES, scoreBuffer.buffer());
        select.setBuffer(ARG_TOPK_OUT, out);
        select.setGlobal(SparseKernels.topKGlobal(c, topKLocal));
        select.dispatch();
    }

    // ------------------------------------------------------------------ helpers

    /** {@code LLM_RoPE_Ext} over {@code heads} indexer heads of {@code head_dim} per token, prepared for ubatch. */
    private PreparedDispatch prepareRope(int heads, GpuBuffer x, GpuBuffer ropePos) {
        RopeSpec r = cfg.rope();
        int dim = cfg.headDim();
        boolean hasFf = ropeFactors != null;
        return prepare(RopeKernels.ROPE_EXT,
                elementGlobal(RopeKernels.ropeExtWorkItems(r.type(), ubatch, heads, r.nDims())), new long[]{local},
                RopeKernels.ropeExtArgs(r, ubatch, heads, dim, heads * dim, cfg.ropeOffset(), ubatch, 0, hasFf,
                        ropePos, hasFf ? ropeFactors : ctx.placeholders().readOnlyFloats(), x));
    }

    private void runRope(PreparedDispatch d, int nTokens, int heads) {
        RopeSpec r = cfg.rope();
        d.setInt(RopeKernels.ARG_N_TOKENS, nTokens);
        d.setInt(RopeKernels.ARG_POS_STRIDE, nTokens);
        d.setGlobal(elementGlobal(RopeKernels.ropeExtWorkItems(r.type(), nTokens, heads, r.nDims())));
        d.dispatch();
    }

    private long[] elementGlobal(long items) {
        return new long[]{RopeKernels.roundUp(Math.max(1L, items), local)};
    }

    private PreparedDispatch prepare(String kernel, long[] global, long[] localSize, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, localSize, args);
        prepared.add(d);
        return d;
    }

    private ScratchSlot temp(String name, long floats) {
        return zeroBased(ctx.scratch().temp(name, floats), name);
    }

    private ScratchSlot zeroBased(ScratchSlot s, String what) {
        if (s.offset() != 0) {
            throw new IllegalStateException(where + ": indexer buffer " + what + " (" + s + ") must start at element 0"
                    + " of its buffer");
        }
        return s;
    }

    private void projection(GpuWeight w, int rows, String what) {
        Objects.requireNonNull(w, what);
        if (w.rows() != rows) {
            throw new IllegalStateException(where + ": " + what + " " + w.name() + " has " + w.rows() + " rows, the"
                    + " indexer needs " + rows);
        }
        if (!GpuOps.supportsWeightType(w.type())) {
            throw new GpuModelUnsupportedException(where + ": no GPU matvec kernels for " + w.name() + " of type "
                    + w.type());
        }
    }

    private void vector(GpuWeight v, int n, String what) {
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || v.cols() * v.rows() != n) {
            throw new IllegalStateException(where + ": " + what + " " + v.name() + " must be a single-buffer F32"
                    + " vector of " + n + " values, got " + v);
        }
    }

    private void checkCache(GpuBuffer keys, KvDType type, int rows, int stride, int offset) {
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(type, "keyType");
        if (rows < 1) {
            throw new IllegalStateException(where + ": the indexer cache needs at least one row, got " + rows);
        }
        if (stride < 1 || offset < 0 || (long) offset + cfg.headDim() > stride) {
            throw new IllegalStateException(where + ": key rows need a positive stride with the " + cfg.headDim()
                    + "-value key inside it, got stride " + stride + " at offset " + offset);
        }
        try {
            SparseKernels.checkIndexerCache(type, stride);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException(where + ": " + e.getMessage(), e);
        }
        long need = (long) rows * type.rowBytes(stride);
        if (keys.sizeInBytes() < need) {
            throw new IllegalStateException(where + ": indexer cache of " + keys.sizeInBytes() + " bytes is smaller"
                    + " than " + rows + " " + type + " rows of " + stride + " (" + need + " bytes)");
        }
    }

    private void checkTokens(int nTokens) {
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException(where + ": indexer step of " + nTokens + " tokens is outside [1, "
                    + ubatch + "]");
        }
    }

    private void ensureBound(boolean bound, String stage) {
        ensureOpen();
        if (!bound) {
            throw new IllegalStateException(where + ": indexer " + stage + " stage or its LoRA deltas are not"
                    + " bound");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(where + ": indexer is closed");
        }
    }

    /** Releases the prepared dispatches; scratch, weights and caches belong to the context. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = Closeables.closeAll(prepared, null);
        prepared.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuDsaIndexer{" + where + ", " + cfg + ", chunk=" + chunk + '}';
    }
}
