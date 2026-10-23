package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

/**
 * Tile plan of the flash-attention kernels for one attention shape on one device. Immutable.
 *
 * <p>{@link #decode} plans {@code LLM_FlashDecode_*}: {@code g_tile} query heads per workgroup (GQA packing, each K/V
 * row read once per tile). The split over keys is chosen per step from the live key count ({@link #decodeSplit}), as
 * llama.cpp sizes {@code parallel_blocks} from the live KV length and the SM count
 * (ggml-cuda/fattn-common.cuh:1141-1214): a step over {@code n} keys aims at two workgroups per compute unit, with
 * {@code split_len} between {@link #minSplitLen()} and {@link #maxSplitLen()}. Shared memory holds
 * {@code g_tile * (d_k + split_len) <= g_tile * (d_k + maxSplitLen)} floats; the output accumulators hold
 * {@code g_tile * d_v <= 128 * 32} values.
 *
 * <p>{@link #prefill} plans {@code LLM_FlashAttn_*}: {@code R = bq * g_tile} query rows per workgroup, key tiles of
 * {@code bk}, head dimension staged in chunks of {@value #CHUNK}. Shared memory holds
 * {@code R*64 + bk*64 + R*bk + 3R} floats (Q chunk, K/V chunk, score tile, row max / sum / rescale) and the output
 * accumulators {@code R * d_v <= 128 * 32} values. When no tiling fits, the plan is {@link Plan#PER_ROW_DECODE}: the
 * prefill runs the decode kernel once per query row with this object's decode parameters (never a CPU fallback).
 *
 * <p>For head dims that are positive multiples of 4 up to {@value #MAX_HEAD_DIM} and {@code smemFloats} of at least
 * {@code FaSmem.MIN_FLOATS} (3072), both factories always return a plan. The constraints here are the ones the
 * kernels check ({@code nf_fd_plan_ok}, {@code nf_fa_plan_ok}).
 */
public final class FlashAttnTiling {

    /** Which kernel runs and how. */
    public enum Plan {
        /** {@code LLM_FlashDecode_*} for one query row per call. */
        DECODE,
        /** {@code LLM_FlashAttn_*} over many query rows. */
        TILED,
        /** Prefill through {@code LLM_FlashDecode_*}, one call per query row (tiled budget infeasible). */
        PER_ROW_DECODE
    }

    /** Split count and keys per split of one call. Immutable. */
    public static final class Split {
        private final int count;
        private final int length;

        Split(int count, int length) {
            this.count = count;
            this.length = length;
        }

        /** {@code n_split}. */
        public int count() { return count; }
        /** {@code split_len}. */
        public int length() { return length; }
        /** True when one split covers the range, so the kernel writes the output itself ({@code direct_out = 1}). */
        public boolean direct() { return count == 1; }

        @Override
        public String toString() {
            return "Split{" + count + " x " + length + '}';
        }
    }

    /** Workgroup size of every flash kernel ({@code NF_FA_WG}). */
    public static final int WG = LlmKernelConstants.FA_WG;
    /** Head-dimension chunk of the tiled kernel ({@code NF_FA_CHUNK}). */
    public static final int CHUNK = 64;
    /** Largest head dim for which a plan is guaranteed. */
    public static final int MAX_HEAD_DIM = 1024;
    /** Output accumulators of one workgroup ({@code NF_FA_WG * NF_FA_ACC_MAX}). */
    public static final int ACC_PER_GROUP = LlmKernelConstants.FA_WG * LlmKernelConstants.FA_ACC_MAX;
    /** Largest {@code g_tile} ({@code NF_FD_MAX_G}). */
    public static final int MAX_G_TILE = LlmKernelConstants.FD_MAX_G;

    /** Decode splits are multiples of this many keys. */
    static final int SPLIT_QUANTUM = 32;
    /**
     * Decode splits are at least this long when shared memory allows: one key per work-item, the granularity of
     * llama.cpp's vector decode kernel ({@code nbatch_fa = D}, 128 for the common head dim, fattn-vec.cuh:543).
     */
    static final int MIN_SPLIT = LlmKernelConstants.FA_WG;
    /** Upper bound of prefill splits per call. */
    static final int MAX_PREFILL_SPLITS = 16;
    /** A prefill split covers at least this many key tiles. */
    static final int MIN_TILES_PER_SPLIT = 4;
    private static final int[] BQ_CHOICES = {32, 16, 8, 4, 2, 1};
    private static final int[] BK_CHOICES = {32, 16};

    private final Plan plan;
    private final int nHead;
    private final int nHeadKv;
    private final int dK;
    private final int dV;
    private final int smemFloats;
    private final int computeUnits;
    private final int gTile;
    private final int keyCapacity;
    private final int wantedSplits;
    private final int minSplitLen;
    private final int maxSplitLen;
    private final int maxSplits;
    private final int bq;
    private final int bk;

    private FlashAttnTiling(Plan plan, int nHead, int nHeadKv, int dK, int dV, int smemFloats, int computeUnits,
                            int gTile, int keyCapacity, int wantedSplits, int minSplitLen, int maxSplitLen,
                            int maxSplits, int bq, int bk) {
        this.plan = plan;
        this.nHead = nHead;
        this.nHeadKv = nHeadKv;
        this.dK = dK;
        this.dV = dV;
        this.smemFloats = smemFloats;
        this.computeUnits = computeUnits;
        this.gTile = gTile;
        this.keyCapacity = keyCapacity;
        this.wantedSplits = wantedSplits;
        this.minSplitLen = minSplitLen;
        this.maxSplitLen = maxSplitLen;
        this.maxSplits = maxSplits;
        this.bq = bq;
        this.bk = bk;
    }

    /**
     * Decode plan.
     *
     * @param nKvMax       largest key range a step scans (cache rows); bounds {@link #decodeSplit} and sizes
     *                     {@link #maxSplits()}
     * @param smemFloats   {@code LlmKernels.faSmemFloats()}
     * @param computeUnits device compute units (SMs / CUs), used to choose the split count of each step
     * @throws IllegalArgumentException for invalid shapes, or when even one head per workgroup does not fit
     */
    public static FlashAttnTiling decode(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                         int computeUnits) {
        validate(nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
        return planDecode(Plan.DECODE, nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
    }

    /**
     * Prefill plan: {@link Plan#TILED} when a tiling fits, else {@link Plan#PER_ROW_DECODE} with decode parameters
     * for {@code nKvMax} keys.
     *
     * @throws IllegalArgumentException for invalid shapes, or when neither kernel fits
     */
    public static FlashAttnTiling prefill(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                          int computeUnits) {
        validate(nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
        int g0 = nHead / nHeadKv;
        int bestRows = 0;
        int bestBk = 0;
        int bestG = 0;
        int lastG = -1;
        for (int tiles = 1; tiles <= g0; tiles++) {
            int g = ceilDiv(g0, tiles);
            if (g == lastG || g > MAX_G_TILE) {
                continue;
            }
            lastG = g;
            for (int bqc : BQ_CHOICES) {
                int rows = bqc * g;
                if ((long) rows * dV > ACC_PER_GROUP) {
                    continue;
                }
                for (int bkc : BK_CHOICES) {
                    if (tiledSmem(rows, bkc) > smemFloats) {
                        continue;
                    }
                    if (rows > bestRows || (rows == bestRows && (bkc > bestBk || (bkc == bestBk && g > bestG)))) {
                        bestRows = rows;
                        bestBk = bkc;
                        bestG = g;
                    }
                }
            }
        }
        if (bestRows == 0) {
            return planDecode(Plan.PER_ROW_DECODE, nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
        }
        return new FlashAttnTiling(Plan.TILED, nHead, nHeadKv, dK, dV, smemFloats, computeUnits,
                bestG, nKvMax, 0, 0, 0, 0, bestRows / bestG, bestBk);
    }

    private static FlashAttnTiling planDecode(Plan plan, int nHead, int nHeadKv, int dK, int dV, int nKvMax,
                                              int smemFloats, int computeUnits) {
        int g0 = nHead / nHeadKv;
        int g = 0;
        // Largest balanced head tile whose queries take at most half the shared floats, leaving the rest for scores.
        for (int cap = Math.min(g0, MAX_G_TILE); cap >= 1 && g == 0; cap--) {
            int balanced = ceilDiv(g0, ceilDiv(g0, cap));
            if ((long) balanced * dK * 2 <= smemFloats && decodeFits(balanced, dK, dV, smemFloats)) {
                g = balanced;
            }
        }
        if (g == 0 && decodeFits(1, dK, dV, smemFloats)) {
            g = 1;
        }
        if (g == 0) {
            throw new IllegalArgumentException("no flash-attention plan for d_k=" + dK + ", d_v=" + dV + " with "
                    + smemFloats + " shared floats: one head needs d_k + " + SPLIT_QUANTUM
                    + " floats and d_v <= " + ACC_PER_GROUP);
        }
        int maxLen = maxSplitLen(g, dK, smemFloats);
        int minLen = Math.min(maxLen, MIN_SPLIT);
        long groups = (long) nHeadKv * ceilDiv(g0, g);
        int wanted = (int) Math.max(1L, ceilDiv(2L * computeUnits, groups));
        // decodeSplit gives count <= wanted unless split_len is clamped to maxLen, where count <= ceil(n / maxLen),
        // and count <= ceil(n / minLen) always; n <= nKvMax bounds all three.
        int maxSplits = Math.max(ceilDiv(nKvMax, maxLen), Math.min(wanted, ceilDiv(nKvMax, minLen)));
        return new FlashAttnTiling(plan, nHead, nHeadKv, dK, dV, smemFloats, computeUnits,
                g, nKvMax, wanted, minLen, maxLen, maxSplits, 0, 0);
    }

    private static boolean decodeFits(int g, int dK, int dV, int smemFloats) {
        return (long) g * dV <= ACC_PER_GROUP && maxSplitLen(g, dK, smemFloats) >= SPLIT_QUANTUM;
    }

    /** Longest decode split for {@code g} heads: {@code roundDown32((smem - g*d_k) / g)}. */
    private static int maxSplitLen(int g, int dK, int smemFloats) {
        long free = (long) smemFloats - (long) g * dK;
        if (free <= 0) {
            return 0;
        }
        return (int) (free / g / SPLIT_QUANTUM * SPLIT_QUANTUM);
    }

    /** Shared floats of the tiled kernel for {@code rows} query rows and key tile {@code bk}. */
    static long tiledSmem(int rows, int bk) {
        return (long) rows * CHUNK + (long) bk * CHUNK + (long) rows * bk + 3L * rows;
    }

    private static void validate(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                 int computeUnits) {
        if (nHead <= 0 || nHeadKv <= 0 || nHead % nHeadKv != 0) {
            throw new IllegalArgumentException("flash attention needs n_head a positive multiple of n_head_kv, got "
                    + nHead + "/" + nHeadKv);
        }
        if (dK <= 0 || dV <= 0 || dK % 4 != 0 || dV % 4 != 0) {
            throw new IllegalArgumentException("flash attention needs head dims that are positive multiples of 4, got d_k="
                    + dK + ", d_v=" + dV);
        }
        if (nKvMax <= 0) {
            throw new IllegalArgumentException("key range must be positive, got " + nKvMax);
        }
        if (smemFloats <= 0 || computeUnits <= 0) {
            throw new IllegalArgumentException("shared floats and compute units must be positive, got " + smemFloats
                    + " and " + computeUnits);
        }
    }

    public Plan plan() { return plan; }
    public int nHead() { return nHead; }
    public int nHeadKv() { return nHeadKv; }
    public int dK() { return dK; }
    public int dV() { return dV; }
    public int smemFloats() { return smemFloats; }
    /** Query heads per workgroup ({@code g_tile}). */
    public int gTile() { return gTile; }
    /** Head tiles per KV head: {@code ceil((n_head / n_head_kv) / g_tile)}. */
    public int groupTiles() { return ceilDiv(nHead / nHeadKv, gTile); }
    /** Largest key range a step may scan ({@code nKvMax} of the factory). */
    public int keyCapacity() { return keyCapacity; }
    /** Shortest decode {@code split_len} (a multiple of 32); 0 for {@link Plan#TILED}. */
    public int minSplitLen() { return minSplitLen; }
    /** Longest decode {@code split_len} shared memory allows (a multiple of 32); 0 for {@link Plan#TILED}. */
    public int maxSplitLen() { return maxSplitLen; }
    /** Most decode splits any step over at most {@link #keyCapacity()} keys uses; 0 for {@link Plan#TILED}. */
    public int maxSplits() { return maxSplits; }
    /** Query tokens per workgroup ({@code bq}); 0 unless {@link Plan#TILED}. */
    public int bq() { return bq; }
    /** Keys per tile ({@code bk}); 0 unless {@link Plan#TILED}. */
    public int bk() { return bk; }
    /** Query rows per workgroup: {@code bq * g_tile} for {@link Plan#TILED}, {@code g_tile} otherwise. */
    public int rows() { return plan == Plan.TILED ? bq * gTile : gTile; }

    /** Most shared floats the plan uses (excluding the kernels' fixed reduction and statistics arrays). */
    public long smemUsed() {
        return plan == Plan.TILED ? tiledSmem(bq * gTile, bk) : (long) gTile * (dK + maxSplitLen);
    }

    /**
     * Splits of a decode step over {@code nKeys} keys (rows {@code [kv_start, kv_start + nKeys)}): the step aims at
     * {@code wanted = ceil(2 * CU / (n_head_kv * groupTiles))} splits, so {@code split_len =
     * clamp(roundUp32(ceil(nKeys / wanted)), minSplitLen, maxSplitLen)} and {@code n_split = ceil(nKeys / split_len)}
     * (at least 1). The caller passes both {@code n_split} and {@code split_len} to the kernel on every call.
     *
     * @throws IllegalStateException    for a {@link Plan#TILED} plan
     * @throws IllegalArgumentException when {@code nKeys} is negative or exceeds {@link #keyCapacity()}
     */
    public Split decodeSplit(int nKeys) {
        if (plan == Plan.TILED) {
            throw new IllegalStateException("decodeSplit needs a decode plan, this is " + plan);
        }
        if (nKeys < 0 || nKeys > keyCapacity) {
            throw new IllegalArgumentException("key count " + nKeys + " is outside [0, " + keyCapacity
                    + "], the key range of " + this);
        }
        int len = roundUp(ceilDiv(nKeys, wantedSplits), SPLIT_QUANTUM);
        len = Math.min(Math.max(len, minSplitLen), maxSplitLen);
        int count = Math.max(1, ceilDiv(nKeys, len));
        if (count > maxSplits) {
            throw new IllegalStateException(nKeys + " keys need " + count + " splits of " + len + ", above the "
                    + maxSplits + " the partial buffer of " + this + " holds");
        }
        return new Split(count, len);
    }

    /** Global size of a decode call: {@code {n_split * 128, n_head_kv * groupTiles}}. */
    public long[] decodeGlobal(int nSplit) {
        return new long[]{(long) nSplit * WG, (long) nHeadKv * groupTiles()};
    }

    /** Local size of a decode call. */
    public static long[] decodeLocal() {
        return new long[]{WG, 1};
    }

    /** Partial-buffer floats of one decode query row at {@link #maxSplits()}: {@code n_head * maxSplits * (d_v + 2)}. */
    public long decodePartialFloats() {
        if (plan == Plan.TILED) {
            throw new IllegalStateException("decodePartialFloats needs a decode plan, this is " + plan);
        }
        return partialFloats(nHead, maxSplits, dV);
    }

    /**
     * Splits of a tiled prefill call. One split unless the call has fewer than two workgroups per compute unit; then
     * up to {@value #MAX_PREFILL_SPLITS} splits of at least {@value #MIN_TILES_PER_SPLIT} key tiles that fit
     * {@code partialCapacityFloats} ({@code n_tokens * n_head * n_split * (d_v + 2)}). A split call needs dense
     * output rows ({@code out_row_stride == n_head * d_v}), because the combine writes rows contiguously.
     *
     * @throws IllegalStateException    unless the plan is {@link Plan#TILED}
     * @throws IllegalArgumentException for a non-positive token count or negative key count or capacity
     */
    public Split prefillSplit(int nTokens, int nKeys, long partialCapacityFloats) {
        if (plan != Plan.TILED) {
            throw new IllegalStateException("prefillSplit needs a TILED plan, this is " + plan);
        }
        if (nTokens <= 0 || nKeys < 0 || partialCapacityFloats < 0) {
            throw new IllegalArgumentException("prefill split needs positive tokens and non-negative keys and capacity, got "
                    + nTokens + "/" + nKeys + "/" + partialCapacityFloats);
        }
        if (nKeys == 0) {
            return new Split(1, bk);
        }
        long groups = (long) ceilDiv(nTokens, bq) * nHeadKv * groupTiles();
        long count = 1;
        if (groups < 2L * computeUnits) {
            long wanted = ceilDiv(2L * computeUnits, groups);
            long byKeys = nKeys / ((long) MIN_TILES_PER_SPLIT * bk);
            long byCapacity = partialCapacityFloats / partialFloats(nTokens * (long) nHead, 1, dV);
            count = Math.max(1, Math.min(Math.min(wanted, MAX_PREFILL_SPLITS), Math.min(byKeys, byCapacity)));
        }
        int len = roundUp(ceilDiv(nKeys, (int) count), bk);
        return new Split(ceilDiv(nKeys, len), len);
    }

    /** Global size of a tiled prefill call: {@code {ceil(n_tokens / bq) * 128, n_head_kv * groupTiles, n_split}}. */
    public long[] prefillGlobal(int nTokens, int nSplit) {
        if (plan != Plan.TILED) {
            throw new IllegalStateException("prefillGlobal needs a TILED plan, this is " + plan);
        }
        return new long[]{(long) ceilDiv(nTokens, bq) * WG, (long) nHeadKv * groupTiles(), nSplit};
    }

    /** Local size of a tiled prefill call. */
    public static long[] prefillLocal() {
        return new long[]{WG, 1, 1};
    }

    /** Global size of {@code LLM_FlashAttn_Combine} for {@code nRows} rows. */
    public static long[] combineGlobal(int nRows) {
        return new long[]{(long) nRows * WG};
    }

    /** Local size of {@code LLM_FlashAttn_Combine}. */
    public static long[] combineLocal() {
        return new long[]{WG};
    }

    /** Partial-buffer floats for {@code rows} output rows and {@code nSplit} splits: {@code rows * nSplit * (dV + 2)}. */
    public static long partialFloats(long rows, int nSplit, int dV) {
        return rows * nSplit * (dV + 2L);
    }

    /** {@code ceil(a / b)} for {@code a >= 0, b > 0}, without overflow near {@code Integer.MAX_VALUE}. */
    private static int ceilDiv(int a, int b) {
        return a / b + (a % b != 0 ? 1 : 0);
    }

    private static long ceilDiv(long a, long b) {
        return (a + b - 1) / b;
    }

    private static int roundUp(int a, int m) {
        return ceilDiv(a, m) * m;
    }

    @Override
    public String toString() {
        String head = "FlashAttnTiling{" + plan + ", heads=" + nHead + "/" + nHeadKv + ", d=" + dK + "/" + dV
                + ", g_tile=" + gTile;
        if (plan == Plan.TILED) {
            return head + ", bq=" + bq + ", bk=" + bk + ", smem=" + smemUsed() + "/" + smemFloats + '}';
        }
        return head + ", split_len=" + minSplitLen + ".." + maxSplitLen + ", wanted_splits=" + wantedSplits
                + ", max_splits=" + maxSplits + ", keys<=" + keyCapacity + ", smem=" + smemUsed() + "/" + smemFloats
                + '}';
    }
}
