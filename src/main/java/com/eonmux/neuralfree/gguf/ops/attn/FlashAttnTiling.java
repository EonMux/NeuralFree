package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvDType;

/**
 * Tile plan of the flash-attention kernels for one attention shape on one device. Immutable.
 *
 * <p>{@link #decode} plans {@code LLM_FlashDecode_*}: one query head per workgroup, so a launch holds
 * {@code n_head} workgroups before splitting (the GQA group's K/V rows come from L2 across the concurrently running
 * workgroups). The split over keys is chosen per step from the live key count ({@link #decodeSplit}), as
 * llama.cpp sizes {@code parallel_blocks} from the live KV length and the SM count
 * (ggml-cuda/fattn-common.cuh:1141-1214): a step over {@code n} keys aims at two workgroups per compute unit, with
 * {@code split_len} between {@link #minSplitLen()} and {@link #maxSplitLen()}. Shared memory holds
 * {@code d_k + split_len + PASS2_GROUPS * d_v} floats; the output accumulators hold {@code d_v <= 128 * 32} values.
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
 *
 * <p>{@link #decodePaged} plans the paged decode kernels {@code LLM_FlashDecodePaged_*} one launch over the batch's
 * sequences: the constraints of {@link #decode}, with the split length chosen over the largest per-sequence slot
 * range, and the n_seq of the launch entering the grid and partial sizes {@link #decodePagedGlobal} /
 * {@link #decodePagedPartialFloats}.
 *
 * <p>{@link #decodeSubgroup}, {@link #decodePagedSubgroup} and {@link #prefillSubgroup} plan the subgroup kernels
 * ({@link SubgroupFlashAttention}) instead, in work-groups of {@link SubgroupFlashAttention#WORK_GROUP} items that keep
 * their state in registers ({@link #subgroup}): a decode plan runs one head per work-group with no shared budget
 * bounding its split length, a {@link Plan#TILED} plan {@link SubgroupFlashAttention#PREFILL_ROWS} query rows per
 * work-group over key blocks of {@link SubgroupFlashAttention#PREFILL_BLOCK}; the split policies are otherwise the
 * portable ones. {@link #decodeGlobal}, {@link #decodePagedGlobal}, {@link #prefillGlobal}, {@link #decodeLocalSize}
 * and {@link #prefillLocalSize} follow the plan's form.
 *
 * <p>The portable paged decode plan also sizes the CUDA grouped-query paged decode {@code LLM_FlashDecodePagedGqa_*}
 * ({@link #gqaDecodeServes}, {@link #gqaDecodeUsable}): one work-group per (sequence, KV head, split) holding every
 * query head of the group, with the split chosen over the launch's work-groups ({@link #decodeGqaSplit}) inside the
 * partial buffer this plan sizes.
 */
public final class FlashAttnTiling {

    /** Which kernel runs and how. */
    public enum Plan {
        /** {@code LLM_FlashDecode_*} for one query row per call. */
        DECODE,
        /** {@code LLM_FlashAttn_*} over many query rows. */
        TILED,
        /** {@code LLM_FlashAttnMma_*} over many query rows on a tensor-core backend. */
        MMA,
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
        /**
         * True when one split covers the range, so the kernel writes the output itself ({@link AttnKernels#FLAG_DIRECT}
         * set).
         */
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
    /** Output accumulators of one workgroup ({@code NF_FA_WG * NF_FA_ACC_MAX}); the tiled and MMA kernels' pair bound. */
    public static final int ACC_PER_GROUP = LlmKernelConstants.FA_WG * LlmKernelConstants.FA_ACC_MAX;
    /**
     * Output pairs of one decode workgroup ({@code (NF_FA_WG / (NF_FA_WG / 32)) * NF_FA_ACC_MAX}): the value pass
     * spreads the {@code g_tile * d_v} pairs over each lane group's 32 items (pair pi at lane {@code pi % 32},
     * accumulator {@code pi / 32}), so every lane group can hold the full accumulator for its key stream.
     */
    public static final int DECODE_ACC_PER_GROUP = (LlmKernelConstants.FA_WG / (LlmKernelConstants.FA_WG / 32))
            * LlmKernelConstants.FA_ACC_MAX;
    /** Largest {@code g_tile} ({@code NF_FD_MAX_G}). */
    public static final int MAX_G_TILE = LlmKernelConstants.FD_MAX_G;
    /**
     * Head-dim limit of the decode kernels' shuffle fast path, one coalesced row float4 per lane
     * ({@code NF_FA_FAST_DIM}); the kernel gate refuses wider rows and serves them with the scalar body.
     */
    public static final int FAST_DIM = LlmKernelConstants.FA_FAST_DIM;

    /** Decode splits are multiples of this many keys. */
    static final int SPLIT_QUANTUM = 32;
    /**
     * Decode splits are at least this long when shared memory allows: the split quantum, so short contexts
     * still split over several workgroups (llama.cpp picks {@code parallel_blocks} from the live key count the
     * same way, fattn-common.cuh:1141-1214).
     */
    static final int MIN_SPLIT = SPLIT_QUANTUM;
    /**
     * Lane groups of one decode workgroup ({@code NF_FA_WG / 32}) that share the value pass's key loop;
     * their accumulators occupy this many times {@code g_tile * d_v} shared floats.
     */
    public static final int PASS2_GROUPS = LlmKernelConstants.FA_WG / 32;
    /**
     * Longest split of a subgroup decode plan. Its kernels hold no keys in shared memory, so the bound only keeps
     * the split policy independent of the plan's key range: the dense and paged plans of a layer then split a step
     * over the same keys alike.
     */
    static final int SUBGROUP_MAX_SPLIT = 1 << 16;
    /**
     * Compute units the attention plans assume on a device that reports none
     * ({@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}: Mesa ANV and other Vulkan drivers without a shader-core
     * property); see {@link #planningUnits}.
     */
    public static final int UNREPORTED_COMPUTE_UNITS = 32;
    /** Upper bound of prefill splits per call. */
    static final int MAX_PREFILL_SPLITS = 16;
    /** A prefill split covers at least this many key tiles. */
    static final int MIN_TILES_PER_SPLIT = 4;
    private static final int[] BQ_CHOICES = {32, 16, 8, 4, 2, 1};
    private static final int[] BK_CHOICES = {32, 16};
    /** Query rows per mma work-group ({@code NF_FM_BM}); {@code bq * g_tile == MMA_ROWS} always. */
    public static final int MMA_ROWS = 64;
    /**
     * Keys per mma tile ({@code NF_FM_BN} of kernels/llm/flash_mma.cl and of its Vulkan twin
     * shaders/llm/flash_mma_cm2.comp), the granularity the mma plan rounds its split lengths to; the Vulkan twin
     * also masks a split's partial last tile.
     */
    public static final int MMA_BK = 64;
    /** Longest split of the grouped-query paged decode ({@code NF_FDG_MAX_SPLIT}, its staged key rows). */
    public static final int GQA_MAX_SPLIT = 512;
    /** Most query heads per KV head of the grouped-query paged decode ({@code NF_FDG_MAX_G}). */
    public static final int GQA_MAX_G = 4;
    /**
     * Work-groups per compute unit a grouped-query paged decode launch aims at when it splits the keys: the
     * kernel keeps four work-groups resident per unit, and two per unit keep every unit busy while the combine
     * of a split launch stays short.
     */
    static final int GQA_GROUPS_PER_UNIT = 2;
    /** Largest mma head dim (d_k and d_v multiples of 16 of at most this). */
    public static final int MMA_MAX_HEAD_DIM = 128;
    /** Head tiles of the mma plan: at most this many query heads per work-group. */
    public static final int MMA_MAX_G_TILE = 8;

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
    private final boolean decodeFastPath;
    /** Whether the plan's launches run the subgroup kernels ({@link SubgroupFlashAttention}). */
    private final boolean subgroup;

    private FlashAttnTiling(Plan plan, int nHead, int nHeadKv, int dK, int dV, int smemFloats, int computeUnits,
                            int gTile, int keyCapacity, int wantedSplits, int minSplitLen, int maxSplitLen,
                            int maxSplits, int bq, int bk, boolean decodeFastPath, boolean subgroup) {
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
        this.decodeFastPath = decodeFastPath;
        this.subgroup = subgroup;
    }

    /**
     * Whether the decode kernels' warp fast path ({@code nf_fd_fast_ok} in flash_decode.cl and flash_decode.comp)
     * serves these caches and head dims on a backend that runs it ({@code LlmKernels.decodeWarpPath()}: CUDA and HIP,
     * whose dialect defines {@code NF_HAS_SG_SHUFFLE}, and Vulkan devices whose {@code __sg} modules run on 32-lane
     * subgroups): plain F32/F16/BF16 cache element types only, head
     * dims of at most {@link #FAST_DIM} (one coalesced row float4 per lane), and the decode plans'
     * {@code g_tile} of 1 (at most {@link #PASS2_GROUPS} lane groups of registers). Quantized, FP8 and latent
     * caches, wider head dims and backends without the warp path run the portable scalar body.
     */
    public static boolean decodeFastPath(boolean warpPath, KvDType typeK, KvDType typeV, int dK, int dV) {
        return warpPath
                && (typeK == KvDType.F32 || typeK == KvDType.F16 || typeK == KvDType.BF16)
                && (typeV == KvDType.F32 || typeV == KvDType.F16 || typeV == KvDType.BF16)
                && dK <= FAST_DIM && dV <= FAST_DIM;
    }

    /**
     * Decode plan.
     *
     * @param nKvMax         largest key range a step scans (cache rows); bounds {@link #decodeSplit} and sizes
     *                       {@link #maxSplits()}
     * @param smemFloats     {@code LlmKernels.faSmemFloats()}
     * @param computeUnits   device compute units (SMs / CUs), used to choose the split count of each step
     * @param decodeFastPath whether the device runs the decode kernels' shuffle fast path for this shape
     *                       ({@link #decodeFastPath}); it caps {@link #maxSplitLen()} at the workgroup size
     * @throws IllegalArgumentException for invalid shapes, or when even one head per workgroup does not fit
     */
    public static FlashAttnTiling decode(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                         int computeUnits, boolean decodeFastPath) {
        validate(nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
        return planDecode(Plan.DECODE, nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits, decodeFastPath);
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
            // The per-row fallback keeps the portable split policy: a shape no tiled kernel fits has no
            // fast-path shape either (tiny shared budgets), and the scalar body serves it.
            return planDecode(Plan.PER_ROW_DECODE, nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits, false);
        }
        return new FlashAttnTiling(Plan.TILED, nHead, nHeadKv, dK, dV, smemFloats, computeUnits,
                bestG, nKvMax, 0, 0, 0, 0, bestRows / bestG, bestBk, false, false);
    }

    /**
     * Tensor-core prefill plan ({@link Plan#MMA}, kernel {@code LLM_FlashAttnMma_*}): the mma work-group carries
     * {@value MMA_ROWS} query rows ({@code bq * g_tile == MMA_ROWS}, {@code g_tile <= MMA_MAX_G_TILE}) over key
     * tiles of {@value MMA_BK}; the head dims must be positive multiples of 16 of at most {@value
     * #MMA_MAX_HEAD_DIM} and the staged tiles must fit {@code smemFloats}. Null when the shape does not support
     * the mma kernel, so the caller keeps the {@link #prefill} plan.
     *
     * @throws IllegalArgumentException for shapes that fail the shared validation (a supported shape never does)
     */
    public static FlashAttnTiling mma(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                      int computeUnits) {
        validate(nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
        if (dK > MMA_MAX_HEAD_DIM || dV > MMA_MAX_HEAD_DIM || dK % 16 != 0 || dV % 16 != 0
                || mmaSmemFloats(dK, dV) > smemFloats) {
            return null;
        }
        int g0 = nHead / nHeadKv;
        int gTile = Math.min(Math.min(g0, MMA_MAX_G_TILE), MMA_ROWS);
        while (MMA_ROWS % gTile != 0) {
            gTile--;
        }
        int bq = MMA_ROWS / gTile;
        return new FlashAttnTiling(Plan.MMA, nHead, nHeadKv, dK, dV, smemFloats, computeUnits,
                gTile, nKvMax, 0, 0, 0, 0, bq, MMA_BK, false, false);
    }

    /**
     * Subgroup prefill plan ({@link Plan#TILED} over {@code LLM_FlashAttnSg_*}): {@code bq * g_tile ==}
     * {@link SubgroupFlashAttention#PREFILL_ROWS} query rows per work-group, with {@code g_tile} the largest power of
     * two of at most {@code min(n_head / n_head_kv, PREFILL_ROWS)} (a subgroup's two rows are then one token's two
     * heads whenever the GQA group has two or more), and key blocks of {@code bk =}
     * {@link SubgroupFlashAttention#PREFILL_BLOCK}, the granularity of {@link #prefillSplit}.
     *
     * @throws IllegalArgumentException for invalid shapes, or head dims the subgroup kernels do not serve
     */
    public static FlashAttnTiling prefillSubgroup(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                                  int computeUnits) {
        validate(nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
        requireSubgroupDims(dK, dV);
        int g0 = nHead / nHeadKv;
        int gTile = Integer.highestOneBit(Math.min(g0, SubgroupFlashAttention.PREFILL_ROWS));
        return new FlashAttnTiling(Plan.TILED, nHead, nHeadKv, dK, dV, smemFloats, computeUnits,
                gTile, nKvMax, 0, 0, 0, 0, SubgroupFlashAttention.PREFILL_ROWS / gTile,
                SubgroupFlashAttention.PREFILL_BLOCK, false, true);
    }

    /**
     * The compute units the attention plans of a device split for: the count it reports, or
     * {@link #UNREPORTED_COMPUTE_UNITS} when it reports none. Planned for one unit, such a device never splits a
     * decode step, since a step aims at {@code ceil(2 * units / n_head)} splits: a 4096-key step of a 32-head layer
     * then runs 32 work-groups of four 16-lane subgroups on the Arrow-Lake iGPU under Mesa ANV, whose 64 EUs (8
     * Xe-cores) hold 512 such subgroup threads. llama.cpp's Vulkan backend assumes 16 shader cores in that case, which
     * under this formula still leaves a 32-head layer unsplit. Measured on that device (Vulkan, 32-token decode at
     * a depth of 4096, tokens/s), planning for 1, 16, 32 and 64 units: Qwen3 4B Q8_0 (32 heads) 8.03, 7.95, 9.43 and
     * 9.35 (llama.cpp Vulkan 9.34); Qwen3 0.6B Q8_0 (16 heads) 18.9 at 1, 37.5 at 32 and 34.8 at 64 units. 32 units
     * fill 256 subgroup threads with one step of either model; 64, the count the device reports through OpenCL (its
     * EUs), adds partials to merge without a faster step.
     */
    public static int planningUnits(DeviceCapabilities caps) {
        int units = caps.getComputeUnits();
        return units > DeviceCapabilities.UNKNOWN_COMPUTE_UNITS ? units : UNREPORTED_COMPUTE_UNITS;
    }

    /** The head-dim conditions of the subgroup kernels ({@link SubgroupFlashAttention#shapeServes}). */
    private static void requireSubgroupDims(int dK, int dV) {
        if (dK != dV || dK % SubgroupFlashAttention.DIMS_PER_LANE != 0 || dK > SubgroupFlashAttention.MAX_HEAD_DIM) {
            throw new IllegalArgumentException("the subgroup attention kernels need d_k == d_v, a multiple of "
                    + SubgroupFlashAttention.DIMS_PER_LANE + " of at most " + SubgroupFlashAttention.MAX_HEAD_DIM
                    + ", got d_k=" + dK + ", d_v=" + dV);
        }
    }

    /**
     * Shared-memory floats the mma kernel stages for d_k and d_v (kernels/llm/flash_mma.cl nf_fm_smem_floats): the
     * K tile, which first holds the Q tile ({@code MMA_ROWS <= MMA_BK}), and the V tile, {@code MMA_BK} rows of
     * {@code d + 8} halves each, plus the tile's key positions.
     */
    public static long mmaSmemFloats(int dK, int dV) {
        long halves = (long) MMA_BK * (dK + 8) + (long) MMA_BK * (dV + 8);
        return halves / 2 + MMA_BK;
    }

    /**
     * Decode plan over a batch's paged decode sequences one launch: the plan constraints of {@link #decode}, with
     * the split length chosen over {@code maxKvLen}, the largest per-sequence slot range any step scans (the
     * longest {@code seq_kv_len}; {@link #decodeSplit} sizes each step from that sequence's live length, and every
     * step fits this plan). {@code nSeq} enters only the dispatch and partial sizes of a paged call
     * ({@link #decodePagedGlobal}, {@link #decodePagedPartialFloats}), which the caller fixes per launch.
     *
     * @param nSeq        sequences of the launch
     * @param maxKvLen    largest per-sequence slot range a step scans ({@code ceil(nCtx / pageSize)} at the most)
     * @param smemFloats  {@code LlmKernels.faSmemFloats()}
     * @param computeUnits device compute units (SMs / CUs), used to choose the split count of each step
     * @throws IllegalArgumentException for invalid shapes, or when even one head per workgroup does not fit
     */
    public static FlashAttnTiling decodePaged(int nSeq, int nHead, int nHeadKv, int dK, int dV, int maxKvLen,
                                              int smemFloats, int computeUnits, boolean decodeFastPath) {
        if (nSeq <= 0) {
            throw new IllegalArgumentException("paged decode needs at least one sequence, got " + nSeq);
        }
        validate(nHead, nHeadKv, dK, dV, maxKvLen, smemFloats, computeUnits);
        return planDecode(Plan.DECODE, nHead, nHeadKv, dK, dV, maxKvLen, smemFloats, computeUnits, decodeFastPath);
    }

    private static FlashAttnTiling planDecode(Plan plan, int nHead, int nHeadKv, int dK, int dV, int nKvMax,
                                              int smemFloats, int computeUnits, boolean decodeFastPath) {
        int g0 = nHead / nHeadKv;
        // One query head per work-group: the most work-groups per launch (llama.cpp's fattn-decode shape,
        // where every query head is its own row over key splits). Short-key decode attention is bound by
        // outstanding memory requests, not bandwidth, and the GQA group's K/V rows still come from L2
        // across the concurrently running work-groups of one KV head.
        if (!decodeFits(1, dK, dV, smemFloats, decodeFastPath)) {
            // maxSplitLen shrinks as g grows, so no head tile fits either.
            throw new IllegalArgumentException(noPlanRefusal(dK, dV, smemFloats));
        }
        int g = 1;
        int maxLen = maxSplitLen(g, dK, dV, smemFloats, decodeFastPath);
        int minLen = Math.min(maxLen, MIN_SPLIT);
        long groups = (long) nHeadKv * ceilDiv(g0, g);
        int wanted = (int) Math.max(1L, ceilDiv(2L * computeUnits, groups));
        // decodeSplit gives count <= wanted unless split_len is clamped to maxLen, where count <= ceil(n / maxLen),
        // and count <= ceil(n / minLen) always; n <= nKvMax bounds all three.
        int maxSplits = Math.max(ceilDiv(nKvMax, maxLen), Math.min(wanted, ceilDiv(nKvMax, minLen)));
        return new FlashAttnTiling(plan, nHead, nHeadKv, dK, dV, smemFloats, computeUnits,
                g, nKvMax, wanted, minLen, maxLen, maxSplits, 0, 0, decodeFastPath, false);
    }

    /**
     * Subgroup decode plan ({@link SubgroupFlashAttention}): one query head per work-group, splits between
     * {@link SubgroupFlashAttention#WORK_GROUP} keys (one key per lane of the work-group's first scan) and
     * {@link #SUBGROUP_MAX_SPLIT}, aiming at two work-groups per compute unit as {@link #decode} does.
     *
     * @throws IllegalArgumentException for invalid shapes, or caches and head dims the subgroup kernels do not
     *                                  serve ({@link SubgroupFlashAttention#shapeServes} with equal plain types)
     */
    public static FlashAttnTiling decodeSubgroup(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                                 int computeUnits) {
        validate(nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
        return planSubgroupDecode(nHead, nHeadKv, dK, dV, nKvMax, smemFloats, computeUnits);
    }

    /**
     * Paged subgroup decode plan: {@link #decodeSubgroup} over {@code maxKvLen}, the largest per-sequence slot range,
     * as {@link #decodePaged} is {@link #decode} over it.
     *
     * @throws IllegalArgumentException for a non-positive {@code nSeq} and the refusals of {@link #decodeSubgroup}
     */
    public static FlashAttnTiling decodePagedSubgroup(int nSeq, int nHead, int nHeadKv, int dK, int dV, int maxKvLen,
                                                      int smemFloats, int computeUnits) {
        if (nSeq <= 0) {
            throw new IllegalArgumentException("paged decode needs at least one sequence, got " + nSeq);
        }
        return decodeSubgroup(nHead, nHeadKv, dK, dV, maxKvLen, smemFloats, computeUnits);
    }

    private static FlashAttnTiling planSubgroupDecode(int nHead, int nHeadKv, int dK, int dV, int nKvMax,
                                                      int smemFloats, int computeUnits) {
        requireSubgroupDims(dK, dV);
        int minLen = SubgroupFlashAttention.WORK_GROUP;
        int maxLen = SUBGROUP_MAX_SPLIT;
        int wanted = (int) Math.max(1L, ceilDiv(2L * computeUnits, (long) nHead));
        int maxSplits = Math.max(ceilDiv(nKvMax, maxLen), Math.min(wanted, ceilDiv(nKvMax, minLen)));
        return new FlashAttnTiling(Plan.DECODE, nHead, nHeadKv, dK, dV, smemFloats, computeUnits,
                1, nKvMax, wanted, minLen, maxLen, maxSplits, 0, 0, false, true);
    }

    private static boolean decodeFits(int g, int dK, int dV, int smemFloats, boolean decodeFastPath) {
        return (long) g * dV <= DECODE_ACC_PER_GROUP
                && maxSplitLen(g, dK, dV, smemFloats, decodeFastPath) >= SPLIT_QUANTUM;
    }

    /**
     * Longest decode split for {@code g} heads: {@code roundDown32((smem - g*d_k - PASS2_GROUPS*g*d_v) / g)}.
     * On a device that runs the decode kernels' shuffle fast path the value is capped at the workgroup size:
     * each workgroup warp scans 32 keys of the split (one coalesced row float4 per lane, butterfly-reduced),
     * so longer splits only stretch the per-block serial scan without adding parallel warps. Everywhere else
     * the portable scalar body reads the split's keys strided over the whole workgroup and shares the longest
     * splits the shared memory allows (llama.cpp's fattn-decode shape).
     */
    private static int maxSplitLen(int g, int dK, int dV, int smemFloats, boolean decodeFastPath) {
        long free = (long) smemFloats - (long) g * dK - (long) PASS2_GROUPS * g * dV;
        if (free <= 0) {
            return 0;
        }
        long len = free / g / SPLIT_QUANTUM * SPLIT_QUANTUM;
        return decodeFastPath ? (int) Math.min(len, WG) : (int) len;
    }

    /** Shared floats of the tiled kernel for {@code rows} query rows and key tile {@code bk}. */
    static long tiledSmem(int rows, int bk) {
        return (long) rows * CHUNK + (long) bk * CHUNK + (long) rows * bk + 3L * rows;
    }

    private static void validate(int nHead, int nHeadKv, int dK, int dV, int nKvMax, int smemFloats,
                                 int computeUnits) {
        String shape = shapeRefusal(nHead, nHeadKv, dK, dV);
        if (shape != null) {
            throw new IllegalArgumentException(shape);
        }
        if (nKvMax <= 0) {
            throw new IllegalArgumentException("key range must be positive, got " + nKvMax);
        }
        if (smemFloats <= 0 || computeUnits <= 0) {
            throw new IllegalArgumentException("shared floats and compute units must be positive, got " + smemFloats
                    + " and " + computeUnits);
        }
    }

    /**
     * Why these head counts and head dims are a shape the flash kernels cannot serve — a head count that is not a
     * positive multiple of the KV-head count, or head dims that are not positive multiples of 4 — or null when the
     * shape itself is valid. The shared body of the factories' {@code IllegalArgumentException} checks and of
     * {@link #geometryRefusal}.
     */
    private static String shapeRefusal(int nHead, int nHeadKv, int dK, int dV) {
        if (nHead <= 0 || nHeadKv <= 0 || nHead % nHeadKv != 0) {
            return "flash attention needs n_head a positive multiple of n_head_kv, got " + nHead + "/" + nHeadKv;
        }
        if (dK <= 0 || dV <= 0 || dK % 4 != 0 || dV % 4 != 0) {
            return "flash attention needs head dims that are positive multiples of 4, got d_k=" + dK + ", d_v=" + dV;
        }
        return null;
    }

    /** The refusal of a shape whose one-head decode tile no shared budget stages (see {@link #decodeFits}). */
    private static String noPlanRefusal(int dK, int dV, int smemFloats) {
        return "no flash-attention plan for d_k=" + dK + ", d_v=" + dV + " with " + smemFloats
                + " shared floats: one head needs d_k + " + PASS2_GROUPS + " * d_v + " + SPLIT_QUANTUM
                + " floats and d_v <= " + DECODE_ACC_PER_GROUP;
    }

    /**
     * Why no flash kernel serves these head counts and dims on a device with {@code smemFloats} shared floats — the
     * model geometry ({@link #shapeRefusal}) or a one-head decode tile wider than the shared budget — or null when a
     * plan exists. {@code smemFloats <= 0} skips the budget check; the factories reject it themselves. The GPU bind
     * sites translate a non-null refusal into a {@code GpuModelUnsupportedException}, so a model whose attention
     * geometry no GPU flash kernel serves falls back to the CPU engine; the factories themselves throw
     * {@code IllegalArgumentException} for the same shapes.
     */
    public static String geometryRefusal(int nHead, int nHeadKv, int dK, int dV, int smemFloats) {
        String shape = shapeRefusal(nHead, nHeadKv, dK, dV);
        if (shape != null) {
            return shape;
        }
        if (smemFloats > 0 && !decodeFits(1, dK, dV, smemFloats, false)) {
            return noPlanRefusal(dK, dV, smemFloats);
        }
        return null;
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
    /** Whether the device runs the decode kernels' shuffle fast path for this plan ({@link #decodeFastPath}). */
    public boolean decodeFastPath() { return decodeFastPath; }
    /**
     * Whether the plan runs the subgroup kernels ({@link #decodeSubgroup}, {@link #decodePagedSubgroup},
     * {@link #prefillSubgroup}).
     */
    public boolean subgroup() { return subgroup; }
    /** Query tokens per workgroup ({@code bq}); 0 unless the plan is tiled or mma. */
    public int bq() { return bq; }
    /** Keys per tile ({@code bk}); 0 unless the plan is tiled or mma. */
    public int bk() { return bk; }
    /** Query rows per workgroup: {@code bq * g_tile} for {@link Plan#TILED} and {@link Plan#MMA}, {@code g_tile} otherwise. */
    public int rows() { return plan == Plan.TILED || plan == Plan.MMA ? bq * gTile : gTile; }

    /** Most shared floats the plan uses (excluding the kernels' fixed reduction and statistics arrays). */
    public long smemUsed() {
        if (plan == Plan.MMA) {
            return mmaSmemFloats(dK, dV);
        }
        if (plan == Plan.TILED) {
            return subgroup ? (long) SubgroupFlashAttention.PREFILL_ROWS * SubgroupFlashAttention.MAX_HEAD_DIM
                    : tiledSmem(bq * gTile, bk);
        }
        if (subgroup) {
            return (long) SubgroupFlashAttention.SUBGROUPS * (SubgroupFlashAttention.MAX_HEAD_DIM + 2);
        }
        return (long) gTile * (dK + maxSplitLen) + (long) PASS2_GROUPS * gTile * dV;
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
        if (plan == Plan.TILED || plan == Plan.MMA) {
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

    /**
     * Whether the grouped-query paged decode serves these head counts and dims: {@code d_k == d_v} in
     * {@code {64, 128}} and at most {@link #GQA_MAX_G} query heads per KV head (the kernel's {@code nf_fdg_plan_ok};
     * the caller checks the cache type, the strides and the V alias).
     */
    public static boolean gqaDecodeServes(int nHead, int nHeadKv, int dK, int dV) {
        return nHead > 0 && nHeadKv > 0 && nHead % nHeadKv == 0 && nHead / nHeadKv <= GQA_MAX_G
                && dK == dV && (dK == 64 || dK == 128);
    }

    /**
     * Whether this decode plan can size the grouped-query paged decode at every key count up to
     * {@link #keyCapacity()}: its shape is served and {@link #maxSplits()} splits of at most
     * {@link #GQA_MAX_SPLIT} keys cover the capacity, so {@link #decodeGqaSplit} always fits the partial buffer.
     */
    public boolean gqaDecodeUsable() {
        return (plan == Plan.DECODE || plan == Plan.PER_ROW_DECODE) && gqaDecodeServes(nHead, nHeadKv, dK, dV)
                && (long) maxSplits * GQA_MAX_SPLIT >= keyCapacity;
    }

    /**
     * Splits of a grouped-query paged decode over {@code nSeq} sequences of at most {@code nKeys} keys: the launch
     * aims at {@code wanted = ceil(GQA_GROUPS_PER_UNIT * CU / (n_seq * n_head_kv))} splits, so {@code split_len =
     * clamp(roundUp32(ceil(nKeys / wanted)), 32, GQA_MAX_SPLIT)}, raised to {@code roundUp32(ceil(nKeys /
     * maxSplits))} so the splits fit the partial buffer, and {@code n_split = ceil(nKeys / split_len)} (at least 1).
     *
     * @throws IllegalStateException    unless {@link #gqaDecodeUsable()}
     * @throws IllegalArgumentException when {@code nKeys} is outside [0, {@link #keyCapacity()}] or {@code nSeq}
     *                                  is not positive
     */
    public Split decodeGqaSplit(int nKeys, int nSeq) {
        if (!gqaDecodeUsable()) {
            throw new IllegalStateException("the grouped-query paged decode does not serve " + this);
        }
        if (nKeys < 0 || nKeys > keyCapacity) {
            throw new IllegalArgumentException("key count " + nKeys + " is outside [0, " + keyCapacity
                    + "], the key range of " + this);
        }
        if (nSeq <= 0) {
            throw new IllegalArgumentException("paged decode needs at least one sequence, got " + nSeq);
        }
        long groups = (long) nSeq * nHeadKv;
        int wanted = (int) Math.max(1L, ceilDiv((long) GQA_GROUPS_PER_UNIT * computeUnits, groups));
        int len = Math.min(Math.max(roundUp(ceilDiv(nKeys, wanted), SPLIT_QUANTUM), MIN_SPLIT), GQA_MAX_SPLIT);
        len = Math.max(len, roundUp(ceilDiv(nKeys, maxSplits), SPLIT_QUANTUM));
        return new Split(Math.max(1, ceilDiv(nKeys, len)), len);
    }

    /**
     * Global size of a grouped-query paged decode call: {@code {n_split * 128, n_seq * n_head_kv}}, the sequence
     * folded into dimension 1 in front of the KV head.
     */
    public static long[] decodeGqaGlobal(int nSeq, int nHeadKv, int nSplit) {
        return new long[]{(long) nSplit * WG, (long) nSeq * nHeadKv};
    }

    /** Global size of a decode call: {@code {n_split * wg, n_head_kv * groupTiles}} ({@link #decodeLocalSize}). */
    public long[] decodeGlobal(int nSplit) {
        return new long[]{(long) nSplit * workGroup(), (long) nHeadKv * groupTiles()};
    }

    /**
     * Global size of a paged decode call for {@code nSeq} sequences and {@code nSplit} splits:
     * {@code {n_split * wg, n_seq * n_head_kv * groupTiles}} ({@link #decodeLocalSize}), the sequence folded into dimension 1 in front of
     * the KV head and its tiles ({@code gid1 = s * n_head_kv * groupTiles + kvh * groupTiles + gt}), the fold the
     * {@code LLM_FlashDecodePaged_*} kernels subtract before the KV head index.
     *
     * @throws IllegalStateException    for a {@link Plan#TILED} plan
     * @throws IllegalArgumentException when {@code nSeq} is not positive
     */
    public long[] decodePagedGlobal(int nSeq, int nSplit) {
        if (plan == Plan.TILED || plan == Plan.MMA) {
            throw new IllegalStateException("decodePagedGlobal needs a decode plan, this is " + plan);
        }
        if (nSeq <= 0) {
            throw new IllegalArgumentException("paged decode needs at least one sequence, got " + nSeq);
        }
        return new long[]{(long) nSplit * workGroup(), (long) nSeq * nHeadKv * groupTiles()};
    }

    /** Local size of a portable decode call ({@code LLM_FlashDecode_*}, {@code LLM_FlashDecodePaged_*}). */
    public static long[] decodeLocal() {
        return new long[]{WG, 1};
    }

    /** Local size of this plan's decode calls: {@link #decodeLocal}, or the subgroup kernels' work-group. */
    public long[] decodeLocalSize() {
        return new long[]{workGroup(), 1};
    }

    private int workGroup() {
        return subgroup ? SubgroupFlashAttention.WORK_GROUP : WG;
    }

    /** Partial-buffer floats of one decode query row at {@link #maxSplits()}: {@code n_head * maxSplits * (d_v + 2)}. */
    public long decodePartialFloats() {
        if (plan == Plan.TILED || plan == Plan.MMA) {
            throw new IllegalStateException("decodePartialFloats needs a decode plan, this is " + plan);
        }
        return partialFloats(nHead, maxSplits, dV);
    }

    /**
     * Partial-buffer floats of one paged decode call for {@code nSeq} sequences at {@link #maxSplits()}:
     * {@code n_seq * n_head * maxSplits * (d_v + 2)} ({@code LLM_FlashAttn_Combine} merges paged partials with
     * {@code rows = n_seq * n_head}).
     *
     * @throws IllegalStateException for a {@link Plan#TILED} plan
     */
    public long decodePagedPartialFloats(int nSeq) {
        if (plan == Plan.TILED || plan == Plan.MMA) {
            throw new IllegalStateException("decodePagedPartialFloats needs a decode plan, this is " + plan);
        }
        if (nSeq <= 0) {
            throw new IllegalArgumentException("paged decode needs at least one sequence, got " + nSeq);
        }
        return partialFloats((long) nSeq * nHead, maxSplits, dV);
    }

    /**
     * Splits of a tiled prefill call. One split unless the call has fewer than two workgroups per compute unit; then
     * up to {@value #MAX_PREFILL_SPLITS} splits of at least {@value #MIN_TILES_PER_SPLIT} key tiles that fit
     * {@code partialCapacityFloats} ({@code n_tokens * n_head * n_split * (d_v + 2)}). A split call needs dense
     * output rows ({@code out_row_stride == n_head * d_v}), because the combine writes rows contiguously.
     *
     * @throws IllegalStateException    unless the plan is {@link Plan#TILED} or {@link Plan#MMA}
     * @throws IllegalArgumentException for a non-positive token count or negative key count or capacity
     */
    public Split prefillSplit(int nTokens, int nKeys, long partialCapacityFloats) {
        if (plan != Plan.TILED && plan != Plan.MMA) {
            throw new IllegalStateException("prefillSplit needs a TILED or MMA plan, this is " + plan);
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

    /**
     * Global size of a tiled prefill call: {@code {ceil(n_tokens / bq) * wg, n_head_kv * groupTiles, n_split}} for the
     * plan's {@link #prefillLocalSize}.
     */
    public long[] prefillGlobal(int nTokens, int nSplit) {
        if (plan != Plan.TILED && plan != Plan.MMA) {
            throw new IllegalStateException("prefillGlobal needs a TILED or MMA plan, this is " + plan);
        }
        return new long[]{(long) ceilDiv(nTokens, bq) * workGroup(), (long) nHeadKv * groupTiles(), nSplit};
    }

    /** Local size of a portable tiled or mma prefill call ({@code LLM_FlashAttn*}). */
    public static long[] prefillLocal() {
        return new long[]{WG, 1, 1};
    }

    /** Local size of this plan's prefill calls: {@link #prefillLocal}, or the subgroup kernels' work-group. */
    public long[] prefillLocalSize() {
        return new long[]{workGroup(), 1, 1};
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
        if (plan == Plan.TILED || plan == Plan.MMA) {
            return head + (subgroup ? ", subgroup" : "") + ", bq=" + bq + ", bk=" + bk + ", smem=" + smemUsed() + "/" + smemFloats + '}';
        }
        return head + (subgroup() ? ", subgroup" : "") + ", split_len=" + minSplitLen + ".." + maxSplitLen + ", wanted_splits=" + wantedSplits
                + ", max_splits=" + maxSplits + ", keys<=" + keyCapacity + ", smem=" + smemUsed() + "/" + smemFloats
                + '}';
    }
}
