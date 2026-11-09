package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Objects;

/**
 * Immutable dispatch plan of one split-K GEMM over the fast GEMM lanes: the
 * {@code <T>_MatMul_MMA_KSplit}, {@code <T>_MatMul_MMQ_KSplit}, {@code <T>_MatMul_W4A8_KSplit} or
 * {@code <T>_MatMul_CM_KSplit} partial launch and its {@code LLM_SplitK_Reduce} reduce, with the
 * validated shape and view arguments, the work sizes and the argument arrays.
 *
 * <p>The partial kernel computes, per slice {@code s < splits} over k in
 * {@code [s * kSlice, min((s + 1) * kSlice, K))}, the plane
 * {@code partial[(s * N + t) * M + r] = sum_k W[r][k] * x[t][k]} with the tiles and paths of its
 * family: the MMA GEMM over a dense 16-bit weight ({@link Family#MMA},
 * kernels/llm/common/matmul_mma_family.h), the int8 MMQ GEMM over the staged activation records of
 * {@link GemmKernels#QUANT_MMQ_Q8_1} ({@link Family#MMQ}, kernels/llm/common/matmul_mmq_family.h), the
 * E4M3 GEMM over codes of a 4-bit or FP8 weight ({@link Family#W4A8}, kernels/llm/matmul_fp8.cl) or
 * the dequantizing CM GEMM ({@link Family#CM}, kernels/llm/matmul_cm.cl and
 * shaders/llm/matmul_mmq_cm.comp); the reduce sums the planes into {@code y[yOff + t*yStride + r]}
 * (kernels/llm/split_k_reduce.cl). Every slice covers whole stage depths ({@code kSlice} is a
 * multiple of the family's slice depth, and of the type's block size for the quantized families, so
 * a slice always starts at a block and chunk boundary) and the last slice is not empty
 * ({@code kSlice * (splits - 1) < K}), so the full product is covered exactly once. The fp32 plane
 * sums differ from the plain kernel's sequential accumulation by fp32 addition rounding.</p>
 *
 * <p>The heuristic that picks the split count (see {@link #splitsFor}) is llama.cpp's
 * {@code ggml_vk_guess_split_k}: only for K of 2048 and more, only while the tile grid fills less
 * than about two thirds of the device's compute units, capped at 8 slices. The staged launches of
 * fewer tokens than the device tile ({@link #ofSmallTiles}, {@link #ofW4a8SmallTiles}) split a grid
 * that fills less than about 1.3 waves instead: there each block's serial reduction over K is the
 * wall time, and llama.cpp spreads the same grids over the device with its stream-k kernel — at
 * one wave the ceil wave count double-books the tail SMs, and the slices remove that quantization
 * (kernel medians on the RTX 5080 Laptop, Q4_K 4096x2560 at 9..15 tokens: the 64-tile twin over 60
 * compute units runs 86-93us whole and 59-60us as 8 slices, while the 152-tile 9728x2560 twin runs
 * 101-110us whole and 119-125us sliced, so the split stops helping past about 1.3 waves). A device
 * that does not report its compute units ({@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS})
 * is never split, as llama.cpp does with a {@code shader_core_count} of 0. Partial launch geometry:
 * {@code global = {ceil(M / TM) * WG * splits, ceil(N / tileTokens)}}, {@code local = {WG, 1}}, with
 * the family's tile: the MMA/CM tiles, or the staged MMQ lane's {@link MmqConfig} (TM, TN and WG);
 * reduce geometry: one work-group of {@code local} work-items per four output elements.</p>
 */
public final class SplitKGemmPlan {

    /** Smallest K that splits (llama.cpp ggml-vulkan: k >= 2048). */
    public static final int MIN_SPLIT_K = 2048;
    /** Largest slice count (llama.cpp caps the split at 8x). */
    public static final int MAX_SPLITS = 8;
    /** Numerator of the wave-fill fraction under which a small-token-tile launch splits. */
    static final int SMALL_WAVE_NUM = 13;
    /** Denominator of the wave-fill fraction under which a small-token-tile launch splits. */
    static final int SMALL_WAVE_DEN = 10;
    /** Blocks per compute unit the small-token-tile split aims for (one occupancy-interleaved wave). */
    static final int SMALL_TILE_BLOCKS_PER_UNIT = 8;
    /** Output elements per reduce work-item. */
    public static final int REDUCE_ELEMENTS_PER_ITEM = 4;

    /**
     * The fast GEMM lane a split-K plan runs: the kernel family, its weight-byte accounting and its
     * fixed stage depth. The MMQ family's depth and token tile follow the device's {@link MmqConfig},
     * so its plans are built from an {@link MmqGemmPlan}.
     */
    public enum Family {
        /** The tensor-core MMA GEMM over a dense 16-bit weight (F16 or BF16), 16-deep stages. */
        MMA(GemmKernels.MATMUL_MMA_KSPLIT_SUFFIX),
        /** The dequantizing CM GEMM over a quantized weight, 16-deep stages. */
        CM(GemmKernels.MATMUL_CM_KSPLIT_SUFFIX),
        /** The int8 MMQ GEMM over staged activations, stages of the device's {@link MmqConfig}. */
        MMQ(GemmKernels.MATMUL_MMQ_KSPLIT_SUFFIX),
        /** The E4M3 GEMM over a 4-bit weight, stages of the device's {@link MmqConfig}. */
        W4A8(GemmKernels.MATMUL_W4A8_KSPLIT_SUFFIX);

        final String suffix;

        Family(String suffix) {
            this.suffix = suffix;
        }

        /** Kernel name suffix of the family's split-K twins. */
        String suffix() {
            return suffix;
        }

        /** Whether the family reads quantized weights (block-size slice and weight-byte accounting). */
        boolean quantized() {
            return this != MMA;
        }

        /** Whether the family reads the staged activation records of the device's {@link MmqConfig}. */
        boolean staged() {
            return this == MMQ || this == W4A8;
        }

        /** The family's fixed slice depth, or 0 when it follows the device's {@link MmqConfig} (MMQ, W4A8). */
        int fixedSliceDepth() {
            return this == MMA || this == CM ? MmaGemmPlan.TILE_DEPTH : 0;
        }

        /** Bytes of one weight row of {@code k} elements over {@code type}. */
        long weightRowBytes(GGMLType type, int k) {
            if (!quantized()) {
                return 2L * k;
            }
            return (long) (k / type.getBlockSize()) * type.getTypeSize();
        }

        /** The family's split-K kernel name for {@code type}. */
        String kernelName(GGMLType type) {
            return type.name() + suffix;
        }
    }

    private final Family family;
    private final GGMLType type;
    private final String kernelName;
    private final int m;
    private final int k;
    private final int n;
    private final int xStride;
    private final int xOff;
    private final int splits;
    private final int kSlice;
    private final int tileRows;
    private final int tileTokens;
    private final int workGroupSize;
    private final int stageDepth;
    private final int recordUnits;
    private final int reduceLocal;
    private final MmqConfig stagedConfig;

    private SplitKGemmPlan(Family family, GGMLType type, int m, int k, int n, int xStride, int xOff, int splits,
                           int kSlice, int tileRows, int tileTokens, int workGroupSize, int stageDepth,
                           MmqConfig stagedConfig, int reduceLocal) {
        this.family = family;
        this.type = type;
        this.kernelName = family.kernelName(type);
        this.m = m;
        this.k = k;
        this.n = n;
        this.xStride = xStride;
        this.xOff = xOff;
        this.splits = splits;
        this.kSlice = kSlice;
        this.tileRows = tileRows;
        this.tileTokens = tileTokens;
        this.workGroupSize = workGroupSize;
        this.stageDepth = stageDepth;
        this.recordUnits = family.staged() ? stageDepth / 4 + 2 + stageDepth / 16 : 1;
        this.stagedConfig = stagedConfig;
        this.reduceLocal = reduceLocal;
    }

    /**
     * Plans one MMA split-K GEMM of {@code splits} slices (the raw form of
     * {@link #of(MmaGemmPlan, int, int)} without the heuristic; the tile is the MMA family's).
     *
     * @throws IllegalArgumentException as {@link #of}
     */
    public static SplitKGemmPlan of(GGMLType type, int m, int k, int n, int xStride, int xOff,
                                    int splits, int kSlice, int reduceLocal) {
        return of(Family.MMA, type, m, k, n, xStride, xOff, splits, kSlice, MmaGemmPlan.TILE_TOKENS, reduceLocal);
    }

    /**
     * Plans one split-K GEMM of {@code splits} slices over the MMA or CM family (the MMQ family is
     * planned from its {@link MmqGemmPlan}, whose device config carries the stage depth and token
     * tile).
     *
     * @param family     {@link Family#MMA} or {@link Family#CM}; the type must have that family's kernel
     * @param type       weight type
     * @param m          weight rows (output features), at least 1
     * @param k          reduction length (input features), at least {@code splits}
     * @param n          tokens, at least 1
     * @param xStride    elements between activation rows, at least {@code k}
     * @param xOff       element offset of activation row 0, at least 0
     * @param splits     slice planes, in [1, {@link #MAX_SPLITS}], with every slice whole stages
     * @param kSlice     k length of every slice: a multiple of the family's slice depth in
     *                   [depth, {@code Integer.MAX_VALUE - depth}], with {@code kSlice * (splits - 1) < k}
     * @param tileTokens tokens per workgroup tile of the plain kernel ({@code NF_MMA_TN})
     * @param reduceLocal work-group size of the reduce launch: positive
     * @throws IllegalArgumentException naming the violated condition
     */
    public static SplitKGemmPlan of(Family family, GGMLType type, int m, int k, int n, int xStride, int xOff,
                                    int splits, int kSlice, int tileTokens, int reduceLocal) {
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(type, "type");
        if (family == Family.MMQ || family == Family.W4A8) {
            throw new IllegalArgumentException(family + " split-K plans come from their staged factory"
                    + " (the staged records follow the device's MmqConfig)");
        }
        if (family == Family.MMA ? !GemmKernels.hasMma(type) : !GemmKernels.hasCm(type)) {
            throw new IllegalArgumentException((family == Family.MMA ? "no split-K MMA GEMM kernel"
                    : "no split-K CM GEMM kernel") + " for weight type " + type);
        }
        return of(family, type, m, k, n, xStride, xOff, splits, kSlice, tileTokens, null, reduceLocal);
    }

    /**
     * Plans one raw split-K GEMM over the staged MMQ family (for tests and callers that plan without
     * an {@link MmqGemmPlan}): the slice depth is the type's block size, {@code qStride} is in staged
     * record units and {@code stageDepth} is the device's {@link MmqConfig#stageDepth()}.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static SplitKGemmPlan ofMmq(GGMLType type, int m, int k, int n, int qStride, int qOff,
                                       int splits, int kSlice, int tileTokens, int stageDepth, int reduceLocal) {
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.hasMmq(type)) {
            throw new IllegalArgumentException("no split-K MMQ GEMM kernel for weight type " + type);
        }
        return of(Family.MMQ, type, m, k, n, qStride, qOff, splits, kSlice, tileTokens,
                MmqConfig.of(stageDepth, MmqGemmPlan.TILE_ROWS, tileTokens), reduceLocal);
    }

    /**
     * Plans one raw split-K GEMM over the staged MMQ family with the staged tile {@code config}.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static SplitKGemmPlan ofMmq(GGMLType type, int m, int k, int n, int qStride, int qOff,
                                       int splits, int kSlice, MmqConfig config, int reduceLocal) {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(config, "config");
        if (!GemmKernels.hasMmq(type)) {
            throw new IllegalArgumentException("no split-K MMQ GEMM kernel for weight type " + type);
        }
        return of(Family.MMQ, type, m, k, n, qStride, qOff, splits, kSlice, config.tileTokens(), config,
                reduceLocal);
    }

    /**
     * Plans one raw split-K GEMM over the W4A8 family (the E4M3 GEMM of a 4-bit weight): {@code qStride}
     * is in staged record units and {@code stageDepth} is the device's {@link MmqConfig#stageDepth()}.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static SplitKGemmPlan ofW4a8(GGMLType type, int m, int k, int n, int qStride, int qOff,
                                        int splits, int kSlice, int tileTokens, int stageDepth, int reduceLocal) {
        requireW4a8(type);
        return of(Family.W4A8, type, m, k, n, qStride, qOff, splits, kSlice, tileTokens,
                MmqConfig.of(stageDepth, MmqGemmPlan.TILE_ROWS, tileTokens), reduceLocal);
    }

    /**
     * The split-K heuristic ({@link #splitsFor}) for a W4A8 launch of {@code m} rows over {@code n} tokens
     * with the tile {@code tileTokens} x {@link MmqGemmPlan#TILE_ROWS} at {@code stageDepth}.
     *
     * @param computeUnits compute units of the device, or {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}
     * @return the plan, or null when the heuristic does not split
     * @throws IllegalArgumentException naming a violated condition
     */
    public static SplitKGemmPlan ofW4a8(GGMLType type, int m, int k, int n, int qStride, int qOff,
                                        int tileTokens, int stageDepth, int computeUnits, int reduceLocal) {
        return ofW4a8(type, m, k, n, qStride, qOff,
                MmqConfig.of(stageDepth, MmqGemmPlan.TILE_ROWS, tileTokens), computeUnits, reduceLocal);
    }

    /**
     * The split-K heuristic ({@link #splitsFor}) for a W4A8 launch of {@code m} rows over {@code n} tokens
     * with the staged tile {@code config}: the plain kernel's tile grid, or null when the heuristic does
     * not split.
     *
     * @return the plan, or null when the heuristic does not split
     * @throws IllegalArgumentException naming a violated condition
     */
    public static SplitKGemmPlan ofW4a8(GGMLType type, int m, int k, int n, int qStride, int qOff,
                                        MmqConfig config, int computeUnits, int reduceLocal) {
        requireW4a8(type);
        Objects.requireNonNull(config, "config");
        int rowTiles = (m + config.tileRows() - 1) / config.tileRows();
        int tokenTiles = (n + config.tileTokens() - 1) / config.tileTokens();
        int splits = splitsFor(computeUnits, rowTiles, tokenTiles, k);
        return finished(Family.W4A8, type, m, k, n, qStride, qOff, splits, config.tileTokens(), config, reduceLocal);
    }

    /** {@link #ofSmallTiles} for a staged W4A8 launch of the staged tile {@code config}. */
    public static SplitKGemmPlan ofW4a8SmallTiles(GGMLType type, int m, int k, int n, int qStride, int qOff,
                                                 MmqConfig config, int computeUnits, int reduceLocal) {
        requireW4a8(type);
        Objects.requireNonNull(config, "config");
        int rowTiles = (m + config.tileRows() - 1) / config.tileRows();
        int tokenTiles = (n + config.tileTokens() - 1) / config.tileTokens();
        int splits = smallTileSplitsFor(computeUnits, rowTiles, tokenTiles, k);
        return finished(Family.W4A8, type, m, k, n, qStride, qOff, splits, config.tileTokens(), config, reduceLocal);
    }

    /** Checks that {@code type} has the W4A8 family's split-K twin. */
    private static void requireW4a8(GGMLType type) {
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.isW4a8(type)) {
            throw new IllegalArgumentException("no split-K W4A8 GEMM kernel for weight type " + type);
        }
    }

    private static SplitKGemmPlan of(Family family, GGMLType type, int m, int k, int n, int xStride, int xOff,
                                     int splits, int kSlice, int tileTokens, MmqConfig staged, int reduceLocal) {
        if (family.staged() != (staged != null)) {
            throw new IllegalArgumentException((family.staged() ? "the staged " + family + " family"
                    : "the " + family + " family") + (staged != null ? " does not" : " must")
                    + " follow a device MmqConfig");
        }
        int stageDepth = staged != null ? staged.stageDepth() : MmaGemmPlan.TILE_DEPTH;
        int tileRows = staged != null ? staged.tileRows() : MmaGemmPlan.TILE_ROWS;
        int workGroupSize = staged != null ? staged.workGroupSize() : MmaGemmPlan.WORKGROUP_SIZE;
        int depth = family.fixedSliceDepth() > 0 ? family.fixedSliceDepth() : stageDepth;
        if (m < 1 || k < 1 || n < 1) {
            throw new IllegalArgumentException("split-K GEMM needs M, K, N >= 1, got M=" + m + ", K=" + k
                    + ", N=" + n);
        }
        long rowRecordsNeeded = family.staged() ? (k + stageDepth - 1) / stageDepth : k;
        if (xStride < rowRecordsNeeded) {
            throw new IllegalArgumentException("activation row stride " + xStride + " is shorter than the row length"
                    + " of K=" + k);
        }
        if (xOff < 0) {
            throw new IllegalArgumentException("split-K GEMM view offset must be >= 0, got xOff=" + xOff);
        }
        if (splits < 1 || splits > MAX_SPLITS) {
            throw new IllegalArgumentException("split-K GEMM supports 1 <= splits <= " + MAX_SPLITS
                    + ", got " + splits);
        }
        if (kSlice < depth || kSlice > Integer.MAX_VALUE - depth || kSlice % depth != 0) {
            throw new IllegalArgumentException("split-K slice length must be a positive multiple of " + depth
                    + " below 2^31 - " + depth + ", got " + kSlice);
        }
        if ((long) kSlice * (splits - 1) >= k) {
            throw new IllegalArgumentException("split-K slice length " + kSlice + " leaves slice " + (splits - 1)
                    + " of " + splits + " empty for K=" + k);
        }
        if (reduceLocal < 1) {
            throw new IllegalArgumentException("split-K reduce work-group size must be positive, got " + reduceLocal);
        }
        long partial = (long) splits * m * n;
        if (partial > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("split-K planes " + splits + " * " + m * n
                    + " exceed the int element offsets of the partial kernel");
        }
        return new SplitKGemmPlan(family, type, m, k, n, xStride, xOff, splits, kSlice, tileRows, tileTokens,
                workGroupSize, stageDepth, staged, reduceLocal);
    }

    /**
     * Plans the split of a plain {@link MmaGemmPlan} launch with llama.cpp's heuristic: only K of
     * {@link #MIN_SPLIT_K} and more, only while the tile grid fills less than two thirds of the
     * compute units, capped at {@link #MAX_SPLITS} slices. A grid that fills the device, a K below
     * the threshold, or a device that does not report its compute units
     * ({@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}) stays unsplit (null, the plain kernel's
     * shape).
     *
     * @param computeUnits compute units of the device, or {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}
     * @return the plan, or null when the heuristic does not split
     */
    public static SplitKGemmPlan of(MmaGemmPlan base, int computeUnits, int reduceLocal) {
        int splits = splitsFor(computeUnits, base.rowTiles(), base.tokenTiles(), base.k());
        return finished(Family.MMA, base.type(), base.m(), base.k(), base.n(), base.xStride(), base.xOff(), splits,
                base.tileTokens(), null, reduceLocal);
    }

    /** {@link #of(MmaGemmPlan, int, int)} for a quantized-weight CM launch. */
    public static SplitKGemmPlan of(CmGemmPlan base, int computeUnits, int reduceLocal) {
        int splits = splitsFor(computeUnits, base.rowTiles(), base.tokenTiles(), base.k());
        return finished(Family.CM, base.type(), base.m(), base.k(), base.n(), base.xStride(), base.xOff(), splits,
                base.tileTokens(), null, reduceLocal);
    }

    /** {@link #of(MmaGemmPlan, int, int)} for a staged MMQ launch (tile and stage depth of the base's config). */
    public static SplitKGemmPlan of(MmqGemmPlan base, int computeUnits, int reduceLocal) {
        int splits = splitsFor(computeUnits, base.rowTiles(), base.tokenTiles(), base.k());
        return finished(Family.MMQ, base.type(), base.m(), base.k(), base.n(), base.qStride(), base.qOff(), splits,
                base.config().tileTokens(), base.config(), reduceLocal);
    }

    /**
     * Plans the split of a staged MMQ launch of fewer tokens than the device tile (the
     * {@link MmqGemmPlan#smallOf} regime): there the tile grid covers one wave of the device at most,
     * and each block's serial reduction over K is the wall time, so the launch splits K up to
     * {@link #MAX_SPLITS} to give every compute unit several blocks (llama.cpp runs the same grids
     * through its stream-k kernel, which spreads the reduction over the device the same way). A grid
     * that fills its wave, a K below {@link #MIN_SPLIT_K} or a device without a compute-unit count
     * keeps the plain kernel's shape (null, as {@link #of} returns).
     *
     * @param computeUnits compute units of the device, or {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}
     * @return the plan, or null when the heuristic does not split
     */
    public static SplitKGemmPlan ofSmallTiles(MmqGemmPlan base, int computeUnits, int reduceLocal) {
        int splits = smallTileSplitsFor(computeUnits, base.rowTiles(), base.tokenTiles(), base.k());
        return finished(Family.MMQ, base.type(), base.m(), base.k(), base.n(), base.qStride(), base.qOff(), splits,
                base.config().tileTokens(), base.config(), reduceLocal);
    }

    /** Steps the count down while its round-up would leave the last slice empty, as llama.cpp does. */
    private static SplitKGemmPlan finished(Family family, GGMLType type, int m, int k, int n, int xStride, int xOff,
                                           int splits, int tileTokens, MmqConfig staged, int reduceLocal) {
        int depth = family.fixedSliceDepth() > 0 ? family.fixedSliceDepth()
                : staged != null ? staged.stageDepth() : MmaGemmPlan.TILE_DEPTH;
        while (splits > 1 && (long) sliceLength(k, splits, depth) * (splits - 1) >= k) {
            splits--;
        }
        if (splits <= 1) {
            return null;
        }
        return of(family, type, m, k, n, xStride, xOff, splits, sliceLength(k, splits, depth), tileTokens, staged,
                reduceLocal);
    }

    /**
     * The heuristic's slice count for a tile grid of {@code rowTiles * tokenTiles} workgroups over a
     * reduction of {@code k}: 1 below the K threshold and for a device whose compute-unit count is
     * unknown ({@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}); otherwise the compute-unit fill
     * ratio, capped at {@link #MAX_SPLITS}.
     *
     * @param computeUnits compute units of the device, or {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}
     * @throws IllegalArgumentException when {@code computeUnits} is negative
     */
    public static int splitsFor(int computeUnits, int rowTiles, int tokenTiles, int k) {
        if (computeUnits < DeviceCapabilities.UNKNOWN_COMPUTE_UNITS) {
            throw new IllegalArgumentException("split-K heuristic needs a non-negative compute-unit count"
                    + " (" + DeviceCapabilities.UNKNOWN_COMPUTE_UNITS + " when the device does not report one), got "
                    + computeUnits);
        }
        if (computeUnits == DeviceCapabilities.UNKNOWN_COMPUTE_UNITS || k < MIN_SPLIT_K) {
            return 1;
        }
        long tiles = (long) rowTiles * tokenTiles;
        int splits;
        if (tiles <= computeUnits / 2) {
            splits = (int) Math.min(MAX_SPLITS, computeUnits / Math.max(1L, tiles));
        } else if (tiles <= computeUnits * 2L / 3) {
            splits = 3;
        } else {
            return 1;
        }
        return Math.max(1, Math.min(splits, MAX_SPLITS));
    }

    /**
     * The small-token-tile slice count ({@link #ofSmallTiles}) for a tile grid of
     * {@code rowTiles * tokenTiles} workgroups over a reduction of {@code k}: 1 for a K below the
     * threshold, an unknown compute-unit count, or a grid past about 1.3 waves of the device
     * ({@code SMALL_WAVE_NUM / SMALL_WAVE_DEN} of it and up, where the measured staged twins stop
     * gaining from the slices); otherwise the count that gives every compute unit
     * {@code SMALL_TILE_BLOCKS_PER_UNIT} blocks, capped at {@link #MAX_SPLITS}.
     *
     * @param computeUnits compute units of the device, or {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}
     * @throws IllegalArgumentException when {@code computeUnits} is negative
     */
    static int smallTileSplitsFor(int computeUnits, int rowTiles, int tokenTiles, int k) {
        if (computeUnits < DeviceCapabilities.UNKNOWN_COMPUTE_UNITS) {
            throw new IllegalArgumentException("split-K heuristic needs a non-negative compute-unit count"
                    + " (" + DeviceCapabilities.UNKNOWN_COMPUTE_UNITS + " when the device does not report one), got "
                    + computeUnits);
        }
        if (computeUnits == DeviceCapabilities.UNKNOWN_COMPUTE_UNITS || k < MIN_SPLIT_K) {
            return 1;
        }
        long tiles = (long) rowTiles * tokenTiles;
        if (tiles * SMALL_WAVE_DEN > (long) computeUnits * SMALL_WAVE_NUM) {
            return 1;
        }
        long blocks = (long) computeUnits * SMALL_TILE_BLOCKS_PER_UNIT;
        int splits = (int) Math.min(MAX_SPLITS, Math.round((double) blocks / Math.max(1L, tiles)));
        return Math.max(1, splits);
    }

    /**
     * The slice length for {@code splits} slices over {@code k} of 16-deep stages: {@code ceil(k /
     * splits)} rounded up to whole stages. The caller keeps the last slice non-empty ({@code of}
     * steps the split count down, as llama.cpp does for its round-up).
     *
     * @throws IllegalArgumentException when {@code k} is below the stage depth
     */
    public static int sliceLength(int k, int splits) {
        return sliceLength(k, splits, MmaGemmPlan.TILE_DEPTH);
    }

    /** {@link #sliceLength(int, int)} for a stage depth of {@code depth} (see {@link Family}). */
    public static int sliceLength(int k, int splits, int depth) {
        if (depth < 1) {
            throw new IllegalArgumentException("split-K stage depth must be positive, got " + depth);
        }
        if (k < depth) {
            throw new IllegalArgumentException("split-K needs K >= " + depth + ", got " + k);
        }
        if (splits < 1 || splits > MAX_SPLITS) {
            throw new IllegalArgumentException("split-K supports 1 <= splits <= " + MAX_SPLITS + ", got " + splits);
        }
        return ((k + splits - 1) / splits + depth - 1) / depth * depth;
    }

    /** The fast GEMM lane of the planned launch. */
    public Family family() {
        return family;
    }

    public GGMLType type() {
        return type;
    }

    public String kernelName() {
        return kernelName;
    }

    public int m() {
        return m;
    }

    public int k() {
        return k;
    }

    public int n() {
        return n;
    }

    /** Elements (floats, or staged record units for the staged families) between activation rows. */
    public int xStride() {
        return xStride;
    }

    /** Element offset of activation row 0. */
    public int xOff() {
        return xOff;
    }

    /** Slice planes. */
    public int splits() {
        return splits;
    }

    /** k length of every slice (a multiple of the family's slice depth). */
    public int kSlice() {
        return kSlice;
    }

    /** Work-groups along the weight rows: {@code ceil(M / tileRows())}. */
    public int rowTiles() {
        return (m + tileRows - 1) / tileRows;
    }

    /** Weight rows per workgroup tile of the family's plain kernel. */
    public int tileRows() {
        return tileRows;
    }

    /** Work-group size of the family's plain kernel. */
    public int workGroupSize() {
        return workGroupSize;
    }

    /** The staged tile of the {@link Family#MMQ} and {@link Family#W4A8} families, or null for MMA and CM. */
    public MmqConfig stagedConfig() {
        return stagedConfig;
    }

    /** Work-groups along the tokens: {@code ceil(N / tileTokens)} of the family's plain kernel. */
    public int tokenTiles() {
        return (n + tileTokens - 1) / tileTokens;
    }

    /** Floats of one slice plane: {@code M * N}. */
    public long planeElements() {
        return (long) m * n;
    }

    /** Floats of the partial buffer: {@code splits * M * N}. */
    public long partialElements() {
        return (long) splits * m * n;
    }

    /** Bytes of one weight row of the family. */
    public long weightRowBytes() {
        return family.weightRowBytes(type, k);
    }

    /** Bytes of the weight the kernel reads: {@code M} rows. */
    public long weightBytes() {
        return weightRowBytes() * m;
    }

    /**
     * Smallest activation buffer length in the family's units (floats, or 32-bit staged record units
     * for {@link Family#MMQ} and {@link Family#W4A8}): {@code xOff + the rows before the last, plus one row}.
     */
    public long xElementsRequired() {
        long row = family.staged() ? (long) (k / stageDepth) * recordUnits : k;
        long rowsBefore = family.staged() ? (long) (n - 1) * xStride * recordUnits : (long) (n - 1) * xStride;
        return (long) xOff + rowsBefore + row;
    }

    /** Smallest output buffer length in floats: {@code (N-1)*yStride + M} beyond {@code yOff}. */
    public long yElementsRequired(int yStride, int yOff) {
        if (yStride < m || yOff < 0) {
            throw new IllegalArgumentException("split-K output needs yStride >= M=" + m + " and yOff >= 0, got "
                    + yStride + "/" + yOff);
        }
        return (long) yOff + (long) (n - 1) * yStride + m;
    }

    /** {@code {rowTiles() * workGroupSize() * splits, tokenTiles()}}: the partial kernel's global work size. */
    public long[] partialGlobalWorkSize() {
        return new long[]{(long) rowTiles() * workGroupSize * splits, tokenTiles()};
    }

    /** {@code {workGroupSize(), 1}}: the partial kernel's local work size. */
    public long[] partialLocalWorkSize() {
        return new long[]{workGroupSize, 1};
    }

    /**
     * Whether the device can run the partial launches: the family plain kernel's device support plus
     * the group counts of both dispatch dimensions (dimension 0 carries the split).
     *
     * @return null when it can, otherwise the reason
     */
    public String checkDevice(DeviceCapabilities caps) {
        Objects.requireNonNull(caps, "caps");
        String reason = family == Family.MMA ? MmaGemmPlan.deviceSupport(caps)
                : family == Family.CM ? CmGemmPlan.deviceSupport(type, caps)
                : MmqGemmPlan.stagedTileSupport(stagedConfig, type, caps);
        if (reason != null) {
            return reason;
        }
        long[] maxGroups = caps.getMaxWorkGroupCount();
        long groups0 = (long) rowTiles() * splits;
        if (groups0 > maxGroups[0]) {
            return kernelName + " needs " + groups0 + " work-groups in dimension 0 (M=" + m + ", splits="
                    + splits + "); device " + caps.getDeviceName() + " allows " + maxGroups[0];
        }
        if (tokenTiles() > maxGroups[1]) {
            return kernelName + " needs " + tokenTiles() + " work-groups in dimension 1 (N=" + n + "); device "
                    + caps.getDeviceName() + " allows " + maxGroups[1];
        }
        return null;
    }

    /**
     * Partial kernel arguments in declared order ({@link GemmKernels#KSPLIT_ARG_M} ..
     * {@link GemmKernels#KSPLIT_ARG_PARTIAL}), after checking that the weight and activation buffers
     * cover the launch and the partial buffer holds every plane. The activation buffer is the
     * projection input (MMA and CM), or the staged records of the activation staging (MMQ and W4A8).
     *
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when a buffer is smaller than the launch needs
     */
    public Object[] partialArgs(GpuBuffer weight, GpuBuffer x, GpuBuffer partial) {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(partial, "partial");
        requireBytes("weight", weight, weightBytes());
        requireBytes("activation", x, xElementsRequired() * (family.staged() ? Integer.BYTES : Float.BYTES));
        requireBytes("partial", partial, partialElements() * Float.BYTES);
        Object[] args = new Object[GemmKernels.KSPLIT_ARG_COUNT];
        args[GemmKernels.KSPLIT_ARG_M] = m;
        args[GemmKernels.KSPLIT_ARG_K] = k;
        args[GemmKernels.KSPLIT_ARG_N] = n;
        args[GemmKernels.KSPLIT_ARG_X_STRIDE] = xStride;
        args[GemmKernels.KSPLIT_ARG_X_OFF] = xOff;
        args[GemmKernels.KSPLIT_ARG_K_SLICE] = kSlice;
        args[GemmKernels.KSPLIT_ARG_WEIGHT] = weight;
        args[GemmKernels.KSPLIT_ARG_X] = x;
        args[GemmKernels.KSPLIT_ARG_PARTIAL] = partial;
        return args;
    }

    /**
     * Reduce kernel arguments in declared order ({@link GemmKernels#REDUCE_ARG_NE} ..
     * {@link GemmKernels#REDUCE_ARG_Y}), after checking that the output covers the strided result.
     *
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when a buffer is smaller than the launch needs
     */
    public Object[] reduceArgs(GpuBuffer partial, GpuBuffer y, int yStride, int yOff) {
        Objects.requireNonNull(partial, "partial");
        Objects.requireNonNull(y, "y");
        if (partial.sizeInBytes() < partialElements() * Float.BYTES) {
            throw new IllegalArgumentException(kernelName + " reduce: partial buffer holds "
                    + partial.sizeInBytes() + " bytes but the launch needs "
                    + partialElements() * Float.BYTES);
        }
        if (y.sizeInBytes() < yElementsRequired(yStride, yOff) * Float.BYTES) {
            throw new IllegalArgumentException(kernelName + " reduce: output buffer holds " + y.sizeInBytes()
                    + " bytes but the launch needs " + yElementsRequired(yStride, yOff) * Float.BYTES);
        }
        Object[] args = new Object[GemmKernels.SPLIT_K_REDUCE_ARG_COUNT];
        args[GemmKernels.REDUCE_ARG_NE] = (int) planeElements();
        args[GemmKernels.REDUCE_ARG_SPLITS] = splits;
        args[GemmKernels.REDUCE_ARG_M] = m;
        args[GemmKernels.REDUCE_ARG_Y_STRIDE] = yStride;
        args[GemmKernels.REDUCE_ARG_Y_OFF] = yOff;
        args[GemmKernels.REDUCE_ARG_PARTIAL] = partial;
        args[GemmKernels.REDUCE_ARG_Y] = y;
        return args;
    }

    /** The reduce launch's global work size for the plan's reduce work-group size. */
    public long[] reduceGlobalWorkSize() {
        long items = (planeElements() + REDUCE_ELEMENTS_PER_ITEM - 1) / REDUCE_ELEMENTS_PER_ITEM;
        long groups = (items + reduceLocal - 1) / reduceLocal;
        return new long[]{groups * reduceLocal};
    }

    /** The reduce launch's local work size ({@code {reduceLocal}}). */
    public long[] reduceLocalWorkSize() {
        return new long[]{reduceLocal};
    }

    private void requireBytes(String role, GpuBuffer buffer, long needed) {
        long size = buffer.sizeInBytes();
        if (size < needed) {
            throw new IllegalArgumentException(kernelName + ": " + role + " buffer holds " + size
                    + " bytes but the launch (M=" + m + ", K=" + k + ", N=" + n + ", splits=" + splits
                    + ") needs " + needed);
        }
    }

    @Override
    public String toString() {
        return "SplitKGemmPlan{" + kernelName + ", M=" + m + ", K=" + k + ", N=" + n + ", splits=" + splits
                + ", kSlice=" + kSlice + ", xStride=" + xStride + ", xOff=" + xOff + '}';
    }
}
