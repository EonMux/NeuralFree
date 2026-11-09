package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Objects;

/**
 * Immutable dispatch plan of one {@code <T>_MatMul_MMQ} launch: the kernel name, the validated shape
 * and view arguments, the 2-D work sizes and the argument array.
 *
 * <p>The kernel computes {@code y[yOff + t*yStride + r] = sum_{k < K} W[r][k] * x[t][k]}
 * for {@code r < M}, {@code t < N}: ggml_mul_mat of an {@code [M x K]} quantized weight with {@code N}
 * activation rows, with the activations quantized once per projection by {@link GemmKernels#QUANT_MMQ_Q8_1}
 * into the staged records the kernel reads ({@code kernels/llm/common/matmul_mmq_family.h}), so all
 * row tiles share one staging pass (llama.cpp quantize_mmq_q8_1); the results match the dequantizing
 * tiled GEMM up to the 8-bit activation quantization error. The staged activation arguments
 * {@code qStride} / {@code qOff} are in 32-bit record units ({@code config.recordUnits()} per chunk,
 * {@code config.rowRecords(k)} per row). The stage depth, the row tile and the token tile come from
 * the device's {@link MmqConfig}.</p>
 *
 * <p>Geometry: {@code global = {ceil(M / TM) * WG, ceil(N / TN)}}, {@code local = {WG, 1}}, with
 * {@code TM = config.tileRows()}, {@code TN = config.tileTokens()} and {@code WG =
 * config.workGroupSize()}. The dispatch is 2-D and therefore never folded on Vulkan, so
 * {@link #checkDevice} verifies the group counts against {@code maxWorkGroupCount}. {@link #smallOf}
 * plans the same launch over {@link MmqConfig#smallTile}'s twin tiles, the {@code <T>_MatMul_MMQ_S}
 * 16-token twin and the S64 64-token twin the host serves staged launches of fewer tokens than the
 * device tile with (llama.cpp picks its J tile per launch, ggml_cuda_mmq_get_J_best).</p>
 */
public final class MmqGemmPlan {

    /** Weight rows per workgroup tile of the default configs ({@code NF_MMQ_TM}); plans follow {@link MmqConfig#tileRows()}. */
    public static final int TILE_ROWS = 64;
    /** Work-items per workgroup of the default configs ({@code NF_MMQ_WG}); plans follow {@link MmqConfig#workGroupSize()}. */
    public static final int WORKGROUP_SIZE = 256;

    private final MmqConfig config;
    private final GGMLType type;
    private final String kernelName;
    private final int m;
    private final int k;
    private final int n;
    private final int qStride;
    private final int qOff;
    private final int yStride;
    private final int yOff;

    private MmqGemmPlan(MmqConfig config, String kernelName, GGMLType type, int m, int k, int n, int qStride,
                        int qOff, int yStride, int yOff) {
        this.config = config;
        this.type = type;
        this.kernelName = kernelName;
        this.m = m;
        this.k = k;
        this.n = n;
        this.qStride = qStride;
        this.qOff = qOff;
        this.yStride = yStride;
        this.yOff = yOff;
    }

    /**
     * Plans {@code N} tokens through an {@code [M x K]} weight with contiguous staged rows
     * ({@code qStride = K / stageDepth} records, offset 0) and output rows ({@code yStride = M}, no
     * offsets).
     *
     * @throws IllegalArgumentException as {@link #of}
     */
    public static MmqGemmPlan contiguous(MmqConfig config, GGMLType type, int m, int k, int n) {
        return of(config, type, m, k, n, config.rowRecords(k), 0, m, 0);
    }

    /**
     * Plans one MMQ GEMM launch. The shape rules match {@link TiledGemmPlan#of}, with the tile index
     * bounds of the largest config ({@link MmqConfig#MAX_STAGE_DEPTH}, {@link MmqConfig#MAX_TILE_ROWS},
     * {@link MmqConfig#MAX_TILE_TOKENS}).
     *
     * @param config  tile config of the device the kernel runs on ({@link MmqConfig#forDevice})
     * @param type    weight type; must have an MMQ kernel ({@link GemmKernels#hasMmq})
     * @param m       weight rows (output features), in [1, {@code Integer.MAX_VALUE - MAX_TILE_ROWS}]
     * @param k       reduction length (input features), in [1, {@code Integer.MAX_VALUE - MAX_STAGE_DEPTH}]
     *                and a multiple of the type's block size (32 or 256)
     * @param n       tokens, in [1, {@code Integer.MAX_VALUE - MAX_TILE_TOKENS}]
     * @param qStride staged records between activation rows, at least {@code k / config.stageDepth()}
     * @param qOff    staged record offset of activation row 0, at least 0
     * @param yStride elements between output rows, at least {@code m}
     * @param yOff    element offset of output row 0, at least 0
     * @throws IllegalArgumentException naming the violated condition
     */
    public static MmqGemmPlan of(MmqConfig config, GGMLType type, int m, int k, int n, int qStride, int qOff,
                                 int yStride, int yOff) {
        return plan(config, GemmKernels.matmulMmq(type), type, m, k, n, qStride, qOff, yStride, yOff);
    }

    /**
     * Plans the launch of the small-token-tile twin ({@code <T>_MatMul_MMQ_S}, the
     * {@link MmqConfig#smallTile} form of {@code config}): the same staged records and validation
     * as {@link #of}, launched as {@code ceil(N / 16)} token tiles. The caller selects it when the
     * device provides the twin and {@code n} is smaller than the device tile.
     *
     * @param config  tile config of the device the twin belongs to ({@link MmqConfig#forDevice})
     * @param type    weight type; must have an MMQ kernel ({@link GemmKernels#hasMmq})
     * @throws IllegalArgumentException as {@link #of}, or when the device tile is already 16 tokens
     */
    public static MmqGemmPlan smallOf(MmqConfig config, GGMLType type, int m, int k, int n, int qStride, int qOff,
                                      int yStride, int yOff) {
        return plan(MmqConfig.smallTile(config), GemmKernels.matmulMmqSmall(type), type, m, k, n, qStride, qOff,
                yStride, yOff);
    }

    /**
     * Plans the launch of the small-token-tile twin at the given twin width: as {@link #smallOf}
     * over {@link MmqConfig#smallTile(MmqConfig, int)}'s tile — the 16-token {@code <T>_MatMul_MMQ_S}
     * at 16 tokens, the 64-token S64 twin at {@link MmqConfig#TWIN_MAX_TOKENS}. The caller picks the
     * width with {@link MmqConfig#twinTokens(int, int)} and keeps the plain tile when the device tile is
     * not wider than it.
     *
     * @param tokens  twin tile width, 16 or {@link MmqConfig#TWIN_MAX_TOKENS}
     * @throws IllegalArgumentException as {@link #of}, when {@code tokens} is not a twin width or
     *                                  when the device tile is not wider than it
     */
    public static MmqGemmPlan smallOf(MmqConfig config, GGMLType type, int tokens, int m, int k, int n, int qStride,
                                      int qOff, int yStride, int yOff) {
        return plan(MmqConfig.smallTile(config, tokens), GemmKernels.matmulMmqSmall(type, tokens), type, m, k, n,
                qStride, qOff, yStride, yOff);
    }

    /** Shared validation and construction of the plain and small-tile forms. */
    private static MmqGemmPlan plan(MmqConfig config, String kernelName, GGMLType type, int m, int k, int n,
                                    int qStride, int qOff, int yStride, int yOff) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.hasMmq(type)) {
            throw new IllegalArgumentException("no MMQ GEMM kernel for weight type " + type);
        }
        if (m < 1 || k < 1 || n < 1) {
            throw new IllegalArgumentException("MMQ GEMM needs M, K, N >= 1, got M=" + m + ", K=" + k + ", N=" + n);
        }
        if (m > Integer.MAX_VALUE - MmqConfig.MAX_TILE_ROWS || n > Integer.MAX_VALUE - MmqConfig.MAX_TILE_TOKENS
                || k > Integer.MAX_VALUE - MmqConfig.MAX_STAGE_DEPTH) {
            throw new IllegalArgumentException("MMQ GEMM tile indices would overflow int for M=" + m + ", K=" + k
                    + ", N=" + n);
        }
        int blockSize = type.getBlockSize();
        if (k % blockSize != 0) {
            throw new IllegalArgumentException("MMQ GEMM over " + type + " needs K to be a multiple of the block size "
                    + blockSize + ", got K=" + k);
        }
        int rowRecords = config.rowRecords(k);
        if (qStride < rowRecords) {
            throw new IllegalArgumentException("staged activation row stride " + qStride + " is shorter than the "
                    + rowRecords + " records of K=" + k + " at stage depth " + config.stageDepth());
        }
        if (yStride < m) {
            throw new IllegalArgumentException("output row stride " + yStride + " is shorter than M=" + m);
        }
        if (qOff < 0 || yOff < 0) {
            throw new IllegalArgumentException("MMQ GEMM view offsets must be >= 0, got qOff=" + qOff
                    + ", yOff=" + yOff);
        }
        return new MmqGemmPlan(config, kernelName, type, m, k, n, qStride, qOff, yStride, yOff);
    }

    /** Local memory one workgroup of {@code config} uses; MMQ kernels stage no IQ grids. */
    public static long requiredSharedBytes(MmqConfig config, GGMLType type) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.hasMmq(type)) {
            throw new IllegalArgumentException("no MMQ GEMM kernel for weight type " + type);
        }
        return config.sharedBytes();
    }

    /**
     * Whether the device can run MMQ GEMM of {@code config} over {@code type} at all (work-group size
     * and local memory), independent of the shape.
     *
     * @return null when it can, otherwise the reason
     */
    public static String deviceSupport(MmqConfig config, GGMLType type, DeviceCapabilities caps) {
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.hasMmq(type)) {
            return "no MMQ GEMM kernel for weight type " + type;
        }
        return stagedTileSupport(config, type, caps);
    }

    /**
     * Whether the device can run the staged GEMM tile of {@code config} over {@code type}
     * ({@link #deviceSupport} without the kernel-name check): the MMQ GEMM and the E4M3 GEMM of a 4-bit W4A8
     * weight share the tile, so both check the same work-group size and local memory.
     *
     * @return null when it can, otherwise the reason
     */
    public static String stagedTileSupport(MmqConfig config, GGMLType type, DeviceCapabilities caps) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(caps, "caps");
        long maxWg = caps.getMaxWorkGroupSize();
        int workGroup = config.workGroupSize();
        if (maxWg > 0 && maxWg < workGroup) {
            return "MMQ GEMM needs work-groups of " + workGroup + " work-items; device "
                    + caps.getDeviceName() + " allows " + maxWg;
        }
        long shared = caps.getMaxSharedMemoryBytes();
        if (shared > 0 && shared < config.sharedBytes()) {
            return "MMQ GEMM over " + type + " needs " + config.sharedBytes() + " bytes of local memory; device "
                    + caps.getDeviceName() + " offers " + shared;
        }
        return null;
    }

    /**
     * Whether the device can run this launch: {@link #deviceSupport} plus the group counts of both
     * dispatch dimensions.
     *
     * @return null when it can, otherwise the reason
     */
    public String checkDevice(DeviceCapabilities caps) {
        String reason = deviceSupport(config, type, caps);
        if (reason != null) {
            return reason;
        }
        long[] maxGroups = caps.getMaxWorkGroupCount();
        if (rowTiles() > maxGroups[0]) {
            return kernelName + " needs " + rowTiles() + " work-groups in dimension 0 (M=" + m + "); device "
                    + caps.getDeviceName() + " allows " + maxGroups[0];
        }
        if (tokenTiles() > maxGroups[1]) {
            return kernelName + " needs " + tokenTiles() + " work-groups in dimension 1 (N=" + n + "); device "
                    + caps.getDeviceName() + " allows " + maxGroups[1];
        }
        return null;
    }

    /** The tile config the launch is planned for. */
    public MmqConfig config() {
        return config;
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

    /** Staged records between activation rows. */
    public int qStride() {
        return qStride;
    }

    /** Staged record offset of activation row 0. */
    public int qOff() {
        return qOff;
    }

    public int yStride() {
        return yStride;
    }

    public int yOff() {
        return yOff;
    }

    /** Workgroups along the weight rows: {@code ceil(M / TM)}. */
    public int rowTiles() {
        return (m + config.tileRows() - 1) / config.tileRows();
    }

    /** Workgroups along the tokens: {@code ceil(N / TN)}. */
    public int tokenTiles() {
        return (n + config.tileTokens() - 1) / config.tileTokens();
    }

    /** {@code {rowTiles() * WG, tokenTiles()}}. */
    public long[] globalWorkSize() {
        return new long[]{(long) rowTiles() * config.workGroupSize(), tokenTiles()};
    }

    /** {@code {WG, 1}}. */
    public long[] localWorkSize() {
        return new long[]{config.workGroupSize(), 1};
    }

    /** Bytes of one weight row: {@code K / blockSize * typeSize}. */
    public long weightRowBytes() {
        return (long) (k / type.getBlockSize()) * type.getTypeSize();
    }

    /** Bytes of the weight the kernel reads: {@code M} rows. */
    public long weightBytes() {
        return weightRowBytes() * m;
    }

    /**
     * Smallest staged activation buffer length in 32-bit units: {@code qOff + ((N-1)*qStride +
     * rowRecords(K)) * recordUnits()} — the record offset of token {@code N-1}'s first record plus its
     * row of records.
     */
    public long q8ElementsRequired() {
        return (long) qOff + ((long) (n - 1) * qStride + config.rowRecords(k)) * config.recordUnits();
    }

    /** Smallest output buffer length in floats: {@code yOff + (N-1)*yStride + M}. */
    public long yElementsRequired() {
        return (long) yOff + (long) (n - 1) * yStride + m;
    }

    /**
     * Kernel arguments in declared order ({@link GemmKernels#ARG_M} .. {@link GemmKernels#ARG_Y}),
     * after checking that each buffer covers the bytes this launch touches.
     *
     * @param weight the quantized weight
     * @param q8     the staged activation records ({@link GemmKernels#QUANT_MMQ_Q8_1}'s output)
     * @param y      the output
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when a buffer is smaller than the launch needs
     */
    public Object[] args(GpuBuffer weight, GpuBuffer q8, GpuBuffer y) {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(q8, "q8");
        Objects.requireNonNull(y, "y");
        if (q8 == y) {
            throw new IllegalArgumentException(kernelName + ": staged activations and output must be distinct buffers");
        }
        requireBytes("weight", weight, weightBytes());
        requireBytes("staged activations", q8, q8ElementsRequired() * Integer.BYTES);
        requireBytes("output", y, yElementsRequired() * Float.BYTES);
        Object[] args = new Object[GemmKernels.MATMUL_TILED_ARG_COUNT];
        args[GemmKernels.ARG_M] = m;
        args[GemmKernels.ARG_K] = k;
        args[GemmKernels.ARG_N] = n;
        args[GemmKernels.ARG_X_STRIDE] = qStride;
        args[GemmKernels.ARG_X_OFF] = qOff;
        args[GemmKernels.ARG_Y_STRIDE] = yStride;
        args[GemmKernels.ARG_Y_OFF] = yOff;
        args[GemmKernels.ARG_WEIGHT] = weight;
        args[GemmKernels.ARG_X] = q8;
        args[GemmKernels.ARG_Y] = y;
        return args;
    }

    private void requireBytes(String role, GpuBuffer buffer, long needed) {
        long size = buffer.sizeInBytes();
        if (size < needed) {
            throw new IllegalArgumentException(kernelName + ": " + role + " buffer holds " + size
                    + " bytes but the launch (M=" + m + ", K=" + k + ", N=" + n + ") needs " + needed);
        }
    }

    @Override
    public String toString() {
        return "MmqGemmPlan{" + kernelName + ", " + config + ", M=" + m + ", K=" + k + ", N=" + n
                + ", qStride=" + qStride + ", qOff=" + qOff + ", yStride=" + yStride + ", yOff=" + yOff + '}';
    }
}
