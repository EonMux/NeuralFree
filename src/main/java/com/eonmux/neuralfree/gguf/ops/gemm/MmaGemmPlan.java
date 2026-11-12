package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Objects;

/**
 * Immutable dispatch plan of one {@code <T>_MatMul_MMA} launch over a dense 16-bit weight (F16 or
 * BF16): the validated shape and view arguments, the 2-D work sizes and the argument array.
 *
 * <p>The kernel computes {@code y[yOff + t*yStride + r] = sum_{k < K} W[r][k] * x[xOff + t*xStride + k]}
 * for {@code r < M}, {@code t < N}: ggml_mul_mat with 16-bit products and an fp32 accumulator on CUDA
 * sm_80+ ({@code mma.m16n8k16} fragments loaded by {@code ldmatrix}; BF16 on the bfloat16
 * instruction), on Vulkan through 16x16x16 fp16 cooperative matrices (BF16 weights convert into the
 * fp16 tiles), and as an fp32 scalar twin elsewhere
 * ({@code kernels/llm/common/matmul_mma_family.h}).</p>
 *
 * <p>Geometry: {@code global = {ceil(M / 64) * 256, ceil(N / 32)}}, {@code local = {256, 1}}. The
 * dispatch is 2-D and therefore never folded on Vulkan, so {@link #checkDevice} verifies the group
 * counts against {@code maxWorkGroupCount}. The split-K twin of the same kernel family is planned by
 * {@link SplitKGemmPlan}.</p>
 *
 * <p>The large tile ({@code <T>_MatMul_MMA_L}, kernels/llm/matmul_mma_large.cl, CUDA sm_80+ only) takes the
 * same arguments over {@link #LARGE_TILE_ROWS} x {@link #LARGE_TILE_TOKENS} tiles, {@code global =
 * {ceil(M / 128) * 256, ceil(N / 64)}}: its weight and activation stages load as 16-byte rows into a two-stage
 * ring with one barrier per 32-deep stage, and the token tiles of a weight row tile run back to back, so a
 * prefill streams each weight row tile from DRAM once. It adds the same 16-deep chunk products in the same
 * order as the 64 x 32 tile, so the two agree bit for bit. The dispatch takes it for every launch of at least
 * {@link #LARGE_TILE_TOKENS} tokens that the split-K heuristic leaves unsplit ({@link SplitKGemmPlan#of}): it
 * measured 3.3-4.6x faster than the 64 x 32 tile at 512 tokens (43-54 TFLOP/s), 1.3-2.8x at 64 to 256 tokens and
 * 1.1-1.3x on grids of 8 to 20 tiles, on the RTX 5080 Laptop; a split-K launch (1024 rows over 2560 at 64 tokens: 0.040 ms against
 * 0.077 ms) keeps the split.</p>
 */
public final class MmaGemmPlan {

    /** Weight rows per workgroup tile ({@code NF_MMA_TM}). */
    public static final int TILE_ROWS = 64;
    /** Tokens per workgroup tile ({@code NF_MMA_TN}). */
    public static final int TILE_TOKENS = 32;
    /** Reduction depth of one stage ({@code NF_MMA_TK}). */
    public static final int TILE_DEPTH = 16;
    /** Work-items per workgroup ({@code NF_MMA_WG}). */
    public static final int WORKGROUP_SIZE = 256;
    /**
     * Tile elements of one workgroup ({@code 64 * 24 + 16 * 40}); the element is a raw 16-bit value on CUDA
     * and a float elsewhere, so the float-sized upper bound of the local memory is twice this count.
     */
    public static final long TILE_ELEMENTS = 64L * 24 + 16L * 40;
    /**
     * Local memory of the GLSL cooperative-matrix twin (shaders/llm/matmul_mma.comp): fp16 tiles of
     * {@code 64 * 24} and {@code 32 * 40} halfs plus the {@code 8 * 16 * 16} float copy-out stage.
     */
    public static final long COOPMAT_SHARED_BYTES = 2L * (64L * 24 + 32L * 40) + 4L * (8L * 16 * 16);
    /**
     * Largest local memory of the kernel over the dialects: the C family's tiles as float elements
     * ({@code NF_MMA_AS}/{@code NF_MMA_BS}) or the cooperative-matrix twin's footprint.
     */
    public static final long TILE_SHARED_BYTES = Math.max(4L * TILE_ELEMENTS, COOPMAT_SHARED_BYTES);
    /** Weight rows per work-group tile of the large tile ({@code NF_MMAL_TM}). */
    public static final int LARGE_TILE_ROWS = 128;
    /** Tokens per work-group tile of the large tile ({@code NF_MMAL_TN}). */
    public static final int LARGE_TILE_TOKENS = 64;

    private final GGMLType type;
    private final String kernelName;
    private final int m;
    private final int k;
    private final int n;
    private final int xStride;
    private final int xOff;
    private final int yStride;
    private final int yOff;

    private MmaGemmPlan(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff) {
        this.type = type;
        this.kernelName = GemmKernels.matmulMma(type);
        this.m = m;
        this.k = k;
        this.n = n;
        this.xStride = xStride;
        this.xOff = xOff;
        this.yStride = yStride;
        this.yOff = yOff;
    }

    /**
     * Plans {@code N} tokens through an {@code [M x K]} weight of {@code type} with contiguous
     * activation and output rows ({@code xStride = K}, {@code yStride = M}, no offsets).
     *
     * @throws IllegalArgumentException as {@link #of}
     */
    public static MmaGemmPlan contiguous(GGMLType type, int m, int k, int n) {
        return of(type, m, k, n, k, 0, m, 0);
    }

    /**
     * Plans one MMA GEMM launch. The shape rules match {@link TiledGemmPlan#of}.
     *
     * @param type    weight type; must have an MMA kernel ({@link GemmKernels#hasMma}, F16 or BF16)
     * @param m       weight rows (output features), in [1, {@code Integer.MAX_VALUE - 64}]
     * @param k       reduction length (input features), in [1, {@code Integer.MAX_VALUE - 16}]; any
     *                value, partial stages load 0
     * @param n       tokens, in [1, {@code Integer.MAX_VALUE - 32}]
     * @param xStride elements between activation rows, at least {@code k}
     * @param xOff    element offset of activation row 0, at least 0
     * @param yStride elements between output rows, at least {@code m}
     * @param yOff    element offset of output row 0, at least 0
     * @throws IllegalArgumentException naming the violated condition
     */
    public static MmaGemmPlan of(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff) {
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.hasMma(type)) {
            throw new IllegalArgumentException("no MMA GEMM kernel for weight type " + type);
        }
        if (m < 1 || k < 1 || n < 1) {
            throw new IllegalArgumentException("MMA GEMM needs M, K, N >= 1, got M=" + m + ", K=" + k + ", N=" + n);
        }
        if (m > Integer.MAX_VALUE - TILE_ROWS || n > Integer.MAX_VALUE - TILE_TOKENS
                || k > Integer.MAX_VALUE - TILE_DEPTH) {
            throw new IllegalArgumentException("MMA GEMM tile indices would overflow int for M=" + m + ", K=" + k
                    + ", N=" + n);
        }
        if (xStride < k) {
            throw new IllegalArgumentException("activation row stride " + xStride + " is shorter than K=" + k);
        }
        if (yStride < m) {
            throw new IllegalArgumentException("output row stride " + yStride + " is shorter than M=" + m);
        }
        if (xOff < 0 || yOff < 0) {
            throw new IllegalArgumentException("MMA GEMM view offsets must be >= 0, got xOff=" + xOff
                    + ", yOff=" + yOff);
        }
        return new MmaGemmPlan(type, m, k, n, xStride, xOff, yStride, yOff);
    }

    /** The weight type of the planned launch (F16 or BF16). */
    public GGMLType type() {
        return type;
    }

    /** {@code <T>_MatMul_MMA} of the weight type. */
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

    public int xStride() {
        return xStride;
    }

    public int xOff() {
        return xOff;
    }

    public int yStride() {
        return yStride;
    }

    public int yOff() {
        return yOff;
    }

    /** Workgroups along the weight rows: {@code ceil(M / 64)}. */
    public int rowTiles() {
        return (m + TILE_ROWS - 1) / TILE_ROWS;
    }

    /** Tokens per workgroup tile ({@code NF_MMA_TN}). */
    public int tileTokens() {
        return TILE_TOKENS;
    }

    /** Workgroups along the tokens: {@code ceil(N / 32)}. */
    public int tokenTiles() {
        return (n + TILE_TOKENS - 1) / TILE_TOKENS;
    }

    /** {@code {rowTiles() * 256, tokenTiles()}}. */
    public long[] globalWorkSize() {
        return new long[]{(long) rowTiles() * WORKGROUP_SIZE, tokenTiles()};
    }

    /** {@code {256, 1}}. */
    public long[] localWorkSize() {
        return new long[]{WORKGROUP_SIZE, 1};
    }

    /** Bytes of one weight row: {@code 2 * K}. */
    public long weightRowBytes() {
        return 2L * k;
    }

    /** Bytes of the weight the kernel reads: {@code M} rows. */
    public long weightBytes() {
        return weightRowBytes() * m;
    }

    /** Smallest activation buffer length in floats: {@code xOff + (N-1)*xStride + K}. */
    public long xElementsRequired() {
        return (long) xOff + (long) (n - 1) * xStride + k;
    }

    /** Smallest output buffer length in floats: {@code yOff + (N-1)*yStride + M}. */
    public long yElementsRequired() {
        return (long) yOff + (long) (n - 1) * yStride + m;
    }

    /**
     * Whether the device can run MMA GEMM at all (work-group size and local memory), independent of
     * the shape and of whether the device selects the kernel at all (the dispatch checks the tensor
     * capability before picking this family).
     *
     * @return null when it can, otherwise the reason
     */
    public static String deviceSupport(DeviceCapabilities caps) {
        Objects.requireNonNull(caps, "caps");
        long maxWg = caps.getMaxWorkGroupSize();
        if (maxWg > 0 && maxWg < WORKGROUP_SIZE) {
            return "MMA GEMM needs work-groups of " + WORKGROUP_SIZE + " work-items; device "
                    + caps.getDeviceName() + " allows " + maxWg;
        }
        long shared = caps.getMaxSharedMemoryBytes();
        if (shared > 0 && shared < TILE_SHARED_BYTES) {
            return "MMA GEMM needs " + TILE_SHARED_BYTES + " bytes of local memory; device "
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
        String reason = deviceSupport(caps);
        if (reason != null) {
            return reason;
        }
        long[] maxGroups = caps.getMaxWorkGroupCount();
        if (rowTiles() > maxGroups[0]) {
            return kernelName() + " needs " + rowTiles() + " work-groups in dimension 0 (M=" + m + "); device "
                    + caps.getDeviceName() + " allows " + maxGroups[0];
        }
        if (tokenTiles() > maxGroups[1]) {
            return kernelName() + " needs " + tokenTiles() + " work-groups in dimension 1 (N=" + n + "); device "
                    + caps.getDeviceName() + " allows " + maxGroups[1];
        }
        return null;
    }

    /**
     * Kernel arguments in declared order ({@link GemmKernels#ARG_M} .. {@link GemmKernels#ARG_Y}),
     * after checking that each buffer covers the bytes this launch touches.
     *
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when a buffer is smaller than the launch needs
     */
    public Object[] args(GpuBuffer weight, GpuBuffer x, GpuBuffer y) {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        if (x == y) {
            throw new IllegalArgumentException(kernelName() + ": activation and output must be distinct buffers");
        }
        requireBytes("weight", weight, weightBytes());
        requireBytes("activation", x, xElementsRequired() * Float.BYTES);
        requireBytes("output", y, yElementsRequired() * Float.BYTES);
        Object[] args = new Object[GemmKernels.MATMUL_TILED_ARG_COUNT];
        args[GemmKernels.ARG_M] = m;
        args[GemmKernels.ARG_K] = k;
        args[GemmKernels.ARG_N] = n;
        args[GemmKernels.ARG_X_STRIDE] = xStride;
        args[GemmKernels.ARG_X_OFF] = xOff;
        args[GemmKernels.ARG_Y_STRIDE] = yStride;
        args[GemmKernels.ARG_Y_OFF] = yOff;
        args[GemmKernels.ARG_WEIGHT] = weight;
        args[GemmKernels.ARG_X] = x;
        args[GemmKernels.ARG_Y] = y;
        return args;
    }

    private void requireBytes(String role, GpuBuffer buffer, long needed) {
        long size = buffer.sizeInBytes();
        if (size < needed) {
            throw new IllegalArgumentException(kernelName() + ": " + role + " buffer holds " + size
                    + " bytes but the launch (M=" + m + ", K=" + k + ", N=" + n + ") needs " + needed);
        }
    }

    @Override
    public String toString() {
        return "MmaGemmPlan{" + kernelName() + ", M=" + m + ", K=" + k + ", N=" + n
                + ", xStride=" + xStride + ", xOff=" + xOff + ", yStride=" + yStride + ", yOff=" + yOff + '}';
    }
}
