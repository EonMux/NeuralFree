package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Objects;

/**
 * Immutable dispatch plan of one {@code <T>_MatMul_Tiled} launch: the kernel name, the validated
 * shape and view arguments, the 2-D work sizes and the argument array.
 *
 * <p>The kernel computes {@code y[yOff + t*yStride + r] = sum_{k < K} W[r][k] * x[xOff + t*xStride + k]}
 * for {@code r < M}, {@code t < N}: ggml_mul_mat of an {@code [M x K]} weight with {@code N} activation
 * rows. A workgroup of {@link #WORKGROUP_SIZE} work-items computes a tile of {@link #TILE_ROWS} weight
 * rows by {@link #TILE_TOKENS} tokens, dequantizing {@link #TILE_DEPTH}-deep weight slices into local
 * float memory ({@code kernels/llm/common/matmul_tiled_family.h}).</p>
 *
 * <p>Geometry: {@code global = {ceil(M / 64) * 256, ceil(N / 32)}}, {@code local = {256, 1}}. The
 * dispatch is 2-D and therefore never folded on Vulkan, so {@link #checkDevice} verifies the group
 * counts against {@code maxWorkGroupCount}.</p>
 */
public final class TiledGemmPlan {

    /** Weight rows per workgroup tile ({@code NF_GEMM_TM}). */
    public static final int TILE_ROWS = 64;
    /** Tokens per workgroup tile ({@code NF_GEMM_TN}). */
    public static final int TILE_TOKENS = 32;
    /** Reduction depth of one local-memory stage ({@code NF_GEMM_TK}). */
    public static final int TILE_DEPTH = 16;
    /** Work-items per workgroup ({@code NF_GEMM_WG}). */
    public static final int WORKGROUP_SIZE = 256;
    /** Local memory of the two padded tiles: {@code 4 * (16 * 65 + 16 * 33)} bytes. */
    public static final long TILE_SHARED_BYTES = 4L * ((long) TILE_DEPTH * (TILE_ROWS + 1)
            + (long) TILE_DEPTH * (TILE_TOKENS + 1));

    private final GGMLType type;
    private final String kernelName;
    private final int m;
    private final int k;
    private final int n;
    private final int xStride;
    private final int xOff;
    private final int yStride;
    private final int yOff;

    private TiledGemmPlan(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff) {
        this.type = type;
        this.kernelName = GemmKernels.matmulTiled(type);
        this.m = m;
        this.k = k;
        this.n = n;
        this.xStride = xStride;
        this.xOff = xOff;
        this.yStride = yStride;
        this.yOff = yOff;
    }

    /**
     * Plans {@code N} tokens through an {@code [M x K]} weight with contiguous activation and output
     * rows ({@code xStride = K}, {@code yStride = M}, no offsets).
     *
     * @throws IllegalArgumentException as {@link #of}
     */
    public static TiledGemmPlan contiguous(GGMLType type, int m, int k, int n) {
        return of(type, m, k, n, k, 0, m, 0);
    }

    /**
     * Plans one tiled GEMM launch.
     *
     * @param type    weight type; must have a tiled kernel ({@link GemmKernels#hasTiled})
     * @param m       weight rows (output features), in [1, {@code Integer.MAX_VALUE - 64}]
     * @param k       reduction length (input features), in [1, {@code Integer.MAX_VALUE - 16}] and a
     *                multiple of the type's block size
     * @param n       tokens, in [1, {@code Integer.MAX_VALUE - 32}]
     * @param xStride elements between activation rows, at least {@code k}
     * @param xOff    element offset of activation row 0, at least 0
     * @param yStride elements between output rows, at least {@code m}
     * @param yOff    element offset of output row 0, at least 0
     * @throws IllegalArgumentException naming the violated condition
     */
    public static TiledGemmPlan of(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff) {
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.hasTiled(type)) {
            throw new IllegalArgumentException("no tiled GEMM kernel for weight type " + type);
        }
        if (m < 1 || k < 1 || n < 1) {
            throw new IllegalArgumentException("tiled GEMM needs M, K, N >= 1, got M=" + m + ", K=" + k + ", N=" + n);
        }
        if (m > Integer.MAX_VALUE - TILE_ROWS || n > Integer.MAX_VALUE - TILE_TOKENS
                || k > Integer.MAX_VALUE - TILE_DEPTH) {
            throw new IllegalArgumentException("tiled GEMM tile indices would overflow int for M=" + m + ", K=" + k
                    + ", N=" + n);
        }
        int blockSize = type.getBlockSize();
        if (k % blockSize != 0) {
            throw new IllegalArgumentException("tiled GEMM over " + type + " needs K to be a multiple of the block size "
                    + blockSize + ", got K=" + k);
        }
        if (xStride < k) {
            throw new IllegalArgumentException("activation row stride " + xStride + " is shorter than K=" + k);
        }
        if (yStride < m) {
            throw new IllegalArgumentException("output row stride " + yStride + " is shorter than M=" + m);
        }
        if (xOff < 0 || yOff < 0) {
            throw new IllegalArgumentException("tiled GEMM view offsets must be >= 0, got xOff=" + xOff + ", yOff=" + yOff);
        }
        return new TiledGemmPlan(type, m, k, n, xStride, xOff, yStride, yOff);
    }

    /**
     * Local memory one workgroup of this weight type may use on any backend: the tiles plus, for the
     * IQ grid types, the grid the Vulkan shader copies into shared memory ({@code iq_tables.glsl}).
     */
    public static long requiredSharedBytes(GGMLType type) {
        return TILE_SHARED_BYTES + iqGridSharedBytes(type);
    }

    private static long iqGridSharedBytes(GGMLType type) {
        switch (type) {
            case IQ2_XXS:
                return 256L * 8;
            case IQ2_XS:
                return 512L * 8;
            case IQ2_S:
                return 1024L * 8;
            case IQ3_XXS:
                return 256L * 4;
            case IQ3_S:
                return 512L * 4;
            case IQ1_S:
            case IQ1_M:
                return 1024L * 4;
            default:
                return 0L;
        }
    }

    /**
     * Whether the device can run tiled GEMM over {@code type} at all (work-group size and local memory),
     * independent of the shape.
     *
     * @return null when it can, otherwise the reason
     */
    public static String deviceSupport(GGMLType type, DeviceCapabilities caps) {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(caps, "caps");
        if (!GemmKernels.hasTiled(type)) {
            return "no tiled GEMM kernel for weight type " + type;
        }
        long maxWg = caps.getMaxWorkGroupSize();
        if (maxWg > 0 && maxWg < WORKGROUP_SIZE) {
            return "tiled GEMM needs work-groups of " + WORKGROUP_SIZE + " work-items; device "
                    + caps.getDeviceName() + " allows " + maxWg;
        }
        long shared = caps.getMaxSharedMemoryBytes();
        long needed = requiredSharedBytes(type);
        if (shared > 0 && shared < needed) {
            return "tiled GEMM over " + type + " needs " + needed + " bytes of local memory; device "
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
        String reason = deviceSupport(type, caps);
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

    /** Bytes of one weight row: {@code K / blockSize * typeSize}. */
    public long weightRowBytes() {
        return (long) (k / type.getBlockSize()) * type.getTypeSize();
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
            throw new IllegalArgumentException(kernelName + ": activation and output must be distinct buffers");
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
            throw new IllegalArgumentException(kernelName + ": " + role + " buffer holds " + size
                    + " bytes but the launch (M=" + m + ", K=" + k + ", N=" + n + ") needs " + needed);
        }
    }

    @Override
    public String toString() {
        return "TiledGemmPlan{" + kernelName + ", M=" + m + ", K=" + k + ", N=" + n
                + ", xStride=" + xStride + ", xOff=" + xOff + ", yStride=" + yStride + ", yOff=" + yOff + '}';
    }
}
