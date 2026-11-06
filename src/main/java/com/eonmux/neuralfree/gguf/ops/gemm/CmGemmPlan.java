package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Objects;

/**
 * Immutable dispatch plan of one {@code <T>_MatMul_CM} launch over a llama.cpp MMQ weight type: the
 * validated shape and view arguments, the 2-D work sizes and the argument array.
 *
 * <p>The kernel computes {@code y[yOff + t*yStride + r] = sum_{k < K} W[r][k] * x[xOff + t*xStride + k]}
 * for {@code r < M}, {@code t < N}: ggml_mul_mat of a quantized weight, dequantizing the weight codes
 * into fp16 tiles and converting the fp32 activations to fp16, with fp16 cooperative-matrix products
 * and an fp32 accumulator (shaders/llm/matmul_mmq_cm.comp; the portable scalar rows of
 * kernels/llm/matmul_cm.cl share the tiled arithmetic). Results differ from the tiled GEMM by the
 * fp16 rounding of the staged tiles. The dispatch selects this family only on Vulkan devices with
 * cooperative-matrix support, and only when the fast quantized-weight lane is enabled
 * ({@link MmqMode}).</p>
 *
 * <p>Geometry: {@code global = {ceil(M / 64) * 256, ceil(N / 32)}}, {@code local = {256, 1}} — the
 * MMA tile, with the MMQ weight-byte accounting of {@link MmqGemmPlan}. The dispatch is 2-D and
 * therefore never folded on Vulkan, so {@link #checkDevice} verifies the group counts against
 * {@code maxWorkGroupCount}.</p>
 */
public final class CmGemmPlan {

    /** Weight rows per workgroup tile ({@code NF_CM_TM}). */
    public static final int TILE_ROWS = MmaGemmPlan.TILE_ROWS;
    /** Tokens per workgroup tile ({@code NF_CM_TN}). */
    public static final int TILE_TOKENS = MmaGemmPlan.TILE_TOKENS;
    /** Reduction depth of one stage ({@code NF_CM_TK}). */
    public static final int TILE_DEPTH = 16;
    /** Work-items per workgroup ({@code NF_CM_WG}). */
    public static final int WORKGROUP_SIZE = MmaGemmPlan.WORKGROUP_SIZE;
    /**
     * Largest local memory of the kernel over the dialects: the cooperative-matrix form's fp16 tiles
     * and copy-out stage (as the MMA twin, {@link MmaGemmPlan#COOPMAT_SHARED_BYTES}) or the tiled
     * twin's tiles (kernels/llm/matmul_cm.cl).
     */
    public static final long TILE_SHARED_BYTES = Math.max(MmaGemmPlan.COOPMAT_SHARED_BYTES,
            TiledGemmPlan.TILE_SHARED_BYTES);

    private final GGMLType type;
    private final String kernelName;
    private final int m;
    private final int k;
    private final int n;
    private final int xStride;
    private final int xOff;
    private final int yStride;
    private final int yOff;

    private CmGemmPlan(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff) {
        this.type = type;
        this.kernelName = GemmKernels.matmulCm(type);
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
    public static CmGemmPlan contiguous(GGMLType type, int m, int k, int n) {
        return of(type, m, k, n, k, 0, m, 0);
    }

    /**
     * Plans one CM GEMM launch. The shape rules match {@link MmqGemmPlan#of}: the tile indices are
     * 16-deep, so {@code k} must stay below {@code Integer.MAX_VALUE - 16} and be a multiple of the
     * type's block size.
     *
     * @param type    weight type; must have a CM kernel ({@link GemmKernels#hasCm}, the MMQ set)
     * @param m       weight rows (output features), in [1, {@code Integer.MAX_VALUE - 64}]
     * @param k       reduction length (input features), in [1, {@code Integer.MAX_VALUE - 16}] and a
     *                multiple of the type's block size (32 or 256)
     * @param n       tokens, in [1, {@code Integer.MAX_VALUE - 32}]
     * @param xStride elements between activation rows, at least {@code k}
     * @param xOff    element offset of activation row 0, at least 0
     * @param yStride elements between output rows, at least {@code m}
     * @param yOff    element offset of output row 0, at least 0
     * @throws IllegalArgumentException naming the violated condition
     */
    public static CmGemmPlan of(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff) {
        Objects.requireNonNull(type, "type");
        if (!GemmKernels.hasCm(type)) {
            throw new IllegalArgumentException("no CM GEMM kernel for weight type " + type);
        }
        if (m < 1 || k < 1 || n < 1) {
            throw new IllegalArgumentException("CM GEMM needs M, K, N >= 1, got M=" + m + ", K=" + k + ", N=" + n);
        }
        if (m > Integer.MAX_VALUE - TILE_ROWS || n > Integer.MAX_VALUE - TILE_TOKENS
                || k > Integer.MAX_VALUE - TILE_DEPTH) {
            throw new IllegalArgumentException("CM GEMM tile indices would overflow int for M=" + m + ", K=" + k
                    + ", N=" + n);
        }
        int blockSize = type.getBlockSize();
        if (k % blockSize != 0) {
            throw new IllegalArgumentException("CM GEMM over " + type + " needs K to be a multiple of the block size "
                    + blockSize + ", got K=" + k);
        }
        if (xStride < k) {
            throw new IllegalArgumentException("activation row stride " + xStride + " is shorter than K=" + k);
        }
        if (yStride < m) {
            throw new IllegalArgumentException("output row stride " + yStride + " is shorter than M=" + m);
        }
        if (xOff < 0 || yOff < 0) {
            throw new IllegalArgumentException("CM GEMM view offsets must be >= 0, got xOff=" + xOff
                    + ", yOff=" + yOff);
        }
        return new CmGemmPlan(type, m, k, n, xStride, xOff, yStride, yOff);
    }

    /** The weight type of the planned launch (an MMQ type). */
    public GGMLType type() {
        return type;
    }

    /** {@code <T>_MatMul_CM} of the weight type. */
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

    /** Work-groups along the weight rows: {@code ceil(M / 64)}. */
    public int rowTiles() {
        return (m + TILE_ROWS - 1) / TILE_ROWS;
    }

    /** Tokens per workgroup tile ({@code NF_CM_TN}). */
    public int tileTokens() {
        return TILE_TOKENS;
    }

    /** Work-groups along the tokens: {@code ceil(N / 32)}. */
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
     * Whether the device can run CM GEMM over {@code type} at all (work-group size and local memory),
     * independent of the shape.
     *
     * @return null when it can, otherwise the reason
     */
    public static String deviceSupport(GGMLType type, DeviceCapabilities caps) {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(caps, "caps");
        if (!GemmKernels.hasCm(type)) {
            return "no CM GEMM kernel for weight type " + type;
        }
        long maxWg = caps.getMaxWorkGroupSize();
        if (maxWg > 0 && maxWg < WORKGROUP_SIZE) {
            return "CM GEMM needs work-groups of " + WORKGROUP_SIZE + " work-items; device "
                    + caps.getDeviceName() + " allows " + maxWg;
        }
        long shared = caps.getMaxSharedMemoryBytes();
        if (shared > 0 && shared < TILE_SHARED_BYTES) {
            return "CM GEMM over " + type + " needs " + TILE_SHARED_BYTES + " bytes of local memory; device "
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
        return "CmGemmPlan{" + kernelName + ", M=" + m + ", K=" + k + ", N=" + n
                + ", xStride=" + xStride + ", xOff=" + xOff + ", yStride=" + yStride + ", yOff=" + yOff + '}';
    }
}
