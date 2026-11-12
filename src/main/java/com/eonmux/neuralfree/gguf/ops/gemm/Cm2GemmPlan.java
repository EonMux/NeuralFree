package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Objects;

/**
 * Immutable dispatch plan of one {@code <T>_MatMul_CM2} launch over a llama.cpp MMQ weight type or dense F16:
 * the validated shape and view arguments, the 2-D work sizes, the activation staging launch and the argument
 * arrays.
 *
 * <p>The launch computes {@code y[yOff + t*yStride + r] = sum_{k < K} W[r][k] * x[xOff + t*xStride + k]}
 * for {@code r < M}, {@code t < N}: ggml_mul_mat of a quantized weight through one workgroup-scope
 * fp16 cooperative matrix per workgroup (VK_NV_cooperative_matrix2, llama.cpp {@code coopmat2} /
 * {@code mul_mm_cm2.comp}; shaders/llm/matmul_mmq_cm2.comp). {@link #STAGE_KERNEL} first rounds the
 * fp32 activation view to contiguous fp16 rows (row stride {@code K}), zero-padded to a whole number of
 * token tiles ({@link #stagedRows()}), as llama.cpp converts its activations to fp16 and pads them to its
 * tile before the coopmat2 GEMM, so the tensor loads read the B tile unclamped; the weight blocks decode
 * per element through the callbacks of {@code common/quant_mmq_ref.glsl}. The tile spans 128 rows x 256
 * tokens (llama.cpp's large coopmat2 tile; a last token tile ending within its first quarter or half runs
 * at that width), with fp16 products accumulated in fp16 and saturated at the largest finite fp16 value
 * before the fp32 store (llama.cpp's coopmat2 contract for quantized weights): results carry the fp16
 * input rounding of {@link CmGemmPlan} plus an accumulation rounding that grows with the absolute product
 * sum. The split-K twin accumulates each slice in fp16 and sums the slices in fp32. The dispatch prefers this
 * family on Vulkan devices that report {@link DeviceFeature#COOPERATIVE_MATRIX_2}, and only when the
 * fast quantized-weight lane is enabled ({@link MmqMode}).</p>
 *
 * <p>{@code F16_MatMul_CM2} reads a dense fp16 weight through the tensor layout without a decode and
 * accumulates in fp32 (each split-K slice too) without the quantized kernels' saturation at +-65504, the accuracy
 * of the MMA GEMM it replaces ({@link MmaGemmPlan}) and of llama.cpp's f32acc coopmat2 pipelines: F16 checkpoints
 * reach sums past the fp16 range. Its rows must be 16-byte aligned, so {@code K} is a multiple of
 * {@link #DENSE_K_MULTIPLE}; the dispatch keeps the MMA GEMM for any other F16 shape ({@link #denseShapeServes}).
 * It is independent of {@link MmqMode}, like the MMA lane; the device switch
 * {@value com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo#COOPMAT2_PROPERTY}{@code =off} turns every
 * CM2 GEMM off.</p>
 *
 * <p>Geometry: {@code global = {ceil(M / 128) * 256, ceil(N / 256)}}, {@code local = {256, 1}} — the
 * dispatch is 2-D and therefore never folded on Vulkan, so {@link #checkDevice} verifies the group
 * counts against {@code maxWorkGroupCount}. The stage depth of 64 ({@link #TILE_DEPTH}) is the
 * cooperative-matrix K edge and the granularity the split-K twin's slice length must be a multiple
 * of.</p>
 */
public final class Cm2GemmPlan {

    /** Weight rows per workgroup tile ({@code NF_CM2_BM}). */
    public static final int TILE_ROWS = 128;
    /** Tokens per workgroup tile ({@code NF_CM2_BN}); the staged activation rows are padded to a multiple. */
    public static final int TILE_TOKENS = 256;
    /**
     * Tokens per workgroup tile of the {@code _N128} kernels ({@code NF_CM2_BN=128}, llama.cpp's medium coopmat2
     * tile): twice the workgroups of a {@link #TILE_TOKENS} grid over the same launch, for a grid too small to fill
     * the device that split-K does not split ({@link #tileTokensFor}). They read the same staged rows, padded to whole
     * {@link #TILE_TOKENS} tiles.
     */
    public static final int MEDIUM_TILE_TOKENS = 128;
    /** Reduction depth of one stage ({@code NF_CM2_BK}), the split-K slice granularity. */
    public static final int TILE_DEPTH = 64;
    /**
     * Reduction elements the heuristic's split-K slices round up to ({@link SplitKGemmPlan#of(Cm2GemmPlan, int,
     * int)}), as llama.cpp's {@code ggml_vk_matmul} rounds its coopmat2 {@code k_split}. A slice of whole 256-element
     * groups starts on a K-quant super-block, so every tile takes the unclamped interior path and the Q4_K / Q5_K
     * scales stage per block, and holds whole unrolled groups of the 32-element types' stages; slices of whole
     * {@link #TILE_DEPTH} stages that start inside a super-block fall to the clamped stage-by-stage loads.
     */
    public static final int SPLIT_SLICE_ALIGN = 256;
    /** Work-items per workgroup ({@code NF_CM2_WG}). */
    public static final int WORKGROUP_SIZE = 256;
    /**
     * The activation staging kernel of the lane ({@code LLM_Stage_F16_Rows}, shaders/llm/stage_f16.comp):
     * {@code (i n, i rows, i k, i x_stride, i x_off, R float x, W half out)}, four elements per work-item.
     */
    public static final String STAGE_KERNEL = "LLM_Stage_F16_Rows";
    /** Work-group size of the staging launch. */
    public static final int STAGE_LOCAL = 256;
    /** Reduction lengths of a dense F16 weight are multiples of this many halves: 16-byte aligned rows. */
    public static final int DENSE_K_MULTIPLE = 8;

    private final GGMLType type;
    private final boolean decodeVector;
    private final int tileTokens;
    private final String kernelName;
    private final int m;
    private final int k;
    private final int n;
    private final int xStride;
    private final int xOff;
    private final int yStride;
    private final int yOff;

    private Cm2GemmPlan(GGMLType type, boolean decodeVector, int tileTokens, int m, int k, int n, int xStride,
                        int xOff, int yStride, int yOff) {
        this.type = type;
        this.decodeVector = decodeVector;
        this.tileTokens = tileTokens;
        this.kernelName = tileTokens == MEDIUM_TILE_TOKENS ? GemmKernels.matmulCm2Medium(type, decodeVector)
                : GemmKernels.matmulCm2(type, decodeVector);
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
    public static Cm2GemmPlan contiguous(GGMLType type, int m, int k, int n) {
        return of(type, m, k, n, k, 0, m, 0);
    }

    /**
     * Plans one CM2 GEMM launch. The shape rules match {@link CmGemmPlan#of}: the type must have a
     * CM2 kernel (the MMQ set without the i-quants), {@code k} a multiple of the type's block size.
     *
     * @param type    weight type; must have a CM2 kernel ({@link GemmKernels#hasCm2})
     * @param m       weight rows (output features), in [1, {@code Integer.MAX_VALUE - 128}]
     * @param k       reduction length (input features), in [1, {@code Integer.MAX_VALUE - 64}] and a
     *                multiple of the type's block size (32 or 256)
     * @param n       tokens, in [1, {@code Integer.MAX_VALUE - 256}]
     * @param xStride elements between activation rows, at least {@code k}
     * @param xOff    element offset of activation row 0, at least 0
     * @param yStride elements between output rows, at least {@code m}
     * @param yOff    element offset of output row 0, at least 0
     * @throws IllegalArgumentException naming the violated condition
     */
    public static Cm2GemmPlan of(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff) {
        return of(type, false, m, k, n, xStride, xOff, yStride, yOff);
    }

    /**
     * {@link #of(GGMLType, int, int, int, int, int, int, int)} in the decode form {@code decodeVector} selects: the
     * {@code <T>_MatMul_CM2V} kernel, whose weight tile decodes four elements per callback, or {@code <T>_MatMul_CM2}.
     * Both forms decode every element to the same fp16 value and share the tile, the arguments and the dispatch.
     *
     * @throws IllegalArgumentException as the plain factory, and for {@code decodeVector} over a type without a CM2V
     *                                  kernel ({@link GemmKernels#hasCm2DecodeVector})
     */
    public static Cm2GemmPlan of(GGMLType type, boolean decodeVector, int m, int k, int n, int xStride, int xOff,
                                 int yStride, int yOff) {
        return of(type, decodeVector, TILE_TOKENS, m, k, n, xStride, xOff, yStride, yOff);
    }

    /**
     * {@link #of(GGMLType, boolean, int, int, int, int, int, int, int)} over the token tile {@code tileTokens}:
     * {@link #TILE_TOKENS} ({@code <T>_MatMul_CM2[V]}) or {@link #MEDIUM_TILE_TOKENS} ({@code <T>_MatMul_CM2[V]_N128}).
     * Both tiles compute every output element over the same reduction stages; the tile sets the grid.
     *
     * @throws IllegalArgumentException as the other factories, and for any other tile width
     */
    public static Cm2GemmPlan of(GGMLType type, boolean decodeVector, int tileTokens, int m, int k, int n,
                                 int xStride, int xOff, int yStride, int yOff) {
        Objects.requireNonNull(type, "type");
        if (tileTokens != TILE_TOKENS && tileTokens != MEDIUM_TILE_TOKENS) {
            throw new IllegalArgumentException("CM2 GEMM tiles span " + TILE_TOKENS + " or " + MEDIUM_TILE_TOKENS
                    + " tokens, got " + tileTokens);
        }
        if (decodeVector && !GemmKernels.hasCm2DecodeVector(type)) {
            throw new IllegalArgumentException("no CM2V GEMM kernel for weight type " + type);
        }
        if (!GemmKernels.hasCm2(type)) {
            throw new IllegalArgumentException("no CM2 GEMM kernel for weight type " + type);
        }
        if (m < 1 || k < 1 || n < 1) {
            throw new IllegalArgumentException("CM2 GEMM needs M, K, N >= 1, got M=" + m + ", K=" + k + ", N=" + n);
        }
        if (m > Integer.MAX_VALUE - TILE_ROWS || n > Integer.MAX_VALUE - TILE_TOKENS
                || k > Integer.MAX_VALUE - TILE_DEPTH) {
            throw new IllegalArgumentException("CM2 GEMM tile indices would overflow int for M=" + m + ", K=" + k
                    + ", N=" + n);
        }
        if (type == GGMLType.F16 && !denseShapeServes(k)) {
            throw new IllegalArgumentException("CM2 GEMM over F16 needs K to be a multiple of " + DENSE_K_MULTIPLE
                    + " (16-byte aligned rows), got K=" + k);
        }
        int blockSize = type.getBlockSize();
        if (k % blockSize != 0) {
            throw new IllegalArgumentException("CM2 GEMM over " + type + " needs K to be a multiple of the block size "
                    + blockSize + ", got K=" + k);
        }
        if (xStride < k) {
            throw new IllegalArgumentException("activation row stride " + xStride + " is shorter than K=" + k);
        }
        if (yStride < m) {
            throw new IllegalArgumentException("output row stride " + yStride + " is shorter than M=" + m);
        }
        if (xOff < 0 || yOff < 0) {
            throw new IllegalArgumentException("CM2 GEMM view offsets must be >= 0, got xOff=" + xOff
                    + ", yOff=" + yOff);
        }
        return new Cm2GemmPlan(type, decodeVector, tileTokens, m, k, n, xStride, xOff, yStride, yOff);
    }

    /**
     * The token tile of a CM2 launch of {@code m} rows by {@code n} tokens over a reduction of {@code k}:
     * {@link #MEDIUM_TILE_TOKENS} when the {@link #TILE_TOKENS} grid fills at most half of the device's compute units
     * (so the 128-token grid, at most twice as many workgroups, still fits one wave of them) and split-K keeps the
     * launch whole
     * ({@link SplitKGemmPlan#splitsFor}: a reduction below {@link SplitKGemmPlan#MIN_SPLIT_K}); otherwise
     * {@link #TILE_TOKENS}. A grid of one token tile gains nothing from the narrower tile, and a device that does not
     * report its compute units keeps the 256-token tile. Where split-K applies it is the better use of the idle compute
     * units: the slices keep the 256-token tile's operand reuse, which the 128-token tile halves on the activation
     * side.
     *
     * @param computeUnits compute units of the device, or {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}
     * @throws IllegalArgumentException for a non-positive shape or a negative compute-unit count
     */
    public static int tileTokensFor(int computeUnits, int m, int n, int k) {
        if (m < 1 || n < 1 || k < 1) {
            throw new IllegalArgumentException("CM2 tile choice needs M, N, K >= 1, got M=" + m + ", N=" + n + ", K="
                    + k);
        }
        if (computeUnits < DeviceCapabilities.UNKNOWN_COMPUTE_UNITS) {
            throw new IllegalArgumentException("CM2 tile choice needs a non-negative compute-unit count, got "
                    + computeUnits);
        }
        long rowTiles = (m + (long) TILE_ROWS - 1) / TILE_ROWS;
        long largeTiles = rowTiles * ((n + (long) TILE_TOKENS - 1) / TILE_TOKENS);
        long mediumTiles = rowTiles * ((n + (long) MEDIUM_TILE_TOKENS - 1) / MEDIUM_TILE_TOKENS);
        if (computeUnits == DeviceCapabilities.UNKNOWN_COMPUTE_UNITS || mediumTiles == largeTiles
                || largeTiles > computeUnits / 2
                || SplitKGemmPlan.splitsFor(computeUnits, (int) rowTiles,
                        (int) (largeTiles / rowTiles), k) > 1) {
            return TILE_TOKENS;
        }
        return MEDIUM_TILE_TOKENS;
    }

    /** Whether the dense F16 kernel serves a reduction of {@code k} elements: rows of 16-byte aligned halves. */
    public static boolean denseShapeServes(int k) {
        return k > 0 && k % DENSE_K_MULTIPLE == 0;
    }

    /** The weight type of the planned launch (a CM2 type). */
    public GGMLType type() {
        return type;
    }

    /** Whether the launch runs the four-element decode form {@code <T>_MatMul_CM2V}. */
    public boolean decodeVector() {
        return decodeVector;
    }

    /** {@code <T>_MatMul_CM2V} or {@code <T>_MatMul_CM2} of the weight type. */
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

    /** Work-groups along the weight rows: {@code ceil(M / 128)}. */
    public int rowTiles() {
        return (m + TILE_ROWS - 1) / TILE_ROWS;
    }

    /** Tokens per workgroup tile ({@code NF_CM2_BN}): {@link #TILE_TOKENS} or {@link #MEDIUM_TILE_TOKENS}. */
    public int tileTokens() {
        return tileTokens;
    }

    /** Work-groups along the tokens: {@code ceil(N / tileTokens())}. */
    public int tokenTiles() {
        return (n + tileTokens - 1) / tileTokens;
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

    /**
     * Staged activation rows: {@code N} padded with zero rows to a whole number of {@link #TILE_TOKENS} tiles, for
     * either tile width, so one staging serves every CM2 launch over the same rows (a {@code GpuOps.SharedQ8} scope's
     * projections may run different tiles).
     */
    public int stagedRows() {
        return (n + TILE_TOKENS - 1) / TILE_TOKENS * TILE_TOKENS;
    }

    /** Halves of the staged fp16 activation rows the GEMM reads: {@code stagedRows() * K}. */
    public long stagedHalves() {
        return (long) stagedRows() * k;
    }

    /** Global work size of the staging launch: four elements of a staged row per work-item. */
    public long[] stageGlobalWorkSize() {
        return new long[]{stagedHalves() / 4};
    }

    /** Local work size of the staging launch: {@code {STAGE_LOCAL}}. */
    public static long[] stageLocalWorkSize() {
        return new long[]{STAGE_LOCAL};
    }

    /**
     * Arguments of this launch's staging pass: the fp32 activation view {@code x} ({@code xOff}, {@code xStride})
     * rounded into the {@link #stagedRows()} contiguous fp16 rows of {@code staged}, after checking that both
     * buffers cover it.
     *
     * @throws IllegalArgumentException when a buffer is smaller than the staging touches or {@code x} and
     *                                  {@code staged} are the same buffer
     */
    public Object[] stageArgs(GpuBuffer x, GpuBuffer staged) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(staged, "staged");
        if (x == staged) {
            throw new IllegalArgumentException("fp16 staging cannot run in place");
        }
        requireBytes("activation", x, xElementsRequired() * Float.BYTES);
        requireBytes("staged activation", staged, stagedHalves() * Short.BYTES);
        return new Object[]{n, stagedRows(), k, xStride, xOff, x, staged};
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
     * Whether the device can run CM2 GEMM over {@code type} at all: the workgroup-scope cooperative
     * matrix feature and work-groups of 256, independent of the shape. The CM2 kernels reserve no
     * shared staging of their own, so unlike {@link CmGemmPlan} there is no local-memory bound.
     *
     * @return null when it can, otherwise the reason
     */
    public static String deviceSupport(GGMLType type, DeviceCapabilities caps) {
        return deviceSupport(type, caps, false);
    }

    /**
     * {@link #deviceSupport(GGMLType, DeviceCapabilities)} of the decode form {@code decodeVector} selects: the
     * {@code <T>_MatMul_CM2V} form also needs {@link DeviceFeature#COOPERATIVE_MATRIX_2_DECODE_VECTOR}.
     *
     * @return null when it can, otherwise the reason
     */
    public static String deviceSupport(GGMLType type, DeviceCapabilities caps, boolean decodeVector) {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(caps, "caps");
        if (!GemmKernels.hasCm2(type)) {
            return "no CM2 GEMM kernel for weight type " + type;
        }
        if (decodeVector && !GemmKernels.hasCm2DecodeVector(type)) {
            return "no CM2V GEMM kernel for weight type " + type;
        }
        if (decodeVector && !caps.hasFeature(DeviceFeature.COOPERATIVE_MATRIX_2_DECODE_VECTOR)) {
            return "CM2V GEMM needs " + DeviceFeature.COOPERATIVE_MATRIX_2_DECODE_VECTOR + "; device "
                    + caps.getDeviceName() + " does not report it";
        }
        if (!caps.hasFeature(DeviceFeature.COOPERATIVE_MATRIX_2)) {
            return "CM2 GEMM needs " + DeviceFeature.COOPERATIVE_MATRIX_2 + "; device " + caps.getDeviceName()
                    + " does not report it";
        }
        long maxWg = caps.getMaxWorkGroupSize();
        if (maxWg > 0 && maxWg < WORKGROUP_SIZE) {
            return "CM2 GEMM needs work-groups of " + WORKGROUP_SIZE + " work-items; device "
                    + caps.getDeviceName() + " allows " + maxWg;
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
        String reason = deviceSupport(type, caps, decodeVector);
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
     * GEMM arguments in declared order ({@link GemmKernels#ARG_M} .. {@link GemmKernels#ARG_Y}) over the
     * staged fp16 rows ({@code x_stride = K}, {@code x_off = 0} halves), after checking that each buffer
     * covers the bytes this launch touches.
     *
     * @throws NullPointerException     for a null buffer
     * @throws IllegalArgumentException when a buffer is smaller than the launch needs
     */
    public Object[] args(GpuBuffer weight, GpuBuffer staged, GpuBuffer y) {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(staged, "staged");
        Objects.requireNonNull(y, "y");
        if (staged == y) {
            throw new IllegalArgumentException(kernelName + ": activation and output must be distinct buffers");
        }
        requireBytes("weight", weight, weightBytes());
        requireBytes("staged activation", staged, stagedHalves() * Short.BYTES);
        requireBytes("output", y, yElementsRequired() * Float.BYTES);
        Object[] args = new Object[GemmKernels.MATMUL_TILED_ARG_COUNT];
        args[GemmKernels.ARG_M] = m;
        args[GemmKernels.ARG_K] = k;
        args[GemmKernels.ARG_N] = n;
        args[GemmKernels.ARG_X_STRIDE] = k;
        args[GemmKernels.ARG_X_OFF] = 0;
        args[GemmKernels.ARG_X] = staged;
        args[GemmKernels.ARG_Y_STRIDE] = yStride;
        args[GemmKernels.ARG_Y_OFF] = yOff;
        args[GemmKernels.ARG_WEIGHT] = weight;
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
        return "Cm2GemmPlan{" + kernelName + ", M=" + m + ", K=" + k + ", N=" + n
                + ", xStride=" + xStride + ", xOff=" + xOff + ", yStride=" + yStride + ", yOff=" + yOff + '}';
    }
}
