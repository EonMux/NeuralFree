package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Kernel names and argument indices of the GEMM family ({@code kernels/llm/matmul_tiled.cl},
 * {@code kernels/llm/sampling_topk.cl}; GLSL twins {@code shaders/llm/matmul_tiled.comp},
 * {@code shaders/llm/topk_partial.comp}).
 *
 * <p>Argument lists (buffers and scalars in declared order):</p>
 * <ul>
 *   <li>{@code <T>_MatMul_Tiled(int M, int K, int N, int x_stride, int x_off, int y_stride, int y_off,
 *       GpuBuffer weight, GpuBuffer x, GpuBuffer y)}: {@code y[y_off + t*y_stride + r] =
 *       sum_k W[r][k] * x[x_off + t*x_stride + k]}; dispatch geometry in {@link TiledGemmPlan}.</li>
 *   <li>{@code LLM_TopK_Partial(int n, int k, float inv_temp, GpuBuffer logits, GpuBuffer cand_val,
 *       GpuBuffer cand_idx)}: per-slice top-k candidates for the host sampler; dispatch geometry in
 *       {@link TopKPartialPlan}.</li>
 * </ul>
 */
public final class GemmKernels {

    /** Name suffix of the per-weight-type tiled GEMM kernels, {@code <T>_MatMul_Tiled}. */
    public static final String MATMUL_TILED_SUFFIX = "_MatMul_Tiled";
    /** Partial top-k over vocabulary slices. */
    public static final String TOPK_PARTIAL = "LLM_TopK_Partial";

    /** {@code <T>_MatMul_Tiled}: weight rows (output features). */
    public static final int ARG_M = 0;
    /** {@code <T>_MatMul_Tiled}: reduction length (input features). */
    public static final int ARG_K = 1;
    /** {@code <T>_MatMul_Tiled}: token count; set per step. */
    public static final int ARG_N = 2;
    /** {@code <T>_MatMul_Tiled}: element stride between activation rows. */
    public static final int ARG_X_STRIDE = 3;
    /** {@code <T>_MatMul_Tiled}: element offset of the first activation row; set per step for views. */
    public static final int ARG_X_OFF = 4;
    /** {@code <T>_MatMul_Tiled}: element stride between output rows. */
    public static final int ARG_Y_STRIDE = 5;
    /** {@code <T>_MatMul_Tiled}: element offset of the first output row; set per step for views. */
    public static final int ARG_Y_OFF = 6;
    /** {@code <T>_MatMul_Tiled}: weight bytes. */
    public static final int ARG_WEIGHT = 7;
    /** {@code <T>_MatMul_Tiled}: F32 activations. */
    public static final int ARG_X = 8;
    /** {@code <T>_MatMul_Tiled}: F32 output. */
    public static final int ARG_Y = 9;
    /** Number of {@code <T>_MatMul_Tiled} arguments. */
    public static final int MATMUL_TILED_ARG_COUNT = 10;

    /** {@code LLM_TopK_Partial}: logits length. */
    public static final int TOPK_ARG_N = 0;
    /** {@code LLM_TopK_Partial}: candidates per slice. */
    public static final int TOPK_ARG_K = 1;
    /** {@code LLM_TopK_Partial}: scale applied to the reported candidate values; set per step. */
    public static final int TOPK_ARG_INV_TEMP = 2;
    /** {@code LLM_TopK_Partial}: F32 logits. */
    public static final int TOPK_ARG_LOGITS = 3;
    /** {@code LLM_TopK_Partial}: F32 candidate values, {@code groups * k}. */
    public static final int TOPK_ARG_CAND_VAL = 4;
    /** {@code LLM_TopK_Partial}: int32 candidate indices, {@code groups * k}; -1 marks an empty slot. */
    public static final int TOPK_ARG_CAND_IDX = 5;
    /** Number of {@code LLM_TopK_Partial} arguments. */
    public static final int TOPK_ARG_COUNT = 6;

    /** Weight types with a {@code <T>_MatMul_Tiled} kernel: F32, F16, BF16 and every GPU quant type. */
    public static final Set<GGMLType> TILED_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.F32, GGMLType.F16, GGMLType.BF16,
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
            GGMLType.IQ4_NL, GGMLType.IQ4_XS, GGMLType.MXFP4,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
            GGMLType.IQ1_S, GGMLType.IQ1_M,
            GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.NVFP4, GGMLType.Q1_0, GGMLType.Q2_0));

    private GemmKernels() {
    }

    /** Whether {@code type} has a tiled GEMM kernel. */
    public static boolean hasTiled(GGMLType type) {
        return type != null && TILED_TYPES.contains(type);
    }

    /**
     * Name of the tiled GEMM kernel of a weight type.
     *
     * @throws IllegalArgumentException when the type has no tiled GEMM kernel
     */
    public static String matmulTiled(GGMLType type) {
        if (!hasTiled(type)) {
            throw new IllegalArgumentException("no tiled GEMM kernel for weight type " + type
                    + "; supported: " + TILED_TYPES);
        }
        return type.name() + MATMUL_TILED_SUFFIX;
    }

    /** Every kernel name of this family, tiled GEMM kernels in {@link GGMLType} order then the top-k. */
    public static List<String> kernelNames() {
        List<String> names = new ArrayList<>(TILED_TYPES.size() + 1);
        for (GGMLType type : TILED_TYPES) {
            names.add(matmulTiled(type));
        }
        names.add(TOPK_PARTIAL);
        return Collections.unmodifiableList(names);
    }

    /**
     * Dispatch geometry of {@code LLM_TopK_Partial}: {@code groups} workgroups of {@code local}
     * work-items, each reducing one vocabulary slice of {@code ceil(n / groups)} logits to its
     * {@code k} best candidates, so the host reads {@code groups * k} candidates instead of {@code n}
     * logits.
     */
    public static final class TopKPartialPlan {

        /** Largest supported {@code k}. */
        public static final int MAX_K = 64;
        /** Largest logits length; keeps the kernel's slice bounds and strided scan within {@code int}. */
        public static final int MAX_N = 1 << 30;
        /** Logits per workgroup the group count aims for. */
        public static final int SLICE_TARGET = 4096;
        /** Largest group count; far below every device's 1-D group limit, so dispatches are never folded. */
        public static final int MAX_GROUPS = 1024;
        /** Smallest work-group size (one full CUDA warp). */
        public static final int MIN_LOCAL = 32;

        private final int n;
        private final int k;
        private final int groups;
        private final int local;

        private TopKPartialPlan(int n, int k, int groups, int local) {
            this.n = n;
            this.k = k;
            this.groups = groups;
            this.local = local;
        }

        /**
         * Plans a partial top-k over {@code n} logits.
         *
         * @param local work-group size: a power of two in [{@link #MIN_LOCAL}, {@code LLM_DIMX}]
         * @throws IllegalArgumentException for {@code n} outside [1, {@link #MAX_N}], {@code k} outside
         *                                  [1, {@link #MAX_K}] or an invalid work-group size
         */
        public static TopKPartialPlan of(int n, int k, int local) {
            if (n < 1) {
                throw new IllegalArgumentException("LLM_TopK_Partial needs at least one logit, got n=" + n);
            }
            if (n > MAX_N) {
                throw new IllegalArgumentException("LLM_TopK_Partial supports at most " + MAX_N + " logits, got n=" + n);
            }
            if (k < 1 || k > MAX_K) {
                throw new IllegalArgumentException("LLM_TopK_Partial supports 1 <= k <= " + MAX_K + ", got k=" + k);
            }
            if (local < MIN_LOCAL || local > LlmKernelConstants.LLM_DIMX || Integer.bitCount(local) != 1) {
                throw new IllegalArgumentException("LLM_TopK_Partial work-group size must be a power of two in ["
                        + MIN_LOCAL + ", " + LlmKernelConstants.LLM_DIMX + "], got " + local);
            }
            long wanted = ((long) n + SLICE_TARGET - 1) / SLICE_TARGET;
            int groups = (int) Math.max(1L, Math.min(MAX_GROUPS, wanted));
            return new TopKPartialPlan(n, k, groups, local);
        }

        public int n() {
            return n;
        }

        public int k() {
            return k;
        }

        /** Workgroups, one per vocabulary slice. */
        public int groups() {
            return groups;
        }

        /** Logits per slice, {@code ceil(n / groups)}; the last slices may be shorter or empty. */
        public int sliceLength() {
            return (int) (((long) n + groups - 1) / groups);
        }

        /** Candidate slots written: {@code groups * k} (the length of both candidate buffers). */
        public int candidateCount() {
            return groups * k;
        }

        /** {@code {groups * local}}. */
        public long[] globalWorkSize() {
            return new long[]{(long) groups * local};
        }

        /** {@code {local}}. */
        public long[] localWorkSize() {
            return new long[]{local};
        }

        /**
         * Whether the device can run this launch: the work-group size and the 1-D group count.
         *
         * @return null when it can, otherwise the reason
         */
        public String checkDevice(DeviceCapabilities caps) {
            Objects.requireNonNull(caps, "caps");
            long maxWg = caps.getMaxWorkGroupSize();
            if (maxWg > 0 && maxWg < local) {
                return TOPK_PARTIAL + " needs work-groups of " + local + " work-items; device "
                        + caps.getDeviceName() + " allows " + maxWg;
            }
            long maxGroups = caps.getMaxWorkGroupCount()[0];
            if (groups > maxGroups) {
                return TOPK_PARTIAL + " needs " + groups + " work-groups; device " + caps.getDeviceName()
                        + " allows " + maxGroups;
            }
            return null;
        }

        /**
         * Kernel arguments in declared order ({@link #TOPK_ARG_N} .. {@link #TOPK_ARG_CAND_IDX}), after
         * checking that the three buffers are distinct and cover {@code n} logits and
         * {@link #candidateCount()} candidate slots.
         *
         * @param invTemp inverse sampling temperature applied to the reported values; positive and finite
         * @throws IllegalArgumentException for a non-positive or non-finite {@code invTemp}, a shared
         *                                  buffer or a buffer smaller than the launch needs
         * @throws NullPointerException     for a null buffer
         */
        public Object[] args(float invTemp, GpuBuffer logits, GpuBuffer candVal, GpuBuffer candIdx) {
            if (!(invTemp > 0.0f) || Float.isInfinite(invTemp)) {
                throw new IllegalArgumentException("LLM_TopK_Partial inv_temp must be positive and finite, got " + invTemp);
            }
            Objects.requireNonNull(logits, "logits");
            Objects.requireNonNull(candVal, "candVal");
            Objects.requireNonNull(candIdx, "candIdx");
            if (logits == candVal || logits == candIdx || candVal == candIdx) {
                throw new IllegalArgumentException(TOPK_PARTIAL + ": logits, cand_val and cand_idx must be distinct buffers");
            }
            requireBytes("logits", logits, (long) n * Float.BYTES);
            requireBytes("cand_val", candVal, (long) candidateCount() * Float.BYTES);
            requireBytes("cand_idx", candIdx, (long) candidateCount() * Integer.BYTES);
            Object[] args = new Object[TOPK_ARG_COUNT];
            args[TOPK_ARG_N] = n;
            args[TOPK_ARG_K] = k;
            args[TOPK_ARG_INV_TEMP] = invTemp;
            args[TOPK_ARG_LOGITS] = logits;
            args[TOPK_ARG_CAND_VAL] = candVal;
            args[TOPK_ARG_CAND_IDX] = candIdx;
            return args;
        }

        private void requireBytes(String role, GpuBuffer buffer, long needed) {
            long size = buffer.sizeInBytes();
            if (size < needed) {
                throw new IllegalArgumentException(TOPK_PARTIAL + ": " + role + " buffer holds " + size
                        + " bytes but the launch (n=" + n + ", k=" + k + ", groups=" + groups + ") needs " + needed);
            }
        }

        @Override
        public String toString() {
            return "TopKPartialPlan{n=" + n + ", k=" + k + ", groups=" + groups + ", local=" + local + '}';
        }
    }
}
