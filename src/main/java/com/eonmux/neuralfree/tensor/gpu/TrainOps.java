package com.eonmux.neuralfree.tensor.gpu;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Operation ids, kernel names and launch local sizes of the {@code train} kernel library.
 *
 * <p>Every {@code int} constant named {@code TR_*} or {@code ML_*} mirrors a {@code #define} of the same name and
 * value in the kernel family's {@code .cl} source under {@code kernels/train} and in its GLSL twin under
 * {@code shaders/train}; the kernels select their operation by these values. Kernel names live in
 * {@link Kernels}.</p>
 */
public final class TrainOps {

    // ----------------------------------------------------------------- eltwise.cl: TR_Unary_F32 op
    public static final int TR_E_NEG = 0;
    public static final int TR_E_ABS = 1;
    public static final int TR_E_SQRT = 2;
    public static final int TR_E_EXP = 3;
    public static final int TR_E_LOG = 4;
    public static final int TR_E_RELU = 5;
    public static final int TR_E_SIGMOID = 6;
    public static final int TR_E_TANH = 7;
    public static final int TR_E_GELU = 8;
    public static final int TR_E_SILU = 9;
    public static final int TR_E_LEAKY_RELU = 10;
    public static final int TR_E_ELU = 11;
    public static final int TR_E_SOFTPLUS = 12;
    public static final int TR_E_MISH = 13;
    public static final int TR_E_POW = 14;
    public static final int TR_E_CLAMP = 15;
    public static final int TR_E_ADD_SCALAR = 16;
    public static final int TR_E_MUL_SCALAR = 17;
    public static final int TR_E_IDENTITY = 18;
    public static final int TR_E_ROUND = 19;
    public static final int TR_E_FLOOR = 20;
    public static final int TR_E_SIN = 21;
    public static final int TR_E_COS = 22;

    // ----------------------------------------------------------------- eltwise.cl: TR_Binary_F32 op
    public static final int TR_E_ADD = 0;
    public static final int TR_E_SUB = 1;
    public static final int TR_E_MUL = 2;
    public static final int TR_E_DIV = 3;

    // TR_UnaryBackward_F32 selects its op with the forward TR_E_* id of the same activation: RELU, GELU, SILU,
    // SIGMOID, TANH, LEAKY_RELU, ELU, SOFTPLUS, MISH, ABS and CLAMP.

    // ----------------------------------------------------------------- eltwise.cl: TR_Compare_F32 op
    public static final int TR_CMP_EQ = 0;
    public static final int TR_CMP_GT = 1;
    public static final int TR_CMP_LT = 2;

    // ----------------------------------------------------------------- eltwise.cl: TR_Where_F32 cond_kind
    /** FLOAT32 condition, selected where {@code cond != 0.0f}; c_off in elements. */
    public static final int TR_WHERE_COND_F32 = 0;
    /** BOOL condition read as bytes of 32-bit words; c_off in bytes. */
    public static final int TR_WHERE_COND_BOOL = 1;

    // ----------------------------------------------------------------- eltwise.cl: TR_Scatter_F32 mode
    public static final int TR_SCATTER_OVERWRITE = 0;
    public static final int TR_SCATTER_ADD = 1;

    // ----------------------------------------------------------------- reduce.cl: TR_ReduceRows*_F32 op
    public static final int TR_R_SUM = 0;
    public static final int TR_R_MAX = 1;
    public static final int TR_R_MIN = 2;
    public static final int TR_R_SUMSQ = 3;
    public static final int TR_R_SQDEV = 4;

    // ----------------------------------------------------------------- reduce.cl: TR_ArgReduceRows_F32 op
    public static final int TR_R_ARGMAX = 0;
    public static final int TR_R_ARGMIN = 1;

    // ----------------------------------------------------------------- softmax.cl: TR_SoftmaxRows_F32 flags
    /** Bit 0: log-softmax. */
    public static final int TR_SM_LOG = 1;
    /** Bit 1: masked softmax, {@code -inf} inputs give exactly 0 and an all-{@code -inf} row gives all 0. */
    public static final int TR_SM_MASKED = 2;

    // ----------------------------------------------------------------- segment.cl: TR_SegmentReduce(Backward)_F32 op
    public static final int TR_SEG_SUM = 0;
    public static final int TR_SEG_MEAN = 1;
    public static final int TR_SEG_MAX = 2;

    // ----------------------------------------------------------------- norm.cl: TR_IndexSelectRows/AddRows oob_mode
    /**
     * An out-of-range index sets the error slot and writes a zero row (TR_IndexSelectRows_F32), and contributes
     * nothing to the gradient (TR_IndexAddRows_F32).
     */
    public static final int TR_IDX_OOB_ERROR = 0;
    /** An out-of-range index is clamped to {@code [0, rows - 1]}. */
    public static final int TR_IDX_OOB_CLAMP = 1;

    // ----------------------------------------------------------------- norm.cl: TR_NormAffineGradCols_F32 mode
    /**
     * LayerNorm affine gradients from the {@code (mean, sdev)} row stats of TR_LayerNormRowsBackward_F32:
     * {@code sum(gy * xhat)} per column, plus {@code sum(gy)} when has_beta.
     */
    public static final int TR_N_AFF_LN = 0;
    /**
     * RMSNorm weight gradient from the {@code rms} row stats of TR_RMSNormRowsBackward_F32:
     * {@code sum(gy * x / rms)} per column.
     */
    public static final int TR_N_AFF_RMS = 1;

    // ----------------------------------------------------------------- norm.cl: TR_ChannelNormBackwardApply_F32 mode
    /** BatchNorm in training mode: per-channel batch statistics and the TR_ChannelGradSums_F32 sums. */
    public static final int TR_N_CB_BATCH = 0;
    /** GroupNorm: per-(batch, group) statistics and the TR_GroupGradSums_F32 sums. */
    public static final int TR_N_CB_GROUP = 1;
    /**
     * BatchNorm in evaluation mode: the running statistics and no sums, {@code gx = d / sqrt(var + eps)} with
     * {@code d = gy * gamma} when has_affine, else {@code d = gy}.
     */
    public static final int TR_N_CB_EVAL = 2;

    // ----------------------------------------------------------------- optim.cl: TR_AdamStep_F32 mode
    public static final int TR_ADAM_MODE_ADAM = 0;
    public static final int TR_ADAM_MODE_ADAMW = 1;

    // ----------------------------------------------------------------- dense.cl: act (ActivationFunctionType ordinals)
    public static final int TR_DENSE_ACT_SIGMOID = 0;
    public static final int TR_DENSE_ACT_RELU = 1;
    public static final int TR_DENSE_ACT_TANH = 2;
    public static final int TR_DENSE_ACT_LEAKY_RELU = 3;
    public static final int TR_DENSE_ACT_ELU = 4;
    public static final int TR_DENSE_ACT_SOFTPLUS = 5;
    public static final int TR_DENSE_ACT_SOFTSIGN = 6;
    public static final int TR_DENSE_ACT_ISRU = 7;
    public static final int TR_DENSE_ACT_GAUSSIAN = 8;
    public static final int TR_DENSE_ACT_SINC = 9;
    public static final int TR_DENSE_ACT_SWISH = 10;
    public static final int TR_DENSE_ACT_SELU = 11;
    public static final int TR_DENSE_ACT_HARD_SIGMOID = 12;
    public static final int TR_DENSE_ACT_BENT_IDENTITY = 13;
    public static final int TR_DENSE_ACT_PRELU = 14;
    public static final int TR_DENSE_ACT_SRELU = 15;
    public static final int TR_DENSE_ACT_SOFTEXPONENTIAL = 16;
    public static final int TR_DENSE_ACT_IDENTITY = 17;

    // ----------------------------------------------------------------- ml/kernel_matrix.cl: type (enum ordinals)
    /** {@code SVM.KernelType} ordinals. */
    public static final int ML_SVM_LINEAR = 0;
    public static final int ML_SVM_RBF = 1;
    public static final int ML_SVM_POLY = 2;
    /** {@code GaussianProcess.KernelType} ordinals. */
    public static final int ML_GP_RBF = 0;
    public static final int ML_GP_MATERN32 = 1;
    public static final int ML_GP_MATERN52 = 2;
    public static final int ML_GP_LINEAR = 3;

    /**
     * Rows and columns of the output tile one TR_Gemm_F32 work-group computes ({@code TR_GM_TILE} in gemm.cl and
     * gemm.comp): a launch over an {@code M x N} product needs {@code ceil(N / GEMM_TILE)} by
     * {@code ceil(M / GEMM_TILE)} work-groups of {@link #localSize} {@code (Kernels.GEMM_F32)}.
     */
    public static final int GEMM_TILE = 64;

    /** Local size of TR_Gemm_F32, the only multi-dimensional train launch ({@code TR_GM_LX} square). */
    private static final long[] GEMM_LOCAL = {16, 16, 1};

    private TrainOps() {
    }

    /**
     * The work-group size a kernel is launched and prewarmed with: {16, 16, 1} for {@link Kernels#GEMM_F32}, and
     * {@link TrainGpu#LINEAR_LOCAL} for every other kernel (all of them are 1-D). A fresh array.
     */
    public static long[] localSize(String kernel) {
        if (kernel == null || kernel.isEmpty()) {
            throw new IllegalArgumentException("kernel name must not be null or empty");
        }
        return Kernels.GEMM_F32.equals(kernel) ? GEMM_LOCAL.clone() : new long[]{TrainGpu.LINEAR_LOCAL};
    }

    /** Names of the train library's kernels, as listed in {@code kernels/train/inventory.txt}. */
    public static final class Kernels {
        // core.cl
        public static final String FILL_F32 = "TR_Fill_F32";
        public static final String COPY_F32 = "TR_Copy_F32";
        public static final String COPY_STRIDED_F32 = "TR_Copy_Strided_F32";
        public static final String CAST_I32_F32 = "TR_Cast_I32_F32";
        public static final String CAST_F32_I32 = "TR_Cast_F32_I32";
        public static final String CAST_BOOL_F32 = "TR_Cast_BOOL_F32";
        public static final String CAST_F32_BOOL = "TR_Cast_F32_BOOL";
        public static final String CAST_F32_F16 = "TR_Cast_F32_F16";
        public static final String CAST_F16_F32 = "TR_Cast_F16_F32";
        public static final String CAST_F32_BF16 = "TR_Cast_F32_BF16";
        public static final String CAST_BF16_F32 = "TR_Cast_BF16_F32";
        public static final String CAST_I64_I32 = "TR_Cast_I64_I32";
        // eltwise.cl
        public static final String UNARY_F32 = "TR_Unary_F32";
        public static final String BINARY_F32 = "TR_Binary_F32";
        public static final String AXPBY_F32 = "TR_Axpby_F32";
        public static final String SCALE_F32 = "TR_Scale_F32";
        public static final String COMPARE_F32 = "TR_Compare_F32";
        public static final String WHERE_F32 = "TR_Where_F32";
        public static final String GATHER_F32 = "TR_Gather_F32";
        public static final String SCATTER_F32 = "TR_Scatter_F32";
        public static final String UNARY_BACKWARD_F32 = "TR_UnaryBackward_F32";
        public static final String MASK_SCALE_F32 = "TR_MaskScale_F32";
        // reduce.cl
        public static final String REDUCE_ROWS_F32 = "TR_ReduceRows_F32";
        public static final String REDUCE_ROWS_SMALL_F32 = "TR_ReduceRowsSmall_F32";
        public static final String REDUCE_ROWS_PARTIAL_F32 = "TR_ReduceRowsPartial_F32";
        public static final String REDUCE_COLS_F32 = "TR_ReduceCols_F32";
        public static final String ARG_REDUCE_ROWS_F32 = "TR_ArgReduceRows_F32";
        public static final String MAX_BACKWARD_ROWS_F32 = "TR_MaxBackwardRows_F32";
        // softmax.cl
        public static final String SOFTMAX_ROWS_F32 = "TR_SoftmaxRows_F32";
        public static final String SOFTMAX_ROWS_BACKWARD_F32 = "TR_SoftmaxRowsBackward_F32";
        // segment.cl
        public static final String SEGMENT_REDUCE_F32 = "TR_SegmentReduce_F32";
        public static final String SEGMENT_REDUCE_BACKWARD_F32 = "TR_SegmentReduceBackward_F32";
        // gemm.cl
        public static final String GEMM_F32 = "TR_Gemm_F32";
        public static final String GEMV_F32 = "TR_Gemv_F32";
        // conv.cl
        public static final String CONV2D_F32 = "TR_Conv2d_F32";
        public static final String CONV_TRANSPOSE2D_F32 = "TR_ConvTranspose2d_F32";
        public static final String CONV2D_WEIGHT_GRAD_F32 = "TR_Conv2dWeightGrad_F32";
        public static final String IM2COL_F32 = "TR_Im2col_F32";
        public static final String COL2IM_F32 = "TR_Col2im_F32";
        // pool.cl
        public static final String MAX_POOL2D_F32 = "TR_MaxPool2d_F32";
        public static final String AVG_POOL2D_F32 = "TR_AvgPool2d_F32";
        public static final String ADAPTIVE_AVG_POOL2D_F32 = "TR_AdaptiveAvgPool2d_F32";
        public static final String MAX_POOL2D_BACKWARD_F32 = "TR_MaxPool2dBackward_F32";
        public static final String AVG_POOL2D_BACKWARD_F32 = "TR_AvgPool2dBackward_F32";
        public static final String ADAPTIVE_AVG_POOL2D_BACKWARD_F32 = "TR_AdaptiveAvgPool2dBackward_F32";
        public static final String MAX_POOL2D_SELECT_F32 = "TR_MaxPool2dSelect_F32";
        // norm.cl
        public static final String LAYER_NORM_ROWS_F32 = "TR_LayerNormRows_F32";
        public static final String RMS_NORM_ROWS_F32 = "TR_RMSNormRows_F32";
        public static final String CHANNEL_STATS_F32 = "TR_ChannelStats_F32";
        public static final String CHANNEL_NORM_APPLY_F32 = "TR_ChannelNormApply_F32";
        public static final String INDEX_SELECT_ROWS_F32 = "TR_IndexSelectRows_F32";
        public static final String LAYER_NORM_ROWS_BACKWARD_F32 = "TR_LayerNormRowsBackward_F32";
        public static final String RMS_NORM_ROWS_BACKWARD_F32 = "TR_RMSNormRowsBackward_F32";
        public static final String NORM_AFFINE_GRAD_COLS_F32 = "TR_NormAffineGradCols_F32";
        public static final String CHANNEL_GRAD_SUMS_F32 = "TR_ChannelGradSums_F32";
        public static final String GROUP_GRAD_SUMS_F32 = "TR_GroupGradSums_F32";
        public static final String CHANNEL_NORM_BACKWARD_APPLY_F32 = "TR_ChannelNormBackwardApply_F32";
        public static final String INDEX_ADD_ROWS_F32 = "TR_IndexAddRows_F32";
        // optim.cl
        public static final String SGD_STEP_F32 = "TR_SgdStep_F32";
        public static final String ADAM_STEP_F32 = "TR_AdamStep_F32";
        public static final String RMSPROP_STEP_F32 = "TR_RmspropStep_F32";
        public static final String ADAGRAD_STEP_F32 = "TR_AdagradStep_F32";
        public static final String LAMB_UPDATE_F32 = "TR_LambUpdate_F32";
        public static final String LAMB_APPLY_F32 = "TR_LambApply_F32";
        public static final String GRAD_NORM_COEF_F32 = "TR_GradNormCoef_F32";
        public static final String SCALE_BY_DEVICE_F32 = "TR_ScaleByDevice_F32";
        public static final String UNSCALE_CHECK_F32 = "TR_UnscaleCheck_F32";
        // dense.cl
        public static final String DENSE_BIAS_ACT_F32 = "TR_DenseBiasAct_F32";
        public static final String DENSE_ACT_BACKWARD_F32 = "TR_DenseActBackward_F32";
        // ml/kmeans.cl
        public static final String ML_KMEANS_DISTANCES = "ML_KMeans_Distances";
        public static final String ML_KMEANS_ASSIGN = "ML_KMeans_Assign";
        public static final String ML_KMEANS_UPDATE_PARTIAL = "ML_KMeans_UpdatePartial";
        public static final String ML_KMEANS_UPDATE_FINALIZE = "ML_KMeans_UpdateFinalize";
        public static final String ML_KMEANS_SHIFT = "ML_KMeans_Shift";
        // ml/kernel_matrix.cl
        public static final String ML_SVM_KERNEL = "ML_SVM_Kernel";
        public static final String ML_GP_KERNEL = "ML_GP_Kernel";
        // ml/linear.cl
        public static final String ML_PERCEPTRON_FORWARD = "ML_Perceptron_Forward";
        public static final String ML_PERCEPTRON_GRAD = "ML_Perceptron_Grad";
        public static final String ML_PERCEPTRON_UPDATE = "ML_Perceptron_Update";
        public static final String ML_LOGREG_FORWARD = "ML_LogReg_Forward";
        public static final String ML_LOGREG_GRAD = "ML_LogReg_Grad";
        public static final String ML_LOGREG_UPDATE = "ML_LogReg_Update";
        // ml/forecast.cl
        public static final String ML_FORECAST_FORWARD = "ML_Forecast_Forward";
        public static final String ML_FORECAST_GRAD = "ML_Forecast_Grad";
        public static final String ML_FORECAST_UPDATE = "ML_Forecast_Update";

        /** Every kernel name above, in declaration order (85 kernels). Unmodifiable. */
        public static final Set<String> ALL = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
                FILL_F32, COPY_F32, COPY_STRIDED_F32, CAST_I32_F32, CAST_F32_I32, CAST_BOOL_F32, CAST_F32_BOOL,
                CAST_F32_F16, CAST_F16_F32, CAST_F32_BF16, CAST_BF16_F32, CAST_I64_I32,
                UNARY_F32, BINARY_F32, AXPBY_F32, SCALE_F32, COMPARE_F32, WHERE_F32, GATHER_F32, SCATTER_F32,
                UNARY_BACKWARD_F32, MASK_SCALE_F32,
                REDUCE_ROWS_F32, REDUCE_ROWS_SMALL_F32, REDUCE_ROWS_PARTIAL_F32, REDUCE_COLS_F32,
                ARG_REDUCE_ROWS_F32, MAX_BACKWARD_ROWS_F32,
                SOFTMAX_ROWS_F32, SOFTMAX_ROWS_BACKWARD_F32, SEGMENT_REDUCE_F32, SEGMENT_REDUCE_BACKWARD_F32,
                GEMM_F32, GEMV_F32, CONV2D_F32, CONV_TRANSPOSE2D_F32, CONV2D_WEIGHT_GRAD_F32, IM2COL_F32, COL2IM_F32,
                MAX_POOL2D_F32, AVG_POOL2D_F32, ADAPTIVE_AVG_POOL2D_F32,
                MAX_POOL2D_BACKWARD_F32, AVG_POOL2D_BACKWARD_F32, ADAPTIVE_AVG_POOL2D_BACKWARD_F32,
                MAX_POOL2D_SELECT_F32,
                LAYER_NORM_ROWS_F32, RMS_NORM_ROWS_F32, CHANNEL_STATS_F32, CHANNEL_NORM_APPLY_F32,
                INDEX_SELECT_ROWS_F32,
                LAYER_NORM_ROWS_BACKWARD_F32, RMS_NORM_ROWS_BACKWARD_F32, NORM_AFFINE_GRAD_COLS_F32,
                CHANNEL_GRAD_SUMS_F32, GROUP_GRAD_SUMS_F32, CHANNEL_NORM_BACKWARD_APPLY_F32, INDEX_ADD_ROWS_F32,
                SGD_STEP_F32, ADAM_STEP_F32, RMSPROP_STEP_F32, ADAGRAD_STEP_F32, LAMB_UPDATE_F32, LAMB_APPLY_F32,
                GRAD_NORM_COEF_F32, SCALE_BY_DEVICE_F32, UNSCALE_CHECK_F32,
                DENSE_BIAS_ACT_F32, DENSE_ACT_BACKWARD_F32,
                ML_KMEANS_DISTANCES, ML_KMEANS_ASSIGN, ML_KMEANS_UPDATE_PARTIAL, ML_KMEANS_UPDATE_FINALIZE,
                ML_KMEANS_SHIFT, ML_SVM_KERNEL, ML_GP_KERNEL,
                ML_PERCEPTRON_FORWARD, ML_PERCEPTRON_GRAD, ML_PERCEPTRON_UPDATE,
                ML_LOGREG_FORWARD, ML_LOGREG_GRAD, ML_LOGREG_UPDATE,
                ML_FORECAST_FORWARD, ML_FORECAST_GRAD, ML_FORECAST_UPDATE)));

        private Kernels() {
        }
    }
}
