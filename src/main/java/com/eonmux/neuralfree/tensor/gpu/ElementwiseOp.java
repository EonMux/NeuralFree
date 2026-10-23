package com.eonmux.neuralfree.tensor.gpu;

/**
 * The operations of the elementwise train kernels ({@code kernels/train/eltwise.cl}), each carrying the op id its
 * kernel selects by. The ids are the {@link TrainOps} constants, so the enums never define values of their own.
 */
public final class ElementwiseOp {

    private ElementwiseOp() {
    }

    /**
     * Ops of TR_Unary_F32. Parameters: {@code p0} is the slope of LEAKY_RELU, the alpha of ELU, the exponent of POW,
     * the lower bound of CLAMP and the scalar of ADD_SCALAR and MUL_SCALAR; {@code p1} is the upper bound of CLAMP.
     * Every other op ignores both.
     */
    public enum Unary {
        /** {@code -x}. */
        NEG(TrainOps.TR_E_NEG),
        /** {@code Math.abs(x)}. */
        ABS(TrainOps.TR_E_ABS),
        /** {@code (float) Math.sqrt(x)}. */
        SQRT(TrainOps.TR_E_SQRT),
        /** {@code (float) Math.exp(x)}. */
        EXP(TrainOps.TR_E_EXP),
        /** {@code (float) Math.log(x)}. */
        LOG(TrainOps.TR_E_LOG),
        /** {@code Math.max(0, x)}. */
        RELU(TrainOps.TR_E_RELU),
        /** {@code (float) (1 / (1 + Math.exp(-x)))}. */
        SIGMOID(TrainOps.TR_E_SIGMOID),
        /** {@code (float) Math.tanh(x)}. */
        TANH(TrainOps.TR_E_TANH),
        /** The tanh form of {@code TensorOps.gelu}. */
        GELU(TrainOps.TR_E_GELU),
        /** {@code x * sigmoid(x)}. */
        SILU(TrainOps.TR_E_SILU),
        /** {@code x > 0 ? x : p0 * x}. */
        LEAKY_RELU(TrainOps.TR_E_LEAKY_RELU),
        /** {@code x > 0 ? x : p0 * (exp(x) - 1)}. */
        ELU(TrainOps.TR_E_ELU),
        /** {@code (float) Math.log(1 + Math.exp(x))}. */
        SOFTPLUS(TrainOps.TR_E_SOFTPLUS),
        /** {@code x * tanh(softplus(x))}. */
        MISH(TrainOps.TR_E_MISH),
        /** {@code (float) Math.pow(x, p0)}, with every {@code Math.pow} special case. */
        POW(TrainOps.TR_E_POW),
        /** {@code Math.max(p0, Math.min(p1, x))}. */
        CLAMP(TrainOps.TR_E_CLAMP),
        /** {@code x + p0}. */
        ADD_SCALAR(TrainOps.TR_E_ADD_SCALAR),
        /** {@code x * p0}. */
        MUL_SCALAR(TrainOps.TR_E_MUL_SCALAR),
        /** {@code x}. */
        IDENTITY(TrainOps.TR_E_IDENTITY),
        /** {@code Math.round(x)} kept as a float ({@code floor(x)}, plus 1 when the fraction is at least 0.5). */
        ROUND(TrainOps.TR_E_ROUND),
        /** {@code (float) Math.floor(x)}. */
        FLOOR(TrainOps.TR_E_FLOOR),
        /** {@code (float) Math.sin(x)}. */
        SIN(TrainOps.TR_E_SIN),
        /** {@code (float) Math.cos(x)}. */
        COS(TrainOps.TR_E_COS);

        private final int id;

        Unary(int id) {
            this.id = id;
        }

        /** The kernel's op id. */
        public int id() {
            return id;
        }
    }

    /** Ops of TR_Binary_F32: {@code a op b} with Java float arithmetic. */
    public enum Binary {
        ADD(TrainOps.TR_E_ADD),
        SUB(TrainOps.TR_E_SUB),
        MUL(TrainOps.TR_E_MUL),
        DIV(TrainOps.TR_E_DIV);

        private final int id;

        Binary(int id) {
            this.id = id;
        }

        /** The kernel's op id. */
        public int id() {
            return id;
        }
    }

    /** Ops of TR_Compare_F32: Java float {@code ==}, {@code >} and {@code <} (NaN compares false, -0 equals +0). */
    public enum Compare {
        EQ(TrainOps.TR_CMP_EQ),
        GT(TrainOps.TR_CMP_GT),
        LT(TrainOps.TR_CMP_LT);

        private final int id;

        Compare(int id) {
            this.id = id;
        }

        /** The kernel's op id. */
        public int id() {
            return id;
        }
    }

    /**
     * Ops of TR_UnaryBackward_F32: {@code gx = gy * f'} in the operation order of the matching {@code GradFunctions}
     * class. The kernel selects the derivative by the forward {@link Unary} id of the same function. Each op reads
     * the forward input {@code x}, the forward output {@code y}, or both, as {@link #readsX()} and {@link #readsY()}
     * state; {@code p0} and {@code p1} are the forward parameters (LEAKY_RELU slope, ELU alpha, CLAMP bounds).
     */
    public enum Backward {
        /** {@code x > 0 ? gy : 0} (ReluGrad). */
        RELU(Unary.RELU, true, false),
        /** {@code gy * gelu'(x)} (GeluGrad). */
        GELU(Unary.GELU, true, false),
        /** {@code gy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))} (SiluGrad). */
        SILU(Unary.SILU, true, false),
        /** {@code gy * y * (1 - y)} (SigmoidGrad). */
        SIGMOID(Unary.SIGMOID, false, true),
        /** {@code gy * (1 - y * y)} (TanhGrad). */
        TANH(Unary.TANH, false, true),
        /** {@code x > 0 ? gy : p0 * gy} (LeakyReluGrad). */
        LEAKY_RELU(Unary.LEAKY_RELU, true, false),
        /** {@code x > 0 ? gy : gy * (y + p0)} (EluGrad). */
        ELU(Unary.ELU, true, true),
        /** {@code gy * sigmoid(x)} (SoftplusGrad). */
        SOFTPLUS(Unary.SOFTPLUS, true, false),
        /** {@code gy * mish'(x)} (MishGrad). */
        MISH(Unary.MISH, true, false),
        /**
         * {@code x > 0 ? gy : x < 0 ? -gy : +0} ({@code gy * sgn(x)}), so {@code +-0} and NaN {@code x} give
         * {@code +0} (AbsGrad).
         */
        ABS(Unary.ABS, true, false),
        /** {@code x < p0 || x > p1 ? 0 : gy} (ClampGrad). */
        CLAMP(Unary.CLAMP, true, false);

        private final Unary forward;
        private final boolean readsX;
        private final boolean readsY;

        Backward(Unary forward, boolean readsX, boolean readsY) {
            this.forward = forward;
            this.readsX = readsX;
            this.readsY = readsY;
        }

        /** The kernel's op id: the forward function's {@link Unary#id()}. */
        public int id() {
            return forward.id();
        }

        /** The forward function this derivative belongs to. */
        public Unary forward() {
            return forward;
        }

        /** Whether the derivative reads the forward input {@code x}. */
        public boolean readsX() {
            return readsX;
        }

        /** Whether the derivative reads the forward output {@code y}. */
        public boolean readsY() {
            return readsY;
        }
    }
}
