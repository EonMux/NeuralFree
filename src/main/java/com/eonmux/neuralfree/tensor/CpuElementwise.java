package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * CPU reference of {@link TensorOps}' elementwise ops: broadcasting arithmetic, unary functions and activations,
 * compares, {@code where}, GLU/GeGLU, activation backwards, the dropout mask-scale and the in-place updates. Every
 * formula is Java float arithmetic in the order the train kernels mirror ({@code kernels/train/eltwise.cl}).
 * Results never require grad, whatever the inputs' flags.
 */
final class CpuElementwise {

    /** {@code (float) Math.sqrt(2 / PI)}, the GELU constant. */
    private static final float SQRT_TWO_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);

    private CpuElementwise() {
    }

    @FunctionalInterface
    interface FloatUnary {
        float apply(float x);
    }

    @FunctionalInterface
    private interface FloatBinary {
        float apply(float a, float b);
    }

    @FunctionalInterface
    private interface FloatCompare {
        boolean test(float a, float b);
    }

    // ------------------------------------------------------------------ binary (broadcasting)

    static Tensor add(Tensor a, Tensor b) {
        return binary(a, b, (x, y) -> x + y);
    }

    static Tensor sub(Tensor a, Tensor b) {
        return binary(a, b, (x, y) -> x - y);
    }

    static Tensor mul(Tensor a, Tensor b) {
        return binary(a, b, (x, y) -> x * y);
    }

    static Tensor div(Tensor a, Tensor b) {
        return binary(a, b, (x, y) -> x / y);
    }

    private static Tensor binary(Tensor a, Tensor b, FloatBinary op) {
        BroadcastEngine.BroadcastInfo broadcast = BroadcastEngine.computeBroadcast(a.shape(), b.shape());
        Shape resultShape = broadcast.resultShape();
        float[] dataA = a.toFloatArray();
        float[] dataB = b.toFloatArray();
        float[] result = new float[CpuShapeOps.hostLength(resultShape.numElements())];
        BroadcastWalk walk = new BroadcastWalk(broadcast);
        for (int i = 0; i < result.length; i++) {
            walk.locate(i);
            result[i] = op.apply(dataA[walk.indexA], dataB[walk.indexB]);
        }
        return CpuShapeOps.floats(a.backend(), resultShape, result, false);
    }

    // ------------------------------------------------------------------ unary

    static Tensor neg(Tensor a) {
        return map(a, x -> -x);
    }

    static Tensor abs(Tensor a) {
        return map(a, Math::abs);
    }

    static Tensor sqrt(Tensor a) {
        return map(a, x -> (float) Math.sqrt(x));
    }

    static Tensor exp(Tensor a) {
        return map(a, x -> (float) Math.exp(x));
    }

    static Tensor log(Tensor a) {
        return map(a, x -> (float) Math.log(x));
    }

    static Tensor pow(Tensor a, float exponent) {
        return map(a, x -> (float) Math.pow(x, exponent));
    }

    static Tensor clamp(Tensor a, float min, float max) {
        return map(a, x -> Math.max(min, Math.min(max, x)));
    }

    static Tensor addScalar(Tensor a, float scalar) {
        return map(a, x -> x + scalar);
    }

    static Tensor mulScalar(Tensor a, float scalar) {
        return map(a, x -> x * scalar);
    }

    static Tensor relu(Tensor a) {
        return map(a, x -> Math.max(0.0f, x));
    }

    static Tensor gelu(Tensor a) {
        return map(a, CpuElementwise::geluOf);
    }

    static Tensor silu(Tensor a) {
        return map(a, x -> x * sigmoidOf(x));
    }

    static Tensor sigmoid(Tensor a) {
        return map(a, CpuElementwise::sigmoidOf);
    }

    static Tensor tanh(Tensor a) {
        return map(a, x -> (float) Math.tanh(x));
    }

    static Tensor sin(Tensor a) {
        return map(a, x -> (float) Math.sin(x));
    }

    static Tensor cos(Tensor a) {
        return map(a, x -> (float) Math.cos(x));
    }

    static Tensor leakyRelu(Tensor a, float negativeSlope) {
        return map(a, x -> x > 0 ? x : negativeSlope * x);
    }

    static Tensor elu(Tensor a, float alpha) {
        return map(a, x -> x > 0 ? x : alpha * ((float) Math.exp(x) - 1));
    }

    static Tensor softplus(Tensor a) {
        return map(a, CpuElementwise::softplusOf);
    }

    static Tensor mish(Tensor a) {
        return map(a, x -> x * (float) Math.tanh(softplusOf(x)));
    }

    /**
     * {@code Math.round(x)} kept as a float, with the kernels' formula: {@code r = floor(x)}, plus 1 when
     * {@code x - r >= 0.5}. Exact for every float; {@code -0}, NaN and the infinities stay as they are, so a later
     * cast to INT32 gives Java's {@code int} result.
     */
    static Tensor round(Tensor a) {
        return map(a, x -> {
            float r = (float) Math.floor(x);
            return x - r >= 0.5f ? r + 1.0f : r;
        });
    }

    static Tensor floor(Tensor a) {
        return map(a, x -> (float) Math.floor(x));
    }

    /** {@code f} applied to every element of {@code a} in row-major order, as a new contiguous tensor. */
    static Tensor map(Tensor a, FloatUnary f) {
        float[] data = a.toFloatArray();
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = f.apply(data[i]);
        }
        return CpuShapeOps.floats(a.backend(), a.shape(), result, false);
    }

    private static float sigmoidOf(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private static float softplusOf(float x) {
        return (float) Math.log(1.0 + Math.exp(x));
    }

    private static float geluOf(float x) {
        float inner = SQRT_TWO_OVER_PI * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + (float) Math.tanh(inner));
    }

    // ------------------------------------------------------------------ compare and select

    static Tensor eq(Tensor a, Tensor b) {
        return compare(a, b, (x, y) -> x == y);
    }

    static Tensor gt(Tensor a, Tensor b) {
        return compare(a, b, (x, y) -> x > y);
    }

    static Tensor lt(Tensor a, Tensor b) {
        return compare(a, b, (x, y) -> x < y);
    }

    private static Tensor compare(Tensor a, Tensor b, FloatCompare op) {
        BroadcastEngine.BroadcastInfo broadcast = BroadcastEngine.computeBroadcast(a.shape(), b.shape());
        Shape resultShape = broadcast.resultShape();
        float[] dataA = a.toFloatArray();
        float[] dataB = b.toFloatArray();
        byte[] result = new byte[CpuShapeOps.hostLength(resultShape.numElements())];
        BroadcastWalk walk = new BroadcastWalk(broadcast);
        for (int i = 0; i < result.length; i++) {
            walk.locate(i);
            result[i] = op.test(dataA[walk.indexA], dataB[walk.indexB]) ? (byte) 1 : (byte) 0;
        }
        return CpuShapeOps.bytes(a.backend(), resultShape, DType.BOOL, result);
    }

    /**
     * {@code cond != 0 ? a : b} per element: a FLOAT32 condition selects {@code a} when {@code != 0.0f} (NaN
     * selects {@code a}, +-0 selects {@code b}), a BOOL condition when its byte is non-zero.
     *
     * @throws IllegalArgumentException      when the shapes differ
     * @throws UnsupportedOperationException when the condition is neither FLOAT32 nor BOOL
     */
    static Tensor where(Tensor condition, Tensor a, Tensor b) {
        Shape condShape = condition.shape();
        if (!condShape.equals(a.shape()) || !condShape.equals(b.shape())) {
            throw new IllegalArgumentException("All tensors must have the same shape for where");
        }
        boolean[] select = conditionValues(condition);
        float[] aData = a.toFloatArray();
        float[] bData = b.toFloatArray();
        float[] result = new float[select.length];
        for (int i = 0; i < select.length; i++) {
            result[i] = select[i] ? aData[i] : bData[i];
        }
        return CpuShapeOps.floats(condition.backend(), condShape, result, false);
    }

    private static boolean[] conditionValues(Tensor condition) {
        boolean[] select = new boolean[CpuShapeOps.hostLength(condition.numElements())];
        if (condition.dtype() == DType.FLOAT32) {
            float[] condData = condition.toFloatArray();
            for (int i = 0; i < select.length; i++) {
                select[i] = condData[i] != 0.0f;
            }
        } else if (condition.dtype() == DType.BOOL) {
            ByteBuffer condData = condition.toByteBuffer();
            for (int i = 0; i < select.length; i++) {
                select[i] = condData.get(i) != 0;
            }
        } else {
            throw new UnsupportedOperationException("where() condition must be FLOAT32 or BOOL, got: "
                    + condition.dtype());
        }
        return select;
    }

    // ------------------------------------------------------------------ gated linear units

    /**
     * The GLU/GeGLU output shape: {@code shape} with {@code dim} (normalized) halved.
     *
     * @throws IllegalArgumentException  when the size of {@code dim} is odd ({@code op} names the unit)
     * @throws IndexOutOfBoundsException when {@code dim} is outside the rank
     */
    static Shape gatedShape(Shape shape, int dim, String op) {
        int dimSize = shape.dim(dim);
        if (dimSize % 2 != 0) {
            throw new IllegalArgumentException(op + " requires dimension size to be even, got " + dimSize);
        }
        int[] outDims = shape.dims();
        outDims[dim] = dimSize / 2;
        return Shape.of(outDims);
    }

    /** {@code first * sigmoid(second)} of the two halves of {@code dim} (normalized, even size). */
    static Tensor glu(Tensor a, int dim) {
        return gated(a, dim, gatedShape(a.shape(), dim, "GLU"), CpuElementwise::sigmoidOf);
    }

    /** {@code first * gelu(second)} of the two halves of {@code dim} (normalized, even size). */
    static Tensor geglu(Tensor a, int dim) {
        return gated(a, dim, gatedShape(a.shape(), dim, "GeGLU"), CpuElementwise::geluOf);
    }

    private static Tensor gated(Tensor a, int dim, Shape outShape, FloatUnary gate) {
        float[] data = a.toFloatArray();
        Shape shape = a.shape();
        int dimSize = shape.dim(dim);
        int halfDim = dimSize / 2;
        int outerSize = CpuShapeOps.product(shape, 0, dim);
        int innerSize = CpuShapeOps.product(shape, dim + 1, shape.rank());
        float[] result = new float[CpuShapeOps.hostLength(outShape.numElements())];
        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                for (int d = 0; d < halfDim; d++) {
                    int aIdx = outer * dimSize * innerSize + d * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (d + halfDim) * innerSize + inner;
                    int outIdx = outer * halfDim * innerSize + d * innerSize + inner;
                    result[outIdx] = data[aIdx] * gate.apply(data[bIdx]);
                }
            }
        }
        return CpuShapeOps.floats(a.backend(), outShape, result, false);
    }

    // ------------------------------------------------------------------ activation backward

    /**
     * {@code gx = gy * f'} with the {@code GradFunctions} formulas (ReluGrad through MishGrad, AbsGrad, ClampGrad) in
     * their float operation order; {@code x} and {@code y} are read only when {@code op} uses them.
     *
     * @throws NullPointerException     when a tensor the op reads is null
     * @throws IllegalArgumentException when the backends differ or a used shape differs from {@code gy}'s
     */
    static Tensor activationBackward(ElementwiseOp.Backward op, Tensor x, Tensor y, Tensor gy, float p0, float p1) {
        float[] xs = op.readsX() ? backwardOperand(Objects.requireNonNull(x, "x must not be null for " + op), gy)
                : null;
        float[] ys = op.readsY() ? backwardOperand(Objects.requireNonNull(y, "y must not be null for " + op), gy)
                : null;
        float[] g = gy.toFloatArray();
        float[] result = new float[g.length];
        for (int i = 0; i < g.length; i++) {
            result[i] = backward(op, xs == null ? 0.0f : xs[i], ys == null ? 0.0f : ys[i], g[i], p0, p1);
        }
        return CpuShapeOps.floats(gy.backend(), gy.shape(), result, false);
    }

    private static float[] backwardOperand(Tensor t, Tensor gy) {
        CpuShapeOps.requireSameBackend(t, gy);
        if (!t.shape().equals(gy.shape())) {
            throw new IllegalArgumentException("Saved tensor shape " + t.shape() + " does not match gradient shape "
                    + gy.shape());
        }
        return t.toFloatArray();
    }

    private static float backward(ElementwiseOp.Backward op, float x, float y, float g, float p0, float p1) {
        switch (op) {
            case RELU:
                return x > 0 ? g : 0.0f;
            case GELU: {
                float x3 = x * x * x;
                float inner = SQRT_TWO_OVER_PI * (x + 0.044715f * x3);
                float tanhInner = (float) Math.tanh(inner);
                float sech2 = 1.0f - tanhInner * tanhInner;
                float dInner = SQRT_TWO_OVER_PI * (1.0f + 3.0f * 0.044715f * x * x);
                return g * (0.5f * (1.0f + tanhInner) + 0.5f * x * sech2 * dInner);
            }
            case SILU: {
                float sigmoid = sigmoidOf(x);
                return g * (sigmoid * (1.0f + x * (1.0f - sigmoid)));
            }
            case SIGMOID:
                return g * y * (1.0f - y);
            case TANH:
                return g * (1.0f - y * y);
            case LEAKY_RELU:
                return x > 0 ? g : p0 * g;
            case ELU:
                return x > 0 ? g : g * (y + p0);
            case SOFTPLUS:
                return g * sigmoidOf(x);
            case MISH: {
                float tanhSp = (float) Math.tanh(softplusOf(x));
                float sigmoid = sigmoidOf(x);
                float sech2 = 1.0f - tanhSp * tanhSp;
                return g * (tanhSp + x * sech2 * sigmoid);
            }
            case ABS:
                return x > 0 ? g : (x < 0 ? -g : 0.0f);
            case CLAMP:
                return x < p0 || x > p1 ? 0.0f : g;
            default:
                throw new IllegalArgumentException("no CPU derivative for " + op);
        }
    }

    // ------------------------------------------------------------------ dropout

    /**
     * {@code m == +-0 ? +0 : (x * m) * scale} per element in row-major order, the formula and evaluation order of
     * {@code TR_MaskScale_F32} ({@code kernels/train/eltwise.cl}). Shapes and backends are checked by the caller.
     */
    static Tensor maskScale(Tensor x, Tensor mask, float scale) {
        float[] xs = x.toFloatArray();
        float[] ms = mask.toFloatArray();
        float[] result = new float[xs.length];
        for (int i = 0; i < xs.length; i++) {
            result[i] = ms[i] == 0.0f ? 0.0f : (xs[i] * ms[i]) * scale;
        }
        return CpuShapeOps.floats(x.backend(), x.shape(), result, false);
    }

    // ------------------------------------------------------------------ in place

    /** {@code dst += alpha * src} element by element, written back into the elements {@code dst} addresses. */
    static void addInPlace(Tensor dst, Tensor src, float alpha) {
        float[] dstData = dst.toFloatArray();
        float[] srcData = src.toFloatArray();
        for (int i = 0; i < dstData.length; i++) {
            dstData[i] += alpha * srcData[i];
        }
        GpuOptimizerOps.storeFloats(dst, dstData);
    }

    /** {@code dst *= scalar} element by element, written back into the elements {@code dst} addresses. */
    static void mulInPlace(Tensor dst, float scalar) {
        float[] data = dst.toFloatArray();
        for (int i = 0; i < data.length; i++) {
            data[i] *= scalar;
        }
        GpuOptimizerOps.storeFloats(dst, data);
    }

    /** Writes {@code +0.0f} into every element {@code dst} addresses. */
    static void zeroInPlace(Tensor dst) {
        GpuOptimizerOps.storeFloats(dst, new float[CpuShapeOps.hostLength(dst.numElements())]);
    }

    /**
     * Row-major walk of a broadcast result: {@link #locate} sets the element indices of both operands for one
     * result element (the operands' row-major positions under their broadcast strides).
     */
    private static final class BroadcastWalk {
        private final int[] dims;
        private final long[] stridesA;
        private final long[] stridesB;
        private final int[] indices;
        int indexA;
        int indexB;

        BroadcastWalk(BroadcastEngine.BroadcastInfo broadcast) {
            this.dims = broadcast.resultShape().dims();
            this.stridesA = broadcast.stridesA();
            this.stridesB = broadcast.stridesB();
            this.indices = new int[dims.length];
        }

        void locate(long i) {
            long linearIdx = i;
            for (int j = dims.length - 1; j >= 0; j--) {
                indices[j] = (int) (linearIdx % dims[j]);
                linearIdx /= dims[j];
            }
            indexA = (int) offset(stridesA);
            indexB = (int) offset(stridesB);
        }

        private long offset(long[] strides) {
            long idx = 0;
            int shift = indices.length - strides.length;
            for (int k = 0; k < strides.length; k++) {
                if (strides[k] != 0) {
                    idx += indices[shift + k] * strides[k];
                }
            }
            return idx;
        }
    }
}
