package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.BroadcastEngine;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Backward functions of the recording {@link TensorOps} ops and tensor views. Every backward is a composition of
 * {@link TensorOps} calls and tensor views, so it runs on the device when the train library is available on the
 * gradient's backend and on the CPU reference path otherwise; no backward reads a tensor to the host. Each class names
 * the formula it computes; the tolerance class of each op is the one its {@code TensorOps} dispatcher documents
 * ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).
 *
 * <p><b>Create graph.</b> Every class supports create-graph ({@link GradientFunction#supportsCreateGraph()}): its
 * backward consists of recording operations, the backward primitives included (they record their
 * {@link SecondOrderGradFunctions} inside a create-graph pass), reads saved tensors through {@code saved}, which
 * reattaches them to the forward tensors, and passes {@code gradOutput} through as the tensor itself, never as a
 * detached handle (rule G4).</p>
 *
 * <p><b>Ownership.</b></p>
 * <ul>
 *   <li>{@code saveForBackward} keeps its own handle of each saved tensor (a detached view sharing the storage), so
 *       the caller may close its tensors after the forward pass; the caller's tensors are never closed.
 *       {@link GradientFunction#clearSavedTensors()} closes these handles, and a later {@code backward} throws
 *       {@link IllegalStateException}.</li>
 *   <li>{@code backward} never closes {@code gradOutput} and closes every temporary it creates, also when it throws.
 *       Each non-null returned gradient is either {@code gradOutput} itself (an operand whose shape needs no
 *       reduction or reshaping), which the caller already owns (the graph takes a new reference of it), or a new
 *       {@link Tensor} object owned by the caller, who closes it when it is no longer needed (including a gradient
 *       it discards). A new gradient may be a view sharing {@code gradOutput}'s storage (reshape, expand and dense
 *       slice parts), so closing it never invalidates {@code gradOutput} or another gradient; a consumer that updates
 *       gradients in place must first give shared storage its own copy.</li>
 *   <li>A null entry means no gradient flows to that input: index operands, and operands the constructor marks as
 *       not requiring a gradient.</li>
 * </ul>
 */
public final class GradFunctions {

    private GradFunctions() {
    }

    /**
     * Saved-tensor bookkeeping: the handles {@link #save} creates, released by {@link #clearSavedTensors()}, each kept
     * beside a reference to the tensor it was saved from (its origin, held by identity only and never closed here).
     * Tensors saved directly through {@link #getContext()} stay the caller's, are never closed and have no origin.
     */
    abstract static class SavingGradFn implements GradientFunction {
        private final GradientContext ctx = new GradientContext();
        private final List<Tensor> handles = new ArrayList<>();
        private final List<Tensor> origins = new ArrayList<>();

        public GradientContext getContext() {
            return ctx;
        }

        /** Saves a handle of each tensor; on failure every handle created here is closed. */
        final void save(Tensor... tensors) {
            Objects.requireNonNull(tensors, "Tensors array cannot be null");
            List<Tensor> created = new ArrayList<>(tensors.length);
            try {
                for (int i = 0; i < tensors.length; i++) {
                    created.add(Objects.requireNonNull(tensors[i], "saved tensor " + i + " must not be null").detach());
                }
                ctx.saveForBackward(created.toArray(new Tensor[0]));
            } catch (RuntimeException e) {
                throw closeAfterFailure(created, e);
            }
            handles.addAll(created);
            origins.addAll(Arrays.asList(tensors));
        }

        /**
         * Saved tensor {@code i}, borrowed (never closed by the caller). Inside a create-graph pass a tensor saved with
         * {@link #save} is returned through {@link Autograd#reattach}, so operations on it route second-order
         * gradients to the forward tensor it was saved from.
         */
        final Tensor saved(int i) {
            List<Tensor> saved = ctx.getSavedTensors();
            if (i >= saved.size()) {
                throw new IllegalStateException(getClass().getSimpleName() + " has no saved tensor " + i
                        + ": nothing was saved, or the saved tensors were released after an earlier backward");
            }
            Tensor handle = saved.get(i);
            if (i >= origins.size() || !Autograd.isCreatingGraph()) {
                return handle;
            }
            return Autograd.reattach(handle, origins.get(i));
        }

        @Override
        public List<Tensor> savedTensors() {
            return ctx.getSavedTensors();
        }

        @Override
        public void clearSavedTensors() {
            ctx.clear();
            origins.clear();
            List<Tensor> owned = new ArrayList<>(handles);
            handles.clear();
            closeAll(owned);
        }
    }

    /**
     * A {@link SavingGradFn} whose backward is a composition of recording operations that never detaches a tensor
     * that may require grad (rule G4), so it records a differentiable graph of itself in a create-graph pass.
     */
    abstract static class ComposedGradFn extends SavingGradFn {
        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /** A gradient function that saves nothing and whose backward is a composition as for {@link ComposedGradFn}. */
    abstract static class StatelessGradFn implements GradientFunction {
        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    // ------------------------------------------------------------------ arithmetic

    /** {@code a + b}: {@code gy} reduced to each operand's shape. */
    public static class AddGrad extends ComposedGradFn {
        private final Shape shapeA;
        private final Shape shapeB;
        private final boolean needA;
        private final boolean needB;

        public AddGrad(Shape shapeA, Shape shapeB) {
            this(shapeA, shapeB, true, true);
        }

        /** {@code needA}/{@code needB}: whether the operand requires a gradient (else its entry is null). */
        public AddGrad(Shape shapeA, Shape shapeB, boolean needA, boolean needB) {
            this.shapeA = Objects.requireNonNull(shapeA, "shapeA");
            this.shapeB = Objects.requireNonNull(shapeB, "shapeB");
            this.needA = needA;
            this.needB = needB;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return pair(gradOutput, () -> needA ? reduceBroadcastGrad(gradOutput, shapeA) : null,
                    () -> needB ? reduceBroadcastGrad(gradOutput, shapeB) : null);
        }
    }

    /** {@code a - b}: {@code gy} and {@code -gy} reduced to the operands' shapes. */
    public static class SubGrad extends ComposedGradFn {
        private final Shape shapeA;
        private final Shape shapeB;
        private final boolean needA;
        private final boolean needB;

        public SubGrad(Shape shapeA, Shape shapeB) {
            this(shapeA, shapeB, true, true);
        }

        /** {@code needA}/{@code needB}: whether the operand requires a gradient (else its entry is null). */
        public SubGrad(Shape shapeA, Shape shapeB, boolean needA, boolean needB) {
            this.shapeA = Objects.requireNonNull(shapeA, "shapeA");
            this.shapeB = Objects.requireNonNull(shapeB, "shapeB");
            this.needA = needA;
            this.needB = needB;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return pair(gradOutput, () -> needA ? reduceBroadcastGrad(gradOutput, shapeA) : null,
                    () -> needB ? reduceOwned(TensorOps.neg(gradOutput), shapeB) : null);
        }
    }

    /** {@code a * b}: {@code gy * b} and {@code gy * a}, reduced to the operands' shapes. */
    public static class MulGrad extends ComposedGradFn {
        private final Shape shapeA;
        private final Shape shapeB;
        private final boolean needA;
        private final boolean needB;

        public MulGrad(Shape shapeA, Shape shapeB) {
            this(shapeA, shapeB, true, true);
        }

        /** {@code needA}/{@code needB}: whether the operand requires a gradient (else its entry is null). */
        public MulGrad(Shape shapeA, Shape shapeB, boolean needA, boolean needB) {
            this.shapeA = Objects.requireNonNull(shapeA, "shapeA");
            this.shapeB = Objects.requireNonNull(shapeB, "shapeB");
            this.needA = needA;
            this.needB = needB;
        }

        public void saveForBackward(Tensor a, Tensor b) {
            save(a, b);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor a = saved(0);
            Tensor b = saved(1);
            return pair(gradOutput, () -> needA ? reduceOwned(TensorOps.mul(gradOutput, b), shapeA) : null,
                    () -> needB ? reduceOwned(TensorOps.mul(gradOutput, a), shapeB) : null);
        }
    }

    /** {@code a / b}: {@code gy / b} and {@code gy * -(a / (b * b))}, reduced to the operands' shapes. */
    public static class DivGrad extends ComposedGradFn {
        private final Shape shapeA;
        private final Shape shapeB;
        private final boolean needA;
        private final boolean needB;

        public DivGrad(Shape shapeA, Shape shapeB) {
            this(shapeA, shapeB, true, true);
        }

        /** {@code needA}/{@code needB}: whether the operand requires a gradient (else its entry is null). */
        public DivGrad(Shape shapeA, Shape shapeB, boolean needA, boolean needB) {
            this.shapeA = Objects.requireNonNull(shapeA, "shapeA");
            this.shapeB = Objects.requireNonNull(shapeB, "shapeB");
            this.needA = needA;
            this.needB = needB;
        }

        public void saveForBackward(Tensor a, Tensor b) {
            save(a, b);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor a = saved(0);
            Tensor b = saved(1);
            return pair(gradOutput, () -> needA ? reduceOwned(TensorOps.div(gradOutput, b), shapeA) : null,
                    () -> needB ? reduceOwned(divisorGrad(gradOutput, a, b), shapeB) : null);
        }

        private static Tensor divisorGrad(Tensor gradOutput, Tensor a, Tensor b) {
            try (Tensor bSquared = TensorOps.mul(b, b);
                 Tensor aOverBSquared = TensorOps.div(a, bSquared);
                 Tensor negated = TensorOps.neg(aOverBSquared)) {
                return TensorOps.mul(gradOutput, negated);
            }
        }
    }

    /**
     * {@code pow(x, e)}: zeros for {@code e == 0} (the forward is constant), {@code gy * (x * 2)} for {@code e == 2},
     * {@code (gy / sqrt(x)) * 0.5} for {@code e == 0.5} (division class), else {@code gy * (pow(x, e - 1) * e)}.
     */
    public static class PowGrad extends ComposedGradFn {
        private final float exponent;

        public PowGrad(float exponent) {
            this.exponent = exponent;
        }

        public void saveForBackward(Tensor base) {
            save(base);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor base = saved(0);
            if (exponent == 0.0f) {
                return single(zerosOf(gradOutput, base.shape()));
            }
            if (exponent == 2.0f) {
                try (Tensor twice = TensorOps.mul(base, 2.0f)) {
                    return single(TensorOps.mul(gradOutput, twice));
                }
            }
            if (exponent == 0.5f) {
                try (Tensor root = TensorOps.sqrt(base);
                     Tensor quotient = TensorOps.div(gradOutput, root)) {
                    return single(TensorOps.mul(quotient, 0.5f));
                }
            }
            try (Tensor power = TensorOps.pow(base, exponent - 1.0f);
                 Tensor coef = TensorOps.mul(power, exponent)) {
                return single(TensorOps.mul(gradOutput, coef));
            }
        }
    }

    /** {@code -x}: {@code -gy}. */
    public static class NegGrad extends StatelessGradFn {
        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(TensorOps.neg(gradOutput));
        }
    }

    /** {@code sqrt(x)} from its output {@code y}: {@code gy / (y * 2)}. */
    public static class SqrtGrad extends ComposedGradFn {
        public void saveForBackward(Tensor output) {
            save(output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            try (Tensor twice = TensorOps.mul(saved(0), 2.0f)) {
                return single(TensorOps.div(gradOutput, twice));
            }
        }
    }

    /** {@code exp(x)} from its output {@code y}: {@code gy * y}. */
    public static class ExpGrad extends ComposedGradFn {
        public void saveForBackward(Tensor output) {
            save(output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(TensorOps.mul(gradOutput, saved(0)));
        }
    }

    /** {@code log(x)}: {@code gy / x}. */
    public static class LogGrad extends ComposedGradFn {
        public void saveForBackward(Tensor input) {
            save(input);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(TensorOps.div(gradOutput, saved(0)));
        }
    }

    /**
     * {@code sin(x)} from its saved input {@code x}: {@code gy * cos(x)}, composed of the recorded {@code cos} and
     * {@code mul} (transcendental class).
     */
    public static class SinGrad extends ComposedGradFn {
        public void saveForBackward(Tensor input) {
            save(input);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            try (Tensor derivative = TensorOps.cos(saved(0))) {
                return single(TensorOps.mul(gradOutput, derivative));
            }
        }
    }

    /**
     * {@code cos(x)} from its saved input {@code x}: {@code -(gy * sin(x))}, composed of the recorded {@code sin},
     * {@code mul} and {@code neg} (transcendental class).
     */
    public static class CosGrad extends ComposedGradFn {
        public void saveForBackward(Tensor input) {
            save(input);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            try (Tensor sine = TensorOps.sin(saved(0));
                 Tensor product = TensorOps.mul(gradOutput, sine)) {
                return single(TensorOps.neg(product));
            }
        }
    }

    /** {@code x + s}: {@code gy} in the input's shape. */
    public static class AddScalarGrad extends StatelessGradFn {
        private final Shape inputShape;

        public AddScalarGrad(Shape inputShape) {
            this.inputShape = Objects.requireNonNull(inputShape, "inputShape");
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(viewAs(gradOutput, inputShape));
        }
    }

    /** {@code x * s}: {@code gy * s}. */
    public static class MulScalarGrad extends StatelessGradFn {
        private final float scalar;

        public MulScalarGrad(float scalar) {
            this.scalar = scalar;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(TensorOps.mul(gradOutput, scalar));
        }
    }

    // ------------------------------------------------------------------ elementwise activations

    /**
     * Activation derivatives through {@link TensorOps#activationBackward}, from the saved input and/or output; in a
     * create-graph pass the primitive records {@link SecondOrderGradFunctions.ActivationBackwardGrad}.
     */
    abstract static class ActivationGrad extends ComposedGradFn {
        private final ElementwiseOp.Backward op;
        private final float p0;
        private final float p1;

        ActivationGrad(ElementwiseOp.Backward op, float p0, float p1) {
            this.op = op;
            this.p0 = p0;
            this.p1 = p1;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = op.readsX() ? saved(0) : null;
            Tensor y = op.readsY() ? saved(op.readsX() ? 1 : 0) : null;
            return single(TensorOps.activationBackward(op, x, y, gradOutput, p0, p1));
        }
    }

    /** {@code |x|}: {@code gy * sgn(x)}, {@code x > 0 ? gy : x < 0 ? -gy : 0} (0 at {@code +-0} and NaN). */
    public static class AbsGrad extends ActivationGrad {
        public AbsGrad() {
            super(ElementwiseOp.Backward.ABS, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    /** {@code clamp(x, min, max)}: {@code x < min || x > max ? 0 : gy}. */
    public static class ClampGrad extends ActivationGrad {
        public ClampGrad(float min, float max) {
            super(ElementwiseOp.Backward.CLAMP, min, max);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    /** {@code relu(x)}: {@code x > 0 ? gy : 0}. */
    public static class ReluGrad extends ActivationGrad {
        public ReluGrad() {
            super(ElementwiseOp.Backward.RELU, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    /** {@code gelu(x)} (tanh form): {@code gy * gelu'(x)}. */
    public static class GeluGrad extends ActivationGrad {
        public GeluGrad() {
            super(ElementwiseOp.Backward.GELU, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    /** {@code silu(x)}: {@code gy * (s * (1 + x * (1 - s)))} with {@code s = sigmoid(x)}. */
    public static class SiluGrad extends ActivationGrad {
        public SiluGrad() {
            super(ElementwiseOp.Backward.SILU, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    /** {@code sigmoid(x)} from its output {@code y}: {@code gy * y * (1 - y)}. */
    public static class SigmoidGrad extends ActivationGrad {
        public SigmoidGrad() {
            super(ElementwiseOp.Backward.SIGMOID, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor output) {
            save(output);
        }
    }

    /** {@code tanh(x)} from its output {@code y}: {@code gy * (1 - y * y)}. */
    public static class TanhGrad extends ActivationGrad {
        public TanhGrad() {
            super(ElementwiseOp.Backward.TANH, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor output) {
            save(output);
        }
    }

    /** {@code leakyRelu(x, slope)}: {@code x > 0 ? gy : slope * gy}. */
    public static class LeakyReluGrad extends ActivationGrad {
        public LeakyReluGrad(float negativeSlope) {
            super(ElementwiseOp.Backward.LEAKY_RELU, negativeSlope, 0.0f);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    /** {@code elu(x, alpha)} from its input and output: {@code x > 0 ? gy : gy * (y + alpha)}. */
    public static class EluGrad extends ActivationGrad {
        public EluGrad(float alpha) {
            super(ElementwiseOp.Backward.ELU, alpha, 0.0f);
        }

        public void saveForBackward(Tensor input, Tensor output) {
            save(input, output);
        }
    }

    /** {@code softplus(x)}: {@code gy * sigmoid(x)}. */
    public static class SoftplusGrad extends ActivationGrad {
        public SoftplusGrad() {
            super(ElementwiseOp.Backward.SOFTPLUS, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    /** {@code mish(x)}: {@code gy * (t + x * (1 - t * t) * sigmoid(x))} with {@code t = tanh(softplus(x))}. */
    public static class MishGrad extends ActivationGrad {
        public MishGrad() {
            super(ElementwiseOp.Backward.MISH, 0.0f, 0.0f);
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }
    }

    // ------------------------------------------------------------------ softmax family and gated units

    /** {@code softmax(x, dim)} from its output {@code y}: {@code y * (gy - sum(y * gy))} along {@code dim}. */
    public static class SoftmaxGrad extends ComposedGradFn {
        private final int dim;

        public SoftmaxGrad(int dim) {
            this.dim = dim;
        }

        public void saveForBackward(Tensor output) {
            save(output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(TensorOps.softmaxBackward(saved(0), gradOutput, dim));
        }
    }

    /** {@code logSoftmax(x, dim)} from its output {@code y}: {@code gy - exp(y) * sum(gy)} along {@code dim}. */
    public static class LogSoftmaxGrad extends ComposedGradFn {
        private final int dim;

        public LogSoftmaxGrad(int dim) {
            this.dim = dim;
        }

        public void saveForBackward(Tensor output) {
            save(output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(TensorOps.logSoftmaxBackward(saved(0), gradOutput, dim));
        }
    }

    /**
     * {@code glu(x, dim) = a * sigmoid(b)} over the halves {@code a}, {@code b} of {@code dim}: {@code ga = gy * s}
     * and {@code gb = ((gy * a) * s) * (1 - s)} with {@code s = sigmoid(b)}, concatenated along {@code dim}.
     */
    public static class GluGrad extends ComposedGradFn {
        private final int dim;

        public GluGrad(int dim) {
            this.dim = dim;
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(gatedBackward(saved(0), gradOutput, dim, false));
        }
    }

    /**
     * {@code geglu(x, dim) = a * gelu(b)} over the halves {@code a}, {@code b} of {@code dim}: {@code ga = gy *
     * gelu(b)} and {@code gb = (gy * a) * gelu'(b)}, concatenated along {@code dim}.
     */
    public static class GegluGrad extends ComposedGradFn {
        private final int dim;

        public GegluGrad(int dim) {
            this.dim = dim;
        }

        public void saveForBackward(Tensor input) {
            save(input);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(gatedBackward(saved(0), gradOutput, dim, true));
        }
    }

    private static Tensor gatedBackward(Tensor input, Tensor gradOutput, int dim, boolean gelu) {
        int d = axis(dim, input.shape().rank());
        int size = input.shape().dim(d);
        if (size % 2 != 0) {
            throw new IllegalArgumentException((gelu ? "GeGLU" : "GLU") + " dimension " + d + " has odd size " + size);
        }
        if (size == 0) {
            return zerosOf(gradOutput, input.shape());
        }
        int half = size / 2;
        try (Tensor a = input.slice(d, 0, half);
             Tensor b = input.slice(d, half, size);
             Tensor gate = gelu ? TensorOps.gelu(b) : TensorOps.sigmoid(b);
             Tensor gradA = TensorOps.mul(gradOutput, gate);
             Tensor gradTimesA = TensorOps.mul(gradOutput, a);
             Tensor gradB = gelu
                     ? TensorOps.activationBackward(ElementwiseOp.Backward.GELU, b, null, gradTimesA, 0.0f, 0.0f)
                     : TensorOps.activationBackward(ElementwiseOp.Backward.SIGMOID, null, gate, gradTimesA, 0.0f,
                     0.0f)) {
            return TensorOps.cat(Arrays.asList(gradA, gradB), d);
        }
    }

    // ------------------------------------------------------------------ matrix products

    /**
     * {@code a @ b} with the rank rules of {@link TensorOps#matmul}: with {@code A} and {@code B} the operands as
     * matrices (a rank-1 {@code a} as {@code [1, K]}, a rank-1 {@code b} as {@code [K, 1]}) and {@code G} the gradient
     * as {@code batch + [M, N]}, {@code ga = G @ B^T} and {@code gb = A^T @ G} over transposed views of the last two
     * dims, each summed over the batch dims its operand was broadcast along and reshaped to the operand's shape.
     */
    public static class MatmulGrad extends ComposedGradFn {
        private final boolean needA;
        private final boolean needB;

        public MatmulGrad() {
            this(true, true);
        }

        /** {@code needA}/{@code needB}: whether the operand requires a gradient (else its entry is null). */
        public MatmulGrad(boolean needA, boolean needB) {
            this.needA = needA;
            this.needB = needB;
        }

        public void saveForBackward(Tensor a, Tensor b) {
            save(a, b);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor a = saved(0);
            Tensor b = saved(1);
            Shape shapeA = a.shape();
            Shape shapeB = b.shape();
            boolean matrices = shapeA.rank() <= 2 && shapeB.rank() <= 2;
            int m = shapeA.rank() > 1 ? shapeA.dim(shapeA.rank() - 2) : 1;
            int n = shapeB.rank() > 1 ? shapeB.dim(shapeB.rank() - 1) : 1;
            Shape gradMatrix = Shape.of(m, n);
            List<Tensor> views = new ArrayList<>(3);
            try {
                Tensor am = shapeA.rank() == 1 ? owned(views, a.unsqueeze(0)) : a;
                Tensor bm = shapeB.rank() == 1 ? owned(views, b.unsqueeze(1)) : b;
                Tensor gm = matrices && !gradOutput.shape().equals(gradMatrix)
                        ? owned(views, gradOutput.reshape(gradMatrix)) : gradOutput;
                return pair(gradOutput, () -> needA ? gradOfLeft(gm, bm, am.shape(), shapeA) : null,
                        () -> needB ? gradOfRight(am, gm, bm.shape(), shapeB) : null);
            } finally {
                closeAll(views);
            }
        }

        /** {@code view}, recorded in {@code views} for closing. */
        private static Tensor owned(List<Tensor> views, Tensor view) {
            views.add(view);
            return view;
        }

        private static Tensor gradOfLeft(Tensor gm, Tensor bm, Shape matrixShape, Shape shape) {
            try (Tensor bT = transposeLastTwo(bm)) {
                return reduceAndReshape(TensorOps.matmul(gm, bT), matrixShape, shape);
            }
        }

        private static Tensor gradOfRight(Tensor am, Tensor gm, Shape matrixShape, Shape shape) {
            try (Tensor aT = transposeLastTwo(am)) {
                return reduceAndReshape(TensorOps.matmul(aT, gm), matrixShape, shape);
            }
        }
    }

    /** {@code bmm(a, b)} for {@code [B, M, K] @ [B, K, N]}: {@code ga = bmm(gy, b^T)}, {@code gb = bmm(a^T, gy)}. */
    public static class BmmGrad extends ComposedGradFn {
        private final boolean needA;
        private final boolean needB;

        public BmmGrad() {
            this(true, true);
        }

        /** {@code needA}/{@code needB}: whether the operand requires a gradient (else its entry is null). */
        public BmmGrad(boolean needA, boolean needB) {
            this.needA = needA;
            this.needB = needB;
        }

        public void saveForBackward(Tensor a, Tensor b) {
            save(a, b);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor a = saved(0);
            Tensor b = saved(1);
            return pair(gradOutput, () -> {
                if (!needA) {
                    return null;
                }
                try (Tensor bT = b.transpose(0, 2, 1)) {
                    return TensorOps.bmm(gradOutput, bT);
                }
            }, () -> {
                if (!needB) {
                    return null;
                }
                try (Tensor aT = a.transpose(0, 2, 1)) {
                    return TensorOps.bmm(aT, gradOutput);
                }
            });
        }
    }

    // ------------------------------------------------------------------ reductions

    /** {@code sum(x, dims)} ({@code null} or no dims: every element): {@code gy} broadcast back to the input shape. */
    public static class SumGrad extends StatelessGradFn {
        private final Shape inputShape;
        private final int[] dims;

        public SumGrad(Shape inputShape, int[] dims) {
            this.inputShape = Objects.requireNonNull(inputShape, "inputShape");
            this.dims = reducedDims(inputShape, dims);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(broadcastBack(gradOutput, inputShape, dims));
        }
    }

    /**
     * {@code mean(x, dims)} ({@code null} or no dims: every element): {@code gy * (1 / n)}, with {@code n} the number
     * of reduced elements, broadcast back to the input shape.
     */
    public static class MeanGrad extends StatelessGradFn {
        private final Shape inputShape;
        private final int[] dims;
        private final long numReducedElements;

        public MeanGrad(Shape inputShape, int[] dims) {
            this.inputShape = Objects.requireNonNull(inputShape, "inputShape");
            this.dims = reducedDims(inputShape, dims);
            long count = 1;
            for (int d : this.dims) {
                count *= inputShape.dim(d);
            }
            this.numReducedElements = this.dims.length == 0 ? inputShape.numElements() : count;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            float scale = 1.0f / numReducedElements;
            try (Tensor scaled = TensorOps.mul(gradOutput, scale)) {
                return single(broadcastBack(scaled, inputShape, dims));
            }
        }
    }

    /**
     * {@code max(x, dim)}: zeros of the input's shape except {@code gy} at the first index along {@code dim} whose
     * value equals the saved maximum ({@link TensorOps#maxBackward}).
     */
    public static class MaxGrad extends ComposedGradFn {
        private final int dim;
        private final Shape inputShape;

        public MaxGrad(int dim, Shape inputShape) {
            this.dim = dim;
            this.inputShape = Objects.requireNonNull(inputShape, "inputShape");
        }

        public void saveForBackward(Tensor input, Tensor output) {
            if (!input.shape().equals(inputShape)) {
                throw new IllegalArgumentException("saved input shape " + input.shape() + " does not match "
                        + inputShape);
            }
            save(input, output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(TensorOps.maxBackward(saved(0), saved(1), gradOutput, dim));
        }
    }

    // ------------------------------------------------------------------ views and data movement

    /** {@code reshape}: {@code gy} viewed in the original shape. */
    public static class ReshapeGrad extends StatelessGradFn {
        private final Shape originalShape;

        public ReshapeGrad(Shape originalShape) {
            this.originalShape = Objects.requireNonNull(originalShape, "originalShape");
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return single(gradOutput.reshape(originalShape));
        }
    }

    /**
     * {@code transpose}/{@code permute}: {@code gy} permuted by the inverse axes ({@code null}: all axes reversed), as
     * a row-major dense tensor: the permuted view when it already is dense, else its contiguous copy. A leaf reached
     * through a transpose, such as {@code Linear}'s weight under {@code x @ W^T}, therefore receives a row-major
     * gradient of its own layout, as {@link CatGrad} and {@link StackGrad} deliver theirs. Tolerance: bit-identical
     * (data movement).
     */
    public static class TransposeGrad extends StatelessGradFn {
        private final int[] axes;

        public TransposeGrad(int[] axes) {
            this.axes = axes == null ? null : axes.clone();
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            if (axes == null) {
                return single(dense(gradOutput.transpose()));
            }
            int[] inverseAxes = new int[axes.length];
            for (int i = 0; i < axes.length; i++) {
                inverseAxes[axes[i]] = i;
            }
            return single(dense(gradOutput.permute(inverseAxes)));
        }
    }

    /**
     * {@code slice(axis, start, end)} (negative bounds count from the end): {@code gy} placed at
     * {@code [start, end)} of a zero tensor of the original shape, as the concatenation of zeros, {@code gy} and zeros
     * along {@code axis} (an exact copy on both paths). An empty slice ({@code start == end}) gets zeros of the
     * original shape, and a zero-element original gets {@code gy} reshaped to it.
     */
    public static class SliceGrad extends StatelessGradFn {
        private final Shape originalShape;
        private final int axis;
        private final int start;
        private final int end;

        public SliceGrad(Shape originalShape, int axis, int start, int end) {
            this.originalShape = Objects.requireNonNull(originalShape, "originalShape");
            if (axis < 0 || axis >= originalShape.rank()) {
                throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + originalShape.rank());
            }
            int size = originalShape.dim(axis);
            int s = start < 0 ? start + size : start;
            int e = end < 0 ? end + size : end;
            if (s < 0 || e > size || s > e) {
                throw new IllegalArgumentException("Invalid slice bounds: start=" + start + ", end=" + end + ", dim="
                        + size);
            }
            this.axis = axis;
            this.start = s;
            this.end = e;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            if (originalShape.numElements() == 0) {
                return single(viewAs(gradOutput, originalShape));
            }
            if (start == end) {
                return single(zerosOf(gradOutput, originalShape));
            }
            int after = originalShape.dim(axis) - end;
            if (start == 0 && after == 0) {
                return single(viewAs(gradOutput, originalShape));
            }
            try (Tensor before = start > 0 ? zerosOf(gradOutput, withDim(originalShape, axis, start)) : null;
                 Tensor rest = after > 0 ? zerosOf(gradOutput, withDim(originalShape, axis, after)) : null) {
                List<Tensor> parts = new ArrayList<>(3);
                if (before != null) {
                    parts.add(before);
                }
                parts.add(gradOutput);
                if (rest != null) {
                    parts.add(rest);
                }
                return single(TensorOps.cat(parts, axis));
            }
        }
    }

    /**
     * {@code cat(tensors, dim)}: the slices of {@code gy} between consecutive split points (the cumulative sizes
     * along {@code dim}), each a dense tensor: a view when the slice already is dense, else a contiguous copy.
     */
    public static class CatGrad extends StatelessGradFn {
        private final int dim;
        private final int[] splitPoints;
        private final boolean[] needs;

        public CatGrad(int dim, int[] splitPoints) {
            this(dim, splitPoints, null);
        }

        /** {@code needs[i]}: whether input {@code i} requires a gradient (else its entry is null); null: all do. */
        public CatGrad(int dim, int[] splitPoints, boolean[] needs) {
            this.dim = dim;
            this.splitPoints = Objects.requireNonNull(splitPoints, "splitPoints").clone();
            if (needs != null && needs.length != splitPoints.length) {
                throw new IllegalArgumentException("needs has " + needs.length + " entries for " + splitPoints.length
                        + " inputs");
            }
            this.needs = needs == null ? null : needs.clone();
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Shape shape = gradOutput.shape();
            int d = axis(dim, shape.rank());
            if (splitPoints.length > 0 && splitPoints[splitPoints.length - 1] != shape.dim(d)) {
                throw new IllegalArgumentException("split points end at " + splitPoints[splitPoints.length - 1]
                        + " but the gradient has " + shape.dim(d) + " entries along dimension " + d);
            }
            List<Tensor> grads = new ArrayList<>(splitPoints.length);
            try {
                int start = 0;
                for (int i = 0; i < splitPoints.length; i++) {
                    int end = splitPoints[i];
                    if (end < start) {
                        throw new IllegalArgumentException("split points must not decrease: " + Arrays.toString(
                                splitPoints));
                    }
                    if (needs != null && !needs[i]) {
                        grads.add(null);
                    } else if (end == start) {
                        grads.add(zerosOf(gradOutput, withDim(shape, d, 0)));
                    } else {
                        grads.add(dense(gradOutput.slice(d, start, end)));
                    }
                    start = end;
                }
            } catch (RuntimeException e) {
                throw closeAfterFailure(grads, e);
            }
            return grads;
        }
    }

    /**
     * {@code stack(tensors, dim)} ({@code dim} in {@code [-rank-1, rank]} of the inputs): entry {@code i} of
     * {@code gy} along {@code dim} with that dim removed, each a dense tensor as for {@link CatGrad}.
     */
    public static class StackGrad extends StatelessGradFn {
        private final int dim;
        private final int numTensors;
        private final boolean[] needs;

        public StackGrad(int dim, int numTensors) {
            this(dim, numTensors, null);
        }

        /** {@code needs[i]}: whether input {@code i} requires a gradient (else its entry is null); null: all do. */
        public StackGrad(int dim, int numTensors, boolean[] needs) {
            if (needs != null && needs.length != numTensors) {
                throw new IllegalArgumentException("needs has " + needs.length + " entries for " + numTensors
                        + " inputs");
            }
            this.dim = dim;
            this.numTensors = numTensors;
            this.needs = needs == null ? null : needs.clone();
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            int d = axis(dim, gradOutput.shape().rank());
            if (gradOutput.shape().dim(d) != numTensors) {
                throw new IllegalArgumentException("gradient has " + gradOutput.shape().dim(d) + " entries along "
                        + "dimension " + d + " for " + numTensors + " stacked tensors");
            }
            List<Tensor> grads = new ArrayList<>(numTensors);
            try {
                for (int i = 0; i < numTensors; i++) {
                    if (needs != null && !needs[i]) {
                        grads.add(null);
                        continue;
                    }
                    try (Tensor entry = gradOutput.slice(d, i, i + 1)) {
                        grads.add(dense(entry.squeeze(d)));
                    }
                }
            } catch (RuntimeException e) {
                throw closeAfterFailure(grads, e);
            }
            return grads;
        }
    }

    /**
     * {@code gather(input, dim, index)}: {@code gy} added into zeros of the input's shape at the gathered positions,
     * in row-major index order ({@link TensorOps#scatterAdd}); the index operand gets no gradient.
     */
    public static class GatherGrad extends ComposedGradFn {
        private final int dim;
        private final Shape inputShape;

        public GatherGrad(int dim, Shape inputShape) {
            this.dim = dim;
            this.inputShape = Objects.requireNonNull(inputShape, "inputShape");
        }

        public void saveForBackward(Tensor index) {
            save(index);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor index = saved(0);
            try (Tensor zeros = zerosOf(gradOutput, inputShape)) {
                return single(TensorOps.scatterAdd(zeros, dim, index, gradOutput));
            }
        }
    }

    /**
     * {@code scatter(input, dim, index, src)}: the input gets {@code gy} with zeros written at the scattered
     * positions ({@link TensorOps#scatter} of zeros into a copy of {@code gy}); the source gets {@code gy} gathered at
     * those positions ({@link TensorOps#gather}); the index operand gets no gradient.
     */
    public static class ScatterGrad extends ComposedGradFn {
        private final int dim;
        private final boolean needInput;
        private final boolean needSrc;

        public ScatterGrad(int dim) {
            this(dim, true, true);
        }

        /** {@code needInput}/{@code needSrc}: whether the operand requires a gradient (else its entry is null). */
        public ScatterGrad(int dim, boolean needInput, boolean needSrc) {
            this.dim = dim;
            this.needInput = needInput;
            this.needSrc = needSrc;
        }

        /** Saves the index; {@code src} must have the index's shape, which is the shape of its gradient. */
        public void saveForBackward(Tensor index, Tensor src) {
            if (!index.shape().equals(src.shape())) {
                throw new IllegalArgumentException("Index and source must have the same shape");
            }
            save(index);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor index = saved(0);
            List<Tensor> grads = pair(gradOutput, () -> {
                if (!needInput) {
                    return null;
                }
                try (Tensor zeros = zerosOf(gradOutput, index.shape())) {
                    return TensorOps.scatter(gradOutput, dim, index, zeros);
                }
            }, () -> needSrc ? TensorOps.gather(gradOutput, dim, index) : null);
            return Arrays.asList(grads.get(0), null, grads.get(1));
        }
    }

    // ------------------------------------------------------------------ shared compositions

    /**
     * {@code grad} as the gradient of an operand of shape {@code target} that was broadcast to {@code grad}'s shape:
     * {@code grad} itself when the shapes match (the graph takes the new reference, rule G4), else
     * {@link TensorOps#sumToShape}. {@code grad} stays open.
     */
    private static Tensor reduceBroadcastGrad(Tensor grad, Shape target) {
        return grad.shape().equals(target) ? grad : TensorOps.sumToShape(grad, target);
    }

    /** {@code fresh}, which the caller owns, reduced to {@code target}: itself when the shapes match, else summed. */
    private static Tensor reduceOwned(Tensor fresh, Shape target) {
        if (fresh.shape().equals(target)) {
            return fresh;
        }
        try (Tensor owned = fresh) {
            return TensorOps.sumToShape(owned, target);
        }
    }

    /** {@code fresh} reduced to {@code matrixShape}, then viewed as {@code shape} (same element count). */
    private static Tensor reduceAndReshape(Tensor fresh, Shape matrixShape, Shape shape) {
        Tensor reduced = reduceOwned(fresh, matrixShape);
        if (reduced.shape().equals(shape)) {
            return reduced;
        }
        try (Tensor owned = reduced) {
            return owned.reshape(shape);
        }
    }

    /** A view of {@code t} with its last two dims swapped. */
    private static Tensor transposeLastTwo(Tensor t) {
        int rank = t.shape().rank();
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return t.transpose(axes);
    }

    /**
     * {@code grad}, which holds one value per reduced row ({@code dims} normalized; empty: every dim), broadcast back
     * to {@code inputShape} as a new dense tensor: the expanded view when no dim grows, else its contiguous copy.
     * {@code grad} stays open.
     */
    private static Tensor broadcastBack(Tensor grad, Shape inputShape, int[] dims) {
        Shape kept = BroadcastEngine.reduceShape(inputShape, dims.length == 0 ? allDims(inputShape.rank()) : dims,
                true);
        try (Tensor keptView = grad.reshape(kept)) {
            return dense(keptView.expand(inputShape));
        }
    }

    /** {@code dims} normalized against {@code shape}; null or empty means every dim (returned as empty). */
    private static int[] reducedDims(Shape shape, int[] dims) {
        if (dims == null || dims.length == 0) {
            return new int[0];
        }
        int[] normalized = BroadcastEngine.normalizeAxes(dims, shape.rank());
        boolean[] seen = new boolean[shape.rank()];
        for (int d : normalized) {
            if (seen[d]) {
                throw new IllegalArgumentException("dim " + d + " is reduced twice in " + Arrays.toString(dims));
            }
            seen[d] = true;
        }
        return normalized;
    }

    private static int[] allDims(int rank) {
        int[] dims = new int[rank];
        for (int i = 0; i < rank; i++) {
            dims[i] = i;
        }
        return dims;
    }

    /** {@code view}, which the caller owns, as a dense tensor: itself when dense, else a contiguous copy. */
    private static Tensor dense(Tensor view) {
        if (view.isDense()) {
            return view;
        }
        try (Tensor owned = view) {
            return owned.contiguous();
        }
    }

    /** Zeros of {@code shape} on {@code like}'s backend, in its dtype. */
    private static Tensor zerosOf(Tensor like, Shape shape) {
        return Tensor.zeros(like.backend(), shape, like.dtype());
    }

    /**
     * {@code grad}'s elements in {@code shape}: {@code grad} itself when the shape is unchanged (the graph takes the
     * new reference, rule G4), else a new reshaped view.
     */
    private static Tensor viewAs(Tensor grad, Shape shape) {
        return grad.shape().equals(shape) ? grad : grad.reshape(shape);
    }

    private static Shape withDim(Shape shape, int axis, int size) {
        int[] dims = shape.dims();
        dims[axis] = size;
        return Shape.of(dims);
    }

    private static int axis(int dim, int rank) {
        int d = dim < 0 ? dim + rank : dim;
        if (d < 0 || d >= rank) {
            throw new IndexOutOfBoundsException("Dimension " + dim + " out of bounds for rank " + rank);
        }
        return d;
    }

    private static List<Tensor> single(Tensor grad) {
        return Collections.singletonList(grad);
    }

    /**
     * The two gradients in order; when the second throws, the first is closed unless it is {@code gradOutput} itself
     * (the backward's incoming gradient, which stays its caller's).
     */
    private static List<Tensor> pair(Tensor gradOutput, Supplier<Tensor> first, Supplier<Tensor> second) {
        Tensor a = first.get();
        try {
            return Arrays.asList(a, second.get());
        } catch (RuntimeException e) {
            throw closeAfterFailure(Collections.singletonList(a == gradOutput ? null : a), e);
        }
    }

    /** Closes the non-null tensors, attaching any close failure to {@code failure}, and returns it. */
    private static RuntimeException closeAfterFailure(List<Tensor> tensors, RuntimeException failure) {
        for (Tensor t : tensors) {
            if (t != null) {
                try {
                    t.close();
                } catch (RuntimeException e) {
                    failure.addSuppressed(e);
                }
            }
        }
        return failure;
    }

    /** Closes every tensor; the first close failure is rethrown after all were attempted. */
    private static void closeAll(List<Tensor> tensors) {
        RuntimeException first = null;
        for (Tensor t : tensors) {
            try {
                t.close();
            } catch (RuntimeException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }
}
