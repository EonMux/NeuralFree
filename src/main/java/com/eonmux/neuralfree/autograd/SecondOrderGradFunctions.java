package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.GpuReduceOps;
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
 * Backward functions of the {@link TensorOps} backward primitives ({@code activationBackward},
 * {@code softmaxBackward}, {@code logSoftmaxBackward}, {@code maxBackward}, {@code indexAddRows},
 * {@code segmentReduceBackward}), so that a backward pass with {@code createGraph = true} records a differentiable
 * graph of itself. A primitive records one of these functions only inside a create-graph pass
 * ({@link Autograd#isCreatingGraph()}) and only when one of its differentiable operands requires grad; the factories
 * below build them with those operands as the recorded inputs.
 *
 * <p>Notation: {@code b} is the primitive's result, {@code g2} the gradient arriving at {@code b}; each class returns
 * {@code g2} contracted with the partial derivative of {@code b} with respect to each recorded input. Every backward
 * is a composition of recording {@link TensorOps} calls and tensor views (rule G2) and never detaches a tensor that
 * may require grad (rule G4), so every class supports create-graph again (third order). The ownership contract is
 * that of {@link GradFunctions}: {@code g2} is never closed, every temporary is closed, and each returned gradient is
 * a new tensor owned by the caller. Tolerance: the class of the last operation
 * states; the formulas hold exactly in real arithmetic.</p>
 */
public final class SecondOrderGradFunctions {

    private static final float SQRT_TWO_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);
    private static final float GELU_CUBIC = 0.044715f;

    private SecondOrderGradFunctions() {
    }

    // ------------------------------------------------------------------ factories used by TensorOps

    /**
     * The recorded inputs of {@code activationBackward(op, x, y, gy)}: {@code (x, y, gy)}, where {@code x} is kept
     * only when {@code b} depends differentiably on it (SOFTPLUS, SILU, GELU, MISH) and {@code y} only when it
     * depends on it (SIGMOID, TANH, ELU); the other positions are null.
     */
    public static List<Tensor> activationOperands(ElementwiseOp.Backward op, Tensor x, Tensor y, Tensor gy) {
        return Arrays.asList(ActivationBackwardGrad.hasXPartial(op) ? x : null,
                ActivationBackwardGrad.hasYPartial(op) ? y : null, gy);
    }

    /** The function recorded by {@code activationBackward(op, x, y, gy, p0, p1)} over {@link #activationOperands}. */
    public static GradientFunction activationBackward(ElementwiseOp.Backward op, Tensor x, Tensor y, Tensor gy,
                                                      float p0, float p1) {
        boolean needX = ActivationBackwardGrad.hasXPartial(op) && requires(x);
        boolean needY = ActivationBackwardGrad.hasYPartial(op) && requires(y);
        ActivationBackwardGrad fn = new ActivationBackwardGrad(op, p0, p1, needX, needY, requires(gy));
        fn.saveForBackward(x, y, gy);
        return fn;
    }

    /**
     * The function recorded by {@code softmaxBackward(y, gy, dim)} ({@code log == false}) or
     * {@code logSoftmaxBackward(y, gy, dim)} ({@code log == true}) over the inputs {@code (y, gy)}.
     */
    public static GradientFunction softmaxBackward(Tensor y, Tensor gy, int dim, boolean log) {
        int d = axis(dim, y.shape().rank());
        if (log) {
            LogSoftmaxBackwardGrad fn = new LogSoftmaxBackwardGrad(d, requires(y), requires(gy));
            fn.saveForBackward(y, gy);
            return fn;
        }
        SoftmaxBackwardGrad fn = new SoftmaxBackwardGrad(d, requires(y), requires(gy));
        fn.saveForBackward(y, gy);
        return fn;
    }

    /** The function recorded by {@code maxBackward(x, maxOut, gy, dim)} over the input {@code (gy)}. */
    public static GradientFunction maxBackward(Tensor x, Tensor maxOut, Tensor gy, int dim) {
        MaxBackwardGrad fn = new MaxBackwardGrad(axis(dim, x.shape().rank()), gy.shape());
        fn.saveForBackward(x, maxOut);
        return fn;
    }

    /** The function recorded by {@code indexAddRows(gy, indices, rows, clamp, skipRow)} over the input {@code (gy)}. */
    public static GradientFunction indexAddRows(Tensor gy, Tensor indices, int rows, boolean clamp, int skipRow) {
        IndexAddRowsGrad fn = new IndexAddRowsGrad(gy.shape(), rows, clamp, skipRow);
        fn.saveForBackward(indices);
        return fn;
    }

    /**
     * The function recorded by {@code segmentReduceBackward(src, csrPtr, csrEdges, out, gy, op)} over the input
     * {@code (gy)}.
     */
    public static GradientFunction segmentReduceBackward(Tensor src, Tensor csrPtr, Tensor csrEdges, Tensor out,
                                                         Tensor gy, GpuReduceOps.SegmentOp op) {
        SegmentReduceBackwardGrad fn = new SegmentReduceBackwardGrad(op, gy.shape(), src.shape());
        fn.saveForBackward(src, csrPtr, csrEdges, out);
        return fn;
    }

    // ------------------------------------------------------------------ activations

    /**
     * {@code b = activationBackward(op, x, y, gy)}, recorded with the inputs {@code (x, y, gy)} of
     * {@link #activationOperands}. With {@code v = g2 * gy}, {@code s = sigmoid(x)}:
     * <ul>
     *   <li>{@code gy}: {@code activationBackward(op, x, y, g2)} for every op ({@code b} is linear in {@code gy}).</li>
     *   <li>SIGMOID ({@code b = gy y (1 - y)}): {@code gy_y = v (1 - 2y)}.</li>
     *   <li>TANH ({@code b = gy (1 - y^2)}): {@code gy_y = -2 v y}.</li>
     *   <li>ELU ({@code b = x > 0 ? gy : gy (y + alpha)}): {@code gy_y = x > 0 ? 0 : v}; x selects only, so it gets
     *   no gradient.</li>
     *   <li>SOFTPLUS ({@code b = gy s}): {@code gx = v s (1 - s)}.</li>
     *   <li>SILU ({@code f = x s}, {@code f' = s (1 + x (1 - s))}):
     *   {@code gx = v f''} with {@code f'' = s (1 - s) (2 + x (1 - 2s))}.</li>
     *   <li>GELU, tanh form ({@code f = x (1 + t) / 2}, {@code t = tanh(u)}, {@code u = c (x + k x^3)},
     *   {@code c = sqrt(2 / pi)}, {@code k = 0.044715}, {@code u' = c (1 + 3k x^2)}):
     *   {@code gx = v f''} with {@code f'' = (1 - t^2) (u' + 3ck x^2 - x t u'^2)}.</li>
     *   <li>MISH ({@code f = x T}, {@code T = tanh(softplus(x))}, {@code f' = T + x (1 - T^2) s}):
     *   {@code gx = v f''} with {@code f'' = (1 - T^2) s (2 + x (1 - s - 2 T s))}.</li>
     *   <li>RELU, LEAKY_RELU, ABS, CLAMP ({@code b = gy m(x)} with a piecewise-constant {@code m}): only
     *   {@code gy}.</li>
     * </ul>
     * The factors {@code s (1 - s)} and {@code 1 - t^2} are applied with the SIGMOID and TANH primitives. A y-partial
     * reaches {@code x} through the forward node, since SIGMOID, TANH and ELU save their output. Saved: {@code x} and
     * {@code y} as the op reads them, and {@code gy} when an x- or y-partial is needed.
     */
    public static class ActivationBackwardGrad extends GradFunctions.SavingGradFn {
        private final ElementwiseOp.Backward op;
        private final float p0;
        private final float p1;
        private final boolean needX;
        private final boolean needY;
        private final boolean needGy;
        private int xSlot = -1;
        private int ySlot = -1;
        private int gySlot = -1;

        /** {@code needX}/{@code needY}/{@code needGy}: whether that input requires a gradient (else its entry null). */
        public ActivationBackwardGrad(ElementwiseOp.Backward op, float p0, float p1, boolean needX, boolean needY,
                                      boolean needGy) {
            this.op = Objects.requireNonNull(op, "op");
            if (needX && !hasXPartial(op)) {
                throw new IllegalArgumentException(op + " has no x partial derivative");
            }
            if (needY && !hasYPartial(op)) {
                throw new IllegalArgumentException(op + " has no y partial derivative");
            }
            this.p0 = p0;
            this.p1 = p1;
            this.needX = needX;
            this.needY = needY;
            this.needGy = needGy;
        }

        /** Whether {@code b} depends differentiably on the forward input {@code x}. */
        static boolean hasXPartial(ElementwiseOp.Backward op) {
            switch (op) {
                case SOFTPLUS:
                case SILU:
                case GELU:
                case MISH:
                    return true;
                default:
                    return false;
            }
        }

        /** Whether {@code b} depends differentiably on the forward output {@code y}. */
        static boolean hasYPartial(ElementwiseOp.Backward op) {
            return op == ElementwiseOp.Backward.SIGMOID || op == ElementwiseOp.Backward.TANH
                    || op == ElementwiseOp.Backward.ELU;
        }

        /** Saves the operands the op reads ({@code x}, {@code y}) and {@code gy} when a partial needs it. */
        public void saveForBackward(Tensor x, Tensor y, Tensor gy) {
            List<Tensor> kept = new ArrayList<>(3);
            int nextX = -1;
            int nextY = -1;
            int nextGy = -1;
            if (op.readsX()) {
                nextX = kept.size();
                kept.add(Objects.requireNonNull(x, "x must not be null for " + op));
            }
            if (op.readsY()) {
                nextY = kept.size();
                kept.add(Objects.requireNonNull(y, "y must not be null for " + op));
            }
            if (needX || needY) {
                nextGy = kept.size();
                kept.add(Objects.requireNonNull(gy, "gy must not be null"));
            }
            save(kept.toArray(new Tensor[0]));
            xSlot = nextX;
            ySlot = nextY;
            gySlot = nextGy;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = xSlot >= 0 ? saved(xSlot) : null;
            Tensor y = ySlot >= 0 ? saved(ySlot) : null;
            Tensor gy = gySlot >= 0 ? saved(gySlot) : null;
            return gradients(
                    () -> needX ? xPartial(x, gy, gradOutput) : null,
                    () -> needY ? yPartial(x, y, gy, gradOutput) : null,
                    () -> needGy ? TensorOps.activationBackward(op, x, y, gradOutput, p0, p1) : null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        private Tensor xPartial(Tensor x, Tensor gy, Tensor g2) {
            try (Tensor v = TensorOps.mul(g2, gy)) {
                switch (op) {
                    case SOFTPLUS:
                        try (Tensor s = TensorOps.sigmoid(x)) {
                            return TensorOps.activationBackward(ElementwiseOp.Backward.SIGMOID, null, s, v, 0f, 0f);
                        }
                    case SILU:
                        return siluPartial(x, v);
                    case GELU:
                        return geluPartial(x, v);
                    case MISH:
                        return mishPartial(x, v);
                    default:
                        throw new IllegalStateException(op + " has no x partial derivative");
                }
            }
        }

        /** {@code v * s (1 - s) (2 + x (1 - 2s))}. */
        private static Tensor siluPartial(Tensor x, Tensor v) {
            try (Tensor s = TensorOps.sigmoid(x);
                 Tensor minusTwoS = TensorOps.mul(s, -2.0f);
                 Tensor oneMinusTwoS = TensorOps.add(minusTwoS, 1.0f);
                 Tensor xTerm = TensorOps.mul(x, oneMinusTwoS);
                 Tensor bracket = TensorOps.add(xTerm, 2.0f);
                 Tensor w = TensorOps.mul(v, bracket)) {
                return TensorOps.activationBackward(ElementwiseOp.Backward.SIGMOID, null, s, w, 0f, 0f);
            }
        }

        /** {@code v * (1 - t^2) (u' + 3ck x^2 - x t u'^2)}. */
        private static Tensor geluPartial(Tensor x, Tensor v) {
            try (Tensor x2 = TensorOps.mul(x, x);
                 Tensor x3 = TensorOps.mul(x2, x);
                 Tensor cubic = TensorOps.mul(x3, GELU_CUBIC);
                 Tensor inner = TensorOps.add(x, cubic);
                 Tensor u = TensorOps.mul(inner, SQRT_TWO_OVER_PI);
                 Tensor t = TensorOps.tanh(u);
                 Tensor du0 = TensorOps.mul(x2, 3.0f * GELU_CUBIC);
                 Tensor du1 = TensorOps.add(du0, 1.0f);
                 Tensor du = TensorOps.mul(du1, SQRT_TWO_OVER_PI);
                 Tensor curvature = TensorOps.mul(x2, 3.0f * SQRT_TWO_OVER_PI * GELU_CUBIC);
                 Tensor duSquared = TensorOps.mul(du, du);
                 Tensor xt = TensorOps.mul(x, t);
                 Tensor slopeTerm = TensorOps.mul(xt, duSquared);
                 Tensor partial = TensorOps.add(du, curvature);
                 Tensor bracket = TensorOps.sub(partial, slopeTerm);
                 Tensor w = TensorOps.mul(v, bracket)) {
                return TensorOps.activationBackward(ElementwiseOp.Backward.TANH, null, t, w, 0f, 0f);
            }
        }

        /** {@code v * (1 - T^2) s (2 + x (1 - s - 2 T s))}. */
        private static Tensor mishPartial(Tensor x, Tensor v) {
            try (Tensor sp = TensorOps.softplus(x);
                 Tensor tt = TensorOps.tanh(sp);
                 Tensor s = TensorOps.sigmoid(x);
                 Tensor twoT = TensorOps.mul(tt, 2.0f);
                 Tensor onePlusTwoT = TensorOps.add(twoT, 1.0f);
                 Tensor sTerm = TensorOps.mul(s, onePlusTwoT);
                 Tensor negSTerm = TensorOps.neg(sTerm);
                 Tensor inner = TensorOps.add(negSTerm, 1.0f);
                 Tensor xInner = TensorOps.mul(x, inner);
                 Tensor bracket = TensorOps.add(xInner, 2.0f);
                 Tensor vs = TensorOps.mul(v, s);
                 Tensor w = TensorOps.mul(vs, bracket)) {
                return TensorOps.activationBackward(ElementwiseOp.Backward.TANH, null, tt, w, 0f, 0f);
            }
        }

        private Tensor yPartial(Tensor x, Tensor y, Tensor gy, Tensor g2) {
            try (Tensor v = TensorOps.mul(g2, gy)) {
                switch (op) {
                    case SIGMOID:
                        try (Tensor minusTwoY = TensorOps.mul(y, -2.0f);
                             Tensor oneMinusTwoY = TensorOps.add(minusTwoY, 1.0f)) {
                            return TensorOps.mul(v, oneMinusTwoY);
                        }
                    case TANH:
                        try (Tensor vy = TensorOps.mul(v, y)) {
                            return TensorOps.mul(vy, -2.0f);
                        }
                    case ELU:
                        try (Tensor zeros = Tensor.zeros(x.backend(), x.shape());
                             Tensor positive = TensorOps.gt(x, zeros)) {
                            return TensorOps.where(positive, zeros, v);
                        }
                    default:
                        throw new IllegalStateException(op + " has no y partial derivative");
                }
            }
        }
    }

    // ------------------------------------------------------------------ softmax family

    /**
     * {@code b = softmaxBackward(y, gy, dim) = y (gy - s)}, {@code s = sum_dim(y gy)} (kept), recorded with the inputs
     * {@code (y, gy)}: {@code gy}: {@code softmaxBackward(y, g2, dim)} (the Jacobian is symmetric); {@code y}:
     * {@code g2 (gy - s) - gy sum_dim(g2 y)}. Saved: {@code y}, and {@code gy} when {@code y} needs a gradient.
     * Tolerance: reduction, K = the size of {@code dim}.
     */
    public static class SoftmaxBackwardGrad extends GradFunctions.SavingGradFn {
        private final int dim;
        private final boolean needY;
        private final boolean needGy;

        /** {@code dim} normalized; {@code needY}/{@code needGy}: whether that input requires a gradient. */
        public SoftmaxBackwardGrad(int dim, boolean needY, boolean needGy) {
            this.dim = dim;
            this.needY = needY;
            this.needGy = needGy;
        }

        /** Saves {@code y}, and {@code gy} when {@code y} needs a gradient. */
        public void saveForBackward(Tensor y, Tensor gy) {
            if (needY) {
                save(y, gy);
            } else {
                save(y);
            }
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor y = saved(0);
            Tensor gy = needY ? saved(1) : null;
            return gradients(() -> needY ? outputPartial(y, gy, gradOutput) : null,
                    () -> needGy ? TensorOps.softmaxBackward(y, gradOutput, dim) : null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        private Tensor outputPartial(Tensor y, Tensor gy, Tensor g2) {
            try (Tensor yGy = TensorOps.mul(y, gy);
                 Tensor s = keptSum(yGy, dim);
                 Tensor centered = TensorOps.sub(gy, s);
                 Tensor left = TensorOps.mul(g2, centered);
                 Tensor g2Y = TensorOps.mul(g2, y);
                 Tensor t = keptSum(g2Y, dim);
                 Tensor right = TensorOps.mul(gy, t)) {
                return TensorOps.sub(left, right);
            }
        }
    }

    /**
     * {@code b = logSoftmaxBackward(y, gy, dim) = gy - exp(y) sum_dim(gy)} (kept), recorded with the inputs
     * {@code (y, gy)}: {@code gy}: {@code g2 - sum_dim(exp(y) g2)}; {@code y}: {@code -(exp(y) g2) sum_dim(gy)}.
     * Saved: {@code y}, and {@code gy} when {@code y} needs a gradient. Tolerance: reduction, K = the size of
     * {@code dim}.
     */
    public static class LogSoftmaxBackwardGrad extends GradFunctions.SavingGradFn {
        private final int dim;
        private final boolean needY;
        private final boolean needGy;

        /** {@code dim} normalized; {@code needY}/{@code needGy}: whether that input requires a gradient. */
        public LogSoftmaxBackwardGrad(int dim, boolean needY, boolean needGy) {
            this.dim = dim;
            this.needY = needY;
            this.needGy = needGy;
        }

        /** Saves {@code y}, and {@code gy} when {@code y} needs a gradient. */
        public void saveForBackward(Tensor y, Tensor gy) {
            if (needY) {
                save(y, gy);
            } else {
                save(y);
            }
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor y = saved(0);
            Tensor gy = needY ? saved(1) : null;
            try (Tensor e = TensorOps.exp(y);
                 Tensor eG2 = TensorOps.mul(e, gradOutput)) {
                return gradients(() -> needY ? outputPartial(eG2, gy) : null,
                        () -> needGy ? gradientPartial(eG2, gradOutput) : null);
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        private Tensor outputPartial(Tensor eG2, Tensor gy) {
            try (Tensor total = keptSum(gy, dim);
                 Tensor product = TensorOps.mul(eG2, total)) {
                return TensorOps.neg(product);
            }
        }

        private Tensor gradientPartial(Tensor eG2, Tensor g2) {
            try (Tensor total = keptSum(eG2, dim)) {
                return TensorOps.sub(g2, total);
            }
        }
    }

    // ------------------------------------------------------------------ routing primitives

    /**
     * {@code b = maxBackward(x, out, gy, dim)}, recorded with the input {@code (gy)}: {@code b} is linear in
     * {@code gy} with a routing that is piecewise constant in {@code x} and {@code out}, so only {@code gy} gets a
     * gradient: {@code sum_dim(where(m, g2, 0))} in {@code gy}'s shape, where {@code m = maxBackward(x, out, ones)}
     * marks the routed element of every row (none for a NaN maximum). Saved: {@code x} and {@code out}. Tolerance:
     * bit-identical (one routed term per row).
     */
    public static class MaxBackwardGrad extends GradFunctions.SavingGradFn {
        private final int dim;
        private final Shape gradientShape;

        /** {@code dim} normalized against {@code x}; {@code gradientShape} the shape of {@code gy}. */
        public MaxBackwardGrad(int dim, Shape gradientShape) {
            this.dim = dim;
            this.gradientShape = Objects.requireNonNull(gradientShape, "gradientShape");
        }

        public void saveForBackward(Tensor x, Tensor maxOut) {
            save(x, maxOut);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = saved(0);
            Tensor maxOut = saved(1);
            try (Tensor ones = Tensor.ones(x.backend(), gradientShape);
                 Tensor mask = TensorOps.maxBackward(x, maxOut, ones, dim);
                 Tensor zeros = Tensor.zeros(x.backend(), x.shape());
                 Tensor routed = TensorOps.where(mask, gradOutput, zeros);
                 Tensor rows = TensorOps.sum(routed, dim)) {
                return Collections.singletonList(rows.reshape(gradientShape));
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code b = indexAddRows(gy, indices, rows, clamp, skipRow)}, recorded with the input {@code (gy)}: {@code b} is
     * linear in {@code gy}, and its adjoint selects rows: {@code indexSelectRows(g2', indices, clamp)} in
     * {@code gy}'s {@code [n, dim]} shape, where {@code g2'} is {@code g2} with row {@code skipRow} replaced by zeros
     * (a concatenation of slices), so a selection of the skipped row gets 0. Without {@code clamp} every index must
     * lie in the table, as the forward {@code indexSelectRows} that produced the first-order gradient required; an
     * out-of-range index throws {@link IndexOutOfBoundsException}. Saved: the indices. Tolerance: bit-identical.
     */
    public static class IndexAddRowsGrad extends GradFunctions.SavingGradFn {
        private final Shape gradientShape;
        private final int rows;
        private final boolean clamp;
        private final int skipRow;

        /** {@code gradientShape} the {@code [n, dim]} shape of {@code gy}; the rest as the primitive's arguments. */
        public IndexAddRowsGrad(Shape gradientShape, int rows, boolean clamp, int skipRow) {
            this.gradientShape = Objects.requireNonNull(gradientShape, "gradientShape");
            if (gradientShape.rank() != 2) {
                throw new IllegalArgumentException("indexAddRows takes a [n, dim] gradient, got " + gradientShape);
            }
            this.rows = rows;
            this.clamp = clamp;
            this.skipRow = skipRow;
        }

        public void saveForBackward(Tensor indices) {
            save(indices);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor indices = saved(0);
            Tensor table = skipRow >= 0 && skipRow < rows ? withZeroRow(gradOutput) : gradOutput;
            try (Tensor selected = TensorOps.indexSelectRows(table, indices, clamp)) {
                return Collections.singletonList(selected.reshape(gradientShape));
            } finally {
                if (table != gradOutput) {
                    table.close();
                }
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        /** {@code g2} with row {@code skipRow} replaced by zeros, as a new tensor. */
        private Tensor withZeroRow(Tensor g2) {
            int dim = g2.shape().dim(1);
            List<Tensor> parts = new ArrayList<>(3);
            try {
                if (skipRow > 0) {
                    parts.add(g2.slice(0, 0, skipRow));
                }
                parts.add(Tensor.zeros(g2.backend(), Shape.of(1, dim), g2.dtype()));
                if (skipRow + 1 < rows) {
                    parts.add(g2.slice(0, skipRow + 1, rows));
                }
                return TensorOps.cat(parts, 0);
            } finally {
                closeAll(parts);
            }
        }
    }

    /**
     * {@code b = segmentReduceBackward(src, csrPtr, csrEdges, out, gy, op)}, recorded with the input {@code (gy)}:
     * {@code b} is linear in {@code gy} (the MAX routing is piecewise constant in {@code src} and {@code out}), and its
     * adjoint gathers per segment: SUM {@code segmentReduce(g2, SUM)}, MEAN {@code segmentReduce(g2, MEAN)} (the sum
     * over the segment's source rows divided by its edge count), MAX {@code segmentReduce(where(m, g2, 0), SUM)} with
     * {@code m = segmentReduceBackward(src, csrPtr, csrEdges, out, ones, MAX)} marking the routed source element of
     * every segment and feature; empty segments get 0. Saved: the CSR arrays and, for MAX, {@code src} and
     * {@code out}. Tolerance: SUM and MAX reduction over the segment's edges, MEAN division.
     */
    public static class SegmentReduceBackwardGrad extends GradFunctions.SavingGradFn {
        private final GpuReduceOps.SegmentOp op;
        private final Shape gradientShape;
        private final Shape sourceShape;

        /** {@code gradientShape} the shape of {@code gy} (segments along dim 0); {@code sourceShape} that of src. */
        public SegmentReduceBackwardGrad(GpuReduceOps.SegmentOp op, Shape gradientShape, Shape sourceShape) {
            this.op = Objects.requireNonNull(op, "op");
            this.gradientShape = Objects.requireNonNull(gradientShape, "gradientShape");
            this.sourceShape = Objects.requireNonNull(sourceShape, "sourceShape");
            if (gradientShape.rank() == 0) {
                throw new IllegalArgumentException("segment gradients hold segments along dim 0, got a scalar");
            }
        }

        /** Saves the CSR arrays, and for MAX the source and the forward result. */
        public void saveForBackward(Tensor src, Tensor csrPtr, Tensor csrEdges, Tensor out) {
            if (op == GpuReduceOps.SegmentOp.MAX) {
                save(csrPtr, csrEdges, src, Objects.requireNonNull(out, "MAX saves the forward output"));
            } else {
                save(csrPtr, csrEdges);
            }
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor csrPtr = saved(0);
            Tensor csrEdges = saved(1);
            int segments = gradientShape.dim(0);
            if (op != GpuReduceOps.SegmentOp.MAX) {
                return Collections.singletonList(
                        TensorOps.segmentReduce(gradOutput, csrPtr, csrEdges, segments, op, 0.0f));
            }
            Tensor src = saved(2);
            Tensor out = saved(3);
            try (Tensor ones = Tensor.ones(src.backend(), gradientShape);
                 Tensor mask = TensorOps.segmentReduceBackward(src, csrPtr, csrEdges, out, ones, op);
                 Tensor zeros = Tensor.zeros(src.backend(), sourceShape);
                 Tensor routed = TensorOps.where(mask, gradOutput, zeros)) {
                return Collections.singletonList(TensorOps.segmentReduce(routed, csrPtr, csrEdges, segments,
                        GpuReduceOps.SegmentOp.SUM, 0.0f));
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    // ------------------------------------------------------------------ helpers

    /** {@code sum_dim(t)} with {@code dim} kept as size 1, as a new tensor. */
    private static Tensor keptSum(Tensor t, int dim) {
        try (Tensor s = TensorOps.sum(t, dim)) {
            return s.unsqueeze(dim);
        }
    }

    private static boolean requires(Tensor t) {
        return t != null && t.requiresGrad();
    }

    private static int axis(int dim, int rank) {
        int d = dim < 0 ? dim + rank : dim;
        if (d < 0 || d >= rank) {
            throw new IndexOutOfBoundsException("Dimension " + dim + " out of bounds for rank " + rank);
        }
        return d;
    }

    /** The gradients in input order; when one throws, those computed before it are closed. */
    @SafeVarargs
    private static List<Tensor> gradients(Supplier<Tensor>... suppliers) {
        List<Tensor> grads = new ArrayList<>(suppliers.length);
        try {
            for (Supplier<Tensor> s : suppliers) {
                grads.add(s.get());
            }
            return grads;
        } catch (RuntimeException e) {
            for (Tensor t : grads) {
                if (t != null) {
                    try {
                        t.close();
                    } catch (RuntimeException suppressed) {
                        e.addSuppressed(suppressed);
                    }
                }
            }
            throw e;
        }
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
