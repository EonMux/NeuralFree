package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Automatic differentiation through {@link TensorOps}: recording, {@code backward} into leaf gradients,
 * {@link ComputationGraph#noGrad()}, {@link Autograd#grad}, backward from a leaf, and the error a second backward
 * through a released graph raises.
 *
 * <p>An operation records one node of the thread's graph ({@link ComputationGraph#current()}) when grad mode is on and
 * an input requires grad; {@code backward} releases the nodes it traverses, so the graph's size returns to its value
 * before the computation. Every value printed from a non-leaf tensor is read before {@code backward}, which closes the
 * graph's intermediates. Each section closes the tensors it creates and the gradients {@code backward} left on
 * them.</p>
 */
public final class AutogradDemo {

    /** Central-difference step of the numerical check (GradientCheckUtil's float step). */
    private static final float NUMERICAL_EPS = 1e-3f;
    /**
     * Largest accepted difference of the numerical check: the inputs lie away from relu's kink at 0, so the central
     * difference is exact up to the rounding of {@code x ± eps}, at most {@code ulp(x) / (2 eps)}, about 6e-5 here.
     */
    private static final float NUMERICAL_TOLERANCE = 1e-3f;

    private AutogradDemo() {}

    public static void run(ComputeBackend backend) {
        DemoUtils.printHeader("Autograd System Demo");
        System.out.println("Demonstrating automatic differentiation\n");

        Random random = new Random(42);

        System.out.println("--- Gradient Computation ---\n");
        demonstrateRecording(backend, random);

        System.out.println();
        demonstrateMatrixGradients(backend, random);

        System.out.println();
        demonstrateNestedGradients(backend, random);

        System.out.println("\n--- Grad Mode and Graph Lifetime ---\n");
        demonstrateNoGrad(backend, random);

        System.out.println();
        demonstrateAutogradGrad(backend, random);

        System.out.println();
        demonstrateLeafBackward(backend);

        System.out.println();
        demonstrateReleasedGraph(backend, random);

        System.out.println("\n--- Verification ---\n");
        demonstrateComparison(backend);
    }

    /** A leaf of the given shape with N(0, 1) values from {@code seed}, requiring gradients; the caller closes it. */
    private static Tensor leaf(ComputeBackend backend, Shape shape, long seed) {
        return DemoUtils.consume(Tensor.randn(backend, shape, seed), t -> t.withRequiresGrad(true));
    }

    /** A vector leaf holding {@code values}, requiring gradients; the caller closes it. */
    private static Tensor leaf(ComputeBackend backend, float[] values) {
        return DemoUtils.consume(Tensor.fromArray(backend, values), t -> t.withRequiresGrad(true));
    }

    private static int graphSize() {
        return ComputationGraph.current().size();
    }

    /** The simple class name of {@code t}'s gradient function, or "none" for a leaf or an unrecorded result. */
    private static String gradFnName(Tensor t) {
        return t.gradFn() == null ? "none" : t.gradFn().getClass().getSimpleName();
    }

    private static void demonstrateRecording(ComputeBackend backend, Random random) {
        System.out.println("Recording and backward (scalars):");

        Tensor x = leaf(backend, Shape.scalar(), random.nextLong());
        Tensor y = null;
        Tensor z = null;
        Tensor loss = null;
        try {
            y = leaf(backend, Shape.scalar(), random.nextLong());
            float xVal = DemoUtils.scalar(x);
            float yVal = DemoUtils.scalar(y);

            System.out.println("x = " + xVal);
            System.out.println("y = " + yVal);

            int before = graphSize();
            z = TensorOps.mul(x, y);
            loss = TensorOps.add(z, x);

            System.out.println("\nComputing: loss = x * y + x");
            System.out.println("loss = " + DemoUtils.scalar(loss));
            System.out.println("loss.requiresGrad() = " + loss.requiresGrad() + ", gradFn = " + gradFnName(loss)
                    + ", nodes recorded = " + (graphSize() - before));

            loss.backward();
            System.out.println("Nodes left after backward = " + (graphSize() - before));

            float xGrad = DemoUtils.scalar(x.grad());
            float yGrad = DemoUtils.scalar(y.grad());

            System.out.println("\nGradients (analytical):");
            System.out.println("∂loss/∂x = y + 1 = " + (yVal + 1));
            System.out.println("∂loss/∂y = x = " + xVal);

            System.out.println("\nGradients (computed):");
            System.out.println("∂loss/∂x = " + xGrad);
            System.out.println("∂loss/∂y = " + yGrad);
        } finally {
            DemoUtils.closeWithGrads(loss, z, x, y);
        }
    }

    private static void demonstrateMatrixGradients(ComputeBackend backend, Random random) {
        System.out.println("Matrix gradient computation:");

        Tensor a = leaf(backend, Shape.matrix(3, 4), random.nextLong());
        Tensor b = null;
        Tensor c = null;
        Tensor d = null;
        Tensor loss = null;
        try {
            b = leaf(backend, Shape.matrix(4, 2), random.nextLong());

            System.out.println("A: shape " + shapeToString(a.shape()));
            System.out.println("B: shape " + shapeToString(b.shape()));

            c = TensorOps.matmul(a, b);
            d = TensorOps.relu(c);
            loss = TensorOps.mean(d);

            System.out.println("\nComputing: loss = mean(relu(A @ B))");
            System.out.println("loss = " + DemoUtils.scalar(loss));

            loss.backward();

            Tensor aGrad = a.grad();
            Tensor bGrad = b.grad();

            System.out.println("\nGradient shapes:");
            System.out.println("∂loss/∂A: " + shapeToString(aGrad.shape()));
            System.out.println("∂loss/∂B: " + shapeToString(bGrad.shape()));
            System.out.println("||∂loss/∂A|| = "
                    + String.format("%.4f", DemoUtils.l2Norm(Collections.singletonList(aGrad))));
        } finally {
            DemoUtils.closeWithGrads(loss, d, c, a, b);
        }
    }

    private static void demonstrateNestedGradients(ComputeBackend backend, Random random) {
        System.out.println("Nested computation with multiple operations:");

        Tensor x = leaf(backend, Shape.vector(5), random.nextLong());
        Tensor w = null;
        Tensor prod = null;
        Tensor sum = null;
        Tensor sq = null;
        try {
            w = leaf(backend, Shape.vector(5), random.nextLong());

            prod = TensorOps.mul(x, w);
            sum = TensorOps.sum(prod);
            sq = TensorOps.mul(sum, sum);

            System.out.println("\nComputing: loss = (sum(x * w))^2");
            System.out.println("loss = " + DemoUtils.scalar(sq));

            sq.backward();

            float[] xGrad = x.grad().toFloatArray();
            float[] wGrad = w.grad().toFloatArray();

            System.out.println("\nGradients:");
            System.out.println("∂loss/∂x = 2 * sum(x*w) * w");
            System.out.println("∂loss/∂x = " + formatVector(xGrad));

            System.out.println("\n∂loss/∂w = 2 * sum(x*w) * x");
            System.out.println("∂loss/∂w = " + formatVector(wGrad));
        } finally {
            DemoUtils.closeWithGrads(sq, sum, prod, x, w);
        }
    }

    /**
     * The same operation on a leaf records a node in grad mode and nothing under {@code noGrad}. The recorded result
     * is never back-propagated, so a {@link ComputationGraph.StepScope} discards its node and closes it.
     */
    private static void demonstrateNoGrad(ComputeBackend backend, Random random) {
        System.out.println("Grad mode on and off (y = x * x):");

        Tensor x = leaf(backend, Shape.vector(3), random.nextLong());
        Tensor untracked = null;
        try {
            int before = graphSize();
            try (ComputationGraph.StepScope step = ComputationGraph.current().step()) {
                Tensor recorded = TensorOps.mul(x, x);
                System.out.println("Grad mode on:  requiresGrad = " + recorded.requiresGrad() + ", gradFn = "
                        + gradFnName(recorded) + ", nodes recorded = " + (graphSize() - before));
            }
            System.out.println("After its StepScope closed: nodes left = " + (graphSize() - before));

            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                untracked = TensorOps.mul(x, x);
            }
            System.out.println("Under noGrad:  requiresGrad = " + untracked.requiresGrad() + ", gradFn = "
                    + gradFnName(untracked) + ", nodes recorded = " + (graphSize() - before));
        } finally {
            DemoUtils.closeAll(untracked, x);
        }
    }

    /**
     * {@code Autograd.grad} returns the gradient to the caller and leaves the leaf's gradient slot untouched, where
     * {@code backward} would accumulate into it.
     */
    private static void demonstrateAutogradGrad(ComputeBackend backend, Random random) {
        System.out.println("Autograd.grad (torch.autograd.grad): dy/dx of y = sum(x * x) = 2x");

        Tensor x = leaf(backend, Shape.vector(3), random.nextLong());
        Tensor square = null;
        Tensor y = null;
        List<Tensor> grads = Collections.emptyList();
        try {
            System.out.println("x       = " + formatVector(x.toFloatArray()));
            square = TensorOps.mul(x, x);
            y = TensorOps.sum(square);

            grads = Autograd.grad(y, Collections.singletonList(x), null, false, false);

            System.out.println("dy/dx   = " + formatVector(grads.get(0).toFloatArray()));
            System.out.println("x.grad() after Autograd.grad = " + (x.grad() == null ? "null (slot untouched)"
                    : formatVector(x.grad().toFloatArray())));
        } finally {
            try {
                DemoUtils.closeAll(grads.toArray(new Tensor[0]));
            } finally {
                DemoUtils.closeWithGrads(y, square, x);
            }
        }
    }

    /** {@code backward} from a leaf accumulates the seed (ones of its shape) into its own gradient slot. */
    private static void demonstrateLeafBackward(ComputeBackend backend) {
        System.out.println("Backward from a leaf (seeded with ones):");

        Tensor x = leaf(backend, new float[]{0.5f, -1.0f, 2.0f});
        try {
            x.backward();
            System.out.println("After one backward:  x.grad() = " + formatVector(x.grad().toFloatArray()));
            x.backward();
            System.out.println("After two backwards: x.grad() = " + formatVector(x.grad().toFloatArray())
                    + " (gradients accumulate until zeroed)");
        } finally {
            DemoUtils.closeWithGrads(x);
        }
    }

    /**
     * {@code retainGraph = true} keeps the graph for a further backward; a backward without it releases the graph,
     * and a backward through the released graph throws.
     */
    private static void demonstrateReleasedGraph(ComputeBackend backend, Random random) {
        System.out.println("Retained and released graphs (y = sum(x * x), dy/dx = 2x):");

        Tensor x = leaf(backend, Shape.vector(3), random.nextLong());
        Tensor square = null;
        Tensor y = null;
        try {
            System.out.println("x = " + formatVector(x.toFloatArray()));
            square = TensorOps.mul(x, x);
            y = TensorOps.sum(square);

            ComputationGraph.current().backward(y, true);
            System.out.println("backward(retainGraph = true): x.grad() = " + formatVector(x.grad().toFloatArray()));

            y.backward();
            System.out.println("backward():                   x.grad() = " + formatVector(x.grad().toFloatArray())
                    + " (accumulated; the graph is now released)");

            try {
                y.backward();
                System.out.println("A third backward unexpectedly succeeded");
            } catch (IllegalStateException e) {
                System.out.println("A third backward throws IllegalStateException: " + e.getMessage());
            }
        } finally {
            DemoUtils.closeWithGrads(y, square, x);
        }
    }

    /** Compares the gradient of {@code sum(relu(x))} with central differences computed without recording. */
    private static void demonstrateComparison(ComputeBackend backend) {
        System.out.println("Numerical gradient verification of loss = sum(relu(x)):");

        float[] xData = {-0.75f, 0.5f, 1.25f};
        Tensor x = leaf(backend, xData);
        Tensor relu = null;
        Tensor loss = null;
        float[] analyticalGrad;
        try {
            relu = TensorOps.relu(x);
            loss = TensorOps.sum(relu);
            loss.backward();

            analyticalGrad = x.grad().toFloatArray();
        } finally {
            DemoUtils.closeWithGrads(loss, relu, x);
        }

        float[] numericalGrad = new float[xData.length];
        for (int i = 0; i < xData.length; i++) {
            float orig = xData[i];

            xData[i] = orig + NUMERICAL_EPS;
            double lossPlus = computeLoss(backend, xData);

            xData[i] = orig - NUMERICAL_EPS;
            double lossMinus = computeLoss(backend, xData);

            numericalGrad[i] = (float) ((lossPlus - lossMinus) / (2.0 * NUMERICAL_EPS));
            xData[i] = orig;
        }

        System.out.println("\nComparing analytical vs numerical gradients:");
        System.out.printf("%-10s %-15s %-15s %-10s%n", "Index", "Analytical", "Numerical", "Error");

        float maxError = 0;
        for (int i = 0; i < analyticalGrad.length; i++) {
            float error = Math.abs(analyticalGrad[i] - numericalGrad[i]);
            maxError = Math.max(maxError, error);
            System.out.printf("%-10d %-15.6f %-15.6f %-10.6f%n",
                i, analyticalGrad[i], numericalGrad[i], error);
        }

        System.out.println("\nMax error: " + String.format("%.2e", maxError));
        System.out.println("Gradient check: " + (maxError < NUMERICAL_TOLERANCE ? "PASSED" : "FAILED"));
    }

    /** {@code sum(relu(x))} for the given values, under {@code noGrad}; the sum is taken in double on the host. */
    private static double computeLoss(ComputeBackend backend, float[] values) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor x = Tensor.fromArray(backend, values);
             Tensor relu = TensorOps.relu(x)) {
            double sum = 0;
            for (float v : relu.toFloatArray()) {
                sum += v;
            }
            return sum;
        }
    }

    private static String formatVector(float[] values) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            sb.append(String.format("%.3f", values[i]));
            if (i < values.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    private static String shapeToString(Shape shape) {
        StringBuilder sb = new StringBuilder("[");
        int[] dims = shape.dims();
        for (int i = 0; i < dims.length; i++) {
            sb.append(dims[i]);
            if (i < dims.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
