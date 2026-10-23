package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Compares the gradients autograd computes with central finite differences: for the inputs of a function
 * ({@link #gradCheck}), for its second-order gradients ({@link #gradGradCheck}) and for the parameters of modules
 * ({@link #gradCheckParameters}).
 *
 * <p>The scalar checked is the float sum of the output (or loss) elements. The analytical gradient of every input
 * comes from one backward pass of {@code TensorOps.sum(output)} over fresh leaf copies of the inputs; the numerical
 * gradient of element {@code j} is {@code (f(x + eps e_j) - f(x - eps e_j)) / (2 eps)}, evaluated without recording.
 * The values are read back to the host for the comparison (the pass/fail decision).</p>
 *
 * <p><b>Tensor lifetime.</b> The inputs stay the caller's and are not modified. Every copy, output and gradient the
 * check creates is closed before it returns. Only the nodes the check recorded are released or discarded, so the
 * caller's graph and the calling thread's grad-enabled state are left as they were. A function must return a new
 * tensor or one of its arguments; its intermediate results that are not recorded nodes are its own to close.</p>
 */
public final class GradientCheckUtil {

    private static final float DEFAULT_EPSILON = 1e-3f;
    private static final float DEFAULT_TOLERANCE = 1e-2f;
    /** Seed of the fixed weights of {@link #gradGradCheck}; input {@code i} draws from {@code WEIGHT_SEED + i}. */
    private static final long WEIGHT_SEED = 0x6E6667726164L;

    private GradientCheckUtil() {
    }

    public static class GradientCheckResult {
        private final boolean passed;
        private final float maxDifference;
        private final float relativeError;
        private final String message;

        GradientCheckResult(boolean passed, float maxDifference, float relativeError, String message) {
            this.passed = passed;
            this.maxDifference = maxDifference;
            this.relativeError = relativeError;
            this.message = message;
        }

        public boolean passed() {
            return passed;
        }

        public float maxDifference() {
            return maxDifference;
        }

        public float relativeError() {
            return relativeError;
        }

        public String message() {
            return message;
        }

        @Override
        public String toString() {
            return String.format("GradientCheckResult{passed=%s, maxDiff=%.6e, relError=%.6e, msg='%s'}",
                    passed, maxDifference, relativeError, message);
        }
    }

    @FunctionalInterface
    public interface TensorFunction {
        Tensor apply(List<Tensor> inputs);
    }

    public static GradientCheckResult gradCheck(TensorFunction fn, List<Tensor> inputs) {
        return gradCheck(fn, inputs, DEFAULT_EPSILON, DEFAULT_TOLERANCE);
    }

    /**
     * Checks {@code fn}'s gradients at {@code inputs}. An element fails when both its absolute difference and its
     * relative difference (to the larger magnitude, or absolute below 1e-8) exceed {@code tolerance}, or when the
     * difference is NaN. A function that returns null fails the check.
     *
     * @throws IllegalArgumentException when {@code fn} is null, {@code inputs} is null or empty, or the inputs live on
     *                                  different backends
     * @throws IllegalStateException    when {@code fn} returns null during a finite-difference evaluation
     */
    public static GradientCheckResult gradCheck(TensorFunction fn, List<Tensor> inputs, float epsilon,
                                                float tolerance) {
        ComputeBackend backend = requireInputs(fn, inputs);
        List<float[]> values = hostValues(inputs);
        List<Shape> shapes = shapes(inputs);

        float[][] analytical = analyticalGradients(fn, backend, values, shapes);
        if (analytical == null) {
            return new GradientCheckResult(false, Float.NaN, Float.NaN, "Function returned null");
        }

        float maxDiff = 0.0f;
        float maxRelError = 0.0f;
        int failingIndex = -1;
        for (int i = 0; i < inputs.size(); i++) {
            float[] numerical = numericalGradient(fn, backend, values, shapes, i, epsilon);
            for (int j = 0; j < numerical.length; j++) {
                float diff = Math.abs(analytical[i][j] - numerical[j]);
                float denom = Math.max(Math.abs(analytical[i][j]), Math.abs(numerical[j]));
                float relError = denom > 1e-8f ? diff / denom : diff;
                maxDiff = Math.max(maxDiff, diff);
                maxRelError = Math.max(maxRelError, relError);
                if (fails(diff, relError, tolerance) && failingIndex < 0) {
                    failingIndex = i;
                }
            }
        }

        boolean passed = failingIndex < 0;
        String message = passed
                ? "All gradients match within tolerance"
                : String.format("Gradient mismatch at input %d (diff=%.6e, rel=%.6e)", failingIndex, maxDiff,
                        maxRelError);
        return new GradientCheckResult(passed, maxDiff, maxRelError, message);
    }

    /** The common backend of {@code inputs}, after checking {@code fn} and {@code inputs} as the checks document. */
    private static ComputeBackend requireInputs(TensorFunction fn, List<Tensor> inputs) {
        if (fn == null) {
            throw new IllegalArgumentException("Function cannot be null");
        }
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException("Inputs cannot be null or empty");
        }
        ComputeBackend backend = inputs.get(0).backend();
        for (Tensor input : inputs) {
            if (input.backend() != backend) {
                throw new IllegalArgumentException("All inputs must use the same backend");
            }
        }
        return backend;
    }

    private static List<float[]> hostValues(List<Tensor> inputs) {
        List<float[]> values = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            values.add(input.toFloatArray());
        }
        return values;
    }

    private static List<Shape> shapes(List<Tensor> inputs) {
        List<Shape> shapes = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            shapes.add(input.shape());
        }
        return shapes;
    }

    /** Whether a compared element fails: both differences exceed {@code tolerance}, or the difference is NaN. */
    static boolean fails(float diff, float relError, float tolerance) {
        return (diff > tolerance && relError > tolerance) || Float.isNaN(diff);
    }

    /**
     * The gradient of {@code sum(fn(x))} for every input, from one recorded backward pass over leaf copies; zeros for
     * an input the output does not depend on through recorded operations. Null when {@code fn} returns null.
     */
    private static float[][] analyticalGradients(TensorFunction fn, ComputeBackend backend, List<float[]> values,
                                                 List<Shape> shapes) {
        ComputationGraph graph = ComputationGraph.current();
        long mark = graph.mark();
        List<Tensor> leaves = new ArrayList<>(values.size());
        Tensor output = null;
        Tensor loss = null;
        RuntimeException failure = null;
        try (ComputationGraph.EnableGradScope scope = ComputationGraph.enableGrad()) {
            for (int i = 0; i < values.size(); i++) {
                leaves.add(leaf(backend, values.get(i), shapes.get(i)));
            }
            output = fn.apply(leaves);
            if (output == null) {
                return null;
            }
            loss = TensorOps.sum(output);
            if (loss.requiresGrad()) {
                loss.backward();
            }
            float[][] grads = new float[leaves.size()][];
            for (int i = 0; i < leaves.size(); i++) {
                Tensor grad = leaves.get(i).grad();
                grads[i] = grad != null ? grad.toFloatArray() : new float[values.get(i).length];
            }
            return grads;
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            release(graph, mark, Arrays.asList(loss, output), leaves, failure);
        }
    }

    /**
     * Releases what one analytical pass created: the graphs of {@code results} and every other node recorded since
     * {@code mark}, then the results, then the leaves with their gradients. Every step is tried; failures are added
     * to {@code primary} when given, else the first is thrown with later ones suppressed.
     */
    private static void release(ComputationGraph graph, long mark, List<Tensor> results, List<Tensor> leaves,
                                RuntimeException primary) {
        List<Runnable> steps = new ArrayList<>();
        for (Tensor result : results) {
            if (result != null) {
                steps.add(() -> graph.release(result));
            }
        }
        steps.add(() -> graph.discardRecordedSince(mark));
        for (Tensor result : results) {
            if (result != null) {
                steps.add(result::close);
            }
        }
        for (Tensor leaf : leaves) {
            steps.add(() -> {
                Tensor grad = leaf.grad();
                if (grad != null) {
                    leaf.setGrad(null);
                    grad.close();
                }
            });
            steps.add(leaf::close);
        }
        RuntimeException failure = null;
        for (Runnable step : steps) {
            try {
                step.run();
            } catch (RuntimeException e) {
                if (primary != null) {
                    primary.addSuppressed(e);
                } else if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** The central finite-difference gradient of {@code sum(fn(x))} with respect to input {@code index}. */
    private static float[] numericalGradient(TensorFunction fn, ComputeBackend backend, List<float[]> values,
                                             List<Shape> shapes, int index, float epsilon) {
        float[] base = values.get(index);
        float[] gradient = new float[base.length];
        List<Tensor> arguments = new ArrayList<>(values.size());
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            for (int i = 0; i < values.size(); i++) {
                arguments.add(i == index ? null : tensor(backend, values.get(i), shapes.get(i)));
            }
            for (int j = 0; j < base.length; j++) {
                float[] plus = Arrays.copyOf(base, base.length);
                plus[j] = base[j] + epsilon;
                float[] minus = Arrays.copyOf(base, base.length);
                minus[j] = base[j] - epsilon;
                float lossPlus = outputSum(fn, backend, arguments, index, plus, shapes.get(index), "plus");
                float lossMinus = outputSum(fn, backend, arguments, index, minus, shapes.get(index), "minus");
                gradient[j] = (lossPlus - lossMinus) / (2.0f * epsilon);
            }
            return gradient;
        } finally {
            for (Tensor argument : arguments) {
                if (argument != null) {
                    argument.close();
                }
            }
        }
    }

    /** The float sum, in element order, of {@code fn} applied with input {@code index} replaced by {@code data}. */
    private static float outputSum(TensorFunction fn, ComputeBackend backend, List<Tensor> arguments, int index,
                                   float[] data, Shape shape, String side) {
        try (Tensor perturbed = tensor(backend, data, shape)) {
            List<Tensor> call = new ArrayList<>(arguments);
            call.set(index, perturbed);
            Tensor output = fn.apply(call);
            if (output == null) {
                throw new IllegalStateException("Function returned null during forward pass (" + side + ")");
            }
            try {
                float sum = 0.0f;
                for (float v : output.toFloatArray()) {
                    sum += v;
                }
                return sum;
            } finally {
                if (!call.contains(output)) {
                    output.close();
                }
            }
        }
    }

    // ------------------------------------------------------------------ second order

    public static GradientCheckResult gradGradCheck(TensorFunction fn, List<Tensor> inputs) {
        return gradGradCheck(fn, inputs, DEFAULT_EPSILON, DEFAULT_TOLERANCE);
    }

    /**
     * Checks {@code fn}'s second-order gradients at {@code inputs}: finite differences of the
     * first-order gradients against a double backward pass.
     *
     * <p>With {@code L = sum(fn(x))}, {@code G_i = dL/dx_i} and fixed weights {@code w_i} of each input's shape
     * (uniform in {@code [0.5, 1.5)}, drawn from {@code new Random(WEIGHT_SEED + i)}), the checked scalar is
     * {@code S = sum_i sum(G_i * w_i)}, and the compared quantity is {@code dS/dx_j}, the Hessian of {@code L} applied
     * to {@code w}. <b>Analytical side:</b> inside one {@link ComputationGraph.StepScope} with grad mode on,
     * {@code G = Autograd.grad(L, x, null, false, true)} over fresh leaf copies of the inputs records the first-order
     * pass, {@code S} is formed with recording operations, and {@code Autograd.grad(S, x, null, false, false)} gives
     * {@code dS/dx} (zeros where no gradient arrives, all zeros when {@code L} or {@code S} does not require grad).
     * <b>Numerical side:</b> for every element {@code j} of every input, {@code S} is evaluated at
     * {@code x +- epsilon e_j} from a first-order backward pass (accumulated on the host in double) and
     * {@code (S(x + epsilon e_j) - S(x - epsilon e_j)) / (2 epsilon)} is the numerical value. An element fails as in
     * {@link #gradCheck(TensorFunction, List, float, float)}.</p>
     *
     * <p>Lifetime follows {@link #gradCheck}: the inputs stay the caller's and unchanged, every tensor the check
     * creates is closed, and the graph's size and the thread's grad mode are as before. Each evaluation takes
     * {@code 2 * (number of input elements)} first-order passes, so the inputs should be small. Inputs should keep
     * clear of the kinks of piecewise functions by more than {@code epsilon}.</p>
     *
     * @throws IllegalArgumentException      when {@code fn} is null, {@code inputs} is null or empty, the inputs live
     *                                       on different backends, {@code epsilon} is not positive and finite or
     *                                       {@code tolerance} is negative or NaN
     * @throws IllegalStateException         when {@code fn} returns null during a finite-difference evaluation
     * @throws UnsupportedOperationException when a node of {@code fn}'s graph does not support create-graph
     */
    public static GradientCheckResult gradGradCheck(TensorFunction fn, List<Tensor> inputs, float epsilon,
                                                    float tolerance) {
        ComputeBackend backend = requireInputs(fn, inputs);
        if (!(epsilon > 0.0f) || Float.isInfinite(epsilon)) {
            throw new IllegalArgumentException("Epsilon must be positive and finite, got " + epsilon);
        }
        if (!(tolerance >= 0.0f)) {
            throw new IllegalArgumentException("Tolerance must be non-negative, got " + tolerance);
        }
        List<float[]> values = hostValues(inputs);
        List<Shape> shapes = shapes(inputs);
        List<float[]> weights = new ArrayList<>(values.size());
        for (int i = 0; i < values.size(); i++) {
            Random random = new Random(WEIGHT_SEED + i);
            float[] w = new float[values.get(i).length];
            for (int j = 0; j < w.length; j++) {
                w[j] = 0.5f + random.nextFloat();
            }
            weights.add(w);
        }

        float[][] analytical = weightedSecondOrder(fn, backend, values, shapes, weights);
        if (analytical == null) {
            return new GradientCheckResult(false, Float.NaN, Float.NaN, "Function returned null");
        }
        float maxDiff = 0.0f;
        float maxRelError = 0.0f;
        String firstFailure = null;
        for (int i = 0; i < values.size(); i++) {
            float[] base = values.get(i);
            for (int j = 0; j < base.length; j++) {
                List<float[]> plus = perturbed(values, i, j, epsilon);
                List<float[]> minus = perturbed(values, i, j, -epsilon);
                double sPlus = weightedGradientSum(fn, backend, plus, shapes, weights, "plus");
                double sMinus = weightedGradientSum(fn, backend, minus, shapes, weights, "minus");
                float numerical = (float) ((sPlus - sMinus) / (2.0 * epsilon));
                float diff = Math.abs(analytical[i][j] - numerical);
                float denom = Math.max(Math.abs(analytical[i][j]), Math.abs(numerical));
                float relError = denom > 1e-8f ? diff / denom : diff;
                maxDiff = Math.max(maxDiff, diff);
                maxRelError = Math.max(maxRelError, relError);
                if (fails(diff, relError, tolerance) && firstFailure == null) {
                    firstFailure = String.format("Second-order mismatch at input %d element %d (analytical=%.6e,"
                        + " numerical=%.6e)", i, j, analytical[i][j], numerical);
                }
            }
        }
        boolean passed = firstFailure == null;
        return new GradientCheckResult(passed, maxDiff, maxRelError,
            passed ? "All second-order gradients match within tolerance" : firstFailure);
    }

    /**
     * {@code dS/dx_i} for every input from a create-graph pass and a second pass (see {@link #gradGradCheck}); null
     * when {@code fn} returns null.
     */
    private static float[][] weightedSecondOrder(TensorFunction fn, ComputeBackend backend, List<float[]> values,
                                                 List<Shape> shapes, List<float[]> weights) {
        List<Tensor> leaves = new ArrayList<>(values.size());
        List<Tensor> created = new ArrayList<>();
        RuntimeException failure = null;
        try (ComputationGraph.StepScope step = ComputationGraph.current().step();
             ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad()) {
            for (int i = 0; i < values.size(); i++) {
                leaves.add(leaf(backend, values.get(i), shapes.get(i)));
            }
            Tensor output = fn.apply(leaves);
            if (output == null) {
                return null;
            }
            created.add(output);
            Tensor loss = TensorOps.sum(output);
            created.add(loss);
            float[][] result = zeros(values);
            if (!loss.requiresGrad()) {
                return result;
            }
            List<Tensor> first = Autograd.grad(loss, leaves, null, false, true);
            created.addAll(first);
            Tensor s = weightedSum(backend, first, shapes, weights, created);
            if (s == null || !s.requiresGrad()) {
                return result;
            }
            List<Tensor> second = Autograd.grad(s, leaves, null, false, false);
            created.addAll(second);
            for (int i = 0; i < second.size(); i++) {
                if (second.get(i) != null) {
                    result[i] = second.get(i).toFloatArray();
                }
            }
            return result;
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            closeEach(created, leaves, failure);
        }
    }

    /**
     * {@code sum_i sum(G_i * w_i)} over the non-null {@code grads} with recording operations, every tensor it creates
     * added to {@code created}; null when all are null.
     */
    private static Tensor weightedSum(ComputeBackend backend, List<Tensor> grads, List<Shape> shapes,
                                      List<float[]> weights, List<Tensor> created) {
        Tensor total = null;
        for (int i = 0; i < grads.size(); i++) {
            Tensor g = grads.get(i);
            if (g == null) {
                continue;
            }
            Tensor w = tensor(backend, weights.get(i), shapes.get(i));
            created.add(w);
            Tensor product = TensorOps.mul(g, w);
            created.add(product);
            Tensor term = TensorOps.sum(product);
            created.add(term);
            if (total == null) {
                total = term;
            } else {
                total = TensorOps.add(total, term);
                created.add(total);
            }
        }
        return total;
    }

    /** {@code S} at {@code values} from a first-order backward pass, accumulated in double on the host. */
    private static double weightedGradientSum(TensorFunction fn, ComputeBackend backend, List<float[]> values,
                                              List<Shape> shapes, List<float[]> weights, String side) {
        float[][] grads = analyticalGradients(fn, backend, values, shapes);
        if (grads == null) {
            throw new IllegalStateException("Function returned null during a finite-difference evaluation (" + side
                + ")");
        }
        double s = 0.0;
        for (int i = 0; i < grads.length; i++) {
            float[] w = weights.get(i);
            for (int j = 0; j < w.length; j++) {
                s += (double) grads[i][j] * w[j];
            }
        }
        return s;
    }

    /** {@code values} with element {@code j} of input {@code i} shifted by {@code delta} (the others shared). */
    private static List<float[]> perturbed(List<float[]> values, int i, int j, float delta) {
        List<float[]> copy = new ArrayList<>(values);
        float[] shifted = Arrays.copyOf(values.get(i), values.get(i).length);
        shifted[j] = values.get(i)[j] + delta;
        copy.set(i, shifted);
        return copy;
    }

    private static float[][] zeros(List<float[]> values) {
        float[][] result = new float[values.size()][];
        for (int i = 0; i < result.length; i++) {
            result[i] = new float[values.get(i).length];
        }
        return result;
    }

    /**
     * Closes the non-null {@code created} tensors and the {@code leaves}; a tensor the step closed already is
     * unaffected, since closing is idempotent. Failures go to {@code primary} when given, else the first is thrown.
     */
    private static void closeEach(List<Tensor> created, List<Tensor> leaves, RuntimeException primary) {
        RuntimeException first = null;
        List<Tensor> all = new ArrayList<>(created);
        all.addAll(leaves);
        for (Tensor t : all) {
            if (t == null) {
                continue;
            }
            try {
                t.close();
            } catch (RuntimeException e) {
                if (primary != null) {
                    primary.addSuppressed(e);
                } else if (first == null) {
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

    // ------------------------------------------------------------------ parameters

    /**
     * Checks the gradients a backward pass gives {@code params} against central finite differences of the float sum
     * of the elements of {@code loss.get()}.
     *
     * <p><b>Analytical side.</b> {@code loss} is evaluated once with grad mode on and back-propagated with a seed of
     * ones ({@link Tensor#backward()}), which gives the gradient of the element sum; each parameter's gradient slot is
     * read back, a missing gradient counting as zeros. <b>Numerical side.</b> Under {@link ComputationGraph#noGrad()},
     * for up to {@code maxElementsPerParameter} elements of each parameter (all of them when it has no more, otherwise
     * elements evenly spaced in row-major order) the element is set to {@code w + epsilon} and to
     * {@code w - epsilon} in place through {@code storage().copyFromHostFloat} on the backend's transfer queue (the
     * optimizers' CPU write path), the loss is evaluated each time, and the element is restored to {@code w}; the
     * numerical gradient is {@code (f(w + epsilon) - f(w - epsilon)) / (2 epsilon)}. An element fails as in
     * {@link #gradCheck(TensorFunction, List, float, float)}.</p>
     *
     * <p><b>Determinism.</b> Every call of {@code loss} must compute the same function of the parameters: put the
     * modules in evaluation mode ({@link Module#eval()}), or seed every stochastic draw (dropout masks, noise) inside
     * {@code loss} so each call draws the same values. The buffers of {@code module} ({@link Module#namedBuffers()}:
     * normalization running statistics, spectral-norm power-iteration vectors) are snapshotted before the first
     * evaluation and restored after every evaluation, so each evaluation starts from the same buffer state and the
     * check leaves the buffers as it found them.</p>
     *
     * <p><b>State.</b> The gradients in the slots of {@code params} and of the trainable parameters of {@code module}
     * are stashed before the analytical pass, which therefore starts from empty (zero) slots; afterwards the slots
     * get their stashed gradients back and the analytical gradients are closed. Every node recorded during the check
     * is removed ({@link ComputationGraph.StepScope}), and each tensor {@code loss} returns is closed after use, so
     * {@code loss} returns a new tensor on every call. Other leaves the loss depends on (inputs that require grad,
     * parameters of modules other than {@code module}) accumulate the analytical gradient as after any backward pass,
     * so inputs should not require grad. No substitution ({@link Parameter#substitute}) may be open on a checked
     * parameter or a parameter of {@code module}, and {@code loss} must not step an optimizer.</p>
     *
     * @param loss                    evaluates the loss with the current parameter values and returns it as a new
     *                                FLOAT32 tensor of any shape
     * @param module                  the module whose buffers and gradient slots the check preserves; may be null
     * @param params                  the parameters to check: trainable, FLOAT32 and each listed once
     * @param epsilon                 the finite-difference step, positive
     * @param tolerance               the absolute and relative tolerance, non-negative
     * @param maxElementsPerParameter the number of elements checked per parameter at most, positive
     * @return the comparison; a failed result when {@code loss} returns null on the analytical pass
     * @throws IllegalArgumentException when {@code loss} or {@code params} is null, {@code params} is empty or holds
     *                                  null, a frozen, non-FLOAT32 or repeated parameter, or a parameter whose view
     *                                  addresses an element more than once; when a buffer of {@code module} is not
     *                                  FLOAT32; or when {@code epsilon}, {@code tolerance} or
     *                                  {@code maxElementsPerParameter} is out of range
     * @throws IllegalStateException    when the tensor of a checked parameter or of a trainable parameter of
     *                                  {@code module} is not its leaf (a substitution is open), or when {@code loss}
     *                                  returns null during a finite-difference evaluation
     */
    public static GradientCheckResult gradCheckParameters(Supplier<Tensor> loss, Module module, List<Parameter> params,
                                                          float epsilon, float tolerance, int maxElementsPerParameter) {
        return ParameterGradientCheck.run(loss, module, params, epsilon, tolerance, maxElementsPerParameter);
    }

    // ------------------------------------------------------------------ tensors

    /** A new tensor of {@code shape} holding {@code data}. */
    private static Tensor tensor(ComputeBackend backend, float[] data, Shape shape) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    /** A new tensor of {@code shape} holding {@code data} that requires grad. */
    private static Tensor leaf(ComputeBackend backend, float[] data, Shape shape) {
        try (Tensor values = tensor(backend, data, shape)) {
            return values.withRequiresGrad(true);
        }
    }

    /** A new INT32 tensor of {@code shape} holding {@code data}. */
    private static Tensor indices(ComputeBackend backend, int[] data, Shape shape) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    /** A uniform random tensor mapped elementwise by {@code value -> value * scale + offset}. */
    private static Tensor shiftedRand(ComputeBackend backend, Shape shape, float scale, float offset, boolean abs) {
        float[] data;
        try (Tensor r = Tensor.rand(backend, shape)) {
            data = r.toFloatArray();
        }
        for (int i = 0; i < data.length; i++) {
            float v = abs ? Math.abs(data[i]) : data[i];
            data[i] = v * scale + offset;
        }
        return tensor(backend, data, shape);
    }

    // ------------------------------------------------------------------ standard checks

    private static boolean check(TensorFunction fn, List<Tensor> inputs, float epsilon, float tolerance) {
        try {
            return gradCheck(fn, inputs, epsilon, tolerance).passed();
        } finally {
            for (Tensor input : inputs) {
                input.close();
            }
        }
    }

    private static boolean check(TensorFunction fn, List<Tensor> inputs) {
        return check(fn, inputs, DEFAULT_EPSILON, DEFAULT_TOLERANCE);
    }

    public static boolean checkAdd(ComputeBackend backend) {
        return check(inputs -> TensorOps.add(inputs.get(0), inputs.get(1)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4)), Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkSub(ComputeBackend backend) {
        return check(inputs -> TensorOps.sub(inputs.get(0), inputs.get(1)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4)), Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkDiv(ComputeBackend backend) {
        return check(inputs -> TensorOps.div(inputs.get(0), inputs.get(1)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4)),
                        shiftedRand(backend, Shape.of(4, 4), 1f, 0.5f, true)),
                1e-3f, 5e-2f);
    }

    public static boolean checkMul(ComputeBackend backend) {
        return check(inputs -> TensorOps.mul(inputs.get(0), inputs.get(1)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4)), Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkMatmul(ComputeBackend backend) {
        return check(inputs -> TensorOps.matmul(inputs.get(0), inputs.get(1)),
                Arrays.asList(Tensor.rand(backend, Shape.of(3, 4)), Tensor.rand(backend, Shape.of(4, 3))),
                1e-3f, 5e-2f);
    }

    public static boolean checkBmm(ComputeBackend backend) {
        return check(inputs -> TensorOps.bmm(inputs.get(0), inputs.get(1)),
                Arrays.asList(Tensor.rand(backend, Shape.of(1, 2, 2)), Tensor.rand(backend, Shape.of(1, 2, 2))),
                1e-3f, 5e-2f);
    }

    public static boolean checkExp(ComputeBackend backend) {
        return check(inputs -> TensorOps.exp(inputs.get(0)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkLog(ComputeBackend backend) {
        return check(inputs -> TensorOps.log(inputs.get(0)),
                Arrays.asList(shiftedRand(backend, Shape.of(4, 4), 1f, 0.1f, true)));
    }

    public static boolean checkSqrt(ComputeBackend backend) {
        return check(inputs -> TensorOps.sqrt(inputs.get(0)),
                Arrays.asList(shiftedRand(backend, Shape.of(4, 4), 1f, 0.1f, true)));
    }

    public static boolean checkAbs(ComputeBackend backend) {
        return check(inputs -> TensorOps.abs(inputs.get(0)),
                Arrays.asList(shiftedRand(backend, Shape.of(4, 4), 2f, -1f, false)), 1e-3f, 5e-2f);
    }

    public static boolean checkClamp(ComputeBackend backend) {
        return check(inputs -> TensorOps.clamp(inputs.get(0), 0.2f, 0.8f),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))), 1e-5f, 5e-3f);
    }

    public static boolean checkNeg(ComputeBackend backend) {
        return check(inputs -> TensorOps.neg(inputs.get(0)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkPow(ComputeBackend backend) {
        return check(inputs -> TensorOps.pow(inputs.get(0), 2.0f),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))), 1e-4f, 5e-3f);
    }

    public static boolean checkSum(ComputeBackend backend) {
        return check(inputs -> TensorOps.sum(inputs.get(0)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkMean(ComputeBackend backend) {
        return check(inputs -> TensorOps.mean(inputs.get(0)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkMax(ComputeBackend backend) {
        return check(inputs -> TensorOps.max(inputs.get(0), 1),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkReshape(ComputeBackend backend) {
        return check(inputs -> inputs.get(0).reshape(Shape.of(2, 8)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkTranspose(ComputeBackend backend) {
        return check(inputs -> inputs.get(0).transpose(),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static boolean checkCat(ComputeBackend backend) {
        return check(inputs -> TensorOps.cat(Arrays.asList(inputs.get(0), inputs.get(1)), 0),
                Arrays.asList(Tensor.rand(backend, Shape.of(2, 4)), Tensor.rand(backend, Shape.of(2, 4))));
    }

    public static boolean checkGather(ComputeBackend backend) {
        try (Tensor index = indices(backend, new int[]{0, 2, 1, 3, 3, 0, 2, 1}, Shape.of(2, 4))) {
            return check(inputs -> TensorOps.gather(inputs.get(0), 1, index),
                    Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))), 1e-5f, 5e-3f);
        }
    }

    public static boolean checkScatter(ComputeBackend backend) {
        float[] inputData = new float[16];
        for (int i = 0; i < 16; i++) {
            inputData[i] = i * 0.5f;
        }
        float[] srcData = new float[8];
        for (int i = 0; i < 8; i++) {
            srcData[i] = (i + 1) * 0.3f;
        }
        try (Tensor index = indices(backend, new int[]{0, 2, 1, 3, 3, 0, 2, 1}, Shape.of(2, 4))) {
            return check(inputs -> TensorOps.scatter(inputs.get(0), 1, index, inputs.get(1)),
                    Arrays.asList(tensor(backend, inputData, Shape.of(4, 4)), tensor(backend, srcData, Shape.of(2, 4))),
                    1e-3f, 1e-2f);
        }
    }

    public static boolean checkGradientAccumulation(ComputeBackend backend) {
        return check(inputs -> TensorOps.add(inputs.get(0), inputs.get(0)),
                Arrays.asList(Tensor.rand(backend, Shape.of(4, 4))));
    }

    /**
     * {@code sum(x * y)}: the product is an intermediate result, closed here when it was not recorded and otherwise
     * released by the backward pass.
     */
    public static boolean checkNestedOperations(ComputeBackend backend) {
        return check(inputs -> {
            Tensor product = TensorOps.mul(inputs.get(0), inputs.get(1));
            try {
                return TensorOps.sum(product);
            } finally {
                if (ComputationGraph.current().getNode(product) == null) {
                    product.close();
                }
            }
        }, Arrays.asList(Tensor.rand(backend, Shape.of(4, 4)), Tensor.rand(backend, Shape.of(4, 4))));
    }

    public static void runAllChecks(ComputeBackend backend) {
        System.out.println("Running gradient checks...");

        System.out.println("  add: " + (checkAdd(backend) ? "PASS" : "FAIL"));
        System.out.println("  sub: " + (checkSub(backend) ? "PASS" : "FAIL"));
        System.out.println("  mul: " + (checkMul(backend) ? "PASS" : "FAIL"));
        System.out.println("  div: " + (checkDiv(backend) ? "PASS" : "FAIL"));
        System.out.println("  matmul: " + (checkMatmul(backend) ? "PASS" : "FAIL"));
        System.out.println("  bmm: " + (checkBmm(backend) ? "PASS" : "FAIL"));
        System.out.println("  exp: " + (checkExp(backend) ? "PASS" : "FAIL"));
        System.out.println("  log: " + (checkLog(backend) ? "PASS" : "FAIL"));
        System.out.println("  sqrt: " + (checkSqrt(backend) ? "PASS" : "FAIL"));
        System.out.println("  abs: " + (checkAbs(backend) ? "PASS" : "FAIL"));
        System.out.println("  clamp: " + (checkClamp(backend) ? "PASS" : "FAIL"));
        System.out.println("  neg: " + (checkNeg(backend) ? "PASS" : "FAIL"));
        System.out.println("  pow: " + (checkPow(backend) ? "PASS" : "FAIL"));
        System.out.println("  sum: " + (checkSum(backend) ? "PASS" : "FAIL"));
        System.out.println("  mean: " + (checkMean(backend) ? "PASS" : "FAIL"));
        System.out.println("  max: " + (checkMax(backend) ? "PASS" : "FAIL"));
        System.out.println("  reshape: " + (checkReshape(backend) ? "PASS" : "FAIL"));
        System.out.println("  transpose: " + (checkTranspose(backend) ? "PASS" : "FAIL"));
        System.out.println("  cat: " + (checkCat(backend) ? "PASS" : "FAIL"));
        System.out.println("  gather: " + (checkGather(backend) ? "PASS" : "FAIL"));
        System.out.println("  scatter: " + (checkScatter(backend) ? "PASS" : "FAIL"));
        System.out.println("  gradient_accumulation: " + (checkGradientAccumulation(backend) ? "PASS" : "FAIL"));
        System.out.println("  nested_operations: " + (checkNestedOperations(backend) ? "PASS" : "FAIL"));

        System.out.println("Gradient checks complete.");
    }
}
