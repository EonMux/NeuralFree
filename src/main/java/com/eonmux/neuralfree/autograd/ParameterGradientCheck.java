package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.GradientCheckUtil.GradientCheckResult;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Strides;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

/**
 * The parameter gradient check behind {@link GradientCheckUtil#gradCheckParameters}: parameter validation, the loss
 * evaluations with buffer restoration, the gradient-slot stash and the element comparison. Its contract is the one
 * {@code gradCheckParameters} documents.
 */
final class ParameterGradientCheck {

    private ParameterGradientCheck() {
    }

    /** {@link GradientCheckUtil#gradCheckParameters}: see its contract. */
    static GradientCheckResult run(Supplier<Tensor> loss, Module module, List<Parameter> params, float epsilon,
                                   float tolerance, int maxElementsPerParameter) {
        validateParameterCheck(loss, params, epsilon, tolerance, maxElementsPerParameter);
        List<CheckedParameter> checked = checkedParameters(params);
        BufferSnapshot buffers = BufferSnapshot.take(module);
        GradientStash stash = GradientStash.take(module, params);
        Set<Tensor> keep = identitySet();
        for (CheckedParameter p : checked) {
            keep.add(p.leaf);
        }
        keep.addAll(buffers.tensors);
        keep.addAll(stash.leaves);
        Evaluator evaluator = new Evaluator(loss, buffers, keep);
        RuntimeException failure = null;
        try (ComputationGraph.StepScope step = ComputationGraph.current().step()) {
            float[][] analytical = evaluator.parameterGradients(params);
            if (analytical == null) {
                return new GradientCheckResult(false, Float.NaN, Float.NaN, "Loss supplier returned null");
            }
            Comparison comparison = new Comparison(tolerance);
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                for (int i = 0; i < checked.size(); i++) {
                    compareParameter(evaluator, checked.get(i), i, analytical[i], epsilon, maxElementsPerParameter,
                        comparison);
                }
            }
            return comparison.result();
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            stash.restore(failure);
        }
    }

    private static void validateParameterCheck(Supplier<Tensor> loss, List<Parameter> params, float epsilon,
                                               float tolerance, int maxElementsPerParameter) {
        if (loss == null) {
            throw new IllegalArgumentException("Loss supplier cannot be null");
        }
        if (params == null || params.isEmpty()) {
            throw new IllegalArgumentException("Parameters cannot be null or empty");
        }
        if (!(epsilon > 0.0f) || Float.isInfinite(epsilon)) {
            throw new IllegalArgumentException("Epsilon must be positive and finite, got " + epsilon);
        }
        if (!(tolerance >= 0.0f)) {
            throw new IllegalArgumentException("Tolerance must be non-negative, got " + tolerance);
        }
        if (maxElementsPerParameter < 1) {
            throw new IllegalArgumentException("maxElementsPerParameter must be positive, got "
                + maxElementsPerParameter);
        }
    }

    /** {@code params} validated as checkable, each with its leaf and its current values. */
    private static List<CheckedParameter> checkedParameters(List<Parameter> params) {
        Set<Parameter> seen = identitySet();
        List<CheckedParameter> checked = new ArrayList<>(params.size());
        for (int i = 0; i < params.size(); i++) {
            Parameter p = params.get(i);
            if (p == null) {
                throw new IllegalArgumentException("Parameter " + i + " is null");
            }
            String label = label(p, i);
            if (!seen.add(p)) {
                throw new IllegalArgumentException("Parameter " + label + " is listed more than once");
            }
            if (!p.requiresGrad()) {
                throw new IllegalArgumentException("Parameter " + label + " is frozen (requiresGrad=false)");
            }
            if (p.dtype() != DType.FLOAT32) {
                throw new IllegalArgumentException("Parameter " + label + " has dtype " + p.dtype()
                    + "; gradient checks perturb FLOAT32 parameters only");
            }
            Tensor leaf = requireLeaf(p, label);
            checked.add(new CheckedParameter(label, leaf, leaf.toFloatArray(), requireDistinctElements(leaf, label)));
        }
        return checked;
    }

    /** The tensor of {@code p}, which must be its leaf: no substitution is open. */
    private static Tensor requireLeaf(Parameter p, String label) {
        Tensor t = p.tensor();
        if (!t.requiresGrad() || t.gradFn() != null) {
            throw new IllegalStateException("Parameter " + label + " does not read its own leaf (a substitution is"
                + " open); close the substitution before checking gradients");
        }
        return t;
    }

    /** {@code t}'s strides, after checking that no element is addressed more than once. */
    private static Strides requireDistinctElements(Tensor t, String label) {
        Strides strides = t.strides();
        int[] dims = t.shape().dims();
        for (int k = 0; k < dims.length; k++) {
            if (dims[k] > 1 && strides.stride(k) == 0) {
                throw new IllegalArgumentException("Parameter " + label + " is a view " + t.shape() + " with strides "
                    + strides + " that addresses an element more than once");
            }
        }
        return strides;
    }

    private static String label(Parameter p, int index) {
        return p.name() != null ? "'" + p.name() + "'" : "#" + index;
    }

    /**
     * Adds the finite-difference comparison of the sampled elements of {@code p} to {@code comparison}; runs under
     * {@code noGrad}.
     */
    private static void compareParameter(Evaluator evaluator, CheckedParameter p, int index, float[] analytical,
                                         float epsilon, int maxElements, Comparison comparison) {
        int[] dims = p.leaf.shape().dims();
        for (int j : sampledElements(p.values.length, maxElements)) {
            long address = elementAddress(dims, p.strides, j);
            float numerical = evaluator.centralDifference(p.leaf, address, p.values[j], epsilon);
            comparison.add(p.label, j, analytical[j], numerical);
        }
    }

    /** All of {@code [0, n)} when {@code n <= max}, otherwise {@code max} indices evenly spaced over it. */
    private static int[] sampledElements(int n, int max) {
        int count = Math.min(n, max);
        int[] indices = new int[count];
        for (int k = 0; k < count; k++) {
            indices[k] = (int) ((long) k * n / count);
        }
        return indices;
    }

    /** The storage element of row-major element {@code index} of a view with {@code dims} and {@code strides}. */
    private static long elementAddress(int[] dims, Strides strides, int index) {
        long address = strides.offset();
        long rest = index;
        for (int k = dims.length - 1; k >= 0; k--) {
            address += (rest % dims[k]) * strides.stride(k);
            rest /= dims[k];
        }
        return address;
    }

    /**
     * Writes {@code value} to storage element {@code address} of {@code t} (see
     * {@link GradientCheckUtil#gradCheckParameters}).
     */
    private static void writeElement(Tensor t, long address, float value) {
        GpuEvent event = t.storage().copyFromHostFloat(new float[]{value}, address, t.backend().transferQueue());
        if (event != null) {
            event.close();
        }
    }

    private static <T> Set<T> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }

    /** Adds {@code e} to {@code primary} when there is one; otherwise to {@code first}, which it returns. */
    private static RuntimeException collect(RuntimeException primary, RuntimeException first, RuntimeException e) {
        if (primary != null) {
            primary.addSuppressed(e);
            return first;
        }
        if (first == null) {
            return e;
        }
        first.addSuppressed(e);
        return first;
    }

    /** A parameter being checked: its label, its leaf, its values in row-major order and the leaf's strides. */
    private static final class CheckedParameter {
        private final String label;
        private final Tensor leaf;
        private final float[] values;
        private final Strides strides;

        CheckedParameter(String label, Tensor leaf, float[] values, Strides strides) {
            this.label = label;
            this.leaf = leaf;
            this.values = values;
            this.strides = strides;
        }
    }

    /** Evaluations of the loss, each followed by closing its result and restoring the module buffers. */
    private static final class Evaluator {
        private final Supplier<Tensor> loss;
        private final BufferSnapshot buffers;
        private final Set<Tensor> keep;

        Evaluator(Supplier<Tensor> loss, BufferSnapshot buffers, Set<Tensor> keep) {
            this.loss = loss;
            this.buffers = buffers;
            this.keep = keep;
        }

        /**
         * The gradients of the loss's element sum for {@code params}, from one backward pass with grad mode on; null
         * when the supplier returns null.
         */
        float[][] parameterGradients(List<Parameter> params) {
            Tensor value = null;
            RuntimeException failure = null;
            try (ComputationGraph.EnableGradScope scope = ComputationGraph.enableGrad()) {
                value = loss.get();
                if (value == null) {
                    return null;
                }
                if (value.requiresGrad()) {
                    value.backward();
                }
                float[][] grads = new float[params.size()][];
                for (int i = 0; i < grads.length; i++) {
                    Parameter p = params.get(i);
                    Tensor grad = p.grad();
                    grads[i] = grad != null ? grad.toFloatArray() : new float[Math.toIntExact(p.numElements())];
                }
                return grads;
            } catch (RuntimeException e) {
                failure = e;
                throw e;
            } finally {
                finish(value, failure);
            }
        }

        /**
         * {@code (f(w + epsilon) - f(w - epsilon)) / (2 epsilon)} for the element of {@code leaf} at storage element
         * {@code address}, whose value is {@code w}; the element holds {@code w} again afterwards, also on failure.
         */
        float centralDifference(Tensor leaf, long address, float w, float epsilon) {
            float plus;
            float minus;
            RuntimeException failure = null;
            try {
                writeElement(leaf, address, w + epsilon);
                plus = lossSum("plus");
                writeElement(leaf, address, w - epsilon);
                minus = lossSum("minus");
            } catch (RuntimeException e) {
                failure = e;
                throw e;
            } finally {
                try {
                    writeElement(leaf, address, w);
                } catch (RuntimeException e) {
                    RuntimeException restoreFailure = collect(failure, null, e);
                    if (restoreFailure != null) {
                        throw restoreFailure;
                    }
                }
            }
            return (plus - minus) / (2.0f * epsilon);
        }

        /** The float sum, in element order, of one evaluation of the loss. */
        private float lossSum(String side) {
            Tensor value = null;
            RuntimeException failure = null;
            try {
                value = loss.get();
                if (value == null) {
                    throw new IllegalStateException("Loss supplier returned null during a finite-difference"
                        + " evaluation (" + side + ")");
                }
                float sum = 0.0f;
                for (float v : value.toFloatArray()) {
                    sum += v;
                }
                return sum;
            } catch (RuntimeException e) {
                failure = e;
                throw e;
            } finally {
                finish(value, failure);
            }
        }

        /** Closes {@code value} unless the check must keep it open, then restores the buffers. */
        private void finish(Tensor value, RuntimeException primary) {
            RuntimeException first = null;
            if (value != null && !keep.contains(value)) {
                try {
                    value.close();
                } catch (RuntimeException e) {
                    first = collect(primary, first, e);
                }
            }
            try {
                buffers.restore();
            } catch (RuntimeException e) {
                first = collect(primary, first, e);
            }
            if (first != null) {
                throw first;
            }
        }
    }

    /** The FLOAT32 buffers of a module with their values when the check started. */
    private static final class BufferSnapshot {
        private final List<Tensor> tensors;
        private final List<float[]> values;

        private BufferSnapshot(List<Tensor> tensors, List<float[]> values) {
            this.tensors = tensors;
            this.values = values;
        }

        /**
         * The buffers of {@code module} (none when it is null) and their current values.
         *
         * @throws IllegalArgumentException when a buffer is not FLOAT32
         */
        static BufferSnapshot take(Module module) {
            List<Tensor> tensors = new ArrayList<>();
            List<float[]> values = new ArrayList<>();
            if (module != null) {
                for (Map.Entry<String, Tensor> entry : module.namedBuffers().entrySet()) {
                    Tensor buffer = entry.getValue();
                    if (buffer.dtype() != DType.FLOAT32) {
                        throw new IllegalArgumentException("Buffer '" + entry.getKey() + "' has dtype "
                            + buffer.dtype() + "; gradient checks restore FLOAT32 buffers only");
                    }
                    tensors.add(buffer);
                    values.add(buffer.toFloatArray());
                }
            }
            return new BufferSnapshot(Collections.unmodifiableList(tensors), Collections.unmodifiableList(values));
        }

        /** Writes the snapshotted values back into every buffer, without recording. */
        void restore() {
            try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
                for (int i = 0; i < tensors.size(); i++) {
                    GpuOptimizerOps.storeFloats(tensors.get(i), values.get(i));
                }
            }
        }
    }

    /**
     * The gradients taken out of parameter slots for the duration of a check. A slot that held a gradient holds
     * zeros of its shape instead (the backward pass adds into them, {@code 0 + g == g} exactly), so the graph keeps
     * tracking the leaf as one it gave a gradient to ({@link ComputationGraph#zeroGrad()}) even when the check's
     * backward pass does not reach it; an empty slot stays empty.
     */
    private static final class GradientStash {
        private final List<Tensor> leaves;
        private final List<Tensor> stashed;

        private GradientStash(List<Tensor> leaves, List<Tensor> stashed) {
            this.leaves = leaves;
            this.stashed = stashed;
        }

        /**
         * Takes the gradients of {@code params} and of the trainable parameters of {@code module} (each leaf once)
         * out of their slots. Every leaf is validated before any slot changes; when a placeholder cannot be made,
         * the slots changed so far get their gradients back before the failure is thrown.
         */
        static GradientStash take(Module module, List<Parameter> params) {
            Set<Tensor> seen = identitySet();
            List<Parameter> all = new ArrayList<>(params);
            if (module != null) {
                all.addAll(module.parameters());
            }
            List<Tensor> leaves = new ArrayList<>();
            for (int i = 0; i < all.size(); i++) {
                Parameter p = all.get(i);
                if (p.requiresGrad()) {
                    Tensor leaf = requireLeaf(p, label(p, i));
                    if (seen.add(leaf)) {
                        leaves.add(leaf);
                    }
                }
            }
            List<Tensor> stashed = new ArrayList<>(leaves.size());
            GradientStash stash = new GradientStash(Collections.unmodifiableList(leaves), stashed);
            try {
                for (Tensor leaf : leaves) {
                    Tensor grad = leaf.grad();
                    Tensor placeholder = grad == null ? null : Tensor.zeros(leaf.backend(), leaf.shape());
                    try {
                        leaf.setGrad(placeholder);
                    } catch (RuntimeException e) {
                        if (placeholder != null) {
                            placeholder.close();
                        }
                        throw e;
                    }
                    stashed.add(grad);
                }
            } catch (RuntimeException e) {
                stash.restore(e);
                throw e;
            }
            return stash;
        }

        /**
         * Gives every slot taken so far its stashed gradient back and closes the gradient the check left there.
         * Every slot is tried; failures are added to {@code primary} when given, else the first is thrown with later
         * ones suppressed.
         */
        void restore(RuntimeException primary) {
            RuntimeException first = null;
            for (int i = 0; i < stashed.size(); i++) {
                try {
                    Tensor leaf = leaves.get(i);
                    Tensor produced = leaf.grad();
                    leaf.setGrad(stashed.get(i));
                    if (produced != null && produced != stashed.get(i)) {
                        produced.close();
                    }
                } catch (RuntimeException e) {
                    first = collect(primary, first, e);
                }
            }
            if (first != null) {
                throw first;
            }
        }
    }

    /** The running comparison of analytical and numerical gradient elements. */
    private static final class Comparison {
        private final float tolerance;
        private float maxDifference;
        private float maxRelativeError;
        private String firstFailure;

        Comparison(float tolerance) {
            this.tolerance = tolerance;
        }

        void add(String parameter, int element, float analytical, float numerical) {
            float diff = Math.abs(analytical - numerical);
            float denom = Math.max(Math.abs(analytical), Math.abs(numerical));
            float relError = denom > 1e-8f ? diff / denom : diff;
            maxDifference = Math.max(maxDifference, diff);
            maxRelativeError = Math.max(maxRelativeError, relError);
            if (GradientCheckUtil.fails(diff, relError, tolerance) && firstFailure == null) {
                firstFailure = String.format("Gradient mismatch at parameter %s element %d (analytical=%.6e,"
                    + " numerical=%.6e)", parameter, element, analytical, numerical);
            }
        }

        GradientCheckResult result() {
            boolean passed = firstFailure == null;
            return new GradientCheckResult(passed, maxDifference, maxRelativeError,
                passed ? "All parameter gradients match within tolerance" : firstFailure);
        }
    }
}
