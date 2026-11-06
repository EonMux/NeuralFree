package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Strides;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorValues;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

/**
 * The double-precision numerical side of {@link GradientCheckUtil#gradCheckFloat64} and
 * {@link GradientCheckUtil#gradCheckParametersFloat64}: central differences of a FLOAT64 evaluation of the checked
 * function, with the step taken and the loss summed in {@code double}. Every evaluation runs under
 * {@link ComputationGraph#noGrad()} and its result is closed after use.
 */
final class Float64Reference {

    private Float64Reference() {
    }

    // ------------------------------------------------------------------ inputs

    /**
     * {@code (S(x + epsilon e_j) - S(x - epsilon e_j)) / (2 epsilon)} for every element {@code j} of input
     * {@code index}, where {@code S} is the double sum of {@code reference} applied to FLOAT64 copies of the inputs
     * ({@code values}, widened exactly) on {@code backend}.
     *
     * @throws IllegalStateException when {@code reference} returns null or a tensor that is not FLOAT64
     */
    static float[] inputGradient(GradientCheckUtil.TensorFunction reference, ComputeBackend backend,
                                 List<float[]> values, List<Shape> shapes, int index, double epsilon) {
        double[] base = widen(values.get(index));
        float[] gradient = new float[base.length];
        List<Tensor> arguments = new ArrayList<>(values.size());
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            for (int i = 0; i < values.size(); i++) {
                arguments.add(i == index ? null : tensor(backend, widen(values.get(i)), shapes.get(i)));
            }
            for (int j = 0; j < base.length; j++) {
                double[] point = base.clone();
                point[j] = base[j] + epsilon;
                double plus = outputSum(reference, backend, arguments, index, point, shapes.get(index), "plus");
                point[j] = base[j] - epsilon;
                double minus = outputSum(reference, backend, arguments, index, point, shapes.get(index), "minus");
                gradient[j] = (float) ((plus - minus) / (2.0 * epsilon));
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

    /** The double sum of {@code reference} applied with input {@code index} replaced by {@code data}. */
    private static double outputSum(GradientCheckUtil.TensorFunction reference, ComputeBackend backend,
                                    List<Tensor> arguments, int index, double[] data, Shape shape, String side) {
        try (Tensor perturbed = tensor(backend, data, shape)) {
            List<Tensor> call = new ArrayList<>(arguments);
            call.set(index, perturbed);
            Tensor output = reference.apply(call);
            if (output == null) {
                throw new IllegalStateException("Reference function returned null (" + side + ")");
            }
            try {
                return sum(output, "Reference function");
            } finally {
                if (!call.contains(output)) {
                    output.close();
                }
            }
        }
    }

    // ------------------------------------------------------------------ parameters

    /**
     * The FLOAT64 counterparts, in {@code reference}, of the parameters a check perturbs, and the reference loss
     * whose central differences are the numerical gradients ({@link GradientCheckUtil#gradCheckParametersFloat64}).
     * The reference's buffers are snapshotted when it is bound and restored after every evaluation.
     */
    static final class Parameters implements ParameterGradientCheck.NumericalSide {
        private final Supplier<Tensor> loss;
        private final List<Counterpart> counterparts;
        private final List<Tensor> buffers;
        private final List<double[]> bufferValues;
        private final Set<Tensor> keep;
        private final double epsilon;

        private Parameters(Supplier<Tensor> loss, List<Counterpart> counterparts, List<Tensor> buffers,
                           List<double[]> bufferValues, double epsilon) {
            this.loss = loss;
            this.counterparts = counterparts;
            this.buffers = buffers;
            this.bufferValues = bufferValues;
            this.epsilon = epsilon;
            this.keep = ParameterGradientCheck.identitySet();
            for (Counterpart c : counterparts) {
                keep.add(c.leaf);
            }
            keep.addAll(buffers);
        }

        /**
         * Binds {@code params}, parameters of {@code module}, to the parameters of {@code reference} under the same
         * dotted names, after checking that {@code reference} holds the state of {@code module}: the same parameter
         * and buffer names, shapes and values (compared exactly in double), with every counterpart of a checked
         * parameter FLOAT64 and its own leaf, and every buffer FLOAT32 or FLOAT64.
         *
         * @throws IllegalArgumentException as {@link GradientCheckUtil#gradCheckParametersFloat64} documents
         * @throws IllegalStateException    when a counterpart is substituted
         */
        static Parameters bind(Module module, List<Parameter> params, Module reference, Supplier<Tensor> loss,
                               double epsilon) {
            Map<String, Parameter> names = sameState(module, reference);
            Map<Parameter, String> nameOf = new IdentityHashMap<>();
            for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
                nameOf.putIfAbsent(e.getValue(), e.getKey());
            }
            List<Counterpart> counterparts = new ArrayList<>(params.size());
            for (int i = 0; i < params.size(); i++) {
                Parameter p = params.get(i);
                String name = nameOf.get(p);
                if (name == null) {
                    throw new IllegalArgumentException("Parameter " + ParameterGradientCheck.label(p, i)
                        + " is not a parameter of the module, so it has no counterpart in the reference");
                }
                counterparts.add(Counterpart.of(name, names.get(name)));
            }
            List<Tensor> buffers = new ArrayList<>();
            List<double[]> values = new ArrayList<>();
            for (Tensor buffer : reference.namedBuffers().values()) {
                buffers.add(buffer);
                values.add(doubles(buffer));
            }
            return new Parameters(loss, Collections.unmodifiableList(counterparts),
                Collections.unmodifiableList(buffers), Collections.unmodifiableList(values), epsilon);
        }

        @Override
        public float centralDifference(int parameter, int element) {
            Counterpart c = counterparts.get(parameter);
            long address = ParameterGradientCheck.elementAddress(c.leaf.shape().dims(), c.strides, element);
            double w = c.values[element];
            double plus;
            double minus;
            RuntimeException failure = null;
            try {
                write(c.leaf, address, w + epsilon);
                plus = lossSum("plus");
                write(c.leaf, address, w - epsilon);
                minus = lossSum("minus");
            } catch (RuntimeException e) {
                failure = e;
                throw e;
            } finally {
                try {
                    write(c.leaf, address, w);
                } catch (RuntimeException e) {
                    RuntimeException restoreFailure = ParameterGradientCheck.collect(failure, null, e);
                    if (restoreFailure != null) {
                        throw restoreFailure;
                    }
                }
            }
            return (float) ((plus - minus) / (2.0 * epsilon));
        }

        /** The double sum of one evaluation of the reference loss; the reference's buffers are restored after it. */
        private double lossSum(String side) {
            Tensor value = null;
            RuntimeException failure = null;
            try {
                value = loss.get();
                if (value == null) {
                    throw new IllegalStateException("Reference loss supplier returned null during a"
                        + " finite-difference evaluation (" + side + ")");
                }
                return sum(value, "Reference loss supplier");
            } catch (RuntimeException e) {
                failure = e;
                throw e;
            } finally {
                finish(value, failure);
            }
        }

        private void finish(Tensor value, RuntimeException primary) {
            RuntimeException first = null;
            if (value != null && !keep.contains(value)) {
                try {
                    value.close();
                } catch (RuntimeException e) {
                    first = ParameterGradientCheck.collect(primary, first, e);
                }
            }
            try {
                restoreBuffers();
            } catch (RuntimeException e) {
                first = ParameterGradientCheck.collect(primary, first, e);
            }
            if (first != null) {
                throw first;
            }
        }

        private void restoreBuffers() {
            for (int i = 0; i < buffers.size(); i++) {
                Tensor buffer = buffers.get(i);
                double[] v = bufferValues.get(i);
                if (buffer.dtype() == DType.FLOAT64) {
                    GpuOptimizerOps.storeDoubles(buffer, v);
                } else {
                    GpuOptimizerOps.storeFloats(buffer, narrow(v));
                }
            }
        }
    }

    /** The reference counterpart of a checked parameter: its name, leaf, values and the leaf's strides. */
    private static final class Counterpart {
        private final Tensor leaf;
        private final double[] values;
        private final Strides strides;

        private Counterpart(Tensor leaf, double[] values, Strides strides) {
            this.leaf = leaf;
            this.values = values;
            this.strides = strides;
        }

        static Counterpart of(String name, Parameter p) {
            String label = "'" + name + "' of the reference";
            if (p.dtype() != DType.FLOAT64) {
                throw new IllegalArgumentException("Parameter " + label + " has dtype " + p.dtype()
                    + "; the reference of a float64 check holds FLOAT64 parameters (Module.to(DType.FLOAT64))");
            }
            Tensor t = p.tensor();
            if (t.gradFn() != null || (p.requiresGrad() && !t.requiresGrad())) {
                throw new IllegalStateException("Parameter " + label + " does not read its own leaf (a"
                    + " substitution is open); close the substitution before checking gradients");
            }
            return new Counterpart(t, t.toDoubleArray(), ParameterGradientCheck.requireDistinctElements(t, label));
        }
    }

    /**
     * The reference's parameters by name, after checking that {@code reference} holds {@code module}'s state.
     *
     * @throws IllegalArgumentException when the names, shapes or values of the parameters or buffers differ, or a
     *                                  buffer is neither FLOAT32 nor FLOAT64
     */
    private static Map<String, Parameter> sameState(Module module, Module reference) {
        Map<String, Parameter> own = module.namedParameters();
        Map<String, Parameter> other = reference.namedParameters();
        requireSameNames("parameter", own.keySet(), other.keySet());
        for (Map.Entry<String, Parameter> e : own.entrySet()) {
            requireSameValues("parameter", e.getKey(), e.getValue().tensor(), other.get(e.getKey()).tensor());
        }
        Map<String, Tensor> ownBuffers = module.namedBuffers();
        Map<String, Tensor> otherBuffers = reference.namedBuffers();
        requireSameNames("buffer", ownBuffers.keySet(), otherBuffers.keySet());
        for (Map.Entry<String, Tensor> e : otherBuffers.entrySet()) {
            DType dtype = e.getValue().dtype();
            if (dtype != DType.FLOAT32 && dtype != DType.FLOAT64) {
                throw new IllegalArgumentException("Buffer '" + e.getKey() + "' of the reference has dtype " + dtype
                    + "; gradient checks restore FLOAT32 and FLOAT64 buffers only");
            }
            requireSameValues("buffer", e.getKey(), ownBuffers.get(e.getKey()), e.getValue());
        }
        return other;
    }

    private static void requireSameNames(String kind, Set<String> own, Set<String> other) {
        if (!own.equals(other)) {
            throw new IllegalArgumentException("The reference's " + kind + " names " + other + " differ from the"
                + " module's " + own + "; the reference must be a module of the same architecture");
        }
    }

    private static void requireSameValues(String kind, String name, Tensor own, Tensor other) {
        if (!own.shape().equals(other.shape())) {
            throw new IllegalArgumentException("Reference " + kind + " '" + name + "' has shape " + other.shape()
                + ", the module's has " + own.shape());
        }
        double[] a = doubles(own);
        double[] b = doubles(other);
        for (int i = 0; i < a.length; i++) {
            if (Double.compare(a[i], b[i]) != 0) {
                throw new IllegalArgumentException("Reference " + kind + " '" + name + "' differs from the module's"
                    + " at element " + i + " (" + b[i] + " vs " + a[i] + "); load the module's state into the"
                    + " reference, for example reference.loadStateDict(module.stateDict())");
            }
        }
    }

    // ------------------------------------------------------------------ values

    /** The double sum, in element order, of the FLOAT64 {@code value}. */
    private static double sum(Tensor value, String what) {
        if (value.dtype() != DType.FLOAT64) {
            throw new IllegalStateException(what + " returned a " + value.dtype() + " tensor; a float64 check needs"
                + " the reference to compute in FLOAT64 from FLOAT64 arguments and parameters");
        }
        double total = 0.0;
        for (double v : value.toDoubleArray()) {
            total += v;
        }
        return total;
    }

    /** The values of a FLOAT64 tensor, or of any other tensor through {@link TensorValues#toFloats}, as doubles. */
    private static double[] doubles(Tensor t) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return t.dtype() == DType.FLOAT64 ? t.toDoubleArray() : widen(TensorValues.toFloats(t));
        }
    }

    private static double[] widen(float[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static float[] narrow(double[] values) {
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = (float) values[i];
        }
        return out;
    }

    /** A new FLOAT64 tensor of {@code shape} holding {@code data}. */
    private static Tensor tensor(ComputeBackend backend, double[] data, Shape shape) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    /** Writes {@code value} to storage element {@code address} of the FLOAT64 {@code t}. */
    private static void write(Tensor t, long address, double value) {
        GpuEvent event = t.storage().copyFromHostDouble(new double[]{value}, address, t.backend().transferQueue());
        if (event != null) {
            event.close();
        }
    }
}
