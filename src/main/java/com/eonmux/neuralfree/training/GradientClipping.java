package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.GpuTensorOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Strides;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * Gradient clipping and gradient scaling shared by {@link Trainer} and the mixed-precision trainer, on the device
 * through {@link GpuOptimizerOps} and {@link GpuTensorOps#scaleInPlace} when the train kernels are available, else on
 * the host. Every gradient update and copy runs under {@link ComputationGraph#noGrad()}, outside the autograd graph, so
 * the in-place device updates are allowed also for a gradient that requires grad (one a create-graph pass delivered).
 *
 * <p><b>Semantics.</b> Both paths implement the trainers' rules. By norm: the global L2 norm is summed over the
 * parameters in order, so a gradient shared by several parameters counts once per parameter, and when the norm
 * exceeds {@code maxNorm} every gradient is multiplied by {@code maxNorm / (norm + 1e-6f)} once. By value: every
 * element is clamped to {@code [-maxValue, maxValue]}, NaN staying NaN. Scaling ({@link #scale}): every gradient is
 * multiplied by the factor once.</p>
 *
 * <p><b>Device path.</b> The device ops update gradients in place, once per distinct view. Autograd may hand one
 * gradient tensor to several parameters, and two parameters may hold different views of one buffer. A parameter whose
 * gradient shares elements with an earlier parameter's gradient without being the same view, or that addresses an
 * element more than once (a broadcast view), first gets its own contiguous device copy through
 * {@link Parameter#setGrad}; that copy is what the host path's fresh per-parameter results amount to. Equal views stay
 * shared, and the in-place update reaches each of them once, as the host path's per-parameter copies do.</p>
 *
 * <p><b>Host path.</b> Taken when the train kernels are unavailable on the parameters' backend, a parameter or gradient
 * is not FLOAT32, or the gradients live on more than one backend (a norm across backends needs the host). Each changed
 * gradient is replaced by a new tensor, once per gradient slot: a parameter listed twice, or two parameters over one
 * tensor (which share its gradient slot), get one replacement, computed from the values read before any replacement.
 * </p>
 */
public final class GradientClipping {

    private GradientClipping() {
    }

    /**
     * Applies the clipping {@code config} enables: by norm when {@link TrainerConfig#gradientClipNorm()} is positive,
     * else by value when {@link TrainerConfig#gradientClipValue()} is positive, else nothing.
     */
    public static void clip(List<Parameter> params, TrainerConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        if (config.gradientClipNorm() > 0) {
            clipByNorm(params, config.gradientClipNorm());
        } else if (config.gradientClipValue() > 0) {
            clipByValue(params, config.gradientClipValue());
        }
    }

    /**
     * Scales the gradients of {@code params} so their global L2 norm is at most {@code maxNorm}; parameters without a
     * gradient are skipped.
     *
     * @throws IllegalArgumentException when {@code maxNorm} is negative or NaN
     */
    public static void clipByNorm(List<Parameter> params, float maxNorm) {
        requireNonNegative(maxNorm, "maxNorm");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            List<Tensor> device = deviceGradients(params);
            if (device != null && GpuOptimizerOps.clipGradNorm(device, maxNorm)) {
                return;
            }
            clipByNormOnHost(withGradients(params), maxNorm);
        }
    }

    /**
     * Clamps every gradient element of {@code params} to {@code [-maxValue, maxValue]}; parameters without a gradient
     * are skipped.
     *
     * @throws IllegalArgumentException when {@code maxValue} is negative or NaN
     */
    public static void clipByValue(List<Parameter> params, float maxValue) {
        requireNonNegative(maxValue, "maxValue");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            List<Tensor> device = deviceGradients(params);
            if (device != null && GpuOptimizerOps.clipGradValue(device, maxValue)) {
                return;
            }
            clipByValueOnHost(withGradients(params), maxValue);
        }
    }

    /**
     * Multiplies the gradient of every parameter of {@code params} that has one by {@code factor}, each gradient once
     * (the mean of accumulated gradients is their sum times {@code 1 / n}). On the device the gradients
     * {@link #deviceGradients} prepares are scaled in place, once per distinct view; on the host each gradient slot
     * gets a new tensor holding {@code g * factor}. Both round each product once. Runs under
     * {@link ComputationGraph#noGrad()}.
     *
     * @throws IllegalArgumentException when {@code factor} is not finite
     */
    public static void scale(List<Parameter> params, float factor) {
        if (!Float.isFinite(factor)) {
            throw new IllegalArgumentException("factor must be finite, got " + factor);
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            List<Tensor> device = deviceGradients(params);
            if (device != null && scaleOnDevice(device, factor)) {
                return;
            }
            List<Parameter> withGrad = withGradients(params);
            replaceEach(withGrad, readGradients(withGrad), data -> {
                for (int i = 0; i < data.length; i++) {
                    data[i] *= factor;
                }
                return data;
            });
        }
    }

    /**
     * Scales each distinct view of {@code grads} in place once; returns false, changing nothing, when the first
     * scaling reports the kernels unavailable.
     */
    private static boolean scaleOnDevice(List<Tensor> grads, float factor) {
        Map<GpuBuffer, List<Tensor>> scaled = new IdentityHashMap<>();
        boolean any = false;
        for (Tensor g : grads) {
            List<Tensor> sameBuffer = scaled.computeIfAbsent(g.storage().gpuBuffer(), b -> new ArrayList<>());
            if (containsView(sameBuffer, g)) {
                continue;
            }
            if (!GpuTensorOps.scaleInPlace(g, factor)) {
                if (!any) {
                    return false;
                }
                throw new IllegalStateException("in-place scaling became unavailable for a FLOAT32 gradient "
                        + g.shape() + " on " + g.backend().backendType() + " after earlier gradients were scaled");
            }
            any = true;
            sameBuffer.add(g);
        }
        return true;
    }

    private static boolean containsView(List<Tensor> views, Tensor g) {
        for (Tensor v : views) {
            if (v == g || GpuOptimizerOps.sameView(v, g)) {
                return true;
            }
        }
        return false;
    }

    /**
     * The gradients of the parameters that have one, in parameter order, ready for the in-place gradient-list ops of
     * {@link GpuOptimizerOps}: every listed tensor is either a view equal to an earlier one or shares no element with
     * any earlier one, and none addresses an element twice (see the class description for the copies this makes).
     * Returns null, changing nothing, when those ops would take their host path: a parameter with a gradient is not
     * eligible for {@link GpuOptimizerOps#shouldUseGpu}, or the gradients live on more than one backend.
     */
    public static List<Tensor> deviceGradients(List<Parameter> params) {
        List<Parameter> withGrad = withGradients(params);
        if (withGrad.isEmpty()) {
            return Collections.emptyList();
        }
        ComputeBackend backend = withGrad.get(0).backend();
        for (Parameter p : withGrad) {
            if (p.backend() != backend || p.grad().backend() != backend || !GpuOptimizerOps.shouldUseGpu(p)) {
                return null;
            }
        }
        List<Tensor> grads = new ArrayList<>(withGrad.size());
        Map<GpuBuffer, List<Tensor>> byBuffer = new IdentityHashMap<>();
        Map<Parameter, Tensor> chosen = new IdentityHashMap<>();
        Map<Parameter, Tensor> copies = new IdentityHashMap<>();
        try {
            for (Parameter p : withGrad) {
                Tensor g = chosen.get(p);
                if (g == null) {
                    g = p.grad();
                    List<Tensor> sameBuffer = byBuffer.computeIfAbsent(g.storage().gpuBuffer(), b -> new ArrayList<>());
                    if (needsOwnCopy(g, sameBuffer)) {
                        g = contiguousCopy(g);
                        copies.put(p, g);
                    }
                    chosen.put(p, g);
                }
                byBuffer.computeIfAbsent(g.storage().gpuBuffer(), b -> new ArrayList<>()).add(g);
                grads.add(g);
            }
        } catch (RuntimeException e) {
            for (Tensor copy : copies.values()) {
                copy.close();
            }
            throw e;
        }
        // Copies are installed only after every original was examined: setGrad closes the gradient it replaces,
        // which another parameter may hold as well.
        for (Map.Entry<Parameter, Tensor> copy : copies.entrySet()) {
            copy.getKey().setGrad(copy.getValue());
        }
        return grads;
    }

    // ------------------------------------------------------------------ device preparation

    /**
     * Whether {@code g} addresses an element twice, or shares elements with an earlier gradient of the same buffer
     * ({@code earlier}) without being the same view.
     */
    private static boolean needsOwnCopy(Tensor g, List<Tensor> earlier) {
        if (addressesAnElementTwice(g)) {
            return true;
        }
        for (Tensor e : earlier) {
            if (!GpuOptimizerOps.sameView(e, g) && GpuOptimizerOps.sharesElements(e, g)) {
                return true;
            }
        }
        return false;
    }

    /** A broadcast view: stride 0 on a dimension longer than 1. */
    private static boolean addressesAnElementTwice(Tensor t) {
        Shape shape = t.shape();
        Strides strides = t.strides();
        for (int k = 0; k < shape.rank(); k++) {
            if (shape.dim(k) > 1 && strides.stride(k) == 0) {
                return true;
            }
        }
        return false;
    }

    /** A dense device copy of the gradient {@code g}, made under {@link ComputationGraph#noGrad()}. */
    private static Tensor contiguousCopy(Tensor g) {
        Tensor copy;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            copy = GpuCoreOps.contiguous(g);
        }
        if (copy == null) {
            throw new IllegalStateException("contiguous device copy unavailable for a FLOAT32 gradient " + g.shape()
                    + " on " + g.backend().backendType() + " although the train kernels are available");
        }
        return copy;
    }

    // ------------------------------------------------------------------ host path

    /**
     * The norm rule on the host: a float sum of squares in parameter order, then, when its square root exceeds
     * {@code maxNorm}, one multiplication of every element by {@code maxNorm / (norm + 1e-6f)} into a new gradient per
     * gradient slot. A parameter listed more than once counts once per listing in the norm, as on the device, and is
     * scaled once.
     */
    private static void clipByNormOnHost(List<Parameter> params, float maxNorm) {
        Map<Tensor, float[]> values = readGradients(params);
        float totalNorm = 0.0f;
        for (Parameter param : params) {
            for (float g : values.get(param.grad())) {
                totalNorm += g * g;
            }
        }
        totalNorm = (float) Math.sqrt(totalNorm);
        if (!(totalNorm > maxNorm)) {
            return;
        }
        float scale = maxNorm / (totalNorm + 1e-6f);
        replaceEach(params, values, data -> {
            for (int i = 0; i < data.length; i++) {
                data[i] *= scale;
            }
            return data;
        });
    }

    /**
     * The value rule on the host: elements above {@code maxValue} become {@code maxValue}, elements below
     * {@code -maxValue} become {@code -maxValue}, into a new gradient per gradient slot.
     */
    private static void clipByValueOnHost(List<Parameter> params, float maxValue) {
        replaceEach(params, readGradients(params), data -> {
            for (int i = 0; i < data.length; i++) {
                if (data[i] > maxValue) {
                    data[i] = maxValue;
                } else if (data[i] < -maxValue) {
                    data[i] = -maxValue;
                }
            }
            return data;
        });
    }

    /**
     * Gives every gradient slot of {@code params} (parameters with a gradient) a new gradient holding
     * {@code update} applied to a copy of its values from {@code values}. A slot is updated once: a parameter whose
     * current gradient was installed by this call (the parameter was listed before, or shares its tensor, and so its
     * slot, with an earlier parameter) is skipped.
     */
    private static void replaceEach(List<Parameter> params, Map<Tensor, float[]> values,
                                    UnaryOperator<float[]> update) {
        Set<Tensor> installed = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Parameter param : params) {
            Tensor grad = param.grad();
            if (installed.contains(grad)) {
                continue;
            }
            replaceGradient(param, grad, update.apply(values.get(grad).clone()));
            installed.add(param.grad());
        }
    }

    /**
     * The values of every distinct gradient tensor of {@code params}, read before any is replaced: replacing a
     * parameter's gradient closes the old tensor ({@link Parameter#setGrad}), which another parameter may also hold.
     */
    private static Map<Tensor, float[]> readGradients(List<Parameter> params) {
        Map<Tensor, float[]> values = new IdentityHashMap<>();
        for (Parameter param : params) {
            values.computeIfAbsent(param.grad(), Tensor::toFloatArray);
        }
        return values;
    }

    /**
     * Gives {@code param} a new gradient tensor holding {@code data} in the shape of {@code grad}; the gradient it
     * replaces is closed by {@link Parameter#setGrad}.
     */
    private static void replaceGradient(Parameter param, Tensor grad, float[] data) {
        Tensor next;
        try (Tensor flat = Tensor.fromArray(grad.backend(), data)) {
            next = flat.reshape(grad.shape());
        }
        try {
            param.setGrad(next);
        } catch (RuntimeException e) {
            next.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------ shared

    private static List<Parameter> withGradients(List<Parameter> params) {
        Objects.requireNonNull(params, "params must not be null");
        List<Parameter> out = new ArrayList<>(params.size());
        for (int i = 0; i < params.size(); i++) {
            Parameter p = Objects.requireNonNull(params.get(i), "parameter " + i + " must not be null");
            if (p.grad() != null) {
                out.add(p);
            }
        }
        return out;
    }

    private static void requireNonNegative(float value, String name) {
        if (!(value >= 0f)) {
            throw new IllegalArgumentException(name + " must be non-negative, got " + value);
        }
    }
}
