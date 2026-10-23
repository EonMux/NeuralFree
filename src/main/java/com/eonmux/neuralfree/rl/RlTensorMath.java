package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.GpuTensorOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * Batch math shared by the reinforcement-learning algorithms, composed from {@link TensorOps}: it runs on the train
 * library where the backend has it and on the CPU reference otherwise, and it never reads a tensor back. Arguments
 * are not closed; every returned tensor is new and owned by the caller. The host readbacks the algorithms make are
 * theirs, each at a documented synchronization point (an action, a token or a logged scalar).
 *
 * <p>A host upload waits for the device work already queued, so the compositions here upload nothing:
 * constants are filled on the device ({@link #constant}) and host data such as action indices is uploaded by the
 * callers before their first device operation ({@link #columnIndex}).</p>
 *
 * <p><b>Autograd.</b> The compositions record through {@link TensorOps} like any module math, and every one of them
 * has a finite gradient wherever its value is finite ({@link #rowEntropy} never multiplies a zero gradient by an
 * infinite logarithm). {@link #optimizerStep} is the one update every algorithm runs inside its
 * {@link ComputationGraph.StepScope}: zero the gradients, back-propagate the loss, step. The parameter
 * synchronizations ({@link #copyParameters}, {@link #softUpdate}) are in-place writes and run under
 * {@link ComputationGraph#noGrad()}, as optimizer steps do.</p>
 */
final class RlTensorMath {

    private RlTensorMath() {
    }

    // ------------------------------------------------------------------ uploads of host data

    /** A {@code [rows.length, cols]} FLOAT32 tensor of equal-length host rows (one upload). */
    static Tensor matrix(ComputeBackend backend, float[][] rows) {
        Objects.requireNonNull(rows, "rows must not be null");
        if (rows.length == 0) {
            throw new IllegalArgumentException("a batch needs at least one row");
        }
        return Tensor.fromArray(backend, rows);
    }

    /** A {@code [1, values.length]} FLOAT32 tensor: one sample as a batch of one. */
    static Tensor row(ComputeBackend backend, float[] values) {
        Objects.requireNonNull(values, "values must not be null");
        return Tensor.fromArray(backend, new float[][]{values});
    }

    /** A FLOAT32 condition vector holding {@code 1} where the episode continues and {@code 0} where it ended. */
    static Tensor liveMask(ComputeBackend backend, boolean[] dones) {
        float[] live = new float[dones.length];
        for (int i = 0; i < dones.length; i++) {
            live[i] = dones[i] ? 0.0f : 1.0f;
        }
        return Tensor.fromArray(backend, live);
    }

    /**
     * A FLOAT32 tensor of {@code shape} holding {@code value} everywhere, made without a host upload: a device fill
     * of zeros plus {@code value} ({@code 0 + v == v} exactly for every {@code v} except {@code -0}, which becomes
     * {@code +0}).
     */
    static Tensor constant(ComputeBackend backend, Shape shape, float value) {
        try (Tensor zeros = Tensor.zeros(backend, shape)) {
            return TensorOps.add(zeros, value);
        }
    }

    // ------------------------------------------------------------------ shapes

    /** {@code t} as a rank-1 view of its elements in row-major order. */
    static Tensor flatten(Tensor t) {
        return t.reshape(Shape.vector((int) t.numElements()));
    }

    /** The first element of {@code t} in row-major order, as a {@code [1]} view (a model's scalar output). */
    static Tensor firstElement(Tensor t, String what) {
        if (t.numElements() == 0) {
            throw new IllegalStateException(what + " returned an empty tensor");
        }
        try (Tensor flat = flatten(t)) {
            return flat.slice(0, 0, 1);
        }
    }

    // ------------------------------------------------------------------ selections

    /**
     * The {@code [cols.length, 1]} INT32 index of one column per row, for {@link #pickColumns}; the columns are
     * validated here, on the host, where they come from.
     *
     * @throws IndexOutOfBoundsException when a column is outside {@code [0, k)}
     */
    static Tensor columnIndex(ComputeBackend backend, int[] cols, int k) {
        Objects.requireNonNull(cols, "column indices must not be null");
        if (cols.length == 0) {
            throw new IllegalArgumentException("a batch needs at least one row");
        }
        for (int i = 0; i < cols.length; i++) {
            if (cols[i] < 0 || cols[i] >= k) {
                throw new IndexOutOfBoundsException("index " + cols[i] + " out of range [0, " + k + ") at row " + i);
            }
        }
        try (Tensor flatIndex = Tensor.fromArray(backend, cols)) {
            return flatIndex.reshape(Shape.matrix(cols.length, 1));
        }
    }

    /** {@code out[i] = m[i, index[i, 0]]} for a {@code [n, k]} matrix and a {@link #columnIndex} of {@code n} rows. */
    static Tensor pickColumns(Tensor m, Tensor index) {
        int n = m.shape().dim(0);
        if (!index.shape().equals(Shape.matrix(n, 1))) {
            throw new IllegalArgumentException("expected a [" + n + ", 1] column index, got " + index.shape());
        }
        try (Tensor picked = TensorOps.gather(m, 1, index)) {
            return picked.reshape(Shape.vector(n));
        }
    }

    /** {@code live != 0 ? x : 0} per element ({@code x} is never read where the episode ended, so NaN stays out). */
    static Tensor whereLive(Tensor live, Tensor x) {
        try (Tensor zeros = Tensor.zeros(x.backend(), x.shape())) {
            return TensorOps.where(live, x, zeros);
        }
    }

    // ------------------------------------------------------------------ elementwise compositions

    /** {@code (float) Math.log(Math.max(p, floor))} per element. */
    static Tensor clampedLog(Tensor p, float floor) {
        try (Tensor clamped = TensorOps.clamp(p, floor, Float.POSITIVE_INFINITY)) {
            return TensorOps.log(clamped);
        }
    }

    /**
     * {@code Math.min(a, b)} per element for two tensors of one shape, with Java semantics (NaN when either is NaN,
     * {@code -0} below {@code +0}): the pair is stacked on a new last axis and reduced by the MIN reduction.
     */
    static Tensor pairMin(Tensor a, Tensor b) {
        try (Tensor pair = TensorOps.stack(Arrays.asList(a, b), a.shape().rank())) {
            return TensorOps.min(pair, a.shape().rank());
        }
    }

    /**
     * Row maxima of a {@code [n, k]} matrix that skip NaN entries, as a loop that starts from {@code -inf} and keeps
     * a strictly greater value does: NaN entries are replaced by {@code -inf} before the MAX reduction, so a row of
     * only NaN gives {@code -inf}.
     */
    static Tensor rowMaxSkippingNaN(Tensor m) {
        try (Tensor ordered = TensorOps.eq(m, m);
             Tensor floor = constant(m.backend(), m.shape(), Float.NEGATIVE_INFINITY);
             Tensor cleaned = TensorOps.where(ordered, m, floor)) {
            return TensorOps.max(cleaned, 1);
        }
    }

    /**
     * Per-row entropy {@code -sum(p * log(p))} of a {@code [n, k]} probability matrix over the entries with
     * {@code p > threshold}; the other entries contribute 0. The logarithm reads {@code p > threshold ? p : 1}, so an
     * excluded entry gives {@code p * 0} and its gradient {@code log(1) + p * 0 = 0}: no {@code 0 * -inf} NaN reaches
     * the value or the gradient. An included entry has the gradient {@code -(log(p) + 1)}.
     */
    static Tensor rowEntropy(Tensor probs, float threshold) {
        try (Tensor limit = constant(probs.backend(), Shape.scalar(), threshold);
             Tensor counted = TensorOps.gt(probs, limit);
             Tensor ones = constant(probs.backend(), probs.shape(), 1.0f);
             Tensor safe = TensorOps.where(counted, probs, ones);
             Tensor logP = TensorOps.log(safe);
             Tensor terms = TensorOps.mul(probs, logP);
             Tensor sum = TensorOps.sum(terms, 1)) {
            return TensorOps.neg(sum);
        }
    }

    // ------------------------------------------------------------------ host decisions on read-back vectors

    /** Index of the first strict maximum, starting from element 0 (a NaN at 0 is never replaced). */
    static int firstMaxIndex(float[] values) {
        int best = 0;
        float bestValue = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestValue) {
                bestValue = values[i];
                best = i;
            }
        }
        return best;
    }

    /** The first index whose running probability sum reaches {@code r}, or {@code fallback} when none does. */
    static int sampleIndex(float[] probs, float r, int fallback) {
        float cumulative = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return i;
            }
        }
        return fallback;
    }

    // ------------------------------------------------------------------ optimization

    /**
     * The optimizer {@code factory} builds over {@code params}, or null when {@code factory} is null (the algorithm
     * then evaluates its losses without updating). The algorithm owns the result and closes it.
     *
     * @throws IllegalArgumentException when the factory returns null or an optimizer that does not update every
     *                                  parameter of {@code params}; the optimizer is closed first
     */
    static Optimizer buildOptimizer(Function<List<Parameter>, ? extends Optimizer> factory, List<Parameter> params,
                                    String what) {
        if (factory == null) {
            return null;
        }
        Optimizer optimizer = factory.apply(Collections.unmodifiableList(new ArrayList<>(params)));
        if (optimizer == null) {
            throw new IllegalArgumentException("the " + what + " optimizer factory returned null");
        }
        Set<Parameter> covered = Collections.newSetFromMap(new IdentityHashMap<>());
        covered.addAll(optimizer.parameters());
        for (Parameter p : params) {
            if (!covered.contains(p)) {
                optimizer.close();
                throw new IllegalArgumentException("the " + what + " optimizer factory must build its optimizer over "
                        + "the parameters it is given; parameter '" + p.name() + "' is missing");
            }
        }
        return optimizer;
    }

    /** {@code first}'s parameters followed by {@code second}'s, for one optimizer over both networks. */
    static List<Parameter> parametersOf(Module first, Module second) {
        List<Parameter> params = new ArrayList<>(first.parameters());
        params.addAll(second.parameters());
        return params;
    }

    /**
     * One update: the loss value is read back (one scalar), then {@link #backwardAndStep}. Returns
     * the loss value.
     *
     * @throws IllegalStateException when {@code loss} does not require grad (no trainable parameter reaches it, or
     *                               grad mode is disabled)
     */
    static float optimizerStep(Optimizer optimizer, Tensor loss, String what) {
        float value = loss.toFloatArray()[0];
        backwardAndStep(optimizer, loss, what);
        return value;
    }

    /**
     * One update without a readback: the optimizer's gradients are zeroed, {@code loss} is back-propagated (which
     * releases its graph) and the optimizer steps.
     *
     * @throws IllegalStateException when {@code loss} does not require grad (no trainable parameter reaches it, or
     *                               grad mode is disabled)
     */
    static void backwardAndStep(Optimizer optimizer, Tensor loss, String what) {
        requireTrainable(loss, what);
        optimizer.zeroGrad();
        loss.backward();
        optimizer.step();
    }

    /**
     * Refuses an update whose loss cannot be back-propagated, so a caller can check before it changes any state.
     *
     * @throws IllegalStateException when {@code loss} does not require grad (no trainable parameter reaches it, or
     *                               grad mode is disabled)
     */
    static void requireTrainable(Tensor loss, String what) {
        if (!loss.requiresGrad()) {
            throw new IllegalStateException("the " + what + " loss does not require grad: no trainable parameter "
                    + "reaches it, or grad mode is disabled");
        }
    }

    /** {@code optimizer}'s {@link Optimizer#close()} as a resource for {@link #closeAll}; null for a null optimizer. */
    static AutoCloseable closer(Optimizer optimizer) {
        return optimizer == null ? null : optimizer::close;
    }

    /** Closes every resource that is not null, then rethrows the first failure with the later ones suppressed. */
    static void closeAll(AutoCloseable... resources) {
        RuntimeException failure = null;
        for (AutoCloseable r : resources) {
            if (r == null) {
                continue;
            }
            try {
                r.close();
            } catch (Exception e) {
                RuntimeException wrapped = e instanceof RuntimeException ? (RuntimeException) e
                        : new IllegalStateException("closing " + r + " failed", e);
                if (failure == null) {
                    failure = wrapped;
                } else {
                    failure.addSuppressed(wrapped);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /**
     * {@code loss + aux}, taking {@code loss} over: {@code loss} is the {@code [1]} loss of one forward pass of
     * {@code model} and {@code aux} the sum of the auxiliary loss terms that forward pass left pending
     * ({@link Module#takeAuxiliaryLoss()}), recorded like the loss. Returns {@code loss} itself when none is pending,
     * otherwise a new tensor (and {@code loss} is closed; the graph keeps routing through it by identity).
     */
    static Tensor withAuxiliaryLoss(Module model, Tensor loss) {
        Tensor aux;
        try {
            aux = model.takeAuxiliaryLoss();
        } catch (RuntimeException e) {
            loss.close();
            throw e;
        }
        if (aux == null) {
            return loss;
        }
        try (Tensor base = loss;
             Tensor auxTerm = aux;
             Tensor auxRow = auxTerm.reshape(Shape.vector(1))) {
            return TensorOps.add(base, auxRow);
        }
    }

    // ------------------------------------------------------------------ parameter synchronization

    /**
     * Overwrites every parameter of {@code target} with the same-named parameter of {@code source}, in place and
     * under {@link ComputationGraph#noGrad()}: a device copy when the train library is available, otherwise the CPU
     * state-dict copy. A target parameter shared under several names is written once.
     *
     * @throws IllegalArgumentException when the two modules' parameter names, shapes, dtypes or backends differ
     */
    static void copyParameters(Module source, Module target) {
        Map<String, Parameter> src = source.namedParameters();
        Map<String, Parameter> dst = target.namedParameters();
        requireMatchingParameters(src, dst);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (!onDevice(dst)) {
                target.loadStateDict(source.stateDict());
                return;
            }
            Set<Parameter> written = Collections.newSetFromMap(new IdentityHashMap<>());
            for (Map.Entry<String, Parameter> e : dst.entrySet()) {
                if (written.add(e.getValue())) {
                    GpuCoreOps.copyInto(src.get(e.getKey()).tensor(), e.getValue().tensor());
                }
            }
        }
    }

    /**
     * Polyak averaging {@code target = tau * source + (1 - tau) * target} for every same-named parameter, in place and
     * under {@link ComputationGraph#noGrad()} ({@code tau == 1} copies exactly and {@code tau == 0} leaves the target
     * unchanged). A target parameter shared under several names (weight tying) is averaged once, as
     * {@code namedParameters()} lists it once per name.
     *
     * @throws IllegalArgumentException when {@code tau} is outside {@code [0, 1]} or the parameters do not match
     */
    static void softUpdate(Module source, Module target, float tau) {
        if (!(tau >= 0.0f && tau <= 1.0f)) {
            throw new IllegalArgumentException("tau must be in [0, 1]: " + tau);
        }
        Map<String, Parameter> src = source.namedParameters();
        Map<String, Parameter> dst = target.namedParameters();
        requireMatchingParameters(src, dst);
        if (tau == 0.0f) {
            return;
        }
        if (tau == 1.0f) {
            copyParameters(source, target);
            return;
        }
        float keep = 1.0f - tau;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Set<Parameter> averaged = Collections.newSetFromMap(new IdentityHashMap<>());
            for (Map.Entry<String, Parameter> e : dst.entrySet()) {
                if (!averaged.add(e.getValue())) {
                    continue;
                }
                Tensor t = e.getValue().tensor();
                Tensor s = src.get(e.getKey()).tensor();
                if (!GpuTensorOps.axpbyInPlace(t, tau, s, keep)) {
                    TensorOps.mulInPlace(t, keep);
                    TensorOps.addInPlace(t, s, tau);
                }
            }
        }
    }

    private static void requireMatchingParameters(Map<String, Parameter> src, Map<String, Parameter> dst) {
        if (!src.keySet().equals(dst.keySet())) {
            throw new IllegalArgumentException("parameter names differ: " + src.keySet() + " vs " + dst.keySet());
        }
        for (Map.Entry<String, Parameter> e : dst.entrySet()) {
            Tensor s = src.get(e.getKey()).tensor();
            Tensor t = e.getValue().tensor();
            if (!s.shape().equals(t.shape())) {
                throw new IllegalArgumentException("parameter " + e.getKey() + " has shape " + s.shape()
                        + " in the source and " + t.shape() + " in the target");
            }
            if (s.dtype() != t.dtype()) {
                throw new IllegalArgumentException("parameter " + e.getKey() + " is " + s.dtype()
                        + " in the source and " + t.dtype() + " in the target");
            }
            if (s.backend() != t.backend()) {
                throw new IllegalArgumentException("parameter " + e.getKey() + " lives on another backend in the "
                        + "source than in the target");
            }
        }
    }

    /** Whether every parameter is FLOAT32 on a backend where the train library is available. */
    private static boolean onDevice(Map<String, Parameter> params) {
        for (Parameter p : params.values()) {
            if (p.tensor().dtype() != DType.FLOAT32 || TrainGpu.forBackend(p.tensor().backend()) == null) {
                return false;
            }
        }
        return true;
    }
}
