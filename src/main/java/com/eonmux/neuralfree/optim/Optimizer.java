package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Base class of the optimizers: holds the parameters, the hyper-parameters ({@link #defaults()}) and the step count.
 *
 * <p><b>State dict.</b> {@link #stateDict()} is a snapshot the caller owns, as in PyTorch: the step count and the
 * hyper-parameters as boxed values, and one new tensor per per-parameter state buffer, keyed by its family prefix
 * followed by the parameter index (for example {@code moment_0}). Each tensor is a detached copy with its own
 * storage, so later steps do not change it and closing it does not affect the optimizer; the caller closes every
 * tensor, for example with {@link #closeStateDict(Map)}. {@link #loadStateDict(Map)} copies values into the live
 * buffers.</p>
 */
public abstract class Optimizer {
    /** State dict key of the step count. */
    public static final String STEP_COUNT_KEY = "step_count";
    /** Hyper-parameter key of the learning rate, which the state dict does not record (it is not fixed). */
    private static final String LR_KEY = "lr";

    protected final List<Parameter> params;
    protected final Map<String, Object> defaults;
    protected int stepCount;
    protected volatile boolean closed = false;

    protected Optimizer(List<Parameter> params, Map<String, Object> defaults) {
        Objects.requireNonNull(params, "Parameters list cannot be null");
        Objects.requireNonNull(defaults, "Defaults map cannot be null");

        this.params = new ArrayList<>(params);
        this.defaults = new HashMap<>(defaults);
        this.stepCount = 0;

        for (int i = 0; i < this.params.size(); i++) {
            if (this.params.get(i) == null) {
                throw new IllegalArgumentException("Parameter at index " + i + " is null");
            }
        }
    }

    public abstract void step();

    public void zeroGrad() {
        ensureOpen();
        for (Parameter param : params) {
            param.zeroGrad();
        }
    }

    public List<Parameter> parameters() {
        ensureOpen();
        return Collections.unmodifiableList(params);
    }

    public Map<String, Object> defaults() {
        ensureOpen();
        return Collections.unmodifiableMap(defaults);
    }

    public int stepCount() {
        ensureOpen();
        return stepCount;
    }

    /**
     * The per-parameter state buffers, keyed by state-dict family prefix (for example {@code "moment_"}), in the
     * order the state dict lists them for each parameter. Element {@code i} of a list belongs to parameter
     * {@code i}; a list shorter than the parameter list, or a null element, means that parameter has no buffer of
     * the family (for example SGD without momentum). The base optimizer keeps none.
     */
    protected Map<String, List<Tensor>> stateBuffers() {
        return Collections.emptyMap();
    }

    /**
     * A snapshot of this optimizer's state that the caller owns: {@value #STEP_COUNT_KEY} as an {@link Integer},
     * every hyper-parameter of {@link #defaults()} except the learning rate as its boxed {@link Number} or
     * {@link Boolean}, and, for each parameter index {@code i} and each state buffer family, a new FLOAT32 tensor
     * {@code <family>i} holding a copy of the buffer (made on the device when the train kernels are available,
     * otherwise through the host). The copies share no storage with the optimizer. The caller closes every tensor
     * value ({@link #closeStateDict(Map)}). An optimizer without parameters returns the step count and the
     * hyper-parameters and no tensors.
     */
    public Map<String, Object> stateDict() {
        ensureOpen();
        Map<String, Object> state = new LinkedHashMap<>();
        state.put(STEP_COUNT_KEY, stepCount);
        for (Map.Entry<String, Object> entry : new TreeMap<>(defaults).entrySet()) {
            Object value = entry.getValue();
            if (!LR_KEY.equals(entry.getKey()) && (value instanceof Number || value instanceof Boolean)) {
                state.put(entry.getKey(), value);
            }
        }
        Map<String, List<Tensor>> buffers = stateBuffers();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int i = 0; i < params.size(); i++) {
                for (Map.Entry<String, List<Tensor>> family : buffers.entrySet()) {
                    Tensor live = OptimizerStates.bufferAt(family.getValue(), i);
                    if (live != null) {
                        state.put(family.getKey() + i, OptimizerStates.snapshot(live));
                    }
                }
            }
        } catch (RuntimeException e) {
            OptimizerStates.closeAfterFailure(state, e);
            throw e;
        }
        return state;
    }

    /**
     * Restores state from a map shaped like {@link #stateDict()}, copying the values in under
     * {@link ComputationGraph#noGrad()}; the map and its tensors stay the caller's. Every entry is validated before
     * anything is written, so a refused map leaves the optimizer unchanged.
     * <ul>
     *   <li>{@value #STEP_COUNT_KEY}: a whole number in [0, 2^31-1], as a {@link Number}, a one-element FLOAT32
     *       tensor or a one-element {@code float[]}.</li>
     *   <li>{@code <family>i}: a FLOAT32 tensor of exactly the buffer's shape (on any backend) or a row-major
     *       {@code float[]} of the buffer's element count.</li>
     *   <li>Hyper-parameters and keys of no family are ignored: hyper-parameters are fixed at construction, and the
     *       learning rate is set with {@link #setLearningRate(float)} or a scheduler.</li>
     * </ul>
     * Buffers whose keys are absent keep their values.
     *
     * @throws NullPointerException     when {@code state} is null
     * @throws IllegalArgumentException when a key is null, the step count is invalid, a key names a buffer this
     *                                  optimizer does not keep (a parameter index past the parameter list, or a family
     *                                  its hyper-parameters do not use), or a buffer value has another type, dtype or
     *                                  shape
     */
    public void loadStateDict(Map<String, ?> state) {
        ensureOpen();
        Objects.requireNonNull(state, "State dict cannot be null");
        String owner = getClass().getSimpleName();
        Integer restoredStep = state.containsKey(STEP_COUNT_KEY)
            ? OptimizerStates.stepCount(state.get(STEP_COUNT_KEY)) : null;
        Map<String, List<Tensor>> buffers = stateBuffers();
        List<Tensor> targets = new ArrayList<>();
        List<Object> values = new ArrayList<>();
        for (Map.Entry<String, ?> entry : state.entrySet()) {
            String key = entry.getKey();
            if (key == null) {
                throw new IllegalArgumentException(owner + " state dict keys cannot be null");
            }
            Tensor target = OptimizerStates.target(owner, buffers, params.size(), key);
            if (target != null) {
                OptimizerStates.requireCompatible(owner, key, target, entry.getValue());
                targets.add(target);
                values.add(entry.getValue());
            }
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int i = 0; i < targets.size(); i++) {
                OptimizerStates.copyInto(targets.get(i), values.get(i));
            }
        }
        if (restoredStep != null) {
            this.stepCount = restoredStep;
        }
    }

    /**
     * Closes every {@link Tensor} value of a state dict returned by {@link #stateDict()}; other values are left
     * alone. Every tensor is closed even when one close fails; the first failure is rethrown with the others
     * suppressed.
     *
     * @throws NullPointerException when {@code state} is null
     */
    public static void closeStateDict(Map<String, ?> state) {
        Objects.requireNonNull(state, "State dict cannot be null");
        RuntimeException failure = null;
        for (Object value : state.values()) {
            if (value instanceof Tensor) {
                try {
                    ((Tensor) value).close();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    public float getLearningRate() {
        Object lr = defaults.get(LR_KEY);
        if (lr instanceof Float) {
            return (Float) lr;
        } else if (lr instanceof Double) {
            return ((Double) lr).floatValue();
        } else if (lr instanceof Number) {
            return ((Number) lr).floatValue();
        }
        throw new IllegalStateException("Learning rate not set or not a valid number");
    }

    /**
     * Sets the learning rate. Zero is accepted, as in PyTorch, so a schedule may anneal to it.
     *
     * @throws IllegalArgumentException when {@code lr} is negative, NaN or infinite
     */
    public void setLearningRate(float lr) {
        ensureOpen();
        if (!(lr >= 0f) || Float.isInfinite(lr)) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative, got: " + lr);
        }
        defaults.put(LR_KEY, lr);
    }

    protected void incrementStep() {
        stepCount++;
    }

    protected void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Optimizer has been closed");
        }
    }

    public void close() {
        if (!closed) {
            closed = true;
            params.clear();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "Optimizer{closed=true}";
        }
        return getClass().getSimpleName() +
               "{params=" + params.size() +
               ", stepCount=" + stepCount +
               ", defaults=" + defaults + '}';
    }
}
