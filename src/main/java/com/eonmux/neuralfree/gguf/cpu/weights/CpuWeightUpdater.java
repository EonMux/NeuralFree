package com.eonmux.neuralfree.gguf.cpu.weights;

import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Live weight updates into the loaded tensors of one CPU model (vLLM's in-process weight transfer engine,
 * inference side): every {@link #apply} swaps the view {@link CpuWeights} serves for one tensor. The model's
 * forward pass binds its tensors at load, so {@code rebind} runs after the writes of an update round to rebind it
 * (at {@link #finish} of a started round, directly for a bare apply), and the next forward pass reads the new
 * views. Only tensors the model has mapped take an update; the CPU loads its tensors on first use, so a tensor
 * that never served holds nothing to replace.
 *
 * <p>Updates address tensors by their stored name, with the model-level aliases ({@code output.weight} for a file
 * that stores {@code lm_head.weight}) resolving like the forward path's lookups. Not thread-safe: run it on the
 * model's own thread, between forward passes ({@link CpuWeights} itself stays safe for concurrent readers).
 */
public final class CpuWeightUpdater implements WeightUpdater {

    private final CpuWeights weights;
    private final String modelName;
    /** Rebuilds the model's forward pass over the weights; every write of a round needs it once. */
    private final Runnable rebind;
    private String version = DEFAULT_VERSION;
    private boolean updating;
    private boolean dirty;
    private boolean shut;

    /** An updater over {@code weights} of the model {@code modelName} names; its writes reach a per-call lookup. */
    public CpuWeightUpdater(CpuWeights weights, String modelName) {
        this(weights, modelName, null);
    }

    /**
     * An updater over {@code weights} whose committed updates run {@code rebind} once afterwards, so a forward
     * pass that binds its tensors at load picks them up.
     */
    public CpuWeightUpdater(CpuWeights weights, String modelName, Runnable rebind) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.modelName = modelName == null || modelName.isEmpty() ? "the model" : modelName;
        this.rebind = rebind;
    }

    @Override
    public void start() {
        ensureAlive();
        updating = true;
        dirty = false;
    }

    @Override
    public void apply(WeightUpdateTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        ensureAlive();
        TensorRef target = weights.catalog().get(tensor.name());
        if (target == null) {
            throw new IllegalArgumentException(modelName + " has no tensor " + tensor.name() + " to update");
        }
        weights.update(target.name(), tensor.dtype(), tensor.ne(), tensor.dataBuffer());
        if (updating) {
            dirty = true;
        } else if (rebind != null) {
            rebind.run();
        }
    }

    @Override
    public void finish(String version) {
        ensureAlive();
        if (version != null) {
            this.version = version;
        }
        updating = false;
        rebind();
    }

    @Override
    public void setVersion(String version) {
        Objects.requireNonNull(version, "version");
        ensureAlive();
        this.version = version;
    }

    @Override
    public String version() {
        return version;
    }

    @Override
    public boolean updating() {
        return updating;
    }

    @Override
    public void shutdown() {
        shut = true;
        updating = false;
    }

    @Override
    public void close() {
        shutdown();
    }

    /** Runs the rebind when a round's writes are uncommitted to the forward pass. */
    private void rebind() {
        if (dirty) {
            dirty = false;
            if (rebind != null) {
                rebind.run();
            }
        }
    }

    private void ensureAlive() {
        if (shut) {
            throw new IllegalStateException("the weight updater of " + modelName + " is shut down;"
                    + " initialize the weight transfer engine again to update weights");
        }
    }
}
