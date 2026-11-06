package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * Live weight updates into the resident device tensors of one GGUF GPU model (vLLM's in-process weight transfer
 * engine, inference side): every {@link #apply} writes new contents over the buffers the model's
 * {@link GpuWeightUploader} holds, without reallocating them, and finishes the model's queue, so the next forward
 * pass serves the update. Tensors the load did not place on the device — a streamed trunk, host-placed layers of a
 * CPU+GPU split, expert tensors of the MoE cache, a split or tensor-parallel model — refuse by name; their bytes
 * come from the model file, not from buffers an update can reach.
 *
 * <p>Updates address tensors by their stored name, with the model-level aliases ({@code lm_head.weight} for
 * {@code output.weight}) resolving like the forward path's lookups. Not thread-safe: run it on the model's own
 * thread, between forward passes.
 */
public final class GpuWeightUpdater implements WeightUpdater {

    private final GpuWeightUploader uploader;
    private final String modelName;
    private String version = DEFAULT_VERSION;
    private boolean updating;
    private boolean shut;

    /** An updater over {@code uploader} of the model {@code modelName} names. */
    public GpuWeightUpdater(GpuWeightUploader uploader, String modelName) {
        this.uploader = Objects.requireNonNull(uploader, "uploader");
        this.modelName = modelName == null || modelName.isEmpty() ? "the model" : modelName;
    }

    @Override
    public void start() {
        ensureAlive();
        updating = true;
    }

    @Override
    public void apply(WeightUpdateTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        ensureAlive();
        uploader.updateTensor(storedName(tensor.name()), tensor.dtype(), tensor.ne(), tensor.dataBuffer());
    }

    @Override
    public void finish(String version) {
        ensureAlive();
        if (version != null) {
            this.version = version;
        }
        updating = false;
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

    /** The stored name of an update target: the name itself, or the model-level alias that resolves. */
    private String storedName(String requested) {
        if (uploader.plan(requested) != null) {
            return requested;
        }
        for (String alias : TensorNames.modelAliases(requested)) {
            if (uploader.plan(alias) != null) {
                return alias;
            }
        }
        throw new IllegalArgumentException(modelName + " has no tensor " + requested + " to update");
    }

    private void ensureAlive() {
        if (shut) {
            throw new IllegalStateException("the weight updater of " + modelName + " is shut down;"
                    + " initialize the weight transfer engine again to update weights");
        }
    }
}
