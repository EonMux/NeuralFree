package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Objects;

/**
 * A {@link FinetuneOptimizer.Update} on the device: the moments are device tensors next to the parameters, and every
 * step is one {@code TR_AdamStep_F32} (AdamW mode) or {@code TR_SgdStep_F32} launch per parameter on the gradient the
 * backward pass left on the device ({@link GpuOptimizerOps}). Nothing is read back; the host arrays follow through
 * {@link DeviceParameters}.
 *
 * <p>The kernels evaluate the formulas of {@link FinetuneOptimizer} operation by operation, in the same order and
 * without contraction; the square root and the divisions are the device's (correctly rounded on CUDA, within the
 * optimizer tolerance class of the {@code train} contract elsewhere).</p>
 *
 * <p><b>Memory plan.</b> AdamW needs two moments per weight. {@link #open} places them on the device only when the
 * free device memory covers them, the gradients the backward pass allocates and a reserve of
 * {@link #ACTIVATION_RESERVE_FRACTION} of the device memory for the window's activations; otherwise (or when an
 * allocation runs out of memory) it returns null and the caller keeps the host update. SGD has no state.</p>
 */
final class DeviceFinetuneUpdate implements FinetuneOptimizer.Update {

    private static final Logger LOGGER = LogManager.getLogger(DeviceFinetuneUpdate.class);

    /** The share of the device memory the plan leaves free for a window's activations. */
    static final double ACTIVATION_RESERVE_FRACTION = 0.125;

    private final DeviceParameters parameters;
    private final FinetuneOptimizer.Kind kind;
    private final float weightDecay;
    private final Tensor[] firstMoments;
    private final Tensor[] secondMoments;
    private boolean closed;

    private DeviceFinetuneUpdate(DeviceParameters parameters, FinetuneOptimizer.Kind kind, float weightDecay,
                                 Tensor[] firstMoments, Tensor[] secondMoments) {
        this.parameters = parameters;
        this.kind = kind;
        this.weightDecay = weightDecay;
        this.firstMoments = firstMoments;
        this.secondMoments = secondMoments;
    }

    /**
     * The device update of {@code kind} over {@code parameters}, or null when the backend has no {@code train} kernels
     * or the memory plan keeps AdamW's moments on the host.
     */
    static DeviceFinetuneUpdate open(DeviceParameters parameters, ComputeBackend backend, FinetuneOptimizer.Kind kind,
                                     float weightDecay) {
        Objects.requireNonNull(parameters, "parameters");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(kind, "kind");
        if (TrainGpu.forBackend(backend) == null) {
            return null;
        }
        int n = parameters.size();
        if (kind == FinetuneOptimizer.Kind.SGD) {
            return new DeviceFinetuneUpdate(parameters, kind, weightDecay, null, null);
        }
        long weightBytes = 0;
        for (int i = 0; i < n; i++) {
            weightBytes += parameters.tensor(i).numElements() * Float.BYTES;
        }
        long total = backend.capabilities().getTotalVramBytes();
        long needed = 3 * weightBytes + (long) (total * ACTIVATION_RESERVE_FRACTION);
        long free = backend.getFreeVramBytes();
        if (free < needed) {
            LOGGER.warn("AdamW moments stay on the host: the device has {} MiB free, the moments and gradients need"
                    + " {} MiB plus a {} MiB activation reserve", free >> 20, (3 * weightBytes) >> 20,
                    (needed - 3 * weightBytes) >> 20);
            return null;
        }
        Tensor[] first = new Tensor[n];
        Tensor[] second = new Tensor[n];
        try {
            for (int i = 0; i < n; i++) {
                Tensor p = parameters.tensor(i);
                first[i] = Tensor.zeros(backend, p.shape());
                second[i] = Tensor.zeros(backend, p.shape());
            }
        } catch (GpuOutOfMemoryException e) {
            closeAll(first);
            closeAll(second);
            LOGGER.warn("AdamW moments stay on the host: allocating them on the device ran out of memory ({})",
                    e.getMessage());
            return null;
        } catch (RuntimeException e) {
            closeAll(first);
            closeAll(second);
            throw e;
        }
        return new DeviceFinetuneUpdate(parameters, kind, weightDecay, first, second);
    }

    @Override
    public void apply(float learningRate, float biasCorrection1, float biasCorrection2) {
        if (closed) {
            throw new IllegalStateException("the optimizer is closed");
        }
        parameters.syncDevice();
        GpuOptimizerOps.AdamParams adam = kind == FinetuneOptimizer.Kind.ADAMW
                ? GpuOptimizerOps.AdamParams.adamW(learningRate, FinetuneOptimizer.BETA1, FinetuneOptimizer.BETA2,
                        FinetuneOptimizer.EPS, weightDecay, biasCorrection1, biasCorrection2)
                : null;
        GpuOptimizerOps.SgdParams sgd = new GpuOptimizerOps.SgdParams(learningRate, 0f, weightDecay, false);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int i = 0; i < parameters.size(); i++) {
                Tensor p = parameters.tensor(i);
                Tensor g = p.grad();
                Tensor zeros = g == null ? Tensor.zeros(p.backend(), p.shape()) : null;
                try {
                    Tensor grad = g != null ? g : zeros;
                    boolean ran = adam != null
                            ? GpuOptimizerOps.adamStep(p, grad, firstMoments[i], secondMoments[i], adam)
                            : GpuOptimizerOps.sgdStep(p, grad, null, sgd);
                    if (!ran) {
                        throw new IllegalStateException("the train kernels are unavailable on "
                                + p.backend().backendType() + " for the optimizer step of "
                                + parameters.hostParameter(i).name());
                    }
                } finally {
                    if (zeros != null) {
                        zeros.close();
                    }
                }
            }
        }
        parameters.deviceValuesChanged();
    }

    @Override
    public FinetuneOptimizer.Placement placement() {
        return FinetuneOptimizer.Placement.DEVICE;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (firstMoments != null) {
            closeAll(firstMoments);
            closeAll(secondMoments);
        }
    }

    private static void closeAll(Tensor[] tensors) {
        RuntimeException failure = null;
        for (Tensor t : tensors) {
            if (t != null) {
                try {
                    t.close();
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
}
