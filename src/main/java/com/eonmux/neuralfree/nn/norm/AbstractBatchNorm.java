package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.GpuTensorOps;
import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Batch normalization over the channels on dim 1 of an input {@code [B, C, *]}, shared by {@link BatchNorm1d} and
 * {@link BatchNorm2d} (PyTorch's {@code _BatchNorm}); the subclasses fix the accepted input ranks.
 *
 * <p><b>State.</b> With {@code affine}, the parameters {@code weight} (ones) and {@code bias} (zeros); always the
 * persistent buffers {@code running_mean} (zeros), {@code running_var} (ones) and {@code num_batches_tracked} (a
 * scalar zero), registered in that order, so {@link #stateDict()} lists and {@link #loadStateDict} restores all three
 * as PyTorch does.</p>
 *
 * <p><b>Batch counter.</b> {@code num_batches_tracked} is a FLOAT32 scalar rather than PyTorch's int64, because the
 * tensor layer has no integer in-place add on any backend (the device {@code axpby} and the CPU in-place fallbacks are
 * FLOAT32 only, and the train kernel library has no integer arithmetic kernel). It is registered with
 * {@code keepDtype} ({@link Module#registerBuffer(String, Tensor, boolean, boolean)}), so
 * {@link #to(com.eonmux.neuralfree.tensor.DType)} leaves it FLOAT32 as PyTorch leaves its int64 counter. A FLOAT32
 * counter counts every training batch exactly up to {@code 2^24 = 16,777,216}; from there on {@code count + 1} rounds
 * back to {@code count} and the counter stays at {@code 2^24}.</p>
 *
 * <p><b>Running update.</b> {@code running_mean = (1 - f) * running_mean + f * mean} and {@code running_var =
 * (1 - f) * running_var + f * var * n / (n - 1)} with {@code n = B * prod(*)}: the running variance blends in the
 * unbiased batch variance, as PyTorch's {@code _BatchNorm} does, while the batch itself is normalized with the biased
 * variance ({@link NormOps#batchNorm}). The factor is {@code f = momentum}, or, when the momentum is null, {@code f =
 * 1 / num_batches_tracked}: the cumulative moving average of every batch seen so far, PyTorch's
 * {@code momentum=None}. A training batch with a single value per channel ({@code n == 1}) is rejected.</p>
 */
public abstract class AbstractBatchNorm extends Module {
    static final String RUNNING_MEAN = "running_mean";
    static final String RUNNING_VAR = "running_var";
    static final String NUM_BATCHES_TRACKED = "num_batches_tracked";
    static final float DEFAULT_EPSILON = 1e-5f;
    static final float DEFAULT_MOMENTUM = 0.1f;

    private final int numFeatures;
    private final float epsilon;
    private final Float momentum;
    private final boolean affine;

    private Parameter weight;
    private Parameter bias;
    private boolean initialized;

    /**
     * @param numFeatures the channel count {@code C}, positive
     * @param epsilon     added to the variance before the square root, positive and finite
     * @param momentum    the weight of the batch statistic in the running update, in {@code [0, 1]}; null for the
     *                    cumulative moving average
     * @param affine      whether to learn a per-channel scale and shift
     * @throws IllegalArgumentException for an out-of-range argument
     */
    AbstractBatchNorm(int numFeatures, float epsilon, Float momentum, boolean affine) {
        if (numFeatures <= 0) {
            throw new IllegalArgumentException("numFeatures must be positive: " + numFeatures);
        }
        if (!(epsilon > 0) || Float.isInfinite(epsilon)) {
            throw new IllegalArgumentException("epsilon must be positive and finite: " + epsilon);
        }
        if (momentum != null && !(momentum >= 0 && momentum <= 1)) {
            throw new IllegalArgumentException("momentum must be in [0, 1] or null: " + momentum);
        }
        this.numFeatures = numFeatures;
        this.epsilon = epsilon;
        this.momentum = momentum;
        this.affine = affine;
    }

    /**
     * Creates the affine parameters (weight ones, bias zeros), the running statistics (mean zeros, variance ones)
     * and the batch counter (zero) on {@code backend}; runs under no-grad.
     */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (affine) {
                Tensor weightTensor = Tensor.ones(backend, Shape.vector(numFeatures));
                this.weight = registerParameter("weight", new Parameter(weightTensor, "weight"));

                Tensor biasTensor = Tensor.zeros(backend, Shape.vector(numFeatures));
                this.bias = registerParameter("bias", new Parameter(biasTensor, "bias"));
            }

            registerBuffer(RUNNING_MEAN, Tensor.zeros(backend, Shape.vector(numFeatures)));
            registerBuffer(RUNNING_VAR, Tensor.ones(backend, Shape.vector(numFeatures)));
            registerBuffer(NUM_BATCHES_TRACKED, Tensor.zeros(backend, Shape.scalar()), true, true);
        }
        this.initialized = true;
    }

    /**
     * Throws {@link IllegalArgumentException} unless {@code inputShape} is a rank this module accepts with
     * {@link #numFeatures()} channels on dim 1.
     */
    abstract void checkInput(Shape inputShape);

    /**
     * Normalizes {@code input} per channel. In training mode {@code num_batches_tracked} is first incremented by one
     * under no-grad (on the device without a host read when the counter's backend has the train kernels, else on the
     * CPU path), as PyTorch counts the batch before it normalizes; the batch mean and biased variance over the
     * {@code B * prod(*)} elements of each channel then normalize the input, and the running statistics are updated
     * in place with the factor of the class description. In evaluation mode the running statistics normalize it and
     * nothing is updated or counted. The affine step is {@code weight * y + bias}.
     *
     * <p>With a null momentum the factor {@code 1 / num_batches_tracked} is a host scalar argument of
     * {@link NormOps#batchNorm}, so each training forward reads the counter back (a sync point, as PyTorch's
     * {@code float(num_batches_tracked)}); a fixed momentum reads nothing.</p>
     *
     * <p>Computed by {@link NormOps#batchNorm}: on the train kernels when every operand is FLOAT32 on the input's
     * backend (the running statistics then stay on the device), else on its CPU reference, which writes the updated
     * running statistics back into their tensors; the two paths agree within the rounding of a reduction. The
     * running-statistic update and the counter increment are buffer writes outside autograd. The call records one
     * autograd node when grad mode is on and the input or an affine parameter requires grad, so the backward reaches
     * the input, the weight and the bias: through the batch statistics in training mode, and with the running
     * statistics as constants in evaluation mode.</p>
     *
     * @throws IllegalStateException    when the module is not initialized
     * @throws IllegalArgumentException for an input of an unaccepted rank or channel count, or, in training mode, an
     *                                  input with a single value per channel ({@code B * prod(*) == 1})
     */
    @Override
    public final Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        checkInput(input.shape());

        Tensor gamma = affine ? weight.tensor() : null;
        Tensor beta = affine ? bias.tensor() : null;
        Tensor runningMean = buffer(RUNNING_MEAN);
        Tensor runningVar = buffer(RUNNING_VAR);
        if (!isTraining()) {
            return NormOps.batchNorm(input, gamma, beta, runningMean, runningVar, false, 0.0f, epsilon);
        }
        Tensor counter = buffer(NUM_BATCHES_TRACKED);
        incrementCounter(counter);
        float factor = momentum != null ? momentum : cumulativeFactor(counter);
        return NormOps.batchNorm(input, gamma, beta, runningMean, runningVar, true, factor, epsilon);
    }

    /**
     * {@code 1 / num_batches_tracked} (computed in double and rounded once, as PyTorch's {@code 1.0 / float(n)}),
     * read after the increment.
     *
     * @throws IllegalStateException when the count is below 1 or NaN, which only a loaded negative count produces
     */
    private float cumulativeFactor(Tensor counter) {
        float count = readCount(counter);
        if (!(count >= 1.0f)) {
            throw new IllegalStateException(moduleName() + " cannot average over " + count + " batches: '"
                + NUM_BATCHES_TRACKED + "' must hold a non-negative count before a training forward");
        }
        return (float) (1.0 / count);
    }

    /**
     * {@code counter = counter + 1} in place under no-grad: on the device the sum is computed by the scalar-add
     * kernel and written back with {@code axpby} ({@code beta = 0}, so the write is exactly the sum); without the
     * train kernels both steps run on the CPU path and the sum is stored with {@link GpuOptimizerOps#storeFloats}.
     */
    private static void incrementCounter(Tensor counter) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor next = TensorOps.add(counter, 1.0f)) {
            if (!GpuTensorOps.axpbyInPlace(counter, 1.0f, next, 0.0f)) {
                GpuOptimizerOps.storeFloats(counter, next.toFloatArray());
            }
        }
    }

    /** The value of the scalar FLOAT32 counter, read to the host. */
    private static float readCount(Tensor counter) {
        return counter.toFloatArray()[0];
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException(moduleName() + " not initialized. Call initialize(backend) first.");
        }
    }

    public int numFeatures() {
        return numFeatures;
    }

    public float epsilon() {
        return epsilon;
    }

    /**
     * The fixed momentum of the running update.
     *
     * @throws IllegalStateException when the module uses the cumulative moving average ({@link #cumulativeAverage()})
     */
    public float momentum() {
        if (momentum == null) {
            throw new IllegalStateException(moduleName() + " has no fixed momentum: it updates its running"
                + " statistics with the cumulative moving average 1 / num_batches_tracked");
        }
        return momentum;
    }

    /** Whether the running statistics are the cumulative moving average (constructed with a null momentum). */
    public boolean cumulativeAverage() {
        return momentum == null;
    }

    public boolean hasAffine() {
        return affine;
    }

    public Parameter weight() {
        if (!affine) {
            throw new IllegalStateException("This " + moduleName() + " has no affine parameters");
        }
        ensureInitialized();
        return weight;
    }

    public Parameter bias() {
        if (!affine) {
            throw new IllegalStateException("This " + moduleName() + " has no affine parameters");
        }
        ensureInitialized();
        return bias;
    }

    public Tensor runningMean() {
        ensureInitialized();
        return buffer(RUNNING_MEAN);
    }

    public Tensor runningVar() {
        ensureInitialized();
        return buffer(RUNNING_VAR);
    }

    /**
     * The number of training-mode forward passes counted in {@code num_batches_tracked} (restored by
     * {@link #loadStateDict}, exact up to {@code 2^24}), read to the host: a sync point (a metric).
     */
    public int numBatchesTrained() {
        ensureInitialized();
        return (int) readCount(buffer(NUM_BATCHES_TRACKED));
    }

    @Override
    public String toString() {
        return moduleName() + "{" +
            "numFeatures=" + numFeatures +
            ", epsilon=" + epsilon +
            ", momentum=" + momentum +
            ", affine=" + affine +
            ", initialized=" + initialized +
            '}';
    }
}
