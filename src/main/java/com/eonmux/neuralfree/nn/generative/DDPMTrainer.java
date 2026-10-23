package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Trainer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Denoising diffusion (DDPM) training and ancestral sampling of a noise predictor. The noising, the predictor's
 * forward, the MSE loss and the reverse steps are {@link TensorOps} compositions on the data's backend (device when the
 * train kernels are available, CPU reference otherwise); the timestep and the Gaussian noise are drawn from the
 * trainer's {@link Random} on the host.
 */
public final class DDPMTrainer implements AutoCloseable {

    /**
     * The result of {@link #trainStep}: the loss value and timestep, and the predicted noise and noisy input as
     * {@link Tensor#detach()} handles taken before the backward pass, so they stay valid after it and do not require
     * grad. The caller owns and closes the two tensors.
     */
    public static final class TrainingStepResult {
        private final float loss;
        private final int timestep;
        private final Tensor predictedNoise;
        private final Tensor noisyInput;

        public TrainingStepResult(float loss, int timestep, Tensor predictedNoise, Tensor noisyInput) {
            this.loss = loss;
            this.timestep = timestep;
            this.predictedNoise = predictedNoise;
            this.noisyInput = noisyInput;
        }

        public float loss() {
            return loss;
        }

        public int timestep() {
            return timestep;
        }

        public Tensor predictedNoise() {
            return predictedNoise;
        }

        public Tensor noisyInput() {
            return noisyInput;
        }

        @Override
        public String toString() {
            return "TrainingStepResult{loss=" + loss + ", timestep=" + timestep + "}";
        }
    }

    public static final class SampleResult {
        private final Tensor samples;
        private final int numSteps;

        public SampleResult(Tensor samples, int numSteps) {
            this.samples = Objects.requireNonNull(samples, "Samples cannot be null");
            this.numSteps = numSteps;
        }

        public Tensor samples() {
            return samples;
        }

        public int numSteps() {
            return numSteps;
        }

        @Override
        public String toString() {
            return "SampleResult{numSteps=" + numSteps + ", shape=" + samples.shape() + "}";
        }
    }

    private final Module noisePredictor;
    private final NoiseScheduler scheduler;
    private final Optimizer optimizer;
    private final ComputeBackend backend;
    private final Random random;

    private int stepCount = 0;
    private float lastLoss = 0.0f;
    private volatile boolean closed = false;

    public DDPMTrainer(Module noisePredictor, NoiseScheduler scheduler,
                       Optimizer optimizer, ComputeBackend backend) {
        this(noisePredictor, scheduler, optimizer, backend, new Random());
    }

    public DDPMTrainer(Module noisePredictor, NoiseScheduler scheduler,
                       Optimizer optimizer, ComputeBackend backend, Random random) {
        Objects.requireNonNull(noisePredictor, "NoisePredictor cannot be null");
        Objects.requireNonNull(scheduler, "Scheduler cannot be null");
        Objects.requireNonNull(optimizer, "Optimizer cannot be null");
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        this.noisePredictor = noisePredictor;
        this.scheduler = scheduler;
        this.optimizer = optimizer;
        this.backend = backend;
        this.random = random;
    }

    /**
     * One denoising training step inside a {@link Trainer.Temporaries}: zeroes the optimizer's gradients, draws
     * a timestep and Gaussian noise from the trainer's {@link Random}, noises {@code xStart},
     * predicts the noise, computes {@code mean((predicted - noise)^2)} on the device, reads the loss value back for
     * the result, back-propagates it plus the auxiliary loss terms the noise predictor's forward pass left pending
     * ({@link Trainer#withAuxiliaryLoss}; terms pending before the step are cleared first, and the step's terms are
     * cleared when it fails), closes the step (the graph returns to its size before the step) and steps the
     * optimizer. The result's loss is the denoising loss without the auxiliary terms.
     *
     * @throws IllegalArgumentException when the prediction does not hold as many elements as {@code xStart}
     * @throws IllegalStateException    when the loss plus the auxiliary terms does not require grad (the noise
     *                                  predictor has no trainable parameter, or grad mode is disabled)
     */
    public TrainingStepResult trainStep(Tensor xStart) {
        ensureOpen();
        Objects.requireNonNull(xStart, "xStart cannot be null");

        optimizer.zeroGrad();
        int timestep = random.nextInt(scheduler.numTimesteps());

        TrainingStepResult result = null;
        try {
            try (Trainer.Temporaries step = new Trainer.Temporaries(noisePredictor)) {
                result = denoisingStep(xStart, timestep, step);
            }
            optimizer.step();
        } catch (RuntimeException e) {
            if (result != null) {
                release(result.predictedNoise());
                release(result.noisyInput());
            }
            throw e;
        }
        lastLoss = result.loss();
        stepCount++;
        return result;
    }

    /**
     * The forward and backward pass of {@link #trainStep}; the caller scopes it with {@code step}, which closes the
     * loss and the tensors made to add the auxiliary terms.
     */
    private TrainingStepResult denoisingStep(Tensor xStart, int timestep, Trainer.Temporaries step) {
        Tensor xNoisy = null;
        Tensor predictedNoise = null;
        Tensor predictedHandle = null;
        Tensor noisyHandle = null;
        try (Tensor noise = Tensor.randn(xStart.backend(), xStart.shape(), random.nextLong())) {
            noisePredictor.clearAuxiliaryLoss();
            xNoisy = scheduler.addNoise(xStart, noise, timestep);
            predictedNoise = noisePredictor.forward(xNoisy);
            predictedHandle = predictedNoise.detach();
            noisyHandle = xNoisy.detach();
            Tensor lossTensor = step.keep(computeMSELoss(predictedNoise, noise));
            float loss = lossTensor.toFloatArray()[0];
            Tensor objective = Trainer.withAuxiliaryLoss(noisePredictor, lossTensor, step);
            if (!objective.requiresGrad()) {
                throw new IllegalStateException("The denoising loss does not require grad: the noise predictor has"
                    + " no trainable parameter, or grad mode is disabled");
            }
            objective.backward();
            return new TrainingStepResult(loss, timestep, predictedHandle, noisyHandle);
        } catch (RuntimeException e) {
            release(predictedHandle);
            release(noisyHandle);
            try {
                noisePredictor.clearAuxiliaryLoss();
            } catch (RuntimeException clearFailure) {
                e.addSuppressed(clearFailure);
            }
            throw e;
        } finally {
            release(predictedNoise, xStart);
            release(xNoisy, xStart);
        }
    }

    /** {@link #sample(Shape, int)} over every timestep of the scheduler. */
    public SampleResult sample(Shape shape) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return sample(shape, scheduler.numTimesteps());
        }
    }

    /**
     * Ancestral sampling from {@code N(0, I)} of {@code shape} over {@code numSteps} evenly spaced timesteps (always
     * ending at 0), under {@link ComputationGraph#noGrad()} (an inference API: nothing is recorded); the samples
     * belong to the caller.
     *
     * @throws IllegalArgumentException when {@code numSteps} is outside {@code [1, numTimesteps]}
     */
    public SampleResult sample(Shape shape, int numSteps) {
        ensureOpen();
        Objects.requireNonNull(shape, "Shape cannot be null");
        if (numSteps <= 0 || numSteps > scheduler.numTimesteps()) {
            throw new IllegalArgumentException(
                "numSteps must be in [1, " + scheduler.numTimesteps() + "], got: " + numSteps);
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return sampleSteps(shape, numSteps);
        }
    }

    private SampleResult sampleSteps(Shape shape, int numSteps) {
        Tensor x = Tensor.randn(backend, shape, random.nextLong());

        int stepRatio = scheduler.numTimesteps() / numSteps;
        List<Integer> timesteps = new ArrayList<>();
        for (int i = scheduler.numTimesteps() - 1; i >= 0; i -= stepRatio) {
            timesteps.add(i);
        }
        if (timesteps.isEmpty() || timesteps.get(timesteps.size() - 1) != 0) {
            timesteps.add(0);
        }

        for (int i = 0; i < timesteps.size(); i++) {
            int t = timesteps.get(i);
            int tPrev = i + 1 < timesteps.size() ? timesteps.get(i + 1) : -1;
            Tensor next;
            try {
                next = denoiseStep(x, t, tPrev);
            } catch (RuntimeException e) {
                x.close();
                throw e;
            }
            x.close();
            x = next;
        }

        return new SampleResult(x, timesteps.size());
    }

    /**
     * One ancestral reverse step from {@code timestep} to {@code previous} ({@code -1} after the last step, where
     * {@code abar_prev = 1}): with {@code alpha = abar_t / abar_prev} and {@code beta = 1 - alpha},
     * {@code x' = (x - beta / sqrt(1 - abar_t) * eps(x)) / sqrt(alpha)}, plus {@code sqrt(var) * z} with
     * {@code var = (1 - abar_prev) / (1 - abar_t) * beta} and {@code z ~ N(0, 1)} drawn from the trainer's
     * {@link Random} when {@code previous >= 0}. Over consecutive timesteps this is the DDPM posterior mean and
     * variance ({@code alpha = alpha_t}); over strided timesteps it is the respaced chain. A new tensor; {@code x}
     * stays with the caller.
     */
    private Tensor denoiseStep(Tensor x, int timestep, int previous) {
        double alphaCumprod = scheduler.getAlphaCumprod(timestep);
        double alphaCumprodPrev = previous >= 0 ? scheduler.getAlphaCumprod(previous) : 1.0;
        double alpha = alphaCumprod / alphaCumprodPrev;
        double beta = 1.0 - alpha;
        float noiseCoeff = (float) (beta / Math.sqrt(1.0 - alphaCumprod));
        float sqrtRecipAlpha = (float) (1.0 / Math.sqrt(alpha));
        float sqrtPosteriorVariance = (float) Math.sqrt(
            Math.max(0.0, (1.0 - alphaCumprodPrev) / (1.0 - alphaCumprod) * beta));

        Tensor mean;
        Tensor predictedNoise = noisePredictor.forward(x);
        try (Tensor predTerm = TensorOps.mul(predictedNoise, noiseCoeff);
             Tensor residual = TensorOps.sub(x, predTerm)) {
            mean = TensorOps.mul(residual, sqrtRecipAlpha);
        } finally {
            release(predictedNoise, x);
        }
        if (previous < 0) {
            return mean;
        }
        try (Tensor noise = Tensor.randn(backend, x.shape(), random.nextLong());
             Tensor noiseTerm = TensorOps.mul(noise, sqrtPosteriorVariance)) {
            return TensorOps.add(mean, noiseTerm);
        } finally {
            mean.close();
        }
    }

    /**
     * {@code mean((predicted - target)^2)} over all elements, with {@code target} viewed in the prediction's shape.
     *
     * @throws IllegalArgumentException when the element counts differ
     */
    private static Tensor computeMSELoss(Tensor predicted, Tensor target) {
        if (predicted.numElements() != target.numElements()) {
            throw new IllegalArgumentException("Predicted noise of shape " + predicted.shape()
                + " does not match the noise of shape " + target.shape());
        }
        try (Tensor targetView = target.reshape(predicted.shape());
             Tensor diff = TensorOps.sub(predicted, targetView);
             Tensor squared = TensorOps.mul(diff, diff)) {
            return TensorOps.mean(squared);
        }
    }

    /** Closes {@code t} unless it is one of {@code keep}. */
    private static void release(Tensor t, Tensor... keep) {
        if (t == null) {
            return;
        }
        for (Tensor k : keep) {
            if (t == k) {
                return;
            }
        }
        t.close();
    }

    public Module noisePredictor() {
        return noisePredictor;
    }

    public NoiseScheduler scheduler() {
        return scheduler;
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public int stepCount() {
        return stepCount;
    }

    public float lastLoss() {
        return lastLoss;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("DDPMTrainer has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
        }
    }

    @Override
    public String toString() {
        return "DDPMTrainer{scheduler=" + scheduler.name()
            + ", numTimesteps=" + scheduler.numTimesteps()
            + ", stepCount=" + stepCount + "}";
    }
}
