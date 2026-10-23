package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * DDIM sampling with a trained noise predictor: deterministic for {@code eta = 0}, with injected noise drawn from the
 * caller's {@link Random} otherwise. Every step is a {@link TensorOps} composition on the
 * sampler's backend, run under {@link ComputationGraph#noGrad()}: sampling records nothing.
 */
public final class DDIMSampler implements AutoCloseable {

    /**
     * The samples and the state after every step ({@link #intermediates()}, the last of which is {@link #samples()}),
     * all owned by the caller.
     */
    public static final class SamplingResult {
        private final Tensor samples;
        private final int numSteps;
        private final List<Tensor> intermediates;

        public SamplingResult(Tensor samples, int numSteps, List<Tensor> intermediates) {
            this.samples = Objects.requireNonNull(samples, "Samples cannot be null");
            this.numSteps = numSteps;
            this.intermediates = intermediates != null ? intermediates : new ArrayList<>();
        }

        public Tensor samples() {
            return samples;
        }

        public int numSteps() {
            return numSteps;
        }

        public List<Tensor> intermediates() {
            return new ArrayList<>(intermediates);
        }

        @Override
        public String toString() {
            return "SamplingResult{numSteps=" + numSteps + ", shape=" + samples.shape() + "}";
        }
    }

    private final Module noisePredictor;
    private final NoiseScheduler scheduler;
    private final ComputeBackend backend;
    private final float eta;
    private volatile boolean closed = false;

    public DDIMSampler(Module noisePredictor, NoiseScheduler scheduler, ComputeBackend backend) {
        this(noisePredictor, scheduler, backend, 0.0f);
    }

    public DDIMSampler(Module noisePredictor, NoiseScheduler scheduler,
                       ComputeBackend backend, float eta) {
        Objects.requireNonNull(noisePredictor, "NoisePredictor cannot be null");
        Objects.requireNonNull(scheduler, "Scheduler cannot be null");
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (eta < 0 || eta > 1) {
            throw new IllegalArgumentException("eta must be in [0, 1], got: " + eta);
        }
        this.noisePredictor = noisePredictor;
        this.scheduler = scheduler;
        this.backend = backend;
        this.eta = eta;
    }

    /** {@link #sample(Shape, int, Random)} with an unseeded {@link Random}, under {@link ComputationGraph#noGrad()}. */
    public SamplingResult sample(Shape shape, int numSteps) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return sample(shape, numSteps, new Random());
        }
    }

    /**
     * Samples from {@code N(0, I)} of {@code shape} over {@code numSteps} evenly spaced timesteps (always ending at
     * 0), under {@link ComputationGraph#noGrad()}. The initial noise is closed; the state after every step is kept in
     * the result's intermediates.
     *
     * @throws IllegalArgumentException when {@code numSteps} is outside {@code [1, numTimesteps]}
     */
    public SamplingResult sample(Shape shape, int numSteps, Random random) {
        ensureOpen();
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (numSteps <= 0 || numSteps > scheduler.numTimesteps()) {
            throw new IllegalArgumentException(
                "numSteps must be in [1, " + scheduler.numTimesteps() + "], got: " + numSteps);
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return sampleSteps(shape, numSteps, random);
        }
    }

    private SamplingResult sampleSteps(Shape shape, int numSteps, Random random) {
        int stepRatio = scheduler.numTimesteps() / numSteps;
        List<Integer> timesteps = new ArrayList<>();
        for (int i = scheduler.numTimesteps() - 1; i >= 0; i -= stepRatio) {
            timesteps.add(i);
        }
        if (timesteps.isEmpty() || timesteps.get(timesteps.size() - 1) != 0) {
            timesteps.add(0);
        }

        Tensor noise = Tensor.randn(backend, shape, random.nextLong());
        List<Tensor> intermediates = new ArrayList<>();
        try {
            Tensor x = noise;
            for (int i = 0; i < timesteps.size(); i++) {
                int t = timesteps.get(i);
                int tPrev = i < timesteps.size() - 1 ? timesteps.get(i + 1) : -1;
                x = ddimStep(x, t, tPrev, random);
                intermediates.add(x);
            }
            return new SamplingResult(x, timesteps.size(), intermediates);
        } catch (RuntimeException e) {
            for (Tensor t : intermediates) {
                t.close();
            }
            throw e;
        } finally {
            noise.close();
        }
    }

    /**
     * One DDIM step from {@code x} (which stays with the caller) to a new tensor:
     * {@code sqrt(abar_prev) * x0 + sqrt(1 - abar_prev - sigma^2) * eps(x) + sigma * z} with {@code x0} predicted from
     * {@code eps(x)}, {@code sigma = eta * sqrt((1 - abar_prev) / (1 - abar_t)) * sqrt(1 - abar_t / abar_prev)} and
     * {@code abar_prev = alphaCumprod(tPrev)}, or 1 after the last timestep ({@code tPrev = -1}).
     */
    private Tensor ddimStep(Tensor x, int t, int tPrev, Random random) {
        float alphaCumprodT = scheduler.getAlphaCumprod(t);
        float alphaCumprodTPrev = tPrev >= 0 ? scheduler.getAlphaCumprod(tPrev) : 1.0f;
        float sqrtAlphaCumprodTPrev = (float) Math.sqrt(alphaCumprodTPrev);

        float sigma = eta
            * (float) Math.sqrt((1.0 - alphaCumprodTPrev) / (1.0 - alphaCumprodT))
            * (float) Math.sqrt(1.0 - alphaCumprodT / alphaCumprodTPrev);

        float dirCoeff = (float) Math.sqrt(Math.max(0.0, 1.0 - alphaCumprodTPrev - (double) sigma * sigma));

        Tensor predictedNoise = noisePredictor.forward(x);
        Tensor meanWithDir;
        try (Tensor x0Pred = scheduler.predictStartFromNoise(x, t, predictedNoise);
             Tensor dir = TensorOps.mul(predictedNoise, dirCoeff);
             Tensor mean = TensorOps.mul(x0Pred, sqrtAlphaCumprodTPrev)) {
            meanWithDir = TensorOps.add(mean, dir);
        } finally {
            if (predictedNoise != x) {
                predictedNoise.close();
            }
        }

        if (eta > 0 && sigma > 0) {
            try (Tensor noise = Tensor.randn(backend, x.shape(), random.nextLong());
                 Tensor noiseTerm = TensorOps.mul(noise, sigma)) {
                return TensorOps.add(meanWithDir, noiseTerm);
            } finally {
                meanWithDir.close();
            }
        }
        return meanWithDir;
    }

    public Module noisePredictor() {
        return noisePredictor;
    }

    public NoiseScheduler scheduler() {
        return scheduler;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public float eta() {
        return eta;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("DDIMSampler has been closed");
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
        return "DDIMSampler{scheduler=" + scheduler.name()
            + ", eta=" + eta + "}";
    }
}
