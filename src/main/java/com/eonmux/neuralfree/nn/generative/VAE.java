package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.autograd.CheckpointFunction;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * Variational autoencoder base: subclasses supply {@link #encode} (mean and log variance) and {@link #decode}. The
 * reparameterization, the ELBO terms and the KL divergence are recorded {@link TensorOps} compositions on the
 * encoder outputs' backend (device when the train kernels are available, CPU reference otherwise); the
 * reparameterization noise is drawn from the module's {@link Random} on the host through
 * {@link CheckpointFunction#replayable}, so a checkpointed recompute sees the same noise.
 *
 * <p>Training: {@link #forward} in training mode adds {@code beta * KL} to the module's pending auxiliary loss
 * ({@link Module#setAuxiliaryLoss}), so a training loop that adds {@link Module#takeAuxiliaryLoss()} to its
 * reconstruction loss (the {@code Trainer} does) trains the full ELBO; several forwards before one backward pass each
 * add their term. {@link #forwardVAE} sets no auxiliary term; its caller builds the loss with
 * {@link #elboLoss} and back-propagates {@link VAELoss#totalLossTensor()}.</p>
 */
public abstract class VAE extends Module {

    public static final class EncodeResult {
        private final Tensor mean;
        private final Tensor logVar;

        public EncodeResult(Tensor mean, Tensor logVar) {
            this.mean = Objects.requireNonNull(mean, "Mean cannot be null");
            this.logVar = Objects.requireNonNull(logVar, "Log variance cannot be null");
        }

        public Tensor mean() {
            return mean;
        }

        public Tensor logVar() {
            return logVar;
        }

        @Override
        public String toString() {
            return "EncodeResult{mean=" + mean.shape() + ", logVar=" + logVar.shape() + "}";
        }
    }

    /**
     * The ELBO terms of {@link #elboLoss}: the reconstruction loss, the KL divergence and the total
     * {@code reconstruction + beta * kl}, as host floats (read back for the caller) and as scalar
     * tensors. {@link #totalLossTensor()} is the recorded loss to back-propagate; the two term tensors are recorded
     * too. The record owns its three tensors and {@link #close()} closes them (closing is idempotent).
     */
    public static final class VAELoss implements AutoCloseable {
        private final float reconstructionLoss;
        private final float klDivergence;
        private final float totalLoss;
        private final Tensor reconstructionLossTensor;
        private final Tensor klDivergenceTensor;
        private final Tensor totalLossTensor;

        public VAELoss(float reconstructionLoss, float klDivergence, float totalLoss,
                       Tensor reconstructionLossTensor, Tensor klDivergenceTensor, Tensor totalLossTensor) {
            this.reconstructionLoss = reconstructionLoss;
            this.klDivergence = klDivergence;
            this.totalLoss = totalLoss;
            this.reconstructionLossTensor = reconstructionLossTensor;
            this.klDivergenceTensor = klDivergenceTensor;
            this.totalLossTensor = totalLossTensor;
        }

        public float reconstructionLoss() {
            return reconstructionLoss;
        }

        public float klDivergence() {
            return klDivergence;
        }

        public float totalLoss() {
            return totalLoss;
        }

        public Tensor reconstructionLossTensor() {
            return reconstructionLossTensor;
        }

        public Tensor klDivergenceTensor() {
            return klDivergenceTensor;
        }

        /** The recorded {@code reconstruction + beta * kl}, the loss to back-propagate. */
        public Tensor totalLossTensor() {
            return totalLossTensor;
        }

        /** Closes the three loss tensors (those present). */
        @Override
        public void close() {
            closeAll(totalLossTensor, klDivergenceTensor, reconstructionLossTensor);
        }

        @Override
        public String toString() {
            return String.format("VAELoss{recon=%.6f, kl=%.6f, total=%.6f}",
                reconstructionLoss, klDivergence, totalLoss);
        }
    }

    /**
     * The result of {@link #forwardVAE}, owned by the caller. Its tensors are outputs of the recorded graph: they are
     * valid until a loss computed from them is back-propagated, whose backward pass closes them; take
     * {@link Tensor#detach()} handles first of any value read afterwards.
     */
    public static final class VAEResult {
        private final Tensor reconstruction;
        private final Tensor latent;
        private final EncodeResult encodeResult;

        public VAEResult(Tensor reconstruction, Tensor latent, EncodeResult encodeResult) {
            this.reconstruction = Objects.requireNonNull(reconstruction, "Reconstruction cannot be null");
            this.latent = Objects.requireNonNull(latent, "Latent cannot be null");
            this.encodeResult = Objects.requireNonNull(encodeResult, "Encode result cannot be null");
        }

        public Tensor reconstruction() {
            return reconstruction;
        }

        public Tensor latent() {
            return latent;
        }

        public Tensor mean() {
            return encodeResult.mean();
        }

        public Tensor logVar() {
            return encodeResult.logVar();
        }

        public EncodeResult encodeResult() {
            return encodeResult;
        }

        @Override
        public String toString() {
            return "VAEResult{reconstruction=" + reconstruction.shape() +
                ", latent=" + latent.shape() + "}";
        }
    }

    private final int latentDim;
    private final float beta;
    private Random random;
    private volatile boolean closed = false;

    protected VAE(int latentDim) {
        this(latentDim, 1.0f);
    }

    protected VAE(int latentDim, float beta) {
        if (latentDim <= 0) {
            throw new IllegalArgumentException("Latent dimension must be positive, got: " + latentDim);
        }
        if (beta < 0) {
            throw new IllegalArgumentException("Beta must be non-negative, got: " + beta);
        }
        this.latentDim = latentDim;
        this.beta = beta;
        this.random = new Random();
    }

    protected abstract EncodeResult encode(Tensor input);

    protected abstract Tensor decode(Tensor latent);

    /**
     * {@code mean + exp(0.5 * logVar) * epsilon} with {@code epsilon ~ N(0, 1)} drawn from the module's
     * {@link Random} on the host through {@link CheckpointFunction#replayable}; the arithmetic runs
     * on the device when the train kernels are available and is recorded with respect to {@code mean} and
     * {@code logVar}.
     */
    protected Tensor reparameterize(Tensor mean, Tensor logVar) {
        ensureOpen();
        Objects.requireNonNull(mean, "Mean cannot be null");
        Objects.requireNonNull(logVar, "Log variance cannot be null");

        if (!mean.shape().equals(logVar.shape())) {
            throw new IllegalArgumentException("Mean and log variance must have the same shape, got: " +
                mean.shape() + " vs " + logVar.shape());
        }

        Random source = random;
        try (Tensor epsilon = CheckpointFunction.replayable(
                () -> Tensor.randn(mean.backend(), mean.shape(), source.nextLong()));
             Tensor halfLogVar = TensorOps.mul(logVar, 0.5f);
             Tensor std = TensorOps.exp(halfLogVar);
             Tensor noise = TensorOps.mul(std, epsilon)) {
            return TensorOps.add(mean, noise);
        }
    }

    public VAEResult forwardVAE(Tensor input) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");

        EncodeResult encoded = encode(input);
        Tensor latent = null;
        try {
            latent = reparameterize(encoded.mean(), encoded.logVar());
            Tensor reconstruction = decode(latent);
            return new VAEResult(reconstruction, latent, encoded);
        } catch (RuntimeException e) {
            release(latent, input, encoded.mean(), encoded.logVar());
            release(encoded.mean(), input);
            release(encoded.logVar(), input, encoded.mean());
            throw e;
        }
    }

    /**
     * {@link #forwardVAE}'s reconstruction; the latent, mean and log variance are closed. In training mode with a
     * positive {@code beta}, {@code beta * KL(mean, logVar)} is added to the module's pending auxiliary loss
     * ({@link Module#setAuxiliaryLoss}), next to the terms of earlier forwards not yet taken; in eval mode, or with a
     * {@code beta} of 0, nothing is added.
     */
    @Override
    public Tensor forward(Tensor input) {
        VAEResult result = forwardVAE(input);
        Tensor reconstruction = result.reconstruction();
        try {
            if (isTraining() && beta > 0) {
                setAuxiliaryLoss(weightedKl(result.mean(), result.logVar()));
            }
        } catch (RuntimeException e) {
            release(reconstruction, input);
            throw e;
        } finally {
            release(result.latent(), input, reconstruction);
            release(result.mean(), input, reconstruction, result.latent());
            release(result.logVar(), input, reconstruction, result.latent(), result.mean());
        }
        return reconstruction;
    }

    /** {@code beta * KL(mean, logVar)}, recorded. */
    private Tensor weightedKl(Tensor mean, Tensor logVar) {
        try (Tensor kl = computeKLDivergence(mean, logVar)) {
            return TensorOps.mul(kl, beta);
        }
    }

    public VAELoss elboLoss(Tensor input, Tensor reconstruction, Tensor mean, Tensor logVar) {
        return elboLoss(input, reconstruction, mean, logVar, "mse");
    }

    /**
     * The ELBO terms: the reconstruction loss ({@code "bce"}: mean binary cross-entropy of the clamped
     * reconstruction; otherwise mean squared error) and the KL divergence
     * {@code -0.5 * sum(1 + logVar - mean^2 - exp(logVar))} over all elements, both as device scalars, and the
     * recorded total {@code reconstruction + beta * kl}; the float fields are the two scalars read back for the caller
     * and {@code totalLoss = reconstruction + beta * kl} computed on the host. The caller owns the
     * returned record and closes it.
     */
    public VAELoss elboLoss(Tensor input, Tensor reconstruction, Tensor mean, Tensor logVar,
                            String reconstructionLossType) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(reconstruction, "Reconstruction cannot be null");
        Objects.requireNonNull(mean, "Mean cannot be null");
        Objects.requireNonNull(logVar, "Log variance cannot be null");

        Tensor reconLossTensor = "bce".equalsIgnoreCase(reconstructionLossType)
            ? computeBCELoss(input, reconstruction)
            : computeMSELoss(input, reconstruction);
        Tensor klTensor = null;
        Tensor totalTensor = null;
        try {
            klTensor = computeKLDivergence(mean, logVar);
            try (Tensor weightedKl = TensorOps.mul(klTensor, beta)) {
                totalTensor = TensorOps.add(reconLossTensor, weightedKl);
            }
            float reconLoss = reconLossTensor.toFloatArray()[0];
            float klLoss = klTensor.toFloatArray()[0];
            float totalLoss = reconLoss + beta * klLoss;
            return new VAELoss(reconLoss, klLoss, totalLoss, reconLossTensor, klTensor, totalTensor);
        } catch (RuntimeException e) {
            closeAll(totalTensor, klTensor, reconLossTensor);
            throw e;
        }
    }

    /** {@code mean((input - reconstruction)^2)}. */
    private static Tensor computeMSELoss(Tensor input, Tensor reconstruction) {
        try (Tensor diff = TensorOps.sub(input, reconstruction);
             Tensor squared = TensorOps.mul(diff, diff)) {
            return TensorOps.mean(squared);
        }
    }

    /**
     * {@code -mean(x * log(r) + (1 - x) * log(1 - r))} with {@code r} the reconstruction clamped to
     * {@code [1e-7, 1 - 1e-7]}.
     */
    private static Tensor computeBCELoss(Tensor input, Tensor reconstruction) {
        try (Tensor r = TensorOps.clamp(reconstruction, 1e-7f, 1.0f - 1e-7f);
             Tensor logR = TensorOps.log(r);
             Tensor negR = TensorOps.neg(r);
             Tensor oneMinusR = TensorOps.add(negR, 1.0f);
             Tensor logOneMinusR = TensorOps.log(oneMinusR);
             Tensor negX = TensorOps.neg(input);
             Tensor oneMinusX = TensorOps.add(negX, 1.0f);
             Tensor positive = TensorOps.mul(input, logR);
             Tensor negative = TensorOps.mul(oneMinusX, logOneMinusR);
             Tensor terms = TensorOps.add(positive, negative);
             Tensor meanTerm = TensorOps.mean(terms)) {
            return TensorOps.neg(meanTerm);
        }
    }

    /** {@code -0.5 * sum(1 + logVar - mean^2 - exp(logVar))}, the KL divergence from {@code N(0, I)}. */
    private static Tensor computeKLDivergence(Tensor mean, Tensor logVar) {
        try (Tensor meanSquared = TensorOps.mul(mean, mean);
             Tensor variance = TensorOps.exp(logVar);
             Tensor onePlusLogVar = TensorOps.add(logVar, 1.0f);
             Tensor minusMeanSquared = TensorOps.sub(onePlusLogVar, meanSquared);
             Tensor inner = TensorOps.sub(minusMeanSquared, variance);
             Tensor klPerElement = TensorOps.mul(inner, -0.5f)) {
            return TensorOps.sum(klPerElement);
        }
    }

    /** Closes every non-null tensor. */
    private static void closeAll(Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t != null) {
                t.close();
            }
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

    /**
     * Decodes {@code batchSize} latents drawn from {@code N(0, I)} with the module's {@link Random}, in eval mode and
     * under {@link ComputationGraph#noGrad()} (an inference API: nothing is recorded); the training mode is restored
     * afterwards and the samples belong to the caller.
     */
    public Tensor sample(ComputeBackend backend, int batchSize) {
        ensureOpen();
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive, got: " + batchSize);
        }

        Shape latentShape = Shape.of(batchSize, latentDim);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor z = Tensor.randn(backend, latentShape, random.nextLong());
            Tensor samples;
            try {
                samples = decodeForInference(z);
            } catch (RuntimeException e) {
                z.close();
                throw e;
            }
            release(z, samples);
            return samples;
        }
    }

    /**
     * Decodes the latent {@code z} in eval mode under {@link ComputationGraph#noGrad()} (an inference API: nothing is
     * recorded); the training mode is restored afterwards and the samples belong to the caller.
     */
    public Tensor sampleAt(ComputeBackend backend, Tensor z) {
        ensureOpen();
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(z, "Latent vector cannot be null");

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return decodeForInference(z);
        }
    }

    /** {@link #decode} in eval mode, restoring the training mode afterwards. */
    private Tensor decodeForInference(Tensor z) {
        boolean wasTraining = isTraining();
        eval();
        try {
            return decode(z);
        } finally {
            if (wasTraining) {
                train();
            }
        }
    }

    public int latentDim() {
        return latentDim;
    }

    public float beta() {
        return beta;
    }

    public void setRandom(Random random) {
        this.random = Objects.requireNonNull(random, "Random cannot be null");
    }

    public Random getRandom() {
        return random;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("VAE has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "VAE{closed=true}";
        }
        return "VAE{latentDim=" + latentDim + ", beta=" + beta + ", training=" + isTraining() + "}";
    }
}
