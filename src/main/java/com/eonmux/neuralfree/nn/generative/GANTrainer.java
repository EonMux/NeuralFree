package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Trainer;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Adversarial training of a generator against a discriminator. Every loss and the gradient penalty are recorded
 * {@link TensorOps} compositions on the batch's backend (device when the train kernels are available, CPU reference
 * otherwise). The host reads back one scalar per loss for the step result; latent noise and the
 * penalty's interpolation weights come from the {@link Random} passed to {@link #trainStep}.
 *
 * <p>Each discriminator step and the generator step run in their own {@link Trainer.Temporaries} (a
 * {@link ComputationGraph.StepScope} that also closes the step's auxiliary loss tensors): the loss is back-propagated,
 * and whatever the step recorded and the backward pass did not release (such as what a gradient penalty's create-graph
 * pass kept) is removed when the step ends, so the graph returns to its size before the step.
 * The discriminator step generates its fake batch under {@link ComputationGraph#noGrad()} (the generator gets no
 * gradient from it); the generator step back-propagates through the discriminator into the generator, and also
 * leaves gradients in the discriminator's parameters, which the next discriminator step clears before it uses
 * them.</p>
 *
 * <p>WGAN-GP penalises the gradient of the discriminator with respect to its input: with {@code x̂ = α·real +
 * (1 − α)·fake} (one {@code α ~ U[0, 1)} per sample) as a new leaf, {@code g = ∂ sum(D(x̂)) / ∂x̂} is computed by
 * {@link Autograd#grad} with {@code createGraph}, and {@code gp = mean_b((sqrt(Σ g_b² + 1e-8) − 1)²)} over the
 * non-batch elements of each sample enters the loss {@code wgan + λ·gp}, so its gradient reaches the discriminator's
 * parameters through the recorded backward pass. Every gradient function of the discriminator must support
 * create-graph; otherwise the step throws {@link UnsupportedOperationException} before any gradient is computed and
 * leaves the parameters unchanged.</p>
 *
 * <p><b>Auxiliary losses.</b> As in every training loop, the auxiliary loss terms the trained model's forward passes
 * leave pending ({@link Module#takeAuxiliaryLoss()}, such as a mixture-of-experts load-balancing term) are added to
 * its loss before the backward pass ({@link Trainer#withAuxiliaryLoss}): a discriminator step adds the
 * discriminator's terms of every forward pass of the step (real, fake and, for WGAN-GP, the interpolated batch), and
 * the generator step adds the generator's terms. The terms of the model a step does not train are cleared
 * ({@link Module#clearAuxiliaryLoss()}): the generator's constants from the discriminator step's fake batch, and the
 * discriminator's terms from the generator step's forward pass. Each step also clears both models' terms left
 * pending before it, and, when it fails, the terms it left pending. The reported losses are the adversarial losses
 * without the auxiliary terms.</p>
 *
 * <p>Ownership: the generated batch, the discriminator outputs and every loss intermediate are closed within the
 * step.</p>
 */
public final class GANTrainer implements AutoCloseable {

    private static final float BCE_EPS = 1e-7f;

    public enum GANLoss {
        VANILLA,
        WGAN,
        WGAN_GP,
        LSGAN,
        HINGE
    }

    private final Module generator;
    private final Module discriminator;
    private final Optimizer genOptimizer;
    private final Optimizer discOptimizer;
    private final GANLoss lossType;
    private final float gradientPenaltyWeight;
    private final int discStepsPerGenStep;

    private int stepCount = 0;
    private float lastGenLoss = 0.0f;
    private float lastDiscLoss = 0.0f;
    private volatile boolean closed = false;

    public GANTrainer(Module generator, Module discriminator,
                      Optimizer genOptimizer, Optimizer discOptimizer,
                      GANLoss lossType) {
        this(generator, discriminator, genOptimizer, discOptimizer, lossType, 10.0f, 5);
    }

    public GANTrainer(Module generator, Module discriminator,
                      Optimizer genOptimizer, Optimizer discOptimizer,
                      GANLoss lossType, float gradientPenaltyWeight,
                      int discStepsPerGenStep) {
        Objects.requireNonNull(generator, "Generator cannot be null");
        Objects.requireNonNull(discriminator, "Discriminator cannot be null");
        Objects.requireNonNull(genOptimizer, "Generator optimizer cannot be null");
        Objects.requireNonNull(discOptimizer, "Discriminator optimizer cannot be null");
        Objects.requireNonNull(lossType, "Loss type cannot be null");

        if (gradientPenaltyWeight < 0) {
            throw new IllegalArgumentException("Gradient penalty weight must be non-negative");
        }
        if (discStepsPerGenStep < 1) {
            throw new IllegalArgumentException("Discriminator steps per generator step must be at least 1");
        }

        this.generator = generator;
        this.discriminator = discriminator;
        this.genOptimizer = genOptimizer;
        this.discOptimizer = discOptimizer;
        this.lossType = lossType;
        this.gradientPenaltyWeight = gradientPenaltyWeight;
        this.discStepsPerGenStep = discStepsPerGenStep;
    }

    /**
     * {@code discStepsPerGenStep} discriminator steps, then one generator step; each step zeroes its optimizer's
     * gradients, back-propagates its loss plus its model's auxiliary loss terms inside a {@link Trainer.Temporaries}
     * and then steps the optimizer.
     *
     * @throws IllegalArgumentException      when the real batch has rank 0 or no sample
     * @throws IllegalStateException         when a loss does not require grad (no trainable parameter reaches it, or
     *                                       grad mode is disabled)
     * @throws UnsupportedOperationException for WGAN-GP, when a gradient function of the discriminator does not
     *                                       support create-graph
     */
    public GANStepResult trainStep(Tensor realBatch, Random random) {
        ensureOpen();
        Objects.requireNonNull(realBatch, "Real batch cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (realBatch.shape().rank() == 0 || realBatch.shape().dim(0) == 0) {
            throw new IllegalArgumentException("Real batch of shape " + realBatch.shape()
                + " must hold at least one sample along dim 0");
        }

        float totalDiscLoss = 0.0f;
        for (int d = 0; d < discStepsPerGenStep; d++) {
            discOptimizer.zeroGrad();
            float discLoss = trainDiscriminator(realBatch, random);
            totalDiscLoss += discLoss;
            discOptimizer.step();
        }
        lastDiscLoss = totalDiscLoss / discStepsPerGenStep;

        genOptimizer.zeroGrad();
        lastGenLoss = trainGenerator(realBatch, random);
        genOptimizer.step();

        stepCount++;

        return new GANStepResult(lastGenLoss, lastDiscLoss, stepCount);
    }

    private float trainDiscriminator(Tensor realBatch, Random random) {
        try (Trainer.Temporaries step = new Trainer.Temporaries()) {
            Tensor fakeBatch = null;
            Tensor realOutput = null;
            Tensor fakeOutput = null;
            Tensor interpolated = null;
            Tensor loss = null;
            try {
                discriminator.clearAuxiliaryLoss();
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    fakeBatch = generateFake(realBatch, random);
                }
                generator.clearAuxiliaryLoss();
                realOutput = discriminator.forward(realBatch);
                fakeOutput = discriminator.forward(fakeBatch);
                if (lossType == GANLoss.WGAN_GP) {
                    interpolated = interpolate(realBatch, fakeBatch, random);
                    loss = wganGpLoss(realOutput, fakeOutput, interpolated);
                } else {
                    loss = discriminatorLoss(realOutput, fakeOutput);
                }
                return readAndBackward(discriminator, loss, step, "discriminator");
            } catch (RuntimeException e) {
                clearAuxiliaryLosses(e);
                throw e;
            } finally {
                release(loss);
                release(fakeOutput, realBatch);
                release(realOutput, realBatch);
                release(fakeBatch, realBatch);
                releaseLeaf(interpolated);
            }
        }
    }

    private float trainGenerator(Tensor realBatch, Random random) {
        try (Trainer.Temporaries step = new Trainer.Temporaries()) {
            Tensor fakeBatch = null;
            Tensor fakeOutput = null;
            Tensor loss = null;
            try {
                generator.clearAuxiliaryLoss();
                discriminator.clearAuxiliaryLoss();
                fakeBatch = generateFake(realBatch, random);
                fakeOutput = discriminator.forward(fakeBatch);
                discriminator.clearAuxiliaryLoss();
                loss = generatorLoss(fakeOutput);
                return readAndBackward(generator, loss, step, "generator");
            } catch (RuntimeException e) {
                clearAuxiliaryLosses(e);
                throw e;
            } finally {
                release(loss);
                release(fakeOutput, realBatch);
                release(fakeBatch, realBatch);
            }
        }
    }

    /**
     * The loss value for the step result (one scalar read, without auxiliary terms), then {@code backward()} of the
     * loss plus {@code model}'s pending auxiliary loss terms ({@link Trainer#withAuxiliaryLoss}); the tensors made
     * for the sum are kept in {@code step}.
     *
     * @throws IllegalStateException when the loss plus the auxiliary terms does not require grad
     */
    private static float readAndBackward(Module model, Tensor loss, Trainer.Temporaries step, String name) {
        float value = loss.toFloatArray()[0];
        Tensor objective = Trainer.withAuxiliaryLoss(model, loss, step);
        if (!objective.requiresGrad()) {
            throw new IllegalStateException("The " + name + " loss does not require grad: no trainable parameter"
                + " reaches it, or grad mode is disabled");
        }
        objective.backward();
        return value;
    }

    /**
     * Clears the pending auxiliary loss terms of both models after a failed step, adding any failure to
     * {@code failure} as suppressed.
     */
    private void clearAuxiliaryLosses(RuntimeException failure) {
        for (Module model : Arrays.asList(discriminator, generator)) {
            try {
                model.clearAuxiliaryLoss();
            } catch (RuntimeException e) {
                failure.addSuppressed(e);
            }
        }
    }

    /** The discriminator loss of every loss type except WGAN-GP ({@link #wganGpLoss}). */
    private Tensor discriminatorLoss(Tensor realOutput, Tensor fakeOutput) {
        switch (lossType) {
            case VANILLA:
                return computeVanillaDiscLoss(realOutput, fakeOutput);
            case WGAN:
                return computeWganDiscLoss(realOutput, fakeOutput);
            case LSGAN:
                return computeLsganDiscLoss(realOutput, fakeOutput);
            case HINGE:
                return computeHingeDiscLoss(realOutput, fakeOutput);
            default:
                throw new IllegalStateException("Unknown loss type: " + lossType);
        }
    }

    private Tensor generatorLoss(Tensor fakeOutput) {
        switch (lossType) {
            case VANILLA:
                return bceLoss(fakeOutput, true);
            case WGAN:
            case WGAN_GP:
            case HINGE:
                try (Tensor mean = TensorOps.mean(fakeOutput)) {
                    return TensorOps.neg(mean);
                }
            case LSGAN:
                try (Tensor mse = mseLoss(fakeOutput, 1.0f)) {
                    return TensorOps.mul(mse, 0.5f);
                }
            default:
                throw new IllegalStateException("Unknown loss type: " + lossType);
        }
    }

    private Tensor generateFake(Tensor realBatch, Random random) {
        int batchSize = realBatch.shape().dim(0);
        int latentDim = getLatentDim();

        ComputeBackend backend = realBatch.backend();
        Tensor latent = Tensor.randn(backend, Shape.matrix(batchSize, latentDim), random.nextLong());
        Tensor fake;
        try {
            fake = generator.forward(latent);
        } catch (RuntimeException e) {
            latent.close();
            throw e;
        }
        release(latent, fake);
        return fake;
    }

    private int getLatentDim() {
        List<Parameter> genParams = generator.parameters();
        if (genParams.isEmpty()) {
            return 100;
        }

        Parameter firstParam = genParams.get(0);
        int[] shape = firstParam.shape().dims();
        if (shape.length >= 2) {
            return shape[1];
        }
        return 100;
    }

    /** {@code BCE(real, 1) + BCE(fake, 0)}. */
    private static Tensor computeVanillaDiscLoss(Tensor realOutput, Tensor fakeOutput) {
        try (Tensor realLoss = bceLoss(realOutput, true);
             Tensor fakeLoss = bceLoss(fakeOutput, false)) {
            return TensorOps.add(realLoss, fakeLoss);
        }
    }

    /** {@code mean(fake) + mean(-real)}. */
    private static Tensor computeWganDiscLoss(Tensor realOutput, Tensor fakeOutput) {
        try (Tensor negReal = TensorOps.neg(realOutput);
             Tensor fakeLoss = TensorOps.mean(fakeOutput);
             Tensor realLoss = TensorOps.mean(negReal)) {
            return TensorOps.add(fakeLoss, realLoss);
        }
    }

    /** {@code (MSE(real, 1) + MSE(fake, 0)) * 0.5}. */
    private static Tensor computeLsganDiscLoss(Tensor realOutput, Tensor fakeOutput) {
        try (Tensor realLoss = mseLoss(realOutput, 1.0f);
             Tensor fakeLoss = mseLoss(fakeOutput, 0.0f);
             Tensor sum = TensorOps.add(realLoss, fakeLoss)) {
            return TensorOps.mul(sum, 0.5f);
        }
    }

    /**
     * {@code mean(clamp(1 - real, 0, MAX)) + mean(clamp(1 + fake, 0, MAX))}, the hinge loss
     * {@code E[relu(1 - D(x))] + E[relu(1 + D(G(z)))]}.
     */
    private static Tensor computeHingeDiscLoss(Tensor realOutput, Tensor fakeOutput) {
        try (Tensor negReal = TensorOps.neg(realOutput);
             Tensor realMargin = TensorOps.add(negReal, 1.0f);
             Tensor realPart = TensorOps.clamp(realMargin, 0.0f, Float.MAX_VALUE);
             Tensor realLoss = TensorOps.mean(realPart);
             Tensor fakeMargin = TensorOps.add(fakeOutput, 1.0f);
             Tensor fakePart = TensorOps.clamp(fakeMargin, 0.0f, Float.MAX_VALUE);
             Tensor fakeLoss = TensorOps.mean(fakePart)) {
            return TensorOps.add(realLoss, fakeLoss);
        }
    }

    /** {@code wgan + gradientPenaltyWeight * gp} (class description). */
    private Tensor wganGpLoss(Tensor realOutput, Tensor fakeOutput, Tensor interpolated) {
        try (Tensor wgan = computeWganDiscLoss(realOutput, fakeOutput);
             Tensor gp = gradientPenalty(interpolated);
             Tensor weighted = TensorOps.mul(gp, gradientPenaltyWeight)) {
            return TensorOps.add(wgan, weighted);
        }
    }

    /**
     * {@code α·real + (1 − α)·fake} with one {@code α} per sample drawn from {@code random} on the host,
     * broadcast over the sample's elements and computed under {@link ComputationGraph#noGrad()}, returned as a new
     * leaf that requires grad. The caller releases it with {@link #releaseLeaf}.
     *
     * @throws IllegalArgumentException when the generated batch does not hold the real batch's element count
     */
    private static Tensor interpolate(Tensor realBatch, Tensor fakeBatch, Random random) {
        Shape shape = realBatch.shape();
        if (fakeBatch.numElements() != realBatch.numElements()) {
            throw new IllegalArgumentException("Generated batch of shape " + fakeBatch.shape()
                + " does not match the real batch of shape " + shape);
        }
        int batchSize = shape.dim(0);
        float[] alpha = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            alpha[i] = random.nextFloat();
        }
        int[] alphaDims = new int[shape.rank()];
        Arrays.fill(alphaDims, 1);
        alphaDims[0] = batchSize;
        Tensor mixed;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor alphaFlat = Tensor.fromArray(realBatch.backend(), alpha);
             Tensor alphaT = alphaFlat.reshape(alphaDims);
             Tensor fake = fakeBatch.reshape(shape);
             Tensor negAlpha = TensorOps.neg(alphaT);
             Tensor oneMinusAlpha = TensorOps.add(negAlpha, 1.0f);
             Tensor realPart = TensorOps.mul(alphaT, realBatch);
             Tensor fakePart = TensorOps.mul(oneMinusAlpha, fake)) {
            mixed = TensorOps.add(realPart, fakePart);
        }
        Tensor leaf = mixed.withRequiresGrad(true);
        if (leaf != mixed) {
            mixed.close();
        }
        return leaf;
    }

    /**
     * {@code mean_b((sqrt(Σ_{non-batch} g_b² + 1e-8) − 1)²)} with {@code g = ∂ sum(D(x̂)) / ∂x̂} from a create-graph
     * pass ({@link Autograd#grad}), so the penalty is recorded with respect to the discriminator's parameters; the
     * {@code 1e-8} keeps the square root differentiable at a zero gradient. A discriminator output that does not
     * depend on {@code x̂} gives {@code g = 0}.
     */
    private Tensor gradientPenalty(Tensor interpolated) {
        Tensor gradient;
        Tensor output = discriminator.forward(interpolated);
        try (Tensor total = TensorOps.sum(output)) {
            gradient = Autograd.grad(total, Collections.singletonList(interpolated), null, true, true).get(0);
        } finally {
            release(output, interpolated);
        }
        if (gradient == null) {
            gradient = Tensor.zeros(interpolated.backend(), interpolated.shape());
        }
        int batchSize = interpolated.shape().dim(0);
        int perSample = (int) (interpolated.numElements() / batchSize);
        try (Tensor g = gradient;
             Tensor rows = g.reshape(batchSize, perSample);
             Tensor squares = TensorOps.mul(rows, rows);
             Tensor sumSquares = TensorOps.sum(squares, 1);
             Tensor shifted = TensorOps.add(sumSquares, 1e-8f);
             Tensor norm = TensorOps.sqrt(shifted);
             Tensor deviation = TensorOps.add(norm, -1.0f);
             Tensor squaredDeviation = TensorOps.mul(deviation, deviation)) {
            return TensorOps.mean(squaredDeviation);
        }
    }

    /**
     * Clears and closes the gradient the loss's backward pass accumulated into the interpolation leaf, then closes
     * the leaf; nothing for null.
     */
    private static void releaseLeaf(Tensor leaf) {
        if (leaf == null) {
            return;
        }
        try {
            Tensor grad = leaf.grad();
            if (grad != null) {
                leaf.setGrad(null);
                grad.close();
            }
        } finally {
            leaf.close();
        }
    }

    /**
     * {@code mean(BCE(clamp(p, 1e-7, 1 - 1e-7), t))} for an all-one ({@code -log p}) or all-zero
     * ({@code -log(1 - p)}) target.
     */
    private static Tensor bceLoss(Tensor prediction, boolean targetOne) {
        try (Tensor p = TensorOps.clamp(prediction, BCE_EPS, 1.0f - BCE_EPS)) {
            if (targetOne) {
                return meanNegLog(p);
            }
            try (Tensor negP = TensorOps.neg(p);
                 Tensor oneMinusP = TensorOps.add(negP, 1.0f)) {
                return meanNegLog(oneMinusP);
            }
        }
    }

    /** {@code mean(-log(x))}. */
    private static Tensor meanNegLog(Tensor x) {
        try (Tensor logX = TensorOps.log(x);
             Tensor negLog = TensorOps.neg(logX)) {
            return TensorOps.mean(negLog);
        }
    }

    /** {@code mean((prediction - target)^2)} against a constant target. */
    private static Tensor mseLoss(Tensor prediction, float target) {
        try (Tensor diff = TensorOps.add(prediction, -target);
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

    /**
     * The generator's output for {@code latent} in eval mode, computed under {@link ComputationGraph#noGrad()} (an
     * inference API: nothing is recorded and the result does not require grad); the generator's training mode is
     * restored afterwards.
     */
    public Tensor generate(Tensor latent) {
        ensureOpen();
        Objects.requireNonNull(latent, "Latent cannot be null");

        boolean wasTraining = generator.isTraining();
        generator.eval();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return generator.forward(latent);
        } finally {
            if (wasTraining) {
                generator.train();
            }
        }
    }

    public Module generator() {
        ensureOpen();
        return generator;
    }

    public Module discriminator() {
        ensureOpen();
        return discriminator;
    }

    public GANLoss lossType() {
        return lossType;
    }

    public int stepCount() {
        return stepCount;
    }

    public float lastGeneratorLoss() {
        return lastGenLoss;
    }

    public float lastDiscriminatorLoss() {
        return lastDiscLoss;
    }

    public float gradientPenaltyWeight() {
        return gradientPenaltyWeight;
    }

    public int discriminatorStepsPerGeneratorStep() {
        return discStepsPerGenStep;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GANTrainer has been closed");
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
        return "GANTrainer{" +
            "lossType=" + lossType +
            ", stepCount=" + stepCount +
            ", lastGenLoss=" + lastGenLoss +
            ", lastDiscLoss=" + lastDiscLoss +
            ", closed=" + closed +
            '}';
    }

    public static final class GANStepResult {
        private final float generatorLoss;
        private final float discriminatorLoss;
        private final int stepCount;

        public GANStepResult(float generatorLoss, float discriminatorLoss, int stepCount) {
            this.generatorLoss = generatorLoss;
            this.discriminatorLoss = discriminatorLoss;
            this.stepCount = stepCount;
        }

        public float generatorLoss() {
            return generatorLoss;
        }

        public float discriminatorLoss() {
            return discriminatorLoss;
        }

        public int stepCount() {
            return stepCount;
        }

        @Override
        public String toString() {
            return "GANStepResult{" +
                "generatorLoss=" + generatorLoss +
                ", discriminatorLoss=" + discriminatorLoss +
                ", stepCount=" + stepCount +
                '}';
        }
    }
}
