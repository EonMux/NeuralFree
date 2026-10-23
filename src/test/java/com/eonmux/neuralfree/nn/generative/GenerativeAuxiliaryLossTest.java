package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;

/**
 * The auxiliary loss contract of {@link GANTrainer} and {@link DDPMTrainer} on the CPU path: each training step adds
 * the auxiliary loss terms its trained model's recorded forward passes leave pending to the loss it back-propagates,
 * clears the terms of the model it does not train, and leaves no term pending. Every scenario runs twice from the same
 * weights and random streams, once with a model whose forward pass leaves {@code c * sum(w * w)} pending and once
 * without; with plain SGD at learning rate {@code lr}, the weights after the step then differ by exactly
 * {@code -lr * k * 2c * w0}, where {@code k} is the number of recorded forward passes of that model whose terms the
 * step's loss collects and {@code w0} the weight before the step. The reported losses are equal, since they leave the
 * auxiliary terms out.
 */
public class GenerativeAuxiliaryLossTest {

    private static final float LR = 0.1f;
    private static final float COEFFICIENT = 0.5f;
    private static final float TOLERANCE = 1e-5f;

    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        TrainGpu.release(host);
    }

    @Test
    public void ganGeneratorStepAddsTheGeneratorsAuxiliaryLoss() {
        GanRun plain = gan(GANTrainer.GANLoss.WGAN, 0f, 0f);
        GanRun auxiliary = gan(GANTrainer.GANLoss.WGAN, COEFFICIENT, 0f);
        assertAuxiliaryUpdate("generator weight", plain.generatorInitial, plain.generatorAfter,
            auxiliary.generatorAfter, 1);
        assertArrayClose("discriminator weight", plain.discriminatorAfter, auxiliary.discriminatorAfter);
        assertEquals("generator loss", plain.generatorLoss, auxiliary.generatorLoss, TOLERANCE);
        assertEquals("discriminator loss", plain.discriminatorLoss, auxiliary.discriminatorLoss, TOLERANCE);
    }

    @Test
    public void ganDiscriminatorStepAddsTheTermsOfEveryDiscriminatorForward() {
        GanRun plain = gan(GANTrainer.GANLoss.WGAN, 0f, 0f);
        GanRun auxiliary = gan(GANTrainer.GANLoss.WGAN, 0f, COEFFICIENT);
        assertAuxiliaryUpdate("discriminator weight", plain.discriminatorInitial, plain.discriminatorAfter,
            auxiliary.discriminatorAfter, 2);
        assertEquals("discriminator loss", plain.discriminatorLoss, auxiliary.discriminatorLoss, TOLERANCE);
    }

    @Test
    public void wganGpDiscriminatorStepAddsTheInterpolatedForwardsTerm() {
        GanRun plain = gan(GANTrainer.GANLoss.WGAN_GP, 0f, 0f);
        GanRun auxiliary = gan(GANTrainer.GANLoss.WGAN_GP, 0f, COEFFICIENT);
        assertAuxiliaryUpdate("discriminator weight", plain.discriminatorInitial, plain.discriminatorAfter,
            auxiliary.discriminatorAfter, 3);
        assertEquals("discriminator loss", plain.discriminatorLoss, auxiliary.discriminatorLoss, TOLERANCE);
    }

    @Test
    public void ddpmStepAddsTheNoisePredictorsAuxiliaryLoss() {
        DdpmRun plain = ddpm(0f);
        DdpmRun auxiliary = ddpm(COEFFICIENT);
        assertAuxiliaryUpdate("noise predictor weight", plain.initial, plain.after, auxiliary.after, 1);
        assertEquals("denoising loss", plain.loss, auxiliary.loss, TOLERANCE);
    }

    /** One GAN step with one discriminator step per generator step; the models' terms must be left cleared. */
    private GanRun gan(GANTrainer.GANLoss lossType, float generatorCoefficient, float discriminatorCoefficient) {
        GanRun run = new GanRun();
        AuxiliaryLinear generator = new AuxiliaryLinear(host, 3, 4, 61, generatorCoefficient);
        AuxiliaryLinear discriminator = new AuxiliaryLinear(host, 4, 1, 62, discriminatorCoefficient);
        SGD genOptimizer = new SGD(generator.parameters(), LR);
        SGD discOptimizer = new SGD(discriminator.parameters(), LR);
        try (GANTrainer trainer = new GANTrainer(generator, discriminator, genOptimizer, discOptimizer, lossType,
                10.0f, 1);
             Tensor real = input(63, 5, 4)) {
            leaveStaleConstant(generator);
            leaveStaleConstant(discriminator);
            run.generatorInitial = generator.weight();
            run.discriminatorInitial = discriminator.weight();
            int nodes = ComputationGraph.current().size();
            GANTrainer.GANStepResult step = trainer.trainStep(real, new Random(64));
            assertEquals("the GAN step leaves no node", nodes, ComputationGraph.current().size());
            assertNull("the GAN step leaves no generator term pending", generator.takeAuxiliaryLoss());
            assertNull("the GAN step leaves no discriminator term pending", discriminator.takeAuxiliaryLoss());
            run.generatorAfter = generator.weight();
            run.discriminatorAfter = discriminator.weight();
            run.generatorLoss = step.generatorLoss();
            run.discriminatorLoss = step.discriminatorLoss();
        } finally {
            genOptimizer.close();
            discOptimizer.close();
            generator.close();
            discriminator.close();
        }
        return run;
    }

    /** One DDPM step; the predictor's terms must be left cleared. */
    private DdpmRun ddpm(float coefficient) {
        DdpmRun run = new DdpmRun();
        AuxiliaryLinear predictor = new AuxiliaryLinear(host, 4, 4, 91, coefficient);
        SGD optimizer = new SGD(predictor.parameters(), LR);
        try (DDPMTrainer trainer = new DDPMTrainer(predictor, new LinearScheduler(host, 10), optimizer, host,
                new Random(92));
             Tensor x0 = input(93, 3, 4)) {
            leaveStaleConstant(predictor);
            run.initial = predictor.weight();
            int nodes = ComputationGraph.current().size();
            DDPMTrainer.TrainingStepResult step = trainer.trainStep(x0);
            try (Tensor predicted = step.predictedNoise(); Tensor noisy = step.noisyInput()) {
                assertEquals("the DDPM step leaves no node", nodes, ComputationGraph.current().size());
                assertNull("the DDPM step leaves no term pending", predictor.takeAuxiliaryLoss());
                run.after = predictor.weight();
                run.loss = step.loss();
            }
        } finally {
            optimizer.close();
            predictor.close();
        }
        return run;
    }

    /**
     * Runs a forward pass of {@code model} under {@link ComputationGraph#noGrad()}, which leaves a constant term
     * pending before the step; the step must clear it rather than add it to its loss.
     */
    private void leaveStaleConstant(AuxiliaryLinear model) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor x = input(65, 2, model.inFeatures());
             Tensor y = model.forward(x)) {
            assertFalse("a no-grad forward records nothing", y.requiresGrad());
        }
    }

    /** {@code withAuxiliary - plain == -lr * k * 2c * initial}, element by element. */
    private static void assertAuxiliaryUpdate(String what, float[] initial, float[] plain, float[] withAuxiliary,
                                              int forwards) {
        assertEquals(what + " size", plain.length, withAuxiliary.length);
        for (int i = 0; i < plain.length; i++) {
            float expected = -LR * forwards * 2f * COEFFICIENT * initial[i];
            assertEquals(what + "[" + i + "] moves by the auxiliary gradient", expected,
                withAuxiliary[i] - plain[i], TOLERANCE);
        }
    }

    private static void assertArrayClose(String what, float[] expected, float[] actual) {
        assertEquals(what + " size", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + "[" + i + "]", expected[i], actual[i], TOLERANCE);
        }
    }

    private Tensor input(long seed, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        try (Tensor flat = Tensor.fromArray(host, TrainParity.uniform(seed, n, -1f, 1f))) {
            return flat.reshape(dims);
        }
    }

    private static final class GanRun {
        float[] generatorInitial;
        float[] discriminatorInitial;
        float[] generatorAfter;
        float[] discriminatorAfter;
        float generatorLoss;
        float discriminatorLoss;
    }

    private static final class DdpmRun {
        float[] initial;
        float[] after;
        float loss;
    }

    /**
     * A {@link Linear} layer whose every forward pass, for a non-zero coefficient {@code c}, leaves the auxiliary loss
     * term {@code c * sum(w * w)} of its weight pending (a constant under {@link ComputationGraph#noGrad()}).
     */
    private static final class AuxiliaryLinear extends Module {
        private final Linear inner;
        private final int inFeatures;
        private final float coefficient;

        AuxiliaryLinear(ComputeBackend backend, int in, int out, long seed, float coefficient) {
            Linear layer = new Linear(in, out);
            layer.initialize(backend, new Random(seed));
            this.inner = registerModule("inner", layer);
            this.inFeatures = in;
            this.coefficient = coefficient;
        }

        int inFeatures() {
            return inFeatures;
        }

        float[] weight() {
            return inner.getParameter("weight").tensor().toFloatArray();
        }

        @Override
        public Tensor forward(Tensor input) {
            if (coefficient != 0f) {
                Tensor w = inner.getParameter("weight").tensor();
                try (Tensor squares = TensorOps.mul(w, w);
                     Tensor total = TensorOps.sum(squares)) {
                    setAuxiliaryLoss(TensorOps.mul(total, coefficient));
                }
            }
            return inner.forward(input);
        }
    }
}
