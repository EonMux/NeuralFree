package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.Assert.*;

public class GenerativeTest {

    private ComputeBackend backend;
    private boolean openCLAvailable = false;

    @Before
    public void setUp() {
        openCLAvailable = false;
    }

    @Test
    public void testGANTrainerCreation() {
        Module generator = new MockGenerator(100, 10);
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        GANTrainer trainer = new GANTrainer(generator, discriminator, genOptim, discOptim, GANTrainer.GANLoss.VANILLA);
        assertNotNull(trainer);
        assertEquals(GANTrainer.GANLoss.VANILLA, trainer.lossType());
        assertEquals(0, trainer.stepCount());
    }

    @Test
    public void testGANTrainerWithCustomParams() {
        Module generator = new MockGenerator(100, 10);
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        GANTrainer trainer = new GANTrainer(generator, discriminator, genOptim, discOptim,
            GANTrainer.GANLoss.WGAN_GP, 5.0f, 3);
        assertNotNull(trainer);
        assertEquals(GANTrainer.GANLoss.WGAN_GP, trainer.lossType());
        assertEquals(5.0f, trainer.gradientPenaltyWeight(), 1e-6f);
        assertEquals(3, trainer.discriminatorStepsPerGeneratorStep());
    }

    @Test
    public void testGANTrainerNullGenerator() {
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(new ArrayList<>(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        try {
            new GANTrainer(null, discriminator, genOptim, discOptim, GANTrainer.GANLoss.VANILLA);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Generator"));
        }
    }

    @Test
    public void testGANTrainerNullDiscriminator() {
        Module generator = new MockGenerator(100, 10);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(new ArrayList<>(), 0.001f);

        try {
            new GANTrainer(generator, null, genOptim, discOptim, GANTrainer.GANLoss.VANILLA);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Discriminator"));
        }
    }

    @Test
    public void testGANTrainerNegativeGradientPenalty() {
        Module generator = new MockGenerator(100, 10);
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        try {
            new GANTrainer(generator, discriminator, genOptim, discOptim,
                GANTrainer.GANLoss.WGAN_GP, -1.0f, 5);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Gradient penalty"));
        }
    }

    @Test
    public void testGANTrainerInvalidDiscSteps() {
        Module generator = new MockGenerator(100, 10);
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        try {
            new GANTrainer(generator, discriminator, genOptim, discOptim,
                GANTrainer.GANLoss.VANILLA, 10.0f, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Discriminator steps"));
        }
    }

    @Test
    public void testGANTrainerAccessors() {
        Module generator = new MockGenerator(100, 10);
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        GANTrainer trainer = new GANTrainer(generator, discriminator, genOptim, discOptim, GANTrainer.GANLoss.LSGAN);

        assertEquals(generator, trainer.generator());
        assertEquals(discriminator, trainer.discriminator());
        assertEquals(GANTrainer.GANLoss.LSGAN, trainer.lossType());
    }

    @Test
    public void testGANTrainerStepResult() {
        GANTrainer.GANStepResult result = new GANTrainer.GANStepResult(0.5f, 0.3f, 1);

        assertEquals(0.5f, result.generatorLoss(), 1e-6f);
        assertEquals(0.3f, result.discriminatorLoss(), 1e-6f);
        assertEquals(1, result.stepCount());
        assertNotNull(result.toString());
    }

    @Test
    public void testGANTrainerToString() {
        Module generator = new MockGenerator(100, 10);
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        GANTrainer trainer = new GANTrainer(generator, discriminator, genOptim, discOptim, GANTrainer.GANLoss.VANILLA);
        String str = trainer.toString();
        assertTrue(str.contains("GANTrainer"));
        assertTrue(str.contains("VANILLA"));
    }

    @Test
    public void testGANLossEnum() {
        assertEquals(5, GANTrainer.GANLoss.values().length);
        assertNotNull(GANTrainer.GANLoss.valueOf("VANILLA"));
        assertNotNull(GANTrainer.GANLoss.valueOf("WGAN"));
        assertNotNull(GANTrainer.GANLoss.valueOf("WGAN_GP"));
        assertNotNull(GANTrainer.GANLoss.valueOf("LSGAN"));
        assertNotNull(GANTrainer.GANLoss.valueOf("HINGE"));
    }

    @Test
    public void testSpectralNormCreation() {
        MockLinearModule linear = new MockLinearModule(10, 5);
        SpectralNorm sn = new SpectralNorm(linear);

        assertNotNull(sn);
        assertEquals(linear, sn.wrappedModule());
        assertEquals(1.0f, sn.normCoeff(), 1e-6f);
        assertEquals(1, sn.numPowerIterations());
    }

    @Test
    public void testSpectralNormWithCustomParams() {
        MockLinearModule linear = new MockLinearModule(10, 5);
        SpectralNorm sn = new SpectralNorm(linear, 0.9f, 5);

        assertNotNull(sn);
        assertEquals(0.9f, sn.normCoeff(), 1e-6f);
        assertEquals(5, sn.numPowerIterations());
    }

    @Test
    public void testSpectralNormNullModule() {
        try {
            new SpectralNorm(null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Wrapped module"));
        }
    }

    @Test
    public void testSpectralNormNegativeNormCoeff() {
        MockLinearModule linear = new MockLinearModule(10, 5);
        try {
            new SpectralNorm(linear, -1.0f, 1);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Norm coefficient"));
        }
    }

    @Test
    public void testSpectralNormInvalidPowerIterations() {
        MockLinearModule linear = new MockLinearModule(10, 5);
        try {
            new SpectralNorm(linear, 1.0f, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("power iterations"));
        }
    }

    @Test
    public void testSpectralNormNotInitialized() {
        MockLinearModule linear = new MockLinearModule(10, 5);
        SpectralNorm sn = new SpectralNorm(linear);

        try {
            sn.forward(null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testSpectralNormToString() {
        MockLinearModule linear = new MockLinearModule(10, 5);
        SpectralNorm sn = new SpectralNorm(linear);
        String str = sn.toString();
        assertTrue(str.contains("SpectralNorm"));
        assertTrue(str.contains("MockLinearModule"));
    }

    @Test
    public void testGANTrainerClose() {
        Module generator = new MockGenerator(100, 10);
        Module discriminator = new MockDiscriminator(10, 1);
        SGD genOptim = new SGD(generator.parameters(), 0.001f);
        SGD discOptim = new SGD(discriminator.parameters(), 0.001f);

        GANTrainer trainer = new GANTrainer(generator, discriminator, genOptim, discOptim, GANTrainer.GANLoss.VANILLA);
        trainer.close();

        try {
            trainer.generator();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }

    private static class MockGenerator extends Module {
        private final int latentDim;
        private final int outputDim;
        private Parameter weight;
        private Parameter bias;

        MockGenerator(int latentDim, int outputDim) {
            this.latentDim = latentDim;
            this.outputDim = outputDim;
        }

        public void initialize(ComputeBackend backend) {
            Tensor w = Tensor.zeros(backend, Shape.matrix(outputDim, latentDim));
            Tensor b = Tensor.zeros(backend, Shape.vector(outputDim));
            weight = registerParameter("weight", new Parameter(w, "weight"));
            bias = registerParameter("bias", new Parameter(b, "bias"));
        }

        @Override
        public Tensor forward(Tensor input) {
            if (weight == null) {
                float[] inputData = input.toFloatArray();
                int batchSize = input.shape().dim(0);
                float[] outputData = new float[batchSize * outputDim];
                try (Tensor flat = Tensor.fromArray(input.backend(), outputData)) {
                    return flat.reshape(batchSize, outputDim);
                }
            }
            return input;
        }
    }

    private static class MockDiscriminator extends Module {
        private final int inputDim;
        private final int outputDim;
        private Parameter weight;
        private Parameter bias;

        MockDiscriminator(int inputDim, int outputDim) {
            this.inputDim = inputDim;
            this.outputDim = outputDim;
        }

        public void initialize(ComputeBackend backend) {
            Tensor w = Tensor.zeros(backend, Shape.matrix(outputDim, inputDim));
            Tensor b = Tensor.zeros(backend, Shape.vector(outputDim));
            weight = registerParameter("weight", new Parameter(w, "weight"));
            bias = registerParameter("bias", new Parameter(b, "bias"));
        }

        @Override
        public Tensor forward(Tensor input) {
            if (weight == null) {
                float[] inputData = input.toFloatArray();
                int batchSize = input.shape().dim(0);
                float[] outputData = new float[batchSize * outputDim];
                for (int i = 0; i < outputData.length; i++) {
                    outputData[i] = 0.5f;
                }
                try (Tensor flat = Tensor.fromArray(input.backend(), outputData)) {
                    return flat.reshape(batchSize, outputDim);
                }
            }
            return input;
        }
    }

    private static class MockLinearModule extends Module {
        private final int inFeatures;
        private final int outFeatures;
        private Parameter weight;
        private Parameter bias;

        MockLinearModule(int inFeatures, int outFeatures) {
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
        }

        public void initialize(ComputeBackend backend) {
            Tensor w = Tensor.randn(backend, Shape.matrix(outFeatures, inFeatures), System.currentTimeMillis());
            weight = registerParameter("weight", new Parameter(w, "weight"));
            Tensor b = Tensor.zeros(backend, Shape.vector(outFeatures));
            bias = registerParameter("bias", new Parameter(b, "bias"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    // =================================================================================================
    // VAE Tests
    // =================================================================================================

    @Test
    public void testVAECreation() {
        MockVAE vae = new MockVAE(10, 20, 32);
        assertNotNull(vae);
        assertEquals(32, vae.latentDim());
        assertEquals(1.0f, vae.beta(), 1e-6f);
    }

    @Test
    public void testVAEWithCustomBeta() {
        MockVAE vae = new MockVAE(10, 20, 16, 0.5f);
        assertNotNull(vae);
        assertEquals(16, vae.latentDim());
        assertEquals(0.5f, vae.beta(), 1e-6f);
    }

    @Test
    public void testVAEInvalidLatentDim() {
        try {
            new MockVAE(10, 20, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Latent dimension"));
        }
    }

    @Test
    public void testVAEInvalidBeta() {
        try {
            new MockVAE(10, 20, 16, -0.5f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Beta"));
        }
    }

    @Test
    public void testVAEEncodeResultNullMean() {
        try {
            new VAE.EncodeResult(null, null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertNotNull(e.getMessage());
        }
    }

    @Test
    public void testVAEEncodeResultNullLogVar() {
        try {
            new VAE.EncodeResult(null, null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertNotNull(e.getMessage());
        }
    }

    @Test
    public void testVAELossComponents() {
        float reconLoss = 0.5f;
        float klLoss = 0.3f;
        float totalLoss = 0.8f;

        VAE.VAELoss loss = new VAE.VAELoss(reconLoss, klLoss, totalLoss, null, null, null);
        assertEquals(reconLoss, loss.reconstructionLoss(), 1e-6f);
        assertEquals(klLoss, loss.klDivergence(), 1e-6f);
        assertEquals(totalLoss, loss.totalLoss(), 1e-6f);
    }

    @Test
    public void testVAELossToString() {
        VAE.VAELoss loss = new VAE.VAELoss(0.5f, 0.3f, 0.8f, null, null, null);
        String str = loss.toString();
        assertTrue(str.contains("recon"));
        assertTrue(str.contains("kl"));
        assertTrue(str.contains("total"));
    }

    @Test
    public void testVAESetRandom() {
        MockVAE vae = new MockVAE(10, 20, 8);
        Random newRandom = new Random(42);
        vae.setRandom(newRandom);
        assertEquals(newRandom, vae.getRandom());
    }

    @Test
    public void testVAESetRandomNull() {
        MockVAE vae = new MockVAE(10, 20, 8);

        try {
            vae.setRandom(null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Random"));
        }
    }

    @Test
    public void testVAEToString() {
        MockVAE vae = new MockVAE(10, 20, 16, 0.5f);
        String str = vae.toString();
        assertTrue(str.contains("VAE"));
        assertTrue(str.contains("16"));
        assertTrue(str.contains("0.5"));
    }

    @Test
    public void testVAEClose() {
        MockVAE vae = new MockVAE(10, 20, 8);
        vae.close();
        assertTrue(vae.toString().contains("closed"));
    }

    @Test
    public void testVAEToStringAfterClose() {
        MockVAE vae = new MockVAE(10, 20, 8);
        vae.close();
        String str = vae.toString();
        assertTrue(str.contains("closed=true"));
    }

    @Test
    public void testVAEBetaVAE() {
        MockVAE vae = new MockVAE(10, 20, 8, 4.0f);
        assertEquals(4.0f, vae.beta(), 1e-6f);
    }

    @Test
    public void testVAETrainingMode() {
        MockVAE vae = new MockVAE(10, 20, 8);
        assertTrue(vae.isTraining());
        vae.eval();
        assertFalse(vae.isTraining());
        vae.train();
        assertTrue(vae.isTraining());
    }

    @Test
    public void testVAEForwardNullInput() {
        MockVAE vae = new MockVAE(10, 20, 8);
        try {
            vae.forwardVAE(null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Input"));
        }
    }

    @Test
    public void testVAESampleNullBackend() {
        MockVAE vae = new MockVAE(10, 20, 8);
        try {
            vae.sample(null, 4);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Backend"));
        }
    }

    @Test
    public void testVAESampleInvalidBatchSize() {
        MockVAE vae = new MockVAE(10, 20, 8);
        try {
            vae.sample(null, 0);
            fail("Expected NullPointerException or IllegalArgumentException");
        } catch (NullPointerException | IllegalArgumentException e) {
            assertTrue(e.getMessage() != null);
        }
    }

    @Test
    public void testVAEElboLossNullInput() {
        MockVAE vae = new MockVAE(10, 20, 8);
        try {
            vae.elboLoss(null, null, null, null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Input"));
        }
    }

    private static class MockVAE extends VAE {
        private final int inputDim;
        private final int outputDim;

        MockVAE(int inputDim, int outputDim, int latentDim) {
            super(latentDim);
            this.inputDim = inputDim;
            this.outputDim = outputDim;
        }

        MockVAE(int inputDim, int outputDim, int latentDim, float beta) {
            super(latentDim, beta);
            this.inputDim = inputDim;
            this.outputDim = outputDim;
        }

        @Override
        protected EncodeResult encode(Tensor input) {
            ComputeBackend backend = input != null ? input.backend() : null;
            int batchSize = input != null ? input.shape().dim(0) : 4;
            return new EncodeResult(
                Tensor.zeros(backend, Shape.matrix(batchSize, latentDim())),
                Tensor.zeros(backend, Shape.matrix(batchSize, latentDim()))
            );
        }

        @Override
        protected Tensor decode(Tensor latent) {
            ComputeBackend backend = latent != null ? latent.backend() : null;
            int batchSize = latent != null ? latent.shape().dim(0) : 4;
            return Tensor.zeros(backend, Shape.matrix(batchSize, outputDim));
        }
    }
}
