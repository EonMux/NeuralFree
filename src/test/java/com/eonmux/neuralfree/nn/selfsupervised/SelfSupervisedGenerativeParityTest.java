package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.generative.AutoregressiveGenerator;
import com.eonmux.neuralfree.nn.generative.DDPMTrainer;
import com.eonmux.neuralfree.nn.generative.GANTrainer;
import com.eonmux.neuralfree.nn.generative.LinearScheduler;
import com.eonmux.neuralfree.nn.generative.SamplingConfig;
import com.eonmux.neuralfree.nn.generative.SpectralNorm;
import com.eonmux.neuralfree.nn.generative.VAE;
import com.eonmux.neuralfree.nn.layers.Embedding;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The forward passes and training steps of the self-supervised and generative modules (CLIP, contrastive, DINO, MAE,
 * MLM, GAN, VAE, spectral norm, DDPM, autoregressive generation) on their CPU path and on each GPU API. Every
 * scenario draws its weights, inputs and random streams from fixed seeds on the host, so both paths see the same
 * values. Float results compare in a composition class, {@code |gpu - cpu| <= 1e-4 * max(1, |cpu|)} with NaN and
 * infinities matching exactly (chains of reduction, division and transcendental steps of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}); host random draws (masks, sampled tokens) compare
 * exactly.
 *
 * <p>Every scenario runs in grad mode. Forward-only scenarios run inside a {@link ComputationGraph.StepScope}; the
 * training steps (GAN, DDPM) run without one and must leave the graph at its size before the step themselves. The
 * gradient invariants check that each module's recorded graph reaches the parameters it trains, and only those.</p>
 */
@RunWith(Enclosed.class)
public class SelfSupervisedGenerativeParityTest {

    static final double COMPOSITION_TOLERANCE = 1e-4;

    /** The CPU path, on a backend without the train library. */
    public static class HostPath {
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
        public void everyScenarioIsFinite() {
            Results results = Scenarios.all(host);
            for (Map.Entry<String, float[]> entry : results.floats.entrySet()) {
                for (float v : entry.getValue()) {
                    assertTrue(entry.getKey() + " holds a non-finite value " + v, Float.isFinite(v));
                }
            }
        }

        @Test
        public void invariantsHold() {
            Scenarios.checkInvariants("cpu path", host);
        }
    }

    /** The device path on each GPU API, against the CPU path. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private ComputeBackend host;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                TrainGpu.release(host);
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void everyScenarioMatchesCpuPath() {
            Results ref = Scenarios.all(host);
            Results got = Scenarios.all(gpu);
            assertEquals("float result keys", ref.floats.keySet(), got.floats.keySet());
            assertEquals("int result keys", ref.ints.keySet(), got.ints.keySet());
            for (Map.Entry<String, float[]> entry : ref.floats.entrySet()) {
                assertComposition(api + " " + entry.getKey(), entry.getValue(), got.floats.get(entry.getKey()));
            }
            for (Map.Entry<String, int[]> entry : ref.ints.entrySet()) {
                assertArrayEquals(api + " " + entry.getKey(), entry.getValue(), got.ints.get(entry.getKey()));
            }
        }

        @Test
        public void invariantsHold() {
            Scenarios.checkInvariants(api.toString(), gpu);
        }
    }

    /** Named results of one backend's run. */
    static final class Results {
        final Map<String, float[]> floats = new LinkedHashMap<>();
        final Map<String, int[]> ints = new LinkedHashMap<>();

        void put(String name, Tensor t) {
            floats.put(name, t.toFloatArray());
        }

        void put(String name, float... values) {
            floats.put(name, values);
        }
    }

    static void assertComposition(String what, float[] ref, float[] got) {
        assertEquals(what + ": length", ref.length, got.length);
        for (int i = 0; i < ref.length; i++) {
            float r = ref[i];
            float g = got[i];
            boolean special = Float.isNaN(r) || Float.isInfinite(r);
            boolean ok = special
                ? Float.floatToIntBits(r) == Float.floatToIntBits(g)
                : Math.abs(g - r) <= COMPOSITION_TOLERANCE * Math.max(1.0, Math.abs(r));
            if (!ok) {
                fail(what + ": element " + i + ": cpu " + r + ", gpu " + g);
            }
        }
    }

    /** Applies a {@link Linear} to the last dimension of an input of any rank. */
    static final class TokenwiseLinear extends Module {
        private final Linear linear;
        private final int in;
        private final int out;

        TokenwiseLinear(ComputeBackend backend, int in, int out, long seed) {
            this.in = in;
            this.out = out;
            this.linear = registerModule("linear", Scenarios.linear(backend, in, out, seed));
        }

        @Override
        public Tensor forward(Tensor input) {
            int[] dims = input.shape().dims();
            dims[dims.length - 1] = out;
            try (Tensor rows = input.reshape((int) (input.numElements() / in), in);
                 Tensor projected = linear.forward(rows)) {
                return projected.reshape(dims);
            }
        }
    }

    /** A VAE with linear encoder heads and a linear decoder. */
    static final class LinearVAE extends VAE {
        private final Linear meanHead;
        private final Linear logVarHead;
        private final Linear decoder;

        LinearVAE(ComputeBackend backend, int dataDim, int latentDim) {
            super(latentDim, 0.5f);
            this.meanHead = registerModule("mean", Scenarios.linear(backend, dataDim, latentDim, 71));
            this.logVarHead = registerModule("log_var", Scenarios.linear(backend, dataDim, latentDim, 72));
            this.decoder = registerModule("decoder", Scenarios.linear(backend, latentDim, dataDim, 73));
        }

        @Override
        protected EncodeResult encode(Tensor input) {
            return new EncodeResult(meanHead.forward(input), logVarHead.forward(input));
        }

        @Override
        protected Tensor decode(Tensor latent) {
            return decoder.forward(latent);
        }
    }

    static final class Scenarios {
        private Scenarios() {
        }

        static Results all(ComputeBackend b) {
            Results r = new Results();
            inStep(() -> clip(b, r));
            inStep(() -> contrastive(b, r));
            inStep(() -> dino(b, r));
            inStep(() -> mae(b, r));
            inStep(() -> mlm(b, r));
            for (GANTrainer.GANLoss loss : GANTrainer.GANLoss.values()) {
                gan(b, loss, r);
            }
            inStep(() -> vae(b, r));
            inStep(() -> spectralNorm(b, r));
            ddpm(b, r);
            inStep(() -> autoregressive(b, r));
            return r;
        }

        /** Runs {@code scenario} in a step scope, which removes whatever its forward passes recorded. */
        static void inStep(Runnable scenario) {
            try (ComputationGraph.StepScope step = ComputationGraph.current().step()) {
                scenario.run();
            }
        }

        static Linear linear(ComputeBackend b, int in, int out, long seed) {
            Linear l = new Linear(in, out);
            l.initialize(b, new Random(seed));
            return l;
        }

        static Tensor input(ComputeBackend b, long seed, float lo, float hi, int... dims) {
            int n = 1;
            for (int d : dims) {
                n *= d;
            }
            try (Tensor flat = Tensor.fromArray(b, TrainParity.uniform(seed, n, lo, hi))) {
                return flat.reshape(dims);
            }
        }

        static void clip(ComputeBackend b, Results r) {
            try (CLIPTrainer clip = new CLIPTrainer(linear(b, 6, 5, 11), linear(b, 4, 3, 12), 5, 3, 4);
                 Tensor image = input(b, 14, -1f, 1f, 3, 6);
                 Tensor text = input(b, 15, -1f, 1f, 3, 4)) {
                clip.setRandom(new Random(13));
                clip.initialize(b);
                CLIPTrainer.CLIPOutput out = clip.forwardCLIP(image, text);
                try (Tensor img = out.imageEmbedding(); Tensor txt = out.textEmbedding();
                     Tensor perImage = out.logitsPerImage(); Tensor perText = out.logitsPerText();
                     Tensor loss = out.clipLoss()) {
                    r.put("clip.logitsPerImage", perImage);
                    r.put("clip.logitsPerText", perText);
                    r.put("clip.loss", loss);
                    r.put("clip.similarity", clip.computeSimilarity(img, txt));
                }
            }
        }

        static void contrastive(ComputeBackend b, Results r) {
            try (ContrastiveLearner learner = new ContrastiveLearner(linear(b, 6, 5, 21), 5, 3, 0.5f);
                 Tensor view1 = input(b, 23, -1f, 1f, 4, 6);
                 Tensor view2 = input(b, 24, -1f, 1f, 4, 6)) {
                learner.setRandom(new Random(22));
                learner.initialize(b);
                ContrastiveLearner.ContrastiveOutput out = learner.forwardContrastive(view1, view2);
                try (Tensor p1 = out.projection1(); Tensor p2 = out.projection2();
                     Tensor h1 = out.representation1(); Tensor h2 = out.representation2();
                     Tensor loss = learner.computeNTXentLoss(p1, p2)) {
                    r.put("contrastive.projection1", p1);
                    r.put("contrastive.representation2", h2);
                    r.put("contrastive.loss", loss);
                }
                learner.setRandom(new Random(25));
                try (Tensor augmented = learner.augment(view1, 0.1f, 0.25f)) {
                    r.put("contrastive.augment", augmented);
                }
            }
        }

        static void dino(ComputeBackend b, Results r) {
            Linear student = linear(b, 6, 8, 31);
            try (DINOTrainer dino = new DINOTrainer(student, 8);
                 Tensor g1 = input(b, 33, -1f, 1f, 3, 6);
                 Tensor g2 = input(b, 34, -1f, 1f, 3, 6);
                 Tensor local = input(b, 35, -1f, 1f, 3, 6)) {
                dino.initializeTeacher(linear(b, 6, 8, 32));
                DINOTrainer.DINOOutput out = dino.forwardDINO(g1, g2, local);
                try (Tensor s = out.studentOutput(); Tensor t = out.teacherOutput(); Tensor loss = out.dinoLoss()) {
                    r.put("dino.student", s);
                    r.put("dino.teacher", t);
                    r.put("dino.loss", loss);
                }
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    TensorOps.mulInPlace(student.getParameter("weight").tensor(), 0.5f);
                }
                dino.updateTeacher();
                r.put("dino.teacherWeight", dino.teacher().getParameter("weight").tensor());
                r.put("dino.center", dino.namedBuffers().get("center"));
            }
        }

        static void mae(ComputeBackend b, Results r) {
            try (MaskedAutoencoder mae = new MaskedAutoencoder(new TokenwiseLinear(b, 8, 8, 41),
                    new TokenwiseLinear(b, 8, 8, 42), 2, 4, 6, 2, 0.5f);
                 Tensor image = input(b, 44, -1f, 1f, 2, 2, 4, 6)) {
                mae.setRandom(new Random(43));
                MaskedAutoencoder.MAEOutput out = mae.forwardMAE(image);
                try (Tensor recon = out.reconstruction(); Tensor encoded = out.encoded();
                     Tensor decoded = out.decoded(); Tensor loss = out.maskedLoss()) {
                    assertFalse("the MAE reconstruction output is not recorded", recon.requiresGrad());
                    r.put("mae.reconstruction", recon);
                    r.put("mae.decoded", decoded);
                    r.put("mae.loss", loss);
                    r.ints.put("mae.masked", out.mask().maskedIndices());
                }
            }
        }

        /** The masked-language model with an untied decoder and with the decoder tied to the embedding table. */
        static void mlm(ComputeBackend b, Results r) {
            int[] ids = {0, 3, 5, 2, 9, 0, 1, 8, 4, 9};
            MaskedLanguageModel.MLMMask mask = new MaskedLanguageModel.MLMMask(
                new int[]{1, 3, 6, 8}, new int[]{2, 7, 1, 9}, 5, 0.15f);
            for (boolean tied : new boolean[]{false, true}) {
                String key = tied ? "mlm.tied." : "mlm.";
                Embedding embedding = new Embedding(10, 4);
                embedding.initialize(b, new Random(51));
                try (MaskedLanguageModel mlm = tied ? new MaskedLanguageModel(embedding, 10, 4, embedding.weight())
                         : new MaskedLanguageModel(embedding, 10, 4);
                     Tensor flat = Tensor.fromArray(b, ids);
                     Tensor inputIds = flat.reshape(2, 5)) {
                    mlm.setRandom(new Random(52));
                    mlm.initialize(b);
                    MaskedLanguageModel.MLMOutput out = mlm.forwardMLM(inputIds, mask);
                    try (Tensor logits = out.logits(); Tensor predictions = out.predictions();
                         Tensor loss = out.mlmLoss()) {
                        r.put(key + "logits", logits);
                        r.put(key + "predictions", predictions);
                        r.put(key + "loss", loss);
                    }
                }
            }
        }

        /**
         * One GAN step per loss type, each completing with finite losses. WGAN-GP differentiates the discriminator's
         * input gradient (create-graph), so its step needs create-graph support on every gradient function of the
         * discriminator's path; its penalty's gradient must reach the discriminator, whose
         * weight the step changes.
         */
        static void gan(ComputeBackend b, GANTrainer.GANLoss lossType, Results r) {
            Linear generator = linear(b, 3, 4, 61);
            Linear discriminator = linear(b, 4, 1, 62);
            SGD genOptimizer = new SGD(generator.parameters(), 0.01f);
            SGD discOptimizer = new SGD(discriminator.parameters(), 0.01f);
            try (GANTrainer trainer = new GANTrainer(generator, discriminator, genOptimizer, discOptimizer,
                    lossType, 10.0f, 2);
                 Tensor real = input(b, 63, -1f, 1f, 5, 4)) {
                int nodes = ComputationGraph.current().size();
                float[] discWeight = discriminator.getParameter("weight").tensor().toFloatArray();
                GANTrainer.GANStepResult step = trainer.trainStep(real, new Random(64));
                assertTrue("gan " + lossType + " generator loss is finite", Float.isFinite(step.generatorLoss()));
                assertTrue("gan " + lossType + " discriminator loss is finite",
                    Float.isFinite(step.discriminatorLoss()));
                if (lossType == GANTrainer.GANLoss.WGAN_GP) {
                    assertFalse("a WGAN-GP step updates the discriminator", Arrays.equals(discWeight,
                        discriminator.getParameter("weight").tensor().toFloatArray()));
                }
                r.put("gan." + lossType + ".losses", step.generatorLoss(), step.discriminatorLoss());
                assertEquals("gan " + lossType + " step leaves no node", nodes, ComputationGraph.current().size());
            } finally {
                genOptimizer.close();
                discOptimizer.close();
                generator.close();
                discriminator.close();
            }
        }

        static void vae(ComputeBackend b, Results r) {
            try (LinearVAE vae = new LinearVAE(b, 4, 2);
                 Tensor x = input(b, 75, 0.05f, 0.95f, 3, 4)) {
                vae.setRandom(new Random(74));
                VAE.VAEResult out = vae.forwardVAE(x);
                try (Tensor recon = out.reconstruction(); Tensor latent = out.latent();
                     Tensor mean = out.mean(); Tensor logVar = out.logVar()) {
                    r.put("vae.reconstruction", recon);
                    r.put("vae.latent", latent);
                    try (VAE.VAELoss mse = vae.elboLoss(x, recon, mean, logVar);
                         VAE.VAELoss bce = vae.elboLoss(x, recon, mean, logVar, "bce")) {
                        r.put("vae.losses", mse.reconstructionLoss(), mse.klDivergence(), mse.totalLoss(),
                            bce.reconstructionLoss(), bce.totalLoss());
                        r.put("vae.totalTensors", mse.totalLossTensor().toFloatArray()[0],
                            bce.totalLossTensor().toFloatArray()[0]);
                        assertTrue("KL divergence is non-negative", mse.klDivergence() >= 0f);
                    }
                }
                vae.setRandom(new Random(76));
                try (Tensor samples = vae.sample(b, 2)) {
                    assertFalse("VAE samples are not recorded", samples.requiresGrad());
                    r.put("vae.samples", samples);
                }
            }
        }

        static void spectralNorm(ComputeBackend b, Results r) {
            try (SpectralNorm sn = new SpectralNorm(linear(b, 5, 3, 81), 1.0f, 3);
                 Tensor x = input(b, 83, -1f, 1f, 2, 5)) {
                sn.initialize(b, new Random(82));
                try (Tensor y1 = sn.forward(x); Tensor y2 = sn.forward(x)) {
                    r.put("spectral.y1", y1);
                    r.put("spectral.y2", y2);
                    float[] u = sn.namedBuffers().get("u").toFloatArray();
                    float[] v = sn.namedBuffers().get("v").toFloatArray();
                    sn.eval();
                    try (Tensor evaluated = sn.forward(x)) {
                        TrainParity.assertBitIdentical("an evaluation forward normalizes with the stored vectors",
                                y2.toFloatArray(), evaluated.toFloatArray());
                    } finally {
                        sn.train();
                    }
                    TrainParity.assertBitIdentical("an evaluation forward leaves u", u,
                            sn.namedBuffers().get("u").toFloatArray());
                    TrainParity.assertBitIdentical("an evaluation forward leaves v", v,
                            sn.namedBuffers().get("v").toFloatArray());
                }
                r.put("spectral.u", sn.namedBuffers().get("u"));
                r.put("spectral.v", sn.namedBuffers().get("v"));
            }
        }

        static void ddpm(ComputeBackend b, Results r) {
            Linear predictor = linear(b, 4, 4, 91);
            SGD optimizer = new SGD(predictor.parameters(), 0.01f);
            try (DDPMTrainer trainer = new DDPMTrainer(predictor, new LinearScheduler(b, 10), optimizer, b,
                    new Random(92));
                 Tensor x0 = input(b, 93, -1f, 1f, 3, 4)) {
                int nodes = ComputationGraph.current().size();
                DDPMTrainer.TrainingStepResult step = trainer.trainStep(x0);
                assertEquals("the DDPM step leaves no node", nodes, ComputationGraph.current().size());
                try (Tensor predicted = step.predictedNoise(); Tensor noisy = step.noisyInput()) {
                    assertFalse("the step result holds detached handles", predicted.requiresGrad());
                    r.put("ddpm.loss", step.loss());
                    r.put("ddpm.predicted", predicted);
                    r.put("ddpm.noisy", noisy);
                    r.ints.put("ddpm.timestep", new int[]{step.timestep()});
                }
                DDPMTrainer.SampleResult sample = trainer.sample(Shape.of(2, 4), 5);
                assertEquals("DDPM sampling records nothing", nodes, ComputationGraph.current().size());
                try (Tensor samples = sample.samples()) {
                    r.put("ddpm.samples", samples);
                }
            } finally {
                optimizer.close();
                predictor.close();
            }
        }

        static void autoregressive(ComputeBackend b, Results r) {
            Embedding model = new Embedding(12, 12);
            model.initialize(b, new Random(101));
            int[] prompt = {1, 4, 7};
            try (AutoregressiveGenerator generator = new AutoregressiveGenerator(model, 12, 12, 1, 1, 16)) {
                r.ints.put("ar.greedy", generator.generate(prompt, 9, SamplingConfig.builder()
                    .topK(0).topP(1.0f).temperature(0.8f).repetitionPenalty(1.3f).seed(5).build()));
                r.ints.put("ar.topK", generator.generate(prompt, 9, SamplingConfig.builder()
                    .topK(4).repetitionPenalty(1.2f).seed(6).build()));
                r.ints.put("ar.topP", generator.generate(prompt, 9, SamplingConfig.builder()
                    .topK(0).topP(0.8f).seed(7).build()));
                r.ints.put("ar.beam", generator.generate(prompt, 9, SamplingConfig.builder()
                    .topK(0).beamWidth(3).seed(8).build()));
            } finally {
                model.close();
            }
        }

        /**
         * Exact properties every path keeps: transposed logits, exact patch round trips, an untouched spectrally
         * normalized weight, and gradients that reach exactly the trained parameters.
         */
        static void checkInvariants(String what, ComputeBackend b) {
            inStep(() -> clipTranspose(what, b));
            inStep(() -> patchRoundTrip(what, b));
            inStep(() -> spectralNormLeavesWeight(what, b));
            inStep(() -> GradientFlowInvariants.clip(what, b));
            inStep(() -> GradientFlowInvariants.dino(what, b));
            inStep(() -> GradientFlowInvariants.mae(what, b));
            inStep(() -> GradientFlowInvariants.spectralNorm(what, b));
            inStep(() -> GradientFlowInvariants.vae(what, b));
            inStep(() -> GradientFlowInvariants.augment(what, b));
            GradientFlowInvariants.inference(what, b);
        }

        private static void clipTranspose(String what, ComputeBackend b) {
            try (CLIPTrainer clip = new CLIPTrainer(linear(b, 6, 5, 11), linear(b, 4, 3, 12), 5, 3, 4);
                 Tensor image = input(b, 14, -1f, 1f, 3, 6);
                 Tensor text = input(b, 15, -1f, 1f, 3, 4)) {
                clip.setRandom(new Random(13));
                clip.initialize(b);
                CLIPTrainer.CLIPOutput out = clip.forwardCLIP(image, text);
                try (Tensor img = out.imageEmbedding(); Tensor txt = out.textEmbedding();
                     Tensor perImage = out.logitsPerImage(); Tensor perText = out.logitsPerText();
                     Tensor loss = out.clipLoss()) {
                    float[] a = perImage.toFloatArray();
                    float[] t = perText.toFloatArray();
                    float[] transposed = new float[a.length];
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            transposed[j * 3 + i] = a[i * 3 + j];
                        }
                    }
                    TrainParity.assertBitIdentical(what + " clip logitsPerText is the transpose", transposed, t);
                }
            }
        }

        private static void patchRoundTrip(String what, ComputeBackend b) {
            int batch = 2;
            int channels = 2;
            int height = 5;
            int width = 7;
            try (MaskedAutoencoder mae = new MaskedAutoencoder(new TokenwiseLinear(b, 8, 8, 41),
                    new TokenwiseLinear(b, 8, 8, 42), 2, height, width, channels, 0.5f);
                 Tensor image = input(b, 45, -1f, 1f, batch, channels, height, width);
                 Tensor patches = mae.patchify(image);
                 Tensor restored = mae.unpatchify(patches)) {
                assertEquals(what + " patch shape", Shape.of(batch, 6, 8), patches.shape());
                float[] x = image.toFloatArray();
                float[] expected = new float[x.length];
                for (int i = 0; i < x.length; i++) {
                    int w = i % width;
                    int h = (i / width) % height;
                    expected[i] = (h < 4 && w < 6) ? x[i] : 0.0f;
                }
                TrainParity.assertBitIdentical(what + " mae unpatchify(patchify(x))", expected,
                    restored.toFloatArray());
            }
        }

        private static void spectralNormLeavesWeight(String what, ComputeBackend b) {
            Linear wrapped = linear(b, 5, 3, 81);
            Parameter weight = wrapped.getParameter("weight");
            float[] before = weight.tensor().toFloatArray();
            try (SpectralNorm sn = new SpectralNorm(wrapped, 1.0f, 3);
                 Tensor x = input(b, 83, -1f, 1f, 2, 5)) {
                sn.initialize(b, new Random(82));
                try (Tensor y = sn.forward(x)) {
                    assertEquals(what + " spectral norm output shape", Shape.of(2, 3), y.shape());
                }
                TrainParity.assertBitIdentical(what + " spectral norm never writes the weight", before,
                    weight.tensor().toFloatArray());
            }
        }
    }
}
