package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Embedding;
import com.eonmux.neuralfree.nn.layers.Linear;
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
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of every nn/selfsupervised module with parameters: {@link CLIPTrainer} (with its
 * learned logit scale), {@link ContrastiveLearner}, {@link DINOTrainer}, {@link MaskedAutoencoder} and
 * {@link MaskedLanguageModel}, each on its training loss, on the CPU path and on each GPU API.
 *
 * <p>Each check compares the recorded gradient of the loss with central differences ({@link GradientCheckUtil}: step
 * {@value #EPSILON}, tolerance {@value #TOLERANCE}) over linear encoders with seeded weights and inputs in
 * {@code [-1, 1]}, and for the masked language model over a seeded embedding of token ids. Randomness is fixed: the
 * autoencoder's patch mask is drawn from a {@link Random} re-seeded in every evaluation, and DINO's center buffer is
 * restored after every evaluation by the check (one forward before the check moves it off zero). Kinks: the losses
 * are softmax, log-softmax and squared-error compositions; their probability floors ({@code 1e-10}) are far below the
 * probabilities of these small logits (temperatures 0.5 for CLIP, the contrastive loss and the DINO student, 0.2 for
 * the DINO teacher), and DINO's row-max shift cancels in the softmax.</p>
 */
@RunWith(Enclosed.class)
public class SelfSupervisedGradCheckTest {

    static final float EPSILON = 1e-3f;
    static final float TOLERANCE = 1e-2f;
    static final int ELEMENTS = 6;

    /** The seed of the autoencoder's patch mask, re-applied before every evaluation. */
    static final long MASK_SEED = 335;

    /** The checks, on the backend of the subclass. */
    public abstract static class Checks {

        protected abstract ComputeBackend backend();

        /** The symmetric CLIP loss in both encoders, both projections and the logit scale, and in both inputs. */
        @Test
        public void clip() {
            ComputeBackend b = backend();
            CLIPTrainer clip = new CLIPTrainer(linear(b, 6, 5, 301), linear(b, 4, 3, 302), 5, 3, 4, true);
            try (Tensor images = uniform(b, 303, 3, 6);
                 Tensor texts = uniform(b, 304, 3, 4)) {
                clip.setRandom(new Random(305));
                clip.setTemperature(0.5f);
                clip.initialize(b);
                assertEquals("encoders, projections and the logit scale", 9, clip.parameters().size());
                assertParameters("CLIPTrainer", () -> clipLoss(clip, images, texts), clip, ELEMENTS);
                assertInputs("CLIPTrainer inputs", clip, in -> clipLoss(clip, in.get(0), in.get(1)), images, texts);
            } finally {
                clip.close();
            }
        }

        @Test
        public void contrastiveLearner() {
            ComputeBackend b = backend();
            ContrastiveLearner learner = new ContrastiveLearner(linear(b, 5, 4, 311), 4, 3, 0.5f);
            try (Tensor view1 = uniform(b, 312, 3, 5);
                 Tensor view2 = uniform(b, 313, 3, 5)) {
                learner.setRandom(new Random(314));
                learner.initialize(b);
                assertParameters("ContrastiveLearner", () -> ntXent(learner, view1, view2), learner, ELEMENTS);
                assertInputs("ContrastiveLearner views", learner, in -> ntXent(learner, in.get(0), in.get(1)),
                    view1, view2);
            } finally {
                learner.close();
            }
        }

        /** Two global views and one local view; the teacher is a frozen copy of the student. */
        @Test
        public void dino() {
            ComputeBackend b = backend();
            Linear student = linear(b, 5, 4, 321);
            DINOTrainer dino = new DINOTrainer(student, 4, 0.9f, 0.9f, 0.2f, 0.5f);
            try (Tensor global1 = uniform(b, 322, 3, 5);
                 Tensor global2 = uniform(b, 323, 3, 5);
                 Tensor local = uniform(b, 324, 3, 5)) {
                dino.initializeTeacher(linear(b, 5, 4, 325));
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                     Tensor warmUp = dinoLoss(dino, global1, global2, local)) {
                    assertTrue("finite DINO loss", Float.isFinite(warmUp.toFloatArray()[0]));
                }
                assertParameters("DINOTrainer", () -> dinoLoss(dino, global1, global2, local), dino,
                    student.parameters(), ELEMENTS);
            } finally {
                dino.close();
            }
        }

        /** The masked loss and the recorded reconstruction of a 4x4 image in 2x2 patches, half of them masked. */
        @Test
        public void maskedAutoencoder() {
            ComputeBackend b = backend();
            MaskedAutoencoder mae = new MaskedAutoencoder(linear(b, 4, 3, 331), linear(b, 3, 4, 332), 2, 4, 4, 1,
                0.5f);
            try (Tensor image = uniform(b, 333, 2, 1, 4, 4);
                 Tensor r = uniform(b, 334, 2, 4, 4)) {
                assertParameters("MaskedAutoencoder masked loss", () -> maskedLoss(mae, image), mae, ELEMENTS);
                assertInputs("MaskedAutoencoder image", mae, in -> maskedLoss(mae, in.get(0)), image);
                assertParameters("MaskedAutoencoder reconstruction", () -> {
                    mae.setRandom(new Random(MASK_SEED));
                    return weighted(mae.forward(image), r);
                }, mae, ELEMENTS);
            } finally {
                mae.close();
            }
        }

        /**
         * Four masked positions of a 2x5 batch over an embedding encoder, through BERT's prediction head, with an
         * untied decoder and with the decoder tied to the embedding table (whose gradient then sums both uses). The
         * dense layer's weights and bias are drawn from {@code [-1, 1]}, so the layer norm's rows keep a variance far
         * above its epsilon.
         */
        @Test
        public void maskedLanguageModel() {
            ComputeBackend b = backend();
            MaskedLanguageModel.MLMMask mask = new MaskedLanguageModel.MLMMask(
                new int[]{1, 3, 6, 8}, new int[]{2, 7, 1, 9}, 5, 0.15f);
            for (boolean tied : new boolean[]{false, true}) {
                Embedding embedding = new Embedding(10, 4);
                embedding.initialize(b, new Random(341));
                MaskedLanguageModel mlm = tied ? new MaskedLanguageModel(embedding, 10, 4, embedding.weight())
                    : new MaskedLanguageModel(embedding, 10, 4);
                try (Tensor ids = ints(b, new int[]{0, 3, 5, 2, 9, 0, 1, 8, 4, 9}, 2, 5)) {
                    mlm.setRandom(new Random(343));
                    mlm.initialize(b);
                    randomize(mlm, 342, 0.5f);
                    randomize(mlm.dense(), 344, 1.0f);
                    assertParameters("MaskedLanguageModel" + (tied ? " tied" : ""), () -> mlmLoss(mlm, ids, mask),
                        mlm, 12);
                } finally {
                    mlm.close();
                }
            }
        }
    }

    /** The CPU path. */
    public static class HostPath extends Checks {
        private ComputeBackend host;

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }

        @Override
        protected ComputeBackend backend() {
            return host;
        }
    }

    /** The device path on each GPU API. */
    @RunWith(Parameterized.class)
    public static class Device extends Checks {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
            session = TrainParity.open(api);
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            if (session != null) {
                session.close();
            }
        }

        @Override
        protected ComputeBackend backend() {
            return session.backend();
        }
    }

    /** The CLIP loss; the embeddings and logits are closed. */
    static Tensor clipLoss(CLIPTrainer clip, Tensor images, Tensor texts) {
        CLIPTrainer.CLIPOutput out = clip.forwardCLIP(images, texts);
        try (Tensor imageEmbedding = out.imageEmbedding();
             Tensor textEmbedding = out.textEmbedding();
             Tensor logitsPerImage = out.logitsPerImage();
             Tensor logitsPerText = out.logitsPerText()) {
            return out.clipLoss();
        }
    }

    /** The NT-Xent loss of the two views' projections; the projections and representations are closed. */
    static Tensor ntXent(ContrastiveLearner learner, Tensor view1, Tensor view2) {
        ContrastiveLearner.ContrastiveOutput out = learner.forwardContrastive(view1, view2);
        try (Tensor p1 = out.projection1();
             Tensor p2 = out.projection2();
             Tensor h1 = out.representation1();
             Tensor h2 = out.representation2()) {
            return learner.computeNTXentLoss(p1, p2);
        }
    }

    /** The DINO loss; the student and teacher outputs are closed. */
    static Tensor dinoLoss(DINOTrainer dino, Tensor global1, Tensor global2, Tensor local) {
        DINOTrainer.DINOOutput out = dino.forwardDINO(global1, global2, local);
        try (Tensor student = out.studentOutput();
             Tensor teacher = out.teacherOutput()) {
            return out.dinoLoss();
        }
    }

    /** The autoencoder's masked loss under the mask of {@link #MASK_SEED}; the other outputs are closed. */
    static Tensor maskedLoss(MaskedAutoencoder mae, Tensor image) {
        mae.setRandom(new Random(MASK_SEED));
        MaskedAutoencoder.MAEOutput out = mae.forwardMAE(image);
        try (Tensor reconstruction = out.reconstruction();
             Tensor encoded = out.encoded();
             Tensor decoded = out.decoded()) {
            return out.maskedLoss();
        }
    }

    /** The masked-language-model loss; the logits and predictions are closed. */
    static Tensor mlmLoss(MaskedLanguageModel mlm, Tensor ids, MaskedLanguageModel.MLMMask mask) {
        MaskedLanguageModel.MLMOutput out = mlm.forwardMLM(ids, mask);
        try (Tensor logits = out.logits();
             Tensor predictions = out.predictions()) {
            return out.mlmLoss();
        }
    }

    /** A {@link Linear} on {@code b} with seeded weights in {@code [-0.5, 0.5]}. */
    static Linear linear(ComputeBackend b, int in, int out, long seed) {
        Linear layer = new Linear(in, out);
        layer.initialize(b, new Random(seed));
        randomize(layer, seed, 0.5f);
        return layer;
    }

    /** Overwrites every parameter of {@code module} with seeded values in {@code [-scale, scale]}. */
    static void randomize(Module module, long seed, float scale) {
        Map<String, Object> state = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : module.stateDict().entrySet()) {
            int n = ((float[]) e.getValue()).length;
            state.put(e.getKey(), TrainParity.uniform(seed * 31 + e.getKey().hashCode(), n, -scale, scale));
        }
        module.loadStateDict(state);
    }

    /** {@link #assertParameters(String, Supplier, Module, List, int)} over every parameter of {@code module}. */
    static void assertParameters(String what, Supplier<Tensor> loss, Module module, int elements) {
        assertParameters(what, loss, module, module.parameters(), elements);
    }

    /**
     * Asserts that the recorded gradients of {@code loss} in {@code params} match central differences on up to
     * {@code elements} elements per parameter, with {@code module}'s buffers restored after every evaluation, and
     * that the check leaves the graph unchanged.
     */
    static void assertParameters(String what, Supplier<Tensor> loss, Module module, List<Parameter> params,
                                 int elements) {
        assertFalse(what + " has no parameters", params.isEmpty());
        int before = ComputationGraph.current().size();
        GradientCheckUtil.GradientCheckResult result =
            GradientCheckUtil.gradCheckParameters(loss, module, params, EPSILON, TOLERANCE, elements);
        assertTrue(what + " parameter gradients: " + result, result.passed());
        assertEquals(what + ": the check leaves the graph as it found it", before, ComputationGraph.current().size());
    }

    /**
     * Asserts that the recorded gradient of {@code sum(fn(inputs))} in every input matches central differences; the
     * gradients the check leaves in {@code module}'s parameters are cleared.
     */
    static void assertInputs(String what, Module module, GradientCheckUtil.TensorFunction fn, Tensor... inputs) {
        int before = ComputationGraph.current().size();
        GradientCheckUtil.GradientCheckResult result =
            GradientCheckUtil.gradCheck(fn, Arrays.asList(inputs), EPSILON, TOLERANCE);
        module.zeroGrad();
        assertTrue(what + " gradients: " + result, result.passed());
        assertEquals(what + ": the check leaves the graph as it found it", before, ComputationGraph.current().size());
    }

    /** {@code out * r} as a new tensor; {@code out} is closed. */
    static Tensor weighted(Tensor out, Tensor r) {
        try (Tensor o = out) {
            return TensorOps.mul(o, r);
        }
    }

    /** A FLOAT32 tensor of {@code dims} with seeded values in {@code [-1, 1]}. */
    static Tensor uniform(ComputeBackend b, long seed, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        try (Tensor flat = Tensor.fromArray(b, TrainParity.uniform(seed, n, -1.0f, 1.0f))) {
            return flat.reshape(Shape.of(dims));
        }
    }

    static Tensor ints(ComputeBackend b, int[] values, int... dims) {
        try (Tensor flat = Tensor.fromArray(b, values)) {
            return flat.reshape(Shape.of(dims));
        }
    }
}
