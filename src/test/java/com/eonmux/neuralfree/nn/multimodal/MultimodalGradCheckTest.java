package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
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
 * Finite-difference gradient checks of every nn/multimodal module with parameters: {@link EncoderBlock},
 * {@link CrossAttentionFusion}, {@link ModalityRouter}, {@link TextEncoder}, {@link VisionEncoder},
 * {@link AudioEncoder} and {@link VisionLanguageModel}, on the CPU path and on each GPU API.
 *
 * <p>Each check builds the module from a fixed seed, overwrites its parameters with seeded values in
 * {@code [-scale, scale]}, puts it in evaluation mode (dropout off) and compares the recorded gradient of
 * {@code sum(out * r)}, for a fixed random {@code r}, with central differences ({@link GradientCheckUtil}: step
 * {@value #EPSILON}, tolerance {@value #TOLERANCE}). Inputs are drawn from {@code [-1, 1]}. Kinks: every module of this
 * family is smooth in its parameters (GELU, softmax, sigmoid, layer norm, linear maps); the audio spectrogram's
 * {@code clamp} and {@code log} act on the waveform only, so the audio encoder is checked in its parameters and not
 * in the waveform, and the router's positive-weight masks are constants of the softmax weights, which are all
 * positive. Every check must leave the graph at its size before the check.</p>
 */
@RunWith(Enclosed.class)
public class MultimodalGradCheckTest {

    static final float EPSILON = 1e-3f;
    static final float TOLERANCE = 1e-2f;
    static final int ELEMENTS = 4;

    /** The checks, on the backend of the subclass. */
    public abstract static class Checks {

        protected abstract ComputeBackend backend();

        @Test
        public void encoderBlock() {
            ComputeBackend b = backend();
            EncoderBlock block = new EncoderBlock(b, 8, 2, new Random(101));
            try (Tensor x = uniform(b, 102, 2, 3, 8);
                 Tensor r = uniform(b, 103, 2, 3, 8)) {
                randomize(block, 104, 0.5f);
                block.eval();
                assertParameters("EncoderBlock", () -> weighted(block.forward(x), r), block, ELEMENTS);
                assertInputs("EncoderBlock input", block, in -> weighted(block.forward(in.get(0)), r), x);
            } finally {
                block.close();
            }
        }

        /** Gated fusion with distinct key and value widths, so the value norm and the gate are both on the path. */
        @Test
        public void crossAttentionFusion() {
            ComputeBackend b = backend();
            CrossAttentionFusion fusion = new CrossAttentionFusion(6, 5, 4, 8, 2, 0.1f, true);
            try (Tensor q = uniform(b, 111, 2, 3, 6);
                 Tensor k = uniform(b, 112, 2, 4, 5);
                 Tensor v = uniform(b, 113, 2, 4, 4);
                 Tensor r = uniform(b, 114, 2, 3, 8)) {
                fusion.initialize(b, new Random(115));
                randomize(fusion, 116, 0.5f);
                fusion.eval();
                assertParameters("CrossAttentionFusion", () -> weighted(fusion.forward(q, k, v, null), r), fusion,
                    ELEMENTS);
                assertInputs("CrossAttentionFusion inputs", fusion,
                    in -> weighted(fusion.forward(in.get(0), in.get(1), in.get(2), null), r), q, k, v);
            } finally {
                fusion.close();
            }
        }

        /**
         * The routed combination {@code combineEmbeddings(e, routeWeights(x))} with softmax weights (all positive,
         * no top-k), in the router's parameters, the routed sequence and the embeddings.
         */
        @Test
        public void modalityRouter() {
            ComputeBackend b = backend();
            ModalityRouter router = new ModalityRouter(8, 3, "softmax", 0.7f, false, 1);
            try (Tensor x = uniform(b, 121, 3, 5, 8);
                 Tensor vision = uniform(b, 122, 3, 4);
                 Tensor text = uniform(b, 123, 3, 4);
                 Tensor audio = uniform(b, 124, 3, 4);
                 Tensor r = uniform(b, 125, 3, 4)) {
                router.initialize(b, new Random(126), ModalityRouter.MODALITY_VISION, ModalityRouter.MODALITY_TEXT,
                    ModalityRouter.MODALITY_AUDIO);
                randomize(router, 127, 0.5f);
                router.eval();
                assertParameters("ModalityRouter", () -> weighted(routed(router, x, vision, text, audio), r), router,
                    ELEMENTS);
                assertInputs("ModalityRouter inputs", router,
                    in -> weighted(routed(router, in.get(0), in.get(1), in.get(2), in.get(3)), r),
                    x, vision, text, audio);
            } finally {
                router.close();
            }
        }

        @Test
        public void textEncoder() {
            ComputeBackend b = backend();
            TextEncoder encoder = new TextEncoder(11, 8, 1, 2, 6, 0.1f, "transformer");
            try (Tensor ids = ids(b, new int[]{3, 0, 7, 10, 5, 5, 1, 9}, 2, 4);
                 Tensor r = uniform(b, 131, 2, 4, 8)) {
                encoder.initialize(b, new Random(132));
                randomize(encoder, 133, 0.5f);
                encoder.eval();
                assertParameters("TextEncoder", () -> weighted(encoder.encode(ids), r), encoder, ELEMENTS);
            } finally {
                encoder.close();
            }
        }

        /** Four 4x4 patches of a 3-channel 8x8 image plus the class token. */
        @Test
        public void visionEncoder() {
            ComputeBackend b = backend();
            VisionEncoder encoder = new VisionEncoder(8, 4, 3, 8, 1, 2, "transformer", true, 0.1f);
            try (Tensor image = uniform(b, 141, 2, 3, 8, 8);
                 Tensor r = uniform(b, 142, 2, 5, 8)) {
                encoder.initialize(b, new Random(143));
                randomize(encoder, 144, 0.5f);
                encoder.eval();
                assertParameters("VisionEncoder", () -> weighted(encoder.encode(image), r), encoder, ELEMENTS);
                assertInputs("VisionEncoder image", encoder, in -> weighted(encoder.encode(in.get(0)), r), image);
            } finally {
                encoder.close();
            }
        }

        /**
         * 80 samples with {@code nFft = 32} and hop 16 give 4 frames of 4 mel bands; every mel filter spans distinct
         * frequency bins, so no band sits at the log floor.
         */
        @Test
        public void audioEncoder() {
            ComputeBackend b = backend();
            AudioEncoder encoder = new AudioEncoder(8000, 4, 32, 16, 8, 1, 2, "transformer", 0.1f);
            try (Tensor wave = uniform(b, 151, 2, 80);
                 Tensor r = uniform(b, 152, 2, 2, 8)) {
                encoder.initialize(b, new Random(153));
                randomize(encoder, 154, 0.3f);
                encoder.eval();
                assertParameters("AudioEncoder", () -> weighted(encoder.encode(wave), r), encoder, ELEMENTS);
            } finally {
                encoder.close();
            }
        }

        @Test
        public void visionLanguageModel() {
            ComputeBackend b = backend();
            VisionLanguageModel model = new VisionLanguageModel(12, 8, 8, 8, 1, 2, 6);
            try (Tensor image = uniform(b, 161, 1, 3, 8, 8);
                 Tensor ids = ids(b, new int[]{4, 11, 2}, 1, 3);
                 Tensor r = uniform(b, 162, 1, 3, 12)) {
                model.initialize(b, new Random(163), 8, 4, 3, 1, 1);
                randomize(model, 164, 0.5f);
                model.eval();
                assertParameters("VisionLanguageModel", () -> weighted(model.forward(image, ids), r), model, 3);
            } finally {
                model.close();
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

    /** {@code combineEmbeddings({vision, text, audio}, routeWeights(x))}; the weights are closed. */
    static Tensor routed(ModalityRouter router, Tensor x, Tensor vision, Tensor text, Tensor audio) {
        Map<String, Tensor> embeddings = new LinkedHashMap<>();
        embeddings.put(ModalityRouter.MODALITY_VISION, vision);
        embeddings.put(ModalityRouter.MODALITY_TEXT, text);
        embeddings.put(ModalityRouter.MODALITY_AUDIO, audio);
        try (Tensor weights = router.routeWeights(x)) {
            return router.combineEmbeddings(embeddings, weights);
        }
    }

    /**
     * Asserts that the recorded gradients of {@code loss} in every parameter of {@code module} match central
     * differences on up to {@code elements} elements per parameter, and that the check leaves the graph unchanged.
     */
    static void assertParameters(String what, Supplier<Tensor> loss, Module module, int elements) {
        List<Parameter> params = module.parameters();
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

    /** Overwrites every parameter of {@code module} with seeded values in {@code [-scale, scale]}. */
    static void randomize(Module module, long seed, float scale) {
        Map<String, Object> state = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : module.stateDict().entrySet()) {
            int n = ((float[]) e.getValue()).length;
            state.put(e.getKey(), TrainParity.uniform(seed * 31 + e.getKey().hashCode(), n, -scale, scale));
        }
        module.loadStateDict(state);
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

    /** An INT32 tensor of {@code dims} holding {@code values}. */
    static Tensor ids(ComputeBackend b, int[] values, int... dims) {
        try (Tensor flat = Tensor.fromArray(b, values)) {
            return flat.reshape(Shape.of(dims));
        }
    }
}
