package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.IndexGradFunctions;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.graph.GraphData;
import com.eonmux.neuralfree.nn.structured.GaussianSplatting;
import com.eonmux.neuralfree.nn.structured.GraphTransformer;
import com.eonmux.neuralfree.nn.structured.NeuralODE;
import com.eonmux.neuralfree.nn.structured.NeuralRadianceField;
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

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

import static com.eonmux.neuralfree.nn.multimodal.MultimodalStructuredParityTest.Scenarios.ids;
import static com.eonmux.neuralfree.nn.multimodal.MultimodalStructuredParityTest.Scenarios.randomize;
import static com.eonmux.neuralfree.nn.multimodal.MultimodalStructuredParityTest.Scenarios.tensor;
import static com.eonmux.neuralfree.nn.multimodal.MultimodalStructuredParityTest.assertNear;
import static com.eonmux.neuralfree.nn.multimodal.MultimodalStructuredParityTest.toDouble;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The gradients of the nn/multimodal and nn/structured modules: the router's recorded weights and their combination
 * against host references, and every module's parameter gradients on each GPU API against the CPU path.
 *
 * <p>Each gradient scenario builds a module from a fixed seed, overwrites its parameters with seeded values, runs one
 * forward in grad mode and back-propagates {@code sum(out * r)} for a fixed random {@code r}, inside a
 * {@link ComputationGraph.StepScope} that must leave the graph at its size before the step; every parameter must get a
 * finite gradient. A backward pass chains the transposed GEMMs, the norm and softmax backward primitives and the
 * reductions of the forward over the same magnitudes, so device gradients are held to
 * {@code |dev - cpu| <= 5e-3 * max(1, |cpu|)}, the forward bound of {@link MultimodalStructuredParityTest} widened
 * for the second pass through the chain.</p>
 */
@RunWith(Enclosed.class)
public class MultimodalStructuredGradParityTest {

    static final double GRAD_TOL = 5e-3;

    /** The router on the CPU path, against host references. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
            ComputationGraph.setGradEnabled(true);
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(true);
            TrainGpu.release(host);
        }

        /** Without top-k, {@code routeWeights} is the recorded form of {@code route}'s weights. */
        @Test
        public void routeWeightsEqualRoutedWeights() {
            ModalityRouter router = new ModalityRouter(16, 3, "softmax", 0.7f, false, 1);
            try {
                router.initialize(host, new Random(22));
                randomize(router, 22, 0.5f);
                int before = ComputationGraph.current().size();
                try (Tensor seq = tensor(host, TrainParity.uniform(24, 4 * 5 * 16, -1, 1), 4, 5, 16)) {
                    ModalityRouter.RoutingResult routed = router.route(seq);
                    assertEquals("route records nothing", before, ComputationGraph.current().size());
                    try (ComputationGraph.StepScope step = ComputationGraph.current().step();
                         Tensor weights = router.routeWeights(seq)) {
                        assertTrue("the weights are recorded", weights.requiresGrad());
                        assertArrayEquals(routed.weights(), weights.toFloatArray(), 0.0f);
                    }
                }
            } finally {
                router.close();
            }
        }

        /** With top-k, the entries outside each row's top {@code k} are 0 and the selected ones keep their values. */
        @Test
        public void routeWeightsMaskEntriesOutsideTopK() {
            ModalityRouter router = new ModalityRouter(16, 3, "sigmoid", 0.5f, true, 2);
            try {
                router.initialize(host, new Random(23));
                randomize(router, 23, 0.5f);
                try (Tensor flat = tensor(host, TrainParity.uniform(25, 4 * 16, -1, 1), 4, 16);
                     ComputationGraph.StepScope step = ComputationGraph.current().step()) {
                    float[] all = router.route(flat).weights();
                    float[] kept;
                    try (Tensor weights = router.routeWeights(flat)) {
                        kept = weights.toFloatArray();
                    }
                    for (int b = 0; b < 4; b++) {
                        int dropped = smallest(all, b * 3, 3);
                        for (int m = 0; m < 3; m++) {
                            float expected = m == dropped ? 0.0f : all[b * 3 + m];
                            assertEquals("row " + b + " modality " + m, expected, kept[b * 3 + m], 0.0f);
                        }
                    }
                }
            } finally {
                router.close();
            }
        }

        /**
         * {@code combineEmbeddings(Map, Tensor)} against the per-row weighted mean in double, with a row whose weights
         * are all non-positive (the zero row); the gradient reaches the weight leaf, is 0 on the non-positive entries
         * and finite everywhere, and the unregistered name is skipped.
         */
        @Test
        public void combineWithWeightTensorMatchesReferenceAndGradient() {
            ModalityRouter router = new ModalityRouter(16, 3);
            try {
                router.initialize(host, new Random(26));
                float[] w = {0.2f, 0.5f, 0.3f, -0.4f, 0.0f, -1.0f, 0.7f, -0.2f, 0.1f};
                float[] vision = TrainParity.uniform(27, 3 * 4, -1, 1);
                float[] text = TrainParity.uniform(28, 3 * 4, -1, 1);
                float[] audio = TrainParity.uniform(29, 3 * 4, -1, 1);
                double[] expected = new double[12];
                float[][] e = {vision, text, audio};
                for (int b = 0; b < 3; b++) {
                    double total = 0;
                    for (int m = 0; m < 3; m++) {
                        total += Math.max(0, w[b * 3 + m]);
                    }
                    for (int d = 0; d < 4; d++) {
                        double acc = 0;
                        for (int m = 0; m < 3; m++) {
                            acc += Math.max(0, w[b * 3 + m]) * e[m][b * 4 + d];
                        }
                        expected[b * 4 + d] = total > 0 ? acc / total : 0.0;
                    }
                }
                try (Tensor raw = tensor(host, w, 3, 3);
                     Tensor weights = raw.withRequiresGrad(true);
                     Tensor v = tensor(host, vision, 3, 4);
                     Tensor t = tensor(host, text, 3, 4);
                     Tensor a = tensor(host, audio, 3, 4);
                     Tensor stray = tensor(host, new float[]{9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, 3, 4)) {
                    Map<String, Tensor> embeddings = new LinkedHashMap<>();
                    embeddings.put(ModalityRouter.MODALITY_VISION, v);
                    embeddings.put(ModalityRouter.MODALITY_TEXT, t);
                    embeddings.put(ModalityRouter.MODALITY_AUDIO, a);
                    embeddings.put("depth", stray);
                    int before = ComputationGraph.current().size();
                    try (ComputationGraph.StepScope step = ComputationGraph.current().step();
                         Tensor combined = router.combineEmbeddings(embeddings, weights);
                         Tensor loss = TensorOps.sum(combined)) {
                        assertNear("tensor-weighted combination", expected, combined.toFloatArray(), 1e-6);
                        loss.backward();
                    }
                    assertEquals("the step leaves the graph as it found it", before,
                        ComputationGraph.current().size());
                    assertNotNull("the weights get a gradient", weights.grad());
                    float[] g = weights.grad().toFloatArray();
                    for (int i = 0; i < g.length; i++) {
                        assertTrue("finite weight gradient " + i, Float.isFinite(g[i]));
                        if (!(w[i] > 0)) {
                            assertEquals("non-positive weight " + i + " gets no gradient", 0.0f, g[i], 0.0f);
                        }
                    }
                    weights.grad().close();
                }
            } finally {
                router.close();
            }
        }

        @Test
        public void combineWithoutRegisteredModalityIsNull() {
            ModalityRouter router = new ModalityRouter(16, 3);
            try {
                router.initialize(host, new Random(30));
                try (Tensor weights = tensor(host, new float[]{1, 1, 1}, 1, 3);
                     Tensor e = tensor(host, new float[]{1, 2}, 1, 2)) {
                    Map<String, Tensor> embeddings = new LinkedHashMap<>();
                    embeddings.put("depth", e);
                    assertNull(router.combineEmbeddings(embeddings, weights));
                }
            } finally {
                router.close();
            }
        }

        @Test
        public void everyGradientScenarioRunsOnTheCpuPath() {
            for (float[][] grads : Scenarios.all(host)) {
                assertFalse("a scenario has parameters", grads.length == 0);
            }
        }

        @Test
        public void visionEncoderEmbeddingDropoutAppliesOnlyInTraining() {
            VisionEncoder withDropout = Settings.vision(host, 0.3f);
            VisionEncoder plain = Settings.vision(host, 0.0f);
            try (Tensor image = Settings.image(host)) {
                Settings.dropoutOnlyInTraining("vision encoder", withDropout, plain,
                    m -> ((VisionEncoder) m).encode(image));
            } finally {
                withDropout.close();
                plain.close();
            }
        }

        @Test
        public void textEncoderEmbeddingDropoutAppliesOnlyInTraining() {
            TextEncoder withDropout = Settings.text(host, 0.3f);
            TextEncoder plain = Settings.text(host, 0.0f);
            try (Tensor tokens = ids(host, new int[]{3, 7, 1, 30, 0, 5, 5, 12, 31, 2}, 2, 5)) {
                Settings.dropoutOnlyInTraining("text encoder", withDropout, plain,
                    m -> ((TextEncoder) m).encode(tokens));
            } finally {
                withDropout.close();
                plain.close();
            }
        }

        @Test
        public void audioEncoderEmbeddingDropoutAppliesOnlyInTraining() {
            AudioEncoder withDropout = Settings.audio(host, 0.3f);
            AudioEncoder plain = Settings.audio(host, 0.0f);
            try (Tensor wave = tensor(host, TrainParity.uniform(17, 2 * 256, -1, 1), 2, 256)) {
                Settings.dropoutOnlyInTraining("audio encoder", withDropout, plain,
                    m -> ((AudioEncoder) m).encode(wave));
            } finally {
                withDropout.close();
                plain.close();
            }
        }

        @Test
        public void crossAttentionOutputDropoutAppliesOnlyInTraining() {
            CrossAttentionFusion withDropout = Settings.fusion(host, 0.3f);
            CrossAttentionFusion plain = Settings.fusion(host, 0.0f);
            try (Tensor query = Settings.query(host);
                 Tensor keyValue = Settings.keyValue(host)) {
                Settings.dropoutOnlyInTraining("cross-attention", withDropout, plain,
                    m -> ((CrossAttentionFusion) m).fuse(query, keyValue));
            } finally {
                withDropout.close();
                plain.close();
            }
        }

        @Test
        public void graphTransformerEdgeBiasEntersTheLogits() {
            Settings.edgeBias(host);
        }

        @Test
        public void graphTransformerRejectsMissingOrMismatchedEdgeFeatures() {
            Settings.rejectedEdgeFeatures(host);
        }

        @Test
        public void combineWithHostWeightsAppliesEachRow() {
            Settings.combineHostWeights(host);
        }

        private static int smallest(float[] values, int offset, int n) {
            int index = 0;
            for (int m = 1; m < n; m++) {
                if (values[offset + m] < values[offset + index]) {
                    index = m;
                }
            }
            return index;
        }
    }

    /** Parameter gradients on each GPU API against the CPU path. */
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
            ComputationGraph.setGradEnabled(true);
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            try {
                TrainGpu.release(host);
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void settingGradientsMatchCpuPath() {
            List<float[][]> cpu = Settings.gradientScenarios(host);
            List<float[][]> dev = Settings.gradientScenarios(gpu);
            assertEquals("scenario count", cpu.size(), dev.size());
            for (int s = 0; s < cpu.size(); s++) {
                assertEquals(Settings.NAMES[s] + ": parameter count", cpu.get(s).length, dev.get(s).length);
                for (int p = 0; p < cpu.get(s).length; p++) {
                    assertNear(api + " " + Settings.NAMES[s] + " parameter " + p + " gradient vs cpu path",
                        toDouble(cpu.get(s)[p]), dev.get(s)[p], GRAD_TOL);
                }
            }
        }

        @Test
        public void gradientsMatchCpuPath() {
            List<float[][]> cpu = Scenarios.all(host);
            List<float[][]> dev = Scenarios.all(gpu);
            assertEquals("scenario count", cpu.size(), dev.size());
            for (int s = 0; s < cpu.size(); s++) {
                assertEquals(Scenarios.NAMES[s] + ": parameter count", cpu.get(s).length, dev.get(s).length);
                for (int p = 0; p < cpu.get(s).length; p++) {
                    assertNear(api + " " + Scenarios.NAMES[s] + " parameter " + p + " gradient vs cpu path",
                        toDouble(cpu.get(s)[p]), dev.get(s)[p], GRAD_TOL);
                }
            }
        }
    }

    /** The gradient scenarios, each built and back-propagated on the given backend from fixed seeds. */
    static final class Scenarios {
        static final String[] NAMES = {
            "vision-language model", "audio encoder", "router weights and combine", "gaussian splatting",
            "graph transformer", "neural ODE rk4", "radiance field rays"
        };

        private Scenarios() {
        }

        static List<float[][]> all(ComputeBackend b) {
            List<float[][]> out = new ArrayList<>();
            out.add(vlm(b));
            out.add(audio(b));
            out.add(router(b));
            out.add(splat(b));
            out.add(graph(b));
            out.add(ode(b));
            out.add(nerf(b));
            return out;
        }

        static float[][] vlm(ComputeBackend b) {
            VisionLanguageModel model = new VisionLanguageModel(32, 16, 16, 16, 1, 4, 8);
            try {
                model.initialize(b, new Random(28), 8, 4, 3, 1, 1);
                randomize(model, 28, 0.3f);
                try (Tensor image = tensor(b, TrainParity.uniform(29, 3 * 8 * 8, -1, 1), 1, 3, 8, 8);
                     Tensor tokens = ids(b, new int[]{4, 9, 17, 2}, 1, 4)) {
                    return gradients(model, 81, m -> model.forward(image, tokens));
                }
            } finally {
                model.close();
            }
        }

        static float[][] audio(ComputeBackend b) {
            AudioEncoder encoder = new AudioEncoder(8000, 6, 64, 32, 16, 1, 4, "transformer", 0.1f);
            try {
                encoder.initialize(b, new Random(16));
                randomize(encoder, 16, 0.3f);
                try (Tensor wave = tensor(b, TrainParity.uniform(17, 2 * 256, -1, 1), 2, 256)) {
                    return gradients(encoder, 82, m -> encoder.encode(wave));
                }
            } finally {
                encoder.close();
            }
        }

        /** {@code combineEmbeddings(Map, routeWeights(x))} of a top-k router, so the gradient reaches its layer. */
        static float[][] router(ComputeBackend b) {
            ModalityRouter router = new ModalityRouter(16, 3, "softmax", 0.7f, true, 2);
            try {
                router.initialize(b, new Random(22));
                randomize(router, 22, 0.5f);
                try (Tensor seq = tensor(b, TrainParity.uniform(24, 4 * 5 * 16, -1, 1), 4, 5, 16);
                     Tensor vision = tensor(b, TrainParity.uniform(26, 4 * 16, -1, 1), 4, 16);
                     Tensor text = tensor(b, TrainParity.uniform(27, 4 * 16, -1, 1), 4, 16);
                     Tensor audio = tensor(b, TrainParity.uniform(31, 4 * 16, -1, 1), 4, 16)) {
                    Map<String, Tensor> embeddings = new LinkedHashMap<>();
                    embeddings.put(ModalityRouter.MODALITY_VISION, vision);
                    embeddings.put(ModalityRouter.MODALITY_TEXT, text);
                    embeddings.put(ModalityRouter.MODALITY_AUDIO, audio);
                    return gradients(router, 83, m -> {
                        try (Tensor weights = router.routeWeights(seq)) {
                            return router.combineEmbeddings(embeddings, weights);
                        }
                    });
                }
            } finally {
                router.close();
            }
        }

        /** The splats of {@link MultimodalStructuredParityTest}'s scenario, including the one at {@code w = 0}. */
        static float[][] splat(ComputeBackend b) {
            GaussianSplatting splats = new GaussianSplatting(3, 16, 16);
            try {
                splats.initialize(b, new Random(32));
                Map<String, Object> state = new LinkedHashMap<>();
                state.put("means", new float[]{0.13f, -0.21f, 0f, 0.4f, 0.35f, -1f, -0.45f, 0.1f, 1f});
                state.put("scales", new float[]{1.1f, 1.3f, 0.7f, 0.9f, 1.2f, 0.8f, 1.4f, 0.6f, 1.0f});
                state.put("rotations", new float[]{1, 0, 0, 0, 0.9f, 0.3f, 0.2f, 0.1f, 0.8f, -0.1f, 0.4f, 0.2f});
                state.put("opacities", new float[]{0.3f, 1.0f, -0.4f});
                state.put("shCoeffs", TrainParity.uniform(32, 3 * 48, -1, 1));
                splats.loadStateDict(state);
                float[] pose = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1};
                try (Tensor p = tensor(b, pose, 4, 4)) {
                    return gradients(splats, 84, m -> splats.render(p));
                }
            } finally {
                splats.close();
            }
        }

        static float[][] graph(ComputeBackend b) {
            float[] edges = {0, 2, 3, 1, 4, 2, 0, 5, 0, 1, 1, 1, 0, 0, 2, 3, 3, 1};
            GraphTransformer model = new GraphTransformer(6, 8, 5, 2, 2, 0.0f, false);
            try {
                model.initialize(b, new Random(33));
                randomize(model, 33, 0.5f);
                try (GraphData g = new GraphData(tensor(b, TrainParity.uniform(35, 36, -1, 1), 6, 6),
                         tensor(b, edges, 2, 9))) {
                    return gradients(model, 85, m -> model.forward(g));
                }
            } finally {
                model.close();
            }
        }

        static float[][] ode(ComputeBackend b) {
            NeuralODE ode = new NeuralODE(4, 8, 2, NeuralODE.Solver.RK4, 1e-5f, 1e-8f);
            try {
                ode.initialize(b, new Random(37));
                randomize(ode, 37, 0.5f);
                try (Tensor state = tensor(b, TrainParity.uniform(38, 8, -1, 1), 2, 4)) {
                    return gradients(ode, 86, m -> ode.forwardWithTrajectory(state, 0.0f, 0.5f, 6));
                }
            } finally {
                ode.close();
            }
        }

        static float[][] nerf(ComputeBackend b) {
            NeuralRadianceField nerf = new NeuralRadianceField(3, 2, 16, 3, new int[]{2}, 0.5f, 2.0f, 6);
            try {
                nerf.initialize(b, new Random(5));
                randomize(nerf, 5, 0.4f);
                try (Tensor origins = tensor(b, TrainParity.uniform(42, 9, -0.5f, 0.5f), 3, 3);
                     Tensor directions = tensor(b, TrainParity.uniform(43, 9, -1, 1), 3, 3)) {
                    return gradients(nerf, 87, m -> nerf.renderRays(origins, directions));
                }
            } finally {
                nerf.close();
            }
        }

        /**
         * Runs {@code forward} in grad mode inside a step scope, back-propagates {@code sum(out * r)} with {@code r}
         * uniform in {@code [-1, 1)} from {@code seed}, and returns every parameter's gradient in
         * {@link Module#parameters()} order; the gradients are then cleared.
         */
        private static float[][] gradients(Module module, long seed, Function<Module, Tensor> forward) {
            ComputationGraph graph = ComputationGraph.current();
            int before = graph.size();
            try (ComputationGraph.StepScope step = graph.step();
                 Tensor out = forward.apply(module);
                 Tensor r = tensor(out.backend(), TrainParity.uniform(seed, (int) out.numElements(), -1, 1),
                     out.shape().dims());
                 Tensor weighted = TensorOps.mul(out, r);
                 Tensor loss = TensorOps.sum(weighted)) {
                assertTrue("the forward is recorded", out.requiresGrad());
                loss.backward();
            }
            assertEquals("the step leaves the graph as it found it", before, graph.size());
            List<Parameter> params = module.parameters();
            float[][] grads = new float[params.size()][];
            for (int i = 0; i < grads.length; i++) {
                Parameter p = params.get(i);
                assertNotNull("gradient of parameter " + i + " (" + p.name() + ")", p.grad());
                grads[i] = p.grad().toFloatArray();
                for (float v : grads[i]) {
                    assertTrue("finite gradient of parameter " + i, Float.isFinite(v));
                }
            }
            module.zeroGrad();
            return grads;
        }
    }

    /**
     * The settings VisionEncoder, TextEncoder, AudioEncoder, CrossAttentionFusion, GraphTransformer and ModalityRouter
     * apply: the embedding and attention-output dropouts (training mode only), GraphTransformer's additive per-head
     * edge bias and the per-row host weights of {@code combineEmbeddings(Map, float[])}.
     */
    static final class Settings {
        static final String[] NAMES = {
            "vision encoder with embedding dropout (training)", "cross-attention with output dropout (training)",
            "graph transformer with edge bias"
        };
        private static final float[] EDGES = {0, 2, 3, 1, 4, 2, 0, 5, 0, 1, 1, 1, 0, 0, 2, 3, 3, 1};
        private static final int EDGE_DIM = 3;

        private Settings() {
        }

        /** The scenario's vision encoder with {@code dropout}, parameters randomized by name (equal for any rate). */
        static VisionEncoder vision(ComputeBackend b, float dropout) {
            VisionEncoder encoder = new VisionEncoder(8, 4, 3, 16, 1, 4, "transformer", true, dropout);
            try {
                encoder.initialize(b, new Random(12));
                randomize(encoder, 12, 0.3f);
                return encoder;
            } catch (RuntimeException e) {
                encoder.close();
                throw e;
            }
        }

        /** The scenario's text encoder with {@code dropout}, parameters randomized by name (equal for any rate). */
        static TextEncoder text(ComputeBackend b, float dropout) {
            TextEncoder encoder = new TextEncoder(32, 16, 1, 4, 8, dropout, "transformer");
            try {
                encoder.initialize(b, new Random(11));
                randomize(encoder, 11, 0.5f);
                return encoder;
            } catch (RuntimeException e) {
                encoder.close();
                throw e;
            }
        }

        /** The scenario's audio encoder with {@code dropout}, parameters randomized by name (equal for any rate). */
        static AudioEncoder audio(ComputeBackend b, float dropout) {
            AudioEncoder encoder = new AudioEncoder(8000, 6, 64, 32, 16, 1, 4, "transformer", dropout);
            try {
                encoder.initialize(b, new Random(16));
                randomize(encoder, 16, 0.3f);
                return encoder;
            } catch (RuntimeException e) {
                encoder.close();
                throw e;
            }
        }

        /** The scenario's gated cross-attention fusion with {@code dropout}, parameters randomized by name. */
        static CrossAttentionFusion fusion(ComputeBackend b, float dropout) {
            CrossAttentionFusion fusion = new CrossAttentionFusion(8, 12, 12, 16, 4, dropout, true);
            try {
                fusion.initialize(b, new Random(18));
                randomize(fusion, 18, 0.5f);
                return fusion;
            } catch (RuntimeException e) {
                fusion.close();
                throw e;
            }
        }

        static Tensor image(ComputeBackend b) {
            return tensor(b, TrainParity.uniform(14, 2 * 3 * 8 * 8, -1, 1), 2, 3, 8, 8);
        }

        static Tensor query(ComputeBackend b) {
            return tensor(b, TrainParity.uniform(19, 2 * 3 * 8, -1, 1), 2, 3, 8);
        }

        static Tensor keyValue(ComputeBackend b) {
            return tensor(b, TrainParity.uniform(20, 2 * 5 * 12, -1, 1), 2, 5, 12);
        }

        /**
         * A module with dropout against the same module without it (same seed and parameters; the dropout seed is
         * drawn after every weight): bit-identical in evaluation mode; in training mode the forward records exactly
         * one {@code MaskScaleGrad}, the step leaves the graph at its size before, and the output differs from the
         * evaluation output (a 30% mask over about a hundred elements drops some).
         */
        static void dropoutOnlyInTraining(String what, Module withDropout, Module plain,
                                          Function<Module, Tensor> forward) {
            float[] evaluated;
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                withDropout.eval();
                plain.eval();
                evaluated = values(forward.apply(withDropout));
                assertArrayEquals(what + ": evaluation mode ignores the dropout", values(forward.apply(plain)),
                    evaluated, 0.0f);
            }
            withDropout.train();
            ComputationGraph graph = ComputationGraph.current();
            int before = graph.size();
            float[] trained;
            try (ComputationGraph.StepScope step = graph.step();
                 Tensor out = forward.apply(withDropout)) {
                assertEquals(what + ": recorded dropouts", 1, countMaskScales(graph, before));
                trained = out.toFloatArray();
            }
            assertEquals(what + ": the step leaves the graph as it found it", before, graph.size());
            boolean differs = false;
            for (int i = 0; i < trained.length; i++) {
                differs |= trained[i] != evaluated[i];
            }
            assertTrue(what + ": training mode applies the dropout", differs);
        }

        /**
         * GraphTransformer with edge features against the same model without them (equal shared parameters): zero
         * edge layers give a bit-identical output, a per-head constant bias (weights 0) leaves it unchanged up to
         * rounding (the softmax over a target's edges is shift-invariant), and seeded edge layers change it. A loss on
         * the output gives every edge layer finite gradients, a non-zero one for the weight; the bias shifts every
         * logit of a head alike, so its gradient is zero up to rounding.
         */
        static void edgeBias(ComputeBackend b) {
            GraphTransformer plain = graphTransformer(b, false);
            GraphTransformer edged = graphTransformer(b, true);
            try (GraphData g = edgeGraph(b)) {
                float[] reference;
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    reference = values(plain.forward(g));
                    edged.loadStateDict(edgeLayers(new float[EDGE_DIM * 2], new float[2]));
                    assertArrayEquals("zero edge bias", reference, values(edged.forward(g)), 0.0f);
                    edged.loadStateDict(edgeLayers(new float[EDGE_DIM * 2], new float[]{0.7f, -1.3f}));
                    assertNear("per-head constant edge bias", toDouble(reference), values(edged.forward(g)), 1e-4);
                    randomize(edged, 33, 0.5f);
                    float[] biased = values(edged.forward(g));
                    boolean differs = false;
                    for (int i = 0; i < biased.length; i++) {
                        differs |= biased[i] != reference[i];
                    }
                    assertTrue("seeded edge layers change the attention", differs);
                }
                ComputationGraph graph = ComputationGraph.current();
                int before = graph.size();
                try (ComputationGraph.StepScope step = graph.step();
                     Tensor out = edged.forward(g);
                     Tensor loss = TensorOps.sum(out)) {
                    loss.backward();
                }
                assertEquals("the step leaves the graph as it found it", before, graph.size());
                for (int layer = 0; layer < 2; layer++) {
                    for (String name : new String[]{"weight", "bias"}) {
                        String key = "edge_" + layer + "." + name;
                        Parameter p = edged.getParameter(key);
                        assertNotNull(key + " gradient", p.grad());
                        boolean nonZero = false;
                        for (float v : p.grad().toFloatArray()) {
                            assertTrue(key + " finite gradient", Float.isFinite(v));
                            nonZero |= v != 0.0f;
                        }
                        if (name.equals("weight")) {
                            assertTrue(key + " receives a non-zero gradient", nonZero);
                        }
                    }
                }
                edged.zeroGrad();
            } finally {
                plain.close();
                edged.close();
            }
        }

        /** Missing edge features, a wrong width and a wrong edge feature dim are rejected before recording. */
        static void rejectedEdgeFeatures(ComputeBackend b) {
            GraphTransformer edged = graphTransformer(b, true);
            float[] narrow = TrainParity.uniform(36, 9 * 2, -1, 1);
            try (GraphData bare = new GraphData(tensor(b, TrainParity.uniform(35, 36, -1, 1), 6, 6),
                     tensor(b, EDGES, 2, 9));
                 GraphData mismatched = new GraphData(tensor(b, TrainParity.uniform(35, 36, -1, 1), 6, 6),
                     tensor(b, EDGES, 2, 9), tensor(b, narrow, 9, 2), null)) {
                int before = ComputationGraph.current().size();
                expectIllegalArgument("no edge features", () -> edged.forward(bare));
                expectIllegalArgument("edge feature width 2", () -> edged.forward(mismatched));
                assertEquals("rejected graphs record nothing", before, ComputationGraph.current().size());
            } finally {
                edged.close();
            }
            expectIllegalArgument("edgeFeatureDim 0", () -> new GraphTransformer(6, 8, 5, 2, 2, 0.0f, true, 0));
        }

        /**
         * {@code combineEmbeddings(Map, float[])} with {@code [batch, numModalities]} weights weights every row by its
         * own row (the non-positive row gives 0) and equals the tensor overload; one row applies to every embedding
         * row; a weight count that fits neither and a non-finite weight are rejected; without a registered modality
         * the result is null.
         */
        static void combineHostWeights(ComputeBackend b) {
            ModalityRouter router = new ModalityRouter(16, 3);
            try {
                router.initialize(b, new Random(26));
                float[] w = {0.2f, 0.5f, 0.3f, -0.4f, 0.0f, -1.0f, 0.7f, -0.2f, 0.1f};
                float[] row = {0.25f, 0.6f, 0.15f};
                float[][] e = {TrainParity.uniform(27, 12, -1, 1), TrainParity.uniform(28, 12, -1, 1),
                    TrainParity.uniform(29, 12, -1, 1)};
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                     Tensor v = tensor(b, e[0], 3, 4);
                     Tensor t = tensor(b, e[1], 3, 4);
                     Tensor a = tensor(b, e[2], 3, 4);
                     Tensor weightTensor = tensor(b, w, 3, 3);
                     Tensor stray = tensor(b, new float[]{1, 2}, 1, 2)) {
                    Map<String, Tensor> embeddings = new LinkedHashMap<>();
                    embeddings.put(ModalityRouter.MODALITY_VISION, v);
                    embeddings.put(ModalityRouter.MODALITY_TEXT, t);
                    embeddings.put(ModalityRouter.MODALITY_AUDIO, a);
                    float[] perRow = values(router.combineEmbeddings(embeddings, w));
                    assertNear("per-row host weights", weightedMean(w, e, true), perRow, 1e-6);
                    assertArrayEquals("host weights equal the tensor overload",
                        values(router.combineEmbeddings(embeddings, weightTensor)), perRow, 0.0f);
                    assertNear("one row of host weights", weightedMean(row, e, false),
                        values(router.combineEmbeddings(embeddings, row)), 1e-6);
                    expectIllegalArgument("four weights", () -> router.combineEmbeddings(embeddings,
                        new float[]{1, 1, 1, 1}));
                    expectIllegalArgument("NaN weight", () -> router.combineEmbeddings(embeddings,
                        new float[]{1, Float.NaN, 1}));
                    Map<String, Tensor> unregistered = new LinkedHashMap<>();
                    unregistered.put("depth", stray);
                    assertNull(router.combineEmbeddings(unregistered, row));
                }
            } finally {
                router.close();
            }
        }

        /**
         * The training-mode gradients of the dropout modules (equal seeds give equal masks on every backend) and of
         * the edge-biased graph transformer, in {@link #NAMES} order.
         */
        static List<float[][]> gradientScenarios(ComputeBackend b) {
            List<float[][]> out = new ArrayList<>();
            VisionEncoder encoder = vision(b, 0.3f);
            try (Tensor image = image(b)) {
                encoder.train();
                out.add(Scenarios.gradients(encoder, 88, m -> encoder.encode(image)));
            } finally {
                encoder.close();
            }
            CrossAttentionFusion fusion = fusion(b, 0.3f);
            try (Tensor query = query(b);
                 Tensor keyValue = keyValue(b)) {
                fusion.train();
                out.add(Scenarios.gradients(fusion, 89, m -> fusion.fuse(query, keyValue)));
            } finally {
                fusion.close();
            }
            GraphTransformer edged = graphTransformer(b, true);
            try (GraphData g = edgeGraph(b)) {
                out.add(Scenarios.gradients(edged, 90, m -> edged.forward(g)));
            } finally {
                edged.close();
            }
            return out;
        }

        private static GraphTransformer graphTransformer(ComputeBackend b, boolean edges) {
            GraphTransformer model = edges ? new GraphTransformer(6, 8, 5, 2, 2, 0.0f, true, EDGE_DIM)
                : new GraphTransformer(6, 8, 5, 2, 2, 0.0f, false);
            model.initialize(b, new Random(33));
            randomize(model, 33, 0.5f);
            return model;
        }

        /** The graph of the graph-transformer scenario with {@code [9, 3]} edge features. */
        private static GraphData edgeGraph(ComputeBackend b) {
            return new GraphData(tensor(b, TrainParity.uniform(35, 36, -1, 1), 6, 6), tensor(b, EDGES, 2, 9),
                tensor(b, TrainParity.uniform(36, 9 * EDGE_DIM, -1, 1), 9, EDGE_DIM), null);
        }

        /** Both edge layers' weights and biases set to the given values. */
        private static Map<String, Object> edgeLayers(float[] weight, float[] bias) {
            Map<String, Object> state = new LinkedHashMap<>();
            for (int layer = 0; layer < 2; layer++) {
                state.put("edge_" + layer + ".weight", weight.clone());
                state.put("edge_" + layer + ".bias", bias.clone());
            }
            return state;
        }

        /**
         * The double reference of the positive-weight mean over three {@code [3, 4]} embeddings, with
         * {@code weights} one row per embedding row ({@code perRow}) or one row for all.
         */
        private static double[] weightedMean(float[] weights, float[][] e, boolean perRow) {
            double[] expected = new double[12];
            for (int b = 0; b < 3; b++) {
                int offset = perRow ? b * 3 : 0;
                double total = 0;
                for (int m = 0; m < 3; m++) {
                    total += Math.max(0, weights[offset + m]);
                }
                for (int d = 0; d < 4; d++) {
                    double acc = 0;
                    for (int m = 0; m < 3; m++) {
                        acc += Math.max(0, weights[offset + m]) * e[m][b * 4 + d];
                    }
                    expected[b * 4 + d] = total > 0 ? acc / total : 0.0;
                }
            }
            return expected;
        }

        private static int countMaskScales(ComputationGraph graph, int from) {
            List<ComputationGraph.GraphNode> nodes = graph.getNodes();
            int count = 0;
            for (ComputationGraph.GraphNode node : nodes.subList(from, nodes.size())) {
                if (node.gradFn() instanceof IndexGradFunctions.MaskScaleGrad) {
                    count++;
                }
            }
            return count;
        }

        /** The host values of {@code t}, which is closed. */
        private static float[] values(Tensor t) {
            try (Tensor owned = t) {
                return owned.toFloatArray();
            }
        }

        private static void expectIllegalArgument(String what, Runnable call) {
            try {
                call.run();
                fail(what + ": expected IllegalArgumentException");
            } catch (IllegalArgumentException expected) {
                assertNotNull(what + ": message", expected.getMessage());
            }
        }
    }
}
