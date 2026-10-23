package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.graph.GraphData;
import com.eonmux.neuralfree.nn.structured.GaussianSplatting;
import com.eonmux.neuralfree.nn.structured.GraphTransformer;
import com.eonmux.neuralfree.nn.structured.NeuralODE;
import com.eonmux.neuralfree.nn.structured.NeuralRadianceField;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The nn/multimodal and nn/structured modules on their CPU path and on each GPU API. Each scenario builds a module on
 * one backend from a fixed seed, overwrites every parameter with seeded values (so biases and norms are non-trivial)
 * and runs it on fixed inputs; the device result is compared with the CPU path's.
 *
 * <p>The modules chain GEMMs of length at most 64, layer norms, softmaxes and transcendentals over values of
 * magnitude about 1 ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract} reduction/GEMM and transcendental
 * classes), so forwards are held to {@code |dev - cpu| <= 2e-3 * max(1, |cpu|)}; the NeuralODE adjoint, a chain of
 * exact vector-Jacobian products, is held to the same bound. The host path is also checked against double-precision
 * references where the formula is short: the attention, the radiance field forward (sinusoidal encodings included) and
 * the adjoint of linear dynamics with its parameter gradients.</p>
 *
 * <p>The parameters require grad, so the forwards run in grad mode record their nodes; every scenario set runs inside
 * a {@link ComputationGraph.StepScope}, which must leave the graph at its size before the step. Generation records
 * nothing.</p>
 */
@RunWith(Enclosed.class)
public class MultimodalStructuredParityTest {

    static final double TOL = 2e-3;
    static final double REFERENCE_TOL = 5e-4;

    /** The CPU path, on a backend without the train library. */
    public static class HostPath {
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

        @Test
        public void maskedAttentionMatchesDoubleReference() {
            int batch = 2;
            int queryLen = 3;
            int keyLen = 4;
            int hidden = 8;
            int heads = 2;
            float[] q = TrainParity.uniform(1, batch * queryLen * hidden, -1, 1);
            float[] k = TrainParity.uniform(2, batch * keyLen * hidden, -1, 1);
            float[] v = TrainParity.uniform(3, batch * keyLen * hidden, -1, 1);
            float[] mask = {1, 0, 1, 1, 0, 0, 0, 0};
            float[] got;
            try (Tensor qt = Scenarios.tensor(host, q, batch, queryLen, hidden);
                 Tensor kt = Scenarios.tensor(host, k, batch, keyLen, hidden);
                 Tensor vt = Scenarios.tensor(host, v, batch, keyLen, hidden);
                 Tensor m = Scenarios.tensor(host, mask, batch, keyLen);
                 Tensor out = ScaledDotProductAttention.attend(qt, kt, vt, heads, m, true)) {
                got = out.toFloatArray();
            }
            double[] ref = References.attention(q, k, v, mask, batch, queryLen, keyLen, hidden, heads);
            assertNear("attention", ref, got, REFERENCE_TOL);
            for (int i = queryLen * hidden; i < 2 * queryLen * hidden; i++) {
                assertEquals("fully masked row is 0", 0.0f, got[i], 0.0f);
            }
        }

        @Test
        public void radianceFieldMatchesDoubleReference() {
            checkRadianceFieldReference(new NeuralRadianceField(3, 2, 16, 3, new int[]{2}, 0.5f, 2.0f, 6), 1.0f);
            checkRadianceFieldReference(new NeuralRadianceField(10, 4, 16, 2, new int[]{1}, 0.5f, 2.0f, 6), 3.0f);
        }

        private void checkRadianceFieldReference(NeuralRadianceField nerf, float coordRange) {
            int batch = 5;
            float[] input = TrainParity.uniform(41, batch * 6, -coordRange, coordRange);
            try {
                nerf.initialize(host, new Random(5));
                Scenarios.randomize(nerf, 5, 0.4f);
                float[] got;
                try (Tensor in = Scenarios.tensor(host, input, batch, 6);
                     Tensor out = nerf.forward(in)) {
                    got = out.toFloatArray();
                }
                double[] ref = References.radianceField(nerf, nerf.stateDict(), input, batch);
                assertNear("radiance field", ref, got, REFERENCE_TOL);
            } finally {
                nerf.close();
            }
        }

        @Test
        public void adjointOfLinearDynamicsMatchesDoubleReference() {
            float[] w = TrainParity.uniform(61, 16, -0.6f, 0.6f);
            float[] bias = TrainParity.uniform(63, 4, -0.2f, 0.2f);
            float[] z = TrainParity.uniform(40, 4, -0.5f, 0.5f);
            float[] grad = TrainParity.uniform(62, 4, -1, 1);
            float[][] got = Scenarios.adjoint(host, Scenarios.linearDynamics(w, bias), z, grad);

            double dt = -1.0 / 5;
            double[] a = toDouble(grad);
            double[] state = toDouble(z);
            double[] gw = new double[16];
            double[] gb = new double[4];
            for (int step = 0; step < 5; step++) {
                double[] nextA = new double[4];
                double[] nextZ = new double[4];
                for (int j = 0; j < 4; j++) {
                    double aw = 0;
                    double f = bias[j];
                    for (int i = 0; i < 4; i++) {
                        aw += a[i] * w[i * 4 + j];
                        f += w[j * 4 + i] * state[i];
                        gw[i * 4 + j] -= dt * a[i] * state[j];
                    }
                    gb[j] -= dt * a[j];
                    nextA[j] = a[j] - dt * aw;
                    nextZ[j] = state[j] + dt * f;
                }
                a = nextA;
                state = nextZ;
            }
            assertNear("linear adjoint", a, got[0], REFERENCE_TOL);
            assertNear("linear adjoint weight gradient", gw, got[1], REFERENCE_TOL);
            assertNear("linear adjoint bias gradient", gb, got[2], REFERENCE_TOL);
            assertArrayEquals("backwardAdjoint is the adjoint's state part", got[0], got[3], 0.0f);
        }

        @Test
        public void generateAppendsSampledTokens() {
            Scenarios.checkGenerate(host);
        }

        @Test
        public void everyScenarioRunsOnTheCpuPath() {
            for (float[] out : Scenarios.allInStep(host)) {
                for (float x : out) {
                    assertTrue("finite output", Float.isFinite(x));
                }
            }
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
        public void forwardsMatchCpuPath() {
            float[][] cpu = Scenarios.allInStep(host);
            float[][] dev = Scenarios.allInStep(gpu);
            assertEquals("scenario count", cpu.length, dev.length);
            for (int i = 0; i < cpu.length; i++) {
                assertNear(api + " " + Scenarios.NAMES[i] + " vs cpu path", toDouble(cpu[i]), dev[i], TOL);
            }
        }

        @Test
        public void adjointMatchesCpuPath() {
            Map<String, Object> state = Scenarios.linearDynamics(TrainParity.uniform(71, 16, -0.6f, 0.6f),
                TrainParity.uniform(73, 4, -0.2f, 0.2f));
            float[] z = TrainParity.uniform(74, 4, -0.5f, 0.5f);
            float[] grad = TrainParity.uniform(72, 4, -1, 1);
            float[][] cpu = Scenarios.adjoint(host, state, z, grad);
            float[][] dev = Scenarios.adjoint(gpu, state, z, grad);
            String[] parts = {"state adjoint", "weight gradient", "bias gradient", "backwardAdjoint"};
            for (int i = 0; i < parts.length; i++) {
                assertNear(api + " " + parts[i] + " vs cpu path", toDouble(cpu[i]), dev[i], TOL);
            }
        }

        @Test
        public void generateAppendsSampledTokens() {
            Scenarios.checkGenerate(gpu);
        }
    }

    static void assertNear(String what, double[] ref, float[] got, double tol) {
        assertEquals(what + ": length", ref.length, got.length);
        for (int i = 0; i < ref.length; i++) {
            double bound = tol * Math.max(1.0, Math.abs(ref[i]));
            if (!(Math.abs(got[i] - ref[i]) <= bound)) {
                fail(what + ": element " + i + " is " + got[i] + ", reference " + ref[i] + " (bound " + bound + ")");
            }
        }
    }

    static double[] toDouble(float[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }

    /** The module scenarios, each built and run on the given backend from fixed seeds. */
    static final class Scenarios {
        static final String[] NAMES = {
            "text encoder", "text encoder (wrapped positions)", "vision encoder", "vision encoder (cropped, no CLS)",
            "audio encoder", "cross-attention (masked)", "cross-attention (separate values)",
            "cross-attention (value width differs)", "router softmax",
            "router sigmoid", "router combine", "vision-language model", "gaussian splatting",
            "graph transformer (projected)", "graph transformer (residual)", "neural ODE euler", "neural ODE rk4",
            "neural ODE dopri5", "radiance field", "radiance field rays"
        };

        private Scenarios() {
        }

        /** {@link #all} in grad mode inside a step scope, which must return the graph to its size before. */
        static float[][] allInStep(ComputeBackend b) {
            ComputationGraph graph = ComputationGraph.current();
            int before = graph.size();
            float[][] out;
            try (ComputationGraph.StepScope step = graph.step()) {
                out = all(b);
            }
            assertEquals("the step scope releases every recorded forward node", before, graph.size());
            return out;
        }

        static float[][] all(ComputeBackend b) {
            float[][] text = text(b);
            float[][] vision = vision(b);
            float[][] fusion = fusion(b);
            float[][] router = router(b);
            float[][] graph = graph(b);
            float[][] nerf = nerf(b);
            return new float[][]{
                text[0], text[1], vision[0], vision[1], audio(b), fusion[0], fusion[1], fusion[2], router[0], router[1],
                router[2], vlm(b), splat(b), graph[0], graph[1], ode(b, NeuralODE.Solver.EULER),
                ode(b, NeuralODE.Solver.RK4), ode(b, NeuralODE.Solver.DOPRI5), nerf[0], nerf[1]
            };
        }

        static float[][] text(ComputeBackend b) {
            TextEncoder encoder = new TextEncoder(32, 16, 2, 4, 8);
            try {
                encoder.initialize(b, new Random(11));
                randomize(encoder, 11, 0.5f);
                return new float[][]{
                    run(encoder, ids(b, new int[]{3, 7, 1, 30, 0, 5, 5, 12, 31, 2}, 2, 5)),
                    run(encoder, ids(b, new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 1, 10))
                };
            } finally {
                encoder.close();
            }
        }

        static float[][] vision(ComputeBackend b) {
            VisionEncoder withCls = new VisionEncoder(8, 4, 3, 16, 2, 4, "transformer", true, 0.1f);
            VisionEncoder cropped = new VisionEncoder(8, 4, 3, 16, 1, 4, "transformer", false, 0.0f);
            try {
                withCls.initialize(b, new Random(12));
                randomize(withCls, 12, 0.3f);
                cropped.initialize(b, new Random(13));
                randomize(cropped, 13, 0.3f);
                return new float[][]{
                    run(withCls, tensor(b, TrainParity.uniform(14, 2 * 3 * 8 * 8, -1, 1), 2, 3, 8, 8)),
                    run(cropped, tensor(b, TrainParity.uniform(15, 3 * 9 * 10, -1, 1), 1, 3, 9, 10))
                };
            } finally {
                withCls.close();
                cropped.close();
            }
        }

        static float[] audio(ComputeBackend b) {
            AudioEncoder encoder = new AudioEncoder(8000, 6, 64, 32, 16, 1, 4, "transformer", 0.1f);
            try {
                encoder.initialize(b, new Random(16));
                randomize(encoder, 16, 0.3f);
                return run(encoder, tensor(b, TrainParity.uniform(17, 2 * 256, -1, 1), 2, 256));
            } finally {
                encoder.close();
            }
        }

        static float[][] fusion(ComputeBackend b) {
            CrossAttentionFusion fusion = new CrossAttentionFusion(8, 12, 12, 16, 4, 0.1f, true);
            CrossAttentionFusion narrowValues = new CrossAttentionFusion(8, 12, 10, 16, 4, 0.1f, true);
            try {
                fusion.initialize(b, new Random(18));
                randomize(fusion, 18, 0.5f);
                narrowValues.initialize(b, new Random(44));
                randomize(narrowValues, 44, 0.5f);
                float[] q = TrainParity.uniform(19, 2 * 3 * 8, -1, 1);
                float[] kv = TrainParity.uniform(20, 2 * 5 * 12, -1, 1);
                float[] v = TrainParity.uniform(21, 2 * 5 * 12, -1, 1);
                float[] narrow = TrainParity.uniform(45, 2 * 5 * 10, -1, 1);
                float[] mask = {1, 1, 0, 1, 0, 0, 0, 0, 0, 0};
                try (Tensor qt = tensor(b, q, 2, 3, 8);
                     Tensor kt = tensor(b, kv, 2, 5, 12);
                     Tensor vt = tensor(b, v, 2, 5, 12);
                     Tensor nt = tensor(b, narrow, 2, 5, 10);
                     Tensor m = tensor(b, mask, 2, 5);
                     Tensor masked = fusion.forward(qt, kt, kt, m);
                     Tensor separate = fusion.forward(qt, kt, vt);
                     Tensor narrowed = narrowValues.forward(qt, kt, nt, m)) {
                    return new float[][]{masked.toFloatArray(), separate.toFloatArray(), narrowed.toFloatArray()};
                }
            } finally {
                fusion.close();
                narrowValues.close();
            }
        }

        static float[][] router(ComputeBackend b) {
            ModalityRouter softmax = new ModalityRouter(16, 3, "softmax", 0.7f, false, 1);
            ModalityRouter sigmoid = new ModalityRouter(16, 3, "sigmoid", 0.5f, true, 2);
            try {
                softmax.initialize(b, new Random(22));
                randomize(softmax, 22, 0.5f);
                sigmoid.initialize(b, new Random(23));
                randomize(sigmoid, 23, 0.5f);
                try (Tensor seq = tensor(b, TrainParity.uniform(24, 4 * 5 * 16, -1, 1), 4, 5, 16);
                     Tensor flat = tensor(b, TrainParity.uniform(25, 4 * 16, -1, 1), 4, 16);
                     Tensor vision = tensor(b, TrainParity.uniform(26, 2 * 16, -1, 1), 2, 16);
                     Tensor text = tensor(b, TrainParity.uniform(27, 2 * 16, -1, 1), 2, 16)) {
                    ModalityRouter.RoutingResult a = softmax.route(seq);
                    ModalityRouter.RoutingResult c = sigmoid.route(flat);
                    Map<String, Tensor> embeddings = new LinkedHashMap<>();
                    embeddings.put(ModalityRouter.MODALITY_VISION, vision);
                    embeddings.put(ModalityRouter.MODALITY_TEXT, text);
                    try (Tensor combined = softmax.combineEmbeddings(embeddings, new float[]{0.25f, 0.6f, 0.15f})) {
                        return new float[][]{
                            concat(a.weights(), selections(a)), concat(c.weights(), selections(c)),
                            combined.toFloatArray()
                        };
                    }
                }
            } finally {
                softmax.close();
                sigmoid.close();
            }
        }

        static float[] vlm(ComputeBackend b) {
            VisionLanguageModel model = new VisionLanguageModel(32, 16, 16, 16, 2, 4, 8);
            try {
                model.initialize(b, new Random(28), 8, 4, 3, 1, 1);
                randomize(model, 28, 0.3f);
                try (Tensor image = tensor(b, TrainParity.uniform(29, 3 * 8 * 8, -1, 1), 1, 3, 8, 8);
                     Tensor tokens = ids(b, new int[]{4, 9, 17, 2}, 1, 4);
                     Tensor logits = model.forward(image, tokens)) {
                    return logits.toFloatArray();
                }
            } finally {
                model.close();
            }
        }

        static void checkGenerate(ComputeBackend b) {
            VisionLanguageModel model = new VisionLanguageModel(32, 16, 16, 16, 1, 4, 8);
            try {
                model.initialize(b, new Random(30), 8, 4, 3, 1, 1);
                int before = ComputationGraph.current().size();
                try (Tensor image = tensor(b, TrainParity.uniform(31, 2 * 3 * 8 * 8, -1, 1), 2, 3, 8, 8);
                     Tensor tokens = ids(b, new int[]{4, 9, 17, 2, 6, 1}, 2, 3);
                     Tensor generated = model.generate(image, tokens, 2, 0.8f)) {
                    assertTrue("generate runs in grad mode here", ComputationGraph.isGradEnabled());
                    assertEquals("generate records nothing", before, ComputationGraph.current().size());
                    assertFalse("generated ids do not require grad", generated.requiresGrad());
                    assertEquals(Shape.of(2, 5), generated.shape());
                    assertEquals(DType.INT32, generated.dtype());
                    int[] values = generated.toIntArray();
                    assertArrayEquals("prompt kept", new int[]{4, 9, 17, 2, 6, 1},
                        new int[]{values[0], values[1], values[2], values[5], values[6], values[7]});
                    for (int value : values) {
                        assertTrue("token in vocabulary", value >= 0 && value < 32);
                    }
                    assertArrayEquals("caller ids untouched", new int[]{4, 9, 17, 2, 6, 1}, tokens.toIntArray());
                }
            } finally {
                model.close();
            }
        }

        /**
         * Three Gaussians through a pose whose last row is {@code [0, 0, 1, 1]}, so {@code w = z + 1}: one at
         * {@code w = 1} with identity rotation, one at {@code w = 0} (no perspective divide), one at {@code w = 2};
         * centres and footprints are away from pixel boundaries.
         */
        static float[] splat(ComputeBackend b) {
            GaussianSplatting splats = new GaussianSplatting(3, 16, 16);
            try {
                splats.initialize(b, new Random(32));
                Map<String, Object> state = new LinkedHashMap<>();
                state.put("means", new float[]{0.13f, -0.21f, 0f, 0.4f, 0.35f, -1f, -0.45f, 0.1f, 1f});
                state.put("scales", new float[]{1.1f, 1.3f, 0.7f, 0.9f, 1.2f, 0.8f, 1.4f, 0.6f, 1.0f});
                state.put("rotations", concat(new float[]{1, 0, 0, 0}, concat(
                    unit(0.9f, 0.3f, 0.2f, 0.1f), unit(0.8f, -0.1f, 0.4f, 0.2f))));
                state.put("opacities", new float[]{0.3f, 1.0f, -0.4f});
                float[] sh = new float[3 * 48];
                float[] colors = {0.5f, -0.2f, 1.1f, -0.7f, 0.3f, 0.9f, 0.1f, 0.8f, -1.2f};
                for (int g = 0; g < 3; g++) {
                    System.arraycopy(colors, g * 3, sh, g * 48, 3);
                }
                state.put("shCoeffs", sh);
                splats.loadStateDict(state);
                float[] pose = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1};
                try (Tensor p = tensor(b, pose, 4, 4);
                     Tensor image = splats.forward(p)) {
                    return image.toFloatArray();
                }
            } finally {
                splats.close();
            }
        }

        /** A 6-node graph with a duplicate edge, a self loop and node 5 without incoming edges. */
        static float[][] graph(ComputeBackend b) {
            float[] edges = {0, 2, 3, 1, 4, 2, 0, 5, 0, 1, 1, 1, 0, 0, 2, 3, 3, 1};
            GraphTransformer projected = new GraphTransformer(6, 8, 5, 2, 2, 0.0f, false);
            GraphTransformer residual = new GraphTransformer(8, 8, 8, 4, 1, 0.0f, false);
            try {
                projected.initialize(b, new Random(33));
                randomize(projected, 33, 0.5f);
                projected.eval();
                residual.initialize(b, new Random(34));
                randomize(residual, 34, 0.5f);
                residual.eval();
                float[] a;
                try (GraphData g = new GraphData(tensor(b, TrainParity.uniform(35, 36, -1, 1), 6, 6),
                         tensor(b, edges, 2, 9));
                     Tensor out = projected.forward(g)) {
                    a = out.toFloatArray();
                }
                try (GraphData g = new GraphData(tensor(b, TrainParity.uniform(36, 48, -1, 1), 6, 8),
                         tensor(b, edges, 2, 9));
                     Tensor out = residual.forward(g)) {
                    return new float[][]{a, out.toFloatArray()};
                }
            } finally {
                projected.close();
                residual.close();
            }
        }

        static float[] ode(ComputeBackend b, NeuralODE.Solver solver) {
            NeuralODE ode = new NeuralODE(4, 8, 2, solver, 1e-5f, 1e-8f);
            try {
                ode.initialize(b, new Random(37));
                randomize(ode, 37, 0.5f);
                try (Tensor state = tensor(b, TrainParity.uniform(38, 8, -1, 1), 2, 4);
                     Tensor out = ode.forward(state, 0.0f, 0.5f)) {
                    return out.toFloatArray();
                }
            } finally {
                ode.close();
            }
        }

        /** The state dict of one-layer linear dynamics {@code f(z) = z W^T + b}. */
        static Map<String, Object> linearDynamics(float[] weight, float[] bias) {
            Map<String, Object> state = new LinkedHashMap<>();
            state.put("layer_0.weight", weight);
            state.put("layer_0.bias", bias);
            return state;
        }

        /**
         * The adjoint of one-layer dynamics with the given state dict over 6 time points from {@code t1 = 1} to
         * {@code t0 = 0}, from the final state {@code z} and output gradient {@code grad}: the state adjoint, the
         * weight and bias gradients of {@code adjoint}, and the result of {@code backwardAdjoint}. Neither call may
         * write a gradient slot or leave a node in the graph.
         */
        static float[][] adjoint(ComputeBackend b, Map<String, Object> state, float[] z, float[] grad) {
            NeuralODE ode = new NeuralODE(4, 4, 1, NeuralODE.Solver.EULER, 1e-5f, 1e-8f);
            try {
                ode.initialize(b, new Random(39));
                ode.loadStateDict(state);
                int before = ComputationGraph.current().size();
                try (Tensor g = tensor(b, grad, 1, 4);
                     Tensor finalState = tensor(b, z, 1, 4);
                     NeuralODE.AdjointResult result = ode.adjoint(g, finalState, 0.0f, 1.0f, 6);
                     Tensor stateOnly = ode.backwardAdjoint(g, finalState, 0.0f, 1.0f, 6)) {
                    assertEquals("the adjoint leaves no node behind", before, ComputationGraph.current().size());
                    for (Parameter p : ode.parameters()) {
                        assertNull("the adjoint writes no gradient slot", p.grad());
                    }
                    assertFalse(result.stateAdjoint().requiresGrad());
                    return new float[][]{
                        result.stateAdjoint().toFloatArray(),
                        result.gradientOf(ode.layers().get(0).weight()).toFloatArray(),
                        result.gradientOf(ode.layers().get(0).bias()).toFloatArray(),
                        stateOnly.toFloatArray()
                    };
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
                try (Tensor in = tensor(b, TrainParity.uniform(41, 30, -1, 1), 5, 6);
                     Tensor out = nerf.forward(in);
                     Tensor origins = tensor(b, TrainParity.uniform(42, 9, -0.5f, 0.5f), 3, 3);
                     Tensor directions = tensor(b, TrainParity.uniform(43, 9, -1, 1), 3, 3);
                     Tensor rays = nerf.renderRays(origins, directions)) {
                    return new float[][]{out.toFloatArray(), rays.toFloatArray()};
                }
            } finally {
                nerf.close();
            }
        }

        /** Overwrites every parameter with uniform values in {@code [-scale, scale)} seeded by its name. */
        static void randomize(Module module, long seed, float scale) {
            Map<String, Object> state = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : module.stateDict().entrySet()) {
                int n = ((float[]) e.getValue()).length;
                state.put(e.getKey(), TrainParity.uniform(seed * 31 + e.getKey().hashCode(), n, -scale, scale));
            }
            module.loadStateDict(state);
        }

        /** Runs {@code module} on {@code input}, closing both tensors. */
        private static float[] run(Module module, Tensor input) {
            try (Tensor in = input;
                 Tensor out = module.forward(in)) {
                return out.toFloatArray();
            }
        }

        static Tensor tensor(ComputeBackend b, float[] data, int... dims) {
            try (Tensor flat = Tensor.fromArray(b, data)) {
                return flat.reshape(Shape.of(dims));
            }
        }

        static Tensor ids(ComputeBackend b, int[] data, int... dims) {
            try (Tensor flat = Tensor.fromArray(b, data)) {
                return flat.reshape(Shape.of(dims));
            }
        }

        private static float[] selections(ModalityRouter.RoutingResult result) {
            int[] selected = result.selectedModalities();
            float[] out = new float[selected.length];
            for (int i = 0; i < selected.length; i++) {
                out[i] = selected[i];
            }
            return out;
        }

        private static float[] unit(float w, float x, float y, float z) {
            float n = (float) Math.sqrt(w * w + x * x + y * y + z * z);
            return new float[]{w / n, x / n, y / n, z / n};
        }

        private static float[] concat(float[] a, float[] b) {
            float[] out = new float[a.length + b.length];
            System.arraycopy(a, 0, out, 0, a.length);
            System.arraycopy(b, 0, out, a.length, b.length);
            return out;
        }
    }

    /** Double-precision references of the module formulas. */
    static final class References {
        private References() {
        }

        static double[] attention(float[] q, float[] k, float[] v, float[] mask, int batch, int queryLen, int keyLen,
                                  int hidden, int heads) {
            int headDim = hidden / heads;
            double scale = 1.0 / Math.sqrt(headDim);
            double[] out = new double[batch * queryLen * hidden];
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < heads; h++) {
                    for (int i = 0; i < queryLen; i++) {
                        double[] p = new double[keyLen];
                        double max = Double.NEGATIVE_INFINITY;
                        for (int j = 0; j < keyLen; j++) {
                            if (mask[b * keyLen + j] == 0) {
                                p[j] = Double.NEGATIVE_INFINITY;
                                continue;
                            }
                            double dot = 0;
                            for (int d = 0; d < headDim; d++) {
                                dot += q[(b * queryLen + i) * hidden + h * headDim + d]
                                    * k[(b * keyLen + j) * hidden + h * headDim + d];
                            }
                            p[j] = dot * scale;
                            max = Math.max(max, p[j]);
                        }
                        double sum = 0;
                        for (int j = 0; j < keyLen; j++) {
                            p[j] = p[j] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(p[j] - max);
                            sum += p[j];
                        }
                        for (int d = 0; d < headDim; d++) {
                            double acc = 0;
                            for (int j = 0; j < keyLen; j++) {
                                acc += sum > 0 ? p[j] / sum * v[(b * keyLen + j) * hidden + h * headDim + d] : 0;
                            }
                            out[(b * queryLen + i) * hidden + h * headDim + d] = acc;
                        }
                    }
                }
            }
            return out;
        }

        static double[] radianceField(NeuralRadianceField nerf, Map<String, Object> state, float[] input, int batch) {
            int hidden = nerf.hiddenDim();
            double[] out = new double[batch * 4];
            for (int r = 0; r < batch; r++) {
                double[] pos = encoded(input, r * 6, nerf.posEncodingDim());
                double[] dir = encoded(input, r * 6 + 3, nerf.dirEncodingDim());
                double[] h = pos;
                for (int i = 0; i < nerf.numLayers(); i++) {
                    int in = nerf.layers().get(i).inFeatures();
                    double[] x = in == h.length ? h : join(h, pos);
                    h = relu(dense(state, "layer_" + i, x, hidden));
                }
                double density = relu(dense(state, "density_layer", h, 1))[0];
                double[] c = relu(dense(state, "rgb_layer_1", join(h, dir), hidden / 2));
                double[] rgb = dense(state, "rgb_layer_2", c, 3);
                for (int j = 0; j < 3; j++) {
                    out[r * 4 + j] = 1.0 / (1.0 + Math.exp(-rgb[j]));
                }
                out[r * 4 + 3] = density;
            }
            return out;
        }

        /** {@code [c, (sin, cos) of (float) ((2^k * c) * (float) PI) per coordinate and frequency]}. */
        private static double[] encoded(float[] input, int offset, int frequencies) {
            double[] out = new double[3 + 6 * frequencies];
            for (int j = 0; j < 3; j++) {
                float c = input[offset + j];
                out[j] = c;
                for (int k = 0; k < frequencies; k++) {
                    float angle = (float) Math.pow(2.0, k) * c * (float) Math.PI;
                    out[3 + j * 2 * frequencies + 2 * k] = Math.sin(angle);
                    out[3 + j * 2 * frequencies + 2 * k + 1] = Math.cos(angle);
                }
            }
            return out;
        }

        private static double[] dense(Map<String, Object> state, String layer, double[] x, int outFeatures) {
            float[] w = (float[]) state.get(layer + ".weight");
            float[] b = (float[]) state.get(layer + ".bias");
            double[] y = new double[outFeatures];
            for (int o = 0; o < outFeatures; o++) {
                double acc = b[o];
                for (int i = 0; i < x.length; i++) {
                    acc += w[o * x.length + i] * x[i];
                }
                y[o] = acc;
            }
            return y;
        }

        private static double[] relu(double[] x) {
            double[] y = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                y[i] = Math.max(0, x[i]);
            }
            return y;
        }

        private static double[] join(double[] a, double[] b) {
            double[] out = new double[a.length + b.length];
            System.arraycopy(a, 0, out, 0, a.length);
            System.arraycopy(b, 0, out, a.length, b.length);
            return out;
        }
    }
}
