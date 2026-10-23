package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.graph.GraphData;
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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of every nn/structured module with parameters: {@link GaussianSplatting} through
 * its rasterizer ({@link RasterizeGrad}), {@link GraphTransformer}, {@link NeuralODE} with each solver and its adjoint
 * method, and {@link NeuralRadianceField}, on the CPU path and on each GPU API.
 *
 * <p>Each check compares the recorded gradient of {@code sum(out * r)}, for a fixed random {@code r}, with central
 * differences ({@link GradientCheckUtil}: step {@value #EPSILON}, tolerance {@value #TOLERANCE}). How the kinks stay
 * away from every evaluation point:</p>
 * <ul>
 *   <li><b>Splats.</b> The rasterizer's three-sigma boxes are piecewise constant in the splats. The fixture's centres
 *   and radii were chosen so that, for all three splats, {@code 3 sqrt(max(varX, varY))} is at least 0.11 from an
 *   integer and every box bound {@code px +- r}, {@code py +- r} at least 0.08 from one, while a step of
 *   {@value #EPSILON} in any parameter or pose entry moves {@code px}, {@code py} by at most {@code 6e-3} and the
 *   radius by at most {@code 5e-3}; every box lies inside the 12x12 image, the variances are far above their 0.001
 *   floor and the determinants far above theirs.</li>
 *   <li><b>Graph transformer.</b> Its only kink is the feed-forward ReLU after {@code norm2_i}. A layer norm output
 *   has {@code sum |y_j| <= (max|gamma| + max|beta|) * H}, so with {@code gamma in [0.8, 1.2]},
 *   {@code beta in [-0.1, 0.1]}, {@code H = 8}, {@code |ff1 weight| <= 0.05} and {@code ff1} biases of {@code +-1},
 *   every ReLU input is at least 0.48 from 0 whatever the other parameters and inputs. The attention's max shift is
 *   detached.</li>
 *   <li><b>Radiance field.</b> Every ReLU input is a bias of {@code +-1} (density: {@code +1}) plus {@code W x} with
 *   {@code |W| <= 0.4 / sum|x|} for the bound of the layer's input (positions within 1.05, encodings within 1, ReLU
 *   outputs within 1.4), so it stays at least 0.6 from 0. The encoding's {@code sin} and {@code cos} are
 *   smooth.</li>
 *   <li><b>Neural ODE.</b> Linear layers joined by tanh: smooth.</li>
 * </ul>
 *
 * <p>The adjoint method ({@link NeuralODE#adjoint}) integrates the adjoint backward with explicit Euler steps at the
 * reverse-integrated state, so it matches the gradient of the discretized forward only to {@code O(dt)}. Its check
 * first validates the backpropagated gradient of the Euler forward against finite differences, then requires the
 * adjoint to be within {@value #ADJOINT_GAP} (relative) of it at 101 time points and the gap to shrink at least by
 * half when the step is divided by four (its first-order rate predicts a quarter).</p>
 */
@RunWith(Enclosed.class)
public class StructuredGradCheckTest {

    static final float EPSILON = 1e-3f;
    static final float TOLERANCE = 1e-2f;
    static final int ELEMENTS = 4;
    static final double ADJOINT_GAP = 0.02;

    static final int IMAGE_SIZE = 12;
    static final float[] SPLAT_MEANS = {-0.31f, 0.22f, 0.0f, 0.18f, -0.27f, 0.4f, 0.05f, 0.08f, -0.3f};
    static final float[] SPLAT_SCALES = {0.9f, 0.6f, 0.7f, 0.55f, 0.8f, 0.5f, 0.7f, 0.45f, 0.9f};
    static final float[] SPLAT_ROTATIONS = {1.0f, 0.12f, -0.08f, 0.2f, 0.95f, -0.1f, 0.15f, -0.25f, 0.9f, 0.2f, 0.1f,
        0.05f};
    static final float[] SPLAT_OPACITIES = {0.4f, -0.3f, 0.9f};
    static final float[] IDENTITY_POSE = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    /** The checks, on the backend of the subclass. */
    public abstract static class Checks {

        protected abstract ComputeBackend backend();

        /** Every element of every splat parameter, and the camera pose. */
        @Test
        public void gaussianSplatting() {
            ComputeBackend b = backend();
            GaussianSplatting splats = new GaussianSplatting(3, IMAGE_SIZE, IMAGE_SIZE);
            try (Tensor pose = tensor(b, IDENTITY_POSE, 4, 4);
                 Tensor r = uniform(b, 201, IMAGE_SIZE, IMAGE_SIZE, 4)) {
                splats.initialize(b, new Random(202));
                Map<String, Object> state = new LinkedHashMap<>();
                state.put("means", SPLAT_MEANS);
                state.put("scales", SPLAT_SCALES);
                state.put("rotations", SPLAT_ROTATIONS);
                state.put("opacities", SPLAT_OPACITIES);
                state.put("shCoeffs", TrainParity.uniform(203, 3 * 48, -1.0f, 1.0f));
                splats.loadStateDict(state);
                assertParameters("GaussianSplatting", () -> weighted(splats.render(pose), r), splats, 3 * 48);
                assertInputs("GaussianSplatting pose", splats, in -> weighted(splats.render(in.get(0)), r), pose);
            } finally {
                splats.close();
            }
        }

        /** Two layers (the second reads the hidden width, with its residual) and per-edge features. */
        @Test
        public void graphTransformer() {
            ComputeBackend b = backend();
            GraphTransformer model = new GraphTransformer(4, 8, 3, 2, 2, 0.0f, true);
            int[] edges = {0, 2, 3, 1, 4, 2, 0, 5, 0, 1, 1, 1, 0, 0, 2, 3, 3, 1};
            try (Tensor x = uniform(b, 211, 6, 4);
                 Tensor r = uniform(b, 212, 6, 3);
                 GraphData graph = new GraphData(uniform(b, 211, 6, 4), ints(b, edges, 2, 9), uniform(b, 213, 9, 1),
                     null)) {
                model.initialize(b, new Random(214));
                load(model, StructuredGradCheckTest::graphTransformerValue);
                model.eval();
                assertParameters("GraphTransformer", () -> weighted(model.forward(graph, x), r), model, ELEMENTS);
                assertInputs("GraphTransformer node features", model,
                    in -> weighted(model.forward(graph, in.get(0)), r), x);
            } finally {
                model.close();
            }
        }

        @Test
        public void neuralOdeSolvers() {
            ComputeBackend b = backend();
            for (NeuralODE.Solver solver : NeuralODE.Solver.values()) {
                NeuralODE ode = new NeuralODE(3, 6, 2, solver, 1e-5f, 1e-8f);
                try (Tensor z0 = uniform(b, 221, 2, 3);
                     Tensor r = uniform(b, 222, 2, 3)) {
                    ode.initialize(b, new Random(223));
                    randomize(ode, 224, 0.5f);
                    String what = "NeuralODE " + solver;
                    assertParameters(what, () -> weighted(ode.forwardWithTrajectory(z0, 0.0f, 0.5f, 6), r), ode,
                        ELEMENTS);
                    assertInputs(what + " initial state", ode,
                        in -> weighted(ode.forwardWithTrajectory(in.get(0), 0.0f, 0.5f, 6), r), z0);
                } finally {
                    ode.close();
                }
            }
        }

        /** {@link NeuralODE#adjoint} against the backpropagated gradient of the discretized Euler forward. */
        @Test
        public void neuralOdeAdjoint() {
            ComputeBackend b = backend();
            NeuralODE ode = new NeuralODE(3, 6, 2, NeuralODE.Solver.EULER, 1e-5f, 1e-8f);
            try (Tensor z0 = uniform(b, 231, 2, 3);
                 Tensor r = uniform(b, 232, 2, 3)) {
                ode.initialize(b, new Random(233));
                randomize(ode, 234, 0.5f);
                assertParameters("NeuralODE Euler, 101 points",
                    () -> weighted(ode.forwardWithTrajectory(z0, 0.0f, 0.5f, 101), r), ode, ELEMENTS);

                float[][] fine = backpropagated(ode, z0, r, 101);
                double scale = 1.0;
                for (float[] g : fine) {
                    for (float v : g) {
                        scale = Math.max(scale, Math.abs(v));
                    }
                }
                double fineGap = adjointGap(ode, z0, r, 101, fine);
                double coarseGap = adjointGap(ode, z0, r, 26, backpropagated(ode, z0, r, 26));
                assertTrue("adjoint vs backpropagated gradient at 101 points: gap " + fineGap + ", scale " + scale,
                    fineGap <= ADJOINT_GAP * scale);
                assertTrue("the adjoint gap shrinks with the step: 26 points " + coarseGap + ", 101 points " + fineGap,
                    fineGap <= 0.5 * coarseGap + 1e-5 * scale);
            } finally {
                ode.close();
            }
        }

        /** Three rays of four samples through a two-layer trunk with a skip connection. */
        @Test
        public void neuralRadianceField() {
            ComputeBackend b = backend();
            NeuralRadianceField nerf = new NeuralRadianceField(2, 1, 8, 2, new int[]{1}, 0.5f, 1.5f, 4);
            try (Tensor origins = scaled(b, 241, 0.3f, 3, 3);
                 Tensor directions = scaled(b, 242, 0.5f, 3, 3);
                 Tensor r = uniform(b, 243, 3, 4)) {
                nerf.initialize(b, new Random(244));
                load(nerf, StructuredGradCheckTest::radianceFieldValue);
                assertParameters("NeuralRadianceField", () -> weighted(nerf.renderRays(origins, directions), r), nerf,
                    ELEMENTS);
            } finally {
                nerf.close();
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

    /**
     * The gradients of {@code sum(z(0.5) * r)} for the Euler forward over {@code points} time points, by one backward
     * pass: one row per parameter of {@code ode}, then the initial state's. The parameters' gradient slots are cleared.
     */
    static float[][] backpropagated(NeuralODE ode, Tensor z0, Tensor r, int points) {
        List<Parameter> params = ode.parameters();
        float[][] grads = new float[params.size() + 1][];
        int before = ComputationGraph.current().size();
        try (ComputationGraph.StepScope step = ComputationGraph.current().step();
             Tensor leaf = z0.withRequiresGrad(true)) {
            try (Tensor weightedOut = weighted(ode.forwardWithTrajectory(leaf, 0.0f, 0.5f, points), r);
                 Tensor loss = TensorOps.sum(weightedOut)) {
                loss.backward();
            }
            for (int i = 0; i < params.size(); i++) {
                assertNotNull("gradient of " + params.get(i).name(), params.get(i).grad());
                grads[i] = params.get(i).grad().toFloatArray();
            }
            Tensor stateGrad = leaf.grad();
            assertNotNull("gradient of the initial state", stateGrad);
            grads[params.size()] = stateGrad.toFloatArray();
            leaf.setGrad(null);
            stateGrad.close();
        } finally {
            ode.zeroGrad();
        }
        assertEquals("the backward step leaves the graph as it found it", before, ComputationGraph.current().size());
        return grads;
    }

    /**
     * The largest difference between {@link NeuralODE#adjoint} over {@code points} time points, seeded with
     * {@code r} at the Euler forward's final state, and {@code reference} (laid out as {@link #backpropagated}).
     */
    static double adjointGap(NeuralODE ode, Tensor z0, Tensor r, int points, float[][] reference) {
        List<Parameter> params = ode.parameters();
        int before = ComputationGraph.current().size();
        Tensor finalState;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            finalState = ode.forwardWithTrajectory(z0, 0.0f, 0.5f, points);
        }
        double gap = 0.0;
        try (Tensor zT = finalState;
             NeuralODE.AdjointResult adjoint = ode.adjoint(r, zT, 0.0f, 0.5f, points)) {
            for (int i = 0; i < params.size(); i++) {
                Tensor g = adjoint.gradientOf(params.get(i));
                assertNotNull("adjoint gradient of " + params.get(i).name(), g);
                gap = Math.max(gap, maxDifference(g.toFloatArray(), reference[i]));
            }
            gap = Math.max(gap, maxDifference(adjoint.stateAdjoint().toFloatArray(), reference[params.size()]));
        }
        assertEquals("the adjoint leaves the graph as it found it", before, ComputationGraph.current().size());
        return gap;
    }

    private static double maxDifference(float[] a, float[] b) {
        assertEquals("gradient length", b.length, a.length);
        double max = 0.0;
        for (int i = 0; i < a.length; i++) {
            assertTrue("finite adjoint gradient", Float.isFinite(a[i]));
            max = Math.max(max, Math.abs((double) a[i] - b[i]));
        }
        return max;
    }

    /** The graph-transformer fixture of the class description, per state-dict entry. */
    static float[] graphTransformerValue(String key, int n) {
        long seed = 250L + key.hashCode();
        boolean bias = key.endsWith(".bias");
        if (key.startsWith("norm2_")) {
            return bias ? TrainParity.uniform(seed, n, -0.1f, 0.1f) : TrainParity.uniform(seed, n, 0.8f, 1.2f);
        }
        if (key.startsWith("ff1_")) {
            return bias ? alternatingSigns(n) : TrainParity.uniform(seed, n, -0.05f, 0.05f);
        }
        return TrainParity.uniform(seed, n, -0.5f, 0.5f);
    }

    /**
     * The radiance-field fixture of the class description, per state-dict entry: weight bounds 0.4 over the input
     * bound of each ReLU layer (15.15 for the trunk's first layer, 26.2 for the skip layer, 11.2 for the density layer
     * and 18.7 for the first colour layer).
     */
    static float[] radianceFieldValue(String key, int n) {
        long seed = 260L + key.hashCode();
        String layer = key.substring(0, key.lastIndexOf('.'));
        boolean bias = key.endsWith(".bias");
        switch (layer) {
            case "layer_0":
                return bias ? alternatingSigns(n) : TrainParity.uniform(seed, n, -0.025f, 0.025f);
            case "layer_1":
                return bias ? alternatingSigns(n) : TrainParity.uniform(seed, n, -0.015f, 0.015f);
            case "density_layer":
                return bias ? filled(n, 1.0f) : TrainParity.uniform(seed, n, -0.035f, 0.035f);
            case "rgb_layer_1":
                return bias ? alternatingSigns(n) : TrainParity.uniform(seed, n, -0.02f, 0.02f);
            case "rgb_layer_2":
                return TrainParity.uniform(seed, n, -0.5f, 0.5f);
            default:
                throw new IllegalStateException("unexpected radiance-field parameter " + key);
        }
    }

    /** A value per state-dict entry of a module: its key and element count to its values. */
    interface Fixture {
        float[] values(String key, int n);
    }

    /** Overwrites every parameter of {@code module} with {@code fixture}'s values. */
    static void load(Module module, Fixture fixture) {
        Map<String, Object> state = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : module.stateDict().entrySet()) {
            state.put(e.getKey(), fixture.values(e.getKey(), ((float[]) e.getValue()).length));
        }
        module.loadStateDict(state);
    }

    /** Overwrites every parameter of {@code module} with seeded values in {@code [-scale, scale]}. */
    static void randomize(Module module, long seed, float scale) {
        load(module, (key, n) -> TrainParity.uniform(seed * 31 + key.hashCode(), n, -scale, scale));
    }

    private static float[] alternatingSigns(int n) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i % 2 == 0 ? 1.0f : -1.0f;
        }
        return values;
    }

    private static float[] filled(int n, float value) {
        float[] values = new float[n];
        Arrays.fill(values, value);
        return values;
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

    /** A FLOAT32 tensor of {@code dims} with seeded values in {@code [-1, 1]}. */
    static Tensor uniform(ComputeBackend b, long seed, int... dims) {
        return scaled(b, seed, 1.0f, dims);
    }

    /** A FLOAT32 tensor of {@code dims} with seeded values in {@code [-bound, bound]}. */
    static Tensor scaled(ComputeBackend b, long seed, float bound, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        return tensor(b, TrainParity.uniform(seed, n, -bound, bound), dims);
    }

    static Tensor tensor(ComputeBackend b, float[] values, int... dims) {
        try (Tensor flat = Tensor.fromArray(b, values)) {
            return flat.reshape(Shape.of(dims));
        }
    }

    static Tensor ints(ComputeBackend b, int[] values, int... dims) {
        try (Tensor flat = Tensor.fromArray(b, values)) {
            return flat.reshape(Shape.of(dims));
        }
    }
}
