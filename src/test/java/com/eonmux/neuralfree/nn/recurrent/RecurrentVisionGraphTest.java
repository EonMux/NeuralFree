package com.eonmux.neuralfree.nn.recurrent;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.vision.BottleneckBlock;
import com.eonmux.neuralfree.nn.vision.DepthwiseSeparableConv;
import com.eonmux.neuralfree.nn.vision.ResNet;
import com.eonmux.neuralfree.nn.vision.ResidualBlock;
import com.eonmux.neuralfree.nn.vision.UNet;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
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
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The recurrent and vision modules build an autograd graph: a forward pass in grad mode records a
 * graph whose backward pass delivers a gradient of the parameter's shape to every parameter and to the input, and
 * releases every node it recorded, so the thread's graph is back at its size before the step. {@link CpuPath} checks
 * this on the host reference backend, together with the exact output-scale gradient of the state-space model, finite
 * gradients for a state whose decay underflows, and a forward under {@code noGrad} that records nothing.
 * {@link GpuParity} compares the gradients of every parameter and of the input on each available GPU API with the same
 * module, built from the same seeds, on the host backend, under the composite bound of
 * {@link RecurrentVisionParityTest#assertComposite}: 1e-3 for the recurrent modules (transcendental steps chained
 * through GEMM backward passes) and 1e-2 for the vision blocks (the gradient-check tolerance of
 * {@code GradientCheckUtil}; batch-normalization backward passes rescale earlier differences).
 */
@RunWith(Enclosed.class)
public class RecurrentVisionGraphTest {

    private static final long WEIGHT_SEED = 5151L;
    private static final float RECURRENT_GRAD_TOL = 1e-3f;
    private static final float VISION_GRAD_TOL = 1e-2f;
    private static final String INPUT = "<input>";

    // ------------------------------------------------------------------ host reference backend

    public static class CpuPath {
        private final RecurrentVisionParityTest.Fixture fx = new RecurrentVisionParityTest.Fixture();
        private final ComputeBackend host = TrainParity.hostBackend();

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
        }

        @After
        public void tearDown() {
            fx.close();
        }

        @Test
        public void recurrentModulesBackPropagateToEveryParameterAndTheInput() {
            checkReach(fx.module(new LSTM(3, 4, 2, true, 0.0f, true)), b -> ((LSTM) b.module).initialize(b.backend,
                    new Random(WEIGHT_SEED)), 2, 5, 3);
            checkReach(fx.module(new GRU(3, 4, 2, true, 0.0f, true)), b -> ((GRU) b.module).initialize(b.backend,
                    new Random(WEIGHT_SEED)), 2, 5, 3);
            checkReach(fx.module(new RNN(3, 4, 2, true, 0.0, RNNCell.RELU, true)), b -> ((RNN) b.module).initialize(
                    b.backend, new Random(WEIGHT_SEED)), 2, 5, 3);
            checkReach(fx.module(new StateSpaceModel(3, 4, 5, 2, true, 0.0f)), b -> {
                ((StateSpaceModel) b.module).initialize(b.backend, new Random(WEIGHT_SEED));
                fx.fillSsmState(b.module);
            }, 2, 4, 3);
            checkReach(fx.module(new StateSpaceModel(3, 4, 2)), b -> {
                ((StateSpaceModel) b.module).initialize(b.backend, new Random(WEIGHT_SEED));
                fx.fillSsmState(b.module);
            }, 2, 4, 3);
        }

        @Test
        public void visionModulesBackPropagateToEveryParameterAndTheInput() {
            checkReach(fx.module(new ResidualBlock(4, 8, new int[]{2, 2})), b -> ((ResidualBlock) b.module).initialize(
                    b.backend, new Random(WEIGHT_SEED)), 2, 4, 6, 6);
            checkReach(fx.module(new BottleneckBlock(8, 16, 32, new int[]{2, 2}, true, 4)),
                    b -> ((BottleneckBlock) b.module).initialize(b.backend, new Random(WEIGHT_SEED)), 2, 8, 6, 6);
            checkReach(fx.module(new DepthwiseSeparableConv(4, 6, new int[]{3, 3}, new int[]{1, 1},
                    new int[]{1, 1})), b -> ((DepthwiseSeparableConv) b.module).initialize(b.backend), 2, 4, 5, 5);
            checkReach(fx.module(new UNet(1, 2, new int[]{2, 4}, true)), b -> ((UNet) b.module).initialize(b.backend,
                    new Random(WEIGHT_SEED)), 1, 1, 9, 9);
            checkReach(fx.module(ResNet.resnet18(10)), b -> ((ResNet) b.module).initialize(b.backend,
                    new Random(WEIGHT_SEED)), 2, 3, 32, 32);
        }

        /**
         * {@code out = y * d} broadcasts the readout {@code y [batch, seqLen, 1]} over {@code d [hiddenSize]}, so with
         * an all-ones upstream gradient every element of {@code d}'s gradient is the sum of the readouts.
         */
        @Test
        public void stateSpaceOutputScaleGradientIsTheSummedReadout() {
            StateSpaceModel ssm = fx.module(new StateSpaceModel(3, 4, 5));
            ssm.initialize(host, new Random(WEIGHT_SEED));
            fx.fillBiases(ssm);
            fx.fillSsmState(ssm);
            float[] x = TrainParity.uniform(41, 2 * 4 * 3, -1f, 1f);
            Tensor out = fx.keep(ssm.forward(fx.upload(host, x, 2, 4, 3)));
            try (Tensor seed = Tensor.ones(host, out.shape())) {
                out.backward(seed);
            }

            float[] readout = RecurrentVisionParityTest.Ref.ssmReadout(x, 2, 4, 3, 4, 5, ssm.namedParameters());
            float total = 0.0f;
            for (float y : readout) {
                total += y;
            }
            float[] expected = new float[4];
            Arrays.fill(expected, total);
            Parameter d = ssm.getParameter("layer_0.d");
            assertNotNull("d receives a gradient", d.grad());
            RecurrentVisionParityTest.assertComposite("CPU StateSpaceModel dL/dd", expected, d.grad().toFloatArray(),
                    RECURRENT_GRAD_TOL);

            Parameter aLog = ssm.getParameter("layer_0.a_log");
            assertNotNull("a_log receives a gradient", aLog.grad());
            assertFinite("dL/da_log", aLog.grad().toFloatArray());
            assertTrue("the decaying states give a_log a non-zero gradient", anyNonZero(aLog.grad().toFloatArray()));
        }

        /**
         * A state whose {@code a = exp(a_log)} underflows to zero takes the {@code delta} branch; the unselected
         * ratio must not turn its gradient into NaN.
         */
        @Test
        public void stateSpaceStateWithAnUnderflowedDecayKeepsFiniteGradients() {
            StateSpaceModel ssm = fx.module(new StateSpaceModel(3, 4, 5));
            ssm.initialize(host, new Random(WEIGHT_SEED));
            fx.fillBiases(ssm);
            fx.fillSsmState(ssm);
            Parameter aLog = ssm.getParameter("layer_0.a_log");
            float[] values = aLog.tensor().toFloatArray();
            values[1] = -200.0f;
            fx.write(aLog.tensor(), values);

            float[] x = TrainParity.uniform(42, 2 * 4 * 3, -1f, 1f);
            Tensor input = fx.leaf(host, x, 2, 4, 3);
            Tensor out = fx.keep(ssm.forward(input));
            float[] ref = RecurrentVisionParityTest.Ref.ssm(x, 2, 4, 3, 4, 5, ssm.namedParameters());
            TrainParity.assertTranscendental("CPU StateSpaceModel with a = 0", ref, out.toFloatArray());
            try (Tensor seed = Tensor.ones(host, out.shape())) {
                out.backward(seed);
            }

            for (Map.Entry<String, Parameter> e : ssm.namedParameters().entrySet()) {
                assertNotNull(e.getKey() + " receives a gradient", e.getValue().grad());
                assertFinite("dL/d" + e.getKey(), e.getValue().grad().toFloatArray());
            }
            assertFinite("dL/dinput", input.grad().toFloatArray());
        }

        @Test
        public void forwardUnderNoGradRecordsNothing() {
            LSTM lstm = fx.module(new LSTM(3, 4, 1));
            lstm.initialize(host, new Random(WEIGHT_SEED));
            StateSpaceModel ssm = fx.module(new StateSpaceModel(3, 4, 5));
            ssm.initialize(host, new Random(WEIGHT_SEED));
            Tensor x = fx.leaf(host, TrainParity.uniform(43, 2 * 4 * 3, -1f, 1f), 2, 4, 3);
            ComputationGraph graph = ComputationGraph.current();
            int before = graph.size();
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Tensor lstmOut = fx.keep(lstm.forward(x));
                Tensor ssmOut = fx.keep(ssm.forward(x));
                assertFalse(lstmOut.requiresGrad());
                assertNull(lstmOut.gradFn());
                assertFalse(ssmOut.requiresGrad());
                assertNull(ssmOut.gradFn());
            }
            assertEquals("nothing recorded under noGrad", before, graph.size());
        }

        @Test
        public void lstmCellGradientsReachBothPreviousStates() {
            LSTMCell cell = fx.module(new LSTMCell(4, 5));
            cell.initialize(host, new Random(WEIGHT_SEED));
            fx.fillBiases(cell);
            Tensor x = fx.leaf(host, TrainParity.uniform(44, 3 * 4, -1f, 1f), 3, 4);
            Tensor h = fx.leaf(host, TrainParity.uniform(45, 3 * 5, -1f, 1f), 3, 5);
            Tensor c = fx.leaf(host, TrainParity.uniform(46, 3 * 5, -1f, 1f), 3, 5);
            int before = ComputationGraph.current().size();

            LSTMCell.LSTMCellOutput out = cell.forwardWithCell(x, h, c);
            Tensor hidden = fx.keep(out.hidden());
            fx.keep(out.cell());
            assertNotNull(hidden.gradFn());
            assertNotNull(out.cell().gradFn());
            try (Tensor seed = Tensor.ones(host, hidden.shape())) {
                hidden.backward(seed);
            }

            assertNotNull("the input receives a gradient", x.grad());
            assertNotNull("hPrev receives a gradient through the recurrent layer", h.grad());
            assertNotNull("cPrev receives a gradient through the forget gate", c.grad());
            for (Map.Entry<String, Parameter> e : cell.namedParameters().entrySet()) {
                assertNotNull(e.getKey() + " receives a gradient", e.getValue().grad());
                assertEquals(e.getValue().shape(), e.getValue().grad().shape());
            }
            assertEquals("the backward pass releases every node of the step", before,
                    ComputationGraph.current().size());
        }

        @Test
        public void gruAndRnnCellGradientsReachThePreviousState() {
            GRUCell gru = fx.module(new GRUCell(4, 5));
            gru.initialize(host, new Random(WEIGHT_SEED));
            RNNCell rnn = fx.module(new RNNCell(4, 5, true, RNNCell.TANH));
            rnn.initialize(host, new Random(WEIGHT_SEED));
            Module[] cells = {gru, rnn};
            for (Module cell : cells) {
                fx.fillBiases(cell);
                Tensor x = fx.leaf(host, TrainParity.uniform(49, 3 * 4, -1f, 1f), 3, 4);
                Tensor h = fx.leaf(host, TrainParity.uniform(50, 3 * 5, -1f, 1f), 3, 5);
                int before = ComputationGraph.current().size();
                Tensor next = fx.keep(cell == gru ? gru.forward(x, h) : rnn.forward(x, h));
                try (Tensor seed = Tensor.ones(host, next.shape())) {
                    next.backward(seed);
                }
                String what = cell.getClass().getSimpleName();
                assertNotNull(what + ": the input receives a gradient", x.grad());
                assertNotNull(what + ": hPrev receives a gradient", h.grad());
                for (Map.Entry<String, Parameter> e : cell.namedParameters().entrySet()) {
                    assertNotNull(what + "." + e.getKey() + " receives a gradient", e.getValue().grad());
                }
                assertEquals(what + ": the backward pass releases every node of the step", before,
                        ComputationGraph.current().size());
            }
        }

        /** Initializes {@code module}, back-propagates a fixed upstream gradient and checks what it reached. */
        private void checkReach(Module module, Initializer init, int... dims) {
            init.apply(new Built(module, host));
            fx.fillBiases(module);
            ComputationGraph graph = ComputationGraph.current();
            int before = graph.size();
            Map<String, float[]> grads = gradients(fx, module, host, dims);
            String what = module.getClass().getSimpleName();
            assertEquals(what + ": the backward pass releases every node it recorded", before, graph.size());
            for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
                float[] grad = grads.get(e.getKey());
                assertNotNull(what + "." + e.getKey() + " receives a gradient", grad);
                assertEquals(what + "." + e.getKey() + " gradient size", e.getValue().numElements(), grad.length);
                assertFinite(what + " dL/d" + e.getKey(), grad);
            }
            assertNotNull(what + ": the input receives a gradient", grads.get(INPUT));
            assertFinite(what + " dL/dinput", grads.get(INPUT));
        }
    }

    // ------------------------------------------------------------------ GPU against the host path

    @RunWith(Parameterized.class)
    public static class GpuParity {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private final RecurrentVisionParityTest.Fixture fx = new RecurrentVisionParityTest.Fixture();
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private ComputeBackend host;

        public GpuParity(TrainParity.Api api) {
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
            try {
                fx.close();
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void stackedBidirectionalLstmGradients() {
            compare("LSTM", () -> new LSTM(3, 5, 2, true, 0.0f, true), b -> ((LSTM) b.module).initialize(b.backend,
                    new Random(WEIGHT_SEED)), RECURRENT_GRAD_TOL, 2, 6, 3);
        }

        @Test
        public void stackedBidirectionalGruGradients() {
            compare("GRU", () -> new GRU(3, 5, 2, true, 0.0f, true), b -> ((GRU) b.module).initialize(b.backend,
                    new Random(WEIGHT_SEED)), RECURRENT_GRAD_TOL, 2, 6, 3);
        }

        @Test
        public void bidirectionalReluRnnGradients() {
            compare("RNN", () -> new RNN(3, 5, 2, true, 0.0, RNNCell.RELU, true), b -> ((RNN) b.module).initialize(
                    b.backend, new Random(WEIGHT_SEED)), RECURRENT_GRAD_TOL, 2, 6, 3);
        }

        @Test
        public void stackedStateSpaceModelGradients() {
            compare("StateSpaceModel", () -> new StateSpaceModel(3, 4, 6, 2, true, 0.0f), b -> {
                ((StateSpaceModel) b.module).initialize(b.backend, new Random(WEIGHT_SEED));
                fx.fillSsmState(b.module);
            }, RECURRENT_GRAD_TOL, 2, 5, 3);
        }

        @Test
        public void bottleneckWithSqueezeExcitationGradients() {
            compare("BottleneckBlock", () -> new BottleneckBlock(8, 16, 32, new int[]{2, 2}, true, 4),
                    b -> ((BottleneckBlock) b.module).initialize(b.backend, new Random(WEIGHT_SEED)), VISION_GRAD_TOL,
                    2, 8, 6, 6);
        }

        @Test
        public void unetWithBatchNormGradients() {
            compare("UNet", () -> new UNet(3, 2, new int[]{4, 8, 16}, true), b -> ((UNet) b.module).initialize(
                    b.backend, new Random(WEIGHT_SEED)), VISION_GRAD_TOL, 2, 3, 12, 12);
        }

        /** Builds the module on the host and the GPU from the same seeds and compares every gradient. */
        private void compare(String what, Supplier<Module> factory, Initializer init, float tol, int... dims) {
            Map<String, float[]> ref = gradientsOn(factory, init, host, dims);
            Map<String, float[]> dev = gradientsOn(factory, init, gpu, dims);
            assertEquals(what + ": gradient set", ref.keySet(), dev.keySet());
            for (Map.Entry<String, float[]> e : ref.entrySet()) {
                RecurrentVisionParityTest.assertComposite(api + " " + what + " dL/d" + e.getKey(), e.getValue(),
                        dev.get(e.getKey()), tol);
            }
        }

        private Map<String, float[]> gradientsOn(Supplier<Module> factory, Initializer init, ComputeBackend backend,
                                                 int... dims) {
            Module module = fx.module(factory.get());
            init.apply(new Built(module, backend));
            fx.fillBiases(module);
            return gradients(fx, module, backend, dims);
        }
    }

    // ------------------------------------------------------------------ shared steps

    /** A freshly built module and the backend it is initialized on. */
    static final class Built {
        final Module module;
        final ComputeBackend backend;

        Built(Module module, ComputeBackend backend) {
            this.module = module;
            this.backend = backend;
        }
    }

    /** Initializes a freshly built module on its backend. */
    interface Initializer {
        void apply(Built built);
    }

    /**
     * One forward pass of {@code module} over a leaf input of shape {@code dims} (seeded values in [-1, 1]) and one
     * backward pass from a seeded upstream gradient of the output's shape. Returns the gradient of every parameter
     * that received one, by name, and the input's under {@link #INPUT}; the gradients are left in their slots, which
     * the fixture closes.
     */
    static Map<String, float[]> gradients(RecurrentVisionParityTest.Fixture fx, Module module, ComputeBackend backend,
                                          int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        Tensor input = fx.leaf(backend, TrainParity.uniform(47, n, -1f, 1f), dims);
        Tensor out = fx.keep(module.forward(input));
        assertTrue("the output of a grad-mode forward requires grad", out.requiresGrad());
        assertNotNull("the output of a grad-mode forward is a recorded node output", out.gradFn());
        Shape shape = out.shape();
        float[] seedValues = TrainParity.uniform(48, (int) shape.numElements(), -1f, 1f);
        out.backward(fx.upload(backend, seedValues, dims(shape)));
        Map<String, float[]> grads = new LinkedHashMap<>();
        for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
            Tensor grad = e.getValue().grad();
            if (grad != null) {
                grads.put(e.getKey(), grad.toFloatArray());
            }
        }
        if (input.grad() != null) {
            grads.put(INPUT, input.grad().toFloatArray());
        }
        return grads;
    }

    private static int[] dims(Shape shape) {
        int[] dims = new int[shape.rank()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape.dim(i);
        }
        return dims;
    }

    static void assertFinite(String what, float[] values) {
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) {
                fail(what + ": element " + i + " is " + values[i]);
            }
        }
    }

    private static boolean anyNonZero(float[] values) {
        for (float v : values) {
            if (v != 0.0f) {
                return true;
            }
        }
        return false;
    }
}
