package com.eonmux.neuralfree.nn.recurrent;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.vision.BottleneckBlock;
import com.eonmux.neuralfree.nn.vision.ResNet;
import com.eonmux.neuralfree.nn.vision.UNet;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

/**
 * The recurrent and vision modules as device compositions. {@link CpuPath} checks the recurrent modules on the host
 * reference backend against independent step-by-step definitions of their recurrences (transcendental class of
 * the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}) and the vision modules' shapes and
 * construction. {@link GpuParity} runs every module on each available GPU API against the same module, built from the
 * same seeds, on the host backend. Their outputs chain GEMM, convolution, normalization and transcendental steps, so
 * they are compared under the composite bound {@code |gpu - ref| <= tol * max(1, |ref|)}: 1e-4 for the recurrent
 * modules and 2e-3 for the vision networks, whose batch normalization rescales earlier differences.
 *
 * <p>The forward passes run in grad mode, so they compare the recorded path that training uses; each test's
 * {@link Fixture} discards what it recorded when it closes. Parameter values are written under
 * {@link ComputationGraph#noGrad()}, since parameter tensors require grad. Gradients are covered by
 * {@link RecurrentVisionGraphTest}.</p>
 */
@RunWith(Enclosed.class)
public class RecurrentVisionParityTest {

    private static final long WEIGHT_SEED = 4242L;
    private static final float RECURRENT_TOL = 1e-4f;
    private static final float VISION_TOL = 2e-3f;

    // ------------------------------------------------------------------ CPU path against the definitions

    public static class CpuPath {
        private final Fixture fx = new Fixture();
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
        public void lstmMatchesTheGateRecurrence() {
            LSTM lstm = fx.module(new LSTM(3, 4, 1));
            lstm.initialize(host, new Random(WEIGHT_SEED));
            fx.fillBiases(lstm);
            float[] x = TrainParity.uniform(11, 2 * 5 * 3, -1f, 1f);
            Tensor out = fx.keep(lstm.forward(fx.upload(host, x, 2, 5, 3)));
            assertEquals(Shape.of(2, 5, 4), out.shape());
            float[] ref = Ref.lstm(x, 2, 5, 3, 4, Ref.params(lstm, "forward_0"));
            TrainParity.assertTranscendental("CPU LSTM", ref, out.toFloatArray());
        }

        @Test
        public void gruAppliesTheResetGateToTheRecurrentCandidate() {
            GRU gru = fx.module(new GRU(3, 4, 1));
            gru.initialize(host, new Random(WEIGHT_SEED));
            fx.fillBiases(gru);
            float[] x = TrainParity.uniform(12, 2 * 5 * 3, -1f, 1f);
            Tensor out = fx.keep(gru.forward(fx.upload(host, x, 2, 5, 3)));
            assertEquals(Shape.of(2, 5, 4), out.shape());
            float[] ref = Ref.gru(x, 2, 5, 3, 4, Ref.params(gru, "forward_0"));
            TrainParity.assertTranscendental("CPU GRU", ref, out.toFloatArray());
        }

        @Test
        public void bidirectionalRnnConcatenatesTheReversedPass() {
            RNN rnn = fx.module(new RNN(3, 4, 1, true));
            rnn.initialize(host, new Random(WEIGHT_SEED));
            fx.fillBiases(rnn);
            float[] x = TrainParity.uniform(13, 2 * 5 * 3, -1f, 1f);
            Tensor out = fx.keep(rnn.forward(fx.upload(host, x, 2, 5, 3)));
            assertEquals(Shape.of(2, 5, 8), out.shape());
            float[] forward = Ref.rnn(x, 2, 5, 3, 4, Ref.params(rnn, "forward_0"), false);
            float[] backward = Ref.rnn(x, 2, 5, 3, 4, Ref.params(rnn, "backward_0"), true);
            float[] ref = new float[2 * 5 * 8];
            for (int row = 0; row < 2 * 5; row++) {
                System.arraycopy(forward, row * 4, ref, row * 8, 4);
                System.arraycopy(backward, row * 4, ref, row * 8 + 4, 4);
            }
            TrainParity.assertTranscendental("CPU bidirectional RNN", ref, out.toFloatArray());
        }

        @Test
        public void stateSpaceModelMatchesTheSelectiveScanWithAPaddedState() {
            checkSelectiveScan(3, 4, 5);
        }

        @Test
        public void stateSpaceModelMatchesTheSelectiveScanWithATruncatedState() {
            checkSelectiveScan(3, 4, 2);
        }

        private void checkSelectiveScan(int inputSize, int hiddenSize, int stateDim) {
            StateSpaceModel ssm = fx.module(new StateSpaceModel(inputSize, hiddenSize, stateDim));
            ssm.initialize(host, new Random(WEIGHT_SEED));
            fx.fillBiases(ssm);
            fx.fillSsmState(ssm);
            float[] x = TrainParity.uniform(14, 2 * 4 * inputSize, -1f, 1f);
            Tensor out = fx.keep(ssm.forward(fx.upload(host, x, 2, 4, inputSize)));
            assertEquals(Shape.of(2, 4, hiddenSize), out.shape());
            float[] ref = Ref.ssm(x, 2, 4, inputSize, hiddenSize, stateDim, ssm.namedParameters());
            TrainParity.assertTranscendental("CPU StateSpaceModel", ref, out.toFloatArray());
        }

        @Test
        public void recurrentModulesRejectInputsThatAreNotBatchSequenceFeatures() {
            LSTM lstm = fx.module(new LSTM(3, 4, 1));
            lstm.initialize(host, new Random(WEIGHT_SEED));
            Tensor matrix = fx.upload(host, new float[15], 5, 3);
            Tensor wrongFeatures = fx.upload(host, new float[40], 2, 5, 4);
            assertThrows(IllegalArgumentException.class, () -> lstm.forward(matrix));
            assertThrows(IllegalArgumentException.class, () -> lstm.forward(wrongFeatures));
        }

        @Test
        public void recurrentModulesRejectASecondInitialize() {
            LSTM lstm = fx.module(new LSTM(3, 4, 1));
            GRU gru = fx.module(new GRU(3, 4, 1));
            RNN rnn = fx.module(new RNN(3, 4, 1));
            StateSpaceModel ssm = fx.module(new StateSpaceModel(3, 4, 2));
            lstm.initialize(host, new Random(WEIGHT_SEED));
            gru.initialize(host, new Random(WEIGHT_SEED));
            rnn.initialize(host, new Random(WEIGHT_SEED));
            ssm.initialize(host, new Random(WEIGHT_SEED));
            assertThrows(IllegalStateException.class, () -> lstm.initialize(host, new Random(WEIGHT_SEED)));
            assertThrows(IllegalStateException.class, () -> gru.initialize(host, new Random(WEIGHT_SEED)));
            assertThrows(IllegalStateException.class, () -> rnn.initialize(host, new Random(WEIGHT_SEED)));
            assertThrows(IllegalStateException.class, () -> ssm.initialize(host, new Random(WEIGHT_SEED)));
        }

        @Test
        public void rnnRejectsAnUnknownActivationAtConstruction() {
            assertThrows(IllegalArgumentException.class, () -> new RNN(3, 4, 1, false, 0.0, "sigmoid", true));
        }

        @Test
        public void unetCropsTheSkipConnectionToTheUpsampledSize() {
            UNet unet = fx.module(new UNet(1, 2, new int[]{2, 4}, false));
            unet.initialize(host, new Random(WEIGHT_SEED));
            assertNotNull(unet.getParameter("encoder_blocks.0.0.weight"));
            Tensor out = fx.keep(unet.forward(fx.upload(host, TrainParity.uniform(15, 81, -1f, 1f), 1, 1, 9, 9)));
            assertEquals(Shape.of(1, 2, 8, 8), out.shape());
            assertThrows(IllegalStateException.class, () -> unet.initialize(host, new Random(WEIGHT_SEED)));
        }

        @Test
        public void resnetClassifiesThePooledFeatures() {
            ResNet resnet = fx.module(ResNet.resnet18(10));
            resnet.initialize(host, new Random(WEIGHT_SEED));
            Tensor out = fx.keep(resnet.forward(fx.upload(host, TrainParity.uniform(16, 2 * 3 * 32 * 32, -1f, 1f),
                    2, 3, 32, 32)));
            assertEquals(Shape.of(2, 10), out.shape());
            assertThrows(IllegalStateException.class, () -> resnet.initialize(host, new Random(WEIGHT_SEED)));
        }

        @Test
        public void bottleneckWithSqueezeExcitationKeepsTheProjectedShape() {
            BottleneckBlock block = fx.module(new BottleneckBlock(8, 16, 32, new int[]{2, 2}, true, 4));
            block.initialize(host, new Random(WEIGHT_SEED));
            Tensor out = fx.keep(block.forward(fx.upload(host, TrainParity.uniform(17, 2 * 8 * 6 * 6, -1f, 1f),
                    2, 8, 6, 6)));
            assertEquals(Shape.of(2, 32, 3, 3), out.shape());
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
        private final Fixture fx = new Fixture();
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
        public void stackedBidirectionalLstm() {
            compare("LSTM", () -> new LSTM(3, 5, 2, true, 0.0f, true), (m, b) -> ((LSTM) m).initialize(b,
                    new Random(WEIGHT_SEED)), RECURRENT_TOL, 2, 6, 3);
        }

        @Test
        public void stackedBidirectionalGru() {
            compare("GRU", () -> new GRU(3, 5, 2, true, 0.0f, true), (m, b) -> ((GRU) m).initialize(b,
                    new Random(WEIGHT_SEED)), RECURRENT_TOL, 2, 6, 3);
        }

        @Test
        public void bidirectionalReluRnn() {
            compare("RNN", () -> new RNN(3, 5, 2, true, 0.0, "relu", true), (m, b) -> ((RNN) m).initialize(b,
                    new Random(WEIGHT_SEED)), RECURRENT_TOL, 2, 6, 3);
        }

        @Test
        public void stackedStateSpaceModel() {
            compare("StateSpaceModel", () -> new StateSpaceModel(3, 4, 6, 2, true, 0.0f), (m, b) -> {
                ((StateSpaceModel) m).initialize(b, new Random(WEIGHT_SEED));
                fx.fillSsmState(m);
            }, RECURRENT_TOL, 2, 5, 3);
        }

        @Test
        public void cellsWithExplicitStates() {
            float[] x = TrainParity.uniform(21, 3 * 4, -1f, 1f);
            float[] h = TrainParity.uniform(22, 3 * 5, -1f, 1f);
            float[] c = TrainParity.uniform(23, 3 * 5, -1f, 1f);
            float[][] outs = new float[2][];
            float[][] cells = new float[2][];
            float[][] grus = new float[2][];
            ComputeBackend[] backends = {host, gpu};
            for (int k = 0; k < 2; k++) {
                ComputeBackend b = backends[k];
                LSTMCell lstm = fx.module(new LSTMCell(4, 5));
                lstm.initialize(b, new Random(WEIGHT_SEED));
                fx.fillBiases(lstm);
                LSTMCell.LSTMCellOutput out = lstm.forwardWithCell(fx.upload(b, x, 3, 4), fx.upload(b, h, 3, 5),
                        fx.upload(b, c, 3, 5));
                outs[k] = fx.keep(out.hidden()).toFloatArray();
                cells[k] = fx.keep(out.cell()).toFloatArray();
                GRUCell gru = fx.module(new GRUCell(4, 5));
                gru.initialize(b, new Random(WEIGHT_SEED));
                fx.fillBiases(gru);
                grus[k] = fx.keep(gru.forward(fx.upload(b, x, 3, 4), fx.upload(b, h, 3, 5))).toFloatArray();
            }
            assertComposite(api + " LSTMCell hidden", outs[0], outs[1], RECURRENT_TOL);
            assertComposite(api + " LSTMCell cell", cells[0], cells[1], RECURRENT_TOL);
            assertComposite(api + " GRUCell", grus[0], grus[1], RECURRENT_TOL);
        }

        @Test
        public void unetWithBatchNorm() {
            compare("UNet", () -> new UNet(3, 2, new int[]{4, 8, 16}, true), (m, b) -> ((UNet) m).initialize(b,
                    new Random(WEIGHT_SEED)), VISION_TOL, 2, 3, 12, 12);
        }

        @Test
        public void bottleneckWithSqueezeExcitationAndProjection() {
            compare("BottleneckBlock", () -> new BottleneckBlock(8, 16, 32, new int[]{2, 2}, true, 4),
                    (m, b) -> ((BottleneckBlock) m).initialize(b, new Random(WEIGHT_SEED)), VISION_TOL, 2, 8, 6, 6);
        }

        @Test
        public void resnet18Classifier() {
            compare("ResNet-18", () -> ResNet.resnet18(10), (m, b) -> ((ResNet) m).initialize(b,
                    new Random(WEIGHT_SEED)), VISION_TOL, 2, 3, 32, 32);
        }

        /** Builds the module on the host and the GPU from the same seeds and compares one forward pass. */
        private void compare(String what, Supplier<Module> factory, Initializer init, float tol, int... dims) {
            int n = 1;
            for (int d : dims) {
                n *= d;
            }
            float[] x = TrainParity.uniform(31, n, -1f, 1f);
            float[][] results = new float[2][];
            Shape[] shapes = new Shape[2];
            ComputeBackend[] backends = {host, gpu};
            for (int k = 0; k < 2; k++) {
                Module module = fx.module(factory.get());
                init.apply(module, backends[k]);
                fx.fillBiases(module);
                Tensor out = fx.keep(module.forward(fx.upload(backends[k], x, dims)));
                shapes[k] = out.shape();
                results[k] = out.toFloatArray();
            }
            assertEquals(what + " shape", shapes[0], shapes[1]);
            assertComposite(api + " " + what, results[0], results[1], tol);
        }
    }

    /** Initializes a freshly built module on a backend. */
    interface Initializer {
        void apply(Module module, ComputeBackend backend);
    }

    /** {@code |gpu - ref| <= tol * max(1, |ref|)} element-wise; NaN matches NaN and infinities match exactly. */
    static void assertComposite(String what, float[] ref, float[] gpu, float tol) {
        assertEquals(what + ": length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            boolean ok;
            if (Float.isNaN(ref[i]) || Float.isNaN(gpu[i])) {
                ok = Float.isNaN(ref[i]) && Float.isNaN(gpu[i]);
            } else if (Float.isInfinite(ref[i]) || Float.isInfinite(gpu[i])) {
                ok = ref[i] == gpu[i];
            } else {
                ok = Math.abs((double) gpu[i] - ref[i]) <= tol * Math.max(1.0, Math.abs(ref[i]));
            }
            if (!ok) {
                fail(what + ": element " + i + ": ref " + ref[i] + ", gpu " + gpu[i]);
            }
        }
    }

    // ------------------------------------------------------------------ step-by-step reference definitions

    /** Float definitions of the recurrences, one batch row at a time, in the modules' arithmetic order. */
    static final class Ref {

        private Ref() {
        }

        /** {@code {W_ih, b_ih, W_hh, b_hh}} of the cell registered under {@code prefix}. */
        static float[][] params(Module module, String prefix) {
            Map<String, Parameter> p = module.namedParameters();
            return new float[][]{
                p.get(prefix + ".ih.weight").tensor().toFloatArray(),
                p.get(prefix + ".ih.bias").tensor().toFloatArray(),
                p.get(prefix + ".hh.weight").tensor().toFloatArray(),
                p.get(prefix + ".hh.bias").tensor().toFloatArray()
            };
        }

        /** {@code x W^T + b} for one row. */
        static float[] linear(float[] x, int xOffset, int in, float[] w, float[] b, int out) {
            float[] y = new float[out];
            for (int o = 0; o < out; o++) {
                float sum = 0.0f;
                for (int i = 0; i < in; i++) {
                    sum += x[xOffset + i] * w[o * in + i];
                }
                y[o] = sum + b[o];
            }
            return y;
        }

        static float sigmoid(float x) {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
        }

        static float tanh(float x) {
            return (float) Math.tanh(x);
        }

        static float[] lstm(float[] x, int batch, int seqLen, int in, int hidden, float[][] p) {
            float[] out = new float[batch * seqLen * hidden];
            for (int b = 0; b < batch; b++) {
                float[] h = new float[hidden];
                float[] c = new float[hidden];
                for (int t = 0; t < seqLen; t++) {
                    float[] gi = linear(x, (b * seqLen + t) * in, in, p[0], p[1], 4 * hidden);
                    float[] gh = linear(h, 0, hidden, p[2], p[3], 4 * hidden);
                    float[] nextH = new float[hidden];
                    for (int j = 0; j < hidden; j++) {
                        float i = sigmoid(gi[j] + gh[j]);
                        float f = sigmoid(gi[hidden + j] + gh[hidden + j]);
                        float g = tanh(gi[2 * hidden + j] + gh[2 * hidden + j]);
                        float o = sigmoid(gi[3 * hidden + j] + gh[3 * hidden + j]);
                        c[j] = f * c[j] + i * g;
                        nextH[j] = o * tanh(c[j]);
                    }
                    h = nextH;
                    System.arraycopy(h, 0, out, (b * seqLen + t) * hidden, hidden);
                }
            }
            return out;
        }

        static float[] gru(float[] x, int batch, int seqLen, int in, int hidden, float[][] p) {
            float[] out = new float[batch * seqLen * hidden];
            for (int b = 0; b < batch; b++) {
                float[] h = new float[hidden];
                for (int t = 0; t < seqLen; t++) {
                    float[] gi = linear(x, (b * seqLen + t) * in, in, p[0], p[1], 3 * hidden);
                    float[] gh = linear(h, 0, hidden, p[2], p[3], 3 * hidden);
                    float[] nextH = new float[hidden];
                    for (int j = 0; j < hidden; j++) {
                        float r = sigmoid(gi[j] + gh[j]);
                        float z = sigmoid(gi[hidden + j] + gh[hidden + j]);
                        float n = tanh(gi[2 * hidden + j] + r * gh[2 * hidden + j]);
                        nextH[j] = z * h[j] + (1 - z) * n;
                    }
                    h = nextH;
                    System.arraycopy(h, 0, out, (b * seqLen + t) * hidden, hidden);
                }
            }
            return out;
        }

        static float[] rnn(float[] x, int batch, int seqLen, int in, int hidden, float[][] p, boolean reverse) {
            float[] out = new float[batch * seqLen * hidden];
            for (int b = 0; b < batch; b++) {
                float[] h = new float[hidden];
                for (int k = 0; k < seqLen; k++) {
                    int t = reverse ? seqLen - 1 - k : k;
                    float[] gi = linear(x, (b * seqLen + t) * in, in, p[0], p[1], hidden);
                    float[] gh = linear(h, 0, hidden, p[2], p[3], hidden);
                    float[] nextH = new float[hidden];
                    for (int j = 0; j < hidden; j++) {
                        nextH[j] = tanh(gi[j] + gh[j]);
                    }
                    h = nextH;
                    System.arraycopy(h, 0, out, (b * seqLen + t) * hidden, hidden);
                }
            }
            return out;
        }

        static float[] ssm(float[] x, int batch, int seqLen, int in, int hidden, int stateDim,
                           Map<String, Parameter> params) {
            float[] y = ssmReadout(x, batch, seqLen, in, hidden, stateDim, params);
            float[] d = params.get("layer_0.d").tensor().toFloatArray();
            float[] out = new float[batch * seqLen * hidden];
            for (int row = 0; row < batch * seqLen; row++) {
                for (int j = 0; j < hidden; j++) {
                    out[row * hidden + j] = y[row] * d[j];
                }
            }
            return out;
        }

        /** The per-step readout {@code y_t = sum_s(c_t * s_t)} of the first layer, one value per (batch, step) row. */
        static float[] ssmReadout(float[] x, int batch, int seqLen, int in, int hidden, int stateDim,
                                  Map<String, Parameter> params) {
            Function<String, float[]> p = name -> params.get("layer_0." + name).tensor().toFloatArray();
            float[] wIn = p.apply("input_proj.weight");
            float[] bIn = p.apply("input_proj.bias");
            float[] wDelta = p.apply("delta_proj.weight");
            float[] bDelta = p.apply("delta_proj.bias");
            float[] wB = p.apply("b_proj.weight");
            float[] bB = p.apply("b_proj.bias");
            float[] wC = p.apply("c_proj.weight");
            float[] bC = p.apply("c_proj.bias");
            float[] aLog = p.apply("a_log");
            float[] readout = new float[batch * seqLen];
            for (int b = 0; b < batch; b++) {
                float[] state = new float[stateDim];
                for (int t = 0; t < seqLen; t++) {
                    int row = (b * seqLen + t) * in;
                    float[] u = linear(x, row, in, wIn, bIn, hidden);
                    for (int j = 0; j < hidden; j++) {
                        u[j] = tanh(u[j]);
                    }
                    float[] delta = linear(u, 0, hidden, wDelta, bDelta, stateDim);
                    float[] bv = linear(u, 0, hidden, wB, bB, stateDim);
                    float[] cv = linear(u, 0, hidden, wC, bC, stateDim);
                    float y = 0.0f;
                    for (int s = 0; s < stateDim; s++) {
                        delta[s] = sigmoid(delta[s]);
                        float a = (float) Math.exp(aLog[s]);
                        float aBar = (float) Math.exp(delta[s] * a);
                        float factor = Math.abs(a) < 1e-8f ? delta[s] : (aBar - 1.0f) / a;
                        float xs = s < in ? x[row + s] : 0.0f;
                        state[s] = aBar * state[s] + factor * bv[s] * xs;
                        y += cv[s] * state[s];
                    }
                    readout[b * seqLen + t] = y;
                }
            }
            return readout;
        }
    }

    // ------------------------------------------------------------------ resources

    /**
     * The resources of one test, closed in reverse order of acquisition. It opens a
     * {@link ComputationGraph.StepScope} on creation and closes it first, so every node a test records (and its
     * output) is discarded before the modules and tensors are closed.
     */
    static final class Fixture implements AutoCloseable {
        private final ComputationGraph.StepScope step = ComputationGraph.current().step();
        private final List<AutoCloseable> owned = new ArrayList<>();

        <T extends AutoCloseable> T keep(T resource) {
            owned.add(resource);
            return resource;
        }

        <M extends Module> M module(M m) {
            return keep(m);
        }

        Tensor upload(ComputeBackend backend, float[] data, int... dims) {
            return keep(Tensor.fromStorage(backend, Shape.of(dims), DType.FLOAT32,
                    TensorStorage.fromHostFloat(backend, data)));
        }

        /** A leaf that requires grad holding {@code data}; its gradient is closed with it. */
        Tensor leaf(ComputeBackend backend, float[] data, int... dims) {
            Tensor leaf = upload(backend, data, dims).withRequiresGrad(true);
            owned.add(() -> {
                Tensor grad = leaf.grad();
                if (grad != null) {
                    grad.close();
                }
                leaf.close();
            });
            return leaf;
        }

        /** Gives every bias parameter non-zero values that depend only on its name, so both backends match. */
        void fillBiases(Module module) {
            for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
                if (e.getKey().endsWith("bias")) {
                    Tensor t = e.getValue().tensor();
                    write(t, TrainParity.uniform(e.getKey().hashCode(), (int) t.numElements(), -0.5f, 0.5f));
                }
            }
        }

        /**
         * Sets every {@code a_log} so that one state takes the near-zero discretization branch ({@code |a| < 1e-8})
         * and the others decay, and gives every {@code d} distinct values.
         */
        void fillSsmState(Module module) {
            for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
                Tensor t = e.getValue().tensor();
                int n = (int) t.numElements();
                if (e.getKey().endsWith("a_log")) {
                    float[] values = TrainParity.uniform(e.getKey().hashCode(), n, -1.5f, -0.2f);
                    values[0] = -20.0f;
                    write(t, values);
                } else if (e.getKey().endsWith(".d")) {
                    write(t, TrainParity.uniform(e.getKey().hashCode(), n, 0.5f, 1.5f));
                }
            }
        }

        /**
         * Overwrites a parameter tensor's elements (a dense FLOAT32 tensor at storage offset 0), an assignment under
         * {@link ComputationGraph#noGrad()}.
         */
        void write(Tensor target, float[] values) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                GpuEvent done = target.storage().copyFromHostFloat(values, target.backend().transferQueue());
                if (done != null) {
                    done.close();
                }
            }
        }

        @Override
        public void close() {
            RuntimeException failure = null;
            try {
                step.close();
            } catch (RuntimeException e) {
                failure = new IllegalStateException("discarding the recorded graph failed", e);
            }
            for (int i = owned.size() - 1; i >= 0; i--) {
                try {
                    owned.get(i).close();
                } catch (Exception e) {
                    if (failure == null) {
                        failure = new IllegalStateException("closing a test resource failed", e);
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            owned.clear();
            if (failure != null) {
                throw failure;
            }
        }
    }
}
