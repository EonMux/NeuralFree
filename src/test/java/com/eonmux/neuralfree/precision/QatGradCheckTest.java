package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Parameter;
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

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The gradients of {@link QATModule}, the module class of the precision package with parameters, on the CPU path and
 * on each GPU API.
 *
 * <p>Fake quantization is piecewise constant, so its finite-difference derivative is 0 almost everywhere; training
 * uses the clipped straight-through estimator instead ({@code FakeQuantGrad}): the gradient passes
 * unchanged where the element's code {@code round(x / scale) + zeroPoint} is inside {@code [qMin, qMax]} and is 0
 * where it is clamped. These checks therefore compare the recorded gradients with that expectation, computed on the
 * host in double, and use finite differences ({@link GradientCheckUtil}) only where the module is smooth: with fake
 * quantization disabled, where it is its base module.</p>
 *
 * <p>Fixture (INT4 symmetric, codes {@code [-8, 7]}). The observers see one calibration batch and are then frozen: its
 * inputs reach 0.875, so the input scale is {@code 0.875 / 7 = 1/8} exactly; the base layer's weights and biases are
 * multiples of 1/4, so its outputs on multiples of 1/8 are exact, and the calibration outputs reach 1.125, so the
 * output scale is {@code 1.125 / 7}. The checked batch holds multiples of 1/8 up to 1.5: its input codes are exact
 * integers, six of them beyond {@code [-8, 7]}, and its output codes are at least 0.25 from the clamp boundaries
 * {@code 7.5} and {@code -8.5}, two of them beyond. Forward values are asserted before gradients, so a drift of the
 * fixture shows as a forward failure, and the fake-quantized result must require grad, so an estimator that is not
 * recorded fails before its gradients are read.</p>
 */
@RunWith(Enclosed.class)
public class QatGradCheckTest {

    static final float EPSILON = 1e-3f;
    static final float TOLERANCE = 1e-2f;
    static final double GRAD_TOLERANCE = 1e-5;
    static final int Q_MIN = -8;
    static final int Q_MAX = 7;

    static final float[] WEIGHT = {0.5f, -0.25f, 0.75f, -0.5f, 1.0f, 0.25f};
    static final float[] BIAS = {0.25f, -0.25f};
    static final float[] CALIBRATION = {0.875f, -0.5f, 0.25f, -0.375f, 0.875f, 0.625f};
    static final float[] BATCH = {
        1.5f, -0.25f, 0.5f,
        -1.25f, 1.375f, 0.125f,
        0.75f, -1.5f, 1.25f,
        -0.625f, 0.375f, -1.125f};
    /** Weight-observer calibration: largest magnitude 0.875 overall, 0.4375 in the second row. */
    static final float[] CALIBRATION_WEIGHT = {0.875f, -0.5f, 0.25f, 0.4375f, -0.125f, 0.0625f};
    /** The weight checked after calibration: 1.5 is beyond the per-tensor clamp, 0.5 beyond the second row's. */
    static final float[] CHECKED_WEIGHT = {1.5f, -0.375f, 0.125f, -0.5f, 0.5f, 0.125f};
    static final float[] WEIGHT_CHECK_INPUT = {0.5f, -0.25f, 1.0f, -0.75f, 0.125f, 0.25f};
    static final int ROWS = 4;
    static final int IN = 3;
    static final int OUT = 2;

    /** The checks, on the backend of the subclass. */
    public abstract static class Checks {

        protected abstract ComputeBackend backend();

        /**
         * Activation fake quantization: {@code y = fq_out(fq_in(x) W^T + b)}; the expected gradients of
         * {@code sum(y * r)} are {@code g = r * m_out}, {@code dW = g^T fq_in(x)}, {@code db = sum_rows g} and
         * {@code dx = (g W) * m_in}.
         */
        @Test
        public void activationStraightThroughEstimator() {
            ComputeBackend b = backend();
            Linear base = fixtureLayer(b);
            QATModule qat = QATModule.builder().baseModule(base).quantType(Quantizer.QuantizationType.INT4_SYMMETRIC)
                .quantizeWeights(false).quantizeActivations(true).build();
            float[] r = TrainParity.uniform(601, ROWS * OUT, -1.0f, 1.0f);
            try (Tensor calibration = tensor(b, CALIBRATION, 2, IN);
                 Tensor batch = tensor(b, BATCH, ROWS, IN);
                 Tensor weights = tensor(b, r, ROWS, OUT)) {
                calibrate(qat, calibration);
                Expected expected = new Expected(1.125f / Q_MAX, r);
                int before = ComputationGraph.current().size();
                try (ComputationGraph.StepScope step = ComputationGraph.current().step();
                     Tensor x = batch.withRequiresGrad(true)) {
                    try (Tensor out = qat.forward(x)) {
                        assertNear("fake-quantized output", expected.output, out.toFloatArray(), 1e-6);
                        assertTrue("the fake-quantized output is recorded", out.requiresGrad());
                        try (Tensor weighted = TensorOps.mul(out, weights);
                             Tensor loss = TensorOps.sum(weighted)) {
                            loss.backward();
                        }
                    }
                    assertNear("weight gradient", expected.weightGrad, gradient(base.getParameter("weight")),
                        GRAD_TOLERANCE);
                    assertNear("bias gradient", expected.biasGrad, gradient(base.getParameter("bias")),
                        GRAD_TOLERANCE);
                    Tensor inputGrad = x.grad();
                    assertNotNull("the input receives a gradient", inputGrad);
                    assertNear("input gradient", expected.inputGrad, inputGrad.toFloatArray(), GRAD_TOLERANCE);
                    x.setGrad(null);
                    inputGrad.close();
                } finally {
                    qat.zeroGrad();
                }
                assertEquals("the step leaves the graph as it found it", before, ComputationGraph.current().size());
            } finally {
                qat.close();
            }
        }

        /** {@link QATModule#fakeQuantizeWeight}: the gradient is {@code r} where the weight's code is in range. */
        @Test
        public void weightStraightThroughEstimator() {
            ComputeBackend b = backend();
            QATModule qat = QATModule.builder().baseModule(fixtureLayer(b))
                .quantType(Quantizer.QuantizationType.INT4_SYMMETRIC).quantizeWeights(true).quantizeActivations(false)
                .build();
            float[] values = new float[ROWS * IN / 2];
            System.arraycopy(BATCH, 0, values, 0, values.length);
            float[] r = TrainParity.uniform(611, values.length, -1.0f, 1.0f);
            try (Tensor calibration = tensor(b, CALIBRATION, 2, IN);
                 Tensor candidate = tensor(b, values, 2, IN);
                 Tensor weights = tensor(b, r, 2, IN)) {
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                     Tensor observed = qat.fakeQuantizeWeight(calibration, "weight")) {
                    assertNear("calibration weights are on the grid", toDouble(CALIBRATION), observed.toFloatArray(),
                        0.0);
                }
                qat.freezeObservers();
                double[] quantized = new double[values.length];
                double[] expectedGrad = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    int code = Math.round(values[i] / 0.125f);
                    boolean inRange = code >= Q_MIN && code <= Q_MAX;
                    quantized[i] = Math.max(Q_MIN, Math.min(Q_MAX, code)) * 0.125;
                    expectedGrad[i] = inRange ? r[i] : 0.0;
                }
                int before = ComputationGraph.current().size();
                try (ComputationGraph.StepScope step = ComputationGraph.current().step();
                     Tensor w = candidate.withRequiresGrad(true)) {
                    try (Tensor q = qat.fakeQuantizeWeight(w, "weight")) {
                        assertNear("fake-quantized weight", quantized, q.toFloatArray(), 0.0);
                        assertTrue("the fake-quantized weight is recorded", q.requiresGrad());
                        try (Tensor weighted = TensorOps.mul(q, weights);
                             Tensor loss = TensorOps.sum(weighted)) {
                            loss.backward();
                        }
                    }
                    Tensor grad = w.grad();
                    assertNotNull("the weight receives a gradient", grad);
                    assertNear("weight gradient", expectedGrad, grad.toFloatArray(), 0.0);
                    w.setGrad(null);
                    grad.close();
                }
                assertEquals("the step leaves the graph as it found it", before, ComputationGraph.current().size());
            } finally {
                qat.close();
            }
        }

        /**
         * Weight fake quantization inside {@link QATModule#forward}, per tensor: the observers calibrated on
         * {@link #CALIBRATION_WEIGHT} (largest magnitude 0.875) give the scale {@code 1/8}; the checked weight
         * {@link #CHECKED_WEIGHT} then has one code beyond the clamp (1.5 gives 12, clamped to 7).
         */
        @Test
        public void weightFakeQuantizationInTheForwardPassPerTensor() {
            checkWeightFakeQuantization(false, new float[]{0.125f},
                new double[]{0.875, -0.375, 0.125, -0.5, 0.5, 0.125},
                new boolean[]{false, true, true, true, true, true});
        }

        /**
         * As {@link #weightFakeQuantizationInTheForwardPassPerTensor} with one observer per output channel: the second
         * row's calibration magnitude 0.4375 gives it the scale {@code 1/16}, so its 0.5 (code 8) is clamped to 0.4375.
         */
        @Test
        public void weightFakeQuantizationInTheForwardPassPerChannel() {
            checkWeightFakeQuantization(true, new float[]{0.125f, 0.0625f},
                new double[]{0.875, -0.375, 0.125, -0.5, 0.4375, 0.125},
                new boolean[]{false, true, true, true, false, true});
        }

        /**
         * Calibrates the weight observers on {@link #CALIBRATION_WEIGHT}, loads {@link #CHECKED_WEIGHT} and checks
         * {@code y = x fq(W)^T + b} and the gradients of {@code sum(y * r)}: {@code dW = (r^T x)} where the weight's
         * code is in range and 0 where it is clamped, {@code db = sum_rows r}; the parameter keeps its own values.
         */
        private void checkWeightFakeQuantization(boolean perChannel, float[] scales, double[] quantized,
                                                 boolean[] inRange) {
            ComputeBackend b = backend();
            Linear base = layer(b, CALIBRATION_WEIGHT);
            QATModule qat = QATModule.builder().baseModule(base).quantType(Quantizer.QuantizationType.INT4_SYMMETRIC)
                .quantizeWeights(true).quantizeActivations(false).perChannelQuantization(perChannel).build();
            float[] r = TrainParity.uniform(631, 2 * OUT, -1.0f, 1.0f);
            try (Tensor input = tensor(b, WEIGHT_CHECK_INPUT, 2, IN);
                 Tensor weights = tensor(b, r, 2, OUT)) {
                calibrate(qat, input);
                if (perChannel) {
                    float[] channelScales = qat.weightChannelScales().get("weight");
                    assertNotNull("per-channel scales of the weight", channelScales);
                    assertNear("per-channel scales", toDouble(scales), channelScales, 0.0);
                } else {
                    assertEquals("per-tensor scale", scales[0], qat.weightScaleObservers().get("weight"), 0.0f);
                }
                Map<String, Object> state = new LinkedHashMap<>();
                state.put("weight", CHECKED_WEIGHT.clone());
                state.put("bias", BIAS.clone());
                base.loadStateDict(state);
                double[] expectedOutput = new double[2 * OUT];
                double[] expectedWeightGrad = new double[OUT * IN];
                double[] expectedBiasGrad = new double[OUT];
                for (int row = 0; row < 2; row++) {
                    for (int o = 0; o < OUT; o++) {
                        double y = BIAS[o];
                        double g = r[row * OUT + o];
                        expectedBiasGrad[o] += g;
                        for (int i = 0; i < IN; i++) {
                            y += WEIGHT_CHECK_INPUT[row * IN + i] * quantized[o * IN + i];
                            if (inRange[o * IN + i]) {
                                expectedWeightGrad[o * IN + i] += g * WEIGHT_CHECK_INPUT[row * IN + i];
                            }
                        }
                        expectedOutput[row * OUT + o] = y;
                    }
                }
                int before = ComputationGraph.current().size();
                try (ComputationGraph.StepScope step = ComputationGraph.current().step()) {
                    try (Tensor out = qat.forward(input)) {
                        assertNear("output with fake-quantized weights", expectedOutput, out.toFloatArray(), 1e-6);
                        try (Tensor weighted = TensorOps.mul(out, weights);
                             Tensor loss = TensorOps.sum(weighted)) {
                            loss.backward();
                        }
                    }
                    assertNear("weight gradient", expectedWeightGrad, gradient(base.getParameter("weight")),
                        GRAD_TOLERANCE);
                    assertNear("bias gradient", expectedBiasGrad, gradient(base.getParameter("bias")),
                        GRAD_TOLERANCE);
                    assertNear("the weight keeps its own values", toDouble(CHECKED_WEIGHT),
                        base.getParameter("weight").tensor().toFloatArray(), 0.0);
                } finally {
                    qat.zeroGrad();
                }
                assertEquals("the step leaves the graph as it found it", before, ComputationGraph.current().size());
            } finally {
                qat.close();
            }
        }

        /** With fake quantization disabled the module is its base layer, whose gradients match finite differences. */
        @Test
        public void disabledFakeQuantizationMatchesFiniteDifferences() {
            ComputeBackend b = backend();
            Linear base = fixtureLayer(b);
            QATModule qat = QATModule.builder().baseModule(base).quantType(Quantizer.QuantizationType.INT4_SYMMETRIC)
                .quantizeWeights(true).quantizeActivations(true).build();
            try (Tensor batch = tensor(b, BATCH, ROWS, IN);
                 Tensor weights = tensor(b, TrainParity.uniform(621, ROWS * OUT, -1.0f, 1.0f), ROWS, OUT)) {
                qat.enableFakeQuant(false);
                int before = ComputationGraph.current().size();
                GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(() -> {
                    try (Tensor out = qat.forward(batch)) {
                        return TensorOps.mul(out, weights);
                    }
                }, qat, qat.parameters(), EPSILON, TOLERANCE, IN * OUT);
                assertTrue("QATModule without fake quantization: " + result, result.passed());
                assertEquals("the check leaves the graph as it found it", before, ComputationGraph.current().size());
            } finally {
                qat.close();
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
     * The host expectation of {@link Checks#activationStraightThroughEstimator} for the checked batch, in double: the
     * output codes use the output scale and Java's {@code Math.round} as the module does.
     */
    static final class Expected {
        final double[] output = new double[ROWS * OUT];
        final double[] weightGrad = new double[OUT * IN];
        final double[] biasGrad = new double[OUT];
        final double[] inputGrad = new double[ROWS * IN];

        Expected(float outputScale, float[] r) {
            double[] quantizedInput = new double[ROWS * IN];
            boolean[] inputInRange = new boolean[ROWS * IN];
            for (int i = 0; i < quantizedInput.length; i++) {
                int code = Math.round(BATCH[i] / 0.125f);
                inputInRange[i] = code >= Q_MIN && code <= Q_MAX;
                quantizedInput[i] = Math.max(Q_MIN, Math.min(Q_MAX, code)) * 0.125;
            }
            double[] outputGrad = new double[ROWS * OUT];
            for (int row = 0; row < ROWS; row++) {
                for (int o = 0; o < OUT; o++) {
                    double y = BIAS[o];
                    for (int i = 0; i < IN; i++) {
                        y += quantizedInput[row * IN + i] * WEIGHT[o * IN + i];
                    }
                    int code = Math.round((float) y / outputScale);
                    int clamped = Math.max(Q_MIN, Math.min(Q_MAX, code));
                    output[row * OUT + o] = clamped * (double) outputScale;
                    outputGrad[row * OUT + o] = code == clamped ? r[row * OUT + o] : 0.0;
                }
            }
            for (int row = 0; row < ROWS; row++) {
                for (int o = 0; o < OUT; o++) {
                    double g = outputGrad[row * OUT + o];
                    biasGrad[o] += g;
                    for (int i = 0; i < IN; i++) {
                        weightGrad[o * IN + i] += g * quantizedInput[row * IN + i];
                        inputGrad[row * IN + i] += g * WEIGHT[o * IN + i];
                    }
                }
            }
            for (int i = 0; i < inputGrad.length; i++) {
                if (!inputInRange[i]) {
                    inputGrad[i] = 0.0;
                }
            }
        }
    }

    /** Runs the calibration batch through {@code qat} under {@code noGrad} and freezes its observers. */
    static void calibrate(QATModule qat, Tensor calibration) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor out = qat.forward(calibration)) {
            for (float v : out.toFloatArray()) {
                assertTrue("finite calibration output", Float.isFinite(v));
            }
        }
        qat.freezeObservers();
    }

    /** The fixture {@link Linear}: {@link #WEIGHT} and {@link #BIAS}. */
    static Linear fixtureLayer(ComputeBackend b) {
        return layer(b, WEIGHT);
    }

    /** A {@link Linear} {@code IN -> OUT} holding {@code weight} and {@link #BIAS}. */
    static Linear layer(ComputeBackend b, float[] weight) {
        Linear layer = new Linear(IN, OUT);
        layer.initialize(b, new Random(600));
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("weight", weight.clone());
        state.put("bias", BIAS.clone());
        layer.loadStateDict(state);
        return layer;
    }

    static float[] gradient(Parameter p) {
        assertNotNull("the base layer's parameter exists", p);
        assertNotNull(p.name() + " receives a gradient", p.grad());
        return p.grad().toFloatArray();
    }

    static void assertNear(String what, double[] expected, float[] actual, double tolerance) {
        assertEquals(what + ": length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (!(Math.abs(actual[i] - expected[i]) <= tolerance * Math.max(1.0, Math.abs(expected[i])))) {
                fail(what + ": element " + i + ": expected " + expected[i] + ", got " + actual[i]);
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

    static Tensor tensor(ComputeBackend b, float[] values, int... dims) {
        try (Tensor flat = Tensor.fromArray(b, values)) {
            return flat.reshape(Shape.of(dims));
        }
    }
}
