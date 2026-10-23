package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.layers.Linear;
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
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * FP8 quantization-aware training: {@link FakeQuantGrad#applyFp8} and {@link FakeQuantGrad#applyFp8PerChannel}
 * fake-quantize to the values the post-training FP8 codes reconstruct, {@code decode(fp8Code(x, scale)) * scale}
 * ({@link Quantizer#fp8Code}, {@link Fp8Codec#decode}), with the clipped straight-through gradient, and a
 * {@link QATModule} of an FP8 type trains with the values its {@link QATModule#toQuantizer()} exports.
 *
 * <p>The expectations are the codec's values, computed on the host. The CPU path must match them bit for bit. On a
 * device the division {@code x / scale} is in the 4-ulp class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract} and every later step is exact, so a device value
 * must be the codec's value of the host quotient or of a quotient within 4 ulp of it. The inputs cover the rounding
 * ties (to even), the subnormal range, saturation at {@code +-fmax}, signed zeros, NaN and the infinities; every
 * non-zero result is a normal FLOAT32.</p>
 */
@RunWith(Enclosed.class)
public class QatFp8Test {

    /** E4M3FN inputs, as multiples of the scale: ties, subnormals, saturation and the special values. */
    static final float[] E4M3_QUOTIENTS = {
        0f, -0f, 1f, -1f, 1.0625f, 1.1875f, -1.1875f, 448f, 449f, 464f, 1e6f, -1e6f,
        0x1p-9f, 0x1p-10f, 3 * 0x1p-10f, 0x1p-6f, 0.0137f, -3.3f, 1e-10f, -1e-10f,
        Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
    /** E5M2 inputs, as multiples of the scale. */
    static final float[] E5M2_QUOTIENTS = {
        0f, -0f, 1.125f, 1.375f, -1.375f, 57344f, 61440f, 1e9f, -1e9f,
        0x1p-16f, 0x1p-17f, 3 * 0x1p-17f, 0x1p-14f, 0.0137f, -3.3f, 1e-10f, -1e-10f,
        Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
    /** A weight whose largest magnitude 56 gives the E4M3 scale {@code 56 / 448 = 1/8}. */
    static final float[] WEIGHT = {56f, -21.3f, 0.07f, 3.3f, -0.001f, 13.75f};

    /** The checks, on the backend of the subclass. */
    public abstract static class Checks {

        protected abstract ComputeBackend backend();

        /** Whether results must equal the host expectation bit for bit (the CPU path). */
        protected abstract boolean exact();

        @Test
        public void e4m3MatchesTheCodec() {
            checkPerTensor(Quantizer.QuantizationType.FP8_E4M3, E4M3_QUOTIENTS, 1.0f);
            checkPerTensor(Quantizer.QuantizationType.FP8_E4M3, E4M3_QUOTIENTS, 0.25f);
            checkPerTensor(Quantizer.QuantizationType.FP8_E4M3, E4M3_QUOTIENTS, 0.75f);
        }

        @Test
        public void e5m2MatchesTheCodec() {
            checkPerTensor(Quantizer.QuantizationType.FP8_E5M2, E5M2_QUOTIENTS, 1.0f);
            checkPerTensor(Quantizer.QuantizationType.FP8_E5M2, E5M2_QUOTIENTS, 4.0f);
            checkPerTensor(Quantizer.QuantizationType.FP8_E5M2, E5M2_QUOTIENTS, 0.75f);
        }

        /** Two channels of dimension 0 with the scales 0.5 and 3: each row matches the codec with its own scale. */
        @Test
        public void perChannelUsesTheScaleOfEachRow() {
            Quantizer.QuantizationType type = Quantizer.QuantizationType.FP8_E4M3;
            float[] scales = {0.5f, 3.0f};
            int columns = 5;
            float[] x = new float[2 * columns];
            float[] elementScales = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                elementScales[i] = scales[i / columns];
                x[i] = E4M3_QUOTIENTS[i + 2] * elementScales[i];
            }
            try (Tensor input = QatGradCheckTest.tensor(backend(), x, 2, columns);
                 Tensor y = FakeQuantGrad.applyFp8PerChannel(input, scales, type)) {
                assertCodecValues("per-channel " + type, type, x, elementScales, y.toFloatArray());
            }
        }

        /**
         * The clipped straight-through estimator: the gradient of {@code sum(fq(x) * r)} (with the NaN values of the
         * infinite and NaN inputs replaced by 0) is {@code r} where {@code |x / scale| <= 448} and 0 where the value
         * saturates or is not finite.
         */
        @Test
        public void gradientIsZeroWhereTheValueSaturates() {
            ComputeBackend b = backend();
            float scale = 0.5f;
            float[] quotients = {1f, -3.3f, 400f, 500f, -600f, 0.01f, Float.POSITIVE_INFINITY, Float.NaN};
            float[] x = new float[quotients.length];
            for (int i = 0; i < x.length; i++) {
                x[i] = quotients[i] * scale;
            }
            float[] r = TrainParity.uniform(701, x.length, -1.0f, 1.0f);
            double[] expected = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                expected[i] = Math.abs(quotients[i]) <= 448f ? r[i] : 0.0;
            }
            int before = ComputationGraph.current().size();
            try (Tensor weights = Tensor.fromArray(b, r);
                 Tensor source = Tensor.fromArray(b, x);
                 ComputationGraph.StepScope step = ComputationGraph.current().step();
                 Tensor input = source.withRequiresGrad(true)) {
                try (Tensor q = FakeQuantGrad.applyFp8(input, scale, Quantizer.QuantizationType.FP8_E4M3)) {
                    assertTrue("the fake-quantized tensor is recorded", q.requiresGrad());
                    try (Tensor zeros = Tensor.zeros(b, q.shape());
                         Tensor notNaN = TensorOps.eq(q, q);
                         Tensor finite = TensorOps.where(notNaN, q, zeros);
                         Tensor weighted = TensorOps.mul(finite, weights);
                         Tensor loss = TensorOps.sum(weighted)) {
                        loss.backward();
                    }
                }
                Tensor grad = input.grad();
                assertNotNull("the input receives a gradient", grad);
                QatGradCheckTest.assertNear("FP8 straight-through gradient", expected, grad.toFloatArray(), 0.0);
                input.setGrad(null);
                grad.close();
            }
            assertEquals("the step leaves the graph as it found it", before, ComputationGraph.current().size());
        }

        /**
         * A {@link QATModule} of each FP8 type observes the weight's largest magnitude 56, so its scale is
         * {@code 56 / fmax} with zero point 0; its fake-quantized weight is the dequantized export of
         * {@link QATModule#toQuantizer()} for the same weight.
         */
        @Test
        public void trainingFakeQuantizesToTheExportedValues() {
            for (Quantizer.QuantizationType type : new Quantizer.QuantizationType[]{
                Quantizer.QuantizationType.FP8_E4M3, Quantizer.QuantizationType.FP8_E5M2}) {
                checkTrainingMatchesExport(type);
            }
        }

        private void checkTrainingMatchesExport(Quantizer.QuantizationType type) {
            ComputeBackend b = backend();
            Linear base = QatGradCheckTest.layer(b, WEIGHT);
            QATModule qat = QATModule.builder().baseModule(base).quantType(type).quantizeActivations(false).build();
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor weight = QatGradCheckTest.tensor(b, WEIGHT, 2, 3);
                 Tensor trained = qat.fakeQuantizeWeight(weight, "weight")) {
                float scale = Quantizer.fp8Scale(type, 56f);
                assertEquals(type + " observer scale", scale, qat.weightScaleObservers().get("weight"), 0f);
                assertEquals(type + " observer zero point", 0, (int) qat.weightZeroPointObservers().get("weight"));

                Quantizer quantizer = qat.toQuantizer();
                Map<String, Quantizer.QuantizedTensor> codes = quantizer.quantizeModel(base);
                assertEquals(type + " exported scale", scale, quantizer.getQuantParams("weight").scale(), 0f);
                float[] exported;
                try (Tensor dequantized = quantizer.dequantizeTensor(codes.get("weight"))) {
                    exported = dequantized.toFloatArray();
                }
                float[] actual = trained.toFloatArray();
                float[] scales = new float[WEIGHT.length];
                Arrays.fill(scales, scale);
                assertCodecValues(type + " training", type, WEIGHT, scales, actual);
                if (exact()) {
                    TrainParity.assertBitIdentical(type + " training and export", exported, actual);
                }
            } finally {
                qat.close();
            }
        }

        private void checkPerTensor(Quantizer.QuantizationType type, float[] quotients, float scale) {
            float[] x = new float[quotients.length];
            float[] scales = new float[quotients.length];
            for (int i = 0; i < x.length; i++) {
                x[i] = quotients[i] * scale;
                scales[i] = scale;
            }
            try (Tensor input = Tensor.fromArray(backend(), x);
                 Tensor y = FakeQuantGrad.applyFp8(input, scale, type)) {
                assertCodecValues(type + " with scale " + scale, type, x, scales, y.toFloatArray());
            }
        }

        /**
         * Asserts that {@code actual[i]} is the codec value of {@code x[i] / scales[i]}: bit for bit on the CPU path,
         * else the value of that quotient or of one within 4 ulp of it.
         */
        private void assertCodecValues(String what, Quantizer.QuantizationType type, float[] x, float[] scales,
                                       float[] actual) {
            assertEquals(what + ": length", x.length, actual.length);
            Fp8Codec codec = Quantizer.codec(type);
            for (int i = 0; i < x.length; i++) {
                float expected = codecValue(codec, x[i], scales[i]);
                if (sameValue(expected, actual[i])) {
                    continue;
                }
                boolean neighbour = false;
                if (!exact()) {
                    float quotient = x[i] / scales[i];
                    float below = quotient;
                    float above = quotient;
                    for (int ulp = 1; ulp <= 4 && !neighbour; ulp++) {
                        below = Math.nextDown(below);
                        above = Math.nextUp(above);
                        neighbour = sameValue(quotientValue(codec, below, x[i], scales[i]), actual[i])
                            || sameValue(quotientValue(codec, above, x[i], scales[i]), actual[i]);
                    }
                }
                if (!neighbour) {
                    fail(what + ": element " + i + " (x = " + x[i] + "): expected " + expected + ", got "
                        + actual[i]);
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

        @Override
        protected boolean exact() {
            return true;
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

        @Override
        protected boolean exact() {
            return false;
        }
    }

    /** The value post-training FP8 codes reconstruct for {@code x}: {@code decode(fp8Code(x, scale)) * scale}. */
    static float codecValue(Fp8Codec codec, float x, float scale) {
        return codec.decode(Quantizer.fp8Code(codec, x, scale)) * scale;
    }

    /**
     * {@link #codecValue} with the quotient {@code quotient} in place of {@code x / scale}, the saturation of a
     * finite {@code x} and the encoding of an infinite one kept.
     */
    static float quotientValue(Fp8Codec codec, float quotient, float x, float scale) {
        float q = quotient;
        if (Float.isInfinite(q) && !Float.isInfinite(x)) {
            q = Math.copySign(Float.MAX_VALUE, q);
        }
        return codec.decode(codec.encode(q)) * scale;
    }

    /** Equal bit patterns, or both NaN. */
    static boolean sameValue(float expected, float actual) {
        if (Float.isNaN(expected)) {
            return Float.isNaN(actual);
        }
        return Float.floatToIntBits(expected) == Float.floatToIntBits(actual);
    }
}
