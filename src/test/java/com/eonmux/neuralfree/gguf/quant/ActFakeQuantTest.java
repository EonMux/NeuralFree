package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.precision.FakeQuantGrad;
import com.eonmux.neuralfree.precision.Quantizer;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ActFakeQuant} on hand vectors: every scheme's rounding, saturation and ties, the scale floors, the
 * {@code scale_ub} clamp, the reciprocal-versus-quotient and NVFP4 operation orders, and the shared zero rules.
 * Expected values are compared bit for bit.
 */
public class ActFakeQuantTest {

    private static float[] run(ActQuantScheme s, float scale, int zp, float g, float... x) {
        float[] y = x.clone();
        ActFakeQuant.apply(s, scale, zp, g, y, 0, 1, y.length);
        return y;
    }

    private static float[] group(int n, float... head) {
        return Arrays.copyOf(head, n);
    }

    private static void assertBits(String what, float[] expected, float[] actual) {
        assertEquals(what + ": length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + " [" + i + "] expected " + expected[i] + " got " + actual[i],
                    Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(actual[i]));
        }
    }

    @Test
    public void int8TokenRoundsTiesToEvenAtUnitScale() {
        float[] y = run(ActQuantScheme.INT8_TOKEN, 0f, 0, 0f, 127f, -63.5f, 0.5f, 1.5f, -2.5f, 0f);
        assertBits("int8 token", new float[]{127f, -64f, 0f, 2f, -2f, 0f}, y);
    }

    @Test
    public void int8TokenZeroRowGivesPositiveZero() {
        float[] y = run(ActQuantScheme.INT8_TOKEN, 0f, 0, 0f, -0f, 0f, -0f);
        assertBits("int8 token zero row", new float[]{0f, 0f, 0f}, y);
    }

    @Test
    public void int8TokenReadsTheNaNQuotientOfAVanishingScaleAsZero() {
        float tiny = 1e-40f;
        float d = tiny / 127f;
        assertTrue("the scale stays positive", d > 0f);
        assertTrue("its reciprocal overflows", Float.isInfinite(127f / tiny));
        float[] y = run(ActQuantScheme.INT8_TOKEN, 0f, 0, 0f, tiny, 0f, -tiny);
        assertBits("int8 token, vanishing scale", new float[]{127f * d, 0f, -128f * d}, y);
    }

    @Test
    public void int8TokenRowOfZeroScaleGivesPositiveZero() {
        float[] y = run(ActQuantScheme.INT8_TOKEN, 0f, 0, 0f, Float.MIN_VALUE, -Float.MIN_VALUE);
        assertEquals("the scale underflows to zero", 0f, Float.MIN_VALUE / 127f, 0f);
        assertBits("int8 token, zero scale", new float[]{0f, 0f}, y);
    }

    @Test
    public void int8StaticSaturatesAndRoundsToEven() {
        float[] y = run(ActQuantScheme.INT8_STATIC, 0.5f, 0, 0f, 100f, -100f, 0.25f, 0.75f, -0.25f);
        assertBits("int8 static", new float[]{63.5f, -64f, 0f, 1f, 0f}, y);
    }

    @Test
    public void int8StaticAsymSaturatesTheShiftedLevel() {
        float[] y = run(ActQuantScheme.INT8_STATIC_ASYM, 0.5f, 10, 0f, 1f, -70f, 70f, 0f);
        assertBits("int8 static asym", new float[]{1f, -69f, 58.5f, 0f}, y);
    }

    @Test
    public void int8GroupTruncatesInsteadOfRounding() {
        float[] x = group(256, 127f, 2.7f, -2.7f, -0.5f, 126.5f);
        float[] y = x.clone();
        ActFakeQuant.apply(ActQuantScheme.INT8_GROUP128, 0f, 0, 0f, y, 0, 1, 256);
        float[] expected = group(256, 127f, 2f, -2f, 0f, 126f);
        assertBits("int8 group", expected, y);
        assertNotEquals("rounding would give 3", 3f, y[1], 0f);
    }

    @Test
    public void int8GroupFloorsTheScaleOfAZeroGroup() {
        float[] y = new float[128];
        y[0] = 1e-12f;
        ActFakeQuant.apply(ActQuantScheme.INT8_GROUP128, 0f, 0, 0f, y, 0, 1, 128);
        float d = ActFakeQuant.GROUP_EPS / 127f;
        float[] expected = new float[128];
        expected[0] = d;
        assertBits("int8 group below eps", expected, y);
    }

    @Test
    public void fp8TokenRoundsTiesToEvenAndSubnormals() {
        float[] y = run(ActQuantScheme.FP8_TOKEN, Float.POSITIVE_INFINITY, 0, 0f,
                448f, -224f, 1.0625f, 1.1875f, 0x1p-10f, 3 * 0x1p-10f);
        assertBits("fp8 token", new float[]{448f, -224f, 1f, 1.25f, 0f, 0x1p-8f}, y);
    }

    @Test
    public void fp8TokenFloorsTheScale() {
        float min = ActFakeQuant.FP8_MIN_SCALE;
        assertEquals(0x36924925, Float.floatToRawIntBits(min));
        float[] y = run(ActQuantScheme.FP8_TOKEN, Float.POSITIVE_INFINITY, 0, 0f, 1e-3f, -5e-4f);
        assertBits("fp8 token below 1/512", new float[]{224f * min, -112f * min}, y);
    }

    @Test
    public void fp8TokenScaleUpperBoundClampsAnOutlierRow() {
        float d = 1f / 448f;
        float[] y = run(ActQuantScheme.FP8_TOKEN, 1f, 0, 0f, 1000f, 0.5f, -0.25f);
        assertBits("fp8 token with ub 1", new float[]{448f * d, 224f * d, -112f * d}, y);
        float[] unbounded = run(ActQuantScheme.FP8_TOKEN, Float.POSITIVE_INFINITY, 0, 0f, 1000f, 0.5f, -0.25f);
        assertEquals("without a bound the outlier sets the scale", 448f * (1000f / 448f), unbounded[0], 0f);
    }

    @Test
    public void fp8TokenMatchesTheTrainingFakeQuantizerAtTheFlooredScale() {
        float[][] rows = {{1e-3f, -5e-4f, 2e-4f, 0f}, {3.3f, -1.7f, 0.02f, 0.4f}};
        ComputeBackend host = TrainParity.hostBackend();
        try {
            for (float[] row : rows) {
                float amax = 0f;
                for (float v : row) {
                    amax = Math.max(amax, Math.abs(v));
                }
                float scale = Math.max(amax / 448f, ActFakeQuant.FP8_MIN_SCALE);
                float[] expected;
                try (Tensor in = Tensor.fromArray(host, row);
                     Tensor q = FakeQuantGrad.applyFp8(in, scale, Quantizer.QuantizationType.FP8_E4M3)) {
                    expected = q.toFloatArray();
                }
                assertBits("fp8 token against FakeQuantGrad", expected,
                        run(ActQuantScheme.FP8_TOKEN, Float.POSITIVE_INFINITY, 0, 0f, row));
            }
        } finally {
            TrainGpu.release(host);
        }
    }

    @Test
    public void fp8StaticMultipliesByTheRoundedReciprocal() {
        float s = 3f;
        float x = Float.intBitsToFloat(0x4063ffff);
        assertEquals("x / s rounds down to the code 1.125", 1.125f, ActFakeQuant.e4m3(x / s), 0f);
        float[] y = run(ActQuantScheme.FP8_STATIC, s, 0, 0f, x, 448f * 3f * 2f, -1f);
        assertBits("fp8 static", new float[]{3.75f, 448f * 3f, -ActFakeQuant.e4m3(1f / 3f) * 3f}, y);
    }

    @Test
    public void fp8GroupFloorsTheScaleOfAVanishingOrZeroGroup() {
        float[] y = new float[256];
        y[0] = 1e-12f;
        ActFakeQuant.apply(ActQuantScheme.FP8_GROUP128, 0f, 0, 0f, y, 0, 1, 256);
        float d = ActFakeQuant.GROUP_EPS / 448f;
        assertEquals(0x2EDBE6FF, Float.floatToRawIntBits(ActFakeQuant.GROUP_EPS));
        float[] expected = new float[256];
        expected[0] = 4.5f * d;
        assertBits("fp8 group below eps, then a zero group", expected, y);
    }

    @Test
    public void nvfp4DequantizesLeftToRight() {
        float g = 0.7f;
        float[] x = group(16, 7.3f, 3.65f);
        float[] y = x.clone();
        ActFakeQuant.apply(ActQuantScheme.NVFP4, 0f, 0, g, y, 0, 1, 16);
        float[] expected = group(16, Float.intBitsToFloat(0x40eb3333), Float.intBitsToFloat(0x406b3333));
        assertBits("nvfp4", expected, y);
        assertNotEquals("the other association rounds differently", 0x40eb3333,
                Float.floatToRawIntBits(6f * (1.75f * g)));
    }

    @Test
    public void nvfp4ZeroBlockGivesPositiveZero() {
        float[] y = new float[16];
        y[3] = -0f;
        ActFakeQuant.apply(ActQuantScheme.NVFP4, 0f, 0, 2f, y, 0, 1, 16);
        assertBits("nvfp4 zero block", new float[16], y);
    }

    @Test
    public void mxfp4RoundsTheBlockMaximumBeforeTakingItsExponent() {
        assertEquals(1, ActFakeQuant.mxfp4Exponent(7f));
        assertEquals(0, ActFakeQuant.mxfp4Exponent(6.5f));
        assertEquals(2, ActFakeQuant.mxfp4Exponent(14f));
        assertEquals(-2, ActFakeQuant.mxfp4Exponent(0.875f));
        assertEquals(0, ActFakeQuant.mxfp4Exponent(5.9f));
        assertEquals(127, ActFakeQuant.mxfp4Exponent(Float.MAX_VALUE));
        assertEquals(-127, ActFakeQuant.mxfp4Exponent(0f));
        float[] x = group(32, 7f, 1f, -0.3f, -2.9f);
        float[] y = x.clone();
        ActFakeQuant.apply(ActQuantScheme.MXFP4, 0f, 0, 0f, y, 0, 1, 32);
        assertBits("mxfp4", group(32, 8f, 1f, 0f, -3f), y);
    }

    @Test
    public void mxfp8TakesTheCeilingExponentOfTheScaledMaximum() {
        assertEquals(0, ActFakeQuant.mxfp8Exponent(448f));
        assertEquals(1, ActFakeQuant.mxfp8Exponent(449f));
        assertEquals(-127, ActFakeQuant.mxfp8Exponent(0f));
        float[] x = group(32, 449f, 1.0625f * 2f, -3f);
        float[] y = x.clone();
        ActFakeQuant.apply(ActQuantScheme.MXFP8, 0f, 0, 0f, y, 0, 1, 32);
        assertBits("mxfp8", group(32, 448f, 2f, -3f), y);
    }

    @Test
    public void appliesEveryRowAndRunIndependently() {
        float[] x = new float[2 * 64];
        x[0] = 7f;
        x[32] = 1.0625f;
        x[64] = 100f;
        x[64 + 33] = 1f;
        float[] y = x.clone();
        ActFakeQuant.apply(ActQuantScheme.MXFP4, 0f, 0, 0f, y, 0, 2, 64);
        float[] expected = new float[2 * 64];
        expected[0] = 8f;
        expected[32] = 1f;
        expected[64] = 96f;
        expected[64 + 33] = 1f;
        assertBits("two rows of two blocks", expected, y);
    }

    @Test
    public void rejectsBadArguments() {
        float[] x = new float[128];
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.INT8_GROUP128, 0f, 0, 0f, x, 0, 1, 96));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.INT8_STATIC, 0f, 0, 0f, x, 0, 1, 128));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.FP8_STATIC, Float.POSITIVE_INFINITY, 0, 0f, x, 0, 1, 128));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.FP8_TOKEN, 0f, 0, 0f, x, 0, 1, 128));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.NVFP4, 0f, 0, Float.NaN, x, 0, 1, 128));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.INT8_STATIC_ASYM, 1f, (1 << 24) + 1, 0f, x, 0, 1, 128));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.INT8_STATIC_ASYM, 1f, Integer.MIN_VALUE, 0f, x, 0, 1, 128));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.INT8_TOKEN, 0f, 0, 0f, x, 1, 1, 128));
        assertThrows(IllegalArgumentException.class,
                () -> ActFakeQuant.apply(ActQuantScheme.INT8_TOKEN, 0f, 0, 0f, x, 0, -1, 128));
    }

    @Test
    public void leavesElementsOutsideTheRangeUntouched() {
        float[] x = {5f, 127f, 0.4f, 9f};
        ActFakeQuant.apply(ActQuantScheme.INT8_TOKEN, 0f, 0, 0f, x, 1, 1, 2);
        assertArrayEquals(new float[]{5f, 127f, 0f, 9f}, x, 0f);
    }
}
