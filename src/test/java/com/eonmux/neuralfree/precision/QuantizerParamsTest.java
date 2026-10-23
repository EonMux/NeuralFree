package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationParams;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationType;
import com.eonmux.neuralfree.precision.Quantizer.QuantizedTensor;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link QuantizationParams} and {@link QuantizedTensor}: integer codes saturate instead of wrapping, invalid
 * parameters and codes are rejected at construction, and per-channel parameters code each slice of dimension 0 with
 * its own scale and zero point exactly as {@link FakeQuantGrad} fake-quantizes it.
 */
public class QuantizerParamsTest {
    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        TrainGpu.release(host);
    }

    /**
     * A quotient beyond the int range saturates to the bound of its sign, also when the zero point has that sign:
     * {@code 1e4 / (1e-3 / 255)} is about {@code 2.55e9}, so with zero point 127 the code is 127 (reconstructing 0),
     * and {@code -1e4 / 1e-6} with zero point -100 is -128.
     */
    @Test
    public void integerCodeSaturatesInsteadOfWrapping() {
        QuantizationParams positive = new QuantizationParams(1e-3f / 255, 127, -128, 127,
            QuantizationType.INT8_ASYMMETRIC);
        assertEquals(127, Quantizer.integerCode(1e4f, positive));
        assertEquals(-128, Quantizer.integerCode(-1e4f, positive));
        QuantizationParams negative = new QuantizationParams(1e-6f, -100, -128, 127, QuantizationType.INT8_ASYMMETRIC);
        assertEquals(-128, Quantizer.integerCode(-1e4f, negative));
        assertEquals(127, Quantizer.integerCode(1e4f, negative));
        assertEquals("NaN codes to the zero point", -100, Quantizer.integerCode(Float.NaN, negative));

        Quantizer quantizer = Quantizer.builder().quantType(QuantizationType.INT8_ASYMMETRIC).build();
        try (Tensor t = Tensor.fromArray(host, new float[]{1e4f, -1e4f})) {
            QuantizedTensor codes = quantizer.quantizeTensor(t, positive);
            assertArrayEquals(new int[]{127, -128}, codes.data());
            try (Tensor restored = quantizer.dequantizeTensor(codes)) {
                assertArrayEquals(new float[]{0f, -255 * positive.scale()}, restored.toFloatArray(), 0f);
            }
        }
    }

    @Test
    public void rejectsInvalidParameters() {
        QuantizationType int8 = QuantizationType.INT8_ASYMMETRIC;
        assertThrows(IllegalArgumentException.class, () -> new QuantizationParams(0f, 0, -128, 127, int8));
        assertThrows(IllegalArgumentException.class, () -> new QuantizationParams(-1f, 0, -128, 127, int8));
        assertThrows(IllegalArgumentException.class, () -> new QuantizationParams(Float.NaN, 0, -128, 127, int8));
        assertThrows(IllegalArgumentException.class, () -> new QuantizationParams(1f, 0, 10, -10, int8));
        assertThrows(IllegalArgumentException.class, () -> new QuantizationParams(1f, 200, -128, 127, int8));
        assertThrows("INT8 codes are signed", IllegalArgumentException.class,
            () -> new QuantizationParams(1f, 128, 0, 255, int8));
        assertThrows(IllegalArgumentException.class,
            () -> new QuantizationParams(1f, 0, -128, 127, QuantizationType.INT4_SYMMETRIC));
        assertThrows(IllegalArgumentException.class,
            () -> new QuantizationParams(1f, 1, 0, 255, QuantizationType.FP8_E4M3));
        assertThrows(IllegalArgumentException.class,
            () -> new QuantizationParams(1f, 0, -128, 127, QuantizationType.FP8_E5M2));
        assertThrows(NullPointerException.class, () -> new QuantizationParams(1f, 0, -128, 127, null));
        assertThrows(IllegalArgumentException.class,
            () -> QuantizationParams.perChannel(new float[]{1f, 2f}, new int[]{0}, -128, 127, int8));
        assertThrows(IllegalArgumentException.class,
            () -> QuantizationParams.perChannel(new float[0], new int[0], -128, 127, int8));
        assertThrows(IllegalArgumentException.class,
            () -> QuantizationParams.perChannel(new float[]{1f, 0f}, new int[]{0, 0}, -128, 127, int8));
    }

    @Test
    public void rejectsInvalidCodes() {
        QuantizationParams params = new QuantizationParams(0.5f, 0, -8, 7, QuantizationType.INT4_SYMMETRIC);
        assertThrows("fewer codes than elements", IllegalArgumentException.class,
            () -> new QuantizedTensor(new int[]{1, 2}, Shape.vector(3), params));
        assertThrows("more codes than elements", IllegalArgumentException.class,
            () -> new QuantizedTensor(new int[]{1, 2, 3, 4}, Shape.vector(3), params));
        assertThrows("a code outside the range", IllegalArgumentException.class,
            () -> new QuantizedTensor(new int[]{1, 8, 3}, Shape.vector(3), params));
        QuantizationParams fp8 = Quantizer.fp8Params(QuantizationType.FP8_E4M3, 1f);
        assertThrows("an FP8 code outside [0, 255]", IllegalArgumentException.class,
            () -> new QuantizedTensor(new int[]{256}, Shape.vector(1), fp8));
        QuantizationParams channels = QuantizationParams.perChannel(new float[]{1f, 1f, 1f}, new int[3], -8, 7,
            QuantizationType.INT4_SYMMETRIC);
        assertThrows("one channel per entry of dimension 0", IllegalArgumentException.class,
            () -> new QuantizedTensor(new int[6], Shape.of(2, 3), channels));
        assertThrows(NullPointerException.class, () -> new QuantizedTensor(null, Shape.vector(1), params));
    }

    @Test
    public void perChannelParametersHoldOneScaleAndZeroPointPerChannel() {
        float[] scales = {0.1f, 1f};
        int[] zeroPoints = {0, 5};
        QuantizationParams params = QuantizationParams.perChannel(scales, zeroPoints, -128, 127,
            QuantizationType.INT8_ASYMMETRIC);
        scales[0] = 7f;
        zeroPoints[1] = 9;
        assertTrue(params.isPerChannel());
        assertEquals(2, params.channels());
        assertEquals("the scales are copied", 0.1f, params.scale(0), 0f);
        assertEquals("the zero points are copied", 5, params.zeroPoint(1));
        assertThrows(IllegalStateException.class, params::scale);
        assertThrows(IllegalStateException.class, params::zeroPoint);
        assertTrue(params.toString().contains("scales=[0.1, 1.0]"));
        QuantizationParams perTensor = new QuantizationParams(0.5f, 3, -128, 127, QuantizationType.INT8_ASYMMETRIC);
        assertFalse(perTensor.isPerChannel());
        assertEquals(1, perTensor.channels());
        assertEquals(0.5f, perTensor.scale(0), 0f);
    }

    /** Integer per-channel codes reconstruct exactly what {@link FakeQuantGrad#applyPerChannel} computes. */
    @Test
    public void integerPerChannelMatchesFakeQuantization() {
        float[] values = {0.33f, -0.71f, 12.9f, 3.4f, -140f, 0.4f};
        float[] scales = {0.1f, 1f};
        int[] zeroPoints = {0, 5};
        QuantizationParams params = QuantizationParams.perChannel(scales, zeroPoints, -128, 127,
            QuantizationType.INT8_ASYMMETRIC);
        Quantizer quantizer = Quantizer.builder().quantType(QuantizationType.INT8_ASYMMETRIC).build();
        try (Tensor t = QatGradCheckTest.tensor(host, values, 2, 3)) {
            QuantizedTensor codes = quantizer.quantizeTensor(t, params);
            assertArrayEquals(new int[]{3, -7, 127, 8, -128, 5}, codes.data());
            try (Tensor restored = quantizer.dequantizeTensor(codes);
                 Tensor fake = FakeQuantGrad.applyPerChannel(t, scales, zeroPoints, -128, 127)) {
                assertArrayEquals(fake.toFloatArray(), restored.toFloatArray(), 0f);
            }
        }
    }

    /** FP8 per-channel codes reconstruct exactly what {@link FakeQuantGrad#applyFp8PerChannel} computes. */
    @Test
    public void fp8PerChannelMatchesFakeQuantization() {
        float[] values = {56f, -21.3f, 0.07f, 3.3f, -0.001f, 13.75f};
        QuantizationType type = QuantizationType.FP8_E4M3;
        float[] scales = {Quantizer.fp8Scale(type, 56f), Quantizer.fp8Scale(type, 13.75f)};
        QuantizationParams params = QuantizationParams.perChannel(scales, new int[2], 0, 255, type);
        Quantizer quantizer = Quantizer.builder().quantType(type).build();
        try (Tensor t = QatGradCheckTest.tensor(host, values, 2, 3)) {
            QuantizedTensor codes = quantizer.quantizeTensor(t, params);
            try (Tensor restored = quantizer.dequantizeTensor(codes);
                 Tensor fake = FakeQuantGrad.applyFp8PerChannel(t, scales, type)) {
                assertArrayEquals(fake.toFloatArray(), restored.toFloatArray(), 0f);
            }
        }
    }

    @Test
    public void perChannelParametersMustMatchDimensionZero() {
        QuantizationParams params = QuantizationParams.perChannel(new float[]{1f, 1f, 1f}, new int[3], -128, 127,
            QuantizationType.INT8_SYMMETRIC);
        Quantizer quantizer = Quantizer.builder().build();
        try (Tensor t = QatGradCheckTest.tensor(host, new float[6], 2, 3)) {
            assertThrows(IllegalArgumentException.class, () -> quantizer.quantizeTensor(t, params));
        }
    }
}
