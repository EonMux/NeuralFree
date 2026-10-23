package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link Quantizer} post-training quantization: every parameter of a multi-layer model is calibrated, parameterized
 * and quantized under its own dotted path; the FP8 types store OCP FP8 bit patterns.
 */
public class QuantizerTest {

    /** One parameter {@code param} of shape {@code [n]}, registered under {@code name} with the parameter name {@code paramName}. */
    static final class Weight extends Module {
        Weight(ComputeBackend backend, String name, String paramName, float[] values) {
            Tensor tensor = Tensor.fromArray(backend, values);
            registerParameter(name, paramName == null ? new Parameter(tensor) : new Parameter(tensor, paramName));
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
    }

    /**
     * Two layers whose parameters share the local name {@code "weight"}: ranges {@code [-1, 0.25]} and {@code [-3, 4]}
     * give the INT8 symmetric scales {@code 1 / 127} and {@code 4 / 127} and codes of their own.
     */
    @Test
    public void keysEveryParameterByItsPath() {
        Module model = new Sequential(
            new Weight(host, "weight", "weight", new float[]{-1f, 0.25f}),
            new Weight(host, "weight", "weight", new float[]{4f, -3f}));
        Quantizer quantizer = Quantizer.builder().quantType(Quantizer.QuantizationType.INT8_SYMMETRIC).build();

        Map<String, Quantizer.QuantizedTensor> codes = quantizer.quantizeModel(model);

        assertEquals(Arrays.asList("0.weight", "1.weight"), new ArrayList<>(codes.keySet()));
        assertEquals(1f / 127, quantizer.getQuantParams("0.weight").scale(), 0f);
        assertEquals(4f / 127, quantizer.getQuantParams("1.weight").scale(), 0f);
        assertNull("no parameter has the bare local name as its path", quantizer.getQuantParams("weight"));
        assertArrayEquals(new int[]{-127, 32}, codes.get("0.weight").data());
        assertArrayEquals(new int[]{127, -95}, codes.get("1.weight").data());
    }

    /** Parameters built without a name are still told apart by their paths. */
    @Test
    public void keysUnnamedParametersByTheirPath() {
        Module model = new Sequential(
            new Weight(host, "w", null, new float[]{2f}),
            new Weight(host, "w", null, new float[]{-8f}));
        Quantizer quantizer = Quantizer.builder().quantType(Quantizer.QuantizationType.INT8_SYMMETRIC).build();

        Map<String, Quantizer.QuantizedTensor> codes = quantizer.quantizeModel(model);

        assertEquals(Arrays.asList("0.w", "1.w"), new ArrayList<>(codes.keySet()));
        assertArrayEquals(new int[]{127}, codes.get("0.w").data());
        assertArrayEquals(new int[]{-127}, codes.get("1.w").data());
    }

    /**
     * The FP8 types store bit patterns of {@code x / scale} with {@code scale = amax / fmax}: weights {@code {-1, 0.25}}
     * give amax 1, so -1 saturates to {@code -fmax} and 0.25 is {@code 0.25 * fmax} (112 in E4M3, 14336 in E5M2), both
     * exactly representable.
     */
    @Test
    public void fp8StoresFloatBitPatterns() {
        Object[][] cases = {
            {Quantizer.QuantizationType.FP8_E4M3, 448f, new int[]{0xFE, 0x6E}, 448f, 112f},
            {Quantizer.QuantizationType.FP8_E5M2, 57344f, new int[]{0xFB, 0x73}, 57344f, 14336f}};
        for (Object[] c : cases) {
            Quantizer.QuantizationType type = (Quantizer.QuantizationType) c[0];
            float fmax = (Float) c[1];
            Module model = new Weight(host, "weight", "weight", new float[]{-1f, 0.25f});
            Quantizer quantizer = Quantizer.builder().quantType(type).build();

            Quantizer.QuantizedTensor codes = quantizer.quantizeModel(model).get("weight");
            assertNotNull(type.name(), codes);
            assertEquals(type, codes.quantType());
            Quantizer.QuantizationParams params = quantizer.getQuantParams("weight");
            assertEquals(type.name() + " zero point", 0, params.zeroPoint());
            assertEquals(type.name() + " code min", 0, params.quantMin());
            assertEquals(type.name() + " code max", 255, params.quantMax());
            assertEquals(type.name() + " scale", 1f / fmax, params.scale(), 0f);
            assertArrayEquals(type.name(), (int[]) c[2], codes.data());

            try (Tensor restored = quantizer.dequantizeTensor(codes)) {
                float[] values = restored.toFloatArray();
                assertEquals(type.name() + " negative weight", -(Float) c[3] * params.scale(), values[0], 0f);
                assertEquals(type.name() + " positive weight", (Float) c[4] * params.scale(), values[1], 0f);
            }
            model.close();
        }
    }

    /** An all-zero FP8 parameter has amax 0 and scale 1. */
    @Test
    public void fp8ZeroRangeHasUnitScale() {
        Module model = new Weight(host, "weight", "weight", new float[]{0f, 0f});
        Quantizer quantizer = Quantizer.builder().quantType(Quantizer.QuantizationType.FP8_E4M3).build();
        Quantizer.QuantizedTensor codes = quantizer.quantizeModel(model).get("weight");
        assertEquals(1f, codes.params().scale(), 0f);
        assertArrayEquals(new int[]{0, 0}, codes.data());
        model.close();
    }

    /** NaN weights keep NaN through FP8 codes; a finite weight whose quotient overflows saturates. */
    @Test
    public void fp8NaNAndOverflow() {
        Quantizer quantizer = Quantizer.builder().quantType(Quantizer.QuantizationType.FP8_E4M3).build();
        Quantizer.QuantizationParams params = Quantizer.fp8Params(Quantizer.QuantizationType.FP8_E4M3, 1e-30f);
        try (Tensor t = Tensor.fromArray(host, new float[]{Float.NaN, 1e10f, -1e10f})) {
            Quantizer.QuantizedTensor codes = quantizer.quantizeTensor(t, params);
            int[] data = codes.data();
            assertTrue("NaN code", Fp8Codec.E4M3FN.isNaN(data[0]));
            assertEquals(0x7E, data[1]);
            assertEquals(0xFE, data[2]);
            try (Tensor restored = quantizer.dequantizeTensor(codes)) {
                float[] values = restored.toFloatArray();
                assertTrue(Float.isNaN(values[0]));
                assertEquals(448f * params.scale(), values[1], 0f);
                assertEquals(-448f * params.scale(), values[2], 0f);
            }
        }
    }

    /** MIN_MAX FP8 calibration of a parameter with an infinite value has no finite scale and is rejected. */
    @Test(expected = IllegalArgumentException.class)
    public void fp8RejectsInfiniteRange() {
        Module model = new Weight(host, "weight", "weight", new float[]{1f, Float.POSITIVE_INFINITY});
        try {
            Quantizer.builder().quantType(Quantizer.QuantizationType.FP8_E5M2).build().quantizeModel(model);
        } finally {
            model.close();
        }
    }

    /**
     * Calibration of a parameter without non-NaN values derives PyTorch's empty-range parameters, scale 1 and zero
     * point 0, with every method and type, as a QAT observer that never saw a value exports them; the NaN values code
     * to the zero point and reconstruct 0.
     */
    @Test
    public void onlyNaNValuesGiveTheEmptyRangeParams() {
        Module model = new Weight(host, "weight", "weight", new float[]{Float.NaN, Float.NaN});
        try {
            for (Quantizer.QuantizationType type : new Quantizer.QuantizationType[]{
                Quantizer.QuantizationType.INT8_SYMMETRIC, Quantizer.QuantizationType.INT8_ASYMMETRIC,
                Quantizer.QuantizationType.INT4_ASYMMETRIC}) {
                for (Quantizer.CalibrationMethod method : new Quantizer.CalibrationMethod[]{
                    Quantizer.CalibrationMethod.MIN_MAX, Quantizer.CalibrationMethod.PERCENTILE}) {
                    String label = type + " " + method;
                    Quantizer quantizer = Quantizer.builder().quantType(type).calibrationMethod(method).build();
                    Quantizer.QuantizedTensor codes = quantizer.quantizeModel(model).get("weight");
                    assertEquals(label + " scale", 1f, codes.params().scale(), 0f);
                    assertEquals(label + " zero point", 0, codes.params().zeroPoint());
                    try (Tensor restored = quantizer.dequantizeTensor(codes)) {
                        assertArrayEquals(label, new float[]{0f, 0f}, restored.toFloatArray(), 0f);
                    }
                }
            }
            Quantizer fp8 = Quantizer.builder().quantType(Quantizer.QuantizationType.FP8_E4M3).build();
            assertEquals(1f, fp8.quantizeModel(model).get("weight").params().scale(), 0f);
        } finally {
            model.close();
        }
    }

    /** The FP8 types have no integer parameters; their scale is {@code amax / fmax}. */
    @Test
    public void integerParamsRejectTheFp8Types() {
        for (Quantizer.QuantizationType type : new Quantizer.QuantizationType[]{
            Quantizer.QuantizationType.FP8_E4M3, Quantizer.QuantizationType.FP8_E5M2}) {
            try {
                Quantizer.integerParams(type, -1f, 0.5f, 1e-10f);
                fail(type + " has no integer parameters");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains(type.name()));
            }
        }
        assertEquals(1f / 448, Quantizer.fp8Scale(Quantizer.QuantizationType.FP8_E4M3, 1f), 0f);
        assertEquals(1f, Quantizer.fp8Scale(Quantizer.QuantizationType.FP8_E5M2, 0f), 0f);
    }

    /** A parameter the quantizer has not calibrated is calibrated from its own values, not a default range. */
    @Test
    public void calibratesParametersAddedAfterCalibration() {
        Quantizer quantizer = Quantizer.builder().quantType(Quantizer.QuantizationType.INT8_SYMMETRIC).build();
        Module calibration = new Weight(host, "a", "a", new float[]{1f});
        Module model = new Sequential(new Weight(host, "a", "a", new float[]{1f}),
            new Weight(host, "b", "b", new float[]{-5f}));
        try {
            quantizer.calibrate(calibration);
            quantizer.quantizeModel(model);
            assertEquals(5f / 127, quantizer.getQuantParams("1.b").scale(), 0f);
        } finally {
            calibration.close();
            model.close();
        }
    }
}
