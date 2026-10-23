package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Test;

import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The observer of {@link QATModule}: PyTorch's affine parameters over a range widened to contain 0 (shared with
 * {@link Quantizer} through {@link Quantizer#asymmetricParams}), and the observer state as persistent buffers that
 * {@link Module#stateDict()} saves and {@link Module#loadStateDict(Map, boolean)} restores.
 */
public class QatObserverStateTest {
    private static final float EPS = 1e-5f;

    private final ComputeBackend backend = TrainParity.hostBackend();

    @After
    public void tearDown() {
        backend.close();
    }

    @Test
    public void asymmetricParametersCoverARangeOfOneSign() {
        Quantizer.QuantizationParams positive = Quantizer.asymmetricParams(0.2f, 0.9f, -128, 127, EPS,
            Quantizer.QuantizationType.INT8_ASYMMETRIC);
        assertEquals(0.9f / 255, positive.scale(), 1e-9f);
        assertEquals(-128, positive.zeroPoint());
        Quantizer.QuantizationParams negative = Quantizer.asymmetricParams(-0.9f, -0.2f, -128, 127, EPS,
            Quantizer.QuantizationType.INT8_ASYMMETRIC);
        assertEquals(0.9f / 255, negative.scale(), 1e-9f);
        assertEquals(127, negative.zeroPoint());
        Quantizer.QuantizationParams mixed = Quantizer.asymmetricParams(-1f, 3f, -8, 7, EPS,
            Quantizer.QuantizationType.INT4_ASYMMETRIC);
        assertEquals(4f / 15, mixed.scale(), 1e-7f);
        assertEquals(-8 - Math.round(-1f / (4f / 15)), mixed.zeroPoint());
    }

    /** Sigmoid-like activations in [0.2, 0.9] keep their full range instead of saturating at 0.7. */
    @Test
    public void positiveActivationsAreNotCutOff() {
        float[] values = {0.2f, 0.5f, 0.9f, 0.7f};
        QATModule qat = QATModule.builder().baseModule(new Identity()).quantizeWeights(false)
            .quantType(Quantizer.QuantizationType.INT8_ASYMMETRIC).build();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor x = Tensor.fromArray(backend, values);
             Tensor y = qat.forward(x)) {
            float[] out = y.toFloatArray();
            float halfStep = 0.9f / 255 / 2;
            for (int i = 0; i < values.length; i++) {
                assertEquals("element " + i, values[i], out[i], halfStep + 1e-6f);
            }
        } finally {
            qat.close();
        }
    }

    /** A strict load of a calibrated module's state into a fresh one reproduces its frozen forward exactly. */
    @Test
    public void observerStateRoundTripsThroughTheStateDict() {
        QATModule source = qat(1L);
        QATModule target = qat(2L);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor x = Tensor.fromArray(backend, new float[][]{{0.5f, -1f, 2f}, {0.25f, 0.75f, -0.5f}})) {
            source.forward(x).close();
            source.freezeObservers();
            Map<String, Object> state = source.stateDict();
            assertTrue(state.containsKey("weight_fake_quant.weight.scale"));
            assertTrue(state.containsKey("weight_fake_quant.weight.min_val"));
            assertTrue(state.containsKey("activation_fake_quant.input.zero_point"));
            assertTrue(state.containsKey("activation_fake_quant.output.max_val"));
            assertTrue("a calibrated scale is not the default", ((float[]) state.get(
                "weight_fake_quant.weight.scale"))[0] != 1.0f);

            target.loadStateDict(state, true);
            target.freezeObservers();
            assertEquals(source.weightScaleObservers(), target.weightScaleObservers());
            assertEquals(source.weightZeroPointObservers(), target.weightZeroPointObservers());
            try (Tensor expected = source.forward(x); Tensor actual = target.forward(x)) {
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 0f);
            }
        } finally {
            source.close();
            target.close();
        }
    }

    /**
     * PyTorch's calibration with {@code disable_fake_quant} and {@code enable_observer}: the forward pass returns the
     * base output unquantized while every weight and activation observer folds in its range, and the export carries
     * the observed activation parameters instead of the empty-range scale 1 and zero point 0.
     */
    @Test
    public void disabledFakeQuantizationStillCalibrates() {
        QATModule qat = qat(3L);
        Module base = qat.baseModule();
        Quantizer.QuantizationType type = Quantizer.QuantizationType.INT8_ASYMMETRIC;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor x = Tensor.fromArray(backend, new float[][]{{0.5f, -1f, 2f}, {0.25f, 0.75f, -0.5f}})) {
            qat.enableFakeQuant(false);
            assertEquals("a new site has the empty-range scale", 1.0f,
                ((float[]) qat.stateDict().get("activation_fake_quant.input.scale"))[0], 0f);
            float[] output;
            try (Tensor expected = base.forward(x); Tensor actual = qat.forward(x)) {
                output = expected.toFloatArray();
                assertArrayEquals("the base output, unquantized", output, actual.toFloatArray(), 0f);
            }

            Quantizer.QuantizationParams input = Quantizer.integerParams(type, -1f, 2f, EPS);
            Quantizer.QuantizationParams out = Quantizer.integerParams(type, min(output), max(output), EPS);
            float[] weight = base.getParameter("weight").tensor().toFloatArray();
            Quantizer.QuantizationParams weights = Quantizer.integerParams(type, min(weight), max(weight), EPS);
            Map<String, Object> state = qat.stateDict();
            assertSite(state, "activation_fake_quant.input", input);
            assertSite(state, "activation_fake_quant.output", out);
            assertSite(state, "weight_fake_quant.weight", weights);

            Quantizer exported = qat.toQuantizer();
            assertEquals(input.scale(), exported.getActivationQuantParams("input").scale(), 0f);
            assertEquals(input.zeroPoint(), exported.getActivationQuantParams("input").zeroPoint());
            assertEquals(out.scale(), exported.getActivationQuantParams("output").scale(), 0f);
            assertEquals(out.zeroPoint(), exported.getActivationQuantParams("output").zeroPoint());
        } finally {
            qat.close();
        }
    }

    private static void assertSite(Map<String, Object> state, String prefix, Quantizer.QuantizationParams expected) {
        assertEquals(prefix + " scale", expected.scale(), ((float[]) state.get(prefix + ".scale"))[0], 0f);
        assertEquals(prefix + " zero point", expected.zeroPoint(),
            ((float[]) state.get(prefix + ".zero_point"))[0], 0f);
    }

    private static float min(float[] values) {
        float min = Float.POSITIVE_INFINITY;
        for (float v : values) {
            min = Math.min(min, v);
        }
        return min;
    }

    private static float max(float[] values) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : values) {
            max = Math.max(max, v);
        }
        return max;
    }

    private QATModule qat(long seed) {
        Linear base = new Linear(3, 2);
        base.initialize(backend, new Random(seed));
        return QATModule.builder().baseModule(base).quantType(Quantizer.QuantizationType.INT8_ASYMMETRIC).build();
    }

    private static final class Identity extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }
}
