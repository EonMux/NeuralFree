package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationParams;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationType;
import com.eonmux.neuralfree.precision.Quantizer.QuantizedTensor;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link QATModule#toQuantizer()} exports the parameters the observers hold, as PyTorch's {@code convert} does: the
 * dequantized export of every fake-quantized weight equals the value the module fake-quantizes it to, also when the
 * observers saw a wider range than the final weights and were then frozen, per tensor and per channel, and when
 * enabled observers fold in a weight that moved past their range; a weight the base module reaches under several
 * paths exports its one site's parameters under every path; biases stay in full precision, and the activation sites
 * export their own parameters.
 */
public class QatExportTest {
    /** A range the observers see before freezing: wider than {@link #FINAL} in every row. */
    private static final float[] WIDE = {4f, -3f, 0.5f, -8f, 2f, 6f};
    /** The base layer's final weight, of shape {@code [2, 3]}. */
    private static final float[] FINAL = {0.3f, -0.7f, 0.11f, 1.2f, -0.05f, 0.9f};
    private static final QuantizationType[] TYPES = {
        QuantizationType.INT8_SYMMETRIC, QuantizationType.INT8_ASYMMETRIC, QuantizationType.INT4_SYMMETRIC,
        QuantizationType.INT4_ASYMMETRIC, QuantizationType.FP8_E4M3, QuantizationType.FP8_E5M2};

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
    public void exportEqualsFakeQuantizationWithFrozenWiderObservers() {
        for (QuantizationType type : TYPES) {
            checkExport(type, false);
            checkExport(type, true);
        }
    }

    private void checkExport(QuantizationType type, boolean perChannel) {
        String label = type + (perChannel ? " per channel" : " per tensor");
        Linear base = QatGradCheckTest.layer(host, FINAL);
        QATModule qat = QATModule.builder().baseModule(base).quantType(type).perChannelQuantization(perChannel)
            .quantizeActivations(false).build();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor wide = QatGradCheckTest.tensor(host, WIDE, 2, 3)) {
            qat.fakeQuantizeWeight(wide, "weight").close();
            qat.freezeObservers();

            Quantizer quantizer = qat.toQuantizer();
            Map<String, QuantizedTensor> codes = quantizer.quantizeModel(base);
            assertEquals(label + ": only the fake-quantized weight is coded",
                Collections.singletonList("weight"), new ArrayList<>(codes.keySet()));
            assertNull(label + ": the bias stays in full precision", quantizer.getQuantParams("bias"));

            QuantizationParams params = quantizer.getQuantParams("weight");
            assertEquals(label + ": per-channel export", perChannel, params.isPerChannel());
            assertObserved(label, qat, params, perChannel);
            Quantizer calibrating = Quantizer.builder().quantType(type).build();
            calibrating.quantizeModel(base);
            assertNotEquals(label + ": not a recalibration of the final weight",
                calibrating.getQuantParams("weight").scale(), params.scale(0), 0f);

            Tensor weight = base.namedParameters().get("weight").tensor();
            try (Tensor exported = quantizer.dequantizeTensor(codes.get("weight"));
                 Tensor trained = qat.fakeQuantizeWeight(weight, "weight")) {
                assertArrayEquals(label + ": export equals fake quantization", trained.toFloatArray(),
                    exported.toFloatArray(), 0f);
            }
        } finally {
            qat.close();
        }
    }

    /** The exported weight parameters are the observers' stored scales and zero points. */
    private static void assertObserved(String label, QATModule qat, QuantizationParams params, boolean perChannel) {
        if (perChannel) {
            assertArrayEquals(label + ": scales", qat.weightChannelScales().get("weight"), params.scales(), 0f);
            assertArrayEquals(label + ": zero points", qat.weightChannelZeroPoints().get("weight"),
                params.zeroPoints());
        } else {
            assertEquals(label + ": scale", qat.weightScaleObservers().get("weight"), params.scale(), 0f);
            assertEquals(label + ": zero point", (int) qat.weightZeroPointObservers().get("weight"),
                params.zeroPoint());
        }
    }

    /** The input and output sites export their observed per-tensor parameters. */
    @Test
    public void exportsActivationParameters() {
        Linear base = QatGradCheckTest.layer(host, FINAL);
        QATModule qat = QATModule.builder().baseModule(base).quantType(QuantizationType.INT8_SYMMETRIC).build();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor input = QatGradCheckTest.tensor(host, new float[]{1f, -2.54f, 0.5f}, 1, 3);
             Tensor output = qat.forward(input)) {
            assertNotNull(output);
            Quantizer quantizer = qat.toQuantizer();
            QuantizationParams in = quantizer.getActivationQuantParams("input");
            assertFalse(in.isPerChannel());
            QuantizationParams expected = Quantizer.integerParams(QuantizationType.INT8_SYMMETRIC, -2.54f, 1f, 1e-5f);
            assertEquals(expected.scale(), in.scale(), 0f);
            assertEquals(0, in.zeroPoint());
            assertNotNull(quantizer.getActivationQuantParams("output"));
            assertEquals(Arrays.asList("input", "output"),
                new ArrayList<>(quantizer.activationQuantParams().keySet()));
            assertNull(quantizer.getActivationQuantParams("hidden"));
            assertTrue(Quantizer.builder().build().activationQuantParams().isEmpty());
        } finally {
            qat.close();
        }
    }

    /**
     * A site that was never observed exports PyTorch's empty-range parameters, scale 1 and zero point 0, once its
     * observers are frozen (enabled weight observers fold in the current weight at export).
     */
    @Test
    public void unobservedSiteExportsUnitScale() {
        Linear base = QatGradCheckTest.layer(host, FINAL);
        QATModule qat = QATModule.builder().baseModule(base).quantType(QuantizationType.INT8_ASYMMETRIC).build();
        try {
            qat.freezeObservers();
            Quantizer quantizer = qat.toQuantizer();
            assertEquals(1f, quantizer.getQuantParams("weight").scale(), 0f);
            assertEquals(0, quantizer.getQuantParams("weight").zeroPoint());
            assertEquals(1f, quantizer.getActivationQuantParams("input").scale(), 0f);
        } finally {
            qat.close();
        }
    }

    /**
     * The exported quantizer codes with its fixed parameters only: it does not calibrate, and a model without any
     * of its weight sites is rejected.
     */
    @Test
    public void observedQuantizerDoesNotCalibrate() {
        Linear base = QatGradCheckTest.layer(host, FINAL);
        QATModule qat = QATModule.builder().baseModule(base).quantizeActivations(false).build();
        Module other = new QuantizerTest.Weight(host, "w", "w", new float[]{1f, 2f});
        try {
            Quantizer quantizer = qat.toQuantizer();
            assertTrue(quantizer.isCalibrated());
            assertThrows(IllegalStateException.class, () -> quantizer.calibrate(base));
            assertThrows(IllegalArgumentException.class, () -> quantizer.quantizeModel(other));
            assertThrows(IllegalArgumentException.class, () -> quantizer.quantizeModel(qat));
        } finally {
            other.close();
            qat.close();
        }
    }

    /**
     * With the observers enabled, the export first folds in the current weights, as PyTorch's {@code from_float}
     * does: a weight stepped past the range of the last forward pass is coded with the widened range, exactly as the
     * next forward pass fake-quantizes it, not clamped to the stale range.
     */
    @Test
    public void enabledObserversFoldTheFinalWeightsAtExport() {
        for (QuantizationType type : TYPES) {
            checkEnabledExport(type, false);
            checkEnabledExport(type, true);
        }
    }

    private void checkEnabledExport(QuantizationType type, boolean perChannel) {
        String label = type + (perChannel ? " per channel" : " per tensor");
        Linear base = QatGradCheckTest.layer(host, FINAL);
        QATModule qat = QATModule.builder().baseModule(base).quantType(type).perChannelQuantization(perChannel)
            .quantizeActivations(false).build();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor input = QatGradCheckTest.tensor(host, new float[]{1f, -0.5f, 2f}, 1, 3)) {
            qat.forward(input).close();
            float stale = storedScale(qat, perChannel);
            Tensor weight = base.namedParameters().get("weight").tensor();
            GpuOptimizerOps.storeFloats(weight, WIDE);

            Quantizer quantizer = qat.toQuantizer();
            QuantizationParams params = quantizer.getQuantParams("weight");
            assertNotEquals(label + ": the export folds in the stepped weight", stale, params.scale(0), 0f);
            assertObserved(label, qat, params, perChannel);
            Map<String, QuantizedTensor> codes = quantizer.quantizeModel(base);
            try (Tensor exported = quantizer.dequantizeTensor(codes.get("weight"));
                 Tensor next = qat.fakeQuantizeWeight(weight, "weight")) {
                assertArrayEquals(label + ": export equals the next fake quantization", next.toFloatArray(),
                    exported.toFloatArray(), 0f);
            }
        } finally {
            qat.close();
        }
    }

    /** The stored scale of the {@code weight} site (of channel 0 per channel). */
    private static float storedScale(QATModule qat, boolean perChannel) {
        return perChannel ? qat.weightChannelScales().get("weight")[0] : qat.weightScaleObservers().get("weight");
    }

    /**
     * One {@link Linear} registered twice ({@code Sequential(layer, relu, layer)}) is one weight site, under its first
     * path {@code 0.weight}: the export carries that site's parameters under both paths, and both codes reconstruct
     * the value the module fake-quantizes the shared weight to.
     */
    @Test
    public void sharedWeightExportsItsSiteUnderEveryPath() {
        for (QuantizationType type : TYPES) {
            checkSharedExport(type, false);
            checkSharedExport(type, true);
        }
    }

    private void checkSharedExport(QuantizationType type, boolean perChannel) {
        String label = type + (perChannel ? " per channel" : " per tensor");
        Linear layer = new Linear(2, 2);
        layer.initialize(host, new Random(17));
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("weight", new float[]{0.6f, -1.3f, 0.2f, 0.9f});
        state.put("bias", new float[]{0.1f, -0.1f});
        layer.loadStateDict(state);
        Sequential base = new Sequential(layer, new ReLU(), layer);
        QATModule qat = QATModule.builder().baseModule(base).quantType(type).perChannelQuantization(perChannel)
            .quantizeActivations(false).build();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor input = QatGradCheckTest.tensor(host, new float[]{1.5f, -2f}, 1, 2)) {
            qat.forward(input).close();
            qat.freezeObservers();
            assertFalse(label + ": no observer for the second path",
                qat.stateDict().containsKey("weight_fake_quant.2_weight.scale"));
            assertTrue(label + ": the first path's observer",
                qat.stateDict().containsKey("weight_fake_quant.0_weight.scale"));

            Quantizer quantizer = qat.toQuantizer();
            QuantizationParams first = quantizer.getQuantParams("0.weight");
            QuantizationParams second = quantizer.getQuantParams("2.weight");
            assertNotNull(label + ": the second path is exported", second);
            assertArrayEquals(label + ": scales", first.scales(), second.scales(), 0f);
            assertArrayEquals(label + ": zero points", first.zeroPoints(), second.zeroPoints());
            assertNotEquals(label + ": observed, not the empty-range scale", 1f, second.scale(0), 0f);

            Map<String, QuantizedTensor> codes = quantizer.quantizeModel(base);
            Tensor weight = layer.namedParameters().get("weight").tensor();
            try (Tensor trained = qat.fakeQuantizeWeight(weight, "2.weight");
                 Tensor viaFirst = qat.fakeQuantizeWeight(weight, "0.weight");
                 Tensor exportedFirst = quantizer.dequantizeTensor(codes.get("0.weight"));
                 Tensor exportedSecond = quantizer.dequantizeTensor(codes.get("2.weight"))) {
                float[] expected = viaFirst.toFloatArray();
                assertArrayEquals(label + ": both paths name one site", expected, trained.toFloatArray(), 0f);
                assertArrayEquals(label + ": first path", expected, exportedFirst.toFloatArray(), 0f);
                assertArrayEquals(label + ": second path", expected, exportedSecond.toFloatArray(), 0f);
            }
        } finally {
            qat.close();
        }
    }
}
