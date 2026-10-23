package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

public class PrecisionTest {

    @Test
    public void testMixedPrecisionTrainerBuilder() {
        MixedPrecisionTrainer.Builder builder = MixedPrecisionTrainer.builder()
            .precisionMode(MixedPrecisionTrainer.PrecisionMode.FP16)
            .dynamicLossScaling(true)
            .initialLossScale(65536.0f)
            .lossScaleFactor(2.0f)
            .lossScaleWindow(2000)
            .minLossScale(1.0f)
            .maxLossScale(1000000.0f);
        
        assertNotNull(builder);
    }

    @Test
    public void testMixedPrecisionTrainerBuilderWithBF16() {
        MixedPrecisionTrainer.Builder builder = MixedPrecisionTrainer.builder()
            .precisionMode(MixedPrecisionTrainer.PrecisionMode.BF16)
            .dynamicLossScaling(false)
            .initialLossScale(1024.0f)
            .lossScaleFactor(2.0f)
            .lossScaleWindow(1000)
            .minLossScale(1.0f)
            .maxLossScale(1000000.0f);
        
        assertNotNull(builder);
    }

    @Test
    public void testMixedPrecisionTrainerBuilderSetsNullModel() {
        MixedPrecisionTrainer.Builder builder = MixedPrecisionTrainer.builder()
            .model(null);
        assertNotNull(builder);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMixedPrecisionTrainerInvalidLossScale() {
        MixedPrecisionTrainer.builder()
            .initialLossScale(-1.0f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMixedPrecisionTrainerInvalidLossScaleFactor() {
        MixedPrecisionTrainer.builder()
            .lossScaleFactor(0.5f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMixedPrecisionTrainerInvalidLossScaleWindow() {
        MixedPrecisionTrainer.builder()
            .lossScaleWindow(0);
    }

    @Test
    public void testMixedPrecisionTrainerPrecisionModes() {
        assertEquals(3, MixedPrecisionTrainer.PrecisionMode.values().length);
        assertNotNull(MixedPrecisionTrainer.PrecisionMode.valueOf("FP32"));
        assertNotNull(MixedPrecisionTrainer.PrecisionMode.valueOf("FP16"));
        assertNotNull(MixedPrecisionTrainer.PrecisionMode.valueOf("BF16"));
    }

    @Test
    public void testMixedPrecisionTrainerFP16Mode() {
        MixedPrecisionTrainer.Builder builder = MixedPrecisionTrainer.builder()
            .precisionMode(MixedPrecisionTrainer.PrecisionMode.FP16);
        assertNotNull(builder);
    }

    @Test
    public void testMixedPrecisionTrainerBF16Mode() {
        MixedPrecisionTrainer.Builder builder = MixedPrecisionTrainer.builder()
            .precisionMode(MixedPrecisionTrainer.PrecisionMode.BF16);
        assertNotNull(builder);
    }

    @Test
    public void testQuantizerCreation() {
        Quantizer quantizer = Quantizer.builder()
            .quantType(Quantizer.QuantizationType.INT8_SYMMETRIC)
            .calibrationMethod(Quantizer.CalibrationMethod.MIN_MAX)
            .build();
        
        assertNotNull(quantizer);
        assertEquals(Quantizer.QuantizationType.INT8_SYMMETRIC, quantizer.quantType());
        assertFalse(quantizer.isCalibrated());
    }

    @Test
    public void testQuantizerWithCustomParams() {
        Quantizer quantizer = Quantizer.builder()
            .quantType(Quantizer.QuantizationType.INT4_SYMMETRIC)
            .calibrationMethod(Quantizer.CalibrationMethod.PERCENTILE)
            .percentile(99.5f)
            .numCalibrationSamples(500)
            .build();
        
        assertNotNull(quantizer);
        assertEquals(Quantizer.QuantizationType.INT4_SYMMETRIC, quantizer.quantType());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testQuantizerInvalidPercentileLow() {
        Quantizer.builder()
            .percentile(0.0f)
            .build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testQuantizerInvalidPercentileHigh() {
        Quantizer.builder()
            .percentile(101.0f)
            .build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testQuantizerInvalidNumCalibrationSamples() {
        Quantizer.builder()
            .numCalibrationSamples(0)
            .build();
    }

    @Test
    public void testQuantizerQuantizationTypes() {
        assertEquals(6, Quantizer.QuantizationType.values().length);
        assertNotNull(Quantizer.QuantizationType.valueOf("INT8_SYMMETRIC"));
        assertNotNull(Quantizer.QuantizationType.valueOf("INT8_ASYMMETRIC"));
        assertNotNull(Quantizer.QuantizationType.valueOf("INT4_SYMMETRIC"));
        assertNotNull(Quantizer.QuantizationType.valueOf("INT4_ASYMMETRIC"));
        assertNotNull(Quantizer.QuantizationType.valueOf("FP8_E4M3"));
        assertNotNull(Quantizer.QuantizationType.valueOf("FP8_E5M2"));
    }

    @Test
    public void testQuantizerCalibrationMethods() {
        assertEquals(4, Quantizer.CalibrationMethod.values().length);
        assertNotNull(Quantizer.CalibrationMethod.valueOf("MIN_MAX"));
        assertNotNull(Quantizer.CalibrationMethod.valueOf("PERCENTILE"));
        assertNotNull(Quantizer.CalibrationMethod.valueOf("ENTROPY"));
        assertNotNull(Quantizer.CalibrationMethod.valueOf("MSE"));
    }

    @Test
    public void testQuantizationParams() {
        Quantizer.QuantizationParams params = new Quantizer.QuantizationParams(
            0.1f, 0, -128, 127, Quantizer.QuantizationType.INT8_SYMMETRIC
        );
        
        assertEquals(0.1f, params.scale(), 0.001f);
        assertEquals(0, params.zeroPoint());
        assertEquals(-128, params.quantMin());
        assertEquals(127, params.quantMax());
        assertEquals(Quantizer.QuantizationType.INT8_SYMMETRIC, params.quantType());
    }

    @Test
    public void testQuantizationParamsToString() {
        Quantizer.QuantizationParams params = new Quantizer.QuantizationParams(
            0.5f, 10, -128, 127, Quantizer.QuantizationType.INT8_ASYMMETRIC
        );
        
        String str = params.toString();
        assertTrue(str.contains("QuantizationParams"));
        assertTrue(str.contains("scale=0.5"));
    }

    @Test
    public void testQuantizedTensor() {
        int[] data = {100, -50, 75, -25};
        Shape shape = Shape.vector(4);
        Quantizer.QuantizationParams params = new Quantizer.QuantizationParams(
            0.01f, 0, -128, 127, Quantizer.QuantizationType.INT8_ASYMMETRIC
        );
        
        Quantizer.QuantizedTensor qt = new Quantizer.QuantizedTensor(data, shape, params);
        
        assertArrayEquals(data, qt.data());
        assertEquals(shape, qt.shape());
        assertEquals(params, qt.params());
        assertEquals(4, qt.numElements());
    }

    @Test
    public void testQuantizedTensorDefensiveCopy() {
        int[] data = {100, -50};
        Shape shape = Shape.vector(2);
        Quantizer.QuantizationParams params = new Quantizer.QuantizationParams(
            0.01f, 0, -128, 127, Quantizer.QuantizationType.INT8_ASYMMETRIC
        );
        
        Quantizer.QuantizedTensor qt = new Quantizer.QuantizedTensor(data, shape, params);
        
        data[0] = 999;
        
        assertEquals(100, qt.data()[0]);
    }

    @Test
    public void testQuantizedTensorToString() {
        int[] data = {100, -50};
        Shape shape = Shape.vector(2);
        Quantizer.QuantizationParams params = new Quantizer.QuantizationParams(
            0.01f, 0, -128, 127, Quantizer.QuantizationType.INT8_ASYMMETRIC
        );
        
        Quantizer.QuantizedTensor qt = new Quantizer.QuantizedTensor(data, shape, params);
        
        String str = qt.toString();
        assertTrue(str.contains("QuantizedTensor"));
        assertTrue(str.contains("numElements=2"));
    }

    @Test
    public void testQATModuleCreation() {
        Module baseModule = new SimpleTestModule();
        
        QATModule qat = QATModule.builder()
            .baseModule(baseModule)
            .quantType(Quantizer.QuantizationType.INT8_SYMMETRIC)
            .quantizeWeights(true)
            .quantizeActivations(true)
            .build();
        
        assertNotNull(qat);
        assertEquals(baseModule, qat.baseModule());
        assertEquals(Quantizer.QuantizationType.INT8_SYMMETRIC, qat.quantType());
        assertTrue(qat.isQuantizeWeights());
        assertTrue(qat.isQuantizeActivations());
    }

    @Test(expected = NullPointerException.class)
    public void testQATModuleNullBaseModuleRejected() {
        QATModule.builder()
            .baseModule(null)
            .build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testQATModuleInvalidObserverEpsZero() {
        Module baseModule = new SimpleTestModule();
        QATModule.builder()
            .baseModule(baseModule)
            .observerEps(0.0f)
            .build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testQATModuleInvalidObserverEpsNegative() {
        Module baseModule = new SimpleTestModule();
        QATModule.builder()
            .baseModule(baseModule)
            .observerEps(-0.001f)
            .build();
    }

    @Test
    public void testQATModuleFlags() {
        Module baseModule = new SimpleTestModule();
        
        QATModule qat = QATModule.builder()
            .baseModule(baseModule)
            .quantizeWeights(false)
            .quantizeActivations(false)
            .perChannelQuantization(true)
            .build();
        
        assertFalse(qat.isQuantizeWeights());
        assertFalse(qat.isQuantizeActivations());
    }

    /**
     * With fake quantization disabled the forward pass is the base module's: nothing is observed while the observers
     * are disabled too, enabled observers calibrate the activation sites, and frozen observers keep that calibration.
     */
    @Test
    public void testQATModuleEnableDisableObservers() {
        ComputeBackend host = TrainParity.hostBackend();
        QATModule qat = QATModule.builder().baseModule(new SimpleTestModule()).quantizeWeights(false).build();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor calibration = Tensor.fromArray(host, new float[]{-1f, 0.5f, 2f});
             Tensor wider = Tensor.fromArray(host, new float[]{-8f, 0.5f, 8f})) {
            qat.enableObservers(false);
            qat.enableFakeQuant(false);
            assertSame("the base forward pass", calibration, qat.forward(calibration));
            assertTrue("disabled observers observe nothing", qat.toQuantizer().activationQuantParams().isEmpty());

            qat.enableObservers(true);
            assertSame("the base forward pass", calibration, qat.forward(calibration));
            Quantizer.QuantizationParams expected = Quantizer.integerParams(
                Quantizer.QuantizationType.INT8_SYMMETRIC, -1f, 2f, 1e-5f);
            for (String site : new String[]{"input", "output"}) {
                Quantizer.QuantizationParams observed = qat.toQuantizer().getActivationQuantParams(site);
                assertEquals(site + " scale", expected.scale(), observed.scale(), 0f);
                assertEquals(site + " zero point", expected.zeroPoint(), observed.zeroPoint());
            }

            qat.freezeObservers();
            assertSame("the base forward pass", wider, qat.forward(wider));
            assertEquals("frozen observers keep their calibration", expected.scale(),
                qat.toQuantizer().getActivationQuantParams("input").scale(), 0f);
        } finally {
            qat.close();
            TrainGpu.release(host);
        }
    }

    @Test
    public void testQATModuleToQuantizer() {
        Module baseModule = new SimpleTestModule();
        
        QATModule qat = QATModule.builder()
            .baseModule(baseModule)
            .quantType(Quantizer.QuantizationType.INT8_ASYMMETRIC)
            .build();
        
        Quantizer quantizer = qat.toQuantizer();
        assertNotNull(quantizer);
        assertEquals(Quantizer.QuantizationType.INT8_ASYMMETRIC, quantizer.quantType());
    }

    @Test
    public void testQATModuleModuleName() {
        Module baseModule = new SimpleTestModule();
        
        QATModule qat = QATModule.builder()
            .baseModule(baseModule)
            .build();
        
        String name = qat.moduleName();
        assertTrue(name.contains("QATModule"));
        assertTrue(name.contains("SimpleTestModule"));
    }

    @Test
    public void testQATModuleToString() {
        Module baseModule = new SimpleTestModule();
        
        QATModule qat = QATModule.builder()
            .baseModule(baseModule)
            .quantType(Quantizer.QuantizationType.INT4_SYMMETRIC)
            .build();
        
        String str = qat.toString();
        assertTrue(str.contains("QATModule"));
        assertTrue(str.contains("INT4_SYMMETRIC"));
    }

    @Test
    public void testQATModuleWeightScaleObservers() {
        Module baseModule = new SimpleTestModule();
        
        QATModule qat = QATModule.builder()
            .baseModule(baseModule)
            .build();
        
        Map<String, Float> observers = qat.weightScaleObservers();
        assertNotNull(observers);
        assertTrue(observers.isEmpty() || observers.containsKey("weight"));
    }

    /**
     * The clipped straight-through estimator through an identity QAT module (INT8 symmetric, observers calibrated on
     * a range of +-1 and frozen, scale 1/127): 0.5 has an in-range code and receives the gradient, 2 and -3 are
     * clamped and NaN has no code, so they receive 0. The graph returns to its size before the step.
     */
    @Test
    public void testQATModuleStraightThroughEstimator() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        QATModule qat = QATModule.builder().baseModule(new SimpleTestModule()).quantizeWeights(false).build();
        try (Tensor calibration = Tensor.fromArray(host, new float[]{-1f, 0.5f, 1f});
             Tensor base = Tensor.fromArray(host, new float[]{0.5f, 2f, -3f, Float.NaN});
             Tensor x = base.withRequiresGrad(true)) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                qat.forward(calibration).close();
            }
            qat.freezeObservers();
            int graphBefore = ComputationGraph.current().size();
            Tensor y = qat.forward(x);
            assertTrue("fake quantization records", y.requiresGrad());
            Tensor loss = TensorOps.sum(y);
            loss.backward();
            assertArrayEquals(new float[]{1f, 0f, 0f, 0f}, x.grad().toFloatArray(), 0.0f);
            assertEquals("graph size", graphBefore, ComputationGraph.current().size());
            loss.close();
            y.close();
            x.grad().close();
            x.setGrad(null);
        } finally {
            qat.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * Gradients reach the base module's parameters through the fake-quantized activations: with input scale 1/64,
     * output scale 1/32 and {@code y = x w} ({@code w = 2}), every value is on its grid, so
     * {@code dL/dw = sum(x)} and {@code dL/dx = w} exactly for {@code L = sum(y)}.
     */
    @Test
    public void testQATModuleGradientReachesTheBaseParameters() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        ScaleModule scaleModule = new ScaleModule(host, 2f);
        QATModule qat = QATModule.builder().baseModule(scaleModule).quantizeWeights(false).build();
        try (Tensor calibration = Tensor.fromArray(host, new float[][]{{127f / 64}, {-127f / 64}});
             Tensor base = Tensor.fromArray(host, new float[][]{{0.5f}, {-1f}, {0.25f}});
             Tensor x = base.withRequiresGrad(true)) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                qat.forward(calibration).close();
            }
            qat.freezeObservers();
            Tensor y = qat.forward(x);
            assertArrayEquals(new float[]{1f, -2f, 0.5f}, y.toFloatArray(), 0.0f);
            Tensor loss = TensorOps.sum(y);
            loss.backward();
            assertArrayEquals(new float[]{-0.25f}, scaleModule.weight.grad().toFloatArray(), 0.0f);
            assertArrayEquals(new float[]{2f, 2f, 2f}, x.grad().toFloatArray(), 0.0f);
            loss.close();
            y.close();
            x.grad().close();
            x.setGrad(null);
        } finally {
            qat.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /** Without grad on the input, fake quantization does not record; invalid parameters are refused. */
    @Test
    public void testFakeQuantGradArguments() {
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor x = Tensor.fromArray(host, new float[]{0.3f, -0.7f})) {
            try (Tensor y = FakeQuantGrad.apply(x, 0.5f, 0, -128, 127)) {
                assertFalse(y.requiresGrad());
                assertArrayEquals(new float[]{0.5f, -0.5f}, y.toFloatArray(), 0.0f);
            }
            try {
                FakeQuantGrad.apply(x, 0f, 0, -128, 127);
                fail("a zero scale was accepted");
            } catch (IllegalArgumentException expected) {
            }
            try {
                FakeQuantGrad.apply(x, 0.5f, 0, 1, 0);
                fail("an empty code range was accepted");
            } catch (IllegalArgumentException expected) {
            }
        } finally {
            TrainGpu.release(host);
        }
    }

    /** {@code x w} with one parameter {@code w} of shape [1, 1]. */
    private static final class ScaleModule extends Module {
        final Parameter weight;

        ScaleModule(ComputeBackend b, float w) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, new float[][]{{w}}), "weight"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    private static class SimpleTestModule extends Module {
        
        SimpleTestModule() {
        }
        
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
        
        @Override
        public String moduleName() {
            return "SimpleTestModule";
        }
    }
}
