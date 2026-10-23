package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;

public class DiffusionFrameworkTest {

    private static final class MockNoisePredictor extends Module {
        private final ComputeBackend backend;

        MockNoisePredictor(ComputeBackend backend) {
            this.backend = backend;
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }

        @Override
        public List<Parameter> parameters() {
            return new ArrayList<>();
        }

        @Override
        public ComputeBackend backend() {
            return backend;
        }
    }

    private static final class MockOptimizer extends Optimizer {
        MockOptimizer() {
            super(Collections.emptyList(), new HashMap<>());
            defaults.put("lr", 0.001f);
        }

        @Override
        public void step() {
        }
    }

    private static final class MockBackend implements ComputeBackend {
        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities("MockDevice", 8L * 1024 * 1024 * 1024,
                2L * 1024 * 1024 * 1024, 16, 1500, 1024L, true, "1.0", true, true, false, 65536L,
                null, null, null);
        }

        @Override
        public ComputeQueue computeQueue() {
            return null;
        }

        @Override
        public ComputeQueue transferQueue() {
            return null;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            return null;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return null;
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return null;
        }

        /** Declines kernel artifacts so tensor ops take their CPU path. */
        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return false;
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            if (artifact == null) {
                throw new IllegalArgumentException("kernel artifact must not be null");
            }
            throw new UnsupportedKernelException("mock backend declines " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }

        @Override
        public void close() {
        }
    }

    @Test
    public void testNoiseSchedulerInterfaceExists() {
        assertNotNull(NoiseScheduler.class);
    }

    @Test
    public void testLinearSchedulerCreation() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);

        assertEquals(1000, scheduler.numTimesteps());
        assertEquals("LinearScheduler", scheduler.name());
        assertNotNull(scheduler.getBetas());
        assertNotNull(scheduler.getAlphas());
        assertNotNull(scheduler.getAlphasCumprod());
        assertEquals(1000, scheduler.getBetas().length);
    }

    @Test
    public void testLinearSchedulerCustomRange() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000, 0.0001f, 0.02f);

        assertEquals(0.0001f, scheduler.betaStart(), 1e-7f);
        assertEquals(0.02f, scheduler.betaEnd(), 1e-7f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinearSchedulerInvalidTimesteps() {
        ComputeBackend backend = new MockBackend();
        new LinearScheduler(backend, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinearSchedulerInvalidTimestepsNegative() {
        ComputeBackend backend = new MockBackend();
        new LinearScheduler(backend, -100);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinearSchedulerInvalidBetaRange() {
        ComputeBackend backend = new MockBackend();
        new LinearScheduler(backend, 1000, 0.5f, 0.1f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinearSchedulerInvalidBetaNegative() {
        ComputeBackend backend = new MockBackend();
        new LinearScheduler(backend, 1000, -0.1f, 0.5f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinearSchedulerInvalidBetaTooLarge() {
        ComputeBackend backend = new MockBackend();
        new LinearScheduler(backend, 1000, 0.1f, 1.5f);
    }

    @Test
    public void testCosineSchedulerCreation() {
        ComputeBackend backend = new MockBackend();
        CosineScheduler scheduler = new CosineScheduler(backend, 1000);

        assertEquals(1000, scheduler.numTimesteps());
        assertEquals("CosineScheduler", scheduler.name());
        assertEquals(0.008f, scheduler.s(), 1e-7f);
    }

    @Test
    public void testCosineSchedulerCustomS() {
        ComputeBackend backend = new MockBackend();
        CosineScheduler scheduler = new CosineScheduler(backend, 1000, 0.01f);

        assertEquals(0.01f, scheduler.s(), 1e-7f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCosineSchedulerInvalidS() {
        ComputeBackend backend = new MockBackend();
        new CosineScheduler(backend, 1000, -0.01f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCosineSchedulerInvalidSZero() {
        ComputeBackend backend = new MockBackend();
        new CosineScheduler(backend, 1000, 0.0f);
    }

    @Test
    public void testSquaredCosineSchedulerCreation() {
        ComputeBackend backend = new MockBackend();
        SquaredCosineScheduler scheduler = new SquaredCosineScheduler(backend, 1000);

        assertEquals(1000, scheduler.numTimesteps());
        assertEquals("SquaredCosineScheduler", scheduler.name());
    }

    @Test
    public void testSchedulerGetAlphaCumprod() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);

        float alpha0 = scheduler.getAlphaCumprod(0);
        float alpha999 = scheduler.getAlphaCumprod(999);

        assertTrue(alpha0 > 0 && alpha0 < 1);
        assertTrue(alpha999 > 0 && alpha999 < 1);
        assertTrue(alpha999 < alpha0);
    }

    @Test
    public void testSchedulerGetBeta() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);

        float beta0 = scheduler.getBeta(0);
        float beta999 = scheduler.getBeta(999);

        assertTrue(beta0 > 0 && beta0 < 1);
        assertTrue(beta999 > 0 && beta999 < 1);
    }

    @Test
    public void testSchedulerGetNoiseLevel() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);

        float noiseLevel0 = scheduler.getNoiseLevel(0);
        float noiseLevel999 = scheduler.getNoiseLevel(999);

        assertTrue(noiseLevel0 >= 0 && noiseLevel0 < 1);
        assertTrue(noiseLevel999 > 0 && noiseLevel999 <= 1);
        assertTrue(noiseLevel999 > noiseLevel0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSchedulerInvalidTimestepNeg() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        scheduler.getAlphaCumprod(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSchedulerInvalidTimestepTooLarge() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        scheduler.getAlphaCumprod(1000);
    }

    @Test
    public void testSchedulerToString() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler linear = new LinearScheduler(backend, 1000);
        CosineScheduler cosine = new CosineScheduler(backend, 1000);
        SquaredCosineScheduler squared = new SquaredCosineScheduler(backend, 1000);

        assertTrue(linear.toString().contains("LinearScheduler"));
        assertTrue(cosine.toString().contains("CosineScheduler"));
        assertTrue(squared.toString().contains("SquaredCosineScheduler"));
    }

    @Test
    public void testDDPMTrainerCreation() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        Optimizer optimizer = new MockOptimizer();
        DDPMTrainer trainer = new DDPMTrainer(predictor, scheduler, optimizer, backend);

        assertNotNull(trainer);
        assertEquals(predictor, trainer.noisePredictor());
        assertEquals(scheduler, trainer.scheduler());
        assertEquals(backend, trainer.backend());
    }

    @Test(expected = NullPointerException.class)
    public void testDDPMTrainerNullPredictor() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        Optimizer optimizer = new MockOptimizer();
        new DDPMTrainer(null, scheduler, optimizer, backend);
    }

    @Test(expected = NullPointerException.class)
    public void testDDPMTrainerNullScheduler() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        Optimizer optimizer = new MockOptimizer();
        new DDPMTrainer(predictor, null, optimizer, backend);
    }

    @Test(expected = NullPointerException.class)
    public void testDDPMTrainerNullBackend() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        Optimizer optimizer = new MockOptimizer();
        new DDPMTrainer(predictor, scheduler, optimizer, null);
    }

    @Test
    public void testDDPMTrainerAccessors() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        Optimizer optimizer = new MockOptimizer();
        DDPMTrainer trainer = new DDPMTrainer(predictor, scheduler, optimizer, backend);

        assertEquals(0, trainer.stepCount());
        assertEquals(0.0f, trainer.lastLoss(), 1e-7f);
    }

    @Test
    public void testDDPMTrainerToString() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        Optimizer optimizer = new MockOptimizer();
        DDPMTrainer trainer = new DDPMTrainer(predictor, scheduler, optimizer, backend);

        assertTrue(trainer.toString().contains("DDPMTrainer"));
        assertTrue(trainer.toString().contains("1000"));
    }

    @Test
    public void testDDPMTrainerClose() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        Optimizer optimizer = new MockOptimizer();
        DDPMTrainer trainer = new DDPMTrainer(predictor, scheduler, optimizer, backend);
        trainer.close();
    }

    @Test
    public void testDDPMTrainerTrainingStepResult() {
        DDPMTrainer.TrainingStepResult result = 
            new DDPMTrainer.TrainingStepResult(0.5f, 500, null, null);

        assertEquals(0.5f, result.loss(), 1e-7f);
        assertEquals(500, result.timestep());
        assertNull(result.predictedNoise());
        assertNull(result.noisyInput());
    }

    @Test(expected = NullPointerException.class)
    public void testDDPMSampleResultNullSamples() {
        new DDPMTrainer.SampleResult(null, 50);
    }

    @Test
    public void testDDIMSamplerCreation() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        DDIMSampler sampler = new DDIMSampler(predictor, scheduler, backend);

        assertNotNull(sampler);
        assertEquals(predictor, sampler.noisePredictor());
        assertEquals(scheduler, sampler.scheduler());
        assertEquals(backend, sampler.backend());
        assertEquals(0.0f, sampler.eta(), 1e-7f);
    }

    @Test
    public void testDDIMSamplerCustomEta() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        DDIMSampler sampler = new DDIMSampler(predictor, scheduler, backend, 0.5f);

        assertEquals(0.5f, sampler.eta(), 1e-7f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDDIMSamplerInvalidEtaNegative() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        new DDIMSampler(predictor, scheduler, backend, -0.5f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDDIMSamplerInvalidEtaTooLarge() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        new DDIMSampler(predictor, scheduler, backend, 1.5f);
    }

    @Test(expected = NullPointerException.class)
    public void testDDIMSamplerNullPredictor() {
        ComputeBackend backend = new MockBackend();
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        new DDIMSampler(null, scheduler, backend);
    }

    @Test
    public void testDDIMSamplerToString() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        DDIMSampler sampler = new DDIMSampler(predictor, scheduler, backend);

        assertTrue(sampler.toString().contains("DDIMSampler"));
        assertTrue(sampler.toString().contains("eta"));
    }

    @Test
    public void testDDIMSamplerClose() {
        ComputeBackend backend = new MockBackend();
        Module predictor = new MockNoisePredictor(backend);
        LinearScheduler scheduler = new LinearScheduler(backend, 1000);
        DDIMSampler sampler = new DDIMSampler(predictor, scheduler, backend);
        sampler.close();
    }

    @Test(expected = NullPointerException.class)
    public void testDDIMSamplingResultNullSamples() {
        new DDIMSampler.SamplingResult(null, 25, new ArrayList<>());
    }

    @Test
    public void testCFGWrapperCreation() {
        ComputeBackend backend = new MockBackend();
        Module model = new MockNoisePredictor(backend);
        CFGWrapper wrapper = new CFGWrapper(model, 7.5f);

        assertNotNull(wrapper);
        assertEquals(model, wrapper.conditionalModel());
        assertEquals(model, wrapper.unconditionalModel());
        assertEquals(7.5f, wrapper.guidanceScale(), 1e-7f);
        assertTrue(wrapper.isSharedModel());
    }

    @Test
    public void testCFGWrapperSeparateModels() {
        ComputeBackend backend = new MockBackend();
        Module conditional = new MockNoisePredictor(backend);
        Module unconditional = new MockNoisePredictor(backend);
        CFGWrapper wrapper = new CFGWrapper(conditional, unconditional, 7.5f);

        assertEquals(conditional, wrapper.conditionalModel());
        assertEquals(unconditional, wrapper.unconditionalModel());
        assertFalse(wrapper.isSharedModel());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCFGWrapperInvalidGuidanceScale() {
        ComputeBackend backend = new MockBackend();
        Module model = new MockNoisePredictor(backend);
        new CFGWrapper(model, 0.5f);
    }

    @Test(expected = NullPointerException.class)
    public void testCFGWrapperNullConditionalModel() {
        ComputeBackend backend = new MockBackend();
        Module model = new MockNoisePredictor(backend);
        new CFGWrapper(null, model, 7.5f);
    }

    @Test(expected = NullPointerException.class)
    public void testCFGWrapperNullUnconditionalModel() {
        ComputeBackend backend = new MockBackend();
        Module model = new MockNoisePredictor(backend);
        new CFGWrapper(model, null, 7.5f);
    }

    @Test
    public void testCFGWrapperToString() {
        ComputeBackend backend = new MockBackend();
        Module model = new MockNoisePredictor(backend);
        CFGWrapper wrapper = new CFGWrapper(model, 7.5f);

        assertTrue(wrapper.toString().contains("CFGWrapper"));
        assertTrue(wrapper.toString().contains("7.5"));
    }

    @Test
    public void testCFGWrapperClose() {
        ComputeBackend backend = new MockBackend();
        Module model = new MockNoisePredictor(backend);
        CFGWrapper wrapper = new CFGWrapper(model, 7.5f);
        wrapper.close();
    }

    @Test
    public void testLinearScheduleStatic() {
        float[] betas = NoiseScheduler.linearSchedule(1000, 0.0001f, 0.02f);

        assertEquals(1000, betas.length);
        assertEquals(0.0001f, betas[0], 1e-7f);
        assertEquals(0.02f, betas[999], 1e-7f);

        for (int i = 1; i < betas.length; i++) {
            assertTrue(betas[i] > betas[i - 1]);
        }
    }

    @Test
    public void testCosineScheduleStatic() {
        float[] betas = NoiseScheduler.cosineSchedule(1000, 0.008f);

        assertEquals(1000, betas.length);

        for (float beta : betas) {
            assertTrue(beta > 0 && beta < 1);
        }
    }

    @Test
    public void testSquaredCosineScheduleStatic() {
        float[] betas = NoiseScheduler.squaredCosineSchedule(1000);

        assertEquals(1000, betas.length);

        for (float beta : betas) {
            assertTrue(beta >= 0 && beta < 1);
        }
    }

    @Test(expected = NullPointerException.class)
    public void testLinearSchedulerNullBackend() {
        new LinearScheduler(null, 1000);
    }

    @Test(expected = NullPointerException.class)
    public void testCosineSchedulerNullBackend() {
        new CosineScheduler(null, 1000);
    }

    @Test(expected = NullPointerException.class)
    public void testSquaredCosineSchedulerNullBackend() {
        new SquaredCosineScheduler(null, 1000);
    }
}
