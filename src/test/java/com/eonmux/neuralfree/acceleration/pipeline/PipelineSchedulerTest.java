package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData.ActivationFunctionType;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainKernels;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.tensor.gpu.TrainParity.Api;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Enclosed.class)
public class PipelineSchedulerTest {

    /** Construction on a backend that declines every kernel artifact, and the backend-independent API. */
    public static class Unavailable {
        private ComputeBackend backend;

        @Before
        public void setUp() {
            backend = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            TrainGpu.release(backend);
            backend.close();
        }

        @Test
        public void constructorThrowsWithTheLibraryReason() {
            String reason = TrainKernels.availability(backend).reason();
            assertFalse("the declining backend has no train kernels", TrainKernels.availability(backend).available());

            try {
                new PipelineScheduler(backend, PipelineConfig.defaultConfig());
                fail("Expected UnsupportedKernelException");
            } catch (UnsupportedKernelException e) {
                assertNotNull(e.getMessage());
                assertTrue("message carries the reason '" + reason + "': " + e.getMessage(),
                        e.getMessage().contains(reason));
            }
        }

        @Test
        public void constructorThrowsForEveryConfig() {
            PipelineConfig noPrefetch = PipelineConfig.builder().enablePrefetch(false).build();
            try {
                new PipelineScheduler(backend, noPrefetch);
                fail("Expected UnsupportedKernelException");
            } catch (UnsupportedKernelException expected) {
                assertTrue(expected.getMessage().contains(TrainKernels.availability(backend).reason()));
            }
        }

        @Test(expected = NullPointerException.class)
        public void nullBackendRejected() {
            new PipelineScheduler(null, PipelineConfig.defaultConfig());
        }

        @Test(expected = NullPointerException.class)
        public void nullConfigRejected() {
            new PipelineScheduler(backend, null);
        }

        @Test
        public void trainingResultBuilderSetsEveryField() {
            TrainingResult result = TrainingResult.builder()
                    .forwardOutput(new float[]{1.0f, 2.0f})
                    .updatedWeights(new float[]{0.1f, 0.2f})
                    .updatedBiases(new float[]{0.5f})
                    .gradientWeights(new float[]{0.01f, 0.02f})
                    .gradientBiases(new float[]{0.05f})
                    .executionTimeNanos(1_000_000L)
                    .build();

            assertArrayEquals(new float[]{1.0f, 2.0f}, result.forwardOutput(), 0f);
            assertArrayEquals(new float[]{0.1f, 0.2f}, result.updatedWeights(), 0f);
            assertArrayEquals(new float[]{0.5f}, result.updatedBiases(), 0f);
            assertArrayEquals(new float[]{0.01f, 0.02f}, result.gradientWeights(), 0f);
            assertArrayEquals(new float[]{0.05f}, result.gradientBiases(), 0f);
            assertEquals(1_000_000L, result.executionTimeNanos());
            assertEquals(1.0, result.executionTimeMillis(), 0.001);
        }

        @Test
        public void trainingResultCopiesArraysInAndOut() {
            float[] output = {1.0f, 2.0f};
            TrainingResult result = TrainingResult.builder().forwardOutput(output).build();

            output[0] = 999.0f;
            assertEquals(1.0f, result.forwardOutput()[0], 0f);
            result.forwardOutput()[1] = 999.0f;
            assertEquals(2.0f, result.forwardOutput()[1], 0f);
        }

        @Test
        public void trainingResultEquality() {
            TrainingResult a = TrainingResult.builder().forwardOutput(new float[]{1f}).executionTimeNanos(5).build();
            TrainingResult b = TrainingResult.builder().forwardOutput(new float[]{1f}).executionTimeNanos(5).build();
            TrainingResult c = TrainingResult.builder().forwardOutput(new float[]{2f}).executionTimeNanos(5).build();

            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
            assertNotEquals(a, c);
        }
    }

    /** The scheduler on every GPU API with the train kernels; skipped where an API has no such device. */
    @RunWith(Parameterized.class)
    public static class OnBackend {
        private static final int[] LAYERS = {4, 6, 2};

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> apis() {
            return TrainParity.apis();
        }

        private final Api api;
        private TrainParity.Session session;
        private PipelineScheduler scheduler;

        public OnBackend(Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            scheduler = new PipelineScheduler(session.backend(), PipelineConfig.defaultConfig());
        }

        @After
        public void tearDown() {
            try {
                if (scheduler != null) {
                    scheduler.close();
                }
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void constructionKeepsTheConfig() {
            PipelineConfig config = PipelineConfig.defaultConfig();

            assertEquals(config.enablePrefetch(), scheduler.config().enablePrefetch());
            assertEquals(config.prefetchDepth(), scheduler.config().prefetchDepth());
            assertNotNull(scheduler.ringBufferManager());
            assertEquals(config.prefetchDepth(), scheduler.getCurrentPrefetchDepth());
            assertFalse(scheduler.isOomFallbackTriggered());
        }

        @Test
        public void constructionWithoutPrefetch() {
            PipelineConfig config = PipelineConfig.builder().enablePrefetch(false).build();
            try (PipelineScheduler noPrefetch = new PipelineScheduler(session.backend(), config)) {
                assertEquals(0, noPrefetch.getCurrentPrefetchDepth());
                assertNotNull(noPrefetch.ringBufferManager());
            }
        }

        @Test
        public void forwardPassReturnsOneOutputRowPerInputRow() {
            NeuralNetworkData network = new NeuralNetworkData(LAYERS);
            int rows = 3;

            float[] output = scheduler.executeForwardPass(network, TrainParity.uniform(1, rows * LAYERS[0], -1, 1));

            assertEquals(rows * LAYERS[2], output.length);
            for (float v : output) {
                assertTrue("finite TANH output in [-1, 1]: " + v, v >= -1f && v <= 1f);
            }
        }

        @Test
        public void trainingIterationReturnsRealGradients() {
            NeuralNetworkData network = new NeuralNetworkData(LAYERS);
            network.setLearningRate(0.5f);
            int rows = 2;
            float[] input = TrainParity.uniform(2, rows * LAYERS[0], -1, 1);
            float[] target = TrainParity.uniform(3, rows * LAYERS[2], -1, 1);
            float[] weights = network.flattenWeights(network.getWeights());
            float[] biases = network.flattenBiases(network.getBiases());

            TrainingResult result = scheduler.executeTrainingIteration(network, input, target);

            assertEquals(rows * LAYERS[2], result.forwardOutput().length);
            assertEquals(weights.length, result.updatedWeights().length);
            assertEquals(weights.length, result.gradientWeights().length);
            assertEquals(biases.length, result.updatedBiases().length);
            assertEquals(biases.length, result.gradientBiases().length);
            assertTrue(result.executionTimeNanos() > 0);
            assertTrue("weight gradients are not all zero", maxAbs(result.gradientWeights()) > 0f);
            assertTrue("bias gradients are not all zero", maxAbs(result.gradientBiases()) > 0f);
            assertSgdUpdate(weights, result.gradientWeights(), result.updatedWeights(), 0.5f);
            assertSgdUpdate(biases, result.gradientBiases(), result.updatedBiases(), 0.5f);
        }

        @Test
        public void backwardPassReturnsWeightGradients() {
            NeuralNetworkData network = new NeuralNetworkData(LAYERS);
            float[] input = TrainParity.uniform(4, LAYERS[0], -1, 1);
            float[] target = TrainParity.uniform(5, LAYERS[2], -1, 1);

            float[] forward = scheduler.executeForwardPass(network, input);
            float[] gradients = scheduler.executeBackwardPass(network, input, target, forward);

            assertEquals(network.flattenWeights(network.getWeights()).length, gradients.length);
            assertArrayEquals(gradients, scheduler.executeBackwardPass(network, input, target, null), 0f);
        }

        @Test
        public void workspaceFollowsTheBatchAndShape() {
            NeuralNetworkData small = new NeuralNetworkData(LAYERS);
            NeuralNetworkData deep = new NeuralNetworkData(new int[]{3, 5, 4, 2});

            assertEquals(LAYERS[2],
                    scheduler.executeForwardPass(small, TrainParity.uniform(6, LAYERS[0], -1, 1)).length);
            assertEquals(5 * LAYERS[2],
                    scheduler.executeForwardPass(small, TrainParity.uniform(7, 5 * LAYERS[0], -1, 1)).length);
            TrainingResult result = scheduler.executeTrainingIteration(deep, TrainParity.uniform(8, 6, -1, 1),
                    TrainParity.uniform(9, 4, -1, 1));
            assertEquals(3 * 5 + 5 * 4 + 4 * 2, result.gradientWeights().length);
            assertEquals(5 + 4 + 2, result.gradientBiases().length);
        }

        @Test(expected = IllegalArgumentException.class)
        public void partialInputRowRejected() {
            scheduler.executeForwardPass(new NeuralNetworkData(LAYERS), new float[LAYERS[0] + 1]);
        }

        @Test(expected = IllegalArgumentException.class)
        public void emptyInputRejected() {
            scheduler.executeForwardPass(new NeuralNetworkData(LAYERS), new float[0]);
        }

        @Test(expected = IllegalArgumentException.class)
        public void mismatchedTargetRejected() {
            NeuralNetworkData network = new NeuralNetworkData(LAYERS);
            scheduler.executeTrainingIteration(network, new float[2 * LAYERS[0]], new float[LAYERS[2]]);
        }

        @Test(expected = IllegalArgumentException.class)
        public void mismatchedForwardOutputRejected() {
            NeuralNetworkData network = new NeuralNetworkData(LAYERS);
            scheduler.executeBackwardPass(network, new float[LAYERS[0]], new float[LAYERS[2]], new float[1]);
        }

        @Test(expected = NullPointerException.class)
        public void nullInputRejected() {
            scheduler.executeForwardPass(new NeuralNetworkData(LAYERS), null);
        }

        @Test
        public void activationOrdinalsValidated() {
            for (ActivationFunctionType act : ActivationFunctionType.values()) {
                scheduler.validateActivationFunction(act.ordinal());
            }
            assertInvalidActivation(-1);
            assertInvalidActivation(ActivationFunctionType.values().length);
        }

        @Test
        public void networkCapabilitiesValidated() {
            NeuralNetworkData network = new NeuralNetworkData(LAYERS);
            network.setActivationTypes(new int[]{ActivationFunctionType.SINC.ordinal(),
                    ActivationFunctionType.IDENTITY.ordinal()});

            scheduler.validateNetworkCapabilities(network);
            scheduler.warnIfFeatureNotSupported("test feature", false);
        }

        @Test
        public void closeIsIdempotent() {
            scheduler.close();
            scheduler.close();
        }

        @Test(expected = IllegalStateException.class)
        public void forwardAfterCloseThrows() {
            scheduler.close();
            scheduler.executeForwardPass(new NeuralNetworkData(LAYERS), new float[LAYERS[0]]);
        }

        @Test(expected = IllegalStateException.class)
        public void trainingAfterCloseThrows() {
            scheduler.close();
            scheduler.executeTrainingIteration(new NeuralNetworkData(LAYERS), new float[LAYERS[0]],
                    new float[LAYERS[2]]);
        }

        @Test
        public void closingLeavesTheBackendUsable() {
            NeuralNetworkData network = new NeuralNetworkData(LAYERS);
            float[] input = TrainParity.uniform(10, LAYERS[0], -1, 1);
            float[] before = scheduler.executeForwardPass(network, input);
            scheduler.close();

            try (PipelineScheduler second = new PipelineScheduler(session.backend(), PipelineConfig.defaultConfig())) {
                assertArrayEquals(before, second.executeForwardPass(network, input), 0f);
            }
        }

        private void assertInvalidActivation(int ordinal) {
            try {
                scheduler.validateActivationFunction(ordinal);
                fail("Expected IllegalArgumentException for activation " + ordinal);
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains(String.valueOf(ordinal)));
            }
        }

        private static void assertSgdUpdate(float[] before, float[] gradients, float[] after, float lr) {
            for (int i = 0; i < before.length; i++) {
                float expected = before[i] - lr * gradients[i];
                assertEquals("parameter " + i, expected, after[i], 1e-6f * Math.max(1f, Math.abs(expected)));
            }
        }

        private static float maxAbs(float[] values) {
            float max = 0f;
            for (float v : values) {
                max = Math.max(max, Math.abs(v));
            }
            return max;
        }
    }
}
