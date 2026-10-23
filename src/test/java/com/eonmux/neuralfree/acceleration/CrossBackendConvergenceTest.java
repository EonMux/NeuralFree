package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.pipeline.PipelineCorrectnessTest.DenseReference;
import com.eonmux.neuralfree.acceleration.pipeline.PipelineScheduler;
import com.eonmux.neuralfree.acceleration.pipeline.TrainingResult;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.tensor.gpu.TrainParity.Api;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Trains a {1, 16, 1} linear network on {@code y = 2x} with the pipeline on every GPU API with the train kernels
 * (OpenCL, CUDA, Vulkan; each skipped where absent) and checks that the loss converges and that the weights stay
 * with the CPU reference, the common anchor that makes the backends agree with each other.
 */
@RunWith(Parameterized.class)
public class CrossBackendConvergenceTest {

    private static final int[] LAYER_SIZES = {1, 16, 1};
    private static final int SAMPLES = 100;
    private static final int EPOCHS = 300;
    private static final float LEARNING_RATE = 0.2f;
    private static final long SEED = 42;
    private static final float FINAL_LOSS_BOUND = 1e-3f;
    /** Bound on the GPU-to-reference parameter difference, relative to {@code max(1, max |ref|)}. */
    private static final float WEIGHT_TOLERANCE = 1e-3f;
    private static final float PREDICTION_TOLERANCE = 0.05f;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> apis() {
        return TrainParity.apis();
    }

    private final Api api;
    private TrainParity.Session session;
    private PipelineScheduler scheduler;

    public CrossBackendConvergenceTest(Api api) {
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
    public void fullBatchTrainingConvergesWithTheReference() {
        float[] inputs = inputs();
        float[] targets = targets(inputs);
        NeuralNetworkData network = createNetwork();
        NeuralNetworkData reference = createNetwork();
        float[] losses = new float[EPOCHS];

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            TrainingResult result = scheduler.executeTrainingIteration(network, inputs, targets);
            losses[epoch] = loss(result.forwardOutput(), targets);
            network.setParameters(result.updatedWeights(), result.updatedBiases());

            DenseReference.Step expected = DenseReference.of(reference).step(inputs, targets, SAMPLES,
                    LEARNING_RATE);
            reference.setParameters(expected.updatedWeights, expected.updatedBiases);
        }

        float initial = losses[0];
        float last = losses[EPOCHS - 1];
        assertTrue(api + " loss decreases: " + initial + " -> " + last, last < initial);
        assertTrue(api + " final loss " + last + " below " + FINAL_LOSS_BOUND, last < FINAL_LOSS_BOUND);
        assertTrue(api + " loss is finite throughout: " + Arrays.toString(losses), allFinite(losses));
        assertParametersMatch(network, reference);
        assertPredictsDouble(network);
    }

    @Test
    public void miniBatchTrainingConvergesWithTheReference() {
        float[] inputs = inputs();
        float[] targets = targets(inputs);
        NeuralNetworkData network = createNetwork();
        NeuralNetworkData reference = createNetwork();
        int batch = 25;

        for (int epoch = 0; epoch < EPOCHS / 2; epoch++) {
            for (int start = 0; start < SAMPLES; start += batch) {
                float[] x = Arrays.copyOfRange(inputs, start, start + batch);
                float[] t = Arrays.copyOfRange(targets, start, start + batch);

                TrainingResult result = scheduler.executeTrainingIteration(network, x, t);
                network.setParameters(result.updatedWeights(), result.updatedBiases());
                DenseReference.Step expected = DenseReference.of(reference).step(x, t, batch, LEARNING_RATE);
                reference.setParameters(expected.updatedWeights, expected.updatedBiases);
            }
        }

        assertTrue(api + " final loss", loss(scheduler.executeForwardPass(network, inputs), targets)
                < FINAL_LOSS_BOUND);
        assertParametersMatch(network, reference);
        assertPredictsDouble(network);
    }

    /** A seeded {@link #LAYER_SIZES} identity network, so every run and the reference start alike. */
    private static NeuralNetworkData createNetwork() {
        NeuralNetworkData network = new NeuralNetworkData(LAYER_SIZES);
        int weights = LAYER_SIZES[0] * LAYER_SIZES[1] + LAYER_SIZES[1] * LAYER_SIZES[2];
        int biases = LAYER_SIZES[1] + LAYER_SIZES[2];
        network.setParameters(TrainParity.uniform(SEED, weights, -0.5f, 0.5f), new float[biases]);
        network.setLearningRate(LEARNING_RATE);
        network.setActivationTypes(new int[]{NeuralNetworkData.ActivationFunctionType.IDENTITY.ordinal(),
                NeuralNetworkData.ActivationFunctionType.IDENTITY.ordinal()});
        return network;
    }

    /** {@link #SAMPLES} inputs evenly spaced over [-1, 1]. */
    private static float[] inputs() {
        float[] x = new float[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            x[i] = -1.0f + 2.0f * i / (SAMPLES - 1);
        }
        return x;
    }

    private static float[] targets(float[] inputs) {
        float[] y = new float[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            y[i] = 2.0f * inputs[i];
        }
        return y;
    }

    /** The pipeline's loss {@code sum((a - t)^2) / (2 * rows)}, in double. */
    private static float loss(float[] outputs, float[] targets) {
        double sum = 0;
        for (int i = 0; i < outputs.length; i++) {
            double d = (double) outputs[i] - targets[i];
            sum += d * d;
        }
        return (float) (sum / (2.0 * outputs.length));
    }

    private void assertParametersMatch(NeuralNetworkData network, NeuralNetworkData reference) {
        assertClose(api + " weights", reference.flattenWeights(reference.getWeights()),
                network.flattenWeights(network.getWeights()));
        assertClose(api + " biases", reference.flattenBiases(reference.getBiases()),
                network.flattenBiases(network.getBiases()));
    }

    private void assertPredictsDouble(NeuralNetworkData network) {
        for (float x : new float[]{-0.75f, 0.0f, 0.5f, 1.0f}) {
            float prediction = network.predict(network, new float[]{x})[0];
            assertEquals(api + " prediction for " + x, 2.0f * x, prediction, PREDICTION_TOLERANCE);
        }
    }

    private static void assertClose(String what, float[] ref, float[] gpu) {
        assertEquals(what + " length", ref.length, gpu.length);
        float scale = 1f;
        for (float v : ref) {
            scale = Math.max(scale, Math.abs(v));
        }
        float bound = WEIGHT_TOLERANCE * scale;
        for (int i = 0; i < ref.length; i++) {
            if (!(Math.abs(ref[i] - gpu[i]) <= bound)) {
                fail(what + "[" + i + "]: reference " + ref[i] + ", GPU " + gpu[i] + " (bound " + bound + ")");
            }
        }
    }

    private static boolean allFinite(float[] values) {
        for (float v : values) {
            if (!Float.isFinite(v)) {
                return false;
            }
        }
        return true;
    }
}
