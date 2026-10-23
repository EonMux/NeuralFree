package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.pipeline.PipelineCorrectnessTest.DenseReference;
import com.eonmux.neuralfree.acceleration.pipeline.PipelineScheduler;
import com.eonmux.neuralfree.acceleration.pipeline.TrainingResult;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.ai.v2.learning.BackPropagate;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.tensor.gpu.TrainParity.Api;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

/**
 * End to end: a single linear neuron learns {@code y = 2x} on every GPU API with the train kernels, through the
 * pipeline scheduler and through {@link BackPropagate}, and tracks the CPU reference step for step.
 */
@RunWith(Parameterized.class)
public class MultiplyByTwoE2ETest {

    private static final int TRAINING_ITERATIONS = 1000;
    private static final float LEARNING_RATE = 0.05f;
    private static final int SEED = 42;
    /** Convergence bound on the learned weight, bias and predictions; the data is fit exactly by w = 2, b = 0. */
    private static final float CONVERGENCE_TOLERANCE = 1e-3f;
    /**
     * Bound on the GPU-to-reference difference, relative to {@code max(1, |ref|)}: a one-input neuron rounds each
     * product and sum exactly as the reference does, so only a contracted multiply-add in the update can differ.
     */
    private static final float REFERENCE_TOLERANCE = 1e-5f;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> apis() {
        return TrainParity.apis();
    }

    private final Api api;
    private TrainParity.Session session;

    public MultiplyByTwoE2ETest(Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        session = TrainParity.open(api);
    }

    @After
    public void tearDown() {
        if (session != null) {
            session.close();
        }
    }

    @Test
    public void perSampleTrainingTracksTheReferenceAndConverges() {
        NeuralNetworkData gpuNetwork = createMultiplyByTwoNetwork(SEED);
        NeuralNetworkData cpuNetwork = createMultiplyByTwoNetwork(SEED);
        float[] inputs = keys(gpuNetwork.getTrainingMap());
        float[] targets = values(gpuNetwork.getTrainingMap());
        Random random = new Random(SEED + 1000);

        try (PipelineScheduler scheduler = new PipelineScheduler(session.backend(), PipelineConfig.defaultConfig())) {
            for (int iter = 0; iter < TRAINING_ITERATIONS; iter++) {
                int idx = random.nextInt(inputs.length);
                float[] x = {inputs[idx]};
                float[] t = {targets[idx]};

                TrainingResult result = scheduler.executeTrainingIteration(gpuNetwork, x, t);
                gpuNetwork.setParameters(result.updatedWeights(), result.updatedBiases());
                DenseReference.Step expected = DenseReference.of(cpuNetwork).step(x, t, 1, LEARNING_RATE);
                cpuNetwork.setParameters(expected.updatedWeights, expected.updatedBiases);
            }
        }

        float gpuWeight = gpuNetwork.getWeights()[0][0];
        float gpuBias = gpuNetwork.getBiases()[0][0];
        float cpuWeight = cpuNetwork.getWeights()[0][0];
        float cpuBias = cpuNetwork.getBiases()[0][0];
        assertEquals(api + " weight vs reference", cpuWeight, gpuWeight, REFERENCE_TOLERANCE * Math.max(1f,
                Math.abs(cpuWeight)));
        assertEquals(api + " bias vs reference", cpuBias, gpuBias, REFERENCE_TOLERANCE);
        assertConverged(api + " per-sample", gpuNetwork);
        assertConverged("reference per-sample", cpuNetwork);
    }

    @Test
    public void fullBatchStepMatchesTheReference() {
        NeuralNetworkData network = createMultiplyByTwoNetwork(SEED);
        float[] inputs = keys(network.getTrainingMap());
        float[] targets = values(network.getTrainingMap());

        DenseReference.Step expected = DenseReference.of(network).step(inputs, targets, inputs.length,
                LEARNING_RATE);
        TrainingResult result;
        try (PipelineScheduler scheduler = new PipelineScheduler(session.backend(), PipelineConfig.defaultConfig())) {
            result = scheduler.executeTrainingIteration(network, inputs, targets);
        }

        assertClose(api + " output", expected.output, result.forwardOutput());
        assertClose(api + " weight gradient", expected.gradientWeights, result.gradientWeights());
        assertClose(api + " bias gradient", expected.gradientBiases, result.gradientBiases());
        assertClose(api + " updated weight", expected.updatedWeights, result.updatedWeights());
        assertClose(api + " updated bias", expected.updatedBiases, result.updatedBiases());
    }

    @Test
    public void backPropagateLearnsMultiplyByTwo() {
        NeuralNetworkData network = createMultiplyByTwoNetwork(SEED);
        network.setBatchSize(network.getTrainingMap().size());

        try (BackPropagate backPropagate = new BackPropagate(network, session.backend())) {
            assertNotNull("construction runs the first step", backPropagate.getLastResult());
            for (int step = 1; step < TRAINING_ITERATIONS / 2; step++) {
                backPropagate.step();
            }
            backPropagate.close();
            assertNull(backPropagate.getPipelineScheduler());
        }

        assertConverged(api + " BackPropagate", network);
    }

    private NeuralNetworkData createMultiplyByTwoNetwork(long seed) {
        NeuralNetworkData network = new NeuralNetworkData(new int[]{1, 1});
        Random random = new Random(seed);
        network.setParameters(new float[]{(random.nextFloat() - 0.5f) * 0.5f}, new float[]{0.0f});
        network.setLearningRate(LEARNING_RATE);
        network.setBatchSize(1);
        network.setTileSize(1);
        network.setActivationTypes(new int[]{NeuralNetworkData.ActivationFunctionType.IDENTITY.ordinal()});

        Map<Float, Float> trainingMap = new LinkedHashMap<>();
        for (int i = -5; i <= 5; i++) {
            trainingMap.put((float) i, 2.0f * i);
        }
        network.setTrainingMap(trainingMap);
        return network;
    }

    private static void assertConverged(String what, NeuralNetworkData network) {
        float weight = network.getWeights()[0][0];
        float bias = network.getBiases()[0][0];
        assertEquals(what + " weight", 2.0f, weight, CONVERGENCE_TOLERANCE);
        assertEquals(what + " bias", 0.0f, bias, CONVERGENCE_TOLERANCE);
        for (float x : new float[]{1.0f, 3.0f, -2.0f}) {
            float prediction = network.predict(network, new float[]{x})[0];
            assertEquals(what + " prediction for " + x, 2.0f * x, prediction, 10 * CONVERGENCE_TOLERANCE);
        }
    }

    private static void assertClose(String what, float[] ref, float[] gpu) {
        assertEquals(what + " length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            assertEquals(what + "[" + i + "]", ref[i], gpu[i], 1e-5f * Math.max(1f, Math.abs(ref[i])));
        }
    }

    private static float[] keys(Map<Float, Float> map) {
        float[] out = new float[map.size()];
        int i = 0;
        for (Float k : map.keySet()) {
            out[i++] = k;
        }
        return out;
    }

    private static float[] values(Map<Float, Float> map) {
        float[] out = new float[map.size()];
        int i = 0;
        for (Float v : map.values()) {
            out[i++] = v;
        }
        return out;
    }
}
