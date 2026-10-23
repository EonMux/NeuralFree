package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData.ActivationFunctionType;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.tensor.gpu.TrainParity.Api;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * The GPU pipeline against {@link DenseReference}, the CPU definition of a dense pass, on every GPU API with the
 * train kernels: forward outputs for every activation, gradients, and SGD-updated parameters.
 */
@RunWith(Parameterized.class)
public class PipelineCorrectnessTest {

    /**
     * Error bound relative to {@code max(1, max |ref|)} of the compared array: the GEMMs accumulate with fma where the
     * reference rounds each product, and the activations are in the transcendental tolerance class.
     */
    private static final float TOLERANCE = 1e-4f;
    /** Three layers, for the multi-layer and repeated-step checks. */
    private static final int[] LAYERS = {5, 7, 6, 3};
    /**
     * Two layers, for the sweeps over every activation: with inputs in [-1, 1], weights in [-0.3, 0.3] and biases in
     * [-0.1, 0.1], the pre-activations stay below 11 in magnitude even through SOFTEXPONENTIAL, so every reference
     * value is finite.
     */
    private static final int[] SWEEP = {5, 7, 3};
    private static final int ROWS = 4;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> apis() {
        return TrainParity.apis();
    }

    private final Api api;
    private TrainParity.Session session;
    private PipelineScheduler scheduler;

    public PipelineCorrectnessTest(Api api) {
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
    public void forwardMatchesReferenceForEveryActivation() {
        for (ActivationFunctionType act : ActivationFunctionType.values()) {
            NeuralNetworkData network = network(SWEEP, 11 + act.ordinal(), act, 0.3f, 0.1f);
            float[] input = TrainParity.uniform(101 + act.ordinal(), ROWS * SWEEP[0], -1f, 1f);

            float[] gpu = scheduler.executeForwardPass(network, input);
            float[] ref = DenseReference.of(network).forward(input, ROWS);

            assertClose(api + " forward " + act, ref, gpu);
        }
    }

    @Test
    public void forwardMatchesReferenceWithMixedActivations() {
        NeuralNetworkData network = network(LAYERS, 3, ActivationFunctionType.TANH, 0.6f, 0.3f);
        network.setActivationTypes(new int[]{ActivationFunctionType.RELU.ordinal(),
                ActivationFunctionType.SWISH.ordinal(), ActivationFunctionType.SIGMOID.ordinal()});
        float[] input = TrainParity.uniform(7, ROWS * LAYERS[0], -1.5f, 1.5f);

        assertClose(api + " mixed forward", DenseReference.of(network).forward(input, ROWS),
                scheduler.executeForwardPass(network, input));
    }

    @Test
    public void singleRowForwardMatchesCpuPredict() {
        NeuralNetworkData network = network(LAYERS, 5, ActivationFunctionType.ELU, 0.6f, 0.3f);
        float[] input = TrainParity.uniform(9, LAYERS[0], -1f, 1f);

        assertClose(api + " single row", network.predict(network, input),
                scheduler.executeForwardPass(network, input));
    }

    @Test
    public void trainingIterationMatchesReferenceForEveryActivation() {
        for (ActivationFunctionType act : ActivationFunctionType.values()) {
            NeuralNetworkData network = network(SWEEP, 31 + act.ordinal(), act, 0.3f, 0.1f);
            network.setLearningRate(0.1f);
            float[] input = TrainParity.uniform(201 + act.ordinal(), ROWS * SWEEP[0], -1f, 1f);
            float[] target = TrainParity.uniform(301 + act.ordinal(), ROWS * SWEEP[2], -1f, 1f);

            DenseReference ref = DenseReference.of(network);
            DenseReference.Step expected = ref.step(input, target, ROWS, network.getLearningRate());
            TrainingResult result = scheduler.executeTrainingIteration(network, input, target);

            String what = api + " " + act;
            assertClose(what + " output", expected.output, result.forwardOutput());
            assertClose(what + " weight gradients", expected.gradientWeights, result.gradientWeights());
            assertClose(what + " bias gradients", expected.gradientBiases, result.gradientBiases());
            assertClose(what + " updated weights", expected.updatedWeights, result.updatedWeights());
            assertClose(what + " updated biases", expected.updatedBiases, result.updatedBiases());
        }
    }

    @Test
    public void backwardPassReturnsTheTrainingGradients() {
        NeuralNetworkData network = network(LAYERS, 41, ActivationFunctionType.TANH, 0.6f, 0.3f);
        float[] input = TrainParity.uniform(43, ROWS * LAYERS[0], -1f, 1f);
        float[] target = TrainParity.uniform(47, ROWS * LAYERS[3], -1f, 1f);

        float[] forward = scheduler.executeForwardPass(network, input);
        float[] gradients = scheduler.executeBackwardPass(network, input, target, forward);
        DenseReference.Step expected = DenseReference.of(network).step(input, target, ROWS, 0f);

        assertClose(api + " backward", expected.gradientWeights, gradients);
        assertArrayEquals(api + " the same pass gives the same gradients",
                scheduler.executeTrainingIteration(network, input, target).gradientWeights(), gradients, 0f);
    }

    @Test
    public void trainingDoesNotModifyTheNetwork() {
        NeuralNetworkData network = network(LAYERS, 51, ActivationFunctionType.SIGMOID, 0.6f, 0.3f);
        float[] weights = network.flattenWeights(network.getWeights());
        float[] biases = network.flattenBiases(network.getBiases());
        float[] input = TrainParity.uniform(53, ROWS * LAYERS[0], -1f, 1f);
        float[] target = TrainParity.uniform(59, ROWS * LAYERS[3], 0f, 1f);

        scheduler.executeTrainingIteration(network, input, target);

        assertArrayEquals(weights, network.flattenWeights(network.getWeights()), 0f);
        assertArrayEquals(biases, network.flattenBiases(network.getBiases()), 0f);
    }

    @Test
    public void repeatedStepsTrackTheReference() {
        NeuralNetworkData network = network(LAYERS, 61, ActivationFunctionType.TANH, 0.6f, 0.3f);
        network.setLearningRate(0.2f);
        float[] input = TrainParity.uniform(67, ROWS * LAYERS[0], -1f, 1f);
        float[] target = TrainParity.uniform(71, ROWS * LAYERS[3], -0.8f, 0.8f);
        NeuralNetworkData reference = network(LAYERS, 61, ActivationFunctionType.TANH, 0.6f, 0.3f);

        for (int step = 0; step < 20; step++) {
            TrainingResult result = scheduler.executeTrainingIteration(network, input, target);
            network.setParameters(result.updatedWeights(), result.updatedBiases());
            DenseReference.Step expected = DenseReference.of(reference).step(input, target, ROWS, 0.2f);
            reference.setParameters(expected.updatedWeights, expected.updatedBiases);
        }

        assertClose(api + " weights after 20 steps", reference.flattenWeights(reference.getWeights()),
                network.flattenWeights(network.getWeights()));
        assertClose(api + " biases after 20 steps", reference.flattenBiases(reference.getBiases()),
                network.flattenBiases(network.getBiases()));
    }

    @Test
    public void referenceAgreesWithCpuPredict() {
        NeuralNetworkData network = network(LAYERS, 81, ActivationFunctionType.GAUSSIAN, 0.6f, 0.3f);
        float[] row = TrainParity.uniform(83, LAYERS[0], -1f, 1f);

        assertArrayEquals(network.predict(network, row), DenseReference.of(network).forward(row, 1), 0f);
    }

    /**
     * A network of {@code sizes} with every layer on {@code act}, weights uniform in {@code ±weightRange} and biases,
     * non-zero, uniform in {@code ±biasRange}.
     */
    private static NeuralNetworkData network(int[] sizes, long seed, ActivationFunctionType act, float weightRange,
                                             float biasRange) {
        NeuralNetworkData network = new NeuralNetworkData(sizes);
        int params = 0;
        int biases = 0;
        for (int l = 0; l < sizes.length - 1; l++) {
            params += sizes[l] * sizes[l + 1];
            biases += sizes[l + 1];
        }
        network.setParameters(TrainParity.uniform(seed, params, -weightRange, weightRange),
                TrainParity.uniform(seed + 1000, biases, -biasRange, biasRange));
        int[] acts = new int[sizes.length - 1];
        Arrays.fill(acts, act.ordinal());
        network.setActivationTypes(acts);
        return network;
    }

    static void assertClose(String what, float[] ref, float[] gpu) {
        assertEquals(what + " length", ref.length, gpu.length);
        float scale = 1f;
        for (float v : ref) {
            scale = Math.max(scale, Math.abs(v));
        }
        float bound = TOLERANCE * scale;
        for (int i = 0; i < ref.length; i++) {
            if (!(Math.abs(ref[i] - gpu[i]) <= bound)) {
                fail(what + "[" + i + "]: reference " + ref[i] + ", GPU " + gpu[i] + " (bound " + bound + ")");
            }
        }
    }

    /**
     * The CPU definition of the pipeline's dense passes, in float arithmetic: per layer
     * {@code z = sum_k x_k * W[o, k]} (ascending k), {@code z + b}, {@code a = act(z)}; the loss
     * {@code sum((a - t)^2) / (2 * rows)} with output gradient {@code s * a + (-s) * t}, {@code s = 1f / rows};
     * {@code dZ = dY * act'(z, a)}; {@code dW[o, k] = sum_r dZ[r, o] * x[r, k]}, {@code db[o] = sum_r dZ[r, o]} and
     * {@code dY[r, k] = sum_o dZ[r, o] * W[o, k]}, all in ascending order; {@code p - lr * g}. Weights are the
     * row-major {@code [out, in]} matrices of {@link NeuralNetworkData}.
     */
    public static final class DenseReference {
        private final int[] sizes;
        private final ActivationFunctionType[] acts;
        private final float[][] weights;
        private final float[][] biases;

        private DenseReference(NeuralNetworkData network) {
            this.sizes = network.getLayerSizes();
            int[] ordinals = network.getActivationTypes();
            this.acts = new ActivationFunctionType[ordinals.length];
            for (int l = 0; l < ordinals.length; l++) {
                acts[l] = ActivationFunctionType.fromOrdinal(ordinals[l]);
            }
            this.weights = new float[ordinals.length][];
            this.biases = new float[ordinals.length][];
            for (int l = 0; l < ordinals.length; l++) {
                weights[l] = network.getWeights()[l].clone();
                biases[l] = network.getBiases()[l].clone();
            }
        }

        /** A reference of the network's current parameters and activations. */
        public static DenseReference of(NeuralNetworkData network) {
            return new DenseReference(network);
        }

        /** The results of one SGD step, flattened as {@link TrainingResult} holds them. */
        public static final class Step {
            public final float[] output;
            public final float[] gradientWeights;
            public final float[] gradientBiases;
            public final float[] updatedWeights;
            public final float[] updatedBiases;

            Step(float[] output, float[] gradientWeights, float[] gradientBiases, float[] updatedWeights,
                 float[] updatedBiases) {
                this.output = output;
                this.gradientWeights = gradientWeights;
                this.gradientBiases = gradientBiases;
                this.updatedWeights = updatedWeights;
                this.updatedBiases = updatedBiases;
            }
        }

        /** The output rows of {@code rows} input rows. */
        public float[] forward(float[] input, int rows) {
            float[][][] za = forwardAll(input, rows);
            return za[1][za[1].length - 1];
        }

        /** One SGD step with learning rate {@code lr} ({@code 0} gives the gradients and unchanged parameters). */
        public Step step(float[] input, float[] target, int rows, float lr) {
            int layers = weights.length;
            float[][][] za = forwardAll(input, rows);
            float[][] z = za[0];
            float[][] a = za[1];
            float[] output = a[layers - 1];
            if (target.length != output.length) {
                throw new IllegalArgumentException("target holds " + target.length + " values, expected "
                        + output.length);
            }

            float s = 1.0f / rows;
            float[] dy = new float[output.length];
            for (int i = 0; i < dy.length; i++) {
                float sa = s * output[i];
                float st = -s * target[i];
                dy[i] = sa + st;
            }

            float[][] gw = new float[layers][];
            float[][] gb = new float[layers][];
            for (int l = layers - 1; l >= 0; l--) {
                int in = sizes[l];
                int out = sizes[l + 1];
                float[] x = l == 0 ? input : a[l - 1];
                float[] dz = new float[dy.length];
                for (int i = 0; i < dz.length; i++) {
                    dz[i] = dy[i] * acts[l].derivative(z[l][i], a[l][i]);
                }
                gw[l] = new float[out * in];
                for (int o = 0; o < out; o++) {
                    for (int k = 0; k < in; k++) {
                        float sum = 0f;
                        for (int r = 0; r < rows; r++) {
                            sum += dz[r * out + o] * x[r * in + k];
                        }
                        gw[l][o * in + k] = sum;
                    }
                }
                gb[l] = new float[out];
                for (int o = 0; o < out; o++) {
                    float sum = 0f;
                    for (int r = 0; r < rows; r++) {
                        sum += dz[r * out + o];
                    }
                    gb[l][o] = sum;
                }
                if (l > 0) {
                    float[] prev = new float[rows * in];
                    for (int r = 0; r < rows; r++) {
                        for (int k = 0; k < in; k++) {
                            float sum = 0f;
                            for (int o = 0; o < out; o++) {
                                sum += dz[r * out + o] * weights[l][o * in + k];
                            }
                            prev[r * in + k] = sum;
                        }
                    }
                    dy = prev;
                }
            }

            float[] gradW = concat(gw);
            float[] gradB = concat(gb);
            float[] newW = concat(weights);
            float[] newB = concat(biases);
            for (int i = 0; i < newW.length; i++) {
                newW[i] = newW[i] - lr * gradW[i];
            }
            for (int i = 0; i < newB.length; i++) {
                newB[i] = newB[i] - lr * gradB[i];
            }
            return new Step(output, gradW, gradB, newW, newB);
        }

        /** {@code {z, a}}: per layer the pre-activations and activations of every row. */
        private float[][][] forwardAll(float[] input, int rows) {
            int layers = weights.length;
            if (rows < 1 || input.length != rows * sizes[0]) {
                throw new IllegalArgumentException("input holds " + input.length + " values, not " + rows
                        + " rows of width " + sizes[0]);
            }
            float[][] z = new float[layers][];
            float[][] a = new float[layers][];
            float[] x = input;
            for (int l = 0; l < layers; l++) {
                int in = sizes[l];
                int out = sizes[l + 1];
                z[l] = new float[rows * out];
                a[l] = new float[rows * out];
                for (int r = 0; r < rows; r++) {
                    for (int o = 0; o < out; o++) {
                        float sum = 0f;
                        for (int k = 0; k < in; k++) {
                            sum += x[r * in + k] * weights[l][o * in + k];
                        }
                        float v = sum + biases[l][o];
                        z[l][r * out + o] = v;
                        a[l][r * out + o] = acts[l].activate(v);
                    }
                }
                x = a[l];
            }
            return new float[][][]{z, a};
        }

        private static float[] concat(float[][] parts) {
            int n = 0;
            for (float[] p : parts) {
                n += p.length;
            }
            float[] out = new float[n];
            int i = 0;
            for (float[] p : parts) {
                System.arraycopy(p, 0, out, i, p.length);
                i += p.length;
            }
            return out;
        }
    }
}
