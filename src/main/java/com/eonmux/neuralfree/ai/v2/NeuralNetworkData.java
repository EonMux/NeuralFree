package com.eonmux.neuralfree.ai.v2;

import com.eonmux.neuralfree.utils.NeuralUtil;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * A fully connected network: per layer {@code l} a weight matrix {@code [layerSizes[l + 1], layerSizes[l]]} stored
 * row-major (element {@code (o, i)} at {@code o * layerSizes[l] + i}), a bias vector of {@code layerSizes[l + 1]}
 * elements and an {@link ActivationFunctionType} ordinal. Layer {@code l} computes
 * {@code a = act(x * W^T + b)} for each input row {@code x}. {@link #getWeights()} and {@link #getBiases()} return
 * the live arrays, which training code updates in place ({@link #setParameters}).
 */
public class NeuralNetworkData {

    /** Rows of the default training map that one batch holds at most. */
    private static final int DEFAULT_BATCH_SIZE = 64;
    private static final int DEFAULT_TILE_SIZE = 32;
    private static final float PI = 3.14159265358979323846f;

    /**
     * Activation functions of the dense layers. Each constant is the CPU reference of the device kernels
     * {@code TR_DenseBiasAct_F32} and {@code TR_DenseActBackward_F32} ({@code kernels/train/dense.cl}): the same
     * formulas, constants and float rounding steps, with Java {@code Math.max}/{@code Math.min} semantics. The
     * ordinal is the kernel's activation id.
     */
    public enum ActivationFunctionType {
        /** {@code 1 / (1 + exp(-x))}; derivative {@code a * (1 - a)}. */
        SIGMOID {
            @Override
            public float activate(float x) {
                return sigmoid(x);
            }

            @Override
            public float derivative(float x, float a) {
                return a * (1.0f - a);
            }
        },
        /** {@code Math.max(0, x)}; derivative {@code x > 0 ? 1 : 0}. */
        RELU {
            @Override
            public float activate(float x) {
                return Math.max(0.0f, x);
            }

            @Override
            public float derivative(float x, float a) {
                return x > 0.0f ? 1.0f : 0.0f;
            }
        },
        /** {@code tanh(x)}; derivative {@code 1 - a * a}. */
        TANH {
            @Override
            public float activate(float x) {
                return (float) Math.tanh(x);
            }

            @Override
            public float derivative(float x, float a) {
                return 1.0f - a * a;
            }
        },
        /** {@code Math.max(0.01 * x, x)}; derivative {@code x > 0 ? 1 : 0.01}. */
        LEAKY_RELU {
            @Override
            public float activate(float x) {
                return Math.max(0.01f * x, x);
            }

            @Override
            public float derivative(float x, float a) {
                return x > 0.0f ? 1.0f : 0.01f;
            }
        },
        /** {@code x >= 0 ? x : exp(x) - 1}; derivative {@code x >= 0 ? 1 : exp(x)}. */
        ELU {
            @Override
            public float activate(float x) {
                return x >= 0.0f ? x : 1.0f * (exp(x) - 1.0f);
            }

            @Override
            public float derivative(float x, float a) {
                return x >= 0.0f ? 1.0f : 1.0f * exp(x);
            }
        },
        /** {@code Math.max(x, 0) + log(1 + exp(-|x|))}; derivative {@code 1 / (1 + exp(-x))}. */
        SOFTPLUS {
            @Override
            public float activate(float x) {
                return Math.max(x, 0.0f) + (float) Math.log(1.0f + exp(-Math.abs(x)));
            }

            @Override
            public float derivative(float x, float a) {
                return sigmoid(x);
            }
        },
        /** {@code x / (1 + |x|)}; derivative {@code 1 / (1 + |x|)^2}. */
        SOFTSIGN {
            @Override
            public float activate(float x) {
                return x / (1.0f + Math.abs(x));
            }

            @Override
            public float derivative(float x, float a) {
                float d = 1.0f + Math.abs(x);
                return 1.0f / (d * d);
            }
        },
        /** {@code x / sqrt(1 + x * x)}; derivative {@code 1 / sqrt(1 + x * x)^3}. */
        ISRU {
            @Override
            public float activate(float x) {
                return x / sqrt1pSquare(x);
            }

            @Override
            public float derivative(float x, float a) {
                float s = sqrt1pSquare(x);
                return 1.0f / (s * s * s);
            }
        },
        /** {@code exp(-(x * x))}; derivative {@code -2 * x * exp(-(x * x))}. */
        GAUSSIAN {
            @Override
            public float activate(float x) {
                return exp(-(x * x));
            }

            @Override
            public float derivative(float x, float a) {
                return -2.0f * x * exp(-(x * x));
            }
        },
        /** {@code sin(pi x) / (pi x)}, with Taylor series for {@code |pi x| < 1}; see {@link #sinc}. */
        SINC {
            @Override
            public float activate(float x) {
                return sinc(x);
            }

            @Override
            public float derivative(float x, float a) {
                return sincDerivative(x);
            }
        },
        /** {@code x / (1 + exp(-x))}; derivative {@code s * (1 + x * (1 - s))}, {@code s = sigmoid(x)}. */
        SWISH {
            @Override
            public float activate(float x) {
                return x / (1.0f + exp(-(1.0f * x)));
            }

            @Override
            public float derivative(float x, float a) {
                float s = 1.0f / (1.0f + exp(-(1.0f * x)));
                return s * (1.0f + 1.0f * x * (1.0f - s));
            }
        },
        /**
         * {@code x > 0 ? scale * x : scale * (alpha * (exp(x) - 1))}; derivative {@code scale} or
         * {@code scale * alpha * exp(x)}.
         */
        SELU {
            @Override
            public float activate(float x) {
                return x > 0.0f ? SELU_SCALE * x : SELU_SCALE * (SELU_ALPHA * (exp(x) - 1.0f));
            }

            @Override
            public float derivative(float x, float a) {
                return x > 0.0f ? SELU_SCALE : SELU_SCALE * SELU_ALPHA * exp(x);
            }
        },
        /** {@code Math.min(Math.max(0.2 * x + 0.5, 0), 1)}; derivative {@code 0.2} inside {@code [0, 1]}, else 0. */
        HARD_SIGMOID {
            @Override
            public float activate(float x) {
                float y = 0.2f * x + 0.5f;
                return Math.min(Math.max(y, 0.0f), 1.0f);
            }

            @Override
            public float derivative(float x, float a) {
                float y = 0.2f * x + 0.5f;
                return (y >= 0.0f && y <= 1.0f) ? 0.2f : 0.0f;
            }
        },
        /** {@code (sqrt(x * x + 1) - 1) / 2 + x}; derivative {@code x / (2 * sqrt(x * x + 1)) + 1}. */
        BENT_IDENTITY {
            @Override
            public float activate(float x) {
                return (sqrt1pSquare(x) - 1.0f) / 2.0f + x;
            }

            @Override
            public float derivative(float x, float a) {
                return x / (2.0f * sqrt1pSquare(x)) + 1.0f;
            }
        },
        /** {@code x >= 0 ? x : 0.25 * x}; derivative {@code x >= 0 ? 1 : 0.25}. */
        PRELU {
            @Override
            public float activate(float x) {
                return x >= 0.0f ? x : 0.25f * x;
            }

            @Override
            public float derivative(float x, float a) {
                return x >= 0.0f ? 1.0f : 0.25f;
            }
        },
        /** Identity on {@code (-0.5, 0.5)}, slope 0.5 beyond the thresholds {@code -0.5} and {@code 0.5}. */
        SRELU {
            @Override
            public float activate(float x) {
                if (x <= SRELU_TL) {
                    return SRELU_TL + SRELU_AL * (x - SRELU_TL);
                }
                if (x >= SRELU_TR) {
                    return SRELU_TR + SRELU_AR * (x - SRELU_TR);
                }
                return x;
            }

            @Override
            public float derivative(float x, float a) {
                if (x <= SRELU_TL) {
                    return SRELU_AL;
                }
                if (x >= SRELU_TR) {
                    return SRELU_AR;
                }
                return 1.0f;
            }
        },
        /** {@code (exp(x) - 1) / 1 + 1} (alpha 1); derivative {@code exp(x)}. */
        SOFTEXPONENTIAL {
            @Override
            public float activate(float x) {
                return (exp(1.0f * x) - 1.0f) / 1.0f + 1.0f;
            }

            @Override
            public float derivative(float x, float a) {
                return exp(1.0f * x);
            }
        },
        /** {@code x}; derivative 1. */
        IDENTITY {
            @Override
            public float activate(float x) {
                return x;
            }

            @Override
            public float derivative(float x, float a) {
                return 1.0f;
            }
        };

        private static final float SELU_ALPHA = 1.6732632423543772848f;
        private static final float SELU_SCALE = 1.0507009873554804934f;
        private static final float SRELU_TL = -0.5f;
        private static final float SRELU_AL = 0.5f;
        private static final float SRELU_TR = 0.5f;
        private static final float SRELU_AR = 0.5f;

        /** The activation of pre-activation {@code x}. */
        public abstract float activate(float x);

        /**
         * The derivative at pre-activation {@code x}, where {@code a = activate(x)} (read by SIGMOID and TANH only).
         */
        public abstract float derivative(float x, float a);

        /**
         * The activation with the given ordinal.
         *
         * @throws IllegalArgumentException when the ordinal is outside {@code [0, values().length)}
         */
        public static ActivationFunctionType fromOrdinal(int ordinal) {
            ActivationFunctionType[] all = values();
            if (ordinal < 0 || ordinal >= all.length) {
                throw new IllegalArgumentException("activation ordinal " + ordinal + " is outside [0, "
                        + all.length + ")");
            }
            return all[ordinal];
        }

        private static float exp(float x) {
            return (float) Math.exp(x);
        }

        private static float sigmoid(float x) {
            return 1.0f / (1.0f + exp(-x));
        }

        /** {@code sqrt(1 + x * x)} without overflow: {@code |x| * sqrt(1 + t * t)}, {@code t = 1 / |x|}, above 1. */
        private static float sqrt1pSquare(float x) {
            float ax = Math.abs(x);
            if (!(ax > 1.0f)) {
                return (float) Math.sqrt(1.0f + x * x);
            }
            float t = 1.0f / ax;
            return ax * (float) Math.sqrt(1.0f + t * t);
        }

        /**
         * {@code sin(u) / u} with {@code u = pi * x}; for {@code |u| < 1} the Taylor series to {@code u^8}, which
         * the kernels use to avoid the division by a vanishing {@code u}.
         */
        private static float sinc(float x) {
            float u = PI * x;
            if (Math.abs(u) < 1.0f) {
                float u2 = u * u;
                return 1.0f - u2 * (0.16666666666666667f - u2 * (0.0083333333333333333f
                        - u2 * (1.9841269841269841e-4f - u2 * 2.7557319223985891e-6f)));
            }
            return (float) Math.sin(u) / u;
        }

        /**
         * {@code pi * (cos(u) / u - sin(u) / (u * u))} with {@code u = pi * x}; for {@code |u| < 1} the Taylor
         * series to {@code u^9}, which removes the cancellation of the two quotients near 0.
         */
        private static float sincDerivative(float x) {
            float u = PI * x;
            if (Math.abs(u) < 1.0f) {
                float u2 = u * u;
                return PI * -(u * (0.33333333333333333f - u2 * (0.033333333333333333f
                        - u2 * (1.1904761904761905e-3f - u2 * (2.2045855379188713e-5f
                        - u2 * 2.5052108385441719e-7f)))));
            }
            float s = (float) Math.sin(u);
            float c = (float) Math.cos(u);
            return PI * (c / u - s / (u * u));
        }
    }

    private float learningRate = 0.01f;
    private int batchSize;
    private int tileSize;
    private int[] activationTypes;

    private final int numLayers;
    private final int[] layerSizes;
    private final float[][] weights;
    private final float[][] biases;

    private Map<Float, Float> trainingMap;
    private Map<Float, Float> testingMap;

    /**
     * A network with the given layer widths, TANH activations, Glorot-Bengio weights, zero biases and a random
     * training map of {@code layerSizes[0]} entries.
     *
     * @throws IllegalArgumentException when fewer than two widths are given or a width is not positive
     */
    public NeuralNetworkData(int[] layerSizes) {
        Objects.requireNonNull(layerSizes, "layerSizes must not be null");
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("a network needs at least an input and an output width, got "
                    + Arrays.toString(layerSizes));
        }
        for (int width : layerSizes) {
            if (width < 1) {
                throw new IllegalArgumentException("layer widths must be positive, got " + Arrays.toString(layerSizes));
            }
        }
        this.trainingMap = NeuralUtil.randomFloatMap(layerSizes[0]);
        this.batchSize = Math.min(DEFAULT_BATCH_SIZE, trainingMap.size());
        this.tileSize = Math.min(DEFAULT_TILE_SIZE, trainingMap.size());

        this.layerSizes = layerSizes.clone();
        this.numLayers = layerSizes.length - 1;
        this.weights = new float[numLayers][];
        this.biases = new float[numLayers][];

        int[] actTypes = new int[numLayers];
        Arrays.fill(actTypes, ActivationFunctionType.TANH.ordinal());
        this.activationTypes = actTypes;

        assignWeightsBengio();
    }

    /**
     * Glorot-Bengio initialisation: weights uniform in {@code ±sqrt(6) / (in + out)} (networks of fewer than twelve
     * layers) or {@code ±sqrt(6) / (in + 12)} (deeper networks, where layers past the twelfth start at 0); zero
     * biases.
     */
    private void assignWeightsBengio() {
        int maxNodesForStandard = 12;
        for (int layer = 0; layer < this.numLayers; layer++) {
            int inputSize = this.layerSizes[layer];
            int outputSize = this.layerSizes[layer + 1];

            this.weights[layer] = new float[inputSize * outputSize];
            this.biases[layer] = new float[outputSize];

            float lowerBound;
            float upperBound;
            if (this.numLayers < maxNodesForStandard) {
                lowerBound = (float) (-1 * Math.sqrt(6) / (inputSize + outputSize));
                upperBound = (float) Math.sqrt(6) / (inputSize + outputSize);
            } else {
                lowerBound = (float) (-1 * Math.sqrt(6) / (inputSize + 12));
                upperBound = (float) Math.sqrt(6) / (inputSize + 12);
            }

            for (int i = 0; i < inputSize * outputSize; i++) {
                if (layer <= 12) {
                    this.weights[layer][i] = NeuralUtil.generateRandomFloat(lowerBound, upperBound);
                } else {
                    this.weights[layer][i] = 0.0f;
                }
            }
        }
    }

    /** Every layer's weights concatenated in layer order, each row-major {@code [out, in]}. */
    public float[] flattenWeights(float[][] weights) {
        int totalSize = Arrays.stream(weights).mapToInt(array -> array.length).sum();
        float[] flattenedWeights = new float[totalSize];
        int index = 0;
        for (float[] weightMatrix : weights) {
            System.arraycopy(weightMatrix, 0, flattenedWeights, index, weightMatrix.length);
            index += weightMatrix.length;
        }
        return flattenedWeights;
    }

    /** Every layer's biases concatenated in layer order. */
    public float[] flattenBiases(float[][] biases) {
        int totalSize = Arrays.stream(biases).mapToInt(array -> array.length).sum();
        float[] flattenedBiases = new float[totalSize];
        int index = 0;
        for (float[] biasVector : biases) {
            System.arraycopy(biasVector, 0, flattenedBiases, index, biasVector.length);
            index += biasVector.length;
        }
        return flattenedBiases;
    }

    /** The inverse of {@link #flattenWeights} for the given layer widths. */
    public float[][] unflattenWeights(float[] flattenedWeights, int[] layerSizes) {
        float[][] weights = new float[layerSizes.length - 1][];
        int index = 0;
        for (int layer = 0; layer < layerSizes.length - 1; layer++) {
            float[] weightMatrix = new float[layerSizes[layer] * layerSizes[layer + 1]];
            System.arraycopy(flattenedWeights, index, weightMatrix, 0, weightMatrix.length);
            weights[layer] = weightMatrix;
            index += weightMatrix.length;
        }
        return weights;
    }

    /** The inverse of {@link #flattenBiases} for the given layer widths. */
    public float[][] unflattenBiases(float[] flattenedBiases, int[] layerSizes) {
        float[][] biases = new float[layerSizes.length - 1][];
        int index = 0;
        for (int layer = 0; layer < layerSizes.length - 1; layer++) {
            float[] biasVector = new float[layerSizes[layer + 1]];
            System.arraycopy(flattenedBiases, index, biasVector, 0, biasVector.length);
            biases[layer] = biasVector;
            index += biasVector.length;
        }
        return biases;
    }

    /**
     * Copies flattened weights and biases ({@link #flattenWeights} and {@link #flattenBiases} layout) into this
     * network's live arrays, for example the updated parameters of a training step.
     *
     * @throws IllegalArgumentException when a length differs from this network's parameter count
     */
    public void setParameters(float[] flattenedWeights, float[] flattenedBiases) {
        Objects.requireNonNull(flattenedWeights, "flattenedWeights must not be null");
        Objects.requireNonNull(flattenedBiases, "flattenedBiases must not be null");
        int weightCount = Arrays.stream(weights).mapToInt(array -> array.length).sum();
        int biasCount = Arrays.stream(biases).mapToInt(array -> array.length).sum();
        if (flattenedWeights.length != weightCount || flattenedBiases.length != biasCount) {
            throw new IllegalArgumentException("expected " + weightCount + " weights and " + biasCount
                    + " biases, got " + flattenedWeights.length + " and " + flattenedBiases.length);
        }
        int w = 0;
        int b = 0;
        for (int layer = 0; layer < numLayers; layer++) {
            System.arraycopy(flattenedWeights, w, weights[layer], 0, weights[layer].length);
            System.arraycopy(flattenedBiases, b, biases[layer], 0, biases[layer].length);
            w += weights[layer].length;
            b += biases[layer].length;
        }
    }

    /**
     * CPU forward pass of {@code nnData} for one input row: per layer the products summed in ascending input order,
     * plus the bias, through the layer's activation.
     *
     * @throws IllegalArgumentException when the input length differs from the network's input width
     */
    public float[] predict(NeuralNetworkData nnData, float[] input) {
        Objects.requireNonNull(nnData, "nnData must not be null");
        Objects.requireNonNull(input, "input must not be null");
        int[] sizes = nnData.getLayerSizes();
        if (input.length != sizes[0]) {
            throw new IllegalArgumentException("input of " + input.length + " values for a network of input width "
                    + sizes[0]);
        }
        int[] acts = nnData.getActivationTypes();
        float[] currentLayerOutput = input;
        for (int layer = 0; layer < nnData.getNumLayers(); layer++) {
            int inputSize = sizes[layer];
            int outputSize = sizes[layer + 1];
            float[] weightMatrix = nnData.getWeights()[layer];
            float[] biasVector = nnData.getBiases()[layer];
            ActivationFunctionType act = ActivationFunctionType.fromOrdinal(acts[layer]);

            float[] nextLayerOutput = new float[outputSize];
            for (int i = 0; i < outputSize; i++) {
                float neuronInput = 0.0f;
                for (int j = 0; j < inputSize; j++) {
                    neuronInput += currentLayerOutput[j] * weightMatrix[i * inputSize + j];
                }
                neuronInput += biasVector[i];
                nextLayerOutput[i] = act.activate(neuronInput);
            }
            currentLayerOutput = nextLayerOutput;
        }
        return currentLayerOutput;
    }

    public Map<Float, Float> getTrainingMap() {
        return trainingMap;
    }

    public void setTrainingMap(Map<Float, Float> trainingMap) {
        this.trainingMap = trainingMap;
    }

    public Map<Float, Float> getTestingMap() {
        return testingMap;
    }

    public void setTestingMap(Map<Float, Float> testingMap) {
        this.testingMap = testingMap;
    }

    public int getNumLayers() {
        return numLayers;
    }

    /** The layer widths, input first (a copy). */
    public int[] getLayerSizes() {
        return layerSizes.clone();
    }

    public float[][] getWeights() {
        return weights;
    }

    public float[][] getBiases() {
        return biases;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getTileSize() {
        return tileSize;
    }

    public void setTileSize(int tileSize) {
        this.tileSize = tileSize;
    }

    /** The activation ordinal of each layer (a copy). */
    public int[] getActivationTypes() {
        return activationTypes.clone();
    }

    /**
     * Sets one activation ordinal per layer.
     *
     * @throws IllegalArgumentException when the count differs from the layer count or an ordinal is not an
     *                                  {@link ActivationFunctionType}
     */
    public void setActivationTypes(int[] activationTypes) {
        Objects.requireNonNull(activationTypes, "activationTypes must not be null");
        if (activationTypes.length != numLayers) {
            throw new IllegalArgumentException(activationTypes.length + " activation types for " + numLayers
                    + " layers");
        }
        for (int act : activationTypes) {
            ActivationFunctionType.fromOrdinal(act);
        }
        this.activationTypes = activationTypes.clone();
    }
}
