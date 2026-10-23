package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;

import java.util.List;

/**
 * An independent double-precision host copy of the reinforcement-learning perceptron
 * {@code Linear(in, h) -> ReLU -> Linear(h, h) -> ReLU -> Linear(h, out)}, read from the six parameters the algorithms
 * hand their optimizer factories ({@code [h, in]} weight, {@code [h]} bias, then the next two layers, each weight
 * {@code [out, in]} row-major as {@code Linear} stores it). It evaluates the network and predicts one SGD update
 * {@code p - lr * dL/dp} by back-propagating given output gradients, so a test can check an algorithm's update against
 * the analytic gradient of its documented loss. Instances are immutable.
 */
final class HostMlp {
    private static final int LAYERS = 3;

    /** Per layer, the {@code [out][in]} weights. */
    private final double[][][] weights;
    /** Per layer, the {@code [out]} biases. */
    private final double[][] biases;

    private HostMlp(double[][][] weights, double[][] biases) {
        this.weights = weights;
        this.biases = biases;
    }

    /**
     * The current values of {@code params} from index {@code offset}: three layers' weight and bias in order.
     *
     * @throws IllegalArgumentException when the parameters do not have the perceptron's shapes
     */
    static HostMlp of(List<Parameter> params, int offset) {
        double[][][] w = new double[LAYERS][][];
        double[][] b = new double[LAYERS][];
        for (int layer = 0; layer < LAYERS; layer++) {
            Parameter weight = params.get(offset + 2 * layer);
            Parameter bias = params.get(offset + 2 * layer + 1);
            Shape shape = weight.tensor().shape();
            if (shape.rank() != 2 || !bias.tensor().shape().equals(Shape.vector(shape.dim(0)))) {
                throw new IllegalArgumentException("layer " + layer + " has weight " + shape + " and bias "
                        + bias.tensor().shape());
            }
            if (layer > 0 && shape.dim(1) != w[layer - 1].length) {
                throw new IllegalArgumentException("layer " + layer + " takes " + shape.dim(1) + " inputs, the layer "
                        + "before has " + w[layer - 1].length + " outputs");
            }
            w[layer] = rows(weight.tensor().toFloatArray(), shape.dim(0), shape.dim(1));
            b[layer] = toDoubles(bias.tensor().toFloatArray());
        }
        return new HostMlp(w, b);
    }

    /** The network's output for one input row. */
    double[] forward(float[] input) {
        return pass(toDoubles(input)).output();
    }

    /**
     * The network after one SGD step {@code p - lr * g}, where {@code g} is the gradient of
     * {@code sum_i outputGrads[i] . f(inputs[i])} in every parameter. A ReLU passes the gradient where its input is
     * positive.
     */
    HostMlp sgdStep(float[][] inputs, double[][] outputGrads, double lr) {
        if (inputs.length != outputGrads.length) {
            throw new IllegalArgumentException(inputs.length + " inputs but " + outputGrads.length + " gradients");
        }
        double[][][] gw = new double[LAYERS][][];
        double[][] gb = new double[LAYERS][];
        for (int layer = 0; layer < LAYERS; layer++) {
            gw[layer] = new double[weights[layer].length][weights[layer][0].length];
            gb[layer] = new double[biases[layer].length];
        }
        for (int i = 0; i < inputs.length; i++) {
            Pass pass = pass(toDoubles(inputs[i]));
            double[] upstream = outputGrads[i].clone();
            for (int layer = LAYERS - 1; layer >= 0; layer--) {
                double[] layerInput = pass.activations[layer];
                for (int o = 0; o < upstream.length; o++) {
                    gb[layer][o] += upstream[o];
                    for (int k = 0; k < layerInput.length; k++) {
                        gw[layer][o][k] += upstream[o] * layerInput[k];
                    }
                }
                if (layer == 0) {
                    break;
                }
                double[] below = new double[layerInput.length];
                double[] preActivation = pass.preActivations[layer - 1];
                for (int k = 0; k < below.length; k++) {
                    if (preActivation[k] > 0.0) {
                        double sum = 0.0;
                        for (int o = 0; o < upstream.length; o++) {
                            sum += weights[layer][o][k] * upstream[o];
                        }
                        below[k] = sum;
                    }
                }
                upstream = below;
            }
        }
        double[][][] w = new double[LAYERS][][];
        double[][] b = new double[LAYERS][];
        for (int layer = 0; layer < LAYERS; layer++) {
            w[layer] = new double[weights[layer].length][];
            for (int o = 0; o < weights[layer].length; o++) {
                w[layer][o] = new double[weights[layer][o].length];
                for (int k = 0; k < w[layer][o].length; k++) {
                    w[layer][o][k] = weights[layer][o][k] - lr * gw[layer][o][k];
                }
            }
            b[layer] = new double[biases[layer].length];
            for (int o = 0; o < b[layer].length; o++) {
                b[layer][o] = biases[layer][o] - lr * gb[layer][o];
            }
        }
        return new HostMlp(w, b);
    }

    /** The layer inputs, the hidden pre-activations and the output of one forward pass. */
    private static final class Pass {
        /** {@code activations[l]} is the input of layer {@code l}, and {@code activations[LAYERS]} the output. */
        final double[][] activations = new double[LAYERS + 1][];
        /** {@code preActivations[l]} is the output of layer {@code l} before its ReLU, for the hidden layers. */
        final double[][] preActivations = new double[LAYERS - 1][];

        double[] output() {
            return activations[LAYERS];
        }
    }

    private Pass pass(double[] input) {
        if (input.length != weights[0][0].length) {
            throw new IllegalArgumentException("expected " + weights[0][0].length + " inputs, got " + input.length);
        }
        Pass pass = new Pass();
        pass.activations[0] = input;
        for (int layer = 0; layer < LAYERS; layer++) {
            double[] in = pass.activations[layer];
            double[] out = new double[weights[layer].length];
            for (int o = 0; o < out.length; o++) {
                double sum = biases[layer][o];
                for (int k = 0; k < in.length; k++) {
                    sum += weights[layer][o][k] * in[k];
                }
                out[o] = sum;
            }
            if (layer < LAYERS - 1) {
                pass.preActivations[layer] = out;
                double[] relu = new double[out.length];
                for (int o = 0; o < out.length; o++) {
                    relu[o] = Math.max(out[o], 0.0);
                }
                pass.activations[layer + 1] = relu;
            } else {
                pass.activations[layer + 1] = out;
            }
        }
        return pass;
    }

    private static double[][] rows(float[] values, int rows, int cols) {
        double[][] out = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out[r][c] = values[r * cols + c];
            }
        }
        return out;
    }

    private static double[] toDoubles(float[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }
}
