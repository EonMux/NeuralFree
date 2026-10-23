package com.eonmux.neuralfree.ai.activations;

import com.eonmux.neuralfree.ai.core.ActivationFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.log;
import static java.lang.Math.pow;

/**
 * Hyperbolic activation function suitable for up to 10 hidden layers
 */
public class ArcTanhFunction implements ActivationFunction {
    /**
     * Returns an output that is a transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * <p>
     * Where the transformation is: (1/2 * log(x+1)) - (1/2 * log(1-x))
     *
     * @param inputs  Input values
     * @param weights Weight values
     * @param bias    Bias value
     * @return Output value
     */
    @Override
    public float activate(FloatDataSet inputs, FloatDataSet weights, float bias) {
        float x = 0f;

        //Calculate all the input*weight
        for (int i = 0; i < inputs.size(); i++) {
            x += inputs.get(i) * weights.get(i);
        }

        // (1/2 * log(x+1)) - (1/2 * log(1-x))
        return ((float) ((0.5f * log(x + 1)) - (0.5f * log(1 - x)))) + bias;
    }

    /**
     * Returns the derivative of the transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * with respect to a given variable
     * <p>
     * Where the transformation is: (1/2 * log(x+1)) - (1/2 * log(1-x))
     *
     * @param respectTo     The variable we are taking the derivative with respect to
     * @param inputs        The input value(s) (Or the previous layer's output value(s))
     * @param weights       The weight(s) given to those input(s)
     * @param specificIndex If we are taking the derivative with respect to just one value in the summation this will be set
     *                      to the index value in the array of that value, else it will be -1
     * @param bias          The bias of this neuron
     * @param delta         The delta parameter
     * @return derivative value
     */
    @Override
    public float differentiate(RespectTo respectTo, FloatDataSet inputs, FloatDataSet weights, int specificIndex, float bias, float delta) {
        //Temp variables
        float x = 0f, w = 0f, sum = 0f, sumX = 0f, sumW = 0f, derivative = 0f;

        // w * ((1/2 * log(x+1)) - (1/2 * log(1-x))) + b
        switch (respectTo) {
            case WEIGHT:
                // d/dw
                // (-Log[1 - x] + Log[1 + x])/2
                if (specificIndex != -1) {
                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);
                    derivative = (float) ((-log(1 - x) + log(1 + x)) / 2);
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        w = weights.get(i);
                        derivative += ((-log(1 - x) + log(1 + x)) / 2);
                    }
                }
                return derivative;
            case BIAS:
                // d/db
                return 1;
            case INPUT:
                // d/dx
                // w/(1 - x^2)
                // w/(1 - Power(x,2))
                if (specificIndex != -1) {
                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);
                    derivative = (float) (w / (1 - pow(x, 2)));
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        w = weights.get(i);
                        derivative += (w / (1 - pow(x, 2)));
                    }
                }
                return derivative;
            case ACTUAL:
            case DELTA:
            default:
                return 0;
        }
    }
}
