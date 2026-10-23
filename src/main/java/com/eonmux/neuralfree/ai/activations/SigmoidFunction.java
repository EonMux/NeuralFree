package com.eonmux.neuralfree.ai.activations;

import com.eonmux.neuralfree.ai.core.ActivationFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.E;
import static java.lang.Math.pow;

/**
 * An optimized version of the sigmoid function that is suitable for up to 6 hidden layers
 */
public class SigmoidFunction implements ActivationFunction {
    /**
     * Returns an output that is a transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * <p>
     * Where the transformation is: 2 * (1/(e^(-x)+1)) - 1
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

        // 2 * (1/(e^(-x)+1)) - 1
        return ((float) (2 * (1f / ((1f / pow(E, x)) + 1f)) - 1)) + bias;
    }

    /**
     * Returns the derivative of the transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * with respect to a given variable
     * <p>
     * Where the transformation is: 2 * (1/(e^(-x)+1)) - 1
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

        // (2 * (1/(e^(-(w*x + b))+1)) - 1)
        switch (respectTo) {
            case WEIGHT:
                // d/dw
                // (2 E^(-b - w x) x)/(1 + E^(-b - w x))^2
                // (2*Power(E,-b - w*x)*x)/Power(1 + Power(E,-b - w*x),2)
                if (specificIndex != -1) {
                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);
                    derivative = (float) ((2 * pow(E, -bias - w * x) * x) / pow(1 + pow(E, -bias - w * x), 2));
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        w = weights.get(i);
                        derivative = (float) ((2 * pow(E, -bias - w * x) * x) / pow(1 + pow(E, -bias - w * x), 2));
                    }
                }
                return derivative;
            case BIAS:
                // d/db
                return 1;
            case INPUT:
                // d/dx
                // (2 E^(-b - w x) w)/(1 + E^(-b - w x))^2
                // (2*Power(E,-b - w*x)*w)/Power(1 + Power(E,-b - w*x),2)
                if (specificIndex != -1) {
                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);
                    derivative = (float) ((2 * pow(E, -bias - w * x) * w) / pow(1 + pow(E, -bias - w * x), 2));
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        w = weights.get(i);
                        derivative += ((2 * pow(E, -bias - w * x) * w) / pow(1 + pow(E, -bias - w * x), 2));
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
