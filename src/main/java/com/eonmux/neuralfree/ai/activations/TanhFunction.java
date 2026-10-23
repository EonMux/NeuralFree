package com.eonmux.neuralfree.ai.activations;

import com.eonmux.neuralfree.ai.core.ActivationFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.*;

/**
 * Hyperbolic activation function suitable for up to 10 hidden layers
 */
public class TanhFunction implements ActivationFunction {
    /**
     * Returns an output that is a transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * <p>
     * Where the transformation is: e^x/(e^(-x) + e^x) - e^(-x)/(e^(-x) + e^x)
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

        // e^x/(e^(-x) + e^x) - e^(-x)/(e^(-x) + e^x)
        return ((float) tanh(x));
    }

    /**
     * Returns the derivative of the transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * with respect to a given variable
     * <p>
     * Where the transformation is: e^x/(e^(-x) + e^x) - e^(-x)/(e^(-x) + e^x)
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

        // w * (e^x/(e^(-x) + e^x) - e^(-x)/(e^(-x) + e^x)) + b
        switch (respectTo) {
            case WEIGHT:
                // d/dw
                // (-1 + E^(2 x))/(1 + E^(2 x))
                // (-1 + Power(E,2*x))/(1 + Power(E,2*x))
                if (specificIndex != -1) {
                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);
                    derivative = (float) ((-1 + pow(E, 2 * x)) / (1 + pow(E, 2 * x)));
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        w = weights.get(i);
                        derivative += ((-1 + pow(E, 2 * x)) / (1 + pow(E, 2 * x)));
                    }
                }
                return derivative;
            case BIAS:
                // d/db
                return 1;
            case INPUT:
                // d/dx
                // (4 E^(2 x) w)/(1 + E^(2 x))^2
                // (4*Power(E,2*x)*w)/Power(1 + Power(E,2*x),2)
                if (specificIndex != -1) {
                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);
                    derivative = (float) ((4 * pow(E, 2 * x) * w) / pow(1 + pow(E, 2 * x), 2));
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        w = weights.get(i);
                        derivative += ((4 * pow(E, 2 * x) * w) / pow(1 + pow(E, 2 * x), 2));
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
