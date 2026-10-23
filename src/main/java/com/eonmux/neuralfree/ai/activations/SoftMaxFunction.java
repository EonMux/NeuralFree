package com.eonmux.neuralfree.ai.activations;

import com.eonmux.neuralfree.ai.core.ActivationFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.E;
import static java.lang.Math.pow;

/**
 * The SoftMax activation is used in the final output layer where a probability is wanted
 * instead of a binary classification
 */
public class SoftMaxFunction implements ActivationFunction {
    /**
     * Returns an output that is a transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * <p>
     * Where the transformation is: e^(x_1)/sum(e^(x))
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

        // e^(x_1)/sum(e^(x))
        return (float) (pow(E, inputs.get(0)) / pow(E, x));
    }

    /**
     * Returns the derivative of the transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * with respect to a given variable
     * <p>
     * Where the transformation is: e^(x_1)/sum(e^(x))
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

        /*  w * (e^(x_1)/sum(e^(x))) + b
         *  This is like..
         *  w * (e^x /(e^y + e^z + e^j)) + b
         */
        switch (respectTo) {
            case WEIGHT:
                // d/dw
                // E^Subscript[x, 1]/Sum[E^x, x]
                // e^(x_1) / Sum[E^x]
                if (specificIndex != -1) {

                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        sumX += pow(E, x);
                    }

                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);

                    derivative = (float) (pow(E, x) / sumX);
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        for (int j = 0; j < inputs.size(); j++) {
                            x = inputs.get(j);
                            sumX += pow(E, x);
                        }

                        x = inputs.get(i);
                        w = weights.get(i);

                        derivative += (pow(E, x) / sumX);
                    }
                }
                return derivative;
            case BIAS:
                // d/db
                return 1;
            case INPUT:
                // d/dx
                // This one wolfram doesn't handle nicely..
                // (E^x * (Sum(E^x) - E^x) * w)/(Sum(E^x))^2
                if (specificIndex != -1) {
                    for (int i = 0; i < inputs.size(); i++) {
                        x = inputs.get(i);
                        sumX += pow(E, x);
                    }

                    x = inputs.get(specificIndex);
                    w = weights.get(specificIndex);

                    derivative = (float) ((pow(E, x) * (sumX - pow(E, x)) * w) / pow(sumX, 2));
                } else {
                    derivative = 0f;
                    for (int i = 0; i < inputs.size(); i++) {
                        for (int j = 0; j < inputs.size(); j++) {
                            x = inputs.get(j);
                            sumX += pow(E, x);
                        }

                        x = inputs.get(i);
                        w = weights.get(i);

                        derivative += ((pow(E, x) * (sumX - pow(E, x)) * w) / pow(sumX, 2));
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
