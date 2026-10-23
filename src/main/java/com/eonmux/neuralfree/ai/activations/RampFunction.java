package com.eonmux.neuralfree.ai.activations;

import com.eonmux.neuralfree.ai.core.ActivationFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

/**
 * The ramp function is linear above 0, and 0 when less than or equal to zero
 * Otherwise known as ReLU it is useful for the hidden layers in DNN as it is very efficient
 * Avoid usage of this outside of hidden layers as it is effectively linear
 */
public class RampFunction implements ActivationFunction {

    /**
     * Returns an output that is a transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * <p>
     * Where the transformation is: x > 0 ? x : 0
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

        // x > 0 ? x : 0
        return (x > 0 ? x : 0) + bias;
    }

    /**
     * Returns the derivative of the transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * with respect to a given variable
     * <p>
     * Where the transformation is: x > 0 ? x : 0
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

        // (sum(w*x) + b) > 0 ? (sum(w*x) + b) : 0

        //Handy sums
        for (int i = 0; i < inputs.size(); i++) {
            float curWeight = weights.get(i);
            float curInput = inputs.get(i);

            sumX += curInput;
            sumW += curWeight;
            sum += curWeight * curInput;
        }

        switch (respectTo) {
            case WEIGHT:
                // d/dw_index
                if (specificIndex != -1) {
                    // (sum(w*x) + b) > 0 ? x_index : 0
                    return (sum + bias) > 0 ? inputs.get(specificIndex) : 0;
                } else {
                    // (sum(w * x) + b) > 0 ? (sum(x)) : 0
                    return (sum + bias) > 0 ? sumX : 0;
                }
            case BIAS:
                // d/db
                return 1;
            case INPUT:
                // d/dx
                if (specificIndex != -1) {
                    // (sum(w * x) + b) > 0 ? w_index : 0
                    return (sum + bias) > 0 ? weights.get(specificIndex) : 0;
                } else {
                    // (sum(w * x) + b) > 0 ? (sum(w)) : 0
                    return (sum + bias) > 0 ? sumW : 0;
                }
            case ACTUAL:
            case DELTA:
            default:
                return 0;
        }
    }
}
