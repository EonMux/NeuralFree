package com.eonmux.neuralfree.ai.activations;

import com.eonmux.neuralfree.ai.core.ActivationFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

/**
 * The step function is most suitable for the output layer
 */
public class StepFunction implements ActivationFunction {

    /**
     * Returns an output that is a transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * <p>
     * Where the transformation is: x < 0 ? 0 : 1
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

        // x < 0 ? 0 : 1
        return (x < 0f ? 0f : 1f) + bias;
    }

    /**
     * Returns the derivative of the transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * with respect to a given variable
     * <p>
     * Where the transformation is: x < 0 ? 0 : 1
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
        // sum(w*x) + b < 0 ? 0 : 1
        // Technically it becomes infinite at exactly 0 but good luck actually representing that
        // derivative of a constant is 0
        switch (respectTo) {
            case WEIGHT:
                // d/dw_index
                // 0
                return 0;
            case BIAS:
                // d/db
                // 1
                return 1;
            case INPUT:
                // d/dx
                // 0
                return 0;
            case ACTUAL:
            case DELTA:
            default:
                return 0;
        }
    }
}
