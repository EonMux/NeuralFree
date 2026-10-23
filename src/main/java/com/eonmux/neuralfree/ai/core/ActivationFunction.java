package com.eonmux.neuralfree.ai.core;

import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

/**
 * An activation function over weighted inputs plus a bias: its value ({@code activate}) and its derivative with
 * respect to one of the variables of {@link RespectTo} ({@code differentiate}).
 */
public interface ActivationFunction {
    /**
     * Variable to take the derivative with respect to
     * <li>{@link #INPUT}</li>
     * <li>{@link #WEIGHT}</li>
     * <li>{@link #BIAS}</li>
     * <li>{@link #ACTUAL}</li>
     * <li>{@link #DELTA}</li>
     */
    enum RespectTo {
        /**
         * An input value
         */
        INPUT,
        /**
         * The weight
         */
        WEIGHT,
        /**
         * The bias
         */
        BIAS,
        /**
         * A known good result
         */
        ACTUAL,
        /**
         * A parameter
         */
        DELTA
    }

    /**
     * Returns an output that is a transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     *
     * Where the transformation is:
     * @param inputs  Input values
     * @param weights Weight values
     * @param bias    Bias value
     * @return Output value
     */
    float activate(FloatDataSet inputs, FloatDataSet weights, float bias);


    /**
     * Returns the derivative of the transformation of
     * (Sum1->N(InputN*WeightN) + Bias)
     * with respect to a given variable
     *
     * Where the transformation is:
     * @param respectTo     The variable we are taking the derivative with respect to
     * @param inputs        The input value(s) (Or the previous layer's output value(s))
     * @param weights       The weight(s) given to those input(s)
     * @param specificIndex If we are taking the derivative with respect to just one value in the summation this will be set
     *                      to the index value in the array of that value, else it will be -1
     * @param bias          The bias of this neuron
     * @param delta         The delta parameter
     * @return derivative value
     */
    float differentiate(RespectTo respectTo, FloatDataSet inputs, FloatDataSet weights, int specificIndex, float bias, float delta);


}
