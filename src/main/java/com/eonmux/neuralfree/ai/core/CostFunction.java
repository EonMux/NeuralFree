package com.eonmux.neuralfree.ai.core;

import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

/**
 * This should really be called an Error function but for some reason in neural networks it is called a Cost function
 * Maybe because it is the most "Costly" operation in training?
 */
public interface CostFunction {
    /**
     * Variable to take the derivative with respect to
     * <li>{@link #INPUT}</li>
     * <li>{@link #WEIGHT}</li>
     * <li>{@link #BIAS}</li>
     * <li>{@link #ACTUAL}</li>
     * <li>{@link #DELTA}</li>
     */
    public enum RespectTo {
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
     * Calculates how far off our results are from the expected results
     * <p>
     * Where the error calculation formula is:
     *
     * @param f_x   The results of an evaluation of given inputs
     * @param y     The expected results of an evaluation of given inputs
     * @param delta The delta parameter
     * @return The Deviation from Expected Values
     */
    float calculateCost(FloatDataSet f_x, FloatDataSet y, float delta);

    /**
     * Returns the derivative of the error calculation
     * with respect to a given variable
     * <p>
     * Where the error calculation formula is:
     *
     * @param fX            The results of an evaluation of given inputs
     * @param y             The expected results of an evaluation of given inputs
     * @param delta         The delta parameter
     * @param respectTo     The variable we are taking the derivative with respect to
     * @param specificIndex If we are taking the derivative with respect to just one value in the summation this will be set
     *                      to the index value in the array of that value, else it will be -1
     * @return derivative value
     */
    float differentiate(FloatDataSet fX, FloatDataSet y, float delta, RespectTo respectTo, int specificIndex);
}
