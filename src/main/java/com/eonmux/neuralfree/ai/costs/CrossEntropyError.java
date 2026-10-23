package com.eonmux.neuralfree.ai.costs;

import com.eonmux.neuralfree.ai.core.CostFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.*;

/**
 * Cross-entropy is widely used as a loss function when optimizing classification models
 * It is faster than MSE for these types of problems
 * Similar to KL Divergence, it is more for suited for probability type classification
 * Cross-entropy loss increases as the predicted probability diverges from the actual label.
 */
public class CrossEntropyError implements CostFunction {
    /**
     * Calculates how far off our results are from the expected results
     * <p>
     * Where the error calculation formula is: SUM(y - ln(f(x)))
     *
     * @param f_x   The results of an evaluation of given inputs
     * @param y     The expected results of an evaluation of given inputs
     * @param delta The delta parameter
     * @return The Deviation from Expected Values
     */
    @Override
    public float calculateCost(FloatDataSet f_x, FloatDataSet y, float delta) {
        float total = 0f;
        float n = f_x.size();

        // SUM(y - ln(f(x)))
        for (int x = 0; x < n; x++) {
            total += y.get(x) * log(f_x.get(x));
        }
        return total;
    }

    /**
     * Returns the derivative of the error calculation
     * with respect to a given variable
     * <p>
     * Where the error calculation formula is: SUM(y - ln(f(x)))
     *
     * @param fX            The results of an evaluation of given inputs
     * @param y             The expected results of an evaluation of given inputs
     * @param delta         The delta parameter
     * @param respectTo     The variable we are taking the derivative with respect to
     * @param specificIndex If we are taking the derivative with respect to just one value in the summation this will be set
     *                      to the index value in the array of that value, else it will be -1
     * @return derivative value
     */
    @Override
    public float differentiate(FloatDataSet fX, FloatDataSet y, float delta, RespectTo respectTo, int specificIndex) {
        float derivative = 0f;
        float n = fX.size();
        
        // SUM(y - ln(f(x)))
        for (int x = 0; x < n; x++) {
            switch (respectTo) {
                case INPUT:
                    // d/df(x)
                    // -1/x
                    derivative += (1f / fX.get(x)) * -1f;
                    break;
                case ACTUAL:
                    // d/dy
                    // 1
                    derivative += 1f;
                    break;
                default:
                    break;
            }
        }
        return derivative;
    }
}
