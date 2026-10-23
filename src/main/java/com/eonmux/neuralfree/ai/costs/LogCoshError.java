package com.eonmux.neuralfree.ai.costs;

import com.eonmux.neuralfree.ai.core.CostFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.*;

/**
 * Log Cosh works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction.
 * It has all the advantages of Huber loss, and it’s twice differentiable everywhere, unlike Huber loss.
 */
public class LogCoshError implements CostFunction {
    /**
     * Calculates how far off our results are from the expected results
     * <p>
     * Where the error calculation formula is: SUM(ln(cosh(f(x)-y)))
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

        // SUM(ln(cosh(f(x)-y)))
        for (int x = 0; x < n; x++) {
            total += log(cosh(f_x.get(x) - y.get(x)));
        }
        return total;
    }

    /**
     * Returns the derivative of the error calculation
     * with respect to a given variable
     * <p>
     * Where the error calculation formula is: SUM(ln(cosh(f(x)-y)))
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
        // SUM(ln(cosh(f(x)-y)))
        switch (respectTo) {
            case INPUT:
                // d/df(x)
                // tanh(x-y)
                for (int x = 0; x < n; x++) {
                    derivative += tanh(fX.get(x) - y.get(x));
                }
                break;
            case ACTUAL:
                // d/dy
                // -tanh(x-y)
                for (int x = 0; x < n; x++) {
                    derivative += tanh(fX.get(x) - y.get(x)) * -1f;
                }
                break;
            default:
                break;
        }
        return derivative;
    }
}
