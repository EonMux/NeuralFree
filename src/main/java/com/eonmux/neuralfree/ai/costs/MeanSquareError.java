package com.eonmux.neuralfree.ai.costs;

import com.eonmux.neuralfree.ai.core.CostFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.*;

/**
 * The most commonly used type of cost function, not quite as robust as the absolute version but is faster to solve
 * Otherwise known as the Quadratic Loss Function
 */
public class MeanSquareError implements CostFunction {
    /**
     * Calculates how far off our results are from the expected results
     * <p>
     * Where the error calculation formula is: SUM((y - f(x))^2)/(number of results * 2)
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

        // SUM((y - f(x))^2)/(number of results * 2)
        for (int x = 0; x < n; x++) {
            total += pow(y.get(x) - f_x.get(x), 2f);
        }
        return total / (n * 2);
    }

    /**
     * Returns the derivative of the error calculation
     * with respect to a given variable
     * <p>
     * Where the error calculation formula is: SUM((y - f(x))^2)/(number of results * 2)
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

        // SUM((y - f(x))^2)/(n * 2)
        // Note for means it is easier to divide at the end
        derivative = 0f;
        switch (respectTo) {
            case INPUT:
                // d/df(x)
                // -(y-x)/n
                for (int x = 0; x < n; x++) {
                    derivative += (y.get(x) - fX.get(x)) * -1;
                }
                break;
            case ACTUAL:
                // d/dy
                // (y-x)/n
                for (int x = 0; x < n; x++) {
                    derivative += y.get(x) - fX.get(x);
                }
                break;
            default:
                break;
        }
        return derivative / n;
    }
}
