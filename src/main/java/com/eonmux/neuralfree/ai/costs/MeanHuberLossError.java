package com.eonmux.neuralfree.ai.costs;

import com.eonmux.neuralfree.ai.core.CostFunction;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

import static java.lang.Math.*;

/**
 * Huber loss curves around the minima which decreases the gradient
 * It’s more robust to outliers than MSE. Therefore, it combines good properties from both MSE and MAE.
 * However, the problem with Huber loss is that we might need to train parameter delta which is an iterative process.
 */
public class MeanHuberLossError implements CostFunction {
    /**
     * Calculates how far off our results are from the expected results
     * <p>
     * Where the error calculation formula is: SUM (d >= |y - f(x)|? (0.5 * (y-f(x))^2 / number of results) : (((d * |y - f(x)|) - (0.5* d^2)) / number of results)
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

        // SUM (d >= |y - f(x)|? (0.5 * (y-f(x))^2 / number of results) : (((d * |y - f(x)|) - (0.5* d^2)) / number of results)
        for (int x = 0; x < n; x++) {
            if (delta >= (abs(y.get(x) - f_x.get(x)))) {
                total += 0.5f * pow(y.get(x) - f_x.get(x), 2f);
            } else {
                total += (delta * abs(y.get(x) - f_x.get(x))) - 0.5f * pow(delta, 2);
            }
        }
        return total / n;
    }

    /**
     * Returns the derivative of the error calculation
     * with respect to a given variable
     * <p>
     * Where the error calculation formula is: SUM (d >= |y - f(x)|? (0.5 * (y-f(x))^2 / number of results) : (((d * |y - f(x)|) - (0.5* d^2)) / number of results)
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

        // SUM (d >= |y - f(x)|? (0.5 * (y-f(x))^2 / number of results) : (((d * |y - f(x)|) - (0.5* d^2)) / number of results)
        switch (respectTo) {
            case INPUT:
                for (int x = 0; x < n; x++) {
                    if (delta >= (abs(y.get(x) - fX.get(x)))) {
                        // d/df(x)
                        // -(y-x)/n
                        derivative += ((y.get(x) - fX.get(x)) * -1);
                    } else {
                        // d/df(x)
                        // d * (f(x)-y)/|f(x)-y|
                        derivative += delta * (fX.get(x) - y.get(x)) / (abs((fX.get(x) - y.get(x))));
                    }
                }
                break;
            case ACTUAL:
                for (int x = 0; x < n; x++) {
                    if (delta >= (abs(y.get(x) - fX.get(x)))) {
                        // d/dy
                        // (y-x)/n
                        derivative += (y.get(x) - fX.get(x));
                    } else {
                        // d/dy
                        // d * (y-f(x))/|x-y|
                        derivative += delta * (y.get(x) - fX.get(x)) / abs(fX.get(x) - y.get(x));
                    }
                }
                break;
            case DELTA:
                for (int x = 0; x < n; x++) {
                    if ((delta < (abs(y.get(x) - fX.get(x))))) {
                        // d/dd
                        // |f(x) - y| - d
                        derivative += abs(fX.get(x) - y.get(x)) - delta;
                    }
                }
                break;
            default:
                break;
        }
        return derivative / n;
    }
}
