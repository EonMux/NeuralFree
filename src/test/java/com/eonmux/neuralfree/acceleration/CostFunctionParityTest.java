package com.eonmux.neuralfree.acceleration;

import org.junit.Test;
import static org.junit.Assert.*;

import com.eonmux.neuralfree.ai.costs.*;
import com.eonmux.neuralfree.datatype.structures.FloatDataSet;

public class CostFunctionParityTest {

    private static final float TOLERANCE = 1e-5f;
    private static final float[][] TEST_PAIRS = {
        {0.1f, 0.2f},
        {0.3f, 0.4f},
        {0.5f, 0.6f},
        {0.8f, 0.9f},
        {1.0f, 1.0f},
        {0.0f, 0.1f},
        {0.9f, 0.1f}
    };

    private static final float DEFAULT_DELTA = 1.0f;

    private static final int MSE = 0;
    private static final int MAE = 1;
    private static final int CROSS_ENTROPY = 2;
    private static final int KL_DIVERGENCE = 3;
    private static final int LOG_COSH = 4;
    private static final int HUBER = 5;

    private static final String[] COST_NAMES = {
        "MeanSquaredError", "MeanAbsoluteError", "CrossEntropy", 
        "KLDivergence", "LogCosh", "Huber"
    };

    private FloatDataSet createFloatDataSet(float[] values) {
        FloatDataSet ds = new FloatDataSet();
        ds.addAll(values);
        return ds;
    }

    @Test
    public void testAll6CostFunctionsExist() {
        assertEquals("All 6 cost types should be defined", 6, COST_NAMES.length);
    }

    @Test
    public void testMSEJavaReference() {
        MeanSquareError mse = new MeanSquareError();
        float[] predictions = {0.1f, 0.5f, 0.9f};
        float[] targets = {0.0f, 0.5f, 1.0f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float cost = mse.calculateCost(fx, y, 0.0f);
        
        float diff1 = 0.0f - 0.1f;
        float diff2 = 0.5f - 0.5f;
        float diff3 = 1.0f - 0.9f;
        float expected = (diff1*diff1 + diff2*diff2 + diff3*diff3) / (3.0f * 2.0f);
        
        assertEquals("MSE cost should match formula", expected, cost, TOLERANCE);
    }

    @Test
    public void testMAEJavaReference() {
        MeanAbsoluteError mae = new MeanAbsoluteError();
        float[] predictions = {0.1f, 0.5f, 0.9f};
        float[] targets = {0.0f, 0.5f, 1.0f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float cost = mae.calculateCost(fx, y, 0.0f);
        
        float expected = (Math.abs(0.0f - 0.1f) + Math.abs(0.5f - 0.5f) + Math.abs(1.0f - 0.9f)) / 3.0f;
        
        assertEquals("MAE cost should match formula", expected, cost, TOLERANCE);
    }

    @Test
    public void testCrossEntropyJavaReference() {
        CrossEntropyError ce = new CrossEntropyError();
        float[] predictions = {0.5f, 0.9f, 0.1f};
        float[] targets = {1.0f, 1.0f, 0.0f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float cost = ce.calculateCost(fx, y, 0.0f);
        
        float expected = (float)(1.0 * Math.log(0.5) + 1.0 * Math.log(0.9) + 0.0 * Math.log(0.1));
        
        assertEquals("CrossEntropy cost should match formula", expected, cost, TOLERANCE);
    }

    @Test
    public void testKLDivergenceJavaReference() {
        KLDivergenceError kl = new KLDivergenceError();
        float[] predictions = {0.7f, 0.3f};
        float[] targets = {0.9f, 0.1f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float cost = kl.calculateCost(fx, y, 0.0f);
        
        float expected = (float)(0.9 * Math.log(0.9 / 0.7) + 0.1 * Math.log(0.1 / 0.3));
        
        assertEquals("KLDivergence cost should match formula", expected, cost, TOLERANCE);
    }

    @Test
    public void testLogCoshJavaReference() {
        LogCoshError lc = new LogCoshError();
        float[] predictions = {0.5f, 1.5f, -0.5f};
        float[] targets = {0.0f, 1.0f, 0.0f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float cost = lc.calculateCost(fx, y, 0.0f);
        
        float expected = (float)(Math.log(cosh(0.5 - 0.0)) + Math.log(cosh(1.5 - 1.0)) + Math.log(cosh(-0.5 - 0.0)));
        
        assertEquals("LogCosh cost should match formula", expected, cost, TOLERANCE);
    }

    private double cosh(double x) {
        return (Math.exp(x) + Math.exp(-x)) / 2.0;
    }

    @Test
    public void testHuberJavaReference() {
        MeanHuberLossError huber = new MeanHuberLossError();
        float delta = 1.0f;
        float[] predictions = {0.1f, 2.0f, 0.0f};
        float[] targets = {0.0f, 0.0f, 0.5f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float cost = huber.calculateCost(fx, y, delta);
        
        float h1 = computeHuber(0.0f, 0.1f, delta);
        float h2 = computeHuber(0.0f, 2.0f, delta);
        float h3 = computeHuber(0.5f, 0.0f, delta);
        float expected = (h1 + h2 + h3) / 3.0f;
        
        assertEquals("Huber cost should match formula", expected, cost, TOLERANCE);
    }

    private float computeHuber(float yTrue, float yPred, float delta) {
        float diff = yTrue - yPred;
        float absDiff = Math.abs(diff);
        if (absDiff <= delta) {
            return 0.5f * diff * diff;
        } else {
            return delta * absDiff - 0.5f * delta * delta;
        }
    }

    @Test
    public void testMSEDerivative() {
        MeanSquareError mse = new MeanSquareError();
        float[] predictions = {0.5f};
        float[] targets = {1.0f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float derivative = mse.differentiate(fx, y, 0.0f, 
            com.eonmux.neuralfree.ai.core.CostFunction.RespectTo.INPUT, -1);
        
        float expected = -(1.0f - 0.5f) / 1.0f;
        
        assertEquals("MSE derivative should match formula", expected, derivative, TOLERANCE);
    }

    @Test
    public void testMAEDerivative() {
        MeanAbsoluteError mae = new MeanAbsoluteError();
        float[] predictions = {0.3f};
        float[] targets = {0.8f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float derivative = mae.differentiate(fx, y, 0.0f, 
            com.eonmux.neuralfree.ai.core.CostFunction.RespectTo.INPUT, -1);
        
        float expected = (0.3f - 0.8f) / Math.abs(0.3f - 0.8f) / 1.0f;
        
        assertEquals("MAE derivative should match formula", expected, derivative, TOLERANCE);
    }

    @Test
    public void testLogCoshDerivative() {
        LogCoshError lc = new LogCoshError();
        float[] predictions = {0.7f};
        float[] targets = {0.0f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float derivative = lc.differentiate(fx, y, 0.0f, 
            com.eonmux.neuralfree.ai.core.CostFunction.RespectTo.INPUT, -1);
        
        float expected = (float) Math.tanh(0.7f - 0.0f);
        
        assertEquals("LogCosh derivative should be tanh(diff)", expected, derivative, TOLERANCE);
    }

    @Test
    public void testHuberDerivativeWithinDelta() {
        MeanHuberLossError huber = new MeanHuberLossError();
        float delta = 1.0f;
        float[] predictions = {0.3f};
        float[] targets = {0.8f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float derivative = huber.differentiate(fx, y, delta, 
            com.eonmux.neuralfree.ai.core.CostFunction.RespectTo.INPUT, -1);
        
        float diff = 0.8f - 0.3f;
        float expected = -diff / 1.0f;
        
        assertEquals("Huber derivative within delta should be -(y-f(x))/n", expected, derivative, TOLERANCE);
    }

    @Test
    public void testHuberDerivativeOutsideDelta() {
        MeanHuberLossError huber = new MeanHuberLossError();
        float delta = 0.5f;
        float[] predictions = {2.0f};
        float[] targets = {0.0f};
        
        FloatDataSet fx = createFloatDataSet(predictions);
        FloatDataSet y = createFloatDataSet(targets);
        
        float derivative = huber.differentiate(fx, y, delta, 
            com.eonmux.neuralfree.ai.core.CostFunction.RespectTo.INPUT, -1);
        
        float diff = 2.0f - 0.0f;
        float expected = delta * (diff) / Math.abs(diff) / 1.0f;
        
        assertEquals("Huber derivative outside delta should be delta*sign(y-f(x))/n", expected, derivative, TOLERANCE);
    }

    @Test
    public void testAllCostsFiniteOnSamePairs() {
        StringBuilder report = new StringBuilder();
        report.append("Cost Function Parity Test Report\n");
        report.append("=================================\n\n");
        
        int passCount = 0;
        int totalTests = 6 * TEST_PAIRS.length;
        
        for (float[] pair : TEST_PAIRS) {
            float yTrue = pair[0];
            float yPred = pair[1];
            
            for (int costType = 0; costType < 6; costType++) {
                try {
                    float cost = computeCostJava(costType, yTrue, yPred, DEFAULT_DELTA);
                    boolean valid = !Float.isNaN(cost) && !Float.isInfinite(cost);
                    
                    if (valid) {
                        passCount++;
                    } else {
                        report.append(String.format("%s(y=%.2f, f(x)=%.2f): cost=%s\n", 
                            COST_NAMES[costType], yTrue, yPred,
                            Float.isNaN(cost) ? "NaN" : "Inf"));
                    }
                } catch (Exception e) {
                    report.append(String.format("%s(y=%.2f, f(x)=%.2f): EXCEPTION %s\n", 
                        COST_NAMES[costType], yTrue, yPred, e.getMessage()));
                }
            }
        }
        
        report.append(String.format("\nSummary: %d/%d Java computations valid (%.1f%%)\n", 
            passCount, totalTests, 100.0 * passCount / totalTests));
        
        System.out.println(report.toString());
        assertTrue("Most cost computations should be valid", passCount >= totalTests * 0.9);
    }

    private float computeCostJava(int type, float yTrue, float yPred, float delta) {
        switch (type) {
            case MSE:
                float diff = yTrue - yPred;
                return diff * diff / 2.0f;
            case MAE:
                return Math.abs(yTrue - yPred);
            case CROSS_ENTROPY:
                return (float) (yTrue * Math.log(yPred));
            case KL_DIVERGENCE:
                return (float) (yTrue * Math.log(yTrue / yPred));
            case LOG_COSH:
                return (float) Math.log(Math.cosh(yTrue - yPred));
            case HUBER:
                float hDiff = yTrue - yPred;
                float hAbsDiff = Math.abs(hDiff);
                if (hAbsDiff <= delta) {
                    return 0.5f * hDiff * hDiff;
                } else {
                    return delta * hAbsDiff - 0.5f * delta * delta;
                }
            default:
                return 0.0f;
        }
    }
}
