package com.eonmux.neuralfree.demo.tools;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The row math of perplexity.cpp as {@link PerplexityMath} implements it: softmax and log-softmax against
 * directly computed values, the 16-bit row encoding round trip within its quantization step, the KL row
 * comparison of a model against itself vanishing, and the report statistics.
 */
public class PerplexityMathTest {

    private static final float[] LOGITS = {0.5f, -1.25f, 2.0f, 0.0f, 3.5f, -0.25f};

    @Test
    public void softmaxAtMatchesTheDirectComputation() {
        double sum = 0;
        for (float l : LOGITS) {
            sum += Math.exp(l);
        }
        for (int token = 0; token < LOGITS.length; token++) {
            double expected = Math.exp(LOGITS[token]) / sum;
            assertEquals(expected, PerplexityMath.softmaxAt(LOGITS, token), 1e-6);
        }
    }

    @Test
    public void logSoftmaxIsTheLogOfTheSoftmax() {
        for (int token = 0; token < LOGITS.length; token++) {
            assertEquals(Math.log(PerplexityMath.softmaxAt(LOGITS, token)),
                    PerplexityMath.logSoftmax(LOGITS, token), 1e-6);
        }
    }

    @Test
    public void nearestIntRoundsHalvesToEven() {
        assertEquals(0, PerplexityMath.nearestInt(0.0f));
        assertEquals(0, PerplexityMath.nearestInt(0.4999f));
        assertEquals(0, PerplexityMath.nearestInt(0.5f));
        assertEquals(1, PerplexityMath.nearestInt(0.7f));
        assertEquals(2, PerplexityMath.nearestInt(1.5f));
        assertEquals(2, PerplexityMath.nearestInt(2.5f));
        assertEquals(4, PerplexityMath.nearestInt(3.5f));
        assertEquals(65535, PerplexityMath.nearestInt(65535.0f));
    }

    @Test
    public void theQuantizedRowDecodesToTheLogSoftmaxWithinItsStep() {
        int nVocab = LOGITS.length;
        short[] row = new short[PerplexityMath.rowLength(nVocab)];
        PerplexityMath.quantLogSoftmaxRow(LOGITS, 2, row, 0);
        float scale = PerplexityMath.rowScale(row, 0);
        float minLogProb = PerplexityMath.rowMinLogProb(row, 0);
        assertTrue("a nonzero scale spreads the row", scale > 0);
        for (int i = 0; i < nVocab; i++) {
            float decoded = scale * PerplexityMath.rowEntry(row, 0, i) + minLogProb;
            double expected = PerplexityMath.logSoftmax(LOGITS, i);
            // one quantization step is range/65535 in log-probability
            assertEquals(expected, decoded, (16.0f / 65535.0f) + 1e-5);
        }
        // an odd vocabulary pads the row to an even entry count
        assertEquals(2 * ((nVocab + 1) / 2) + 4, PerplexityMath.rowLength(nVocab));
        assertEquals(10, PerplexityMath.rowLength(5));
    }

    @Test
    public void theKlOfAModelAgainstItselfVanishes() {
        int nVocab = LOGITS.length;
        short[] base = new short[PerplexityMath.rowLength(nVocab)];
        PerplexityMath.quantLogSoftmaxRow(LOGITS, 2, base, 0);
        PerplexityMath.Kld kld = new PerplexityMath.Kld();
        float[] out = new float[2];
        PerplexityMath.klRow(LOGITS, base, 0, 2, kld, out);
        // the quantized base row can differ by one step of range/65535 in log-probability
        assertEquals("the identical distribution diverges by its quantization error only", 0.0, out[0], 1e-4);
        assertEquals("the token probability difference is quantization noise", 0.0, out[1], 1e-4);
        assertEquals("both models pick the same top token", 1, kld.nSameTop);
        assertEquals(1, kld.count);
        assertEquals(0.0, kld.maxPDiff, 1e-4);
    }

    @Test
    public void aShiftedBaseGivesAPositiveDivergenceAndTheWrongTop() {
        int nVocab = LOGITS.length;
        short[] base = new short[PerplexityMath.rowLength(nVocab)];
        // the base model boosts a different token to the top
        float[] baseLogits = LOGITS.clone();
        baseLogits[0] = LOGITS[4] + 5.0f;
        PerplexityMath.quantLogSoftmaxRow(baseLogits, 0, base, 0);
        PerplexityMath.Kld kld = new PerplexityMath.Kld();
        float[] out = new float[2];
        PerplexityMath.klRow(LOGITS, base, 0, 2, kld, out);
        assertTrue("a different distribution diverges positively: " + out[0], out[0] > 1.0);
        assertTrue("the top tokens differ", kld.nSameTop == 0);
        assertTrue("the winning probabilities differ", Math.abs(kld.sumPDiff) > 0.1);
    }

    @Test
    public void theRunningMeansFollowTheReferenceFormulas() {
        PerplexityMath.Kld kld = new PerplexityMath.Kld();
        double[] empty = kld.meanAndUncertainty(4.0, 8.0);
        assertEquals(0.0, empty[0], 0.0);
        assertEquals(0.0, empty[1], 0.0);
        assertEquals(0.0, kld.covariance(1, 2, 3), 0.0);

        kld.count = 20;
        double sum = 10;
        double sum2 = 6;
        double[] mu = kld.meanAndUncertainty(sum, sum2);
        double f = sum / 20;
        double df = sum2 / 20 - f * f;
        assertEquals(f, mu[0], 0.0);
        assertEquals(Math.sqrt(df / 19), mu[1], 1e-12);
        double cov = kld.covariance(100, 50, 260);
        assertEquals((260 / 20.0 - (100 / 20.0) * (50 / 20.0)) / 19.0, cov, 1e-12);
    }

    @Test
    public void percentilesInterpolateTheSortedValues() {
        float[] sorted = {1, 2, 3, 4, 5};
        assertEquals(1.0f, PerplexityMath.percentile(sorted, 0.0f), 0.0);
        assertEquals(5.0f, PerplexityMath.percentile(sorted, 1.0f), 0.0);
        assertEquals(3.0f, PerplexityMath.percentile(sorted, 0.5f), 1e-6);
        assertEquals(1.5f, PerplexityMath.percentile(sorted, 0.125f), 1e-6);
        assertEquals(4.5f, PerplexityMath.percentile(sorted, 0.875f), 1e-6);
    }
}
