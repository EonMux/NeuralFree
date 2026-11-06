package com.eonmux.neuralfree.demo.tools;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ImatrixPerplexity}: the log-softmax of llama.cpp {@code log_softmax}, the NLL accumulation and the
 * final estimate {@code PPL = exp(mean NLL) +/- stderr * PPL} over hand-computed rows.
 */
public class ImatrixPerplexityTest {

    @Test
    public void logSoftmaxMatchesTheDistributionOfTheRow() {
        // logits ln(1..4): the softmax is {0.1, 0.2, 0.3, 0.4}
        float[] logits = {(float) Math.log(1), (float) Math.log(2), (float) Math.log(3), (float) Math.log(4)};
        assertEquals(Math.log(0.3), ImatrixPerplexity.logSoftmax(logits, 2), 1e-6);
        assertEquals(Math.log(0.1), ImatrixPerplexity.logSoftmax(logits, 0), 1e-6);
        // a flat row distributes evenly
        assertEquals(Math.log(0.25), ImatrixPerplexity.logSoftmax(new float[]{2, 2, 2, 2}, 1), 1e-6);
    }

    @Test
    public void logSoftmaxRefusesAnEmptyRowAndATokenOutsideIt() {
        assertThrows(IllegalArgumentException.class, () -> ImatrixPerplexity.logSoftmax(new float[0], 0));
        assertThrows(IllegalArgumentException.class, () -> ImatrixPerplexity.logSoftmax(new float[]{1, 2}, 2));
        assertThrows(IllegalArgumentException.class, () -> ImatrixPerplexity.logSoftmax(new float[]{1, 2}, -1));
    }

    @Test
    public void theEstimateFollowsTheReferenceFormula() {
        ImatrixPerplexity ppl = new ImatrixPerplexity();
        // two rows whose winning token carries different probability mass
        float[] row1 = {1, 1, 1, 1};
        float[] row2 = {0, 0, 0, 3, 0};
        ppl.observe(row1, 0);
        ppl.observe(row2, 3);
        assertEquals(2, ppl.count());
        double v1 = -ImatrixPerplexity.logSoftmax(row1, 0);
        double v2 = -ImatrixPerplexity.logSoftmax(row2, 3);
        double mean = (v1 + v2) / 2;
        double[] e = ppl.estimate();
        assertEquals(Math.exp(mean), e[0], 1e-9);
        double variance = (v1 * v1 + v2 * v2) / 2 - mean * mean;
        assertEquals(Math.sqrt(variance) * e[0], e[1], 1e-9);
        String report = ppl.report();
        assertTrue(report, report.startsWith("Final estimate: PPL = "));
    }

    @Test
    public void aSingleRowHasNoVarianceTermAndNoRowsEstimateToNaN() {
        ImatrixPerplexity ppl = new ImatrixPerplexity();
        assertNull(ppl.report());
        double[] none = ppl.estimate();
        assertTrue(Double.isNaN(none[0]));
        assertTrue(Double.isNaN(none[1]));
        ppl.observe(new float[]{1, 2, 3}, 1);
        assertEquals(Math.exp(-ImatrixPerplexity.logSoftmax(new float[]{1, 2, 3}, 1)), ppl.estimate()[0], 1e-9);
        assertNull(ppl.report()); // variance 0 over one row: the reference prints no error term
    }
}
