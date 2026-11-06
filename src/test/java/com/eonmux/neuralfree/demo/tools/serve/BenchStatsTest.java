package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The report statistics against numpy's definitions: the mean, the population standard deviation, the
 * median and the linear-interpolated percentiles, including the empty sample the reference prints.
 */
public class BenchStatsTest {

    private static final double EPS = 1e-12;

    @Test
    public void theEmptySampleIsZeroEverywhere() {
        List<Double> none = Collections.emptyList();
        assertEquals(0.0, BenchStats.mean(none), EPS);
        assertEquals(0.0, BenchStats.std(none), EPS);
        assertEquals(0.0, BenchStats.median(none), EPS);
        assertEquals(0.0, BenchStats.percentile(none, 99), EPS);
    }

    @Test
    public void theMeanAndThePopulationStandardDeviationMatchNumpy() {
        List<Double> values = Arrays.asList(1.0, 2.0, 3.0, 4.0);
        assertEquals(2.5, BenchStats.mean(values), EPS);
        // np.std default ddof=0: variance 1.25
        assertEquals(Math.sqrt(1.25), BenchStats.std(values), EPS);
    }

    @Test
    public void aSingleSampleHasNoSpread() {
        assertEquals(0.0, BenchStats.std(Collections.singletonList(7.0)), EPS);
    }

    @Test
    public void theMedianIsTheFiftiethPercentile() {
        assertEquals(2.0, BenchStats.median(Arrays.asList(3.0, 1.0, 2.0)), EPS);
        assertEquals(2.5, BenchStats.median(Arrays.asList(4.0, 1.0, 3.0, 2.0)), EPS);
    }

    @Test
    public void percentilesInterpolateLinearlyBetweenOrderStatistics() {
        List<Double> values = Arrays.asList(4.0, 1.0, 3.0, 2.0);
        assertEquals(1.0, BenchStats.percentile(values, 0), EPS);
        assertEquals(1.75, BenchStats.percentile(values, 25), EPS);
        assertEquals(2.5, BenchStats.percentile(values, 50), EPS);
        assertEquals(4.0, BenchStats.percentile(values, 100), EPS);
        assertEquals(3.97, BenchStats.percentile(values, 99), EPS);
    }

    @Test
    public void aPercentileOutsideTheRangeIsRefused() {
        try {
            BenchStats.percentile(Collections.singletonList(1.0), 101);
            fail("a percentile above 100 must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("[0, 100]"));
        }
        try {
            BenchStats.percentile(Collections.singletonList(1.0), -1);
            fail("a negative percentile must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("[0, 100]"));
        }
    }

    @Test
    public void theGammaDrawHasTheRequestedMean() {
        // shape*scale = 0.5*2 = 1: the boost branch below one and Marsaglia-Tsang above it agree
        Random random = new Random(1234);
        double sum = 0;
        int draws = 20000;
        for (int i = 0; i < draws; i++) {
            sum += ArrivalSchedule.nextGamma(0.5, 2.0, random);
        }
        assertEquals(1.0, sum / draws, 0.05);
    }

    @Test
    public void aNonPositiveGammaShapeIsRefused() {
        try {
            ArrivalSchedule.nextGamma(0.0, 1.0, new Random(1));
            fail("a gamma draw needs a positive shape");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("positive shape"));
        }
    }
}
