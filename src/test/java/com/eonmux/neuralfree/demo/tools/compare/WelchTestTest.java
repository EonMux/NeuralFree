package com.eonmux.neuralfree.demo.tools.compare;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link WelchTest} against {@code scipy.stats.ttest_ind(b, a, equal_var=False)} and {@code scipy.stats.t.sf}
 * (SciPy 1.18.1).
 */
public class WelchTestTest {

    @Test
    public void aClearShiftMatchesSciPy() {
        WelchTest.Result r = WelchTest.test(new double[]{100, 102, 98, 101, 99},
                new double[]{110, 108, 112, 111, 109});
        assertEquals(10.0, r.t(), 1e-12);
        assertEquals(8.0, r.df(), 1e-12);
        assertEquals(8.488181527628489e-06, r.p(), 1e-15);
    }

    @Test
    public void unequalSizesAndVariancesMatchSciPy() {
        WelchTest.Result r = WelchTest.test(new double[]{480.1, 485.3, 479.9},
                new double[]{481.0, 484.2, 486.5, 483.3});
        assertEquals(0.9435237159591918, r.t(), 1e-12);
        assertEquals(3.589411956978738, r.df(), 1e-12);
        assertEquals(0.4044822679832878, r.p(), 1e-12);
    }

    @Test
    public void theStudentTailMatchesSciPyAtAFractionalDegree() {
        assertEquals(0.12613852257591343, WelchTest.twoSidedP(2.0, 3.5), 1e-13);
        assertEquals(1.0, WelchTest.twoSidedP(0.0, 5.0), 1e-15);
    }

    @Test
    public void constantSetsDecideByTheirMeans() {
        assertEquals(1.0, WelchTest.test(new double[]{5, 5}, new double[]{5, 5, 5}).p(), 0.0);
        WelchTest.Result r = WelchTest.test(new double[]{5, 5}, new double[]{6, 6});
        assertEquals(0.0, r.p(), 0.0);
        assertTrue(r.t() > 0);
    }

    @Test
    public void fewerThanTwoSamplesLeaveNoTest() {
        assertNull(WelchTest.test(new double[]{1}, new double[]{1, 2}));
        assertNull(WelchTest.test(new double[]{1, 2}, new double[0]));
    }

    @Test
    public void aNonFiniteSampleIsRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> WelchTest.test(new double[]{1, Double.NaN}, new double[]{1, 2}));
    }
}
