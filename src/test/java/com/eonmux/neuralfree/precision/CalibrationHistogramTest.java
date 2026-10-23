package com.eonmux.neuralfree.precision;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link CalibrationHistogram}: the power-of-two grid, and merging that equals building from all values at once.
 */
public class CalibrationHistogramTest {

    /**
     * Signed range {@code [-0.25, 3]}: {@code 2^-10} would need 3329 bins, so the width is {@code 2^-9} and the
     * bins run from grid index -128 to 1536.
     */
    @Test
    public void choosesTheSmallestPowerOfTwoWidthThatFits() {
        CalibrationHistogram h = CalibrationHistogram.of(new float[]{0.5f, -0.25f, 3f, Float.NaN}, false);
        assertEquals(-9, h.widthExponent());
        assertEquals(-128, h.firstIndex());
        assertEquals(1665, h.binCount());
        assertEquals(128, h.zeroBin());
        assertEquals("NaN is skipped", 3, h.total());
        assertEquals(1, h.count(128 + 256));
        assertEquals(1, h.count(0));
        assertEquals(1, h.count(1664));
        assertEquals(-0.25, h.minValue(), 0.0);
        assertEquals(3.0, h.maxValue(), 0.0);
        assertEquals(-0.25, h.lowerEdge(0), 0.0);
        assertEquals(0.0, h.lowerEdge(h.zeroBin()), 0.0);
    }

    /** A magnitude histogram bins {@code |x|} from 0. */
    @Test
    public void magnitudeHistogramStartsAtZero() {
        CalibrationHistogram h = CalibrationHistogram.of(new float[]{-1f, 0.5f, 0.75f}, true);
        assertEquals(0, h.firstIndex());
        assertEquals(-10, h.widthExponent());
        assertEquals(1025, h.binCount());
        assertEquals(0.5, h.minValue(), 0.0);
        assertEquals(1.0, h.maxValue(), 0.0);
        assertEquals(1, h.count(512));
        assertEquals(1, h.count(768));
        assertEquals(1, h.count(1024));
    }

    @Test
    public void allZeroValuesHaveOneBin() {
        CalibrationHistogram h = CalibrationHistogram.of(new float[]{0f, -0f, 0f}, false);
        assertEquals(1, h.binCount());
        assertEquals(3, h.count(0));
        CalibrationHistogram empty = CalibrationHistogram.of(new float[]{Float.NaN}, true);
        assertEquals(0, empty.total());
        assertEquals(1, empty.binCount());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsInfiniteValues() {
        CalibrationHistogram.of(new float[]{1f, Float.NEGATIVE_INFINITY}, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMergingDifferentModes() {
        CalibrationHistogram.of(new float[]{1f}, false).merge(CalibrationHistogram.of(new float[]{1f}, true));
    }

    /**
     * Merging rebins to the wider grid exactly: any order and grouping of the observations gives the histogram of all
     * values built at once, in both modes, including an all-zero and an empty observation.
     */
    @Test
    public void mergingDoesNotDependOnOrder() {
        Random random = new Random(7);
        float[] narrow = values(random, 500, 0.01f, 0f);
        float[] wide = values(random, 700, 40f, -10f);
        float[] shifted = values(random, 300, 3f, 5f);
        float[] zeros = {0f, 0f, -0f};
        float[] none = {Float.NaN};
        for (boolean magnitudes : new boolean[]{false, true}) {
            CalibrationHistogram all = CalibrationHistogram.of(concat(narrow, wide, shifted, zeros, none), magnitudes);
            CalibrationHistogram a = CalibrationHistogram.of(narrow, magnitudes);
            CalibrationHistogram b = CalibrationHistogram.of(wide, magnitudes);
            CalibrationHistogram c = CalibrationHistogram.of(shifted, magnitudes);
            CalibrationHistogram z = CalibrationHistogram.of(zeros, magnitudes);
            CalibrationHistogram n = CalibrationHistogram.of(none, magnitudes);
            assertSameHistogram(all, z.merge(a).merge(b).merge(c).merge(n));
            assertSameHistogram(all, c.merge(b).merge(n).merge(a).merge(z));
            assertSameHistogram(all, n.merge(b.merge(z)).merge(a.merge(c)));
        }
    }

    /** Rebinning keeps every count: a widening merge still holds every value in the bin of its own grid index. */
    @Test
    public void wideningMergeKeepsEveryValueInItsBin() {
        CalibrationHistogram small = CalibrationHistogram.of(new float[]{0.001f, -0.002f, 0.75f}, false);
        CalibrationHistogram merged = small.merge(CalibrationHistogram.of(new float[]{1000f}, false));
        assertTrue(merged.widthExponent() > small.widthExponent());
        assertEquals(4, merged.total());
        long sum = 0;
        for (long count : merged.counts()) {
            sum += count;
        }
        assertEquals(4, sum);
        long[] counts = merged.counts();
        double w = merged.width();
        assertEquals(1, counts[(int) (Math.floor(-0.002f / w) - merged.firstIndex())]);
        assertEquals(1, counts[(int) (Math.floor(1000f / w) - merged.firstIndex())]);
    }

    private static void assertSameHistogram(CalibrationHistogram expected, CalibrationHistogram actual) {
        assertEquals(expected.widthExponent(), actual.widthExponent());
        assertEquals(expected.firstIndex(), actual.firstIndex());
        assertEquals(expected.total(), actual.total());
        assertEquals(expected.minValue(), actual.minValue(), 0.0);
        assertEquals(expected.maxValue(), actual.maxValue(), 0.0);
        assertArrayEquals(expected.counts(), actual.counts());
    }

    private static float[] values(Random random, int n, float spread, float offset) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = offset + spread * (random.nextFloat() - 0.5f);
        }
        return v;
    }

    private static float[] concat(float[]... parts) {
        int length = 0;
        for (float[] part : parts) {
            length += part.length;
        }
        float[] all = new float[length];
        int at = 0;
        for (float[] part : parts) {
            System.arraycopy(part, 0, all, at, part.length);
            at += part.length;
        }
        return all;
    }
}
