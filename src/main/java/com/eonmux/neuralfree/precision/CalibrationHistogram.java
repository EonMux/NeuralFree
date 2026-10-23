package com.eonmux.neuralfree.precision;

import java.util.Objects;

/**
 * An immutable histogram of calibration values on a power-of-two grid, in one of two modes: the signed values (for
 * the asymmetric types) or their magnitudes {@code |x|} (for the symmetric types, whose clip ranges are symmetric).
 *
 * <p><b>Grid.</b> Bins have the width {@code w = 2^k} and are aligned to multiples of it: a value {@code v} falls in
 * the bin of grid index {@code floor(v / w)}, which covers {@code [index * w, (index + 1) * w)}. The histogram spans
 * the bins from the one holding {@code min(minValue, 0)} to the one holding {@code max(maxValue, 0)}, so 0 is always a
 * bin edge inside the range, and {@code k} is the smallest exponent that keeps that span within {@link #MAX_BINS}
 * bins. A span of {@code 2^k}-wide bins over the data therefore holds between 1025 and 2048 bins whenever a value is
 * not 0. A histogram whose values are all 0 (or that holds none) has one bin.</p>
 *
 * <p><b>Merging.</b> {@link #merge} rebins both histograms to the wider grid by merging adjacent pairs of bins
 * ({@code floor(index / 2)} per doubling, an exact operation) and adds the counts. Because the chosen exponent is the
 * smallest that fits the merged range, merging gives exactly the histogram of all the merged values built at once:
 * the result does not depend on the order or grouping of the merged observations.</p>
 */
final class CalibrationHistogram {
    /** The largest number of bins a histogram spans. */
    static final int MAX_BINS = 2048;

    /** The width exponent of a histogram whose values are all 0: below every double exponent, so any data widens it. */
    private static final int ALL_ZERO_EXPONENT = -1100;

    private final boolean magnitudes;
    private final int widthExponent;
    private final long firstIndex;
    private final long[] counts;
    private final long total;
    private final double minValue;
    private final double maxValue;

    private CalibrationHistogram(boolean magnitudes, int widthExponent, long firstIndex, long[] counts, long total,
                                 double minValue, double maxValue) {
        this.magnitudes = magnitudes;
        this.widthExponent = widthExponent;
        this.firstIndex = firstIndex;
        this.counts = counts;
        this.total = total;
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    /**
     * The histogram of the non-NaN entries of {@code values}, or of their magnitudes when {@code magnitudes} is set.
     *
     * @throws IllegalArgumentException when a value is infinite, since a histogram range must be finite
     */
    static CalibrationHistogram of(float[] values, boolean magnitudes) {
        Objects.requireNonNull(values, "Values cannot be null");
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        long n = 0;
        for (float value : values) {
            if (Float.isNaN(value)) {
                continue;
            }
            if (Float.isInfinite(value)) {
                throw new IllegalArgumentException("Histogram calibration needs finite values, found " + value);
            }
            double v = magnitudes ? Math.abs((double) value) : value;
            min = Math.min(min, v);
            max = Math.max(max, v);
            n++;
        }
        double low = Math.min(min, 0.0);
        double high = Math.max(max, 0.0);
        int exponent = fitExponent(low, high, ALL_ZERO_EXPONENT);
        long first = gridIndex(low, exponent);
        long[] binCounts = new long[Math.toIntExact(gridIndex(high, exponent) - first + 1)];
        for (float value : values) {
            if (!Float.isNaN(value)) {
                double v = magnitudes ? Math.abs((double) value) : value;
                binCounts[Math.toIntExact(gridIndex(v, exponent) - first)]++;
            }
        }
        return new CalibrationHistogram(magnitudes, exponent, first, binCounts, n, min, max);
    }

    /**
     * The histogram of the values of this histogram and of {@code other} together, on the smallest grid that fits
     * both ranges (see the class description).
     *
     * @throws IllegalArgumentException when the two histograms are not of the same mode
     */
    CalibrationHistogram merge(CalibrationHistogram other) {
        Objects.requireNonNull(other, "Histogram cannot be null");
        if (other.magnitudes != magnitudes) {
            throw new IllegalArgumentException("Cannot merge a magnitude histogram with a signed histogram");
        }
        double min = Math.min(minValue, other.minValue);
        double max = Math.max(maxValue, other.maxValue);
        double low = Math.min(min, 0.0);
        double high = Math.max(max, 0.0);
        int exponent = fitExponent(low, high, Math.max(widthExponent, other.widthExponent));
        long first = gridIndex(low, exponent);
        long[] merged = new long[Math.toIntExact(gridIndex(high, exponent) - first + 1)];
        addRebinned(merged, first, exponent);
        other.addRebinned(merged, first, exponent);
        return new CalibrationHistogram(magnitudes, exponent, first, merged, total + other.total, min, max);
    }

    private void addRebinned(long[] target, long targetFirst, int targetExponent) {
        int shift = targetExponent - widthExponent;
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] != 0) {
                long index = firstIndex + i;
                long rebinned = shift >= Long.SIZE - 1 ? index >> (Long.SIZE - 1) : index >> shift;
                target[Math.toIntExact(rebinned - targetFirst)] += counts[i];
            }
        }
    }

    /**
     * The smallest exponent {@code k >= atLeast} for which the grid indices of {@code low <= 0} and {@code high >= 0}
     * span at most {@link #MAX_BINS} bins; {@code atLeast} when both are 0.
     */
    private static int fitExponent(double low, double high, int atLeast) {
        if (low == 0.0 && high == 0.0) {
            return atLeast;
        }
        int exponent = Math.max(atLeast, Math.getExponent(Math.max(-low, high)) - 11);
        while (gridIndex(high, exponent) - gridIndex(low, exponent) + 1 > MAX_BINS) {
            exponent++;
        }
        return exponent;
    }

    /** {@code floor(value / 2^exponent)}, exact since the division is a power-of-two scaling. */
    private static long gridIndex(double value, int exponent) {
        if (value == 0.0) {
            return 0L;
        }
        return (long) Math.floor(Math.scalb(value, -exponent));
    }

    /** True for a histogram of magnitudes {@code |x|}, false for one of signed values. */
    boolean magnitudes() {
        return magnitudes;
    }

    /** The number of bins. */
    int binCount() {
        return counts.length;
    }

    /** The count of bin {@code bin}, {@code 0 <= bin < binCount()}. */
    long count(int bin) {
        return counts[bin];
    }

    /** A copy of the bin counts. */
    long[] counts() {
        return counts.clone();
    }

    /** The number of histogrammed values. */
    long total() {
        return total;
    }

    /** The exponent {@code k} of the bin width {@code 2^k}. */
    int widthExponent() {
        return widthExponent;
    }

    /** The grid index of bin 0, {@code <= 0}. */
    long firstIndex() {
        return firstIndex;
    }

    /** The bin whose lower edge is 0. */
    int zeroBin() {
        return Math.toIntExact(-firstIndex);
    }

    /** The bin width {@code 2^k}; 0 for a histogram whose values are all 0. */
    double width() {
        return Math.scalb(1.0, widthExponent);
    }

    /**
     * The lower edge of bin {@code bin}, {@code 0 <= bin <= binCount()}; {@code bin == binCount()} gives the upper
     * end.
     */
    double lowerEdge(int bin) {
        return Math.scalb((double) (firstIndex + bin), widthExponent);
    }

    /** The smallest histogrammed value ({@code +Infinity} when there is none). */
    double minValue() {
        return minValue;
    }

    /** The largest histogrammed value ({@code -Infinity} when there is none). */
    double maxValue() {
        return maxValue;
    }

    /**
     * The value that stands for the entries of bin {@code bin} in error estimates: the midpoint of the part of the
     * bin inside {@code [minValue, maxValue]}.
     */
    double representative(int bin) {
        double low = Math.max(lowerEdge(bin), minValue);
        double high = Math.min(lowerEdge(bin + 1), maxValue);
        return low + (high - low) / 2;
    }
}
