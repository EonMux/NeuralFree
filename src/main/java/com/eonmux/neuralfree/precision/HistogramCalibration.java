package com.eonmux.neuralfree.precision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Clip-range selection over a {@link CalibrationHistogram}: the PERCENTILE, MSE and ENTROPY calibration methods of
 * {@link Quantizer}. Symmetric methods take a magnitude histogram and return a threshold {@code t} (the clip range
 * {@code [-t, t]}); asymmetric methods take a signed histogram and return {@code {lo, hi}} with
 * {@code lo <= 0 <= hi}, since the affine parameters always widen the range to contain 0. Every result lies within
 * the histogrammed values' range ({@code [0, maxValue]}, or {@code [min(minValue, 0), max(maxValue, 0)]}), and a
 * histogram without values, or whose values are all 0, gives the range {@code {0}}.
 *
 * <p>The MSE and ENTROPY candidate clip points are the bin edges inside the value range plus the range's own ends,
 * so the min-max range is always a candidate. Ties between candidates go to the wider range. PERCENTILE interpolates
 * within a bin instead.</p>
 */
final class HistogramCalibration {
    /**
     * The fewest bins an ENTROPY candidate keeps, as in TensorRT's calibration (fewer when the histogram is smaller).
     */
    static final int MIN_ENTROPY_BINS = 128;

    /** The quantize-dequantize map of one clip range: the value a code of that range reconstructs for {@code x}. */
    @FunctionalInterface
    interface RoundTrip {
        float apply(float x);
    }

    /** The {@link RoundTrip} of the symmetric clip range {@code [-threshold, threshold]}. */
    @FunctionalInterface
    interface SymmetricQuantizer {
        RoundTrip forThreshold(float threshold);
    }

    /** The {@link RoundTrip} of the clip range {@code [lo, hi]}, {@code lo <= 0 <= hi}. */
    @FunctionalInterface
    interface RangeQuantizer {
        RoundTrip forRange(float lo, float hi);
    }

    private HistogramCalibration() {
    }

    /**
     * The {@code q}-quantile of the histogrammed values, {@code 0 <= q <= 1}: the value below which a fraction
     * {@code q} of them lies, interpolated linearly inside the bin that reaches it (values are taken as uniform within
     * a bin) and clamped to {@code [minValue, maxValue]}. {@code q = 0} gives {@code minValue} and {@code q = 1}
     * gives {@code maxValue} exactly; a histogram without values gives 0.
     */
    static float percentile(CalibrationHistogram histogram, double q) {
        if (histogram.total() == 0) {
            return 0.0f;
        }
        if (q <= 0.0) {
            return (float) histogram.minValue();
        }
        if (q >= 1.0) {
            return (float) histogram.maxValue();
        }
        double target = q * histogram.total();
        long before = 0;
        for (int bin = 0; bin < histogram.binCount(); bin++) {
            long count = histogram.count(bin);
            if (count == 0) {
                continue;
            }
            if (before + count >= target) {
                double value = histogram.lowerEdge(bin) + histogram.width() * ((target - before) / count);
                return (float) Math.max(histogram.minValue(), Math.min(histogram.maxValue(), value));
            }
            before += count;
        }
        return (float) histogram.maxValue();
    }

    /**
     * MSE calibration of a symmetric type: the threshold minimizing the expected squared quantization error
     * {@code sum(count_b * (r_b - roundTrip(r_b))^2)} over the bins of a magnitude histogram, with
     * {@link CalibrationHistogram#representative} values {@code r_b}. A negative value is assumed to incur the
     * error of its magnitude, which holds exactly for sign-symmetric codes (FP8) and up to the one extra negative
     * code for the integer types.
     */
    static float symmetricMse(CalibrationHistogram histogram, SymmetricQuantizer quantizer) {
        requireMode(histogram, true);
        double amax = histogram.maxValue();
        if (histogram.total() == 0 || !(amax > 0.0)) {
            return 0.0f;
        }
        Bins bins = new Bins(histogram);
        float best = (float) amax;
        double bestError = bins.error(quantizer.forThreshold(best));
        for (int edge = histogram.binCount(); edge >= 1; edge--) {
            double candidate = histogram.lowerEdge(edge);
            if (!(candidate < amax)) {
                continue;
            }
            float threshold = (float) candidate;
            double error = bins.error(quantizer.forThreshold(threshold));
            if (error < bestError) {
                bestError = error;
                best = threshold;
            }
        }
        return best;
    }

    /**
     * MSE calibration of an asymmetric type: {@code {lo, hi}} minimizing the expected squared quantization error over
     * a signed histogram (as {@link #symmetricMse}). The two-dimensional search is a coordinate descent from the
     * min-max range: the best {@code hi} for the current {@code lo}, then the best {@code lo} for that {@code hi},
     * each an exhaustive scan of the candidate edges, accepting only strict improvements, until neither end moves.
     */
    static float[] asymmetricMse(CalibrationHistogram histogram, RangeQuantizer quantizer) {
        requireMode(histogram, false);
        double low = Math.min(histogram.minValue(), 0.0);
        double high = Math.max(histogram.maxValue(), 0.0);
        if (histogram.total() == 0 || (low == 0.0 && high == 0.0)) {
            return new float[]{0.0f, 0.0f};
        }
        Bins bins = new Bins(histogram);
        float[] lows = lowCandidates(histogram, low);
        float[] highs = highCandidates(histogram, high);
        float lo = (float) low;
        float hi = (float) high;
        double error = bins.error(quantizer.forRange(lo, hi));
        boolean moved = true;
        while (moved) {
            moved = false;
            float bestHi = hi;
            for (float candidate : highs) {
                double e = bins.error(quantizer.forRange(lo, candidate));
                if (e < error) {
                    error = e;
                    bestHi = candidate;
                }
            }
            float bestLo = lo;
            for (float candidate : lows) {
                double e = bins.error(quantizer.forRange(candidate, bestHi));
                if (e < error) {
                    error = e;
                    bestLo = candidate;
                }
            }
            moved = bestHi != hi || bestLo != lo;
            hi = bestHi;
            lo = bestLo;
        }
        return new float[]{lo, hi};
    }

    /**
     * ENTROPY calibration of a symmetric type (TensorRT's KL-divergence calibration) over a magnitude histogram of
     * {@code B} bins. For each candidate bin count {@code i} from {@code min(128, B)} to {@code B}:
     * <ul>
     *   <li>the reference distribution {@code P} is bins {@code [0, i)} with the count of every bin from {@code i} on
     *       (the clipped outliers) added to bin {@code i - 1};</li>
     *   <li>the candidate distribution {@code Q} quantizes bins {@code [0, i)} (without the outliers) into
     *       {@code levels} groups, bin {@code j} in group {@code floor(j * levels / i)}, and expands it back: each
     *       non-empty bin of a group gets the group's mean over its non-empty bins, an empty bin stays 0;</li>
     *   <li>both are normalized and {@code KL(P || Q) = sum(p * ln(p / q))} is taken, infinite where
     *       {@code q = 0 < p}.</li>
     * </ul>
     * The threshold is the upper edge {@code i * w} of the candidate with the smallest divergence (the largest
     * {@code i} on ties), clamped to {@code maxValue}. {@code levels} is {@code 2^(bits - 1)}: 128 for 8-bit codes.
     */
    static float symmetricEntropy(CalibrationHistogram histogram, int levels) {
        requireMode(histogram, true);
        double amax = histogram.maxValue();
        if (histogram.total() == 0 || !(amax > 0.0)) {
            return 0.0f;
        }
        Divergence divergence = new Divergence(histogram, levels);
        int bins = histogram.binCount();
        int best = bins;
        double bestDivergence = Double.POSITIVE_INFINITY;
        for (int end = bins; end >= Math.min(MIN_ENTROPY_BINS, bins); end--) {
            double kl = divergence.of(0, end);
            if (kl < bestDivergence) {
                bestDivergence = kl;
                best = end;
            }
        }
        return (float) Math.min(histogram.lowerEdge(best), amax);
    }

    /**
     * ENTROPY calibration of an asymmetric type over a signed histogram: the bin range {@code [a, b)} containing the
     * zero edge and at least {@code min(128, B)} bins whose divergence (as {@link #symmetricEntropy}, with the outliers
     * below {@code a} added to bin {@code a} and those from {@code b} on to bin {@code b - 1}, and bin {@code j} in
     * group {@code floor((j - a) * levels / (b - a))}) is smallest. The search is a coordinate descent from the full
     * range, as in {@link #asymmetricMse}. The result is the range's edges clamped to the value range;
     * {@code levels} is {@code 2^bits}: 256 for 8-bit codes.
     */
    static float[] asymmetricEntropy(CalibrationHistogram histogram, int levels) {
        requireMode(histogram, false);
        double low = Math.min(histogram.minValue(), 0.0);
        double high = Math.max(histogram.maxValue(), 0.0);
        if (histogram.total() == 0 || (low == 0.0 && high == 0.0)) {
            return new float[]{0.0f, 0.0f};
        }
        Divergence divergence = new Divergence(histogram, levels);
        int bins = histogram.binCount();
        int zero = histogram.zeroBin();
        int minWidth = Math.min(MIN_ENTROPY_BINS, bins);
        int start = 0;
        int end = bins;
        double kl = divergence.of(start, end);
        boolean moved = true;
        while (moved) {
            int bestEnd = end;
            for (int candidate = bins; candidate >= Math.max(zero, start + minWidth); candidate--) {
                double d = divergence.of(start, candidate);
                if (d < kl) {
                    kl = d;
                    bestEnd = candidate;
                }
            }
            int bestStart = start;
            for (int candidate = 0; candidate <= Math.min(zero, bestEnd - minWidth); candidate++) {
                double d = divergence.of(candidate, bestEnd);
                if (d < kl) {
                    kl = d;
                    bestStart = candidate;
                }
            }
            moved = bestEnd != end || bestStart != start;
            end = bestEnd;
            start = bestStart;
        }
        return new float[]{
            (float) Math.max(histogram.lowerEdge(start), low),
            (float) Math.min(histogram.lowerEdge(end), high)};
    }

    /** The lower clip candidates, widest first: {@code low}, then the edges in {@code (low, 0]} ascending. */
    private static float[] lowCandidates(CalibrationHistogram histogram, double low) {
        List<Float> candidates = new ArrayList<>();
        candidates.add((float) low);
        for (int edge = 0; edge <= histogram.binCount(); edge++) {
            double value = histogram.lowerEdge(edge);
            if (value > low && value <= 0.0) {
                candidates.add((float) value);
            }
        }
        return toArray(candidates);
    }

    /** The upper clip candidates, widest first: {@code high}, then the edges in {@code [0, high)} descending. */
    private static float[] highCandidates(CalibrationHistogram histogram, double high) {
        List<Float> candidates = new ArrayList<>();
        candidates.add((float) high);
        for (int edge = histogram.binCount(); edge >= 0; edge--) {
            double value = histogram.lowerEdge(edge);
            if (value < high && value >= 0.0) {
                candidates.add((float) value);
            }
        }
        return toArray(candidates);
    }

    private static float[] toArray(List<Float> values) {
        float[] array = new float[values.size()];
        for (int i = 0; i < array.length; i++) {
            array[i] = values.get(i);
        }
        return array;
    }

    private static void requireMode(CalibrationHistogram histogram, boolean magnitudes) {
        if (histogram.magnitudes() != magnitudes) {
            throw new IllegalArgumentException(magnitudes
                ? "Symmetric calibration needs a magnitude histogram"
                : "Asymmetric calibration needs a signed histogram");
        }
    }

    /** The non-empty bins of a histogram as representative values and counts, for error sums. */
    private static final class Bins {
        private final float[] values;
        private final long[] counts;

        Bins(CalibrationHistogram histogram) {
            int nonEmpty = 0;
            for (int bin = 0; bin < histogram.binCount(); bin++) {
                if (histogram.count(bin) != 0) {
                    nonEmpty++;
                }
            }
            values = new float[nonEmpty];
            counts = new long[nonEmpty];
            int next = 0;
            for (int bin = 0; bin < histogram.binCount(); bin++) {
                if (histogram.count(bin) != 0) {
                    values[next] = (float) histogram.representative(bin);
                    counts[next] = histogram.count(bin);
                    next++;
                }
            }
        }

        /** {@code sum(count * (value - roundTrip(value))^2)} over the non-empty bins. */
        double error(RoundTrip roundTrip) {
            double sum = 0.0;
            for (int i = 0; i < values.length; i++) {
                double difference = (double) values[i] - roundTrip.apply(values[i]);
                sum += counts[i] * difference * difference;
            }
            return sum;
        }
    }

    /** {@code KL(P || Q)} of a bin range of one histogram, as {@link #symmetricEntropy} defines it. */
    private static final class Divergence {
        private final long[] counts;
        private final long[] prefix;
        private final int levels;
        private final double[] groupSums;
        private final int[] groupSizes;

        Divergence(CalibrationHistogram histogram, int levels) {
            if (levels < 1) {
                throw new IllegalArgumentException("ENTROPY calibration needs at least one level, got " + levels);
            }
            this.counts = histogram.counts();
            this.prefix = new long[counts.length + 1];
            for (int i = 0; i < counts.length; i++) {
                prefix[i + 1] = prefix[i] + counts[i];
            }
            this.levels = levels;
            this.groupSums = new double[levels];
            this.groupSizes = new int[levels];
        }

        /** The divergence of the bin range {@code [start, end)}, {@code start < end}. */
        double of(int start, int end) {
            long below = prefix[start];
            long above = prefix[counts.length] - prefix[end];
            long kept = prefix[end] - prefix[start];
            if (kept == 0) {
                return Double.POSITIVE_INFINITY;
            }
            double totalP = below + kept + above;
            double totalQ = kept;
            int width = end - start;
            Arrays.fill(groupSums, 0.0);
            Arrays.fill(groupSizes, 0);
            for (int bin = start; bin < end; bin++) {
                if (counts[bin] != 0) {
                    int group = group(bin - start, width);
                    groupSums[group] += counts[bin];
                    groupSizes[group]++;
                }
            }
            double kl = 0.0;
            for (int bin = start; bin < end; bin++) {
                long p = counts[bin];
                if (bin == start) {
                    p += below;
                }
                if (bin == end - 1) {
                    p += above;
                }
                if (p == 0) {
                    continue;
                }
                if (counts[bin] == 0) {
                    return Double.POSITIVE_INFINITY;
                }
                int group = group(bin - start, width);
                double pNorm = p / totalP;
                double qNorm = (groupSums[group] / groupSizes[group]) / totalQ;
                kl += pNorm * Math.log(pNorm / qNorm);
            }
            return kl;
        }

        private int group(int offset, int width) {
            return (int) ((long) offset * levels / width);
        }
    }
}
