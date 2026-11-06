package com.eonmux.neuralfree.demo.tools.serve;

import java.util.Arrays;
import java.util.List;

/**
 * The sample statistics of the benchmark reports, with numpy's conventions: the mean and the
 * population standard deviation ({@code ddof=0}), the median as the 50th percentile, and the
 * percentile with linear interpolation between the order statistics. An empty sample is 0 for
 * every statistic, the value the reference's {@code np.mean(x or 0)} prints when no request
 * succeeded.
 */
public final class BenchStats {

    private BenchStats() {
    }

    /** The arithmetic mean of the sample, 0 when it is empty. */
    public static double mean(List<Double> values) {
        if (values.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    /** The population standard deviation ({@code ddof=0}), 0 when the sample is empty or of size one. */
    public static double std(List<Double> values) {
        if (values.size() < 2) {
            return 0.0;
        }
        double mean = mean(values);
        double squares = 0.0;
        for (double v : values) {
            double d = v - mean;
            squares += d * d;
        }
        return Math.sqrt(squares / values.size());
    }

    /** The 50th percentile. */
    public static double median(List<Double> values) {
        return percentile(values, 50.0);
    }

    /**
     * The {@code p}-th percentile ({@code 0 <= p <= 100}) with numpy's linear interpolation between the
     * order statistics, 0 when the sample is empty.
     */
    public static double percentile(List<Double> values, double p) {
        if (values.isEmpty()) {
            return 0.0;
        }
        if (p < 0.0 || p > 100.0) {
            throw new IllegalArgumentException("a percentile must be in [0, 100], got " + p);
        }
        double[] sorted = new double[values.size()];
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = values.get(i);
        }
        Arrays.sort(sorted);
        double position = (p / 100.0) * (sorted.length - 1);
        int low = (int) Math.floor(position);
        int high = (int) Math.ceil(position);
        if (low == high) {
            return sorted[low];
        }
        double fraction = position - low;
        return sorted[low] + (sorted[high] - sorted[low]) * fraction;
    }
}
