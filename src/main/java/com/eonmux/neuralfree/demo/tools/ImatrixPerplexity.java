package com.eonmux.neuralfree.demo.tools;

import java.util.Locale;

/**
 * The perplexity estimate of the calibration corpus ({@code process_logits} of llama.cpp
 * {@code tools/imatrix/imatrix.cpp}): per predicted token the negative log-softmax of the model's logits row,
 * accumulated as the NLL sum and the NLL square sum over the predicted rows. {@link #estimate} returns the
 * reference's final estimate, the exponential mean NLL with its standard error scaled by it.
 */
final class ImatrixPerplexity {

    private double nll;
    private double nll2;
    private long count;

    /** Accounts one logits row and the token it must predict. */
    void observe(float[] logits, int token) {
        double v = -logSoftmax(logits, token);
        nll += v;
        nll2 += v * v;
        count++;
    }

    /** The rows accounted so far. */
    long count() {
        return count;
    }

    /** The cumulative estimate so far, {@code exp(mean NLL)}; NaN without rows. */
    double runningPpl() {
        return count == 0 ? Double.NaN : Math.exp(nll / count);
    }

    /**
     * The final estimate {@code {ppl, error}}: {@code exp(nll/count)} and the reference's error term, the
     * standard deviation of the per-token NLL over {@code count - 1} scaled by the PPL. The error is NaN when
     * the variance is not positive (too few rows or a degenerate distribution), the state the reference reports
     * instead of an error number.
     */
    double[] estimate() {
        if (count == 0) {
            return new double[]{Double.NaN, Double.NaN};
        }
        double meanNll = nll / count;
        double ppl = Math.exp(meanNll);
        double variance = nll2 / count - meanNll * meanNll;
        double error = variance > 0.0 && count > 1 ? Math.sqrt(variance / (count - 1)) * ppl : Double.NaN;
        return new double[]{ppl, error};
    }

    /** The final estimate as the reference prints it, or {@code null} without a positive variance. */
    String report() {
        double[] e = estimate();
        if (Double.isNaN(e[1])) {
            return null;
        }
        return String.format(Locale.ROOT, "Final estimate: PPL = %.4f +/- %.5f", e[0], e[1]);
    }

    /**
     * The log-softmax of {@code token} under {@code logits}, the maximum subtracted before the exponentials
     * (llama.cpp {@code log_softmax}); an empty row has no distribution and fails the run.
     *
     * @throws IllegalArgumentException when {@code logits} is empty or {@code token} is outside the row
     */
    static double logSoftmax(float[] logits, int token) {
        if (logits.length == 0) {
            throw new IllegalArgumentException("log softmax of an empty logits row");
        }
        if (token < 0 || token >= logits.length) {
            throw new IllegalArgumentException("token " + token + " outside the " + logits.length + "-token row");
        }
        float max = logits[0];
        for (float v : logits) {
            if (v > max) {
                max = v;
            }
        }
        double sumExp = 0.0;
        for (float v : logits) {
            sumExp += Math.exp(v - max);
        }
        return logits[token] - max - Math.log(sumExp);
    }
}
