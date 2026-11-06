package com.eonmux.neuralfree.gguf.engine.watermark;

/**
 * The survival functions the watermark p-values use: the upper tail of a Gamma distribution with unit scale, at an
 * integer shape (a closed sum, as vLLM's {@code _gamma_survival_integer_shape}) and at a real shape (the regularized
 * upper incomplete gamma, as {@code torch.special.gammaincc}).
 */
final class WatermarkGamma {

    private static final int MAX_ITERATIONS = 100_000;
    private static final double EPSILON = 1e-15;
    private static final double TINY = 1e-300;

    private WatermarkGamma() {
    }

    /**
     * The survival probability of a Gamma(shape, 1) variable at {@code score}, for an integer shape:
     * {@code exp(-score) * sum(score^k / k!, k = 0..shape-1)}, summed in log space so a large score or shape does not
     * overflow. Answers 1 for a non-positive score or shape.
     */
    static double integerSurvival(double score, int shape) {
        if (shape <= 0 || score <= 0) {
            return 1.0;
        }
        double logScore = Math.log(score);
        // the terms rise while score > k and fall after, so the maximum sits where the running log crosses it
        double maxLogTerm = 0.0;
        double logTerm = 0.0;
        for (int index = 1; index < shape; index++) {
            logTerm += logScore - Math.log(index);
            maxLogTerm = Math.max(maxLogTerm, logTerm);
        }
        double sum = 0.0;
        logTerm = 0.0;
        for (int index = 0; index < shape; index++) {
            if (index > 0) {
                logTerm += logScore - Math.log(index);
            }
            sum += Math.exp(logTerm - maxLogTerm);
        }
        double logSum = maxLogTerm + Math.log(sum);
        return Math.min(1.0, Math.exp(Math.min(0.0, -score + logSum)));
    }

    /**
     * The regularized upper incomplete gamma {@code Q(a, x) = 1 - P(a, x)}, by the series below
     * {@code x < a + 1} and the continued fraction above it (Numerical Recipes, {@code gser}/{@code gcf}).
     *
     * @throws IllegalArgumentException when {@code shape} is not positive
     */
    static double upperRegularized(double shape, double x) {
        if (!(shape > 0.0) || Double.isNaN(shape)) {
            throw new IllegalArgumentException("gamma shape must be positive, got " + shape);
        }
        if (Double.isNaN(x) || x <= 0.0) {
            return 1.0;
        }
        double logGamma = logGamma(shape);
        if (x < shape + 1.0) {
            return 1.0 - series(shape, x, logGamma);
        }
        return continuedFraction(shape, x, logGamma);
    }

    /** P(a, x) by its power series (the lower tail, well conditioned for x below a + 1). */
    private static double series(double a, double x, double logGamma) {
        double ap = a;
        double term = 1.0 / a;
        double sum = term;
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (Math.abs(term) < Math.abs(sum) * EPSILON) {
                break;
            }
        }
        return sum * Math.exp(-x + a * Math.log(x) - logGamma);
    }

    /** Q(a, x) by its continued fraction (the upper tail, well conditioned for x at or above a + 1). */
    private static double continuedFraction(double a, double x, double logGamma) {
        double b = x + 1.0 - a;
        double c = 1.0 / TINY;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i < MAX_ITERATIONS; i++) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (Math.abs(d) < TINY) {
                d = TINY;
            }
            c = b + an / c;
            if (Math.abs(c) < TINY) {
                c = TINY;
            }
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1.0) < EPSILON) {
                break;
            }
        }
        return Math.exp(-x + a * Math.log(x) - logGamma) * h;
    }

    /** The logarithm of the gamma function (Lanczos, as the common implementations use it). */
    static double logGamma(double x) {
        double[] coefficients = {
                76.18009172947146, -86.50532032941677, 24.01409824083091,
                -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
        };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double sum = 1.000000000190015;
        for (double coefficient : coefficients) {
            y += 1.0;
            sum += coefficient / y;
        }
        return -tmp + Math.log(2.5066282746310005 * sum / x);
    }
}
