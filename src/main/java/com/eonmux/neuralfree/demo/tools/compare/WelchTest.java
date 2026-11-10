package com.eonmux.neuralfree.demo.tools.compare;

/**
 * Welch's unequal-variance t-test, the two-sided p-value of the hypothesis that two sample sets share their mean, as
 * {@code scipy.stats.ttest_ind(new, base, equal_var=False)} computes it: the statistic over the samples' standard
 * errors, the Welch–Satterthwaite degrees of freedom, and the Student-t tail through the regularized incomplete beta
 * function. Stateless.
 */
public final class WelchTest {

    private static final int MAX_ITERATIONS = 300;
    private static final double EPS = 1.0e-15;
    private static final double TINY = 1.0e-300;
    private static final double[] LANCZOS = {
        676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    };

    private WelchTest() {
    }

    /** The test's statistic, degrees of freedom and two-sided p-value. Immutable. */
    public static final class Result {
        private final double t;
        private final double df;
        private final double p;

        Result(double t, double df, double p) {
            this.t = t;
            this.df = df;
            this.p = p;
        }

        /** The t statistic of {@code mean(b) - mean(a)}. */
        public double t() {
            return t;
        }

        /** The Welch–Satterthwaite degrees of freedom, infinite when both sets are constant. */
        public double df() {
            return df;
        }

        /** The two-sided p-value. */
        public double p() {
            return p;
        }
    }

    /**
     * The test of {@code b} against {@code a}, or null when either set holds fewer than two samples, which leaves no
     * variance to estimate. Two constant sets give p = 1 when their means agree and p = 0 otherwise.
     *
     * @throws IllegalArgumentException when a sample is not finite
     */
    public static Result test(double[] a, double[] b) {
        if (a.length < 2 || b.length < 2) {
            return null;
        }
        double ma = mean(a);
        double mb = mean(b);
        double sa = variance(a, ma) / a.length;
        double sb = variance(b, mb) / b.length;
        double se2 = sa + sb;
        if (se2 == 0.0) {
            return new Result(ma == mb ? 0.0 : Math.copySign(Double.POSITIVE_INFINITY, mb - ma),
                    Double.POSITIVE_INFINITY, ma == mb ? 1.0 : 0.0);
        }
        double t = (mb - ma) / Math.sqrt(se2);
        double df = se2 * se2 / (sa * sa / (a.length - 1) + sb * sb / (b.length - 1));
        return new Result(t, df, twoSidedP(t, df));
    }

    /** The two-sided tail probability of Student's t with {@code df} degrees of freedom beyond {@code |t|}. */
    static double twoSidedP(double t, double df) {
        if (!(df > 0)) {
            throw new IllegalArgumentException("degrees of freedom " + df);
        }
        double x = df / (df + t * t);
        return Math.min(1.0, regularizedBeta(x, df / 2.0, 0.5));
    }

    /** The mean of {@code v}. */
    static double mean(double[] v) {
        double sum = 0;
        for (double x : v) {
            if (!Double.isFinite(x)) {
                throw new IllegalArgumentException("sample " + x + " is not finite");
            }
            sum += x;
        }
        return sum / v.length;
    }

    /** The unbiased sample variance of {@code v} around {@code mean}. */
    static double variance(double[] v, double mean) {
        double sum = 0;
        for (double x : v) {
            double d = x - mean;
            sum += d * d;
        }
        return sum / (v.length - 1);
    }

    /** The regularized incomplete beta function {@code I_x(a, b)}, by its continued fraction. */
    static double regularizedBeta(double x, double a, double b) {
        if (x <= 0.0) {
            return 0.0;
        }
        if (x >= 1.0) {
            return 1.0;
        }
        double lnFront = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log1p(-x);
        double front = Math.exp(lnFront);
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return front * betaFraction(x, a, b) / a;
        }
        return 1.0 - front * betaFraction(1.0 - x, b, a) / b;
    }

    /** The continued fraction of the incomplete beta function, by the modified Lentz method. */
    private static double betaFraction(double x, double a, double b) {
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (Math.abs(d) < TINY) {
            d = TINY;
        }
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= MAX_ITERATIONS; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            d = Math.abs(d) < TINY ? TINY : d;
            c = 1.0 + aa / c;
            c = Math.abs(c) < TINY ? TINY : c;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            d = Math.abs(d) < TINY ? TINY : d;
            c = 1.0 + aa / c;
            c = Math.abs(c) < TINY ? TINY : c;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1.0) < EPS) {
                return h;
            }
        }
        return h;
    }

    /** The natural logarithm of the gamma function for {@code x > 0}, by the Lanczos approximation (g = 7). */
    static double logGamma(double x) {
        if (x < 0.5) {
            return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1.0 - x);
        }
        double z = x - 1.0;
        double sum = 0.99999999999980993;
        for (int i = 0; i < LANCZOS.length; i++) {
            sum += LANCZOS[i] / (z + i + 1.0);
        }
        double t = z + LANCZOS.length - 0.5;
        return 0.5 * Math.log(2.0 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
    }
}
