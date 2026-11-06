package com.eonmux.neuralfree.demo.tools.serve;

import java.util.Random;

/**
 * The arrival times of a benchmark workload, as vLLM's {@code get_request} draws them: the
 * {@code requestRate} of infinity sends everything at time zero, the burstiness of infinity spaces
 * requests exactly {@code 1/rate} apart, and every other burstiness samples the interval from the
 * gamma distribution of shape {@code burstiness} and scale {@code 1/(rate * burstiness)} — an
 * exponential (Poisson process) at the default burstiness of 1. Without a ramp-up the cumulative
 * delays are rescaled to end at exactly {@code n/rate}, closing the sampling gap the reference
 * closes so a run's throughput reflects its rate; with a ramp-up every request carries its own rate
 * (linear or exponential from {@code startRps} to {@code endRps}) and the delays stay raw. Pure:
 * the schedule is a double array of cumulative seconds.
 */
public final class ArrivalSchedule {

    private final double[] delays;
    private final double[] rates;

    private ArrivalSchedule(double[] delays, double[] rates) {
        this.delays = delays;
        this.rates = rates;
    }

    /**
     * The cumulative send times of {@code count} requests, in seconds from the first send.
     *
     * @param requestRate the requests per second, or infinity to send at once
     * @param burstiness  the gamma shape, or infinity for the constant interval
     * @throws IllegalArgumentException for a non-positive rate (a ramp-up start of 0 is a rate of
     *                                  0 only for the requests the ramp-up still rates) or a
     *                                  non-positive burstiness
     */
    public static ArrivalSchedule of(int count, double requestRate, double burstiness,
                                     String rampUpStrategy, Integer rampUpStartRps, Integer rampUpEndRps,
                                     Random random) {
        if (burstiness <= 0.0 || Double.isNaN(burstiness)) {
            throw new IllegalArgumentException("a positive burstiness factor is expected, got " + burstiness);
        }
        double[] delays = new double[count];
        double[] rates = new double[count];
        for (int i = 0; i < count; i++) {
            double rate = rateAt(i, count, requestRate, rampUpStrategy, rampUpStartRps, rampUpEndRps);
            if (!(rate > 0.0)) {
                throw new IllegalArgumentException("obtained non-positive request rate " + rate
                        + " for request " + i);
            }
            rates[i] = rate;
            if (Double.isInfinite(rate)) {
                delays[i] = 0.0;
            } else if (Double.isInfinite(burstiness)) {
                delays[i] = 1.0 / rate;
            } else {
                delays[i] = nextGamma(burstiness, 1.0 / (rate * burstiness), random);
            }
        }
        for (int i = 1; i < count; i++) {
            delays[i] += delays[i - 1];
        }
        boolean rampUp = rampUpStrategy != null;
        if (!rampUp && !Double.isInfinite(requestRate) && count > 0 && delays[count - 1] != 0.0) {
            double target = count / requestRate;
            double normalize = target / delays[count - 1];
            for (int i = 0; i < count; i++) {
                delays[i] *= normalize;
            }
        }
        return new ArrivalSchedule(delays, rates);
    }

    /** The schedule of a self-timed trace: each request's own cumulative arrival timestamp. */
    public static ArrivalSchedule ofTimestamps(java.util.List<BenchSample> samples) {
        double[] delays = new double[samples.size()];
        double[] rates = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            Double timestamp = samples.get(i).timestamp();
            if (timestamp == null) {
                throw new IllegalArgumentException("--self-timed needs every trace record to carry a"
                        + " timestamp; request " + i + " has none");
            }
            if (timestamp < 0.0) {
                throw new IllegalArgumentException("a trace timestamp cannot be negative, got "
                        + timestamp + " for request " + i);
            }
            delays[i] = timestamp;
            rates[i] = 0.0;
        }
        return new ArrivalSchedule(delays, rates);
    }

    /** The rate request {@code index} of {@code count} sends at: the ramp-up's, else the constant one. */
    private static double rateAt(int index, int count, double requestRate, String rampUpStrategy,
                                 Integer startRps, Integer endRps) {
        if (rampUpStrategy == null) {
            return requestRate;
        }
        double progress = count <= 1 ? 0.0 : (double) index / (count - 1);
        if ("linear".equals(rampUpStrategy)) {
            return startRps + (endRps - startRps) * progress;
        }
        double ratio = (double) endRps / startRps;
        return startRps * Math.pow(ratio, progress);
    }

    /**
     * One gamma draw of {@code shape} and {@code scale}, Marsaglia and Tsang's squeeze for shape
     * {@code >= 1} and the boost {@code gamma(shape) = gamma(shape + 1) * U^(1/shape)} below it. The
     * reference draws from numpy's generator; this draw shares its distribution but not its sequence,
     * so runs are reproducible here, not identical to numpy's.
     */
    static double nextGamma(double shape, double scale, Random random) {
        if (shape <= 0.0) {
            throw new IllegalArgumentException("a gamma draw needs a positive shape, got " + shape);
        }
        if (shape < 1.0) {
            double boost = Math.pow(random.nextDouble(), 1.0 / shape);
            return boost == 0.0 ? 0.0 : nextGamma(shape + 1.0, scale, random) * boost;
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        while (true) {
            double x;
            double v;
            do {
                x = nextNormal(random);
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            double u = random.nextDouble();
            double xsq = x * x;
            if (u < 1.0 - 0.0331 * xsq * xsq) {
                return d * v * scale;
            }
            if (Math.log(u) < 0.5 * xsq + d * (1.0 - v + Math.log(v))) {
                return d * v * scale;
            }
        }
    }

    /** One standard normal draw, Box and Muller's polar form over {@link Random}. */
    private static double nextNormal(Random random) {
        double u;
        double v;
        double s;
        do {
            u = 2.0 * random.nextDouble() - 1.0;
            v = 2.0 * random.nextDouble() - 1.0;
            s = u * u + v * v;
        } while (s == 0.0 || s >= 1.0);
        return u * Math.sqrt(-2.0 * Math.log(s) / s);
    }

    /** The cumulative send time of request {@code index}, in seconds from the first send. */
    public double delayAt(int index) {
        return delays[index];
    }

    /** The rate request {@code index} sends at. */
    public double rateAt(int index) {
        return rates[index];
    }

    /** The number of scheduled requests. */
    public int size() {
        return delays.length;
    }
}
