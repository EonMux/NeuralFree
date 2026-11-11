package com.eonmux.neuralfree.gguf.audio;

import java.util.Locale;

/**
 * Windowed-sinc resampler between PCM sample rates: each output frame is a weighted sum of the input frames under
 * a Hann-windowed sinc kernel of 16 lobes per side, stretched to the rate ratio when downsampling, with the
 * kernel cutoff at the lower of the two Nyquist frequencies, so down- and up-sampling are band-limited. The frame
 * count is {@code ceil(n * target / source)} frames. Static and stateless.
 *
 * <p>With the rates reduced to {@code source / target = down / up}, output frame {@code t} sits at input position
 * {@code t * down / up}, whose fractional part is one of {@code up} phases {@code p / up}. The kernel weights of
 * every phase are tabulated once per call (polyphase form), so each output frame is a dot product over the table
 * row of its phase instead of a sine and cosine per tap. The frame's center and phase come from the same
 * floating-point position as the direct form, so the taps line up exactly; a weight differs from the directly
 * evaluated one only by the rounding of the position, which is exact for rates whose ratio is a power-of-two
 * fraction or an integer (48 to 16 kHz, 8 to 16 kHz). Rate pairs with more than {@link #MAX_PHASES} phases, and
 * clips shorter than their table, evaluate the kernel directly.</p>
 */
public final class AudioResampler {

    /** Kernel taps per side; the sinc lobes are windowed over {@code [-TAPS, TAPS]}. */
    static final int TAPS = 16;
    /** The most phases a weight table holds (44.1 to 16 kHz needs 160, 11.025 to 16 kHz 640). */
    static final int MAX_PHASES = 4096;
    /** How far {@code frac * up} may sit from its phase before the frame is evaluated directly. */
    private static final double PHASE_SLACK = 1e-3;

    private AudioResampler() {
    }

    /**
     * The signal resampled to {@code targetRate} Hz; the input is returned unchanged when the rates match.
     *
     * @throws IllegalArgumentException when a rate is not positive
     */
    public static AudioClip resample(AudioClip clip, int targetRate) {
        if (targetRate <= 0) {
            throw new IllegalArgumentException("target rate must be positive, got " + targetRate);
        }
        if (clip.sampleRate() == targetRate) {
            return clip;
        }
        int nIn = clip.length();
        long nOut = ((long) nIn * targetRate + clip.sampleRate() - 1) / clip.sampleRate();
        if (nOut == 0) {
            return AudioClip.silent(0, targetRate);
        }
        float[] in = clip.samples();
        float[] out = new float[(int) nOut];
        // Position in input frames of output frame t: t * source / target.
        double step = (double) clip.sampleRate() / targetRate;
        // Kernel half-width in input frames: TAPS scaled by the stretch factor (larger when downsampling).
        double halfWidth = TAPS * Math.max(1.0, step);
        double cutoff = 0.5 / Math.max(1.0, step);
        int span = (int) Math.ceil(halfWidth);
        int up = targetRate / gcd(clip.sampleRate(), targetRate);
        double[][] table = up <= MAX_PHASES && up < nOut ? phaseTable(up, span, halfWidth, cutoff) : null;
        for (int t = 0; t < out.length; t++) {
            double pos = t * step;
            int center = (int) Math.floor(pos);
            double frac = pos - center;
            double scaled = frac * up;
            long phase = Math.round(scaled);
            double sum = table != null && Math.abs(scaled - phase) <= PHASE_SLACK
                    ? dot(in, center, span, table[(int) phase])
                    : direct(in, center, frac, span, halfWidth, cutoff);
            out[t] = (float) sum;
        }
        return new AudioClip(out, targetRate);
    }

    /**
     * The kernel weights of phases {@code 0..up} (the last one is a whole frame, where the position's rounding put
     * the center one frame early), each over taps {@code -span..span}.
     */
    private static double[][] phaseTable(int up, int span, double halfWidth, double cutoff) {
        double[][] table = new double[up + 1][2 * span + 1];
        for (int p = 0; p <= up; p++) {
            double frac = (double) p / up;
            for (int k = -span; k <= span; k++) {
                table[p][k + span] = kernel(k - frac, halfWidth, cutoff);
            }
        }
        return table;
    }

    /** One output frame from a tabulated phase: the input frames around {@code center} under {@code weights}. */
    private static double dot(float[] in, int center, int span, double[] weights) {
        int lo = Math.max(-span, -center);
        int hi = Math.min(span, in.length - 1 - center);
        double sum = 0.0;
        for (int k = lo; k <= hi; k++) {
            sum += in[center + k] * weights[k + span];
        }
        return sum;
    }

    /** One output frame with the kernel evaluated at every tap. */
    private static double direct(float[] in, int center, double frac, int span, double halfWidth, double cutoff) {
        double sum = 0.0;
        for (int k = -span; k <= span; k++) {
            int src = center + k;
            if (src < 0 || src >= in.length) {
                continue;
            }
            sum += in[src] * kernel(k - frac, halfWidth, cutoff);
        }
        return sum;
    }

    private static int gcd(int a, int b) {
        while (b != 0) {
            int r = a % b;
            a = b;
            b = r;
        }
        return a;
    }

    /** Hann-windowed sinc at input-frame distance {@code d} with half-width {@code w} and cutoff {@code fc}. */
    private static double kernel(double d, double w, double fc) {
        if (d == 0.0) {
            return 2.0 * fc;
        }
        double sinc = Math.sin(2.0 * Math.PI * fc * d) / (Math.PI * d);
        double x = d / w;
        double hann = 0.5 * (1.0 + Math.cos(Math.PI * x));
        return sinc * hann;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "AudioResampler{taps=%d}", TAPS);
    }
}
