package com.eonmux.neuralfree.gguf.audio;

import java.util.Locale;

/**
 * Windowed-sinc resampler between PCM sample rates: each output frame is a weighted sum of the input frames under
 * a Hann-windowed sinc kernel of 16 lobes per side, stretched to the rate ratio when downsampling, with the
 * kernel cutoff at the lower of the two Nyquist frequencies, so down- and up-sampling are band-limited. The frame
 * count is {@code ceil(n * target / source)} frames. Static and stateless.
 */
public final class AudioResampler {

    /** Kernel taps per side; the sinc lobes are windowed over {@code [-TAPS, TAPS]}. */
    static final int TAPS = 16;

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
        for (int t = 0; t < out.length; t++) {
            double pos = t * step;
            int center = (int) Math.floor(pos);
            double frac = pos - center;
            double sum = 0.0;
            int span = (int) Math.ceil(halfWidth);
            for (int k = -span; k <= span; k++) {
                int src = center + k;
                if (src < 0 || src >= nIn) {
                    continue;
                }
                double d = k - frac;
                sum += in[src] * kernel(d, halfWidth, cutoff);
            }
            out[t] = (float) sum;
        }
        return new AudioClip(out, targetRate);
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
