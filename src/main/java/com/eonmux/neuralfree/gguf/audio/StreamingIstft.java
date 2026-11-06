package com.eonmux.neuralfree.gguf.audio;

import java.util.Arrays;
import java.util.Objects;

/**
 * Streaming inverse STFT of complex spectra to mono PCM, the twin of llama.cpp's
 * {@code mtmd_audio_streaming_istft} (tools/mtmd/mtmd-audio.cpp:1539-1623): one weighted overlap-add with a
 * periodic Hann window and per-sample window-square normalization, feeding {@code hop_length} samples per frame
 * while the buffers hold {@code n_fft}.
 *
 * <ul>
 *   <li>{@link #processFrame} takes one spectrum frame as {@code 2 * (n_fft/2 + 1)} interleaved
 *       {@code (real, imaginary)} bin values, mirrors the negative-frequency bins by conjugation, runs the
 *       inverse FFT and overlap-adds it, and returns the next {@code hop_length} samples. The first
 *       {@code (n_fft - hop) / 2} returned samples in the stream are dropped, matching the centered iSTFT of
 *       torch ({@code center=true}) with the shared iSTFT's own trimming convention
 *       ({@code (n_fft - hop) / 2} here, {@code n_fft / 2} for torch).</li>
 *   <li>{@link #flush} drains the remaining {@code n_fft - hop} overlap samples after the last frame; the
 *       callers trim the tail their reference trims (soprano keeps {@code (frames - 1) * hop} samples,
 *       lfm3a drops another {@code (n_fft - hop) / 2}).</li>
 * </ul>
 *
 * <p>The inverse FFT is the repository FFT convention inverted ({@link Fft}): the shared sin/cos table
 * {@code theta = 2*PI*i/n}, the Cooley-Tukey radix-2 recursion with the exponent {@code exp(+2*PI*i*k/len)},
 * the accumulated {@code 1/n} scaling of the reference recursion ({@code scale = 0.5} per level), and a direct
 * DFT for odd lengths. A returned sample divides by its window-square sum when that sum exceeds {@code 1e-8},
 * and is the raw overlap value otherwise (mtmd-audio.cpp:1597-1602). Not thread-safe; {@link #reset} restores
 * the cold-start state.
 */
public final class StreamingIstft {

    /** Normalization floor of the window-square sum (mtmd-audio.cpp:1597). */
    static final float WINDOW_SUM_FLOOR = 1e-8f;

    private final int nFft;
    private final int hopLength;
    private final int nBins;
    private final float[] sinTable;
    private final float[] cosTable;
    private final float[] hann;
    private final float[] overlap;
    private final float[] windowSum;
    private int paddingToRemove;

    /**
     * @param nFft      transform size
     * @param hopLength samples between frames, at most {@code nFft}
     * @throws IllegalArgumentException when a size is not positive or the hop exceeds the transform size
     */
    public StreamingIstft(int nFft, int hopLength) {
        if (nFft <= 0) {
            throw new IllegalArgumentException("n_fft must be positive, got " + nFft);
        }
        if (hopLength <= 0) {
            throw new IllegalArgumentException("hop_length must be positive, got " + hopLength);
        }
        if (hopLength > nFft) {
            throw new IllegalArgumentException("hop_length " + hopLength + " exceeds n_fft " + nFft
                    + " (mtmd-audio.cpp:1549)");
        }
        this.nFft = nFft;
        this.hopLength = hopLength;
        this.nBins = nFft / 2 + 1;
        this.sinTable = new float[nFft];
        this.cosTable = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            double theta = (2.0 * Math.PI * i) / nFft;
            sinTable[i] = (float) Math.sin(theta);
            cosTable[i] = (float) Math.cos(theta);
        }
        this.hann = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            hann[i] = (float) (0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / nFft)));
        }
        this.overlap = new float[nFft];
        this.windowSum = new float[nFft];
        this.paddingToRemove = (nFft - hopLength) / 2;
    }

    /** Transform size ({@code n_fft}). */
    public int nFft() {
        return nFft;
    }

    /** Samples between frames ({@code hop_length}). */
    public int hopLength() {
        return hopLength;
    }

    /** Spectrum bins per frame ({@code n_fft / 2 + 1}). */
    public int nBins() {
        return nBins;
    }

    /**
     * Overlap-adds one spectrum frame and returns the next samples: {@code hop_length} of them, minus the
     * stream's leading {@code (n_fft - hop) / 2} padding samples while they last.
     *
     * @param frame {@code 2 * nBins()} interleaved {@code (real, imaginary)} bin values; left untouched
     * @return a fresh array of at most {@code hopLength} samples (shorter only while the head padding drains)
     * @throws IllegalArgumentException when the frame does not hold exactly {@code 2 * nBins()} values
     */
    public float[] processFrame(float[] frame) {
        Objects.requireNonNull(frame, "frame");
        if (frame.length != 2 * nBins) {
            throw new IllegalArgumentException("a spectrum frame needs " + (2 * nBins) + " interleaved (real,"
                    + " imaginary) values, got " + frame.length);
        }
        float[] full = new float[2 * nFft];
        for (int j = 0; j < nBins; j++) {
            full[2 * j] = frame[2 * j];
            full[2 * j + 1] = frame[2 * j + 1];
        }
        for (int j = 1; j < nBins - 1; j++) {
            int mirror = nFft - j;
            full[2 * mirror] = frame[2 * j];
            full[2 * mirror + 1] = -frame[2 * j + 1];
        }
        ifft(full);

        for (int j = 0; j < nFft; j++) {
            windowSum[j] += hann[j] * hann[j];
            overlap[j] += full[2 * j] * hann[j];
        }
        return extract(hopLength, true);
    }

    /**
     * Drains the remaining overlap after the last frame: the {@code n_fft - hop} samples the buffers still
     * hold, in hop-sized chunks with the same normalization as {@link #processFrame}, no head trimming.
     */
    public float[] flush() {
        float[] out = new float[0];
        int remaining = nFft - hopLength;
        while (remaining > 0) {
            int chunk = Math.min(remaining, hopLength);
            out = concat(out, extract(chunk, false));
            remaining -= chunk;
        }
        return out;
    }

    /** Clears the overlap, the window sums and the head-padding budget (the cold-start state). */
    public void reset() {
        Arrays.fill(overlap, 0.0f);
        Arrays.fill(windowSum, 0.0f);
        paddingToRemove = (nFft - hopLength) / 2;
    }

    /** One extraction of {@code count} normalized samples and the buffer shift of the reference. */
    private float[] extract(int count, boolean trimHead) {
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            out[i] = windowSum[i] > WINDOW_SUM_FLOOR ? overlap[i] / windowSum[i] : overlap[i];
        }
        shift(count);
        if (!trimHead) {
            return out;
        }
        int toRemove = Math.min(paddingToRemove, out.length);
        paddingToRemove -= toRemove;
        if (toRemove == 0) {
            return out;
        }
        float[] trimmed = new float[out.length - toRemove];
        System.arraycopy(out, toRemove, trimmed, 0, trimmed.length);
        return trimmed;
    }

    /** Moves the buffers left by {@code count} samples and zeroes the freed tail (mtmd-audio.cpp:1605-1612). */
    private void shift(int count) {
        System.arraycopy(overlap, count, overlap, 0, nFft - count);
        Arrays.fill(overlap, nFft - count, nFft, 0.0f);
        System.arraycopy(windowSum, count, windowSum, 0, nFft - count);
        Arrays.fill(windowSum, nFft - count, nFft, 0.0f);
    }

    /**
     * Inverse transform of the {@code 2 * nFft} interleaved view, in place: the recursion of {@link Fft} with
     * the conjugate exponent {@code exp(+2*PI*i*k/len)} and the reference's per-level {@code 0.5} scaling
     * (mtmd-audio.cpp:251-252), which accumulates to {@code 1/n}.
     */
    private void ifft(float[] view) {
        float[] re = new float[nFft];
        float[] im = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            re[i] = view[2 * i];
            im[i] = view[2 * i + 1];
        }
        transform(re, im, nFft);
        for (int i = 0; i < nFft; i++) {
            view[2 * i] = re[i];
            view[2 * i + 1] = im[i];
        }
    }

    /** The recursion behind {@link #ifft}: {@code len} a power-of-two divisor of {@code nFft}, else a DFT. */
    private void transform(float[] re, float[] im, int len) {
        if (len == 1) {
            return;
        }
        int half = len / 2;
        if (len != 2 * half) {
            dft(re, im, len);
            return;
        }
        float[] evenRe = new float[half];
        float[] evenIm = new float[half];
        float[] oddRe = new float[half];
        float[] oddIm = new float[half];
        for (int t = 0; t < half; t++) {
            evenRe[t] = re[2 * t];
            evenIm[t] = im[2 * t];
            oddRe[t] = re[2 * t + 1];
            oddIm[t] = im[2 * t + 1];
        }
        transform(evenRe, evenIm, half);
        transform(oddRe, oddIm, half);
        int step = nFft / len;
        for (int k = 0; k < half; k++) {
            int idx = k * step;
            float wRe = cosTable[idx];
            float wIm = sinTable[idx];
            float reOdd = oddRe[k];
            float imOdd = oddIm[k];
            float prodRe = wRe * reOdd - wIm * imOdd;
            float prodIm = wRe * imOdd + wIm * reOdd;
            re[k] = 0.5f * (evenRe[k] + prodRe);
            im[k] = 0.5f * (evenIm[k] + prodIm);
            re[k + half] = 0.5f * (evenRe[k] - prodRe);
            im[k + half] = 0.5f * (evenIm[k] - prodIm);
        }
    }

    /** Direct DFT with the conjugate exponent, for odd lengths ({@link Fft}'s {@code dft}, sign-flipped). */
    private void dft(float[] re, float[] im, int len) {
        int step = nFft / len;
        float[] outRe = new float[len];
        float[] outIm = new float[len];
        for (int k = 0; k < len; k++) {
            float accRe = 0.0f;
            float accIm = 0.0f;
            for (int t = 0; t < len; t++) {
                int idx = (k * t * step) % nFft;
                float c = cosTable[idx];
                float s = sinTable[idx];
                accRe += re[t] * c - im[t] * s;
                accIm += re[t] * s + im[t] * c;
            }
            outRe[k] = 0.5f * accRe;
            outIm[k] = 0.5f * accIm;
        }
        System.arraycopy(outRe, 0, re, 0, len);
        System.arraycopy(outIm, 0, im, 0, len);
    }

    private static float[] concat(float[] a, float[] b) {
        float[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    @Override
    public String toString() {
        return "StreamingIstft{n_fft=" + nFft + ", hop=" + hopLength + "}";
    }
}
