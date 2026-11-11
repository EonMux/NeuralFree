package com.eonmux.neuralfree.gguf.audio;

import java.util.stream.IntStream;

/**
 * The log-mel computation of llama.cpp's mtmd audio frontend (tools/mtmd/mtmd-audio.cpp:273-565): pad, apply the
 * preemphasis, frame with a Hann window, take the FFT power (or magnitude), fold the bins through the mel
 * filterbank with a floor, take the log, and apply the flavor's normalization (per-feature mean/std, or the
 * whisper clamp of {@code (max - 8, +4) / 4} for the center-padded framings). Static and stateless.
 */
public final class LogMel {

    /** The per-feature std epsilon of the conformer flavor (mtmd-audio.cpp:520). */
    static final float STD_EPS = 1e-5f;
    /** Whisper tail of zeros appended before framing (mtmd-audio.cpp:427). */
    static final int WHISPER_TAIL_SECONDS = 30;
    /** Frames below which one thread computes the spectrogram; also the least frames one parallel range takes. */
    static final int FRAMES_PER_RANGE = 64;

    private LogMel() {
    }

    /**
     * Spectrogram of the real signal {@code samples[0..nSamples)} at the framing {@code params}: the mel-major
     * values and frame count of the result describe every flavor's output.
     *
     * @throws IllegalArgumentException when the filterbank does not match the framing, a sample is not finite, or
     *                                  the input is too short for the padding mode
     */
    public static MelChunk compute(float[] samples, int nSamples, MelFrameParams params,
                                   MelFilterbank filters) {
        return compute(samples, nSamples, params, filters, null);
    }

    /**
     * As {@link #compute(float[], int, MelFrameParams, MelFilterbank)} with an explicit window of
     * {@code params.hannWindowSize()} values (the parakeet window from the GGUF); {@code window} null builds the
     * standard Hann window.
     */
    public static MelChunk compute(float[] samples, int nSamples, MelFrameParams params,
                                   MelFilterbank filters, float[] window) {
        if (window != null && window.length != params.hannWindowSize()) {
            throw new IllegalArgumentException("window override holds " + window.length + " values, expected "
                    + params.hannWindowSize());
        }
        if (params.nMel() != filters.nMel() || params.nFftBins() != filters.nFftBins()) {
            throw new IllegalArgumentException("filterbank " + filters.nMel() + "x" + filters.nFftBins()
                    + " does not match the framing " + params.nMel() + "x" + params.nFftBins());
        }
        if (samples == null || nSamples < 0 || nSamples > samples.length) {
            throw new IllegalArgumentException("n_samples " + nSamples + " outside the array");
        }
        for (int i = 0; i < nSamples; i++) {
            if (!Float.isFinite(samples[i])) {
                throw new IllegalArgumentException("sample " + i + " is not finite (" + samples[i] + ")");
            }
        }
        float[] padded = pad(samples, nSamples, params);
        int nLen = (padded.length - params.frameSize()) / params.hopLength() + 1;
        if (nLen < 1) {
            throw new IllegalArgumentException("not enough samples after padding: " + nSamples + " inputs give "
                    + padded.length + " padded samples for a frame of " + params.frameSize());
        }
        float[] hann = window != null ? window : hannWindow(params);
        float[] data = new float[(int) ((long) params.nMel() * nLen)];
        frameLoop(padded, params, filters, hann, data, nLen);
        int effectiveLen = nSamples / params.hopLength();
        if (params.normPerFeature()) {
            normalizePerFeature(data, params.nMel(), nLen, effectiveLen, params);
        } else if (params.padding() != MelFrameParams.Padding.NONE) {
            whisperClamp(data);
        }
        return new MelChunk(params.nMel(), nLen, data);
    }

    /**
     * Whisper normalization of mel-major {@code data} in place (mtmd-audio.cpp:533-551): values are clamped to
     * {@code max - 8} and mapped by {@code (v + 4) / 4}.
     */
    public static void whisperClamp(float[] data) {
        double mmax = -1e20;
        for (int i = 0; i < data.length; i++) {
            if (data[i] > mmax) {
                mmax = data[i];
            }
        }
        mmax -= 8.0;
        for (int i = 0; i < data.length; i++) {
            if (data[i] < mmax) {
                data[i] = (float) mmax;
            }
            data[i] = (float) ((data[i] + 4.0) / 4.0);
        }
    }

    /** One padded copy of the signal (mtmd-audio.cpp:412-456): the mode's padding, then the preemphasis. */
    private static float[] pad(float[] samples, int nSamples, MelFrameParams params) {
        int frame = params.frameSize();
        switch (params.padding()) {
            case NONE: {
                float[] out = new float[nSamples];
                System.arraycopy(samples, 0, out, 0, nSamples);
                return preemph(out, params);
            }
            case CENTER: {
                int half = frame / 2;
                float[] out = new float[nSamples + 2 * half];
                System.arraycopy(samples, 0, out, half, nSamples);
                return preemph(out, params);
            }
            case WHISPER: {
                int half = frame / 2;
                int tail = WHISPER_TAIL_SECONDS * params.sampleRate();
                if (nSamples < half + 1) {
                    throw new IllegalArgumentException("the whisper framing reflects " + half
                            + " samples at the start, so at least " + (half + 1) + " inputs are required, got "
                            + nSamples);
                }
                float[] out = new float[nSamples + tail + 2 * half];
                // samples[half], samples[half - 1], ..., samples[1]
                for (int i = 0; i < half; i++) {
                    out[i] = samples[half - i];
                }
                System.arraycopy(samples, 0, out, half, nSamples);
                return preemph(out, params);
            }
            default:
                throw new IllegalStateException("unhandled padding " + params.padding());
        }
    }

    /** In-place preemphasis on a padded copy (mtmd-audio.cpp:447-456); coefficient 0.97, frame/2 head skip. */
    private static float[] preemph(float[] padded, MelFrameParams params) {
        float coefficient = params.preemph();
        if (coefficient == 0.0f) {
            return padded;
        }
        int half = params.frameSize() / 2;
        float prev = padded[half];
        for (int i = half + 1; i + half < padded.length; i++) {
            float cur = padded[i];
            padded[i] = cur - coefficient * prev;
            prev = cur;
        }
        return padded;
    }

    /** Hann window of {@code params}, periodic or symmetric (mtmd-audio.cpp:27-33). */
    static float[] hannWindow(MelFrameParams params) {
        int len = params.hannWindowSize();
        int offset = params.hannPeriodic() ? 0 : -1;
        float[] out = new float[len];
        for (int i = 0; i < len; i++) {
            out[i] = (float) (0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (len + offset))));
        }
        return out;
    }

    /**
     * The worker loop of mtmd-audio.cpp:290-366 over the padded signal: windowed frames (the window is centered in
     * the frame when shorter), FFT, power or magnitude, mel fold with the floor, log. Frames are independent and
     * each writes only its own column, so contiguous frame ranges run on the common pool (mtmd-audio.cpp splits
     * frames across its worker threads the same way); every frame computes exactly what a serial loop computes.
     */
    static void frameLoop(float[] padded, MelFrameParams params, MelFilterbank filters, float[] hann,
                                  float[] data, int nLen) {
        Fft fft = new Fft(params.frameSize());
        int ranges = (int) Math.min(Runtime.getRuntime().availableProcessors(),
                (nLen + FRAMES_PER_RANGE - 1) / FRAMES_PER_RANGE);
        if (ranges <= 1) {
            frameRange(padded, params, filters, hann, data, nLen, fft, 0, nLen);
            return;
        }
        IntStream.range(0, ranges).parallel().forEach(r -> frameRange(padded, params, filters, hann, data, nLen,
                fft, (int) ((long) nLen * r / ranges), (int) ((long) nLen * (r + 1) / ranges)));
    }

    /** Frames {@code [from, to)} of {@link #frameLoop}, with work arrays of their own. */
    private static void frameRange(float[] padded, MelFrameParams params, MelFilterbank filters, float[] hann,
                                   float[] data, int nLen, Fft fft, int from, int to) {
        int frame = params.frameSize();
        int hop = params.hopLength();
        int win = params.hannWindowSize();
        boolean leftAligned = params.hannLeftAligned();
        int winPad = leftAligned ? 0 : (frame - win) / 2;
        int nBins = params.nFftBins();
        int nMel = params.nMel();
        float[] fftOut = new float[2 * frame];
        float[] power = new float[nBins];
        float[] real = new float[frame];
        for (int t = from; t < to; t++) {
            int offset = t * hop;
            for (int j = 0; j < frame; j++) {
                real[j] = 0.0f;
            }
            for (int j = 0; j < win; j++) {
                int src = offset + winPad + j;
                if (src >= 0 && src < padded.length) {
                    real[winPad + j] = hann[j] * padded[src];
                }
            }
            fft.forward(real, fftOut);
            boolean finite = true;
            for (int k = 0; k < nBins; k++) {
                float re = fftOut[2 * k];
                float im = fftOut[2 * k + 1];
                float p = re * re + im * im;
                power[k] = params.magnitude() ? (float) Math.sqrt(p) : p;
                finite &= Float.isFinite(power[k]);
            }
            for (int m = 0; m < nMel; m++) {
                // a zero tap adds an exact zero to the sum unless the power is infinite or NaN
                int lo = finite ? filters.firstBin(m) : 0;
                int hi = finite ? filters.endBin(m) : nBins;
                double sum = 0.0;
                for (int k = lo; k < hi; k++) {
                    sum += power[k] * filters.weight(m, k);
                }
                sum = params.floorAdd() ? sum + params.melFloor() : Math.max(sum, params.melFloor());
                data[m * nLen + t] = params.naturalLog() ? (float) Math.log(sum) : (float) Math.log10(sum);
            }
        }
    }

    /**
     * Per-feature normalization of mtmd-audio.cpp:506-532 with the statistics of frames {@code [0, effectiveLen)}:
     * later frames are zeroed, or normalized alike with {@link MelFrameParams#normalizeTail()}.
     */
    private static void normalizePerFeature(float[] data, int nMel, int nLen, int effectiveLen,
                                            MelFrameParams params) {
        if (effectiveLen <= 1) {
            throw new IllegalArgumentException("per-feature normalization needs more than " + effectiveLen
                    + " effective frames");
        }
        for (int m = 0; m < nMel; m++) {
            double mean = 0.0;
            for (int t = 0; t < effectiveLen; t++) {
                mean += data[m * nLen + t];
            }
            mean /= effectiveLen;
            double var = 0.0;
            for (int t = 0; t < effectiveLen; t++) {
                double v = data[m * nLen + t] - mean;
                var += v * v;
            }
            var /= effectiveLen - 1;
            double mstd = params.stdEpsAfterSqrt() ? Math.sqrt(var) + STD_EPS : Math.sqrt(var + STD_EPS);
            int normalized = params.normalizeTail() ? nLen : effectiveLen;
            for (int t = 0; t < normalized; t++) {
                data[m * nLen + t] = (float) ((data[m * nLen + t] - mean) / mstd);
            }
            for (int t = normalized; t < nLen; t++) {
                data[m * nLen + t] = 0.0f;
            }
        }
    }
}
