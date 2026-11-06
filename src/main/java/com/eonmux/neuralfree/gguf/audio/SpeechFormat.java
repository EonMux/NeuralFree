package com.eonmux.neuralfree.gguf.audio;

import java.util.Objects;
import java.util.function.Consumer;

/**
 * Frame geometry and rate of a spectral-to-waveform speech decoder: {@code n_fft}, {@code hop_length} and the
 * sample rate. A wavtokenizer-dec model carries them as the optional GGUF keys {@code audio.sample_rate},
 * {@code audio.n_fft} and {@code audio.hop_length} (all three or none; llama.cpp's wavtokenizer-dec loader
 * defines none of them, so a file without them cannot synthesize a waveform by itself).
 *
 * <p>{@link #decode} turns one spectrum row per frame into PCM: the shared streaming iSTFT
 * ({@link StreamingIstft}), then the centered-iSTFT trimming of torch ({@code center=True}, the convention the
 * references decode with, soprano.cpp:62-78): {@code n_fft / 2} head samples in total — the shared iSTFT's
 * {@code (n_fft - hop) / 2} plus {@code hop / 2} — and exactly {@code (frames - 1) * hop} samples kept.
 * {@link #stream} runs the same computation frame by frame and hands the kept samples out as they become known,
 * so a route can send audio while the decoder walks the frames. Immutable; validate the frame geometry of a file
 * at open and the batches per call.
 */
public final class SpeechFormat {

    /** The optional GGUF keys of a wavtokenizer-dec speech decoder, all three or none. */
    public static final String SAMPLE_RATE_KEY = "audio.sample_rate";
    public static final String N_FFT_KEY = "audio.n_fft";
    public static final String HOP_LENGTH_KEY = "audio.hop_length";

    private final int sampleRate;
    private final int nFft;
    private final int hopLength;

    /**
     * @throws IllegalArgumentException when a value is not positive, the hop exceeds {@code n_fft}, or the rate
     *                                  does not fit the WAV byte-rate field
     */
    public SpeechFormat(int sampleRate, int nFft, int hopLength) {
        if (sampleRate <= 0 || sampleRate > 0x7FFFFFFF / 2) {
            throw new IllegalArgumentException("sample rate must be positive and fit the WAV byte-rate field, got "
                    + sampleRate);
        }
        if (nFft <= 0) {
            throw new IllegalArgumentException("n_fft must be positive, got " + nFft);
        }
        if (hopLength <= 0) {
            throw new IllegalArgumentException("hop_length must be positive, got " + hopLength);
        }
        if (hopLength > nFft) {
            throw new IllegalArgumentException("hop_length " + hopLength + " exceeds n_fft " + nFft);
        }
        this.sampleRate = sampleRate;
        this.nFft = nFft;
        this.hopLength = hopLength;
    }

    /**
     * Reads the three optional keys of a wavtokenizer-dec file: null when none is present, the format when all
     * three are.
     *
     * @throws com.eonmux.neuralfree.gguf.hparams.GgufFormatException when only some of the keys are present
     */
    public static SpeechFormat ofGguf(com.eonmux.neuralfree.gguf.hparams.GgufKv kv) {
        Objects.requireNonNull(kv, "kv");
        boolean rate = kv.has(SAMPLE_RATE_KEY);
        boolean fft = kv.has(N_FFT_KEY);
        boolean hop = kv.has(HOP_LENGTH_KEY);
        if (!rate && !fft && !hop) {
            return null;
        }
        StringBuilder missing = new StringBuilder();
        if (!rate) {
            missing.append(missing.length() == 0 ? "" : ", ").append(SAMPLE_RATE_KEY);
        }
        if (!fft) {
            missing.append(missing.length() == 0 ? "" : ", ").append(N_FFT_KEY);
        }
        if (!hop) {
            missing.append(missing.length() == 0 ? "" : ", ").append(HOP_LENGTH_KEY);
        }
        if (missing.length() > 0) {
            throw new com.eonmux.neuralfree.gguf.hparams.GgufFormatException("the speech decoder keys are all three"
                    + " or none, missing: " + missing);
        }
        return new SpeechFormat(kv.u32Required(SAMPLE_RATE_KEY), kv.u32Required(N_FFT_KEY),
                kv.u32Required(HOP_LENGTH_KEY));
    }

    /** Frames per second of the decoded audio. */
    public int sampleRate() {
        return sampleRate;
    }

    /** Transform size ({@code n_fft}). */
    public int nFft() {
        return nFft;
    }

    /** Samples between frames ({@code hop_length}); one code is one frame. */
    public int hopLength() {
        return hopLength;
    }

    /** Spectrum bins per frame ({@code n_fft / 2 + 1}). */
    public int nBins() {
        return nFft / 2 + 1;
    }

    /** Values of one spectrum row: {@code n_bins} interleaved {@code (real, imaginary)} pairs. */
    public int featureWidth() {
        return 2 * nBins();
    }

    /**
     * PCM of one spectrum batch: each row through the streaming iSTFT, with the trimming of the references'
     * spectral decoders (soprano.cpp:62-78, the torch {@code center=True} convention): {@code n_fft / 2} head
     * samples in total — the shared iSTFT's {@code (n_fft - hop) / 2} plus {@code hop / 2} — and exactly
     * {@code (frames - 1) * hop} samples kept. A one-frame batch answers an empty clip.
     *
     * @param spectra {@code [frames][featureWidth()]} interleaved {@code (real, imaginary)} rows
     * @throws IllegalArgumentException when a row has another width or the batch is empty
     */
    public AudioClip decode(float[][] spectra) {
        requireFrames(spectra);
        StreamingIstft istft = new StreamingIstft(nFft, hopLength);
        float[] all = new float[spectra.length * hopLength + (nFft - hopLength)];
        int n = 0;
        for (float[] frame : spectra) {
            for (float v : istft.processFrame(frame)) {
                all[n++] = v;
            }
        }
        for (float v : istft.flush()) {
            all[n++] = v;
        }
        int head = nFft / 2;
        int kept = Math.min((spectra.length - 1) * hopLength, Math.max(n - head, 0));
        float[] trimmed = new float[kept];
        System.arraycopy(all, head, trimmed, 0, kept);
        return new AudioClip(trimmed, sampleRate);
    }

    /**
     * {@link #decode(float[][])} delivered as it decodes: every row goes through {@link #stream}, and each run of
     * samples the centered trim keeps is passed to {@code sink} as an {@link AudioClip} at {@link #sampleRate()}.
     * The concatenation of the clips is exactly {@link #decode(float[][])}'s clip, but the first rows produce no
     * clip because the head padding is still draining.
     *
     * @param spectra {@code [frames][featureWidth()]} interleaved {@code (real, imaginary)} rows
     * @param sink    receives consecutive runs of the decoded waveform
     * @throws IllegalArgumentException when a row has another width or the batch is empty
     */
    public void decode(float[][] spectra, Consumer<AudioClip> sink) {
        requireFrames(spectra);
        Objects.requireNonNull(sink, "sink");
        Stream stream = stream();
        for (float[] frame : spectra) {
            emit(stream.push(frame), sink);
        }
        emit(stream.finish(), sink);
    }

    /**
     * A stateful frame-by-frame decoding: the same computation as {@link #decode(float[][])}, keeping exactly the
     * one-shot samples. A pushed frame can release samples only up to the boundary before its own hop, because the
     * one-shot trim drops the last frame's hop; {@link Stream#finish()} knows the frame count and releases the
     * rest. Not thread-safe; one instance decodes one sequence.
     */
    public Stream stream() {
        return new Stream(this);
    }

    /** The increment form of {@link #decode(float[][])}: see {@link #stream()}. */
    public static final class Stream {

        private final SpeechFormat format;
        private final StreamingIstft istft;
        private final long head;
        private final long hop;
        private float[] carry = new float[64];
        private int carryLength;
        /** Absolute index of the first sample the stream has neither emitted nor dropped. */
        private long pendingStart;
        /** Absolute end of the samples the iSTFT has produced so far. */
        private long received;
        private int frames;
        private boolean finished;

        private Stream(SpeechFormat format) {
            this.format = format;
            this.istft = new StreamingIstft(format.nFft, format.hopLength);
            this.head = format.nFft / 2;
            this.hop = format.hopLength;
        }

        /**
         * Decodes one spectrum row and returns the samples the centered trim now keeps.
         *
         * @param frame {@code featureWidth()} interleaved {@code (real, imaginary)} values
         * @return a fresh array of ready samples, empty while the head padding drains
         * @throws IllegalArgumentException when the row has another width
         * @throws IllegalStateException    after {@link #finish()}
         */
        public float[] push(float[] frame) {
            if (finished) {
                throw new IllegalStateException("the speech stream is finished");
            }
            Objects.requireNonNull(frame, "frame");
            int width = format.featureWidth();
            if (frame.length != width) {
                throw new IllegalArgumentException("spectrum frame holds " + frame.length + " values, the decoder"
                        + " takes " + width);
            }
            append(istft.processFrame(frame));
            frames++;
            // the last frame's hop is trimmed, so the current frame's own output is never released yet
            return take(head + (frames - 1) * hop);
        }

        /**
         * Drains the iSTFT tail and returns the last samples the centered trim keeps; later calls answer nothing.
         * The kept stream ends at {@code (frames - 1) * hop} past the head, and everything beyond it is dropped.
         */
        public float[] finish() {
            if (finished) {
                return new float[0];
            }
            finished = true;
            append(istft.flush());
            float[] out = take(head + Math.max(frames - 1, 0) * hop);
            carryLength = 0;
            pendingStart = received;
            return out;
        }

        private void append(float[] samples) {
            if (carryLength + samples.length > carry.length) {
                carry = java.util.Arrays.copyOf(carry, carryLength + samples.length);
            }
            System.arraycopy(samples, 0, carry, carryLength, samples.length);
            carryLength += samples.length;
            received += samples.length;
        }

        /** Drops the head prefix, then releases the pending samples before {@code endExclusive}. */
        private float[] take(long endExclusive) {
            if (pendingStart < head) {
                int drop = (int) Math.min(carryLength, head - pendingStart);
                System.arraycopy(carry, drop, carry, 0, carryLength - drop);
                carryLength -= drop;
                pendingStart += drop;
            }
            long available = Math.min(endExclusive, received);
            int ready = (int) Math.max(0, Math.min(carryLength, available - pendingStart));
            if (ready == 0) {
                return new float[0];
            }
            float[] out = java.util.Arrays.copyOfRange(carry, 0, ready);
            System.arraycopy(carry, ready, carry, 0, carryLength - ready);
            carryLength -= ready;
            pendingStart += ready;
            return out;
        }
    }

    private void requireFrames(float[][] spectra) {
        Objects.requireNonNull(spectra, "spectra");
        if (spectra.length == 0) {
            throw new IllegalArgumentException("speech decoding needs at least one frame");
        }
        int width = featureWidth();
        for (int f = 0; f < spectra.length; f++) {
            Objects.requireNonNull(spectra[f], "spectra[" + f + "]");
            if (spectra[f].length != width) {
                throw new IllegalArgumentException("spectrum frame " + f + " holds " + spectra[f].length
                        + " values, the decoder takes " + width);
            }
        }
    }

    private void emit(float[] samples, Consumer<AudioClip> sink) {
        if (samples.length > 0) {
            sink.accept(new AudioClip(samples, sampleRate));
        }
    }

    @Override
    public String toString() {
        return "SpeechFormat{rate=" + sampleRate + ", n_fft=" + nFft + ", hop=" + hopLength + "}";
    }
}
