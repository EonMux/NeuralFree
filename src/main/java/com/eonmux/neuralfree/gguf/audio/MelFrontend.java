package com.eonmux.neuralfree.gguf.audio;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The log-mel flavors of llama.cpp's mtmd audio preprocessors (tools/mtmd/mtmd-audio.cpp, and the voxtral
 * realtime stream of mtmd.cpp), one constant per preprocessor family: the whisper framings of the
 * whisper/ultravox/voxtral/qwen2a/qwen3a towers, the NeMo conformer framings, the model-specific flavors of
 * gemma4a, lfm3a, granite-speech, parakeet, mimo, voxtral realtime and dots3note, the raw-waveform framings of
 * muse spark and gemma4ua, and the Hugging Face {@code Gemma3nAudioFeatureExtractor} framings of gemma3na (which
 * llama.cpp does not run). Each flavor turns a mono clip at the projector's rate into one or more
 * {@link MelChunk}s, with the padding, log, floor and normalization the reference applies. Static methods only.
 */
public final class MelFrontend {

    /** Frames per chunk of the whisper flavor (tools/mtmd/mtmd-audio.cpp:624). */
    private static final int WHISPER_CHUNK_FRAMES = 3000;
    /** Frames of the whisper normalization tail (tools/mtmd/mtmd-audio.cpp:427). */
    private static final int WHISPER_TAIL_SECONDS = 30;
    /** Chunk geometry of the qwen3a flavor (tools/mtmd/mtmd-audio.cpp:727-728). */
    private static final int QWEN3A_WINDOW_FRAMES = 800;
    private static final int QWEN3A_CHUNK_FRAMES = 100;
    /** Seconds per chunk of the gemma4a flavor (tools/mtmd/mtmd-audio.cpp:1290). */
    private static final int GEMMA4A_CHUNK_SECONDS = 30;
    /** The parakeet preemphasis (tools/mtmd/mtmd-audio.cpp:1414). */
    private static final float PARAKEET_PREEMPH = 0.97f;
    /** The parakeet log epsilon, added before the log (tools/mtmd/mtmd-audio.cpp:1323). */
    private static final float PARAKEET_FLOOR = 5.960464477539063e-08f;
    /** The mimo mel floor, clamped before the natural log (tools/mtmd/mtmd-audio.cpp:900). */
    private static final float MIMO_FLOOR = 1e-7f;
    /** The voxtral realtime mel floor (mtmd_audio_log_mel_frame). */
    private static final float VOXTRAL_RT_FLOOR = 1e-10f;
    /** The voxtral realtime log-mel ceiling, minus 8 in the clamp (mtmd.cpp log_mel_max). */
    private static final float VOXTRAL_RT_LOG_MAX = 1.5f;
    /** Raw samples folded into one waveform frame of the gemma4ua embedder (mtmd-audio.cpp:1515). */
    private static final int GEMMA4UA_FRAME_SAMPLES = 640;
    /** Seconds per window of the gemma3n flavor (the extractor's {@code max_length} of 480000 samples). */
    private static final int GEMMA3N_CHUNK_SECONDS = 30;
    /** The gemma3n extractor pads every clip to a multiple of this many samples ({@code pad_to_multiple_of}). */
    private static final int GEMMA3N_PAD_MULTIPLE = 128;
    /** The gemma3n mel band edges in Hz ({@code min_frequency}, {@code max_frequency}). */
    private static final double GEMMA3N_FMIN = 125.0;
    private static final double GEMMA3N_FMAX = 7600.0;
    /** The gemma3n HTK-flavored preemphasis coefficient. */
    private static final double GEMMA3N_PREEMPH = 0.97;
    /** The gemma3n floor of the mel magnitudes before the natural log ({@code mel_floor}). */
    private static final double GEMMA3N_MEL_FLOOR = 1e-5;

    private MelFrontend() {
    }

    /**
     * Chunks of {@code clip} under the flavor's framing. The input must already be mono at
     * {@code config.sampleRate}.
     *
     * @throws IllegalArgumentException when the clip is empty or the config is incomplete for the flavor
     */
    public static List<MelChunk> chunks(Flavor flavor, AudioClip clip, Config config) {
        Objects.requireNonNull(flavor, "flavor");
        Objects.requireNonNull(clip, "clip");
        Objects.requireNonNull(config, "config");
        if (clip.length() == 0) {
            throw new IllegalArgumentException("audio clip is empty");
        }
        switch (flavor) {
            case WHISPER:
                return whisper(clip, config);
            case QWEN3A:
                return qwen3a(clip, config);
            case CONFORMER:
                return conformer(clip, config, false);
            case D1OMNI:
                return conformer(clip, config, true);
            case GEMMA4A:
                return gemma4a(clip, config);
            case LFM3A:
                return lfm3a(clip, config);
            case GRANITE_SPEECH:
                return graniteSpeech(clip, config);
            case PARAKEET:
                return parakeet(clip, config);
            case MIMO_AUDIO:
                return mimoAudio(clip, config);
            case VOXTRAL_RT:
                return voxtralRt(clip, config);
            case MUSE_SPARK_AUDIO:
                return museSpark(clip, config);
            case GEMMA4UA:
                return gemma4ua(clip);
            case DOTS3NOTE:
                return dots3note(clip, config);
            case GEMMA3N:
                return gemma3n(clip, config);
            case QWEN3TTS_SPKENC:
                return qwen3ttsSpkenc(clip, config);
            default:
                throw new IllegalStateException("unhandled mel flavor " + flavor);
        }
    }

    /** Which preprocessor family runs before an encoder. */
    public enum Flavor {
        /** Whisper framings: the whisper, ultravox, voxtral, qwen2a and glma towers. */
        WHISPER,
        /** Qwen3-ASR framings: reflection padding and 100-frame conv chunks. */
        QWEN3A,
        /** NeMo conformer framings: the lfm2a and conformer towers. */
        CONFORMER,
        /** The d1omni variant of the conformer framings, clipped to 30 s and frame-trimmed. */
        D1OMNI,
        /** The gemma4a framings: HTK mel, magnitude and semicausal left padding. */
        GEMMA4A,
        /** The lfm3a framings: right-padded frames and a {@code (v + 4) / 4} scale without clamping. */
        LFM3A,
        /** The granite-speech framings: HTK mel of half the bins, stacked into frame pairs. */
        GRANITE_SPEECH,
        /** The parakeet framings: GGUF-provided filterbank and window with a preemphasis. */
        PARAKEET,
        /** The mimo framings: HTK mel, magnitude, natural log, reflection padding, no normalization. */
        MIMO_AUDIO,
        /** The voxtral realtime framings: centered frames, log10, a fixed per-frame clamp and scale. */
        VOXTRAL_RT,
        /** The muse-spark passthrough: the raw waveform as one row, zero-padded to whole tokens. */
        MUSE_SPARK_AUDIO,
        /** The gemma4ua passthrough: raw 640-sample waveform frames, one token per frame. */
        GEMMA4UA,
        /** The dots3note framings: 60 s chunks of whisper-style mel, valid frames only. */
        DOTS3NOTE,
        /**
         * The Gemma 3n framings: 30 s windows padded to 128 samples, in-frame HTK preemphasis, magnitude spectra of
         * the doubled FFT, an HTK mel of 125-7600 Hz and a natural log floored at 1e-5.
         */
        GEMMA3N,
        /**
         * The qwen3tts speaker-encoder framing: {@code (n_fft - hop) / 2} reflected samples on both ends, magnitude
         * spectra, a Slaney mel of 0 to Nyquist and a natural log floored at 1e-5, over the whole clip.
         */
        QWEN3TTS_SPKENC
    }

    /**
     * The framing parameters a flavor reads; {@code melFilterRows} and {@code windowVector} feed parakeet,
     * {@code rowSamples} the flavors that frame whole output rows (muse spark, gemma4ua, voxtral realtime).
     */
    public static final class Config {
        final int nMelBins;
        final int sampleRate;
        final int nFft;
        final int windowLen;
        final int hopLen;
        final int chunkSeconds;
        final int rowSamples;
        final float[] melFilterRows;
        final float[] windowVector;

        private Config(Builder b) {
            this(b.nMelBins, b.sampleRate, b.nFft, b.windowLen, b.hopLen, b.chunkSeconds, b.rowSamples,
                    b.melFilterRows, b.windowVector);
        }

        private Config(int nMelBins, int sampleRate, int nFft, int windowLen, int hopLen, int chunkSeconds,
                       int rowSamples, float[] melFilterRows, float[] windowVector) {
            this.nMelBins = nMelBins;
            this.sampleRate = sampleRate;
            this.nFft = nFft;
            this.windowLen = windowLen;
            this.hopLen = hopLen;
            this.chunkSeconds = chunkSeconds;
            this.rowSamples = rowSamples;
            this.melFilterRows = melFilterRows;
            this.windowVector = windowVector;
        }

        public static Builder builder(int nMelBins, int sampleRate, int nFft, int windowLen, int hopLen) {
            return new Builder(nMelBins, sampleRate, nFft, windowLen, hopLen);
        }

        /**
         * Framing of the raw-waveform flavors (no FFT): {@code nMel} rows at {@code rate}, {@code rowSamples}
         * samples per output row (the muse-spark fold, the gemma4ua frame).
         *
         * @throws IllegalArgumentException for a non-positive size
         */
        public static Config passthrough(int nMel, int sampleRate, int rowSamples) {
            if (nMel <= 0 || sampleRate <= 0 || rowSamples <= 0) {
                throw new IllegalArgumentException("passthrough framing n_mel " + nMel + ", rate " + sampleRate
                        + ", row samples " + rowSamples + " must be positive");
            }
            return new Config(nMel, sampleRate, 0, 0, 0, 0, rowSamples, null, null);
        }

        /** Builder of one framing config. */
        public static final class Builder {
            private final int nMelBins;
            private final int sampleRate;
            private final int nFft;
            private final int windowLen;
            private final int hopLen;
            private int chunkSeconds;
            private int rowSamples;
            private float[] melFilterRows;
            private float[] windowVector;

            private Builder(int nMelBins, int sampleRate, int nFft, int windowLen, int hopLen) {
                if (nMelBins <= 0 || sampleRate <= 0 || nFft <= 1 || windowLen <= 0 || windowLen > nFft
                        || hopLen <= 0) {
                    throw new IllegalArgumentException("framing config n_mel " + nMelBins + ", rate " + sampleRate
                            + ", n_fft " + nFft + ", window " + windowLen + ", hop " + hopLen + " must satisfy"
                            + " n_mel > 0, rate > 0, 1 < n_fft, 0 < window <= n_fft, hop > 0");
                }
                this.nMelBins = nMelBins;
                this.sampleRate = sampleRate;
                this.nFft = nFft;
                this.windowLen = windowLen;
                this.hopLen = hopLen;
            }

            /** Fixed chunk length of the preprocessor in seconds (dots3note); 0 = whole clip. */
            public Builder chunkSeconds(int v) {
                this.chunkSeconds = v;
                return this;
            }

            /** Samples per output row the frames are aligned to (voxtral realtime); 0 = no alignment. */
            public Builder rowSamples(int v) {
                if (v < 0) {
                    throw new IllegalArgumentException("row samples must not be negative, got " + v);
                }
                this.rowSamples = v;
                return this;
            }

            /** GGUF-provided mel filterbank rows ({@code a.mel_filters}, {@code nMel * (nFft/2 + 1)} values). */
            public Builder melFilterRows(float[] v) {
                this.melFilterRows = v;
                return this;
            }

            /** GGUF-provided window ({@code a.window}, {@code windowLen} values). */
            public Builder windowVector(float[] v) {
                this.windowVector = v;
                return this;
            }

            /** Values of one framing config. */
            public Config build() {
                return new Config(this);
            }
        }

        public int nMelBins() {
            return nMelBins;
        }

        public int sampleRate() {
            return sampleRate;
        }

        public int nFft() {
            return nFft;
        }

        public int windowLen() {
            return windowLen;
        }

        public int hopLen() {
            return hopLen;
        }

        /** Fixed chunk length of the preprocessor in seconds; 0 = the whole clip is one chunk. */
        public int chunkSeconds() {
            return chunkSeconds;
        }

        /** Samples per output row of the flavors that frame whole rows; 0 when the flavor does not. */
        public int rowSamples() {
            return rowSamples;
        }

        /**
         * A copy of this config with the GGUF-provided parakeet filterbank and window bound; both arrays must be
         * present when the flavor reads them.
         */
        public Config withFrontend(float[] filterRows, float[] window) {
            return new Config(nMelBins, sampleRate, nFft, windowLen, hopLen, chunkSeconds, rowSamples, filterRows,
                    window);
        }
    }

    // ================================================================== whisper (whisper, ultravox, voxtral, qwen2a, glma)

    /**
     * mtmd_audio_preprocessor_whisper: whisper padding (start reflection and the 30 s zero tail), log10, the
     * whisper clamp, and a split into 3000-frame chunks without the trailing incomplete one.
     */
    private static List<MelChunk> whisper(AudioClip clip, Config c) {
        float[] samples = ensureMinSamples(clip.samples(), c.nFft / 2 + 1);
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.WHISPER)
                .naturalLog(false)
                .hannWindowSize(c.windowLen)
                .build();
        MelFilterbank filters = standardFilterbank(c);
        MelChunk mel = LogMel.compute(samples, samples.length, params, filters);
        List<MelChunk> out = new ArrayList<>();
        for (int off = 0; off + WHISPER_CHUNK_FRAMES <= mel.nLen(); off += WHISPER_CHUNK_FRAMES) {
            out.add(slice(mel, off, WHISPER_CHUNK_FRAMES));
        }
        return out;
    }

    /** A copy of {@code samples}, zero-padded to at least {@code min} frames (mtmd-audio.cpp:585-593). */
    private static float[] ensureMinSamples(float[] samples, int min) {
        if (samples.length >= min) {
            return samples;
        }
        float[] padded = new float[min];
        System.arraycopy(samples, 0, padded, 0, samples.length);
        return padded;
    }

    /** Slaney-scale filterbank with the area norm, the default of every flavor except gemma4a and granite. */
    private static MelFilterbank standardFilterbank(Config c) {
        return new MelFilterbank.Builder()
                .nMel(c.nMelBins)
                .nFft(c.nFft)
                .sampleRate(c.sampleRate)
                .slaneyAreaNorm(true)
                .build();
    }

    /** Rows {@code [off, off + len)} of a mel-major spectrogram. */
    private static MelChunk slice(MelChunk mel, int off, int len) {
        float[] data = new float[mel.nMel() * len];
        for (int m = 0; m < mel.nMel(); m++) {
            System.arraycopy(mel.data(), m * mel.nLen() + off, data, m * len, len);
        }
        return new MelChunk(mel.nMel(), len, data);
    }

    // ================================================================== qwen3a

    /**
     * mtmd_audio_preprocessor_qwen3a: reflection padding of n_fft/2 at both ends, log10 with the whisper clamp over
     * the whole mel, then windows of 800 frames padded to the next multiple of 100.
     */
    private static List<MelChunk> qwen3a(AudioClip clip, Config c) {
        float[] samples = clip.samples();
        int n = samples.length;
        int pad = c.nFft / 2;
        float[] padded = new float[n + 2 * pad];
        for (int i = 0; i < pad; i++) {
            int src = pad - i;
            padded[i] = src < n ? samples[src] : 0.0f;
        }
        System.arraycopy(samples, 0, padded, pad, n);
        for (int i = 0; i < pad; i++) {
            int src = n - 2 - i;
            padded[n + pad + i] = src >= 0 ? samples[src] : 0.0f;
        }
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(false)
                .hannWindowSize(c.windowLen)
                .build();
        MelChunk mel = LogMel.compute(padded, padded.length, params, standardFilterbank(c));
        LogMel.whisperClamp(mel.data());
        int nEff = Math.min(mel.nLen(), n / c.hopLen + 1);
        List<MelChunk> out = new ArrayList<>();
        for (int off = 0; off < nEff; off += QWEN3A_WINDOW_FRAMES) {
            int winEff = Math.min(QWEN3A_WINDOW_FRAMES, nEff - off);
            int nPadded = ((winEff + QWEN3A_CHUNK_FRAMES - 1) / QWEN3A_CHUNK_FRAMES) * QWEN3A_CHUNK_FRAMES;
            float[] data = new float[c.nMelBins * nPadded];
            int copyLen = Math.min(winEff, mel.nLen() - off);
            for (int m = 0; m < c.nMelBins; m++) {
                if (copyLen > 0) {
                    System.arraycopy(mel.data(), m * mel.nLen() + off, data, m * nPadded, copyLen);
                }
            }
            out.add(new MelChunk(c.nMelBins, nPadded, data));
        }
        return out;
    }

    // ================================================================== conformer (lfm2a, d1omni)

    /**
     * mtmd_audio_preprocessor_conformer: center padding, preemphasis 0.97, the symmetric Hann window, natural log
     * with an added floor, and per-feature normalization. The d1omni variant clips to 30 s, pads to half a second
     * and keeps {@code n / hop} frames.
     */
    private static List<MelChunk> conformer(AudioClip clip, Config c, boolean d1omni) {
        float[] samples = clip.samples();
        int n = samples.length;
        if (d1omni) {
            int nMax = WHISPER_TAIL_SECONDS * c.sampleRate;
            int nMin = c.sampleRate / 2;
            int kept = Math.min(n, nMax);
            float[] clipped = new float[Math.max(kept, nMin)];
            System.arraycopy(samples, 0, clipped, 0, kept);
            samples = clipped;
            n = samples.length;
        }
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.CENTER)
                .preemph(0.97f)
                .naturalLog(true)
                .floorAdd(true)
                .normPerFeature(true)
                .stdEpsAfterSqrt(true)
                .hannWindowSize(c.windowLen)
                .hannPeriodic(false)
                .build();
        MelChunk mel = LogMel.compute(samples, samples.length, params, standardFilterbank(c));
        if (!d1omni) {
            return List.of(slice(mel, 0, mel.nLen()));
        }
        int nFrames = n / c.hopLen;
        return List.of(slice(mel, 0, Math.min(mel.nLen(), nFrames)));
    }

    // ================================================================== gemma4a

    /**
     * mtmd_audio_preprocessor_gemma4a: HTK mel without the area norm, magnitude spectra with a 1e-3 floor and
     * natural log, 30 s chunks with the semicausal left padding, trimmed to the PyTorch frame count.
     */
    private static List<MelChunk> gemma4a(AudioClip clip, Config c) {
        float[] samples = clip.samples();
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(true)
                .magnitude(true)
                .melFloor(0.001f)
                .hannWindowSize(c.windowLen)
                .hannLeftAligned(true)
                .build();
        MelFilterbank htk = new MelFilterbank.Builder()
                .nMel(c.nMelBins)
                .nFft(c.nFft)
                .sampleRate(c.sampleRate)
                .fmax(0.5 * c.sampleRate)
                .slaneyAreaNorm(false)
                .useHtk(true)
                .build();
        int chunkSamples = GEMMA4A_CHUNK_SECONDS * c.sampleRate;
        int padLeft = c.windowLen / 2;
        List<MelChunk> out = new ArrayList<>();
        for (int off = 0; off < samples.length; off += chunkSamples) {
            int len = Math.min(chunkSamples, samples.length - off);
            int nWithLeft = len + padLeft;
            long ptFrames = (long) ((nWithLeft - (c.windowLen + 1)) / c.hopLen) + 1;
            long nPaddedNeeded = (ptFrames - 1) * c.hopLen + (long) c.nFft;
            int totalPad = (int) Math.max(nPaddedNeeded - len, padLeft);
            float[] padded = new float[totalPad + len];
            System.arraycopy(samples, off, padded, padLeft, len);
            MelChunk mel = LogMel.compute(padded, padded.length, params, htk);
            out.add(slice(mel, 0, (int) Math.min(mel.nLen(), ptFrames)));
        }
        return out;
    }

    // ================================================================== lfm3a

    /**
     * mtmd_audio_preprocessor_lfm3a: frames right-padded so the last frame is complete, log10 with a 1e-12 floor,
     * and the {@code (v + 4) / 4} scale without clamping.
     */
    private static List<MelChunk> lfm3a(AudioClip clip, Config c) {
        float[] samples = clip.samples();
        int n = samples.length;
        int win = c.windowLen;
        int nPadded = n <= win ? win : n + (c.hopLen - (n - win) % c.hopLen) % c.hopLen;
        float[] padded = new float[nPadded];
        System.arraycopy(samples, 0, padded, 0, n);
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(false)
                .melFloor(1e-12f)
                .hannWindowSize(c.windowLen)
                .build();
        MelChunk mel = LogMel.compute(padded, padded.length, params, standardFilterbank(c));
        for (int i = 0; i < mel.data().length; i++) {
            mel.data()[i] = (mel.data()[i] + 4.0f) / 4.0f;
        }
        return List.of(slice(mel, 0, mel.nLen()));
    }

    // ================================================================== granite speech

    /**
     * mtmd_audio_preprocessor_granite_speech: HTK mel of half the bins without the area norm, granite reflection
     * padding, log10 with a 1e-10 floor and the whisper clamp, then frame pairs stacked into twice the bins.
     */
    private static List<MelChunk> graniteSpeech(AudioClip clip, Config c) {
        float[] samples = clip.samples();
        int n = samples.length;
        int halfMel = c.nMelBins / 2;
        if (halfMel <= 0) {
            throw new IllegalArgumentException("granite speech needs an even n_mel, got " + c.nMelBins);
        }
        MelFilterbank htk = new MelFilterbank.Builder()
                .nMel(halfMel)
                .nFft(c.nFft)
                .sampleRate(c.sampleRate)
                .slaneyAreaNorm(false)
                .useHtk(true)
                .build();
        int pad = c.nFft / 2;
        float[] padded = new float[n + 2 * pad];
        for (int i = 0; i < pad; i++) {
            int src = Math.min(i + 1, n - 1);
            padded[pad - 1 - i] = samples[src];
        }
        System.arraycopy(samples, 0, padded, pad, n);
        for (int i = 0; i < pad; i++) {
            int src = Math.max(n - 2 - i, 0);
            padded[pad + n + i] = samples[src];
        }
        MelFrameParams params = MelFrameParams.builder(halfMel, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(false)
                .melFloor(1e-10f)
                .hannWindowSize(c.windowLen)
                .build();
        MelChunk mel = LogMel.compute(padded, padded.length, params, htk);
        LogMel.whisperClamp(mel.data());
        int nFrames = mel.nLen();
        if (nFrames % 2 == 1) {
            nFrames--;
        }
        int nStacked = nFrames / 2;
        float[] data = new float[c.nMelBins * nStacked];
        for (int t = 0; t < nStacked; t++) {
            for (int m = 0; m < halfMel; m++) {
                data[m * nStacked + t] = mel.value(m, 2 * t);
                data[(m + halfMel) * nStacked + t] = mel.value(m, 2 * t + 1);
            }
        }
        return List.of(new MelChunk(c.nMelBins, nStacked, data));
    }

    // ================================================================== gemma3n

    /**
     * Hugging Face {@code Gemma3nAudioFeatureExtractor}: one chunk per 30 s of audio (the extractor's truncation
     * length; later audio continues in further windows), each zero-padded to a multiple of 128 samples and to at
     * least one frame, framed by {@link #gemma3nWindow}.
     */
    private static List<MelChunk> gemma3n(AudioClip clip, Config c) {
        float[] samples = clip.samples();
        MelFilterbank htk = new MelFilterbank.Builder()
                .nMel(c.nMelBins)
                .nFft(c.nFft)
                .sampleRate(c.sampleRate)
                .fmin(GEMMA3N_FMIN)
                .fmax(GEMMA3N_FMAX)
                .slaneyAreaNorm(false)
                .useHtk(true)
                .build();
        float[] window = new float[c.windowLen];
        for (int i = 0; i < window.length; i++) {
            float phase = (float) (2.0 * Math.PI) * i / c.windowLen;
            window[i] = 0.5f * (1f - (float) Math.cos(phase));
        }
        int chunkSamples = GEMMA3N_CHUNK_SECONDS * c.sampleRate;
        List<MelChunk> out = new ArrayList<>();
        for (int off = 0; off < samples.length; off += chunkSamples) {
            int len = Math.min(chunkSamples, samples.length - off);
            int padded = (len + GEMMA3N_PAD_MULTIPLE - 1) / GEMMA3N_PAD_MULTIPLE * GEMMA3N_PAD_MULTIPLE;
            float[] x = new float[Math.max(padded, c.windowLen + 1)];
            System.arraycopy(samples, off, x, 0, len);
            out.add(gemma3nWindow(x, c, htk, window));
        }
        return out;
    }

    /**
     * {@code _extract_spectrogram} of one padded window: frames of {@code window + 1} samples every hop; frame
     * value 0 is {@code 0.03 * s[0]} and value {@code i} is {@code s[i] - 0.97 * s[i - 1]} (the last sample only
     * feeds the preemphasis), times the periodic Hann window, in float as the extractor computes them; the
     * magnitude of the zero-padded FFT in double (NumPy's {@code rfft}) through the filterbank, floored at 1e-5,
     * natural log.
     */
    private static MelChunk gemma3nWindow(float[] x, Config c, MelFilterbank filters, float[] window) {
        int win = c.windowLen;
        int nFrames = (x.length - (win + 1)) / c.hopLen + 1;
        int nBins = c.nFft / 2 + 1;
        float[] data = new float[c.nMelBins * nFrames];
        DoubleFft fft = new DoubleFft(c.nFft);
        float[] frame = new float[win];
        double[] magnitude = new double[nBins];
        float first = (float) (1.0 - GEMMA3N_PREEMPH);
        float coefficient = (float) GEMMA3N_PREEMPH;
        for (int t = 0; t < nFrames; t++) {
            int start = t * c.hopLen;
            frame[0] = x[start] * first * window[0];
            for (int i = 1; i < win; i++) {
                frame[i] = (x[start + i] - coefficient * x[start + i - 1]) * window[i];
            }
            fft.magnitudes(frame, win, magnitude);
            for (int m = 0; m < c.nMelBins; m++) {
                double sum = 0.0;
                for (int k = 0; k < nBins; k++) {
                    sum += magnitude[k] * filters.weight(m, k);
                }
                data[m * nFrames + t] = (float) Math.log(Math.max(sum, GEMMA3N_MEL_FLOOR));
            }
        }
        return new MelChunk(c.nMelBins, nFrames, data);
    }

    // ================================================================== parakeet

    /**
     * The parakeet preprocessor (mtmd_audio_preprocessor_parakeet): a 0.97 preemphasis, centered zero padding,
     * the GGUF-provided window and filterbank, a natural log of {@code energy + 2^-24}, then per-feature
     * normalization ({@code (v - mean) / (std + 1e-5)}, unbiased std) with the statistics of the
     * {@code n_samples / hop} valid frames applied to every frame. One chunk.
     */
    private static List<MelChunk> parakeet(AudioClip clip, Config c) {
        if (c.melFilterRows == null || c.windowVector == null) {
            throw new IllegalArgumentException("the parakeet flavor needs the GGUF mel filterbank (a.mel_filters)"
                    + " and window (a.window), got "
                    + (c.melFilterRows == null ? "no filterbank" : "no window"));
        }
        if (c.windowVector.length != c.windowLen || c.windowLen > c.nFft) {
            throw new IllegalArgumentException("the parakeet window holds " + c.windowVector.length + " values,"
                    + " expected " + c.windowLen + " within the " + c.nFft + "-point FFT");
        }
        MelFilterbank filters = MelFilterbank.of(c.nMelBins, c.nFft / 2 + 1, c.melFilterRows);
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.CENTER)
                .preemph(PARAKEET_PREEMPH)
                .naturalLog(true)
                .floorAdd(true)
                .melFloor(PARAKEET_FLOOR)
                .hannWindowSize(c.windowLen)
                .normPerFeature(true)
                .stdEpsAfterSqrt(true)
                .normalizeTail(true)
                .build();
        MelChunk mel = LogMel.compute(clip.samples(), clip.length(), params, filters, c.windowVector);
        return List.of(slice(mel, 0, mel.nLen()));
    }

    // ================================================================== mimo audio

    /**
     * mtmd_audio_preprocessor_mimo_audio: torchaudio MelSpectrogram(power=1, center=True) — reflection padding of
     * n_fft/2 at both ends, an HTK filterbank without the area norm, a magnitude spectrogram and a natural log of
     * the energy clamped to 1e-7, without normalization.
     */
    private static List<MelChunk> mimoAudio(AudioClip clip, Config c) {
        float[] samples = clip.samples();
        int n = samples.length;
        int pad = c.nFft / 2;
        float[] padded = new float[n + 2 * pad];
        for (int i = 0; i < pad; i++) {
            int src = pad - i;
            padded[i] = src < n ? samples[src] : 0.0f;
        }
        System.arraycopy(samples, 0, padded, pad, n);
        for (int i = 0; i < pad; i++) {
            int src = n - 2 - i;
            padded[n + pad + i] = src >= 0 ? samples[src] : 0.0f;
        }
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(true)
                .magnitude(true)
                .melFloor(MIMO_FLOOR)
                .hannWindowSize(c.windowLen)
                .build();
        MelFilterbank htk = new MelFilterbank.Builder()
                .nMel(c.nMelBins)
                .nFft(c.nFft)
                .sampleRate(c.sampleRate)
                .fmax(0.5 * c.sampleRate)
                .slaneyAreaNorm(false)
                .useHtk(true)
                .build();
        MelChunk mel = LogMel.compute(padded, padded.length, params, htk);
        return List.of(slice(mel, 0, mel.nLen()));
    }

    // ================================================================== voxtral realtime

    /** Zero mel frames before the clip: the left context of the voxtral realtime conv stem. */
    public static final int VOXTRAL_RT_CONTEXT_FRAMES = 2;

    /**
     * The voxtral realtime mel of a whole clip as one encoder input (mtmd.cpp {@code mtmd_audio_stream_voxtral_rt}
     * flushed at the end): the clip zero-padded to whole output rows ({@code config.rowSamples}), frame {@code k}
     * centered on sample {@code k * hop} with silence outside the clip, {@code mtmd_audio_log_mel_frame}
     * (log10 of the energy floored at 1e-10), every value clamped below at {@code 1.5 - 8} and scaled by
     * {@code (v + 4) / 4}; the chunk starts with the two zero context frames the stream's conv stem begins with.
     */
    private static List<MelChunk> voxtralRt(AudioClip clip, Config c) {
        int rowSamples = c.rowSamples;
        if (rowSamples <= 0 || rowSamples % c.hopLen != 0) {
            throw new IllegalArgumentException("the voxtral realtime flavor needs the samples per row as a"
                    + " multiple of the hop " + c.hopLen + ", got " + rowSamples);
        }
        float[] samples = clip.samples();
        int n = samples.length;
        int aligned = Math.toIntExact(((long) n + rowSamples - 1) / rowSamples * rowSamples);
        int nFrames = aligned / c.hopLen;
        int pad = c.nFft / 2;
        float[] padded = new float[aligned + 2 * pad];
        System.arraycopy(samples, 0, padded, pad, n);
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(false)
                .melFloor(VOXTRAL_RT_FLOOR)
                .hannWindowSize(c.windowLen)
                .build();
        MelChunk mel = LogMel.compute(padded, padded.length, params, standardFilterbank(c));
        int ctx = VOXTRAL_RT_CONTEXT_FRAMES;
        int nLen = ctx + nFrames;
        float[] data = new float[c.nMelBins * nLen];
        float floor = VOXTRAL_RT_LOG_MAX - 8.0f;
        for (int m = 0; m < c.nMelBins; m++) {
            for (int t = 0; t < nFrames; t++) {
                float v = Math.max(mel.value(m, t), floor);
                data[m * nLen + ctx + t] = (v + 4.0f) / 4.0f;
            }
        }
        return List.of(new MelChunk(c.nMelBins, nLen, data));
    }

    // ================================================================== muse spark / gemma4ua passthrough

    /**
     * mtmd_audio_preprocessor_muse_spark_audio: the raw waveform as a one-row "mel" of every sample, zero-padded
     * to a whole number of tokens ({@code config.rowSamples}, the samples the conv pyramid folds per token).
     */
    private static List<MelChunk> museSpark(AudioClip clip, Config c) {
        int fold = c.rowSamples;
        if (fold <= 0) {
            throw new IllegalArgumentException("the muse-spark flavor needs the samples per token, got " + fold);
        }
        float[] samples = clip.samples();
        int nPadded = Math.toIntExact(((long) samples.length + fold - 1) / fold * fold);
        return List.of(new MelChunk(1, nPadded, Arrays.copyOf(samples, nPadded)));
    }

    /**
     * mtmd_audio_preprocessor_gemma4ua: raw 640-sample frames, one token per frame, the tail frame zero-padded;
     * stored mel-major ({@code data[f * nTokens + t]}).
     */
    private static List<MelChunk> gemma4ua(AudioClip clip) {
        float[] samples = clip.samples();
        int n = samples.length;
        int frame = GEMMA4UA_FRAME_SAMPLES;
        int nTokens = (n + frame - 1) / frame;
        float[] data = new float[frame * nTokens];
        for (int t = 0; t < nTokens; t++) {
            for (int f = 0; f < frame; f++) {
                int src = t * frame + f;
                data[f * nTokens + t] = src < n ? samples[src] : 0.0f;
            }
        }
        return List.of(new MelChunk(frame, nTokens, data));
    }

    // ================================================================== dots3note

    /**
     * mtmd_audio_preprocessor_dots3note: 60 s chunks, each with its own whisper-style log-mel (start reflection,
     * end reflection only for a full chunk), keeping the {@code n / hop} valid frames with the clamp over them.
     */
    private static List<MelChunk> dots3note(AudioClip clip, Config c) {
        if (c.chunkSeconds() <= 0) {
            throw new IllegalArgumentException("the dots3note flavor needs the chunk seconds in the config, got "
                    + c.chunkSeconds());
        }
        float[] samples = clip.samples();
        int n = samples.length;
        long chunkSamples = (long) c.chunkSeconds() * c.sampleRate;
        int pad = c.nFft / 2;
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(false)
                .hannWindowSize(c.windowLen)
                .build();
        List<MelChunk> out = new ArrayList<>();
        for (long start = 0; start < n; start += chunkSamples) {
            int nChunk = (int) Math.min(chunkSamples, n - start);
            int nValid = nChunk / c.hopLen;
            if (nValid == 0) {
                continue; // sub-hop tail, contributes no frames
            }
            float[] padded = new float[nChunk + 2 * pad];
            for (int i = 0; i < pad; i++) {
                int src = pad - i;
                padded[i] = src < nChunk ? samples[(int) start + src] : 0.0f;
            }
            System.arraycopy(samples, (int) start, padded, pad, nChunk);
            if (nChunk == chunkSamples) {
                for (int i = 0; i < pad; i++) {
                    int src = nChunk - 2 - i;
                    padded[nChunk + pad + i] = src >= 0 ? samples[(int) start + src] : 0.0f;
                }
            }
            MelChunk mel = LogMel.compute(padded, padded.length, params, standardFilterbank(c));
            // whisper clamp over the valid frames only, then keep them
            double mmax = -1e20;
            for (int m = 0; m < c.nMelBins; m++) {
                for (int t = 0; t < nValid; t++) {
                    mmax = Math.max(mmax, mel.value(m, t));
                }
            }
            mmax -= 8.0;
            float[] data = new float[c.nMelBins * nValid];
            for (int m = 0; m < c.nMelBins; m++) {
                for (int t = 0; t < nValid; t++) {
                    double v = Math.max(mel.value(m, t), mmax);
                    data[m * nValid + t] = (float) ((v + 4.0) / 4.0);
                }
            }
            out.add(new MelChunk(c.nMelBins, nValid, data));
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("the clip holds less than one " + c.hopLen + "-sample frame");
        }
        return out;
    }

    // ================================================================== qwen3tts speaker encoder

    /**
     * mtmd_audio_preprocessor_qwen3tts_spk: the reference {@code (n_fft - hop) / 2} samples are reflected onto both
     * ends (a {@code center=False} STFT), the whole clip is one chunk, and the mel takes magnitudes, a Slaney
     * filterbank and a natural log floored at 1e-5, with no normalization.
     */
    private static List<MelChunk> qwen3ttsSpkenc(AudioClip clip, Config c) {
        int pad = (c.nFft - c.hopLen) / 2;
        if (pad < 0) {
            throw new IllegalArgumentException("the qwen3tts speaker framing needs n_fft >= hop, got " + c.nFft
                    + " and " + c.hopLen);
        }
        float[] samples = clip.samples();
        int n = samples.length;
        if (n < pad + 1) {
            throw new IllegalArgumentException("the speaker reference holds " + n + " samples; the reflections"
                    + " need at least " + (pad + 1));
        }
        float[] padded = new float[n + 2 * pad];
        for (int i = 0; i < pad; i++) {
            padded[i] = samples[pad - i];
        }
        System.arraycopy(samples, 0, padded, pad, n);
        for (int i = 0; i < pad; i++) {
            padded[n + pad + i] = samples[n - 2 - i];
        }
        MelFrameParams params = MelFrameParams.builder(c.nMelBins, c.nFft, c.hopLen, c.sampleRate)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(true)
                .magnitude(true)
                .melFloor(1e-5f)
                .hannWindowSize(c.windowLen)
                .build();
        return List.of(LogMel.compute(padded, padded.length, params, standardFilterbank(c)));
    }
}
