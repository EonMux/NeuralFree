package com.eonmux.neuralfree.gguf.audio;

/**
 * How a log-mel spectrogram frames the waveform and maps the spectra to mel bins: the knobs of llama.cpp's
 * {@code filter_params} (tools/mtmd/mtmd-audio.cpp:273-288) plus the Hann window length and kind. Immutable;
 * built through {@link #builder} and the flavor presets of {@link MelFrontend}.
 */
public final class MelFrameParams {

    /** Zero samples added around the signal before framing (mtmd-audio.cpp:413-444). */
    public enum Padding {
        /** Frame the samples as they are; the caller has padded. */
        NONE,
        /** Zero-pad {@code frameSize / 2} on both ends (torch center=True). */
        CENTER,
        /** Reflect {@code frameSize / 2} samples at the start, then pad the 30 s whisper tail of zeros. */
        WHISPER
    }

    private final int nMel;
    private final int frameSize;
    private final int hopLength;
    private final int sampleRate;
    private final Padding padding;
    private final float preemph;
    private final boolean magnitude;
    private final boolean naturalLog;
    private final float melFloor;
    private final boolean floorAdd;
    private final int hannWindowSize;
    private final boolean hannPeriodic;
    private final boolean hannLeftAligned;
    private final boolean normPerFeature;
    private final boolean stdEpsAfterSqrt;
    private final boolean normalizeTail;

    private MelFrameParams(Builder b) {
        this.nMel = b.nMel;
        this.frameSize = b.frameSize;
        this.hopLength = b.hopLength;
        this.sampleRate = b.sampleRate;
        this.padding = b.padding;
        this.preemph = b.preemph;
        this.magnitude = b.magnitude;
        this.naturalLog = b.naturalLog;
        this.melFloor = b.melFloor;
        this.floorAdd = b.floorAdd;
        this.hannWindowSize = b.hannWindowSize;
        this.hannPeriodic = b.hannPeriodic;
        this.hannLeftAligned = b.hannLeftAligned;
        this.normPerFeature = b.normPerFeature;
        this.stdEpsAfterSqrt = b.stdEpsAfterSqrt;
        this.normalizeTail = b.normalizeTail;
    }

    public static Builder builder(int nMel, int frameSize, int hopLength, int sampleRate) {
        return new Builder(nMel, frameSize, hopLength, sampleRate);
    }

    /** Builder of one framing. */
    public static final class Builder {
        private final int nMel;
        private final int frameSize;
        private final int hopLength;
        private final int sampleRate;
        private Padding padding = Padding.NONE;
        private float preemph;
        private boolean magnitude;
        private boolean naturalLog = true;
        private float melFloor = 5.960464477539063e-08f;
        private boolean floorAdd;
        private int hannWindowSize = -1;
        private boolean hannPeriodic = true;
        private boolean hannLeftAligned;
        private boolean normPerFeature;
        private boolean stdEpsAfterSqrt;
        private boolean normalizeTail;

        private Builder(int nMel, int frameSize, int hopLength, int sampleRate) {
            if (nMel <= 0 || frameSize <= 1 || hopLength <= 0 || sampleRate <= 0) {
                throw new IllegalArgumentException("n_mel " + nMel + ", frame size " + frameSize + ", hop "
                        + hopLength + " and sample rate " + sampleRate + " must satisfy n_mel > 0, frameSize > 1,"
                        + " hop > 0, sampleRate > 0");
            }
            this.nMel = nMel;
            this.frameSize = frameSize;
            this.hopLength = hopLength;
            this.sampleRate = sampleRate;
            this.hannWindowSize = frameSize;
        }

        /** Padding mode (default NONE). */
        public Builder padding(Padding v) {
            this.padding = v;
            return this;
        }

        /** Preemphasis coefficient; 0 disables (default). */
        public Builder preemph(float v) {
            this.preemph = v;
            return this;
        }

        /** Take {@code sqrt} of the power before the mel sum (magnitude); default power. */
        public Builder magnitude(boolean v) {
            this.magnitude = v;
            return this;
        }

        /** Natural log when true, {@code log10} otherwise (default natural). */
        public Builder naturalLog(boolean v) {
            this.naturalLog = v;
            return this;
        }

        /** Floor applied to the mel energy before the log (default {@code 2^-24}). */
        public Builder melFloor(float v) {
            this.melFloor = v;
            return this;
        }

        /** Add the floor instead of clamping to it (default clamp). */
        public Builder floorAdd(boolean v) {
            this.floorAdd = v;
            return this;
        }

        /** Length the Hann window is built with, padded with zeros to {@code frameSize} (default {@code frameSize}). */
        public Builder hannWindowSize(int v) {
            if (v <= 0 || v > frameSize) {
                throw new IllegalArgumentException("hann window size " + v + " must be in (0, " + frameSize + "]");
            }
            this.hannWindowSize = v;
            return this;
        }

        /** Periodic Hann when true (default), the symmetric torch {@code periodic=False} window otherwise. */
        public Builder hannPeriodic(boolean v) {
            this.hannPeriodic = v;
            return this;
        }

        /**
         * Place the window at the start of the frame with the tail zeroed (the gemma4a framings) instead of
         * centered in the frame (default).
         */
        public Builder hannLeftAligned(boolean v) {
            this.hannLeftAligned = v;
            return this;
        }

        /** Per-feature normalization of each mel row to zero mean and unit std over the unnormalized frames
         *  (default off). */
        public Builder normPerFeature(boolean v) {
            this.normPerFeature = v;
            return this;
        }

        /** With {@link #normPerFeature}: add the epsilon after the sqrt (default inside). */
        public Builder stdEpsAfterSqrt(boolean v) {
            this.stdEpsAfterSqrt = v;
            return this;
        }

        /**
         * With {@link #normPerFeature}: normalize the frames past the unnormalized ones with the same statistics
         * (the parakeet preprocessor) instead of zeroing them (default).
         */
        public Builder normalizeTail(boolean v) {
            this.normalizeTail = v;
            return this;
        }

        /** Values of one framing. */
        public MelFrameParams build() {
            return new MelFrameParams(this);
        }
    }

    /** Mel bin count. */
    public int nMel() {
        return nMel;
    }

    /** Windowed samples per frame ({@code n_fft}). */
    public int frameSize() {
        return frameSize;
    }

    /** Frames per FFT bin count: {@code frameSize / 2 + 1}. */
    public int nFftBins() {
        return frameSize / 2 + 1;
    }

    /** Samples between frame starts. */
    public int hopLength() {
        return hopLength;
    }

    /** Sample rate the input is expected at. */
    public int sampleRate() {
        return sampleRate;
    }

    public Padding padding() {
        return padding;
    }

    public float preemph() {
        return preemph;
    }

    public boolean magnitude() {
        return magnitude;
    }

    public boolean naturalLog() {
        return naturalLog;
    }

    public float melFloor() {
        return melFloor;
    }

    public boolean floorAdd() {
        return floorAdd;
    }

    public int hannWindowSize() {
        return hannWindowSize;
    }

    public boolean hannPeriodic() {
        return hannPeriodic;
    }

    public boolean hannLeftAligned() {
        return hannLeftAligned;
    }

    public boolean normPerFeature() {
        return normPerFeature;
    }

    public boolean stdEpsAfterSqrt() {
        return stdEpsAfterSqrt;
    }

    public boolean normalizeTail() {
        return normalizeTail;
    }
}
