package com.eonmux.neuralfree.gguf.audio;

import java.util.Locale;

/**
 * Mel filterbank of llama.cpp's mtmd audio frontend (tools/mtmd/mtmd-audio.cpp:35-130): {@code nMel} triangular
 * filters over the mel grid of {@code nMel + 2} edges, evaluated at the FFT bin frequencies. The mel scale is HTK
 * or Slaney (librosa default), with the optional Slaney area normalization; every entry is scaled by a constant.
 * Immutable; {@link #weight} reads one filter tap.
 */
public final class MelFilterbank {

    private final int nMel;
    private final int nFftBins;
    private final float[] data;
    /** Per filter, the first bin with a non-zero tap ({@code nFftBins} for an all-zero filter). */
    private final int[] firstBin;
    /** Per filter, one past the last bin with a non-zero tap ({@code 0} for an all-zero filter). */
    private final int[] endBin;

    private MelFilterbank(int nMel, int nFftBins, float[] data) {
        this.nMel = nMel;
        this.nFftBins = nFftBins;
        this.data = data;
        this.firstBin = new int[nMel];
        this.endBin = new int[nMel];
        for (int m = 0; m < nMel; m++) {
            int first = nFftBins;
            int end = 0;
            for (int k = 0; k < nFftBins; k++) {
                if (data[m * nFftBins + k] != 0.0f) {
                    first = Math.min(first, k);
                    end = k + 1;
                }
            }
            firstBin[m] = first;
            endBin[m] = end;
        }
    }

    /**
     * A filterbank of given taps (the parakeet {@code a.mel_filters} tensor): {@code nMel} rows of
     * {@code nFftBins} values, filter-major, copied.
     *
     * @throws IllegalArgumentException for non-positive sizes, a length mismatch or a non-finite tap
     */
    public static MelFilterbank of(int nMel, int nFftBins, float[] taps) {
        if (nMel <= 0 || nFftBins <= 0) {
            throw new IllegalArgumentException("a filterbank needs positive sizes, got " + nMel + "x" + nFftBins);
        }
        if (taps == null || taps.length != (long) nMel * nFftBins) {
            throw new IllegalArgumentException("a " + nMel + "x" + nFftBins + " filterbank needs "
                    + (long) nMel * nFftBins + " taps, got " + (taps == null ? "none" : taps.length));
        }
        for (int i = 0; i < taps.length; i++) {
            if (!Float.isFinite(taps[i])) {
                throw new IllegalArgumentException("filterbank tap " + i + " is not finite (" + taps[i] + ")");
            }
        }
        return new MelFilterbank(nMel, nFftBins, taps.clone());
    }

    /** Builder of one filterbank; defaults follow the whisper framings. */
    public static final class Builder {
        private int nMel;
        private int nFft;
        private int sampleRate;
        private double fmin;
        private double fmax = -1.0;
        private boolean slaneyAreaNorm;
        private double scale = 1.0;
        private boolean useHtk;

        /** Filter count. */
        public Builder nMel(int v) {
            this.nMel = v;
            return this;
        }

        /** FFT size the filters are evaluated at; bins are {@code nFft / 2 + 1}. */
        public Builder nFft(int v) {
            this.nFft = v;
            return this;
        }

        /** Sample rate the bins map to. */
        public Builder sampleRate(int v) {
            this.sampleRate = v;
            return this;
        }

        /** Lowest filter edge in Hz (default 0). */
        public Builder fmin(double v) {
            this.fmin = v;
            return this;
        }

        /** Highest filter edge in Hz; non-positive means Nyquist (default). */
        public Builder fmax(double v) {
            this.fmax = v;
            return this;
        }

        /** Multiplies every filter by {@code 2 / (f_right - f_left)}, the librosa "slaney" norm. */
        public Builder slaneyAreaNorm(boolean v) {
            this.slaneyAreaNorm = v;
            return this;
        }

        /** Constant applied to every tap (default 1). */
        public Builder scale(double v) {
            this.scale = v;
            return this;
        }

        /** HTK formula when true, the Slaney scale otherwise (default). */
        public Builder useHtk(boolean v) {
            this.useHtk = v;
            return this;
        }

        /** Filters of the builder's arguments. */
        public MelFilterbank build() {
            if (nMel <= 0 || nFft <= 1) {
                throw new IllegalArgumentException("n_mel " + nMel + " and n_fft " + nFft + " must satisfy"
                        + " n_mel > 0 and n_fft > 1");
            }
            if (sampleRate <= 0) {
                throw new IllegalArgumentException("sample rate must be positive, got " + sampleRate);
            }
            double fmaxEff = fmax > 0.0 ? fmax : 0.5 * sampleRate;
            double binHzStep = (double) sampleRate / nFft;
            double mLo = useHtk ? hzToMelHtk(fmin) : hzToMelSlaney(fmin);
            double mHi = useHtk ? hzToMelHtk(fmaxEff) : hzToMelSlaney(fmaxEff);
            int nBins = nFft / 2 + 1;
            float[] out = new float[(int) ((long) nMel * nBins)];
            for (int m = 0; m < nMel; m++) {
                double frac = (double) m / (nMel + 1);
                double fLeft = melToHz(useHtk, mLo + (mHi - mLo) * frac);
                double fCenter = melToHz(useHtk, mLo + (mHi - mLo) * ((double) (m + 1) / (nMel + 1)));
                double fRight = melToHz(useHtk, mLo + (mHi - mLo) * ((double) (m + 2) / (nMel + 1)));
                double denomL = Math.max(1e-30, fCenter - fLeft);
                double denomR = Math.max(1e-30, fRight - fCenter);
                double enorm = slaneyAreaNorm ? 2.0 / Math.max(1e-30, fRight - fLeft) : 1.0;
                for (int k = 0; k < nBins; k++) {
                    double f = k * binHzStep;
                    double w = 0.0;
                    if (f >= fLeft && f <= fCenter) {
                        w = (f - fLeft) / denomL;
                    } else if (f > fCenter && f <= fRight) {
                        w = (fRight - f) / denomR;
                    }
                    out[m * nBins + k] = (float) (w * enorm * scale);
                }
            }
            return new MelFilterbank(nMel, nBins, out);
        }
    }

    private static double hzToMelHtk(double f) {
        return 2595.0 * Math.log10(1.0 + f / 700.0);
    }

    private static double hzToMelSlaney(double f) {
        final double minLogHz = 1000.0;
        final double linSlope = 3.0 / 200.0;
        final double minLogMel = minLogHz * linSlope;
        final double logStep = Math.log(6.4) / 27.0;
        return f < minLogHz ? f * linSlope : minLogMel + Math.log(f / minLogHz) / logStep;
    }

    private static double melToHz(boolean htk, double m) {
        if (htk) {
            return 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0);
        }
        final double minLogHz = 1000.0;
        final double linSlope = 3.0 / 200.0;
        final double minLogMel = minLogHz * linSlope;
        final double logStep = Math.log(6.4) / 27.0;
        return m < minLogMel ? m / linSlope : minLogHz * Math.exp((m - minLogMel) * logStep);
    }

    /** Filter count. */
    public int nMel() {
        return nMel;
    }

    /** FFT bins the filters cover ({@code nFft / 2 + 1}). */
    public int nFftBins() {
        return nFftBins;
    }

    /** Tap of filter {@code m} at bin {@code k}; both in range. */
    public float weight(int m, int k) {
        if (m < 0 || m >= nMel || k < 0 || k >= nFftBins) {
            throw new IndexOutOfBoundsException("filterbank tap (" + m + ',' + k + ") outside " + nMel + 'x'
                    + nFftBins);
        }
        return data[m * nFftBins + k];
    }

    /**
     * The first bin of filter {@code m}'s non-zero taps: every tap outside {@code [firstBin(m), endBin(m))} is zero,
     * so a fold over finite powers may skip those bins without changing its sum.
     */
    public int firstBin(int m) {
        return firstBin[m];
    }

    /** One past the last bin of filter {@code m}'s non-zero taps; {@code 0} when every tap is zero. */
    public int endBin(int m) {
        return endBin[m];
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "MelFilterbank{nMel=%d, nFftBins=%d}", nMel, nFftBins);
    }
}
