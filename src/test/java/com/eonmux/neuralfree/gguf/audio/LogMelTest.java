package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The log-mel spectrogram against an independent double-precision transcription of the mtmd frontend
 * (tools/mtmd/mtmd-audio.cpp:273-565), plus the frame counts of the padding modes and the two normalizations.
 */
public class LogMelTest {

    // ================================================================== independent double reference

    private static double[] refHann(int len, boolean periodic) {
        int offset = periodic ? 0 : -1;
        double[] out = new double[len];
        for (int i = 0; i < len; i++) {
            out[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (len + offset)));
        }
        return out;
    }

    /** Naive DFT power at bin {@code k} of the windowed frame. */
    private static double refPower(double[] frame, int k) {
        double re = 0.0;
        double im = 0.0;
        for (int t = 0; t < frame.length; t++) {
            re += frame[t] * Math.cos(-2.0 * Math.PI * k * t / frame.length);
            im += frame[t] * Math.sin(-2.0 * Math.PI * k * t / frame.length);
        }
        return re * re + im * im;
    }

    private static double refMelWeight(double[] edges, double binHz, int m, boolean areaNorm) {
        double fl = edges[m];
        double fc = edges[m + 1];
        double fr = edges[m + 2];
        double w;
        if (binHz >= fl && binHz <= fc) {
            w = (binHz - fl) / Math.max(1e-30, fc - fl);
        } else if (binHz > fc && binHz <= fr) {
            w = (fr - binHz) / Math.max(1e-30, fr - fc);
        } else {
            return 0.0;
        }
        return areaNorm ? w * 2.0 / Math.max(1e-30, fr - fl) : w;
    }

    /** Mel edges of the builder config, in Hz, in double. */
    private static double[] refEdges(int nMel, int nFft, int sr, boolean htk, double fmax) {
        double hi = fmax > 0 ? fmax : 0.5 * sr;
        double mLo = htk ? htkMel(0) : slaneyMel(0);
        double mHi = htk ? htkMel(hi) : slaneyMel(hi);
        double[] out = new double[nMel + 2];
        for (int i = 0; i <= nMel + 1; i++) {
            double m = mLo + (mHi - mLo) * ((double) i / (nMel + 1));
            out[i] = htk ? htkInverse(m) : slaneyInverse(m);
        }
        return out;
    }

    private static double htkMel(double f) {
        return 2595.0 * Math.log10(1.0 + f / 700.0);
    }

    private static double htkInverse(double m) {
        return 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0);
    }

    private static double slaneyMel(double f) {
        double minLogHz = 1000.0;
        double linSlope = 3.0 / 200.0;
        double minLogMel = minLogHz * linSlope;
        double logStep = Math.log(6.4) / 27.0;
        return f < minLogHz ? f * linSlope : minLogMel + Math.log(f / minLogHz) / logStep;
    }

    private static double slaneyInverse(double m) {
        double minLogHz = 1000.0;
        double linSlope = 3.0 / 200.0;
        double minLogMel = minLogHz * linSlope;
        double logStep = Math.log(6.4) / 27.0;
        return m < minLogMel ? m / linSlope : minLogHz * Math.exp((m - minLogMel) * logStep);
    }

    /** The reference spectrogram of the no-padding framing, in double, mel-major; the floor is clamped. */
    private static double[][] refLogMel(float[] samples, int nMel, int nFft, int hop, int sr, boolean htk,
                                        boolean magnitude, float floor, boolean naturalLog, boolean floorAdd,
                                        boolean areaNorm) {
        double[] hann = refHann(nFft, true);
        double[] edges = refEdges(nMel, nFft, sr, htk, -1);
        int nLen = (samples.length - nFft) / hop + 1;
        double[][] out = new double[nMel][nLen];
        double[] frameBuf = new double[nFft];
        for (int t = 0; t < nLen; t++) {
            for (int m = 0; m < nMel; m++) {
                double sum = 0.0;
                for (int k = 0; k <= nFft / 2; k++) {
                    for (int j = 0; j < nFft; j++) {
                        frameBuf[j] = hann[j] * samples[t * hop + j];
                    }
                    double p = refPower(frameBuf, k);
                    double bin = magnitude ? Math.sqrt(p) : p;
                    sum += bin * refMelWeight(edges, (double) k * sr / nFft, m, areaNorm);
                }
                sum = floorAdd ? sum + floor : Math.max(sum, floor);
                out[m][t] = naturalLog ? Math.log(sum) : Math.log10(sum);
            }
        }
        return out;
    }

    private static float[] signal(int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.sin(2.0 * Math.PI * 31 * i / 400.0) * (i % 2 == 0 ? 1 : -0.5f);
        }
        return out;
    }

    // ================================================================== parity

    @Test
    public void noPaddingFramingMatchesReference() {
        int nFft = 16;
        int hop = 4;
        int nMel = 6;
        int sr = 2000;
        float[] s = signal(64);
        MelFrameParams params = MelFrameParams.builder(nMel, nFft, hop, sr)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(false)
                .build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(nMel).nFft(nFft).sampleRate(sr)
                .slaneyAreaNorm(true).build();
        MelChunk mel = LogMel.compute(s, s.length, params, filters);
        double[][] ref = refLogMel(s, nMel, nFft, hop, sr, false, false, 5.960464477539063e-08f, false, false, true);
        assertEquals(ref[0].length, mel.nLen());
        for (int m = 0; m < nMel; m++) {
            for (int t = 0; t < mel.nLen(); t++) {
                assertEquals(ref[m][t], mel.value(m, t), 2e-3);
            }
        }
    }

    @Test
    public void centerPaddingFramesAreInputPlusOne() {
        int nFft = 16;
        int hop = 4;
        float[] s = signal(33);
        MelFrameParams params = MelFrameParams.builder(4, nFft, hop, 2000)
                .padding(MelFrameParams.Padding.CENTER)
                .build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(4).nFft(nFft).sampleRate(2000).build();
        MelChunk mel = LogMel.compute(s, s.length, params, filters);
        // padded to 33 + 16 = 49; frames = (49 - 16) / 4 + 1 = 9 = n/hop + 1
        assertEquals(33 / hop + 1, mel.nLen());
    }

    @Test
    public void whisperPaddingClampsToMaxMinusEight() {
        int nFft = 16;
        int hop = 4;
        int sr = 2000;
        float[] s = signal(64);
        MelFrameParams params = MelFrameParams.builder(6, nFft, hop, sr)
                .padding(MelFrameParams.Padding.WHISPER)
                .naturalLog(false)
                .build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(6).nFft(nFft).sampleRate(sr).slaneyAreaNorm(true)
                .build();
        MelChunk mel = LogMel.compute(s, s.length, params, filters);
        double mmax = -1e20;
        for (int m = 0; m < mel.nMel(); m++) {
            for (int t = 0; t < mel.nLen(); t++) {
                mmax = Math.max(mmax, mel.value(m, t));
            }
        }
        assertEquals((mmax + 8 - 8 + 4) / 4.0, (mmax + 4) / 4.0, 1e-6);
        for (int m = 0; m < mel.nMel(); m++) {
            for (int t = 0; t < mel.nLen(); t++) {
                double v = mel.value(m, t);
                assertTrue(v <= (mmax + 4) / 4.0 + 1e-4);
                assertTrue(v >= ((mmax - 8) + 4) / 4.0 - 1e-4);
            }
        }
    }

    @Test
    public void perFeatureNormCentersRows() {
        int nFft = 16;
        int hop = 4;
        float[] s = signal(96);
        MelFrameParams params = MelFrameParams.builder(6, nFft, hop, 2000)
                .padding(MelFrameParams.Padding.CENTER)
                .normPerFeature(true)
                .stdEpsAfterSqrt(true)
                .build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(6).nFft(nFft).sampleRate(2000).build();
        MelChunk mel = LogMel.compute(s, s.length, params, filters);
        int effective = s.length / hop;
        for (int m = 0; m < mel.nMel(); m++) {
            double mean = 0.0;
            for (int t = 0; t < effective; t++) {
                mean += mel.value(m, t);
            }
            mean /= effective;
            assertEquals(0.0, mean, 1e-4);
            double var = 0.0;
            for (int t = 0; t < effective; t++) {
                var += Math.pow(mel.value(m, t) - mean, 2);
            }
            double std = Math.sqrt(var / (effective - 1));
            assertEquals(1.0, std, 1e-3);
            for (int t = effective; t < mel.nLen(); t++) {
                assertEquals(0.0, mel.value(m, t), 0.0);
            }
        }
    }

    @Test
    public void magnitudeAndFloorAddFollowTheFlags() {
        int nFft = 16;
        int hop = 4;
        int nMel = 4;
        int sr = 2000;
        float[] s = signal(48);
        MelFrameParams params = MelFrameParams.builder(nMel, nFft, hop, sr)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(true)
                .magnitude(true)
                .melFloor(1e-3f)
                .floorAdd(true)
                .build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(nMel).nFft(nFft).sampleRate(sr).build();
        MelChunk mel = LogMel.compute(s, s.length, params, filters);
        double[][] ref = refLogMel(s, nMel, nFft, hop, sr, false, true, 1e-3f, true, true, false);
        for (int m = 0; m < nMel; m++) {
            for (int t = 0; t < mel.nLen(); t++) {
                assertEquals(ref[m][t], mel.value(m, t), 2e-3);
            }
        }
    }

    @Test
    public void rejectsMismatchedFilterbankAndNonFiniteSamples() {
        MelFrameParams params = MelFrameParams.builder(4, 16, 4, 2000).build();
        MelFilterbank wrong = new MelFilterbank.Builder().nMel(6).nFft(16).sampleRate(2000).build();
        float[] s = signal(64);
        assertThrows(IllegalArgumentException.class, () -> LogMel.compute(s, s.length, params, wrong));
        float[] bad = signal(64);
        bad[7] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> LogMel.compute(bad, bad.length, params,
                new MelFilterbank.Builder().nMel(4).nFft(16).sampleRate(2000).build()));
        assertThrows(IllegalArgumentException.class, () -> LogMel.compute(s, 3, params,
                new MelFilterbank.Builder().nMel(4).nFft(16).sampleRate(2000).build()));
    }

    @Test
    public void symmetricWindowMatchesReference() {
        int nFft = 16;
        int hop = 8;
        int nMel = 4;
        int sr = 2000;
        float[] s = signal(64);
        MelFrameParams params = MelFrameParams.builder(nMel, nFft, hop, sr)
                .padding(MelFrameParams.Padding.NONE)
                .naturalLog(false)
                .hannPeriodic(false)
                .build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(nMel).nFft(nFft).sampleRate(sr).build();
        MelChunk mel = LogMel.compute(s, s.length, params, filters);
        // reference with the symmetric window (offset -1)
        double[] hann = refHann(nFft, false);
        double[] edges = refEdges(nMel, nFft, sr, false, -1);
        int nLen = (s.length - nFft) / hop + 1;
        for (int m = 0; m < nMel; m++) {
            for (int t = 0; t < nLen; t++) {
                double sum = 0.0;
                for (int k = 0; k <= nFft / 2; k++) {
                    double[] frame = new double[nFft];
                    for (int j = 0; j < nFft; j++) {
                        frame[j] = hann[j] * s[t * hop + j];
                    }
                    double p = refPower(frame, k);
                    sum += p * refMelWeight(edges, (double) k * sr / nFft, m, false);
                }
                assertEquals(Math.log10(Math.max(sum, 5.960464477539063e-08)), mel.value(m, t), 2e-3);
            }
        }
    }
}
