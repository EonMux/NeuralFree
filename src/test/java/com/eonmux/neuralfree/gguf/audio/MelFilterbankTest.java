package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The mel filterbank against the formulas of tools/mtmd/mtmd-audio.cpp:35-130: the mel grid, the triangle
 * supports, the Slaney area norm and the HTK variant.
 */
public class MelFilterbankTest {

    /** Mel edge of filter {@code m} under the builder's scale, in Hz, re-derived in double. */
    private static double[] edges(MelFilterbank fb, int nFft, int sr, boolean htk, double fmin, double fmax) {
        double hi = fmax > 0 ? fmax : 0.5 * sr;
        double mLo = htk ? 2595.0 * Math.log10(1.0 + fmin / 700.0) : slaney(fmin);
        double mHi = htk ? 2595.0 * Math.log10(1.0 + hi / 700.0) : slaney(hi);
        int nMel = fb.nMel();
        double[] out = new double[nMel + 2];
        for (int i = 0; i <= nMel + 1; i++) {
            double m = mLo + (mHi - mLo) * ((double) i / (nMel + 1));
            out[i] = htk ? 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0) : slaneyInverse(m);
        }
        return out;
    }

    private static double slaney(double f) {
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

    @Test
    public void slaneyPeakMatchesMelGrid() {
        int nMel = 40;
        int nFft = 400;
        int sr = 16000;
        MelFilterbank fb = new MelFilterbank.Builder().nMel(nMel).nFft(nFft).sampleRate(sr).slaneyAreaNorm(true)
                .build();
        double[] e = edges(fb, nFft, sr, false, 0, -1);
        double binStep = (double) sr / nFft;
        for (int m = 0; m < nMel; m++) {
            // the peak sits at the center edge (k * binStep == f_center for some bin when the grid aligns)
            int peak = -1;
            double best = 0.0;
            for (int k = 0; k <= nFft / 2; k++) {
                if (fb.weight(m, k) > best) {
                    best = fb.weight(m, k);
                    peak = k;
                }
            }
            double fPeak = peak * binStep;
            assertTrue("filter " + m + " peak at " + fPeak + " vs edge " + e[m + 1], best > 0);
            assertTrue("filter " + m + " peak off its center edge: " + fPeak + " vs " + e[m + 1],
                    Math.abs(fPeak - e[m + 1]) <= binStep);
        }
    }

    @Test
    public void slaneyAreaNormScalesByBandwidth() {
        int nMel = 8;
        int nFft = 400;
        int sr = 16000;
        MelFilterbank normed = new MelFilterbank.Builder().nMel(nMel).nFft(nFft).sampleRate(sr).slaneyAreaNorm(true)
                .build();
        MelFilterbank plain = new MelFilterbank.Builder().nMel(nMel).nFft(nFft).sampleRate(sr).slaneyAreaNorm(false)
                .build();
        double[] e = edges(normed, nFft, sr, false, 0, -1);
        for (int m = 0; m < nMel; m++) {
            double enorm = 2.0 / Math.max(1e-30, e[m + 2] - e[m]);
            for (int k = 0; k <= nFft / 2; k++) {
                assertEquals((float) (plain.weight(m, k) * enorm), normed.weight(m, k), 1e-4);
            }
        }
    }

    @Test
    public void slaneyFilterIntegratesToUnitArea() {
        // with the area norm, sum(filter[k] * binStep) equals 1: the triangle's area 0.5 * base * height with
        // height = 2 / base
        int nFft = 400;
        int sr = 16000;
        MelFilterbank fb = new MelFilterbank.Builder().nMel(16).nFft(nFft).sampleRate(sr).slaneyAreaNorm(true)
                .build();
        double binStep = (double) sr / nFft;
        for (int m = 1; m < fb.nMel() - 1; m++) {
            double area = 0.0;
            for (int k = 0; k <= nFft / 2; k++) {
                area += fb.weight(m, k) * binStep;
            }
            assertEquals(1.0, area, 2e-2);
        }
    }

    @Test
    public void htkCenterMatchesHtkFormula() {
        int nMel = 20;
        int nFft = 512;
        int sr = 16000;
        MelFilterbank fb = new MelFilterbank.Builder().nMel(nMel).nFft(nFft).sampleRate(sr).slaneyAreaNorm(false)
                .useHtk(true).build();
        double[] e = edges(fb, nFft, sr, true, 0, -1);
        double binStep = (double) sr / nFft;
        for (int m = 0; m < nMel; m++) {
            int peak = -1;
            double best = 0.0;
            for (int k = 0; k <= nFft / 2; k++) {
                if (fb.weight(m, k) > best) {
                    best = fb.weight(m, k);
                    peak = k;
                }
            }
            assertTrue(best > 0);
            assertTrue(Math.abs(peak * binStep - e[m + 1]) <= binStep);
        }
    }

    @Test
    public void binsMatchNfftHalfPlusOne() {
        MelFilterbank fb = new MelFilterbank.Builder().nMel(80).nFft(400).sampleRate(16000).build();
        assertEquals(80, fb.nMel());
        assertEquals(201, fb.nFftBins());
        assertThrows(IndexOutOfBoundsException.class, () -> fb.weight(80, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> fb.weight(0, 201));
    }

    @Test
    public void scaleMultipliesEveryTap() {
        MelFilterbank one = new MelFilterbank.Builder().nMel(8).nFft(400).sampleRate(16000).build();
        MelFilterbank scaled = new MelFilterbank.Builder().nMel(8).nFft(400).sampleRate(16000).scale(0.5).build();
        for (int k = 0; k < 201; k++) {
            assertEquals(0.5f * one.weight(3, k), scaled.weight(3, k), 1e-6);
        }
    }
}
