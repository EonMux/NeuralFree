package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** The FFT against the naive DFT, Parseval's identity and the impulse response. */
public class FftTest {

    private static double naiveDftRe(double[] in, int k) {
        double re = 0.0;
        for (int t = 0; t < in.length; t++) {
            re += in[t] * Math.cos(-2.0 * Math.PI * k * t / in.length);
        }
        return re;
    }

    private static double naiveDftIm(double[] in, int k) {
        double im = 0.0;
        for (int t = 0; t < in.length; t++) {
            im += in[t] * Math.sin(-2.0 * Math.PI * k * t / in.length);
        }
        return im;
    }

    @Test
    public void matchesNaiveDftOnRandomSignal() {
        int n = 64;
        Fft fft = new Fft(n);
        float[] in = new float[n];
        double[] ref = new double[n];
        long seed = 42;
        for (int i = 0; i < n; i++) {
            seed = seed * 6364136223846793005L + 1442695040888963407L;
            in[i] = (float) ((seed >> 33) % 2001 / 1000.0 - 1.0);
            ref[i] = in[i];
        }
        float[] out = new float[2 * n];
        fft.forward(in, out);
        for (int k = 0; k <= n / 2; k++) {
            assertEquals(naiveDftRe(ref, k), out[2 * k], 2e-3);
            assertEquals(naiveDftIm(ref, k), out[2 * k + 1], 2e-3);
        }
    }

    @Test
    public void impulseGivesFlatSpectrum() {
        int n = 32;
        Fft fft = new Fft(n);
        float[] in = new float[n];
        in[0] = 1.0f;
        float[] out = new float[2 * n];
        fft.forward(in, out);
        for (int k = 0; k <= n / 2; k++) {
            assertEquals(1.0f, out[2 * k], 1e-5);
            assertEquals(0.0f, out[2 * k + 1], 1e-5);
        }
    }

    @Test
    public void parsevalHolds() {
        int n = 128;
        Fft fft = new Fft(n);
        float[] in = new float[n];
        double energy = 0.0;
        for (int i = 0; i < n; i++) {
            in[i] = (float) Math.sin(2.0 * Math.PI * 7 * i / n + 0.3);
            energy += in[i] * in[i];
        }
        float[] out = new float[2 * n];
        fft.forward(in, out);
        double spectrum = out[0] * out[0];
        double scale = 2.0;
        for (int k = 1; k < n / 2; k++) {
            spectrum += out[2 * k] * out[2 * k] + out[2 * k + 1] * out[2 * k + 1];
        }
        spectrum *= scale;
        // nyquist bin counted once
        spectrum += out[n] * out[n];
        // sum over all bins of |X|^2 equals n * the signal energy
        assertEquals(n * energy, spectrum, n * energy * 1e-3);
    }

    @Test
    public void realSineIsHermitian() {
        int n = 64;
        Fft fft = new Fft(n);
        float[] in = new float[n];
        for (int i = 0; i < n; i++) {
            in[i] = (float) Math.cos(2.0 * Math.PI * 5 * i / n);
        }
        float[] out = new float[2 * n];
        fft.forward(in, out);
        // k and n-k carry conjugate values; the stored low half is real-symmetric in magnitude
        for (int k = 0; k <= n / 2; k++) {
            double mag = Math.hypot(out[2 * k], out[2 * k + 1]);
            double expect = k == 5 ? n / 2.0 : 0.0;
            assertEquals(expect, mag, 1e-2);
        }
    }

    @Test
    public void rejectsShortArrays() {
        Fft fft = new Fft(16);
        assertThrows(IllegalArgumentException.class, () -> fft.forward(new float[8], new float[32]));
        assertThrows(IllegalArgumentException.class, () -> fft.forward(new float[16], new float[16]));
    }

    @Test
    public void oddSizeFallsBackToDft() {
        Fft fft = new Fft(15);
        float[] in = new float[15];
        double[] ref = new double[15];
        for (int i = 0; i < 15; i++) {
            in[i] = (float) Math.sin(i);
            ref[i] = in[i];
        }
        float[] out = new float[30];
        fft.forward(in, out);
        assertEquals(naiveDftRe(ref, 3), out[6], 1e-2);
        assertEquals(naiveDftIm(ref, 3), out[7], 1e-2);
    }
}
