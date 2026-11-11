package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * The iterative FFT against the recursive radix-2 form of mtmd-audio.cpp it replaces (a copy kept here as the
 * reference): bit-identical outputs for power-of-two sizes, the Whisper {@code n_fft} of 400 (16 x 25), odd sizes
 * that are a single DFT and sizes in between.
 */
public class FftRecursionParityTest {

    @Test
    public void matchesTheRecursiveFormBitForBit() {
        int[] sizes = {2, 3, 4, 8, 25, 50, 64, 100, 400, 512, 1024, 6, 12, 400 * 3};
        Random random = new Random(7);
        for (int n : sizes) {
            Fft fft = new Fft(n);
            RecursiveFft reference = new RecursiveFft(n);
            for (int trial = 0; trial < 5; trial++) {
                float[] in = new float[n];
                for (int i = 0; i < n; i++) {
                    in[i] = trial == 4 && i % 3 == 0 ? -0.0f : random.nextFloat() * 2f - 1f;
                }
                float[] want = new float[2 * n];
                float[] got = new float[2 * n];
                reference.forward(in, want);
                fft.forward(in, got);
                for (int i = 0; i < 2 * n; i++) {
                    assertEquals("n=" + n + " trial " + trial + " index " + i,
                            Float.floatToRawIntBits(want[i]), Float.floatToRawIntBits(got[i]));
                }
            }
        }
    }

    /** The recursive form: four new arrays per node, a direct DFT at the odd leaves. */
    private static final class RecursiveFft {
        private final int n;
        private final float[] sinTable;
        private final float[] cosTable;

        RecursiveFft(int n) {
            this.n = n;
            this.sinTable = new float[n];
            this.cosTable = new float[n];
            for (int i = 0; i < n; i++) {
                double theta = (2.0 * Math.PI * i) / n;
                sinTable[i] = (float) Math.sin(theta);
                cosTable[i] = (float) Math.cos(theta);
            }
        }

        void forward(float[] in, float[] out) {
            float[] re = new float[n];
            float[] im = new float[n];
            System.arraycopy(in, 0, re, 0, n);
            fft(re, im, n);
            for (int k = 0; k < n; k++) {
                out[2 * k] = re[k];
                out[2 * k + 1] = im[k];
            }
        }

        private void fft(float[] re, float[] im, int len) {
            if (len == 1) {
                return;
            }
            int half = len / 2;
            if (len - 2 * half == 1) {
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
            fft(evenRe, evenIm, half);
            fft(oddRe, oddIm, half);
            int step = n / len;
            for (int k = 0; k < half; k++) {
                int idx = k * step;
                float wRe = cosTable[idx];
                float wIm = -sinTable[idx];
                float reOdd = oddRe[k];
                float imOdd = oddIm[k];
                float prodRe = wRe * reOdd - wIm * imOdd;
                float prodIm = wRe * imOdd + wIm * reOdd;
                re[k] = evenRe[k] + prodRe;
                im[k] = evenIm[k] + prodIm;
                re[k + half] = evenRe[k] - prodRe;
                im[k + half] = evenIm[k] - prodIm;
            }
        }

        private void dft(float[] re, float[] im, int len) {
            int step = n / len;
            float[] outRe = new float[len];
            float[] outIm = new float[len];
            for (int k = 0; k < len; k++) {
                float accRe = 0.0f;
                float accIm = 0.0f;
                for (int t = 0; t < len; t++) {
                    int idx = (k * t * step) % n;
                    float c = cosTable[idx];
                    float s = sinTable[idx];
                    accRe += re[t] * c + im[t] * s;
                    accIm += -re[t] * s + im[t] * c;
                }
                outRe[k] = accRe;
                outIm[k] = accIm;
            }
            System.arraycopy(outRe, 0, re, 0, len);
            System.arraycopy(outIm, 0, im, 0, len);
        }
    }
}
