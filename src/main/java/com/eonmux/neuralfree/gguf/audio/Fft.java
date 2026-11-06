package com.eonmux.neuralfree.gguf.audio;

/**
 * Forward FFT with the conventions of llama.cpp's mtmd audio frontend (tools/mtmd/mtmd-audio.cpp:17-33): the
 * shared sin/cos table {@code theta = 2*PI*i/n} and the Cooley-Tukey radix-2 recursion; odd sizes fall back to a
 * direct DFT. The real input of the mel framings is transformed as the complex signal with zero imaginary part.
 * One instance is state-free; {@link #forward} allocates its work arrays, so callers that frame repeatedly keep
 * one instance and reuse the output array.
 */
public final class Fft {

    private final int n;
    private final float[] sinTable;
    private final float[] cosTable;

    /**
     * @param n transform size; the mel framings use the power of two {@code n_fft}
     * @throws IllegalArgumentException when {@code n} is below 2
     */
    public Fft(int n) {
        if (n < 2) {
            throw new IllegalArgumentException("fft size must be at least 2, got " + n);
        }
        this.n = n;
        this.sinTable = new float[n];
        this.cosTable = new float[n];
        for (int i = 0; i < n; i++) {
            double theta = (2.0 * Math.PI * i) / n;
            sinTable[i] = (float) Math.sin(theta);
            cosTable[i] = (float) Math.cos(theta);
        }
    }

    /** Transform size. */
    public int size() {
        return n;
    }

    /**
     * Forward transform of the real signal {@code in[0..n)}: {@code out[2k]} and {@code out[2k+1]} receive the real
     * and imaginary part of bin {@code k}, {@code k <= n/2}. {@code in} is left untouched; {@code out} needs
     * {@code 2n} floats.
     *
     * @throws IllegalArgumentException when an array is too short
     */
    public void forward(float[] in, float[] out) {
        if (in.length < n) {
            throw new IllegalArgumentException("fft input needs " + n + " floats, got " + in.length);
        }
        if (out.length < 2 * n) {
            throw new IllegalArgumentException("fft output needs " + (2 * n) + " floats, got " + out.length);
        }
        float[] re = new float[n];
        float[] im = new float[n];
        System.arraycopy(in, 0, re, 0, n);
        fft(re, im, n);
        for (int k = 0; k < n; k++) {
            out[2 * k] = re[k];
            out[2 * k + 1] = im[k];
        }
    }

    /**
     * In-place radix-2 transform of the complex pairs {@code (re[t], im[t])}, {@code t < len}; both views are
     * {@code len} floats starting at index 0. The exponent of the k-th butterfly is {@code exp(-2*PI*i*k/len)},
     * read from the shared table at stride {@code n/len}.
     */
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

    /** Direct DFT of mtmd-audio.cpp {@code dft_impl<false, false>} for odd sizes. */
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
