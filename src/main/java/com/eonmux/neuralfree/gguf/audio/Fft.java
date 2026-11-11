package com.eonmux.neuralfree.gguf.audio;

/**
 * Forward FFT with the conventions of llama.cpp's mtmd audio frontend (tools/mtmd/mtmd-audio.cpp:17-33): the
 * shared sin/cos table {@code theta = 2*PI*i/n} and the Cooley-Tukey radix-2 decimation in time; the odd factor of
 * the size falls back to a direct DFT. The real input of the mel framings is transformed as the complex signal with
 * zero imaginary part.
 *
 * <p>The transform runs iteratively in place over one work array pair: the {@code 2^s} decimated subsequences
 * {@code x[r + 2^s * t]} (the leaves of the radix-2 recursion, laid out in bit-reversed order of {@code r}) take
 * their direct DFT of the odd length, then {@code s} butterfly stages combine neighbouring blocks. Every leaf DFT
 * and every butterfly evaluates the same float expression with the same twiddle index as the recursive form, so
 * the output is bit-identical to it. One instance is state-free; {@link #forward} allocates its work arrays, so
 * callers that frame repeatedly keep one instance and reuse the output array.</p>
 */
public final class Fft {

    private final int n;
    /** The odd factor of {@code n}: the length of the leaf DFTs. */
    private final int base;
    /** Radix-2 stages: {@code n = base * 2^levels}. */
    private final int levels;
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
        this.levels = Integer.numberOfTrailingZeros(n);
        this.base = n >> levels;
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
        leaves(in, re, im);
        for (int len = 2 * base; len <= n; len *= 2) {
            butterflies(re, im, len);
        }
        for (int k = 0; k < n; k++) {
            out[2 * k] = re[k];
            out[2 * k + 1] = im[k];
        }
    }

    /**
     * Gathers each decimated subsequence into its block of {@code base} entries — block {@code j} holds residue
     * {@code bitReverse(j)} — and replaces it with its direct DFT when {@code base > 1} (a length-1 leaf is its own
     * transform).
     */
    private void leaves(float[] in, float[] re, float[] im) {
        int count = 1 << levels;
        float[] scratchRe = base > 1 ? new float[base] : null;
        float[] scratchIm = base > 1 ? new float[base] : null;
        for (int j = 0; j < count; j++) {
            int residue = levels == 0 ? 0 : Integer.reverse(j) >>> (Integer.SIZE - levels);
            int at = j * base;
            for (int t = 0; t < base; t++) {
                re[at + t] = in[residue + t * count];
            }
            if (base > 1) {
                dft(re, im, at, scratchRe, scratchIm);
            }
        }
    }

    /**
     * One radix-2 stage: every block of {@code len} entries holds the transforms of its even half then its odd half
     * and becomes the transform of the whole block. The exponent of the k-th butterfly is
     * {@code exp(-2*PI*i*k/len)}, read from the shared table at stride {@code n/len}. The butterflies of a stage are
     * independent, so the early stages (many short blocks) run twiddle-major: each twiddle is read once and applied
     * across every block.
     */
    private void butterflies(float[] re, float[] im, int len) {
        int half = len / 2;
        int step = n / len;
        if (half < step) {
            for (int k = 0; k < half; k++) {
                int idx = k * step;
                float wRe = cosTable[idx];
                float wIm = -sinTable[idx];
                for (int e = k; e < n; e += len) {
                    butterfly(re, im, e, e + half, wRe, wIm);
                }
            }
            return;
        }
        for (int b = 0; b < n; b += len) {
            for (int k = 0; k < half; k++) {
                int idx = k * step;
                butterfly(re, im, b + k, b + k + half, cosTable[idx], -sinTable[idx]);
            }
        }
    }

    /** The butterfly of the even entry {@code e} and the odd entry {@code o} under the twiddle {@code (wRe, wIm)}. */
    private static void butterfly(float[] re, float[] im, int e, int o, float wRe, float wIm) {
        float reOdd = re[o];
        float imOdd = im[o];
        float prodRe = wRe * reOdd - wIm * imOdd;
        float prodIm = wRe * imOdd + wIm * reOdd;
        float reEven = re[e];
        float imEven = im[e];
        re[e] = reEven + prodRe;
        im[e] = imEven + prodIm;
        re[o] = reEven - prodRe;
        im[o] = imEven - prodIm;
    }

    /** Direct DFT of mtmd-audio.cpp {@code dft_impl<false, false>} over the leaf block at {@code at}. */
    private void dft(float[] re, float[] im, int at, float[] outRe, float[] outIm) {
        int len = base;
        int step = n / len;
        for (int k = 0; k < len; k++) {
            float accRe = 0.0f;
            float accIm = 0.0f;
            // idx = (k * t * step) % n, advanced by k * step (below n) per t instead of recomputed
            int stride = k * step;
            int idx = 0;
            for (int t = 0; t < len; t++, idx = idx + stride >= n ? idx + stride - n : idx + stride) {
                float c = cosTable[idx];
                float s = sinTable[idx];
                accRe += re[at + t] * c + im[at + t] * s;
                accIm += -re[at + t] * s + im[at + t] * c;
            }
            outRe[k] = accRe;
            outIm[k] = accIm;
        }
        System.arraycopy(outRe, 0, re, at, len);
        System.arraycopy(outIm, 0, im, at, len);
    }
}
