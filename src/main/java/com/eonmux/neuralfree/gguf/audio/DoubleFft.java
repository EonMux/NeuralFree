package com.eonmux.neuralfree.gguf.audio;

/**
 * Forward FFT of a real signal in double precision, for the frontends whose reference takes its spectra with
 * NumPy's float64 {@code rfft} (the Gemma 3n feature extractor): an iterative radix-2 transform over power-of-two
 * sizes with a double twiddle table. Immutable after construction; {@link #magnitudes} allocates nothing beyond its
 * work arrays, which belong to the instance, so one instance serves one thread.
 */
final class DoubleFft {

    private final int n;
    private final double[] cos;
    private final double[] sin;
    private final int[] bitReverse;
    private final double[] re;
    private final double[] im;

    /**
     * @param n transform size, a power of two of at least 2
     * @throws IllegalArgumentException for any other size
     */
    DoubleFft(int n) {
        if (n < 2 || Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("fft size must be a power of two of at least 2, got " + n);
        }
        this.n = n;
        this.cos = new double[n / 2];
        this.sin = new double[n / 2];
        for (int i = 0; i < n / 2; i++) {
            double theta = 2.0 * Math.PI * i / n;
            cos[i] = Math.cos(theta);
            sin[i] = Math.sin(theta);
        }
        int bits = Integer.numberOfTrailingZeros(n);
        this.bitReverse = new int[n];
        for (int i = 0; i < n; i++) {
            bitReverse[i] = Integer.reverse(i) >>> (32 - bits);
        }
        this.re = new double[n];
        this.im = new double[n];
    }

    /**
     * The magnitudes {@code |X[k]|} of bins {@code k <= n/2} of the real signal {@code in[0..len)}, zero-padded to
     * the transform size, into {@code out}.
     *
     * @throws IllegalArgumentException when {@code len} exceeds the size or {@code out} holds fewer than
     *                                  {@code n/2 + 1} values
     */
    void magnitudes(float[] in, int len, double[] out) {
        if (len < 0 || len > n || len > in.length) {
            throw new IllegalArgumentException("fft input of " + len + " samples does not fit size " + n);
        }
        if (out.length < n / 2 + 1) {
            throw new IllegalArgumentException("fft output needs " + (n / 2 + 1) + " values, got " + out.length);
        }
        for (int i = 0; i < n; i++) {
            int src = bitReverse[i];
            re[i] = src < len ? in[src] : 0.0;
            im[i] = 0.0;
        }
        for (int size = 2; size <= n; size <<= 1) {
            int half = size >> 1;
            int step = n / size;
            for (int start = 0; start < n; start += size) {
                for (int j = 0; j < half; j++) {
                    double wr = cos[j * step];
                    double wi = -sin[j * step];
                    int a = start + j;
                    int b = a + half;
                    double tr = wr * re[b] - wi * im[b];
                    double ti = wr * im[b] + wi * re[b];
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
        for (int k = 0; k <= n / 2; k++) {
            out[k] = Math.hypot(re[k], im[k]);
        }
    }
}
