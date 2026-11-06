package com.eonmux.neuralfree.demo.tools;

/**
 * The mt19937 engine, the Mersenne Twister llama.cpp's task selection draws from ({@code std::mt19937},
 * seeded with 1 in perplexity.cpp), with the two bounded draws of the reference: libstdc++'s
 * {@code uniform_int_distribution} downscaling (the HellaSwag draw over the shrinking task pool) and
 * perplexity.cpp's {@code (int)(scale * rng() * size)} pick (the WinoGrande and multiple-choice selection).
 * The same dataset therefore selects the same tasks here as llama.cpp selects there.
 */
final class Mt19937 {

    /** The engine's output range, {@code mt19937::min() .. mt19937::max()}. */
    private static final long MIN = 0L;
    private static final long MAX = 0xFFFFFFFFL;

    private static final int N = 624;
    private static final int M = 397;
    private static final int MATRIX_A = 0x9908b0df;
    private static final int UPPER_MASK = 0x80000000;
    private static final int LOWER_MASK = 0x7fffffff;

    private final int[] state = new int[N];
    private int index = N;

    /** The generator at {@code seed}, as {@code std::mt19937(seed)} initializes it. */
    Mt19937(long seed) {
        int s = (int) seed;
        state[0] = s;
        for (int i = 1; i < N; i++) {
            int prev = state[i - 1];
            state[i] = 1812433253 * (prev ^ (prev >>> 30)) + i;
        }
    }

    /** The next raw 32-bit output as an unsigned value, as {@code rng()} produces it. */
    long nextUnsigned() {
        return nextInner() & 0xFFFFFFFFL;
    }

    /**
     * A draw in {@code [0, count)}, libstdc++'s {@code uniform_int_distribution(0, count - 1)} over this
     * engine: downscaling by {@code urngrange / count} with rejection above the representable multiple.
     *
     * @throws UnsupportedOperationException when {@code count} exceeds the engine's draw range, which the
     *                                       reference's task pools never do
     */
    long nextBounded(long count) {
        if (count < 1) {
            throw new IllegalArgumentException("a bounded draw needs a positive count, got " + count);
        }
        if (count > MAX) {
            throw new UnsupportedOperationException("a task pool of " + count + " entries exceeds the "
                    + "mt19937 draw range of " + MAX);
        }
        if (count == MAX) {
            // __urange == __urngrange: the raw draw maps one to one
            return nextUnsigned() - MIN;
        }
        long scaling = MAX / count;
        long past = count * scaling;
        long ret;
        do {
            ret = nextUnsigned() - MIN;
        } while (ret >= past);
        return ret / scaling;
    }

    /**
     * A draw in {@code [0, size)}, perplexity.cpp's {@code (int)(scale * rng() * size)} with
     * {@code scale = 1 / (1 + (float) mt19937::max())}: the float roundings of the reference, truncation
     * included, so the selection sequence is the one llama.cpp runs.
     */
    int nextFloatScaled(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("a float-scaled draw needs a positive size, got " + size);
        }
        float scale = 1.0f / (1.0f + (float) MAX);
        return (int) (scale * (float) nextUnsigned() * (float) size);
    }

    /** The next tempered output of the block generator. */
    private int nextInner() {
        if (index >= N) {
            twist();
        }
        int y = state[index++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
        return y;
    }

    /** Refills the state, the standard two-pass twist over the old block. */
    private void twist() {
        int i;
        for (i = 0; i < N - M; i++) {
            int y = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
            state[i] = state[i + M] ^ (y >>> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0);
        }
        for (; i < N - 1; i++) {
            int y = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
            state[i] = state[i + (M - N)] ^ (y >>> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0);
        }
        int y = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
        state[N - 1] = state[M - 1] ^ (y >>> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0);
        index = 0;
    }
}
