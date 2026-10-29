package com.eonmux.neuralfree.gguf.engine.sampling;

/**
 * std::mt19937, llama.cpp's sampler RNG (every seeded stage constructs one from {@code get_rng_seed(seed)}, llama.cpp
 * src/llama-sampler.cpp). The standard 624x32 Mersenne Twister: the seed expansion and the twist follow the C++
 * standard's {@code mersenne_twister_engine} layout (u = 11, d, s = 7, b, t = 15, c, l = 18, a = 0x9908B0DF,
 * f = 1812433253), so a seeded engine's draw sequence matches llama.cpp's bit for bit.
 *
 * <p>The float and double images are llama.cpp's libstdc++ <em>generate_canonical</em> draws: a uniform float consumes
 * one twist output ({@code x / 2^32}, <em>float</em> arithmetic, retried when the rounding reaches 1.0f), a uniform
 * double consumes two ({@code (x1 + x2 * 2^32) / 2^64}, double arithmetic, the same retry); the retry is libstdc++'s
 * {@code nextafter(1.0)} guard. {@link #distDraw} is {@code llama_sample_dist} (src/llama-sampler.cpp:217) through
 * libstdc++ {@code std::discrete_distribution}: normalized prefix sums in double, and the first bin whose cumulative
 * probability reaches the drawn uniform - candidates of probability 0 are never selected.
 */
public final class Mt19937 {

    private static final int N = 624;
    private static final int M = 397;
    private static final int MATRIX_A = 0x9908b0df;
    private static final int UPPER = 0x80000000;
    private static final int LOWER = 0x7fffffff;
    private static final float TWO_POW_32_F = 4294967296.0f;
    private static final double TWO_POW_64 = 18446744073709551616.0; // 2^64

    private final int[] state = new int[N];
    private int pos = N;

    /** A fresh engine in llama.cpp's initial state: state 0 is the {@code uint32} seed, then the standard expansion. */
    public Mt19937(long seed) {
        state[0] = (int) (seed & 0xffffffffL);
        for (int i = 1; i < N; i++) {
            int prev = state[i - 1];
            state[i] = 1812433253 * (prev ^ (prev >>> 30)) + i;
        }
    }

    /** Re-seeds in place to the engine a fresh {@code std::mt19937(seed)} is in. */
    public void seed(long seed) {
        state[0] = (int) (seed & 0xffffffffL);
        for (int i = 1; i < N; i++) {
            int prev = state[i - 1];
            state[i] = 1812433253 * (prev ^ (prev >>> 30)) + i;
        }
        pos = N;
    }

    /** The next 32-bit twist output ({@code mt19937::operator()}). */
    public int next() {
        if (pos >= N) {
            twist();
        }
        int x = state[pos++];
        x ^= (x >>> 11);                                  // u = 11, d = 0xffffffff (no-op on ints)
        x ^= (x << 7) & 0x9d2c5680;                       // s = 7, b
        x ^= (x << 15) & 0xefc60000;                      // t = 15, c
        x ^= x >>> 18;                                    // l = 18
        return x;
    }

    /** The standard two-loop twist, which reads untouched values past index {@code N - M}. */
    private void twist() {
        for (int i = 0; i < N - M; i++) {                 // 0..226: state[i + M] is untouched
            state[i] = state[i + M] ^ twistStep(i);
        }
        for (int i = N - M; i < N - 1; i++) {             // 227..622: state[i + M - N] is already new
            state[i] = state[i + M - N] ^ twistStep(i);
        }
        int y = (state[N - 1] & UPPER) | (state[0] & LOWER);
        state[N - 1] = state[M - 1] ^ (y >>> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0);
        pos = 0;
    }

    /** The fold of {@code state[i]}'s favorite word: upper bit of state[i], low 31 bits of state[i + 1]. */
    private int twistStep(int i) {
        int y = (state[i] & UPPER) | (state[i + 1] & LOWER);
        int mag = (y & 1) != 0 ? MATRIX_A : 0;
        return (y >>> 1) ^ mag;
    }

    /**
     * libstdc++ <em>generate_canonical&lt;float, 24&gt;</em>: one 32-bit draw over 2^32, rounded to float, retried when
     * the rounding reaches 1.0f. {@code uniform_real_distribution<float>(0, 1)} calls this and adds nothing.
     */
    public float nextCanonicalFloat() {
        do {
            float v = (float) (next() & 0xffffffffL) / TWO_POW_32_F;
            if (v < 1.0f) {
                return v;
            }
        } while (true);
    }

    /**
     * libstdc++ <em>generate_canonical&lt;double, 53&gt;</em>: two 32-bit draws, {@code (x1 + x2 * 2^32) / 2^64} in
     * double, retried when the rounding reaches 1.0. {@code uniform_real_distribution<double>(0, 1)} calls this.
     */
    public double nextCanonicalDouble() {
        double v = 0.0;
        do {
            long low = next() & 0xffffffffL;
            long high = next() & 0xffffffffL;
            // the unsigned 64-bit fold as one double rounding: high's value scaled by the exact power of two,
            // plus the low word — the same nearest-double the unsigned integer converts to
            v = ((double) low + (double) high * 4294967296.0) / TWO_POW_64;
        } while (v >= 1.0);
        return v;
    }

    /**
     * llama_sample_dist (src/llama-sampler.cpp:217): the index of the first candidate whose cumulative probability
     * reaches {@code nextCanonicalDouble()}, the probabilities re-normalized in double as libstdc++
     * {@code std::discrete_distribution} does. Candidates of probability 0 are never selected; the last candidate wins
     * when the accumulated probability has not crossed the draw (its cumulative is 1.0 exactly in libstdc++).
     */
    public int distDraw(float[] probs) {
        double sum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            sum += probs[i];
        }
        double target = nextCanonicalDouble();
        double run = 0.0;
        for (int i = 0; i < probs.length; i++) {
            run += probs[i] / sum;
            if (run >= target) {
                return i;
            }
        }
        return probs.length - 1;
    }
}
