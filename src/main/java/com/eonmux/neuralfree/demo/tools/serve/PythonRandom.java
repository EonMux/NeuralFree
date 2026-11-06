package com.eonmux.neuralfree.demo.tools.serve;

import java.util.List;

/**
 * CPython's {@code random.Random} generator: the Mersenne Twister with CPython's seeding and the
 * {@code getrandbits} draw that {@code shuffle} and {@code choices} consume. The file datasets
 * shuffle and sample in the reference with this generator, so the same seed puts the same records
 * in the same order here as it does in vLLM. Not thread-safe.
 */
public final class PythonRandom {

    private static final int N = 624;
    private static final int M = 397;
    private static final int MATRIX_A = 0x9908b0df;
    private static final int UPPER_MASK = 0x80000000;
    private static final int LOWER_MASK = 0x7fffffff;
    private static final double RECIPROCAL_TWO_POW_53 = 1.0 / 9007199254740992.0;

    private final int[] state = new int[N];
    private int index = N;

    /** A generator seeded as {@code random.Random(seed)} seeds one, the seed taken absolutely. */
    public PythonRandom(long seed) {
        initByArray(seedWords(seed));
    }

    /** The seed's absolute value as little-endian 32-bit words, the reference's key array. */
    private static int[] seedWords(long seed) {
        long value = seed < 0 ? -seed : seed;
        if (value == 0) {
            return new int[]{0};
        }
        int words = 0;
        for (long v = value; v != 0; v >>>= 32) {
            words++;
        }
        int[] key = new int[words];
        for (int i = 0; i < words; i++) {
            key[i] = (int) (value >>> (32 * i));
        }
        return key;
    }

    /** CPython's {@code init_by_array}. */
    private void initByArray(int[] key) {
        initGenrand(19650218);
        int i = 1;
        int j = 0;
        int k = Math.max(N, key.length);
        for (; k > 0; k--) {
            state[i] = (int) ((state[i] ^ ((state[i - 1] ^ (state[i - 1] >>> 30)) * 1664525L))
                    + key[j] + j);
            i++;
            j++;
            if (i >= N) {
                state[0] = state[N - 1];
                i = 1;
            }
            if (j >= key.length) {
                j = 0;
            }
        }
        for (k = N - 1; k > 0; k--) {
            state[i] = (int) ((state[i] ^ ((state[i - 1] ^ (state[i - 1] >>> 30)) * 1566083941L))
                    - i);
            i++;
            if (i >= N) {
                state[0] = state[N - 1];
                i = 1;
            }
        }
        state[0] = UPPER_MASK;
    }

    /** CPython's {@code init_genrand}. */
    private void initGenrand(int seed) {
        state[0] = seed;
        for (int i = 1; i < N; i++) {
            state[i] = (int) (1812433253L * (state[i - 1] ^ (state[i - 1] >>> 30)) + i);
        }
        index = N;
    }

    /** One 32-bit word of the twisted state, CPython's {@code genrand_uint32}. */
    private int nextUint32() {
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

    /** The block twist of the reference's generator. */
    private void twist() {
        int[] mag01 = {0, MATRIX_A};
        int kk = 0;
        for (; kk < N - M; kk++) {
            int y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
            state[kk] = state[kk + M] ^ (y >>> 1) ^ mag01[y & 1];
        }
        for (; kk < N - 1; kk++) {
            int y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
            state[kk] = state[kk + (M - N)] ^ (y >>> 1) ^ mag01[y & 1];
        }
        int y = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
        state[N - 1] = state[M - 1] ^ (y >>> 1) ^ mag01[y & 1];
        index = 0;
    }

    /** CPython's {@code random()}: 53 random bits over 2^53. */
    double nextDouble() {
        int a = nextUint32() >>> 5;
        int b = nextUint32() >>> 6;
        return (a * 67108864.0 + b) * RECIPROCAL_TWO_POW_53;
    }

    /** CPython's {@code getrandbits(k)} for the widths {@code _randbelow} asks for. */
    long getRandBits(int bits) {
        if (bits <= 32) {
            return bits == 0 ? 0 : (nextUint32() & 0xffffffffL) >>> (32 - bits);
        }
        long value = 0;
        int shift = 0;
        int remaining = bits;
        while (remaining > 0) {
            long word = nextUint32() & 0xffffffffL;
            int take = Math.min(remaining, 32);
            if (take < 32) {
                word >>>= (32 - take);
            }
            value |= word << shift;
            shift += take;
            remaining -= take;
        }
        return value;
    }

    /** CPython's {@code _randbelow_with_getrandbits(n)} for a positive bound. */
    int nextIntBelow(int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("a random bound must be positive, got " + bound);
        }
        int bits = 32 - Integer.numberOfLeadingZeros(bound);
        while (true) {
            long r = getRandBits(bits);
            if (r < bound) {
                return (int) r;
            }
        }
    }

    /** In-place Fisher-Yates shuffle, CPython's {@code shuffle}. */
    public void shuffle(List<?> list) {
        for (int i = list.size() - 1; i > 0; i--) {
            int j = nextIntBelow(i + 1);
            if (i != j) {
                java.util.Collections.swap(list, i, j);
            }
        }
    }

    /** CPython's {@code choice}: a uniform index over the list. */
    <T> T choice(List<T> list) {
        if (list.isEmpty()) {
            throw new IllegalArgumentException("cannot choose from an empty list");
        }
        return list.get(nextIntBelow(list.size()));
    }

    /** CPython's {@code choices(population, k)} with equal weights: {@code floor(random() * n)} draws. */
    <T> List<T> choices(List<T> population, int k) {
        if (population.isEmpty()) {
            throw new IllegalArgumentException("cannot choose from an empty population");
        }
        java.util.ArrayList<T> out = new java.util.ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            out.add(population.get((int) (nextDouble() * population.size())));
        }
        return out;
    }
}
