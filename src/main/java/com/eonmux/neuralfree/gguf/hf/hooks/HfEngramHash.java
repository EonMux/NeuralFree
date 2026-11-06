package com.eonmux.neuralfree.gguf.hf.hooks;

import java.util.Objects;

/**
 * The Engram hash multipliers of the DeepSeek-V4.1 converter (deepseek41.py:45-52): numpy's
 * {@code default_rng(10007 * layer_id).integers(0, bound, size=max_ngram_size, dtype=int64)}, then {@code v * 2 + 1}
 * so every multiplier is odd. The stream is numpy's {@code SeedSequence} mixing into a {@code PCG64} bit generator
 * (PCG-XSL-RR 128/64) and Lemire's bounded integer method, reproduced bit for bit.
 */
final class HfEngramHash {

    private static final int INIT_A = 0x43b0d7e5;
    private static final int MULT_A = 0x931e8875;
    private static final int INIT_B = 0x8b51f9dd;
    private static final int MULT_B = 0x58f38ded;
    private static final int MIX_MULT_L = 0xca01f9dd;
    private static final int MIX_MULT_R = 0x4973f715;
    private static final int XSHIFT = 16;
    private static final int POOL_SIZE = 4;

    /** {@code PCG_DEFAULT_MULTIPLIER_128}, high and low words. */
    private static final long MULTIPLIER_HIGH = 2549297995355413924L;
    private static final long MULTIPLIER_LOW = 4865540595714422341L;

    private HfEngramHash() {
    }

    /**
     * The multiplier rows of {@code layerIds}, {@code maxNgram} per layer in layer order (deepseek41.py:45-52).
     *
     * @param layerIds         the Engram layer ids
     * @param maxNgram         the n-gram size
     * @param compressedVocab  the compressed vocabulary size
     * @throws IllegalArgumentException when the compressed vocabulary size is not positive
     */
    static long[] multipliers(int[] layerIds, int maxNgram, long compressedVocab) {
        Objects.requireNonNull(layerIds, "layerIds");
        if (compressedVocab <= 0) {
            throw new IllegalArgumentException("the compressed vocabulary size must be positive, got "
                    + compressedVocab);
        }
        if (maxNgram < 0) {
            throw new IllegalArgumentException("the n-gram size must be non-negative, got " + maxNgram);
        }
        long bound = Math.max(1, (Long.MAX_VALUE / compressedVocab) / 2);
        long[] out = new long[Math.multiplyExact(layerIds.length, maxNgram)];
        int at = 0;
        for (int layer : layerIds) {
            Pcg64 rng = new Pcg64(generateState(10007L * layer));
            for (int i = 0; i < maxNgram; i++) {
                out[at++] = boundedLemire(rng, bound - 1) * 2 + 1;
            }
        }
        return out;
    }

    /** {@code SeedSequence(entropy).generate_state(4, np.uint64)}: four little-endian 64-bit words. */
    static long[] generateState(long entropy) {
        if (entropy < 0) {
            throw new IllegalArgumentException("the seed entropy must be non-negative, got " + entropy);
        }
        int[] pool = new int[POOL_SIZE];
        int[] hash = {INIT_A};
        int[] words = entropyWords(entropy);
        for (int i = 0; i < POOL_SIZE; i++) {
            pool[i] = hashmix(i < words.length ? words[i] : 0, hash);
        }
        for (int src = 0; src < POOL_SIZE; src++) {
            for (int dst = 0; dst < POOL_SIZE; dst++) {
                if (src != dst) {
                    pool[dst] = mix(pool[dst], hashmix(pool[src], hash));
                }
            }
        }
        for (int src = POOL_SIZE; src < words.length; src++) {
            for (int dst = 0; dst < POOL_SIZE; dst++) {
                pool[dst] = mix(pool[dst], hashmix(words[src], hash));
            }
        }
        int[] state = new int[8];
        hash[0] = INIT_B;
        for (int i = 0; i < state.length; i++) {
            int value = pool[i % POOL_SIZE];
            value ^= hash[0];
            hash[0] *= MULT_B;
            value *= hash[0];
            value ^= value >>> XSHIFT;
            state[i] = value;
        }
        long[] out = new long[4];
        for (int i = 0; i < out.length; i++) {
            out[i] = (state[2 * i] & 0xFFFFFFFFL) | ((long) (state[2 * i + 1] & 0xFFFFFFFFL) << 32);
        }
        return out;
    }

    /** {@code _int_to_uint32_array}: the non-negative integer as little-endian 32-bit words. */
    private static int[] entropyWords(long entropy) {
        if (entropy == 0) {
            return new int[]{0};
        }
        int[] words = new int[2];
        int count = 0;
        long value = entropy;
        while (value > 0) {
            words[count++] = (int) value;
            value >>>= 32;
        }
        int[] out = new int[count];
        System.arraycopy(words, 0, out, 0, count);
        return out;
    }

    private static int hashmix(int value, int[] hash) {
        value ^= hash[0];
        hash[0] *= MULT_A;
        value *= hash[0];
        value ^= value >>> XSHIFT;
        return value;
    }

    private static int mix(int x, int y) {
        int result = MIX_MULT_L * x - MIX_MULT_R * y;
        result ^= result >>> XSHIFT;
        return result;
    }

    /**
     * {@code random_bounded_uint64(state, 0, rng, mask, use_masked=False)} for a range above 32 bits: Lemire's
     * algorithm with the rejection loop of distributions.c:1216-1260.
     */
    private static long boundedLemire(Pcg64 rng, long range) {
        long exclusive = range + 1;
        while (true) {
            long value = rng.next64();
            long low = value * exclusive;
            long high = multiplyHigh(value, exclusive);
            if (Long.compareUnsigned(low, exclusive) >= 0) {
                return high;
            }
            long threshold = Long.remainderUnsigned(-1L - range, exclusive);
            if (Long.compareUnsigned(low, threshold) >= 0) {
                return high;
            }
        }
    }

    /** The high 64 bits of the unsigned 128-bit product of {@code a} and {@code b}. */
    static long multiplyHigh(long a, long b) {
        return Math.multiplyHigh(a, b) + ((a >> 63) & b) + ((b >> 63) & a);
    }

    /** A PCG64 state seeded like numpy's {@code PCG64(SeedSequence)}. */
    private static final class Pcg64 {
        private long stateHigh;
        private long stateLow;
        private final long incrementHigh;
        private final long incrementLow;

        Pcg64(long[] seed) {
            long initHigh = seed[0];
            long initLow = seed[1];
            incrementHigh = (seed[2] << 1) | (seed[3] >>> 63);
            incrementLow = (seed[3] << 1) | 1L;
            step();
            stateLow += initLow;
            stateHigh += initHigh + (Long.compareUnsigned(stateLow, initLow) < 0 ? 1 : 0);
            step();
        }

        /** {@code rng->state = rng->state * PCG_DEFAULT_MULTIPLIER_128 + rng->inc}. */
        private void step() {
            long low = stateLow * MULTIPLIER_LOW;
            long high = multiplyHigh(stateLow, MULTIPLIER_LOW)
                    + stateLow * MULTIPLIER_HIGH + stateHigh * MULTIPLIER_LOW;
            long nextLow = low + incrementLow;
            long carry = Long.compareUnsigned(nextLow, low) < 0 ? 1 : 0;
            stateHigh = high + incrementHigh + carry;
            stateLow = nextLow;
        }

        /** {@code pcg_output_xsl_rr_128_64}: {@code rotr64(high ^ low, high >> 58)} after one step. */
        long next64() {
            step();
            return Long.rotateRight(stateHigh ^ stateLow, (int) (stateHigh >>> 58));
        }
    }
}
