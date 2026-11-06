package com.eonmux.neuralfree.gguf.engine.watermark;

/**
 * Philox4x32-10 (Random123) as the watermark pseudorandom function: a 64-bit key, a context of preceding token ids
 * and a candidate token id map to a uniform draw. The construction matches vLLM's {@code PhiloxPRF} exactly, so
 * watermarks produced and detected here agree with vLLM's.
 *
 * <p>The context keys the stream first: the state starts from the context domain tag and the context width, the
 * context words are mixed in four at a time, and the final block uses the candidate token's id (the low two bits pick
 * the output word) so every vocabulary entry of one context draws an independent value from one keyed stream. The
 * uniform conversion keeps the high 24 bits of the 32-bit output, the same image vLLM's {@code uint32_to_uniform}
 * produces in {@code float32}. Immutable and thread-safe.
 */
public final class PhiloxPrf implements WatermarkPrf {

    /** The PRF version tag vLLM records ({@code prf_version}). */
    public static final String VERSION = "philox4x32-10-v1";

    private static final long UINT32_MASK = 0xFFFFFFFFL;
    private static final long PHILOX_M0 = 0xD2511F53L;
    private static final long PHILOX_M1 = 0xCD9E8D57L;
    private static final long PHILOX_W0 = 0x9E3779B9L;
    private static final long PHILOX_W1 = 0xBB67AE85L;
    private static final long CONTEXT_DOMAIN = 0x574D4358L;
    private static final long TOKEN_DOMAIN = 0x574D544BL;
    private static final double UNIFORM_DENOMINATOR = 16777217.0; // 2^24 + 1

    private final long key;

    /** A PRF over the 64-bit unsigned {@code key} (a {@code long} bit pattern). */
    public PhiloxPrf(long key) {
        this.key = key;
    }

    /** The secret key as its 64-bit unsigned value. */
    public long key() {
        return key;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Override
    public double uniform(int[] context, int tokenId) {
        long[] state = contextState(context);
        return uniformOf(state, tokenId);
    }

    @Override
    public void uniforms(int[] context, int[] tokenIds, int count, double[] out) {
        if (out.length < count) {
            throw new IllegalArgumentException("output array of " + out.length + " values cannot hold " + count);
        }
        long[] state = contextState(context);
        for (int i = 0; i < count; i++) {
            out[i] = uniformOf(state, tokenIds[i]);
        }
    }

    /** The four state words the context keys, before the per-token block. */
    private long[] contextState(int[] context) {
        long key0 = key & UINT32_MASK;
        long key1 = (key >>> 32) & UINT32_MASK;
        long state0 = CONTEXT_DOMAIN;
        long state1 = context.length & UINT32_MASK;
        long state2 = 0L;
        long state3 = 0L;
        long[] block = new long[4];
        long[] scratch = new long[4];
        for (int offset = 0; offset < context.length; offset += 4) {
            for (int index = 0; index < 4; index++) {
                int position = offset + index;
                block[index] = position < context.length
                        ? context[position] & UINT32_MASK
                        : (UINT32_MASK - index) & UINT32_MASK;
            }
            philox(state0 ^ block[0], state1 ^ block[1], state2 ^ block[2], state3 ^ block[3],
                    (key0 ^ offset) & UINT32_MASK, key1, scratch);
            state0 = scratch[0];
            state1 = scratch[1];
            state2 = scratch[2];
            state3 = scratch[3];
        }
        return new long[]{state0, state1, state2, state3};
    }

    /** The uniform draw of {@code tokenId} under a context's keyed state. */
    private double uniformOf(long[] state, int tokenId) {
        long tokenWords = tokenId & UINT32_MASK;
        long key0 = key & UINT32_MASK;
        long key1 = (key >>> 32) & UINT32_MASK;
        long[] output = new long[4];
        philox(tokenWords >> 2, state[0], state[1], state[2],
                (key0 ^ state[3]) & UINT32_MASK, (key1 ^ TOKEN_DOMAIN) & UINT32_MASK, output);
        int wordIndex = (int) (tokenWords & 3L);
        return uint32ToUniform(output[wordIndex]);
    }

    /** Philox4x32-10 over the four counter words and the two key words; the result lands in {@code out}. */
    private static void philox(long counter0, long counter1, long counter2, long counter3, long keyWord0,
                               long keyWord1, long[] out) {
        long c0 = counter0 & UINT32_MASK;
        long c1 = counter1 & UINT32_MASK;
        long c2 = counter2 & UINT32_MASK;
        long c3 = counter3 & UINT32_MASK;
        long k0 = keyWord0 & UINT32_MASK;
        long k1 = keyWord1 & UINT32_MASK;
        long[] product = new long[2];
        for (int round = 0; round < 10; round++) {
            mulhilo32(PHILOX_M0, c0, product);
            long high0 = product[0];
            long low0 = product[1];
            mulhilo32(PHILOX_M1, c2, product);
            long high1 = product[0];
            long low1 = product[1];
            long next0 = (high1 ^ c1 ^ k0) & UINT32_MASK;
            long next1 = low1;
            long next2 = (high0 ^ c3 ^ k1) & UINT32_MASK;
            long next3 = low0;
            c0 = next0;
            c1 = next1;
            c2 = next2;
            c3 = next3;
            if (round != 9) {
                k0 = (k0 + PHILOX_W0) & UINT32_MASK;
                k1 = (k1 + PHILOX_W1) & UINT32_MASK;
            }
        }
        out[0] = c0;
        out[1] = c1;
        out[2] = c2;
        out[3] = c3;
    }

    /** The 32x32-to-64-bit multiply: {@code product[0]} is the high half, {@code product[1]} the low half. */
    private static void mulhilo32(long multiplier, long value, long[] product) {
        long valueLow = value & 0xFFFFL;
        long valueHigh = (value >>> 16) & 0xFFFFL;
        long multiplierLow = multiplier & 0xFFFFL;
        long multiplierHigh = (multiplier >>> 16) & 0xFFFFL;
        long lowProduct = valueLow * multiplierLow;
        long crossProduct = valueHigh * multiplierLow + valueLow * multiplierHigh;
        long lowWithCarry = lowProduct + ((crossProduct & 0xFFFFL) << 16);
        long low = lowWithCarry & UINT32_MASK;
        long high = (valueHigh * multiplierHigh + (crossProduct >>> 16) + (lowWithCarry >>> 32)) & UINT32_MASK;
        product[0] = high;
        product[1] = low;
    }

    /** The high 24 bits as a {@code float32}-exact value in {@code (0, 1)} (vLLM's {@code uint32_to_uniform}). */
    private static double uint32ToUniform(long value) {
        long mantissa = (value & UINT32_MASK) >>> 8;
        return (float) (((double) mantissa + 1.0) / UNIFORM_DENOMINATOR);
    }
}
