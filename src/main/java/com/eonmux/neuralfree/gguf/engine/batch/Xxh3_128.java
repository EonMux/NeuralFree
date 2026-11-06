package com.eonmux.neuralfree.gguf.engine.batch;

/**
 * XXH3's 128-bit digest (the {@code xxh3_128_digest} vLLM's {@code xxhash} and {@code xxhash_cbor} prefix-cache
 * hash functions use), ported from the reference implementation (xxhash.h): the length dispatch of
 * {@code XXH3_128bits} — 0–16, 17–128, 129–240 and the long-input accumulator loop — over the default secret, in
 * the seedless form ({@code XXH3_128bits}); the seed stays 0, the way vLLM's digest call runs, and
 * {@code init_none_hash} seeds through the hash input instead ({@link BlockHasher}). All arithmetic wraps (Java's
 * int and long do), the multiplies are unsigned, and every read is little-endian. Not thread-safe per call; the
 * methods are stateless.
 */
final class Xxh3_128 {

    private static final long PRIME32_1 = 0x9E3779B1L;
    private static final long PRIME32_2 = 0x85EBCA77L;
    private static final long PRIME32_3 = 0xC2B2AE3DL;
    private static final long PRIME64_1 = 0x9E3779B185EBCA87L;
    private static final long PRIME64_2 = 0xC2B2AE3D27D4EB4FL;
    private static final long PRIME64_3 = 0x165667B19E3779F9L;
    private static final long PRIME64_4 = 0x85EBCA77C2B2AE63L;
    private static final long PRIME64_5 = 0x27D4EB2F165667C5L;
    private static final long PRIME_MX1 = 0x165667919E3779F9L;
    private static final long PRIME_MX2 = 0x9FB21C651E98DF25L;

    /** Stripes of 64 bytes one accumulate pass takes; the secret advances 8 bytes per stripe. */
    private static final int STRIPE_LEN = 64;
    private static final int SECRET_CONSUME_RATE = 8;
    /** Where merging the accumulators starts in the secret ({@code XXH_SECRET_MERGEACCS_START}). */
    private static final int SECRET_MERGEACCS_START = 11;
    /** The last stripe's secret offset ({@code XXH_SECRET_LASTACC_START}). */
    private static final int SECRET_LASTACC_START = 7;
    /** The 129–240 rule's fixed offsets ({@code XXH3_MIDSIZE_STARTOFFSET} / {@code LASTOFFSET}, secret size 136). */
    private static final int MIDSIZE_STARTOFFSET = 3;
    private static final int MIDSIZE_LASTOFFSET = 17;
    private static final int SECRET_SIZE_MIN = 136;
    /** The default secret (xxhash.h's {@code XXH3_kSecret}), 192 bytes. */
    private static final byte[] SECRET = {
            (byte) 0xb8, (byte) 0xfe, 0x6c, 0x39, 0x23, (byte) 0xa4, 0x4b, (byte) 0xbe, 0x7c, 0x01, (byte) 0x81,
            0x2c, (byte) 0xf7, 0x21, (byte) 0xad, 0x1c,
            (byte) 0xde, (byte) 0xd4, 0x6d, (byte) 0xe9, (byte) 0x83, (byte) 0x90, (byte) 0x97, (byte) 0xdb, 0x72,
            0x40, (byte) 0xa4, (byte) 0xa4, (byte) 0xb7, (byte) 0xb3, 0x67, 0x1f,
            (byte) 0xcb, 0x79, (byte) 0xe6, 0x4e, (byte) 0xcc, (byte) 0xc0, (byte) 0xe5, 0x78, (byte) 0x82, 0x5a,
            (byte) 0xd0, 0x7d, (byte) 0xcc, (byte) 0xff, 0x72, 0x21,
            (byte) 0xb8, 0x08, 0x46, 0x74, (byte) 0xf7, 0x43, 0x24, (byte) 0x8e, (byte) 0xe0, 0x35, (byte) 0x90,
            (byte) 0xe6, (byte) 0x81, 0x3a, 0x26, 0x4c,
            0x3c, 0x28, 0x52, (byte) 0xbb, (byte) 0x91, (byte) 0xc3, 0x00, (byte) 0xcb, (byte) 0x88, (byte) 0xd0,
            0x65, (byte) 0x8b, 0x1b, 0x53, 0x2e, (byte) 0xa3,
            0x71, 0x64, 0x48, (byte) 0x97, (byte) 0xa2, 0x0d, (byte) 0xf9, 0x4e, 0x38, 0x19, (byte) 0xef, 0x46,
            (byte) 0xa9, (byte) 0xde, (byte) 0xac, (byte) 0xd8,
            (byte) 0xa8, (byte) 0xfa, 0x76, 0x3f, (byte) 0xe3, (byte) 0x9c, 0x34, 0x3f, (byte) 0xf9, (byte) 0xdc,
            (byte) 0xbb, (byte) 0xc7, (byte) 0xc7, 0x0b, 0x4f, 0x1d,
            (byte) 0x8a, 0x51, (byte) 0xe0, 0x4b, (byte) 0xcd, (byte) 0xb4, 0x59, 0x31, (byte) 0xc8, (byte) 0x9f,
            0x7e, (byte) 0xc9, (byte) 0xd9, 0x78, 0x73, 0x64,
            (byte) 0xea, (byte) 0xc5, (byte) 0xac, (byte) 0x83, 0x34, (byte) 0xd3, (byte) 0xeb, (byte) 0xc3,
            (byte) 0xc5, (byte) 0x81, (byte) 0xa0, (byte) 0xff, (byte) 0xfa, 0x13, 0x63, (byte) 0xeb,
            0x17, 0x0d, (byte) 0xdd, 0x51, (byte) 0xb7, (byte) 0xf0, (byte) 0xda, 0x49, (byte) 0xd3, 0x16, 0x55,
            0x26, 0x29, (byte) 0xd4, 0x68, (byte) 0x9e,
            0x2b, 0x16, (byte) 0xbe, 0x58, 0x7d, 0x47, (byte) 0xa1, (byte) 0xfc, (byte) 0x8f, (byte) 0xf8, (byte) 0xb8,
            (byte) 0xd1, 0x7a, (byte) 0xd0, 0x31, (byte) 0xce,
            0x45, (byte) 0xcb, 0x3a, (byte) 0x8f, (byte) 0x95, 0x16, 0x04, 0x28, (byte) 0xaf, (byte) 0xd7, (byte) 0xfb,
            (byte) 0xca, (byte) 0xbb, 0x4b, 0x40, 0x7e,
    };

    /** The seedless {@code XXH3_128bits}: the 16-byte digest of {@code input}, the canonical form (high 64 bits
     * big-endian first, then the low 64 bits), as {@code XXH128_canonicalFromHash} and Python's
     * {@code xxh3_128_digest} write it. */
    static byte[] digest(byte[] input) {
        long[] h = digest128(input);
        byte[] out = new byte[16];
        putLong(out, 0, h[1]);
        putLong(out, 8, h[0]);
        return out;
    }

    /** {@code input}'s digest as {low64, high64} (xxhash.h's {@code XXH128_hash_t} order). */
    private static long[] digest128(byte[] input) {
        int len = input.length;
        if (len <= 16) {
            return len0to16(input, len);
        }
        if (len <= 128) {
            return len17to128(input, len);
        }
        if (len <= 240) {
            return len129to240(input, len);
        }
        return hashLong(input, len);
    }

    // ------------------------------------------------------------------ short inputs

    /** {@code XXH3_len_0to16_128b}. */
    private static long[] len0to16(byte[] input, int len) {
        if (len > 8) {
            return len9to16(input, len);
        }
        if (len >= 4) {
            return len4to8(input, len);
        }
        if (len > 0) {
            return len1to3(input, len);
        }
        long bitflipl = read64(SECRET, 64) ^ read64(SECRET, 72);
        long bitfliph = read64(SECRET, 80) ^ read64(SECRET, 88);
        return new long[] {avalanche64(bitflipl), avalanche64(bitfliph)};
    }

    /** {@code XXH3_len_1to3_128b}. */
    private static long[] len1to3(byte[] input, int len) {
        int c1 = input[0] & 0xff;
        int c2 = input[len >> 1] & 0xff;
        int c3 = input[len - 1] & 0xff;
        int combined = (c1 << 16) | (c2 << 24) | c3 | (len << 8);
        int combinedH = Integer.rotateLeft(Integer.reverseBytes(combined), 13);
        long bitflipl = Integer.toUnsignedLong(read32(SECRET, 0) ^ read32(SECRET, 4));
        long bitfliph = Integer.toUnsignedLong(read32(SECRET, 8) ^ read32(SECRET, 12));
        return new long[] {avalanche64(Integer.toUnsignedLong(combined) ^ bitflipl),
                avalanche64(Integer.toUnsignedLong(combinedH) ^ bitfliph)};
    }

    /** {@code XXH3_len_4to8_128b}; the seed terms are 0 ({@code seed ^= swap32(seed) << 32}). */
    private static long[] len4to8(byte[] input, int len) {
        int inputLo = read32(input, 0);
        int inputHi = read32(input, len - 4);
        long input64 = Integer.toUnsignedLong(inputLo) | ((inputHi & 0xFFFFFFFFL) << 32);
        long bitflip = read64(SECRET, 16) ^ read64(SECRET, 24);
        long keyed = input64 ^ bitflip;
        long[] m = mult64to128(keyed, PRIME64_1 + (len << 2));
        m[1] += (m[0] << 1);
        m[0] ^= (m[1] >>> 3);
        m[0] = xorshift(m[0], 35) * PRIME_MX2;
        m[0] = xorshift(m[0], 28);
        m[1] = avalanche3(m[1]);
        return m;
    }

    /** {@code XXH3_len_9to16_128b}; the seed terms are 0. */
    private static long[] len9to16(byte[] input, int len) {
        long bitflipl = read64(SECRET, 32) ^ read64(SECRET, 40);
        long bitfliph = read64(SECRET, 48) ^ read64(SECRET, 56);
        long inputLo = read64(input, 0);
        long inputHi = read64(input, len - 8);
        long[] m = mult64to128(inputLo ^ inputHi ^ bitflipl, PRIME64_1);
        m[0] += (long) (len - 1) << 54;
        inputHi ^= bitfliph;
        m[1] += inputHi + (inputHi & 0xFFFFFFFFL) * (PRIME32_2 - 1);
        m[0] ^= Long.reverseBytes(m[1]);
        long[] h = mult64to128(m[0], PRIME64_2);
        h[1] += m[1] * PRIME64_2;
        return new long[] {avalanche3(h[0]), avalanche3(h[1])};
    }

    // ------------------------------------------------------------------ mid-size inputs

    /** {@code XXH3_len_17to128_128b}. */
    private static long[] len17to128(byte[] input, int len) {
        long accLo = (long) len * PRIME64_1;
        long accHi = 0;
        if (len > 32) {
            if (len > 64) {
                if (len > 96) {
                    long[] mixed = mix32B(accLo, accHi, input, 48, input, len - 64, SECRET, 96);
                    accLo = mixed[0];
                    accHi = mixed[1];
                }
                long[] mixed = mix32B(accLo, accHi, input, 32, input, len - 48, SECRET, 64);
                accLo = mixed[0];
                accHi = mixed[1];
            }
            long[] mixed = mix32B(accLo, accHi, input, 16, input, len - 32, SECRET, 32);
            accLo = mixed[0];
            accHi = mixed[1];
        }
        long[] mixed = mix32B(accLo, accHi, input, 0, input, len - 16, SECRET, 0);
        accLo = mixed[0];
        accHi = mixed[1];
        long lo = accLo + accHi;
        long hi = accLo * PRIME64_1 + accHi * PRIME64_4 + ((long) len * PRIME64_2);
        return new long[] {avalanche3(lo), -avalanche3(hi)};
    }

    /** {@code XXH3_len_129to240_128b}. */
    private static long[] len129to240(byte[] input, int len) {
        long accLo = (long) len * PRIME64_1;
        long accHi = 0;
        for (int i = 32; i < 160; i += 32) {
            long[] mixed = mix32B(accLo, accHi, input, i - 32, input, i - 16, SECRET, i - 32);
            accLo = mixed[0];
            accHi = mixed[1];
        }
        accLo = avalanche3(accLo);
        accHi = avalanche3(accHi);
        for (int i = 160; i <= len; i += 32) {
            long[] mixed = mix32B(accLo, accHi, input, i - 32, input, i - 16, SECRET, MIDSIZE_STARTOFFSET + i - 160);
            accLo = mixed[0];
            accHi = mixed[1];
        }
        long[] mixed = mix32B(accLo, accHi, input, len - 16, input, len - 32,
                SECRET, SECRET_SIZE_MIN - MIDSIZE_LASTOFFSET - 16);
        accLo = mixed[0];
        accHi = mixed[1];
        long lo = accLo + accHi;
        long hi = accLo * PRIME64_1 + accHi * PRIME64_4 + ((long) len * PRIME64_2);
        return new long[] {avalanche3(lo), -avalanche3(hi)};
    }

    /** {@code XXH128_mix32B}: {acc.low, acc.high} mixed with 32 input bytes at two offsets. */
    private static long[] mix32B(long accLo, long accHi, byte[] in1, int off1, byte[] in2, int off2, byte[] secret,
                                 int secretOff) {
        accLo += mix16B(in1, off1, secret, secretOff);
        accLo ^= read64(in2, off2) + read64(in2, off2 + 8);
        accHi += mix16B(in2, off2, secret, secretOff + 16);
        accHi ^= read64(in1, off1) + read64(in1, off1 + 8);
        return new long[] {accLo, accHi};
    }

    /** {@code XXH3_mix16B}; the seed terms are 0 ({@code secret ± 0}). */
    private static long mix16B(byte[] input, int off, byte[] secret, int secretOff) {
        long inputLo = read64(input, off);
        long inputHi = read64(input, off + 8);
        return mul128Fold(inputLo ^ read64(secret, secretOff), inputHi ^ read64(secret, secretOff + 8));
    }

    // ------------------------------------------------------------------ long inputs

    /** {@code XXH3_hashLong_128b_internal} over the default secret. */
    private static long[] hashLong(byte[] input, int len) {
        long[] acc = {PRIME32_3, PRIME64_1, PRIME64_2, PRIME64_3, PRIME64_4, PRIME32_2, PRIME64_5, PRIME32_1};
        hashLongLoop(acc, input, len);
        long lo = mergeAccs(acc, SECRET, SECRET_MERGEACCS_START, len * PRIME64_1);
        long hi = mergeAccs(acc, SECRET, SECRET.length - STRIPE_LEN - SECRET_MERGEACCS_START,
                ~((long) len * PRIME64_2));
        return new long[] {lo, hi};
    }

    /** {@code XXH3_hashLong_internal_loop}. */
    private static void hashLongLoop(long[] acc, byte[] input, int len) {
        int nbStripesPerBlock = (SECRET.length - STRIPE_LEN) / SECRET_CONSUME_RATE;
        int blockLen = STRIPE_LEN * nbStripesPerBlock;
        int nbBlocks = (len - 1) / blockLen;
        for (int n = 0; n < nbBlocks; n++) {
            accumulate(acc, input, n * blockLen, nbStripesPerBlock, SECRET, 0);
            scramble(acc, SECRET, SECRET.length - STRIPE_LEN);
        }
        int nbStripes = ((len - 1) - (blockLen * nbBlocks)) / STRIPE_LEN;
        accumulate(acc, input, nbBlocks * blockLen, nbStripes, SECRET, 0);
        accumulate(acc, input, len - STRIPE_LEN, 1, SECRET, SECRET.length - STRIPE_LEN - SECRET_LASTACC_START);
    }

    /** {@code XXH3_accumulate_512_scalar} over {@code nbStripes} 64-byte stripes. */
    private static void accumulate(long[] acc, byte[] input, int off, int nbStripes, byte[] secret, int secretOff) {
        for (int s = 0; s < nbStripes; s++) {
            for (int i = 0; i < 8; i++) {
                long dataVal = read64(input, off + s * STRIPE_LEN + i * 8);
                long dataKey = dataVal ^ read64(secret, secretOff + s * SECRET_CONSUME_RATE + i * 8);
                acc[i ^ 1] += dataVal;
                acc[i] += (dataKey & 0xFFFFFFFFL) * (dataKey >>> 32);
            }
        }
    }

    /** {@code XXH3_scrambleAcc_scalar}. */
    private static void scramble(long[] acc, byte[] secret, int secretOff) {
        for (int i = 0; i < 8; i++) {
            acc[i] = xorshift(acc[i], 47) ^ read64(secret, secretOff + i * 8);
            acc[i] *= PRIME32_1;
        }
    }

    /** {@code XXH3_mergeAccs}. */
    private static long mergeAccs(long[] acc, byte[] secret, int secretOff, long start) {
        long result = start;
        for (int i = 0; i < 4; i++) {
            result += mul128Fold(acc[2 * i] ^ read64(secret, secretOff + 16 * i),
                    acc[2 * i + 1] ^ read64(secret, secretOff + 16 * i + 8));
        }
        return avalanche3(result);
    }

    // ------------------------------------------------------------------ primitives

    /** {@code XXH3_avalanche}. */
    private static long avalanche3(long h) {
        return xorshift(xorshift(h, 37) * PRIME_MX1, 32);
    }

    /** {@code XXH64_avalanche}. */
    private static long avalanche64(long h) {
        h ^= h >>> 33;
        h *= PRIME64_2;
        h ^= h >>> 29;
        h *= PRIME64_3;
        h ^= h >>> 32;
        return h;
    }

    private static long xorshift(long v, int shift) {
        return v ^ (v >>> shift);
    }

    /** {@code XXH3_mul128_fold64}. */
    private static long mul128Fold(long lhs, long rhs) {
        long[] product = mult64to128(lhs, rhs);
        return product[0] ^ product[1];
    }

    /** Unsigned 64x64 multiply as {low64, high64} ({@code XXH_mult64to128}). */
    private static long[] mult64to128(long lhs, long rhs) {
        long loLo = (lhs & 0xFFFFFFFFL) * (rhs & 0xFFFFFFFFL);
        long hiLo = (lhs >>> 32) * (rhs & 0xFFFFFFFFL);
        long loHi = (lhs & 0xFFFFFFFFL) * (rhs >>> 32);
        long hiHi = (lhs >>> 32) * (rhs >>> 32);
        long cross = (loLo >>> 32) + (hiLo & 0xFFFFFFFFL) + (loHi & 0xFFFFFFFFL);
        long upper = hiHi + (hiLo >>> 32) + (loHi >>> 32) + (cross >>> 32);
        return new long[] {(loLo & 0xFFFFFFFFL) | (cross << 32), upper};
    }

    private static int read32(byte[] bytes, int off) {
        return (bytes[off] & 0xff) | ((bytes[off + 1] & 0xff) << 8) | ((bytes[off + 2] & 0xff) << 16)
                | ((bytes[off + 3] & 0xff) << 24);
    }

    private static long read64(byte[] bytes, int off) {
        return Integer.toUnsignedLong(read32(bytes, off)) | ((long) read32(bytes, off + 4) << 32);
    }

    private static void putLong(byte[] bytes, int off, long value) {
        for (int i = 0; i < 8; i++) {
            bytes[off + i] = (byte) (value >>> ((7 - i) * 8));
        }
    }
}
