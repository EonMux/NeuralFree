package com.eonmux.neuralfree.gguf.inspect;

/**
 * Streaming xxHash64 (xxHash's {@code XXH64}), the non-cryptographic hash of llama.cpp's {@code gguf-hash}.
 * The state machine follows xxHash's reference implementation: a 32-byte internal buffer feeds four
 * accumulators, and {@link #digest()} finalizes a copy of the state, so a state may be digested and updated
 * again. Hashes are unsigned 64-bit values held in a {@code long}; {@link #hex(long)} prints them the way the
 * reference prints them (16 lower-case hex digits, most significant first).
 *
 * <p>Not thread-safe.</p>
 */
public final class Xxh64 {

    private static final long PRIME64_1 = 0x9E3779B185EBCA87L;
    private static final long PRIME64_2 = 0xC2B2AE3D27D4EB4FL;
    private static final long PRIME64_3 = 0x165667B19E3779F9L;
    private static final long PRIME64_4 = 0x85EBCA77C2B2AE63L;
    private static final long PRIME64_5 = 0x27D4EB2F165667C5L;

    private static final int STRIPE_BYTES = 32;

    private final long seed;
    private final long[] v = new long[4];
    private final byte[] mem = new byte[STRIPE_BYTES];
    private int memSize;
    private long totalLength;

    /** A state for seed 0, the seed of every hash of the GGUF tools. */
    public Xxh64() {
        this(0L);
    }

    /** A state for the given seed. */
    public Xxh64(long seed) {
        this.seed = seed;
        v[0] = seed + PRIME64_1 + PRIME64_2;
        v[1] = seed + PRIME64_2;
        v[2] = seed;
        v[3] = seed - PRIME64_1;
    }

    /** One-shot hash of {@code len} bytes of {@code input} from {@code offset}, with seed 0. */
    public static long hash(byte[] input, int offset, int len) {
        return hash(input, offset, len, 0L);
    }

    /** One-shot hash of {@code len} bytes of {@code input} from {@code offset}. */
    public static long hash(byte[] input, int offset, int len, long seed) {
        Xxh64 state = new Xxh64(seed);
        state.update(input, offset, len);
        return state.digest();
    }

    /** The 16 lower-case hex digits of a hash, most significant byte first. */
    public static String hex(long hash) {
        StringBuilder s = new StringBuilder(16);
        for (int shift = 60; shift >= 0; shift -= 4) {
            s.append(Character.forDigit((int) (hash >>> shift) & 0xF, 16));
        }
        return s.toString();
    }

    /** Feeds {@code len} bytes of {@code input} from {@code offset} into the state. */
    public void update(byte[] input, int offset, int len) {
        if (input == null) {
            throw new IllegalArgumentException("input is null");
        }
        if (offset < 0 || len < 0 || len > input.length - offset) {
            throw new IllegalArgumentException("range [" + offset + ", +" + len + ") exceeds array length "
                    + input.length);
        }
        totalLength += len;
        int p = offset;
        int end = offset + len;
        if (len < STRIPE_BYTES - memSize) {
            System.arraycopy(input, p, mem, memSize, len);
            memSize += len;
            return;
        }
        if (memSize > 0) {
            int fill = STRIPE_BYTES - memSize;
            System.arraycopy(input, p, mem, memSize, fill);
            for (int i = 0; i < 4; i++) {
                v[i] = round(v[i], readLongLE(mem, i * 8));
            }
            p += fill;
            memSize = 0;
        }
        if (p + STRIPE_BYTES <= end) {
            int limit = end - STRIPE_BYTES;
            do {
                for (int i = 0; i < 4; i++) {
                    v[i] = round(v[i], readLongLE(input, p));
                    p += 8;
                }
            } while (p <= limit);
        }
        if (p < end) {
            memSize = end - p;
            System.arraycopy(input, p, mem, 0, memSize);
        }
    }

    /** The hash of everything fed so far; leaves the state usable for further updates. */
    public long digest() {
        long h;
        if (totalLength >= STRIPE_BYTES) {
            h = Long.rotateLeft(v[0], 1) + Long.rotateLeft(v[1], 7) + Long.rotateLeft(v[2], 12)
                    + Long.rotateLeft(v[3], 18);
            h = mergeRound(h, v[0]);
            h = mergeRound(h, v[1]);
            h = mergeRound(h, v[2]);
            h = mergeRound(h, v[3]);
        } else {
            h = seed + PRIME64_5;
        }
        h += totalLength;
        int p = 0;
        int len = memSize;
        while (len >= 8) {
            h ^= round(0L, readLongLE(mem, p));
            p += 8;
            h = Long.rotateLeft(h, 27) * PRIME64_1 + PRIME64_4;
            len -= 8;
        }
        if (len >= 4) {
            h ^= (readIntLE(mem, p) & 0xFFFF_FFFFL) * PRIME64_1;
            p += 4;
            h = Long.rotateLeft(h, 23) * PRIME64_2 + PRIME64_3;
            len -= 4;
        }
        while (len > 0) {
            h ^= (mem[p] & 0xFFL) * PRIME64_5;
            p++;
            h = Long.rotateLeft(h, 11) * PRIME64_1;
            len--;
        }
        h ^= h >>> 33;
        h *= PRIME64_2;
        h ^= h >>> 29;
        h *= PRIME64_3;
        h ^= h >>> 32;
        return h;
    }

    private static long round(long acc, long input) {
        acc += input * PRIME64_2;
        acc = Long.rotateLeft(acc, 31);
        acc *= PRIME64_1;
        return acc;
    }

    private static long mergeRound(long acc, long val) {
        acc ^= round(0L, val);
        return acc * PRIME64_1 + PRIME64_4;
    }

    private static long readLongLE(byte[] b, int off) {
        return (b[off] & 0xFFL)
                | (b[off + 1] & 0xFFL) << 8
                | (b[off + 2] & 0xFFL) << 16
                | (b[off + 3] & 0xFFL) << 24
                | (b[off + 4] & 0xFFL) << 32
                | (b[off + 5] & 0xFFL) << 40
                | (b[off + 6] & 0xFFL) << 48
                | (b[off + 7] & 0xFFL) << 56;
    }

    private static int readIntLE(byte[] b, int off) {
        return (b[off] & 0xFF)
                | (b[off + 1] & 0xFF) << 8
                | (b[off + 2] & 0xFF) << 16
                | (b[off + 3] & 0xFF) << 24;
    }
}
