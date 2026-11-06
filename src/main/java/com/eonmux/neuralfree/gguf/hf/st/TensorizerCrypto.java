package com.eonmux.neuralfree.gguf.hf.st;

import java.util.Arrays;

/**
 * The libsodium primitive tensorizer encrypts with: XSalsa20-Poly1305 ({@code crypto_secretbox_detached}).
 *
 * <p>A tensorizer chunk is encrypted in place, its 16-byte Poly1305 tag stored apart from it in the tensor's crypt
 * info. Decryption derives the XSalsa20 subkey with HSalsa20 from the first 16 nonce bytes, takes the tag key from
 * the first 32 bytes of the stream's block 0, and verifies the tag over the ciphertext. The plaintext then comes from
 * XORing the first 32 ciphertext bytes with block 0's second half and the rest with the stream from block 1 on, the
 * secretbox construction of NaCl. All arithmetic is local to this class; nothing here is used unless a checkpoint is
 * encrypted.</p>
 */
final class TensorizerCrypto {

    static final int KEY_BYTES = 32;
    static final int NONCE_BYTES = 24;
    static final int MAC_BYTES = 16;

    private static final int SIGMA0 = 0x61707865;
    private static final int SIGMA1 = 0x3320646e;
    private static final int SIGMA2 = 0x79622d32;
    private static final int SIGMA3 = 0x6b206574;

    private TensorizerCrypto() {
    }

    /**
     * Opens a detached secretbox chunk.
     *
     * @param key        32-byte secretbox key
     * @param nonce      24-byte nonce
     * @param mac        16-byte Poly1305 tag
     * @param ciphertext the ciphertext, which is overwritten with the plaintext
     * @throws IllegalStateException when the tag does not match the ciphertext
     */
    static void openDetached(byte[] key, byte[] nonce, byte[] mac, byte[] ciphertext) {
        byte[] subkey = hsalsa20(key, nonce, 0);
        byte[] block0 = salsa20Block(subkey, nonce, 16, 0);
        byte[] tag = poly1305(ciphertext, ciphertext.length, block0);
        if (!constantTimeEquals(tag, mac)) {
            throw new IllegalStateException("tensorizer chunk authentication failed: the Poly1305 tag does not match"
                    + " the ciphertext");
        }
        xorStream(ciphertext, subkey, nonce, 16, block0);
    }

    /** Constant-time equality of two byte arrays. */
    static boolean constantTimeEquals(byte[] a, byte[] b) {
        if (a.length != b.length) {
            return false;
        }
        int diff = 0;
        for (int i = 0; i < a.length; i++) {
            diff |= a[i] ^ b[i];
        }
        return diff == 0;
    }

    /** HSalsa20 of the 16-byte nonce prefix; {@code off} is the nonce offset. */
    static byte[] hsalsa20(byte[] key, byte[] nonce, int off) {
        if (key.length != KEY_BYTES || nonce.length < off + 16) {
            throw new IllegalArgumentException("HSalsa20 needs a 32-byte key and a 16-byte nonce prefix");
        }
        int[] x = new int[16];
        x[0] = SIGMA0;
        x[1] = leInt(key, 0);
        x[2] = leInt(key, 4);
        x[3] = leInt(key, 8);
        x[4] = leInt(key, 12);
        x[5] = SIGMA1;
        x[6] = leInt(nonce, off);
        x[7] = leInt(nonce, off + 4);
        x[8] = leInt(nonce, off + 8);
        x[9] = leInt(nonce, off + 12);
        x[10] = SIGMA2;
        x[11] = leInt(key, 16);
        x[12] = leInt(key, 20);
        x[13] = leInt(key, 24);
        x[14] = leInt(key, 28);
        x[15] = SIGMA3;
        int[] original = x.clone();
        rounds(x);
        byte[] out = new byte[32];
        putInt(out, 0, x[0]);
        putInt(out, 4, x[5]);
        putInt(out, 8, x[10]);
        putInt(out, 12, x[15]);
        putInt(out, 16, x[6]);
        putInt(out, 20, x[7]);
        putInt(out, 24, x[8]);
        putInt(out, 28, x[9]);
        Arrays.fill(original, 0);
        return out;
    }

    /**
     * XORs {@code data} with the secretbox stream whose block 0 is {@code block0}, the 8-byte nonce at {@code off}.
     * The construction takes block 0's second half for the first 32 bytes and continues from block 1.
     */
    private static void xorStream(byte[] data, byte[] key, byte[] nonce, int off, byte[] block0) {
        int head = Math.min(32, data.length);
        for (int i = 0; i < head; i++) {
            data[i] ^= block0[32 + i];
        }
        if (data.length == head) {
            return;
        }
        byte[] block = new byte[64];
        for (int pos = head; pos < data.length; pos += 64) {
            long streamOffset = pos - head;
            salsa20BlockInto(key, nonce, off, 1 + streamOffset / 64, block);
            int within = (int) (streamOffset % 64);
            int n = Math.min(64 - within, data.length - pos);
            for (int i = 0; i < n; i++) {
                data[pos + i] ^= block[within + i];
            }
        }
        Arrays.fill(block, (byte) 0);
    }

    private static byte[] salsa20Block(byte[] key, byte[] nonce, int off, long counter) {
        byte[] out = new byte[64];
        salsa20BlockInto(key, nonce, off, counter, out);
        return out;
    }

    private static void salsa20BlockInto(byte[] key, byte[] nonce, int off, long counter, byte[] out) {
        int[] x = new int[16];
        x[0] = SIGMA0;
        x[1] = leInt(key, 0);
        x[2] = leInt(key, 4);
        x[3] = leInt(key, 8);
        x[4] = leInt(key, 12);
        x[5] = SIGMA1;
        x[6] = leInt(nonce, off);
        x[7] = leInt(nonce, off + 4);
        x[8] = (int) counter;
        x[9] = (int) (counter >>> 32);
        x[10] = SIGMA2;
        x[11] = leInt(key, 16);
        x[12] = leInt(key, 20);
        x[13] = leInt(key, 24);
        x[14] = leInt(key, 28);
        x[15] = SIGMA3;
        int[] original = x.clone();
        rounds(x);
        for (int i = 0; i < 16; i++) {
            putInt(out, i * 4, x[i] + original[i]);
        }
    }

    /** The 20 Salsa20 rounds (10 column/row double rounds) in place. */
    private static void rounds(int[] x) {
        for (int i = 0; i < 10; i++) {
            // column round
            x[4] ^= rotl(x[0] + x[12], 7);
            x[8] ^= rotl(x[4] + x[0], 9);
            x[12] ^= rotl(x[8] + x[4], 13);
            x[0] ^= rotl(x[12] + x[8], 18);
            x[9] ^= rotl(x[5] + x[1], 7);
            x[13] ^= rotl(x[9] + x[5], 9);
            x[1] ^= rotl(x[13] + x[9], 13);
            x[5] ^= rotl(x[1] + x[13], 18);
            x[14] ^= rotl(x[10] + x[6], 7);
            x[2] ^= rotl(x[14] + x[10], 9);
            x[6] ^= rotl(x[2] + x[14], 13);
            x[10] ^= rotl(x[6] + x[2], 18);
            x[3] ^= rotl(x[15] + x[11], 7);
            x[7] ^= rotl(x[3] + x[15], 9);
            x[11] ^= rotl(x[7] + x[3], 13);
            x[15] ^= rotl(x[11] + x[7], 18);
            // row round
            x[1] ^= rotl(x[0] + x[3], 7);
            x[2] ^= rotl(x[1] + x[0], 9);
            x[3] ^= rotl(x[2] + x[1], 13);
            x[0] ^= rotl(x[3] + x[2], 18);
            x[6] ^= rotl(x[5] + x[4], 7);
            x[7] ^= rotl(x[6] + x[5], 9);
            x[4] ^= rotl(x[7] + x[6], 13);
            x[5] ^= rotl(x[4] + x[7], 18);
            x[11] ^= rotl(x[10] + x[9], 7);
            x[8] ^= rotl(x[11] + x[10], 9);
            x[9] ^= rotl(x[8] + x[11], 13);
            x[10] ^= rotl(x[9] + x[8], 18);
            x[12] ^= rotl(x[15] + x[14], 7);
            x[13] ^= rotl(x[12] + x[15], 9);
            x[14] ^= rotl(x[13] + x[12], 13);
            x[15] ^= rotl(x[14] + x[13], 18);
        }
    }

    private static int rotl(int v, int n) {
        return (v << n) | (v >>> (32 - n));
    }

    // ------------------------------------------------------------------ Poly1305

    /** The 16-byte Poly1305 tag of {@code message[0, length)} under the 32-byte one-time {@code key}. */
    static byte[] poly1305(byte[] message, int length, byte[] key) {
        // 26-bit limbs of r, with the clamping constants
        long t0 = leInt(key, 0) & 0xffffffffL;
        long t1 = leInt(key, 4) & 0xffffffffL;
        long t2 = leInt(key, 8) & 0xffffffffL;
        long t3 = leInt(key, 12) & 0xffffffffL;
        long r0 = t0 & 0x3ffffffL;
        long r1 = ((t0 >>> 26) | (t1 << 6)) & 0x3ffff03L;
        long r2 = ((t1 >>> 20) | (t2 << 12)) & 0x3ffc0ffL;
        long r3 = ((t2 >>> 14) | (t3 << 18)) & 0x3f03fffL;
        long r4 = (t3 >>> 8) & 0x00fffffL;
        long s1 = r1 * 5;
        long s2 = r2 * 5;
        long s3 = r3 * 5;
        long s4 = r4 * 5;
        long h0 = 0;
        long h1 = 0;
        long h2 = 0;
        long h3 = 0;
        long h4 = 0;
        long mask26 = 0x3ffffffL;
        for (int i = 0; i < length; i += 16) {
            int n = Math.min(16, length - i);
            byte[] block = new byte[16];
            System.arraycopy(message, i, block, 0, n);
            if (n < 16) {
                block[n] = 1;
            }
            long m0 = leInt(block, 0) & 0xffffffffL;
            long m1 = leInt(block, 4) & 0xffffffffL;
            long m2 = leInt(block, 8) & 0xffffffffL;
            long m3 = leInt(block, 12) & 0xffffffffL;
            long hibit = n < 16 ? 0 : (1L << 24);
            h0 += m0 & mask26;
            h1 += ((m0 >>> 26) | (m1 << 6)) & mask26;
            h2 += ((m1 >>> 20) | (m2 << 12)) & mask26;
            h3 += ((m2 >>> 14) | (m3 << 18)) & mask26;
            h4 += (m3 >>> 8) | hibit;

            long d0 = h0 * r0 + h1 * s4 + h2 * s3 + h3 * s2 + h4 * s1;
            long d1 = h0 * r1 + h1 * r0 + h2 * s4 + h3 * s3 + h4 * s2;
            long d2 = h0 * r2 + h1 * r1 + h2 * r0 + h3 * s4 + h4 * s3;
            long d3 = h0 * r3 + h1 * r2 + h2 * r1 + h3 * r0 + h4 * s4;
            long d4 = h0 * r4 + h1 * r3 + h2 * r2 + h3 * r1 + h4 * r0;

            long c = d0 >>> 26;
            h0 = d0 & mask26;
            d1 += c;
            c = d1 >>> 26;
            h1 = d1 & mask26;
            d2 += c;
            c = d2 >>> 26;
            h2 = d2 & mask26;
            d3 += c;
            c = d3 >>> 26;
            h3 = d3 & mask26;
            d4 += c;
            c = d4 >>> 26;
            h4 = d4 & mask26;
            h0 += c * 5;
            c = h0 >>> 26;
            h0 &= mask26;
            h1 += c;
        }
        // full carry
        long c = h1 >>> 26;
        h1 &= mask26;
        h2 += c;
        c = h2 >>> 26;
        h2 &= mask26;
        h3 += c;
        c = h3 >>> 26;
        h3 &= mask26;
        h4 += c;
        c = h4 >>> 26;
        h4 &= mask26;
        h0 += c * 5;
        c = h0 >>> 26;
        h0 &= mask26;
        h1 += c;

        // compute h + -p and select
        long g0 = h0 + 5;
        c = g0 >>> 26;
        g0 &= mask26;
        long g1 = h1 + c;
        c = g1 >>> 26;
        g1 &= mask26;
        long g2 = h2 + c;
        c = g2 >>> 26;
        g2 &= mask26;
        long g3 = h3 + c;
        c = g3 >>> 26;
        g3 &= mask26;
        long g4 = h4 + c - (1L << 26);

        long select = (g4 >>> 63) - 1; // all ones when h >= p
        long notSelect = ~select;
        h0 = (h0 & notSelect) | (g0 & select);
        h1 = (h1 & notSelect) | (g1 & select);
        h2 = (h2 & notSelect) | (g2 & select);
        h3 = (h3 & notSelect) | (g3 & select);
        h4 = (h4 & notSelect) | (g4 & select);

        long f0 = h0 | (h1 << 26);
        long f1 = (h1 >>> 6) | (h2 << 20);
        long f2 = (h2 >>> 12) | (h3 << 14);
        long f3 = (h3 >>> 18) | (h4 << 8);

        long pad0 = leInt(key, 16) & 0xffffffffL;
        long pad1 = leInt(key, 20) & 0xffffffffL;
        long pad2 = leInt(key, 24) & 0xffffffffL;
        long pad3 = leInt(key, 28) & 0xffffffffL;

        f0 = (f0 & 0xffffffffL) + pad0;
        f1 = (f1 & 0xffffffffL) + pad1 + (f0 >>> 32);
        f2 = (f2 & 0xffffffffL) + pad2 + (f1 >>> 32);
        f3 = (f3 & 0xffffffffL) + pad3 + (f2 >>> 32);

        byte[] tag = new byte[16];
        putInt(tag, 0, (int) f0);
        putInt(tag, 4, (int) f1);
        putInt(tag, 8, (int) f2);
        putInt(tag, 12, (int) f3);
        return tag;
    }

    // ------------------------------------------------------------------ little-endian words

    private static int leInt(byte[] b, int off) {
        return (b[off] & 0xff) | ((b[off + 1] & 0xff) << 8) | ((b[off + 2] & 0xff) << 16)
                | ((b[off + 3] & 0xff) << 24);
    }

    private static void putInt(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
    }
}
