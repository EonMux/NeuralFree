package com.eonmux.neuralfree.gguf.engine.batch;

import java.nio.charset.StandardCharsets;

/**
 * The bytes {@code cbor2.dumps(value, canonical=True)} writes for the block-hash inputs (RFC 8949 preferred
 * serialization): the shortest form of every head — arrays, text and byte strings and integers in the minimal
 * length encoding, {@code null} as {@code f6} — which is all canonical mode adds for these structures. A payload
 * longer than an unsigned 32-bit length is refused.
 */
final class CborWriter {

    private byte[] buffer = new byte[64];
    private int length;

    CborWriter() {
    }

    /** A text string in the minimal major type 3 form. */
    void string(String value) {
        byte[] utf8 = value.getBytes(StandardCharsets.UTF_8);
        head(3, utf8.length);
        writeBytes(utf8);
    }

    /** A byte string in the minimal major type 2 form. */
    void bytes(byte[] value) {
        head(2, value.length);
        writeBytes(value);
    }

    /** An unsigned int in the minimal major type 0 form. */
    void integer(int value) {
        if (value >= 0) {
            head(0, value);
        } else {
            // negative ints encode as -1 - n, major type 1
            head(1, -1 - value);
        }
    }

    /** {@code null} ({@code f6}). */
    void nullValue() {
        writeByte((byte) 0xf6);
    }

    /** An array head of {@code n} elements, major type 4. */
    void array(int n) {
        head(4, n);
    }

    /** The CBOR bytes; no trailer exists. */
    byte[] finish() {
        byte[] out = new byte[length];
        System.arraycopy(buffer, 0, out, 0, length);
        return out;
    }

    /** The minimal head of major type {@code major} carrying the unsigned argument {@code n}. */
    private void head(int major, long n) {
        int marker = major << 5;
        if (n < 0 || n > 0xffffffffL) {
            throw new IllegalArgumentException("a CBOR argument of " + n + " has no unsigned 32-bit form");
        }
        if (n < 24) {
            writeByte((byte) (marker | n));
        } else if (n <= 0xff) {
            writeByte((byte) (marker | 24));
            writeByte((byte) n);
        } else if (n <= 0xffff) {
            writeByte((byte) (marker | 25));
            writeByte((byte) (n >>> 8));
            writeByte((byte) n);
        } else {
            writeByte((byte) (marker | 26));
            writeByte((byte) (n >>> 24));
            writeByte((byte) (n >>> 16));
            writeByte((byte) (n >>> 8));
            writeByte((byte) n);
        }
    }

    private void writeByte(byte b) {
        grow(1);
        buffer[length++] = b;
    }

    private void writeBytes(byte[] bytes) {
        grow(bytes.length);
        System.arraycopy(bytes, 0, buffer, length, bytes.length);
        length += bytes.length;
    }

    private void grow(int n) {
        if (length + n > buffer.length) {
            int size = buffer.length;
            while (size < length + n) {
                size *= 2;
            }
            byte[] bigger = new byte[size];
            System.arraycopy(buffer, 0, bigger, 0, length);
            buffer = bigger;
        }
    }
}
