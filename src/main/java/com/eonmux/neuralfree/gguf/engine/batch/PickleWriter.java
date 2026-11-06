package com.eonmux.neuralfree.gguf.engine.batch;

import java.nio.charset.StandardCharsets;

/**
 * The bytes Python's {@code pickle.dumps(value, protocol=5)} writes for the block-hash inputs: protocol 5 with the
 * C pickler's framing — {@code PROTO}, then one {@code FRAME} whose length covers everything through {@code STOP}
 * once at least 4 bytes follow (a shorter tail rides without a frame header). Values are fresh structures, so no
 * object repeats inside one dump and no {@code MEMOIZE} back-reference appears; every string, bytes value and
 * tuple still carries the {@code MEMOIZE} opcode after it. Ints use {@code BININT1}/{@code BININT2}/{@code BININT}
 * (token ids are 32 bits, so {@code LONG1} never appears), strings {@code SHORT_BINUNICODE}/{@code BINUNICODE},
 * bytes {@code SHORT_BINBYTES}/{@code BINBYTES}. Inputs past one frame (64 KiB) are refused rather than framed
 * wrong.
 */
final class PickleWriter {

    /** The C pickler commits a frame when this much has buffered ({@code _FRAME_SIZE_TARGET}). */
    private static final int FRAME_SIZE_TARGET = 64 * 1024;
    /** A shorter tail rides without a frame header ({@code _FRAME_SIZE_MIN}). */
    private static final int FRAME_SIZE_MIN = 4;

    private byte[] buffer = new byte[64];
    private int length;
    /** How many objects the memo holds: every MEMOIZE adds one. */
    private int memoCount;
    /** The memo index of the strings {@link #tagString} wrote, by value (vLLM's shared tag constants). */
    private final java.util.Map<String, Integer> tags = new java.util.HashMap<>();

    PickleWriter() {
    }

    /**
     * A string that the input construction shares between several entries, vLLM's tag constants ({@code "mm"} and
     * friends are one interned object referenced by every entry that carries them): the first occurrence rides
     * {@code SHORT_BINUNICODE}/{@code BINUNICODE} plus {@code MEMOIZE}, any later one the {@code BINGET}
     * back-reference Python's pickler writes for the shared object.
     */
    void tagString(String value) {
        Integer index = tags.get(value);
        if (index != null) {
            if (index > 0xff) {
                writeByte((byte) 0x6a);
                writeInt32(index);
            } else {
                writeByte((byte) 0x68);
                writeByte((byte) (int) index);
            }
            return;
        }
        string(value);
        tags.put(value, memoCount - 1);
    }

    /** A string as {@code SHORT_BINUNICODE} or {@code BINUNICODE} over its UTF-8 bytes, then {@code MEMOIZE}. */
    void string(String value) {
        byte[] utf8 = value.getBytes(StandardCharsets.UTF_8);
        if (utf8.length <= 0xff) {
            writeByte((byte) 0x8c);
            writeByte((byte) utf8.length);
        } else if (utf8.length <= 0x7fffffff) {
            writeByte((byte) 0x58);
            writeInt32(utf8.length);
        } else {
            throw new IllegalArgumentException("a string of " + utf8.length + " bytes has no pickle short form");
        }
        writeBytes(utf8);
        writeByte((byte) 0x94);
        memoCount++;
    }

    /** Bytes as {@code SHORT_BINBYTES} or {@code BINBYTES}, then {@code MEMOIZE}. */
    void bytes(byte[] value) {
        if (value.length < 0x100) {
            writeByte((byte) 0x43);
            writeByte((byte) value.length);
        } else if (value.length <= 0x7fffffff) {
            writeByte((byte) 0x42);
            writeInt32(value.length);
        } else {
            throw new IllegalArgumentException(value.length + " bytes have no pickle short form");
        }
        writeBytes(value);
        writeByte((byte) 0x94);
        memoCount++;
    }

    /** An int as {@code BININT1} (0–255), {@code BININT2} (up to 65535) or {@code BININT} (signed 32-bit). */
    void integer(int value) {
        if (value >= 0 && value <= 0xff) {
            writeByte((byte) 0x4b);
            writeByte((byte) value);
        } else if (value >= 0 && value <= 0xffff) {
            writeByte((byte) 0x4d);
            writeByte((byte) value);
            writeByte((byte) (value >>> 8));
        } else {
            writeByte((byte) 0x4a);
            writeInt32(value);
        }
    }

    /** {@code null} as Python's {@code None}. */
    void nullValue() {
        writeByte((byte) 0x4e);
    }

    /**
     * Opens a tuple of {@code n} elements: {@code EMPTY_TUPLE} right away for {@code n == 0} (Python memoizes no
     * empty tuple), {@code MARK} for more than 3 elements, nothing for 1–3.
     */
    void tupleOpen(int n) {
        if (n == 0) {
            writeByte((byte) 0x29);
        } else if (n > 3) {
            writeByte((byte) 0x28);
        }
    }

    /** Closes the tuple {@code tupleOpen(n)} opened, its elements written: {@code TUPLEn} plus {@code MEMOIZE}. */
    void tupleClose(int n) {
        if (n == 0) {
            return;
        }
        writeByte((byte) (n <= 3 ? 0x84 + n : 0x74));
        writeByte((byte) 0x94);
        memoCount++;
    }

    /** The dump: {@code PROTO 5}, a frame over the tail when at least 4 bytes follow, and {@code STOP}. */
    byte[] finish() {
        int tail = length + 1;
        if (tail > FRAME_SIZE_TARGET) {
            throw new IllegalArgumentException("a block-hash input of " + tail + " bytes passes the one-frame"
                    + " pickle form");
        }
        byte[] out = new byte[2 + (tail >= FRAME_SIZE_MIN ? 9 : 0) + tail];
        out[0] = (byte) 0x80;
        out[1] = (byte) 0x05;
        int at = 2;
        if (tail >= FRAME_SIZE_MIN) {
            out[at++] = (byte) 0x95;
            // the frame length is an unsigned 64-bit little-endian int; a refused tail cannot reach 2^31
            for (int i = 0; i < 4; i++) {
                out[at++] = (byte) (tail >>> (i * 8));
            }
            for (int i = 4; i < 8; i++) {
                out[at++] = 0;
            }
        }
        System.arraycopy(buffer, 0, out, at, length);
        out[at + length] = 0x2e;
        return out;
    }

    private void writeInt32(int value) {
        writeByte((byte) value);
        writeByte((byte) (value >>> 8));
        writeByte((byte) (value >>> 16));
        writeByte((byte) (value >>> 24));
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
