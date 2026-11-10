package com.eonmux.neuralfree.gguf.engine.sampling;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * UTF-8 bytes decoded as {@code new String(bytes, UTF_8)} decodes them (each malformed sequence becomes one
 * U+FFFD), together with the byte offset where every char starts. A match found in the text maps back to the bytes
 * exactly even when the bytes hold partial or invalid UTF-8, which re-encoding the text cannot do: a replaced
 * sequence of one or two bytes re-encodes as three. Immutable.
 */
final class DecodedBytes {

    private final String text;
    private final int[] byteAt;

    private DecodedBytes(String text, int[] byteAt) {
        this.text = text;
        this.byteAt = byteAt;
    }

    /** Decodes {@code bytes}, recording each char's byte offset. */
    static DecodedBytes of(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        ByteBuffer in = ByteBuffer.wrap(bytes);
        // every byte yields at most one char: a four-byte sequence gives a surrogate pair
        CharBuffer out = CharBuffer.allocate(bytes.length);
        int[] byteAt = new int[bytes.length + 1];
        while (true) {
            int validFrom = in.position();
            int charFrom = out.position();
            CoderResult r = decoder.decode(in, out, true);
            mapValid(bytes, validFrom, in.position(), byteAt, charFrom);
            if (r.isUnderflow()) {
                break;
            }
            if (!r.isMalformed() && !r.isUnmappable()) {
                throw new IllegalStateException("UTF-8 decoding of " + bytes.length + " bytes stopped with " + r);
            }
            byteAt[out.position()] = in.position();
            out.put('�');
            in.position(in.position() + r.length());
        }
        CoderResult flushed = decoder.flush(out);
        if (!flushed.isUnderflow()) {
            throw new IllegalStateException("UTF-8 decoding of " + bytes.length + " bytes failed to flush: "
                    + flushed);
        }
        int chars = out.position();
        byteAt[chars] = bytes.length;
        out.flip();
        return new DecodedBytes(out.toString(), byteAt);
    }

    /** Records the char offsets of the well-formed run {@code bytes[from, to)} that decoded from char {@code at}. */
    private static void mapValid(byte[] bytes, int from, int to, int[] byteAt, int at) {
        int c = at;
        int p = from;
        while (p < to) {
            int lead = bytes[p] & 0xFF;
            int len = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
            byteAt[c++] = p;
            if (len == 4) {
                byteAt[c++] = p;
            }
            p += len;
        }
    }

    /** The decoded text. */
    String text() {
        return text;
    }

    /**
     * The byte offset of char {@code index}: where its UTF-8 sequence (or the malformed bytes it replaces) starts;
     * {@code text().length()} maps to the byte count.
     *
     * @throws IndexOutOfBoundsException outside {@code [0, text().length()]}
     */
    int byteOffset(int index) {
        Objects.checkIndex(index, text.length() + 1);
        return byteAt[index];
    }
}
