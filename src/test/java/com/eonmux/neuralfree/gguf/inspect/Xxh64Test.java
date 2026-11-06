package com.eonmux.neuralfree.gguf.inspect;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link Xxh64} against the reference implementation's published and generated vectors: the one-shot hashes of
 * short strings, and the streaming state across every stripe boundary (lengths 0 to 100 in 3-byte updates), whose
 * expected values come from xxHash's own {@code XXH64_update}/{@code XXH64_digest}.
 */
public class Xxh64Test {

    @Test
    public void oneShotVectors() {
        assertEquals("ef46db3751d8e999", Xxh64.hex(Xxh64.hash(new byte[0], 0, 0)));
        assertEquals("d24ec4f1a98c6e5b", Xxh64.hex(Xxh64.hash(bytes("a"), 0, 1)));
        assertEquals("44bc2cf5ad770999", Xxh64.hex(Xxh64.hash(bytes("abc"), 0, 3)));
        assertEquals("fbcea83c8a378bf1", Xxh64.hex(Xxh64.hash(
                bytes("Nobody inspects the spammish repetition"), 0, 39)));
        assertEquals("45ab6734b21e6968", Xxh64.hex(Xxh64.hash(bytes("hello world"), 0, 11)));
    }

    @Test
    public void oneShotWithOffsetAndLength() {
        byte[] buf = bytes("xxhello worldxx");
        assertEquals("45ab6734b21e6968", Xxh64.hex(Xxh64.hash(buf, 2, 11)));
    }

    @Test
    public void streamingMatchesTheReferenceAcrossStripeBoundaries() {
        byte[] data = new byte[100];
        for (int i = 0; i < data.length; i++) {
            data[i] = (byte) i;
        }
        // expected values of XXH64_update in 3-byte steps, from the reference implementation
        assertEquals("ef46db3751d8e999", streamed(data, 0));
        assertEquals("884a173614b81b8d", streamed(data, 8));
        assertEquals("cbf59c5116ff32b4", streamed(data, 32));
        assertEquals("f7c67301db6713f0", streamed(data, 64));
        assertEquals("450baa11f6739216", streamed(data, 96));
    }

    @Test
    public void streamingMatchesOneShotForEveryLength() {
        byte[] data = new byte[200];
        for (int i = 0; i < data.length; i++) {
            data[i] = (byte) (i * 7);
        }
        for (int n = 0; n <= data.length; n++) {
            assertEquals("length " + n, Xxh64.hex(Xxh64.hash(data, 0, n)), streamed(data, n));
        }
    }

    @Test
    public void updateRefusesARangeOutsideTheArray() {
        Xxh64 state = new Xxh64();
        assertThrows(IllegalArgumentException.class, () -> state.update(new byte[4], 2, 3));
    }

    /** The hash of the first {@code n} bytes, fed 3 bytes at a time. */
    private static String streamed(byte[] data, int n) {
        Xxh64 state = new Xxh64();
        int i = 0;
        while (i < n) {
            int step = Math.min(3, n - i);
            state.update(data, i, step);
            i += step;
        }
        return Xxh64.hex(state.digest());
    }

    private static byte[] bytes(String s) {
        return s.getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }
}
