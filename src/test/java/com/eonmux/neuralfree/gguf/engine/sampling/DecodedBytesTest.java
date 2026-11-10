package com.eonmux.neuralfree.gguf.engine.sampling;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link DecodedBytes} decodes exactly as {@code new String(bytes, UTF_8)} does and maps every char back to the
 * first byte of the sequence it came from: one-to-four-byte characters, surrogate pairs, truncated sequences and
 * stray continuation bytes.
 */
public class DecodedBytesTest {

    private static byte[] bytes(int... v) {
        byte[] b = new byte[v.length];
        for (int i = 0; i < v.length; i++) {
            b[i] = (byte) v[i];
        }
        return b;
    }

    @Test
    public void wellFormedCharactersMapToTheirLeadBytes() {
        byte[] b = "aé叫😀z".getBytes(StandardCharsets.UTF_8);
        DecodedBytes d = DecodedBytes.of(b);
        assertEquals("aé叫😀z", d.text());
        int[] expected = {0, 1, 3, 6, 6, 10, 11};
        for (int i = 0; i < expected.length; i++) {
            assertEquals("char " + i, expected[i], d.byteOffset(i));
        }
    }

    @Test
    public void eachMalformedSequenceIsOneReplacementAtItsOwnOffset() {
        DecodedBytes d = DecodedBytes.of(bytes(0xE5, 0x8F, '<', 0xAB, 't', 0xE5, 0x8F));
        assertEquals("�<�t�", d.text());
        int[] expected = {0, 2, 3, 4, 5, 7};
        for (int i = 0; i < expected.length; i++) {
            assertEquals("char " + i, expected[i], d.byteOffset(i));
        }
        assertThrows(IndexOutOfBoundsException.class, () -> d.byteOffset(6));
        assertThrows(IndexOutOfBoundsException.class, () -> d.byteOffset(-1));
    }

    @Test
    public void randomBytesDecodeLikeTheStringConstructorWithMonotoneOffsets() {
        Random rnd = new Random(7);
        int[] pool = {'a', '<', 0x80, 0x8F, 0xAB, 0xBF, 0xC2, 0xC3, 0xE5, 0xED, 0xF0, 0xF4, 0x9F, 0xFF};
        for (int trial = 0; trial < 5000; trial++) {
            byte[] b = new byte[rnd.nextInt(12)];
            for (int i = 0; i < b.length; i++) {
                b[i] = (byte) pool[rnd.nextInt(pool.length)];
            }
            DecodedBytes d = DecodedBytes.of(b);
            String expected = new String(b, StandardCharsets.UTF_8);
            assertEquals("trial " + trial, expected, d.text());
            assertEquals(b.length, d.byteOffset(expected.length()));
            for (int i = 0; i < expected.length(); i++) {
                int at = d.byteOffset(i);
                int next = d.byteOffset(i + 1);
                char c = expected.charAt(i);
                String where = "trial " + trial + " char " + i;
                if (Character.isHighSurrogate(c)) {
                    assertEquals(where, at, next);
                    continue;
                }
                assertTrue(where, next > at);
                String whole = Character.isLowSurrogate(c) ? expected.substring(i - 1, i + 1)
                        : expected.substring(i, i + 1);
                assertEquals(where, whole, new String(Arrays.copyOfRange(b, at, next), StandardCharsets.UTF_8));
            }
        }
    }
}
