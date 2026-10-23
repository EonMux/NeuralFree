package com.eonmux.neuralfree.tensor;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** FLOAT16 and BFLOAT16 conversions: rounding ties, subnormals, overflow, NaN, and exact round trips. */
public class HalfFloatsTest {

    private static void assertHalf(int expected, float value) {
        assertEquals("f16 of " + value + " (0x" + Integer.toHexString(Float.floatToRawIntBits(value)) + ")",
                Integer.toHexString(expected), Integer.toHexString(HalfFloats.floatToHalf(value)));
    }

    private static void assertBf16(int expected, int floatBits) {
        assertEquals("bf16 of 0x" + Integer.toHexString(floatBits), Integer.toHexString(expected),
                Integer.toHexString(HalfFloats.floatBitsToBFloat16(floatBits)));
    }

    @Test
    public void halfOfExactValues() {
        assertHalf(0x0000, 0.0f);
        assertHalf(0x8000, -0.0f);
        assertHalf(0x3c00, 1.0f);
        assertHalf(0xbc00, -1.0f);
        assertHalf(0x3800, 0.5f);
        assertHalf(0x7bff, 65504.0f);
        assertHalf(0x0400, Math.scalb(1.0f, -14));
    }

    @Test
    public void halfRoundsTiesToEven() {
        assertHalf(0x3c00, 1.0f + Math.scalb(1.0f, -11));
        assertHalf(0x3c02, 1.0f + 3 * Math.scalb(1.0f, -11));
        assertHalf(0x3c01, 1.0f + Math.scalb(1.0f, -11) + Math.scalb(1.0f, -20));
    }

    @Test
    public void halfOverflowsToInfinity() {
        assertHalf(0x7bff, 65519.0f);
        assertHalf(0x7c00, 65520.0f);
        assertHalf(0x7c00, 1e10f);
        assertHalf(0xfc00, -1e10f);
        assertHalf(0x7c00, Float.POSITIVE_INFINITY);
        assertHalf(0xfc00, Float.NEGATIVE_INFINITY);
        assertHalf(0x7c00, Float.MAX_VALUE);
    }

    @Test
    public void halfPreservesSubnormals() {
        float tiny = Math.scalb(1.0f, -24);
        assertHalf(0x0001, tiny);
        assertHalf(0x03ff, 1023 * tiny);
        assertHalf(0x8001, -tiny);
        assertHalf(0x0000, Math.scalb(1.0f, -25));
        assertHalf(0x0001, 1.5f * Math.scalb(1.0f, -25));
        assertHalf(0x0002, 3 * Math.scalb(1.0f, -25));
        assertHalf(0x0002, 5 * Math.scalb(1.0f, -25) - Math.scalb(1.0f, -40));
        assertHalf(0x0000, Float.MIN_VALUE);
        assertHalf(0x8000, -Float.MIN_NORMAL);
        assertEquals(tiny, HalfFloats.halfToFloat(0x0001), 0.0f);
        assertEquals(1023 * tiny, HalfFloats.halfToFloat(0x03ff), 0.0f);
    }

    @Test
    public void halfOfNanIsQuietNanKeepingSign() {
        assertHalf(0x7e00, Float.NaN);
        assertEquals(0xfe00, HalfFloats.floatBitsToHalf(0xffc00000));
        assertEquals(0x7e00, HalfFloats.floatBitsToHalf(0x7f800001));
        assertTrue(Float.isNaN(HalfFloats.halfToFloat(0x7e00)));
        assertTrue(Float.isNaN(HalfFloats.halfToFloat(0x7c01)));
    }

    @Test
    public void everyHalfRoundTrips() {
        for (int h = 0; h < 0x10000; h++) {
            float f = HalfFloats.halfToFloat(h);
            boolean nan = (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0;
            if (nan) {
                assertTrue(Float.isNaN(f));
                assertEquals(h & 0x8000, HalfFloats.floatToHalf(f) & 0x8000);
            } else {
                assertEquals("round trip of 0x" + Integer.toHexString(h), h, HalfFloats.floatToHalf(f));
            }
        }
    }

    @Test
    public void halfMatchesDoublePrecisionReference() {
        Random r = new Random(7);
        for (int i = 0; i < 200_000; i++) {
            float x = Float.intBitsToFloat(r.nextInt());
            if (Float.isNaN(x)) {
                continue;
            }
            assertEquals("f16 of " + x, referenceHalfValue(x), HalfFloats.halfToFloat(HalfFloats.floatToHalf(x)), 0.0f);
        }
        for (int i = 0; i < 200_000; i++) {
            float x = (float) ((r.nextDouble() * 2 - 1) * Math.pow(2, r.nextInt(46) - 30));
            assertEquals("f16 of " + x, referenceHalfValue(x), HalfFloats.halfToFloat(HalfFloats.floatToHalf(x)), 0.0f);
        }
    }

    /** Round to nearest even on the FLOAT16 grid, in double arithmetic (exact for float inputs). */
    private static float referenceHalfValue(float x) {
        if (Float.isInfinite(x)) {
            return x;
        }
        int e = Math.max(Math.getExponent(x), -14);
        double ulp = Math.scalb(1.0, e - 10);
        double q = Math.rint(x / ulp) * ulp;
        if (Math.abs(q) >= 65520.0) {
            return Math.copySign(Float.POSITIVE_INFINITY, x);
        }
        if (q == 0.0) {
            return Math.copySign(0.0f, x);
        }
        return (float) q;
    }

    @Test
    public void bfloat16RoundsTiesToEven() {
        assertBf16(0x3f80, Float.floatToRawIntBits(1.0f));
        assertBf16(0x3f80, 0x3f808000);
        assertBf16(0x3f82, 0x3f818000);
        assertBf16(0x3f81, 0x3f808001);
        assertBf16(0x3f80, 0x3f807fff);
        assertBf16(0x8000, Float.floatToRawIntBits(-0.0f));
    }

    @Test
    public void bfloat16OverflowsToInfinityAndKeepsSubnormals() {
        assertBf16(0x7f80, Float.floatToRawIntBits(Float.MAX_VALUE));
        assertBf16(0xff80, Float.floatToRawIntBits(-Float.MAX_VALUE));
        assertBf16(0x7f80, Float.floatToRawIntBits(Float.POSITIVE_INFINITY));
        assertBf16(0x0001, 0x00010000);
        assertBf16(0x0000, Float.floatToRawIntBits(Float.MIN_VALUE));
        assertEquals(Float.intBitsToFloat(0x00010000), HalfFloats.bfloat16ToFloat(0x0001), 0.0f);
    }

    @Test
    public void bfloat16OfNanIsQuietNanKeepingSign() {
        assertBf16(0x7fc0, Float.floatToRawIntBits(Float.NaN));
        assertBf16(0xffc0, 0xffc00000);
        assertBf16(0x7fc0, 0x7f800001);
        assertBf16(0x7fc0, 0x7fffffff);
        assertTrue(Float.isNaN(HalfFloats.bfloat16ToFloat(0x7fc0)));
    }

    @Test
    public void everyBfloat16RoundTrips() {
        for (int b = 0; b < 0x10000; b++) {
            float f = HalfFloats.bfloat16ToFloat(b);
            if (Float.isNaN(f)) {
                assertEquals((b & 0x8000) | 0x7fc0, HalfFloats.floatToBFloat16(f));
            } else {
                assertEquals(b, HalfFloats.floatToBFloat16(f));
            }
        }
    }
}
