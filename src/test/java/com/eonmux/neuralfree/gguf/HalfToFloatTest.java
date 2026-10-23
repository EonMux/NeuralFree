package com.eonmux.neuralfree.gguf;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Unit tests for {@link GGUFFileParser#halfToFloat(int)} (IEEE-754 binary16 -> binary32).
 *
 * <p>Subnormal halves (zero exponent) decode to mantissa * 2^-24 exactly; Q6_K block scales are
 * typically subnormal, so these cases pin the dequantization scale. All expected values are
 * exactly representable in float32.
 */
public class HalfToFloatTest {

    private static float h2f(int halfBits) {
        return GGUFFileParser.halfToFloat(halfBits);
    }

    private static void assertHalf(float expected, int halfBits) {
        float actual = h2f(halfBits);
        float tol = Math.max(Math.abs(expected) * 1e-6f, 1e-12f);
        assertEquals("halfToFloat(0x" + Integer.toHexString(halfBits) + ")", expected, actual, tol);
    }

    @Test
    public void zeros() {
        assertEquals(0.0f, h2f(0x0000), 0.0f);
        // Negative zero decodes to a zero-magnitude value.
        assertEquals(0.0f, Math.abs(h2f(0x8000)), 0.0f);
    }

    @Test
    public void normals() {
        assertHalf(1.0f, 0x3C00);
        assertHalf(-1.0f, 0xBC00);
        assertHalf(2.0f, 0x4000);
        assertHalf(-2.0f, 0xC000);
        assertHalf(0.5f, 0x3800);
        // Largest finite half value.
        assertHalf(65504.0f, 0x7BFF);
    }

    @Test
    public void smallestNormal() {
        // 2^-14 — the smallest positive *normal* half.
        assertHalf(0x1p-14f, 0x0400);
    }

    @Test
    public void smallestSubnormal() {
        // 0x0001 = 2^-24 ~= 5.9604645e-8, the smallest positive subnormal half.
        float expected = 0x1p-24f;
        float actual = h2f(0x0001);
        assertEquals(expected, actual, expected * 1e-6f);
    }

    @Test
    public void largestSubnormal() {
        // 0x03FF = 1023 * 2^-24 ~= 6.0975552e-5 (just below the smallest normal).
        assertHalf(1023.0f * 0x1p-24f, 0x03FF);
    }

    @Test
    public void infinitiesAndNaN() {
        assertTrue(Float.isInfinite(h2f(0x7C00)) && h2f(0x7C00) > 0);   // +Inf
        assertTrue(Float.isInfinite(h2f(0xFC00)) && h2f(0xFC00) < 0);   // -Inf
        assertTrue(Float.isNaN(h2f(0x7E00)));                            // NaN
    }
}
