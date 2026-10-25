package com.eonmux.neuralfree.gguf.quant;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Tests for the OCP microscaling element codecs of {@link FpBlockDecoders} (E8M0, E4M3, E5M2, E2M1, E2M3, E3M2) and
 * the bitsandbytes codebooks of {@link QuantTables}, against values computed by hand from the bit layouts.
 */
public class FpCodecsTest {

    private static void assertBits(String label, float expected, float actual) {
        assertEquals(label, Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
    }

    // ------------------------------------------------------------------ E8M0

    @Test
    public void e8m0DecodesPowersOfTwo() {
        assertBits("e = 0 is the subnormal 2^-127", Math.scalb(1f, -127), FpBlockDecoders.e8m0ToFp32(0));
        assertBits("e = 1", Math.scalb(1f, -126), FpBlockDecoders.e8m0ToFp32(1));
        assertBits("e = 127", 1f, FpBlockDecoders.e8m0ToFp32(127));
        assertBits("e = 130", 8f, FpBlockDecoders.e8m0ToFp32(130));
        assertBits("e = 254", Math.scalb(1f, 127), FpBlockDecoders.e8m0ToFp32(254));
        assertTrue("e = 255 is NaN", Float.isNaN(FpBlockDecoders.e8m0ToFp32(255)));
        assertTrue("a signed byte is masked", Float.isNaN(FpBlockDecoders.e8m0ToFp32((byte) 0xFF)));
        assertBits("signed byte 0xFE", Math.scalb(1f, 127), FpBlockDecoders.e8m0ToFp32((byte) 0xFE));
    }

    @Test
    public void e8m0IsTwiceTheGgmlHalfConvention() {
        for (int e = 0; e < 255; e++) {
            assertBits("e = " + e, 2f * FpBlockDecoders.e8m0ToFp32Half(e), FpBlockDecoders.e8m0ToFp32(e));
        }
    }

    // ------------------------------------------------------------------ E4M3 / E5M2

    @Test
    public void e4m3CoversEveryClass() {
        assertBits("zero", 0f, FpBlockDecoders.e4m3ToFp32(0x00));
        assertBits("negative zero", -0f, FpBlockDecoders.e4m3ToFp32(0x80));
        assertBits("smallest subnormal", Math.scalb(1f, -9), FpBlockDecoders.e4m3ToFp32(0x01));
        assertBits("largest subnormal", Math.scalb(7f, -9), FpBlockDecoders.e4m3ToFp32(0x07));
        assertBits("smallest normal", Math.scalb(1f, -6), FpBlockDecoders.e4m3ToFp32(0x08));
        assertBits("one", 1f, FpBlockDecoders.e4m3ToFp32(0x38));
        assertBits("max", 448f, FpBlockDecoders.e4m3ToFp32(0x7E));
        assertBits("negative max", -448f, FpBlockDecoders.e4m3ToFp32(0xFE));
        assertBits("signed byte", -448f, FpBlockDecoders.e4m3ToFp32((byte) 0xFE));
        assertTrue("0x7F is NaN", Float.isNaN(FpBlockDecoders.e4m3ToFp32(0x7F)));
        assertTrue("0xFF is NaN", Float.isNaN(FpBlockDecoders.e4m3ToFp32(0xFF)));
    }

    @Test
    public void e4m3EncoderRoundsToEvenAndSaturates() {
        for (int code = 0; code < 256; code++) {
            float v = FpBlockDecoders.e4m3ToFp32(code);
            if (!Float.isNaN(v)) {
                assertEquals("code 0x" + Integer.toHexString(code), code, FpBlockDecoders.fp32ToE4m3(v));
            }
        }
        assertEquals("1.0625 ties to 1.0", 0x38, FpBlockDecoders.fp32ToE4m3(1.0625f));
        assertEquals("1.1875 ties to 1.25", 0x3A, FpBlockDecoders.fp32ToE4m3(1.1875f));
        assertEquals("finite overflow saturates", 0x7E, FpBlockDecoders.fp32ToE4m3(1000f));
        assertEquals("negative overflow saturates", 0xFE, FpBlockDecoders.fp32ToE4m3(-1000f));
        assertEquals("NaN", 0x7F, FpBlockDecoders.fp32ToE4m3(Float.NaN));
    }

    @Test
    public void e5m2DecodesTheHighByteOfAHalf() {
        assertBits("one", 1f, FpBlockDecoders.e5m2ToFp32(0x3C));
        assertBits("max", 57344f, FpBlockDecoders.e5m2ToFp32(0x7B));
        assertBits("smallest subnormal", Math.scalb(1f, -16), FpBlockDecoders.e5m2ToFp32(0x01));
        assertBits("infinity", Float.POSITIVE_INFINITY, FpBlockDecoders.e5m2ToFp32(0x7C));
        assertBits("negative infinity", Float.NEGATIVE_INFINITY, FpBlockDecoders.e5m2ToFp32(0xFC));
        assertTrue(Float.isNaN(FpBlockDecoders.e5m2ToFp32(0x7D)));
        for (int code = 0; code < 256; code++) {
            float v = FpBlockDecoders.e5m2ToFp32(code);
            if (!Float.isNaN(v)) {
                assertBits("code 0x" + Integer.toHexString(code), FpBlockDecoders.fp16ToFp32(code << 8), v);
            }
        }
    }

    // ------------------------------------------------------------------ E2M1

    @Test
    public void e2m1DecodesHalfTheGgmlCodebook() {
        float[] expected = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};
        for (int c = 0; c < 8; c++) {
            assertBits("code " + c, expected[c], FpBlockDecoders.e2m1ToFp32(c));
            if (c > 0) {
                assertBits("code " + (c | 8), -expected[c], FpBlockDecoders.e2m1ToFp32(c | 8));
            }
        }
        assertBits("negative zero code decodes to +0 as in ggml", 0f, FpBlockDecoders.e2m1ToFp32(8));
    }

    @Test
    public void e2m1EncoderRoundsTiesToEven() {
        for (int c = 0; c < 16; c++) {
            if (c != 8) {
                assertEquals("code " + c, c, FpBlockDecoders.fp32ToE2m1(FpBlockDecoders.e2m1ToFp32(c)));
            }
        }
        float[] ties = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5f};
        int[] tieCodes = {0, 2, 2, 4, 4, 6, 6};
        int[] below = {0, 1, 2, 3, 4, 5, 6};
        int[] above = {1, 2, 3, 4, 5, 6, 7};
        for (int i = 0; i < ties.length; i++) {
            float t = ties[i];
            assertEquals("tie " + t, tieCodes[i], FpBlockDecoders.fp32ToE2m1(t));
            assertEquals("negative tie " + t, 8 | tieCodes[i], FpBlockDecoders.fp32ToE2m1(-t));
            assertEquals("below " + t, below[i], FpBlockDecoders.fp32ToE2m1(Math.nextDown(t)));
            assertEquals("above " + t, above[i], FpBlockDecoders.fp32ToE2m1(Math.nextUp(t)));
        }
    }

    @Test
    public void e2m1EncoderSaturatesAndKeepsTheSign() {
        assertEquals(7, FpBlockDecoders.fp32ToE2m1(6.5f));
        assertEquals(7, FpBlockDecoders.fp32ToE2m1(1e30f));
        assertEquals(15, FpBlockDecoders.fp32ToE2m1(-100f));
        assertEquals(7, FpBlockDecoders.fp32ToE2m1(Float.POSITIVE_INFINITY));
        assertEquals(15, FpBlockDecoders.fp32ToE2m1(Float.NEGATIVE_INFINITY));
        assertEquals(7, FpBlockDecoders.fp32ToE2m1(Float.NaN));
        assertEquals(0, FpBlockDecoders.fp32ToE2m1(0f));
        assertEquals(8, FpBlockDecoders.fp32ToE2m1(-0f));
        assertEquals(8, FpBlockDecoders.fp32ToE2m1(-0.1f));
    }

    // ------------------------------------------------------------------ FP6

    @Test
    public void e2m3HasNoSpecialsAndKnownExtremes() {
        assertBits("zero", 0f, FpBlockDecoders.e2m3ToFp32(0x00));
        assertBits("negative zero", -0f, FpBlockDecoders.e2m3ToFp32(0x20));
        assertBits("smallest subnormal", 0.125f, FpBlockDecoders.e2m3ToFp32(0x01));
        assertBits("largest subnormal", 0.875f, FpBlockDecoders.e2m3ToFp32(0x07));
        assertBits("smallest normal", 1f, FpBlockDecoders.e2m3ToFp32(0x08));
        assertBits("max", 7.5f, FpBlockDecoders.e2m3ToFp32(0x1F));
        assertBits("negative max", -7.5f, FpBlockDecoders.e2m3ToFp32(0x3F));
        assertMonotoneAndFinite(true);
    }

    @Test
    public void e3m2HasNoSpecialsAndKnownExtremes() {
        assertBits("zero", 0f, FpBlockDecoders.e3m2ToFp32(0x00));
        assertBits("negative zero", -0f, FpBlockDecoders.e3m2ToFp32(0x20));
        assertBits("smallest subnormal", 0.0625f, FpBlockDecoders.e3m2ToFp32(0x01));
        assertBits("largest subnormal", 0.1875f, FpBlockDecoders.e3m2ToFp32(0x03));
        assertBits("smallest normal", 0.25f, FpBlockDecoders.e3m2ToFp32(0x04));
        assertBits("one", 1f, FpBlockDecoders.e3m2ToFp32(0x0C));
        assertBits("max", 28f, FpBlockDecoders.e3m2ToFp32(0x1F));
        assertBits("negative max", -28f, FpBlockDecoders.e3m2ToFp32(0x3F));
        assertMonotoneAndFinite(false);
    }

    private static void assertMonotoneAndFinite(boolean e2m3) {
        for (int c = 0; c < 64; c++) {
            float v = e2m3 ? FpBlockDecoders.e2m3ToFp32(c) : FpBlockDecoders.e3m2ToFp32(c);
            assertTrue("code " + c + " is finite", Float.isFinite(v));
            if ((c & 0x1F) > 0) {
                float prev = e2m3 ? FpBlockDecoders.e2m3ToFp32(c - 1) : FpBlockDecoders.e3m2ToFp32(c - 1);
                assertTrue("code " + c + " grows in magnitude", Math.abs(v) > Math.abs(prev));
            }
            float mirrored = e2m3 ? FpBlockDecoders.e2m3ToFp32(c ^ 0x20) : FpBlockDecoders.e3m2ToFp32(c ^ 0x20);
            assertBits("code " + c + " sign", -v, mirrored);
        }
    }

    // ------------------------------------------------------------------ bitsandbytes codebooks

    @Test
    public void fp4BnbCodebookIsTheRawListOverTwelve() {
        float[] raw = {0f, 0.0625f, 8f, 12f, 4f, 6f, 2f, 3f, -0f, -0.0625f, -8f, -12f, -4f, -6f, -2f, -3f};
        for (int i = 0; i < 16; i++) {
            assertBits("index " + i, raw[i] / 12f, QuantTables.fp4BnbCode(i));
        }
        assertEquals(8f / 12f, QuantTables.fp4BnbCode(2), 0f);
        assertEquals(1f, QuantTables.fp4BnbCode(3), 0f);
        assertEquals(-1f, QuantTables.fp4BnbCode(11), 0f);
        assertBits("index 8 keeps the negative zero", -0f, QuantTables.fp4BnbCode(8));
    }

    @Test
    public void nf4CodebookIsNormalizedAndIncreasing() {
        assertEquals(-1f, QuantTables.nf4Code(0), 0f);
        assertEquals(0f, QuantTables.nf4Code(7), 0f);
        assertEquals(1f, QuantTables.nf4Code(15), 0f);
        assertEquals(0.07958029955625534f, QuantTables.nf4Code(8), 0f);
        float maxAbs = 0f;
        for (int i = 0; i < 16; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(QuantTables.nf4Code(i)));
            if (i > 0) {
                assertTrue("index " + i, QuantTables.nf4Code(i) > QuantTables.nf4Code(i - 1));
            }
        }
        assertEquals(1f, maxAbs, 0f);
        assertFalse(Float.isNaN(QuantTables.nf4Code(3)));
    }
}
