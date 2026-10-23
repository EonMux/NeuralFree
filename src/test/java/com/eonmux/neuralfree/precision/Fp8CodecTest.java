package com.eonmux.neuralfree.precision;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link Fp8Codec}: the OCP E4M3FN and E5M2 formats as {@code torch.float8_e4m3fn} and {@code torch.float8_e5m2}
 * define them, with round-to-nearest-even and saturating casts.
 */
public class Fp8CodecTest {
    private static final Fp8Codec[] CODECS = {Fp8Codec.E4M3FN, Fp8Codec.E5M2};

    /** Every finite and infinite code decodes and encodes back to itself; NaN codes give the canonical NaN of their sign. */
    @Test
    public void everyCodeRoundTrips() {
        for (Fp8Codec codec : CODECS) {
            for (int code = 0; code < 256; code++) {
                float value = codec.decode(code);
                String label = codec + " code 0x" + Integer.toHexString(code);
                if (codec.isNaN(code)) {
                    assertTrue(label + " decodes to NaN", Float.isNaN(value));
                    assertEquals(label, (code & 0x80) | codec.nanCode(), codec.encode(value));
                } else {
                    assertFalse(label + " is not NaN", Float.isNaN(value));
                    assertEquals(label, code, codec.encode(value));
                }
            }
        }
    }

    /** E4M3FN has exactly one NaN per sign, so every one of its codes, NaN included, round-trips exactly. */
    @Test
    public void e4m3NaNCodesRoundTripExactly() {
        assertTrue(Fp8Codec.E4M3FN.isNaN(0x7F));
        assertTrue(Fp8Codec.E4M3FN.isNaN(0xFF));
        assertEquals(0x7F, Fp8Codec.E4M3FN.encode(Fp8Codec.E4M3FN.decode(0x7F)));
        assertEquals(0xFF, Fp8Codec.E4M3FN.encode(Fp8Codec.E4M3FN.decode(0xFF)));
        int nanCodes = 0;
        for (int code = 0; code < 256; code++) {
            if (Fp8Codec.E4M3FN.isNaN(code)) {
                nanCodes++;
            }
        }
        assertEquals(2, nanCodes);
    }

    @Test
    public void e4m3KnownValues() {
        Fp8Codec c = Fp8Codec.E4M3FN;
        assertEquals(448f, c.maxFinite(), 0f);
        assertEquals(448f, c.decode(0x7E), 0f);
        assertEquals(-448f, c.decode(0xFE), 0f);
        assertEquals(1f, c.decode(0x38), 0f);
        assertEquals(Math.scalb(1f, -9), c.decode(0x01), 0f);
        assertEquals(Math.scalb(7f, -9), c.decode(0x07), 0f);
        assertEquals(Math.scalb(1f, -6), c.decode(0x08), 0f);
        assertEquals(416f, c.decode(0x7D), 0f);
        assertEquals(0x00, c.encode(0f));
        assertEquals(0x80, c.encode(-0f));
        assertEquals(Float.floatToRawIntBits(-0f), Float.floatToRawIntBits(c.decode(0x80)));
    }

    @Test
    public void e5m2KnownValues() {
        Fp8Codec c = Fp8Codec.E5M2;
        assertEquals(57344f, c.maxFinite(), 0f);
        assertEquals(57344f, c.decode(0x7B), 0f);
        assertEquals(1f, c.decode(0x3C), 0f);
        assertEquals(Math.scalb(1f, -16), c.decode(0x01), 0f);
        assertEquals(Math.scalb(1f, -14), c.decode(0x04), 0f);
        assertEquals(Float.POSITIVE_INFINITY, c.decode(0x7C), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.decode(0xFC), 0f);
        assertTrue(c.isNaN(0x7D));
        assertTrue(c.isNaN(0x7E));
        assertTrue(c.isNaN(0x7F));
        assertFalse(c.isNaN(0x7C));
    }

    /**
     * Over every pair of adjacent non-negative finite values: the midpoint rounds to the code with the even
     * mantissa, and the floats just below and above it round down and up.
     */
    @Test
    public void midpointsRoundToEven() {
        for (Fp8Codec codec : CODECS) {
            List<Integer> codes = new ArrayList<>();
            for (int code = 0; code < 0x80; code++) {
                float v = codec.decode(code);
                if (!Float.isNaN(v) && !Float.isInfinite(v)) {
                    codes.add(code);
                }
            }
            for (int i = 0; i + 1 < codes.size(); i++) {
                int low = codes.get(i);
                int high = codes.get(i + 1);
                assertEquals("codes are ordered by value", low + 1, high);
                float mid = (codec.decode(low) + codec.decode(high)) / 2;
                String label = codec + " between 0x" + Integer.toHexString(low) + " and 0x" + Integer.toHexString(high);
                assertEquals(label + " tie", (low & 1) == 0 ? low : high, codec.encode(mid));
                assertEquals(label + " below tie", low, codec.encode(Math.nextDown(mid)));
                assertEquals(label + " above tie", high, codec.encode(Math.nextUp(mid)));
                assertEquals(label + " negative tie", 0x80 | ((low & 1) == 0 ? low : high), codec.encode(-mid));
            }
        }
    }

    @Test
    public void namedTies() {
        Fp8Codec e4m3 = Fp8Codec.E4M3FN;
        assertEquals("1.0625 between 1 and 1.125", 0x38, e4m3.encode(1.0625f));
        assertEquals("1.1875 between 1.125 and 1.25", 0x3A, e4m3.encode(1.1875f));
        assertEquals("1.9375 rounds up into the next binade", 0x40, e4m3.encode(1.9375f));
        assertEquals("432 between 416 and 448", 0x7E, e4m3.encode(432f));
        assertEquals("431 below the tie", 0x7D, e4m3.encode(431f));
        Fp8Codec e5m2 = Fp8Codec.E5M2;
        assertEquals("1.125 between 1 and 1.25", 0x3C, e5m2.encode(1.125f));
        assertEquals("1.375 between 1.25 and 1.5", 0x3E, e5m2.encode(1.375f));
    }

    @Test
    public void subnormalsRoundToEven() {
        Fp8Codec e4m3 = Fp8Codec.E4M3FN;
        assertEquals("half the smallest subnormal ties to 0", 0x00, e4m3.encode(Math.scalb(1f, -10)));
        assertEquals("just above half rounds up", 0x01, e4m3.encode(Math.nextUp(Math.scalb(1f, -10))));
        assertEquals("1.5 units tie to 2", 0x02, e4m3.encode(Math.scalb(1.5f, -9)));
        assertEquals("7.5 units tie to the smallest normal", 0x08, e4m3.encode(Math.scalb(7.5f, -9)));
        assertEquals("a negative value too small for a subnormal keeps its sign", 0x80,
            e4m3.encode(-Math.scalb(1f, -12)));
        assertEquals("float subnormals encode to 0", 0x00, e4m3.encode(Float.MIN_VALUE));
        Fp8Codec e5m2 = Fp8Codec.E5M2;
        assertEquals(0x00, e5m2.encode(Math.scalb(1f, -17)));
        assertEquals(0x02, e5m2.encode(Math.scalb(3f, -17)));
        assertEquals(0x03, e5m2.encode(Math.scalb(3f, -16)));
    }

    /** Finite values beyond the largest finite value saturate to it; infinities and NaN follow each format. */
    @Test
    public void saturationAndSpecials() {
        Fp8Codec e4m3 = Fp8Codec.E4M3FN;
        assertEquals(0x7E, e4m3.encode(447f));
        assertEquals(0x7E, e4m3.encode(464f));
        assertEquals(0x7E, e4m3.encode(1e30f));
        assertEquals(0xFE, e4m3.encode(-1000f));
        assertEquals(0x7E, e4m3.encode(Float.MAX_VALUE));
        assertEquals("no infinity in E4M3FN", 0x7F, e4m3.encode(Float.POSITIVE_INFINITY));
        assertEquals(0xFF, e4m3.encode(Float.NEGATIVE_INFINITY));
        assertEquals(0x7F, e4m3.encode(Float.NaN));

        Fp8Codec e5m2 = Fp8Codec.E5M2;
        assertEquals(0x7B, e5m2.encode(61440f));
        assertEquals(0x7B, e5m2.encode(Float.MAX_VALUE));
        assertEquals(0xFB, e5m2.encode(-1e9f));
        assertEquals(0x7C, e5m2.encode(Float.POSITIVE_INFINITY));
        assertEquals(0xFC, e5m2.encode(Float.NEGATIVE_INFINITY));
        assertEquals(0x7E, e5m2.encode(Float.NaN));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsCodesOutsideAByte() {
        Fp8Codec.E4M3FN.decode(256);
    }
}
