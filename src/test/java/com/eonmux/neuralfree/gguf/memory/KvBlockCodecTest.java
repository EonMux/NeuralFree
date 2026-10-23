package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * KV block codec: dequantization agrees with the independent weight decoders ({@code dequantize_row_*}), the
 * reference quantizers reproduce exactly representable blocks bit for bit (scale sign and qh layout included),
 * random blocks stay within the type's step, and the BF16 / IQ4_NL edge cases follow ggml.
 */
public class KvBlockCodecTest {

    private static final KvDType[] BLOCK_TYPES = {
            KvDType.Q8_0, KvDType.Q4_0, KvDType.Q4_1, KvDType.IQ4_NL, KvDType.Q5_0, KvDType.Q5_1};

    private static float[] gaussian(int n, long seed, double scale) {
        Random r = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) (r.nextGaussian() * scale);
        }
        return x;
    }

    private static float[] roundTrip(KvDType t, float[] x) {
        byte[] b = new byte[t.blockBytes()];
        KvBlockCodec.quantize(t, x, 0, b, 0);
        float[] back = new float[32];
        KvBlockCodec.dequantize(t, b, 0, 0, 32, back, 0);
        return back;
    }

    private static float half(byte[] b, int off) {
        return FpBlockDecoders.fp16ToFp32((b[off] & 0xFF) | (b[off + 1] & 0xFF) << 8);
    }

    private static int u32(byte[] b, int off) {
        return (b[off] & 0xFF) | (b[off + 1] & 0xFF) << 8 | (b[off + 2] & 0xFF) << 16 | (b[off + 3] & 0xFF) << 24;
    }

    @Test
    public void dequantizeMatchesTheWeightDecoders() {
        for (KvDType t : BLOCK_TYPES) {
            for (long seed = 1; seed <= 4; seed++) {
                float[] x = gaussian(32, seed * 31 + t.code(), 2.0);
                byte[] b = new byte[t.blockBytes()];
                KvBlockCodec.quantize(t, x, 0, b, 0);
                float[] mine = new float[32];
                KvBlockCodec.dequantize(t, b, 0, 0, 32, mine, 0);
                float[] ref = new float[32];
                ByteBuffer buf = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
                CpuBlockDecoders.decode(t.ggmlType(), buf, 0, ref, 0, 1);
                assertArrayEquals(t + " seed " + seed, ref, mine, 0f);

                float[] part = new float[7];
                KvBlockCodec.dequantize(t, b, 0, 13, 7, part, 0);
                for (int i = 0; i < 7; i++) {
                    assertEquals(mine[13 + i], part[i], 0f);
                }
                ByteBuffer direct = ByteBuffer.allocate(5 + b.length).order(ByteOrder.LITTLE_ENDIAN);
                KvBlockCodec.quantize(t, x, 0, direct, 5);
                float[] viaBuffer = new float[32];
                KvBlockCodec.dequantize(t, direct, 5, viaBuffer, 0);
                assertArrayEquals(mine, viaBuffer, 0f);
            }
        }
    }

    /** Values (j % 16) - 8: the first -8 has the largest |x|, so d = -8 / -8 = 1 and every code is exact. */
    @Test
    public void q4_0ReproducesItsGrid() {
        float[] x = new float[32];
        for (int j = 0; j < 32; j++) {
            x[j] = (j % 16) - 8;
        }
        assertArrayEquals(x, roundTrip(KvDType.Q4_0, x), 0f);
        byte[] b = new byte[18];
        KvBlockCodec.quantize(KvDType.Q4_0, x, 0, b, 0);
        assertEquals(1f, half(b, 0), 0f);
    }

    /** A positive extreme gives a negative scale (d = max / -8), as quantize_row_q4_0_ref. */
    @Test
    public void q4_0KeepsTheSignOfTheExtreme() {
        float[] x = new float[32];
        x[0] = 8f;
        x[1] = -4f;
        byte[] b = new byte[18];
        KvBlockCodec.quantize(KvDType.Q4_0, x, 0, b, 0);
        assertEquals(-1f, half(b, 0), 0f);
        float[] back = roundTrip(KvDType.Q4_0, x);
        assertEquals(8f, back[0], 0f);
        assertEquals(-4f, back[1], 0f);
        assertEquals(0f, back[2], 0f);
    }

    /** Values j - 16: d = 1, codes j, so the fifth bit is set exactly for j &gt;= 16. */
    @Test
    public void q5_0ReproducesItsGridAndHighBits() {
        float[] x = new float[32];
        for (int j = 0; j < 32; j++) {
            x[j] = j - 16;
        }
        assertArrayEquals(x, roundTrip(KvDType.Q5_0, x), 0f);
        byte[] b = new byte[22];
        KvBlockCodec.quantize(KvDType.Q5_0, x, 0, b, 0);
        assertEquals(1f, half(b, 0), 0f);
        assertEquals(0xFFFF0000, u32(b, 2));
    }

    /** Values 0.5 (j % 16) + 1: d = 7.5 / 15 = 0.5, m = 1. */
    @Test
    public void q4_1ReproducesItsGrid() {
        float[] x = new float[32];
        for (int j = 0; j < 32; j++) {
            x[j] = 0.5f * (j % 16) + 1f;
        }
        assertArrayEquals(x, roundTrip(KvDType.Q4_1, x), 0f);
        byte[] b = new byte[20];
        KvBlockCodec.quantize(KvDType.Q4_1, x, 0, b, 0);
        assertEquals(0.5f, half(b, 0), 0f);
        assertEquals(1f, half(b, 2), 0f);
    }

    /** Values 0.25 j - 2: d = 7.75 / 31 = 0.25, m = -2, codes j. */
    @Test
    public void q5_1ReproducesItsGridAndHighBits() {
        float[] x = new float[32];
        for (int j = 0; j < 32; j++) {
            x[j] = 0.25f * j - 2f;
        }
        assertArrayEquals(x, roundTrip(KvDType.Q5_1, x), 0f);
        byte[] b = new byte[24];
        KvBlockCodec.quantize(KvDType.Q5_1, x, 0, b, 0);
        assertEquals(0.25f, half(b, 0), 0f);
        assertEquals(-2f, half(b, 2), 0f);
        assertEquals(0xFFFF0000, u32(b, 4));
    }

    /** Values 0.5 kvalues[j % 16]: indices are exact and the weighted least-squares scale is exactly 0.5. */
    @Test
    public void iq4_nlReproducesItsCodebook() {
        float[] x = new float[32];
        for (int j = 0; j < 32; j++) {
            x[j] = 0.5f * QuantTables.kvalueIq4nl(j % 16);
        }
        assertArrayEquals(x, roundTrip(KvDType.IQ4_NL, x), 0f);
        byte[] b = new byte[18];
        KvBlockCodec.quantize(KvDType.IQ4_NL, x, 0, b, 0);
        assertEquals(0.5f, half(b, 0), 0f);
        for (int j = 0; j < 16; j++) {
            assertEquals((byte) (j | (j << 4)), b[2 + j]);
        }
    }

    /** A block whose max |x| is below GROUP_MAX_EPS (1e-15) stores d = 0 and zero indices. */
    @Test
    public void iq4_nlNearZeroBlockIsAllZero() {
        float[] x = new float[32];
        x[3] = 5e-16f;
        x[9] = -7e-16f;
        byte[] b = new byte[18];
        KvBlockCodec.quantize(KvDType.IQ4_NL, x, 0, b, 0);
        assertArrayEquals(new byte[18], b);
    }

    @Test
    public void zeroBlocksDecodeToZero() {
        float[] zero = new float[32];
        for (KvDType t : BLOCK_TYPES) {
            float[] back = roundTrip(t, zero);
            for (float v : back) {
                assertEquals(t.toString(), 0f, v, 0f);
            }
        }
    }

    /**
     * Every value lands within half a step of its input (plus the f16 rounding of the stored scale and min); Q4_0 and
     * Q5_0 saturate the extreme opposite the scale's sign, so a full step there; IQ4_NL within a quarter of max |x|.
     */
    @Test
    public void randomBlocksStayWithinTheStep() {
        for (KvDType t : BLOCK_TYPES) {
            for (long seed = 10; seed < 20; seed++) {
                float[] x = gaussian(32, seed, 3.0);
                float amax = 0f;
                float min = Float.MAX_VALUE;
                float max = -Float.MAX_VALUE;
                for (float v : x) {
                    amax = Math.max(amax, Math.abs(v));
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
                float step;
                switch (t) {
                    case Q8_0: step = amax / 127; break;
                    case Q4_0: step = amax / 8; break;
                    case Q5_0: step = amax / 16; break;
                    case Q4_1: step = (max - min) / 15; break;
                    case Q5_1: step = (max - min) / 31; break;
                    default: step = amax / 64; break;
                }
                float[] back = roundTrip(t, x);
                double bound;
                if (t == KvDType.IQ4_NL) {
                    bound = 0.25 * amax;
                } else if (t == KvDType.Q4_0 || t == KvDType.Q5_0) {
                    bound = step + 2e-3 * amax;
                } else {
                    bound = 0.5 * step + 2e-3 * amax;
                }
                for (int j = 0; j < 32; j++) {
                    assertTrue(t + " seed " + seed + " value " + j + ": |" + x[j] + " - " + back[j] + "| > " + bound,
                            Math.abs(x[j] - back[j]) <= bound);
                }
            }
        }
    }

    @Test
    public void bf16RoundsToNearestEvenAndKeepsNaNQuiet() {
        assertEquals(0x3F80, KvBlockCodec.fp32ToBf16(1f));
        assertEquals(0x3F80, KvBlockCodec.fp32ToBf16(Float.intBitsToFloat(0x3F808000)));
        assertEquals(0x3F82, KvBlockCodec.fp32ToBf16(Float.intBitsToFloat(0x3F818000)));
        assertEquals(0x3F81, KvBlockCodec.fp32ToBf16(Float.intBitsToFloat(0x3F808001)));
        assertEquals(0xBF80, KvBlockCodec.fp32ToBf16(-1f));
        assertEquals(0x7F80, KvBlockCodec.fp32ToBf16(Float.POSITIVE_INFINITY));
        assertEquals(0x7FC0, KvBlockCodec.fp32ToBf16(Float.intBitsToFloat(0x7F800001)));
        assertEquals(1.0078125f, FpBlockDecoders.bf16ToFp32(KvBlockCodec.fp32ToBf16(1.0078125f)), 0f);
    }

    @Test
    public void rejectsNonBlockTypesAndRangesOutsideTheBlock() {
        assertThrows(IllegalArgumentException.class,
                () -> KvBlockCodec.quantize(KvDType.BF16, new float[32], 0, new byte[34], 0));
        assertThrows(IllegalArgumentException.class,
                () -> KvBlockCodec.dequantize(KvDType.F16, new byte[34], 0, 0, 32, new float[32], 0));
        assertThrows(IllegalArgumentException.class,
                () -> KvBlockCodec.dequantize(KvDType.Q4_0, new byte[18], 0, 30, 4, new float[4], 0));
    }
}
