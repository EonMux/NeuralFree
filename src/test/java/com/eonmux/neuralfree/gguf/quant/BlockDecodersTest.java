package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the CPU block decoders: block geometry, the half conversions, and every decoder family on
 * hand-built blocks whose values follow directly from the ggml block definitions.
 */
public class BlockDecodersTest {

    private static final int H_ONE = 0x3C00;   // half 1.0
    private static final int H_HALF = 0x3800;  // half 0.5
    private static final int H_TWO = 0x4000;   // half 2.0

    private static ByteBuffer block(int bytes) {
        return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static float[] decode(GGMLType type, ByteBuffer src, int nBlocks) {
        float[] out = new float[nBlocks * type.getBlockSize()];
        CpuBlockDecoders.decode(type, src, 0, out, 0, nBlocks);
        return out;
    }

    /** Independent binary16 reference: sign, exponent and mantissa evaluated directly. */
    private static float refHalf(int h) {
        int s = (h >>> 15) & 1;
        int e = (h >>> 10) & 0x1F;
        int m = h & 0x3FF;
        float v;
        if (e == 0) {
            v = m * 0x1p-24f;
        } else if (e == 31) {
            v = m == 0 ? Float.POSITIVE_INFINITY : Float.NaN;
        } else {
            v = Math.scalb(1.0f + m / 1024.0f, e - 15);
        }
        return s == 1 ? -v : v;
    }

    // ------------------------------------------------------------------ geometry and dispatch

    @Test
    public void blockSizesMatchGgmlStructs() {
        int[][] expected = {
            {GGMLType.Q1_0.ordinal(), BlockLayout.Q1_0_BYTES, BlockLayout.QK1_0},
            {GGMLType.Q2_0.ordinal(), BlockLayout.Q2_0_BYTES, BlockLayout.QK2_0},
            {GGMLType.Q4_0.ordinal(), BlockLayout.Q4_0_BYTES, BlockLayout.QK4_0},
            {GGMLType.Q4_1.ordinal(), BlockLayout.Q4_1_BYTES, BlockLayout.QK4_1},
            {GGMLType.Q5_0.ordinal(), BlockLayout.Q5_0_BYTES, BlockLayout.QK5_0},
            {GGMLType.Q5_1.ordinal(), BlockLayout.Q5_1_BYTES, BlockLayout.QK5_1},
            {GGMLType.Q8_0.ordinal(), BlockLayout.Q8_0_BYTES, BlockLayout.QK8_0},
            {GGMLType.Q8_1.ordinal(), BlockLayout.Q8_1_BYTES, BlockLayout.QK8_1},
            {GGMLType.MXFP4.ordinal(), BlockLayout.MXFP4_BYTES, BlockLayout.QK_MXFP4},
            {GGMLType.NVFP4.ordinal(), BlockLayout.NVFP4_BYTES, BlockLayout.QK_NVFP4},
            {GGMLType.TQ1_0.ordinal(), BlockLayout.TQ1_0_BYTES, BlockLayout.QK_K},
            {GGMLType.TQ2_0.ordinal(), BlockLayout.TQ2_0_BYTES, BlockLayout.QK_K},
            {GGMLType.Q2_K.ordinal(), BlockLayout.Q2_K_BYTES, BlockLayout.QK_K},
            {GGMLType.Q3_K.ordinal(), BlockLayout.Q3_K_BYTES, BlockLayout.QK_K},
            {GGMLType.Q4_K.ordinal(), BlockLayout.Q4_K_BYTES, BlockLayout.QK_K},
            {GGMLType.Q5_K.ordinal(), BlockLayout.Q5_K_BYTES, BlockLayout.QK_K},
            {GGMLType.Q6_K.ordinal(), BlockLayout.Q6_K_BYTES, BlockLayout.QK_K},
            {GGMLType.Q8_K.ordinal(), BlockLayout.Q8_K_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ2_XXS.ordinal(), BlockLayout.IQ2_XXS_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ2_XS.ordinal(), BlockLayout.IQ2_XS_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ2_S.ordinal(), BlockLayout.IQ2_S_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ3_XXS.ordinal(), BlockLayout.IQ3_XXS_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ3_S.ordinal(), BlockLayout.IQ3_S_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ1_S.ordinal(), BlockLayout.IQ1_S_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ1_M.ordinal(), BlockLayout.IQ1_M_BYTES, BlockLayout.QK_K},
            {GGMLType.IQ4_NL.ordinal(), BlockLayout.IQ4_NL_BYTES, BlockLayout.QK4_NL},
            {GGMLType.IQ4_XS.ordinal(), BlockLayout.IQ4_XS_BYTES, BlockLayout.QK_K},
        };
        for (int[] e : expected) {
            GGMLType t = GGMLType.values()[e[0]];
            BlockLayout layout = BlockLayout.of(t);
            assertEquals(t + " block bytes", e[1], layout.blockBytes());
            assertEquals(t + " block elements", e[2], layout.blockElements());
        }
    }

    @Test
    public void everyStorageTypeHasADecoder() {
        for (GGMLType t : GGMLType.values()) {
            boolean removed = t == GGMLType.UNKNOWN || t == GGMLType.Q4_2 || t == GGMLType.Q4_3;
            assertEquals(t.toString(), !removed, CpuBlockDecoders.supports(t));
            assertEquals(t.toString(), !removed, BlockLayout.isDefined(t));
        }
        assertEquals(35, CpuBlockDecoders.supportedTypes().size());
        try {
            CpuBlockDecoders.forType(GGMLType.UNKNOWN);
            fail("UNKNOWN must not have a decoder");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage().contains("UNKNOWN"));
        }
    }

    @Test
    public void rangeAndShapeErrorsAreReported() {
        ByteBuffer b = block(BlockLayout.Q8_0_BYTES);
        float[] out = new float[32];
        try {
            CpuBlockDecoders.decode(GGMLType.Q8_0, b, 1, out, 0, 1);
            fail("source overrun");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage().contains("limit"));
        }
        try {
            CpuBlockDecoders.decode(GGMLType.Q8_0, b, 0, new float[31], 0, 1);
            fail("destination overrun");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage().contains("31"));
        }
        try {
            CpuBlockDecoders.decodeElements(GGMLType.Q8_0, b, 0, out, 0, 16);
            fail("partial block");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("block size 32"));
        }
    }

    @Test
    public void byteOrderAndOffsetAreHonoured() {
        ByteBuffer le = block(3 + BlockLayout.Q8_0_BYTES);
        le.putShort(3, (short) H_TWO);
        for (int j = 0; j < 32; j++) {
            le.put(3 + 2 + j, (byte) (j - 16));
        }
        ByteBuffer be = ByteBuffer.wrap(le.array()).order(ByteOrder.BIG_ENDIAN);
        float[] a = new float[32];
        float[] c = new float[32];
        CpuBlockDecoders.decode(GGMLType.Q8_0, le, 3, a, 0, 1);
        CpuBlockDecoders.decode(GGMLType.Q8_0, be, 3, c, 0, 1);
        for (int j = 0; j < 32; j++) {
            assertEquals((j - 16) * 2.0f, a[j], 0f);
            assertEquals(a[j], c[j], 0f);
        }
        assertEquals(ByteOrder.BIG_ENDIAN, be.order());
    }

    // ------------------------------------------------------------------ scalar conversions

    @Test
    public void halfToFloatMatchesBinary16ForEveryPattern() {
        for (int h = 0; h < 0x10000; h++) {
            float ref = refHalf(h);
            float got = FpBlockDecoders.fp16ToFp32(h);
            if (Float.isNaN(ref)) {
                assertTrue(Float.isNaN(got));
            } else {
                assertEquals("half 0x" + Integer.toHexString(h), Float.floatToIntBits(ref), Float.floatToIntBits(got));
                assertEquals("round trip 0x" + Integer.toHexString(h), h, FpBlockDecoders.fp32ToFp16(got));
            }
        }
    }

    @Test
    public void floatToHalfRoundsToNearestEven() {
        assertEquals(0x3C00, FpBlockDecoders.fp32ToFp16(1.0f + 0x1p-11f));          // tie, even stays
        assertEquals(0x3C02, FpBlockDecoders.fp32ToFp16(1.0f + 3 * 0x1p-11f));      // tie, odd rounds up
        assertEquals(0x3C01, FpBlockDecoders.fp32ToFp16(1.0f + 0x1p-11f + 0x1p-20f));
        assertEquals(0x7C00, FpBlockDecoders.fp32ToFp16(65520.0f));                // overflow to +inf
        assertEquals(0x7BFF, FpBlockDecoders.fp32ToFp16(65519.0f));
        assertEquals(0x0001, FpBlockDecoders.fp32ToFp16(0x1p-24f));                // smallest subnormal
        assertEquals(0x8000, FpBlockDecoders.fp32ToFp16(-0.0f));
        assertEquals(0x7E00, FpBlockDecoders.fp32ToFp16(Float.NaN));
    }

    @Test
    public void fp4ScaleFormats() {
        assertEquals(1.0f, FpBlockDecoders.e8m0ToFp32Half(128), 0f);              // 0.5 * 2^(128-127)
        assertEquals(0x1p-128f, FpBlockDecoders.e8m0ToFp32Half(0), 0f);
        assertEquals(0x1p-127f, FpBlockDecoders.e8m0ToFp32Half(1), 0f);
        assertEquals(0.5f, FpBlockDecoders.ue4m3ToFp32(0x38), 0f);                // exp 7, man 0 -> 1.0, halved
        assertEquals(0.75f, FpBlockDecoders.ue4m3ToFp32(0x3C), 0f);               // 1.5 halved
        assertEquals(0x1p-10f, FpBlockDecoders.ue4m3ToFp32(0x01), 0f);            // subnormal 2^-9, halved
        assertEquals(0.0f, FpBlockDecoders.ue4m3ToFp32(0x7F), 0f);
        assertEquals(1.5f, FpBlockDecoders.bf16ToFp32(0x3FC0), 0f);
    }

    // ------------------------------------------------------------------ legacy family

    @Test
    public void q4_0UsesSplitNibbles() {
        ByteBuffer b = block(BlockLayout.Q4_0_BYTES);
        b.putShort(0, (short) H_HALF);
        for (int j = 0; j < 16; j++) {
            b.put(2 + j, (byte) ((j & 0xF) | ((15 - j) << 4)));
        }
        float[] y = decode(GGMLType.Q4_0, b, 1);
        for (int j = 0; j < 16; j++) {
            assertEquals((j - 8) * 0.5f, y[j], 0f);
            assertEquals((15 - j - 8) * 0.5f, y[j + 16], 0f);
        }
    }

    @Test
    public void q4_1AndQ5_1AddTheMin() {
        ByteBuffer b = block(BlockLayout.Q4_1_BYTES);
        b.putShort(0, (short) H_TWO).putShort(2, (short) H_HALF);
        for (int j = 0; j < 16; j++) {
            b.put(4 + j, (byte) (j | (j << 4)));
        }
        float[] y = decode(GGMLType.Q4_1, b, 1);
        for (int j = 0; j < 16; j++) {
            assertEquals(j * 2.0f + 0.5f, y[j], 0f);
            assertEquals(j * 2.0f + 0.5f, y[j + 16], 0f);
        }

        ByteBuffer c = block(BlockLayout.Q5_1_BYTES);
        c.putShort(0, (short) H_ONE).putShort(2, (short) H_TWO).putInt(4, 0xFFFF0000);
        float[] z = decode(GGMLType.Q5_1, c, 1);
        for (int j = 0; j < 16; j++) {
            assertEquals(2.0f, z[j], 0f);           // q = 0, fifth bit clear
            assertEquals(16.0f + 2.0f, z[j + 16], 0f); // q = 16, fifth bit set
        }
    }

    @Test
    public void q5_0FifthBitsFollowElementIndex() {
        ByteBuffer b = block(BlockLayout.Q5_0_BYTES);
        b.putShort(0, (short) H_ONE);
        b.putInt(2, 0x00010001);                 // bit 0 -> element 0, bit 16 -> element 16
        for (int j = 0; j < 16; j++) {
            b.put(6 + j, (byte) 0x31);           // low nibble 1, high nibble 3
        }
        float[] y = decode(GGMLType.Q5_0, b, 1);
        assertEquals(1f, y[0], 0f);             // (1 | 16) - 16
        assertEquals(-15f, y[1], 0f);           // 1 - 16
        assertEquals(3f, y[16], 0f);            // (3 | 16) - 16
        assertEquals(-13f, y[17], 0f);          // 3 - 16
    }

    @Test
    public void q8_0AndQ8_1ScaleSignedQuants() {
        ByteBuffer b = block(BlockLayout.Q8_1_BYTES);
        b.putShort(0, (short) H_HALF).putShort(2, (short) 0x7BFF);
        for (int j = 0; j < 32; j++) {
            b.put(4 + j, (byte) (j * 8 - 128));
        }
        float[] y = decode(GGMLType.Q8_1, b, 1);
        for (int j = 0; j < 32; j++) {
            assertEquals((j * 8 - 128) * 0.5f, y[j], 0f);
        }
    }

    // ------------------------------------------------------------------ FP4 and 1-/2-bit types

    @Test
    public void mxfp4AndNvfp4UseTheDoubledE2M1Codebook() {
        ByteBuffer m = block(BlockLayout.MXFP4_BYTES);
        m.put(0, (byte) 128);
        for (int j = 0; j < 16; j++) {
            m.put(1 + j, (byte) (j | ((15 - j) << 4)));
        }
        float[] y = decode(GGMLType.MXFP4, m, 1);
        int[] kv = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        for (int j = 0; j < 16; j++) {
            assertEquals(kv[j], y[j], 0f);
            assertEquals(kv[15 - j], y[j + 16], 0f);
        }

        ByteBuffer n = block(BlockLayout.NVFP4_BYTES);
        n.put(0, (byte) 0x38).put(1, (byte) 0x40).put(2, (byte) 0x00).put(3, (byte) 0x7F);  // 0.5, 1.0, 0, 0
        for (int j = 0; j < 32; j++) {
            n.put(4 + j, (byte) 0x72);           // low code 2 (value 2), high code 7 (value 12)
        }
        float[] z = decode(GGMLType.NVFP4, n, 1);
        for (int j = 0; j < 8; j++) {
            assertEquals(2 * 0.5f, z[j], 0f);
            assertEquals(12 * 0.5f, z[j + 8], 0f);
            assertEquals(2 * 1.0f, z[16 + j], 0f);
            assertEquals(12 * 1.0f, z[16 + j + 8], 0f);
            assertEquals(0f, z[32 + j], 0f);
            assertEquals(0f, z[48 + j + 8], 0f);
        }
    }

    @Test
    public void q1_0AndQ2_0() {
        ByteBuffer b = block(BlockLayout.Q1_0_BYTES);
        b.putShort(0, (short) H_TWO);
        b.put(2, (byte) 0x81);                   // elements 0 and 7 set
        float[] y = decode(GGMLType.Q1_0, b, 1);
        assertEquals(2f, y[0], 0f);
        assertEquals(-2f, y[1], 0f);
        assertEquals(2f, y[7], 0f);
        assertEquals(-2f, y[127], 0f);

        ByteBuffer c = block(BlockLayout.Q2_0_BYTES);
        c.putShort(0, (short) H_HALF);
        c.put(2, (byte) 0xE4);                   // codes 0,1,2,3 for elements 0..3
        float[] z = decode(GGMLType.Q2_0, c, 1);
        assertEquals(-0.5f, z[0], 0f);
        assertEquals(0.0f, z[1], 0f);
        assertEquals(0.5f, z[2], 0f);
        assertEquals(1.0f, z[3], 0f);
        assertEquals(-0.5f, z[4], 0f);
    }

    // ------------------------------------------------------------------ ternary

    /** {@code quantize_row_tq1_0_ref} (llama.cpp ggml-quants.c:2317-2381) for trits already in {-1, 0, 1}. */
    private static ByteBuffer encodeTq1(int[] t, int halfD) {
        ByteBuffer b = block(BlockLayout.TQ1_0_BYTES);
        int x = 0;
        for (int m = 0; m < 32; m++) {
            b.put(m, (byte) packTrits(t, x + m, 32, 5));
        }
        x += 160;
        for (int m = 0; m < 16; m++) {
            b.put(32 + m, (byte) packTrits(t, x + m, 16, 5));
        }
        x += 80;
        for (int j = 0; j < 4; j++) {
            int q = 0;
            for (int m = 0; m < 4; m++) {
                q = q * 3 + t[x + j + m * 4] + 1;
            }
            q *= 3;
            b.put(48 + j, (byte) ((q * 256 + 242) / 243));
        }
        b.putShort(BlockLayout.TQ1_0_D, (short) halfD);
        return b;
    }

    private static int packTrits(int[] t, int first, int stride, int count) {
        int q = 0;
        for (int n = 0; n < count; n++) {
            q = q * 3 + t[first + n * stride] + 1;
        }
        return (q * 256 + 242) / 243;
    }

    @Test
    public void tq1_0RoundTripsTernaryValues() {
        int[] t = new int[256];
        for (int i = 0; i < 256; i++) {
            t[i] = (i * 7 + i / 3) % 3 - 1;
        }
        float[] y = decode(GGMLType.TQ1_0, encodeTq1(t, H_TWO), 1);
        for (int i = 0; i < 256; i++) {
            assertEquals("element " + i, t[i] * 2.0f, y[i], 0f);
        }
    }

    @Test
    public void tq2_0UnpacksFourCodesPerByte() {
        ByteBuffer b = block(BlockLayout.TQ2_0_BYTES);
        for (int j = 0; j < 64; j++) {
            b.put(j, (byte) 0x1B);               // codes 3,2,1,0 for shifts 0,2,4,6
        }
        b.putShort(BlockLayout.TQ2_0_D, (short) H_HALF);
        float[] y = decode(GGMLType.TQ2_0, b, 1);
        for (int half = 0; half < 2; half++) {
            for (int m = 0; m < 32; m++) {
                assertEquals(1.0f, y[half * 128 + m], 0f);
                assertEquals(0.5f, y[half * 128 + 32 + m], 0f);
                assertEquals(0.0f, y[half * 128 + 64 + m], 0f);
                assertEquals(-0.5f, y[half * 128 + 96 + m], 0f);
            }
        }
    }

    // ------------------------------------------------------------------ K-quants

    @Test
    public void q4_KAppliesSixBitScalesAndMins() {
        ByteBuffer b = block(BlockLayout.Q4_K_BYTES);
        b.putShort(0, (short) H_ONE).putShort(2, (short) H_HALF);
        byte[] scales = {1, 2, 3, 4, 1, 0, 0, 0, 0x05, 0x06, 0x07, 0x08};
        for (int i = 0; i < 12; i++) {
            b.put(4 + i, scales[i]);
        }
        for (int i = 0; i < 128; i++) {
            b.put(16 + i, (byte) 0x93);          // low 3, high 9
        }
        float[] y = decode(GGMLType.Q4_K, b, 1);
        int[] sc = {1, 2, 3, 4, 5, 6, 7, 8};
        for (int g = 0; g < 4; g++) {
            float m1 = g == 0 ? 0.5f : 0f;
            for (int l = 0; l < 32; l++) {
                assertEquals(sc[2 * g] * 3 - m1, y[64 * g + l], 0f);
                assertEquals(sc[2 * g + 1] * 9, y[64 * g + 32 + l], 0f);
            }
        }
        assertEquals(5, KQuantBlockDecoders.scaleMinK4(4, b, 4));   // scale 5, min 0
    }

    @Test
    public void q6_KCombinesLowAndHighBits() {
        ByteBuffer b = block(BlockLayout.Q6_K_BYTES);
        for (int i = 0; i < 128; i++) {
            b.put(i, (byte) 0x21);               // low nibble 1, high nibble 2
        }
        for (int i = 0; i < 64; i++) {
            b.put(128 + i, (byte) 0xE4);         // 2-bit fields 0,1,2,3
        }
        for (int i = 0; i < 16; i++) {
            b.put(192 + i, (byte) (i - 8));
        }
        b.putShort(BlockLayout.Q6_K_D, (short) H_ONE);
        float[] y = decode(GGMLType.Q6_K, b, 1);
        for (int n = 0; n < 2; n++) {
            for (int l = 0; l < 32; l++) {
                int is = 8 * n + l / 16;
                assertEquals((is - 8) * ((1 | (0 << 4)) - 32), y[128 * n + l], 0f);
                assertEquals((is + 2 - 8) * ((1 | (1 << 4)) - 32), y[128 * n + l + 32], 0f);
                assertEquals((is + 4 - 8) * ((2 | (2 << 4)) - 32), y[128 * n + l + 64], 0f);
                assertEquals((is + 6 - 8) * ((2 | (3 << 4)) - 32), y[128 * n + l + 96], 0f);
            }
        }
    }

    @Test
    public void q2_KAndQ3_KUseSixteenElementGroups() {
        ByteBuffer b = block(BlockLayout.Q2_K_BYTES);
        for (int i = 0; i < 16; i++) {
            b.put(i, (byte) 0x12);               // scale 2, min 1
        }
        for (int i = 0; i < 64; i++) {
            b.put(16 + i, (byte) 0xE4);          // shifts 0,2,4,6 -> 0,1,2,3
        }
        b.putShort(BlockLayout.Q2_K_D, (short) H_ONE).putShort(BlockLayout.Q2_K_DMIN, (short) H_HALF);
        float[] y = decode(GGMLType.Q2_K, b, 1);
        for (int n = 0; n < 2; n++) {
            for (int j = 0; j < 4; j++) {
                for (int l = 0; l < 32; l++) {
                    assertEquals(2.0f * j - 0.5f, y[128 * n + 32 * j + l], 0f);
                }
            }
        }

        ByteBuffer c = block(BlockLayout.Q3_K_BYTES);
        for (int i = 0; i < 32; i++) {
            c.put(i, (byte) 0x0F);               // high bit set for shifts 0..3 of chunk 0 only
        }
        byte[] packed = {0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, (byte) 0xAA, (byte) 0xAA, (byte) 0xAA, (byte) 0xAA};
        for (int i = 0; i < 12; i++) {
            c.put(BlockLayout.Q3_K_SCALES + i, packed[i]);
        }
        c.putShort(BlockLayout.Q3_K_D, (short) H_ONE);
        byte[] s = new byte[16];
        KQuantBlockDecoders.q3kScales(c, BlockLayout.Q3_K_SCALES, s);
        for (int k = 0; k < 16; k++) {
            assertEquals("scale " + k, k < 8 ? 33 : 34, s[k]);   // low nibble 1 or 2, high bits 2
        }
        float[] z = decode(GGMLType.Q3_K, c, 1);
        for (int l = 0; l < 32; l++) {
            assertEquals(0f, z[l], 0f);                           // chunk 0: q2 = 0, high bit set
            assertEquals((34 - 32) * -4f, z[128 + l], 0f);        // chunk 1: high bit clear, q = -4
        }
    }

    @Test
    public void q8_KUsesAFloatScale() {
        ByteBuffer b = block(BlockLayout.Q8_K_BYTES);
        b.putFloat(0, 0.25f);
        for (int i = 0; i < 256; i++) {
            b.put(4 + i, (byte) (i - 128));
        }
        float[] y = decode(GGMLType.Q8_K, b, 1);
        for (int i = 0; i < 256; i++) {
            assertEquals((i - 128) * 0.25f, y[i], 0f);
        }
    }

    // ------------------------------------------------------------------ I-quants

    @Test
    public void iq2AndIq3GridsAtIndexZero() {
        ByteBuffer xxs = block(BlockLayout.IQ2_XXS_BYTES);
        xxs.putShort(0, (short) H_ONE);
        xxs.putInt(2 + 4, 1);                    // sign pattern index 1 for group 0 of sub-block 0
        float[] y = decode(GGMLType.IQ2_XXS, xxs, 1);
        assertEquals(-1.0f, y[0], 0f);           // 1.0 * 0.5 * 0.25 * 8, ksigns[1] = 0b10000001
        assertEquals(1.0f, y[1], 0f);
        assertEquals(-1.0f, y[7], 0f);
        assertEquals(1.0f, y[8], 0f);
        assertEquals(1.0f, y[255], 0f);

        ByteBuffer xs = block(BlockLayout.IQ2_XS_BYTES);
        xs.putShort(0, (short) H_ONE);
        float[] z = decode(GGMLType.IQ2_XS, xs, 1);
        ByteBuffer s2 = block(BlockLayout.IQ2_S_BYTES);
        s2.putShort(0, (short) H_ONE);
        float[] w = decode(GGMLType.IQ2_S, s2, 1);
        ByteBuffer x3 = block(BlockLayout.IQ3_XXS_BYTES);
        x3.putShort(0, (short) H_ONE);
        float[] v = decode(GGMLType.IQ3_XXS, x3, 1);
        ByteBuffer s3 = block(BlockLayout.IQ3_S_BYTES);
        s3.putShort(0, (short) H_ONE);
        float[] u = decode(GGMLType.IQ3_S, s3, 1);
        for (int i = 0; i < 256; i++) {
            assertEquals(1.0f, z[i], 0f);        // 0.125 * lane 8
            assertEquals(1.0f, w[i], 0f);        // 0.125 * lane 8
            assertEquals(1.0f, v[i], 0f);        // 0.25 * lane 4
            assertEquals(1.0f, u[i], 0f);        // 1 * lane 1
        }
    }

    @Test
    public void iq1_SAndIq1_MAddTheGridDelta() {
        ByteBuffer s = block(BlockLayout.IQ1_S_BYTES);
        s.putShort(0, (short) H_ONE);
        s.putShort(BlockLayout.IQ1_S_QH + 2, (short) 0x8000);   // sub-block 1: negative delta
        float[] y = decode(GGMLType.IQ1_S, s, 1);
        assertEquals(-0.875f, y[0], 0f);
        assertEquals(-1.125f, y[32], 0f);

        ByteBuffer m = block(BlockLayout.IQ1_M_BYTES);
        m.putShort(BlockLayout.IQ1_M_SCALES + 4, (short) 0xC000);
        m.putShort(BlockLayout.IQ1_M_SCALES + 6, (short) 0x3000);
        m.put(BlockLayout.IQ1_M_QH, (byte) 0x08);              // group 0 negative delta
        assertEquals(1.0f, IQuantBlockDecoders.iq1mSuperScale(m, BlockLayout.IQ1_M_SCALES), 0f);
        float[] z = decode(GGMLType.IQ1_M, m, 1);
        assertEquals(-1.125f, z[0], 0f);
        assertEquals(-0.875f, z[8], 0f);
        assertEquals(-0.875f, z[255], 0f);
    }

    @Test
    public void iq4UsesTheNonLinearCodebook() {
        int[] kv = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        ByteBuffer nl = block(BlockLayout.IQ4_NL_BYTES);
        nl.putShort(0, (short) H_HALF);
        for (int j = 0; j < 16; j++) {
            nl.put(2 + j, (byte) (j | ((15 - j) << 4)));
        }
        float[] y = decode(GGMLType.IQ4_NL, nl, 1);
        for (int j = 0; j < 16; j++) {
            assertEquals(kv[j] * 0.5f, y[j], 0f);
            assertEquals(kv[15 - j] * 0.5f, y[j + 16], 0f);
        }

        ByteBuffer xs = block(BlockLayout.IQ4_XS_BYTES);
        xs.putShort(0, (short) H_ONE);
        xs.putShort(BlockLayout.IQ4_XS_SCALES_H, (short) 0xAAAA);  // high bits 2 -> ls = low | 32
        xs.put(BlockLayout.IQ4_XS_SCALES_L, (byte) 0x31);          // sub-block 0 low 1, sub-block 1 low 3
        float[] z = decode(GGMLType.IQ4_XS, xs, 1);
        assertEquals(1 * kv[0], z[0], 0f);
        assertEquals(3 * kv[0], z[32], 0f);
        assertEquals(0f, z[64], 0f);
    }

    // ------------------------------------------------------------------ tensor rows

    @Test
    public void tensorRefDecodesRowsAcrossShards() {
        int rowBytes = BlockLayout.Q8_0_BYTES * 2;
        ByteBuffer s0 = block(rowBytes * 2);
        ByteBuffer s1 = block(rowBytes);
        for (int r = 0; r < 3; r++) {
            ByteBuffer s = r < 2 ? s0 : s1;
            int base = (r < 2 ? r : 0) * rowBytes;
            for (int blk = 0; blk < 2; blk++) {
                s.putShort(base + blk * BlockLayout.Q8_0_BYTES, (short) H_ONE);
                for (int j = 0; j < 32; j++) {
                    s.put(base + blk * BlockLayout.Q8_0_BYTES + 2 + j, (byte) (r * 10 + blk));
                }
            }
        }
        CpuTensorRef w = new CpuTensorRef(GGMLType.Q8_0, new ByteBuffer[]{s0, s1}, new long[]{0, 2L * rowBytes},
                3, 64, rowBytes);
        assertEquals(1, w.shardIndex(2));
        assertEquals(0, w.rowOffset(2));
        assertEquals(rowBytes, w.rowOffset(1));
        float[] rows = new float[3 * 64];
        w.decodeRows(0, 3, rows, 0);
        for (int r = 0; r < 3; r++) {
            assertEquals(r * 10f, rows[r * 64], 0f);
            assertEquals(r * 10f + 1, rows[r * 64 + 63], 0f);
        }
        try {
            new CpuTensorRef(GGMLType.Q8_0, new ByteBuffer[]{s0, s1}, new long[]{0, rowBytes + 2}, 3, 64, rowBytes);
            fail("a shard boundary inside a row must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("multiple of rowBytes"));
        }
        assertFalse(w.rowBuffer(0) == s0);
    }
}
