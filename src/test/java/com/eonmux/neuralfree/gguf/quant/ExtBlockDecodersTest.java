package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the NeuralFree extension block types: their {@link GGMLType} rows and {@link BlockLayout} geometry, and
 * {@link ExtBlockDecoders} through {@link CpuBlockDecoders} on hand-built blocks whose values follow from the
 * block layouts.
 */
public class ExtBlockDecodersTest {

    private static final GGMLType[] EXT_TYPES = {
        GGMLType.F8_E4M3, GGMLType.F8_E4M3_B128, GGMLType.MXFP8, GGMLType.I8_B128, GGMLType.NF4, GGMLType.FP4_BNB
    };

    private static ByteBuffer blocks(GGMLType type, int nBlocks) {
        return ByteBuffer.allocate(nBlocks * type.getTypeSize()).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static float[] decode(GGMLType type, ByteBuffer src, int nBlocks) {
        float[] out = new float[nBlocks * type.getBlockSize()];
        CpuBlockDecoders.decode(type, src, 0, out, 0, nBlocks);
        return out;
    }

    /** An E4M3 code that is never NaN, with the sign bit set for odd {@code j}. */
    private static int e4m3Code(int j) {
        return ((j * 37) % 0x7F) | ((j & 1) << 7);
    }

    // ------------------------------------------------------------------ type rows and geometry

    @Test
    public void typeRowsFollowTheExtensionTable() {
        int[][] expected = {
            {1000, 32, 32}, {1001, 132, 128}, {1002, 33, 32}, {1003, 132, 128}, {1004, 36, 64}, {1005, 36, 64}
        };
        for (int i = 0; i < EXT_TYPES.length; i++) {
            GGMLType t = EXT_TYPES[i];
            assertEquals(t + " id", expected[i][0], t.getId());
            assertEquals(t + " type size", expected[i][1], t.getTypeSize());
            assertEquals(t + " block size", expected[i][2], t.getBlockSize());
            assertTrue(t + " is quantized", t.isQuantized());
            assertTrue(t + " is an extension", t.isExtension());
            assertEquals(t, GGMLType.fromId(expected[i][0]));
            assertTrue(t + " has a layout", BlockLayout.isDefined(t));
            assertEquals(t.getTypeSize(), BlockLayout.of(t).blockBytes());
            assertEquals(t.getBlockSize(), BlockLayout.of(t).blockElements());
            assertTrue(t + " has a decoder", CpuBlockDecoders.supports(t));
        }
        assertEquals(GGMLType.UNKNOWN, GGMLType.fromId(1006));
        assertFalse(GGMLType.Q8_0.isExtension());
        assertFalse(GGMLType.Q2_0.isExtension());
        assertFalse(GGMLType.UNKNOWN.isExtension());
    }

    @Test
    public void bytesForElementsAreWholeBlocks() {
        assertEquals(96, GGMLType.F8_E4M3.getBytesForElements(96));
        assertEquals(2 * 132, GGMLType.F8_E4M3_B128.getBytesForElements(256));
        assertEquals(2 * 33, GGMLType.MXFP8.getBytesForElements(64));
        assertEquals(3 * 132, GGMLType.I8_B128.getBytesForElements(384));
        assertEquals(2 * 36, GGMLType.NF4.getBytesForElements(128));
        assertEquals(36, GGMLType.FP4_BNB.getBytesForElements(64));
        assertEquals(BlockLayout.F8B128_BYTES, BlockLayout.F8B128_QS + BlockLayout.QK_F8B128);
        assertEquals(BlockLayout.MXFP8_BYTES, BlockLayout.MXFP8_QS + BlockLayout.QK_MXFP8);
        assertEquals(BlockLayout.I8B128_BYTES, BlockLayout.I8B128_QS + BlockLayout.QK_I8B128);
        assertEquals(BlockLayout.BNB4_BYTES, BlockLayout.BNB4_QS + BlockLayout.QK_BNB4 / 2);
    }

    // ------------------------------------------------------------------ decoders

    @Test
    public void f8E4m3DecodesEachCode() {
        ByteBuffer b = blocks(GGMLType.F8_E4M3, 2);
        for (int j = 0; j < 64; j++) {
            b.put(BlockLayout.F8_E4M3_QS + j, (byte) e4m3Code(j));
        }
        b.put(5, (byte) 0x38);
        b.put(40, (byte) 0x7E);
        float[] y = decode(GGMLType.F8_E4M3, b, 2);
        for (int j = 0; j < 64; j++) {
            assertEquals("element " + j, FpBlockDecoders.e4m3ToFp32(b.get(j)), y[j], 0f);
        }
        assertEquals(1f, y[5], 0f);
        assertEquals(448f, y[40], 0f);
    }

    @Test
    public void f8E4m3B128AppliesTheBlockScale() {
        ByteBuffer b = blocks(GGMLType.F8_E4M3_B128, 2);
        float[] d = {0.5f, -3f};
        for (int i = 0; i < 2; i++) {
            int o = i * BlockLayout.F8B128_BYTES;
            b.putFloat(o + BlockLayout.F8B128_D, d[i]);
            for (int j = 0; j < BlockLayout.QK_F8B128; j++) {
                b.put(o + BlockLayout.F8B128_QS + j, (byte) e4m3Code(j + i));
            }
        }
        float[] y = decode(GGMLType.F8_E4M3_B128, b, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < BlockLayout.QK_F8B128; j++) {
                float expected = d[i] * FpBlockDecoders.e4m3ToFp32(e4m3Code(j + i));
                assertEquals("block " + i + " element " + j, expected, y[i * BlockLayout.QK_F8B128 + j], 0f);
            }
        }
    }

    @Test
    public void mxfp8AppliesThePowerOfTwoScale() {
        ByteBuffer b = blocks(GGMLType.MXFP8, 2);
        b.put(BlockLayout.MXFP8_E, (byte) 130);
        b.put(BlockLayout.MXFP8_BYTES + BlockLayout.MXFP8_E, (byte) 0xFF);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < BlockLayout.QK_MXFP8; j++) {
                b.put(i * BlockLayout.MXFP8_BYTES + BlockLayout.MXFP8_QS + j, (byte) e4m3Code(j));
            }
        }
        float[] y = decode(GGMLType.MXFP8, b, 2);
        for (int j = 0; j < BlockLayout.QK_MXFP8; j++) {
            assertEquals("element " + j, 8f * FpBlockDecoders.e4m3ToFp32(e4m3Code(j)), y[j], 0f);
            assertTrue("reserved scale 0xFF decodes to NaN", Float.isNaN(y[BlockLayout.QK_MXFP8 + j]));
        }
    }

    @Test
    public void i8B128AppliesTheBlockScaleToSignedCodes() {
        ByteBuffer b = blocks(GGMLType.I8_B128, 1);
        b.putFloat(BlockLayout.I8B128_D, 0.25f);
        for (int j = 0; j < BlockLayout.QK_I8B128; j++) {
            b.put(BlockLayout.I8B128_QS + j, (byte) (j - 64));
        }
        float[] y = decode(GGMLType.I8_B128, b, 1);
        for (int j = 0; j < BlockLayout.QK_I8B128; j++) {
            assertEquals("element " + j, (j - 64) * 0.25f, y[j], 0f);
        }
        assertEquals(-16f, y[0], 0f);
        assertEquals(15.75f, y[127], 0f);
    }

    @Test
    public void bnb4BlocksSplitNibblesOverHalves() {
        for (GGMLType t : new GGMLType[]{GGMLType.NF4, GGMLType.FP4_BNB}) {
            ByteBuffer b = blocks(t, 1);
            b.putFloat(BlockLayout.BNB4_D, 2f);
            for (int j = 0; j < BlockLayout.QK_BNB4 / 2; j++) {
                int lo = j % 16;
                int hi = (j + 5) % 16;
                b.put(BlockLayout.BNB4_QS + j, (byte) (lo | (hi << 4)));
            }
            float[] y = decode(t, b, 1);
            for (int j = 0; j < BlockLayout.QK_BNB4 / 2; j++) {
                assertEquals(t + " element " + j, code(t, j % 16) * 2f, y[j], 0f);
                assertEquals(t + " element " + (j + 32), code(t, (j + 5) % 16) * 2f, y[j + 32], 0f);
            }
        }
    }

    private static float code(GGMLType t, int i) {
        return t == GGMLType.NF4 ? QuantTables.nf4Code(i) : QuantTables.fp4BnbCode(i);
    }

    // ------------------------------------------------------------------ round trips

    /** Values representable in each format are packed by hand and decode back unchanged. */
    @Test
    public void representableValuesRoundTrip() {
        float[] e4m3Values = new float[128];
        for (int j = 0; j < e4m3Values.length; j++) {
            e4m3Values[j] = FpBlockDecoders.e4m3ToFp32(e4m3Code(j));
        }

        ByteBuffer b128 = blocks(GGMLType.F8_E4M3_B128, 1);
        b128.putFloat(BlockLayout.F8B128_D, 0.125f);
        for (int j = 0; j < 128; j++) {
            b128.put(BlockLayout.F8B128_QS + j, (byte) FpBlockDecoders.fp32ToE4m3(e4m3Values[j]));
        }
        float[] y = decode(GGMLType.F8_E4M3_B128, b128, 1);
        for (int j = 0; j < 128; j++) {
            assertEquals("F8_E4M3_B128 element " + j, e4m3Values[j] * 0.125f, y[j], 0f);
        }

        ByteBuffer mx = blocks(GGMLType.MXFP8, 1);
        mx.put(BlockLayout.MXFP8_E, (byte) 127);
        for (int j = 0; j < 32; j++) {
            mx.put(BlockLayout.MXFP8_QS + j, (byte) FpBlockDecoders.fp32ToE4m3(e4m3Values[j]));
        }
        y = decode(GGMLType.MXFP8, mx, 1);
        for (int j = 0; j < 32; j++) {
            assertEquals("MXFP8 element " + j, e4m3Values[j], y[j], 0f);
        }

        ByteBuffer f8 = blocks(GGMLType.F8_E4M3, 1);
        for (int j = 0; j < 32; j++) {
            f8.put(BlockLayout.F8_E4M3_QS + j, (byte) FpBlockDecoders.fp32ToE4m3(e4m3Values[j + 32]));
        }
        y = decode(GGMLType.F8_E4M3, f8, 1);
        for (int j = 0; j < 32; j++) {
            assertEquals("F8_E4M3 element " + j, e4m3Values[j + 32], y[j], 0f);
        }

        ByteBuffer i8 = blocks(GGMLType.I8_B128, 1);
        i8.putFloat(BlockLayout.I8B128_D, 0.5f);
        for (int j = 0; j < 128; j++) {
            float value = ((j * 29) % 256 - 128) * 0.5f;
            i8.put(BlockLayout.I8B128_QS + j, (byte) Math.round(value / 0.5f));
        }
        y = decode(GGMLType.I8_B128, i8, 1);
        for (int j = 0; j < 128; j++) {
            assertEquals("I8_B128 element " + j, ((j * 29) % 256 - 128) * 0.5f, y[j], 0f);
        }

        for (GGMLType t : new GGMLType[]{GGMLType.NF4, GGMLType.FP4_BNB}) {
            float absmax = 4f;
            ByteBuffer bnb = blocks(t, 1);
            bnb.putFloat(BlockLayout.BNB4_D, absmax);
            float[] values = new float[64];
            for (int j = 0; j < 64; j++) {
                values[j] = code(t, (j * 7) % 16) * absmax;
            }
            for (int j = 0; j < 32; j++) {
                int lo = indexOf(t, values[j] / absmax);
                int hi = indexOf(t, values[j + 32] / absmax);
                bnb.put(BlockLayout.BNB4_QS + j, (byte) (lo | (hi << 4)));
            }
            y = decode(t, bnb, 1);
            for (int j = 0; j < 64; j++) {
                assertEquals(t + " element " + j, values[j], y[j], 0f);
            }
        }
    }

    /** The codebook index of {@code value} in the codebook of {@code t}. */
    private static int indexOf(GGMLType t, float value) {
        for (int i = 0; i < 16; i++) {
            if (code(t, i) == value) {
                return i;
            }
        }
        throw new AssertionError(value + " is not a " + t + " code");
    }

    @Test
    public void wholeTensorDequantizationAndRangeErrors() {
        byte[] raw = new byte[2 * BlockLayout.I8B128_BYTES];
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        b.putFloat(0, 1f);
        b.putFloat(BlockLayout.I8B128_BYTES, -2f);
        raw[BlockLayout.I8B128_QS] = 3;
        raw[BlockLayout.I8B128_BYTES + BlockLayout.I8B128_QS + 127] = -5;
        float[] y = CpuBlockDecoders.dequantize(raw, GGMLType.I8_B128, 256);
        assertEquals(3f, y[0], 0f);
        assertEquals(10f, y[255], 0f);
        try {
            CpuBlockDecoders.dequantize(raw, GGMLType.I8_B128, 200);
            fail("partial block");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("I8_B128"));
        }
        try {
            CpuBlockDecoders.decode(GGMLType.NF4, ByteBuffer.allocate(35), 0, new float[64], 0, 1);
            fail("source overrun");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage().contains("limit"));
        }
    }
}
