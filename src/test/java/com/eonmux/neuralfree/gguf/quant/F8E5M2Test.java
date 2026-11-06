package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@link GGMLType#F8_E5M2} row (id 1006): its {@link BlockLayout} geometry (32 elements, 32 bytes,
 * codes at offset 0), the exact decode of every code against the IEEE half its high byte encodes, and
 * the row's {@code .scale} behaviour shared with F8_E4M3 (per-tensor or per-expert, applied by the
 * consumer). The vec-dot integration with the other extension types is covered by the cpu/quant tests,
 * the GPU kernels by {@code ExtTypeFastPathTest}.
 */
public class F8E5M2Test {

    private static final GGMLType F8_E5M2 = GGMLType.F8_E5M2;

    @Test
    public void typeRowFollowsTheExtensionTable() {
        assertEquals(1006, F8_E5M2.getId());
        assertEquals(32, F8_E5M2.getTypeSize());
        assertEquals(32, F8_E5M2.getBlockSize());
        assertTrue(F8_E5M2.isQuantized());
        assertTrue(F8_E5M2.isExtension());
        assertEquals(F8_E5M2, GGMLType.fromId(1006));
        assertEquals("the id after the last extension row is unassigned", GGMLType.UNKNOWN,
                GGMLType.fromId(ExtBlockDecodersTest.nextExtensionId()));
        assertTrue(F8_E5M2 + " has a layout", BlockLayout.isDefined(F8_E5M2));
        assertEquals(32, BlockLayout.of(F8_E5M2).blockBytes());
        assertEquals(32, BlockLayout.of(F8_E5M2).blockElements());
        assertEquals(BlockLayout.F8_E5M2_BYTES, BlockLayout.F8_E5M2_QS + F8_E5M2.getBlockSize());
        assertEquals(96, F8_E5M2.getBytesForElements(96));
        assertTrue("CPU decoder", CpuBlockDecoders.supports(F8_E5M2));
    }

    /**
     * Reference decode of an E5M2 code: the code is the high byte of an IEEE half (sign bit 7, exponent
     * bits 6..2 with bias 15, two mantissa bits), computed without touching the engine's codecs.
     */
    private static float e5m2HighByte(int code) {
        final int sign = (code & 0x80) != 0 ? -1 : 1;
        final int exp = (code >>> 2) & 0x1F;
        final int mant = code & 0x3;
        if (exp == 0) {
            return sign * Math.scalb((float) mant, -16);
        }
        if (exp == 31) {
            if (mant != 0) {
                return Float.NaN;
            }
            return Math.copySign(Float.POSITIVE_INFINITY, sign);
        }
        return sign * Math.scalb((float) (4 + mant), exp - 15 - 2);
    }

    /**
     * Every code of one block, decoded through {@code CpuBlockDecoders}, equals the IEEE half of its
     * high byte; the format's anchor values sit where torch.float8_e5m2 (Fp8Codec.E5M2) puts them.
     */
    @Test
    public void everyCodeDecodesAgainstTheIeeeHalfHighByte() {
        final int codes = 256;
        final int nBlocks = codes / F8_E5M2.getBlockSize();
        ByteBuffer b = blocks(nBlocks);
        for (int c = 0; c < codes; c++) {
            b.put(c, (byte) c);
        }
        float[] y = decode(b, nBlocks);
        for (int c = 0; c < codes; c++) {
            float expected = e5m2HighByte(c);
            if (!Float.isNaN(expected)) {
                assertEquals("code 0x" + Integer.toHexString(c), expected, y[c], 0f);
            } else {
                assertTrue("code 0x" + Integer.toHexString(c) + " decodes to NaN", Float.isNaN(y[c]));
            }
        }
        // spot checks of the format's anchors (torch.float8_e5m2: the high byte of the IEEE half)
        assertEquals(0.0f, y[0x00], 0f);
        assertEquals(-0.0f, y[0x80], 0f);
        assertEquals((float) Math.scalb(1f, -16), y[0x01], 0f);
        assertEquals(1.0f, y[0x3C], 0f);
        assertEquals(0.5f, y[0x38], 0f);
        assertEquals(7.0f, y[0x47], 0f);
        assertEquals(57344f, y[0x7B], 0f);
        assertEquals(Float.POSITIVE_INFINITY, y[0x7C], 0f);
        assertEquals(Float.NEGATIVE_INFINITY, y[0xFC], 0f);
        assertEquals(4.0f, y[0x44], 0f);
        assertEquals(2.0f, y[0x40], 0f);
        assertEquals(-7.0f, y[0xC7], 0f);
    }

    /** Random blocks decode the same way; the tensor's {@code .scale} is not part of the block data. */
    @Test
    public void randomBlocksDecodeThroughCpuBlockDecodersWithoutBlockScale() {
        Random rnd = new Random(0xE5E5E5L);
        final int blocks = 8;
        final int nElements = blocks * F8_E5M2.getBlockSize();
        ByteBuffer b = blocks(blocks);
        for (int j = 0; j < nElements; j++) {
            b.put(j, (byte) rnd.nextInt(256));
        }
        float[] y = decode(b, blocks);
        for (int j = 0; j < nElements; j++) {
            int code = b.get(j) & 0xFF;
            float expected = e5m2HighByte(code);
            if (Float.isNaN(expected)) {
                assertTrue("element " + j, Float.isNaN(y[j]));
            } else {
                assertEquals("code 0x" + Integer.toHexString(code) + " at " + j, expected, y[j], 0f);
            }
        }
        try {
            CpuBlockDecoders.decode(F8_E5M2, ByteBuffer.allocate(F8_E5M2.getTypeSize() * 2), 0,
                    new float[3 * F8_E5M2.getBlockSize()], 0, 3);
            fail("a partial block fails");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("limit"));
        }
    }

    // ------------------------------------------------------------------ helpers

    private static ByteBuffer blocks(int nBlocks) {
        return ByteBuffer.allocate(nBlocks * F8_E5M2.getTypeSize()).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static float[] decode(ByteBuffer src, int nBlocks) {
        float[] out = new float[nBlocks * F8_E5M2.getBlockSize()];
        CpuBlockDecoders.decode(F8_E5M2, src, 0, out, 0, nBlocks);
        return out;
    }
}
