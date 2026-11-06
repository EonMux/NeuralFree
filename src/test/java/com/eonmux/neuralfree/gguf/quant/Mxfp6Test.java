package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.ExtVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@link GGMLType#MXFP6_E3M2} and {@link GGMLType#MXFP6_E2M3} rows (ids 1007 / 1008, Quark OCP-MX
 * MXFP6): their {@link BlockLayout} geometry (32 elements, 25 bytes: the E8M0 scale byte and 24 bytes of
 * 32 packed 6-bit codes, element {@code j} = code {@code j}, four codes per three bytes little-endian),
 * the exact decode of every code against the OCP-MX arithmetic of {@link FpBlockDecoders#e2m3ToFp32} /
 * {@link FpBlockDecoders#e3m2ToFp32} ({@code 2^(e - 127)} scales included), and the load-time rejection of
 * the NaN E8M0 reservation {@code e == 0xFF} (ExtBlockDecoders, the OCP-MX spec).
 * The vec-dot integration with the other extension types is covered by the cpu/quant tests
 * ({@code ExtVecDot} registers both types with Q8_0 activations), the GPU kernels by the
 * {@code ExtTypeFastPath} KV and the shader twins of common/quant_ext.glsl.
 */
public class Mxfp6Test {

    private static final GGMLType E3M2 = GGMLType.MXFP6_E3M2;
    private static final GGMLType E2M3 = GGMLType.MXFP6_E2M3;

    @Test
    public void typeRowsFollowTheExtensionTable() {
        assertEquals(1007, E3M2.getId());
        assertEquals(25, E3M2.getTypeSize());
        assertEquals(32, E3M2.getBlockSize());
        assertEquals(1008, E2M3.getId());
        assertEquals(25, E2M3.getTypeSize());
        assertEquals(32, E2M3.getBlockSize());
        assertTrue(E3M2.isQuantized() && E2M3.isQuantized());
        assertTrue(E3M2.isExtension() && E2M3.isExtension());
        assertEquals(E3M2, GGMLType.fromId(1007));
        assertEquals(E2M3, GGMLType.fromId(1008));
        assertEquals(GGMLType.F8_E4M3_B32, GGMLType.fromId(1009));
        assertEquals("the next extension id stays unassigned", GGMLType.UNKNOWN, GGMLType.fromId(1010));
        assertTrue("128 elements of E3M2 fill 100 bytes (4 blocks of 25)", E3M2.getBytesForElements(128) == 100);
        assertTrue(CpuBlockDecoders.supports(E3M2) && CpuBlockDecoders.supports(E2M3));
        assertTrue("the CPU vec-dot registers both rows", ExtVecDot.supports(E3M2) && ExtVecDot.supports(E2M3));
        assertEquals(GGMLType.Q8_0, CpuActivationQuantizer.activationType(E3M2));
        assertEquals(GGMLType.Q8_0, CpuActivationQuantizer.activationType(E2M3));
    }

    @Test
    public void layoutsCarryTheBlockGeometry() {
        for (GGMLType t : new GGMLType[]{E3M2, E2M3}) {
            BlockLayout layout = BlockLayout.of(t);
            assertEquals(32, layout.blockElements());
            assertEquals(25, layout.blockBytes());
            assertEquals(BlockLayout.MXFP6_BYTES, BlockLayout.MXFP6_E + 1 + 24);
            assertEquals(BlockLayout.MXFP6_QS, BlockLayout.MXFP6_E + 1);
            assertEquals(25, BlockLayout.MXFP6_BYTES);
            assertEquals(32, BlockLayout.QK_MXFP6);
        }
    }

    /** Reference decode of an MXFP6 code, computed without touching the engine's codecs: sign bit 5;
     *  for E2M3 the exponent bits 4..3 (bias 1) and three mantissa bits, for E3M2 the exponent bits 4..2
     *  (bias 3) and two mantissa bits. The formats have no infinities and no NaNs. */
    private static float e2m3_(int code) {
        final int sign = (code & 0x20) != 0 ? -1 : 1;
        final int exp = (code >>> 3) & 0x3;
        final int mant = code & 0x7;
        return exp == 0 ? sign * Math.scalb((float) mant, -3) : sign * Math.scalb((float) (8 + mant), exp - 4);
    }

    private static float e3m2_(int code) {
        final int sign = (code & 0x20) != 0 ? -1 : 1;
        final int exp = (code >>> 2) & 0x7;
        final int mant = code & 0x3;
        return exp == 0 ? sign * Math.scalb((float) mant, -4) : sign * Math.scalb((float) (4 + mant), exp - 5);
    }

    /** {@link FpBlockDecoders} agrees with the reference for every code of both formats. */
    @Test
    public void fp6DecodersAgreeWithTheOcpMxArithmetic() {
        for (int c = 0; c < 64; c++) {
            assertEquals("E2M3 code 0x" + Integer.toHexString(c), e2m3_(c), FpBlockDecoders.e2m3ToFp32(c), 0f);
            assertEquals("E3M2 code 0x" + Integer.toHexString(c), e3m2_(c), FpBlockDecoders.e3m2ToFp32(c), 0f);
        }
    }

    /**
     * A block with {@code element j = code j} (the distinct codes <code>(i + j + 1) % 64</code> per block
     * {@code i}, so the 6-bit packing order is exercised end to end) decodes through
     * {@code CpuBlockDecoders}, package-private {@link ExtBlockDecoders#decodeMxfp6} and the kernel
     * element order: value {@code 2^(e - 127) * fp6(code)} per element.
     */
    @Test
    public void blocksDecodeWithTheSixBitOrderByFourCodesPerThreeBytes() throws Exception {
        Random rnd = new Random(0x6E6EL);
        final int blocks = 4;
        for (GGMLType t : new GGMLType[]{E3M2, E2M3}) {
            final int n = blocks * t.getBlockSize();
            byte[] raw = new byte[blocks * t.getTypeSize()];
            for (int i = 0; i < blocks; i++) {
                raw[i * t.getTypeSize()] = (byte) (112 + rnd.nextInt(8)); // finite E8M0, 2^-15 .. 2^-8
                // element j of block i = code j, packed four codes per three bytes little-endian
                for (int j = 0; j < t.getBlockSize(); j++) {
                    int code = (i + 2 * j + 1) % 64; // distinct per j within a block; wraps test both orders
                    putCode6(raw, i * t.getTypeSize() + BlockLayout.MXFP6_E + 1, j, code);
                }
            }
            ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            float[] y = new float[n];
            CpuBlockDecoders.decode(t, b, 0, y, 0, blocks);
            float[] y2 = new float[n];
            ExtBlockDecoders.decodeMxfp6(b, 0, y2, 0, blocks, t == E3M2);
            // element unpacked by hand through a little-endian 24-bit word at MXFP6_QS + 3 * (j >> 2)
            for (int j = 0; j < n; j++) {
                int bi = j / t.getBlockSize();
                int e = j % t.getBlockSize();
                int base = bi * t.getTypeSize() + BlockLayout.MXFP6_E + 1;
                int word = (raw[base + 3 * (e / 4)] & 0xFF) | (raw[base + 3 * (e / 4) + 1] & 0xFF) << 8
                        | (raw[base + 3 * (e / 4) + 2] & 0xFF) << 16;
                int code = (word >>> (6 * (e & 3))) & 0x3F;
                float expected = FpBlockDecoders.e8m0ToFp32(raw[bi * t.getTypeSize()] & 0xFF)
                        * (t == E3M2 ? e3m2_(code) : e2m3_(code));
                assertEquals("element " + j + " of " + t, expected, y[j], 0f);
                assertEquals("element " + j + " of " + t + " (ExtBlockDecoders)", y[j], y2[j], 0f);
            }
        }
    }

    /** The anchor values of both formats decode exactly; the largest magnitudes are 28 (E3M2) and 7.5 (E2M3). */
    @Test
    public void anchorValuesDecodeExactly() {
        assertEquals(0f, FpBlockDecoders.e3m2ToFp32(0x00), 0f);
        assertEquals(-0f, FpBlockDecoders.e3m2ToFp32(0x20), 0f);
        assertEquals(Math.scalb(1f, -4), FpBlockDecoders.e3m2ToFp32(0x01), 0f);
        assertEquals(1f, FpBlockDecoders.e3m2ToFp32(0x0C), 0f);   // e=3, m=0: 4 * 2^(3-5) = 1
        assertEquals(4f, FpBlockDecoders.e3m2ToFp32(0x14), 0f);   // e=5, m=0: 4 * 2^(5-5) = 4
        assertEquals(16f, FpBlockDecoders.e3m2ToFp32(0x1C), 0f);  // e=7, m=0: 4 * 2^(7-5) = 16
        assertEquals(28f, FpBlockDecoders.e3m2ToFp32(0x1F), 0f);  // e=7, m=3: 7 * 2^(7-5) = 28
        assertEquals(-28f, FpBlockDecoders.e3m2ToFp32(0x3F), 0f); // the sign bit 0x20 is set
        assertEquals(0f, FpBlockDecoders.e2m3ToFp32(0x00), 0f);
        assertEquals(-0f, FpBlockDecoders.e2m3ToFp32(0x20), 0f);
        assertEquals(Math.scalb(1f, -3), FpBlockDecoders.e2m3ToFp32(0x01), 0f);
        assertEquals(1f, FpBlockDecoders.e2m3ToFp32(0x08), 0f);   // e=1, m=0: 8 * 2^(1-4) = 1
        assertEquals(4f, FpBlockDecoders.e2m3ToFp32(0x18), 0f);   // e=3, m=0: 8 * 2^(3-4) = 4
        assertEquals(7.5f, FpBlockDecoders.e2m3ToFp32(0x1F), 0f); // e=3, m=7: 15 * 2^(3-4) = 7.5
        assertEquals(-7.5f, FpBlockDecoders.e2m3ToFp32(0x3F), 0f); // the sign bit 0x20 is set
        for (int c = 0; c < 64; c++) {
            assertTrue("FP6 has no infinities", Float.isFinite(FpBlockDecoders.e3m2ToFp32(c)));
            assertTrue("FP6 has no NaNs", !Float.isNaN(FpBlockDecoders.e3m2ToFp32(c)));
            assertTrue(Float.isFinite(FpBlockDecoders.e2m3ToFp32(c)));
            assertTrue(!Float.isNaN(FpBlockDecoders.e2m3ToFp32(c)));
        }
    }

    /** {@code e == 0xFF} (the E8M0 NaN reservation) is rejected at load, by both decoders. */
    @Test
    public void nanE8m0ExponentByteIsRejectedAtLoad() {
        final int n = GGMLType.MXFP6_E3M2.getBlockSize();
        for (GGMLType t : new GGMLType[]{E3M2, E2M3}) {
            ByteBuffer b = ByteBuffer.allocate(t.getTypeSize()).order(ByteOrder.LITTLE_ENDIAN);
            b.put(0, (byte) 0xFF);
            try {
                CpuBlockDecoders.decode(t, b, 0, new float[n], 0, 1);
                fail(t + " does not accept the E8M0 NaN reservation");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("0xFF"));
            }
            try {
                ExtBlockDecoders.decodeMxfp6(b, 0, new float[n], 0, 1, t == E3M2);
                fail(t + " does not accept the E8M0 NaN reservation");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("0xFF"));
            }
            try {
                CpuBlockDecoders.decode(t, ByteBuffer.allocate(t.getTypeSize() * 2), 0,
                        new float[3 * t.getBlockSize()], 0, 3);
                fail("a partial block fails");
            } catch (IndexOutOfBoundsException expected) {
                assertTrue(expected.getMessage().contains("limit"));
            }
        }
    }

    /** Packs code {@code code} as element {@code j} of the {@code qs} bytes starting at {@code qs}:
     *  four 6-bit codes per three bytes, little-endian bit order. */
    private static void putCode6(byte[] raw, int qs, int j, int code) {
        final int off = qs + 3 * (j >> 2);
        final int word = (raw[off] & 0xFF) | (raw[off + 1] & 0xFF) << 8 | (raw[off + 2] & 0xFF) << 16;
        final int w = word | (code << (6 * (j & 3)));
        raw[off] = (byte) w;
        raw[off + 1] = (byte) (w >> 8);
        raw[off + 2] = (byte) (w >> 16);
    }
}
