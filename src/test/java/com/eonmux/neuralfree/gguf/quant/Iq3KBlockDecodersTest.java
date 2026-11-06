package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_KS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Parity tests of the IQ3_K / IQ3_KS CPU decoders ({@link IQuantBlockDecoders#decodeIq3K} and
 * {@code decodeIq3Ks}) against an independent port of the Avalon reference functions
 * {@code dequantize_row_iq3_k} / {@code dequantize_row_iq3_ks} (Avalon ggml/src/ggml-quants.c:5381 and
 * :5613), on hand-traced blocks with exact expected values and on random blocks. Pure logic; no GPU.
 */
public class Iq3KBlockDecodersTest {

    private static final int HALF_ONE = 0x3C00;

    /** Avalon iq3nl_values (ggml-common.h:1151-1155): the base grid, then its shifted copy. */
    private static final int[] IQ3NL = {-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51};

    // ------------------------------------------------------------------ block builders

    private static ByteBuffer block(GGMLType type) {
        return ByteBuffer.allocate(type == GGMLType.IQ3_K ? IQ3_K_BYTES : IQ3_KS_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
    }

    private static void putHalf(ByteBuffer b, int at, int bits) {
        b.putShort(at, (short) bits);
    }

    /** f16 bit pattern of a positive value in [0.5, 1). */
    private static int randomScale(Random rnd) {
        return FpBlockDecoders.fp32ToFp16(0.5f + rnd.nextFloat() * 0.5f);
    }

    // ------------------------------------------------------------------ reference ports

    /**
     * Line-by-line port of Avalon {@code dequantize_row_iq3_k} (ggml-quants.c:5381-5410): per 32-element
     * sub-block the two 16-element halves have scales {@code d * (2*ls + 1) * sign} from the low/high nibble
     * of {@code scales_l[ib32]} and bits {@code 2*ib32} / {@code 2*ib32+1} of {@code scales_h}; element
     * {@code 32*ib32 + j} takes its two low code bits from qs byte {@code 32*(ib32/4) + (j % 32)} at shift
     * {@code 2*(ib32%4)}, its high bit from bit {@code ib32} of qh byte {@code j % 32}, and its grid from bit
     * {@code 2*ib32 + j/16} of {@code extra}.
     */
    private static float[] refDequantIq3K(ByteBuffer le) {
        float[] y = new float[QK_K];
        final float d = FpBlockDecoders.fp16ToFp32(le.getShort(0) & 0xFFFF);
        int sh = le.getShort(4) & 0xFFFF;
        int extra = le.getShort(2) & 0xFFFF;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            final int sl = le.get(6 + ib32) & 0xFF;
            final float dl1 = d * (2 * (sl & 0xf) + 1) * ((sh & 1) != 0 ? -1 : 1);
            final float dl2 = d * (2 * (sl >>> 4) + 1) * ((sh & 2) != 0 ? -1 : 1);
            sh >>= 2;
            final int grid1 = (extra & 1) * 8;
            final int grid2 = (extra & 2) * 4;
            extra >>= 2;
            final int shiftL = 2 * (ib32 % 4);
            final int shiftH = ib32 % 8;
            final int qs = 32 * (ib32 / 4);
            for (int j = 0; j < 16; ++j) {
                y[32 * ib32 + j] = dl1 * IQ3NL[grid1
                        + ((le.get(14 + qs + j) & 0xFF) >> shiftL & 3)
                        + ((le.get(78 + j) & 0xFF) >> shiftH & 1) * 4];
                y[32 * ib32 + j + 16] = dl2 * IQ3NL[grid2
                        + ((le.get(14 + qs + j + 16) & 0xFF) >> shiftL & 3)
                        + ((le.get(78 + j + 16) & 0xFF) >> shiftH & 1) * 4];
            }
        }
        return y;
    }

    /**
     * Line-by-line port of Avalon {@code dequantize_row_iq3_ks} (ggml-quants.c:5613-5632): sub-block
     * {@code ib32} has the signed 5-bit scale {@code (nibble ib32/4 of scales[ib32%4] | bit ib32 of extra
     * << 4) - 16} and the grid-select bit {@code 8 + ib32} of {@code extra}; the codes index as IQ3_K.
     */
    private static float[] refDequantIq3Ks(ByteBuffer le) {
        float[] y = new float[QK_K];
        final float d = FpBlockDecoders.fp16ToFp32(le.getShort(0) & 0xFFFF);
        final int extra = le.getShort(2) & 0xFFFF;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            final int ls = ((((le.get(4 + ib32 % 4) & 0xFF) >>> (4 * (ib32 / 4))) & 0xf)
                    | (((extra >> ib32) & 1) << 4)) - 16;
            final float dl = d * ls;
            final int grid = ((extra >> (8 + ib32)) & 1) * 8;
            final int shiftL = 2 * (ib32 % 4);
            final int shiftH = ib32 % 8;
            final int qs = 32 * (ib32 / 4);
            for (int j = 0; j < 32; ++j) {
                y[32 * ib32 + j] = dl * IQ3NL[grid
                        + ((le.get(8 + qs + j) & 0xFF) >> shiftL & 3)
                        + ((le.get(72 + j) & 0xFF) >> shiftH & 1) * 4];
            }
        }
        return y;
    }

    /** Decodes one whole block with the NF decoder under test. */
    private static float[] decode(GGMLType type, ByteBuffer blockBits) {
        float[] y = new float[QK_K];
        ByteBuffer le = blockBits.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        if (type == GGMLType.IQ3_K) {
            IQuantBlockDecoders.decodeIq3K(le, 0, y, 0, 1);
        } else {
            IQuantBlockDecoders.decodeIq3Ks(le, 0, y, 0, 1);
        }
        return y;
    }

    // ------------------------------------------------------------------ hand traces

    /** An all-zero IQ3_K block with d = 1 decodes every element to {@code iq3nl_values[0] = -63}. */
    @Test
    public void iq3KZeroBlockIsTheFirstGridValue() {
        ByteBuffer b = block(GGMLType.IQ3_K);
        putHalf(b, 0, HALF_ONE);
        float[] y = decode(GGMLType.IQ3_K, b);
        for (int i = 0; i < QK_K; i++) {
            assertEquals("element " + i, -63f, y[i], 0f);
        }
    }

    /**
     * Sub-block 0 of IQ3_K with scale nibbles 2 and 1, sign bit 1 of {@code scales_h} (the second half),
     * grid-select bit 0 of {@code extra} (the first half only, the second half's bit 1 is clear), qs byte 0
     * = 0x11 and qh byte 0 bit 0: element 0 is {@code 5 * iq3nl[8 + (1 | 4)] = 5 * 17} (shifted grid, low
     * code 1, high bit set); elements 1-15 have low code 0 on the shifted grid, {@code 5 * iq3nl[8]};
     * elements 16-31 (second half, signed, base grid, low code 0) are {@code -3 * iq3nl[0]}.
     */
    @Test
    public void iq3KSubBlock0FieldLayout() {
        ByteBuffer b = block(GGMLType.IQ3_K);
        putHalf(b, 0, HALF_ONE);
        b.putShort(2, (short) 0b01);         // extra: bit 0 shifts the grid of the first half only
        b.putShort(4, (short) 0x2);          // scales_h: bit 1 signs the second half
        b.put(6, (byte) 0x12);               // scales_l[0]: low nibble 2, high nibble 1
        b.put(14, (byte) 0x11);              // qs[0]: low code 1 of elements 0 and 16
        b.put(78, (byte) 0x1);               // qh[0]: high bit of element 0 (ib32 = 0)
        float[] y = decode(GGMLType.IQ3_K, b);
        assertEquals(5f * 17f, y[0], 0f);
        for (int j = 1; j < 16; j++) {
            assertEquals("element " + j, 5f * -59f, y[j], 0f);
        }
        for (int j = 16; j < 32; j++) {
            assertEquals("element " + j, -3f * -63f, y[j], 0f);
        }
        // sub-blocks 1-7: the low code of qs byte 0 reappears at shift 4 as element 64's code (ib32 = 2
        // reads the same byte at shift 2*(2%4)); every other element outside sub-block 0 is untouched
        assertEquals(-40f, y[64], 0f);
        for (int j = 32; j < QK_K; j++) {
            if (j == 64) {
                continue;
            }
            assertEquals("element " + j, -63f, y[j], 0f);
        }
    }

    /**
     * The qs window advances every four sub-blocks and each qs byte serves four elements: qs byte 4 holds
     * the codes of elements 4, 36, 68 and 100 at shifts 0, 2, 4 and 6, and qh byte 4 holds their high bits
     * at bit positions 0, 1, 2 and 3 (and element 132's at bit 4). Element 132 reads qs byte 36 (the second
     * window) at shift 0.
     */
    @Test
    public void iq3KWindowAdvance() {
        ByteBuffer b = block(GGMLType.IQ3_K);
        putHalf(b, 0, HALF_ONE);
        b.put(14 + 4, (byte) 0b00000100);    // qs[4]: bits 2-3 = 1 -> element 36's low code
        b.put(14 + 36, (byte) 0b00000011);   // qs[36]: bits 0-1 = 3 -> element 132's low code
        b.put(78 + 4, (byte) 0b00010000);    // qh[4]: bit 4 -> element 132's high bit
        float[] y = decode(GGMLType.IQ3_K, b);
        // element 36: ib32 = 1, code 1 -> -40; element 132: ib32 = 4, code 3 | 4 -> iq3nl[7] = 47
        assertEquals(-40f, y[36], 0f);
        assertEquals(47f, y[132], 0f);
        // the neighbours share neither code bits nor qh bits: all decode to the first grid value
        assertEquals(-63f, y[4], 0f);
        assertEquals(-63f, y[37], 0f);
        assertEquals(-63f, y[68], 0f);
        assertEquals(-63f, y[100], 0f);
        assertEquals(-63f, y[133], 0f);
        assertEquals(-63f, y[164], 0f);
    }

    /** An all-zero IQ3_KS block with d = 1 has scale {@code ls = -16}: every element is {@code -16 * -63}. */
    @Test
    public void iq3KsZeroBlockHasTheNegativeBiasScale() {
        ByteBuffer b = block(GGMLType.IQ3_KS);
        putHalf(b, 0, HALF_ONE);
        float[] y = decode(GGMLType.IQ3_KS, b);
        for (int i = 0; i < QK_K; i++) {
            assertEquals("element " + i, -16f * -63f, y[i], 0f);
        }
    }

    /**
     * IQ3_KS scale and grid packing: sub-blocks 0 and 4 share scales byte 0 (low and high nibble), extra
     * bit 4 is sub-block 4's scale bit 4 and extra bit 8 is sub-block 0's grid-select bit. Sub-block 0 has
     * {@code ls = 1 - 16 = -15} on the shifted grid, sub-block 1 keeps {@code ls = -16} on the base grid,
     * and sub-block 4 has {@code ls = (2 | 16) - 16 = 2}.
     */
    @Test
    public void iq3KsScaleAndGridPacking() {
        ByteBuffer b = block(GGMLType.IQ3_KS);
        putHalf(b, 0, HALF_ONE);
        b.putShort(2, (short) ((1 << 4) | (1 << 8)));  // extra: bit 4 (scale of sb 4), bit 8 (grid of sb 0)
        b.put(4, (byte) 0x21);                          // scales[0]: sb 0 nibble 1, sb 4 nibble 2
        float[] y = decode(GGMLType.IQ3_KS, b);
        for (int j = 0; j < 32; j++) {
            assertEquals("element " + j, -15f * -59f, y[j], 0f);
        }
        for (int j = 32; j < 64; j++) {
            assertEquals("element " + j, -16f * -63f, y[j], 0f);
        }
        for (int j = 128; j < 160; j++) {
            assertEquals("element " + j, 2f * -63f, y[j], 0f);
        }
    }

    // ------------------------------------------------------------------ reference agreement

    private static void checkAgainstReference(GGMLType type, long seed) {
        Random rnd = new Random(seed);
        int bytes = type == GGMLType.IQ3_K ? IQ3_K_BYTES : IQ3_KS_BYTES;
        for (int trial = 0; trial < 32; trial++) {
            byte[] raw = new byte[bytes];
            rnd.nextBytes(raw);
            int h = randomScale(rnd);
            raw[0] = (byte) h;
            raw[1] = (byte) (h >>> 8);
            ByteBuffer bits = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            float[] expected = type == GGMLType.IQ3_K ? refDequantIq3K(bits) : refDequantIq3Ks(bits);
            assertArrayEquals(type + " trial " + trial, expected, decode(type, bits), 0f);
        }
    }

    @Test
    public void iq3KMatchesTheAvalonReferenceOnRandomBlocks() {
        for (long seed : new long[]{7, 43, 2026}) {
            checkAgainstReference(GGMLType.IQ3_K, seed);
        }
    }

    @Test
    public void iq3KsMatchesTheAvalonReferenceOnRandomBlocks() {
        for (long seed : new long[]{9, 44, 2027}) {
            checkAgainstReference(GGMLType.IQ3_KS, seed);
        }
    }

    /** The random-block trials cover every bit position; the crafted block of the field-layout trace too. */
    @Test
    public void referencePortMatchesTheDecoderOnTheCraftedBlocks() {
        ByteBuffer k = block(GGMLType.IQ3_K);
        putHalf(k, 0, HALF_ONE);
        k.putShort(2, (short) 0x55AA);
        k.putShort(4, (short) 0x3C3C);
        for (int i = 0; i < 8; i++) {
            k.put(6 + i, (byte) 0x93);
            k.put(14 + i, (byte) 0x7B);
            k.put(14 + 32 + i, (byte) 0xE4);
            k.put(78 + i, (byte) 0x5D);
        }
        assertArrayEquals(refDequantIq3K(k), decode(GGMLType.IQ3_K, k), 0f);

        ByteBuffer ks = block(GGMLType.IQ3_KS);
        putHalf(ks, 0, HALF_ONE);
        ks.putShort(2, (short) 0xA95E);
        for (int i = 0; i < 4; i++) {
            ks.put(4 + i, (byte) 0x6C);
            ks.put(8 + i, (byte) 0x1F);
            ks.put(8 + 32 + i, (byte) 0xC9);
            ks.put(72 + i, (byte) 0xB7);
        }
        assertArrayEquals(refDequantIq3Ks(ks), decode(GGMLType.IQ3_KS, ks), 0f);
    }
}
