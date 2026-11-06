package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_D;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link TernaryBlockQuantizers}: byte parity with the reference quantizers of gguf-py ({@code TQ1_0} / {@code TQ2_0}
 * {@code quantize_blocks}, the numpy ports of {@code quantize_row_tq*_0_ref}) over three blocks, hand-computed
 * packing of single bytes, the round trip through {@link TernaryBlockDecoders}, the {@code lroundf} rounding and
 * the offset checks.
 */
public class TernaryBlockQuantizersTest {

    /**
     * The gguf-py bytes of {@link #parityInput()}: {@code TQ1_0.quantize_blocks(x.reshape(3, 256))}, the first block a
     * spread of values, the second the rounding ties around half of its scale, the third all zeros.
     */
    private static final String TQ1_0_GOLDEN = "466dd2467ad36c7a456dcf466dd3467ad36c7a456dcf4689d39b7a296d7a486d7f51"
            + "74d51b7cf25d7c515ed45174d51b339fb5205940ff008080ff008080ff008080ff008080ff008080ff008080ff008080ff008080"
            + "ff008080ff008080ff008080ff008080fd007f7f004080808080808080808080808080808080808080808080808080808080808080"
            + "80808080808080808080808080808080807f7f7f7f0000";

    /** The gguf-py bytes of {@link #parityInput()} as {@code TQ2_0.quantize_blocks}. */
    private static final String TQ2_0_GOLDEN =
            "5861165885166185186186586116588516618518618658651659851461855861146185586116584516618518618658611658"
            + "85166185186186586116598516615940aa005555aa005555aa005555aa005555aa005555aa005555aa005555aa005555aa00"
            + "5555aa005555aa005555aa005555aa005555aa005555aa005555aa0055550040555555555555555555555555555555555555"
            + "555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555550000";

    /** The input of the golden bytes, built exactly as the generating script builds it in float32. */
    private static float[] parityInput() {
        float[] x = new float[768];
        for (int i = 0; i < 256; i++) {
            x[i] = (float) ((i * 37 + 11) % 101 - 50) / 23f;
        }
        float[] ties = {1.0f, -1.0f, Math.nextDown(1.0f), -0.25f};
        for (int i = 0; i < 256; i++) {
            x[256 + i] = ties[i % 4];
        }
        x[256] = 2.0f;
        return x;
    }

    @Test
    public void matchesTheReferenceQuantizersByteForByte() {
        float[] x = parityInput();
        for (GGMLType t : new GGMLType[]{GGMLType.TQ1_0, GGMLType.TQ2_0}) {
            byte[] y = new byte[3 * t.getTypeSize()];
            for (int b = 0; b < 3; b++) {
                TernaryBlockQuantizers.quantize(t, x, b * 256, y, b * t.getTypeSize());
            }
            String expected = t == GGMLType.TQ1_0 ? TQ1_0_GOLDEN : TQ2_0_GOLDEN;
            assertEquals(t.name(), expected, hex(y));
        }
    }

    @Test
    public void theFirstBytesCarryTheHandComputedTrits() {
        // d = amax = 2, id = 0.5: every trit is lroundf(x * 0.5), packed as the digit trit + 1.
        float[] x = new float[256];
        x[0] = 2f;
        x[32] = -2f;
        x[64] = 0f;
        x[96] = 2f;
        x[128] = -2f;
        x[240] = 2f;
        x[244] = -2f;
        x[248] = 0f;
        x[252] = 2f;
        byte[] y = new byte[TQ1_0_BYTES];
        TernaryBlockQuantizers.quantizeTq1_0(x, 0, y, 0);
        // byte 0 packs the digits 2, 0, 1, 2, 0: q = 177 and ceil(177 * 256 / 243) = 187.
        assertEquals(187, y[0] & 0xFF);
        // byte 1 packs five zero trits (digit 1 each): q = 121 and ceil(121 * 256 / 243) = 128.
        assertEquals(128, y[1] & 0xFF);
        // qh byte 0 packs elements 240, 244, 248, 252 (digits 2, 0, 1, 2) shifted one trit up: 177 again.
        assertEquals(187, y[TQ1_0_QH] & 0xFF);
        assertEquals(0x00, y[TQ1_0_D] & 0xFF);
        assertEquals(0x40, y[TQ1_0_D + 1] & 0xFF);

        // TQ2_0 byte 0 packs the bit pairs of elements 0, 32, 64, 96 (digits 2, 0, 1, 2): 2 + 16 + 128 = 146.
        byte[] z = new byte[TQ2_0_BYTES];
        TernaryBlockQuantizers.quantizeTq2_0(x, 0, z, 0);
        assertEquals(146, z[0] & 0xFF);
        // a byte of four zero trits: 1 + 4 + 16 + 64 = 85.
        assertEquals(85, z[1] & 0xFF);
        assertEquals(0x00, z[TQ2_0_D] & 0xFF);
        assertEquals(0x40, z[TQ2_0_D + 1] & 0xFF);
    }

    @Test
    public void roundTripsThroughTheDecoders() {
        float[] x = new float[512];
        float[] values = {2f, -2f, 0f, 1f, -1f, 0.5f};
        for (int i = 0; i < x.length; i++) {
            x[i] = values[i % values.length];
        }
        for (GGMLType t : new GGMLType[]{GGMLType.TQ1_0, GGMLType.TQ2_0}) {
            byte[] y = new byte[2 * t.getTypeSize()];
            TernaryBlockQuantizers.quantize(t, x, 0, y, 0);
            TernaryBlockQuantizers.quantize(t, x, 256, y, t.getTypeSize());
            float[] back = new float[512];
            CpuBlockDecoders.decodeElements(t, ByteBuffer.wrap(y).order(ByteOrder.LITTLE_ENDIAN), 0, back, 0, 512);
            for (int i = 0; i < x.length; i++) {
                assertEquals(t + " element " + i, TernaryBlockQuantizers.lroundf(x[i] * 0.5f) * 2f, back[i], 0f);
            }
        }
    }

    @Test
    public void lroundfRoundsHalvesAwayFromZeroAndNothingBelowThem() {
        assertEquals(1, TernaryBlockQuantizers.lroundf(0.5f));
        assertEquals(-1, TernaryBlockQuantizers.lroundf(-0.5f));
        assertEquals(2, TernaryBlockQuantizers.lroundf(1.5f));
        assertEquals(-2, TernaryBlockQuantizers.lroundf(-1.5f));
        assertEquals(0, TernaryBlockQuantizers.lroundf(Math.nextDown(0.5f)));
        assertEquals(0, TernaryBlockQuantizers.lroundf(-Math.nextDown(0.5f)));
        assertEquals(0, TernaryBlockQuantizers.lroundf(-0.0f));
    }

    @Test
    public void aZeroBlockCarriesAZeroScaleAndZeroTrits() {
        float[] x = new float[256];
        byte[] y = new byte[TQ1_0_BYTES];
        TernaryBlockQuantizers.quantizeTq1_0(x, 0, y, 0);
        assertEquals(0, y[TQ1_0_D] & 0xFF);
        assertEquals(0, y[TQ1_0_D + 1] & 0xFF);
        // every qs byte is five zero trits (128) and every qh byte four shifted one up, ceil(120 * 256 / 243) = 127.
        for (int i = 0; i < TQ1_0_QS_BYTES; i++) {
            assertEquals("qs byte " + i, 128, y[i] & 0xFF);
        }
        for (int i = 0; i < TQ1_0_QH_BYTES; i++) {
            assertEquals("qh byte " + i, 127, y[TQ1_0_QH + i] & 0xFF);
        }
    }

    @Test
    public void quantizeDispatchesAndRefusesOtherTypes() {
        float[] x = parityInput();
        byte[] a = new byte[TQ1_0_BYTES];
        byte[] b = new byte[TQ1_0_BYTES];
        TernaryBlockQuantizers.quantize(GGMLType.TQ1_0, x, 0, a, 0);
        TernaryBlockQuantizers.quantizeTq1_0(x, 0, b, 0);
        assertArrayEquals(a, b);
        byte[] c = new byte[TQ2_0_BYTES];
        byte[] d = new byte[TQ2_0_BYTES];
        TernaryBlockQuantizers.quantize(GGMLType.TQ2_0, x, 0, c, 0);
        TernaryBlockQuantizers.quantizeTq2_0(x, 0, d, 0);
        assertArrayEquals(c, d);
        assertThrows(IllegalArgumentException.class,
                () -> TernaryBlockQuantizers.quantize(GGMLType.Q8_0, x, 0, a, 0));
    }

    @Test
    public void refusesOffsetsWithoutAWholeBlock() {
        float[] x = new float[256];
        assertThrows(IllegalArgumentException.class,
                () -> TernaryBlockQuantizers.quantizeTq1_0(x, 1, new byte[TQ1_0_BYTES], 0));
        assertThrows(IllegalArgumentException.class,
                () -> TernaryBlockQuantizers.quantizeTq1_0(x, -1, new byte[TQ1_0_BYTES], 0));
        assertThrows(IllegalArgumentException.class,
                () -> TernaryBlockQuantizers.quantizeTq2_0(x, 0, new byte[TQ2_0_BYTES - 1], 0));
        assertThrows(IllegalArgumentException.class,
                () -> TernaryBlockQuantizers.quantizeTq2_0(x, 0, new byte[TQ2_0_BYTES], 1));
    }

    private static String hex(byte[] bytes) {
        StringBuilder s = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            s.append(String.format("%02x", b & 0xFF));
        }
        return s.toString();
    }
}
