package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.Layout;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link IntPacking}: both word layouts and their selection, bit-stream values straddling words, the AWQ order, the
 * stable act-order argsort, fp16 representability, and every block encoder decoded back through
 * {@link CpuBlockDecoders} to the exact element values.
 */
public class IntPackingTest {

    @Test
    public void perWordSlotsAreLeastSignificantFirst() {
        int word = 0x87654321;
        for (int i = 0; i < 8; i++) {
            assertEquals(i + 1, IntPacking.slot(word, i, 4));
        }
        assertEquals(8, IntPacking.perWord(4));
        assertEquals(10, IntPacking.perWord(3));
        assertEquals(0x7, IntPacking.mask(3));
    }

    @Test
    public void wordCountsSelectTheLayout() {
        assertEquals(7, IntPacking.wordsFor(64, 3, Layout.PER_WORD));
        assertEquals(6, IntPacking.wordsFor(64, 3, Layout.BIT_STREAM));
        assertEquals(Layout.PER_WORD, IntPacking.layoutOf(64, 7, 3));
        assertEquals(Layout.BIT_STREAM, IntPacking.layoutOf(64, 6, 3));
        assertNull(IntPacking.layoutOf(64, 5, 3));
        assertEquals(Layout.PER_WORD, IntPacking.layoutOf(64, 8, 4));
        assertFalse(IntPacking.ambiguous(64, 8, 4));
        assertTrue(IntPacking.ambiguous(16, 2, 3));
    }

    @Test
    public void bitStreamValuesStraddleWords() {
        int[] values = new int[32];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i * 5 + 3) & 7;
        }
        int[] words = IntSchemeFixtures.packBitStream(values, 3);
        assertEquals(3, words.length);
        for (int e = 0; e < values.length; e++) {
            assertEquals("element " + e, values[e], IntPacking.element(words, 0, 1, Layout.BIT_STREAM, 3, e));
        }
        assertEquals(values[10], IntPacking.element(new int[]{words[0], words[1]}, 0, 1, Layout.BIT_STREAM, 3, 10));
        assertEquals(values[20], IntPacking.element(new int[]{words[1], words[2]}, 0, 1, Layout.BIT_STREAM, 3, 20, 1));
    }

    @Test
    public void stridedPerWordElements() {
        int[] interleaved = {0x76543210, 0x11111111, 0xFEDCBA98, 0x22222222};
        for (int e = 0; e < 16; e++) {
            assertEquals(e, IntPacking.element(interleaved, 0, 2, Layout.PER_WORD, 4, e));
        }
    }

    @Test
    public void awqOrderAndItsInverse() {
        int[] columns = {0, 2, 4, 6, 1, 3, 5, 7};
        for (int s = 0; s < 8; s++) {
            assertEquals(columns[s], IntPacking.awqColumn(s));
            assertEquals(s, IntPacking.awqSlot(IntPacking.awqColumn(s)));
        }
    }

    @Test
    public void stableArgsortKeepsTiesInIndexOrder() {
        int[] keys = {1, 0, 1, 0, 2, 0};
        assertArrayEquals(new int[]{1, 3, 5, 0, 2, 4}, IntPacking.stableArgsort(keys, 3));
        assertThrows(IllegalArgumentException.class, () -> IntPacking.stableArgsort(new int[]{0, 3}, 3));
        assertThrows(IllegalArgumentException.class, () -> IntPacking.stableArgsort(new int[]{-1}, 3));
    }

    @Test
    public void fp16Representability() {
        assertTrue(IntPacking.fp16Exact(0.125f));
        assertTrue(IntPacking.fp16Exact(-65504f));
        assertFalse(IntPacking.fp16Exact(0.1f));
        assertFalse(IntPacking.fp16Exact(1e-9f));
    }

    @Test
    public void q4AndQ5EncodersRoundTrip() {
        int[] v = ramp(32, -8, 16);
        float[] out = decode(GGMLType.Q4_0, b -> IntPacking.putQ4_0(b, 0, 0.5f, v, 0));
        assertArrayEquals(toFloat(v, 0.5f, 0), out, 0f);

        int[] q = ramp(32, 0, 16);
        out = decode(GGMLType.Q4_1, b -> IntPacking.putQ4_1(b, 0, 0.25f, -1.5f, q, 0));
        assertArrayEquals(toFloat(q, 0.25f, -1.5f), out, 0f);

        int[] v5 = ramp(32, -16, 32);
        out = decode(GGMLType.Q5_0, b -> IntPacking.putQ5_0(b, 0, 2f, v5, 0));
        assertArrayEquals(toFloat(v5, 2f, 0), out, 0f);

        int[] q5 = ramp(32, 0, 32);
        out = decode(GGMLType.Q5_1, b -> IntPacking.putQ5_1(b, 0, 0.5f, 3f, q5, 0));
        assertArrayEquals(toFloat(q5, 0.5f, 3f), out, 0f);
    }

    @Test
    public void q8AndI8B128EncodersRoundTrip() {
        int[] v = ramp(32, -128, 256);
        float[] out = decode(GGMLType.Q8_0, b -> IntPacking.putQ8_0(b, 0, 0.0625f, v, 0));
        assertArrayEquals(toFloat(v, 0.0625f, 0), out, 0f);

        int[] w = new int[128];
        for (int i = 0; i < w.length; i++) {
            w[i] = i * 2 - 128;
        }
        float d = 0.1f;
        out = decode(GGMLType.I8_B128, b -> IntPacking.putI8B128(b, 0, d, w, 0));
        for (int i = 0; i < w.length; i++) {
            assertEquals(w[i] * d, out[i], 0f);
        }
    }

    @Test
    public void q2AndTq2EncodersRoundTrip() {
        int[] q = new int[64];
        for (int i = 0; i < q.length; i++) {
            q[i] = (i * 7 + 1) & 3;
        }
        float[] out = decode(GGMLType.Q2_0, b -> IntPacking.putQ2_0(b, 0, -0.5f, q, 0));
        for (int i = 0; i < q.length; i++) {
            assertEquals((q[i] - 1) * -0.5f, out[i], 0f);
        }
        int[] t = new int[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = (i * 5 + i / 37) % 3;
        }
        out = decode(GGMLType.TQ2_0, b -> IntPacking.putTq2_0(b, 0, 1f, t, 0));
        for (int i = 0; i < t.length; i++) {
            assertEquals("element " + i, t[i] - 1, out[i], 0f);
        }
    }

    @Test
    public void floatWritersRoundToNearestEven() {
        ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        IntPacking.putFloat(b, 0, 1, GGMLType.F16, 0.1f);
        assertEquals(FpBlockDecoders.roundToFp16(0.1f), FpBlockDecoders.fp16ToFp32(b.getShort(2)), 0f);
        IntPacking.putFloat(b, 0, 1, GGMLType.F32, 0.1f);
        assertEquals(0.1f, b.getFloat(4), 0f);
        assertThrows(IllegalArgumentException.class, () -> IntPacking.putFloat(b, 0, 0, GGMLType.Q8_0, 1f));
    }

    private interface Encoder {
        void write(ByteBuffer b);
    }

    private static float[] decode(GGMLType type, Encoder e) {
        ByteBuffer b = ByteBuffer.allocate(type.getTypeSize()).order(ByteOrder.LITTLE_ENDIAN);
        e.write(b);
        return CpuBlockDecoders.dequantize(b.array(), type, type.getBlockSize());
    }

    /** {@code n} values cycling through {@code [lo, lo + span)}. */
    private static int[] ramp(int n, int lo, int span) {
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = lo + (i * 7) % span;
        }
        return v;
    }

    private static float[] toFloat(int[] v, float d, float m) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] * d + m;
        }
        return out;
    }
}
