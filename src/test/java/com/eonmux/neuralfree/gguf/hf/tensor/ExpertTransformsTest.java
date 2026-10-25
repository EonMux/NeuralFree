package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.bytes;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.seq;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.values;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.withSiblings;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/**
 * {@link ExpertTransforms}: stacking with output-scale merging, fused-expert splits, and {@code transposeLast2} on
 * element formats (the gpt-oss BF16 gate/up layout), on block-128 FP8 (exact re-blocking) and on other block formats
 * (F32 fallback).
 */
public class ExpertTransformsTest {

    @Test
    public void stackAddsAnOuterDimension() throws IOException {
        RowSource s = ExpertTransforms.stack(Arrays.asList(seq(2, 3), seq(2, 3)));
        assertArrayEquals(new long[]{2, 2, 3}, s.shape());
        float[] v = values(s);
        assertArrayEquals(Arrays.copyOfRange(v, 0, 6), Arrays.copyOfRange(v, 6, 12), 0f);
        assertThrows(IllegalArgumentException.class, () -> ExpertTransforms.stack(Arrays.asList(seq(2, 3), seq(3, 2))));
    }

    @Test
    public void stackMergesPerTensorScalesIntoPerExpertScales() throws IOException {
        RowSource q = ElementTransforms.convert(seq(1, 32), GGMLType.Q8_0);
        RowSource a = withSiblings(q, new float[]{2f}, new float[]{0.5f}, null, null, ActQuantScheme.FP8_STATIC);
        RowSource b = withSiblings(q, new float[]{3f}, new float[]{0.25f}, null, null, ActQuantScheme.FP8_STATIC);
        RowSource s = ExpertTransforms.stack(Arrays.asList(a, b));
        assertEquals(ElementFormat.of(GGMLType.Q8_0), s.format());
        assertArrayEquals(new float[]{2f, 3f}, s.outputScale(), 0f);
        assertArrayEquals(new float[]{0.5f, 0.25f}, s.inputScale(), 0f);
        assertEquals(ActQuantScheme.FP8_STATIC, s.actQuant());
        byte[] one = bytes(q);
        byte[] both = bytes(s);
        assertArrayEquals(one, Arrays.copyOfRange(both, 0, 34));
        assertArrayEquals(one, Arrays.copyOfRange(both, 34, 68));

        RowSource mixed = ExpertTransforms.stack(Arrays.asList(a, q));
        assertNull(mixed.outputScale());
        assertNull(mixed.actQuant());
        float[] base = values(q);
        float[] got = values(mixed);
        for (int i = 0; i < 32; i++) {
            assertEquals(2f * base[i], got[i], 0f);
            assertEquals(base[i], got[32 + i], 0f);
        }
    }

    @Test
    public void splitExpertsAlongTheFusedDimension() throws IOException {
        RowSource[] parts = ExpertTransforms.splitExperts(seq(2, 4, 3), -2, 2);
        assertArrayEquals(new long[]{2, 2, 3}, parts[1].shape());
        assertArrayEquals(new float[]{0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17}, values(parts[0]), 0f);
        assertArrayEquals(new float[]{6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23}, values(parts[1]), 0f);
        assertThrows(IllegalArgumentException.class, () -> ExpertTransforms.splitExperts(seq(2, 3, 3), 1, 2));
    }

    @Test
    public void transposeLast2OfElements() throws IOException {
        RowSource t = ExpertTransforms.transposeLast2(seq(2, 2, 3));
        assertArrayEquals(new long[]{2, 3, 2}, t.shape());
        assertFalse(t.rowLocal());
        assertArrayEquals(new float[]{0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11}, values(t), 0f);
        RowSource act = withSiblings(seq(2, 2), new float[]{2f}, new float[]{1f}, null, null,
                ActQuantScheme.FP8_STATIC);
        RowSource ta = ExpertTransforms.transposeLast2(act);
        assertArrayEquals(new float[]{2f}, ta.outputScale(), 0f);
        assertNull(ta.actQuant());
        assertNull(ta.inputScale());
    }

    @Test
    public void gptOssGateUpMatchesTheBf16Reference() throws IOException {
        RowSource x = ElementTransforms.convert(seq(2, 4, 6), GGMLType.BF16);
        float[] xv = values(x);
        for (int part = 0; part < 2; part++) {
            RowSource got = ExpertTransforms.transposeLast2(ColumnTransforms.interleaveCols(x, 2, part));
            assertEquals(ElementFormat.of(GGMLType.BF16), got.format());
            assertArrayEquals(new long[]{2, 3, 4}, got.shape());
            float[] want = new float[24];
            for (int e = 0; e < 2; e++) {
                for (int f = 0; f < 3; f++) {
                    for (int d = 0; d < 4; d++) {
                        want[(e * 3 + f) * 4 + d] = xv[(e * 4 + d) * 6 + 2 * f + part];
                    }
                }
            }
            assertArrayEquals(want, values(got), 0f);
        }
        RowSource bias = ElementTransforms.convert(seq(2, 6), GGMLType.BF16);
        assertArrayEquals(new float[]{1, 3, 5, 7, 9, 11}, values(ColumnTransforms.interleaveCols(bias, 2, 1)), 0f);
    }

    /** A 128 x 128 F8_E4M3_B128 matrix with block scale {@code d} except row {@code oddRow} ({@code -1}: none). */
    private static RowSource fp8b128(float d, int oddRow) {
        ByteBuffer b = ByteBuffer.allocate(128 * 132).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < 128; r++) {
            b.putFloat(r == oddRow ? 2 * d : d);
            for (int c = 0; c < 128; c++) {
                b.put((byte) ((r * 3 + c) & 0x77));
            }
        }
        return RopeTensors.fromBytes(ElementFormat.of(GGMLType.F8_E4M3_B128), new long[]{128, 128}, b.array());
    }

    @Test
    public void block128Fp8IsReblockedExactly() throws IOException {
        RowSource src = fp8b128(0.5f, -1);
        RowSource t = ExpertTransforms.transposeLast2(src);
        assertEquals(ElementFormat.of(GGMLType.F8_E4M3_B128), t.format());
        float[] in = values(src);
        float[] out = values(t);
        for (int r = 0; r < 128; r++) {
            for (int c = 0; c < 128; c++) {
                assertEquals(in[r * 128 + c], out[c * 128 + r], 0f);
            }
        }
        assertThrows(IOException.class, () -> bytes(ExpertTransforms.transposeLast2(fp8b128(0.5f, 5))));
    }

    @Test
    public void perTensorFp8CodesTransposeByteForByte() throws IOException {
        byte[] codes = new byte[32 * 64];
        for (int r = 0; r < 32; r++) {
            for (int c = 0; c < 64; c++) {
                codes[r * 64 + c] = (byte) ((r * 5 + c) & 0x77);
            }
        }
        RowSource f8 = withSiblings(RopeTensors.fromBytes(ElementFormat.of(GGMLType.F8_E4M3), new long[]{32, 64},
                codes), new float[]{0.5f}, null, null, null, null);
        RowSource t = ExpertTransforms.transposeLast2(f8);
        assertEquals(ElementFormat.of(GGMLType.F8_E4M3), t.format());
        assertArrayEquals(new long[]{64, 32}, t.shape());
        assertArrayEquals(new float[]{0.5f}, t.outputScale(), 0f);
        byte[] out = bytes(t);
        for (int r = 0; r < 32; r++) {
            for (int c = 0; c < 64; c++) {
                assertEquals(codes[r * 64 + c], out[c * 32 + r]);
            }
        }
        RowSource narrow = RopeTensors.fromBytes(ElementFormat.of(GGMLType.F8_E4M3), new long[]{2, 32}, new byte[64]);
        assertEquals(ElementFormat.of(GGMLType.F32), ExpertTransforms.transposeLast2(narrow).format());
    }

    @Test
    public void otherBlockFormatsTransposeAsF32() throws IOException {
        RowSource q8 = ElementTransforms.convert(seq(2, 32), GGMLType.Q8_0);
        RowSource t = ExpertTransforms.transposeLast2(q8);
        assertEquals(ElementFormat.of(GGMLType.F32), t.format());
        float[] in = values(q8);
        float[] out = values(t);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 32; c++) {
                assertEquals(in[r * 32 + c], out[c * 2 + r], 0f);
            }
        }
    }
}
