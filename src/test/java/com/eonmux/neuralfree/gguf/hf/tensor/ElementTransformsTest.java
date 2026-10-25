package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.bytes;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.direct;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.q4Rows;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.seq;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.values;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.withSiblings;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/**
 * {@link ElementTransforms}: element functions, reshapes, row and column scaling, dtype conversion from every source
 * kind, the block-scale fast paths (exact on Q8_0 and I8_B128, power-of-two on NVFP4), ternary requantization and
 * low-rank merges, against hand-computed values.
 */
public class ElementTransformsTest {

    private static RowSource f32(long[] shape, float... v) {
        return RopeTensors.fromFloats(shape, v);
    }

    /** One I8_B128 row of 128 codes {@code j - 64} with block scale {@code d}. */
    private static RowSource i8b128(float d) {
        ByteBuffer b = ByteBuffer.allocate(132).order(ByteOrder.LITTLE_ENDIAN);
        b.putFloat(d);
        for (int j = 0; j < 128; j++) {
            b.put((byte) (j - 64));
        }
        return RopeTensors.fromBytes(ElementFormat.of(GGMLType.I8_B128), new long[]{1, 128}, b.array());
    }

    /** One NVFP4 row of 64 elements with the four given sub-scale bytes and codes {@code 0x21}. */
    private static RowSource nvfp4(int... subScales) {
        byte[] block = new byte[36];
        for (int s = 0; s < 4; s++) {
            block[s] = (byte) subScales[s];
        }
        Arrays.fill(block, 4, 36, (byte) 0x21);
        return RopeTensors.fromBytes(ElementFormat.of(GGMLType.NVFP4), new long[]{1, 64}, block);
    }

    // ------------------------------------------------------------------ element functions and scaling

    @Test
    public void elementFunctions() throws IOException {
        RowSource x = seq(1, 3);
        float e = (float) Math.E;
        assertArrayEquals(new float[]{-1f, -e, (float) -Math.exp(2)}, values(ElementTransforms.negExp(x)), 1e-6f);
        assertArrayEquals(new float[]{1f, e, (float) Math.exp(2)}, values(ElementTransforms.exp(x)), 1e-6f);
        assertArrayEquals(new float[]{1, 3, 5}, values(ElementTransforms.affine(x, 2f, 1f)), 0f);
        assertArrayEquals(new float[]{0.5f, 1.5f, 2.5f}, values(ElementTransforms.addScalar(x, 0.5f)), 0f);
        RowSource bf = ElementTransforms.convert(x, GGMLType.BF16);
        RowSource tripled = ElementTransforms.mulScalar(bf, 3f);
        assertEquals(ElementFormat.of(GGMLType.BF16), tripled.format());
        assertArrayEquals(new float[]{0, 3, 6}, values(tripled), 0f);
    }

    @Test
    public void opsOnBf16ComputeInF32AndRoundOnceIntoThePlannedType() throws IOException {
        float w = 1f / 256;
        RowSource bf = ElementTransforms.convert(f32(new long[]{1}, w), GGMLType.BF16);
        RowSource shifted = ElementTransforms.addScalar(bf, 1f);
        assertEquals("kept in the source's BF16", ElementFormat.of(GGMLType.BF16), shifted.format());
        assertArrayEquals("1 + 2^-8 rounds to 1 in BF16", new float[]{1f}, values(shifted), 0f);
        assertArrayEquals("the f32 result rounded once", new float[]{1f + w},
                values(ElementTransforms.convert(shifted, GGMLType.F32)), 0f);
        assertArrayEquals("1 + 2^-8 is exact in F16", new float[]{1f + w},
                values(ElementTransforms.convert(shifted, GGMLType.F16)), 0f);
        assertArrayEquals("chained ops read the f32 values", new float[]{w},
                values(ElementTransforms.addScalar(shifted, -1f)), 0f);
        assertArrayEquals("a dequantization reads the f32 values", new float[]{1f + w},
                values(ElementTransforms.dequant(shifted, GGMLType.F32)), 0f);

        float h = 1f / 2048;
        RowSource f16 = ElementTransforms.convert(f32(new long[]{1}, h), GGMLType.F16);
        assertArrayEquals("an F16 tensor is computed in F16, as the converter hands it over", new float[]{1f},
                values(ElementTransforms.convert(ElementTransforms.addScalar(f16, 1f), GGMLType.F32)), 0f);
    }

    @Test
    public void reshapeAndSqueezeKeepTheBytes() throws IOException {
        RowSource x = direct(seq(2, 4, 3), 4096);
        RowSource r = ElementTransforms.reshape(x, 8, 3);
        assertArrayEquals(new long[]{8, 3}, r.shape());
        assertEquals(4096, r.directVirtualOffset());
        assertArrayEquals(values(seq(2, 4, 3)), values(r), 0f);
        RowSource wide = ElementTransforms.reshape(x, -1, 6);
        assertArrayEquals(new long[]{4, 6}, wide.shape());
        assertArrayEquals(values(seq(2, 4, 3)), values(wide), 0f);
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.reshape(x, 5, 5));
        RowSource sq = ElementTransforms.squeezeConv(seq(4, 1, 3));
        assertArrayEquals(new long[]{4, 3}, sq.shape());
        assertArrayEquals(values(seq(4, 3)), values(sq), 0f);
    }

    @Test
    public void rowAndColumnScaling() throws IOException {
        RowSource rows = ElementTransforms.mulRowsBy(seq(2, 2, 2), new float[]{1f, -2f});
        assertArrayEquals(new float[]{0, 1, -4, -6, 4, 5, -12, -14}, values(rows), 0f);
        assertArrayEquals(new float[]{0, 0, -2, 3, 0, -5},
                values(ElementTransforms.mulColsBy(seq(2, 3), new float[]{1f, 0f, -1f})), 0f);
        assertArrayEquals(new float[]{0, 2, 6}, values(ElementTransforms.mulRowsBy(seq(3), new float[]{1, 2, 3})), 0f);
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.mulRowsBy(seq(2, 3), new float[3]));
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.mulColsBy(seq(2, 3), new float[2]));
    }

    @Test
    public void mulScalarPrefersSiblingAndBlockScales() throws IOException {
        RowSource f8 = RopeTensors.fromBytes(ElementFormat.of(GGMLType.F8_E4M3), new long[]{1, 32}, new byte[32]);
        RowSource scaledF8 = ElementTransforms.mulScalar(f8, 0.25f);
        assertEquals(ElementFormat.of(GGMLType.F8_E4M3), scaledF8.format());
        assertArrayEquals(new float[]{0.25f}, scaledF8.outputScale(), 0f);
        RowSource withScale = withSiblings(seq(1, 2), new float[]{3f}, null, null, null, null);
        assertArrayEquals(new float[]{6f}, ElementTransforms.mulScalar(withScale, 2f).outputScale(), 0f);
        RowSource q4 = ElementTransforms.mulScalar(q4Rows(1), 4f);
        assertEquals(ElementFormat.of(GGMLType.Q4_0), q4.format());
        float[] in = values(q4Rows(1));
        float[] out = values(q4);
        for (int i = 0; i < in.length; i++) {
            assertEquals(4f * in[i], out[i], 0f);
        }
    }

    @Test
    public void nvfp4SubScalesScaleByPowersOfTwoOnly() throws IOException {
        RowSource src = nvfp4(0x38, 0x39, 0x30, 0x00);
        RowSource doubled = ElementTransforms.mulScalar(src, 2f);
        assertEquals(ElementFormat.of(GGMLType.NVFP4), doubled.format());
        byte[] b = bytes(doubled);
        assertArrayEquals(new byte[]{0x40, 0x41, 0x38, 0x00}, Arrays.copyOf(b, 4));
        float[] in = values(src);
        float[] out = values(doubled);
        for (int i = 0; i < in.length; i++) {
            assertEquals(2f * in[i], out[i], 0f);
        }
        assertThrows(IOException.class, () -> bytes(ElementTransforms.mulScalar(nvfp4(0x78, 0x38, 0x38, 0x38), 2f)));
        assertEquals(ElementFormat.of(GGMLType.F32), ElementTransforms.mulScalar(src, 3f).format());

        RowSource sub = nvfp4(0x04, 0x08, 0x38, 0x00);
        assertArrayEquals(new byte[]{0x08, 0x10, 0x40, 0x00},
                Arrays.copyOf(bytes(ElementTransforms.mulScalar(sub, 2f)), 4));
        assertArrayEquals(new byte[]{0x02, 0x04, 0x30, 0x00},
                Arrays.copyOf(bytes(ElementTransforms.mulScalar(sub, 0.5f)), 4));
        assertThrows(IOException.class, () -> bytes(ElementTransforms.mulScalar(nvfp4(0x01, 0x38, 0x38, 0x38), 0.5f)));
    }

    // ------------------------------------------------------------------ scale folding

    @Test
    public void foldScaleIsExactOnQ8() throws IOException {
        RowSource q8 = ElementTransforms.convert(seq(2, 32), GGMLType.Q8_0);
        RowSource scaled = withSiblings(q8, new float[]{2f}, null, null, null, null);
        RowSource folded = ElementTransforms.foldScale(scaled);
        assertEquals(ElementFormat.of(GGMLType.Q8_0), folded.format());
        assertNull(folded.outputScale());
        byte[] in = bytes(q8);
        byte[] out = bytes(folded);
        for (int r = 0; r < 2; r++) {
            int o = r * 34;
            float d = FpBlockDecoders.fp16ToFp32((in[o] & 0xFF) | (in[o + 1] & 0xFF) << 8);
            float d2 = FpBlockDecoders.fp16ToFp32((out[o] & 0xFF) | (out[o + 1] & 0xFF) << 8);
            assertEquals(2f * d, d2, 0f);
            assertArrayEquals(Arrays.copyOfRange(in, o + 2, o + 34), Arrays.copyOfRange(out, o + 2, o + 34));
        }
        float[] vin = values(q8);
        float[] vout = values(folded);
        for (int i = 0; i < vin.length; i++) {
            assertEquals(2f * vin[i], vout[i], 0f);
        }
    }

    @Test
    public void foldScaleIsExactOnI8B128() throws IOException {
        RowSource src = withSiblings(i8b128(0.3f), new float[]{0.5f}, null, null, null, null);
        RowSource folded = ElementTransforms.foldScale(src);
        assertEquals(ElementFormat.of(GGMLType.I8_B128), folded.format());
        ByteBuffer b = ByteBuffer.wrap(bytes(folded)).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0.3f * 0.5f, b.getFloat(0), 0f);
        float[] v = values(folded);
        for (int j = 0; j < 128; j++) {
            assertEquals((0.3f * 0.5f) * (j - 64), v[j], 0f);
        }
        RowSource expertScales = withSiblings(seq(2, 1, 2), new float[]{2f, 3f}, null, null, null, null);
        assertArrayEquals(new float[]{0, 2, 6, 9}, values(ElementTransforms.foldScale(expertScales)), 0f);
    }

    // ------------------------------------------------------------------ conversion

    @Test
    public void convertRoundsToNearestEvenAndQuantizesLikeTheReference() throws IOException {
        float third = 1f / 3f;
        RowSource x = f32(new long[]{1, 32}, filled(third));
        ByteBuffer h = ByteBuffer.wrap(bytes(ElementTransforms.convert(x, GGMLType.F16)))
                .order(ByteOrder.LITTLE_ENDIAN);
        assertEquals((short) FpBlockDecoders.fp32ToFp16(third), h.getShort(0));
        ByteBuffer bf = ByteBuffer.wrap(bytes(ElementTransforms.convert(x, GGMLType.BF16)))
                .order(ByteOrder.LITTLE_ENDIAN);
        assertEquals((short) HalfFloats.floatToBFloat16(third), bf.getShort(0));
        float[] ramp = values(seq(1, 32));
        byte[] want = new byte[34];
        KvBlockCodec.quantize(KvDType.Q8_0, ramp, 0, want, 0);
        assertArrayEquals(want, bytes(ElementTransforms.convert(seq(1, 32), GGMLType.Q8_0)));
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.convert(seq(1, 3), GGMLType.Q8_0));
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.convert(seq(1, 32), GGMLType.Q4_0));
    }

    @Test
    public void convertDecodesIntegerEightBitAndBlockSources() throws IOException {
        RowSource u8 = RopeTensors.fromBytes(ElementFormat.of(StDType.U8), new long[]{4},
                new byte[]{0, 1, (byte) 200, (byte) 255});
        assertArrayEquals(new float[]{0, 1, 200, 255}, values(ElementTransforms.convert(u8, GGMLType.F32)), 0f);
        RowSource f8 = RopeTensors.fromBytes(ElementFormat.of(StDType.F8_E4M3), new long[]{3},
                new byte[]{0x38, 0x40, (byte) 0xB8});
        assertArrayEquals(new float[]{1, 2, -1}, values(ElementTransforms.convert(f8, GGMLType.F32)), 0f);
        RowSource fnuz = RopeTensors.fromBytes(ElementFormat.of(StDType.F8_E4M3FNUZ), new long[]{2},
                new byte[]{0x40, (byte) 0xC8});
        assertArrayEquals(new float[]{1, -2}, values(ElementTransforms.convert(fnuz, GGMLType.F32)), 0f);
        RowSource f4 = RopeTensors.fromBytes(ElementFormat.of(StDType.F4), new long[]{2}, new byte[]{0x21});
        assertArrayEquals(new float[]{0.5f, 1f}, values(ElementTransforms.convert(f4, GGMLType.F32)), 0f);
        RowSource i32 = RopeTensors.fromInts(new long[]{2}, new int[]{-7, 9});
        assertArrayEquals(new float[]{-7, 9}, values(ElementTransforms.convert(i32, GGMLType.F16)), 0f);
        float[] q4 = values(ElementTransforms.dequant(q4Rows(1), GGMLType.F32));
        for (int j = 0; j < 16; j++) {
            assertEquals(j - 8, q4[j], 0f);
            assertEquals(-8, q4[j + 16], 0f);
        }
    }

    @Test
    public void keepIntegerWidensUnsignedTypes() throws IOException {
        RowSource u8 = RopeTensors.fromBytes(ElementFormat.of(StDType.U8), new long[]{2}, new byte[]{1, (byte) 250});
        RowSource kept = ElementTransforms.keepInteger(u8);
        assertEquals(ElementFormat.of(GGMLType.I16), kept.format());
        assertArrayEquals(new float[]{1, 250}, values(kept), 0f);
        RowSource i64 = RopeTensors.fromLongs(new long[]{2}, new long[]{5, -3});
        assertEquals(ElementFormat.of(GGMLType.I64), ElementTransforms.keepInteger(i64).format());
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.keepInteger(seq(2)));
    }

    // ------------------------------------------------------------------ ternary re-quantization and low-rank merges

    @Test
    public void ternaryRequantMatchesBitnetWeightQuant() throws IOException {
        float[] w = new float[256];
        for (int i = 0; i < w.length; i++) {
            w[i] = (i % 4) - 1.5f;
        }
        RowSource t = ElementTransforms.ternaryRequant(f32(new long[]{1, 256}, w));
        assertEquals(ElementFormat.of(GGMLType.TQ2_0), t.format());
        assertArrayEquals(new float[]{1f}, t.outputScale(), 0f);
        float[] q = values(t);
        float[] want = {-1, 0, 0, 1};
        for (int i = 0; i < 256; i++) {
            assertEquals(want[i % 4], q[i], 0f);
        }
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.ternaryRequant(seq(1, 128)));
    }

    @Test
    public void addLowRankMergesInTheBaseDtype() throws IOException {
        RowSource base = seq(2, 3);
        RowSource a = f32(new long[]{1, 3}, 1, 2, 3);
        RowSource b = f32(new long[]{2, 1}, 1, 2);
        RowSource merged = ElementTransforms.addLowRank(base, a, b, 0.5f);
        assertEquals(ElementFormat.of(GGMLType.F32), merged.format());
        assertArrayEquals(new float[]{0.5f, 2f, 3.5f, 4f, 6f, 8f}, values(merged), 0f);
        RowSource bfBase = ElementTransforms.convert(base, GGMLType.BF16);
        assertEquals(ElementFormat.of(GGMLType.BF16), ElementTransforms.addLowRank(bfBase, a, b, 0.5f).format());
        assertThrows(IllegalArgumentException.class, () -> ElementTransforms.addLowRank(base, b, a, 1f));
    }

    private static float[] filled(float v) {
        float[] out = new float[32];
        Arrays.fill(out, v);
        return out;
    }
}
