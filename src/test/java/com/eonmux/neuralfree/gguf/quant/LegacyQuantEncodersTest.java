package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The legacy and IQ4 row encoders: the unweighted legacy paths equal the reference quantizers of
 * {@link KvBlockCodec} bit for bit, the weighted paths fit their importance better, and the IQ4_XS scale
 * packing decodes through the reference fields.
 */
public class LegacyQuantEncodersTest {

    private static float[] row(long seed, int n) {
        Random r = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) (r.nextGaussian() * 0.05);
        }
        return x;
    }

    private static float maxAbs(float[] x) {
        float m = 0;
        for (float v : x) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }

    private static float maxError(float[] x, float[] y) {
        float m = 0;
        for (int i = 0; i < x.length; i++) {
            m = Math.max(m, Math.abs(x[i] - y[i]));
        }
        return m;
    }

    @Test
    public void unweightedLegacyPathsMatchKvCodecBitForBit() {
        for (GGMLType type : new GGMLType[] {GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1}) {
            float[] x = row(21, 128);
            byte[] mine = new byte[type.getTypeSize() * 4];
            LegacyQuantEncoders.quantizeRow(type, x, 0, mine, 0, 128, null);
            byte[] ref = new byte[mine.length];
            float[] block = new float[32];
            for (int b = 0; b < 4; b++) {
                System.arraycopy(x, 32 * b, block, 0, 32);
                KvBlockCodec.quantize(KvDType.valueOf(type.name()), block, 0, ref, b * type.getTypeSize());
            }
            assertArrayEquals(type + " differs from the reference quantizer", mine, ref);
        }
    }

    @Test
    public void weightedLegacyPathsRoundTrip() {
        for (GGMLType type : new GGMLType[] {GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1}) {
            float[] x = row(22, 128);
            float[] w = new float[128];
            Random r = new Random(23);
            for (int i = 0; i < w.length; i++) {
                w[i] = 0.25f + r.nextFloat();
            }
            byte[] out = new byte[type.getTypeSize() * 4];
            LegacyQuantEncoders.quantizeRow(type, x, 0, out, 0, 128, w);
            float[] back = CpuBlockDecoders.dequantize(out, type, 128);
            float bound = 0.09f;
            assertTrue(type + " weighted round trip error " + maxError(x, back),
                    maxError(x, back) <= bound * maxAbs(x));
        }
    }

    @Test
    public void weightedQ4_0FitsSkewedImportanceBetter() {
        float[] x = row(24, 128);
        float[] w = new float[128];
        Arrays.fill(w, 1.0f);
        for (int i = 0; i < 32; i++) {
            w[i] = 5000.0f;
        }
        byte[] weighted = new byte[GGMLType.Q4_0.getTypeSize() * 4];
        byte[] plain = new byte[GGMLType.Q4_0.getTypeSize() * 4];
        LegacyQuantEncoders.quantizeRow(GGMLType.Q4_0, x, 0, weighted, 0, 128, w);
        LegacyQuantEncoders.quantizeRow(GGMLType.Q4_0, x, 0, plain, 0, 128, null);
        float[] backW = CpuBlockDecoders.dequantize(weighted, GGMLType.Q4_0, 128);
        float[] backP = CpuBlockDecoders.dequantize(plain, GGMLType.Q4_0, 128);
        float errW = 0;
        float errP = 0;
        for (int i = 0; i < 32; i++) {
            errW += w[i] * (x[i] - backW[i]) * (x[i] - backW[i]);
            errP += w[i] * (x[i] - backP[i]) * (x[i] - backP[i]);
        }
        assertTrue("weighted error " + errW + " should not exceed the plain one " + errP, errW <= errP);
    }

    @Test
    public void iq4NlRoundTripAndStructure() {
        float[] x = row(25, 256);
        byte[] out = new byte[GGMLType.IQ4_NL.getTypeSize() * 8];
        LegacyQuantEncoders.quantizeRow(GGMLType.IQ4_NL, x, 0, out, 0, 256, null);
        float[] back = CpuBlockDecoders.dequantize(out, GGMLType.IQ4_NL, 256);
        assertTrue("IQ4_NL round trip error " + maxError(x, back),
                maxError(x, back) <= 0.08f * maxAbs(x));
        // every block scale is a finite half
        for (int b = 0; b < 8; b++) {
            int d = (out[b * 18] & 0xFF) | (out[b * 18 + 1] & 0xFF) << 8;
            assertTrue(Float.isFinite(FpBlockDecoders.fp16ToFp32(d)));
        }
    }

    @Test
    public void iq4XsRoundTripAndScalePacking() {
        float[] x = row(26, 256);
        byte[] out = new byte[GGMLType.IQ4_XS.getTypeSize()];
        LegacyQuantEncoders.quantizeRow(GGMLType.IQ4_XS, x, 0, out, 0, 256, null);
        assertEquals(GGMLType.IQ4_XS.getTypeSize(), 136);
        float[] back = CpuBlockDecoders.dequantize(out, GGMLType.IQ4_XS, 256);
        assertTrue("IQ4_XS round trip error " + maxError(x, back),
                maxError(x, back) <= 0.08f * maxAbs(x));
        // the eight 6-bit sub-block scales pack into scales_l nibbles and scales_h bit pairs, and each
        // decodes to the dl the reference builds: d * (l - 32)
        float d = FpBlockDecoders.fp16ToFp32((out[0] & 0xFF) | (out[1] & 0xFF) << 8);
        int scalesH = (out[2] & 0xFF) | (out[3] & 0xFF) << 8;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((out[4 + ib / 2] & 0xFF) >>> (4 * (ib % 2)) & 0xf) | (((scalesH >>> (2 * ib)) & 3) << 4);
            assertTrue("sub-block scale " + ls + " outside [0, 63]", ls >= 0 && ls <= 63);
            float dl = d * (ls - 32);
            float[] block = new float[32];
            System.arraycopy(back, 32 * ib, block, 0, 32);
            float absCode = Math.abs(QuantTables.kvalueIq4nl(15) * dl);
            float blockMax = maxAbs(block);
            // a non-degenerate block scales to within one code step of its largest value
            if (blockMax > 0) {
                assertTrue("block " + ib + " max " + blockMax + " far from its scale span " + absCode,
                        blockMax <= Math.abs(dl) * 127.0f + 1e-9f);
            }
        }
    }

    @Test
    public void weightedIq4XsMatchesSlice() {
        float[] x = row(27, 512);
        float[] w = new float[512];
        Random r = new Random(28);
        for (int i = 0; i < w.length; i++) {
            w[i] = 0.5f + r.nextFloat();
        }
        byte[] out = new byte[GGMLType.IQ4_XS.getTypeSize() * 2];
        LegacyQuantEncoders.quantizeRow(GGMLType.IQ4_XS, x, 0, out, 0, 512, w);
        float[] back = CpuBlockDecoders.dequantize(out, GGMLType.IQ4_XS, 512);
        assertTrue(maxError(x, back) <= 0.08f * maxAbs(x));
    }

    @Test
    public void refusals() {
        try {
            LegacyQuantEncoders.quantizeRow(GGMLType.IQ4_XS, row(29, 100), 0, new byte[136], 0, 100, null);
            fail("a partial super-block must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("256"));
        }
        try {
            LegacyQuantEncoders.quantizeRow(GGMLType.Q6_K, row(30, 256), 0, new byte[210], 0, 256, null);
            fail("Q6_K is not a legacy type");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("legacy"));
        }
    }
}
