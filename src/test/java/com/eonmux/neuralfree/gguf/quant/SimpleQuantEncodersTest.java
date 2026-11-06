package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The simple row encoders: Q1_0 sign bits, Q2_0 codes, MXFP4's E8M0 scale, the ternary values of TQ1_0
 * and TQ2_0, Q8_K's levels and sums, and the exact F16 / BF16 / F32 rows.
 */
public class SimpleQuantEncodersTest {

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
    public void q1_0SignBitsAndScale() {
        int n = 256;
        float[] x = row(31, n);
        byte[] out = new byte[GGMLType.Q1_0.getTypeSize() * 2];
        SimpleQuantEncoders.quantizeRow(GGMLType.Q1_0, x, 0, out, 0, n, null);
        float sumAbs = 0;
        for (int i = 0; i < 128; i++) {
            sumAbs += Math.abs(x[i]);
        }
        float d = FpBlockDecoders.fp16ToFp32((out[0] & 0xFF) | (out[1] & 0xFF) << 8);
        assertEquals(sumAbs / 128, d, 1e-3f * d + 1e-9f);
        float[] back = CpuBlockDecoders.dequantize(out, GGMLType.Q1_0, n);
        // every value decodes to +d or -d by the sign bit
        for (int i = 0; i < 128; i++) {
            assertEquals(Math.abs(back[i]), d, 1e-6f * Math.max(1, Math.abs(d)));
        }
    }

    @Test
    public void q2_0CodesRoundTrip() {
        int n = 256;
        float[] x = row(32, n);
        byte[] out = new byte[GGMLType.Q2_0.getTypeSize() * 4];
        SimpleQuantEncoders.quantizeRow(GGMLType.Q2_0, x, 0, out, 0, n, null);
        float[] back = CpuBlockDecoders.dequantize(out, GGMLType.Q2_0, n);
        assertTrue("Q2_0 round trip error " + maxError(x, back),
                maxError(x, back) <= 0.52f * maxAbs(x) + 1e-9f);
        // the scale of block 0 is the block's max |x|
        float amax = 0;
        for (int i = 0; i < 64; i++) {
            amax = Math.max(amax, Math.abs(x[i]));
        }
        float d = FpBlockDecoders.fp16ToFp32((out[0] & 0xFF) | (out[1] & 0xFF) << 8);
        assertEquals(amax, d, 1e-3f * amax + 1e-9f);
    }

    @Test
    public void mxfp4RoundTripAndExponent() {
        int n = 256;
        float[] x = row(33, n);
        byte[] out = new byte[GGMLType.MXFP4.getTypeSize() * 8];
        SimpleQuantEncoders.quantizeRow(GGMLType.MXFP4, x, 0, out, 0, n, null);
        float[] back = CpuBlockDecoders.dequantize(out, GGMLType.MXFP4, n);
        assertTrue("MXFP4 round trip error " + maxError(x, back),
                maxError(x, back) <= 0.26f * maxAbs(x) + 1e-9f);
        // e = floor(log2(amax)) - 2 + 127 of block 0, and the values are multiples of the group scale
        float amax = 0;
        for (int i = 0; i < 32; i++) {
            amax = Math.max(amax, Math.abs(x[i]));
        }
        int e = out[0] & 0xFF;
        assertEquals(Math.getExponent(amax) - 2 + 127, e);
        float d = FpBlockDecoders.e8m0ToFp32Half(e);
        for (int i = 0; i < 32; i++) {
            float steps = back[i] / d / 0.5f;
            assertEquals("value " + back[i] + " is not an E2M1 multiple of " + d,
                    Math.round(steps), steps, 1e-4f);
        }
    }

    @Test
    public void ternaryValuesAndScale() {
        int n = 256;
        for (GGMLType type : new GGMLType[] {GGMLType.TQ1_0, GGMLType.TQ2_0}) {
            float[] x = row(34, n);
            byte[] out = new byte[type.getTypeSize()];
            SimpleQuantEncoders.quantizeRow(type, x, 0, out, 0, n, null);
            float[] back = CpuBlockDecoders.dequantize(out, type, n);
            float d = FpBlockDecoders.fp16ToFp32(
                    type == GGMLType.TQ1_0 ? (out[52] & 0xFF) | (out[53] & 0xFF) << 8
                            : (out[64] & 0xFF) | (out[65] & 0xFF) << 8);
            float amax = maxAbs(x);
            assertEquals(type + " scale", amax, d, amax * 1e-3f + 1e-9f);
            for (int i = 0; i < n; i++) {
                float v = back[i] / d;
                int trit = Math.round(v);
                assertEquals(type + " value " + v + " is not a trit", trit, v, 1e-3f);
                assertTrue(trit >= -1 && trit <= 1);
            }
        }
    }

    @Test
    public void q8_KLevelsAndSums() {
        int n = 256;
        float[] x = row(35, n);
        byte[] out = new byte[GGMLType.Q8_K.getTypeSize()];
        SimpleQuantEncoders.quantizeRow(GGMLType.Q8_K, x, 0, out, 0, n, null);
        float d = java.nio.ByteBuffer.wrap(out).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .getFloat(com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_K_D);
        float[] back = CpuBlockDecoders.dequantize(out, GGMLType.Q8_K, n);
        assertTrue(maxError(x, back) <= 0.006f * maxAbs(x));
        // bsums[0] sums the first 16 levels
        int sum = 0;
        for (int i = 0; i < 16; i++) {
            sum += out[4 + i];
        }
        short bsum0 = java.nio.ByteBuffer.wrap(out).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .getShort(com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_K_BSUMS);
        assertEquals(sum, bsum0);
    }

    @Test
    public void floatRowsAreExact() {
        int n = 128;
        float[] x = row(36, n);
        byte[] f32 = new byte[n * 4];
        byte[] f16 = new byte[n * 2];
        byte[] bf16 = new byte[n * 2];
        SimpleQuantEncoders.quantizeRow(GGMLType.F32, x, 0, f32, 0, n, null);
        SimpleQuantEncoders.quantizeRow(GGMLType.F16, x, 0, f16, 0, n, null);
        SimpleQuantEncoders.quantizeRow(GGMLType.BF16, x, 0, bf16, 0, n, null);
        assertArrayEquals(x, CpuBlockDecoders.dequantize(f32, GGMLType.F32, n), 0.0f);
        // random floats are not representable: the F16 row keeps a half ulp, the BF16 row a half ulp
        // of its 8-bit mantissa
        assertTrue(maxError(x, CpuBlockDecoders.dequantize(f16, GGMLType.F16, n))
                <= 1e-3f * maxAbs(x) + 1e-9f);
        assertTrue(maxError(x, CpuBlockDecoders.dequantize(bf16, GGMLType.BF16, n))
                <= 5e-3f * maxAbs(x) + 1e-9f);
    }

    @Test
    public void refusals() {
        try {
            SimpleQuantEncoders.quantizeRow(GGMLType.MXFP4, row(37, 100), 0, new byte[17], 0, 100, null);
            fail("a partial block must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("32"));
        }
        try {
            SimpleQuantEncoders.quantizeRow(GGMLType.Q4_K, row(38, 256), 0, new byte[144], 0, 256, null);
            fail("Q4_K is not a simple type");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("simple"));
        }
    }
}
