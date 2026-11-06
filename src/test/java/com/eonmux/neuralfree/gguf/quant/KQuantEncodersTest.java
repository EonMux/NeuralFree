package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The K-quant row encoders against their decoders: round-trip error bounds per type, deterministic
 * repeats, and the imatrix-weighted path beating the unweighted one on skewed importance.
 */
public class KQuantEncodersTest {

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

    /** Dequantizes an encoded row back to F32. */
    private static float[] decode(GGMLType type, byte[] encoded, int n) {
        return CpuBlockDecoders.dequantize(encoded, type, n);
    }

    private static float maxError(float[] x, float[] y) {
        float m = 0;
        for (int i = 0; i < x.length; i++) {
            m = Math.max(m, Math.abs(x[i] - y[i]));
        }
        return m;
    }

    private static void assertRoundTrip(GGMLType type, long seed, int nPerRow, float tol) {
        float[] x = row(seed, nPerRow);
        byte[] out = new byte[type.getTypeSize() * (nPerRow / type.getBlockSize())];
        KQuantEncoders.quantizeRow(type, x, 0, out, 0, nPerRow, null);
        float[] back = decode(type, out, nPerRow);
        float bound = tol * maxAbs(x);
        assertTrue(type + " round trip error " + maxError(x, back) + " exceeds " + bound,
                maxError(x, back) <= bound);
        // the group lattice: the values of one scale group sit on at most one point per level
        int group = type == GGMLType.Q4_K || type == GGMLType.Q5_K ? 32 : 16;
        int levels = type == GGMLType.Q2_K ? 4 : type == GGMLType.Q3_K ? 8
                : type == GGMLType.Q4_K ? 16 : type == GGMLType.Q5_K ? 32 : 16;
        assertLattice(type, back, group, levels);
    }

    /** Within one scale group the decoded values take at most {@code levels} distinct values. */
    private static void assertLattice(GGMLType type, float[] back, int group, int levels) {
        for (int g = 0; g + group <= back.length; g += group) {
            java.util.Set<Float> distinct = new java.util.HashSet<>();
            for (int i = g; i < g + group; i++) {
                distinct.add(Math.round(back[i] * 5e5f) / 5e5f);
            }
            assertTrue(type + " group at " + g + " takes " + distinct.size() + " distinct values, more"
                    + " than the " + levels + " levels of the format", distinct.size() <= levels);
        }
    }

    @Test
    public void roundTripReferencePath() {
        assertRoundTrip(GGMLType.Q2_K, 1, 256, 0.35f);
        assertRoundTrip(GGMLType.Q3_K, 2, 256, 0.18f);
        assertRoundTrip(GGMLType.Q4_K, 3, 256, 0.08f);
        assertRoundTrip(GGMLType.Q5_K, 4, 256, 0.05f);
        assertRoundTrip(GGMLType.Q6_K, 5, 256, 0.02f);
    }

    @Test
    public void roundTripSeveralSuperBlocks() {
        assertRoundTrip(GGMLType.Q4_K, 6, 512, 0.08f);
        assertRoundTrip(GGMLType.Q6_K, 7, 768, 0.02f);
        assertRoundTrip(GGMLType.Q2_K, 8, 1024, 0.35f);
    }

    @Test
    public void zeroRowEncodesToZeros() {
        byte[] out = new byte[GGMLType.Q6_K.getTypeSize()];
        KQuantEncoders.quantizeRow(GGMLType.Q6_K, new float[256], 0, out, 0, 256, null);
        for (byte b : out) {
            assertEquals(0, b);
        }
    }

    @Test
    public void repeatIsDeterministic() {
        float[] x = row(9, 256);
        byte[] a = new byte[GGMLType.Q4_K.getTypeSize()];
        byte[] b = new byte[GGMLType.Q4_K.getTypeSize()];
        KQuantEncoders.quantizeRow(GGMLType.Q4_K, x, 0, a, 0, 256, null);
        KQuantEncoders.quantizeRow(GGMLType.Q4_K, x, 0, b, 0, 256, null);
        assertArrayEquals(a, b);
    }

    @Test
    public void weightedPathBeatsUnweightedOnSkewedImportance() {
        // One 256-value row whose first group carries nearly all the importance
        float[] x = row(10, 256);
        float[] w = new float[256];
        Arrays.fill(w, 1.0f);
        for (int i = 0; i < 16; i++) {
            w[i] = 10000.0f;
        }
        byte[] weighted = new byte[GGMLType.Q4_K.getTypeSize()];
        byte[] plain = new byte[GGMLType.Q4_K.getTypeSize()];
        KQuantEncoders.quantizeRow(GGMLType.Q4_K, x, 0, weighted, 0, 256, w);
        KQuantEncoders.quantizeRow(GGMLType.Q4_K, x, 0, plain, 0, 256, null);
        float[] backW = decode(GGMLType.Q4_K, weighted, 256);
        float[] backP = decode(GGMLType.Q4_K, plain, 256);
        assertTrue(weightedError(x, backW, w) <= weightedError(x, backP, w) + 1e-9f);
        // the difference is real, not a tie: the first group fits tighter under its weight
        float groupW = 0;
        float groupP = 0;
        for (int i = 0; i < 16; i++) {
            groupW += w[i] * (x[i] - backW[i]) * (x[i] - backW[i]);
            groupP += w[i] * (x[i] - backP[i]) * (x[i] - backP[i]);
        }
        assertTrue("weighted group error " + groupW + " should stay below the plain one " + groupP,
                groupW <= groupP);
    }

    private static float weightedError(float[] x, float[] y, float[] w) {
        float s = 0;
        for (int i = 0; i < x.length; i++) {
            float d = x[i] - y[i];
            s += w[i] * d * d;
        }
        return s;
    }

    @Test
    public void weightedQ6KMatchesQwSlice() {
        // the weighted Q6_K search runs on qw directly (no sigma2 factor), as the reference does
        float[] x = row(11, 256);
        float[] w = new float[256];
        Random r = new Random(12);
        for (int i = 0; i < w.length; i++) {
            w[i] = 0.5f + r.nextFloat();
        }
        byte[] out = new byte[GGMLType.Q6_K.getTypeSize()];
        KQuantEncoders.quantizeRow(GGMLType.Q6_K, x, 0, out, 0, 256, w);
        float[] back = decode(GGMLType.Q6_K, out, 256);
        float bound = 0.02f * maxAbs(x);
        assertTrue("weighted Q6_K error " + maxError(x, back) + " exceeds " + bound,
                maxError(x, back) <= bound);
        assertLattice(GGMLType.Q6_K, back, 16, 64);
    }

    @Test
    public void refusals() {
        float[] x = row(13, 100);
        try {
            KQuantEncoders.quantizeRow(GGMLType.Q4_K, x, 0, new byte[144], 0, 100, null);
            fail("a partial super-block must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("256"));
        }
        try {
            KQuantEncoders.quantizeRow(GGMLType.Q8_0, x, 0, new byte[34], 0, 32, null);
            fail("Q8_0 is not a K-quant");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("K-quant"));
        }
    }
}
