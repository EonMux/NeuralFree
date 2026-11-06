package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The grid-search IQ encoders (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M) through
 * {@link RowQuantizers}: bit-exact blocks against llama.cpp on the fixed inputs of
 * {@code iq-grid-parity.txt}, round trips through {@link CpuBlockDecoders} within per-type error bounds,
 * the importance weights steering the error, the expert-slice offsets, and the refusals by name.
 */
public class IqGridEncodersTest {

    private static final GGMLType[] TYPES = {GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
        GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ1_S, GGMLType.IQ1_M};

    private static final int N = 512;

    /** The LCG input generator of the parity fixture. */
    private static final class Lcg {
        private int s;

        Lcg(int seed) {
            s = seed;
        }

        float next() {
            s = s * 1664525 + 1013904223;
            return (float) (s >>> 8) / 16777216.0f;
        }
    }

    /** The fixture's row {@code x} and weights {@code w} of {@code seed} and zero case {@code z}. */
    private static float[][] fixtureInput(int seed, int z) {
        Lcg r = new Lcg(seed);
        float[] x = new float[N];
        float[] w = new float[N];
        for (int i = 0; i < N; i++) {
            float a = r.next();
            float b = r.next();
            float c = r.next();
            float v = (a + b + c - 1.5f) * 0.1f;
            x[i] = i % 61 == 7 ? v * 6.0f : v;
        }
        for (int i = 0; i < N; i++) {
            float a = r.next();
            float b = r.next();
            w[i] = a * b + 0.02f;
        }
        if (z == 1) {
            Arrays.fill(x, 224, 256, 0f);
            Arrays.fill(x, 480, 512, 0f);
        } else if (z == 2) {
            Arrays.fill(x, 64, 96, 0f);
        }
        return new float[][] {x, w};
    }

    private static byte[] encode(GGMLType type, float[] x, float[] w) {
        byte[] out = new byte[RowQuantizers.bytesPerRow(type, x.length)];
        Arrays.fill(out, (byte) 0xEE);
        RowQuantizers.quantizeRow(type, x, 0, out, 0, x.length, w);
        return out;
    }

    private static String hex(byte[] b) {
        StringBuilder s = new StringBuilder();
        for (byte v : b) {
            s.append(String.format("%02x", v & 0xFF));
        }
        return s.toString();
    }

    @Test
    public void blocksMatchLlamaCppBitForBit() throws IOException {
        int cases = 0;
        try (InputStream in = IqGridEncodersTest.class.getResourceAsStream("iq-grid-parity.txt")) {
            assertNotNull("parity fixture on the test classpath", in);
            BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            for (String line = r.readLine(); line != null; line = r.readLine()) {
                if (line.startsWith("#") || line.trim().isEmpty()) {
                    continue;
                }
                String[] f = line.trim().split(" ");
                GGMLType type = GGMLType.valueOf(f[0]);
                int z = Integer.parseInt(f[1].substring(1));
                int seed = (int) Long.parseLong(f[2].substring(1));
                float[][] input = fixtureInput(seed, z);
                float[] w = f[3].equals("w") ? input[1] : null;
                assertEquals(line.substring(0, 24), f[4], hex(encode(type, input[0], w)));
                cases++;
            }
        }
        assertEquals("every fixture line ran", 31, cases);
    }

    private static float relativeRmse(float[] x, float[] y, float[] w) {
        double err = 0;
        double sig = 0;
        for (int i = 0; i < x.length; i++) {
            double wi = w == null ? 1 : w[i];
            err += wi * (x[i] - y[i]) * (x[i] - y[i]);
            sig += wi * x[i] * x[i];
        }
        return (float) Math.sqrt(err / sig);
    }

    private static float[] gaussianRow(long seed, int n) {
        Random r = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) (r.nextGaussian() * 0.02);
        }
        return x;
    }

    private static float[] uniformWeights(int n) {
        float[] w = new float[n];
        Arrays.fill(w, 1f);
        return w;
    }

    /** Relative RMSE bounds per type over a Gaussian row; each type's bits per weight order them. */
    @Test
    public void roundTripsThroughTheDecoders() {
        float[] bounds = {0.42f, 0.36f, 0.32f, 0.24f, 0.20f, 0.62f, 0.55f};
        float[] x = gaussianRow(7, 4 * 256);
        for (int t = 0; t < TYPES.length; t++) {
            GGMLType type = TYPES[t];
            byte[] q = encode(type, x, uniformWeights(x.length));
            float[] back = CpuBlockDecoders.dequantize(q, type, x.length);
            float e = relativeRmse(x, back, null);
            assertTrue(type + " relative RMSE " + e + " exceeds " + bounds[t], e <= bounds[t]);
            assertTrue(type + " decodes to signal", e < 0.9f);
        }
    }

    @Test
    public void unweightedPathsRoundTrip() {
        float[] x = gaussianRow(8, 2 * 256);
        GGMLType[] types = {GGMLType.IQ2_S, GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ1_M};
        float[] bounds = {0.34f, 0.26f, 0.22f, 0.64f};
        for (int t = 0; t < types.length; t++) {
            byte[] q = encode(types[t], x, null);
            float e = relativeRmse(x, CpuBlockDecoders.dequantize(q, types[t], x.length), null);
            assertTrue(types[t] + " unweighted relative RMSE " + e, e <= bounds[t]);
        }
    }

    /** Weights that favour every fourth value pull the weighted error below the uniform one. */
    @Test
    public void importanceWeightsSteerTheError() {
        float[] x = gaussianRow(9, 4 * 256);
        float[] skew = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            skew[i] = i % 4 == 0 ? 50f : 0.02f;
        }
        for (GGMLType type : TYPES) {
            float skewed = relativeRmse(x, CpuBlockDecoders.dequantize(encode(type, x, skew), type, x.length), skew);
            float flat = relativeRmse(x, CpuBlockDecoders.dequantize(
                    encode(type, x, uniformWeights(x.length)), type, x.length), skew);
            assertTrue(type + " weighted error " + skewed + " not below uniform " + flat, skewed < flat);
        }
    }

    /** A row read at an offset with its weights at an offset equals the same row read from zero. */
    @Test
    public void offsetsSelectTheRowAndItsExpertSlice() {
        float[] x = gaussianRow(10, 256);
        float[] w = new float[256];
        for (int i = 0; i < w.length; i++) {
            w[i] = 0.1f + (i % 13) * 0.3f;
        }
        float[] shifted = new float[256 + 256];
        System.arraycopy(x, 0, shifted, 256, 256);
        float[] experts = new float[3 * 256];
        System.arraycopy(w, 0, experts, 2 * 256, 256);
        for (GGMLType type : TYPES) {
            int bytes = RowQuantizers.bytesPerRow(type, 256);
            byte[] direct = encode(type, x, w);
            byte[] viaOffsets = new byte[bytes + 5];
            RowQuantizers.quantizeRow(type, shifted, 256, viaOffsets, 5, 256, experts, 2 * 256);
            assertArrayEquals(type.name(), direct, Arrays.copyOfRange(viaOffsets, 5, 5 + bytes));
        }
    }

    @Test
    public void zeroRowEncodesToZero() {
        for (GGMLType type : TYPES) {
            float[] x = new float[256];
            byte[] q = encode(type, x, uniformWeights(256));
            float[] back = CpuBlockDecoders.dequantize(q, type, 256);
            for (float v : back) {
                assertEquals(type.name(), 0f, v, 0f);
            }
        }
    }

    /**
     * An all-zero IQ3_S sub-block keeps the grid bytes and signs of the sub-blocks after it at their own
     * offsets: the later sub-blocks decode as close to their values as the earlier ones.
     */
    @Test
    public void iq3SZeroSubBlockKeepsTheLaterSubBlocksInPlace() {
        float[] x = gaussianRow(11, 256);
        Arrays.fill(x, 64, 96, 0f);
        float[] back = CpuBlockDecoders.dequantize(encode(GGMLType.IQ3_S, x, null), GGMLType.IQ3_S, 256);
        float[] later = Arrays.copyOfRange(x, 96, 256);
        float e = relativeRmse(later, Arrays.copyOfRange(back, 96, 256), null);
        assertTrue("IQ3_S sub-blocks after a zero sub-block relative RMSE " + e, e <= 0.25f);
    }

    @Test
    public void typesThatNeedWeightsRefuseWithoutThem() {
        for (GGMLType type : new GGMLType[] {GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ1_S}) {
            try {
                encode(type, gaussianRow(12, 256), null);
                fail(type + " without weights must be refused");
            } catch (IllegalArgumentException e) {
                assertTrue(e.getMessage(), e.getMessage().contains(type.name()));
                assertTrue(e.getMessage(), e.getMessage().contains("importance weights"));
            }
        }
    }

    @Test
    public void partialSuperBlockRefuses() {
        for (GGMLType type : TYPES) {
            try {
                RowQuantizers.quantizeRow(type, new float[128], 0, new byte[256], 0, 128, uniformWeights(128));
                fail(type + " must refuse a 128-value row");
            } catch (IllegalArgumentException e) {
                assertTrue(e.getMessage(), e.getMessage().contains("256"));
            }
        }
    }
}
