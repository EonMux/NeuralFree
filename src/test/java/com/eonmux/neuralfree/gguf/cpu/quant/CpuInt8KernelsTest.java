package com.eonmux.neuralfree.gguf.cpu.quant;

import org.junit.Assume;
import org.junit.Test;

import java.util.Arrays;
import java.util.Properties;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The int8 primitives of the fused CPU paths: the Vector API kernels against their scalar reference bit for bit, the
 * scalar kernel against a long accumulator, the symmetric row quantizer of the int8 attention, and the property gate
 * of {@link CpuInt8Fusion}.
 */
public class CpuInt8KernelsTest {

    private static byte[] bytes(int n, long seed) {
        Random rnd = new Random(seed);
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            b[i] = (byte) (rnd.nextInt(256) - 128);
        }
        return b;
    }

    private static float[] floats(int n, long seed) {
        Random rnd = new Random(seed);
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = rnd.nextFloat() * 4f - 2f;
        }
        return f;
    }

    /** The exact dot of two byte slices in a long, the reference of every integer implementation. */
    private static long exactDot(byte[] a, int aOff, byte[] b, int bOff, int n) {
        long sum = 0;
        for (int i = 0; i < n; i++) {
            sum += (long) a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Test
    public void theScalarDotIsExactAtEveryLength() {
        for (int n = 0; n <= 130; n++) {
            byte[] a = bytes(n + 8, n);
            byte[] b = bytes(n + 8, 1000L + n);
            assertEquals("n=" + n, (int) exactDot(a, 3, b, 5, n),
                    ScalarInt8Kernels.INSTANCE.dot(a, 3, b, 5, n));
        }
        byte[] a = new byte[]{127, -128, 1, -1};
        byte[] b = new byte[]{127, -128, -1, 1};
        assertEquals(127 * 127 + 128 * 128 - 1 - 1, ScalarInt8Kernels.INSTANCE.dot(a, 0, b, 0, 4));
    }

    @Test
    public void theVectorDotEqualsTheScalarDotExactly() {
        Assume.assumeTrue("the Vector API engine is unavailable on this JVM", CpuSimd.available());
        CpuInt8Kernels vector = CpuSimd.int8();
        assertNotSame("the engine serves its Vector API kernels", ScalarInt8Kernels.INSTANCE, vector);
        assertTrue(vector.describe(), vector.describe().startsWith("vector/"));
        for (int n = 0; n <= 300; n++) {
            byte[] a = bytes(n + 11, n);
            byte[] b = bytes(n + 5, 500L + n);
            assertEquals("n=" + n, ScalarInt8Kernels.INSTANCE.dot(a, 4, b, 2, n), vector.dot(a, 4, b, 2, n));
        }
    }

    @Test
    public void theAccumulationsMatchTheScalarOrderExactly() {
        Assume.assumeTrue("the Vector API engine is unavailable on this JVM", CpuSimd.available());
        CpuInt8Kernels vector = CpuSimd.int8();
        for (int n = 0; n <= 300; n++) {
            byte[] codes = bytes(n + 9, 77L + n);
            float[] y = floats(n + 6, 91L + n);
            float[] scalarAcc = floats(n + 13, n);
            float[] vectorAcc = scalarAcc.clone();
            ScalarInt8Kernels.INSTANCE.accumulateScaled(codes, 3, 0.0625f, -1.75f, scalarAcc, 5, n);
            vector.accumulateScaled(codes, 3, 0.0625f, -1.75f, vectorAcc, 5, n);
            assertArrayEquals("scaled accumulation of " + n + " values", scalarAcc, vectorAcc, 0f);
            Arrays.fill(scalarAcc, 0.5f);
            Arrays.fill(vectorAcc, 0.5f);
            ScalarInt8Kernels.INSTANCE.accumulateWeighted(y, 2, 0.375f, scalarAcc, 7, n);
            vector.accumulateWeighted(y, 2, 0.375f, vectorAcc, 7, n);
            assertArrayEquals("weighted accumulation of " + n + " values", scalarAcc, vectorAcc, 0f);
        }
    }

    @Test
    public void theScalarAccumulationsAddTheDequantizedProduct() {
        byte[] codes = new byte[]{1, -2, 3, -4};
        float[] acc = new float[]{10f, 20f, 30f, 40f};
        ScalarInt8Kernels.INSTANCE.accumulateScaled(codes, 0, 0.5f, 2f, acc, 0, 4);
        assertArrayEquals(new float[]{10f + 1f, 20f - 2f, 30f + 3f, 40f - 4f}, acc, 0f);
        float[] y = new float[]{1f, 2f, 3f, 4f};
        float[] out = new float[4];
        ScalarInt8Kernels.INSTANCE.accumulateWeighted(y, 0, -0.5f, out, 0, 4);
        assertArrayEquals(new float[]{-0.5f, -1f, -1.5f, -2f}, out, 0f);
    }

    // ------------------------------------------------------------------ quantization

    @Test
    public void aRowQuantizesToItsAbsmaxScale() {
        float[] x = {0f, 1f, -0.5f, 0.25f};
        float scale = CpuInt8Quantizer.scale(x, 0, 4);
        assertEquals(1.0f / 127.0f, scale, 0f);
        byte[] codes = new byte[4];
        CpuInt8Quantizer.quantize(x, 0, 4, codes, 0, scale);
        assertArrayEquals(new byte[]{0, 127, -64, 32}, codes);
    }

    @Test
    public void theQuantizerRoundsHalfToEvenAndClamps() {
        byte[] codes = new byte[4];
        CpuInt8Quantizer.quantize(new float[]{0.5f, 1.5f, 2.5f, 400f}, 0, 4, codes, 0, 1f);
        assertArrayEquals("2.5 rounds to 2, 1.5 to 2, 0.5 to 0", new byte[]{0, 2, 2, 127}, codes);
        assertEquals(0, CpuInt8Quantizer.code(-0.5f, 1f));
        assertEquals(-127, CpuInt8Quantizer.code(-1000f, 1f));
    }

    @Test
    public void anEmptyRowOrNonPositiveScaleIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> CpuInt8Quantizer.scale(new float[3], 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> CpuInt8Quantizer.quantize(new float[]{1f}, 0, 1, new byte[1], 0, 0f));
    }

    @Test
    public void anExactMultipleOfItsScaleQuantizesWithoutError() {
        // codes over a power-of-two scale: the absmax is the 127th multiple, so the derived scale is exact
        float scale = 0.25f;
        float[] x = new float[8];
        x[0] = 127 * scale;
        x[1] = -64 * scale;
        x[2] = 31 * scale;
        float derived = CpuInt8Quantizer.scale(x, 0, 8);
        assertEquals(scale, derived, 0f);
        byte[] codes = new byte[8];
        CpuInt8Quantizer.quantize(x, 0, 8, codes, 0, derived);
        assertArrayEquals(new byte[]{127, -64, 31, 0, 0, 0, 0, 0}, codes);
    }

    // ------------------------------------------------------------------ selection

    @Test
    public void theFusedPathsAreOnUnlessThePropertySaysOtherwise() {
        Properties p = new Properties();
        assertTrue(CpuInt8Fusion.enabled(p));
        p.setProperty(CpuInt8Fusion.PROP_INT8, "false");
        assertFalse(CpuInt8Fusion.enabled(p));
        p.setProperty(CpuInt8Fusion.PROP_INT8, " TRUE ");
        assertTrue(CpuInt8Fusion.enabled(p));
        p.setProperty(CpuInt8Fusion.PROP_INT8, "sometimes");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> CpuInt8Fusion.enabled(p));
        assertTrue(e.getMessage(), e.getMessage().contains(CpuInt8Fusion.PROP_INT8));
        assertTrue(e.getMessage(), e.getMessage().contains("true or false"));
    }

    @Test
    public void theEngineServesOneKernelSet() {
        CpuInt8Kernels kernels = CpuSimd.int8();
        assertFalse(kernels.describe().isEmpty());
        assertEquals(kernels.describe(), CpuInt8Fusion.kernels().describe());
        if (CpuSimd.available()) {
            assertEquals("vector kernels when the engine is available", "vector", kernels.describe().split("/")[0]);
        } else {
            assertEquals(ScalarInt8Kernels.INSTANCE.describe(), kernels.describe());
        }
    }
}
