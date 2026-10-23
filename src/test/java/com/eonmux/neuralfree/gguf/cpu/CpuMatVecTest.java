package com.eonmux.neuralfree.gguf.cpu;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.Fp16VecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.Fp4VecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.IQuantVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.KQuantVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.LegacyVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.NewFpVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.TernaryVecDot;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link CpuMatVec}: agreement with the type's vec-dot kernel on activations quantized once, closeness to
 * the decoded F32 product, the multi-token, row-block and expert views, accumulation, the output scale and PRISM
 * input transforms, argument validation, and the single kernel table built from the eight vec-dot families.
 */
public class CpuMatVecTest {

    /** F16 bit patterns of small block scales: 1/16, 1/8, 1/4, -1/8, -1/16. */
    private static final short[] HALF_SCALES = {0x2C00, 0x3000, 0x3400, (short) 0xB000, (short) 0xAC00};

    // ------------------------------------------------------------------ fixtures

    /**
     * Tolerance for results the same kernel computes along different call paths ({@code dotRow} versus a
     * {@code dotRows} override that hoists per-call work): float rounding only.
     */
    private static float tolerance(float v) {
        return 1e-5f * Math.max(1.0f, Math.abs(v));
    }

    private static void assertNear(float[] expected, float[] actual) {
        assertNear("", expected, actual);
    }

    private static void assertNear(String what, float[] expected, float[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + " [" + i + "]", expected[i], actual[i], tolerance(expected[i]));
        }
    }

    private static float[] randomVector(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2.0f - 1.0f;
        }
        return v;
    }

    private static ByteBuffer le(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static void putHalf(byte[] b, int at, short half) {
        b[at] = (byte) half;
        b[at + 1] = (byte) (half >> 8);
    }

    private static CpuTensorRef f32(float[] values, int rows, int cols) {
        ByteBuffer buf = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) {
            buf.putFloat(i * 4, values[i]);
        }
        return CpuTensorRef.of(GGMLType.F32, buf, rows, cols);
    }

    /** Q8_0 rows: per 32-value block an F16 scale then 32 random int8 quants. */
    private static CpuTensorRef q8_0(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        int blocks = rows * (cols / 32);
        byte[] b = new byte[blocks * 34];
        for (int k = 0; k < blocks; k++) {
            putHalf(b, k * 34, HALF_SCALES[rnd.nextInt(HALF_SCALES.length)]);
            for (int j = 0; j < 32; j++) {
                b[k * 34 + 2 + j] = (byte) (rnd.nextInt(256) - 128);
            }
        }
        return CpuTensorRef.of(GGMLType.Q8_0, le(b), rows, cols);
    }

    /** Q4_K rows: per 256-value block F16 {@code d} and {@code dmin}, then random 6-bit scales and 4-bit quants. */
    private static CpuTensorRef q4K(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        int blocks = rows * (cols / 256);
        byte[] b = new byte[blocks * 144];
        for (int k = 0; k < blocks; k++) {
            putHalf(b, k * 144, (short) 0x2C00);
            putHalf(b, k * 144 + 2, (short) 0x2400);
            for (int j = 4; j < 144; j++) {
                b[k * 144 + j] = (byte) rnd.nextInt(256);
            }
        }
        return CpuTensorRef.of(GGMLType.Q4_K, le(b), rows, cols);
    }

    /** The type's own kernel row by row, on the input quantized once for the weight type. */
    private static float[] kernelReference(CpuTensorRef w, float[] x, int xOff) {
        CpuVecDot.QuantizedActivations q = CpuActivationQuantizer.quantizeFor(w.type(), x, xOff, (int) w.cols());
        CpuVecDot dot = CpuMatVec.vecDot(w.type());
        float[] y = new float[(int) w.rows()];
        for (int r = 0; r < y.length; r++) {
            y[r] = dot.dotRow(w, r, q);
        }
        return y;
    }

    /** Decoded F32 product in double precision. */
    private static double[] decodedProduct(CpuTensorRef w, float[] x, int xOff) {
        int cols = (int) w.cols();
        float[] row = new float[cols];
        double[] y = new double[(int) w.rows()];
        for (int r = 0; r < y.length; r++) {
            w.decodeRow(r, row, 0);
            double acc = 0.0;
            for (int c = 0; c < cols; c++) {
                acc += (double) row[c] * x[xOff + c];
            }
            y[r] = acc;
        }
        return y;
    }

    /**
     * Bound of the error that 8-bit activation quantization adds to row {@code r}: every activation moves by at most
     * half a quantization step of {@code max|x| / 127}, weighted by {@code |w|}.
     */
    private static double quantizationBound(CpuTensorRef w, long r, float[] x) {
        float[] row = new float[(int) w.cols()];
        w.decodeRow(r, row, 0);
        double maxX = 0.0;
        for (float v : x) {
            maxX = Math.max(maxX, Math.abs(v));
        }
        double l1 = 0.0;
        for (float v : row) {
            l1 += Math.abs(v);
        }
        return l1 * maxX / 127.0 + 1e-4;
    }

    private static void assertCloseToDecoded(CpuTensorRef w, float[] x, float[] y) {
        double[] ref = decodedProduct(w, x, 0);
        for (int r = 0; r < ref.length; r++) {
            assertEquals("row " + r, ref[r], y[r], quantizationBound(w, r, x));
        }
    }

    // ------------------------------------------------------------------ products

    @Test
    public void f32MatvecMatchesDecodedProduct() {
        int rows = 300;
        int cols = 128;
        CpuTensorRef w = f32(randomVector(rows * cols, 1), rows, cols);
        float[] x = randomVector(cols, 2);
        float[] y = new float[rows];
        CpuMatVec.PLAIN.matvec(w, x, 0, y, 0);
        double[] ref = decodedProduct(w, x, 0);
        for (int r = 0; r < rows; r++) {
            assertEquals("row " + r, ref[r], y[r], 1e-4);
        }
    }

    @Test
    public void q8_0MatvecMatchesKernelAndDecodedProduct() {
        CpuTensorRef w = q8_0(257, 256, 3);
        float[] x = randomVector(256, 4);
        float[] y = new float[257];
        CpuMatVec.PLAIN.matvec(w, x, 0, y, 0);
        assertNear(kernelReference(w, x, 0), y);
        assertCloseToDecoded(w, x, y);
    }

    @Test
    public void q4KMatvecMatchesKernelAndDecodedProduct() {
        CpuTensorRef w = q4K(64, 512, 5);
        float[] x = randomVector(512, 6);
        float[] y = new float[64];
        CpuMatVec.PLAIN.matvec(w, x, 0, y, 0);
        assertNear(kernelReference(w, x, 0), y);
        assertCloseToDecoded(w, x, y);
    }

    @Test
    public void matvecHonoursInputAndOutputOffsets() {
        CpuTensorRef w = q8_0(48, 64, 7);
        float[] x = randomVector(64 + 5, 8);
        float[] y = new float[48 + 3];
        Arrays.fill(y, 42.0f);
        CpuMatVec.PLAIN.matvec(w, x, 5, y, 3);
        float[] ref = kernelReference(w, x, 5);
        assertEquals(42.0f, y[0], 0.0f);
        assertEquals(42.0f, y[2], 0.0f);
        assertNear(ref, Arrays.copyOfRange(y, 3, 51));
    }

    @Test
    public void matmulEqualsPerTokenMatvec() {
        int rows = 96;
        int cols = 256;
        int n = 5;
        for (CpuTensorRef w : Arrays.asList(q8_0(rows, cols, 9), q4K(rows, cols, 10))) {
            float[] x = randomVector(n * cols, 11);
            float[] y = new float[n * rows];
            CpuMatVec.PLAIN.matmul(w, x, 0, y, 0, n);
            for (int t = 0; t < n; t++) {
                float[] one = new float[rows];
                CpuMatVec.PLAIN.matvec(w, x, t * cols, one, 0);
                assertNear(w.type() + " token " + t, one, Arrays.copyOfRange(y, t * rows, (t + 1) * rows));
            }
        }
    }

    @Test
    public void stridedMatmulLeavesGapsUntouched() {
        int rows = 40;
        int cols = 64;
        int n = 3;
        int xStride = cols + 3;
        int yStride = rows + 2;
        CpuTensorRef w = q8_0(rows, cols, 12);
        float[] x = randomVector(n * xStride, 13);
        float[] y = new float[n * yStride];
        Arrays.fill(y, -7.0f);
        CpuMatVec.PLAIN.matmul(w, x, 0, xStride, y, 0, yStride, n);
        for (int t = 0; t < n; t++) {
            float[] one = new float[rows];
            CpuMatVec.PLAIN.matvec(w, x, t * xStride, one, 0);
            assertNear(one, Arrays.copyOfRange(y, t * yStride, t * yStride + rows));
            assertEquals(-7.0f, y[t * yStride + rows], 0.0f);
            assertEquals(-7.0f, y[t * yStride + rows + 1], 0.0f);
        }
    }

    @Test
    public void matvecRowsComputesTheRequestedBlock() {
        CpuTensorRef w = q8_0(40, 128, 14);
        float[] x = randomVector(128, 15);
        float[] full = new float[40];
        CpuMatVec.PLAIN.matvec(w, x, 0, full, 0);
        float[] block = new float[7];
        CpuMatVec.PLAIN.matvecRows(w, 10, 7, x, 0, block, 0);
        assertNear(Arrays.copyOfRange(full, 10, 17), block);
    }

    @Test
    public void matvecAddAccumulates() {
        CpuTensorRef w = q4K(16, 256, 16);
        float[] x = randomVector(256, 17);
        float[] plain = new float[16];
        CpuMatVec.PLAIN.matvec(w, x, 0, plain, 0);
        float[] acc = new float[16];
        Arrays.fill(acc, 1.5f);
        CpuMatVec.PLAIN.matvecAdd(w, x, 0, acc, 0);
        for (int r = 0; r < 16; r++) {
            assertEquals(1.5f + plain[r], acc[r], tolerance(plain[r]));
        }
    }

    @Test
    public void matvecExpertSlicesStackedRows() {
        int experts = 3;
        int perExpert = 16;
        CpuTensorRef w = q8_0(experts * perExpert, 64, 18);
        float[] x = randomVector(64, 19);
        float[] full = new float[experts * perExpert];
        CpuMatVec.PLAIN.matvec(w, x, 0, full, 0);
        for (int e = 0; e < experts; e++) {
            float[] y = new float[perExpert];
            CpuMatVec.PLAIN.matvecExpert(w, e, experts, x, 0, y, 0);
            assertNear("expert " + e, Arrays.copyOfRange(full, e * perExpert, (e + 1) * perExpert), y);
        }
        try {
            CpuMatVec.PLAIN.matvecExpert(w, 0, x, 0, new float[perExpert], 0);
            fail("a weight without a known expert count needs one");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("expert"));
        }
    }

    // ------------------------------------------------------------------ transforms

    @Test
    public void outputScaleAppliesPerTensorAndPerExpert() {
        int experts = 3;
        int perExpert = 8;
        CpuTensorRef w = q8_0(experts * perExpert, 64, 20);
        float[] x = randomVector(64, 21);
        float[] plain = new float[experts * perExpert];
        CpuMatVec.PLAIN.matvec(w, x, 0, plain, 0);

        float[] perExpertScale = {2.0f, 3.0f, 0.5f};
        CpuMatVec scaled = new CpuMatVec(ref -> CpuMatVec.Transforms.of(null, perExpertScale, experts));
        float[] y = new float[experts * perExpert];
        scaled.matvec(w, x, 0, y, 0);
        for (int r = 0; r < y.length; r++) {
            assertEquals("row " + r, plain[r] * perExpertScale[r / perExpert], y[r], tolerance(plain[r]));
        }
        float[] e1 = new float[perExpert];
        scaled.matvecExpert(w, 1, x, 0, e1, 0);
        for (int i = 0; i < perExpert; i++) {
            assertEquals(plain[perExpert + i] * 3.0f, e1[i], tolerance(plain[perExpert + i]));
        }

        CpuMatVec tensorScaled = new CpuMatVec(ref -> CpuMatVec.Transforms.of(null, new float[]{0.25f}, 0));
        tensorScaled.matvec(w, x, 0, y, 0);
        for (int r = 0; r < y.length; r++) {
            assertEquals(plain[r] * 0.25f, y[r], tolerance(plain[r]));
        }
    }

    @Test
    public void hadamardInputTransformRotatesActivationsFirst() {
        int rows = 24;
        int cols = 64;
        HadamardSpec spec = HadamardSpec.rotation(32);
        CpuTensorRef w = f32(randomVector(rows * cols, 22), rows, cols);
        float[] x = randomVector(cols, 23);
        float[] rotated = new float[cols];
        CpuHadamard.apply(spec, x, 0, rotated, 0, 1, cols);
        float[] expected = new float[rows];
        CpuMatVec.PLAIN.matvec(w, rotated, 0, expected, 0);

        CpuMatVec prism = new CpuMatVec(ref -> CpuMatVec.Transforms.of(spec, null, 0));
        float[] y = new float[rows];
        prism.matvec(w, x, 0, y, 0);
        assertNear(expected, y);

        float[] xs = new float[2 * cols];
        System.arraycopy(x, 0, xs, 0, cols);
        System.arraycopy(randomVector(cols, 24), 0, xs, cols, cols);
        float[] ys = new float[2 * rows];
        prism.matmul(w, xs, 0, ys, 0, 2);
        assertNear(expected, Arrays.copyOfRange(ys, 0, rows));
    }

    @Test
    public void transformsRejectMalformedScales() {
        try {
            CpuMatVec.Transforms.of(null, new float[2], 0);
            fail("a 2-value scale needs 2 experts");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("scale"));
        }
        try {
            CpuMatVec.Transforms.of(null, new float[]{Float.NaN}, 0);
            fail("non-finite scale");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("finite"));
        }
        assertFalse(CpuMatVec.Transforms.of(null, null, 0).hasScale());
    }

    // ------------------------------------------------------------------ validation

    @Test
    public void rejectsViewsOutsideTheirArrays() {
        CpuTensorRef w = q8_0(8, 64, 25);
        try {
            CpuMatVec.PLAIN.matvec(w, new float[63], 0, new float[8], 0);
            fail("input shorter than a row");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("input"));
        }
        try {
            CpuMatVec.PLAIN.matvec(w, new float[64], 0, new float[7], 0);
            fail("output shorter than the rows");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("output"));
        }
        try {
            CpuMatVec.PLAIN.matvecRows(w, 5, 4, new float[64], 0, new float[4], 0);
            fail("rows past the end");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("outside"));
        }
        try {
            CpuMatVec.PLAIN.matvecExpert(w, 0, 3, new float[64], 0, new float[8], 0);
            fail("3 experts do not split 8 rows");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("expert count"));
        }
    }

    @Test
    public void typesWithoutKernelAreRejected() {
        assertFalse(CpuMatVec.supports(GGMLType.I32));
        assertFalse(CpuMatVec.supports(null));
        try {
            CpuMatVec.vecDot(GGMLType.I32);
            fail("I32 has no matmul kernel");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage().contains("I32"));
        }
    }

    @Test
    public void kernelTableIsTheDisjointUnionOfTheEightFamilies() {
        List<Set<GGMLType>> families = Arrays.asList(
                LegacyVecDot.supportedTypes(), Q8VecDot.supportedTypes(), Fp4VecDot.supportedTypes(),
                KQuantVecDot.supportedTypes(), IQuantVecDot.supportedTypes(), TernaryVecDot.supportedTypes(),
                NewFpVecDot.supportedTypes(), Fp16VecDot.supportedTypes());
        Set<GGMLType> union = EnumSet.noneOf(GGMLType.class);
        int total = 0;
        for (Set<GGMLType> f : families) {
            assertFalse("every family supports at least one type", f.isEmpty());
            union.addAll(f);
            total += f.size();
        }
        assertEquals("families overlap", union.size(), total);
        assertEquals(union, CpuMatVec.supportedTypes());
        for (GGMLType t : new GGMLType[]{GGMLType.F32, GGMLType.F16, GGMLType.BF16, GGMLType.Q4_0, GGMLType.Q8_0,
                GGMLType.Q4_K, GGMLType.Q6_K, GGMLType.IQ4_XS, GGMLType.TQ2_0, GGMLType.MXFP4}) {
            assertTrue(t + " is supported", CpuMatVec.supports(t));
        }
    }
}
