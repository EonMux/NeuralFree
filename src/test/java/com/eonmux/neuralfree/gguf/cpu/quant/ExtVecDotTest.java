package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link ExtVecDot}, the CPU vec-dot of the NeuralFree extension block types: agreement with the dot
 * product of the decoder's weights and the Q8_0-quantized activations, a hand-computed I8_B128 row, multi-shard
 * rows, the {@link CpuMatVec} registration and argument validation.
 */
public class ExtVecDotTest {

    private static final GGMLType[] EXT_TYPES = {
        GGMLType.F8_E4M3, GGMLType.F8_E4M3_B128, GGMLType.MXFP8, GGMLType.I8_B128, GGMLType.NF4, GGMLType.FP4_BNB
    };

    /**
     * The full vec-dot registry: the sanitized deep-test types above plus the block types whose kernels are
     * exercised where their fixtures live (E5M2 and the MXFP6 codes carry NaN patterns the raw random rows here
     * do not sanitize).
     */
    private static final GGMLType[] REGISTRY_TYPES = {
        GGMLType.F8_E4M3, GGMLType.F8_E5M2, GGMLType.F8_E4M3_B128, GGMLType.MXFP8,
        GGMLType.MXFP6_E3M2, GGMLType.MXFP6_E2M3, GGMLType.I8_B128, GGMLType.NF4, GGMLType.FP4_BNB
    };

    /** Random blocks with finite scales, moderate E8M0 exponents and no E4M3 NaN codes. */
    private static ByteBuffer randomRows(GGMLType t, int rows, int cols, Random rnd) {
        BlockLayout layout = BlockLayout.of(t);
        int blocks = rows * (cols / layout.blockElements());
        byte[] raw = new byte[blocks * layout.blockBytes()];
        rnd.nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            int o = i * layout.blockBytes();
            switch (t) {
                case F8_E4M3:
                    clearE4m3NaNs(raw, o + BlockLayout.F8_E4M3_QS, BlockLayout.QK_F8_E4M3);
                    break;
                case F8_E4M3_B128:
                    b.putFloat(o + BlockLayout.F8B128_D, 0.01f + rnd.nextFloat() * 0.1f);
                    clearE4m3NaNs(raw, o + BlockLayout.F8B128_QS, BlockLayout.QK_F8B128);
                    break;
                case MXFP8:
                    raw[o + BlockLayout.MXFP8_E] = (byte) (118 + rnd.nextInt(14));
                    clearE4m3NaNs(raw, o + BlockLayout.MXFP8_QS, BlockLayout.QK_MXFP8);
                    break;
                case I8_B128:
                    b.putFloat(o + BlockLayout.I8B128_D, (rnd.nextFloat() * 2f - 1f) * 0.01f);
                    break;
                default:
                    b.putFloat(o + BlockLayout.BNB4_D, 0.01f + rnd.nextFloat() * 2f);
                    break;
            }
        }
        return b;
    }

    private static void clearE4m3NaNs(byte[] raw, int from, int n) {
        for (int j = from; j < from + n; j++) {
            if ((raw[j] & 0x7F) == 0x7F) {
                raw[j] = (byte) (raw[j] & 0xFE);
            }
        }
    }

    private static float[] gaussian(int n, Random rnd) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) rnd.nextGaussian();
        }
        return x;
    }

    @Test
    public void everyTypeMatchesTheDecodedDotProduct() {
        final int rows = 2;
        final int cols = 256;
        Random rnd = new Random(0xE7D07L);
        for (GGMLType t : EXT_TYPES) {
            CpuTensorRef w = CpuTensorRef.of(t, randomRows(t, rows, cols, rnd), rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            assertEquals(t + " activations", GGMLType.Q8_0, x.type());
            CpuVecDot vd = ExtVecDot.forType(t);
            float[] wf = new float[cols];
            float[] y = new float[rows];
            vd.dotRows(w, 0, rows, x, y, 0);
            for (int r = 0; r < rows; r++) {
                w.decodeRow(r, wf, 0);
                double ref = 0;
                double maxW = 0;
                double sumA = 0;
                for (int j = 0; j < cols; j++) {
                    double a = (double) x.quant(j) * x.scale(j / x.blockSize());
                    ref += wf[j] * a;
                    maxW = Math.max(maxW, Math.abs(wf[j]));
                    sumA += Math.abs(a);
                }
                double tol = 1e-5 * maxW * sumA + 1e-6;
                float got = vd.dotRow(w, r, x);
                assertTrue(t + " row " + r + " is finite", Float.isFinite(got));
                assertEquals(t + " row " + r, ref, got, tol);
                assertEquals(t + " row " + r + " via dotRows", got, y[r], 0f);
            }
        }
    }

    /**
     * One I8_B128 row of two blocks with all weight codes 1 and integer activations that Q8_0 quantizes with
     * {@code d = 1}, so the dot is {@code sum_j d_w(j / 128) * x_j} exactly.
     */
    @Test
    public void i8B128RowMatchesTheHandComputedSum() {
        final int cols = 256;
        ByteBuffer b = ByteBuffer.allocate(2 * BlockLayout.I8B128_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        float[] d = {0.5f, -2f};
        for (int i = 0; i < 2; i++) {
            int o = i * BlockLayout.I8B128_BYTES;
            b.putFloat(o + BlockLayout.I8B128_D, d[i]);
            for (int j = 0; j < BlockLayout.QK_I8B128; j++) {
                b.put(o + BlockLayout.I8B128_QS + j, (byte) 1);
            }
        }
        float[] x = new float[cols];
        for (int j = 0; j < cols; j++) {
            x[j] = j % 32 == 0 ? 127f : (j * 13) % 61 - 30;
        }
        QuantizedActivations xq = CpuActivationQuantizer.quantizeQ8_0(x, 0, cols);
        double expected = 0;
        for (int j = 0; j < cols; j++) {
            assertEquals(1f, xq.scale(j / 32), 0f);
            assertEquals((int) x[j], xq.quant(j));
            expected += d[j / 128] * x[j];
        }
        CpuTensorRef w = CpuTensorRef.of(GGMLType.I8_B128, b, 1, cols);
        assertEquals(expected, ExtVecDot.I8_B128.dotRow(w, 0, xq), 0.0);
    }

    @Test
    public void dotRowsAcrossShardsMatchesSingleRows() {
        final int rows = 4;
        final int cols = 128;
        Random rnd = new Random(7);
        for (GGMLType t : EXT_TYPES) {
            ByteBuffer whole = randomRows(t, rows, cols, rnd);
            long rowBytes = BlockLayout.of(t).rowBytes(cols);
            ByteBuffer first = slice(whole, 0, (int) (2 * rowBytes));
            ByteBuffer second = slice(whole, (int) (2 * rowBytes), (int) (2 * rowBytes));
            CpuTensorRef sharded = new CpuTensorRef(t, new ByteBuffer[]{first, second},
                    new long[]{0L, 2 * rowBytes}, rows, cols, rowBytes);
            CpuTensorRef single = CpuTensorRef.of(t, whole, rows, cols);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, gaussian(cols, rnd), 0, cols);
            CpuVecDot vd = ExtVecDot.forType(t);
            float[] y = new float[rows + 2];
            vd.dotRows(sharded, 0, rows, x, y, 1);
            assertEquals(0f, y[0], 0f);
            assertEquals(0f, y[rows + 1], 0f);
            for (int r = 0; r < rows; r++) {
                assertEquals(t + " row " + r, vd.dotRow(single, r, x), y[r + 1], 0f);
            }
        }
    }

    private static ByteBuffer slice(ByteBuffer src, int from, int len) {
        ByteBuffer d = src.duplicate();
        d.position(from);
        d.limit(from + len);
        return d.slice().order(ByteOrder.LITTLE_ENDIAN);
    }

    @Test
    public void registeredInCpuMatVecWithQ8_0Activations() {
        assertEquals(EnumSet.copyOf(Arrays.asList(REGISTRY_TYPES)), EnumSet.copyOf(ExtVecDot.supportedTypes()));
        for (GGMLType t : EXT_TYPES) {
            assertEquals(t.toString(), GGMLType.Q8_0, CpuActivationQuantizer.activationType(t));
            assertEquals(t.toString(), GGMLType.Q8_0, ExtVecDot.forType(t).activationType());
            assertEquals(t, ExtVecDot.forType(t).weightType());
            assertTrue(t.toString(), CpuMatVec.supports(t));
            assertSame(ExtVecDot.forType(t), CpuMatVec.vecDot(t));
        }
        try {
            ExtVecDot.forType(GGMLType.Q8_0);
            fail("Q8_0 is not an extension type");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Q8_0"));
        }
    }

    @Test
    public void matvecUsesTheExtensionKernel() {
        final int rows = 2;
        final int cols = 128;
        Random rnd = new Random(11);
        for (GGMLType t : EXT_TYPES) {
            CpuTensorRef w = CpuTensorRef.of(t, randomRows(t, rows, cols, rnd), rows, cols);
            float[] xf = gaussian(cols, rnd);
            float[] y = new float[rows];
            CpuMatVec.PLAIN.matvec(w, xf, 0, y, 0);
            QuantizedActivations x = CpuActivationQuantizer.quantizeFor(t, xf, 0, cols);
            for (int r = 0; r < rows; r++) {
                assertEquals(t + " row " + r, ExtVecDot.forType(t).dotRow(w, r, x), y[r], 0f);
            }
        }
    }

    @Test
    public void rejectsMismatchedWeightsAndActivations() {
        final int cols = 128;
        Random rnd = new Random(3);
        CpuTensorRef nf4 = CpuTensorRef.of(GGMLType.NF4, randomRows(GGMLType.NF4, 1, cols, rnd), 1, cols);
        float[] xf = gaussian(cols, rnd);
        try {
            ExtVecDot.NF4.dotRow(nf4, 0, CpuActivationQuantizer.f32(xf, 0, cols));
            fail("F32 activations");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Q8_0"));
        }
        try {
            ExtVecDot.NF4.dotRow(nf4, 0, CpuActivationQuantizer.quantizeQ8_0(xf, 0, 64));
            fail("short activations");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("64"));
        }
        try {
            ExtVecDot.FP4_BNB.dotRow(nf4, 0, CpuActivationQuantizer.quantizeQ8_0(xf, 0, cols));
            fail("wrong weight type");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("NF4"));
        }
    }
}
