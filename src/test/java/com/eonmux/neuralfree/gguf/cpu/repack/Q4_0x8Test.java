package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.LegacyVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.Q4_0x8VecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Assume;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Properties;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the Q4_0 weight repacking ({@link Q4_0x8}, the {@code Q4_0x8VecDot} kernels of
 * {@code com.eonmux.neuralfree.gguf.cpu.quant}, {@link CpuRepack}): the interleaved byte layout of llama.cpp
 * {@code make_block_q4_0x8}, bit-exact agreement of the repacked kernels with the plain Q4_0 kernel, the
 * eligibility and caching of {@link CpuRepack#planFor}, the property gate and the {@link CpuMatVec} kernel
 * source seam. The plan tests assume the Vector API engine, which supplies the repacked kernel.
 */
public class Q4_0x8Test {

    /** One group of 8 blocks with known bytes: block {@code i} carries quants {@code 16i + j} (unsigned). */
    private static ByteBuffer knownGroup() {
        byte[] raw = new byte[8 * BlockLayout.Q4_0_BYTES];
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 8; i++) {
            int off = i * BlockLayout.Q4_0_BYTES;
            b.putShort(off + BlockLayout.Q4_0_D, (short) FpBlockDecoders.fp32ToFp16(0.03125f * (i + 1)));
            for (int j = 0; j < 16; j++) {
                raw[off + BlockLayout.Q4_0_QS + j] = (byte) (16 * i + j);
            }
        }
        return b;
    }

    /** Random Q4_0 rows with finite, moderate scales. */
    private static ByteBuffer randomRows(int rows, int cols, Random rnd) {
        BlockLayout layout = BlockLayout.of(GGMLType.Q4_0);
        int blocks = rows * (cols / layout.blockElements());
        byte[] raw = new byte[blocks * layout.blockBytes()];
        rnd.nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            float v = (rnd.nextFloat() * 2f - 1f) * 0.1f;
            b.putShort(i * layout.blockBytes() + BlockLayout.Q4_0_D,
                    (short) FpBlockDecoders.fp32ToFp16(v));
        }
        return b;
    }

    private static float[] gaussian(int n, Random rnd) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) rnd.nextGaussian();
        }
        return x;
    }

    /** Repacks every row of {@code w} into one buffer and returns a tensor view over it. */
    private static CpuTensorRef repacked(CpuTensorRef w) {
        int groups = (int) (w.cols() / Q4_0x8.ELEMENTS);
        byte[] dst = new byte[(int) (w.rows() * w.rowBytes())];
        for (int r = 0; r < w.rows(); r++) {
            Q4_0x8.repackRows(w.rowBuffer(r), w.rowOffset(r), dst, (int) (r * w.rowBytes()), groups);
        }
        return CpuTensorRef.of(GGMLType.Q4_0, ByteBuffer.wrap(dst).order(ByteOrder.LITTLE_ENDIAN),
                w.rows(), w.cols());
    }

    @Test
    public void theGroupLayoutMatchesLlamaCpp() {
        ByteBuffer src = knownGroup();
        byte[] dst = new byte[Q4_0x8.BYTES];
        Q4_0x8.repackRows(src, 0, dst, 0, 1);
        // d[i] first, in block order (repack.cpp:3131-3133)
        for (int i = 0; i < 8; i++) {
            assertEquals("d[" + i + "] low byte",
                    src.get(i * BlockLayout.Q4_0_BYTES + BlockLayout.Q4_0_D), dst[Q4_0x8.D + 2 * i]);
            assertEquals("d[" + i + "] high byte",
                    src.get(i * BlockLayout.Q4_0_BYTES + BlockLayout.Q4_0_D + 1), dst[Q4_0x8.D + 2 * i + 1]);
        }
        // qs interleaved in 8-byte slices, each byte XOR 0x88 (repack.cpp:3138-3147): the first 64 bytes are
        // blocks 0..7 qs[0..8), the second 64 are blocks 0..7 qs[8..16)
        for (int i = 0; i < 8; i++) {
            for (int k = 0; k < 8; k++) {
                assertEquals("lo slice block " + i + " byte " + k,
                        (byte) (src.get(i * BlockLayout.Q4_0_BYTES + BlockLayout.Q4_0_QS + k) ^ 0x88),
                        dst[Q4_0x8.QS + 8 * i + k]);
                assertEquals("hi slice block " + i + " byte " + k,
                        (byte) (src.get(i * BlockLayout.Q4_0_BYTES + BlockLayout.Q4_0_QS + 8 + k) ^ 0x88),
                        dst[Q4_0x8.QS + 64 + 8 * i + k]);
            }
        }
    }

    @Test
    public void theScalarRepackedKernelMatchesThePlainKernelBitForBit() {
        final int rows = 3;
        final int cols = 1024;
        Random rnd = new Random(0x408);
        CpuTensorRef plain = CpuTensorRef.of(GGMLType.Q4_0, randomRows(rows, cols, rnd), rows, cols);
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_0(gaussian(cols, rnd), 0, cols);
        CpuTensorRef repacked = repacked(plain);
        for (int r = 0; r < rows; r++) {
            assertEquals("row " + r + ": the repacked kernel must match the plain kernel exactly",
                    LegacyVecDot.Q4_0.dotRow(plain, r, x), Q4_0x8VecDot.INSTANCE.dotRow(repacked, r, x), 0f);
        }
        float[] ys = new float[rows];
        float[] yr = new float[rows];
        LegacyVecDot.Q4_0.dotRows(plain, 0, rows, x, ys, 0);
        Q4_0x8VecDot.INSTANCE.dotRows(repacked, 0, rows, x, yr, 0);
        assertArrayEquals("dotRows", ys, yr, 0f);
    }

    @Test
    public void theVectorRepackedKernelMatchesTheScalarOne() {
        Assume.assumeTrue(CpuSimd.available());
        CpuVecDot simd = CpuSimd.q4_0x8();
        assertNotNull("the engine offers the Q4_0x8 kernel", simd);
        final int rows = 2;
        final int cols = 512;
        Random rnd = new Random(0x409);
        CpuTensorRef plain = CpuTensorRef.of(GGMLType.Q4_0, randomRows(rows, cols, rnd), rows, cols);
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_0(gaussian(cols, rnd), 0, cols);
        CpuTensorRef repacked = repacked(plain);
        for (int r = 0; r < rows; r++) {
            assertEquals("row " + r, Q4_0x8VecDot.INSTANCE.dotRow(repacked, r, x), simd.dotRow(repacked, r, x),
                    0f);
        }
    }

    @Test
    public void plansCoverEligibleWeightsAndCacheThem() {
        Assume.assumeTrue(CpuSimd.available());
        CpuRepack repack = new CpuRepack(CpuRepack.Mode.ON);
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q4_0, randomRows(2, 512, new Random(5)), 2, 512);
        CpuMatVec.MatvecPlan plan = repack.planFor(w);
        assertNotNull("a 512-column Q4_0 weight is eligible", plan);
        assertEquals(w.rows(), plan.weight.rows());
        assertEquals(w.cols(), plan.weight.cols());
        assertEquals(w.rowBytes(), plan.weight.rowBytes());
        assertNotNull(plan.kernel);
        assertSame("the plan is cached", plan, repack.planFor(w));
        assertEquals(1, repack.repackedCount());
        // the copy holds the interleaved bytes: row for row the repacker's own output, and not the original
        CpuTensorRef reference = repacked(w);
        for (int r = 0; r < w.rows(); r++) {
            assertArrayEquals("plan row " + r, rowOf(reference, r), rowOf(plan.weight, r));
            assertFalse("plan row " + r + " is repacked", Arrays.equals(rowOf(w, r), rowOf(plan.weight, r)));
        }
        repack.clear();
        assertEquals(0, repack.repackedCount());
    }

    /** Row {@code r} of {@code t} as a fresh array. */
    private static byte[] rowOf(CpuTensorRef t, int r) {
        byte[] row = new byte[(int) t.rowBytes()];
        ByteBuffer d = t.rowBuffer(r).duplicate();
        d.position((int) t.rowOffset(r));
        d.get(row, 0, (int) t.rowBytes());
        return row;
    }

    @Test
    public void ineligibleWeightsGetNoPlan() {
        Assume.assumeTrue(CpuSimd.available());
        CpuRepack repack = new CpuRepack(CpuRepack.Mode.ON);
        assertNull("columns of 15 blocks do not split into groups of 8",
                repack.planFor(CpuTensorRef.of(GGMLType.Q4_0, randomRows(1, 480, new Random(6)), 1, 480)));
        assertNull("only Q4_0 repacks",
                repack.planFor(CpuTensorRef.of(GGMLType.Q8_0, ByteBuffer.allocate(34), 1, 32)));
        assertNull("null gets no plan", repack.planFor(null));
    }

    @Test
    public void thePropertySelectsTheModeAndRejectsOtherValuesByName() {
        Properties p = new Properties();
        assertEquals("absent property keeps the measured types only", CpuRepack.Mode.AUTO,
                CpuRepack.Mode.fromProperties(p));
        p.setProperty(CpuRepack.PROP_REPACK, "on");
        assertEquals(CpuRepack.Mode.ON, CpuRepack.Mode.fromProperties(p));
        p.setProperty(CpuRepack.PROP_REPACK, " OFF ");
        assertEquals(CpuRepack.Mode.OFF, CpuRepack.Mode.fromProperties(p));
        p.setProperty(CpuRepack.PROP_REPACK, "auto");
        assertEquals(CpuRepack.Mode.AUTO, CpuRepack.Mode.fromProperties(p));
        p.setProperty(CpuRepack.PROP_REPACK, "yes");
        try {
            CpuRepack.Mode.fromProperties(p);
            fail("expected the rejection of " + CpuRepack.PROP_REPACK + "=yes");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(CpuRepack.PROP_REPACK));
            assertTrue(expected.getMessage().contains("yes"));
        }
    }

    @Test
    public void aKernelSourcePlanRunsTheRepackedCopyThroughTheMatvec() {
        final int rows = 3;
        final int cols = 512;
        Random rnd = new Random(9);
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q4_0, randomRows(rows, cols, rnd), rows, cols);
        float[] x = gaussian(cols, rnd);
        float[] plain = new float[rows];
        new CpuMatVec(weight -> CpuMatVec.Transforms.NONE).matvec(w, x, 0, plain, 0);

        CpuMatVec.MatvecPlan plan = new CpuMatVec.MatvecPlan(repacked(w), Q4_0x8VecDot.INSTANCE);
        float[] viaPlan = new float[rows];
        new CpuMatVec(weight -> CpuMatVec.Transforms.NONE, weight -> plan).matvec(w, x, 0, viaPlan, 0);
        assertArrayEquals("the repacked product must equal the plain product exactly", plain, viaPlan, 0f);
    }

    @Test
    public void theKernelRejectsWeightsThatDoNotSplitIntoGroups() {
        QuantizedActivations x = CpuActivationQuantizer.quantizeQ8_0(new float[480], 0, 480);
        CpuTensorRef w = CpuTensorRef.of(GGMLType.Q4_0, randomRows(1, 480, new Random(7)), 1, 480);
        try {
            Q4_0x8VecDot.INSTANCE.dotRow(w, 0, x);
            fail("expected the refusal of a 480-column weight");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("256-element groups"));
        }
    }
}
