package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import org.junit.Assume;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;
import java.util.function.IntToLongFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The Vector API kernels of the grid-coded i-quants ({@code SimdIqGrid}): their 64-bit sign step negates whole
 * grid rows without a carry between bytes only because no grid magnitude is zero, and their int16 lanes hold four
 * products of the largest grid magnitude and an int8 activation; at the largest grid entries, every sign set and
 * the largest scales, the batched product and the single-token row loop return the scalar kernel's bits.
 */
public class SimdIqGridTest {

    private static final GGMLType[] GRID_TYPES = {
            GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S, GGMLType.IQ3_XXS, GGMLType.IQ3_S};

    @Test
    public void everyGridMagnitudeIsANonZeroByteOfAtMost62() {
        assertMagnitudes("iq2xxs", 256, 8, QuantTables::iq2xxsGrid);
        assertMagnitudes("iq2xs", 512, 8, QuantTables::iq2xsGrid);
        assertMagnitudes("iq2s", 1024, 8, QuantTables::iq2sGrid);
        assertMagnitudes("iq3xxs", 256, 4, i -> QuantTables.iq3xxsGrid(i) & 0xFFFFFFFFL);
        assertMagnitudes("iq3s", 512, 4, i -> QuantTables.iq3sGrid(i) & 0xFFFFFFFFL);
    }

    private static void assertMagnitudes(String name, int entries, int lanes, IntToLongFunction grid) {
        for (int i = 0; i < entries; i++) {
            final long g = grid.applyAsLong(i);
            for (int j = 0; j < lanes; j++) {
                final int v = (int) (g >>> (8 * j)) & 0xFF;
                assertTrue(name + "[" + i + "] lane " + j + " = " + v, v >= 1 && v <= 62);
            }
        }
    }

    @Test
    public void theLargestGridEntriesSignsAndScalesMatchTheScalarKernels() {
        Assume.assumeTrue(CpuSimd.available());
        final int rows = 5;
        final int cols = 512;
        final int nTokens = 11;
        final Random rnd = new Random(0x16D);
        for (GGMLType t : GRID_TYPES) {
            final ByteBuffer data = SimdVecDotTest.randomRows(t, rows, cols, rnd);
            final byte[] raw = data.array();
            final BlockLayout layout = BlockLayout.of(t);
            for (int blk = 0; blk < rows * cols / layout.blockElements(); blk++) {
                if (blk % 3 != 2) {
                    largest(t, raw, blk * layout.blockBytes());
                }
            }
            final CpuTensorRef w = CpuTensorRef.of(t, data, rows, cols);
            final QuantizedActivations[] acts = new QuantizedActivations[nTokens];
            for (int tok = 0; tok < nTokens; tok++) {
                final float[] x = SimdVecDotTest.gaussian(cols, rnd);
                if (tok < 8) {
                    Arrays.fill(x, tok % 2 == 0 ? 1f : -1f);
                }
                acts[tok] = CpuActivationQuantizer.quantizeFor(t, x, 0, cols);
            }
            final CpuVecDot simd = CpuSimd.vecDot(t);
            final CpuVecDot scalar = IQuantVecDot.forType(t);
            final float[] y = new float[nTokens * rows];
            simd.dotTiles(w, 0, rows, acts, nTokens, y, 0, rows);
            for (int tok = 0; tok < nTokens; tok++) {
                final float[] expected = new float[rows];
                scalar.dotRows(w, 0, rows, acts[tok], expected, 0);
                for (int r = 0; r < rows; r++) {
                    assertEquals(t + " token " + tok + ", row " + r, expected[r], y[tok * rows + r], 0f);
                }
                final float[] single = new float[rows];
                simd.dotRows(w, 0, rows, acts[tok], single, 0);
                assertArrayEquals(t + " token " + tok + ": the single-token row loop", expected, single, 0f);
            }
        }
    }

    /** Rewrites the block at {@code raw[o..)} to its type's largest grid entry, every sign set, the largest scale. */
    private static void largest(GGMLType t, byte[] raw, int o) {
        switch (t) {
            case IQ2_XXS: {
                final int idx = argmax(256, 8, QuantTables::iq2xxsGrid);
                for (int ib32 = 0; ib32 < 8; ib32++) {
                    final int at = o + BlockLayout.IQ2_XXS_QS + 8 * ib32;
                    Arrays.fill(raw, at, at + 4, (byte) idx);
                    Arrays.fill(raw, at + 4, at + 8, (byte) -1);
                }
                return;
            }
            case IQ2_XS: {
                final int v = argmax(512, 8, QuantTables::iq2xsGrid) | 127 << 9;
                for (int k = 0; k < 32; k++) {
                    raw[o + BlockLayout.IQ2_XS_QS + 2 * k] = (byte) v;
                    raw[o + BlockLayout.IQ2_XS_QS + 2 * k + 1] = (byte) (v >>> 8);
                }
                Arrays.fill(raw, o + BlockLayout.IQ2_XS_SCALES, o + BlockLayout.IQ2_XS_BYTES, (byte) -1);
                return;
            }
            case IQ2_S: {
                final int idx = argmax(1024, 8, QuantTables::iq2sGrid);
                final int h = (idx >>> 8) & 3;
                Arrays.fill(raw, o + BlockLayout.IQ2_S_QS, o + BlockLayout.IQ2_S_SIGNS, (byte) idx);
                Arrays.fill(raw, o + BlockLayout.IQ2_S_SIGNS, o + BlockLayout.IQ2_S_QH, (byte) -1);
                Arrays.fill(raw, o + BlockLayout.IQ2_S_QH, o + BlockLayout.IQ2_S_SCALES,
                        (byte) (h | h << 2 | h << 4 | h << 6));
                Arrays.fill(raw, o + BlockLayout.IQ2_S_SCALES, o + BlockLayout.IQ2_S_BYTES, (byte) -1);
                return;
            }
            case IQ3_XXS: {
                final int idx = argmax(256, 4, i -> QuantTables.iq3xxsGrid(i) & 0xFFFFFFFFL);
                Arrays.fill(raw, o + BlockLayout.IQ3_XXS_QS, o + BlockLayout.IQ3_XXS_SAS, (byte) idx);
                Arrays.fill(raw, o + BlockLayout.IQ3_XXS_SAS, o + BlockLayout.IQ3_XXS_BYTES, (byte) -1);
                return;
            }
            case IQ3_S: {
                final int idx = argmax(512, 4, i -> QuantTables.iq3sGrid(i) & 0xFFFFFFFFL);
                Arrays.fill(raw, o + BlockLayout.IQ3_S_QS, o + BlockLayout.IQ3_S_QH, (byte) idx);
                Arrays.fill(raw, o + BlockLayout.IQ3_S_QH, o + BlockLayout.IQ3_S_SIGNS,
                        (byte) ((idx & 256) != 0 ? -1 : 0));
                Arrays.fill(raw, o + BlockLayout.IQ3_S_SIGNS, o + BlockLayout.IQ3_S_BYTES, (byte) -1);
                return;
            }
            default:
                throw new IllegalArgumentException(t.name());
        }
    }

    /** The grid entry whose lanes sum to the most, the largest products a row of the type can hold. */
    private static int argmax(int entries, int lanes, IntToLongFunction grid) {
        int best = 0;
        int bestSum = -1;
        for (int i = 0; i < entries; i++) {
            final long g = grid.applyAsLong(i);
            int sum = 0;
            for (int j = 0; j < lanes; j++) {
                sum += (int) (g >>> (8 * j)) & 0xFF;
            }
            if (sum > bestSum) {
                best = i;
                bestSum = sum;
            }
        }
        return best;
    }
}
