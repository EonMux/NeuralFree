package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link CpuMlaHeadProducts} against one {@link CpuMatVec#matvecRows} per (token, head) pair, the per-pair product it
 * batches: F32 and Q8_0 weights, strided inputs and outputs, row blocks that split a head, and the shape checks.
 */
public class CpuMlaHeadProductsTest {

    private static final int N_HEAD = 3;
    private static final int COLS = 64;
    private static final int ROWS_PER_HEAD = 40;

    private static CpuTensorRef f32Weight(long seed) {
        int rows = N_HEAD * ROWS_PER_HEAD;
        ByteBuffer b = ByteBuffer.allocate(rows * COLS * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        SplittableRandom rng = new SplittableRandom(seed);
        for (int i = 0; i < rows * COLS; i++) {
            b.putFloat((float) (rng.nextDouble() - 0.5));
        }
        b.flip();
        return new CpuTensorRef(GGMLType.F32, new ByteBuffer[]{b}, new long[]{0L}, rows, COLS,
                (long) COLS * Float.BYTES);
    }

    /** Q8_0 rows: per block of 32 a half-precision scale of 1/64 then 32 random int8 quants. */
    private static CpuTensorRef q8Weight(long seed) {
        int rows = N_HEAD * ROWS_PER_HEAD;
        int blocks = COLS / 32;
        int rowBytes = blocks * 34;
        ByteBuffer b = ByteBuffer.allocate(rows * rowBytes).order(ByteOrder.LITTLE_ENDIAN);
        SplittableRandom rng = new SplittableRandom(seed);
        for (int r = 0; r < rows * blocks; r++) {
            b.putShort((short) 0x2400);
            for (int i = 0; i < 32; i++) {
                b.put((byte) (rng.nextInt(255) - 127));
            }
        }
        b.flip();
        return new CpuTensorRef(GGMLType.Q8_0, new ByteBuffer[]{b}, new long[]{0L}, rows, COLS, rowBytes);
    }

    private static float[] random(long seed, int n) {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (rng.nextDouble() * 2.0 - 1.0);
        }
        return v;
    }

    /**
     * Inputs strided as the absorption reads q_nope ({@code [t][h][head_k]} with a wider head), outputs as it writes
     * the absorbed query ({@code [t][h][d_q]} with a wider head), compared pair by pair.
     */
    private static void check(CpuTensorRef w, int nTokens) {
        int xHead = COLS + 16;
        int xToken = N_HEAD * xHead;
        int yHead = ROWS_PER_HEAD + 8;
        int yToken = N_HEAD * yHead;
        int xOff = 5;
        int yOff = 3;
        float[] x = random(11, xOff + nTokens * xToken);
        float[] y = new float[yOff + nTokens * yToken];
        float[] want = new float[y.length];
        CpuMlaHeadProducts.run(w, N_HEAD, ROWS_PER_HEAD, nTokens, x, xOff, xToken, xHead, y, yOff, yToken, yHead);
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                CpuMatVec.PLAIN.matvecRows(w, (long) h * ROWS_PER_HEAD, ROWS_PER_HEAD, x,
                        xOff + t * xToken + h * xHead, want, yOff + t * yToken + h * yHead);
            }
        }
        for (int i = 0; i < y.length; i++) {
            assertEquals("element " + i, want[i], y[i], 1e-5f * Math.max(1.0f, Math.abs(want[i])));
        }
    }

    @Test
    public void f32HeadsMatchThePerPairProducts() {
        check(f32Weight(1), 1);
        check(f32Weight(2), 7);
    }

    @Test
    public void q8HeadsMatchThePerPairProducts() {
        check(q8Weight(3), 1);
        check(q8Weight(4), 9);
    }

    @Test
    public void manyTokensSplitHeadsIntoRowBlocks() {
        check(f32Weight(5), 64);
    }

    @Test
    public void shapesAndViewsAreChecked() {
        CpuTensorRef w = f32Weight(6);
        float[] x = new float[N_HEAD * COLS];
        float[] y = new float[N_HEAD * ROWS_PER_HEAD];
        assertThrows(IllegalArgumentException.class, () -> CpuMlaHeadProducts.run(w, N_HEAD + 1, ROWS_PER_HEAD, 1,
                x, 0, N_HEAD * COLS, COLS, y, 0, N_HEAD * ROWS_PER_HEAD, ROWS_PER_HEAD));
        assertThrows(IllegalArgumentException.class, () -> CpuMlaHeadProducts.run(w, N_HEAD, ROWS_PER_HEAD, 2,
                x, 0, N_HEAD * COLS, COLS, y, 0, N_HEAD * ROWS_PER_HEAD, ROWS_PER_HEAD));
        assertThrows(IllegalArgumentException.class, () -> CpuMlaHeadProducts.run(w, N_HEAD, ROWS_PER_HEAD, 0,
                x, 0, N_HEAD * COLS, COLS, y, 0, N_HEAD * ROWS_PER_HEAD, ROWS_PER_HEAD));
    }
}
