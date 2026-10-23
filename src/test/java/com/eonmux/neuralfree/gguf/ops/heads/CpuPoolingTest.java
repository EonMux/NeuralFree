package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Reference checks of the heads-family CPU math: pooling against the llama.cpp build_pooling definitions (mean as
 * the product with the 1/n vector, CLS/LAST row choice, RANK resolution), row gather against ggml_get_rows, and the
 * switched-LoRA delta against a direct transcription of the two mul_mat_id of granite-switch.cpp, plus the launch
 * geometry helpers.
 */
public class CpuPoolingTest {

    private static final float EPS = 1e-5f;

    private static float[] random(Random r, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = r.nextFloat() * 2f - 1f;
        }
        return a;
    }

    @Test
    public void meanPoolingIsTheProductWithTheInverseCountVector() {
        Random r = new Random(1);
        int nTokens = 5, nEmbd = 7, stride = 9, off = 3;
        float[] x = random(r, off + nTokens * stride);
        float[] out = new float[2 + nEmbd];
        CpuPooling.pool(PoolingKind.MEAN, nTokens, nEmbd, x, off, stride, out, 2);
        float div = 1.0f / nTokens;
        for (int c = 0; c < nEmbd; c++) {
            float ref = 0f;
            for (int t = 0; t < nTokens; t++) {
                ref += x[off + t * stride + c] * div;
            }
            assertEquals(ref, out[2 + c], 0f);
        }
        assertEquals(0f, out[0], 0f);
        assertEquals(0f, out[1], 0f);
    }

    @Test
    public void clsLastAndNonePickRows() {
        Random r = new Random(2);
        int nTokens = 4, nEmbd = 6, stride = 8;
        float[] x = random(r, nTokens * stride);
        float[] cls = new float[nEmbd];
        float[] last = new float[nEmbd];
        float[] none = new float[nTokens * nEmbd];
        CpuPooling.pool(PoolingKind.CLS, nTokens, nEmbd, x, 0, stride, cls, 0);
        CpuPooling.pool(PoolingKind.LAST, nTokens, nEmbd, x, 0, stride, last, 0);
        CpuPooling.pool(PoolingKind.NONE, nTokens, nEmbd, x, 0, stride, none, 0);
        assertArrayEquals(Arrays.copyOfRange(x, 0, nEmbd), cls, 0f);
        assertArrayEquals(Arrays.copyOfRange(x, (nTokens - 1) * stride, (nTokens - 1) * stride + nEmbd), last, 0f);
        for (int t = 0; t < nTokens; t++) {
            assertArrayEquals(Arrays.copyOfRange(x, t * stride, t * stride + nEmbd),
                    Arrays.copyOfRange(none, t * nEmbd, (t + 1) * nEmbd), 0f);
        }
    }

    @Test
    public void rankResolvesLikeBuildPooling() {
        assertEquals(PoolingKind.MEAN, CpuPooling.rowPooling(PoolingKind.RANK, PoolingKind.MEAN, false));
        assertEquals(PoolingKind.MEAN, CpuPooling.rowPooling(PoolingKind.RANK, PoolingKind.MEAN, true));
        assertEquals(PoolingKind.CLS, CpuPooling.rowPooling(PoolingKind.RANK, null, false));
        assertEquals(PoolingKind.LAST, CpuPooling.rowPooling(PoolingKind.RANK, null, true));
        assertEquals(PoolingKind.CLS, CpuPooling.rowPooling(PoolingKind.RANK, PoolingKind.CLS, false));
        assertEquals(PoolingKind.CLS, CpuPooling.rowPooling(PoolingKind.CLS, PoolingKind.MEAN, true));
        assertEquals(PoolingKind.NONE, CpuPooling.rowPooling(PoolingKind.NONE, null, true));
    }

    @Test
    public void poolingRejectsUnresolvedRank() {
        try {
            CpuPooling.pool(PoolingKind.RANK, 1, 1, new float[1], 0, 1, new float[1], 0);
            fail("RANK must be resolved before pooling");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void gatherCopiesSelectedRowsAndZeroesNegativeIndices() {
        Random r = new Random(3);
        int srcRows = 6, rowLen = 5, srcStride = 7, dstStride = 6;
        float[] src = random(r, 1 + srcRows * srcStride);
        int[] rows = {4, -1, 0, 4};
        float[] dst = new float[2 + rows.length * dstStride];
        Arrays.fill(dst, 9f);
        CpuPooling.gatherRows(rows.length, rowLen, rows, src, 1, srcStride, dst, 2, dstStride);
        for (int i = 0; i < rows.length; i++) {
            for (int c = 0; c < rowLen; c++) {
                float expect = rows[i] >= 0 ? src[1 + rows[i] * srcStride + c] : 0f;
                assertEquals(expect, dst[2 + i * dstStride + c], 0f);
            }
            for (int c = rowLen; c < dstStride; c++) {
                assertEquals(9f, dst[2 + i * dstStride + c], 0f);
            }
        }
    }

    @Test
    public void gatherRejectsRowsOutsideTheSource() {
        try {
            CpuPooling.gatherRows(1, 4, new int[] {3}, new float[12], 0, 4, new float[4], 0, 4);
            fail("row 3 of a 3-row source must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void viewsOfOneArrayMustNotOverlap() {
        float[] buf = new float[16];
        CpuPooling.pool(PoolingKind.MEAN, 2, 4, buf, 0, 4, buf, 8);
        try {
            CpuPooling.pool(PoolingKind.MEAN, 2, 4, buf, 0, 4, buf, 7);
            fail("pooling output overlapping its input must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        try {
            CpuPooling.gatherRows(1, 4, new int[] {1}, buf, 0, 4, buf, 6, 4);
            fail("a gathered source row overlapping the destination must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    /** ggml_mul_mat_id(lora_a, x, ids) then ggml_mul_mat_id(lora_b, a, ids), added to y. */
    private static void referenceLora(int nTokens, int inDim, int outDim, int rank, int nSlots, int[] slots,
                                      float[] a, float[] b, float[] x, int xStride, float[] y, int yStride) {
        for (int t = 0; t < nTokens; t++) {
            int s = Math.max(0, Math.min(nSlots - 1, slots[t]));
            float[] mid = new float[rank];
            for (int rr = 0; rr < rank; rr++) {
                float acc = 0f;
                for (int i = 0; i < inDim; i++) {
                    acc += a[(s * rank + rr) * inDim + i] * x[t * xStride + i];
                }
                mid[rr] = acc;
            }
            for (int o = 0; o < outDim; o++) {
                float acc = 0f;
                for (int rr = 0; rr < rank; rr++) {
                    acc += b[(s * outDim + o) * rank + rr] * mid[rr];
                }
                y[t * yStride + o] += acc;
            }
        }
    }

    @Test
    public void switchedLoraMatchesTwoMulMatId() {
        Random r = new Random(4);
        int nTokens = 5, inDim = 11, outDim = 9, rank = 3, nSlots = 4, xStride = 13, yStride = 12;
        int[] slots = {0, 3, 1, 7, -2};
        float[] a = random(r, nSlots * rank * inDim);
        float[] b = random(r, nSlots * outDim * rank);
        float[] x = random(r, nTokens * xStride);
        float[] y = random(r, nTokens * yStride);
        float[] ref = y.clone();
        float[] tmp = new float[1 + nTokens * rank];
        CpuPooling.switchedLoraAdd(nTokens, inDim, outDim, rank, nSlots, slots, a, b, x, 0, xStride, tmp, 1,
                y, 0, yStride);
        referenceLora(nTokens, inDim, outDim, rank, nSlots, slots, a, b, x, xStride, ref, yStride);
        assertArrayEquals(ref, y, EPS);
    }

    @Test
    public void loraGeometry() {
        assertEquals(3L * 8 * HeadsKernels.LORA_LOCAL,
                HeadsKernels.loraGlobal(HeadsKernels.LORA_STAGE_DOWN, 3, 8, 4096));
        assertEquals(12288L, HeadsKernels.loraGlobal(HeadsKernels.LORA_STAGE_UP, 3, 8, 4096));
        assertEquals(256L, HeadsKernels.loraGlobal(HeadsKernels.LORA_STAGE_UP, 1, 8, 5));
        assertEquals(256L, FlowKernels.gatherGlobal(2, 100));
        try {
            HeadsKernels.loraGlobal(2, 1, 1, 1);
            fail("unknown stage must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }
}
