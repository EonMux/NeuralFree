package com.eonmux.neuralfree.gguf.ops.ssm;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Reference checks of the SSM CPU math against direct transcriptions of the llama.cpp operations
 * (ggml_ssm_conv on {@code concat(state, x)}, ggml_ssm_scan, the Mamba D/z gate and the LFM2 centered taps),
 * the decode/prefill equivalence of the stateful ops, and the scan launch geometry.
 */
public class CpuSsmTest {

    private static final float EPS = 1e-5f;

    private static float[] random(Random r, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (r.nextFloat() * 2f - 1f) * scale;
        }
        return a;
    }

    private static float silu(float v) {
        return v / (1f + (float) Math.exp(-v));
    }

    private static float softplus(float v) {
        return v > 20f ? v : (float) Math.log(1f + (float) Math.exp(v));
    }

    /** ggml_ssm_conv over xs = concat(state[c], x[.][c]) plus bias and silu, and the new state. */
    private static float[] referenceConv(int nCh, int dConv, int nTokens, float[] x, float[] w, float[] b,
                                         boolean silu, float[] state, float[] newState) {
        int steps = dConv - 1;
        float[] out = new float[nTokens * nCh];
        for (int c = 0; c < nCh; c++) {
            float[] xs = new float[steps + nTokens];
            System.arraycopy(state, c * steps, xs, 0, steps);
            for (int t = 0; t < nTokens; t++) {
                xs[steps + t] = x[t * nCh + c];
            }
            for (int t = 0; t < nTokens; t++) {
                float sum = 0f;
                for (int k = 0; k < dConv; k++) {
                    sum += xs[t + k] * w[c * dConv + k];
                }
                if (b != null) {
                    sum += b[c];
                }
                out[t * nCh + c] = silu ? silu(sum) : sum;
            }
            System.arraycopy(xs, nTokens, newState, c * steps, steps);
        }
        return out;
    }

    @Test
    public void convMatchesConcatReference() {
        Random r = new Random(1);
        int nCh = 37, dConv = 4, nTokens = 6;
        float[] x = random(r, nTokens * nCh, 1f);
        float[] w = random(r, nCh * dConv, 1f);
        float[] b = random(r, nCh, 0.5f);
        float[] state = random(r, nCh * (dConv - 1), 1f);
        float[] expectState = new float[state.length];
        float[] expect = referenceConv(nCh, dConv, nTokens, x, w, b, true, state, expectState);

        float[] out = new float[nTokens * nCh];
        float[] live = state.clone();
        CpuSsmConv.forward(nCh, dConv, nTokens, SsmKernels.CONV_FLAG_BIAS | SsmKernels.CONV_FLAG_SILU,
                x, 0, nCh, w, b, live, 0, 0, out, 0, nCh);
        assertArrayEquals(expect, out, EPS);
        assertArrayEquals(expectState, live, 0f);
    }

    @Test
    public void convShortPrefillKeepsOldStateTail() {
        Random r = new Random(2);
        int nCh = 5, dConv = 4, nTokens = 1;
        float[] x = random(r, nCh, 1f);
        float[] w = random(r, nCh * dConv, 1f);
        float[] state = random(r, nCh * (dConv - 1), 1f);
        float[] expectState = new float[state.length];
        float[] expect = referenceConv(nCh, dConv, nTokens, x, w, null, false, state, expectState);
        float[] out = new float[nCh];
        CpuSsmConv.forward(nCh, dConv, nTokens, 0, x, 0, nCh, w, null, state, 0, 0, out, 0, nCh);
        assertArrayEquals(expect, out, EPS);
        assertArrayEquals(expectState, state, 0f);
    }

    @Test
    public void convPrefillEqualsTokenByTokenDecode() {
        Random r = new Random(3);
        int nCh = 19, dConv = 3, nTokens = 7;
        float[] x = random(r, nTokens * nCh, 1f);
        float[] w = random(r, nCh * dConv, 1f);
        float[] state0 = random(r, nCh * (dConv - 1), 1f);

        float[] prefillState = state0.clone();
        float[] prefill = new float[nTokens * nCh];
        CpuSsmConv.forward(nCh, dConv, nTokens, SsmKernels.CONV_FLAG_SILU, x, 0, nCh, w, null,
                prefillState, 0, 0, prefill, 0, nCh);

        float[] decodeState = state0.clone();
        float[] decode = new float[nTokens * nCh];
        for (int t = 0; t < nTokens; t++) {
            CpuSsmConv.forward(nCh, dConv, 1, SsmKernels.CONV_FLAG_SILU, x, t * nCh, nCh, w, null,
                    decodeState, 0, 0, decode, t * nCh, nCh);
        }
        assertArrayEquals(prefill, decode, 0f);
        assertArrayEquals(prefillState, decodeState, 0f);
    }

    @Test
    public void convSliceAddressesItsChannelsOfTheStateRow() {
        Random r = new Random(4);
        int dConv = 4, steps = dConv - 1, rowCh = 12, sliceOff = 4, nCh = 5, nTokens = 3, rowStride = rowCh * steps;
        float[] x = random(r, nTokens * 20, 1f);
        float[] w = random(r, nCh * dConv, 1f);
        float[] stateAll = random(r, 2 * rowStride, 1f);
        float[] before = stateAll.clone();

        float[] slice = Arrays.copyOfRange(stateAll, rowStride + sliceOff * steps, rowStride + (sliceOff + nCh) * steps);
        float[] xs = new float[nTokens * nCh];
        for (int t = 0; t < nTokens; t++) {
            System.arraycopy(x, 3 + t * 20, xs, t * nCh, nCh);
        }
        float[] expectSlice = new float[slice.length];
        float[] expect = referenceConv(nCh, dConv, nTokens, xs, w, null, false, slice, expectSlice);

        float[] out = new float[nTokens * 8];
        CpuSsmConv.forward(nCh, dConv, nTokens, 0, x, 3, 20, w, null, stateAll, rowStride, sliceOff, out, 1, 8);
        for (int t = 0; t < nTokens; t++) {
            for (int c = 0; c < nCh; c++) {
                assertEquals(expect[t * nCh + c], out[1 + t * 8 + c], EPS);
            }
        }
        for (int i = 0; i < stateAll.length; i++) {
            int rel = i - (rowStride + sliceOff * steps);
            if (rel >= 0 && rel < nCh * steps) {
                assertEquals(expectSlice[rel], stateAll[i], 0f);
            } else {
                assertEquals("untouched state float " + i, before[i], stateAll[i], 0f);
            }
        }
    }

    /** ggml_ssm_scan transcription (ops.cpp:10092-10308) with the dt bias added first. */
    private static float[] referenceScan(int dState, int headDim, int nHead, int nGroup, int nTokens, boolean aPerState,
                                         float[] x, float[] dt, float[] dtBias, float[] a, float[] bMat, float[] cMat,
                                         float[] s) {
        int nCh = nHead * headDim;
        float[] y = new float[nTokens * nCh];
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < nHead; h++) {
                float dtSp = softplus(dt[t * nHead + h] + (dtBias != null ? dtBias[h] : 0f));
                int g = h / (nHead / nGroup);
                for (int p = 0; p < headDim; p++) {
                    int ii = p + h * headDim;
                    float xDt = x[t * nCh + ii] * dtSp;
                    float sum = 0f;
                    for (int n = 0; n < dState; n++) {
                        float dA = (float) Math.exp(dtSp * (aPerState ? a[n + h * dState] : a[h]));
                        int ig = t * nGroup * dState + n + g * dState;
                        float st = s[n + ii * dState] * dA + bMat[ig] * xDt;
                        sum += st * cMat[ig];
                        s[n + ii * dState] = st;
                    }
                    y[t * nCh + ii] = sum;
                }
            }
        }
        return y;
    }

    private void checkScan(long seed, int dState, int headDim, int nHead, int nGroup, int nTokens, boolean aPerState) {
        Random r = new Random(seed);
        int nCh = nHead * headDim;
        float[] x = random(r, nTokens * nCh, 1f);
        float[] dt = random(r, nTokens * nHead, 2f);
        float[] dtBias = random(r, nHead, 1f);
        float[] a = random(r, aPerState ? nHead * dState : nHead, 1f);
        for (int i = 0; i < a.length; i++) {
            a[i] = -Math.abs(a[i]) - 0.1f;
        }
        float[] bMat = random(r, nTokens * nGroup * dState, 1f);
        float[] cMat = random(r, nTokens * nGroup * dState, 1f);
        float[] s0 = random(r, nCh * dState, 1f);

        float[] sRef = s0.clone();
        float[] yRef = referenceScan(dState, headDim, nHead, nGroup, nTokens, aPerState, x, dt, dtBias, a, bMat, cMat, sRef);

        float[] s = new float[nCh * dState + 3];
        System.arraycopy(s0, 0, s, 3, s0.length);
        float[] y = new float[nTokens * nCh];
        CpuSsmScan.scan(dState, headDim, nHead, nGroup, nTokens, aPerState, x, 0, nCh, dt, 0, nHead, dtBias, a,
                bMat, 0, nGroup * dState, cMat, 0, nGroup * dState, s, 3, y, 0, nCh);
        assertArrayEquals(yRef, y, EPS);
        assertArrayEquals(sRef, Arrays.copyOfRange(s, 3, s.length), EPS);

        float[] sDec = s0.clone();
        float[] yDec = new float[nTokens * nCh];
        for (int t = 0; t < nTokens; t++) {
            CpuSsmScan.scan(dState, headDim, nHead, nGroup, 1, aPerState, x, t * nCh, nCh, dt, t * nHead, nHead, dtBias,
                    a, bMat, t * nGroup * dState, nGroup * dState, cMat, t * nGroup * dState, nGroup * dState,
                    sDec, 0, yDec, t * nCh, nCh);
        }
        assertArrayEquals(y, yDec, 0f);
        assertArrayEquals(Arrays.copyOfRange(s, 3, s.length), sDec, 0f);
    }

    @Test
    public void mamba1ScanMatchesReference() {
        checkScan(10, 16, 1, 24, 1, 5, true);
    }

    @Test
    public void mamba2ScanMatchesReference() {
        checkScan(11, 32, 8, 6, 2, 4, false);
    }

    @Test
    public void scanReadsInterleavedBcViews() {
        Random r = new Random(12);
        int dState = 8, headDim = 4, nHead = 2, nGroup = 1, nTokens = 3, nCh = nHead * headDim;
        int rowW = nCh + 2 * dState;
        float[] xbc = random(r, nTokens * rowW, 1f);
        float[] dt = random(r, nTokens * nHead, 1f);
        float[] a = new float[] {-0.5f, -1.5f};
        float[] s0 = random(r, nCh * dState, 1f);

        float[] x = new float[nTokens * nCh];
        float[] bMat = new float[nTokens * dState];
        float[] cMat = new float[nTokens * dState];
        for (int t = 0; t < nTokens; t++) {
            System.arraycopy(xbc, t * rowW, x, t * nCh, nCh);
            System.arraycopy(xbc, t * rowW + nCh, bMat, t * dState, dState);
            System.arraycopy(xbc, t * rowW + nCh + dState, cMat, t * dState, dState);
        }
        float[] sRef = s0.clone();
        float[] yRef = referenceScan(dState, headDim, nHead, nGroup, nTokens, false, x, dt, null, a, bMat, cMat, sRef);

        float[] s = s0.clone();
        float[] y = new float[nTokens * nCh];
        CpuSsmScan.scan(dState, headDim, nHead, nGroup, nTokens, false, xbc, 0, rowW, dt, 0, nHead, null, a,
                xbc, nCh, rowW, xbc, nCh + dState, rowW, s, 0, y, 0, nCh);
        assertArrayEquals(yRef, y, EPS);
        assertArrayEquals(sRef, s, EPS);
    }

    @Test
    public void gateOutMatchesMambaGraph() {
        Random r = new Random(20);
        int nTokens = 3, headDim = 4, nHead = 3, width = headDim * nHead, zStride = 2 * width + 5;
        float[] y = random(r, nTokens * width, 1f);
        float[] x = random(r, nTokens * (width + 7), 1f);
        float[] d = random(r, nHead, 1f);
        float[] z = random(r, nTokens * zStride, 2f);
        float[] out = new float[2 + nTokens * width];
        CpuSsmScan.gateOut(nTokens, width, headDim, y, 0, x, 1, width + 7, d, z, width, zStride, out, 2);
        for (int t = 0; t < nTokens; t++) {
            for (int c = 0; c < width; c++) {
                float yd = y[t * width + c] + x[1 + t * (width + 7) + c] * d[c / headDim];
                float expect = silu(z[width + t * zStride + c]) * yd;
                assertEquals(expect, out[2 + t * width + c], EPS);
            }
        }
    }

    @Test
    public void centeredConvMatchesLfm2Taps() {
        Random r = new Random(30);
        int nCh = 9, dConv = 3, nTokens = 5;
        float[] bx = random(r, nTokens * nCh, 1f);
        float[] w = random(r, nCh * dConv, 1f);
        float[] out = new float[nTokens * nCh];
        CpuShortConv.centered(nCh, dConv, nTokens, bx, 0, nCh, w, out, 0, nCh);
        for (int t = 0; t < nTokens; t++) {
            for (int c = 0; c < nCh; c++) {
                float left = t > 0 ? 1f : 0f;
                float right = t + 1 < nTokens ? 1f : 0f;
                float prv = t > 0 ? bx[(t - 1) * nCh + c] : 0f;
                float nxt = t + 1 < nTokens ? bx[(t + 1) * nCh + c] : 0f;
                float yv = bx[t * nCh + c] * w[c * dConv + 1];
                yv += prv * w[c * dConv] * left;
                yv += nxt * w[c * dConv + 2] * right;
                assertEquals(yv, out[t * nCh + c], EPS);
            }
        }
    }

    @Test
    public void gatedCausalMatchesLfmShortconv() {
        Random r = new Random(31);
        int nEmbd = 6, lCache = 3, nTokens = 4;
        float[] bcx = random(r, nTokens * 3 * nEmbd, 1f);
        float[] w = random(r, nEmbd * lCache, 1f);
        float[] state = random(r, nEmbd * (lCache - 1), 1f);

        float[] bxRef = new float[nTokens * nEmbd];
        for (int t = 0; t < nTokens; t++) {
            for (int c = 0; c < nEmbd; c++) {
                bxRef[t * nEmbd + c] = bcx[t * 3 * nEmbd + c] * bcx[t * 3 * nEmbd + 2 * nEmbd + c];
            }
        }
        float[] stateRef = new float[state.length];
        float[] conv = referenceConv(nEmbd, lCache, nTokens, bxRef, w, null, false, state, stateRef);

        float[] out = new float[nTokens * nEmbd];
        CpuShortConv.gatedCausal(nEmbd, lCache, nTokens, bcx, 0, 3 * nEmbd, w, state, 0,
                new float[nTokens * nEmbd], out, 0, nEmbd);
        for (int t = 0; t < nTokens; t++) {
            for (int c = 0; c < nEmbd; c++) {
                assertEquals(bcx[t * 3 * nEmbd + nEmbd + c] * conv[t * nEmbd + c], out[t * nEmbd + c], EPS);
            }
        }
        assertArrayEquals(stateRef, state, 0f);
    }

    @Test
    public void scanGeometryPacksSmallStates() {
        SsmKernels.ScanGeometry m1 = SsmKernels.ScanGeometry.of(16, 1, 4096, 1);
        assertEquals(16, m1.lanes());
        assertEquals(2, m1.channelsPerGroup());
        assertEquals(32, m1.localSize());
        assertEquals(2048L * 32, m1.globalSize());
        assertEquals(1, m1.statesPerLane());

        SsmKernels.ScanGeometry m2 = SsmKernels.ScanGeometry.of(128, 64, 80, 8);
        assertEquals(64, m2.lanes());
        assertEquals(1, m2.channelsPerGroup());
        assertEquals(64, m2.localSize());
        assertEquals(2, m2.statesPerLane());
        assertEquals((long) 80 * 64 * 64, m2.globalSize());

        SsmKernels.ScanGeometry odd = SsmKernels.ScanGeometry.of(48, 2, 3, 1);
        assertEquals(64, odd.lanes());
        assertEquals(1, odd.statesPerLane());

        SsmKernels.ScanGeometry tiny = SsmKernels.ScanGeometry.of(8, 1, 5, 1);
        assertEquals(4, tiny.channelsPerGroup());
        assertEquals(32, tiny.localSize());
        assertEquals(2L * 32, tiny.globalSize());

        SsmKernels.ScanGeometry big = SsmKernels.ScanGeometry.of(256, 64, 4, 1);
        assertEquals(4, big.statesPerLane());
        assertTrue(big.statesPerLane() <= SsmKernels.SCAN_MAX_STATES_PER_LANE);
    }

    @Test
    public void invalidShapesAreRejected() {
        expectIllegal(() -> SsmKernels.ScanGeometry.of(16, 4, 6, 4));
        expectIllegal(() -> SsmKernels.ScanGeometry.of(1024, 1, 1, 1));
        expectIllegal(() -> SsmKernels.checkCenteredShape(8, 4));
        expectIllegal(() -> SsmKernels.checkConvShape(8, 1));
        expectIllegal(() -> CpuSsmConv.forward(4, 4, 2, 4, new float[8], 0, 4, new float[16], null,
                new float[12], 0, 0, new float[8], 0, 4));
        expectIllegal(() -> CpuSsmConv.forward(4, 4, 2, 0, new float[7], 0, 4, new float[16], null,
                new float[12], 0, 0, new float[8], 0, 4));
        float[] bcx = new float[2 * 3 * 4];
        expectIllegal(() -> CpuShortConv.gatedCausal(4, 3, 2, bcx, 0, 12, new float[12], new float[8], 0,
                bcx, new float[8], 0, 4));
        expectIllegal(() -> CpuShortConv.gatedCentered(4, 3, 2, bcx, 0, 12, new float[12], new float[8],
                bcx, 0, 4));
    }

    private static void expectIllegal(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage() != null && !expected.getMessage().isEmpty());
        }
    }
}
