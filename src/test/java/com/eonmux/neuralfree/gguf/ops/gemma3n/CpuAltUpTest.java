package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Reference values of the Gemma3n AltUp and activation-sparsity CPU math (the CPU twins of {@code LLM_AltUp_Predict},
 * {@code LLM_AltUp_Correct}, {@code LLM_MagnitudeMatch}, {@code LLM_StreamsMean} and {@code LLM_GaussianTopK}), plus
 * the argument layout of {@link Gemma3nKernels}.
 */
public class CpuAltUpTest {

    private static final float EPS = 1e-6f;

    @Test
    public void predictMixesStreamsWithPerTokenCoefficients() {
        // n_altup = 2, one token, one column: coefs[j*n_altup + k]
        float[] streams = {1f, 2f};
        float[] coefs = {0.5f, 0.25f, -1f, 2f};
        float[] pred = new float[2];
        CpuAltUp.predict(1, 1, 2, 1, coefs, 0, streams, 0, pred, 0);
        assertArrayEquals(new float[]{1f * 0.5f + 2f * 0.25f + 1f, 1f * -1f + 2f * 2f + 2f}, pred, 0f);
    }

    @Test
    public void predictHonoursStreamStrideAndRunsInPlace() {
        int nTokens = 2;
        int nEmbd = 3;
        int nAltup = 3;
        int stride = 8;
        float[] streams = new float[nAltup * stride];
        for (int k = 0; k < nAltup; k++) {
            for (int i = 0; i < nTokens * nEmbd; i++) {
                streams[k * stride + i] = 0.1f * (k + 1) + 0.01f * i;
            }
        }
        float[] coefs = new float[nTokens * nAltup * nAltup];
        for (int i = 0; i < coefs.length; i++) {
            coefs[i] = (i % 5) * 0.2f - 0.3f;
        }
        float[] expected = new float[streams.length];
        for (int t = 0; t < nTokens; t++) {
            for (int e = 0; e < nEmbd; e++) {
                int idx = t * nEmbd + e;
                for (int j = 0; j < nAltup; j++) {
                    float acc = 0f;
                    for (int k = 0; k < nAltup; k++) {
                        acc += streams[k * stride + idx] * coefs[t * nAltup * nAltup + j * nAltup + k];
                    }
                    expected[j * stride + idx] = acc + streams[j * stride + idx];
                }
            }
        }
        float[] separate = new float[streams.length];
        CpuAltUp.predict(nTokens, nEmbd, nAltup, stride, coefs, 0, streams, 0, separate, 0);
        assertArrayEquals(expected, separate, 0f);

        CpuAltUp.predict(nTokens, nEmbd, nAltup, stride, coefs, 0, streams, 0, streams, 0);
        for (int k = 0; k < nAltup; k++) {
            for (int i = 0; i < nTokens * nEmbd; i++) {
                assertEquals(expected[k * stride + i], streams[k * stride + i], 0f);
            }
        }
    }

    @Test
    public void correctAddsScaledInnovationToEveryStream() {
        // n_altup = 2, act = 0: innov = 3 - 2 = 1; c = raw + 1
        float[] coefs = {0.5f, -1f};
        float[] activated = {3f};
        float[] pred = {2f, 5f};
        float[] out = new float[2];
        CpuAltUp.correct(1, 1, 2, 0, 1, coefs, 0, activated, 0, pred, 0, out, 0);
        assertArrayEquals(new float[]{1f * 1.5f + 2f, 1f * 0f + 5f}, out, 0f);

        CpuAltUp.correct(1, 1, 2, 0, 1, coefs, 0, activated, 0, pred, 0, pred, 0);
        assertArrayEquals(out, pred, 0f);
    }

    @Test
    public void correctUsesTheActiveStream() {
        float[] coefs = {0f, 0f, 1f};
        float[] activated = {10f};
        float[] pred = {1f, 4f, 2f};
        float[] out = new float[3];
        CpuAltUp.correct(1, 1, 3, 1, 1, coefs, 0, activated, 0, pred, 0, out, 0);
        // innov = 10 - pred[1] = 6
        assertArrayEquals(new float[]{6f + 1f, 6f + 4f, 6f * 2f + 2f}, out, 0f);
    }

    @Test
    public void magnitudeMatchRescalesEachRowToTheReferenceNorm() {
        int nTokens = 2;
        int nEmbd = 2;
        float[] ref = {3f, 4f, 0f, 2f};          // |ref[0]| = 5, |ref[1]| = 2
        float[] x = {1f, 0f, 0f, 1f, 0f, 3f, 6f, 8f};   // two x rows per token, x_stride = 4
        CpuAltUp.magnitudeMatch(nTokens, nEmbd, 2, 4, ref, 0, x, 0);
        assertArrayEquals(new float[]{5f, 0f, 0f, 2f, 0f, 5f, 6f * 2f / 10f, 8f * 2f / 10f}, x, EPS);
        assertEquals(5f, CpuAltUp.magnitude(x, 0, 2), EPS);
    }

    @Test
    public void streamsMeanAveragesEveryStream() {
        float[] streams = {1f, 2f, 0f, 3f, 4f, 0f, 5f, 9f};   // n_altup = 3, stride 3 (one token, two columns)
        float[] out = new float[2];
        CpuAltUp.streamsMean(1, 2, 3, 3, streams, 0, out, 0);
        assertArrayEquals(new float[]{(1f + 3f + 5f) * (1f / 3f), (2f + 4f + 9f) * (1f / 3f)}, out, 0f);

        CpuAltUp.streamsMean(1, 2, 3, 3, streams, 0, streams, 0);
        assertEquals(out[0], streams[0], 0f);
        assertEquals(out[1], streams[1], 0f);
    }

    @Test
    public void gaussianTopKUsesTheUnbiasedDeviation() {
        float[] x = {1f, 2f, 3f, 4f};
        float mean = 2.5f;
        float std = (float) Math.sqrt(5.0f * (1f / 3f));
        assertEquals(mean + std * 0.5f, CpuActivationSparsity.cutoff(x, 0, 4, 0.5f), EPS);

        CpuActivationSparsity.gaussianTopK(1, 4, 4, 0, 0.5f, x);
        assertArrayEquals(new float[]{0f, 0f, 0f, 4f - (mean + std * 0.5f)}, x, EPS);
    }

    @Test
    public void gaussianTopKFiltersOnlyTheGateHalfOfFusedRows() {
        float[] x = {1f, 2f, 3f, 4f, 9f, 9f, 9f, 9f,
                     4f, 4f, 4f, 8f, 7f, 7f, 7f, 7f};
        CpuActivationSparsity.gaussianTopK(2, 4, 8, 0, CpuActivationSparsity.GEMMA3N_STD_MUL, x);
        for (int t = 0; t < 2; t++) {
            for (int i = 4; i < 8; i++) {
                assertTrue(x[t * 8 + i] == 9f || x[t * 8 + i] == 7f);
            }
        }
        // second row: mean 5, std 2, cutoff 5 + 2*1.6448534 = 8.2897 -> all gated values are cut
        assertArrayEquals(new float[]{0f, 0f, 0f, 0f}, new float[]{x[8], x[9], x[10], x[11]}, 0f);
    }

    @Test
    public void overlappingViewsAreRejected() {
        float[] buf = new float[16];
        float[] coefs = new float[4];
        try {
            CpuAltUp.predict(1, 2, 2, 4, coefs, 0, buf, 0, buf, 1);
            fail("shifted in-place prediction must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("overlaps"));
        }
        try {
            CpuAltUp.magnitudeMatch(1, 4, 1, 4, buf, 0, buf, 2);
            fail("x overlapping ref must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("overlaps"));
        }
        try {
            CpuAltUp.predict(1, 1, Gemma3nKernels.MAX_ALTUP + 1, 1, new float[100], 0, new float[16], 0, new float[16], 0);
            fail("n_altup above the kernel maximum must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("n_altup"));
        }
    }

    @Test
    public void argumentListsFollowTheKernelSignatures() {
        FakeGpuBuffer streams = new FakeGpuBuffer(1, 4L * 4 * 64, false);
        FakeGpuBuffer coefs = new FakeGpuBuffer(2, 4L * 64, false);
        FakeGpuBuffer act = new FakeGpuBuffer(3, 4L * 64, false);
        Object[] p = Gemma3nKernels.predictArgs(2, 8, 4, 64, coefs, 0, streams, 0, streams, 0);
        assertEquals(Gemma3nKernels.PREDICT_ARG_COUNT, p.length);
        assertEquals(2, p[Gemma3nKernels.ARG_N_TOKENS]);
        Object[] c = Gemma3nKernels.correctArgs(2, 8, 4, 0, 64, coefs, 0, act, 0, streams, 0, streams, 0);
        assertEquals(Gemma3nKernels.CORRECT_ARG_COUNT, c.length);
        Object[] m = Gemma3nKernels.magnitudeMatchArgs(2, 8, 3, 64, streams, 0, streams, 64);
        assertEquals(Gemma3nKernels.MAGNITUDE_ARG_COUNT, m.length);
        Object[] g = Gemma3nKernels.gaussianTopKArgs(2, 16, 32, 0, CpuActivationSparsity.GEMMA3N_STD_MUL, act);
        assertEquals(Gemma3nKernels.TOPK_ARG_COUNT, g.length);
        assertEquals(CpuActivationSparsity.GEMMA3N_STD_MUL, g[Gemma3nKernels.ARG_TOPK_K_STD]);
        Object[] s = Gemma3nKernels.streamsMeanArgs(2, 8, 4, 64, streams, 0, streams, 0);
        assertEquals(Gemma3nKernels.MEAN_ARG_COUNT, s.length);

        assertArrayEquals(new long[]{256}, Gemma3nKernels.elementwiseGlobal(2, 8, 256));
        assertArrayEquals(new long[]{6 * 64}, Gemma3nKernels.rowGlobal(6, 64));
        assertEquals(256, Gemma3nKernels.rowLocal(1024));
        assertEquals(64, Gemma3nKernels.rowLocal(64));
        assertEquals(32, Gemma3nKernels.rowLocal(8));
    }
}
