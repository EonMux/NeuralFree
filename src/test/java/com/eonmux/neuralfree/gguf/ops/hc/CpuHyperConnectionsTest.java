package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.CoefficientParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PostParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PreParams;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link CpuHyperConnections}, {@link CpuAttnResidualStack} and the helpers of {@link HcKernels}, against
 * hand-computed values and independent transcriptions of the llama.cpp graph formulations
 * (src/models/deepseek4.cpp:279-446, hy-v4.cpp:175-245, qwen4exp.cpp:369-395, kimi-k3.cpp:198-232).
 */
public class CpuHyperConnectionsTest {

    private static final float EPS = 1e-5f;

    private static float[] random(Random rnd, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (rnd.nextFloat() * 2f - 1f) * scale;
        }
        return a;
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    /**
     * Graph form of build_hc_sinkhorn (deepseek4.cpp:316-351) over a [dst][src] matrix m[src][dst]: softmax over dst
     * (ne0), + eps, column sums over src per dst, then (iters - 1) x (row sums over dst per src, column sums).
     */
    private static float[][] graphSinkhorn(float[][] logits, float eps, int iters) {
        int hc = logits.length;
        float[][] m = new float[hc][hc];
        for (int src = 0; src < hc; src++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int dst = 0; dst < hc; dst++) {
                max = Math.max(max, logits[src][dst]);
            }
            float sum = 0f;
            for (int dst = 0; dst < hc; dst++) {
                m[src][dst] = (float) Math.exp(logits[src][dst] - max);
                sum += m[src][dst];
            }
            for (int dst = 0; dst < hc; dst++) {
                m[src][dst] = m[src][dst] * (1f / sum) + eps;
            }
        }
        normColumns(m, eps);
        for (int it = 1; it < iters; it++) {
            for (int src = 0; src < hc; src++) {
                float sum = eps;
                for (int dst = 0; dst < hc; dst++) {
                    sum += m[src][dst];
                }
                for (int dst = 0; dst < hc; dst++) {
                    m[src][dst] *= 1f / sum;
                }
            }
            normColumns(m, eps);
        }
        return m;
    }

    private static void normColumns(float[][] m, float eps) {
        int hc = m.length;
        for (int dst = 0; dst < hc; dst++) {
            float sum = eps;
            for (int src = 0; src < hc; src++) {
                sum += m[src][dst];
            }
            for (int src = 0; src < hc; src++) {
                m[src][dst] *= 1f / sum;
            }
        }
    }

    @Test
    public void mhcCoefficientsMatchTheGraphFormulation() {
        Random rnd = new Random(7);
        int hc = 4;
        int nTokens = 3;
        float eps = 1e-6f;
        int iters = 20;
        CoefficientParams p = CoefficientParams.mhc(hc, eps, iters);
        assertEquals(24, p.mixDim());
        float[] mixes = random(rnd, nTokens * p.mixDim(), 3f);
        float[] scale = {0.7f, 1.3f, 0.9f};
        float[] base = random(rnd, p.baseCount(), 1f);
        float[] pre = new float[nTokens * hc];
        float[] post = new float[nTokens * hc];
        float[] comb = new float[nTokens * hc * hc];
        CpuHyperConnections.coefficients(p, 0, nTokens, mixes, 0, scale, base, pre, 0, post, 0, comb, 0);

        for (int t = 0; t < nTokens; t++) {
            int m = t * p.mixDim();
            float[][] logits = new float[hc][hc];
            for (int h = 0; h < hc; h++) {
                assertEquals(sigmoid(mixes[m + h] * scale[0] + base[h]) + eps, pre[t * hc + h], 1e-6f);
                assertEquals(2f * sigmoid(mixes[m + hc + h] * scale[1] + base[hc + h]), post[t * hc + h], 1e-6f);
            }
            for (int src = 0; src < hc; src++) {
                for (int dst = 0; dst < hc; dst++) {
                    int idx = 2 * hc + dst + hc * src;
                    logits[src][dst] = mixes[m + idx] * scale[2] + base[idx];
                }
            }
            float[][] ref = graphSinkhorn(logits, eps, iters);
            for (int src = 0; src < hc; src++) {
                for (int dst = 0; dst < hc; dst++) {
                    assertEquals("t=" + t + " dst=" + dst + " src=" + src, ref[src][dst],
                            comb[t * hc * hc + dst + hc * src], 1e-6f);
                }
            }
        }
    }

    @Test
    public void sinkhornConvergesToADoublyStochasticMatrix() {
        Random rnd = new Random(11);
        int hc = 4;
        float[] mixes = random(rnd, 16, 4f);
        float[] base = new float[16];
        float[] c = new float[16];
        CpuHyperConnections.combFromLogits(hc, 1e-9f, 200, mixes, 0, 1f, base, 0, c);
        for (int a = 0; a < hc; a++) {
            float rowSum = 0f;
            float colSum = 0f;
            for (int b = 0; b < hc; b++) {
                rowSum += c[b + hc * a];
                colSum += c[a + hc * b];
            }
            assertEquals(1f, rowSum, 1e-4f);
            assertEquals(1f, colSum, 1e-4f);
        }
    }

    @Test
    public void singleIterationNormalizesColumnsOnly() {
        // iters = 1: softmax over dst for each src, + eps, then one norm_cols (sum over src per dst).
        int hc = 2;
        float eps = 0f;
        float[] logits = {0f, 0f, (float) Math.log(3.0), 0f};
        float[] c = new float[4];
        CpuHyperConnections.combFromLogits(hc, eps, 1, logits, 0, 1f, new float[4], 0, c);
        // softmax: src0 -> [0.5, 0.5], src1 -> [0.75, 0.25]; columns: dst0 sum 1.25, dst1 sum 0.75.
        assertEquals(0.5f / 1.25f, c[0], 1e-6f);
        assertEquals(0.5f / 0.75f, c[1], 1e-6f);
        assertEquals(0.75f / 1.25f, c[2], 1e-6f);
        assertEquals(0.25f / 0.75f, c[3], 1e-6f);
    }

    @Test
    public void ihcAndHeadVariantsWriteOnlyTheirOutputs() {
        int hc = 2;
        float eps = 0.01f;
        float[] mixes = {0f, 1f, 2f, -1f};
        float[] scale = {1f, 0.5f};
        float[] base = {0f, 0f, 0f, 0f};
        float[] pre = new float[2];
        float[] post = new float[2];
        CpuHyperConnections.coefficients(CoefficientParams.ihc(hc, eps, 3f), 0, 1, mixes, 0, scale, base,
                pre, 0, post, 0, null, 0);
        assertEquals(sigmoid(0f) + eps, pre[0], 1e-6f);
        assertEquals(sigmoid(1f) + eps, pre[1], 1e-6f);
        assertEquals(3f * sigmoid(1f) + eps, post[0], 1e-6f);
        assertEquals(3f * sigmoid(-0.5f) + eps, post[1], 1e-6f);

        float[] headPre = new float[4];
        CoefficientParams head = CoefficientParams.head(hc, eps);
        assertEquals(2, head.mixDim());
        CpuHyperConnections.coefficients(head, 0, 2, new float[]{0f, 1f, 2f, 3f}, 0, new float[]{2f},
                new float[]{0.5f, -0.5f}, headPre, 0, null, 0, null, 0);
        assertEquals(sigmoid(0.5f) + eps, headPre[0], 1e-6f);
        assertEquals(sigmoid(1.5f) + eps, headPre[1], 1e-6f);
        assertEquals(sigmoid(4.5f) + eps, headPre[2], 1e-6f);
        assertEquals(sigmoid(5.5f) + eps, headPre[3], 1e-6f);
    }

    @Test
    public void preModesMatchHandComputation() {
        int hc = 2;
        int nEmbd = 3;
        // token 0: streams [1,2,3] and [4,5,6]
        float[] x = {1f, 2f, 3f, 4f, 5f, 6f};
        float[] out = new float[3];
        CpuHyperConnections.preRange(PreParams.weights(), hc, nEmbd, 0, 1, x, 0, new float[]{0.5f, 2f}, 0, out, 0);
        assertArrayEquals(new float[]{8.5f, 11f, 13.5f}, out, 1e-6f);

        CpuHyperConnections.preRange(PreParams.mean(hc), hc, nEmbd, 0, 1, x, 0, null, 0, out, 0);
        assertArrayEquals(new float[]{2.5f, 3.5f, 4.5f}, out, 1e-6f);

        float[] gate = {0f, 0f, 0f, 100f, 100f, -100f};
        CpuHyperConnections.preRange(PreParams.gated(hc), hc, nEmbd, 0, 1, x, 0, gate, 0, out, 0);
        assertEquals(0.5f * (0.5f * 1f + 4f), out[0], 1e-5f);
        assertEquals(0.5f * (0.5f * 2f + 5f), out[1], 1e-5f);
        assertEquals(0.5f * (0.5f * 3f), out[2], 1e-5f);
    }

    @Test
    public void postWithCombMatchesTheGraphFormulationAndRunsInPlace() {
        Random rnd = new Random(3);
        int hc = 4;
        int nEmbd = 5;
        int nTokens = 2;
        float[] y = random(rnd, nTokens * nEmbd, 1f);
        float[] x = random(rnd, nTokens * hc * nEmbd, 1f);
        float[] post = random(rnd, nTokens * hc, 1f);
        float[] comb = random(rnd, nTokens * hc * hc, 1f);
        float[] out = new float[x.length];
        CpuHyperConnections.postRange(PostParams.mixing(), hc, nEmbd, 0, nTokens, y, 0, x, 0, post, 0, comb, 0, out, 0);
        for (int t = 0; t < nTokens; t++) {
            for (int dst = 0; dst < hc; dst++) {
                for (int i = 0; i < nEmbd; i++) {
                    // build_hc_post (deepseek4.cpp:429-441): cur = x*post[dst] + sum_src res[src]*comb[dst + hc*src]
                    float expect = y[t * nEmbd + i] * post[t * hc + dst];
                    for (int src = 0; src < hc; src++) {
                        expect += x[(t * hc + src) * nEmbd + i] * comb[t * hc * hc + dst + hc * src];
                    }
                    assertEquals(expect, out[(t * hc + dst) * nEmbd + i], 1e-6f);
                }
            }
        }
        float[] inPlace = x.clone();
        CpuHyperConnections.postRange(PostParams.mixing(), hc, nEmbd, 0, nTokens, y, 0, inPlace, 0, post, 0, comb, 0,
                inPlace, 0);
        assertArrayEquals(out, inPlace, 0f);
    }

    @Test
    public void identityAndGatedInjectPostAddEachStreamItsOwnResidual() {
        int hc = 2;
        int nEmbd = 2;
        float[] y = {1f, 2f};
        float[] x = {10f, 20f, 30f, 40f};
        float[] post = {0.5f, -1f};
        float[] out = new float[4];
        CpuHyperConnections.postRange(PostParams.identity(), hc, nEmbd, 0, 1, y, 0, x, 0, post, 0, null, 0, out, 0);
        assertArrayEquals(new float[]{10.5f, 21f, 29f, 38f}, out, 1e-6f);

        // qwen4exp combine (qwen4exp.cpp:376-391): w = 2*sigmoid(inject/hc), out = residual + block_out*w.
        float[] inject = {0f, 4f};
        CpuHyperConnections.postRange(PostParams.gatedInject(hc), hc, nEmbd, 0, 1, y, 0, x, 0, inject, 0, null, 0,
                out, 0);
        float w1 = 2f * sigmoid(2f);
        assertArrayEquals(new float[]{11f, 22f, 30f + w1, 40f + 2f * w1}, out, 1e-5f);
    }

    @Test
    public void expandStreamsRepeatsTheEmbeddingRow() {
        float[] out = new float[8];
        CpuHyperConnections.expandStreams(2, 2, 2, new float[]{1f, 2f, 3f, 4f}, 0, out, 0);
        assertArrayEquals(new float[]{1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f}, out, 0f);
    }

    @Test
    public void attentionResidualMixMatchesKimiK3() {
        Random rnd = new Random(5);
        int nEmbd = 6;
        int nTokens = 2;
        float eps = 1e-6f;
        float[] ckpt0 = random(rnd, nTokens * nEmbd, 2f);
        float[] ckpt1 = random(rnd, nTokens * nEmbd, 2f);
        float[] cur = random(rnd, nTokens * nEmbd, 2f);
        float[] w = random(rnd, nEmbd, 1f);

        CpuAttnResidualStack stack = new CpuAttnResidualStack(nEmbd, 2);
        stack.begin(nTokens);
        float[] out = new float[cur.length];
        stack.mixInto(cur, 0, w, eps, out, 0);
        assertArrayEquals("no checkpoint banked: identity", cur, out, 0f);

        stack.push(ckpt0, 0);
        stack.push(ckpt1, 0);
        assertEquals(2, stack.checkpoints());
        stack.mixInto(cur, 0, w, eps, out, 0);

        for (int t = 0; t < nTokens; t++) {
            float[][] rows = {slice(ckpt0, t, nEmbd), slice(ckpt1, t, nEmbd), slice(cur, t, nEmbd)};
            double[] score = new double[3];
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < 3; c++) {
                double ss = 0;
                for (float v : rows[c]) {
                    ss += v * v;
                }
                double r = 1.0 / Math.sqrt(ss / nEmbd + eps);
                for (int i = 0; i < nEmbd; i++) {
                    score[c] += rows[c][i] * r * w[i];
                }
                max = Math.max(max, score[c]);
            }
            double sum = 0;
            for (int c = 0; c < 3; c++) {
                score[c] = Math.exp(score[c] - max);
                sum += score[c];
            }
            for (int i = 0; i < nEmbd; i++) {
                double expect = 0;
                for (int c = 0; c < 3; c++) {
                    expect += rows[c][i] * score[c] / sum;
                }
                assertEquals((float) expect, out[t * nEmbd + i], 1e-5f);
            }
        }

        float[] inPlace = cur.clone();
        stack.mixInto(inPlace, 0, w, eps, inPlace, 0);
        assertArrayEquals(out, inPlace, 0f);

        try {
            stack.push(cur, 0);
            fail("a full stack must reject another checkpoint");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("full"));
        }
    }

    @Test
    public void probabilitiesSumToOne() {
        Random rnd = new Random(9);
        int nEmbd = 8;
        int nTokens = 3;
        int nCkpt = 3;
        float[] stack = random(rnd, nCkpt * nTokens * nEmbd, 3f);
        float[] cur = random(rnd, nTokens * nEmbd, 3f);
        float[] probs = new float[nTokens * (nCkpt + 1)];
        CpuAttnResidualStack.scores(nTokens, nEmbd, nCkpt, EPS, stack, 0, cur, 0, random(rnd, nEmbd, 1f), probs, 0,
                0, nTokens);
        for (int t = 0; t < nTokens; t++) {
            float sum = 0f;
            for (int c = 0; c <= nCkpt; c++) {
                float p = probs[t * (nCkpt + 1) + c];
                assertTrue(p > 0f && p < 1f);
                sum += p;
            }
            assertEquals(1f, sum, 1e-6f);
        }
    }

    @Test
    public void geometryAndValidation() {
        assertEquals(24, HcKernels.mixDim(4, HcKernels.OUT_PRE_POST_COMB));
        assertEquals(8, HcKernels.mixDim(4, HcKernels.OUT_PRE_POST));
        assertEquals(4, HcKernels.mixDim(4, HcKernels.OUT_PRE));
        assertEquals(3, HcKernels.checkpointCount(13, 6));
        assertTrue(HcKernels.isCheckpointLayer(12, 6));
        assertTrue(!HcKernels.isCheckpointLayer(13, 6));
        assertEquals(512L, HcKernels.elementGlobal(1, 300, 256));
        assertEquals(256L, HcKernels.perTokenGlobal(3, 256));
        assertEquals(64, HcKernels.scoreLocal(64));
        assertEquals(32, HcKernels.scoreLocal(16));
        assertEquals(256, HcKernels.scoreLocal(1024));
        assertEquals(3L * 128, HcKernels.scoreGlobal(3, 128));
        try {
            CoefficientParams.mhc(HcKernels.MAX_HC + 1, EPS, 3);
            fail("hc above the kernel limit must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("stream count"));
        }
        try {
            CoefficientParams.mhc(4, EPS, 0);
            fail("Sinkhorn without iterations must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("iteration"));
        }
    }

    @Test
    public void argumentIndicesMatchTheKernelSource() throws IOException {
        String source = readResource("kernels/llm/hc.cl");

        List<String> coef = parameters(source, HcKernels.HC_PRE_POST_COMB);
        assertEquals(HcKernels.COEF_ARG_COUNT, coef.size());
        assertEquals(HcKernels.COEF_ARG_N_TOKENS, coef.indexOf("n_tokens"));
        assertEquals(HcKernels.COEF_ARG_HC, coef.indexOf("hc"));
        assertEquals(HcKernels.COEF_ARG_EPS, coef.indexOf("eps"));
        assertEquals(HcKernels.COEF_ARG_ITERS, coef.indexOf("iters"));
        assertEquals(HcKernels.COEF_ARG_N_OUT, coef.indexOf("n_out"));
        assertEquals(HcKernels.COEF_ARG_POST_MAG, coef.indexOf("post_mag"));
        assertEquals(HcKernels.COEF_ARG_POST_EPS, coef.indexOf("post_eps"));
        assertEquals(HcKernels.COEF_ARG_MIXES_OFF, coef.indexOf("mixes_off"));
        assertEquals(HcKernels.COEF_ARG_PRE_OFF, coef.indexOf("pre_off"));
        assertEquals(HcKernels.COEF_ARG_POST_OFF, coef.indexOf("post_off"));
        assertEquals(HcKernels.COEF_ARG_COMB_OFF, coef.indexOf("comb_off"));
        assertEquals(HcKernels.COEF_ARG_MIXES, coef.indexOf("mixes"));
        assertEquals(HcKernels.COEF_ARG_SCALE, coef.indexOf("scale"));
        assertEquals(HcKernels.COEF_ARG_BASE, coef.indexOf("base"));
        assertEquals(HcKernels.COEF_ARG_PRE, coef.indexOf("pre"));
        assertEquals(HcKernels.COEF_ARG_POST, coef.indexOf("post"));
        assertEquals(HcKernels.COEF_ARG_COMB, coef.indexOf("comb"));

        List<String> pre = parameters(source, HcKernels.HC_PRE);
        assertEquals(HcKernels.PRE_ARG_COUNT, pre.size());
        assertEquals(HcKernels.PRE_ARG_N_TOKENS, pre.indexOf("n_tokens"));
        assertEquals(HcKernels.PRE_ARG_N_EMBD, pre.indexOf("n_embd"));
        assertEquals(HcKernels.PRE_ARG_HC, pre.indexOf("hc"));
        assertEquals(HcKernels.PRE_ARG_MODE, pre.indexOf("mode"));
        assertEquals(HcKernels.PRE_ARG_GSCALE, pre.indexOf("gscale"));
        assertEquals(HcKernels.PRE_ARG_X_OFF, pre.indexOf("x_off"));
        assertEquals(HcKernels.PRE_ARG_W_OFF, pre.indexOf("w_off"));
        assertEquals(HcKernels.PRE_ARG_OUT_OFF, pre.indexOf("out_off"));
        assertEquals(HcKernels.PRE_ARG_X, pre.indexOf("X"));
        assertEquals(HcKernels.PRE_ARG_W_OR_GATE, pre.indexOf("w_or_gate"));
        assertEquals(HcKernels.PRE_ARG_OUT, pre.indexOf("out"));

        List<String> post = parameters(source, HcKernels.HC_POST);
        assertEquals(HcKernels.POST_ARG_COUNT, post.size());
        assertEquals(HcKernels.POST_ARG_N_TOKENS, post.indexOf("n_tokens"));
        assertEquals(HcKernels.POST_ARG_N_EMBD, post.indexOf("n_embd"));
        assertEquals(HcKernels.POST_ARG_HC, post.indexOf("hc"));
        assertEquals(HcKernels.POST_ARG_HAS_COMB, post.indexOf("has_comb"));
        assertEquals(HcKernels.POST_ARG_POST_ACT, post.indexOf("post_act"));
        assertEquals(HcKernels.POST_ARG_POST_IN_SCALE, post.indexOf("post_in_scale"));
        assertEquals(HcKernels.POST_ARG_POST_MAG, post.indexOf("post_mag"));
        assertEquals(HcKernels.POST_ARG_Y_OFF, post.indexOf("y_off"));
        assertEquals(HcKernels.POST_ARG_X_OFF, post.indexOf("x_off"));
        assertEquals(HcKernels.POST_ARG_POST_OFF, post.indexOf("post_off"));
        assertEquals(HcKernels.POST_ARG_COMB_OFF, post.indexOf("comb_off"));
        assertEquals(HcKernels.POST_ARG_XOUT_OFF, post.indexOf("xout_off"));
        assertEquals(HcKernels.POST_ARG_Y, post.indexOf("y"));
        assertEquals(HcKernels.POST_ARG_X, post.indexOf("X"));
        assertEquals(HcKernels.POST_ARG_POST, post.indexOf("post"));
        assertEquals(HcKernels.POST_ARG_COMB, post.indexOf("comb"));
        assertEquals(HcKernels.POST_ARG_XOUT, post.indexOf("Xout"));

        List<String> score = parameters(source, HcKernels.RES_STACK_SCORE);
        assertEquals(HcKernels.SCORE_ARG_COUNT, score.size());
        assertEquals(HcKernels.SCORE_ARG_N_TOKENS, score.indexOf("n_tokens"));
        assertEquals(HcKernels.SCORE_ARG_N_EMBD, score.indexOf("n_embd"));
        assertEquals(HcKernels.SCORE_ARG_N_CKPT, score.indexOf("n_ckpt"));
        assertEquals(HcKernels.SCORE_ARG_EPS, score.indexOf("eps"));
        assertEquals(HcKernels.SCORE_ARG_STACK_OFF, score.indexOf("stack_off"));
        assertEquals(HcKernels.SCORE_ARG_CUR_OFF, score.indexOf("cur_off"));
        assertEquals(HcKernels.SCORE_ARG_PROBS_OFF, score.indexOf("probs_off"));
        assertEquals(HcKernels.SCORE_ARG_STACK, score.indexOf("stack"));
        assertEquals(HcKernels.SCORE_ARG_CUR, score.indexOf("cur"));
        assertEquals(HcKernels.SCORE_ARG_SCORE_W, score.indexOf("score_w"));
        assertEquals(HcKernels.SCORE_ARG_PROBS, score.indexOf("probs"));

        List<String> mix = parameters(source, HcKernels.RES_STACK_MIX);
        assertEquals(HcKernels.MIX_ARG_COUNT, mix.size());
        assertEquals(HcKernels.MIX_ARG_N_TOKENS, mix.indexOf("n_tokens"));
        assertEquals(HcKernels.MIX_ARG_N_EMBD, mix.indexOf("n_embd"));
        assertEquals(HcKernels.MIX_ARG_N_CKPT, mix.indexOf("n_ckpt"));
        assertEquals(HcKernels.MIX_ARG_STACK_OFF, mix.indexOf("stack_off"));
        assertEquals(HcKernels.MIX_ARG_CUR_OFF, mix.indexOf("cur_off"));
        assertEquals(HcKernels.MIX_ARG_PROBS_OFF, mix.indexOf("probs_off"));
        assertEquals(HcKernels.MIX_ARG_OUT_OFF, mix.indexOf("out_off"));
        assertEquals(HcKernels.MIX_ARG_STACK, mix.indexOf("stack"));
        assertEquals(HcKernels.MIX_ARG_CUR, mix.indexOf("cur"));
        assertEquals(HcKernels.MIX_ARG_PROBS, mix.indexOf("probs"));
        assertEquals(HcKernels.MIX_ARG_OUT, mix.indexOf("out"));
    }

    /** Parameter names of {@code NF_KERNEL void <kernel>(...)} in declaration order. */
    private static List<String> parameters(String source, String kernel) {
        String head = "NF_KERNEL void " + kernel + "(";
        int start = source.indexOf(head);
        assertTrue(kernel + " is defined in hc.cl", start >= 0);
        int end = source.indexOf(')', start);
        List<String> names = new ArrayList<>();
        for (String param : source.substring(start + head.length(), end).split(",")) {
            String[] tokens = param.trim().split("[\\s*]+");
            names.add(tokens[tokens.length - 1]);
        }
        return names;
    }

    private static String readResource(String path) throws IOException {
        try (InputStream in = CpuHyperConnectionsTest.class.getClassLoader().getResourceAsStream(path)) {
            assertNotNull(path + " is on the classpath", in);
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            for (int n = in.read(buf); n >= 0; n = in.read(buf)) {
                bytes.write(buf, 0, n);
            }
            return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private static float[] slice(float[] a, int t, int n) {
        float[] r = new float[n];
        System.arraycopy(a, t * n, r, 0, n);
        return r;
    }
}
