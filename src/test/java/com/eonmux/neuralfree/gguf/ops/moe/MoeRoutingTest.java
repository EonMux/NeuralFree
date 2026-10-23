package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Traces of the CPU MoE router and helper twins against hand-derived llama.cpp {@code build_moe_ffn} results
 * (src/llama-graph.cpp:2232-2345), plus the kernel-side reciprocal used by the grovemoe id division.
 */
public class MoeRoutingTest {

    private static final float EPS = 1e-6f;

    private static double sig(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static MoeRouting.Params params(int n, int k, GatingFunc g, boolean norm, float scale, int sel) {
        return new MoeRouting.Params(n, k, g, 1, 1, norm, scale, false, sel, 0);
    }

    /** Mixtral-style softmax top-2 with normalization: weights are the renormalized probabilities. */
    @Test
    public void softmaxTopKNormalized() {
        float[] logits = {1f, 3f, 2f, 0.5f, 3.5f, -1f, 0f, 2.9f};
        MoeRouting.Selection s = MoeRouting.route(params(8, 2, GatingFunc.SOFTMAX, true, 1f, MoeKernels.SEL_PROBS),
                logits, 1, null);
        assertArrayEquals(new int[]{4, 1}, s.ids());
        double a = Math.exp(0.5);
        assertEquals(a / (a + 1.0), s.weight(0, 0), EPS);
        assertEquals(1.0 / (a + 1.0), s.weight(0, 1), EPS);

        MoeRouting.Selection raw = MoeRouting.route(params(8, 2, GatingFunc.SOFTMAX, false, 1f, MoeKernels.SEL_PROBS),
                logits, 1, null);
        double sum = 0;
        for (float x : logits) {
            sum += Math.exp(x - 3.5);
        }
        assertEquals(1.0 / sum, raw.weight(0, 0), EPS);
        assertEquals(Math.exp(-0.5) / sum, raw.weight(0, 1), EPS);
    }

    /** Equal scores select in ascending index order; NaN scores rank last. */
    @Test
    public void tiesGoToTheLowerIndexAndNanRanksLast() {
        float[] equal = {0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
        MoeRouting.Selection s = MoeRouting.route(params(5, 3, GatingFunc.SIGMOID, false, 1f, MoeKernels.SEL_PROBS),
                equal, 1, null);
        assertArrayEquals(new int[]{0, 1, 2}, s.ids());

        float[] withNan = {Float.NaN, -3f, Float.NaN, 7f};
        MoeRouting.Selection n = MoeRouting.route(
                params(4, 3, GatingFunc.SIGMOID, false, 1f, MoeKernels.SEL_RAW_LOGITS), withNan, 1, null);
        assertArrayEquals(new int[]{3, 1, 0}, n.ids());
    }

    /**
     * DeepSeek-V3 group-limited routing: 256 experts in 8 groups of 32, top-4 groups by the sum of each group's two
     * best scores, sigmoid gating, {@code exp_probs_b}, k = 8, normalized weights scaled by 2.5.
     * Selection scores {@code sigmoid(logit) + bias} (all other experts 0.5):
     * <pre>
     * g1: e35 2.8+s(0.5)=3.4225, e41 2.8+s(-0.5)=3.1775  -> 6.6000
     * g2: e64, e65 1.0+0.5                                -> 3.0
     * g3: e126 4+s(2)=4.8808, e127 4+s(-2)=4.1192        -> 9.0000
     * g5: e167 10+0.5=10.5                                -> 11.0
     * g6: e197, e198 3+s(1)=3.7311                        -> 7.4621
     * g7: e224 5.4+0.5=5.9 (best single expert after e167) -> 6.4
     * g0, g4: 0.5 + 0.5                                   -> 1.0
     * </pre>
     * Kept groups {5, 3, 6, 1}: g7 loses to g1 although its best expert beats every expert of g1, g3 and g6.
     * Top-8 over the kept groups: 167, 126, 127, 197, 198 (tie: lower index first), 35, 41, then the lowest-index
     * 0.5 expert of a kept group, 32. Weights are the unbiased sigmoid probabilities, normalized and scaled.
     */
    @Test
    public void deepseekV3GroupLimitedRouting() {
        int n = 256;
        float[] logits = new float[n];
        float[] bias = new float[n];
        logits[35] = 0.5f;
        bias[35] = 2.8f;
        logits[41] = -0.5f;
        bias[41] = 2.8f;
        bias[64] = 1.0f;
        bias[65] = 1.0f;
        logits[126] = 2.0f;
        bias[126] = 4.0f;
        logits[127] = -2.0f;
        bias[127] = 4.0f;
        bias[167] = 10.0f;
        logits[197] = 1.0f;
        bias[197] = 3.0f;
        logits[198] = 1.0f;
        bias[198] = 3.0f;
        bias[224] = 5.4f;

        MoeSpec spec = MoeSpec.builder().experts(n, 8, 2048).gating(GatingFunc.SIGMOID).probBias(true)
                .groups(8, 4).normWeights(true).weightsScale(2.5f).glu(GluOp.SWIGLU).build();
        MoeRouting.Params p = MoeRouting.Params.forExperts(spec);
        assertEquals(MoeKernels.SEL_PROBS, p.selMode());
        assertTrue(p.hasBias());

        float[] scores = new float[n];
        for (int e = 0; e < n; e++) {
            scores[e] = (float) sig(logits[e]) + bias[e];
        }
        MoeRouting.maskGroups(scores, 8, 4);
        for (int g = 0; g < 8; g++) {
            boolean kept = g == 1 || g == 3 || g == 5 || g == 6;
            assertEquals("group " + g, kept, scores[g * 32 + 10] != Float.NEGATIVE_INFINITY);
        }

        MoeRouting.Selection s = MoeRouting.route(p, logits, 1, bias);
        assertArrayEquals(new int[]{167, 126, 127, 197, 198, 35, 41, 32}, s.ids());

        double[] probs = {0.5, sig(2), sig(-2), sig(1), sig(1), sig(0.5), sig(-0.5), 0.5};
        double sum = 0;
        for (double v : probs) {
            sum += v;
        }
        for (int j = 0; j < 8; j++) {
            assertEquals("weight " + j, probs[j] / sum * 2.5, s.weight(0, j), EPS);
        }
    }

    /** Two groups with equal top-2 sums: the lower group is kept. */
    @Test
    public void groupTiesKeepTheLowerGroup() {
        float[] sel = {1f, 2f, 0f, 0f, 2f, 1f, 0f, 0f};
        MoeRouting.maskGroups(sel, 2, 1);
        assertArrayEquals(new float[]{1f, 2f, 0f, 0f, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}, sel, 0f);
    }

    /**
     * SOFTMAX_WEIGHT: probs are the logits, the top-k is taken on them, then softmax over the k selected values.
     * Logits {0.1, 2, -1, 2, 0.5, 3}, k = 3 -> ids {5, 1, 3}; weights softmax(3, 2, 2) =
     * {1 / (1 + 2/e), (1/e) / (1 + 2/e), (1/e) / (1 + 2/e)}; normalization leaves them unchanged.
     */
    @Test
    public void softmaxWeightGating() {
        float[] logits = {0.1f, 2f, -1f, 2f, 0.5f, 3f};
        double d = 1.0 + 2.0 * Math.exp(-1.0);
        for (boolean norm : new boolean[]{false, true}) {
            MoeRouting.Selection s = MoeRouting.route(
                    params(6, 3, GatingFunc.SOFTMAX_WEIGHT, norm, 1f, MoeKernels.SEL_PROBS), logits, 1, null);
            assertArrayEquals(new int[]{5, 1, 3}, s.ids());
            assertEquals(1.0 / d, s.weight(0, 0), EPS);
            assertEquals(Math.exp(-1.0) / d, s.weight(0, 1), EPS);
            assertEquals(Math.exp(-1.0) / d, s.weight(0, 2), EPS);
        }
    }

    /** SQRT_SOFTPLUS gating: weights are sqrt(softplus(logit)), normalized. */
    @Test
    public void sqrtSoftplusGating() {
        float[] logits = {-1f, 2f, 0f, 25f};
        MoeRouting.Selection s = MoeRouting.route(
                params(4, 2, GatingFunc.SQRT_SOFTPLUS, true, 1f, MoeKernels.SEL_PROBS), logits, 1, null);
        assertArrayEquals(new int[]{3, 1}, s.ids());
        double a = Math.sqrt(25.0);
        double b = Math.sqrt(Math.log1p(Math.exp(2.0)));
        assertEquals(a / (a + b), s.weight(0, 0), EPS);
        assertEquals(b / (a + b), s.weight(0, 1), EPS);
    }

    /** llama4: selection on the raw logits, weights sigmoid(logit) without normalization. */
    @Test
    public void llama4RawLogitSelection() {
        float[] logits = {0.3f, -0.2f, 1.7f, 0.9f};
        MoeRouting.Selection s = MoeRouting.route(
                params(4, 1, GatingFunc.SIGMOID, false, 1f, MoeKernels.SEL_RAW_LOGITS), logits, 1, null);
        assertArrayEquals(new int[]{2}, s.ids());
        assertEquals(sig(1.7), s.weight(0, 0), EPS);
    }

    /**
     * grovemoe chunk pass: selection on sigmoid(logits) over all experts, ids scaled by 1/n_group_experts and
     * truncated, weights = softmax probs at the divided ids (llama-graph.cpp:2312-2321).
     */
    @Test
    public void grovemoeChunkIds() {
        MoeSpec spec = MoeSpec.builder().experts(8, 4, 64).gating(GatingFunc.SOFTMAX)
                .selectionMode(MoeSpec.SelectionMode.SIGMOID_LOGITS).normWeights(true).chunks(4, 2, 32, 0.05f)
                .glu(GluOp.SWIGLU).build();
        MoeRouting.Params main = MoeRouting.Params.forExperts(spec);
        MoeRouting.Params chunk = MoeRouting.Params.forChunks(spec);
        assertEquals(0, main.idDiv());
        assertEquals(4, chunk.idDiv());
        assertEquals(2, chunk.k());
        assertEquals(MoeKernels.SEL_SIGMOID_LOGITS, chunk.selMode());

        float[] logits = {0f, 5f, 1f, 4f, 3f, 0f, 0f, 6f};
        MoeRouting.Selection s = MoeRouting.route(chunk, logits, 1, null);
        assertArrayEquals(new int[]{1, 0}, s.ids());
        double e0 = Math.exp(0.0 - 6.0);
        double e1 = Math.exp(5.0 - 6.0);
        assertEquals(e1 / (e0 + e1), s.weight(0, 0), EPS);
        assertEquals(e0 / (e0 + e1), s.weight(0, 1), EPS);

        assertEquals(1, MoeRouting.divId(3, 3));
        assertEquals(2, MoeRouting.divId(6, 3));
        assertEquals(0, MoeRouting.divId(2, 3));
        assertEquals(31, MoeRouting.divId(127, 4));
    }

    /** Hash-routed layers: ids come from the token table, weights from the gating probabilities of those ids. */
    @Test
    public void preselectedIdsTakeGatingWeights() {
        int[] table = {2, 0, 1, 3, 3, 2};
        int[] ids = MoeCombine.hashIds(new int[]{2, 0}, 2, 2, table);
        assertArrayEquals(new int[]{3, 2, 2, 0}, ids);

        MoeSpec spec = MoeSpec.builder().experts(4, 2, 16).gating(GatingFunc.SQRT_SOFTPLUS).probBias(true)
                .normWeights(true).hashRouting(true).glu(GluOp.SWIGLU).build();
        MoeRouting.Params p = MoeRouting.Params.forExperts(spec);
        assertEquals(MoeKernels.SEL_PRESELECTED, p.selMode());
        assertFalse(p.hasBias());
        float[] logits = {0f, 1f, 2f, 3f, 3f, 2f, 1f, 0f};
        MoeRouting.Selection s = MoeRouting.routePreselected(p, logits, 2, ids);
        assertArrayEquals(ids, s.ids());
        double a = Math.sqrt(Math.log1p(Math.exp(3.0)));
        double b = Math.sqrt(Math.log1p(Math.exp(2.0)));
        assertEquals(a / (a + b), s.weight(0, 0), EPS);
        assertEquals(b / (a + b), s.weight(0, 1), EPS);
        double c = Math.sqrt(Math.log1p(Math.exp(1.0)));
        double d = Math.sqrt(Math.log1p(Math.exp(3.0)));
        assertEquals(c / (c + d), s.weight(1, 0), EPS);
        assertEquals(d / (c + d), s.weight(1, 1), EPS);

        try {
            MoeCombine.validateHashTable(new int[]{0, 4}, 2, 1, 4);
            fail("expert id 4 of 4 must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("[0][1]"));
        }
    }

    /** A weight sum below the smallest normal f16 is clamped before dividing. */
    @Test
    public void normalizationClampsTinySums() {
        float[] logits = {-20f, -21f, -30f};
        MoeRouting.Selection s = MoeRouting.route(params(3, 2, GatingFunc.SIGMOID, true, 1f, MoeKernels.SEL_PROBS),
                logits, 1, null);
        assertEquals(sig(-20) / MoeSpec.NORM_SUM_MIN, s.weight(0, 0), 1e-9);
        assertEquals(sig(-21) / MoeSpec.NORM_SUM_MIN, s.weight(0, 1), 1e-9);
    }

    /** weightsScale 0 and 1 leave the weights unscaled; any other value multiplies them. */
    @Test
    public void weightScaleZeroAndOneAreIdentity() {
        float[] logits = {0.5f, -0.5f};
        float base = MoeRouting.route(params(2, 1, GatingFunc.SIGMOID, false, 1f, MoeKernels.SEL_PROBS),
                logits, 1, null).weight(0, 0);
        float zero = MoeRouting.route(params(2, 1, GatingFunc.SIGMOID, false, 0f, MoeKernels.SEL_PROBS),
                logits, 1, null).weight(0, 0);
        float half = MoeRouting.route(params(2, 1, GatingFunc.SIGMOID, false, 0.5f, MoeKernels.SEL_PROBS),
                logits, 1, null).weight(0, 0);
        assertEquals(base, zero, 0f);
        assertEquals(base * 0.5f, half, 0f);
    }

    /**
     * The kernels compute 1/div by integer long division rounded to nearest-even (moe.cl nf_moe_div_id); it must
     * equal IEEE {@code 1.0f / div}, which the Java twin uses.
     */
    @Test
    public void kernelReciprocalMatchesIeeeDivision() {
        for (int div = 1; div <= (1 << 16); div++) {
            assertEquals("1/" + div, Float.floatToIntBits(1.0f / div), kernelReciprocalBits(div));
        }
        for (int div : new int[]{(1 << 20) + 1, (1 << 23) - 1, (1 << 24) - 1, 3 * 7 * 11 * 13 * 17}) {
            assertEquals("1/" + div, Float.floatToIntBits(1.0f / div), kernelReciprocalBits(div));
        }
    }

    private static int kernelReciprocalBits(int div) {
        int g = div;
        int n = 31 - Integer.numberOfLeadingZeros(g);
        if ((g & (g - 1)) == 0) {
            return (127 - n) << 23;
        }
        long rem = 1L << n;
        long q = 0;
        for (int b = 0; b < 24; b++) {
            rem <<= 1;
            q <<= 1;
            if (rem >= g) {
                rem -= g;
                q |= 1;
            }
        }
        if (2 * rem > g || (2 * rem == g && (q & 1) != 0)) {
            q++;
        }
        return q == (1L << 24) ? (127 - n) << 23 : ((126 - n) << 23) | (int) (q & 0x7FFFFF);
    }

    /** Bucketize is a stable sort by expert with invalid ids last; the GPU kernel ranks chunk-wise to the same rows. */
    @Test
    public void bucketizeIsStable() {
        int[] ids = {2, 0, 2, 5, 1, 2, -1, 0};
        MoeCombine.Buckets b = MoeCombine.bucketize(4, 2, 4, ids);
        assertArrayEquals(new int[]{2, 1, 3, 0}, b.counts());
        assertArrayEquals(new int[]{0, 2, 3, 6}, b.offsets());
        assertArrayEquals(new int[]{1, 7, 4, 0, 2, 5, 3, 6}, b.rowToken());
        assertArrayEquals(new int[]{0, 0, 1, 2, 2, 2, -1, -1}, b.rowSlot());
    }

    /** Weighted sum (j ascending), unweighted and accumulate flags; input scaling; add-id. */
    @Test
    public void combineHelpers() {
        float[] experts = {1f, 2f, 10f, 20f, 3f, 4f, 30f, 40f};
        float[] w = {0.25f, 0.5f, 2f, 1f};
        float[] dst = new float[4];
        MoeCombine.weightedSum(2, 2, 2, 0, experts, w, dst);
        assertArrayEquals(new float[]{5.25f, 10.5f, 36f, 48f}, dst, 0f);
        MoeCombine.weightedSum(2, 2, 2, MoeKernels.WSUM_ACCUMULATE | MoeKernels.WSUM_UNWEIGHTED, experts, null, dst);
        assertArrayEquals(new float[]{16.25f, 32.5f, 69f, 92f}, dst, 0f);

        float[] scaled = MoeCombine.scaleInput(1, 2, 2, new float[]{3f, -1f}, new float[]{2f, 0.5f});
        assertArrayEquals(new float[]{6f, -2f, 1.5f, -0.5f}, scaled, 0f);

        float[] data = {1f, 1f, 1f, 1f};
        MoeCombine.addId(2, 2, new int[]{1, 0}, new float[]{10f, 20f, 30f, 40f}, data);
        assertArrayEquals(new float[]{31f, 41f, 11f, 21f}, data, 0f);
    }

    /** Every weight type has the three id-matvec kernels, with the family's threads-per-block. */
    @Test
    public void kernelNamesCoverAllWeightTypes() {
        assertEquals(29, MoeKernels.weightTypes().size());
        assertEquals("Q4_K_MatVecIdGateUp", MoeKernels.matVecIdGateUp(GGMLType.Q4_K));
        assertEquals("NVFP4_BatchMatMulId", MoeKernels.batchMatMulId(GGMLType.NVFP4));
        assertEquals("BF16_MatVecId", MoeKernels.matVecId(GGMLType.BF16));
        assertEquals(16, MoeKernels.threadsPerBlock(GGMLType.Q1_0));
        assertEquals(32, MoeKernels.threadsPerBlock(GGMLType.IQ4_XS));
        assertEquals(4, MoeKernels.threadsPerBlock(GGMLType.MXFP4));
        MoeKernels.checkIdLocal(GGMLType.Q6_K, 64);
        try {
            MoeKernels.checkIdLocal(GGMLType.Q6_K, 16);
            fail("local 16 is below a K-quant block's 32 work-items");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Q6_K"));
        }
        assertArrayEquals(new long[]{4096L * 64, 3}, MoeKernels.batchMatMulIdGlobal(4096, 17, 64));
    }
}
