package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The tensor-type policy of the quantize tool against the reference recipes: the Q4_K_M and Q3_K_M
 * mixtures over a four-layer model, the tied-embedding and output rules, the pure mode, the manual
 * tensor-type overrides, the shape fallback and the imatrix requirement.
 */
public class QuantizeTypesTest {

    /** A 4-layer llama: n_gqa 2, no experts. */
    private static QuantizeTypes.ModelInfo llama() {
        return QuantizeTypes.ModelInfo.of("llama", 4, 4, 2, 0, false);
    }

    private static QuantizeParams.Builder params(QuantizeFtype ftype) {
        return QuantizeParams.builder().ftype(ftype);
    }

    private static long[] dims2(long ne0, long ne1) {
        return new long[] {ne0, ne1};
    }

    /** Decides the targets of {@code names} in order, one call per name, and returns them. */
    private static GGMLType[] decide(QuantizeParams p, QuantizeTypes.ModelInfo model, String[] names,
                                     long[][] dims, GGMLType[] src) {
        QuantizeTypes.Mixture mix = new QuantizeTypes.Mixture(p, model, Arrays.asList(names));
        GGMLType[] out = new GGMLType[names.length];
        for (int i = 0; i < names.length; i++) {
            out[i] = mix.target(names[i], dims[i], src[i], p.ftype.defaultType());
        }
        return out;
    }

    @Test
    public void q4KMixtureOverFourLayers() {
        String[] names = {
            "token_embd.weight", "output.weight",
            "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
            "blk.0.attn_output.weight", "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
            "blk.2.attn_v.weight", "blk.2.ffn_down.weight",
            "blk.3.attn_v.weight", "blk.3.ffn_down.weight",
        };
        long[][] dims = new long[names.length][];
        for (int i = 0; i < names.length; i++) {
            dims[i] = dims2(256, 256);
        }
        GGMLType[] src = new GGMLType[names.length];
        Arrays.fill(src, GGMLType.F32);
        GGMLType[] out = decide(params(QuantizeFtype.Q4_K_M).build(), llama(), names, dims, src);

        assertEquals(GGMLType.Q4_K, out[0]); // token_embd, untied
        assertEquals(GGMLType.Q6_K, out[1]); // output.weight
        assertEquals(GGMLType.F32, out[2]);  // attn_norm is 1-D and stays
        assertEquals(GGMLType.Q4_K, out[3]);
        assertEquals(GGMLType.Q4_K, out[5]); // blk.0.attn_v, the first attn_v of the run
        // the layer counters order the mixture: the first attn_v and ffn_down stay Q4_K, the later
        // ones land on the more-bits layers
        assertEquals(GGMLType.Q4_K, out[10]); // blk.0.ffn_down, ffn_down 0
        assertEquals(GGMLType.Q4_K, out[11]); // blk.2.attn_v, attn_v 1
        assertEquals(GGMLType.Q4_K, out[12]); // blk.2.ffn_down, ffn_down 1
        assertEquals(GGMLType.Q6_K, out[13]); // blk.3.attn_v, attn_v 2: (2 - 0) % 3 == 2
        assertEquals(GGMLType.Q6_K, out[14]); // blk.3.ffn_down, ffn_down 2
        assertFalse(Arrays.asList(out).contains(GGMLType.Q8_0));
    }

    @Test
    public void q3KMixtureBumpsTheFirstAttentionV() {
        String[] names = {"blk.0.attn_v.weight", "blk.1.attn_v.weight", "blk.2.attn_v.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.Q3_K_M).build(), llama(), names, dims, src);
        assertEquals(GGMLType.Q5_K, out[0]);
        assertEquals(GGMLType.Q5_K, out[1]);
        assertEquals(GGMLType.Q4_K, out[2]);
    }

    @Test
    public void pureDisablesTheMixture() {
        String[] names = {"output.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.Q4_K_M).pure(true).build(), llama(), names, dims, src);
        assertAll(GGMLType.Q4_K, out);
    }

    @Test
    public void iq2XxsRecipeMovesTheSensitiveTensors() {
        String[] names = {"token_embd.weight", "output.weight", "blk.0.attn_v.weight",
                "blk.0.attn_output.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256), dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32, GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.IQ2_XXS).build(), llama(), names, dims, src);
        assertEquals(GGMLType.Q2_K, out[0]);
        assertEquals(GGMLType.Q5_K, out[1]);
        assertEquals(GGMLType.Q2_K, out[2]); // n_gqa 2 < 4
        assertEquals(GGMLType.IQ2_XXS, out[3]);
    }

    @Test
    public void iq2XxsWithFourQueryHeadsBumpsAttnV() {
        QuantizeTypes.ModelInfo mqa = QuantizeTypes.ModelInfo.of("llama", 4, 4, 1, 0, false);
        String[] names = {"blk.0.attn_v.weight"};
        long[][] dims = {dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.IQ2_XS).build(), mqa, names, dims, src);
        assertEquals(GGMLType.Q4_K, out[0]);
    }

    @Test
    public void eightExpertsBumpTheSensitiveTensors() {
        QuantizeTypes.ModelInfo moe = QuantizeTypes.ModelInfo.of("llama", 4, 4, 2, 8, false);
        String[] names = {"blk.0.attn_v.weight", "blk.0.attn_k.weight", "blk.0.attn_output.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.IQ2_XXS).build(), moe, names, dims, src);
        // inside the low-bit mixture the 8-expert rule picks Q4_K for v and k, Q5_K for the output
        assertEquals(GGMLType.Q4_K, out[0]);
        assertEquals(GGMLType.Q4_K, out[1]);
        assertEquals(GGMLType.Q5_K, out[2]);
    }

    @Test
    public void eightExpertsBumpAttnVToQ8_0OutsideTheLowBitMixtures() {
        QuantizeTypes.ModelInfo moe = QuantizeTypes.ModelInfo.of("llama", 4, 4, 2, 8, false);
        String[] names = {"blk.0.attn_v.weight"};
        long[][] dims = {dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.Q4_K_M).build(), moe, names, dims, src);
        assertEquals(GGMLType.Q8_0, out[0]);
    }

    @Test
    public void tiedEmbeddingsShareTheOutputType() {
        // a model without output.weight: token_embd follows the output rule
        String[] names = {"token_embd.weight"};
        long[][] dims = {dims2(256, 320)};
        GGMLType[] src = {GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.Q4_K_S).build(), llama(), names, dims, src);
        assertEquals(GGMLType.Q6_K, out[0]);
    }

    @Test
    public void mxfp4MoeSplitsExpertsFromTheRest() {
        String[] names = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
                "blk.0.ffn_gate_inp.weight", "output.weight", "token_embd.weight"};
        long[][] dims = {dims2(256, 512), dims2(256, 512), dims2(256, 8), dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32, GGMLType.F32, GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.MXFP4_MOE).build(), llama(), names, dims, src);
        assertEquals(GGMLType.MXFP4, out[0]);
        assertEquals(GGMLType.MXFP4, out[1]);
        assertEquals(GGMLType.F32, out[2]); // the router is never quantized
        assertEquals(GGMLType.Q8_0, out[3]);
        assertEquals(GGMLType.Q8_0, out[4]);
    }

    @Test
    public void tq2_0PutsTheEmbeddingIntoQ4K() throws Exception {
        String[] names = {"token_embd.weight", "output.weight", "blk.0.attn_q.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(params(QuantizeFtype.TQ2_0).build(), llama(), names, dims, src);
        assertEquals(GGMLType.Q4_K, out[0]);
        assertEquals(GGMLType.TQ2_0, out[2]);
    }

    @Test
    public void manualTensorTypeOverridesWin() {
        QuantizeParams p = params(QuantizeFtype.Q4_K_M).build().toBuilder()
                .tensorTypes(List.of(new QuantizeParams.TensorType(
                        java.util.regex.Pattern.compile("attn_q"), "attn_q", GGMLType.Q8_0)))
                .build();
        String[] names = {"blk.0.attn_q.weight", "blk.0.attn_k.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(p, llama(), names, dims, src);
        assertEquals(GGMLType.Q8_0, out[0]);
        assertEquals(GGMLType.Q4_K, out[1]);
    }

    @Test
    public void outputAndTokenEmbeddingTypeFlags() {
        QuantizeParams p = params(QuantizeFtype.Q4_K_M)
                .outputTensorType(GGMLType.Q8_0)
                .tokenEmbeddingType(GGMLType.Q5_K)
                .build();
        String[] names = {"output.weight", "token_embd.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F32, GGMLType.F32};
        GGMLType[] out = decide(p, llama(), names, dims, src);
        assertEquals(GGMLType.Q8_0, out[0]);
        assertEquals(GGMLType.Q5_K, out[1]);
    }

    @Test
    public void shapeFallbackDemotesByBlock() {
        QuantizeTypes.Mixture mix = new QuantizeTypes.Mixture(params(QuantizeFtype.Q4_K_M).build(),
                llama(), List.of("blk.0.attn_q.weight"));
        // 96 divides 32 but not 256: Q4_K falls back to Q5_0
        GGMLType out = mix.target("blk.0.attn_q.weight", dims2(96, 96), GGMLType.F32, GGMLType.Q4_K);
        assertEquals(GGMLType.Q5_0, out);
        assertEquals(1, mix.fallbackCount());
        // 16 divides nothing bigger: the fallback itself falls to F16
        QuantizeTypes.Mixture mix2 = new QuantizeTypes.Mixture(params(QuantizeFtype.Q4_K_M).build(),
                llama(), List.of("blk.0.attn_q.weight"));
        GGMLType out2 = mix2.target("blk.0.attn_q.weight", dims2(16, 16), GGMLType.F32, GGMLType.Q4_K);
        assertEquals(GGMLType.F16, out2);
    }

    @Test
    public void copyKeepsEverything() {
        String[] names = {"blk.0.attn_q.weight", "token_embd.weight"};
        long[][] dims = {dims2(256, 256), dims2(256, 256)};
        GGMLType[] src = {GGMLType.F16, GGMLType.F16};
        GGMLType[] out = decide(params(QuantizeFtype.COPY).build(), llama(), names, dims, src);
        assertAll(GGMLType.F16, out);
    }

    @Test
    public void eligibilityExcludesTheSmallTensors() {
        QuantizeParams p = params(QuantizeFtype.Q8_0).build();
        QuantizeTypes.Mixture mix = new QuantizeTypes.Mixture(p, llama(),
                List.of("output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_gate_inp.weight",
                        "token_types.weight"));
        assertEquals(GGMLType.F32, mix.target("output_norm.weight", dims2(256, 1), GGMLType.F32,
                GGMLType.Q8_0));
        assertEquals(GGMLType.F32, mix.target("blk.0.attn_norm.weight", dims2(256, 1), GGMLType.F32,
                GGMLType.Q8_0));
        assertEquals(GGMLType.F32, mix.target("blk.0.ffn_gate_inp.weight", dims2(256, 8), GGMLType.F32,
                GGMLType.Q8_0));
        assertEquals(GGMLType.F32, mix.target("token_types.weight", dims2(8, 8), GGMLType.F32,
                GGMLType.Q8_0));
    }

    @Test
    public void imatrixRequirement() {
        assertTrue(QuantizeTypes.requiresImatrix("blk.0.ffn_down.weight", GGMLType.IQ2_XXS,
                QuantizeFtype.IQ2_XXS));
        assertTrue(QuantizeTypes.requiresImatrix("blk.0.ffn_down.weight", GGMLType.Q2_K,
                QuantizeFtype.Q2_K_S));
        assertFalse(QuantizeTypes.requiresImatrix("blk.0.ffn_down.weight", GGMLType.Q2_K,
                QuantizeFtype.Q2_K));
        assertFalse(QuantizeTypes.requiresImatrix("token_embd.weight", GGMLType.IQ2_XXS,
                QuantizeFtype.IQ2_XXS));
        assertFalse(QuantizeTypes.requiresImatrix("output.weight", GGMLType.IQ1_S,
                QuantizeFtype.IQ1_S));
    }

    private static final String[] IQ3_NAMES = {"token_embd.weight", "output.weight", "blk.0.attn_q.weight",
        "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_down.weight",
        "blk.1.ffn_down.weight"};

    private static GGMLType[] iq3Decide(QuantizeFtype ftype, boolean imatrix) {
        long[][] dims = new long[IQ3_NAMES.length][];
        Arrays.fill(dims, dims2(256, 256));
        GGMLType[] src = new GGMLType[IQ3_NAMES.length];
        Arrays.fill(src, GGMLType.F32);
        QuantizeTypes.ModelInfo model = QuantizeTypes.ModelInfo.of("llama", 8, 4, 2, 0, imatrix);
        return decide(params(ftype).build(), model, IQ3_NAMES, dims, src);
    }

    @Test
    public void iq3XxsRecipeFollowsTheImatrix() {
        GGMLType[] without = iq3Decide(QuantizeFtype.IQ3_XXS, false);
        assertEquals(GGMLType.IQ3_S, without[0]);    // token_embd
        assertEquals(GGMLType.Q5_K, without[1]);     // output
        assertEquals(GGMLType.IQ2_S, without[2]);    // attn_q
        assertEquals(GGMLType.IQ2_S, without[3]);    // attn_k
        assertEquals(GGMLType.IQ3_S, without[4]);    // attn_v without an imatrix
        assertEquals(GGMLType.IQ3_S, without[5]);    // attn_output
        assertEquals(GGMLType.Q4_K, without[6]);     // ffn_down 0 < 8/8
        assertEquals(GGMLType.Q3_K, without[7]);     // later ffn_down without an imatrix
        GGMLType[] with = iq3Decide(QuantizeFtype.IQ3_XXS, true);
        assertEquals(GGMLType.IQ3_XXS, with[4]);
        assertEquals(GGMLType.IQ3_XXS, with[7]);
    }

    @Test
    public void iq3MRecipeLiftsAttentionAndTheFirstFfnDown() {
        GGMLType[] out = iq3Decide(QuantizeFtype.IQ3_M, true);
        assertEquals(GGMLType.IQ3_S, out[0]);
        assertEquals(GGMLType.Q6_K, out[1]);
        assertEquals(GGMLType.IQ3_S, out[2]);
        assertEquals(GGMLType.Q4_K, out[4]);
        assertEquals(GGMLType.Q4_K, out[5]);
        assertEquals(GGMLType.Q4_K, out[6]);
        assertEquals(GGMLType.IQ3_S, out[7]);
    }

    @Test
    public void iq1SRecipeDropsAttnOutputToIq2Xxs() {
        GGMLType[] out = iq3Decide(QuantizeFtype.IQ1_S, true);
        assertEquals(GGMLType.Q2_K, out[0]);
        assertEquals(GGMLType.Q5_K, out[1]);
        assertEquals(GGMLType.IQ1_S, out[2]);
        assertEquals(GGMLType.Q2_K, out[4]);
        assertEquals(GGMLType.IQ2_XXS, out[5]);
        assertEquals(GGMLType.Q2_K, out[6]);         // ffn_down 0 < 8/8
        assertEquals(GGMLType.IQ1_S, out[7]);
    }

    private static void assertAll(GGMLType expected, GGMLType[] out) {
        for (GGMLType t : out) {
            assertEquals(expected, t);
        }
    }
}
