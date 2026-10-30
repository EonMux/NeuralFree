package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The part D hooks: the Step3.5 per-layer head/KV arrays, SwiGLU clamps, stored-minus-one norms, the squeezed
 * size-0 dims and the llama3 rope factors (step3.py:138-341), and the MuseSpark stored-minus-one norms, the
 * fused expert gate/up split, the router MetaP multiplier and the residual gate α/β synthesis
 * (muse_spark.py:53-155).
 */
public class HfMoeSwaHooksDTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ Step35Model

    private static final String STEP35_CONFIG = "'layer_types': ['sliding_attention', 'full_attention'],"
            + " 'num_attention_groups': 1,"
            + " 'attention_other_setting': {'num_attention_heads': 1, 'num_attention_groups': 1},"
            + " 'sliding_window': 4, 'head_dim': 4, 'moe_num_experts': 2, 'moe_top_k': 1,"
            + " 'moe_intermediate_size': 4, 'share_expert_dim': 4,"
            + " 'swiglu_limits': [2.0, 0.0], 'swiglu_limits_shared': [3.0, 0.0],"
            + " 'moe_layers_enum': '1', 'moe_every_n_layer': 2, 'num_nextn_predict_layers': 1,"
            + " 'local_rope_theta': 500000.0";

    @Test
    public void step35WritesThePerLayerArraysAndClamps() throws IOException {
        // the MTP count rides the load's includeMtp option (the converter reads it unconditionally)
        HfLoadContext ctx = context(dir(), Arch.STEP35, "Step3p5ForCausalLM", STEP35_CONFIG,
                java.util.Collections.emptyMap(), true);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.STEP35.ggufName());
        HfMoeSwaHooksD.step35Meta(ctx, kv, null);
        assertEquals(Arrays.asList(1, 2), kv.get(GgufKeys.ATTENTION_HEAD_COUNT));
        assertEquals(Arrays.asList(1, 1), kv.get(GgufKeys.ATTENTION_HEAD_COUNT_KV));
        assertEquals(Arrays.asList(true, false), kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
        java.util.List<?> clamp = (java.util.List<?>) kv.get(GgufKeys.SWIGLU_CLAMP_EXP);
        assertEquals(2.0, (Float) clamp.get(0), 0f);
        assertEquals(0.0, (Float) clamp.get(1), 0f);
        assertEquals(1, kv.get(GgufKeys.LEADING_DENSE_BLOCK_COUNT));
        assertEquals(2, kv.get(GgufKeys.MOE_EVERY_N_LAYERS));
        assertEquals(1, kv.get(GgufKeys.NEXTN_PREDICT_LAYERS));
        assertEquals(500000.0, ((Number) kv.get(GgufKeys.ROPE_FREQ_BASE_SWA)).doubleValue(), 0f);
    }

    @Test
    public void step35NormsCarryTheStoredMinusOneOffsetAndSqueezeTheSizeZeroDims() throws IOException {
        Map<String, RowSource> index = new java.util.LinkedHashMap<>();
        index.put("model.layers.0.moe.gate.weight", seq(2, 4, 1, 1));
        index.put("model.layers.0.post_attention_layernorm.weight", seq(2, 4));
        HfLoadContext ctx = context(dir(), Arch.STEP35, "Step3p5ForCausalLM", "", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksD.step35Tensors(ctx, name, ctx.index().get(name), sink);
        }
        List<RowSource> out = new ArrayList<>(sink.tensors.values());
        assertEquals(2, out.size());
        // a conv-shaped routing tensor is squeezed to its layout rows
        assertArrayEquals(new long[]{2, 4}, out.get(0).shape());
        // the norm holds a delta around one
        assertEquals(2, (long) out.get(1).shape()[0]);
        float[] norm = values(out.get(1));
        for (int i = 0; i < norm.length; i++) {
            assertEquals(i + 1.0f, norm[i], 0f);
        }
    }

    @Test
    public void step35GeneratesTheLlama3RopeFactors() throws IOException {
        String config = STEP35_CONFIG + ", 'rope_theta': 10000.0,"
                + " 'rope_scaling': {'rope_type': 'llama3', 'factor': 8.0,"
                + " 'original_max_position_embeddings': 8192, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0}";
        HfLoadContext ctx = context(dir(), Arch.STEP35, "Step3p5ForCausalLM", config,
                java.util.Collections.emptyMap(), false);
        RecordingSink sink = new RecordingSink();
        HfMoeSwaHooksD.step35Extra(ctx, sink);
        // padded to half the storage head width
        assertEquals(1, sink.tensors.size());
        RowSource freqs = sink.tensors.values().iterator().next();
        assertArrayEquals(new long[]{2}, freqs.shape());
        float[] f = values(freqs);
        // wavelengths 6.28..63 stand inside both boundaries: every factor is a finite ramp value
        for (float v : f) {
            assertTrue(v >= 1.0);
        }
    }

    // ------------------------------------------------------------------ MuseSparkModel

    private static final String MUSE_CONFIG = "'output_multiplier': 2.0, 'final_logit_softcapping': 30.0,"
            + " 'post_norm_eps': 1e-05, 'sliding_window': 4, 'layer_types': ['sliding_attention',"
            + " 'full_attention'], 'moe_hidden_size': 2, 'moe_intermediate_size': 4,"
            + " 'routed_scaling_factor': 1.0, 'residual_gate_temperature': 1.0";

    @Test
    public void museSparkWritesItsMetadataAndSynthesizesTheResidualGates() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.MUSE_SPARK, "MuseSparkForConditionalGeneration", MUSE_CONFIG);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.MUSE_SPARK.ggufName());
        HfMoeSwaHooksD.museSparkMeta(ctx, kv, null);
        assertEquals(2.0, ((Number) kv.get(GgufKeys.LOGIT_SCALE)).doubleValue(), 0f);
        assertEquals(Boolean.TRUE, kv.get(GgufKeys.EXPERT_WEIGHTS_NORM));
        assertEquals(Arrays.asList(true, false), kv.get(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN));
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksD.museSparkMeta(
                context(dir(), Arch.MUSE_SPARK, "MuseSparkForConditionalGeneration", MUSE_CONFIG
                        + ", 'layer_rope_theta': [10000.0, 5000.0]"),
                new HfMetadataBuilder(Arch.MUSE_SPARK.ggufName()), null));
        assertThrows(GgufFormatException.class, () -> HfMoeSwaHooksD.museSparkMeta(
                context(dir(), Arch.MUSE_SPARK, "MuseSparkForConditionalGeneration", MUSE_CONFIG
                        + ", 'mlp_layer_types': ['dense', 'sparse']"),
                new HfMetadataBuilder(Arch.MUSE_SPARK.ggufName()), null));
        Map<String, RowSource> index = new java.util.LinkedHashMap<>();
        index.put("model.layers.0.post_attention_residual_gate.gate", seq(4));
        HfLoadContext gate = context(dir(), Arch.MUSE_SPARK, "MuseSparkForConditionalGeneration", MUSE_CONFIG,
                index, false);
        RecordingSink sink = new RecordingSink();
        HfMoeSwaHooksD.museSparkTensors(gate, "model.layers.0.post_attention_residual_gate.gate",
                gate.index().get("model.layers.0.post_attention_residual_gate.gate"), sink);
        // β = σ(w / τ) at τ = 1 and α = √(max(σ(−w / τ) · (1 + β), ε))
        assertEquals(2, sink.tensors.size());
        double[] w = {0, 1, 2, 3};
        for (RowSource rs : sink.tensors.values()) {
            assertArrayEquals(new long[]{4}, rs.shape());
            float[] v = values(rs);
            int kind = (float) (1.0 / (1.0 + Math.exp(-w[2]))) == v[2] ? 1
                    : (float) Math.sqrt(Math.max((1.0 / (1.0 + Math.exp(w[2]))) * (1.0
                    + 1.0 / (1.0 + Math.exp(-w[2]))), 1e-3)) == v[2] ? 0 : -1;
            assertTrue("neither α nor β", kind == 0 || kind == 1);
            double logit;
            for (int i = 0; i < w.length; i++) {
                if (kind == 1) {
                    assertEquals(1.0 / (1.0 + Math.exp(-w[i])), v[i], 1e-5);
                } else {
                    double b = 1.0 / (1.0 + Math.exp(-w[i]));
                    assertEquals(Math.sqrt(Math.max((1.0 / (1.0 + Math.exp(w[i]))) * (1.0 + b), 1e-3)),
                            v[i], 1e-5);
                    logit = b;
                }
            }
        }
    }

    @Test
    public void museSparkSplitsTheFusedExpertsScalesTheRouterAndRaisesTheNorms() throws IOException {
        Map<String, RowSource> index = new java.util.LinkedHashMap<>();
        index.put("model.layers.0.mlp.experts.gate_up_proj", seq(2, 8, 4));
        index.put("model.layers.0.mlp.experts.down_proj", seq(2, 4, 8));
        index.put("model.layers.0.mlp.gate.weight", seq(2, 8));
        index.put("model.layers.0.input_layernorm.weight", seq(8));
        HfLoadContext ctx = context(dir(), Arch.MUSE_SPARK, "MuseSparkForConditionalGeneration",
                MUSE_CONFIG + ", 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeSwaHooksD.museSparkTensors(ctx, name, ctx.index().get(name), sink);
        }
        // gate/up experts (2), the transposed down (.weight appended), the ×2 router and the +1 norm
        assertEquals(5, sink.tensors.size());
        List<RowSource> out = new ArrayList<>(sink.tensors.values());
        assertArrayEquals(new long[]{2, 4, 4}, out.get(0).shape());
        assertArrayEquals(new long[]{2, 4, 4}, out.get(1).shape());
        assertArrayEquals(new long[]{2, 4, 8}, out.get(2).shape());
        // the router is scaled by the output multiplier of 2.0
        float[] router = values(out.get(3));
        for (int i = 0; i < router.length; i++) {
            assertEquals(2.0f * i, router[i], 0f);
        }
        // the delta-around-one norm becomes norm + 1
        float[] norm = values(out.get(4));
        for (int i = 0; i < norm.length; i++) {
            assertEquals(i + 1.0f, norm[i], 0f);
        }
    }

    /**
     * {@code museSparkTensors} of a query row: the stored rotate-half layout unpermutes per head and the synth
     * Q/K norm pair follows with the MetaP scale (1.0 here) as the q rows.
     */
    @Test
    public void museSparkUnpermutesTheQueriesAndSynthesizesTheQkNormPair() throws IOException {
        Map<String, RowSource> index = new java.util.LinkedHashMap<>();
        index.put("model.layers.0.self_attn.q_proj.weight", seq(8, 8));
        HfLoadContext ctx = context(dir(), Arch.MUSE_SPARK, "MuseSparkForConditionalGeneration", MUSE_CONFIG
                + ", 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4", index, false);
        RecordingSink sink = new RecordingSink();
        HfMoeSwaHooksD.museSparkTensors(ctx, "model.layers.0.self_attn.q_proj.weight",
                ctx.index().get("model.layers.0.self_attn.q_proj.weight"), sink);
        assertEquals(3, sink.tensors.size());
        List<RowSource> out = new ArrayList<>(sink.tensors.values());
        assertArrayEquals(new long[]{8, 8}, out.get(0).shape());
        // the synth pair: q rows carry the MetaP qk scale (1.0), k rows carry one
        assertArrayEquals(new long[]{4}, out.get(1).shape());
        assertArrayEquals(new long[]{4}, out.get(2).shape());
        for (float v : values(out.get(1))) {
            assertEquals(1.0f, v, 0f);
        }
        for (float v : values(out.get(2))) {
            assertEquals(1.0f, v, 0f);
        }
    }
}
