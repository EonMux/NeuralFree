package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.iota;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.planAll;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link HfDenseHooksA} against the converter classes it ports: the Gemma 1 and 2 norm shift, tied-{@code lm_head}
 * filter, soft-cappings and CodeGemma infill ids (LC/conversion/gemma.py:53-113) and the Gemma 3n AltUp stacking,
 * embedding padding, {@code *_scale} renames, zero norm shift and activation-sparsity scales (gemma.py:489-629).
 */
public class HfDenseHooksATest {

    private static final UnaryOperator<String> IDENTITY = UnaryOperator.identity();

    private static final String GEMMA2 = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'head_dim': 4, 'num_hidden_layers': 2, 'max_position_embeddings': 1024,"
            + " 'rms_norm_eps': 1e-06, 'vocab_size': 8, 'rope_theta': 1000000.0";

    // ------------------------------------------------------------------ Gemma (v1) and Gemma2

    @Test
    public void gemmaFilterDropsTheTiedLMHead() {
        assertNull(HfDenseHooksA.GEMMA_FILTER.filter("lm_head.weight", IDENTITY));
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfDenseHooksA.GEMMA_FILTER.filter("model.layers.0.self_attn.q_proj.weight", IDENTITY));
    }

    @Test
    public void gemmaTensorsAddOneToEveryNorm() throws IOException {
        RecordingSink out = planAll(context(Arch.GEMMA2, "Gemma2ForCausalLM", GEMMA2,
                Map.of("model.norm.weight", iota(8),
                        "model.layers.0.post_attention_layernorm.weight", iota(8),
                        "model.layers.0.mlp.up_proj.weight", iota(16))), HfDenseHooksA::gemmaTensors);
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, values(out.tensors.get("output_norm.weight")),
                0f);
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 7, 8},
                values(out.tensors.get("blk.0.post_attention_norm.weight")), 0f);
        assertArrayEquals(new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                values(out.tensors.get("blk.0.ffn_up.weight")), 0f);
    }

    @Test
    public void gemmaVocabCarriesTheCodegemmaInfillIds() {
        assertEquals(67, HfDenseHooksA.FIM_PRE_ID);
        assertEquals(69, HfDenseHooksA.FIM_SUF_ID);
        assertEquals(68, HfDenseHooksA.FIM_MID_ID);
        assertEquals(70, HfDenseHooksA.FIM_SEP_ID);
        assertEquals(107, HfDenseHooksA.FIM_EOT_ID);
    }

    @Test
    public void gemma2MetaWritesSoftcapsAndTheSlidingWindow() throws IOException {
        HfLoadContext ctx = context(Arch.GEMMA2, "Gemma2ForCausalLM", GEMMA2
                + ", 'attn_logit_softcapping': 50.0, 'final_logit_softcapping': 30.0, 'sliding_window': 512",
                Map.of());
        HfMetadata m = meta(HfDenseHooksA::gemma2Meta, ctx);
        assertEquals(50.0f, num(m.get(key(Arch.GEMMA2, "attn_logit_softcapping"))), 0f);
        assertEquals(30.0f, num(m.get(key(Arch.GEMMA2, "final_logit_softcapping"))), 0f);
        assertEquals(512, num(m.get(key(Arch.GEMMA2, "attention.sliding_window"))));
        assertEquals(1024, num(m.get(key(Arch.GEMMA2, "context_length"))));
        assertEquals(4, num(m.get(key(Arch.GEMMA2, "attention.key_length"))));
    }

    /** The float value of a boxed metadata number. */
    private static float num(Object v) {
        return ((Number) v).floatValue();
    }

    private static HfMetadata meta(MetaHook hook, HfLoadContext ctx) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        hook.apply(ctx, kv, null);
        return kv.build();
    }

    private interface MetaHook {
        void apply(HfLoadContext ctx, HfMetadataBuilder kv, com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView plan)
                throws IOException;
    }

    private static String key(Arch arch, String tail) {
        return String.format(Arch.GEMMA2 == arch ? tail : tail, arch.ggufName());
    }

    // ------------------------------------------------------------------ Gemma3N

    private static final String GEMMA3N = GEMMA2 + ", 'altup_active_idx': 1, 'altup_num_inputs': 4,"
            + " 'hidden_size_per_layer_input': 8, 'num_kv_shared_layers': 2,"
            + " 'activation_sparsity_pattern': [0.9, 0.5], 'layer_types': ['sliding_attention',"
            + " 'full_attention'], 'sliding_window': 512";

    @Test
    public void gemma3nRenamesScaleParametersToWeights() {
        assertEquals("model.layers.0.altup_correct_coef.weight",
                HfDenseHooksA.GEMMA3N_FILTER.filter("model.layers.0.altup_correct_coef.scale", IDENTITY));
        assertEquals("model.layers.0.plain.weight",
                HfDenseHooksA.GEMMA3N_FILTER.filter("model.layers.0.plain.weight", IDENTITY));
    }

    @Test
    public void gemma3nKeepsNormsUnshifted() throws IOException {
        RecordingSink out = planAll(context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA3N,
                Map.of("model.norm.weight", iota(8))), HfDenseHooksA::gemma3nTensors);
        assertArrayEquals(new float[] {0, 1, 2, 3, 4, 5, 6, 7}, values(out.tensors.get("output_norm.weight")), 0f);
    }

    @Test
    public void gemma3nStacksTheAltUpSlotsByNameId() throws IOException {
        Map<String, RowSource> tensors = new LinkedHashMap<>();
        // out of order on purpose: the converter indexes the slots by the name's id
        tensors.put("model.altup_projections.2.weight", iota(8, 16));
        tensors.put("model.altup_projections.0.weight", iota(8, 16));
        tensors.put("model.altup_projections.1.weight", iota(8, 16));
        RecordingSink out = planAll(context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA3N, tensors),
                HfDenseHooksA::gemma3nTensors);
        String merged = ctx(Arch.GEMMA3N, GEMMA3N).names().mapOrThrow("model.altup_projections.weight");
        assertEquals(1, out.tensors.size());
        assertTrue(out.tensors.containsKey(merged));
        assertArrayEquals(threeSlots(8 * 16), values(out.tensors.get(merged)), 0f);
    }

    @Test
    public void gemma3nStacksTheUnembedProjectionSlots() throws IOException {
        Map<String, RowSource> tensors = new LinkedHashMap<>();
        for (int slot = 0; slot < HfDenseHooksA.ALTUP_SLOTS; slot++) {
            tensors.put("model.altup_unembed_projections." + slot + ".weight", iota(16, 8));
        }
        RecordingSink out = planAll(context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA3N, tensors),
                HfDenseHooksA::gemma3nTensors);
        String merged = ctx(Arch.GEMMA3N, GEMMA3N).names().mapOrThrow("model.altup_unembed_projections.weight");
        assertEquals(1, out.tensors.size());
        assertTrue(out.tensors.containsKey(merged));
        assertArrayEquals(threeSlots(16 * 8), values(out.tensors.get(merged)), 0f);
    }

    /** The stack of 3 identical iota slots. */
    private static float[] threeSlots(int slotSize) {
        float[] single = new float[slotSize];
        for (int i = 0; i < slotSize; i++) {
            single[i] = i;
        }
        float[] out = new float[3 * slotSize];
        for (int s = 0; s < 3; s++) {
            System.arraycopy(single, 0, out, s * slotSize, slotSize);
        }
        return out;
    }

    @Test
    public void gemma3nPadsTheTokenEmbeddingToTheVocabulary() throws IOException {
        RecordingSink out = planAll(context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA3N,
                Map.of("model.embed_tokens.weight", iota(6, 8))), HfDenseHooksA::gemma3nTensors);
        RowSource s = out.tensors.get("token_embd.weight");
        assertEquals(8, s.shape()[0]);
        float[] v = values(s);
        assertArrayEquals(new float[] {0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, v, 0f);
    }

    @Test
    public void gemma3nMetaWritesAltupSparsityAndTheWindowPattern() throws IOException {
        HfLoadContext ctx = context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA3N, Map.of());
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.GEMMA3N.ggufName());
        HfDenseHooksA.gemma3nMeta(ctx, kv, null);
        HfMetadata m = kv.build();
        String g = Arch.GEMMA3N.ggufName();
        assertEquals(1, m.get(String.format(GgufKeys.ALTUP_ACTIVE_IDX, g)));
        assertEquals(4, m.get(String.format(GgufKeys.ALTUP_NUM_INPUTS, g)));
        assertEquals(8, m.get(String.format(GgufKeys.EMBEDDING_LENGTH_PER_LAYER, g)));
        assertEquals(2, m.get(String.format(GgufKeys.ATTENTION_SHARED_KV_LAYERS, g)));
        assertArrayEquals(new Object[] {1.281555f, 0.0f},
                ((java.util.List<?>) m.get(String.format(GgufKeys.ACTIVATION_SPARSITY_SCALE, g))).toArray());
        assertArrayEquals(new Object[] {true, false},
                ((java.util.List<?>) m.get(String.format(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, g)))
                        .toArray());
    }

    @Test
    public void gemma3nRejectsOtherAltupCounts() throws IOException {
        HfLoadContext ctx = context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA2
                + ", 'altup_active_idx': 1, 'altup_num_inputs': 3, 'hidden_size_per_layer_input': 8,"
                + " 'num_kv_shared_layers': 2, 'activation_sparsity_pattern': [0.9, 0.5],"
                + " 'layer_types': ['sliding_attention', 'full_attention'], 'sliding_window': 512", Map.of());
        try {
            HfDenseHooksA.gemma3nMeta(ctx, new HfMetadataBuilder(Arch.GEMMA3N.ggufName()), null);
            fail("altup_num_inputs 3 must be refused");
        } catch (GgufFormatException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("altup_num_inputs"));
        }
    }

    @Test
    public void activationSparsityScaleIsTheNormalInverseCdf() throws IOException {
        assertTrue(HfDenseHooksA.activationSparsityScale(context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA2
                + ", 'activation_sparsity_pattern': [0.9, 0.5, 0.3]", Map.of()).config(), 0)
                .size() == 3);
        assertArrayEquals(new double[] {1.281552, 0.0, -0.524401},
                HfDenseHooksA.activationSparsityScale(context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA2
                        + ", 'activation_sparsity_pattern': [0.9, 0.5, 0.3]", Map.of()).config(), 0).stream()
                        .mapToDouble(Float::doubleValue).toArray(), 1e-5);
    }

    @Test
    public void outOfRangeSparsitiesAreRejected() throws IOException {
        for (String bad : new String[] {"[0.0]", "[1.5]", "['x']", "[]"}) {
            try {
                HfDenseHooksA.activationSparsityScale(context(Arch.GEMMA3N, "Gemma3nForCausalLM", GEMMA2
                        + ", 'activation_sparsity_pattern': " + bad, Map.of()).config(), 0);
                fail("sparsity pattern " + bad + " must be refused");
            } catch (GgufFormatException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("activation_sparsity_pattern"));
            }
        }
    }

    private static HfLoadContext ctx(Arch arch, String members) throws IOException {
        return context(Arch.GEMMA3N, "Gemma3nForCausalLM", members, Map.of());
    }
}
