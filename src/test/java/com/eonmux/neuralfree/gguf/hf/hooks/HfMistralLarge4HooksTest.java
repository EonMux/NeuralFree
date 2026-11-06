package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfMistralLarge4Hooks} on a synthetic Mistral Large 4 checkpoint: the Mistral MoE remap, the pinned layout the
 * runtime implements (the indexer pattern, the KV source map, the LeRoPE and the tanh cap), the metadata, the tensor
 * surgery (LeRoPE factors, SSMax scales, indexer query norm, indexer rename, per-layer {@code wkv_b} split) and the
 * expert name filter.
 */
public class HfMistralLarge4HooksTest {

    private static final String BASE = "'dim': 8, 'hidden_dim': 16, 'n_layers': 2, 'n_heads': 2, 'n_kv_heads': 1,"
            + " 'norm_eps': 1e-05, 'vocab_size': 16, 'max_position_embeddings': 128, 'rope_theta': 10000.0,"
            + " 'qk_rope_head_dim': 2, 'qk_nope_head_dim': 2, 'v_head_dim': 2, 'kv_lora_rank': 4,"
            + " 'nope': 'rope,nope', 'softmax_tanh_gating': 50.0,"
            + " 'moe': {'num_experts': 2, 'expert_hidden_dim': 4, 'num_experts_per_tok': 1,"
            + "   'num_shared_experts': 1, 'route_every_n': 1, 'first_k_dense_replace': 0, 'routed_scale': 1.0},"
            + " 'dsa': {'n_heads': 2, 'head_dim': 2, 'topk': 1, 'sharing': 'full,shared', 'norm_eps': 1e-06,"
            + "   'rope_head_dim': 2, 'match_nope': true, 'ssmax_scaling': true},"
            + " 'kv_sharing': {'source_layers_str': '0,0'},"
            + " 'yarn': {'factor': 4.0, 'beta': 32.0, 'alpha': 1.0,"
            + "   'original_max_position_embeddings': 128, 'apply_scale': false}";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private static float[] offset(float base, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = base + i;
        }
        return v;
    }

    private static HfConfig large4(String extra) {
        return config("{'architectures': ['MistralLarge4ForConditionalGeneration'], " + BASE + extra + "}");
    }

    private static HfLoadContext load(HfConfig cfg, HfTensorIndex index) {
        return context(cfg, entry("MistralLarge4ForConditionalGeneration"), index);
    }

    @Test
    public void theRemapGivesTheDeepseekNamesAndThePatternsExpand() {
        HfConfig fixed = HfMistralLarge4Hooks.fixup(large4(""));
        assertEquals(8, fixed.requireInt("hidden_size"));
        assertEquals(1, fixed.requireInt("num_key_value_heads"));
        assertEquals(4, fixed.requireInt("moe_intermediate_size"));
        assertEquals(2, fixed.requireInt("n_routed_experts"));
        assertEquals(true, fixed.flat().get("norm_topk_prob"));
        assertEquals("softmax", fixed.flat().get("scoring_func"));
        assertTrue(fixed.flat().containsKey("topk_method"));
        assertNull(fixed.flat().get("topk_method"));
        assertEquals(Arrays.asList(true, false), HfMistralLarge4Hooks.ropePattern(fixed));
        assertEquals(Arrays.asList(true, false), HfMistralLarge4Hooks.indexerTypes(fixed));
    }

    @Test
    public void aLayoutTheRuntimeDoesNotImplementIsRefusedByKey() {
        for (String[] broken : new String[][]{
                {"'match_nope': true", "'match_nope': false", "match_nope"},
                {"'ssmax_scaling': true", "'ssmax_scaling': false", "ssmax_scaling"},
                {"'apply_scale': false", "'apply_scale': true", "apply_scale"},
                {"'rope_head_dim': 2", "'rope_head_dim': 4", "dsa.rope_head_dim"}}) {
            String json = BASE.replace(broken[0], broken[1]);
            HfConfig cfg = config("{'architectures': ['MistralLarge4ForConditionalGeneration'], " + json + "}");
            GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfMistralLarge4Hooks.fixup(cfg));
            assertTrue(e.getMessage(), e.getMessage().contains(broken[2]));
        }
        HfConfig noSharing = config("{'architectures': ['MistralLarge4ForConditionalGeneration'], "
                + BASE.replace("'kv_sharing': {'source_layers_str': '0,0'}", "'kv_sharing': {'windows': 2}") + "}");
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfMistralLarge4Hooks.fixup(noSharing));
        assertTrue(e.getMessage(), e.getMessage().contains("kv_sharing"));
    }

    @Test
    public void aBlockFp8CheckpointIsAcceptedAndAnotherQuantizationIsRefused() {
        String block = ", 'quantization': {'qformat_weight': 'fp8_e4m3', 'qscheme_weight': 'BLOCK',"
                + " 'block_size_out': 128, 'block_size_in': 128}";
        assertEquals(2, HfMistralLarge4Hooks.fixup(large4(block)).requireInt("num_hidden_layers"));
        String tensor = ", 'quantization': {'qformat_weight': 'fp8_e4m3', 'qscheme_weight': 'TENSOR'}";
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfMistralLarge4Hooks.fixup(large4(tensor)));
        assertTrue(e.getMessage(), e.getMessage().contains("qscheme_weight"));
    }

    @Test
    public void theMetadataWritesTheIndexerAndTheTanhCap() throws IOException {
        HfLoadContext ctx = load(large4(""), HfTensorIndex.of(Map.of()));
        HfMetadataBuilder b = new HfMetadataBuilder(ctx.arch().ggufName());
        entry("MistralLarge4ForConditionalGeneration").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(2, m.get("mistral-large4.attention.indexer.head_count"));
        assertEquals(2, m.get("mistral-large4.attention.indexer.key_length"));
        assertEquals(1, m.get("mistral-large4.attention.indexer.top_k"));
        assertEquals(Arrays.asList(true, false), m.get("mistral-large4.attention.indexer.types"));
        assertEquals(1e-6f, m.get("mistral-large4.attention.layer_norm_epsilon"));
        assertEquals(Arrays.asList(true, false), m.get("mistral-large4.attention.rope_pattern"));
        assertEquals(Arrays.asList(0, 0), m.get("mistral-large4.attention.kv_source_map"));
        assertEquals(1, m.get("mistral-large4.expert_gating_func"));
        assertEquals(50.0f, m.get("mistral-large4.final_logit_softcapping"));
        assertEquals("yarn", m.get("mistral-large4.rope.scaling.type"));
        assertEquals(4.0f, m.get("mistral-large4.rope.scaling.factor"));
        assertEquals(1.0f, m.get("mistral-large4.rope.scaling.yarn_log_multiplier"));
        assertEquals(4, m.get("mistral-large4.attention.key_length_mla"));
        assertEquals(2, m.get("mistral-large4.attention.value_length_mla"));
        assertEquals(4, m.get("mistral-large4.attention.kv_lora_rank"));
        assertEquals(1, m.get("mistral-large4.attention.head_count_kv"));
        assertEquals(2, m.get("mistral-large4.rope.dimension_count"));
    }

    @Test
    public void theLeropeFactorsAreWrittenAsFrequenciesAndTheScalesSplit() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("lerope.alpha", new long[]{2}, 0f, (float) Math.log(2))
                .f32("model.layers.0.self_attn.ssmax_scaling.s", new long[]{2}, 1f, 2f)
                .f32("model.layers.0.self_attn.ssmax_scaling.b", new long[]{2}, 3f, 4f)
                .f32("model.layers.0.self_attn.indexer.ssmax_scaling.s", new long[]{2}, 5f, 6f)
                .f32("model.layers.0.self_attn.indexer.q_input_norm.weight", new long[]{2}, 7f, 8f)
                .f32("model.layers.0.self_attn.indexer.q_input_norm.bias", new long[]{2}, 9f, 10f)
                .index();
        Sink out = plan(load(large4(""), index));
        assertArrayEquals(new float[]{1f, 0.5f}, floats(out.get("rope_freqs.weight")), 1e-6f);
        assertArrayEquals(new float[]{1f, 2f}, floats(out.get("blk.0.attn_ssmax.weight")), 0f);
        assertArrayEquals(new float[]{3f, 4f}, floats(out.get("blk.0.attn_ssmax.bias")), 0f);
        assertArrayEquals(new float[]{5f, 6f}, floats(out.get("blk.0.indexer.ssmax.weight")), 0f);
        assertArrayEquals(new float[]{7f, 8f}, floats(out.get("blk.0.indexer.q_norm.weight")), 0f);
        assertArrayEquals(new float[]{9f, 10f}, floats(out.get("blk.0.indexer.q_norm.bias")), 0f);
    }

    @Test
    public void theWkvBSplitsPerLayerAndTheNopeLayerWidensItsKeyHead() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.layers.0.attention.wkv_b.weight", new long[]{8, 4}, seq(32))
                .f32("model.layers.1.attention.wkv_b.weight", new long[]{12, 4}, offset(100, 48))
                .index();
        Sink out = plan(load(large4(""), index));
        assertArrayEquals(new long[]{2, 4, 2}, out.get("blk.0.attn_k_b.weight").shape());
        assertArrayEquals(new float[]{0f, 4f, 1f, 5f, 2f, 6f, 3f, 7f,
                16f, 20f, 17f, 21f, 18f, 22f, 19f, 23f}, floats(out.get("blk.0.attn_k_b.weight")), 0f);
        assertArrayEquals(new long[]{2, 2, 4}, out.get("blk.0.attn_v_b.weight").shape());
        assertArrayEquals(new float[]{8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f,
                24f, 25f, 26f, 27f, 28f, 29f, 30f, 31f}, floats(out.get("blk.0.attn_v_b.weight")), 0f);
        // layer 1 is NoPE: its key head takes the rotary width as well (12 = 2 * (4 + 2))
        assertArrayEquals(new long[]{2, 4, 4}, out.get("blk.1.attn_k_b.weight").shape());
        assertArrayEquals(new long[]{2, 2, 4}, out.get("blk.1.attn_v_b.weight").shape());
        assertArrayEquals(new float[]{100f, 104f, 108f, 112f},
                Arrays.copyOf(floats(out.get("blk.1.attn_k_b.weight")), 4), 0f);
        assertArrayEquals(new float[]{116f, 117f, 118f, 119f, 120f, 121f, 122f, 123f},
                Arrays.copyOf(floats(out.get("blk.1.attn_v_b.weight")), 8), 0f);
    }

    @Test
    public void aWkvBOfTheWrongShapeIsRefused() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt.f32("model.layers.0.attention.wkv_b.weight", new long[]{6, 4}, seq(24)).index();
        HfLoadContext ctx = load(large4(""), index);
        Sink out = new Sink();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfMistralLarge4Hooks.tensors(ctx, "model.layers.0.attention.wkv_b.weight",
                        index.get("model.layers.0.attention.wkv_b.weight"), out));
        assertTrue(e.getMessage(), e.getMessage().contains("wkv_b"));
    }

    @Test
    public void theFilterDropsTheDraftTensorsAndRenamesTheExperts() {
        assertNull(HfMistralLarge4Hooks.FILTER.filter("mtp_dspark.backbone.layers.0.x.weight", n -> n));
        assertNull(HfMistralLarge4Hooks.FILTER.filter("model.layers.0.self_attn.sam_model.x", n -> n));
        assertNull(HfMistralLarge4Hooks.FILTER.filter("qwen2_model.layers.0.x", n -> n));
        assertEquals("model.layers.0.mlp.experts.1.gate_proj.weight",
                HfMistralLarge4Hooks.FILTER.filter("layers.0.experts.1.w1.weight", n -> n));
        assertEquals("model.layers.0.mlp.experts.1.up_proj.weight",
                HfMistralLarge4Hooks.FILTER.filter("layers.0.experts.1.w3.weight", n -> n));
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfMistralLarge4Hooks.FILTER.filter("model.layers.0.self_attn.q_proj.weight", n -> n));
    }

    @Test
    public void theEntryResolvesToMistralLarge4() {
        assertEquals(Arch.MISTRAL_LARGE4, entry("MistralLarge4ForConditionalGeneration").arch()
                .resolve(large4(""), HfTensorIndex.of(Map.of())));
        assertTrue(entry("MistralLarge4ForConditionalGeneration").undoPermute());
    }
}
