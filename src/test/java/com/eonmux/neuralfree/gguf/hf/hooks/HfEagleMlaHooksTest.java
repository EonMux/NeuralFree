package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek-MTP-shaped EAGLE draft route ({@link HfEagleMlaHooks} behind {@link HfEagle1Hooks}): the shared
 * {@code EagleDeepSeekMTPModel} name routes a checkpoint that declares {@code kv_lora_rank} to the MLA mapping — the
 * DeepSeek MLA metadata, the {@code kv_b_proj} split, the fusion input norms, the stacked routed experts and the
 * {@code model.}-prefixed names of vLLM's loader — while a checkpoint without it keeps the llama-shaped EAGLE-1
 * mapping. The unsupported variants are refused by name.
 */
public class HfEagleMlaHooksTest {

    private static final String MLA = "{'architectures': ['EagleDeepSeekMTPModel'], 'hidden_size': 64,"
            + " 'intermediate_size': 128, 'num_attention_heads': 4, 'num_key_value_heads': 4,"
            + " 'num_hidden_layers': 1, 'vocab_size': 320, 'max_position_embeddings': 64,"
            + " 'rms_norm_eps': 1e-06, 'kv_lora_rank': 32, 'qk_nope_head_dim': 16, 'qk_rope_head_dim': 16,"
            + " 'v_head_dim': 32, 'rope_theta': 10000%s}";

    private static final String LLAMA = "{'architectures': ['EagleModel'], 'hidden_size': 64,"
            + " 'intermediate_size': 128, 'num_attention_heads': 4, 'num_key_value_heads': 1,"
            + " 'num_hidden_layers': 1, 'vocab_size': 320, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
            + " 'rope_theta': 10000%s}";

    private static final String MOE = ", 'n_routed_experts': 4, 'num_experts_per_tok': 2,"
            + " 'moe_intermediate_size': 48, 'n_shared_experts': 1, 'norm_topk_prob': true,"
            + " 'routed_scaling_factor': 2.0, 'scoring_func': 'sigmoid', 'first_k_dense_replace': 0";

    private static final int TARGET_VOCAB = 320;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private final List<AutoCloseable> closing = new ArrayList<>();

    @org.junit.After
    public void close() throws Exception {
        for (AutoCloseable c : closing) {
            c.close();
        }
    }

    // ------------------------------------------------------------------ the route

    @Test
    public void theMlaCheckpointIsRoutedByItsGeometry() {
        assertTrue(HfEagleMlaHooks.isMla(config(mla(""))));
        assertFalse(HfEagleMlaHooks.isMla(config(llama(""))));
        HfConfig fixed = HfEagle1Hooks.fixup(config(mla("")));
        assertEquals("EagleForCausalLM", fixed.architecture());
        assertEquals(1, fixed.requireInt("num_hidden_layers"));
    }

    /** DeepSeek's configuration defaults {@code kv_lora_rank} to 512, so the head triple alone routes the draft. */
    @Test
    public void aDraftWithoutTheKvRankUsesTheDeepseekDefault() throws Exception {
        HfMetadata md = meta("EagleDeepSeekMTPModel", "{'architectures': ['EagleDeepSeekMTPModel'],"
                + " 'hidden_size': 64, 'num_attention_heads': 4, 'num_key_value_heads': 4,"
                + " 'num_hidden_layers': 1, 'vocab_size': 320, 'qk_nope_head_dim': 16,"
                + " 'qk_rope_head_dim': 16, 'v_head_dim': 32}");
        assertEquals(512, md.get("eagle.attention.kv_lora_rank"));
        assertEquals(528, md.get("eagle.attention.key_length"));
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void metadataWritesTheMlaGeometry() throws Exception {
        HfMetadata md = meta("EagleDeepSeekMTPModel", mla(", 'q_lora_rank': 16"));
        assertEquals(320, md.get("eagle.vocab_size"));
        assertEquals(64, md.get("eagle.embedding_length"));
        assertEquals(1, md.get("eagle.block_count"));
        assertEquals(1, md.get("eagle.attention.head_count_kv"));
        assertEquals(48, md.get("eagle.attention.key_length"));
        assertEquals(32, md.get("eagle.attention.value_length"));
        assertEquals(32, md.get("eagle.attention.key_length_mla"));
        assertEquals(32, md.get("eagle.attention.value_length_mla"));
        assertEquals(16, md.get("eagle.rope.dimension_count"));
        assertEquals(32, md.get("eagle.attention.kv_lora_rank"));
        assertEquals(16, md.get("eagle.attention.q_lora_rank"));
        assertEquals(Collections.singletonList(3), md.get("eagle.target_layers"));
        assertEquals(8, md.get("eagle.target_hidden_size"));
        assertEquals(false, md.get("eagle.norm_before_residual"));
        assertEquals(false, md.get("eagle.norm_before_fc"));
    }

    @Test
    public void aDirectQueryDraftWritesNoQueryRank() throws Exception {
        assertNull(meta("EagleDeepSeekMTPModel", mla("")).get("eagle.attention.q_lora_rank"));
    }

    @Test
    public void anExpertDraftWritesTheMoeKeys() throws Exception {
        HfMetadata md = meta("EagleDeepSeekMTPModel", mla(MOE));
        assertEquals(4, md.get("eagle.expert_count"));
        assertEquals(2, md.get("eagle.expert_used_count"));
        assertEquals(1, md.get("eagle.expert_shared_count"));
        assertEquals(48, md.get("eagle.expert_feed_forward_length"));
        assertEquals(0, md.get("eagle.leading_dense_block_count"));
        assertEquals(2.0f, md.get("eagle.expert_weights_scale"));
        assertEquals(true, md.get("eagle.expert_weights_norm"));
        assertEquals(2, md.get("eagle.expert_gating_func"));
    }

    @Test
    public void theYarnLogMultiplierIsTheDeepSeekScale() throws Exception {
        HfMetadata md = meta("EagleDeepSeekMTPModel", mla(", 'rope_scaling': {'type': 'yarn', 'factor': 4.0,"
                + " 'original_max_position_embeddings': 64, 'mscale_all_dim': 2.0}"));
        assertEquals(0.2, (Float) md.get("eagle.rope.scaling.yarn_log_multiplier"), 1e-6f);
    }

    // ------------------------------------------------------------------ the non-MLA route

    @Test
    public void aLlamaShapedDraftKeepsTheLlamaMapping() throws Exception {
        HfMetadata md = meta("EagleModel", llama(""));
        assertNull(md.get("eagle.attention.kv_lora_rank"));
        assertNull(md.get("eagle.attention.key_length_mla"));
        assertEquals(16, md.get("eagle.rope.dimension_count"));
    }

    // ------------------------------------------------------------------ the tensors

    @Test
    public void theMlaTensorsMapToTheDraftBlock() throws Exception {
        Checkpoint ckpt = new Checkpoint(tmp.newFolder("mla").toPath());
        ckpt.f32("embed_tokens.weight", new long[]{320, 64}, new float[320 * 64]);
        ckpt.f32("enorm.weight", new long[]{64}, new float[64]);
        ckpt.f32("hnorm.weight", new long[]{64}, new float[64]);
        ckpt.f32("fc.weight", new long[]{64, 128}, new float[64 * 128]);
        ckpt.f32("norm.weight", new long[]{64}, new float[64]);
        ckpt.f32("lm_head.weight", new long[]{320, 64}, new float[320 * 64]);
        ckpt.f32("layers.0.input_layernorm.weight", new long[]{64}, new float[64]);
        ckpt.f32("layers.0.self_attn.q_a_proj.weight", new long[]{16, 64}, new float[16 * 64]);
        ckpt.f32("layers.0.self_attn.q_a_layernorm.weight", new long[]{16}, new float[16]);
        ckpt.f32("layers.0.self_attn.q_b_proj.weight", new long[]{128, 16}, new float[128 * 16]);
        ckpt.f32("layers.0.self_attn.kv_a_proj_with_mqa.weight", new long[]{48, 64}, new float[48 * 64]);
        ckpt.f32("layers.0.self_attn.kv_a_layernorm.weight", new long[]{32}, new float[32]);
        ckpt.f32("layers.0.self_attn.kv_b_proj.weight", new long[]{192, 32}, new float[192 * 32]);
        ckpt.f32("layers.0.self_attn.o_proj.weight", new long[]{64, 128}, new float[64 * 128]);
        ckpt.f32("layers.0.post_attention_layernorm.weight", new long[]{64}, new float[64]);
        ckpt.f32("layers.0.mlp.gate_proj.weight", new long[]{128, 64}, new float[128 * 64]);
        ckpt.f32("layers.0.mlp.up_proj.weight", new long[]{128, 64}, new float[128 * 64]);
        ckpt.f32("layers.0.mlp.down_proj.weight", new long[]{64, 128}, new float[64 * 128]);
        HfTensorIndex index = ckpt.index();
        closing.add(ckpt::close);

        Sink planned = plan(context(config(mla(", 'q_lora_rank': 16")), entry("EagleDeepSeekMTPModel"), index));
        assertEquals(new LinkedHashSet<>(Arrays.asList(
                "token_embd.weight", "enc.enorm.weight", "enc.hnorm.weight", "fc.weight", "output_norm.weight",
                "output.weight", "blk.0.attn_norm.weight", "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight",
                "blk.0.attn_q_b.weight", "blk.0.attn_kv_a_mqa.weight", "blk.0.attn_kv_a_norm.weight",
                "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight", "blk.0.attn_output.weight",
                "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight")),
                planned.added.keySet());
        assertArrayEquals(new long[]{4, 32, 16}, planned.get("blk.0.attn_k_b.weight").shape());
        assertArrayEquals(new long[]{4, 32, 32}, planned.get("blk.0.attn_v_b.weight").shape());
    }

    @Test
    public void aDirectQueryCheckpointMapsItsQueryProjection() throws Exception {
        Checkpoint ckpt = new Checkpoint(tmp.newFolder("direct").toPath());
        ckpt.f32("embed_tokens.weight", new long[]{320, 64}, new float[320 * 64]);
        ckpt.f32("fc.weight", new long[]{64, 128}, new float[64 * 128]);
        ckpt.f32("layers.0.self_attn.q_proj.weight", new long[]{128, 64}, new float[128 * 64]);
        ckpt.f32("layers.0.self_attn.kv_a_proj_with_mqa.weight", new long[]{48, 64}, new float[48 * 64]);
        ckpt.f32("layers.0.self_attn.kv_a_layernorm.weight", new long[]{32}, new float[32]);
        ckpt.f32("layers.0.self_attn.kv_b_proj.weight", new long[]{192, 32}, new float[192 * 32]);
        ckpt.f32("layers.0.self_attn.o_proj.weight", new long[]{64, 128}, new float[64 * 128]);
        HfTensorIndex index = ckpt.index();
        closing.add(ckpt::close);

        Sink planned = plan(context(config(mla("")), entry("EagleDeepSeekMTPModel"), index));
        assertTrue(planned.added.keySet().toString(), planned.added.containsKey("blk.0.attn_q.weight"));
        assertFalse(planned.added.keySet().toString(), planned.added.containsKey("blk.0.attn_q_a.weight"));
    }

    @Test
    public void theRoutedExpertsStackIntoTheLayerTensors() throws Exception {
        Checkpoint ckpt = new Checkpoint(tmp.newFolder("moe").toPath());
        ckpt.f32("embed_tokens.weight", new long[]{320, 64}, new float[320 * 64]);
        ckpt.f32("fc.weight", new long[]{64, 128}, new float[64 * 128]);
        ckpt.f32("layers.0.self_attn.q_proj.weight", new long[]{128, 64}, new float[128 * 64]);
        ckpt.f32("layers.0.self_attn.kv_a_proj_with_mqa.weight", new long[]{48, 64}, new float[48 * 64]);
        ckpt.f32("layers.0.self_attn.kv_a_layernorm.weight", new long[]{32}, new float[32]);
        ckpt.f32("layers.0.self_attn.kv_b_proj.weight", new long[]{192, 32}, new float[192 * 32]);
        ckpt.f32("layers.0.self_attn.o_proj.weight", new long[]{64, 128}, new float[64 * 128]);
        ckpt.f32("layers.0.mlp.gate.weight", new long[]{4, 64}, new float[4 * 64]);
        for (int x = 0; x < 4; x++) {
            ckpt.f32("layers.0.mlp.experts." + x + ".gate_proj.weight", new long[]{48, 64}, new float[48 * 64]);
            ckpt.f32("layers.0.mlp.experts." + x + ".up_proj.weight", new long[]{48, 64}, new float[48 * 64]);
            ckpt.f32("layers.0.mlp.experts." + x + ".down_proj.weight", new long[]{64, 48}, new float[64 * 48]);
        }
        ckpt.f32("layers.0.mlp.shared_experts.gate_proj.weight", new long[]{48, 64}, new float[48 * 64]);
        ckpt.f32("layers.0.mlp.shared_experts.up_proj.weight", new long[]{48, 64}, new float[48 * 64]);
        ckpt.f32("layers.0.mlp.shared_experts.down_proj.weight", new long[]{64, 48}, new float[64 * 48]);
        HfTensorIndex index = ckpt.index();
        closing.add(ckpt::close);

        Sink planned = plan(context(config(mla(MOE)), entry("EagleDeepSeekMTPModel"), index));
        assertEquals(new LinkedHashSet<>(Arrays.asList(
                "token_embd.weight", "fc.weight", "blk.0.attn_q.weight", "blk.0.attn_kv_a_mqa.weight",
                "blk.0.attn_kv_a_norm.weight", "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight",
                "blk.0.attn_output.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight",
                "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_shexp.weight",
                "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight")), planned.added.keySet());
        assertArrayEquals(new long[]{4, 48, 64}, planned.get("blk.0.ffn_gate_exps.weight").shape());
    }

    @Test
    public void theLoaderPrefixAndTheLayerIndexAreReduced() {
        assertEquals("model.enorm.weight", HfEagleMlaHooks.rename("enorm.weight"));
        assertEquals("lm_head.weight", HfEagleMlaHooks.rename("lm_head.weight"));
        assertEquals("model.layers.0.self_attn.q_b_proj.weight",
                HfEagleMlaHooks.rename("layers.3.self_attn.q_b_proj.weight"));
        assertEquals("model.layers.0.self_attn.q_b_proj.weight",
                HfEagleMlaHooks.rename("model.layers.0.self_attn.q_b_proj.weight"));
    }

    @Test
    public void anIncompleteExpertSetIsRefusedByName() throws Exception {
        Checkpoint ckpt = new Checkpoint(tmp.newFolder("partial").toPath());
        ckpt.f32("embed_tokens.weight", new long[]{320, 64}, new float[320 * 64]);
        ckpt.f32("fc.weight", new long[]{64, 128}, new float[64 * 128]);
        ckpt.f32("layers.0.mlp.gate.weight", new long[]{4, 64}, new float[4 * 64]);
        ckpt.f32("layers.0.mlp.experts.0.gate_proj.weight", new long[]{48, 64}, new float[48 * 64]);
        HfTensorIndex index = ckpt.index();
        closing.add(ckpt::close);

        HfLoadContext ctx = context(config(mla(MOE)), entry("EagleDeepSeekMTPModel"), index);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> plan(ctx));
        assertTrue(e.getMessage(), e.getMessage().contains("layers.0.mlp.experts.0.down_proj.weight"));
        assertTrue(e.getMessage(), e.getMessage().contains("is missing"));
    }

    // ------------------------------------------------------------------ the refusals

    @Test
    public void aTiedOutputHeadIsRefused() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfEagle1Hooks.fixup(config(mla(", 'tie_word_embeddings': true"))));
        assertTrue(e.getMessage(), e.getMessage().contains("ties its output head"));
    }

    @Test
    public void aHeadCountThatTheMlaCannotDecompressIsRefused() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfEagle1Hooks.fixup(config(mla(", 'num_key_value_heads': 2"))));
        assertTrue(e.getMessage(), e.getMessage().contains("2 KV heads for 4 attention heads"));
    }

    @Test
    public void anInvalidKvLoraRankIsRefused() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfEagle1Hooks.fixup(config("{'architectures': ['EagleDeepSeekMTPModel'], 'num_hidden_layers': 1,"
                        + " 'num_attention_heads': 4, 'num_key_value_heads': 4, 'qk_nope_head_dim': 16,"
                        + " 'qk_rope_head_dim': 16, 'v_head_dim': 32, 'kv_lora_rank': 0}")));
        assertTrue(e.getMessage(), e.getMessage().contains("positive 'kv_lora_rank'"));
    }

    @Test
    public void aMissingHeadDimensionIsRefusedByName() {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfEagle1Hooks.fixup(config(
                "{'architectures': ['EagleDeepSeekMTPModel'], 'num_hidden_layers': 1,"
                        + " 'num_attention_heads': 4, 'num_key_value_heads': 4, 'qk_nope_head_dim': 16,"
                        + " 'v_head_dim': 32, 'kv_lora_rank': 32}")));
        assertTrue(e.getMessage(), e.getMessage().contains("qk_rope_head_dim"));
    }

    @Test
    public void aQueryPathThatDisagreesWithTheRankIsRefused() throws Exception {
        Checkpoint ckpt = new Checkpoint(tmp.newFolder("plain").toPath());
        ckpt.f32("layers.0.self_attn.q_proj.weight", new long[]{128, 64}, new float[128 * 64]);
        HfTensorIndex index = ckpt.index();
        closing.add(ckpt::close);
        HfLoadContext ctx = context(config(mla(", 'q_lora_rank': 16")), entry("EagleDeepSeekMTPModel"), index);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> plan(ctx));
        assertTrue(e.getMessage(), e.getMessage().contains("q_proj"));
        assertTrue(e.getMessage(), e.getMessage().contains("q_lora_rank"));
    }

    // ------------------------------------------------------------------ fixtures

    private static String mla(String extra) {
        return String.format(MLA, extra);
    }

    private static String llama(String extra) {
        return String.format(LLAMA, extra);
    }

    private HfMetadata meta(String name, String json) throws IOException {
        HfConfig cfg = config(json);
        HfArchEntry e = entry(name);
        HfLoadContext ctx = HfLoadContext.builder(e.configFixup().apply(cfg), e, Arch.EAGLE)
                .index(HfTensorIndex.of(Collections.emptyMap())).targetPath(target()).build();
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        return kv.build();
    }

    /** The target: a four-layer GGUF of hidden size 8 with a tokenizer. */
    private Path target() throws IOException {
        Path p = tmp.getRoot().toPath().resolve("target.gguf");
        if (!p.toFile().exists()) {
            GgufTestWriter w = new GgufTestWriter().architecture("llama");
            w.u32(w.archKey("embedding_length"), 8).u32(w.archKey("block_count"), 4)
                    .u32(w.archKey("vocab_size"), TARGET_VOCAB)
                    .str(GgufKeys.TOKENIZER_MODEL, "gpt2")
                    .strArray(GgufKeys.TOKENIZER_LIST, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
                    .writeTo(p);
        }
        return p;
    }

}
