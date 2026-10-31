package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfSsmHybridHooksB} on synthetic LFM2-family checkpoints: the per-layer key/value vector fixing (the conv
 * layers and the MoE full-attention layers), the auto-adjusted feed-forward width, the audio filter, the D1 decision
 * type, the D1Omni trunk+head fold with its token-type count, and the MoE expert stack.
 */
public class HfSsmHybridHooksBTest {

    private static final String LFM2_BASE = "'hidden_size': 8, 'num_hidden_layers': 3, 'vocab_size': 10,"
            + " 'conv_L_cache': 4, 'norm_eps': 1e-05, 'block_ff_dim': 6, 'block_multiple_of': 2,"
            + " 'num_attention_heads': 2, 'num_key_value_heads': 1, 'max_position_embeddings': 16,"
            + " 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private static HfConfig cfg(String json) {
        return HfLlamaHooksTest.config(json);
    }

    @Test
    public void lfm2FixesThePerLayerKeyValueVectorAndAutoAdjustsTheFfn() throws IOException {
        String json = "{'architectures': ['LFM2ForCausalLM'], " + LFM2_BASE
                + ", 'layer_types': ['conv', 'full_attention', 'shortconv'], 'block_auto_adjust_ff_dim': true}";
        HfConfig fixed = HfSsmHybridHooksB.lfm2Fixup(cfg(json));
        List<Integer> kv = fixed.findIntList("num_key_value_heads").orElse(null);
        assertEquals(Arrays.asList(0, 1, 1), kv);
        HfMetadataBuilder b = new HfMetadataBuilder("lfm2");
        HfLoadContext ctx = context(fixed, entry("LFM2ForCausalLM"), HfTensorIndex.of(Map.of()));
        entry("LFM2ForCausalLM").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(Arrays.asList(0, 1, 1), m.get("lfm2.attention.head_count_kv"));
        // (2 * 6 / 3) rounded to a multiple of 2 with no multiplier
        assertEquals(4, m.get("lfm2.feed_forward_length"));
        assertEquals(4, m.get("lfm2.shortconv.l_cache"));
        assertEquals(1e-05f, m.get("lfm2.attention.layer_norm_rms_epsilon"));
    }

    @Test
    public void theMoeFixupZeroesEveryLayerButFullAttentionAndStacksTheExperts() throws IOException {
        String json = "{'architectures': ['Lfm2MoeForCausalLM'], " + LFM2_BASE
                + ", 'layer_types': ['full_attention', 'shortconv', 'conv'], 'block_auto_adjust_ff_dim': false,"
                + " 'moe_intermediate_size': 5, 'num_dense_layers': 1, 'score_func': 'sigmoid',"
                + " 'num_local_experts': 2, 'num_experts_per_tok': 1}";
        HfConfig fixed = HfSsmHybridHooksB.lfm2MoeFixup(cfg(json));
        HfMetadataBuilder b = new HfMetadataBuilder("lfm2moe");
        HfLoadContext ctx = context(fixed, entry("Lfm2MoeForCausalLM"), HfTensorIndex.of(Map.of()));
        entry("Lfm2MoeForCausalLM").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(Arrays.asList(1, 0, 0), m.get("lfm2moe.attention.head_count_kv"));
        assertEquals(1, m.get("lfm2moe.leading_dense_block_count"));
        assertEquals(2, m.get("lfm2moe.expert_gating_func"));
        assertEquals(5, m.get("lfm2moe.expert_feed_forward_length"));

        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.layers.0.feed_forward.experts.0.w1.weight", new long[]{3, 4}, seq(12))
                .f32("model.layers.0.feed_forward.experts.0.w2.weight", new long[]{4, 3}, seq(12))
                .f32("model.layers.0.feed_forward.experts.0.w3.weight", new long[]{3, 4}, seq(12))
                .f32("model.layers.0.feed_forward.experts.1.w1.weight", new long[]{3, 4}, seq(12))
                .f32("model.layers.0.feed_forward.experts.1.w2.weight", new long[]{4, 3}, seq(12))
                .f32("model.layers.0.feed_forward.experts.1.w3.weight", new long[]{3, 4}, seq(12)).index();
        Sink out = plan(context(fixed, entry("Lfm2MoeForCausalLM"), index));
        assertArrayEquals(new long[]{2, 3, 4}, out.get("blk.0.ffn_gate_exps.weight").shape());
        assertNull(out.added.get("blk.1.ffn_gate_exps.weight"));
    }

    @Test
    public void theD1OmniHeadFoldsIntoTheTrunkArrays() throws IOException {
        String json = "{'architectures': ['D1OmniModel'], " + LFM2_BASE
                + ", 'layer_types': ['full_attention', 'shortconv', 'conv'], 'block_auto_adjust_ff_dim': true,"
                + " 'intermediate_size': 6, 'block_ffn_dim_multiplier': 1.0, 'hidden_size': 8, 'head_layers': 1,"
                + " 'head_dim': 4, 'max_length': 16384, 'image_text_length': 896, 'audio_text_length': 15360,"
                + " 'temperatures': {'choice:3-5': 0.5, 'choice:11+': 0.25}}";
        HfConfig fixed = HfSsmHybridHooksB.d1OmniFixup(cfg(json));
        assertEquals(4, fixed.blockCount());
        List<Integer> kvs = fixed.findIntList("num_key_value_heads").orElse(null);
        assertEquals(Arrays.asList(1, 1, 0, 2), kvs);
        HfMetadataBuilder b = new HfMetadataBuilder("lfm2");
        HfLoadContext ctx = context(fixed, entry("D1OmniModel"), HfTensorIndex.of(Map.of()));
        entry("D1OmniModel").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(4, m.get("lfm2.block_count"));
        assertEquals(4, m.get("lfm2.attention.key_length"));
        assertEquals("lfm2-d1-omni", m.get("lfm2.decision.type"));
        assertEquals(1, m.get("lfm2.decision.block_count"));
        assertEquals(0.5f, m.get("lfm2.decision.temperature.choice.3_5"));
        // the ADD_BOS=false write belongs to set_vocab (lfm2.py:199-200), not to set_gguf_parameters
        assertNull(tail(m));
        // the head blocks fold: head.layers. becomes blk.{trunk+bid}
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("head.layers.0.self_attn.in_proj.weight", new long[]{4, 4}, seq(16)).index();
        Sink out = plan(context(fixed, entry("D1OmniModel"), index));
        assertTrue(out.added.containsKey("blk.3.attn_qkv.weight"));
    }

    private static Object tail(HfMetadata m) {
        return m.get("lfm2.tokenizer.ggml.add_bos_token");
    }

    @Test
    public void theFiltersSkipTheAudioAndVocabFreeAudioTokens() throws Exception {
        // audio tensors skip; lfm. renames to model.; head/head renames per the omni filter
        java.util.function.UnaryOperator<String> base = n -> n;
        assertNull(HfSsmHybridHooksB.LFM2_FILTER.filter("lfm.codebook.0.weight", base));
        assertEquals("model.blocks.0.conv.weight",
                HfSsmHybridHooksB.LFM2_FILTER.filter("lfm.blocks.0.conv.weight", base));
        assertEquals("head.layers.3.in_proj.q.weight",
                HfSsmHybridHooksB.D1_OMNI_FILTER.filter("head.head.layers.3.in_proj_q.weight", base));
        Path dir = tmp.newFolder("audio").toPath();
        HfConfig cfg = cfg("{'architectures': ['Lfm25AudioTokenizer'], " + LFM2_BASE
                + ", 'layer_types': ['shortconv'], 'block_auto_adjust_ff_dim': false,'sliding_window': 4,"
                + " 'output_size': 3}");
        HfArchEntry e = entry("Lfm25AudioTokenizer");
        HfLoadContext probe = context(cfg, e, HfTensorIndex.of(Map.of()));
        HfLoadContext ctx = HfLoadContext.builder(probe.config(), e, probe.arch())
                .tokenizerFiles(com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles.locate(dir)).build();
        Map<String, Object> md = e.vocab().build(ctx, new HfPreTokenizerResolver());
        assertEquals("none", md.get(GgufKeys.TOKENIZER_MODEL));
    }
}
