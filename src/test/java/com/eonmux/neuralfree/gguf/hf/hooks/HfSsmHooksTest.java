package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfSsmHooks} on synthetic Mamba, Mamba-2, Falcon-H1, Jamba and PLaMo-2 checkpoints (hook level, the entry
 * hooks of {@code HfArchTableHybridSsm}): the builtin GPT-NeoX fallback at the padded size and the
 * {@code tokenizer.model.v3} refusal, the Mamba-1 metadata defaults with the falconmamba flag, the Mamba-2 conv
 * squeeze, A/D reshape, grouped norm reshape and {@code dt_bias} rename, the tied output drop, the Falcon-H1
 * multiplier folding with the {@code in_proj} row segments, the Jamba attention vector and expert merge, and the
 * PLaMo-2 norm offsets.
 */
public class HfSsmHooksTest {

    private static final String MAMBA_BASE =
            "'hidden_size': 8, 'num_hidden_layers': 2, 'vocab_size': 10";
    private static final String MAMBA2_BASE =
            "'hidden_size': 8, 'num_hidden_layers': 2, 'vocab_size': 10";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private static HfArchEntry e(String hfName) {
        return entry(hfName);
    }

    private static HfConfig c(String json) {
        return HfLlamaHooksTest.config(json);
    }

    private static HfMetadata meta(String json) throws IOException {
        HfConfig cfg = c(json);
        HfLoadContext ctx = context(cfg, e("MambaForCausalLM"), HfTensorIndex.of(Map.of()));
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        e("MambaForCausalLM").meta().apply(ctx, kv, null);
        return kv.build();
    }

    private Map<String, Object> vocab(String json, String hfName, Path dir) throws Exception {
        HfArchEntry entry = e(hfName);
        HfLoadContext ctx = HfLoadContext.builder(context(c(json), entry, HfTensorIndex.of(Map.of())).config(),
                entry, context(c(json), entry, HfTensorIndex.of(Map.of())).arch())
                .tokenizerFiles(HfTokenizerFiles.locate(dir)).build();
        return entry.vocab().build(ctx, new HfPreTokenizerResolver());
    }

    // ------------------------------------------------------------------ vocabularies

    @Test
    public void builtinVocabularyFallsThroughAtThePaddedSizeWhenNoTokenizerFilesExist() throws Exception {
        // Mamba-1: pad 8, Mamba-2: pad 16 (mamba.py:29-36, :128-141)
        Map<String, Object> one = vocab("{'architectures': ['MambaForCausalLM'], " + MAMBA_BASE + "}",
                "MambaForCausalLM", tmp.getRoot().toPath());
        assertEquals("gpt-neox", one.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("gpt_neox", one.get(GgufKeys.TOKENIZER_PRE));
        assertEquals(16, size(one.get(GgufKeys.TOKENIZER_LIST)));

        Map<String, Object> two = vocab("{'architectures': ['Mamba2ForCausalLM'], " + MAMBA2_BASE + "}",
                "Mamba2ForCausalLM", tmp.getRoot().toPath());
        assertEquals("gpt-neox", two.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(16, size(two.get(GgufKeys.TOKENIZER_LIST)));

        Map<String, Object> padded = vocab("{'architectures': ['MambaForCausalLM'], " + MAMBA_BASE
                        + ", 'pad_vocab_size_multiple': 8}",
                "MambaForCausalLM", tmp.getRoot().toPath());
        assertEquals(16, size(padded.get(GgufKeys.TOKENIZER_LIST)));
    }

    private static int size(Object list) {
        return list instanceof List ? ((List<?>) list).size() : -1;
    }

    @Test
    public void v3TokenizerModelRefusesOnlyForTheMamba2Subclasses() throws Exception {
        Path dir = tmp.newFolder("v3").toPath();
        Files.write(dir.resolve("tokenizer.model.v3"), "v3".getBytes(StandardCharsets.UTF_8));
        assertThrows(GgufFormatException.class,
                () -> vocab("{'architectures': ['Mamba2ForCausalLM'], " + MAMBA2_BASE + "}",
                        "Mamba2ForCausalLM", dir));
        // Mamba-1 has no .v3 rule: the GPT-2 JSON, the SentencePiece model and the builtin follow
        Map<String, Object> one = vocab("{'architectures': ['MambaForCausalLM'], " + MAMBA_BASE + "}",
                "MambaForCausalLM", dir);
        assertEquals("gpt-neox", one.get(GgufKeys.TOKENIZER_MODEL));
    }

    // ------------------------------------------------------------------ metadata and tensors

    @Test
    public void mambaMetaWritesTheDefaultSizesAndTheFalconMambaFlag() throws IOException {
        HfMetadata m = meta("{'architectures': ['MambaForCausalLM'], 'model_type': 'falcon_mamba', "
                + MAMBA_BASE + "}");
        assertEquals(1L << 20, m.get("mamba.context_length"));
        assertEquals(8, m.get("mamba.embedding_length"));
        assertEquals(16, m.get("mamba.ssm.state_size"));
        assertEquals(4, m.get("mamba.ssm.conv_kernel"));
        assertEquals(1, m.get("mamba.ssm.time_step_rank"));
        assertEquals(0, m.get("mamba.attention.head_count"));
        assertEquals(0, m.get("mamba.feed_forward_length"));
        assertEquals(2, m.get("mamba.block_count"));
        assertEquals(true, m.get("mamba.ssm.dt_b_c_rms"));
        HfMetadata plain = meta("{'architectures': ['MambaForCausalLM'], " + MAMBA_BASE + "}");
        assertEquals(false, plain.get("mamba.ssm.dt_b_c_rms"));
    }

    @Test
    public void mamba2TensorSqueezesReshapesNegExpsAndRenamesTheDtBias() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("backbone.layers.0.mixer.conv1d.weight", new long[]{4, 1, 3}, seq(12))
                .f32("backbone.layers.0.mixer.A_log", new long[]{2}, new float[]{0f, 1f})
                .f32("backbone.layers.0.mixer.D", new long[]{2}, new float[]{1f, 2f})
                .f32("backbone.layers.1.mixer.norm.weight", new long[]{8}, seq(8))
                .f32("backbone.layers.1.mixer.dt_proj.bias", new long[]{2}, seq(2))
                .f32("backbone.embeddings.weight", new long[]{16, 8}, seq(128)).index();
        HfConfig cfg = c("{'architectures': ['Mamba2ForCausalLM'], " + MAMBA2_BASE + ", 'n_groups': 4}");
        Sink out = plan(context(cfg, e("Mamba2ForCausalLM"), index));
        assertArrayEquals(new long[]{4, 3}, out.get("blk.0.ssm_conv1d.weight").shape());
        assertArrayEquals(new float[]{-1f, -1f * (float) Math.E}, floats(out.get("blk.0.ssm_a.weight")), 1e-4f);
        assertArrayEquals(new long[]{2, 1}, out.get("blk.0.ssm_a.weight").shape());
        assertArrayEquals(new long[]{2, 1}, out.get("blk.0.ssm_d.weight").shape());
        assertArrayEquals(new long[]{4, 2}, out.get("blk.1.ssm_norm.weight").shape());
        assertTrue(out.added.containsKey("blk.1.dt_proj.bias"));
        assertTrue(out.added.containsKey("token_embd.weight"));
    }

    @Test
    public void mambaTensorDropsTheTiedOutput() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        float[] emb = seq(64);
        HfTensorIndex index = ckpt
                .f32("backbone.embeddings.weight", new long[]{8, 8}, emb)
                .f32("backbone.layers.0.mixer.A_log", new long[]{2}, new float[]{0f, 0f})
                .f32("backbone.layers.1.mixer.A_log", new long[]{2}, new float[]{0f, 0f})
                .f32("lm_head.weight", new long[]{8, 8}, emb).index();
        HfConfig cfg = c("{'architectures': ['MambaForCausalLM'], " + MAMBA_BASE + "}");
        Sink out = plan(context(cfg, e("MambaForCausalLM"), index));
        assertTrue(out.dropped.containsKey("lm_head.weight"));
        assertFalse(out.added.containsKey("output.weight"));
    }

    @Test
    public void falconH1FoldsTheMultipliersAndTheInProjRowSegments() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("backbone.layers.0.mixer.in_proj.weight", new long[]{12, 2}, new float[24])
                .f32("backbone.layers.0.mixer.out_proj.weight", new long[]{4, 4}, new float[16])
                .f32("backbone.layers.0.mixer.down_proj.weight", new long[]{3, 4}, new float[12])
                .f32("backbone.layers.0.mixer.k_proj.weight", new long[]{2, 4}, new float[8])
                .f32("backbone.layers.0.mixer.norm.weight", new long[]{4}, new float[4])
                .f32("model.embed_tokens.weight", new long[]{16, 4}, new float[64])
                .f32("lm_head.weight", new long[]{4, 16}, new float[64]).index();
        HfConfig cfg = c("{'architectures': ['FalconH1ForCausalLM'], 'hidden_size': 4, 'num_hidden_layers': 2,"
                + " 'vocab_size': 16, 'mamba_d_ssm': 4, 'n_groups': 1, 'mamba_d_state': 2, 'd_head': 4,"
                + " 'mamba_d_head': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'intermediate_size': 3,"
                + " 'max_position_embeddings': 32, 'rope_parameters': {'rope_theta': 10000.0},"
                + " 'attention_in_multiplier': 2.0, 'ssm_in_multiplier': 3.0, 'ssm_out_multiplier': 5.0,"
                + " 'embedding_multiplier': 7.0, 'lm_head_multiplier': 11.0,"
                + " 'mlp_multipliers': [13.0, 17.0], 'ssm_multipliers': [1, 2, 3, 4, 5]}");
        Sink out = plan(context(cfg, e("FalconH1ForCausalLM"), index));
        // rows: [0,4) x1, [4,8) x2, [8,10) x3 (n_groups * d_state), [10,12) x4, no tail
        float[] want = new float[24];
        for (int r = 0; r < 12; r++) {
            want[2 * r] = want[2 * r + 1] = row(multiplier[r]);
        }
        float[] got = floats(out.get("blk.0.ssm_in.weight"));
        for (int r = 0; r < 12; r++) {
            assertEquals("row " + r, want[2 * r], got[2 * r], 1e-5f);
        }
        assertEquals(17f, floats(out.get("blk.0.ffn_down.weight"))[0], 1e-5f);
        assertEquals(2f, floats(out.get("blk.0.attn_k.weight"))[0], 1e-5f);
        assertEquals(5f, floats(out.get("blk.0.ssm_out.weight"))[0], 1e-5f);
        assertEquals(7f, floats(out.get("token_embd.weight"))[0], 1e-5f);
        assertEquals(11f, floats(out.get("output.weight"))[0], 1e-5f);
        assertArrayEquals(new long[]{1, 4}, out.get("blk.0.ssm_norm.weight").shape());
    }

    private static float[] multiplier = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4};

    private static float row(float v) {
        return v;
    }

    @Test
    public void jambaWritesTheAttentionVectorAndMergesTheExperts() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.layers.0.moe.experts.0.down_proj.weight", new long[]{4, 4}, seq(16))
                .f32("model.layers.0.moe.experts.0.gate_proj.weight", new long[]{4, 4}, seq(16))
                .f32("model.layers.0.moe.experts.0.up_proj.weight", new long[]{4, 4}, seq(16))
                .f32("model.layers.0.moe.experts.1.down_proj.weight", new long[]{4, 4}, seq(16))
                .f32("model.layers.0.moe.experts.1.gate_proj.weight", new long[]{4, 4}, seq(16))
                .f32("model.layers.0.moe.experts.1.up_proj.weight", new long[]{4, 4}, seq(16)).index();
        String json = "{'architectures': ['JambaForCausalLM'], 'hidden_size': 4, 'mamba_expand': 2,"
                + " 'num_hidden_layers': 1, 'max_position_embeddings': 32, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'intermediate_size': 5, 'attn_layer_offset': 0,"
                + " 'attn_layer_period': 1, 'expert_layer_offset': 0, 'expert_layer_period': 1,"
                + " 'num_local_experts': 2, 'num_experts_per_tok': 1}";
        Sink out = plan(context(c(json), e("JambaForCausalLM"), index));
        assertArrayEquals(new long[]{2, 4, 4}, out.get("blk.0.ffn_down_exps.weight").shape());
        assertArrayEquals(new long[]{2, 4, 4}, out.get("blk.0.ffn_up_exps.weight").shape());
        // the merged down holds [expert0; expert1]
        assertArrayEquals(seq(16),
                java.util.Arrays.copyOfRange(floats(out.get("blk.0.ffn_down_exps.weight")), 0, 16), 0f);
        assertArrayEquals(seq(16),
                java.util.Arrays.copyOfRange(floats(out.get("blk.0.ffn_down_exps.weight")), 16, 32), 0f);
        HfMetadataBuilder kv = new HfMetadataBuilder("jamba");
        e("JambaForCausalLM").meta().apply(context(c(json), e("JambaForCausalLM"), index), kv, null);
        assertEquals(Arrays.asList(1), kv.build().get("jamba.attention.head_count_kv"));
    }

    @Test
    public void plamo2AppliesTheNormOffsets() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.layers.layers.0.pre_mixer_norm.weight", new long[]{4}, new float[]{1, 2, 3, 4})
                .f32("model.layers.layers.0.post_mixer_norm.weight", new long[]{4}, new float[]{1, 2, 3, 4})
                .f32("model.layers.layers.0.post_mlp_norm.weight", new long[]{4}, new float[]{1, 2, 3, 4})
                .f32("model.layers.layers.0.mixer.conv1d.weight", new long[]{4, 1, 3}, seq(12))
                .f32("model.layers.layers.0.mixer.dt_bias", new long[]{2}, seq(2)).index();
        HfConfig cfg = c("{'architectures': ['Plamo2ForCausalLM'], 'num_hidden_layers': 1, 'vocab_size': 8}");
        Sink out = plan(context(cfg, e("Plamo2ForCausalLM"), index));
        assertArrayEquals(new float[]{2, 3, 4, 5}, floats(out.get("blk.0.attn_norm.weight")), 1e-5f);
        assertArrayEquals(new float[]{1.2f, 2.2f, 3.2f, 4.2f},
                floats(out.get("blk.0.post_attention_norm.weight")), 1e-5f);
        float mlpTail = 1f + (float) (1.0 / Math.pow(5.0, 1.5));
        assertEquals(mlpTail, floats(out.get("blk.0.post_ffn_norm.weight"))[0], 1e-5f);
        assertArrayEquals(new float[]{5, 6, 7, 8}, floats(out.get("blk.0.ssm_conv1d.weight")), 1e-5f);
        assertTrue(out.added.containsKey("blk.0.dt_proj.bias"));
    }
}
