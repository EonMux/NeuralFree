package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.arch.Arch;
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
import java.util.List;
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
 * {@link HfLimiteHooks} on a synthetic Limite checkpoint: the pinned layout keys, the metadata (head widths, float32
 * RMS epsilon, ggml-convention RoPE base, inclusive window, sigmoid cap, MUDD taps), and the tensor surgery (value
 * embeddings, RoPE pair swap, padded gates, XSA alpha, residual scales, MUDD merge, F32 forcing).
 */
public class HfLimiteHooksTest {

    private static final String BASE = "'pos_mode': 'rope', 'qk_norm': 'rms_pre_rope',"
            + " 'rope_style': 'interleaved_pairs_odd_lane_sign_flip', 'rope_per_layer': false,"
            + " 'rms_norm_has_weight': false, 'rms_norm_eps_mode': 'torch_finfo_default',"
            + " 'mlp_type': 'swiglu', 'attn_gate_applied': 'per_head_before_o_proj',"
            + " 've_head_slice': 'first_num_key_value_heads', 've_applied_before_qk_norm': true,"
            + " 'global_nope': true, 'global_window': -1, 'mudd_r_site': 'resid',"
            + " 'tie_word_embeddings': true, 'final_softcap': 0.0,"
            + " 'softcap_logits': {'kind': 'sigmoid', 'a': 1.5, 'b': 0.25, 'c': 2.0},"
            + " 'hidden_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
            + " 'head_dim': 4, 'rope_n_pairs': 2, 'rope_base_local': 10000.0, 'attention_softmax_scale': 0.5,"
            + " 'sliding_window': 3, 'global_layers': [0], 'vocab_size': 16, 've_dim': 2, 've_gate_scale': 2.0,"
            + " 'attn_gate_scale': 0.5";

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

    private static HfConfig limite(String extra) {
        return config("{'architectures': ['LimiteForCausalLM'], " + BASE + extra + "}");
    }

    @Test
    public void aLayoutValueTheRuntimeDoesNotImplementIsRefusedByName() {
        HfConfig cfg = config("{'architectures': ['LimiteForCausalLM'], "
                + BASE.replace("'mlp_type': 'swiglu'", "'mlp_type': 'gelu'") + "}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfLimiteHooks.fixup(cfg));
        assertTrue(e.getMessage(), e.getMessage().contains("mlp_type"));
    }

    @Test
    public void aSoftcapOfAnotherKindIsRefused() {
        HfConfig cfg = config("{'architectures': ['LimiteForCausalLM'], "
                + BASE.replace("'kind': 'sigmoid'", "'kind': 'tanh'") + "}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfLimiteHooks.fixup(cfg));
        assertTrue(e.getMessage(), e.getMessage().contains("softcap_logits"));
    }

    @Test
    public void theMetadataWritesTheLimiteLayout() throws IOException {
        HfMetadataBuilder b = new HfMetadataBuilder("limite");
        HfLoadContext ctx = context(limite(""), entry("LimiteForCausalLM"), HfTensorIndex.of(Map.of()));
        entry("LimiteForCausalLM").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(4, m.get("limite.attention.key_length"));
        assertEquals(4, m.get("limite.attention.value_length"));
        assertEquals(1.1920929E-7f, m.get("limite.attention.layer_norm_rms_epsilon"));
        assertEquals(0.5f, m.get("limite.attention.scale"));
        assertEquals(0.0f, m.get("limite.attention.xsa_epsilon"));
        assertEquals(4, m.get("limite.rope.dimension_count"));
        assertEquals((float) Math.pow(10000.0, 2.0), m.get("limite.rope.freq_base"));
        assertEquals(4, m.get("limite.attention.sliding_window"));
        assertEquals(Arrays.asList(false, true), m.get("limite.attention.sliding_window_pattern"));
        assertEquals(Arrays.asList(1.5f, 0.25f, 2.0f), m.get("limite.final_logit_sigmoid_capping"));
    }

    @Test
    public void theMuddTapTableIsFlattenedPerLayer() throws IOException {
        String extra = ", 'mudd': true, 'mudd_mlp': true, 'mudd_taps': 2, 'mudd_inter': 4,"
                + " 'mudd_tap_idx': {'1': [0, 1]}";
        HfMetadataBuilder b = new HfMetadataBuilder("limite");
        HfLoadContext ctx = context(limite(extra), entry("LimiteForCausalLM"), HfTensorIndex.of(Map.of()));
        entry("LimiteForCausalLM").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(2, m.get("limite.mudd.tap_count"));
        assertEquals(4, m.get("limite.mudd.feed_forward_length"));
        assertEquals(Arrays.asList(-1, -1, 0, 1), m.get("limite.mudd.tap_indices"));
    }

    @Test
    public void aMuddModelWithoutTheMlpMixersIsRefused() {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfLimiteHooks.fixup(limite(
                ", 'mudd': true, 'mudd_tap_idx': {'1': [0, 1]}, 'mudd_taps': 2")));
        assertTrue(e.getMessage(), e.getMessage().contains("mudd_mlp"));
    }

    @Test
    public void theValueEmbeddingsTakeTheFirstKvHeadsAndPadToTheHeadWidth() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt.f32("model.value_embeds.weight", new long[]{2, 1, 2}, seq(4)).index();
        HfLoadContext ctx = context(limite(""), entry("LimiteForCausalLM"), index);
        Sink out = plan(ctx);
        assertArrayEquals(new long[]{2, 4}, out.get("value_embd.weight").shape());
        assertArrayEquals(new float[]{0f, 2f, 0f, 0f, 4f, 6f, 0f, 0f}, floats(out.get("value_embd.weight")), 0f);
    }

    @Test
    public void theQueryAndKeyRopePairsSwapAndTheScaleFolds() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        float[] q = new float[8 * 8];
        for (int i = 0; i < q.length; i++) {
            q[i] = i;
        }
        float[] k = new float[4 * 8];
        for (int i = 0; i < k.length; i++) {
            k[i] = i;
        }
        HfTensorIndex index = ckpt
                .f32("model.layers.0.self_attn.qkv_scale", new long[]{1}, 2f)
                .f32("model.layers.0.self_attn.q_proj.weight", new long[]{8, 8}, q)
                .f32("model.layers.0.self_attn.k_proj.weight", new long[]{4, 8}, k)
                .f32("model.layers.0.self_attn.v_proj.weight", new long[]{4, 8}, k)
                .index();
        HfLoadContext ctx = context(limite(""), entry("LimiteForCausalLM"), index);
        Sink out = plan(ctx);
        float[] qw = floats(out.get("blk.0.attn_q.weight"));
        // rows of each head swap in pairs, then the qkv scale doubles every value
        assertEquals(16f, qw[0], 0f);      // old row 1, col 0
        assertEquals(0f, qw[8], 0f);       // old row 0, col 0
        assertEquals(48f, qw[16], 0f);     // old row 3, col 0
        assertEquals(32f, qw[24], 0f);     // old row 2, col 0
        assertArrayEquals(new float[]{16f, 18f, 20f, 22f, 24f, 26f, 28f, 30f},
                Arrays.copyOf(floats(out.get("blk.0.attn_k.weight")), 8), 0f);
        assertArrayEquals(new float[]{0f, 2f, 4f, 6f, 8f, 10f, 12f, 14f},
                Arrays.copyOf(floats(out.get("blk.0.attn_v.weight")), 8), 0f);
    }

    @Test
    public void theGatesPadToTheEmbeddingWidthAndTheXsaIsTanh() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.layers.0.self_attn.attn_gate", new long[]{2}, 1f, 2f)
                .f32("model.layers.0.self_attn.ve_gate", new long[]{2, 2}, 3f, 4f, 5f, 6f)
                .f32("model.layers.0.self_attn.xsa_alpha", new long[]{1}, 0f)
                .f32("model.layers.0.resid_lambda_attn", new long[]{1}, 0.5f)
                .f32("model.layers.0.resid_lambda_mlp", new long[]{1}, 0.25f)
                .index();
        HfConfig cfg = limite(", 'xsa': true, 'xsa_normalize_eps': 1e-06, 'xsa_layers': [0]");
        HfLoadContext ctx = context(cfg, entry("LimiteForCausalLM"), index);
        Sink out = plan(ctx);
        assertArrayEquals(new float[]{1f, 2f, 0f, 0f, 0f, 0f, 0f, 0f}, floats(out.get("blk.0.attn_gate.weight")), 0f);
        assertArrayEquals(new long[]{1, 8}, out.get("blk.0.attn_ve_gate.weight").shape());
        assertArrayEquals(new float[]{3f, 4f, 0f, 0f, 0f, 0f, 0f, 0f},
                floats(out.get("blk.0.attn_ve_gate.weight")), 0f);
        assertArrayEquals(new float[]{0f}, floats(out.get("blk.0.attn_xsa_alpha.weight")), 0f);
        assertArrayEquals(new float[]{0.5f}, floats(out.get("blk.0.attn_resid_scale.weight")), 0f);
        assertArrayEquals(new float[]{0.25f}, floats(out.get("blk.0.ffn_resid_scale.weight")), 0f);
    }

    @Test
    public void theMuddMixersMergeTheirSharedSources() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.mudd.dense1", new long[]{2, 3}, seq(6))
                .f32("model.mudd.dense2", new long[]{2, 2, 3}, seq(12))
                .f32("model.mudd.dense2_mlp", new long[]{2, 2, 3}, offset(100, 12))
                .f32("model.mudd.bias", new long[]{2, 2}, 1f, 2f, 3f, 4f)
                .f32("model.mudd.bias_mlp", new long[]{2, 2}, 5f, 6f, 7f, 8f)
                .index();
        String extra = ", 'mudd': true, 'mudd_mlp': true, 'mudd_taps': 2, 'mudd_inter': 4,"
                + " 'mudd_tap_idx': {'1': [0, 1]}";
        HfLoadContext ctx = context(limite(extra), entry("LimiteForCausalLM"), index);
        Sink out = plan(ctx);
        assertArrayEquals(new float[]{6f, 7f, 8f, 9f, 10f, 11f, 106f, 107f, 108f, 109f, 110f, 111f},
                floats(out.get("blk.1.mudd_up.weight")), 0f);
        assertArrayEquals(new float[]{3f, 4f, 7f, 8f}, floats(out.get("blk.1.mudd_up.bias")), 0f);
        assertArrayEquals(new long[]{2, 3}, out.get("blk.1.mudd_down.weight").shape());
        assertNull(out.added.get("blk.0.mudd_down.weight"));
    }

    @Test
    public void theMixingWeightsAreForcedToF32() {
        assertEquals(com.eonmux.neuralfree.gguf.GGMLType.F32, HfLimiteHooks.forceType(
                "blk.3.attn_gate.weight", 1, null, OutType.F16));
        assertEquals(com.eonmux.neuralfree.gguf.GGMLType.F32, HfLimiteHooks.forceType(
                "blk.3.mudd_up.bias", 1, null, OutType.F16));
        assertNull(HfLimiteHooks.forceType("blk.3.attn_q.weight", 2, null, OutType.F16));
    }

    @Test
    public void theEntryResolvesToLimite() {
        assertEquals(Arch.LIMITE, entry("LimiteForCausalLM").arch().resolve(limite(""),
                HfTensorIndex.of(Map.of())));
        List<String> names = entry("LimiteForCausalLM").hfNames();
        assertTrue(names.contains("LimiteForCausalLM"));
    }
}
