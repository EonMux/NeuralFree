package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the LFM2 converter family (LC/conversion/lfm2.py), the LFM2 rows of the Appendix A
 * HK-H1 table: {@code LFM2Model} ({@code LFM2ForCausalLM} and its alias {@code Lfm2ForCausalLM}), the decision
 * {@code D1Model}, the decision {@code D1OmniModel}, the MoE {@code LFM2MoeModel}
 * ({@code Lfm2MoeForCausalLM}) and the audio {@code LFM25AudioTokenizer}. Three layers, hidden size 8, and the exact
 * tensor names and configuration keys of the converters and the generated pattern tables: full-attention layers use
 * the plain {@code model.layers.{bid}.self_attn.*} / {@code .mlp.*} (dense) and {@code .feed_forward.*} (MoE) names,
 * the short-convolution layers use {@code model.layers.{bid}.conv.*}, and the D1Omni head block folds into the trunk
 * as {@code head.layers.{bid}.linear1/linear2/self_attn.in_proj}.
 */
public final class HfSsmFixturesHybridB {

    private static final int LAYERS = 3;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int FFN = 16;
    private static final int CONV = 4;
    private static final int EXPERTS = 2;
    private static final int EXPERT_FFN = 8;
    private static final int VOCAB = 16;

    /** The configuration keys of the LFM2 trunk, after the architectures list. */
    private static final String LFM2_BASE = "'hidden_size': " + DIM
            + ", 'num_hidden_layers': " + LAYERS + ", 'vocab_size': " + VOCAB
            + ", 'num_attention_heads': 2, 'num_key_value_heads': 1, 'max_position_embeddings': 16,"
            + " 'layer_types': ['full_attention', 'shortconv', 'conv'], 'block_ff_dim': " + FFN
            + ", 'block_auto_adjust_ff_dim': false, 'block_multiple_of': 2, 'conv_L_cache': " + CONV
            + ", 'norm_eps': 1e-05, 'rms_norm_eps': 1e-05, 'rope_parameters': {'rope_theta': 10000.0}";

    private HfSsmFixturesHybridB() {
    }

    /**
     * {@code LFM2ForCausalLM}: a full-attention layer, a short-convolution layer and a dense FFN over the plain
     * {@code model.} form (lfm2.py:20-51).
     */
    public static SyntheticHf.Builder lfm2() {
        return lfm2Trunk(builder("{'architectures': ['LFM2ForCausalLM'], " + LFM2_BASE + "}"));
    }

    /**
     * {@code Lfm2ForCausalLM}: the second registered name of the same {@code LFM2Model} checkpoint form.
     */
    public static SyntheticHf.Builder lfm2Alias() {
        return lfm2Trunk(builder("{'architectures': ['Lfm2ForCausalLM'], " + LFM2_BASE + "}"));
    }

    /**
     * {@code Lfm2MoeForCausalLM}: the LFM2 trunk with the routed expert feed-forward on the full-attention layer
     * (the {@code feed_forward.experts.w1/w2/w3} parts the LFM2-MoE path stacks) (lfm2.py:279-345).
     */
    public static SyntheticHf.Builder lfm2Moe() {
        // the lfm2moe fixup keeps the KV heads on the full_attention layer only, so the trunk
        // differs from lfm2(): block 0 attention+FFN, blocks 1-2 shortconv+MoE; the tied
        // output (the lfm2moe set carries no OUTPUT tensor)
        SyntheticHf.Builder b = builder("{'architectures': ['Lfm2MoeForCausalLM'], " + LFM2_BASE
                + ", 'intermediate_size': " + FFN
                + ", 'num_experts_per_tok': 1, 'num_local_experts': " + EXPERTS
                + ", 'moe_intermediate_size': " + EXPERT_FFN + ", 'num_dense_layers': 1}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        b.bf16("model.embedding_norm.weight", DIM);
        // block 0: attention + FFN
        String a = "model.layers.0.";
        b.bf16(a + "input_layernorm.weight", DIM);
        b.bf16(a + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.k_proj.weight", KV, DIM);
        b.bf16(a + "self_attn.v_proj.weight", KV, DIM);
        b.bf16(a + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.q_norm.weight", KV);
        b.bf16(a + "self_attn.k_norm.weight", KV);
        b.bf16(a + "post_attention_layernorm.weight", DIM);
        b.bf16(a + "mlp.gate_proj.weight", FFN, DIM);
        b.bf16(a + "mlp.up_proj.weight", FFN, DIM);
        b.bf16(a + "mlp.down_proj.weight", DIM, FFN);
        // blocks 1-2: shortconv + MoE (after the dense lead)
        for (String p : new String[] {"model.layers.1.", "model.layers.2."}) {
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "conv.conv.weight", DIM, 1, CONV);
            b.bf16(p + "conv.in_proj.weight", DIM, DIM);
            b.bf16(p + "conv.out_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "feed_forward.gate.weight", EXPERTS, DIM);
            b.bf16(p + "feed_forward.expert_bias.bias", EXPERTS);
            for (int e = 0; e < EXPERTS; e++) {
                b.bf16(p + "feed_forward.experts." + e + ".w1.weight", EXPERT_FFN, DIM);
                b.bf16(p + "feed_forward.experts." + e + ".w2.weight", DIM, EXPERT_FFN);
                b.bf16(p + "feed_forward.experts." + e + ".w3.weight", EXPERT_FFN, DIM);
            }
        }
        return b;
    }

    /**
     * {@code D1Model}: the LFM2 trunk with the {@code lfm2-d1} decision type — the same {@code model.} tensor layout
     * as {@link #lfm2()} (lfm2.py:89-131).
     */
    public static SyntheticHf.Builder d1() {
        return lfm2Trunk(builder("{'architectures': ['D1Model'], " + LFM2_BASE + "}"));
    }

    /**
     * {@code D1OmniModel}: the LFM2 trunk plus one decision head block that folds into the block sequence as
     * {@code head.layers.{bid}.linear1/linear2/self_attn.in_proj} (lfm2.py:188-276).
     */
    public static SyntheticHf.Builder d1Omni() {
        // the decision head's centred short convolution requires conv_L_cache 3 (lfm2.cpp:369)
        SyntheticHf.Builder b = lfm2Trunk(builder("{'architectures': ['D1OmniModel'], "
                + LFM2_BASE.replace("'conv_L_cache': " + CONV, "'conv_L_cache': 3")
                + ", 'intermediate_size': " + FFN
                + ", 'head_layers': 1, 'block_ffn_dim_multiplier': 1.0, 'max_length': 16384"
                + ", 'image_text_length': 896, 'audio_text_length': 15360"
                + ", 'temperatures': {'choice:3-5': 0.5}}"));
        // the decision head block folds to blk.3: the LAYA-form block (the head.layers rows)
        // with the claimed biases, plus the type embedding and the scorer head
        b.bf16("head.layers.0.norm1.weight", DIM);
        b.bf16("head.layers.0.norm1.bias", DIM);
        b.bf16("head.layers.0.self_attn.in_proj.weight", 4 * DIM, DIM);
        b.bf16("head.layers.0.self_attn.in_proj.bias", 4 * DIM);
        b.bf16("head.layers.0.self_attn.out_proj.weight", DIM, DIM);
        b.bf16("head.layers.0.self_attn.out_proj.bias", DIM);
        b.bf16("head.layers.0.norm2.weight", DIM);
        b.bf16("head.layers.0.norm2.bias", DIM);
        b.bf16("head.layers.0.linear1.weight", 4 * DIM, DIM);
        b.bf16("head.layers.0.linear1.bias", 4 * DIM);
        b.bf16("head.layers.0.linear2.weight", DIM, 4 * DIM);
        b.bf16("head.layers.0.linear2.bias", DIM, 4 * DIM);
        b.bf16("head.type_emb.weight", 3, DIM);
        b.bf16("head.scorer.0.weight", DIM);
        b.bf16("head.scorer.0.bias", DIM);
        b.bf16("head.scorer.1.weight", DIM, DIM);
        b.bf16("head.scorer.1.bias", DIM);
        b.bf16("head.scorer.3.weight", DIM, DIM);
        b.bf16("head.scorer.3.bias", DIM);
        return b;
    }

    /**
     * {@code Lfm25AudioTokenizer}: the LFM2 trunk of the audio build over the plain {@code model.} form
     * (lfm2.py:495-519), every layer a short-convolution block and the {@code sliding_window}/{@code output_size}
     * keys.
     */
    public static SyntheticHf.Builder audioTokenizer() {
        SyntheticHf.Builder b = builder("{'architectures': ['Lfm25AudioTokenizer'], " + LFM2_BASE
                + ", 'sliding_window': " + CONV + ", 'output_size': " + DIM
                + ", 'layer_types': ['shortconv', 'shortconv', 'shortconv']}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        b.bf16("model.embedding_norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
        // the audio build runs through the LFM2 claims: the attention projections with the
        // per-layer q/k norms, the operator norms and the FFN on every layer
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.up_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, FFN);
        }
        return b;
    }

    /**
     * The tensors of the LFM2 trunk: two attention layers and one convolution layer (the config's
     * {@code layer_types}; the LFM2 claims carry the two operator norms per layer, the per-layer q/k norms and
     * the attention projections on the attention layers, the short-convolution weights on the recurrent ones).
     */
    private static SyntheticHf.Builder lfm2Trunk(SyntheticHf.Builder b) {
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        b.bf16("model.embedding_norm.weight", DIM);
        // the LFM2 set carries no OUTPUT_NORM (the per-block operator norms only)
        b.bf16("lm_head.weight", VOCAB, DIM);
        // the lfm2moe fixup keeps the KV heads on the full_attention layer only: the conv
        // layers are the recurrent ones (arch_tensors/lfm2moe claims)
        lfm2Attention(b, "model.layers.0.");
        lfm2Attention(b, "model.layers.1.");
        lfm2Conv(b, "model.layers.2.");
        return b;
    }
    /** An LFM2 attention layer: the {@code self_attn.*} projections with the per-layer q/k norms and the FFN. */
    private static void lfm2Attention(SyntheticHf.Builder b, String p) {
        b.bf16(p + "input_layernorm.weight", DIM);
        b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(p + "self_attn.q_norm.weight", KV);
        b.bf16(p + "self_attn.k_norm.weight", KV);
        b.bf16(p + "post_attention_layernorm.weight", DIM);
        denseFfn(b, p);
    }

    /** An LFM2 short-convolution layer: the {@code conv.*} weights and the FFN. */
    private static void lfm2Conv(SyntheticHf.Builder b, String p) {
        b.bf16(p + "input_layernorm.weight", DIM);
        b.bf16(p + "conv.conv.weight", DIM, 1, CONV);
        b.bf16(p + "conv.in_proj.weight", DIM, DIM);
        b.bf16(p + "conv.out_proj.weight", DIM, DIM);
        b.bf16(p + "post_attention_layernorm.weight", DIM);
        denseFfn(b, p);
    }

    /** The dense FFN of an LFM2 layer: the {@code mlp.gate_proj}/{@code up_proj}/{@code down_proj} weights. */
    private static void denseFfn(SyntheticHf.Builder b, String p) {
        b.bf16(p + "mlp.gate_proj.weight", FFN, DIM);
        b.bf16(p + "mlp.up_proj.weight", FFN, DIM);
        b.bf16(p + "mlp.down_proj.weight", DIM, FFN);
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }

    /**
     * One end-to-end fixture: its leading HF architecture name and the checkpoint builder.
     */
    public static final class Named {
        /** The HF architecture of the fixture. */
        public final String hfArch;
        /** The builder of the checkpoint. */
        public final Supplier<SyntheticHf.Builder> fixture;

        Named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
            this.hfArch = hfArch;
            this.fixture = fixture;
        }

        @Override
        public String toString() {
            return hfArch;
        }
    }

    private static Named named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
        return new Named(hfArch, fixture);
    }

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(
                named("LFM2ForCausalLM", HfSsmFixturesHybridB::lfm2),
                named("Lfm2ForCausalLM", HfSsmFixturesHybridB::lfm2Alias),
                named("Lfm2MoeForCausalLM", HfSsmFixturesHybridB::lfm2Moe),
                named("D1Model", HfSsmFixturesHybridB::d1),
                named("D1OmniModel", HfSsmFixturesHybridB::d1Omni),
                named("Lfm25AudioTokenizer", HfSsmFixturesHybridB::audioTokenizer));
    }
}
