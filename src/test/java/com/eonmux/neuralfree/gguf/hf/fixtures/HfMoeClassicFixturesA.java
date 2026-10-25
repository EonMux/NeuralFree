package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-MC part A converter classes: {@code Llama4Model}, {@code DbrxModel},
 * {@code GrokModel} and {@code ArcticModel}. Two layers, a hidden size of 8, two query heads, one key/value head, an
 * MoE of two experts and the exact tensor names and configuration keys of the converter sources.
 */
public final class HfMoeClassicFixturesA {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeClassicFixturesA() {
    }

    /** {@code Llama4ForCausalLM}: the MoE width is {@code intermediate_size}, the dense one {@code intermediate_size_mlp}. */
    public static SyntheticHf.Builder llama4() {
        SyntheticHf.Builder b = builder("{'architectures': ['Llama4ForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 4, 'intermediate_size_mlp': 8, 'interleave_moe_layer_step': 1,"
                + " 'num_local_experts': 2, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'vocab_size': 8, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'tie_word_embeddings': false,"
                + " 'layer_types': ['full_attention', 'full_attention']}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            layer(b, p);
            b.bf16(p + "feed_forward.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "feed_forward.experts." + x + ".gate_up_proj.weight", DIM, 2 * FF);
                b.bf16(p + "feed_forward.experts." + x + ".down_proj.weight", FF, DIM);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code DbrxForCausalLM}: nested ffn_config/attn_config and the unsuffixed w1/w2/v1 expert names. */
    public static SyntheticHf.Builder dbrx() {
        SyntheticHf.Builder b = builder("{'architectures': ['DbrxForCausalLM'], 'd_model': 8, 'n_heads': 2,"
                + " 'max_seq_len': 64, 'vocab_size': 8, 'num_hidden_layers': 2,"
                + " 'attn_config': {'kv_n_heads': 1, 'clip_qkv': 4.0, 'rope_theta': 10000.0},"
                + " 'ffn_config': {'ffn_hidden_size': 8, 'moe_num_experts': 2, 'moe_top_k': 1}}");
        b.bf16("tok_embeddings.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "attn_norm.weight", DIM);
            b.bf16(p + "attn.qkv_proj.weight", 2 * DIM + KV, DIM);
            b.bf16(p + "attn.out_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_norm.weight", DIM);
            b.bf16(p + "ffn_norm.weight", DIM);
            b.bf16(p + "mlp.router.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.experts.mlp.w1", EXPERTS, DIM, DIM);
            b.bf16(p + "mlp.experts.mlp.w2", EXPERTS, DIM, DIM);
            b.bf16(p + "mlp.experts.mlp.v1", EXPERTS, DIM, DIM);
        }
        b.bf16("norm.weight", DIM);
        b.bf16("output.weight", 8, 8);
        return b;
    }

    /** {@code ArcticForCausalLM}: a Llama attention with Mixtral per-expert w1/w2/w3 part names. */
    public static SyntheticHf.Builder arctic() {
        SyntheticHf.Builder b = builder("{'architectures': ['ArcticForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'num_local_experts': 2, 'num_experts_per_tok': 1,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 8,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            layer(b, p);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "block_sparse_moe.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "block_sparse_moe.experts." + x + ".w1.weight", FF, DIM);
                b.bf16(p + "block_sparse_moe.experts." + x + ".w2.weight", DIM, FF);
                b.bf16(p + "block_sparse_moe.experts." + x + ".w3.weight", FF, DIM);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Grok1ForCausalLM}: the transformer-decoder names with the MoE chunk pieces per expert. */
    public static SyntheticHf.Builder grok() {
        SyntheticHf.Builder b = builder("{'architectures': ['Grok1ForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'moe_intermediate_size': 4, 'num_local_experts': 2,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 8,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'attn_logit_softcapping': 30.0, 'router_logit_softcapping': 30.0,"
                + " 'embedding_multiplier_scale': 78.383671875,  'output_multiplier_scale': 2.617966}");
        b.bf16("transformer.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "transformer.decoder_layer." + l + ".";
            b.bf16(p + "attn.attn_norm.weight", DIM);
            b.bf16(p + "attn.attn_qkv.weight", 2 * DIM + KV, DIM);
            b.bf16(p + "attn.attn_out.weight", DIM, DIM);
            b.bf16(p + "moe.moe_norm.weight", DIM);
            b.bf16(p + "moe.gate.weight", EXPERTS, DIM);
            b.bf16(p + "rms_norm.weight", DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "moe." + x + ".linear.weight", FF, DIM);
                b.bf16(p + "moe." + x + ".linear_1.weight", DIM, FF);
                b.bf16(p + "moe." + x + ".linear_v.weight", FF, DIM);
            }
        }
        b.bf16("transformer.final_norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** The Llama-style attention of a decoder layer under {@code prefix} (the + suffix). */
    private static void layer(SyntheticHf.Builder b, String p) {
        b.bf16(p + "input_layernorm.weight", DIM);
        b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(p + "post_attention_layernorm.weight", DIM);
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
                named("ArcticForCausalLM", HfMoeClassicFixturesA::arctic),
                named("DbrxForCausalLM", HfMoeClassicFixturesA::dbrx),
                named("Grok1ForCausalLM", HfMoeClassicFixturesA::grok),
                named("Llama4ForCausalLM", HfMoeClassicFixturesA::llama4));
    }
}
