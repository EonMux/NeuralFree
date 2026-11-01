package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-MS part C converter classes: {@code LagunaModel} (laguna.py:17-208, the
 * attention gating and per-layer heads), {@code MapleModel} (maple.py:15-129), {@code MellumModel}
 * (mellum.py:15-131), {@code ExaoneMoEModel} (exaone.py:132-229, on the HK-B2 Exaone4 hooks) and
 * {@code SmallThinkerModel} (smallthinker.py:15-83). Two layers, a hidden size of 8, two query heads, one
 * key/value head, an MoE of two experts and the checkpoint's real names and config keys.
 */
public final class HfMoeSwaFixturesC {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeSwaFixturesC() {
    }

    /** {@code LagunaForCausalLM}: the per-head attention gate projection and the per-expert projections. */
    public static SyntheticHf.Builder laguna() {
        SyntheticHf.Builder b = builder("{'architectures': ['LagunaForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'gating': 'per-head', 'num_attention_heads_per_layer': [2, 2],"
                + " 'moe_intermediate_size': 4, 'shared_expert_intermediate_size': 8,"
                + " 'moe_routed_scaling_factor': 1.0, 'num_experts': 2, 'num_experts_per_tok': 1,"
                + " 'norm_topk_prob': true,"
                + " 'sliding_window': 4, 'mlp_layer_types': ['sparse', 'sparse'],"
                + " 'partial_rotary_factor': 0.5,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "self_attn.g_proj.weight", 2, DIM);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            // the laguna router bias takes the normal mapping path (laguna.py:174)
            b.bf16(p + "mlp.experts.e_score_correction.bias", EXPERTS);
            b.bf16(p + "mlp.shared_expert.gate_proj.weight", 8, DIM);
            b.bf16(p + "mlp.shared_expert.up_proj.weight", 8, DIM);
            b.bf16(p + "mlp.shared_expert.down_proj.weight", DIM, 8);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code MapleForCausalLM}: the per-expert projections under the 7.0 SwiGLU clamp. */
    public static SyntheticHf.Builder maple() {
        SyntheticHf.Builder b = builder("{'architectures': ['MapleForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'hidden_act': 'silu', 'partial_rotary_factor': 0.5,"
                + " 'moe_intermediate_size': 4, 'num_experts': 2, 'num_experts_per_tok': 1,"
                + " 'sliding_window': 4, 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code MellumForCausalLM}: the Qwen-style MTP alias with a sliding window. */
    public static SyntheticHf.Builder mellum() {
        SyntheticHf.Builder b = builder("{'architectures': ['MellumForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'moe_intermediate_size': 4, 'num_local_experts': 2,"
                + " 'num_experts_per_tok': 1, 'use_sliding_window': true, 'sliding_window': 4,"
                + " 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code ExaoneMoEForCausalLM}: the Exaone4 body with the per-expert projections. */
    public static SyntheticHf.Builder exaoneMoe() {
        SyntheticHf.Builder b = builder("{'architectures': ['ExaoneMoEForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'moe_intermediate_size': 4, 'num_experts_per_tok': 1,"
                + " 'num_shared_experts': 1, 'score_func': 'sigmoid', 'norm_topk_prob': true,"
                + " 'routed_scaling_factor': 1.0,"
                + " 'num_local_experts': 2, 'sliding_window': 4,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.q_norm.weight", KV);
            b.bf16(p + "self_attn.k_norm.weight", KV);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.shared_experts.gate_proj.weight", FF, DIM);
            b.bf16(p + "mlp.shared_experts.up_proj.weight", FF, DIM);
            b.bf16(p + "mlp.shared_experts.down_proj.weight", DIM, FF);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code SmallThinkerForCausalLM}: the block-sparse named experts and the window layout. */
    public static SyntheticHf.Builder smallthinker() {
        SyntheticHf.Builder b = builder("{'architectures': ['SmallThinkerForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'moe_ffn_hidden_size': 4, 'moe_num_primary_experts': 2,"
                + " 'moe_num_active_primary_experts': 1, 'sliding_window_layout': ['0', '4'],"
                + " 'sliding_window_size': 4,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "block_sparse_moe.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "block_sparse_moe.experts." + x + ".down.weight", DIM, FF);
                b.bf16(p + "block_sparse_moe.experts." + x + ".gate.weight", FF, DIM);
                b.bf16(p + "block_sparse_moe.experts." + x + ".up.weight", FF, DIM);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** The Llama-style attention of a decoder layer under {@code prefix}. */
    private static void attention(SyntheticHf.Builder b, String p) {
        b.bf16(p + "input_layernorm.weight", DIM);
        b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(p + "post_attention_layernorm.weight", DIM);
    }

    /** One end-to-end fixture: its leading HF architecture name and the checkpoint builder. */
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
                named("LagunaForCausalLM", HfMoeSwaFixturesC::laguna),
                named("MapleForCausalLM", HfMoeSwaFixturesC::maple),
                named("MellumForCausalLM", HfMoeSwaFixturesC::mellum),
                named("ExaoneMoEForCausalLM", HfMoeSwaFixturesC::exaoneMoe),
                named("SmallThinkerForCausalLM", HfMoeSwaFixturesC::smallthinker));
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }
}
