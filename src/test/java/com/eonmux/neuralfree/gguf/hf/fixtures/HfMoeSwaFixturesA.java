package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-MS part A converter classes: {@code AfmoeModel} (afmoe.py:17-80),
 * {@code FlintModel} (flint.py:9-37), {@code Kolibri1Model} (kolibri1.py:9-22), {@code Cohere2MoeModel}
 * (command_r.py:67-183), the Granite model (granite.py:78-200), {@code Kolibri1Model} and the
 * {@code Gemma4UnifiedModel} suppress-tokens row. Two layers, a hidden size of 8, two query heads, one
 * key/value head, an MoE of two experts and the checkpoint's real names and config keys.
 */
public final class HfMoeSwaFixturesA {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeSwaFixturesA() {
    }

    /** {@code AfmoeForCausalLM}: the Llama-style per-expert projections. */
    public static SyntheticHf.Builder afmoe() {
        SyntheticHf.Builder b = builder("{'architectures': ['AfmoeForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'head_dim': 4, 'moe_intermediate_size': 4,"
                + " 'num_local_experts': 2, 'num_experts_per_tok': 1, 'sliding_window': 4,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
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

    /** {@code FlintForCausalLM}: the Qwen2-MoE fused experts, softmax offset and router expert bias. */
    public static SyntheticHf.Builder flint() {
        SyntheticHf.Builder b = builder("{'architectures': ['FlintForCausalLM'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'num_local_experts': 2,"
                + " 'num_experts_per_tok': 1, 'shared_expert_intermediate_size': 8, 'norm_topk_prob': true,"
                + " 'mlp_only_layers': [0], 'sliding_window': 4,"
                + " 'layer_types': ['full_attention', 'sliding_attention'], 'no_rope_layer_indices': [1],"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.experts.gate_up_proj.weight", EXPERTS, DIM, 2 * FF);
            b.bf16(p + "mlp.experts.down_proj.weight", EXPERTS, FF, DIM);
            b.bf16(p + "mlp.shared_expert.gate_proj.weight", 2, DIM);
            b.bf16(p + "mlp.shared_expert.up_proj.weight", 2, DIM);
            b.bf16(p + "mlp.shared_expert.down_proj.weight", DIM, 2);
            b.bf16(p + "mlp.shared_expert_gate.weight", 1, DIM);
            b.bf16(p + "mlp.softmax_offset", DIM);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Kolibri1ForCausalLM}: the Afmoe names under the kolibri alias config keys. */
    public static SyntheticHf.Builder kolibri1() {
        return afmoe().config(SyntheticHf.json("{'architectures': ['Kolibri1ForCausalLM'], 'hidden_size': 8,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,'head_dim': 4,"
                + " 'moe_intermediate_size': 4, 'num_experts_per_tok': 1, 'norm_topk_prob': true,"
                + " 'shared_expert_intermediate_size': 8, 'layer_types': ['sliding_attention',"
                + " 'full_attention'], 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8}"));
    }

    /** {@code Cohere2MoeForCausalLM}: the iterate-per-expert names with zero biases. */
    public static SyntheticHf.Builder cohere2() {
        SyntheticHf.Builder b = builder("{'architectures': ['Cohere2MoeForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 4, 'num_experts': 2, 'num_experts_per_tok': 1,"
                + " 'shared_expert_combination_strategy': 'average', 'head_dim': 4,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'layer_types': ['sliding_attention', 'full_attention'], 'sliding_window': 4,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "self_attn.q_attn_norm.weight", DIM);
            b.bf16(p + "self_attn.k_attn_norm.weight", DIM);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.bias", FF);
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code GraniteSWAForCausalLM}: a dense Granite body with the SWA metadata. */
    public static SyntheticHf.Builder graniteSwa() {
        SyntheticHf.Builder b = builder("{'architectures': ['GraniteSWAForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'head_dim': 4, 'layer_types': ['sliding_attention',"
                + " 'full_attention'], 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8}");
        body(b, false);
        return b;
    }

    /** {@code GraniteMoeSWAForCausalLM}: the MoE Granite body with the shared expert width. */
    public static SyntheticHf.Builder graniteMoeSwa() {
        SyntheticHf.Builder b = builder("{'architectures': ['GraniteMoeSWAForCausalLM'], 'hidden_size': 8,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'moe_intermediate_size': 4, 'num_experts_per_tok': 1,"
                + " 'num_local_experts': 2, 'shared_intermediate_size': 8,"
                + " 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Gemma4UnifiedForConditionalGeneration}: the suppress_tokens of the generation config. */
    public static SyntheticHf.Builder gemma4Unified() {
        SyntheticHf.Builder b = builder("{'architectures': ['Gemma4UnifiedForConditionalGeneration'],"
                + " 'hidden_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'head_dim': 4, 'intermediate_size': 8,"
                + " 'sliding_window': 4, 'vocab_size': 8, 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0,"
                + " 'max_position_embeddings': 64}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "pre_feedforward_layernorm.weight", DIM);
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", 2 * FF, DIM);
            b.bf16(p + "mlp.up_proj.weight", 2 * FF, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, 2 * FF);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        b.file("generation_config.json", SyntheticHf.json("{'suppress_tokens': [5, 6]}"));
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

    /** The dense Granite body. */
    private static void body(SyntheticHf.Builder b, boolean padding) {
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            attention(b, p);
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", 8, DIM);
            b.bf16(p + "mlp.up_proj.weight", 8, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, 8);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
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
                named("AfmoeForCausalLM", HfMoeSwaFixturesA::afmoe),
                named("FlintForCausalLM", HfMoeSwaFixturesA::flint),
                named("Kolibri1ForCausalLM", HfMoeSwaFixturesA::kolibri1),
                named("Cohere2MoeForCausalLM", HfMoeSwaFixturesA::cohere2),
                named("GraniteSWAForCausalLM", HfMoeSwaFixturesA::graniteSwa),
                named("GraniteMoeSWAForCausalLM", HfMoeSwaFixturesA::graniteMoeSwa),
                named("Gemma4UnifiedForConditionalGeneration", HfMoeSwaFixturesA::gemma4Unified));
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }
}
