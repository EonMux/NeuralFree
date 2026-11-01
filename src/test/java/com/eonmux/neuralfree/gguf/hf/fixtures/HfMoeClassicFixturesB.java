package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-MC part B converter classes: {@code BailingMoeModel} and its V2/Sarvam variants
 * (bailingmoe.py:15-219), {@code Dots1Model} (dots1.py:15-33) and {@code Ernie4_5MoeModel} (ernie.py:78-157). Two
 * layers, a hidden size of 8, two query heads, one key/value head, an MoE of two experts and the checkpoint's real
 * names and config keys.
 */
public final class HfMoeClassicFixturesB {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeClassicFixturesB() {
    }

    /** {@code BailingMoeForCausalLM}: fused QKV, attention.dense, per-expert down/gate/up and a normalized norm head. */
    public static SyntheticHf.Builder bailingMoe() {
        SyntheticHf.Builder b = builder("{'architectures': ['BailingMoeForCausalLM'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'num_shared_experts': 1, 'first_k_dense_replace': 0,"
                + " 'num_local_experts': 2, 'num_experts_per_tok': 1, 'norm_topk_prob': false, 'norm_head': true,"
                + " 'num_hidden_layers': 2,"
                + " 'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 8,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.query_key_value.weight", DIM + 2 * KV, DIM);
            b.bf16(p + "self_attn.attention.dense.weight", DIM, DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            // the BAILINGMOE arch maps no dense FFN (arch_tensors.tsv), so every layer is MoE
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

    /** {@code BailingMoeV2ForCausalLM} with one appended NextN layer and the per-layer router bias. */
    public static SyntheticHf.Builder bailingMoeV2() {
        int depth = LAYERS + 1;
        SyntheticHf.Builder b = builder("{'architectures': ['BailingMoeV2ForCausalLM'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'num_shared_experts': 1, 'num_nextn_predict_layers': 1,"
                + " 'first_k_dense_replace': 1, 'num_local_experts': 2, 'num_experts_per_tok': 1,"
                + " 'score_func': 'sigmoid', 'norm_topk_prob': false,"
                + " 'routed_scaling_factor': 1.0, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'vocab_size': 8, 'max_position_embeddings': 64,"
                + " 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < depth; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            // BAILINGMOE2 maps the fused QKV whole (tensor_mapping.py:281) and carries the
            // attention q/k layernorms; no split q/k/v tensors
            b.bf16(p + "attention.query_key_value.weight", DIM + 2 * KV, DIM);
            b.bf16(p + "attention.query_layernorm.weight", KV);
            b.bf16(p + "attention.key_layernorm.weight", KV);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            if (l < 1) {
                // first_k_dense_replace 1: layer 0 is dense, the rest are MoE
                b.bf16(p + "mlp.gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.down_proj.weight", DIM, FF);
                continue;
            }
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.gate.expert_bias", EXPERTS);
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

    /** {@code Dots1ForCausalLM}: the Qwen2Moe naming with n_routed_experts and shared experts by their own names. */
    public static SyntheticHf.Builder dots1() {
        SyntheticHf.Builder b = builder("{'architectures': ['Dots1ForCausalLM'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'n_routed_experts': 2, 'num_experts_per_tok': 1,"
                + " 'score_func': 'sigmoid',"
                + " 'n_shared_experts': 1, 'shared_expert_intermediate_size': 4, 'first_k_dense_replace': 1,"
                + " 'routed_scaling_factor': 1.0, 'norm_topk_prob': false, 'num_hidden_layers': 2,"
                + " 'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 8,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0}");
        // the dots1 vocabulary strictly requires a sentencepiece tokenizer.model
        b.binaryFile("tokenizer.model", HfSpecialFixturesT5.unigramModel());
        b.bf16("model.embed_tokens.weight", 8, 8);
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
            if (l < 1) {
                // first_k_dense_replace 1: layer 0 is dense, the rest are MoE
                b.bf16(p + "mlp.gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.down_proj.weight", DIM, FF);
                continue;
            }
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

    /**
     * {@code Ernie4_5_MoeForCausalLM}: interleaved MoE layers with the MTP tensors of the checkpoint (dropped).
     */
    public static SyntheticHf.Builder ernie45Moe() {
        SyntheticHf.Builder b = builder("{'architectures': ['Ernie4_5_MoeForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'moe_num_experts': 2, 'moe_k': 1, 'moe_layer_interval': 2,"
                + " 'moe_layer_start_index': 1, 'moe_intermediate_size': 4, 'moe_num_shared_experts': 1,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 8,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0}");
        // the ernie vocabulary strictly requires a sentencepiece tokenizer.model
        b.binaryFile("tokenizer.model", HfSpecialFixturesT5.unigramModel());
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            if (l < 1) {
                // moe_layer_start_index 1: layer 0 is dense, the rest are MoE
                b.bf16(p + "mlp.gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.down_proj.weight", DIM, FF);
                continue;
            }
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
        b.bf16("model.mtp_block.0.input_layernorm.weight", DIM);
        b.bf16("model.mtp_block.0.mlp.experts.0.gate_proj.weight", FF, DIM);
        b.bf16("model.mtp_linear_proj.0.weight", DIM, DIM);
        return b;
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
                named("BailingMoeForCausalLM", HfMoeClassicFixturesB::bailingMoe),
                named("BailingMoeV2ForCausalLM", HfMoeClassicFixturesB::bailingMoeV2),
                named("Dots1ForCausalLM", HfMoeClassicFixturesB::dots1),
                named("Ernie4_5_MoeForCausalLM", HfMoeClassicFixturesB::ernie45Moe));
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }
}
