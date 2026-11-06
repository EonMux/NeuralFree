package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the converter classes and the gpt-oss row: {@code Step35Model}
 * (step3.py:102-341, the per-layer arrays and MTP), {@code MuseSparkModel} (muse_spark.py:39-182, the
 * residual gates and the fused experts) and {@code GptOssModel} (gpt_oss.py:15-130, the BF16 and MXFP4
 * expert forms and the attention sinks). Two layers, a hidden size of 8, two query heads, one key/value
 * head, an MoE of two experts and the checkpoint's real names and config keys.
 */
public final class HfMoeSwaFixturesD {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeSwaFixturesD() {
    }

    /** {@code Step3p5ForCausalLM}: the dense lead with MoE layers, the per-layer arrays and the shared expert. */
    public static SyntheticHf.Builder step35() {
        SyntheticHf.Builder b = builder("{'architectures': ['Step3p5ForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'attention_other_setting': {'num_attention_heads': 1, 'num_attention_groups': 1},"
                + " 'sliding_window': 4, 'moe_num_experts': 2, 'moe_top_k': 1,"
                + " 'moe_intermediate_size': 4, 'share_expert_dim': 8,"
                + " 'swiglu_limits': [4.0, 0.0], 'swiglu_limits_shared': [4.0, 0.0],"
                + " 'moe_layers_enum': '1', 'moe_every_n_layer': 2,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_attention_groups': 1,"
                + " 'intermediate_size': 8, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            if (l == 0) {
                // the sliding layer runs the attention_other_setting widths (1 head of 4)
                b.bf16(p + "self_attn.q_proj.weight", 4, DIM);
                b.bf16(p + "self_attn.k_proj.weight", 4, DIM);
                b.bf16(p + "self_attn.v_proj.weight", 4, DIM);
                b.bf16(p + "self_attn.o_proj.weight", DIM, 4);
                b.bf16(p + "mlp.gate_proj.weight", 8, DIM);
                b.bf16(p + "mlp.up_proj.weight", 8, DIM);
                b.bf16(p + "mlp.down_proj.weight", DIM, 8);
            } else {
                b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
                b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
                b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
                b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
                // the step3.5 MoE forms: the moe.gate router and the merged expert stacks
                // (tensor_mapping.py:534/666/758)
                b.bf16(p + "moe.gate.weight", EXPERTS, DIM);
                b.bf16(p + "moe.gate_proj.weight", EXPERTS, FF, DIM);
                b.bf16(p + "moe.up_proj.weight", EXPERTS, FF, DIM);
                b.bf16(p + "moe.down_proj.weight", EXPERTS, DIM, FF);
                b.bf16(p + "mlp.shared_expert.gate_proj.weight", 4, DIM);
                b.bf16(p + "mlp.shared_expert.up_proj.weight", 4, DIM);
                b.bf16(p + "mlp.shared_expert.down_proj.weight", DIM, 4);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code MuseSparkForConditionalGeneration}: the delta norms, the gate rows and the residual gates. */
    public static SyntheticHf.Builder museSpark() {
        SyntheticHf.Builder b = builder("{'architectures': ['MuseSparkForConditionalGeneration'],"
                + " 'hidden_size': 8, 'head_dim': 4, 'output_multiplier': 1.0, 'final_logit_softcapping': 30.0,"
                + " 'post_norm_eps': 1e-05, 'sliding_window': 4,"
                + " 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'moe_hidden_size': 2, 'moe_intermediate_size': 4, 'moe_top_k': 1,"
                + " 'num_experts': 2, 'num_experts_per_tok': 1,"
                + " 'routed_scaling_factor': 1.0, 'residual_gate_temperature': 1.0,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            // the muse-spark norms: the ffn pre/post norms; no post-attention norm
            b.bf16(p + "pre_feedforward_layernorm.weight", DIM);
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
            b.bf16(p + "mlp.pre_expert_norm.weight", DIM);
            b.bf16(p + "mlp.experts.post_expert_norm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_residual_gate.gate", 4);
            b.bf16(p + "post_feedforward_residual_gate.gate", 4);
            b.bf16(p + "self_attn.attn_gate_proj.weight", DIM, DIM);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            // the muse router bias (tensor_patterns.tsv:355) and the latent projections
            // (tensor_patterns.tsv:468/470; the latent width is moe_hidden_size)
            b.bf16(p + "mlp.gate.e_score_correction_bias", EXPERTS);
            b.bf16(p + "mlp.pre_expert_proj.weight", 2, DIM);
            b.bf16(p + "mlp.post_expert_proj.weight", DIM, 2);
            b.bf16(p + "mlp.experts.gate_up_proj", EXPERTS, 2 * FF, DIM);
            b.bf16(p + "mlp.experts.down_proj", EXPERTS, DIM, FF);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code GptOssForCausalLM}: the BF16 fused experts, their biases and the attention sinks. */
    public static SyntheticHf.Builder gptOss() {
        SyntheticHf.Builder b = builder("{'architectures': ['GptOssForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 4, 'sliding_window': 4,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_local_experts': 2,"
                + " 'num_experts_per_tok': 1, 'num_key_value_heads': 1, 'head_dim': 4,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.o_proj.bias", DIM);
            b.bf16(p + "self_attn.sinks", 2);
            b.bf16(p + "mlp.router.weight", EXPERTS, DIM);
            b.bf16(p + "mlp.router.bias", EXPERTS);
            // the merged expert forms the gpt-oss hook and map handle (the hooks-test names)
            b.bf16(p + "mlp.experts.gate_up_proj.weight", EXPERTS, DIM, 2 * FF);
            b.bf16(p + "mlp.experts.gate_up_proj_bias", EXPERTS, 2 * FF);
            b.bf16(p + "mlp.experts.down_proj.weight", EXPERTS, FF, DIM);
            b.bf16(p + "mlp.experts.down_proj.bias", EXPERTS, DIM);
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
                named("Step3p5ForCausalLM", HfMoeSwaFixturesD::step35),
                named("MuseSparkForConditionalGeneration", HfMoeSwaFixturesD::museSpark),
                named("GptOssForCausalLM", HfMoeSwaFixturesD::gptOss));
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }
}
