package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the converter classes: {@code OpenELMModel}, {@code Phi2Model},
 * {@code StarCoder2Model}, {@code Spark2_5Model}, {@code TalkieModel} and {@code GraniteSwitchModel}. Two layers,
 * a hidden size of 8, the exact tensor names and configuration keys of the converter sources. The SentencePiece
 * backends of {@code InternLM2Model}, {@code OrionModel}, {@code PlamoModel} and {@code PanguEmbeddedModel} need a
 * binary {@code tokenizer.model} and are not covered here.
 */
public final class HfDenseClassicFixturesE {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int FF = 4;
    private static final String GPT2_TOKENIZER_CONFIG = SyntheticHf.GPT2_TOKENIZER_CONFIG;

    private HfDenseClassicFixturesE() {
    }

    /** {@code OpenELMForCausalLM}: the per-layer widths, heads and {@code ffn.proj_1} fused gate/up projection. */
    public static SyntheticHf.Builder openelm() {
        SyntheticHf.Builder b = builder("{'architectures': ['OpenELMForCausalLM'], 'model_dim': 8,"
                + " 'head_dim': 4, 'ffn_dim_divisor': 4, 'ffn_multipliers': [0.5, 0.5], 'num_transformer_layers': 2,"
                + " 'num_query_heads': [2, 2], 'num_kv_heads': [1, 1], 'max_context_length': 64,"
                + " 'rope_freq_constant': 10000.0, 'vocab_size': 8}");
        b.bf16("transformer.token_embeddings.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "transformer.layers." + l + ".";
            b.bf16(p + "attn_norm.weight", DIM);
            b.bf16(p + "attn.qkv_proj.weight", 2 * DIM + KV, DIM);
            b.bf16(p + "attn.out_proj.weight", DIM, DIM);
            b.bf16(p + "attn.q_norm.weight", 4);
            b.bf16(p + "attn.k_norm.weight", 4);
            b.bf16(p + "ffn.proj_1.weight", 2 * FF, DIM);
            b.bf16(p + "ffn.proj_2.weight", DIM, FF);
            b.bf16(p + "ffn_norm.weight", DIM);
        }
        b.bf16("transformer.norm.weight", DIM);
        return b;
    }

    /** {@code PhiForCausalLM} (phi-2): the unpermuted Phi attention with its partial rotary factor. */
    public static SyntheticHf.Builder phi2() {
        SyntheticHf.Builder b = builder("{'architectures': ['PhiForCausalLM'], 'hidden_size': 8,"
                + " 'num_attention_heads': 2, 'num_key_value_heads': 2, 'num_hidden_layers': 2,"
                + " 'max_position_embeddings': 64, 'partial_rotary_factor': 0.5, 'layer_norm_epsilon': 1e-05,"
                + " 'vocab_size': 8}");
        b.bf16("transformer.embd.wte.weight", 8, 8);
        b.bf16("transformer.ln_f.weight", DIM);
        b.bf16("transformer.ln_f.bias", DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "transformer.h." + l + ".";
            b.bf16(p + "ln_1.weight", DIM);
            b.bf16(p + "ln_1.bias", DIM);
            b.bf16(p + "attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "attn.k_proj.weight", DIM, DIM);
            b.bf16(p + "attn.v_proj.weight", DIM, DIM);
            b.bf16(p + "attn.out_proj.weight", DIM, DIM);
            b.bf16(p + "attn.out_proj.bias", DIM);
            // phi2's graph loads no post-attention norm (phi2.cpp has no FFN_NORM load)
            b.bf16(p + "mlp.fc1.weight", 2 * FF, DIM);
            b.bf16(p + "mlp.fc1.bias", 2 * FF);
            b.bf16(p + "mlp.fc2.weight", DIM, 2 * FF);
            b.bf16(p + "mlp.fc2.bias", DIM);
        }
        b.bf16("lm_head.weight", 8, 8);
        b.bf16("lm_head.bias", 8);
        return b;
    }

    /** {@code Starcoder2ForCausalLM}: the base-class keys and attention. */
    public static SyntheticHf.Builder starcoder2() {
        SyntheticHf.Builder b = builder("{'architectures': ['Starcoder2ForCausalLM'], 'hidden_size': 8,"
                + " 'intermediate_size': 8, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'vocab_size': 8, 'max_position_embeddings': 64,"
                + " 'layer_norm_epsilon': 1e-05, 'rope_theta': 10000.0, 'partial_rotary_factor': 0.5}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        b.bf16("model.norm.bias", DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "input_layernorm.bias", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.o_proj.bias", DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "post_attention_layernorm.bias", DIM);
            // the starcoder2 mlp is the GPT-2 form: c_fc up and c_proj down, no gate;
            // the arch carries no q/k norms (constants.py STARCODER2)
            b.bf16(p + "mlp.c_fc.weight", FF, DIM);
            b.bf16(p + "mlp.c_fc.bias", FF);
            b.bf16(p + "mlp.c_proj.weight", DIM, FF);
            b.bf16(p + "mlp.c_proj.bias", DIM);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code Spark2_5ForCausalLM}: the fused {@code q_k_v_proj}, the head gate and the sliding attention types. */
    public static SyntheticHf.Builder spark25() {
        SyntheticHf.Builder b = builder("{'architectures': ['Spark2_5ForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 2, 'vocab_size': 8, 'sliding_window': 4,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'layer_types': ['sliding_attention', 'full_attention'], 'gate_attn_act_mode': 'sigmoid',"
                + " 'headwise_attn_output_gate': true, 'hidden_act': 'gelu',"
                + " 'rope_scaling': {'full_attention': {'partial_rotary_factor': 0.5},"
                + " 'sliding_attention': {'partial_rotary_factor': 0.5}}}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_k_v_proj.weight", DIM + KV + KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.g_proj.weight", 2, 1);
            b.bf16(p + "mlp.gate_proj.weight", FF, DIM);
            b.bf16(p + "mlp.up_proj.weight", FF, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, FF);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code TalkieForCausalLM}: the block-prefixed gains, query/key sign flip and head gains. */
    public static SyntheticHf.Builder talkie() {
        SyntheticHf.Builder b = builder("{'architectures': ['TalkieForCausalLM'], 'hidden_size': 8,"
                + " 'head_dim': 4, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 2,"
                + " 'vocab_size': 8, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0}");
        b.bf16("model.embed.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.blocks." + l + ".";
            // the TALKIE arch carries no ATTN_NORM/FFN_NORM (constants.py TALKIE); the
            // query-side head gain is the ATTN_Q_NORM (tensor_patterns.tsv:541)
            b.bf16(p + "attn.attn_query.weight", DIM, DIM);
            b.bf16(p + "attn.attn_key.weight", DIM, DIM);
            b.bf16(p + "attn.attn_value.weight", DIM, DIM);
            b.bf16(p + "attn.attn_resid.weight", DIM, DIM);
            b.f32(p + "attn_gain.a_g", 1);
            b.f32(p + "embed_skip.a_g", 1);
            b.bf16(p + "attn.head_gain.head_g", 2);
            b.bf16(p + "mlp.mlp_gate.weight", FF, DIM);
            b.bf16(p + "mlp.mlp_linear.weight", FF, DIM);
            b.bf16(p + "mlp.mlp_resid.weight", DIM, FF);
            b.f32(p + "mlp_gain.a_g", 1);
        }
        b.bf16("model.lm_head.weight", 8, 8);
        // the logit scale must be positive (talkie.py:33-35 turns the gain into add_logit_scale)
        b.tensor("lm_head_gain.w_g", com.eonmux.neuralfree.io.safetensors.StDType.F32, new long[]{1},
                com.eonmux.neuralfree.gguf.testutil.SyntheticHf.f32(2.0f));
        return b;
    }

    /** {@code GraniteSwitchForCausalLM}: the GraniteMoe base keys over one adapter with its fused projections and
     *  LoRA slices. */
    public static SyntheticHf.Builder graniteSwitch() {
        SyntheticHf.Builder b = builder("{'architectures': ['GraniteSwitchForCausalLM'], 'hidden_size': 8,"
                + " 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4, 'num_hidden_layers': 2,"
                + " 'vocab_size': 8, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'num_adapters': 1, 'max_lora_rank': 2, 'adapter_token_ids': [3],"
                + " 'adapter_substitute_token_ids': [5], 'adapter_names': ['ratescore'],"
                + " 'shared_intermediate_size': 4, 'num_experts_per_tok': 1,"
                + " 'attention_multiplier': 1.0, 'embedding_multiplier': 1.0, 'residual_multiplier': 1.0,"
                + " 'logits_scaling': 1.0}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.qkv_proj.base_layer.weight", DIM + KV + KV, DIM);
            b.bf16(p + "self_attn.qkv_proj.lora_A_slices.0", 1, 1, 2, 8);
            b.bf16(p + "self_attn.qkv_proj.lora_A_slices.1", 1, 1, 2, 8);
            b.bf16(p + "self_attn.qkv_proj.lora_A_slices.2", 1, 1, 2, 8);
            b.bf16(p + "self_attn.qkv_proj.lora_B_slices.0", 1, 1, 8, 2);
            b.bf16(p + "self_attn.qkv_proj.lora_B_slices.1", 1, 1, KV, 2);
            b.bf16(p + "self_attn.qkv_proj.lora_B_slices.2", 1, 1, KV, 2);
            b.bf16(p + "self_attn.o_proj.base_layer.weight", DIM, DIM);
            b.bf16(p + "self_attn.o_proj.lora_A", 1, 1, 2, 8);
            b.bf16(p + "self_attn.o_proj.lora_B", 1, 1, 8, 2);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "shared_mlp.input_linear.base_layer.weight", 2 * FF, DIM);
            b.bf16(p + "shared_mlp.input_linear.lora_A_slices.0", 1, 1, 2, 8);
            b.bf16(p + "shared_mlp.input_linear.lora_A_slices.1", 1, 1, 2, 8);
            b.bf16(p + "shared_mlp.input_linear.lora_B_slices.0", 1, 1, 4, 2);
            b.bf16(p + "shared_mlp.input_linear.lora_B_slices.1", 1, 1, 4, 2);
            b.bf16(p + "shared_mlp.output_linear.base_layer.weight", DIM, FF);
            b.bf16(p + "shared_mlp.output_linear.lora_A", 1, 1, 2, 4);
            b.bf16(p + "shared_mlp.output_linear.lora_B", 1, 1, 8, 2);
        }
        b.bf16("model.norm.weight", DIM);
        return b;
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig))
                .tokenizerJson(SyntheticHf.GPT2_TOKENIZER)
                .tokenizerConfigJson(GPT2_TOKENIZER_CONFIG);
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
                named("OpenELMForCausalLM", HfDenseClassicFixturesE::openelm),
                named("PhiForCausalLM", HfDenseClassicFixturesE::phi2),
                named("Starcoder2ForCausalLM", HfDenseClassicFixturesE::starcoder2),
                named("Spark2_5ForCausalLM", HfDenseClassicFixturesE::spark25),
                named("TalkieForCausalLM", HfDenseClassicFixturesE::talkie),
                named("GraniteSwitchForCausalLM", HfDenseClassicFixturesE::graniteSwitch));
    }
}
