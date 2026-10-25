package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-D2 part B converter classes: {@code ApertusModel}, {@code ArceeModel}, {@code
 * OlmoModel}, {@code Olmo2Model}, {@code Phi3MiniModel} and {@code Qwen3MoeModel}. Two layers, a hidden size of 8, two
 * query heads, one key/value head, an FFN of {@code 2 * dim} and the exact tensor names and configuration keys of the
 * converter sources.
 */
public final class HfDenseFixturesB {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int FFN = 2 * DIM;
    private static final int EXPERTS = 2;
    private static final int EXPERT_FFN = 4;
    private static final int VOCAB = 8;

    private HfDenseFixturesB() {
    }

    /** {@code ApertusForCausalLM}: the Llama layout with the xIELU scalar activation parameters of every layer. */
    public static SyntheticHf.Builder apertus() {
        SyntheticHf.Builder b = builder("{'architectures': ['ApertusForCausalLM'], 'model_type': 'apertus',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8, 'tie_word_embeddings': false}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            llamaLayer(b, p);
            b.f32(p + "mlp.act_fn.alpha_n");
            b.f32(p + "mlp.act_fn.alpha_p");
            b.f32(p + "mlp.act_fn.beta");
            b.f32(p + "mlp.act_fn.eps");
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
        return b;
    }

    /** {@code ArceeForCausalLM}: the Llama layout with no pooling side files of its own. */
    public static SyntheticHf.Builder arcee() {
        SyntheticHf.Builder b = builder("{'architectures': ['ArceeForCausalLM'], 'model_type': 'arcee',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        llama(b);
        return b;
    }

    /** {@code OLMoForCausalLM}: the Llama layout the Olmo class renames, with its optional clip of the QKV scores. */
    public static SyntheticHf.Builder olmo() {
        SyntheticHf.Builder b = builder("{'architectures': ['OLMoForCausalLM'], 'model_type': 'olmo',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8, 'clip_qkv': 100.0}");
        llama(b);
        return b;
    }

    /** {@code Olmo2ForCausalLM}: the Llama layout with the sliding window and its per-layer pattern. */
    public static SyntheticHf.Builder olmo2() {
        SyntheticHf.Builder b = builder("{'architectures': ['Olmo2ForCausalLM'], 'model_type': 'olmo2',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8, 'sliding_window': 4096,"
                + " 'layer_types': ['full_attention', 'sliding_attention']}");
        llama(b);
        return b;
    }

    /** {@code Phi3ForCausalLM}: the Llama layout with the Phi-3 RoPE parameters its metadata requires. */
    public static SyntheticHf.Builder phi3Mini() {
        SyntheticHf.Builder b = builder("{'architectures': ['Phi3ForCausalLM'], 'model_type': 'phi3',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'n_positions': 64, 'rms_norm_eps': 1e-05, 'vocab_size': 8,"
                + " 'rope_parameters': {'rope_theta': 10000.0, 'factor': 4.0, 'original_max_position_embeddings': 4096},"
                + " 'max_position_embeddings': 32768}");
        llama(b);
        return b;
    }

    /** {@code Qwen3MoeForCausalLM}: the Qwen 3 layout with the per-expert and shared-expert MLPs of every layer. */
    public static SyntheticHf.Builder qwen3Moe() {
        SyntheticHf.Builder b = builder("{'architectures': ['Qwen3MoeForCausalLM'], 'model_type': 'qwen3_moe',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06, 'rope_theta': 1000000.0,"
                + " 'vocab_size': 8, 'num_experts': 2, 'moe_intermediate_size': 4,"
                + " 'shared_expert_intermediate_size': 4}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, EXPERT_FFN);
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", EXPERT_FFN, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", EXPERT_FFN, DIM);
            }
            b.bf16(p + "mlp.shared_expert.down_proj.weight", DIM, EXPERT_FFN);
            b.bf16(p + "mlp.shared_expert.gate_proj.weight", EXPERT_FFN, DIM);
            b.bf16(p + "mlp.shared_expert.up_proj.weight", EXPERT_FFN, DIM);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
        return b;
    }

    /** The Llama-style attention of a decoder layer under {@code prefix} (the q/k norms are absent). */
    private static void llamaLayer(SyntheticHf.Builder b, String p) {
        b.bf16(p + "input_layernorm.weight", DIM);
        b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
        b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(p + "post_attention_layernorm.weight", DIM);
        b.bf16(p + "mlp.gate_proj.weight", FFN, DIM);
        b.bf16(p + "mlp.up_proj.weight", FFN, DIM);
        b.bf16(p + "mlp.down_proj.weight", DIM, FFN);
    }

    /** The Llama layout of {@link #llamaLayer} over {@code model.} with the shared final norm and head. */
    private static void llama(SyntheticHf.Builder b) {
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            llamaLayer(b, "model.layers." + l + ".");
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
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
                named("ApertusForCausalLM", HfDenseFixturesB::apertus),
                named("ArceeForCausalLM", HfDenseFixturesB::arcee),
                named("OLMoForCausalLM", HfDenseFixturesB::olmo),
                named("Olmo2ForCausalLM", HfDenseFixturesB::olmo2),
                named("Phi3ForCausalLM", HfDenseFixturesB::phi3Mini),
                named("Qwen3MoeForCausalLM", HfDenseFixturesB::qwen3Moe));
    }
}
