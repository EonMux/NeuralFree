package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-D2 part C converter classes: {@code HunYuanModel}, {@code
 * HunyuanVLTextModel}, {@code Mistral3Model} (the Ministral 3 text branch) and {@code VoxtralRealtimeModel}. Two layers,
 * a hidden size of 8, two query heads, one key/value head, an FFN of {@code 2 * dim} and the exact tensor names and
 * configuration keys of the converter sources.
 */
public final class HfDenseFixturesC {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int FFN = 2 * DIM;
    private static final int VOCAB = 8;

    private HfDenseFixturesC() {
    }

    /**
     * {@code HunYuanDenseV1ForCausalLM}: the dense layout with the unreal {@code num_experts} the class pops, and the
     * end-of-document token id its vocabulary reads.
     */
    public static SyntheticHf.Builder hunyuan() {
        SyntheticHf.Builder b = builder("{'architectures': ['HunYuanDenseV1ForCausalLM'], 'model_type': 'hunyuan_d',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8, 'num_experts': 1, 'pad_token_id': 2, 'eod_token_id': 1,"
                + " 'tie_word_embeddings': false}");
        llama(b);
        // the hunyuan qk-norms (attn_q_norm/attn_k_norm, the q_layernorm/k_layernorm sources)
        for (int l = 0; l < LAYERS; l++) {
            b.bf16("model.layers." + l + ".self_attn.q_layernorm.weight", 4);
            b.bf16("model.layers." + l + ".self_attn.k_layernorm.weight", 4);
        }
        return b;
    }

    /**
     * {@code HunYuanVLForConditionalGeneration}: the dense layout with the dynamic RoPE and its mrope sections the
     * class normalizes to XD-RoPE, plus the alpha the loader rescales from.
     */
    public static SyntheticHf.Builder hunyuanVlText() {
        SyntheticHf.Builder b = builder("{'architectures': ['HunYuanVLForConditionalGeneration'], 'model_type': 'hunyuan_vl',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8, 'tie_word_embeddings': false,"
                + " 'rope_scaling': {'rope_type': 'dynamic', 'mrope_section': [1, 1, 0, 0], 'alpha': 50.0,"
                + " 'rope_theta': 10000.0, 'factor': 1.0}}");
        llama(b);
        for (int l = 0; l < LAYERS; l++) {
            b.bf16("model.layers." + l + ".self_attn.q_layernorm.weight", 4);
            b.bf16("model.layers." + l + ".self_attn.k_layernorm.weight", 4);
        }
        return b;
    }

    /**
     * {@code Ministral3ForCausalLM}: the Llama layout with the required YaRN RoPE the Ministral 3 class asserts, its
     * log multiplier and its attention temperature scale.
     */
    public static SyntheticHf.Builder ministral3() {
        SyntheticHf.Builder b = builder("{'architectures': ['Ministral3ForCausalLM'], 'model_type': 'ministral3',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'vocab_size': 8,"
                + " 'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 4096,"
                + " 'mscale_all_dim': 0.5, 'llama_4_scaling_beta': 1.0}}");
        llama(b);
        return b;
    }

    /**
     * {@code VoxtralRealtimeForConditionalGeneration}: the Llama layout the real-time class converts with, and its
     * default delay token count.
     */
    public static SyntheticHf.Builder voxtralRealtime() {
        SyntheticHf.Builder b = builder("{'architectures': ['VoxtralRealtimeForConditionalGeneration'],"
                + " 'model_type': 'voxtral_realtime', 'hidden_size': 8, 'intermediate_size': 16,"
                + " 'num_attention_heads': 2, 'num_key_value_heads': 1, 'num_hidden_layers': 2,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0, 'vocab_size': 8,"
                + " 'default_num_delay_tokens': 1}");
        llama(b);
        return b;
    }

    /** The Llama layout over {@code model.} with the shared final norm and head. */
    private static void llama(SyntheticHf.Builder b) {
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
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
                named("HunYuanDenseV1ForCausalLM", HfDenseFixturesC::hunyuan),
                named("HunYuanVLForConditionalGeneration", HfDenseFixturesC::hunyuanVlText),
                named("Ministral3ForCausalLM", HfDenseFixturesC::ministral3),
                named("VoxtralRealtimeForConditionalGeneration", HfDenseFixturesC::voxtralRealtime));
    }
}
