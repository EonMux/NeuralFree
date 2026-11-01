package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-D2 part A converter classes: {@code Gemma3Model}, {@code Gemma3NModel},
 * {@code SenseNovaU1Model}, {@code Qwen2VLModel}, {@code Qwen3VLTextModel} and {@code Qwen3ASRTextModel}. Two layers, a
 * hidden size of 8, two query heads, one key/value head, an FFN of {@code 2 * dim} and the exact tensor names and
 * configuration keys of the converter sources.
 */
public final class HfDenseFixturesA {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int FFN = 2 * DIM;
    private static final int VOCAB = 8;
    private static final int HEAD_DIM = 4;

    private HfDenseFixturesA() {
    }

    /** {@code Gemma3ForCausalLM}: the Gemma 3 text layout with its soft-capping and sliding-window configuration. */
    public static SyntheticHf.Builder gemma3() {
        SyntheticHf.Builder b = builder("{'architectures': ['Gemma3ForCausalLM'], 'model_type': 'gemma3',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                + " 'rope_theta': 1000000.0, 'vocab_size': 8, 'final_logit_softcapping': 30.0,"
                + " 'sliding_window_pattern': 1, 'sliding_window': 1024}");
        gemma(b);
        return b;
    }

    /**
     * {@code Gemma3nForCausalLM}: the Gemma 3 layout plus the AltUp projection slots (three id-named matrices each) and
     * the per-layer configuration the metadata reads.
     */
    public static SyntheticHf.Builder gemma3n() {
        SyntheticHf.Builder b = builder("{'architectures': ['Gemma3nForCausalLM'], 'model_type': 'gemma3n',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                + " 'rope_theta': 1000000.0, 'vocab_size': 8, 'sliding_window_pattern': 1,"
                + " 'altup_active_idx': 1, 'altup_num_inputs': 4, 'hidden_size_per_layer_input': 8,"
                + " 'num_kv_shared_layers': 2, 'activation_sparsity_pattern': [0.9, 0.5],"
                + " 'layer_types': ['sliding_attention', 'full_attention'], 'sliding_window': 512}");
        gemma(b);
        for (int slot = 0; slot < 3; slot++) {
            b.bf16("model.altup_projections." + slot + ".weight", 8, 16);
            b.bf16("model.altup_unembed_projections." + slot + ".weight", 16, 8);
        }
        // the per-layer-input (PLE) and AltUp tensors the GEMMA3N catalog requires
        b.bf16("model.embed_tokens_per_layer.weight", VOCAB, 8);
        b.bf16("model.per_layer_model_projection.weight", 8, 8);
        b.bf16("model.per_layer_projection_norm.weight", 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "per_layer_input_gate.weight", 8, 8);
            b.bf16(p + "per_layer_projection.weight", 8, 8);
            b.bf16(p + "post_per_layer_input_norm.weight", 8);
            b.bf16(p + "altup.correction_coefs.weight", 4);
            b.bf16(p + "altup.correct_output_scale", 1);
            b.bf16(p + "altup.prediction_coefs.weight", 4);
            b.bf16(p + "altup.modality_router.weight", 8, 8);
            b.bf16(p + "altup.router_norm.weight", 8);
            b.bf16(p + "laurel.linear_left.weight", 8, 8);
            b.bf16(p + "laurel.linear_right.weight", 8, 8);
            b.bf16(p + "laurel.post_laurel_norm.weight", 8);
        }
        return b;
    }

    /**
     * {@code NEOChatModel}: the SenseNova U1 language model, kept under the {@code language_model.} prefix with the
     * q/k norms and their spatial ({@code _hw}) counterparts the class concatenates.
     */
    public static SyntheticHf.Builder senseNovaU1() {
        SyntheticHf.Builder b = builder("{'architectures': ['NEOChatModel'], 'model_type': 'neo_chat',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'head_dim': 4, 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                + " 'rope_theta': 10000.0, 'rope_theta_hw': 1000000.0, 'vocab_size': 8}");
        b.bf16("language_model.model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "language_model.model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.q_norm.weight", 4);
            b.bf16(p + "self_attn.k_norm.weight", 4);
            b.bf16(p + "self_attn.q_norm_hw.weight", 4);
            b.bf16(p + "self_attn.k_norm_hw.weight", 4);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.up_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, FFN);
        }
        b.bf16("language_model.model.norm.weight", DIM);
        b.bf16("language_model.lm_head.weight", VOCAB, DIM);
        return b;
    }

    /** {@code Qwen2VLForConditionalGeneration}: the Qwen 2 text tower (the {@code thinker.} prefix is optional). */
    public static SyntheticHf.Builder qwen2Vl() {
        SyntheticHf.Builder b = builder("{'architectures': ['Qwen2VLForConditionalGeneration'], 'model_type': 'qwen2_vl',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8, 'rope_scaling': {'rope_type': 'mrope', 'mrope_section': [1, 1, 0]}}");
        qwen(b);
        return b;
    }

    /** {@code Qwen3VLForConditionalGeneration}: the Qwen 3 text tower with the deepstack configuration the class reads. */
    public static SyntheticHf.Builder qwen3VlText() {
        SyntheticHf.Builder b = builder("{'architectures': ['Qwen3VLForConditionalGeneration'], 'model_type': 'qwen3_vl',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8, 'vision_config': {'deepstack_visual_indexes': [0, 1]},"
                + " 'rope_scaling': {'rope_type': 'mrope', 'mrope_section': [1, 1, 0]}}");
        qwen(b);
        for (int l = 0; l < LAYERS; l++) {
            b.bf16("model.layers." + l + ".self_attn.q_norm.weight", 4);
            b.bf16("model.layers." + l + ".self_attn.k_norm.weight", 4);
        }
        return b;
    }

    /** {@code Qwen3ASRForConditionalGeneration}: the Qwen 3 text tower the ASR class converts. */
    public static SyntheticHf.Builder qwen3AsrText() {
        SyntheticHf.Builder b = builder("{'architectures': ['Qwen3ASRForConditionalGeneration'], 'model_type': 'qwen3_asr',"
                + " 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8, 'rope_scaling': {'rope_type': 'mrope', 'mrope_section': [1, 1, 0]}}");
        qwen(b);
        for (int l = 0; l < LAYERS; l++) {
            b.bf16("model.layers." + l + ".self_attn.q_norm.weight", 4);
            b.bf16("model.layers." + l + ".self_attn.k_norm.weight", 4);
        }
        return b;
    }

    /** The Gemma 1/2/3/3n common layout under {@code model.}. */
    private static void gemma(SyntheticHf.Builder b) {
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            // the gemma3 q/k norms (attn_q_norm/attn_k_norm) and the ffn pre/post norms
            // (gemma.py:915-917 pre_feedforward_layernorm -> ffn_norm; the post_feedforward
            // norm maps through the base table to post_ffw_norm)
            b.bf16(p + "self_attn.q_norm.weight", HEAD_DIM);
            b.bf16(p + "self_attn.k_norm.weight", HEAD_DIM);
            b.bf16(p + "pre_feedforward_layernorm.weight", DIM);
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.up_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, FFN);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
    }

    /** The Qwen 2/3 text layout under {@code model.}. */
    private static void qwen(SyntheticHf.Builder b) {
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
                named("Gemma3ForCausalLM", HfDenseFixturesA::gemma3),
                named("Gemma3nForCausalLM", HfDenseFixturesA::gemma3n),
                named("NEOChatModel", HfDenseFixturesA::senseNovaU1),
                named("Qwen2VLForConditionalGeneration", HfDenseFixturesA::qwen2Vl),
                named("Qwen3ASRForConditionalGeneration", HfDenseFixturesA::qwen3AsrText),
                named("Qwen3VLForConditionalGeneration", HfDenseFixturesA::qwen3VlText));
    }
}
