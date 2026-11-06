package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the two DFLASH draft converters this package delivers: {@code DeepseekV41DSparkModel}
 * (deepseek41.py:331-442) and {@code MistralLarge4DSparkModel} (mistral4.py:132-217). Both carry only the
 * {@code mtp.*} / {@code mtp_dspark.*} tensors the converters export; the target model supplies the vocabulary at
 * runtime.
 */
public final class HfDraftFixtures {

    private HfDraftFixtures() {
    }

    private static final int V41_DIM = 32;
    private static final int V41_HEADS = 4;
    private static final int V41_HEAD_DIM = 8;
    private static final int V41_Q_LORA = 8;
    private static final int V41_MOE_FF = 16;
    private static final int V41_VOCAB = 16;

    /**
     * {@code DeepseekV41DSparkModel}: two stages of the V4.1 sparse-attention block with hyper connections, MXFP4
     * experts and the DSpark Markov and confidence heads.
     */
    public static SyntheticHf.Builder deepseekV41Dspark() {
        return deepseekV41("DeepseekV41DSparkModel");
    }

    /**
     * A {@code DeepseekV41ForCausalLM} target carrying the same {@code mtp.*} tensors: the checkpoint {@code --dspark}
     * selects the draft of.
     */
    public static SyntheticHf.Builder deepseekV41TargetDspark() {
        return deepseekV41("DeepseekV41ForCausalLM");
    }

    private static SyntheticHf.Builder deepseekV41(String arch) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['" + arch + "'], 'model_type': 'deepseek_v41', 'hidden_size': " + V41_DIM + ","
                + " 'intermediate_size': 64, 'moe_intermediate_size': " + V41_MOE_FF + ", 'num_attention_heads': "
                + V41_HEADS + ", 'num_key_value_heads': 1, 'head_dim': " + V41_HEAD_DIM + ","
                + " 'num_hidden_layers': 2, 'num_nextn_predict_layers': 2, 'max_position_embeddings': 64,"
                + " 'rms_norm_eps': 1e-06, 'vocab_size': " + V41_VOCAB + ", 'rope_theta': 10000.0,"
                + " 'qk_rope_head_dim': 4, 'q_lora_rank': " + V41_Q_LORA + ", 'sliding_window': 16,"
                + " 'n_routed_experts': 2, 'num_experts_per_tok': 1, 'n_shared_experts': 1,"
                + " 'routed_scaling_factor': 1.5, 'norm_topk_prob': true, 'swiglu_limit': 10.0,"
                + " 'index_n_heads': 2, 'index_head_dim': 4, 'index_topk': 4, 'o_groups': 2, 'o_lora_rank': 8,"
                + " 'compress_ratios': [0, 4], 'compress_rope_theta': 160000.0, 'hc_mult': 2,"
                + " 'hc_sinkhorn_iters': 4, 'hc_eps': 1e-06, 'scoring_func': 'sqrtsoftplus',"
                + " 'kv_source_layer_ids': [0, 1], 'index_source_layer_ids': [1, 1],"
                + " 'candidate_source_layer_id': 0, 'candidate_topk_blocks': 4, 'candidate_block_size': 64,"
                + " 'dspark_block_size': 8, 'dspark_target_layer_ids': [0, 1], 'dspark_noise_token_id': 5,"
                + " 'dspark_n_routed_experts': 2, 'dspark_num_experts_per_tok': 1, 'gate_temp': 1.0}"));
        b.bf16("mtp.0.main_proj.weight", V41_DIM, V41_DIM);
        b.bf16("mtp.0.main_norm.weight", V41_DIM);
        b.bf16("mtp.0.norm.weight", V41_DIM);
        b.bf16("mtp.0.markov_head.embed.weight", V41_VOCAB, 8);
        b.bf16("mtp.0.markov_head.head.weight", V41_VOCAB, 8);
        b.bf16("mtp.0.confidence_head.proj.weight", 1, V41_DIM + 8);
        for (int s = 0; s < 2; s++) {
            String p = "mtp." + s + ".";
            b.bf16(p + "attn_norm.weight", V41_DIM);
            b.bf16(p + "ffn_norm.weight", V41_DIM);
            b.bf16(p + "attn.wq_a.weight", V41_Q_LORA, V41_DIM);
            b.bf16(p + "attn.wq_b.weight", V41_HEADS * V41_HEAD_DIM, V41_Q_LORA);
            b.bf16(p + "attn.q_norm.weight", V41_Q_LORA);
            b.bf16(p + "attn.wkv.weight", V41_DIM, V41_DIM);
            b.bf16(p + "attn.kv_norm.weight", V41_DIM);
            b.bf16(p + "attn.wo_a.weight", 8, V41_DIM);
            b.bf16(p + "attn.wo_b.weight", V41_DIM, 8);
            b.bf16(p + "attn.attn_sink", V41_HEADS);
            b.bf16(p + "hc_attn_fn", 4, 4);
            b.bf16(p + "hc_attn_base", 4);
            b.bf16(p + "hc_attn_scale", 1);
            b.bf16(p + "hc_ffn_fn", 4, 4);
            b.bf16(p + "hc_ffn_base", 4);
            b.bf16(p + "hc_ffn_scale", 1);
            b.bf16(p + "ffn.gate.weight", 2, V41_DIM);
            b.bf16(p + "ffn.shared_experts.w1.weight", V41_MOE_FF, V41_DIM);
            b.bf16(p + "ffn.shared_experts.w2.weight", V41_DIM, V41_MOE_FF);
            b.bf16(p + "ffn.shared_experts.w3.weight", V41_MOE_FF, V41_DIM);
            for (int e = 0; e < 2; e++) {
                b.bf16(p + "ffn.experts." + e + ".w1.weight", V41_MOE_FF, V41_DIM);
                b.bf16(p + "ffn.experts." + e + ".w2.weight", V41_DIM, V41_MOE_FF);
                b.bf16(p + "ffn.experts." + e + ".w3.weight", V41_MOE_FF, V41_DIM);
            }
        }
        return b;
    }

    private static final int ML4_DIM = 8;
    private static final int ML4_HEADS = 2;
    private static final int ML4_HEAD_DIM = 4;
    private static final int ML4_KV_LORA = 4;
    private static final int ML4_V_HEAD = 2;
    private static final int ML4_MOE_FF = 4;
    /** The tekken vocabulary of {@link #tekkenJson()}. */
    private static final int ML4_VOCAB = 277;

    /** {@code MistralLarge4DSparkModel}: one DSpark stage of the MLA backbone plus its Markov head. */
    public static SyntheticHf.Builder mistralLarge4Dspark() {
        return mistralLarge4("MistralLarge4DSparkModel");
    }

    /**
     * A {@code MistralLarge4ForConditionalGeneration} target carrying the same {@code mtp_dspark.*} tensors: the
     * checkpoint {@code --dspark} selects the draft of.
     */
    public static SyntheticHf.Builder mistralLarge4TargetDspark() {
        return mistralLarge4("MistralLarge4ForConditionalGeneration");
    }

    private static SyntheticHf.Builder mistralLarge4(String arch) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['" + arch + "'], 'dim': " + ML4_DIM + ", 'hidden_dim': 16, 'n_layers': 2,"
                + " 'n_heads': " + ML4_HEADS + ", 'n_kv_heads': 1, 'norm_eps': 1e-05, 'vocab_size': " + ML4_VOCAB
                + ", 'max_position_embeddings': 128, 'rope_theta': 10000.0, 'qk_rope_head_dim': 2,"
                + " 'qk_nope_head_dim': 2, 'v_head_dim': " + ML4_V_HEAD + ", 'kv_lora_rank': " + ML4_KV_LORA
                + ", 'nope': 'rope,nope', 'softmax_tanh_gating': 50.0,"
                + " 'moe': {'num_experts': 2, 'expert_hidden_dim': " + ML4_MOE_FF
                + ", 'num_experts_per_tok': 1, 'num_shared_experts': 1, 'route_every_n': 1,"
                + " 'first_k_dense_replace': 0, 'routed_scale': 1.0},"
                + " 'dsa': {'n_heads': 2, 'head_dim': 2, 'topk': 1, 'sharing': 'full,shared',"
                + " 'norm_eps': 1e-06, 'rope_head_dim': 2, 'match_nope': true, 'ssmax_scaling': true},"
                + " 'yarn': {'factor': 4.0, 'beta': 32.0, 'alpha': 1.0,"
                + " 'original_max_position_embeddings': 128, 'apply_scale': false},"
                + " 'mtp': {'n_layers': 1, 'depth': 8, 'ragged_attention': '8',"
                + " 'target_activation_layers': '1,2',"
                + " 'dspark': {'head_type': 'rnn', 'draft_input': 'position_specific'}}}"));
        b.file("tekken.json", tekkenJson());
        String p = "mtp_dspark.backbone.layers.0.";
        b.bf16(p + "attention_norm.weight", ML4_DIM);
        b.bf16(p + "ffn_norm.weight", ML4_DIM);
        b.bf16(p + "attention.wq.weight", ML4_HEADS * ML4_HEAD_DIM, ML4_DIM);
        b.bf16(p + "attention.wkv_a_with_mqa.weight", ML4_KV_LORA + 2, ML4_DIM);
        b.bf16(p + "attention.kv_a_norm.weight", ML4_KV_LORA);
        b.bf16(p + "attention.wkv_b.weight", ML4_HEADS * (2 + ML4_V_HEAD), ML4_KV_LORA);
        b.bf16(p + "attention.wo.weight", ML4_HEADS * ML4_V_HEAD, ML4_DIM);
        b.bf16(p + "feed_forward.gate.weight", 2, ML4_DIM);
        for (int e = 0; e < 2; e++) {
            b.bf16(p + "experts." + e + ".w1.weight", ML4_MOE_FF, ML4_DIM);
            b.bf16(p + "experts." + e + ".w2.weight", ML4_DIM, ML4_MOE_FF);
            b.bf16(p + "experts." + e + ".w3.weight", ML4_MOE_FF, ML4_DIM);
        }
        b.bf16(p + "shared_experts.w1.weight", ML4_MOE_FF, ML4_DIM);
        b.bf16(p + "shared_experts.w2.weight", ML4_DIM, ML4_MOE_FF);
        b.bf16(p + "shared_experts.w3.weight", ML4_MOE_FF, ML4_DIM);
        b.bf16("mtp_dspark.backbone.target_projection.weight", ML4_DIM, ML4_DIM);
        b.bf16("mtp_dspark.backbone.target_norm.weight", ML4_DIM);
        b.bf16("mtp_dspark.backbone.final_norm.weight", ML4_DIM);
        b.bf16("mtp_dspark.backbone.mask_embedding", 7, ML4_DIM);
        b.bf16("mtp_dspark.autoregressive_head.embedding.weight", ML4_VOCAB, 8);
        b.bf16("mtp_dspark.autoregressive_head.output.weight", ML4_VOCAB, 8);
        b.bf16("mtp_dspark.autoregressive_head.joint_proj.weight", 24, ML4_DIM + 16);
        b.bf16("mtp_dspark.autoregressive_head.joint_proj.bias", 24);
        return b;
    }

    /** A tekken v7 tokenizer: 20 special tokens, the 256 single bytes and the piece {@code ab}. */
    static String tekkenJson() {
        StringBuilder vocab = new StringBuilder();
        for (int i = 0; i < 257; i++) {
            byte[] bytes = i < 256 ? new byte[]{(byte) i} : "ab".getBytes(StandardCharsets.UTF_8);
            if (i > 0) {
                vocab.append(',');
            }
            vocab.append("{\"rank\":").append(i).append(",\"token_bytes\":\"")
                    .append(Base64.getEncoder().encodeToString(bytes)).append("\",\"token_str\":null}");
        }
        return "{\"config\":{\"pattern\":\"\\\\p{L}+|\\\\s+\",\"num_vocab_tokens\":257,"
                + "\"default_vocab_size\":277,\"default_num_special_tokens\":20,\"version\":\"v7\"},"
                + "\"vocab\":[" + vocab + "]}";
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

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(
                new Named("DeepseekV41DSparkModel", HfDraftFixtures::deepseekV41Dspark),
                new Named("MistralLarge4DSparkModel", HfDraftFixtures::mistralLarge4Dspark));
    }
}
