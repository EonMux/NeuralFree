package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

/**
 * A synthetic {@code DeepseekV4ForCausalLM} checkpoint whose checkpoint carries the trunk's routed experts and one
 * {@code mtp.0} stage: the target of the {@code --mtp} and {@code --no-nextn} export modes.
 */
public final class HfMtpFixtures {

    private HfMtpFixtures() {
    }

    private static final int DIM = 16;
    private static final int VOCAB = 16;
    private static final int HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int Q_LORA = 8;
    private static final int MOE_FF = 8;
    private static final int EXPERTS = 2;
    private static final int MAIN_LAYERS = 2;

    /**
     * Two trunk blocks with their routed experts and one NextN stage ({@code mtp.0}), with the shared embedding,
     * final norm and output head. The trunk's other tensors are absent: a {@code --mtp} export drops them anyway.
     */
    public static SyntheticHf.Builder deepseekV4WithMtp() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['DeepseekV4ForCausalLM'], 'hidden_size': " + DIM + ", 'intermediate_size': 32,"
                + " 'moe_intermediate_size': " + MOE_FF + ", 'num_attention_heads': " + HEADS + ","
                + " 'num_key_value_heads': 1, 'head_dim': " + HEAD_DIM + ", 'num_hidden_layers': " + MAIN_LAYERS
                + ", 'num_nextn_predict_layers': 1, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                + " 'vocab_size': " + VOCAB + ", 'rope_theta': 10000.0, 'qk_rope_head_dim': 4,"
                + " 'q_lora_rank': " + Q_LORA + ", 'sliding_window': 16, 'n_routed_experts': " + EXPERTS + ","
                + " 'num_experts_per_tok': 1, 'n_shared_experts': 1, 'routed_scaling_factor': 1.5,"
                + " 'norm_topk_prob': true, 'swiglu_limit': 10.0, 'index_n_heads': 2, 'index_head_dim': 4,"
                + " 'index_topk': 4, 'o_groups': 2, 'o_lora_rank': " + Q_LORA + ","
                + " 'compress_ratios': [0, 4, 16], 'compress_rope_theta': 160000.0, 'hc_mult': 2,"
                + " 'hc_sinkhorn_iters': 4, 'hc_eps': 1e-06, 'num_hash_layers': 0,"
                + " 'scoring_func': 'sqrtsoftplus'}"));
        b.bf16("embed.weight", VOCAB, DIM);
        b.bf16("norm.weight", DIM);
        b.bf16("head.weight", VOCAB, DIM);
        for (int bid = 0; bid < MAIN_LAYERS; bid++) {
            experts(b, "layers." + bid + ".ffn.experts.");
        }
        b.bf16("mtp.0.e_proj.weight", DIM, DIM);
        b.bf16("mtp.0.h_proj.weight", DIM, DIM);
        b.bf16("mtp.0.enorm.weight", DIM);
        b.bf16("mtp.0.hnorm.weight", DIM);
        b.bf16("mtp.0.norm.weight", DIM);
        b.bf16("mtp.0.attn_norm.weight", DIM);
        b.bf16("mtp.0.ffn_norm.weight", DIM);
        b.bf16("mtp.0.attn.wq_a.weight", Q_LORA, DIM);
        b.bf16("mtp.0.attn.q_norm.weight", Q_LORA);
        b.bf16("mtp.0.attn.wq_b.weight", HEADS * HEAD_DIM, Q_LORA);
        b.bf16("mtp.0.attn.wkv.weight", DIM, DIM);
        b.bf16("mtp.0.attn.kv_norm.weight", DIM);
        b.bf16("mtp.0.attn.wo_a.weight", Q_LORA, DIM);
        b.bf16("mtp.0.attn.wo_b.weight", DIM, Q_LORA);
        b.bf16("mtp.0.attn.attn_sink", HEADS);
        b.bf16("mtp.0.hc_attn_fn", 4, 4);
        b.bf16("mtp.0.hc_attn_base", 4);
        b.bf16("mtp.0.hc_attn_scale", 1);
        b.bf16("mtp.0.hc_ffn_fn", 4, 4);
        b.bf16("mtp.0.hc_ffn_base", 4);
        b.bf16("mtp.0.hc_ffn_scale", 1);
        b.bf16("mtp.0.ffn.gate.weight", EXPERTS, DIM);
        b.bf16("mtp.0.ffn.shared_experts.w1.weight", MOE_FF, DIM);
        b.bf16("mtp.0.ffn.shared_experts.w2.weight", DIM, MOE_FF);
        b.bf16("mtp.0.ffn.shared_experts.w3.weight", MOE_FF, DIM);
        experts(b, "mtp.0.ffn.experts.");
        return b;
    }

    private static void experts(SyntheticHf.Builder b, String prefix) {
        for (int e = 0; e < EXPERTS; e++) {
            b.bf16(prefix + e + ".w1.weight", MOE_FF, DIM);
            b.bf16(prefix + e + ".w2.weight", DIM, MOE_FF);
            b.bf16(prefix + e + ".w3.weight", MOE_FF, DIM);
        }
    }
}
