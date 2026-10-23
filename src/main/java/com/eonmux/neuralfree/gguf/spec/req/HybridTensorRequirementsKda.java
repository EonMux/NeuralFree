package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;

import java.util.List;

import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.AFTER_DENSE_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ALL;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ATTENTION;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.DENSE_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.RECURRENT;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.and;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.not;

/**
 * Tensor lists of the Kimi delta attention (KDA) + MLA hybrids: kimi-linear, kimi-k3, bailingmoe3 and
 * glm5-next. KDA layers hold depthwise Q/K/V convolutions ({@code ssm_conv1d_q/k/v}, created 4-D or, after
 * quantization, 3-D under the same name), the decay gate ({@code ssm_f_a} [+ {@code ssm_f_b}],
 * {@code ssm_dt.bias}, {@code ssm_a}), {@code ssm_beta}, the output gate and the gated output norm.
 */
public final class HybridTensorRequirementsKda {

    /** Cross-layer attention residuals enabled: {@code attn_res_block_size > 0} (kimi-k3.cpp:73). */
    private static final LayerCondition ATTN_RES = (hp, il) -> hp.attnResBlockSize() > 0;
    /** Latent routed experts: {@code n_expert_latent > 0} (kimi-k3.cpp:154). */
    private static final LayerCondition LATENT_EXPERTS = (hp, il) -> hp.nExpertLatent() > 0;
    /** Low-rank MLA query: {@code q_lora_rank > 0} (bailingmoe3.cpp:90). */
    private static final LayerCondition Q_LORA = (hp, il) -> hp.nLoraQ() > 0;
    /** glm5-next layers running the full indexer (glm5-next.cpp:142). */
    private static final LayerCondition INDEXER_FULL = (hp, il) -> hp.isIndexerFull(il);

    /** The glm5-next k-pool indexer tensors (glm5-next.cpp:146-152). */
    private static final String[] GLM5_INDEXER = {
        "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight", "indexer.attn_k.weight",
        "indexer.attn_q_b.weight", "indexer_compressor_gate.weight", "indexer_compressor_ape.weight"};

    private HybridTensorRequirementsKda() {
    }

    /** kimi-linear.cpp:37-166. */
    public static List<TensorRequirement> kimiLinear() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight", "output.weight");
        b.layer(ALL, "attn_norm.weight");
        kdaConvAndQkv(b);
        b.layer(RECURRENT, "ssm_f_a.weight", "ssm_f_b.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias",
                "ssm_g_a.weight", "ssm_g_b.weight", "ssm_norm.weight", "attn_output.weight");
        b.layer(ATTENTION, "attn_kv_a_norm.weight");
        probedMla(b);
        b.layer(ATTENTION, "attn_output.weight");
        b.layer(ALL, "ffn_norm.weight");
        b.gatedFfn(DENSE_LEAD);
        b.layer(AFTER_DENSE_LEAD, "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight",
                "ffn_up_exps.weight");
        b.layerOptional(AFTER_DENSE_LEAD, "ffn_gate_shexp.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight");
        b.layer(AFTER_DENSE_LEAD, "exp_probs_b.bias");
        return b.build();
    }

    /**
     * kimi-k3.cpp:58-165. {@code output_res_score} is required with cross-layer attention residuals
     * ({@code attn_res_block_size > 0}, kimi-k3.cpp:63-65).
     */
    public static List<TensorRequirement> kimiK3() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight", "output.weight");
        b.modelWhen(hp -> hp.attnResBlockSize() > 0, "output_res_score.weight");
        b.layer(ALL, "attn_norm.weight", "ffn_norm.weight");
        b.layer(ATTN_RES, "attn_res_score.weight", "ffn_res_score.weight");
        kdaConvAndQkv(b);
        b.layer(RECURRENT, "ssm_f_a.weight", "ssm_f_b.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias",
                "ssm_g.weight", "ssm_norm.weight", "attn_output.weight");
        b.layer(ATTENTION, "attn_kv_a_norm.weight");
        probedMla(b);
        b.layerOptional(ATTENTION, "attn_gate.weight");
        b.layer(ATTENTION, "attn_output.weight");
        b.gatedFfn(DENSE_LEAD);
        b.layer(AFTER_DENSE_LEAD, "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
                "ffn_down_exps.weight", "ffn_up_exps.weight");
        LayerCondition latent = and(AFTER_DENSE_LEAD, LATENT_EXPERTS);
        b.layer(latent, "ffn_routed_down.weight", "ffn_routed_up.weight");
        b.layerOptional(latent, "ffn_routed_norm.weight");
        b.layerOptional(AFTER_DENSE_LEAD, "ffn_gate_shexp.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight");
        return b.build();
    }

    /**
     * bailingmoe3.cpp:50-153. The KDA decay gate is the full-rank {@code ssm_f_a} (no {@code ssm_f_b}) and the
     * output gate the full-rank {@code ssm_g_a}. Each NextN block is an MLA + MoE block closed by
     * {@code layer_output_norm} (bailingmoe3.cpp:122-152).
     */
    public static List<TensorRequirement> bailingmoe3() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight");
        b.outputOrTied();
        b.layer(ALL, "attn_norm.weight");
        b.layer(RECURRENT, "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight");
        b.qkv(RECURRENT);
        b.layer(RECURRENT, "ssm_f_a.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias", "ssm_g_a.weight",
                "ssm_norm.weight", "attn_output.weight");
        bailingMla(b, ATTENTION);
        b.layer(ALL, "ffn_norm.weight");
        b.gatedFfn(DENSE_LEAD);
        bailingMoe(b, AFTER_DENSE_LEAD);

        b.nextn();
        b.layer(ALL, "attn_norm.weight");
        bailingMla(b, ALL);
        b.layer(ALL, "ffn_norm.weight");
        bailingMoe(b, ALL);
        b.layer(ALL, "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                "layer_output_norm.weight");
        return b.build();
    }

    /**
     * glm5-next.cpp:71-183. Trunk blocks carry the mHC mixers; attention blocks carry the k-pool indexer,
     * required on full-indexer layers and optional elsewhere (glm5-next.cpp:142-152).
     */
    public static List<TensorRequirement> glm5Next() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight", "output.weight");
        glm5NextBlock(b);
        b.layer(ALL, "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight", "hc_ffn_fn.weight",
                "hc_ffn_base.weight", "hc_ffn_scale.weight");
        b.layer(and(ATTENTION, INDEXER_FULL), GLM5_INDEXER);
        b.layerOptional(and(ATTENTION, not(INDEXER_FULL)), GLM5_INDEXER);
        b.nextn();
        glm5NextBlock(b);
        b.layer(ATTENTION, GLM5_INDEXER);
        b.nextnFusion();
        return b.build();
    }

    /** The glm5-next block body shared by trunk and NextN layers (glm5-next.cpp:82-173). */
    private static void glm5NextBlock(HybridTensorRequirements.Builder b) {
        b.layer(ALL, "attn_norm.weight", "ffn_norm.weight");
        kdaConvAndQkv(b);
        b.layer(RECURRENT, "ssm_f_a.weight", "ssm_f_b.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias",
                "ssm_g_a.weight", "ssm_g_b.weight", "ssm_norm.weight", "attn_output.weight");
        b.layer(ATTENTION, "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_q_a.weight", "attn_q_b.weight",
                "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight");
        b.gatedFfn(DENSE_LEAD);
        b.layer(AFTER_DENSE_LEAD, "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
                "ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_gate_shexp.weight", "ffn_down_shexp.weight",
                "ffn_up_shexp.weight");
    }

    /** The KDA Q/K/V convolutions and {@code create_tensor_qkv} projections of recurrent layers. */
    private static void kdaConvAndQkv(HybridTensorRequirements.Builder b) {
        b.layer(RECURRENT, "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight");
        b.qkv(RECURRENT);
    }

    /**
     * The kimi MLA projections chosen by probing (kimi-linear.cpp:112-133, kimi-k3.cpp:114-131): low-rank Q
     * when {@code attn_q_a_norm} exists, else {@code attn_q}; {@code attn_kv_b}, else the split
     * {@code attn_k_b} / {@code attn_v_b}.
     */
    private static void probedMla(HybridTensorRequirements.Builder b) {
        b.layerFirstOf(ATTENTION, new String[] {"attn_q_a_norm.weight", "attn_q_a.weight", "attn_q_b.weight"},
                new String[] {"attn_q.weight"});
        b.layer(ATTENTION, "attn_kv_a_mqa.weight");
        b.layerFirstOf(ATTENTION, new String[] {"attn_kv_b.weight"},
                new String[] {"attn_k_b.weight", "attn_v_b.weight"});
    }

    /** bailingmoe3 MLA with output gate (bailingmoe3.cpp:90-102, 127-139). */
    private static void bailingMla(HybridTensorRequirements.Builder b, LayerCondition when) {
        LayerCondition qLora = and(when, Q_LORA);
        b.layer(qLora, "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight");
        b.layer(and(when, not(Q_LORA)), "attn_q.weight");
        b.layer(when, "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight",
                "attn_gate.weight", "attn_output.weight");
    }

    /** bailingmoe3 routed and shared experts (bailingmoe3.cpp:111-118, 141-148). */
    private static void bailingMoe(HybridTensorRequirements.Builder b, LayerCondition when) {
        b.layer(when, "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight");
    }
}
