package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;

import java.util.List;

import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ALL;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ATTENTION;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.RECURRENT;

/**
 * Tensor lists of the Gated DeltaNet hybrids (qwen3next, qwen35, qwen35moe, qwen4exp) and of the
 * lightning-attention hybrid minimax-01. The GDN recurrent layers name {@code ssm_a} without a suffix
 * ({@code LLM_TENSOR_SSM_A_NOSCAN}).
 */
public final class HybridTensorRequirementsGdn {

    /** The qwen4exp PLE layer (qwen4exp.cpp:266). */
    private static final LayerCondition PLE_LAYER = (hp, il) -> hp.isPleLayer(il);

    private HybridTensorRequirementsGdn() {
    }

    /**
     * qwen3next.cpp:42-128. A recurrent layer takes its Q/K/V and Z inputs from {@code attn_qkv} +
     * {@code attn_gate} or, in legacy files, from the fused {@code ssm_in}; llama.cpp creates all three as
     * optional and the graph uses whichever path exists (qwen3next.cpp:86-88, 322-333), so one of the two
     * groups is required here ({@code RecurrentResolver} picks the path into {@code RecurrentSpec.legacySsmIn()}).
     * Each NextN block is a trunk block plus the NextN fusion (qwen3next.cpp:108-121).
     */
    public static List<TensorRequirement> qwen3next() {
        HybridTensorRequirements.Builder b = header();
        qwen3nextBlock(b);
        b.nextn();
        qwen3nextBlock(b);
        b.nextnFusion();
        return b.build();
    }

    /**
     * qwen35.cpp:40-128. The recurrent {@code attn_qkv} and {@code attn_gate} are created optional
     * (qwen35.cpp:81-82) but the GDN graph multiplies by both unconditionally (qwen35.cpp:245, 249), so they are
     * required here. The NextN block is always a full-attention block (qwen35.cpp:97-120).
     */
    public static List<TensorRequirement> qwen35() {
        HybridTensorRequirements.Builder b = header();
        b.modelOptional("cls.output.weight", "cls.output.bias");
        qwenAttentionNorms(b);
        qwenAttention(b, ATTENTION);
        qwen35Recurrent(b);
        b.gatedFfn(ALL);
        b.nextn();
        qwenAttentionNorms(b);
        qwenAttention(b, ALL);
        b.gatedFfn(ALL);
        b.nextnFusion();
        return b.build();
    }

    /** qwen35moe.cpp:43-139; recurrent inputs as {@link #qwen35()} (qwen35moe.cpp:83-84, 255, 259). */
    public static List<TensorRequirement> qwen35moe() {
        HybridTensorRequirements.Builder b = header();
        qwenAttentionNorms(b);
        qwenAttention(b, ATTENTION);
        qwen35Recurrent(b);
        qwenMoe(b);
        b.nextn();
        qwenAttentionNorms(b);
        qwenAttention(b, ALL);
        qwenMoe(b);
        b.nextnFusion();
        return b.build();
    }

    /**
     * qwen4exp.cpp:182-301. There is no {@code output_norm}: the final hyper-connection mixer
     * ({@code output_hc_*}, created with the trunk flags, :186-188) carries it, and every block mixes its residual
     * streams through
     * {@code hc_attn_*} / {@code hc_ffn_*}. Attention layers carry the QSA indexer; the PLE layer carries the
     * n-gram embedding module, whose {@code per_layer_token_embd} table is required when the PLE metadata
     * declares heads ({@code ple_n_heads > 0}, qwen4exp.cpp:196-215).
     */
    public static List<TensorRequirement> qwen4exp() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight");
        b.modelTrunk("output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight");
        b.outputOrTied();
        b.modelWhen(hp -> hp.pleNHeads() > 0, "per_layer_token_embd.weight");
        qwen4expBlock(b);
        b.nextn();
        qwen4expBlock(b);
        b.layer(ALL, "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                "nextn.hc_head_norm.weight", "nextn.hc_head_down.weight", "nextn.hc_head_up.weight");
        return b.build();
    }

    /**
     * minimax-01.cpp:29-58. Lightning layers carry {@code attn_norm_2} and the fused {@code attn_qkv} with an
     * output gate; softmax layers use {@code create_tensor_qkv}.
     */
    public static List<TensorRequirement> minimax01() {
        HybridTensorRequirements.Builder b = header();
        b.layer(ALL, "attn_norm.weight");
        b.qkv(ATTENTION);
        b.layer(RECURRENT, "attn_norm_2.weight", "attn_qkv.weight", "attn_gate.weight");
        b.layer(ALL, "attn_output.weight", "ffn_norm.weight", "ffn_gate_inp.weight");
        b.layerOptional(ALL, "ffn_gate_exps.weight");
        b.layer(ALL, "ffn_down_exps.weight", "ffn_up_exps.weight");
        return b.build();
    }

    /** qwen3next.cpp:68-105 ({@code load_block_trunk}). */
    private static void qwen3nextBlock(HybridTensorRequirements.Builder b) {
        qwenAttentionNorms(b);
        qwenAttention(b, ATTENTION);
        b.layerAnyOf(RECURRENT, new String[] {"attn_qkv.weight", "attn_gate.weight"}, new String[] {"ssm_in.weight"});
        b.layer(RECURRENT, "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_ba.weight", "ssm_norm.weight",
                "ssm_out.weight");
        qwenMoe(b);
    }

    /** {@code attn_norm} and the pre-FFN {@code post_attention_norm} of every Qwen GDN block. */
    private static void qwenAttentionNorms(HybridTensorRequirements.Builder b) {
        b.layer(ALL, "attn_norm.weight", "post_attention_norm.weight");
    }

    /** The gated full attention: Q (with gate) / K / V, output, per-head Q/K norms. */
    private static void qwenAttention(HybridTensorRequirements.Builder b, LayerCondition when) {
        b.qkv(when);
        b.layer(when, "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight");
    }

    /** The qwen35-style GDN mixer with separate beta / alpha projections (qwen35.cpp:81-89). */
    private static void qwen35Recurrent(HybridTensorRequirements.Builder b) {
        b.layer(RECURRENT, "attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a",
                "ssm_beta.weight", "ssm_alpha.weight", "ssm_norm.weight", "ssm_out.weight");
    }

    /** The Qwen routed experts plus the sigmoid-gated shared expert (qwen3next.cpp:96-104). */
    private static void qwenMoe(HybridTensorRequirements.Builder b) {
        b.layer(ALL, "ffn_gate_inp.weight", "ffn_down_exps.weight");
        b.gateUpExps(ALL);
        b.layer(ALL, "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                "ffn_down_shexp.weight");
    }

    /** qwen4exp.cpp:217-283 ({@code load_block}). */
    private static void qwen4expBlock(HybridTensorRequirements.Builder b) {
        b.layer(ALL, "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight");
        qwenAttention(b, ATTENTION);
        b.layer(ATTENTION, "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight",
                "indexer.k_norm.weight");
        qwen35Recurrent(b);
        b.layer(PLE_LAYER, "ple_key.weight", "ple_value.weight", "ple_norm_key.weight", "ple_norm_query.weight",
                "ple_norm_conv.weight", "ple_conv1d.weight");
        qwenMoe(b);
    }

    /** {@code token_embd}, {@code output_norm} and the untied-or-tied output head. */
    private static HybridTensorRequirements.Builder header() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight");
        return b.outputOrTied();
    }
}
