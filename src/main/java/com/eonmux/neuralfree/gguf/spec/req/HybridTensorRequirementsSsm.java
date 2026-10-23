package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;

import java.util.List;

import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ALL;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ATTENTION;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.RECURRENT;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.and;

/**
 * Tensor lists of the Mamba-family architectures: mamba, mamba2, jamba, falcon-h1, plamo2, granitehybrid
 * and nemotron_h / nemotron_h_moe. {@code ssm_a}, {@code ssm_d} and some PLaMo2 / Falcon-H1 norms are
 * created without a {@code .weight} suffix; the names below keep that.
 */
public final class HybridTensorRequirementsSsm {

    /** Layers whose FFN routes experts ({@code n_expert > 0} branches). */
    private static final LayerCondition MOE = (hp, il) -> hp.nExpert() > 0;
    /** Layers whose FFN is dense ({@code n_expert == 0} branches). */
    private static final LayerCondition DENSE = (hp, il) -> hp.nExpert() == 0;
    /** MoE layers with a shared expert ({@code n_ff_shexp > 0}). */
    private static final LayerCondition MOE_SHARED = (hp, il) -> hp.nExpert() > 0 && hp.nFfShexp() > 0;
    /** Nemotron-H attention layers: KV heads and no FFN (nemotron-h.cpp:102). */
    private static final LayerCondition NEMOTRON_ATTENTION = (hp, il) -> !hp.isRecurrent(il) && hp.nFf(il) == 0;
    /** Nemotron-H FFN layers (nemotron-h.cpp:103-111). */
    private static final LayerCondition NEMOTRON_FFN = (hp, il) -> !hp.isRecurrent(il) && hp.nFf(il) != 0;

    private HybridTensorRequirementsSsm() {
    }

    /** mamba.cpp:47-80. */
    public static List<TensorRequirement> mamba() {
        HybridTensorRequirements.Builder b = header();
        b.layer(ALL, "attn_norm.weight", "ssm_in.weight", "ssm_conv1d.weight", "ssm_conv1d.bias", "ssm_x.weight",
                "ssm_dt.weight", "ssm_dt.bias", "ssm_a", "ssm_d", "ssm_out.weight");
        return b.build();
    }

    /** mamba2.cpp:49-83. */
    public static List<TensorRequirement> mamba2() {
        HybridTensorRequirements.Builder b = header();
        b.layer(ALL, "attn_norm.weight", "ssm_in.weight", "ssm_conv1d.weight", "ssm_conv1d.bias", "ssm_dt.bias",
                "ssm_a", "ssm_d", "ssm_norm.weight", "ssm_out.weight");
        return b.build();
    }

    /**
     * jamba.cpp:33-100. A layer's FFN is routed when it has {@code ffn_gate_inp} and dense otherwise
     * (jamba.cpp:88-99), which {@code MoeResolver} applies per layer ({@code MoeSpec.isMoeLayer}).
     */
    public static List<TensorRequirement> jamba() {
        HybridTensorRequirements.Builder b = header();
        b.layer(ALL, "attn_norm.weight");
        b.layer(RECURRENT, "ssm_in.weight", "ssm_conv1d.weight", "ssm_conv1d.bias", "ssm_x.weight",
                "ssm_dt_norm.weight", "ssm_dt.weight", "ssm_dt.bias", "ssm_b_norm.weight", "ssm_c_norm.weight",
                "ssm_a", "ssm_d", "ssm_out.weight");
        b.qkv(ATTENTION);
        b.layer(ATTENTION, "attn_output.weight");
        b.layer(ALL, "ffn_norm.weight");
        b.layerFirstOf(ALL,
                new String[] {"ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight",
                        "ffn_up_exps.weight"},
                new String[] {"ffn_gate.weight", "ffn_down.weight", "ffn_up.weight"});
        return b.build();
    }

    /**
     * falcon-h1.cpp:57-103. Every block holds both mixers; {@code ffn_norm} is created without a suffix
     * (falcon-h1.cpp:96).
     */
    public static List<TensorRequirement> falconH1() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight");
        b.outputOrTied();
        b.model("output_norm.weight");
        b.layer(ALL, "ssm_in.weight", "ssm_conv1d.weight");
        b.layerOptional(ALL, "ssm_conv1d.bias");
        b.layer(ALL, "ssm_dt.bias", "ssm_a", "ssm_d");
        b.layerOptional(ALL, "ssm_norm.weight");
        b.layer(ALL, "ssm_out.weight");
        b.qkv(ALL);
        b.layer(ALL, "attn_output.weight");
        b.layerOptional(ALL, "attn_output.bias");
        b.layer(ALL, "attn_norm.weight", "ffn_norm");
        b.modelOptional("rope_freqs.weight");
        b.gatedFfn(ALL);
        b.layerOptional(ALL, "ffn_gate.bias", "ffn_down.bias", "ffn_up.bias");
        return b.build();
    }

    /**
     * plamo2.cpp:49-103. The Mamba dt/B/C norms and the post norms are created without a suffix
     * (plamo2.cpp:78-80, 98, 102); {@code ffn_up} holds the fused gate and up halves.
     */
    public static List<TensorRequirement> plamo2() {
        HybridTensorRequirements.Builder b = header();
        b.layer(ALL, "attn_norm.weight");
        b.layer(RECURRENT, "ssm_in.weight", "ssm_conv1d.weight", "ssm_x.weight", "ssm_dt.weight", "ssm_dt.bias",
                "ssm_a", "ssm_d", "ssm_out.weight", "ssm_dt_norm", "ssm_b_norm", "ssm_c_norm");
        b.layer(ATTENTION, "attn_qkv.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight");
        b.layer(ALL, "post_attention_norm", "ffn_norm.weight", "ffn_down.weight", "ffn_up.weight", "post_ffw_norm");
        return b.build();
    }

    /** granite-hybrid.cpp:56-125. */
    public static List<TensorRequirement> graniteHybrid() {
        HybridTensorRequirements.Builder b = header();
        b.layer(ALL, "attn_norm.weight");
        mamba2Mixer(b);
        b.qkv(ATTENTION);
        b.layer(ATTENTION, "attn_output.weight");
        b.layerOptional(ATTENTION, "attn_output.bias");
        b.layer(ALL, "ffn_norm.weight");
        b.modelOptional("rope_freqs.weight");
        b.layer(MOE, "ffn_gate_inp.weight");
        b.layerOptional(MOE, "ffn_gate_exps.weight");
        b.layer(MOE, "ffn_down_exps.weight", "ffn_up_exps.weight");
        b.layer(MOE_SHARED, "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight");
        b.gatedFfn(DENSE);
        b.layerOptional(DENSE, "ffn_gate.bias", "ffn_down.bias", "ffn_up.bias");
        return b.build();
    }

    /**
     * nemotron-h.cpp:66-190 (nemotron_h and nemotron_h_moe). Trunk layers are Mamba-2, attention or FFN
     * layers; each NextN block folds an attention and a MoE sub-layer (nemotron-h.cpp:149-190).
     */
    public static List<TensorRequirement> nemotronH() {
        HybridTensorRequirements.Builder b = header();
        b.layer(ALL, "attn_norm.weight");
        mamba2Mixer(b);
        b.qkv(NEMOTRON_ATTENTION);
        b.layer(NEMOTRON_ATTENTION, "attn_output.weight");
        b.layerOptional(NEMOTRON_ATTENTION, "attn_output.bias");
        LayerCondition moe = and(NEMOTRON_FFN, MOE);
        b.layer(moe, "ffn_gate_inp.weight", "exp_probs_b.bias");
        b.layerOptional(moe, "ffn_latent_down.weight", "ffn_latent_up.weight");
        b.layer(moe, "ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight");
        LayerCondition mlp = and(NEMOTRON_FFN, DENSE);
        b.layer(mlp, "ffn_down.weight", "ffn_up.weight");
        b.layerOptional(mlp, "ffn_down.bias", "ffn_up.bias");

        b.nextn();
        b.layer(ALL, "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.eh_proj.weight",
                "nextn.shared_head_norm.weight", "attn_norm.weight");
        b.qkv(ALL);
        b.layer(ALL, "attn_output.weight");
        b.layerOptional(ALL, "attn_output.bias");
        b.layer(ALL, "post_attention_norm.weight", "ffn_gate_inp.weight", "exp_probs_b.bias");
        b.layerOptional(ALL, "ffn_latent_down.weight", "ffn_latent_up.weight");
        b.layer(ALL, "ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight");
        b.layerOptional(ALL, "ffn_down_exps.scale", "ffn_up_exps.scale", "ffn_down_exps.input_scale",
                "ffn_up_exps.input_scale", "ffn_down_shexp.scale", "ffn_up_shexp.scale",
                "ffn_down_shexp.input_scale", "ffn_up_shexp.input_scale");
        return b.build();
    }

    /** {@code token_embd}, {@code output_norm} and the untied-or-tied output head. */
    private static HybridTensorRequirements.Builder header() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight");
        return b.outputOrTied();
    }

    /**
     * The Mamba-2 mixer of granitehybrid and nemotron_h recurrent layers: optional conv bias, required
     * grouped {@code ssm_norm} (granite-hybrid.cpp:76-91, nemotron-h.cpp:85-101).
     */
    private static void mamba2Mixer(HybridTensorRequirements.Builder b) {
        b.layer(RECURRENT, "ssm_in.weight", "ssm_conv1d.weight");
        b.layerOptional(RECURRENT, "ssm_conv1d.bias");
        b.layer(RECURRENT, "ssm_dt.bias", "ssm_a", "ssm_d", "ssm_norm.weight", "ssm_out.weight");
    }
}
