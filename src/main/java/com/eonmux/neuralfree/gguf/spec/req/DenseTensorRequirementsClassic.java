package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;

import java.util.List;

/**
 * Tensor lists of the LayerNorm-era and early RMSNorm decoders: the GPT-2, GPT-J, GPT-NeoX, Falcon, MPT, BLOOM,
 * StarCoder, Phi-2, Cohere, OLMo, JAIS, Nemotron, Chameleon, BitNet, Qwen-1, PLaMo, InternLM2, XVERSE and
 * Baichuan families. Each method cites the llama.cpp {@code load_arch_tensors} it transcribes
 * (src/models/&lt;arch&gt;.cpp).
 */
public final class DenseTensorRequirementsClassic {

    /** command-r loads per-head Q/K norms only for its 64-layer (35B+) variants (command-r.cpp:28-31). */
    private static final LayerCondition COMMAND_R_QK_NORM = (hp, il) -> il < hp.nLayer() && hp.nLayer() >= 64;

    /** Layers that mix earlier layer inputs (limite.cpp:62): a MUDD tap count and a first tap >= 0. */
    private static final LayerCondition LIMITE_MUDD =
            (hp, il) -> hp.muddTapCount() > 0 && hp.muddTapIdx(il, 0) >= 0;

    private DenseTensorRequirementsClassic() {
    }

    /**
     * limite.cpp:30-69. The value embeddings are a model-level table looked up by token id; the attention
     * output gates, the XSA alpha and the MUDD mixers are optional per layer; the learned residual scales are
     * required. The output head falls back to the duplicated {@code token_embd.weight} (:36-40).
     */
    public static List<TensorRequirement> limite() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("value_embd.weight")
                .outputOrTied()
                .qkv()
                .layer("attn_output.weight", "attn_resid_scale.weight", "ffn_resid_scale.weight")
                .layerOptional("attn_gate.weight", "attn_ve_gate.weight", "attn_xsa_alpha.weight")
                .gatedFfn()
                .layerWhen(LIMITE_MUDD, "mudd_down.weight", "mudd_up.weight")
                .layerOptionalWhen(LIMITE_MUDD, "mudd_up.bias")
                .build();
    }

    /** falcon.cpp:13-44; {@code attn_norm_2} exists on the 40B variant only. */
    public static List<TensorRequirement> falcon() {        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_norm.bias")
                .layerOptional("attn_norm_2.weight", "attn_norm_2.bias")
                .layer("attn_qkv.weight", "attn_output.weight", "ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /** gpt2.cpp:15-52. */
    public static List<TensorRequirement> gpt2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("position_embd.weight", "output_norm.weight", "output_norm.bias")
                .outputOrTied()
                .modelOptional("cls.output.weight", "cls.output.bias")
                .layer("attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                        "attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_down.weight", "ffn_down.bias", "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /** GPT-J uses separate, bias-free attention projections and biased MLP and output heads. */
    public static List<TensorRequirement> gptj() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias", "output.weight", "output.bias")
                .layer("attn_norm.weight", "attn_norm.bias", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                        "attn_output.weight", "ffn_down.weight", "ffn_down.bias", "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /**
     * OPT (vLLM opt.py, which has no llama.cpp counterpart): learned position rows, separate q/k/v projections and
     * an ungated MLP, all with optional biases. The pre-LN layers carry {@code attn_norm} and {@code ffn_norm}; the
     * 350M post-LN variant carries {@code attn_output_norm} and {@code layer_output_norm} instead. The decoder and
     * output projections exist only when {@code word_embed_proj_dim != hidden_size}; {@code output_norm} is absent
     * on post-LN files and when the checkpoint removed it, and the LM head may be tied.
     */
    public static List<TensorRequirement> opt() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("position_embd.weight")
                .modelOptional("project_in.weight", "project_out.weight", "output_norm.weight", "output_norm.bias")
                .outputOrTied()
                .layerAnyOf(TensorRequirement.TRUNK, new String[] {"attn_norm.weight"},
                        new String[] {"attn_output_norm.weight"})
                .layerAnyOf(TensorRequirement.TRUNK, new String[] {"ffn_norm.weight"},
                        new String[] {"layer_output_norm.weight"})
                .layerOptional("attn_norm.bias", "attn_output_norm.bias", "ffn_norm.bias", "layer_output_norm.bias")
                .layer("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                        "ffn_up.weight", "ffn_down.weight")
                .layerOptional("attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias",
                        "ffn_up.bias", "ffn_down.bias")
                .build();
    }

    /** gptneox.cpp:50-81; the output head is required. */
    public static List<TensorRequirement> gptneox() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias", "output.weight")
                .layer("attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                        "attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_down.weight", "ffn_down.bias", "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /**
     * vLLM's Step1 decoder: separate q/k/v projections (or a fused one) with optional biases, an optional
     * attention-output bias, and the biased gated SiLU feed-forward of {@code mlp_bias}.
     */
    public static List<TensorRequirement> step1() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layer("ffn_norm.weight")
                .gatedFfn()
                .layerOptional("ffn_gate.bias", "ffn_up.bias", "ffn_down.bias")
                .build();
    }

    /** mpt.cpp:15-61; every bias, the Q/K LayerNorms and {@code ffn.act.scales} are optional. */
    public static List<TensorRequirement> mpt() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .modelOptional("position_embd.weight")
                .model("output_norm.weight")
                .modelOptional("output_norm.bias")
                .outputOrTied()
                .layer("attn_norm.weight")
                .layerOptional("attn_norm.bias")
                .layer("attn_qkv.weight")
                .layerOptional("attn_qkv.bias")
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layer("ffn_norm.weight")
                .layerOptional("ffn_norm.bias")
                .layer("ffn_down.weight")
                .layerOptional("ffn_down.bias")
                .layer("ffn_up.weight")
                .layerOptional("ffn_up.bias", "attn_q_norm.weight", "attn_q_norm.bias", "attn_k_norm.weight",
                        "attn_k_norm.bias", "ffn.act.scales")
                .build();
    }

    /** starcoder.cpp:15-54. */
    public static List<TensorRequirement> starcoder() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("position_embd.weight", "output_norm.weight", "output_norm.bias")
                .outputOrTied()
                .modelOptional("cls.output.weight", "cls.output.bias")
                .layer("attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                        "attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_down.weight", "ffn_down.bias", "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /** bloom.cpp:21-59; {@code token_embd_norm} is the embedding LayerNorm. */
    public static List<TensorRequirement> bloom() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("token_embd_norm.weight", "token_embd_norm.bias", "output_norm.weight", "output_norm.bias")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                        "attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_down.weight", "ffn_down.bias", "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /** stablelm.cpp:14-45; the Q/K norms (12B) and the FFN norm are optional. */
    public static List<TensorRequirement> stablelm() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.bias", "output_norm.weight", "output.weight")
                .layer("attn_norm.weight", "attn_norm.bias")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight", "ffn_norm.bias")
                .gatedFfn()
                .build();
    }

    /** phi2.cpp:13-41; the output head carries a bias. */
    public static List<TensorRequirement> phi2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias", "output.weight", "output.bias")
                .layer("attn_norm.weight", "attn_norm.bias")
                .qkv()
                .layer("attn_output.weight", "attn_output.bias", "ffn_down.weight", "ffn_down.bias",
                        "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /** codeshell.cpp:12-47; {@code token_embd} is optional and falls back to the duplicated output head. */
    public static List<TensorRequirement> codeshell() {
        return DenseTensorRequirements.builder()
                .modelOptional("token_embd.weight")
                .model("output_norm.weight", "output_norm.bias", "output.weight")
                .layer("attn_norm.weight", "attn_norm.bias")
                .qkv()
                .layer("attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_down.weight", "ffn_down.bias", "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /** orion.cpp:12-37. */
    public static List<TensorRequirement> orion() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias", "output.weight")
                .layer("attn_norm.weight", "attn_norm.bias")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight", "ffn_norm.bias")
                .gatedFfn()
                .build();
    }

    /** starcoder2.cpp:16-53; the output and FFN biases are created with flags 0, hence required. */
    public static List<TensorRequirement> starcoder2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_norm.bias")
                .qkv()
                .layer("attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_down.weight", "ffn_up.weight", "ffn_down.bias", "ffn_up.bias")
                .build();
    }

    /** command-r.cpp:13-40; the output head is always the duplicated {@code token_embd}. */
    public static List<TensorRequirement> commandR() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .layer("attn_norm.weight")
                .layerWhen(COMMAND_R_QK_NORM, "attn_q_norm.weight", "attn_k_norm.weight")
                .qkv()
                .layer("attn_output.weight")
                .gatedFfn()
                .build();
    }

    /** cohere2.cpp:21-44; the output head is always the duplicated {@code token_embd}. */
    public static List<TensorRequirement> cohere2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight")
                .gatedFfn()
                .build();
    }

    /** olmo.cpp:15-37; the LayerNorms are parameter-free, so no norm tensors exist. */
    public static List<TensorRequirement> olmo() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .outputOrTied()
                .qkv()
                .layer("attn_output.weight")
                .gatedFfn()
                .build();
    }

    /** jais.cpp:15-49. */
    public static List<TensorRequirement> jais() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias", "output.weight")
                .layer("attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                        "attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_down.weight", "ffn_down.bias", "ffn_gate.weight", "ffn_gate.bias",
                        "ffn_up.weight", "ffn_up.bias")
                .build();
    }

    /** jais2.cpp:13-46; ungated MLP with biases. */
    public static List<TensorRequirement> jais2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_norm.bias")
                .qkv()
                .layer("attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias",
                        "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias")
                .build();
    }

    /** nemotron.cpp:12-44; the attention-output and MLP biases are optional. */
    public static List<TensorRequirement> nemotron() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output_norm.bias", "output.weight")
                .layer("attn_norm.weight", "attn_norm.bias")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layer("ffn_norm.weight", "ffn_norm.bias", "ffn_down.weight", "ffn_up.weight")
                .layerOptional("ffn_down.bias", "ffn_up.bias")
                .build();
    }

    /** chameleon.cpp:16-47; per-head Q/K LayerNorm weights, optional biases. */
    public static List<TensorRequirement> chameleon() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight")
                .layerOptional("attn_q_norm.bias", "attn_k_norm.bias")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight")
                .gatedFfn()
                .build();
    }

    /**
     * bitnet.cpp:12-45; separate projections with optional per-tensor {@code .scale} entries, sub-norms
     * before the output and down projections, and no output head (the graph reuses {@code token_embd}).
     */
    public static List<TensorRequirement> bitnet() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .layer("attn_norm.weight", "attn_sub_norm.weight", "attn_q.weight")
                .layerOptional("attn_q.scale")
                .layer("attn_k.weight")
                .layerOptional("attn_k.scale")
                .layer("attn_v.weight")
                .layerOptional("attn_v.scale")
                .layer("attn_output.weight")
                .layerOptional("attn_output.scale")
                .layer("ffn_norm.weight", "ffn_sub_norm.weight", "ffn_gate.weight")
                .layerOptional("ffn_gate.scale")
                .layer("ffn_down.weight")
                .layerOptional("ffn_down.scale")
                .layer("ffn_up.weight")
                .layerOptional("ffn_up.scale")
                .build();
    }

    /** qwen.cpp:13-37; fused QKV with a required bias. */
    public static List<TensorRequirement> qwen() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output.weight")
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_qkv.bias", "attn_output.weight",
                        "ffn_norm.weight")
                .gatedFfn()
                .build();
    }

    /** plamo.cpp:12-33; no FFN norm (the FFN reads the attention norm output). */
    public static List<TensorRequirement> plamo() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight")
                .gatedFfn()
                .build();
    }

    /**
     * InternLM2 (internlm2.cpp:13-35): the plain llama-1 layout with the optional {@code v_head} projection the
     * reward loader writes as {@code cls.output}, and no required LM head on a classifier checkpoint (vLLM's
     * {@code InternLM2ForRewardModel} loads no LM head).
     */
    public static List<TensorRequirement> internlm2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .modelWhen(hp -> hp.classifierScope() == null, "output.weight")
                .modelOptional("cls.output.weight", "cls.output.bias")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight")
                .gatedFfn()
                .build();
    }

    /**
     * The plain llama-1 layout with a required output head: internlm2.cpp:13-35, xverse.cpp:14-35,
     * baichuan.cpp:17-40.
     */
    public static List<TensorRequirement> llamaUntied() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight")
                .gatedFfn()
                .build();
    }
}
