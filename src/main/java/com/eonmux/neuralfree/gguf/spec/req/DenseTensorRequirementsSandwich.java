package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;

import java.util.List;

/**
 * Tensor lists of the decoders with post-branch norms ({@code post_attention_norm}/{@code post_ffw_norm}):
 * the Gemma family, PLaMo-3, GLM-4, OLMo-2, EXAONE-4 and Muse Glimmer. Each method cites the llama.cpp
 * {@code load_arch_tensors} it transcribes (src/models/&lt;arch&gt;.cpp).
 */
public final class DenseTensorRequirementsSandwich {

    private DenseTensorRequirementsSandwich() {
    }

    /** gemma.cpp:13-35; the output head is always the duplicated {@code token_embd}. */
    public static List<TensorRequirement> gemma() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
                .build();
    }

    /** gemma2.cpp:30-54; the output head is always the duplicated {@code token_embd}. */
    public static List<TensorRequirement> gemma2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "post_attention_norm.weight", "ffn_norm.weight", "ffn_gate.weight",
                        "ffn_up.weight", "ffn_down.weight", "post_ffw_norm.weight")
                .build();
    }

    /** gemma3.cpp:34-71; optional {@code dense_2}/{@code dense_3} projections of embedding exports. */
    public static List<TensorRequirement> gemma3() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .modelOptional("dense_2.weight", "dense_3.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "post_attention_norm.weight", "attn_k_norm.weight",
                        "attn_q_norm.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                        "ffn_down.weight", "post_ffw_norm.weight")
                .build();
    }

    /**
     * gemma3n.cpp:21-76: the Gemma-3 block plus AltUp, LAuReL and per-layer embedding tensors.
     * {@code per_layer_model_proj} and {@code per_layer_proj_norm} carry no layer index.
     */
    public static List<TensorRequirement> gemma3n() {
        return DenseTensorRequirements.builder()
                .outputOrTied()
                .tokenEmbd()
                .model("altup_proj.weight", "altup_unembd_proj.weight", "per_layer_token_embd.weight",
                        "per_layer_model_proj.weight", "per_layer_proj_norm.weight", "output_norm.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                        "post_attention_norm.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                        "ffn_down.weight", "post_ffw_norm.weight", "inp_gate.weight", "proj.weight",
                        "post_norm.weight", "altup_correct_coef.weight", "altup_correct_scale.weight",
                        "altup_predict_coef.weight", "altup_router.weight", "altup_router_norm.weight",
                        "laurel_l.weight", "laurel_r.weight", "laurel_post_norm.weight")
                .build();
    }

    /**
     * plamo3.cpp:20-58. llama.cpp names the post norms without a suffix ({@code tn(LLM_TENSOR_ATTN_POST_NORM, i)},
     * plamo3.cpp:50 and :53) while the converter's tensor map emits {@code .weight}; either spelling satisfies
     * the entry. The FFN up projection is the fused gate+up (SwiGLU).
     */
    public static List<TensorRequirement> plamo3() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                        "attn_output.weight")
                .layerAnyOf(TensorRequirement.TRUNK, new String[] {"post_attention_norm"},
                        new String[] {"post_attention_norm.weight"})
                .layer("ffn_norm.weight")
                .layerAnyOf(TensorRequirement.TRUNK, new String[] {"post_ffw_norm"},
                        new String[] {"post_ffw_norm.weight"})
                .layer("ffn_up.weight", "ffn_down.weight")
                .build();
    }

    /**
     * glm4.cpp:15-62. NextN blocks are loaded with {@code TENSOR_SKIP}: their QKV projections become optional
     * (llama-model.cpp:3868-3880) while every other block tensor must still be present
     * (llama-model-loader.cpp:1345); the NextN head tensors follow at :50-60.
     */
    public static List<TensorRequirement> glm4() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layerWhen(DenseTensorRequirements.ALL_LAYERS, "attn_norm.weight")
                .qkvWhen(TensorRequirement.TRUNK)
                .qkvOptionalWhen(TensorRequirement.NEXTN)
                .layerWhen(DenseTensorRequirements.ALL_LAYERS, "attn_output.weight", "post_attention_norm.weight",
                        "ffn_norm.weight", "ffn_down.weight", "ffn_up.weight", "post_ffw_norm.weight")
                .layerWhen(TensorRequirement.NEXTN, "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight")
                .layerOptionalWhen(TensorRequirement.NEXTN, "nextn.embed_tokens.weight",
                        "nextn.shared_head_head.weight", "nextn.shared_head_norm.weight")
                .build();
    }

    /** olmo2.cpp:27-52; no pre-norms, the output head is required. */
    public static List<TensorRequirement> olmo2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output.weight")
                .qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                        "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                        "post_ffw_norm.weight")
                .build();
    }

    /**
     * exaone4.cpp:24-71; no pre-norms. NextN blocks are loaded with {@code TENSOR_SKIP} (see {@link #glm4()})
     * and carry the NextN head tensors of :64-69.
     */
    public static List<TensorRequirement> exaone4() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .qkvWhen(TensorRequirement.TRUNK)
                .qkvOptionalWhen(TensorRequirement.NEXTN)
                .layerWhen(DenseTensorRequirements.ALL_LAYERS, "attn_output.weight")
                .modelOptional("rope_freqs.weight")
                .layerWhen(DenseTensorRequirements.ALL_LAYERS, "post_attention_norm.weight", "attn_q_norm.weight",
                        "attn_k_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
                        "post_ffw_norm.weight")
                .layerWhen(TensorRequirement.NEXTN, "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight")
                .layerOptionalWhen(TensorRequirement.NEXTN, "nextn.shared_head_norm.weight")
                .build();
    }

    /** muse-glimmer.cpp:21-55; sandwich norms, Q/K norms and the elementwise sigmoid attention gate. */
    public static List<TensorRequirement> museGlimmer() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output.weight")
                .layer("attn_norm.weight", "post_attention_norm.weight")
                .qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
                        "ffn_norm.weight", "post_ffw_norm.weight", "ffn_gate.weight", "ffn_down.weight",
                        "ffn_up.weight")
                .build();
    }
}
