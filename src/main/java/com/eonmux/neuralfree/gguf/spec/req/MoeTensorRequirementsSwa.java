package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;

import java.util.List;

import static com.eonmux.neuralfree.gguf.arch.TensorRequirement.NEXTN;
import static com.eonmux.neuralfree.gguf.arch.TensorRequirement.TRUNK;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.AFTER_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.DENSE_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.NO_EXPERTS;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.WITH_EXPERTS;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.and;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.builder;

/**
 * Tensor lists of the sliding-window MoE architectures, referenced from
 * {@link com.eonmux.neuralfree.gguf.arch.ArchTableMoeSwa}. Each method transcribes {@code load_arch_tensors}
 * of llama.cpp src/models/&lt;file&gt;.cpp (cited {@code models/<file>.cpp:lines}); conventions in
 * {@link MoeTensorRequirementsSupport}.
 */
public final class MoeTensorRequirementsSwa {

    private static final LayerCondition SHEXP_BY_COUNT = (hp, il) -> hp.nExpertShared() > 0;
    private static final LayerCondition SHEXP_BY_FF = (hp, il) -> hp.nFfShexp() > 0;

    private MoeTensorRequirementsSwa() {
    }

    /** models/flint.cpp:20-69: QK norms and attention sinks; dense lead, then biased routed plus shared experts. */
    public static List<TensorRequirement> flint() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight", "attn_sinks.weight",
                        "ffn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .build();
    }

    /**
     * models/gemma4.cpp:37-140. The fused {@code attn_qkv} is probed first and the separate projections are created
     * only without it (:79-85); layers without their own KV cache need no {@code attn_k}/{@code attn_k_norm}; the
     * proportional RoPE factors {@code rope_freqs} come with the first full-attention layer (:97-101). The
     * expert branch (:118-135) exists on layers whose {@code ffn_gate_inp} is present, and its tensors are then
     * required; the per-layer-embedding model tensors exist when {@code embedding_length_per_layer_input > 0}
     * (:56-60).
     */
    public static List<TensorRequirement> gemma4() {
        LayerCondition kv = (hp, il) -> il < hp.nLayer() && hp.hasKv(il);
        LayerCondition noKv = (hp, il) -> il < hp.nLayer() && !hp.hasKv(il);
        LayerCondition ple = (hp, il) -> il < hp.nLayer() && hp.nEmbdPerLayer() > 0;
        return builder().outputOrTied().model("token_embd.weight")
                .modelWhen(hp -> hp.nEmbdPerLayer() > 0,
                        "per_layer_token_embd.weight", "per_layer_model_proj.weight", "per_layer_proj_norm.weight")
                .model("output_norm.weight")
                .layer("attn_norm.weight")
                .firstOf(kv, new String[] {"attn_qkv.weight"}, new String[] {"attn_q.weight", "attn_k.weight"})
                .firstOf(noKv, new String[] {"attn_qkv.weight"}, new String[] {"attn_q.weight"})
                .optionalWithout("attn_qkv.weight", TRUNK, "attn_v.weight")
                .optionalWithout("attn_qkv.weight", noKv, "attn_k.weight")
                .layer("attn_output.weight", "attn_q_norm.weight")
                .when(kv, "attn_k_norm.weight").optionalWhen(noKv, "attn_k_norm.weight")
                .layer("post_attention_norm.weight").layerOptional("layer_output_scale.weight")
                .modelWhen(MoeTensorRequirementsSwa::anyTrunkFullAttention, "rope_freqs.weight")
                .layer("ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "post_ffw_norm.weight")
                .layerOptional("ffn_gate_inp.weight")
                .withTrigger("ffn_gate_inp.weight", TRUNK, "ffn_gate_inp.scale", "pre_ffw_norm_2.weight",
                        "post_ffw_norm_1.weight", "post_ffw_norm_2.weight", "ffn_down_exps.weight")
                .gateUpExpsWithTrigger("ffn_gate_inp.weight", TRUNK)
                .when(ple, "inp_gate.weight", "proj.weight", "post_norm.weight")
                .build();
    }

    /** True when some trunk layer uses full (non-sliding-window) attention. */
    static boolean anyTrunkFullAttention(ModelHparams hp) {
        for (int il = 0; il < hp.nLayer(); il++) {
            if (!hp.isSwa(il)) {
                return true;
            }
        }
        return false;
    }

    /** models/maple.cpp:24-58. */
    public static List<TensorRequirement> maple() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight")
                .experts(TRUNK)
                .build();
    }

    /** models/cohere2moe.cpp:40-134: no FFN norm (parallel attention/FFN on the attention norm output). */
    public static List<TensorRequirement> cohere2moe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight")
                .denseFfn(DENSE_LEAD)
                .expertsGateUp(AFTER_LEAD).sharedExperts(and(AFTER_LEAD, SHEXP_BY_COUNT))
                .nextn()
                .build();
    }

    /**
     * models/muse-spark.cpp:25-76: gated attention, learned residual mixing, and a latent MoE (project down, norm,
     * experts, norm, project up) on every layer.
     */
    public static List<TensorRequirement> museSpark() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
                        "attn_resid_alpha.weight", "attn_resid_beta.weight", "ffn_norm.weight", "post_ffw_norm.weight",
                        "ffn_resid_alpha.weight", "ffn_resid_beta.weight")
                .experts(TRUNK).layer("exp_probs_b.bias", "ffn_latent_down.weight", "ffn_latent_up.weight",
                        "ffn_pre_exp_norm.weight", "ffn_post_exp_norm.weight")
                .build();
    }

    /**
     * models/exaone-moe.cpp:28-100. NextN blocks carry a dense FFN instead of the experts (:67-71), so their
     * decoder tensors are listed here (optional, as the blocks load with {@code TENSOR_SKIP}, :49-51).
     */
    public static List<TensorRequirement> exaoneMoe() {
        return builder().embdAndOutputNorm().outputRequired()
                .qkv().layer("attn_output.weight").modelOptional("rope_freqs.weight")
                .layer("attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .nextn()
                .optionalWhen(NEXTN, "attn_qkv.weight", "attn_q.weight", "attn_k.weight",
                        "attn_v.weight", "attn_qkv.bias", "attn_q.bias", "attn_k.bias", "attn_v.bias",
                        "attn_output.weight", "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                        "ffn_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /** models/granite-swa.cpp:56-112: the shared expert stores gate and up fused in {@code ffn_up_shexp}. */
    public static List<TensorRequirement> graniteSwa() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight").layerOptional("attn_output.bias")
                .layer("attn_sinks.weight", "ffn_norm.weight").ropeFactors()
                .denseFfn(NO_EXPERTS).optionalWhen(NO_EXPERTS, "ffn_gate.bias", "ffn_down.bias", "ffn_up.bias")
                .expertsGateUp(WITH_EXPERTS)
                .when(and(WITH_EXPERTS, SHEXP_BY_FF), "ffn_up_shexp.weight", "ffn_down_shexp.weight")
                .build();
    }

    /** models/afmoe.cpp:38-99: sandwich norms and gated attention; dense lead, then biased routed experts. */
    public static List<TensorRequirement> afmoe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight", "post_attention_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
                        "ffn_norm.weight", "post_ffw_norm.weight")
                .experts(AFTER_LEAD).when(AFTER_LEAD, "exp_probs_b.bias")
                .sharedExperts(and(AFTER_LEAD, SHEXP_BY_COUNT))
                .denseFfn(DENSE_LEAD)
                .build();
    }

    /** models/laguna.cpp:68-147: per-head or per-element attention gate; dense lead, then biased experts. */
    public static List<TensorRequirement> laguna() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
                        "ffn_norm.weight")
                .experts(AFTER_LEAD).when(AFTER_LEAD, "exp_probs_b.bias").sharedExperts(AFTER_LEAD)
                .denseFfn(DENSE_LEAD)
                .build();
    }

    /** gpt-oss, models/openai-moe.cpp:22-55: attention sinks, router and expert biases. */
    public static List<TensorRequirement> gptOss() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight", "post_attention_norm.weight").qkv()
                .layer("attn_output.weight", "attn_sinks.weight")
                .experts(TRUNK)
                .layer("attn_output.bias", "ffn_gate_inp.bias", "ffn_gate_exps.bias", "ffn_down_exps.bias",
                        "ffn_up_exps.bias")
                .build();
    }

    /** models/smallthinker.cpp:30-63. */
    public static List<TensorRequirement> smallthinker() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "ffn_norm.weight")
                .experts(TRUNK)
                .build();
    }

    /**
     * models/mimo2.cpp:25-76: dense and expert FFN tensors are both optional; a block is MoE exactly when its
     * {@code ffn_gate_inp} is present.
     */
    public static List<TensorRequirement> mimo2() {
        return builder().embdAndOutputNorm().outputRequired()
                .qkv().layer("attn_output.weight", "attn_norm.weight").layerOptional("attn_sinks.weight")
                .layer("ffn_norm.weight")
                .denseFfnOptional(TRUNK)
                .layerOptional("ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight",
                        "ffn_up_exps.weight", "exp_probs_b.bias")
                .nextn().optionalWhen(NEXTN, "layer_output_norm.weight")
                .build();
    }

    /**
     * models/step35.cpp:37-172: optional QK norms and head-wise attention gate; dense, routed and shared FFN
     * tensors are all optional (a block is MoE exactly when its {@code ffn_gate_inp} is present). The untied
     * {@code output} is created with the trunk flags (:48).
     */
    public static List<TensorRequirement> step35() {
        return builder().embdAndOutputNorm().modelTrunk("output.weight")
                .layer("attn_norm.weight").layerOptional("attn_q_norm.weight", "attn_k_norm.weight").ropeFactors()
                .qkv().layer("attn_output.weight").layerOptional("attn_gate.weight")
                .layer("ffn_norm.weight")
                .denseFfnOptional(TRUNK)
                .layerOptional("ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight",
                        "ffn_up_exps.weight", "exp_probs_b.bias")
                .sharedExpertsOptional(TRUNK)
                .nextn()
                .build();
    }

    /**
     * models/maion-coder.cpp:28-63: sandwich norms; every {@code interleave_moe_layer_step}-th layer
     * ({@code (il + 1) % step == 0}) has a latent MoE, the others a dense FFN.
     */
    public static List<TensorRequirement> maionCoder() {
        LayerCondition moe = (hp, il) -> il < hp.nLayer() && (il + 1) % hp.nMoeLayerStep() == 0;
        LayerCondition dense = (hp, il) -> il < hp.nLayer() && (il + 1) % hp.nMoeLayerStep() != 0;
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight",
                        "ffn_norm.weight", "post_ffw_norm.weight")
                .experts(moe).when(moe, "ffn_latent_down.weight", "ffn_latent_up.weight")
                .denseFfn(dense)
                .build();
    }

    /** models/mellum.cpp:32-89. */
    public static List<TensorRequirement> mellum() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight", "ffn_norm.weight")
                .experts(TRUNK)
                .nextnOf("eh_proj", "enorm", "hnorm", "shared_head_norm")
                .build();
    }

    /** models/beam.cpp:59-101: sandwich norms and a biased per-head attention gate; dense lead, then experts. */
    public static List<TensorRequirement> beam() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight", "post_attention_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
                        "attn_gate.bias", "ffn_norm.weight", "post_ffw_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).when(AFTER_LEAD, "exp_probs_b.bias")
                .build();
    }

    /** models/kolibri1.cpp:28-64: sandwich norms; biased routed experts plus one shared expert on every layer. */
    public static List<TensorRequirement> kolibri1() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight", "post_attention_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight",
                        "post_ffw_norm.weight")
                .experts(TRUNK).layer("exp_probs_b.bias").sharedExperts(TRUNK)
                .build();
    }
}
