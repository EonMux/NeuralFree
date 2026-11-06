package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.arch.TensorRequirement.NEXTN;
import static com.eonmux.neuralfree.gguf.arch.TensorRequirement.TRUNK;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.AFTER_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.DENSE_LAYER;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.DENSE_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.MOE_LAYER;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.NO_EXPERTS;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.WITH_EXPERTS;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.and;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.builder;

/**
 * Tensor lists of the MoE architectures with plain multi-head attention, referenced from
 * {@link com.eonmux.neuralfree.gguf.arch.ArchTableMoeClassic}. Each method transcribes
 * {@code llama_model_<arch>::load_arch_tensors} of llama.cpp src/models/&lt;file&gt;.cpp (cited
 * {@code models/<file>.cpp:lines}); conventions in {@link MoeTensorRequirementsSupport}.
 */
public final class MoeTensorRequirements {

    /** Layers with a shared expert in loaders that test {@code n_ff_shexp > 0}. */
    private static final LayerCondition SHEXP_BY_FF = (hp, il) -> hp.nFfShexp() > 0;
    /** Layers with a shared expert in loaders that test {@code n_expert_shared > 0}. */
    private static final LayerCondition SHEXP_BY_COUNT = (hp, il) -> hp.nExpertShared() > 0;

    private MoeTensorRequirements() {
    }

    /** models/llama.cpp:34-91: dense FFN without experts, otherwise routed experts and an optional shared expert. */
    public static List<TensorRequirement> llama() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight").layerOptional("attn_output.bias")
                .layer("ffn_norm.weight").ropeFactors()
                .denseFfn(NO_EXPERTS).optionalWhen(NO_EXPERTS, "ffn_gate.bias", "ffn_down.bias", "ffn_up.bias")
                .expertsOptionalGate(WITH_EXPERTS)
                .sharedExperts(and(WITH_EXPERTS, SHEXP_BY_FF))
                .build();
    }

    /**
     * models/llama4.cpp:44-94: every {@code interleave_moe_layer_step}-th layer has routed and shared experts, the
     * others a dense FFN (the hparams hook stores this layer split).
     */
    public static List<TensorRequirement> llama4() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "ffn_norm.weight")
                .modelOptional("rope_freqs.weight")
                .experts(MOE_LAYER).sharedExperts(MOE_LAYER)
                .denseFfn(DENSE_LAYER)
                .build();
    }

    /**
     * models/grok.cpp:35-79: optional dense FFN beside the experts; the block output norm is
     * {@code layer_output_norm}, or {@code post_ffw_norm} when that is absent (:75-77).
     */
    public static List<TensorRequirement> grok() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_output_norm.weight", "ffn_norm.weight")
                .denseFfnOptional(TRUNK)
                .expertsOptionalGate(TRUNK)
                .firstOf(TRUNK, new String[] {"layer_output_norm.weight"}, new String[] {"post_ffw_norm.weight"})
                .build();
    }

    /** models/qwen2moe.cpp:16-58: experts plus a sigmoid-gated shared expert on every layer. */
    public static List<TensorRequirement> qwen2moe() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "ffn_norm.weight")
                .experts(TRUNK)
                .layer("ffn_gate_inp_shexp.weight").sharedExperts(TRUNK)
                .build();
    }

    /** models/qwen3vlmoe.cpp:16-57. */
    public static List<TensorRequirement> qwen3vlmoe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight")
                .layer("ffn_norm.weight")
                .experts(TRUNK)
                .build();
    }

    /**
     * The InternS1-Pro text backbone: the Qwen3-VL MoE tensors plus the model-level Fourier coefficient matrices its
     * rotation reads ({@code rope_cos_coef}/{@code rope_sin_coef}, required exactly when hparams select FoPE).
     */
    public static List<TensorRequirement> internS1Pro() {
        List<TensorRequirement> out = new ArrayList<>(qwen3vlmoe());
        out.add(TensorRequirement.modelWhen(ModelHparams::ropeFope,
                TensorNames.weight(TensorNames.ROPE_COS_COEF)));
        out.add(TensorRequirement.modelWhen(ModelHparams::ropeFope,
                TensorNames.weight(TensorNames.ROPE_SIN_COEF)));
        return Collections.unmodifiableList(out);
    }

    /** models/phimoe.cpp:12-45: biased norms and output head, LongRoPE factors. */
    public static List<TensorRequirement> phimoe() {
        return builder().embdAndOutputNorm().model("output_norm.bias", "output.weight", "output.bias")
                .layer("attn_norm.weight", "attn_norm.bias").qkv().layer("attn_output.weight", "attn_output.bias")
                .layer("ffn_norm.weight", "ffn_norm.bias")
                .experts(TRUNK)
                .modelOptional("rope_factors_long.weight", "rope_factors_short.weight")
                .build();
    }

    /** models/dbrx.cpp:13-40: fused QKV; {@code attn_output_norm} is the pre-FFN norm. */
    public static List<TensorRequirement> dbrx() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_output.weight", "attn_output_norm.weight")
                .experts(TRUNK)
                .build();
    }

    /** models/olmoe.cpp:12-46. */
    public static List<TensorRequirement> olmoe() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight")
                .layer("ffn_norm.weight")
                .experts(TRUNK)
                .build();
    }

    /** models/arctic.cpp:16-49: a dense residual FFN beside the normalized expert branch. */
    public static List<TensorRequirement> arctic() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "ffn_norm.weight")
                .denseFfn(TRUNK)
                .layer("ffn_norm_exps.weight").experts(TRUNK)
                .build();
    }

    /** models/deepseek.cpp:17-66: dense lead, then routed and shared experts. */
    public static List<TensorRequirement> deepseek() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "ffn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .build();
    }

    /** models/granite-moe.cpp:21-76. */
    public static List<TensorRequirement> graniteMoe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight").layerOptional("attn_output.bias")
                .layer("ffn_norm.weight").ropeFactors()
                .denseFfn(NO_EXPERTS).optionalWhen(NO_EXPERTS, "ffn_gate.bias", "ffn_down.bias", "ffn_up.bias")
                .expertsOptionalGate(WITH_EXPERTS)
                .sharedExperts(and(WITH_EXPERTS, SHEXP_BY_FF))
                .build();
    }

    /** models/bailingmoe.cpp:18-55. */
    public static List<TensorRequirement> bailingmoe() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "ffn_norm.weight")
                .experts(TRUNK).sharedExperts(TRUNK)
                .build();
    }

    /** models/bailingmoe2.cpp:20-82: fused QKV; the NextN blocks also carry {@code layer_output_norm}. */
    public static List<TensorRequirement> bailingmoe2() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_output.weight", "attn_q_norm.weight",
                        "attn_k_norm.weight", "ffn_norm.weight")
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .denseFfn(DENSE_LEAD)
                .nextn().optionalWhen(NEXTN, "layer_output_norm.weight")
                .build();
    }

    /** models/dots1.cpp:18-66. */
    public static List<TensorRequirement> dots1() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight")
                .layer("ffn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .build();
    }

    /**
     * vLLM's openPangu MoE trunk without MLA: the pangu-embedded attention with optional projection biases, or its
     * static sink attention ({@code attn_sink_k}/{@code attn_sink_v}) with the K norm it normalizes keys by; a dense
     * lead layer and then routed plus shared experts with an optional router bias.
     */
    public static List<TensorRequirement> panguMoe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight").layerOptional("attn_output.bias")
                .layerOptional("attn_k_norm.weight", "attn_sink_k.weight", "attn_sink_v.weight")
                .layer("ffn_norm.weight").layerOptional("post_attention_norm.weight", "post_ffw_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .build();
    }

    /**
     * vLLM's Step3 text trunk (step3_text.py): the low-rank query projections ({@code attn_q_a},
     * {@code attn_q_a_norm}, {@code attn_q_b}) over plain K and V, a dense FFN where no router exists, and routed
     * plus one shared expert where {@code ffn_gate_inp} is present.
     */
    public static List<TensorRequirement> step3Text() {
        return builder().embdAndOutputNorm().outputRequired()
                .layer("attn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                        "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_norm.weight")
                .denseFfnOptional(TRUNK)
                .layerOptional("ffn_gate_inp.weight")
                .withTrigger("ffn_gate_inp.weight", TRUNK,
                        "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight",
                        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")
                .build();
    }

    /** ERNIE 4.5 MoE, models/ernie4-5.cpp:23-68 (the MoE branch of {@code llama_model_ernie4_5}). */
    public static List<TensorRequirement> ernie45Moe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight").layerOptional("attn_output.bias")
                .layer("ffn_norm.weight")
                .expertsOptionalGate(AFTER_LEAD).probsBiasOptional(AFTER_LEAD)
                .sharedExperts(and(AFTER_LEAD, SHEXP_BY_FF))
                .denseFfn(DENSE_LEAD)
                .build();
    }

    /** models/hunyuan-moe.cpp:14-49. */
    public static List<TensorRequirement> hunyuanMoe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight")
                .layer("ffn_norm.weight")
                .experts(TRUNK).sharedExperts(TRUNK)
                .build();
    }

    /**
     * models/hy-v3.cpp:22-87: every FFN tensor is optional; a block is MoE exactly when its {@code ffn_gate_inp}
     * is present. The selection bias is stored without a {@code .weight}/{@code .bias} suffix. The routed gate/up
     * pair comes from {@code create_tensor_gate_up_exps} with {@code TENSOR_NOT_REQUIRED} (:61).
     */
    public static List<TensorRequirement> hyV3() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight")
                .layer("ffn_norm.weight")
                .denseFfnOptional(TRUNK)
                .layerOptional("ffn_gate_inp.weight", "exp_probs_b", "ffn_down_exps.weight")
                .gateUpExpsOptional(TRUNK)
                .sharedExpertsOptional(TRUNK)
                .nextn()
                .build();
    }

    /** models/grovemoe.cpp:16-60: routed experts plus chunk ("adjugate") experts. */
    public static List<TensorRequirement> grovemoe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight")
                .layer("ffn_norm.weight")
                .experts(TRUNK)
                .layer("ffn_gate_chexps.weight", "ffn_down_chexps.weight", "ffn_up_chexps.weight")
                .build();
    }

    /** models/minimax-m2.cpp:14-40. */
    public static List<TensorRequirement> minimaxM2() {
        return builder().embdAndOutputNorm().outputRequired()
                .qkv().layer("attn_output.weight", "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight")
                .layer("ffn_norm.weight")
                .experts(TRUNK).layer("exp_probs_b.bias")
                .build();
    }

    /**
     * models/k2-horizon.cpp:51-116. MoE layers are {@code n_expert > 0 && il >= n_layer_dense_lead}; MoVA layers
     * (MoE layers of a file with value experts) replace {@code attn_v} by gated value experts.
     */
    public static List<TensorRequirement> k2Horizon() {
        LayerCondition moe = (hp, il) -> il < hp.nLayer() && hp.nExpert() > 0 && il >= hp.nLayerDenseLead();
        LayerCondition dense = (hp, il) -> il < hp.nLayer() && !(hp.nExpert() > 0 && il >= hp.nLayerDenseLead());
        LayerCondition mova = and(moe, (hp, il) -> hp.nValueExpert() > 0);
        LayerCondition plainV = (hp, il) -> il < hp.nLayer() && !mova.test(hp, il);
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight", "attn_q.weight", "attn_k.weight")
                .layerOptional("attn_q_norm.weight", "attn_k_norm.weight")
                .when(mova, "attn_v_gate.weight", "attn_v_exps.weight").optionalWhen(mova, "attn_v_gate.bias")
                .when(plainV, "attn_v.weight")
                .layer("attn_output.weight").layerOptional("attn_gate.weight")
                .layer("ffn_norm.weight")
                .experts(moe).probsBiasOptional(moe).sharedExperts(and(moe, SHEXP_BY_COUNT))
                .denseFfn(dense)
                .build();
    }
}
