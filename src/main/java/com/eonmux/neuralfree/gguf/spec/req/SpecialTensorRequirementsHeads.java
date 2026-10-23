package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerSpan;

import java.util.List;

import static com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirements.DEC;

/**
 * Tensors of the decision heads that clef and vllm-decision2 add on top of their backbone's tensor list
 * (combine with {@link SpecialTensorRequirements#extend}). Line references are to llama.cpp src/models/.
 */
public final class SpecialTensorRequirementsHeads {

    /** clef head blocks {@code [0, n_layer_routing + n_layer_joint)} (clef.cpp:58-59). */
    private static final LayerCondition CLEF_HEAD = (hp, il) ->
            il < hp.decisionRoutingLayers() + hp.decisionJointLayers();
    /** Layer slots of the clef head stack; it may exceed {@code n_layer}. */
    private static final LayerSpan CLEF_HEAD_SPAN = hp -> hp.decisionRoutingLayers() + hp.decisionJointLayers();
    /** clef routing blocks {@code [0, n_layer_routing)}, which have no self attention (clef.cpp:62-63). */
    private static final LayerCondition CLEF_ROUTING = (hp, il) -> il < hp.decisionRoutingLayers();
    /** clef joint blocks {@code [n_layer_routing, n_layer_routing + n_layer_joint)} (clef.cpp:64-67). */
    private static final LayerCondition CLEF_JOINT = (hp, il) ->
            il >= hp.decisionRoutingLayers() && CLEF_HEAD.test(hp, il);

    private SpecialTensorRequirementsHeads() {
    }

    /** Decision head of vllm-decision2 (vllm-decision2.cpp:18-39), created after the qwen3 tensors. */
    public static List<TensorRequirement> vllmDecision2Head() {
        return SpecialTensorRequirements.builder()
                .model("decision.key.weight")
                .modelWeightBias("decision.option_norm")
                .modelWeightBias("decision.query_norm")
                .model("decision.query.weight")
                .modelWeightBias("decision.option_mlp")
                .model("decision.query_mlp.weight", "decision.scorer_out.weight")
                .build();
    }

    /** Joint decision head of clef (clef.cpp:26-98), created after the qwen35 tensors. */
    public static List<TensorRequirement> clefHead() {
        return SpecialTensorRequirements.builder()
                .model("decision.proj_memory.weight")
                .weightBiasOver(CLEF_HEAD_SPAN, DEC, true, CLEF_ROUTING, "cross_attn_norm_kv")
                .weightBiasOver(CLEF_HEAD_SPAN, DEC, true, CLEF_JOINT, "attn_norm", "attn_q", "attn_k", "attn_v",
                        "attn_o")
                .weightBiasOver(CLEF_HEAD_SPAN, DEC, true, CLEF_HEAD, "cross_attn_norm", "cross_attn_q",
                        "cross_attn_k", "cross_attn_v", "cross_attn_o", "ffn_norm", "ffn_up", "ffn_down")
                .modelWeightBias("decision.hidden_norm")
                .modelWeightBias("decision.option_summary_norm")
                .modelWeightBias("decision.field_norm")
                .modelWeightBias("decision.option_norm")
                .model("decision.proj_question.weight", "decision.proj_option_question.weight",
                        "decision.proj_global.weight", "decision.proj_option_context.weight",
                        "decision.proj_option_lexical.weight", "decision.scales", "token_types.weight")
                .modelWeightBias("decision.scorer")
                .modelWeightBias("decision.scorer_out")
                .build();
    }
}
