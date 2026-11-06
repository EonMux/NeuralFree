package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;

import java.util.List;

/**
 * Tensor lists of the repeat-block architectures: Dory, whose trunk alternates attention-only and MLP-only blocks,
 * and IQuestLoopCoder, a Llama block with a per-layer gated mixed attention.
 */
public final class DenseTensorRequirementsLoop {

    private DenseTensorRequirementsLoop() {
    }

    /**
     * Dory ({@code dory.py}): an untied output head whose logits carry a per-vocabulary scale, attention layers with
     * Q/K/V, output and gate projections plus their slerp-residual alpha, and MLP-only layers with a SwiGLU FFN and
     * its slerp-residual alpha. There is no final output norm in the graph.
     */
    public static List<TensorRequirement> dory() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output.weight", "output_logit_scale.weight")
                .layerWhen(DenseTensorRequirements.ATTENTION_ONLY, "attn_q.weight", "attn_k.weight", "attn_v.weight",
                        "attn_output.weight", "attn_gate.weight", "attn_q_norm.weight", "attn_slerp_alpha.weight")
                .layerWhen(DenseTensorRequirements.MLP_ONLY, "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                        "ffn_slerp_alpha.weight")
                .build();
    }

    /**
     * IQuestLoopCoder ({@code iquest_loopcoder.py}): a Llama block (attention and SwiGLU FFN projections, the two
     * block norms and the final output norm) with an optional tied head, plus the per-layer gate that mixes the
     * global and local attention of the later passes ({@code gate_projections.<l>.gate_proj}, a
     * {@code [n_head][head_dim]} weight with a per-head bias).
     */
    public static List<TensorRequirement> iquestLoopCoder() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .outputOrTied()
                .model("output_norm.weight")
                .layer("attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                        "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                        "attn_loop_gate.weight", "attn_loop_gate.bias")
                .build();
    }
}
