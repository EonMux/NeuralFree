package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;

import java.util.List;

/**
 * Tensor list of Inkling (vLLM's {@code InklingForCausalLM}), which has no llama.cpp converter: a NoPE attention
 * block with separate Q/K/V projections, a query-side relative projection and its per-head table, per-head Q/K RMS
 * norms, four short-convolution kernels and a gated SiLU MLP. The converted checkpoint splits vLLM's fused
 * {@code attn.qkvr} rows into {@code attn_q}/{@code attn_k}/{@code attn_v}/{@code attn_rel}, and its interleaved
 * {@code gate_up_proj} rows into {@code ffn_gate}/{@code ffn_up}.
 */
public final class InklingTensorRequirements {

    private InklingTensorRequirements() {
    }

    /** Every tensor the Inkling graph reads, in load order. */
    public static List<TensorRequirement> inkling() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output.weight")
                .modelOptional("token_embd_norm.weight")
                .layer("attn_norm.weight", "ffn_norm.weight")
                .layer("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_rel.weight", "attn_output.weight")
                .layer("attn_q_norm.weight", "attn_k_norm.weight", "attn_rel_proj.weight")
                .layer("attn_sconv_k.weight", "attn_sconv_v.weight", "attn_sconv.weight", "ffn_sconv.weight")
                .layer("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
                .layerOptional("attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_rel.bias", "attn_output.bias",
                        "ffn_global_scale.weight")
                .build();
    }
}
