package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;

import java.util.List;

import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.DENSE_LAYER;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.MOE_LAYER;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.builder;

/**
 * Tensor list of the Step5 text decoder (vLLM {@code vllm/models/step5}): GQA attention with per-head Q/K RMS norms
 * and a per-head output gate, the sparse indexer's query/key/value-weight/head-weight projections and its RMS query
 * and LayerNorm key norms, a dense or routed-plus-shared-expert FFN per layer, and the optional NextN/MTP fusion and
 * shared head. The NF layout names the indexer tensors {@code indexer.attn_q_b} (query), {@code indexer.attn_k}
 * (key), {@code indexer.z} (region-summary weights), {@code indexer.proj} (per-head weights),
 * {@code indexer.q_norm} and {@code indexer.k_norm}.
 */
public final class Step5TensorRequirements {

    private Step5TensorRequirements() {
    }

    /** The required and optional tensors of a Step5 file. */
    public static List<TensorRequirement> step5() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
                        "ffn_norm.weight")
                // DSA indexer tensors exist only on full-attention blocks (vLLM builds them when the block has no
                // sliding window); the region-summary projection marks their presence.
                .layerOptional("indexer.z.weight")
                .withTrigger("indexer.z.weight", TensorRequirement.TRUNK, "indexer.attn_q_b.weight",
                        "indexer.attn_k.weight", "indexer.proj.weight", "indexer.q_norm.weight",
                        "indexer.k_norm.weight", "indexer.k_norm.bias")
                .denseFfn(DENSE_LAYER)
                .experts(MOE_LAYER).probsBiasOptional(MOE_LAYER).sharedExperts(MOE_LAYER)
                .nextn()
                .build();
    }
}
