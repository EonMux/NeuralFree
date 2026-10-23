package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;

import java.util.List;

import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.AFTER_DENSE_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ALL;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.ATTENTION;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.DENSE_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.RECURRENT;
import static com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirements.and;

/**
 * Tensor lists of the RWKV architectures (rwkv6, rwkv6qwen2, rwkv7, arwkv7) and the LFM2 short-convolution
 * hybrids (lfm2, lfm2moe).
 */
public final class HybridTensorRequirementsRwkv {

    /** LFM2 blocks before the decision head: {@code il < n_layer - n_layer_decision} (lfm2.cpp:94). */
    private static final LayerCondition LFM_TRUNK = (hp, il) -> il < hp.nLayer() - hp.nLayerDecision();
    /** LFM2 decision-head blocks: the last {@code n_layer_decision} trunk blocks (lfm2.cpp:55). */
    private static final LayerCondition LFM_DECISION = (hp, il) -> hp.nLayerDecision() > 0
            && il >= hp.nLayer() - hp.nLayerDecision();

    private HybridTensorRequirementsRwkv() {
    }

    /**
     * rwkv6.cpp:29-85. A layer needs the fused lerp or at least {@code time_mix_lerp_w} (rwkv6.cpp:65); the
     * other separate lerps are optional.
     */
    public static List<TensorRequirement> rwkv6() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias", "output_norm.weight",
                "output_norm.bias", "output.weight");
        b.layer(ALL, "attn_norm.weight", "attn_norm.bias", "attn_norm_2.weight", "attn_norm_2.bias",
                "time_mix_w1.weight", "time_mix_w2.weight", "time_mix_lerp_x.weight");
        b.layerOptional(ALL, "time_mix_lerp_k.weight", "time_mix_lerp_v.weight", "time_mix_lerp_r.weight",
                "time_mix_lerp_g.weight");
        b.layerAnyOf(ALL, new String[] {"time_mix_lerp_fused.weight"}, new String[] {"time_mix_lerp_w.weight"});
        b.layer(ALL, "time_mix_first.weight", "time_mix_decay.weight", "time_mix_decay_w1.weight",
                "time_mix_decay_w2.weight", "time_mix_key.weight", "time_mix_value.weight",
                "time_mix_receptance.weight", "time_mix_gate.weight", "time_mix_ln.weight", "time_mix_ln.bias",
                "time_mix_output.weight", "channel_mix_lerp_k.weight", "channel_mix_lerp_r.weight",
                "channel_mix_key.weight", "channel_mix_value.weight", "channel_mix_receptance.weight");
        return b.build();
    }

    /**
     * rwkv6qwen2.cpp:29-76. {@code time_mix_first} is optional: without it the layer is a QRWKV layer
     * ({@code RecurrentSpec.qrwkv()}, decided by {@code RecurrentResolver}).
     */
    public static List<TensorRequirement> rwkv6qwen2() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight");
        b.modelOptional("output_norm.bias");
        b.model("output.weight");
        b.layer(ALL, "attn_norm.weight", "time_mix_w1.weight", "time_mix_w2.weight", "time_mix_lerp_x.weight",
                "time_mix_lerp_fused.weight");
        b.layerOptional(ALL, "time_mix_first.weight");
        b.layer(ALL, "time_mix_decay.weight", "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
                "time_mix_key.weight", "time_mix_value.weight", "time_mix_receptance.weight", "time_mix_gate.weight");
        b.layerOptional(ALL, "time_mix_key.bias", "time_mix_value.bias", "time_mix_receptance.bias");
        b.layer(ALL, "time_mix_output.weight", "ffn_norm.weight");
        b.gatedFfn(ALL);
        return b.build();
    }

    /** rwkv7.cpp:49-115. */
    public static List<TensorRequirement> rwkv7() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias", "output_norm.weight",
                "output_norm.bias", "output.weight");
        b.layer(ALL, "attn_norm.weight", "attn_norm.bias", "attn_norm_2.weight", "attn_norm_2.bias");
        rwkv7Lora(b);
        b.layer(ALL, "time_mix_g1.weight", "time_mix_g2.weight", "time_mix_lerp_fused.weight");
        rwkv7Projections(b);
        b.layer(ALL, "time_mix_ln.weight", "time_mix_ln.bias", "time_mix_output.weight",
                "channel_mix_lerp_k.weight", "channel_mix_key.weight", "channel_mix_value.weight");
        return b.build();
    }

    /**
     * arwkv7.cpp:49-111. The output gate and the WKV group norm are optional (the gate applies when both
     * {@code time_mix_g1} and {@code time_mix_g2} exist, rwkv7-base.cpp:49-52); the FFN is the Qwen SwiGLU block.
     */
    public static List<TensorRequirement> arwkv7() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "output_norm.weight", "output.weight");
        b.layer(ALL, "attn_norm.weight");
        rwkv7Lora(b);
        b.layerOptional(ALL, "time_mix_g1.weight", "time_mix_g2.weight");
        b.layer(ALL, "time_mix_lerp_fused.weight");
        rwkv7Projections(b);
        b.layerOptional(ALL, "time_mix_ln.weight", "time_mix_ln.bias");
        b.layer(ALL, "time_mix_output.weight", "ffn_norm.weight");
        b.gatedFfn(ALL);
        return b.build();
    }

    /**
     * lfm2.cpp:49-134. The output norm is stored as {@code token_embd_norm} (llama-arch.cpp:487,
     * {@code LLM_TENSOR_OUTPUT_NORM_LFM2}). A decision model replaces the output head with
     * {@code token_types}, {@code cls.norm}, {@code cls} and {@code cls.output} (lfm2.cpp:78-85) and its last
     * blocks are plain pre-norm blocks with biases (lfm2.cpp:55-71). The decision tensors are required when
     * {@code n_layer_decision > 0}.
     */
    public static List<TensorRequirement> lfm2() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "token_embd_norm.weight");
        b.layer(LFM_DECISION, "attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                "attn_output.weight", "attn_output.bias", "ffn_norm.weight", "ffn_norm.bias", "ffn_up.weight",
                "ffn_up.bias", "ffn_down.weight", "ffn_down.bias");
        b.modelWhen(hp -> hp.nLayerDecision() > 0, "token_types.weight", "cls.norm.weight", "cls.norm.bias",
                "cls.weight", "cls.bias", "cls.output.weight", "cls.output.bias");
        b.outputOrTied();
        lfmTrunk(b);
        return b.build();
    }

    /** lfm2moe.cpp:26-75. */
    public static List<TensorRequirement> lfm2moe() {
        HybridTensorRequirements.Builder b = HybridTensorRequirements.builder();
        b.model("token_embd.weight", "token_embd_norm.weight");
        b.outputOrTied();
        lfmTrunk(b);
        return b.build();
    }

    /** The RWKV7 decay, in-context learning rate and value-residual LoRAs (rwkv7.cpp:76-92). */
    private static void rwkv7Lora(HybridTensorRequirements.Builder b) {
        b.layer(ALL, "time_mix_w0.weight", "time_mix_w1.weight", "time_mix_w2.weight", "time_mix_a0.weight",
                "time_mix_a1.weight", "time_mix_a2.weight", "time_mix_v0.weight", "time_mix_v1.weight",
                "time_mix_v2.weight");
    }

    /** The RWKV7 key normalization and projections (rwkv7.cpp:100-106). */
    private static void rwkv7Projections(HybridTensorRequirements.Builder b) {
        b.layer(ALL, "time_mix_k_k.weight", "time_mix_k_a.weight", "time_mix_r_k.weight", "time_mix_key.weight",
                "time_mix_value.weight", "time_mix_receptance.weight");
    }

    /**
     * The LFM2 trunk blocks (lfm2.cpp:94-130, lfm2moe.cpp:35-70): FFN norm, routed experts after the leading
     * dense blocks or a dense SwiGLU FFN, the operator norm, then attention with per-head Q/K norms or the
     * short convolution; plus the optional ColBERT {@code dense_2} projection (lfm2.cpp:133-134).
     */
    private static void lfmTrunk(HybridTensorRequirements.Builder b) {
        b.layer(LFM_TRUNK, "ffn_norm.weight");
        LayerCondition moe = and(LFM_TRUNK, AFTER_DENSE_LEAD);
        b.layer(moe, "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight",
                "exp_probs_b.bias");
        b.gatedFfn(and(LFM_TRUNK, DENSE_LEAD));
        b.layer(LFM_TRUNK, "attn_norm.weight");
        LayerCondition attention = and(LFM_TRUNK, ATTENTION);
        b.layer(attention, "attn_q_norm.weight", "attn_k_norm.weight");
        b.qkv(attention);
        b.layer(attention, "attn_output.weight");
        b.layer(and(LFM_TRUNK, RECURRENT), "shortconv.conv.weight", "shortconv.in_proj.weight",
                "shortconv.out_proj.weight");
        b.modelOptional("dense_2.weight", "dense_2.bias");
    }
}
