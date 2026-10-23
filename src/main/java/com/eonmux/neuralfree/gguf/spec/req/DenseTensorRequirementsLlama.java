package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;

import java.util.List;

/**
 * Tensor lists of the pre-norm RMSNorm decoders with the llama block layout (attention norm, attention,
 * FFN norm, FFN) and its variants: Q/K norms, attention-output biases, RoPE factor tensors, MoE branches,
 * per-layer LoRA stacks and vision-expert weights. Each method cites the llama.cpp {@code load_arch_tensors}
 * it transcribes (src/models/&lt;arch&gt;.cpp).
 */
public final class DenseTensorRequirementsLlama {

    /** deci layers with an attention or linear-replacement mixer ({@code n_head_kv > 0 || n_head > 0}). */
    private static final LayerCondition DECI_MIXER =
            (hp, il) -> il < hp.nLayer() && (hp.nHeadKv(il) > 0 || hp.nHead(il) > 0);
    /** deci layers with full attention ({@code n_head_kv > 0}). */
    private static final LayerCondition DECI_ATTENTION = (hp, il) -> il < hp.nLayer() && hp.nHeadKv(il) > 0;
    /** deci layers with an FFN ({@code n_ff > 0}). */
    private static final LayerCondition DECI_FFN = (hp, il) -> il < hp.nLayer() && hp.nFf(il) > 0;

    private DenseTensorRequirementsLlama() {
    }

    /**
     * deci.cpp:14-74. Layers with {@code n_head_kv == 0 && n_head > 0} carry only {@code attn_norm} and
     * {@code attn_output} (linear replacement); layers with {@code n_head_kv > 0} carry the full attention;
     * layers with {@code n_ff == 0} have no FFN tensors.
     */
    public static List<TensorRequirement> deci() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layerWhen(DECI_MIXER, "attn_norm.weight")
                .qkvWhen(DECI_ATTENTION)
                .layerWhen(DECI_MIXER, "attn_output.weight")
                .layerOptional("attn_output.bias")
                .layerWhen(DECI_FFN, "ffn_norm.weight")
                .ropeFactors()
                .layerWhen(DECI_FFN, "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight")
                .layerOptional("ffn_gate.bias", "ffn_down.bias", "ffn_up.bias")
                .build();
    }

    /**
     * The llama layout with optional attention-output bias, RoPE factors and the dense-or-experts FFN branch:
     * refact.cpp:15-73, minicpm.cpp:26-84, granite.cpp:59-117.
     */
    public static List<TensorRequirement> llamaWithExperts() {
        return llamaWithExpertsBuilder().build();
    }

    /**
     * mistral3.cpp:31-96: {@link #llamaWithExperts()} plus the adaptive RMS norm pair of Voxtral realtime
     * (:57-62): {@code ffn_norm_ada_down} is optional, and {@code ffn_norm_ada_up} is required where it exists.
     */
    public static List<TensorRequirement> mistral3() {
        return llamaWithExpertsBuilder()
                .layerOptional("ffn_norm_ada_down.weight")
                .layerWithTrigger("ffn_norm_ada_down.weight", "ffn_norm_ada_up.weight")
                .build();
    }

    private static DenseTensorRequirements.Builder llamaWithExpertsBuilder() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layer("ffn_norm.weight")
                .ropeFactors()
                .llamaFfnOrExperts();
    }

    /**
     * granite-switch.cpp:74-132: the granite layout with a fused QKV and, per projection, stacked LoRA
     * adapter factors {@code lora_a}/{@code lora_b} (slot 0 is the zero delta).
     */
    public static List<TensorRequirement> graniteSwitch() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_output.weight", "ffn_norm.weight")
                .gatedFfn()
                .layer("attn_q.lora_a", "attn_q.lora_b", "attn_k.lora_a", "attn_k.lora_b",
                        "attn_v.lora_a", "attn_v.lora_b", "attn_output.lora_a", "attn_output.lora_b",
                        "ffn_gate.lora_a", "ffn_gate.lora_b", "ffn_up.lora_a", "ffn_up.lora_b",
                        "ffn_down.lora_a", "ffn_down.lora_b")
                .build();
    }

    /** arcee.cpp:13-42; ungated ReLU^2 MLP and {@code rope_freqs} only (:37). */
    public static List<TensorRequirement> arcee() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight")
                .ropeFreqs()
                .layer("ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /** ernie4-5.cpp:23-70, dense branch; paddleocr reuses this loader (models.h:2143-2145). */
    public static List<TensorRequirement> ernie45() {
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
                .build();
    }

    /** apertus.cpp:17-54; ungated xIELU MLP and Q/K norms with optional biases. */
    public static List<TensorRequirement> apertus() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight", "output.weight")
                .layer("attn_norm.weight")
                .ropeFactors()
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layer("ffn_norm.weight", "ffn_down.weight", "ffn_up.weight", "attn_q_norm.weight")
                .layerOptional("attn_q_norm.bias")
                .layer("attn_k_norm.weight")
                .layerOptional("attn_k_norm.bias")
                .build();
    }

    /** pangu-embed.cpp:13-52; the attention-output bias is required. */
    public static List<TensorRequirement> panguEmbedded() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "attn_output.bias", "ffn_norm.weight")
                .ropeFactors()
                .gatedFfn()
                .build();
    }

    /** exaone.cpp:12-40; {@code rope_freqs} only (:35). */
    public static List<TensorRequirement> exaone() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight")
                .ropeFreqs()
                .gatedFfn()
                .build();
    }

    /**
     * cogvlm.cpp:12-47; the {@code vis_*} vision-expert weights are required even though text never uses them, and
     * {@code rope_freqs} is the only RoPE factor tensor (:36).
     */
    public static List<TensorRequirement> cogvlm() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_output.weight", "vis_attn_qkv.weight",
                        "vis_attn_output.weight")
                .ropeFreqs()
                .layer("ffn_norm.weight")
                .gatedFfn()
                .layer("vis_gate.weight", "vis_down.weight", "vis_up.weight")
                .build();
    }

    /** The plain llama layout with a tied-or-untied head: smollm3.cpp:13-40. */
    public static List<TensorRequirement> llamaTied() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight")
                .gatedFfn()
                .build();
    }

    /**
     * The llama layout with per-head Q/K RMS norms: hunyuan-vl.cpp:23-54 (also hunyuan-dense, models.h:2260-2262),
     * maincoder.cpp:12-41, sensenova-u1.cpp:41-70.
     */
    public static List<TensorRequirement> llamaQkNorm() {
        return llamaQkNormBuilder().build();
    }

    /** qwen3vl.cpp:16-54: {@link #llamaQkNorm()} plus the optional classifier head {@code cls.output}. */
    public static List<TensorRequirement> qwen3vl() {
        return llamaQkNormBuilder()
                .modelOptional("cls.output.weight")
                .build();
    }

    /** qwen3.cpp:15-51: {@link #qwen3vl()} plus the optional {@code dense_2} projection of embedding exports. */
    public static List<TensorRequirement> qwen3() {
        return llamaQkNormBuilder()
                .modelOptional("cls.output.weight", "dense_2.weight", "dense_2.bias")
                .build();
    }

    private static DenseTensorRequirements.Builder llamaQkNormBuilder() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight", "ffn_norm.weight")
                .gatedFfn();
    }

    /** qwen3moe.cpp:14-56; every layer is a MoE layer with required gate/down/up expert tensors. */
    public static List<TensorRequirement> qwen3moe() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight", "ffn_norm.weight",
                        "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight")
                .build();
    }

    /** qwen2.cpp:19-47 and qwen2vl.cpp:8-36; optional output bias. */
    public static List<TensorRequirement> qwen2() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .modelOptional("output.bias")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight")
                .gatedFfn()
                .build();
    }

    /**
     * seed-oss.cpp:12-43; {@code post_attention_norm} is the pre-FFN norm (no {@code ffn_norm} tensor).
     */
    public static List<TensorRequirement> seedOss() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .qkv()
                .layer("attn_output.weight", "attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight",
                        "ffn_up.weight", "ffn_down.weight")
                .build();
    }

    /** spark2-5.cpp:20-50; per-head sigmoid attention gate {@code attn_gate}. */
    public static List<TensorRequirement> spark25() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_gate.weight", "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                        "ffn_up.weight", "ffn_down.weight")
                .build();
    }

    /**
     * talkie.cpp:13-34; weightless block norms, a per-head Q gain {@code attn_q_norm} and the scalar
     * {@code layer_output_scale} of the embedding skip; no output norm tensor.
     */
    public static List<TensorRequirement> talkie() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output.weight")
                .qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                        "ffn_down.weight", "layer_output_scale.weight")
                .build();
    }

    /**
     * phi3.cpp:27-57; {@code create_tensor_qkv} runs with {@code TENSOR_NOT_REQUIRED}, the FFN up projection
     * is the fused gate+up, and the LONGROPE factors are optional.
     */
    public static List<TensorRequirement> phi3() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkvOptionalWhen(TensorRequirement.TRUNK)
                .layer("attn_output.weight", "ffn_norm.weight", "ffn_down.weight", "ffn_up.weight")
                .modelOptional("rope_factors_long.weight", "rope_factors_short.weight")
                .build();
    }

    /** openelm.cpp:15-44; per-layer head and FFN sizes, fused QKV, output tied to {@code token_embd}. */
    public static List<TensorRequirement> openelm() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                        "attn_output.weight", "ffn_norm.weight")
                .gatedFfn()
                .build();
    }

    /** chatglm.cpp:25-52; the FFN up projection is the fused gate+up (SwiGLU). */
    public static List<TensorRequirement> chatglm() {
        return DenseTensorRequirements.builder()
                .tokenEmbd()
                .model("output_norm.weight")
                .outputOrTied()
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight", "ffn_up.weight", "ffn_down.weight")
                .build();
    }
}
