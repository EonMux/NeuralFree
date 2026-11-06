package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;

/**
 * A tiny EAGLE-1 draft GGUF (LLM_ARCH_EAGLE, the {@code ArchTableDraft} row) for a target of the same width and
 * vocabulary: the {@code fc} fusion over {@code [token embedding ‖ target hidden]}, and one decoder layer with a
 * gated FFN, its own token embedding and output head. Values follow {@link SyntheticModels}: projection matrices in
 * the config's weight type, norm weights 1. With norms the checkpoint has the mistral_eagle.py shape
 * ({@code input_layernorm}, {@code output_norm}); without them the llama_eagle.py / cnets.py shape, which keeps
 * only the layer's {@code ffn_norm}. The file carries no tokenizer, so it binds only as a draft file.
 */
public final class SyntheticEagle {

    /** EAGLE-3 fuses exactly three target layers (eagle3.cpp:9-11). */
    private static final int EAGLE3_TARGET_LAYERS = 3;

    private SyntheticEagle() {
    }

    /** The mistral-shaped draft (input norm and final norm present) over target layer {@code targetLayer}. */
    public static GgufTestWriter eagle(SyntheticModels.Config c, int targetLayer) {
        return eagle(c, true, targetLayer);
    }

    /**
     * The draft with the target's SPM tokenizer metadata, as a converted checkpoint carries it
     * ({@code HfEagle1Hooks.vocab} inherits the target's tokenizer); the plain writers bind only as draft files.
     */
    public static GgufTestWriter eagleWithTokenizer(SyntheticModels.Config c, boolean withNorms, int targetLayer) {
        GgufTestWriter w = eagle(c, withNorms, targetLayer);
        SyntheticModels.spm(w, c.nVocab());
        return w;
    }

    /**
     * The draft of {@code c}'s dimensions; {@code targetLayer} is the single {@code eagle.target_layers} entry,
     * {@code withNorms} selects the checkpoint shape. The draft must hold one decoder layer of the target's width.
     *
     * @throws IllegalArgumentException when the geometry is not a one-layer drafter or the attention geometry is
     *                                  invalid
     */
    public static GgufTestWriter eagle(SyntheticModels.Config c, boolean withNorms, int targetLayer) {
        requireAttention(c, "eagle");
        if (c.nLayer() != 1) {
            throw new IllegalArgumentException("an EAGLE-1 drafter holds one decoder layer, got " + c.nLayer());
        }
        if (targetLayer < 0) {
            throw new IllegalArgumentException("eagle target layer must be >= 0, got " + targetLayer);
        }
        GgufTestWriter w = base("eagle", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.headDim());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.i32Array(w.archKey("target_layers"), targetLayer);
        // the EAGLE row fuses a target hidden of the drafter's own width (the same-width target pair)
        w.u32(w.archKey("target_hidden_size"), c.nEmbd());

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", 2L * c.nEmbd(), c.nEmbd());
        if (withNorms) {
            t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        }
        t.qkvo(0, (long) c.nHead() * c.headDim());
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(0, c.nFf());
        if (withNorms) {
            t.ones("output_norm.weight", c.nEmbd());
        }
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /**
     * A tiny EAGLE-3 draft GGUF for a target of the same width and vocabulary: the {@code fc} encoder over the
     * three target taps ({@link #EAGLE3_TARGET_LAYERS} layers), its one decoder layer over the concatenated
     * {@code [embedding | fused hidden]} input, and its own token embedding and output head. Values follow
     * {@link SyntheticModels}: projection matrices in the config's weight type, norm weights 1.
     *
     * @param targetLayers the three target tap points, in the order the encoder concatenates them
     * @param normBeforeFc whether the encoder RMS-norms the target features before {@code fc} ({@code norm_before_fc})
     * @throws IllegalArgumentException when the geometry is not a one-layer drafter, the three taps are missing, or
     *                                  the attention geometry is invalid
     */
    public static GgufTestWriter eagle3(SyntheticModels.Config c, int[] targetLayers, boolean normBeforeFc) {
        requireAttention(c, "eagle3");
        if (c.nLayer() != 1) {
            throw new IllegalArgumentException("an EAGLE-3 drafter holds one decoder layer, got " + c.nLayer());
        }
        if (targetLayers.length != EAGLE3_TARGET_LAYERS) {
            throw new IllegalArgumentException("eagle3 needs exactly " + EAGLE3_TARGET_LAYERS + " target layers, got "
                    + targetLayers.length);
        }
        GgufTestWriter w = base("eagle3", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.headDim());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.i32Array(w.archKey("target_layers"), targetLayers);
        w.u32(w.archKey("target_hidden_size"), c.nEmbd());
        if (normBeforeFc) {
            w.bool(w.archKey("norm_before_fc"), true);
        }

        long input = 2L * c.nEmbd();
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", (long) EAGLE3_TARGET_LAYERS * c.nEmbd(), c.nEmbd());
        if (normBeforeFc) {
            t.ones("enc.output_norm.weight", (long) EAGLE3_TARGET_LAYERS * c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        t.ones(blk(0, "attn_norm_2.weight"), c.nEmbd());
        t.matrix(blk(0, "attn_q.weight"), input, (long) c.nHead() * c.headDim());
        t.matrix(blk(0, "attn_k.weight"), input, (long) c.nHeadKv() * c.headDim());
        t.matrix(blk(0, "attn_v.weight"), input, (long) c.nHeadKv() * c.headDim());
        t.matrix(blk(0, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(0, c.nFf());
        return w;
    }

    /**
     * A tiny EAGLE-3 draft in vLLM's DeepSeek layout ({@code deepseek_eagle3.Eagle3DeepseekV2ForCausalLM}): the
     * three-tap fusion and one decoder layer of {@link #eagle3}, whose layer runs multi-head latent attention with
     * the absorbed {@code attn_k_b} / {@code attn_v_b} decompression. Every projection reads the concatenated
     * {@code [embedding | fused hidden]} input, so the query and KV LoRA projections are {@code 2 * n_embd} wide.
     *
     * @param targetLayers the three target tap points, in the order the encoder concatenates them
     * @param qLora        the query LoRA rank ({@code attention.q_lora_rank})
     * @param kvLora       the KV LoRA rank ({@code attention.kv_lora_rank})
     * @throws IllegalArgumentException when the geometry is not a one-layer drafter, the three taps are missing, or
     *                                  the attention geometry is invalid
     */
    public static GgufTestWriter eagle3Mla(SyntheticModels.Config c, int[] targetLayers, int qLora, int kvLora) {
        requireAttention(c, "eagle3");
        if (c.nLayer() != 1) {
            throw new IllegalArgumentException("an EAGLE-3 drafter holds one decoder layer, got " + c.nLayer());
        }
        if (targetLayers.length != EAGLE3_TARGET_LAYERS) {
            throw new IllegalArgumentException("eagle3 needs exactly " + EAGLE3_TARGET_LAYERS + " target layers, got "
                    + targetLayers.length);
        }
        if (qLora <= 0 || kvLora <= 0 || c.headDim() <= c.nRot()) {
            throw new IllegalArgumentException("eagle3 MLA needs positive LoRA ranks and a non-rotary head part");
        }
        GgufTestWriter w = base("eagle3", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), kvLora + c.nRot());
        w.u32(w.archKey("attention.value_length"), kvLora);
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), c.headDim());
        w.u32(w.archKey("attention.q_lora_rank"), qLora);
        w.u32(w.archKey("attention.kv_lora_rank"), kvLora);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.i32Array(w.archKey("target_layers"), targetLayers);
        w.u32(w.archKey("target_hidden_size"), c.nEmbd());

        long input = 2L * c.nEmbd();
        long qWidth = (long) c.nHead() * c.headDim();
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", (long) EAGLE3_TARGET_LAYERS * c.nEmbd(), c.nEmbd());
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        t.ones(blk(0, "attn_norm_2.weight"), c.nEmbd());
        t.matrix(blk(0, "attn_q_a.weight"), input, qLora);
        t.ones(blk(0, "attn_q_a_norm.weight"), qLora);
        t.matrix(blk(0, "attn_q_b.weight"), qLora, qWidth);
        t.matrix(blk(0, "attn_kv_a_mqa.weight"), input, kvLora + c.nRot());
        t.ones(blk(0, "attn_kv_a_norm.weight"), kvLora);
        t.matrix(blk(0, "attn_k_b.weight"), c.headDim() - c.nRot(), kvLora, c.nHead());
        t.matrix(blk(0, "attn_v_b.weight"), kvLora, c.headDim(), c.nHead());
        t.matrix(blk(0, "attn_output.weight"), qWidth, c.nEmbd());
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(0, c.nFf());
        return w;
    }

    /**
     * A tiny DeepSeek-MTP-shaped EAGLE draft ({@code deepseek_eagle.EagleDeepseekV3ForCausalLM}, the
     * {@code EagleDeepSeekMTPModel} name): the {@code fc} fusion over the normed token embedding and target hidden
     * row ({@code enc.enorm} / {@code enc.hnorm}), one decoder layer running the DeepSeek latent attention with the
     * absorbed {@code attn_k_b} / {@code attn_v_b} decompression, and the draft's own embedding and output head. The
     * projections read the residual width, not a concatenated input: the attention input of an EAGLE-1 drafter is
     * the fusion output alone.
     *
     * @param targetLayer the single {@code eagle.target_layers} entry
     * @param qLora       the query LoRA rank; 0 stores the plain {@code attn_q} projection instead
     * @param kvLora      the KV LoRA rank
     * @throws IllegalArgumentException when the geometry is not a one-layer drafter, the ranks are invalid, or the
     *                                  attention geometry is invalid
     */
    public static GgufTestWriter eagleMla(SyntheticModels.Config c, int targetLayer, int qLora, int kvLora) {
        return eagleMla(c, targetLayer, qLora, kvLora, false);
    }

    /**
     * {@link #eagleMla(SyntheticModels.Config, int, int, int)} whose FFN is the DeepSeek routed-expert block
     * ({@code moe}) rather than a dense gated FFN: every expert of {@code c.nExpert()} experts is
     * {@code c.nFfExp()} wide, one shared expert is summed beside them and routing is softmax.
     *
     * @param moe whether the layer runs routed experts
     * @throws IllegalArgumentException when the expert geometry is missing or the attention geometry is invalid
     */
    public static GgufTestWriter eagleMla(SyntheticModels.Config c, int targetLayer, int qLora, int kvLora,
                                          boolean moe) {
        requireAttention(c, "eagle");
        if (c.nLayer() != 1) {
            throw new IllegalArgumentException("an EAGLE-1 drafter holds one decoder layer, got " + c.nLayer());
        }
        if (targetLayer < 0) {
            throw new IllegalArgumentException("eagle target layer must be >= 0, got " + targetLayer);
        }
        if (qLora < 0 || kvLora <= 0 || c.headDim() <= c.nRot()) {
            throw new IllegalArgumentException("eagle MLA needs a non-negative query rank, a positive KV rank and"
                    + " a non-rotary head part");
        }
        if (moe && (c.nExpert() <= 0 || c.nFfExp() <= 0)) {
            throw new IllegalArgumentException("eagle MLA needs positive expert count and width for a MoE draft");
        }
        long qWidth = (long) c.nHead() * c.headDim();
        GgufTestWriter w = base("eagle", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), kvLora + c.nRot());
        w.u32(w.archKey("attention.value_length"), kvLora);
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), c.headDim());
        if (qLora > 0) {
            w.u32(w.archKey("attention.q_lora_rank"), qLora);
        }
        w.u32(w.archKey("attention.kv_lora_rank"), kvLora);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.i32Array(w.archKey("target_layers"), targetLayer);
        w.u32(w.archKey("target_hidden_size"), c.nEmbd());
        if (moe) {
            w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
            w.u32(w.archKey("expert_count"), c.nExpert());
            w.u32(w.archKey("expert_used_count"), Math.min(2, c.nExpert()));
            w.u32(w.archKey("expert_shared_count"), 1);
            w.u32(w.archKey("leading_dense_block_count"), 0);
            w.u32(w.archKey("expert_gating_func"), 1);
        }

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", 2L * c.nEmbd(), c.nEmbd());
        t.ones("enc.enorm.weight", c.nEmbd());
        t.ones("enc.hnorm.weight", c.nEmbd());
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        if (qLora > 0) {
            t.matrix(blk(0, "attn_q_a.weight"), c.nEmbd(), qLora);
            t.ones(blk(0, "attn_q_a_norm.weight"), qLora);
            t.matrix(blk(0, "attn_q_b.weight"), qLora, qWidth);
        } else {
            t.matrix(blk(0, "attn_q.weight"), c.nEmbd(), qWidth);
        }
        t.matrix(blk(0, "attn_kv_a_mqa.weight"), c.nEmbd(), kvLora + c.nRot());
        t.ones(blk(0, "attn_kv_a_norm.weight"), kvLora);
        t.matrix(blk(0, "attn_k_b.weight"), c.headDim() - c.nRot(), kvLora, c.nHead());
        t.matrix(blk(0, "attn_v_b.weight"), kvLora, c.headDim(), c.nHead());
        t.matrix(blk(0, "attn_output.weight"), qWidth, c.nEmbd());
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        if (moe) {
            t.random(blk(0, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
            t.matrix(blk(0, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(0, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
            t.matrix(blk(0, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(0, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
            t.matrix(blk(0, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
            t.matrix(blk(0, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
        } else {
            t.gatedFfn(0, c.nFf());
        }
        return w;
    }

    /**
     * A tiny concat-fc draft in vLLM's Cohere layout ({@code cohere_eagle.EagleCohereForCausalLM}): the biased
     * {@code fc} over {@code [embedding | target hidden]}, one LayerNorm block with the parallel residual (no
     * {@code ffn_norm}), the final LayerNorm, and its own embedding and LM head ({@code EagleCohereForCausalLM}
     * shares the target's, a converted file may carry copies). {@code logit_scale} multiplies the draft logits when
     * it is not 1.
     *
     * @throws IllegalArgumentException when the draft is not one layer or the attention geometry is invalid
     */
    public static GgufTestWriter eagleFcCohere(SyntheticModels.Config c, int targetLayer, float logitScale) {
        requireAttention(c, "eagle-cohere");
        requireDraft(c, "eagle-cohere");
        GgufTestWriter w = base("eagle-cohere", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.headDim());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_epsilon"), c.normEps());
        concatFcTarget(w, c, targetLayer);
        if (logitScale != 1f) {
            w.f32(w.archKey("logit_scale"), logitScale);
        }
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", 2L * c.nEmbd(), c.nEmbd());
        t.bias("fc.bias", c.nEmbd());
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        t.qkvo(0, (long) c.nHead() * c.headDim());
        t.gatedFfn(0, c.nFf());
        return w;
    }

    /**
     * A tiny concat-fc draft in vLLM's MiniCPM layout ({@code minicpm_eagle.EagleMiniCPMForCausalLM}): the
     * {@code scale_emb} embedding, the {@code input_norm1} / {@code input_norm2} fusion-part norms, one layer whose
     * residual branches carry {@code residual_scale}, and the {@code output_hidden_scale} division. There is no
     * final norm (vLLM's forward returns the layer output scaled).
     */
    public static GgufTestWriter eagleFcMiniCpm(SyntheticModels.Config c, int targetLayer, float scaleEmb,
                                                float residualScale, float outputHiddenScale) {
        requireAttention(c, "eagle-minicpm");
        requireDraft(c, "eagle-minicpm");
        GgufTestWriter w = base("eagle-minicpm", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.headDim());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        concatFcTarget(w, c, targetLayer);
        w.f32(w.archKey("embedding_scale"), scaleEmb);
        w.f32(w.archKey("residual_scale"), residualScale);
        w.f32(w.archKey("output_hidden_scale"), outputHiddenScale);
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", 2L * c.nEmbd(), c.nEmbd());
        t.ones("enc.input_norm1.weight", c.nEmbd());
        t.ones("enc.input_norm2.weight", c.nEmbd());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        t.qkvo(0, (long) c.nHead() * c.headDim());
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(0, c.nFf());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /**
     * A tiny concat-fc draft in vLLM's Llama 4 layout ({@code llama4_eagle.EagleLlama4ForCausalLM}): the draft
     * layer's {@code no_rope_layers} entry selects RoPE or NoPE (the NoPE layer carries the 8192-token attention
     * temperature), {@code use_qk_norm} the weightless Q/K norm, and the checkpoint has its own embedding and LM
     * head.
     */
    public static GgufTestWriter eagleFcLlama4(SyntheticModels.Config c, int targetLayer, boolean rope,
                                               boolean qkNorm) {
        requireAttention(c, "eagle-llama4");
        requireDraft(c, "eagle-llama4");
        GgufTestWriter w = base("eagle-llama4", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.headDim());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        concatFcTarget(w, c, targetLayer);
        w.u32(w.archKey("attention.sliding_window"), 8192);
        w.boolArray(w.archKey("attention.sliding_window_pattern"), true);
        w.boolArray(w.archKey("attention.rope_pattern"), rope);
        w.bool(w.archKey("attention.use_qk_norm"), qkNorm);
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", 2L * c.nEmbd(), c.nEmbd());
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        t.qkvo(0, (long) c.nHead() * c.headDim());
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(0, c.nFf());
        return w;
    }

    /**
     * A tiny concat-fc draft in vLLM's Mistral Large 3 layout ({@code mistral_large_3_eagle.EagleMistralLarge3ForCausalLM}):
     * the fusion projection over {@code [embedding | target hidden]} and one dense layer of DeepSeek multi-head latent
     * attention with the absorbed {@code attn_k_b} / {@code attn_v_b} decompression, and its own embedding and LM
     * head ({@code EagleMistralLarge3ForCausalLM} shares the target's).
     *
     * @param qLora  the query LoRA rank ({@code attention.q_lora_rank})
     * @param kvLora the KV LoRA rank ({@code attention.kv_lora_rank})
     */
    public static GgufTestWriter eagleFcMistralLarge3(SyntheticModels.Config c, int targetLayer, int qLora,
                                                      int kvLora) {
        requireAttention(c, "eagle-mistral-large3");
        requireDraft(c, "eagle-mistral-large3");
        if (qLora <= 0 || kvLora <= 0 || c.headDim() <= c.nRot()) {
            throw new IllegalArgumentException("eagle-mistral-large3 needs positive LoRA ranks and a non-rotary"
                    + " head part");
        }
        GgufTestWriter w = base("eagle-mistral-large3", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), kvLora + c.nRot());
        w.u32(w.archKey("attention.value_length"), kvLora);
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), c.headDim());
        w.u32(w.archKey("attention.q_lora_rank"), qLora);
        w.u32(w.archKey("attention.kv_lora_rank"), kvLora);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        concatFcTarget(w, c, targetLayer);
        long input = c.nEmbd();
        long qWidth = (long) c.nHead() * c.headDim();
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", 2L * c.nEmbd(), c.nEmbd());
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        t.matrix(blk(0, "attn_q_a.weight"), input, qLora);
        t.ones(blk(0, "attn_q_a_norm.weight"), qLora);
        t.matrix(blk(0, "attn_q_b.weight"), qLora, qWidth);
        t.matrix(blk(0, "attn_kv_a_mqa.weight"), input, kvLora + c.nRot());
        t.ones(blk(0, "attn_kv_a_norm.weight"), kvLora);
        t.matrix(blk(0, "attn_k_b.weight"), c.headDim() - c.nRot(), kvLora, c.nHead());
        t.matrix(blk(0, "attn_v_b.weight"), kvLora, c.headDim(), c.nHead());
        t.matrix(blk(0, "attn_output.weight"), qWidth, c.nEmbd());
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(0, c.nFf());
        return w;
    }

    /** The concat-fc draft keys every family shares: one target tap of the drafter's width. */
    private static void concatFcTarget(GgufTestWriter w, SyntheticModels.Config c, int targetLayer) {
        if (targetLayer < 0) {
            throw new IllegalArgumentException("the target layer must be >= 0, got " + targetLayer);
        }
        w.i32Array(w.archKey("target_layers"), targetLayer);
        w.u32(w.archKey("target_hidden_size"), c.nEmbd());
    }

    private static void requireDraft(SyntheticModels.Config c, String arch) {
        if (c.nLayer() != 1) {
            throw new IllegalArgumentException("a " + arch + " drafter holds one decoder layer, got " + c.nLayer());
        }
    }

    /**
     * A tiny EAGLE-3 draft in the Qwen3 layout of {@code qwen3_eagle3}: the decoder layer of {@link #eagle3} with
     * Qwen3's per-head Q/K norms ({@code attn_q_norm} / {@code attn_k_norm}), its q/k/v projection biases
     * ({@code attention_bias}), the per-tap encoder norms of {@code fc_norm} when {@code fcNorm} is set, the
     * whole-input encoder norm of {@code norm_before_fc} when {@code normBeforeFc} is set, and the draft
     * {@code logit_scale} when it is not 1.
     *
     * @param targetLayers the three target tap points, in the order the encoder concatenates them
     * @param fcNorm       whether the encoder normalizes each target tap before the fusion
     * @param logitScale   the logit scale written to the file, or 1 for none
     * @param normBeforeFc whether the encoder RMS-norms the whole fusion input before {@code fc}
     * @throws IllegalArgumentException when the geometry is not a one-layer drafter, the three taps are missing, or
     *                                  the attention geometry is invalid
     */
    public static GgufTestWriter eagle3Qwen3(SyntheticModels.Config c, int[] targetLayers, boolean fcNorm,
                                             float logitScale, boolean normBeforeFc) {
        requireAttention(c, "eagle3");
        if (c.nLayer() != 1) {
            throw new IllegalArgumentException("an EAGLE-3 drafter holds one decoder layer, got " + c.nLayer());
        }
        if (targetLayers.length != EAGLE3_TARGET_LAYERS) {
            throw new IllegalArgumentException("eagle3 needs exactly " + EAGLE3_TARGET_LAYERS + " target layers, got "
                    + targetLayers.length);
        }
        GgufTestWriter w = base("eagle3", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.headDim());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.i32Array(w.archKey("target_layers"), targetLayers);
        w.u32(w.archKey("target_hidden_size"), c.nEmbd());
        if (logitScale != 1f) {
            w.f32(w.archKey("logit_scale"), logitScale);
        }
        if (normBeforeFc) {
            w.bool(w.archKey("norm_before_fc"), true);
        }
        long input = 2L * c.nEmbd();
        long qRows = (long) c.nHead() * c.headDim();
        long kvRows = (long) c.nHeadKv() * c.headDim();
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", (long) EAGLE3_TARGET_LAYERS * c.nEmbd(), c.nEmbd());
        if (normBeforeFc) {
            t.ones("enc.output_norm.weight", (long) EAGLE3_TARGET_LAYERS * c.nEmbd());
        }
        if (fcNorm) {
            for (int k = 0; k < EAGLE3_TARGET_LAYERS; k++) {
                t.ones("fc_norm." + k + ".weight", c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        t.ones(blk(0, "attn_norm.weight"), c.nEmbd());
        t.ones(blk(0, "attn_norm_2.weight"), c.nEmbd());
        t.matrix(blk(0, "attn_q.weight"), input, qRows);
        t.matrix(blk(0, "attn_k.weight"), input, kvRows);
        t.matrix(blk(0, "attn_v.weight"), input, kvRows);
        t.matrix(blk(0, "attn_output.weight"), qRows, c.nEmbd());
        t.ones(blk(0, "attn_q_norm.weight"), c.headDim());
        t.ones(blk(0, "attn_k_norm.weight"), c.headDim());
        t.bias(blk(0, "attn_q.bias"), qRows);
        t.bias(blk(0, "attn_k.bias"), kvRows);
        t.bias(blk(0, "attn_v.bias"), kvRows);
        t.ones(blk(0, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(0, c.nFf());
        return w;
    }
}
