package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocabWords;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.wpmVocab;

/**
 * Decision-readout fixtures of {@link SyntheticModels} on causal trunks: vllm-decision2 (a per-option decision head),
 * the label and kev reads over the model's own output logits, h2o-lightning (label metadata read), minicpm-ocr-jev
 * (pooled two-class read) and modern-bert (the laya read over a non-causal encoder with a biased decision block).
 * The d1 families live in {@link SyntheticD1Models}. Sources: the llama.cpp models of each fixture and the
 * {@code decision.*} keys {@code HparamsLoader} reads back.
 */
public final class SyntheticDecisionModels {

    private SyntheticDecisionModels() {
    }

    // ================================================================== vllm-decision2

    /** LLM_ARCH_VLLM_DECISION2 over the 2-layer GQA config, with the systemone template and calibration keys. */
    public static GgufTestWriter vllmDecision2() {
        return vllmDecision2(SyntheticModels.Config.qwen2(), SyntheticModels.VLLM_DECISION2_TEMPLATE);
    }

    /** LLM_ARCH_VLLM_DECISION2 over {@code c}, with the systemone template and calibration keys. */
    public static GgufTestWriter vllmDecision2(SyntheticModels.Config c) {
        return vllmDecision2(c, SyntheticModels.VLLM_DECISION2_TEMPLATE);
    }

    /** LLM_ARCH_VLLM_DECISION2 over {@code c} with {@code systemoneTemplate} in the tokenizer metadata. */
    public static GgufTestWriter vllmDecision2(SyntheticModels.Config c, String systemoneTemplate) {
        requireAttention(c, "vllm-decision2");
        if (c.nRot() <= 0 || c.nRot() > c.headDim() || c.nRot() % 2 != 0) {
            throw new IllegalArgumentException("vllm-decision2 rope dimension must be even and in (0, head dim],"
                    + " got " + c.nRot());
        }
        GgufTestWriter w = base("vllm-decision2", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("attention.layer_norm_epsilon"), c.normEps());
        w.str(w.archKey("decision.type"), "vllm-decision2");
        w.str("tokenizer.chat_template.systemone", systemoneTemplate);
        bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|im_start|>", "<|im_end|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());

        // the per-option decision head (vllm-decision2.cpp:8-39): n_embd_h is the head dim
        long h = c.headDim();
        t.matrix("decision.key.weight", c.nEmbd(), h);
        t.layerNorm("decision.option_norm", c.nEmbd());
        t.layerNorm("decision.query_norm", c.nEmbd());
        t.matrix("decision.query.weight", c.nEmbd(), h);
        t.matrix("decision.option_mlp.weight", c.nEmbd(), h);
        t.random("decision.option_mlp.bias", -0.02f, 0.02f, h);
        t.matrix("decision.query_mlp.weight", c.nEmbd(), h);
        t.matrix("decision.scorer_out.weight", h, 1);

        // the server's calibration keys: bucketed choice and bare score temperatures, offsets for 4 levels
        w.f32(w.archKey("decision.temperature.choice.3_5"), 0.7f);
        w.f32(w.archKey("decision.temperature.score"), 0.5f);
        w.f32(w.archKey("decision.score_bias.4.0"), -1.0f);
        w.f32(w.archKey("decision.score_bias.4.3"), 1.0f);
        return w;
    }

    // ================================================================== label decision families

    /** LLM_ARCH_QWEN2 with a label decision readout: an lm_head, a systemone template and the type metadata. */
    public static GgufTestWriter decisionLabels(String type) {
        return decisionLabels(type, SyntheticModels.Config.qwen2());
    }

    /** {@link #decisionLabels(String)} over {@code c}. */
    public static GgufTestWriter decisionLabels(String type, SyntheticModels.Config c) {
        return decisionLabels(type, c, SyntheticModels.LABEL_TEMPLATE);
    }

    /**
     * {@link #decisionLabels(String)} with {@code systemoneTemplate} in the tokenizer metadata; lev also carries its
     * per-option-count temperature buckets.
     */
    public static GgufTestWriter decisionLabels(String type, SyntheticModels.Config c, String systemoneTemplate) {
        return labelDecision(type, c, systemoneTemplate,
                new String[] {"<|endoftext|>", "<|im_start|>", "<|im_end|>"});
    }

    /** {@link #decisionLabels(String, SyntheticModels.Config, String)} with {@code specials} added to the vocabulary. */
    private static GgufTestWriter labelDecision(String type, SyntheticModels.Config c, String systemoneTemplate,
                                                String[] specials) {
        requireAttention(c, type);
        GgufTestWriter w = base("qwen2", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.str(w.archKey("decision.type"), type);
        w.str("tokenizer.chat_template.systemone", systemoneTemplate);
        if ("lev".equals(type)) {
            w.f32(w.archKey("decision.temperature.choice.small"), 0.5f);
            w.f32(w.archKey("decision.temperature.choice.mid"), 1.5f);
            w.f32(w.archKey("decision.temperature.choice.large"), 2.0f);
            w.f32(w.archKey("decision.temperature.noul"), 0.5f);
        }
        bpeVocab(w, c.nVocab(), "qwen2", specials);
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    // ================================================================== kev

    /** LLM_ARCH_QWEN2 with the kev decision readout: the {@code <|box_end|>} marker token of its vocabulary. */
    public static GgufTestWriter decisionKev() {
        return decisionKev(SyntheticModels.Config.qwen2());
    }

    /** {@link #decisionKev()} over {@code c}. */
    public static GgufTestWriter decisionKev(SyntheticModels.Config c) {
        return decisionKev(c, SyntheticModels.KEV_TEMPLATE);
    }

    /** {@link #decisionKev(SyntheticModels.Config)} with {@code systemoneTemplate} in place of the standard one. */
    public static GgufTestWriter decisionKev(SyntheticModels.Config c, String systemoneTemplate) {
        return labelDecision("kev", c, systemoneTemplate,
                new String[] {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|box_end|>"});
    }

    // ================================================================== modern-bert decision

    /** The modern-bert decision fixture: two encoder blocks and one biased decision block, the laya readout. */
    public static GgufTestWriter modernBertDecision() {
        return modernBertDecision(SyntheticModels.Config.bert().toBuilder().nLayer(3).nFf(256).build());
    }

    /**
     * LLM_ARCH_MODERN_BERT with {@code nLayer - 1} encoder blocks and one decision block (modern-bert.cpp:61-132):
     * a fused GEGLU-erf encoder body with RoPE, the biased pre-norm head block, the three question-type embeddings
     * and the classifier head, read at the [MASK] markers of {@link SyntheticModels#LAYA_TEMPLATE}.
     */
    public static GgufTestWriter modernBertDecision(SyntheticModels.Config c) {
        requireAttention(c, "modern-bert decision");
        if (c.nLayer() < 2) {
            throw new IllegalArgumentException("modern-bert decision needs an encoder block and a decision block, got "
                    + c.nLayer() + " layers");
        }
        if (c.nHeadKv() != c.nHead() || (long) c.nHead() * c.headDim() != c.nEmbd()) {
            throw new IllegalArgumentException("modern-bert requires n_head_kv == n_head and n_head * head_dim =="
                    + " n_embd");
        }
        GgufTestWriter w = base("modern-bert", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_epsilon"), c.normEps());
        w.u32(w.archKey("decision.block_count"), 1);
        w.u32(w.archKey("decision.max_head_tokens"), 256);
        w.str(w.archKey("decision.type"), "laya");
        w.str("tokenizer.chat_template.systemone", SyntheticModels.LAYA_TEMPLATE);
        wpmVocab(w, c.nVocab());
        w.u32("tokenizer.ggml.token_type_count", 3)
                .u32("tokenizer.ggml.padding_token_id", 0).u32("tokenizer.ggml.unknown_token_id", 1)
                .u32("tokenizer.ggml.cls_token_id", 2).u32("tokenizer.ggml.seperator_token_id", 3)
                .u32("tokenizer.ggml.mask_token_id", 4);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.ones("token_embd_norm.weight", c.nEmbd());
        t.ones("output_norm.weight", c.nEmbd());
        int head = c.nLayer() - 1;
        for (int il = 0; il < head; il++) {
            if (il > 0) {
                t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            }
            t.matrix(blk(il, "attn_qkv.weight"), c.nEmbd(), 3L * c.nEmbd());
            t.matrix(blk(il, "attn_output.weight"), c.nEmbd(), c.nEmbd());
            t.matrix(blk(il, "ffn_up.weight"), c.nEmbd(), 2L * c.nFf());
            t.matrix(blk(il, "ffn_down.weight"), c.nFf(), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
        }
        t.layerNorm(blk(head, "attn_norm"), c.nEmbd());
        t.matrix(blk(head, "attn_qkv.weight"), c.nEmbd(), 3L * c.nEmbd());
        t.bias(blk(head, "attn_qkv.bias"), 3L * c.nEmbd());
        t.matrix(blk(head, "attn_output.weight"), c.nEmbd(), c.nEmbd());
        t.bias(blk(head, "attn_output.bias"), c.nEmbd());
        t.layerNorm(blk(head, "ffn_norm"), c.nEmbd());
        t.matrix(blk(head, "ffn_up.weight"), c.nEmbd(), c.nFf());
        t.bias(blk(head, "ffn_up.bias"), c.nFf());
        t.matrix(blk(head, "ffn_down.weight"), c.nFf(), c.nEmbd());
        t.bias(blk(head, "ffn_down.bias"), c.nEmbd());
        t.matrix("token_types.weight", c.nEmbd(), 3);
        t.matrix("cls.weight", c.nEmbd(), c.nEmbd());
        t.bias("cls.bias", c.nEmbd());
        t.layerNorm("cls.norm", c.nEmbd());
        t.matrix("cls.output.weight", c.nEmbd(), 1);
        t.bias("cls.output.bias", 1);
        return w;
    }

    // ================================================================== h2o-lightning and minicpm-ocr-jev

    /** The h2o-lightning fixture: the label readout over {@code decision.labels}, the served prompt layout. */
    public static GgufTestWriter h2oLightning() {
        return h2oLightning(SyntheticModels.Config.qwen2());
    }

    /**
     * LLM_ARCH_QWEN2 with the {@code h2o-lightning} decision type: the option labels come from the {@code
     * decision.labels} metadata, one distinct token each with their leading space (Serve.labels), and the answers
     * are read from the model's output logits at the last prompt token.
     */
    public static GgufTestWriter h2oLightning(SyntheticModels.Config c) {
        requireAttention(c, "h2o-lightning");
        GgufTestWriter w = base("qwen2", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.str(w.archKey("decision.type"), "h2o-lightning");
        w.strArray(w.archKey("decision.labels"), " yes", " no");
        w.f32(w.archKey("decision.temperature.noul"), 0.6f);
        w.f32(w.archKey("decision.temperature.choice.3_5"), 0.8f);
        w.f32(w.archKey("decision.temperature.image"), 0.9f);
        w.str("tokenizer.chat_template.systemone", SyntheticModels.H2O_LIGHTNING_TEMPLATE);
        bpeVocabWords(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|im_start|>", "<|im_end|>",
                "<|special|>"}, new String[] {" yes", " no"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /** The minicpm-ocr-jev fixture: the pooled two-class readout over the fixed OCR question. */
    public static GgufTestWriter minicpmOcrJev() {
        return minicpmOcrJev(SyntheticModels.Config.qwen2());
    }

    /**
     * LLM_ARCH_QWEN2 with the {@code openbmb-minicpm-ocr-jev} decision type: LAST pooling over a two-output
     * classifier head (one score per class), no classifier scope — the pooled decision read answers the decision
     * routes only, and the class labels are the fixed inconsistent/consistent pair.
     */
    public static GgufTestWriter minicpmOcrJev(SyntheticModels.Config c) {
        requireAttention(c, "minicpm-ocr-jev");
        GgufTestWriter w = base("qwen2", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("pooling_type"), 3);
        w.u32(w.archKey("embedding_length_out"), 2);
        w.strArray(w.archKey("classifier.output_labels"), "inconsistent", "consistent");
        w.str(w.archKey("decision.type"), "openbmb-minicpm-ocr-jev");
        w.str("tokenizer.chat_template.systemone", SyntheticModels.MINICPM_OCR_JEV_TEMPLATE);
        bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|im_start|>", "<|im_end|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("cls.output.weight", c.nEmbd(), 2);
        t.bias("cls.output.bias", 2);
        return w;
    }
}
