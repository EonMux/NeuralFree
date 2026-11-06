package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocabWords;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * The d1 decision fixtures of {@link SyntheticModels}: the {@code lfm2-d1} and {@code lfm3-d1} trunk-logit reads, the
 * {@code lfm2-d1-omni} mask-marker read over one modern-bert-style decision block and the {@code lfm3-d1-omni}
 * variant. All run non-causal trunks. Sources: the llama.cpp lfm2/lfm3 models and the {@code decision.*} keys
 * {@code HparamsLoader} reads back.
 */
public final class SyntheticD1Models {

    private SyntheticD1Models() {
    }

    // ================================================================== lfm2 decision

    /** The lfm2-d1 fixture: the plain LFM2 trunk with the d1 decision type and the d1 systemone template. */
    public static GgufTestWriter lfm2Decision() {
        return lfm2Decision(SyntheticModels.Config.lfm2());
    }

    /**
     * LLM_ARCH_LFM2 with the {@code lfm2-d1} decision type (lfm2.py:89-137): the text d1 checkpoints carry no
     * decision blocks, so the answer codes are read from the model's own output logits at the last prompt token.
     */
    public static GgufTestWriter lfm2Decision(SyntheticModels.Config c) {
        requireAttention(c, "lfm2 decision");
        GgufTestWriter w = base("lfm2", c);
        long[] heads = new long[c.nLayer()];
        long[] headsKv = new long[c.nLayer()];
        for (int il = 0; il < c.nLayer(); il++) {
            boolean attention = il % 2 == 1;
            heads[il] = attention ? c.nHead() : 0;
            headsKv[il] = attention ? c.nHeadKv() : 0;
        }
        w.u32Array(w.archKey("attention.head_count"), heads);
        w.u32Array(w.archKey("attention.head_count_kv"), headsKv);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("shortconv.l_cache"), c.ssmConv());
        w.str(w.archKey("decision.type"), "lfm2-d1");
        w.str("tokenizer.chat_template.systemone", SyntheticModels.LFM2_D1_TEMPLATE);
        bpeVocab(w, c.nVocab(), "lfm2", new String[] {"<|startoftext|>", "<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.ones("token_embd_norm.weight", c.nEmbd());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            if (il % 2 == 1) {
                t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
                t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
                t.qkvo(il, (long) c.nHead() * c.headDim());
            } else {
                t.matrix(blk(il, "shortconv.conv.weight"), c.ssmConv(), c.nEmbd());
                t.matrix(blk(il, "shortconv.in_proj.weight"), c.nEmbd(), 3L * c.nEmbd());
                t.matrix(blk(il, "shortconv.out_proj.weight"), c.nEmbd(), c.nEmbd());
            }
        }
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    // ============================================================ lfm2-d1-omni

    /** The lfm2-d1-omni fixture: the LFM2 trunk, one modern-bert-style decision block, the mask markers. */
    public static GgufTestWriter lfm2D1Omni() {
        // the centred short convolution of the decision read runs with a kernel of 3 (lfm2.cpp:369)
        return lfm2D1Omni(SyntheticModels.Config.lfm2().toBuilder().nLayer(5).nCtxTrain(512).ssmConv(3).build());
    }

    /**
     * LLM_ARCH_LFM2 with the {@code lfm2-d1-omni} decision type (lfm2.py:195-258): the trunk carries one head block
     * after the body layers ({@code decision.block_count}), run non-causally on the final hidden rows; the read
     * scores the question-type column of every option's mask marker through the media.
     */
    public static GgufTestWriter lfm2D1Omni(SyntheticModels.Config c) {
        requireAttention(c, "lfm2-d1-omni");
        requirePositive("lfm2-d1-omni shortconv.l_cache", c.ssmConv());
        if (c.nLayer() < 2) {
            throw new IllegalArgumentException("lfm2-d1-omni needs a body layer and a decision block, got "
                    + c.nLayer() + " layers");
        }
        GgufTestWriter w = base("lfm2", c);
        long[] heads = new long[c.nLayer()];
        long[] headsKv = new long[c.nLayer()];
        for (int il = 0; il < c.nLayer() - 1; il++) {
            boolean attention = il % 2 == 1;
            heads[il] = attention ? c.nHead() : 0;
            headsKv[il] = attention ? c.nHeadKv() : 0;
        }
        heads[c.nLayer() - 1] = c.nHead();
        headsKv[c.nLayer() - 1] = c.nHeadKv();
        w.u32Array(w.archKey("attention.head_count"), heads);
        w.u32Array(w.archKey("attention.head_count_kv"), headsKv);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("attention.layer_norm_epsilon"), 1e-5f);
        w.u32(w.archKey("shortconv.l_cache"), c.ssmConv());
        w.bool(w.archKey("attention.causal"), false);
        w.u32("tokenizer.ggml.token_type_count", 3);
        w.str(w.archKey("decision.type"), "lfm2-d1-omni");
        w.u32(w.archKey("decision.block_count"), 1);
        w.str("tokenizer.chat_template.systemone", SyntheticModels.LFM2_D1_OMNI_TEMPLATE);
        w.f32(w.archKey("decision.temperature.noul"), 0.7f);
        w.f32(w.archKey("decision.temperature.choice"), 0.9f);
        w.f32(w.archKey("decision.temperature.score"), 0.6f);
        w.f32(w.archKey("decision.temperature.choice.3_5"), 0.8f);
        bpeVocab(w, c.nVocab(), "lfm2", new String[] {"<|startoftext|>", "<|endoftext|>", "<|mask|>",
                "<|reserved_7|>", "<|reserved_8|>", "<|reserved_9|>", "<|reserved_10|>", "<|reserved_11|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .u32("tokenizer.ggml.mask_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.ones("token_embd_norm.weight", c.nEmbd());
        int head = c.nLayer() - 1;
        for (int il = 0; il < head; il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            if (il % 2 == 1) {
                t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
                t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
                t.qkvo(il, (long) c.nHead() * c.headDim());
            } else {
                t.matrix(blk(il, "shortconv.conv.weight"), c.ssmConv(), c.nEmbd());
                t.matrix(blk(il, "shortconv.in_proj.weight"), c.nEmbd(), 3L * c.nEmbd());
                t.matrix(blk(il, "shortconv.out_proj.weight"), c.nEmbd(), c.nEmbd());
            }
        }
        writeDecisionBlock(t, c, head);
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    /** One modern-bert-style decision block with its biased norms, the type embeddings and the classifier head. */
    private static void writeDecisionBlock(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        long kvColumns = (long) c.nHeadKv() * c.headDim();
        t.layerNorm(blk(il, "attn_norm"), c.nEmbd());
        t.matrix(blk(il, "attn_qkv.weight"), c.nEmbd(), c.nEmbd() + 2L * kvColumns);
        t.bias(blk(il, "attn_qkv.bias"), c.nEmbd() + 2L * kvColumns);
        t.matrix(blk(il, "attn_output.weight"), c.nEmbd(), c.nEmbd());
        t.bias(blk(il, "attn_output.bias"), c.nEmbd());
        t.layerNorm(blk(il, "ffn_norm"), c.nEmbd());
        t.matrix(blk(il, "ffn_up.weight"), c.nEmbd(), c.nFf());
        t.bias(blk(il, "ffn_up.bias"), c.nFf());
        t.matrix(blk(il, "ffn_down.weight"), c.nFf(), c.nEmbd());
        t.bias(blk(il, "ffn_down.bias"), c.nEmbd());
        t.matrix("token_types.weight", c.nEmbd(), 3);
        t.matrix("cls.weight", c.nEmbd(), c.nEmbd());
        t.bias("cls.bias", c.nEmbd());
        t.layerNorm("cls.norm", c.nEmbd());
        t.matrix("cls.output.weight", c.nEmbd(), 1);
        t.bias("cls.output.bias", 1);
    }

    // ============================================================ lfm3 d1 families

    /** The lfm3-d1 fixture: the {@code lfm3-d1} decision type on the non-causal LFM3 trunk (the VL d1 family). */
    public static GgufTestWriter lfm3D1() {
        // the centred short convolution of the non-causal trunk runs with an odd kernel
        return lfm3D1(SyntheticModels.Config.lfm3().toBuilder().nCtxTrain(1024).ssmConv(3).build());
    }

    /** {@link #lfm3D1()} over {@code c}. */
    public static GgufTestWriter lfm3D1(SyntheticModels.Config c) {
        return lfm3D1("lfm3-d1", c);
    }

    /** The lfm3-d1-omni fixture: the non-causal LFM3 trunk with the d1 readout and the think marker. */
    public static GgufTestWriter lfm3D1Omni() {
        // the centred short convolution of the non-causal trunk runs with an odd kernel
        return lfm3D1Omni(SyntheticModels.Config.lfm3().toBuilder().nCtxTrain(1024).ssmConv(3).build());
    }

    /** {@link #lfm3D1Omni()} over {@code c}. */
    public static GgufTestWriter lfm3D1Omni(SyntheticModels.Config c) {
        return lfm3D1("lfm3-d1-omni", c);
    }

    /**
     * LLM_ARCH_LFM3 with an lfm3 d1 decision type (lfm3.py:100-200): the stock trunk runs non-causally and the answer
     * codes are read from its output logits at the last prompt token; the temperatures come from the calibration and
     * the score answer is the median level.
     */
    private static GgufTestWriter lfm3D1(String type, SyntheticModels.Config c) {
        requireAttention(c, type);
        requirePositive(type + " shortconv.l_cache", c.ssmConv());
        requirePositive(type + " sliding window", c.slidingWindow());
        if (c.nLayer() != 4) {
            throw new IllegalArgumentException(type + " fixture has 4 layers (attn, conv, attn, conv), got "
                    + c.nLayer() + " layers");
        }
        GgufTestWriter w = base("lfm3", c);
        w.u32Array(w.archKey("attention.head_count"), c.nHead(), 0, c.nHead(), 0);
        w.u32Array(w.archKey("attention.head_count_kv"), c.nHeadKv(), 0, c.nHeadKv(), 0);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("shortconv.l_cache"), c.ssmConv());
        w.u32(w.archKey("attention.sliding_window"), c.slidingWindow());
        w.boolArray(w.archKey("attention.sliding_window_pattern"), true, false, false, false);
        w.f32(w.archKey("rope.freq_base_swa"), c.ropeFreqBase());
        w.bool(w.archKey("attention.causal"), false);
        w.str(w.archKey("decision.type"), type);
        w.f32(w.archKey("decision.temperature.noul"), 0.5f);
        w.f32(w.archKey("decision.temperature.choice"), 0.7f);
        w.f32(w.archKey("decision.temperature.score"), 0.4f);
        w.str("tokenizer.chat_template.systemone", SyntheticModels.LFM2_D1_TEMPLATE + "<think></think>");
        bpeVocabWords(w, c.nVocab(), "lfm2", new String[] {"<|startoftext|>", "<|endoftext|>"}, new String[] {
                "yes", "Yes", "YES", "no", "No", "NO"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("rope_freqs.weight", c.nRot() / 2);
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            if (il % 2 == 0) {
                long kvRows = (long) c.nHeadKv() * c.headDim();
                t.matrix(blk(il, "attn_q.weight"), c.nEmbd(), (long) c.nHead() * c.headDim());
                t.matrix(blk(il, "attn_k.weight"), c.nEmbd(), kvRows);
                t.matrix(blk(il, "attn_v.weight"), c.nEmbd(), kvRows);
                t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
                t.matrix(blk(il, "attn_gate.weight"), c.nEmbd(), c.nHead());
                t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
                t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            } else {
                t.matrix(blk(il, "shortconv.conv.weight"), c.ssmConv(), c.nEmbd());
                t.matrix(blk(il, "shortconv.in_proj.weight"), c.nEmbd(), 3L * c.nEmbd());
                t.matrix(blk(il, "shortconv.out_proj.weight"), c.nEmbd(), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }
}
