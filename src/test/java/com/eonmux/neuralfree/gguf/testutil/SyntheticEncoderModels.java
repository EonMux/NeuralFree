package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.spmVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.wpmVocab;

/**
 * Encoder fixtures of {@link SyntheticModels}: bert (non-causal encoder with learned positions, mean pooling) and
 * t5 (encoder-decoder with relative-position bias and cross-attention). Sources: llama.cpp
 * src/models/{bert,t5}.cpp {@code load_arch_hparams}/{@code load_arch_tensors}.
 */
public final class SyntheticEncoderModels {

    /** Relative-position bucket count of the t5 fixture. */
    private static final int T5_BUCKETS = 32;

    private SyntheticEncoderModels() {
    }

    // ================================================================== bert

    public static GgufTestWriter bert() {
        return bert(SyntheticModels.Config.bert());
    }

    /**
     * LLM_ARCH_BERT: non-causal encoder with learned positions, token-type embeddings, post-norm
     * LayerNorm blocks with biases and mean pooling. Requires {@code n_head_kv == n_head} and
     * {@code n_head * head_dim == n_embd}; there is no output head.
     */
    public static GgufTestWriter bert(SyntheticModels.Config c) {
        requireAttention(c, "bert");
        if (c.nHeadKv() != c.nHead() || (long) c.nHead() * c.headDim() != c.nEmbd()) {
            throw new IllegalArgumentException("bert requires n_head_kv == n_head and n_head * head_dim == n_embd");
        }
        GgufTestWriter w = base("bert", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.f32(w.archKey("attention.layer_norm_epsilon"), c.normEps());
        w.bool(w.archKey("attention.causal"), false);
        w.u32(w.archKey("pooling_type"), SyntheticModels.POOLING_MEAN);
        wpmVocab(w, c.nVocab());
        w.u32("tokenizer.ggml.token_type_count", SyntheticModels.BERT_TOKEN_TYPES)
                .u32("tokenizer.ggml.padding_token_id", 0).u32("tokenizer.ggml.unknown_token_id", 1)
                .u32("tokenizer.ggml.cls_token_id", 2).u32("tokenizer.ggml.seperator_token_id", 3)
                .u32("tokenizer.ggml.mask_token_id", 4);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.random("token_types.weight", -0.1f, 0.1f, c.nEmbd(), SyntheticModels.BERT_TOKEN_TYPES);
        t.random("position_embd.weight", -0.1f, 0.1f, c.nEmbd(), c.nCtxTrain());
        t.layerNorm("token_embd_norm", c.nEmbd());
        for (int il = 0; il < c.nLayer(); il++) {
            for (String p : new String[] {"attn_q", "attn_k", "attn_v", "attn_output"}) {
                t.matrix(blk(il, p + ".weight"), c.nEmbd(), c.nEmbd());
                t.bias(blk(il, p + ".bias"), c.nEmbd());
            }
            t.layerNorm(blk(il, "attn_output_norm"), c.nEmbd());
            t.matrix(blk(il, "ffn_up.weight"), c.nEmbd(), c.nFf());
            t.bias(blk(il, "ffn_up.bias"), c.nFf());
            t.matrix(blk(il, "ffn_down.weight"), c.nFf(), c.nEmbd());
            t.bias(blk(il, "ffn_down.bias"), c.nEmbd());
            t.layerNorm(blk(il, "layer_output_norm"), c.nEmbd());
        }
        return w;
    }

    // ================================================================== t5

    public static GgufTestWriter t5() {
        return t5(SyntheticModels.Config.t5());
    }

    /**
     * LLM_ARCH_T5: an RMS-norm encoder stack and decoder trunk of {@code nLayer} layers each, gated-GELU FFNs, the
     * relative-position bias table on layer 0 of each stack, cross-attention in every decoder layer and an untied LM
     * head (t5.cpp:39-110). Requires {@code n_head_kv == n_head}; the decoder starts from
     * {@link SyntheticModels#T5_DECODER_START}.
     */
    public static GgufTestWriter t5(SyntheticModels.Config c) {
        requireAttention(c, "t5");
        if (c.nHeadKv() != c.nHead()) {
            throw new IllegalArgumentException("t5 requires n_head_kv == n_head");
        }
        GgufTestWriter w = base("t5", c);
        w.u32(w.archKey("decoder_block_count"), c.nLayer());
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.f32(w.archKey("attention.layer_norm_epsilon"), c.normEps());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("attention.relative_buckets_count"), T5_BUCKETS);
        w.u32(w.archKey("decoder_start_token_id"), SyntheticModels.T5_DECODER_START);
        spmVocab(w, c.nVocab(), new String[] {"<pad>", "</s>", "<unk>"},
                new int[] {SyntheticModels.TOKEN_TYPE_CONTROL, SyntheticModels.TOKEN_TYPE_CONTROL,
                        SyntheticModels.TOKEN_TYPE_UNKNOWN});
        w.u32("tokenizer.ggml.padding_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .u32("tokenizer.ggml.unknown_token_id", 2).bool("tokenizer.ggml.add_bos_token", false)
                .bool("tokenizer.ggml.add_eos_token", true);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long attnWidth = (long) c.nHead() * c.headDim();
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t5Attention(t, c, "enc.blk." + il + ".", "attn", il == 0, attnWidth);
            t5Ffn(t, c, "enc.blk." + il + ".");
        }
        t.ones("enc.output_norm.weight", c.nEmbd());
        for (int il = 0; il < c.nLayer(); il++) {
            String p = "dec.blk." + il + ".";
            t5Attention(t, c, p, "attn", il == 0, attnWidth);
            t5Attention(t, c, p, "cross_attn", false, attnWidth);
            t5Ffn(t, c, p);
        }
        t.ones("dec.output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /** One T5 attention of a block: its RMS norm, Q/K/V/O and, when {@code bias}, the relative-position table. */
    private static void t5Attention(SyntheticModels.Tensors t, SyntheticModels.Config c, String block, String kind,
                                    boolean bias, long width) {
        t.ones(block + kind + "_norm.weight", c.nEmbd());
        t.matrix(block + kind + "_q.weight", c.nEmbd(), width);
        t.matrix(block + kind + "_k.weight", c.nEmbd(), width);
        t.matrix(block + kind + "_v.weight", c.nEmbd(), width);
        t.matrix(block + kind + "_o.weight", width, c.nEmbd());
        if (bias) {
            t.random(block + kind + "_rel_b.weight", -0.5f, 0.5f, c.nHead(), T5_BUCKETS);
        }
    }

    /** One T5 gated FFN of a block with its RMS norm. */
    private static void t5Ffn(SyntheticModels.Tensors t, SyntheticModels.Config c, String block) {
        t.ones(block + "ffn_norm.weight", c.nEmbd());
        t.matrix(block + "ffn_gate.weight", c.nEmbd(), c.nFf());
        t.matrix(block + "ffn_up.weight", c.nEmbd(), c.nFf());
        t.matrix(block + "ffn_down.weight", c.nFf(), c.nEmbd());
    }
}
