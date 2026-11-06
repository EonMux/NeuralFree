package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.gemmaVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.spmVocab;

/**
 * Dense attention decoder fixtures of {@link SyntheticModels}: llama, qwen2 (with the MiMo MTP nextn head and the
 * dual-chunk attention variant), qwen2vl (M-RoPE sections), gemma2 (sliding-window attention) and limite (value
 * embeddings, XSA, MUDD). Sources: llama.cpp
 * src/models/{llama,qwen2,qwen2vl,gemma2,limite}.cpp {@code load_arch_hparams}/{@code load_arch_tensors},
 * llama.cpp src/llama-model.cpp {@code load_hparams}/{@code create_tensor_qkv}, and llama.cpp
 * gguf-py/gguf/gguf_writer.py for value types.
 */
public final class SyntheticDenseModels {

    private SyntheticDenseModels() {
    }

    // ================================================================== llama

    public static GgufTestWriter llama() {
        return llama(SyntheticModels.Config.llama(), 0);
    }

    public static GgufTestWriter llama(SyntheticModels.Config c) {
        return llama(c, 0);
    }

    /**
     * {@link #llama(SyntheticModels.Config)} with its scored pieces rotated by {@code pieceShift} ids: the same token
     * texts as {@link #llama(SyntheticModels.Config)}, but at other ids past the specials and byte tokens, so the two
     * files describe token-level-incompatible vocabularies.
     */
    public static GgufTestWriter llamaRotatedVocab(SyntheticModels.Config c, int pieceShift) {
        if (pieceShift == 0) {
            throw new IllegalArgumentException("a rotated vocabulary needs a non-zero piece shift");
        }
        return llama(c, pieceShift);
    }

    /** LLM_ARCH_LLAMA: dense GQA attention + SwiGLU FFN, SPM tokenizer. */
    private static GgufTestWriter llama(SyntheticModels.Config c, int pieceShift) {
        requireAttention(c, "llama");
        if (c.nRot() != c.headDim()) {
            throw new IllegalArgumentException("llama requires rope.dimension_count == head dim (llama-model.cpp "
                    + "load_hparams n_rot check), got " + c.nRot() + " vs " + c.headDim());
        }
        GgufTestWriter w = base("llama", c);
        attentionKeys(w, c, false);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        spmVocab(w, c.nVocab(), new String[] {"<unk>", "<s>", "</s>"},
                new int[] {SyntheticModels.TOKEN_TYPE_UNKNOWN, SyntheticModels.TOKEN_TYPE_CONTROL,
                        SyntheticModels.TOKEN_TYPE_CONTROL}, pieceShift);
        w.u32("tokenizer.ggml.unknown_token_id", 0).u32("tokenizer.ggml.bos_token_id", 1)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", true)
                .bool("tokenizer.ggml.add_eos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    // ================================================================== qwen2

    /**
     * LLM_ARCH_QWEN2 (qwen2.cpp, llama-model.cpp qwen2 tensors): dense GQA attention with the optional Q/K/V biases
     * and a SwiGLU FFN, followed by {@code nextn} appended blocks that carry the same body beside the
     * {@code nextn.eh_proj}, {@code nextn.enorm}, {@code nextn.hnorm} and {@code nextn.shared_head_norm} of a MiMo
     * MTP head (vLLM mimo_mtp.py); the blocks read the trunk's token table and output head.
     */
    public static GgufTestWriter qwen2Mtp(SyntheticModels.Config c, int nextn) {
        requireAttention(c, "qwen2");
        if (nextn < 0) {
            throw new IllegalArgumentException("qwen2Mtp needs a non-negative NextN block count, got " + nextn);
        }
        GgufTestWriter w = base("qwen2", c.toBuilder().nLayer(c.nLayer() + nextn).build());
        attentionKeys(w, c, false);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        if (nextn > 0) {
            w.u32(w.archKey("nextn_predict_layers"), nextn);
        }
        bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|user|>", "<|assistant|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .u32("tokenizer.ggml.padding_token_id", 0).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer() + nextn; il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.bias(blk(il, "attn_q.bias"), c.nEmbd());
            t.bias(blk(il, "attn_k.bias"), (long) c.nHeadKv() * c.headDim());
            t.bias(blk(il, "attn_v.bias"), (long) c.nHeadKv() * c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            if (il >= c.nLayer()) {
                t.matrix(blk(il, "nextn.eh_proj.weight"), 2L * c.nEmbd(), c.nEmbd());
                t.ones(blk(il, "nextn.enorm.weight"), c.nEmbd());
                t.ones(blk(il, "nextn.hnorm.weight"), c.nEmbd());
                t.ones(blk(il, "nextn.shared_head_norm.weight"), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /**
     * LLM_ARCH_QWEN2 whose softmax layers run Qwen2.5-1M dual-chunk attention: the qwen2 tensor set and the
     * NeuralFree {@code attention.dual_chunk.*} keys {@code HparamsLoader} reads back into the attention spec.
     *
     * @param chunkSize   {@code chunk_size} of the block
     * @param localSize   {@code local_size} (the chunk length is {@code chunk_size - local_size})
     * @param originalMax {@code original_max_position_embeddings}; 0 disables the logit scaling
     */
    public static GgufTestWriter qwen2DualChunk(SyntheticModels.Config c, int chunkSize, int localSize,
                                                int originalMax) {
        return qwen2DualChunk(c, chunkSize, localSize, originalMax, false);
    }

    /** {@link #qwen2DualChunk(SyntheticModels.Config, int, int, int)} with the sparse-attention switch set. */
    public static GgufTestWriter qwen2DualChunk(SyntheticModels.Config c, int chunkSize, int localSize,
                                                int originalMax, boolean sparseEnabled) {
        if (localSize <= 0 || chunkSize <= localSize) {
            throw new IllegalArgumentException("dual-chunk attention needs 0 < local_size < chunk_size, got "
                    + localSize + " / " + chunkSize);
        }
        if (originalMax < 0) {
            throw new IllegalArgumentException("dual-chunk original_max_position_embeddings must be >= 0, got "
                    + originalMax);
        }
        GgufTestWriter w = qwen2Mtp(c, 0);
        w.u32(w.archKey("attention.dual_chunk.chunk_size"), chunkSize);
        w.u32(w.archKey("attention.dual_chunk.local_size"), localSize);
        w.u32(w.archKey("attention.dual_chunk.original_max_position_embeddings"), originalMax);
        w.bool(w.archKey("attention.dual_chunk.sparse_attention_enabled"), sparseEnabled);
        w.u32(w.archKey("attention.dual_chunk.sparse_attention_threshold"), 32768);
        w.u32(w.archKey("attention.dual_chunk.sparse_attention_last_q"), 64);
        return w;
    }

    // ================================================================== gemma2

    public static GgufTestWriter gemma2() {
        return gemma2(SyntheticModels.Config.gemma2());
    }

    /**
     * LLM_ARCH_GEMMA2: sandwich norms, attention/final logit soft-capping and the default SWA
     * pattern 2 (even layers sliding, odd layers global). The output head is always tied, as in
     * the llama.cpp loader.
     */
    public static GgufTestWriter gemma2(SyntheticModels.Config c) {
        requireAttention(c, "gemma2");
        requirePositive("gemma2 sliding window", c.slidingWindow());
        GgufTestWriter w = base("gemma2", c);
        attentionKeys(w, c, true);
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("attn_logit_softcapping"), c.attnLogitSoftcap());
        w.f32(w.archKey("final_logit_softcapping"), c.finalLogitSoftcap());
        w.u32(w.archKey("attention.sliding_window"), c.slidingWindow());
        gemmaVocab(w, c.nVocab());

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "post_attention_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            t.ones(blk(il, "post_ffw_norm.weight"), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        return w;
    }

    // ================================================================== qwen2vl

    public static GgufTestWriter qwen2vl() {
        return qwen2vl(SyntheticModels.Config.llama());
    }

    /**
     * LLM_ARCH_QWEN2VL: the qwen2 tensor set with the required four M-RoPE sections
     * ({@code qwen2vl.cpp:3-5} reads only them; the RMS epsilon defaults to 1e-6). The sections are pair counts
     * covering the whole rotation (ggml {@code mrope_cache_init} cycles the pair index over their sum): temporal,
     * height and width splits, no extra section.
     */
    public static GgufTestWriter qwen2vl(SyntheticModels.Config c) {
        requireAttention(c, "qwen2vl");
        if (c.nRot() != c.headDim()) {
            throw new IllegalArgumentException("qwen2vl requires rope.dimension_count == head dim, got " + c.nRot()
                    + " vs " + c.headDim());
        }
        int pairs = c.nRot() / 2;
        int t = pairs / 2;
        int h = pairs / 4;
        int w = pairs - t - h;
        if (t <= 0 || h <= 0 || w <= 0) {
            throw new IllegalArgumentException("qwen2vl rope dim " + c.nRot() + " does not split into four sections");
        }
        GgufTestWriter w2 = base("qwen2vl", c);
        attentionKeys(w2, c, false);
        w2.u32(w2.archKey("rope.dimension_count"), c.nRot());
        w2.f32(w2.archKey("rope.freq_base"), c.ropeFreqBase());
        w2.f32(w2.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w2.u32Array(w2.archKey("rope.dimension_sections"),
                new long[]{t, h, w, 0});
        bpeVocab(w2, c.nVocab(), "qwen2", new String[]{"<|endoftext|>", "<|im_start|>", "<|im_end|>"});
        w2.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t2 = new SyntheticModels.Tensors(w2, c);
        t2.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t2.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t2.qkvo(il, c.nHead() * c.headDim());
            t2.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t2.gatedFfn(il, c.nFf());
        }
        t2.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t2.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w2;
    }

    // ================================================================== limite

    public static GgufTestWriter limite() {
        return limite(SyntheticModels.Config.limite());
    }

    /**
     * LLM_ARCH_LIMITE: value embeddings with a per-KV-head gate on layer 0, the exclusive self attention with
     * per-head alphas, a per-head output gate on layer 1, learned residual scales, MUDD dense connections over
     * taps {0,0} and {0,1} and the sigmoid logit cap. Layer 0 is a sliding-window layer (the only roped one),
     * layer 1 runs full attention without positional encoding (limite.cpp:163-170).
     */
    public static GgufTestWriter limite(SyntheticModels.Config c) {
        requireAttention(c, "limite");
        requirePositive("limite sliding window", c.slidingWindow());
        requirePositive("limite mudd feed-forward length", c.nFfExp());
        GgufTestWriter w = base("limite", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), c.nHeadKv());
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("attention.scale"), 0.125f);
        w.f32(w.archKey("attention.xsa_epsilon"), 1e-6f);
        w.f32Array(w.archKey("final_logit_sigmoid_capping"), 2.0f, 1.0f, 4.0f);
        w.u32(w.archKey("attention.sliding_window"), c.slidingWindow());
        w.boolArray(w.archKey("attention.sliding_window_pattern"), true, false);
        w.u32(w.archKey("mudd.tap_count"), 2);
        w.u32(w.archKey("mudd.feed_forward_length"), c.nFfExp());
        w.i32Array(w.archKey("mudd.tap_indices"), 0, 0, 0, 1);
        bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("value_embd.weight", (long) c.nHeadKv() * c.headDim(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.qkvo(il, (long) c.nHead() * c.headDim());
            t.matrix(blk(il, "attn_xsa_alpha.weight"), c.nHead());
            t.random(blk(il, "attn_resid_scale.weight"), 0.4f, 0.6f, 1);
            t.random(blk(il, "ffn_resid_scale.weight"), 0.4f, 0.6f, 1);
            if (il == 0) {
                t.matrix(blk(il, "attn_ve_gate.weight"), c.nEmbd(), c.nHeadKv());
            } else {
                t.matrix(blk(il, "attn_gate.weight"), c.nEmbd(), c.nHead());
            }
            t.gatedFfn(il, c.nFf());
            t.matrix(blk(il, "mudd_down.weight"), c.nEmbd(), c.nFfExp());
            t.matrix(blk(il, "mudd_up.weight"), c.nFfExp(), 4);
            t.random(blk(il, "mudd_up.bias"), -0.05f, 0.05f, 4);
        }
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }
}
