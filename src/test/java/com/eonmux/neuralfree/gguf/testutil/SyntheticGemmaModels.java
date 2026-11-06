package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * Synthetic GGUF writers of the gemma families with per-layer token embeddings ({@code per_layer_token_embd}), sized
 * like {@link SyntheticModels}, Gemma SPM tokenizer and tied output: gemma3n (sliding-window attention with per-head
 * Q/K norms and sandwich norms, the GELU-tanh gated FFN, the AltUp streams and the Laurel residual) and dense gemma4
 * (a sliding-window block then a full-attention block with proportional RoPE, sandwich norms and the GELU-tanh gated
 * FFN). Source: llama.cpp src/models/gemma3n.cpp and src/models/gemma4.cpp {@code load_arch_hparams} /
 * {@code load_arch_tensors}.
 */
public final class SyntheticGemmaModels {

    /**
     * The gemma3n widths llama.cpp fixes rather than reads (llama-hparams.h {@code n_altup}, {@code laurel_rank},
     * {@code n_embd_altup}): AltUp streams, Laurel rank and the per-layer embedding width of one layer.
     */
    public static final int ALTUP = 4;
    public static final int LAUREL_RANK = 64;
    public static final int PER_LAYER_WIDTH = 256;
    /** The gemma4 per-layer embedding width ({@code embedding_length_per_layer_input}). */
    public static final int GEMMA4_PER_LAYER_WIDTH = 16;

    private SyntheticGemmaModels() {
    }

    /** 2 sliding-window layers of 4 query heads / 2 KV heads of width 16, FFN 128, window 32, final soft-cap 30. */
    public static SyntheticModels.Config gemma3nConfig() {
        return new SyntheticModels.Config.Builder().nLayer(2).nHead(4).nHeadKv(2).headDim(16).nRot(16).nFf(128)
                .normEps(1e-6f).slidingWindow(32).finalLogitSoftcap(30.0f).tiedOutput(true).build();
    }

    public static GgufTestWriter gemma3n() {
        return gemma3n(gemma3nConfig());
    }

    /**
     * LLM_ARCH_GEMMA3N: every tensor {@code load_arch_tensors} requires, the output head tied to
     * {@code token_embd}. Layer {@code il}'s per-layer embedding reads columns {@code [il*256, (il+1)*256)} of the
     * {@code per_layer_token_embd} rows.
     */
    public static GgufTestWriter gemma3n(SyntheticModels.Config c) {
        SyntheticText.requireAttention(c, "gemma3n");
        requirePositive("gemma3n sliding window", c.slidingWindow());
        GgufTestWriter w = SyntheticText.base("gemma3n", c);
        SyntheticText.attentionKeys(w, c, true);
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("final_logit_softcapping"), c.finalLogitSoftcap());
        w.u32(w.archKey("attention.sliding_window"), c.slidingWindow());
        SyntheticText.gemmaVocab(w, c.nVocab());

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long perLayer = (long) PER_LAYER_WIDTH * c.nLayer();
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("altup_proj.weight", c.nEmbd(), c.nEmbd(), ALTUP - 1);
        t.matrix("altup_unembd_proj.weight", c.nEmbd(), c.nEmbd(), ALTUP - 1);
        t.random("per_layer_token_embd.weight", -0.5f, 0.5f, perLayer, c.nVocab());
        t.matrix("per_layer_model_proj.weight", c.nEmbd(), perLayer);
        t.ones("per_layer_proj_norm.weight", PER_LAYER_WIDTH);
        for (int il = 0; il < c.nLayer(); il++) {
            layer(t, c, il);
        }
        t.ones("output_norm.weight", c.nEmbd());
        return w;
    }

    /** 2 layers (sliding window, then full attention) of 4 query heads / 2 KV heads of width 16, window 32. */
    public static SyntheticModels.Config gemma4Config() {
        return gemma3nConfig();
    }

    public static GgufTestWriter gemma4() {
        return gemma4(gemma4Config());
    }

    /**
     * LLM_ARCH_GEMMA4 without experts: every tensor {@code load_arch_tensors} requires, an even number of blocks
     * alternating sliding-window and full attention (the full-attention blocks carry {@code rope_freqs}), no KV
     * sharing, and the per-layer embedding of width {@value #GEMMA4_PER_LAYER_WIDTH}: layer {@code il} reads
     * columns {@code [il*w, (il+1)*w)} of the {@code per_layer_token_embd} rows.
     */
    public static GgufTestWriter gemma4(SyntheticModels.Config c) {
        SyntheticText.requireAttention(c, "gemma4");
        requirePositive("gemma4 sliding window", c.slidingWindow());
        if (c.nLayer() < 2 || c.nLayer() % 2 != 0) {
            throw new IllegalArgumentException("the gemma4 fixture writes an even number of layers, got "
                    + c.nLayer());
        }
        GgufTestWriter w = SyntheticText.base("gemma4", c);
        SyntheticText.attentionKeys(w, c, true);
        w.u32(w.archKey("attention.key_length_swa"), c.headDim());
        w.u32(w.archKey("attention.value_length_swa"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("rope.freq_base_swa"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("final_logit_softcapping"), c.finalLogitSoftcap());
        w.u32(w.archKey("attention.sliding_window"), c.slidingWindow());
        boolean[] swa = new boolean[c.nLayer()];
        for (int il = 0; il < swa.length; il++) {
            swa[il] = il % 2 == 0;
        }
        w.boolArray(w.archKey("attention.sliding_window_pattern"), swa);
        w.u32(w.archKey("embedding_length_per_layer_input"), GEMMA4_PER_LAYER_WIDTH);
        SyntheticText.gemmaVocab(w, c.nVocab());

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long perLayer = (long) GEMMA4_PER_LAYER_WIDTH * c.nLayer();
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.random("per_layer_token_embd.weight", -0.5f, 0.5f, perLayer, c.nVocab());
        t.matrix("per_layer_model_proj.weight", c.nEmbd(), perLayer);
        t.ones("per_layer_proj_norm.weight", GEMMA4_PER_LAYER_WIDTH);
        t.ones("rope_freqs.weight", c.headDim() / 2);
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, (long) c.nHead() * c.headDim());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.ones(blk(il, "post_attention_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            t.ones(blk(il, "post_ffw_norm.weight"), c.nEmbd());
            t.matrix(blk(il, "inp_gate.weight"), c.nEmbd(), GEMMA4_PER_LAYER_WIDTH);
            t.matrix(blk(il, "proj.weight"), GEMMA4_PER_LAYER_WIDTH, c.nEmbd());
            t.ones(blk(il, "post_norm.weight"), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        return w;
    }

    /** One decoder block with its AltUp, Laurel and per-layer embedding tensors. */
    private static void layer(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
        t.qkvo(il, (long) c.nHead() * c.headDim());
        t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
        t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
        t.ones(blk(il, "post_attention_norm.weight"), c.nEmbd());
        t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
        t.gatedFfn(il, c.nFf());
        t.ones(blk(il, "post_ffw_norm.weight"), c.nEmbd());
        t.matrix(blk(il, "inp_gate.weight"), c.nEmbd(), PER_LAYER_WIDTH);
        t.matrix(blk(il, "proj.weight"), PER_LAYER_WIDTH, c.nEmbd());
        t.ones(blk(il, "post_norm.weight"), c.nEmbd());
        t.random(blk(il, "altup_correct_coef.weight"), -0.2f, 0.2f, ALTUP, ALTUP);
        t.random(blk(il, "altup_correct_scale.weight"), 0.5f, 1.0f, c.nEmbd());
        t.random(blk(il, "altup_predict_coef.weight"), -0.2f, 0.2f, ALTUP, (long) ALTUP * ALTUP);
        t.matrix(blk(il, "altup_router.weight"), c.nEmbd(), ALTUP);
        t.ones(blk(il, "altup_router_norm.weight"), c.nEmbd());
        t.matrix(blk(il, "laurel_l.weight"), c.nEmbd(), LAUREL_RANK);
        t.matrix(blk(il, "laurel_r.weight"), LAUREL_RANK, c.nEmbd());
        t.ones(blk(il, "laurel_post_norm.weight"), c.nEmbd());
    }
}
