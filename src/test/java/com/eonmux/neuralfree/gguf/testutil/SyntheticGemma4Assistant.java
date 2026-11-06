package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * A tiny Gemma 4 assistant draft GGUF ({@code LLM_ARCH_GEMMA4_ASSISTANT}, llama.cpp
 * src/models/gemma4-assistant.cpp) for a Gemma 4 target: the {@code nextn.pre_projection} over
 * {@code [target token embedding | target hidden]}, {@code nextn_predict_layers} assistant blocks that attend
 * over the target's KV caches (Q only, sandwich norms, GELU-tanh FFN and {@code layer_output_scale}), the output
 * norm and the tied token-embedding head. Values follow {@link SyntheticModels}: projection matrices in the
 * config's weight type, norm weights 1.
 */
public final class SyntheticGemma4Assistant {

    private SyntheticGemma4Assistant() {
    }

    /**
     * The assistant blocks of the target {@code target}: {@code own} carries the assistant's hidden size,
     * vocabulary, head geometry and FFN length; its sliding-window pattern must start with the sliding-window
     * block, as the target's does.
     *
     * @param blocks assistant block count, the model's {@code nextn_predict_layers}
     * @throws IllegalArgumentException when the block count is not positive or the geometry is invalid
     */
    public static GgufTestWriter gemma4Assistant(SyntheticModels.Config target, SyntheticModels.Config own,
                                                 int blocks) {
        requirePositive("gemma4-assistant blocks", blocks);
        SyntheticText.requireAttention(own, "gemma4-assistant");
        if (own.nLayer() != blocks) {
            throw new IllegalArgumentException("the assistant's block_count (" + own.nLayer()
                    + ") must equal its nextn_predict_layers (" + blocks + ")");
        }
        if (own.nEmbd() == target.nEmbd()) {
            throw new IllegalArgumentException("embedding_length_out must carry the target hidden size "
                    + target.nEmbd() + ", not the assistant's " + own.nEmbd());
        }
        GgufTestWriter w = SyntheticText.base("gemma4-assistant", own);
        SyntheticText.attentionKeys(w, own, true);
        w.u32(w.archKey("attention.key_length_swa"), own.headDim());
        w.u32(w.archKey("attention.value_length_swa"), own.headDim());
        w.u32(w.archKey("rope.dimension_count"), own.nRot());
        w.f32(w.archKey("rope.freq_base"), own.ropeFreqBase());
        w.f32(w.archKey("rope.freq_base_swa"), target.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), own.normEps());
        w.u32(w.archKey("attention.sliding_window"), target.slidingWindow());
        boolean[] swa = new boolean[blocks];
        swa[0] = true;
        w.boolArray(w.archKey("attention.sliding_window_pattern"), swa);
        w.u32(w.archKey("embedding_length_out"), target.nEmbd());
        w.u32(w.archKey("nextn_predict_layers"), blocks);
        SyntheticText.gemmaVocab(w, own.nVocab());

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, own);
        t.matrix("token_embd.weight", own.nEmbd(), own.nVocab());
        t.ones("output_norm.weight", own.nEmbd());
        t.matrix("nextn.pre_projection.weight", 2L * target.nEmbd(), own.nEmbd());
        t.matrix("nextn.post_projection.weight", own.nEmbd(), target.nEmbd());
        for (int il = 0; il < blocks; il++) {
            t.ones(blk(il, "attn_norm.weight"), own.nEmbd());
            t.matrix(blk(il, "attn_q.weight"), own.nEmbd(), (long) own.nHead() * own.headDim());
            t.matrix(blk(il, "attn_output.weight"), (long) own.nHead() * own.headDim(), own.nEmbd());
            t.ones(blk(il, "attn_q_norm.weight"), own.headDim());
            t.ones(blk(il, "post_attention_norm.weight"), own.nEmbd());
            t.ones(blk(il, "layer_output_scale.weight"), 1);
            t.ones(blk(il, "ffn_norm.weight"), own.nEmbd());
            t.gatedFfn(il, own.nFf());
            t.ones(blk(il, "post_ffw_norm.weight"), own.nEmbd());
        }
        t.ones("rope_freqs.weight", own.headDim() / 2);
        return w;
    }
}
