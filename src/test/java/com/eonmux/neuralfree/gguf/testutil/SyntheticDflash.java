package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * A tiny DFlash draft GGUF (LLM_ARCH_DFLASH, llama.cpp src/models/dflash.cpp {@code load_arch_hparams} /
 * {@code load_arch_tensors}) for a target of the same width and vocabulary: the {@code fc} encoder over
 * {@code targetLayers.length} target layers, {@code enc.output_norm}, and per block the attention with per-head Q/K
 * norms and a gated FFN, its own token embeddings and output head. Values follow {@link SyntheticModels}: projection
 * matrices in the config's weight type, norm weights 1.
 */
public final class SyntheticDflash {

    private SyntheticDflash() {
    }

    /** The draft's settings beside the shared dimensions of {@link SyntheticModels.Config}. */
    public static final class Options {
        private final int[] targetLayers;
        private final int blockSize;
        private final boolean causal;
        private final int slidingWindow;
        private final SyntheticModels.Dflash2 dflash2;
        private final boolean shareTargetHead;
        private final int dsparkRank;

        private Options(int[] targetLayers, int blockSize, boolean causal, int slidingWindow,
                        SyntheticModels.Dflash2 dflash2, boolean shareTargetHead, int dsparkRank) {
            this.targetLayers = targetLayers.clone();
            this.blockSize = blockSize;
            this.causal = causal;
            this.slidingWindow = slidingWindow;
            this.dflash2 = dflash2;
            this.shareTargetHead = shareTargetHead;
            this.dsparkRank = dsparkRank;
        }

        /** Bidirectional blocks of {@code blockSize} over {@code targetLayers}, every block full attention. */
        public static Options of(int blockSize, int... targetLayers) {
            requirePositive("dflash block size", blockSize);
            if (targetLayers.length == 0) {
                throw new IllegalArgumentException("a dflash draft reads at least one target layer");
            }
            return new Options(targetLayers, blockSize, false, 0, null, false, 0);
        }

        /** The same draft with {@code dflash.attention.causal}. */
        public Options causal() {
            return new Options(targetLayers, blockSize, true, slidingWindow, dflash2, shareTargetHead, dsparkRank);
        }

        /** The same draft with a sliding window of {@code window} positions on its even blocks. */
        public Options slidingWindow(int window) {
            requirePositive("dflash sliding window", window);
            return new Options(targetLayers, blockSize, causal, window, dflash2, shareTargetHead, dsparkRank);
        }

        /** The same draft with DFlash2's convolutions and selector ({@code SyntheticModels.Dflash2}). */
        public Options dflash2(int convKernel, int convGroup, int selectorRank, int selectorTopK) {
            return new Options(targetLayers, blockSize, causal, slidingWindow,
                    new SyntheticModels.Dflash2(convKernel, convGroup, selectorRank, selectorTopK),
                    shareTargetHead, dsparkRank);
        }

        /** The same draft without its own output head, so the drafter borrows the target's. */
        public Options shareTargetHead() {
            return new Options(targetLayers, blockSize, causal, slidingWindow, dflash2, true, dsparkRank);
        }

        /** The same draft as a DSpark drafter: markov tables of {@code rank} over the full vocabulary. */
        public Options dspark(int rank) {
            requirePositive("dspark markov rank", rank);
            return new Options(targetLayers, blockSize, causal, slidingWindow, dflash2, shareTargetHead, rank);
        }
    }

    /**
     * The draft of {@code c}'s dimensions; its mask token is the last vocabulary id.
     *
     * @throws IllegalArgumentException when the attention geometry is invalid
     */
    public static GgufTestWriter dflash(SyntheticModels.Config c, Options o) {
        requireAttention(c, "dflash");
        GgufTestWriter w = base("dflash", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.headDim());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.i32Array(w.archKey("target_layers"), o.targetLayers);
        w.u32(w.archKey("block_size"), o.blockSize);
        if (o.causal) {
            w.bool(w.archKey("attention.causal"), true);
        }
        if (o.slidingWindow > 0) {
            boolean[] swa = new boolean[c.nLayer()];
            for (int il = 0; il < swa.length; il += 2) {
                swa[il] = true;
            }
            w.u32(w.archKey("attention.sliding_window"), o.slidingWindow);
            w.boolArray(w.archKey("attention.sliding_window_pattern"), swa);
        }
        w.u32("tokenizer.ggml.mask_token_id", c.nVocab() - 1);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", (long) o.targetLayers.length * c.nEmbd(), c.nEmbd());
        t.ones("enc.output_norm.weight", c.nEmbd());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, (long) c.nHead() * c.headDim());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!o.shareTargetHead) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        if (o.dsparkRank > 0) {
            t.matrix("markov_w1.weight", o.dsparkRank, c.nVocab());
            t.matrix("markov_w2.weight", o.dsparkRank, c.nVocab());
        }
        if (o.dflash2 != null) {
            SyntheticAudioModels.dflash2Drafter(w, c, o.dflash2);
        }
        return w;
    }
}
