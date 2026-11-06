package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.spmVocab;

/**
 * The audio and drafting fixtures of {@link SyntheticModels}: the WavTokenizer codec decoder and the DFlash /
 * DFlash2 block-diffusion drafters. Each builder returns a fresh {@link GgufTestWriter}; the {@link SyntheticModels}
 * entry points delegate here.
 */
final class SyntheticAudioModels {

    private SyntheticAudioModels() {
    }

    /**
     * LLM_ARCH_WAVTOKENIZER_DEC: the codec decoder whose tokens map to spectral frames (llama.cpp
     * src/models/wavtokenizer-dec.cpp). With {@code sampleRate > 0} the speech geometry keys are written.
     */
    static GgufTestWriter wavtokenizer(int sampleRate, int nFft, int hopLength) {
        GgufTestWriter w = base("wavtokenizer-dec", SyntheticModels.Config.llama().toBuilder()
                .nEmbd(SyntheticModels.WAVTOK_OUT).nCtxTrain(64)
                .nLayer(SyntheticModels.WAVTOK_POSNET).build());
        w.u32(w.archKey("features_length"), SyntheticModels.WAVTOK_E);
        w.u32(w.archKey("posnet.embedding_length"), SyntheticModels.WAVTOK_C)
                .u32(w.archKey("posnet.block_count"), SyntheticModels.WAVTOK_POSNET);
        w.u32(w.archKey("convnext.embedding_length"), SyntheticModels.WAVTOK_C)
                .u32(w.archKey("convnext.block_count"), SyntheticModels.WAVTOK_CONVNEXT);
        w.u32(w.archKey("feed_forward_length"), SyntheticModels.WAVTOK_FF);
        w.u32(w.archKey("attention.group_norm_groups"), 2);
        w.f32(w.archKey("attention.group_norm_epsilon"), 1e-6f);
        w.f32(w.archKey("attention.layer_norm_epsilon"), 1e-5f);
        if (sampleRate > 0) {
            w.u32("audio.sample_rate", sampleRate).u32("audio.n_fft", nFft).u32("audio.hop_length", hopLength);
        }
        spmVocab(w, SyntheticModels.WAVTOK_CODES, new String[] {"<unk>", "<s>", "</s>"},
                new int[] {SyntheticModels.TOKEN_TYPE_UNKNOWN, SyntheticModels.TOKEN_TYPE_CONTROL,
                        SyntheticModels.TOKEN_TYPE_CONTROL});
        w.u32("tokenizer.ggml.unknown_token_id", 0).u32("tokenizer.ggml.bos_token_id", 1)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false)
                .bool("tokenizer.ggml.add_eos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w,
                SyntheticModels.Config.llama().toBuilder().nEmbd(SyntheticModels.WAVTOK_E).build());
        t.matrix("token_embd.weight", SyntheticModels.WAVTOK_E, SyntheticModels.WAVTOK_CODES);
        t.random("conv1d.weight", -0.1f, 0.1f, 7, SyntheticModels.WAVTOK_E, SyntheticModels.WAVTOK_C);
        t.random("conv1d.bias", -0.02f, 0.02f, 1, SyntheticModels.WAVTOK_C);
        for (int il = 0; il < SyntheticModels.WAVTOK_POSNET; il++) {
            switch (il) {
                case 0:
                case 1:
                case 3:
                case 4:
                    t.random(posnet(il, "norm1.weight"), 0.8f, 1.2f, 1, SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "norm1.bias"), -0.05f, 0.05f, 1, SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "conv1.weight"), -0.1f, 0.1f, 3, SyntheticModels.WAVTOK_C,
                            SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "conv1.bias"), -0.02f, 0.02f, 1, SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "norm2.weight"), 0.8f, 1.2f, 1, SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "norm2.bias"), -0.05f, 0.05f, 1, SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "conv2.weight"), -0.1f, 0.1f, 3, SyntheticModels.WAVTOK_C,
                            SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "conv2.bias"), -0.02f, 0.02f, 1, SyntheticModels.WAVTOK_C);
                    break;
                case 2:
                    t.random(posnet(il, "attn_norm.weight"), 0.8f, 1.2f, 1, SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "attn_norm.bias"), -0.05f, 0.05f, 1, SyntheticModels.WAVTOK_C);
                    for (String p : new String[] {"attn_q", "attn_k", "attn_v", "attn_output"}) {
                        t.random(posnet(il, p + ".weight"), -0.1f, 0.1f, 1, SyntheticModels.WAVTOK_C,
                                SyntheticModels.WAVTOK_C);
                        t.random(posnet(il, p + ".bias"), -0.02f, 0.02f, 1, SyntheticModels.WAVTOK_C);
                    }
                    break;
                default:
                    t.random(posnet(il, "attn_norm.weight"), 0.8f, 1.2f, 1, SyntheticModels.WAVTOK_C);
                    t.random(posnet(il, "attn_norm.bias"), -0.05f, 0.05f, 1, SyntheticModels.WAVTOK_C);
            }
        }
        t.random("token_embd_norm.weight", 0.8f, 1.2f, SyntheticModels.WAVTOK_C);
        t.random("token_embd_norm.bias", -0.05f, 0.05f, SyntheticModels.WAVTOK_C);
        for (int il = 0; il < SyntheticModels.WAVTOK_CONVNEXT; il++) {
            t.random(convnext(il, "dw.weight"), -0.1f, 0.1f, 7, 1, SyntheticModels.WAVTOK_C);
            t.random(convnext(il, "dw.bias"), -0.02f, 0.02f, 1, SyntheticModels.WAVTOK_C);
            t.random(convnext(il, "norm.weight"), 0.8f, 1.2f, SyntheticModels.WAVTOK_C);
            t.random(convnext(il, "norm.bias"), -0.05f, 0.05f, SyntheticModels.WAVTOK_C);
            t.matrix(convnext(il, "pw1.weight"), SyntheticModels.WAVTOK_C, SyntheticModels.WAVTOK_FF);
            t.random(convnext(il, "pw1.bias"), -0.02f, 0.02f, SyntheticModels.WAVTOK_FF);
            t.matrix(convnext(il, "pw2.weight"), SyntheticModels.WAVTOK_FF, SyntheticModels.WAVTOK_C);
            t.random(convnext(il, "pw2.bias"), -0.02f, 0.02f, SyntheticModels.WAVTOK_C);
            t.random(convnext(il, "gamma.weight"), 0.4f, 0.6f, SyntheticModels.WAVTOK_C);
        }
        t.random("output_norm.weight", 0.8f, 1.2f, SyntheticModels.WAVTOK_C);
        t.random("output_norm.bias", -0.05f, 0.05f, SyntheticModels.WAVTOK_C);
        t.matrix("output.weight", SyntheticModels.WAVTOK_C, SyntheticModels.WAVTOK_OUT);
        t.random("output.bias", -0.02f, 0.02f, SyntheticModels.WAVTOK_OUT);
        return w;
    }

    private static String posnet(int il, String base) {
        return "posnet." + il + "." + base;
    }

    private static String convnext(int il, String base) {
        return "convnext." + il + "." + base;
    }

    /**
     * LLM_ARCH_DFLASH block-diffusion drafter (dflash.cpp:298-333) over {@code c.nLayer()} plain attention stages.
     */
    static GgufTestWriter dflashDrafter(SyntheticModels.Config c, int[] targetLayers, int blockSize) {
        return dflashDrafter(c, targetLayers, blockSize, false, 0, null);
    }

    /** {@link #dflashDrafter} with the block-internal causal flag and an interleaved sliding window. */
    static GgufTestWriter dflashDrafter(SyntheticModels.Config c, int[] targetLayers, int blockSize, boolean causal,
                                        int nSwa, boolean[] swaPattern) {
        return dflashDrafter(c, targetLayers, blockSize, causal, nSwa, swaPattern, null);
    }

    /** {@link #dflashDrafter} as a DFlash2 drafter with its dynamic convolutions and selector lattice. */
    static GgufTestWriter dflashDrafter(SyntheticModels.Config c, int[] targetLayers, int blockSize, boolean causal,
                                        int nSwa, boolean[] swaPattern, SyntheticModels.Dflash2 dflash2) {
        requireAttention(c, "dflash");
        if (targetLayers == null || targetLayers.length == 0) {
            throw new IllegalArgumentException("dflash drafter needs at least one target layer");
        }
        if (blockSize < 2) {
            throw new IllegalArgumentException("dflash drafter needs a block size of at least 2, got " + blockSize);
        }
        GgufTestWriter w = base("dflash", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("block_size"), blockSize);
        w.i32Array(w.archKey("target_layers"), targetLayers);
        if (causal) {
            w.bool(w.archKey("attention.causal"), true);
        }
        if (nSwa > 0) {
            w.u32(w.archKey("attention.sliding_window"), nSwa);
            if (swaPattern != null) {
                w.boolArray(w.archKey("attention.sliding_window_pattern"), swaPattern);
            }
        }
        spmVocab(w, c.nVocab(), new String[] {"<unk>", "<s>", "</s>"},
                new int[] {SyntheticModels.TOKEN_TYPE_UNKNOWN, SyntheticModels.TOKEN_TYPE_CONTROL,
                        SyntheticModels.TOKEN_TYPE_CONTROL});
        w.u32("tokenizer.ggml.unknown_token_id", 0).u32("tokenizer.ggml.bos_token_id", 1)
                .u32("tokenizer.ggml.eos_token_id", 2).u32("tokenizer.ggml.mask_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", true).bool("tokenizer.ggml.add_eos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("fc.weight", (long) targetLayers.length * c.nEmbd(), c.nEmbd());
        t.ones("enc.output_norm.weight", c.nEmbd());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, (long) c.nHead() * c.headDim());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.random(blk(il, "attn_sinks.weight"), -0.1f, 0.1f, c.nHead());
            t.ones(blk(il, "post_attention_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            t.ones(blk(il, "post_ffw_norm.weight"), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        if (dflash2 != null) {
            dflash2Drafter(w, c, dflash2);
        }
        return w;
    }

    /**
     * The DFlash2 additions to a dflash drafter writer: the conv and selector keys and their tensors.
     *
     * @throws IllegalArgumentException when a size is non-positive or the group size does not divide {@code n_embd}
     */
    static void dflash2Drafter(GgufTestWriter w, SyntheticModels.Config c, SyntheticModels.Dflash2 d2) {
        requirePositive("dflash2 conv kernel", d2.convKernel);
        requirePositive("dflash2 conv group size", d2.convGroup);
        requirePositive("dflash2 selector rank", d2.selectorRank);
        requirePositive("dflash2 selector top-k", d2.selectorTopK);
        if (c.nEmbd() % d2.convGroup != 0) {
            throw new IllegalArgumentException("dflash2 conv group size " + d2.convGroup + " must divide n_embd "
                    + c.nEmbd());
        }
        w.u32(w.archKey("conv_kernel_size"), d2.convKernel).u32(w.archKey("conv_group_size"), d2.convGroup)
                .u32(w.archKey("selector_rank"), d2.selectorRank)
                .u32(w.archKey("selector_top_k"), d2.selectorTopK);
        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long dyn = 2L * d2.convKernel * (c.nEmbd() / d2.convGroup);
        for (int il = 0; il < c.nLayer(); il++) {
            t.random(blk(il, "attn_conv_base"), -0.1f, 0.1f, 2L * d2.convKernel * c.nEmbd());
            t.matrix(blk(il, "attn_conv_proj.weight"), c.nEmbd(), dyn);
            t.random(blk(il, "ffn_conv_base"), -0.1f, 0.1f, 2L * d2.convKernel * c.nEmbd());
            t.matrix(blk(il, "ffn_conv_proj.weight"), c.nEmbd(), dyn);
        }
        t.matrix("selector_predecessor.weight", d2.selectorRank, c.nVocab());
        t.matrix("selector_successor.weight", d2.selectorRank, c.nVocab());
        t.matrix("selector_hidden.weight", c.nEmbd(), d2.selectorRank);
    }
}
