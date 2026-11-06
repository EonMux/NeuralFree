package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;

/**
 * Synthetic GGUF writer of qwen4exp with the n-gram per-layer embedding, sized like {@link SyntheticModels}: block 0
 * a gated delta net carrying the PLE module, block 1 a gated full attention with the pooled (QSA, ratio
 * {@value #QSA_RATIO}) indexer or dense, four hyper-connection streams of low rank {@value #HC_LOW_RANK} around both
 * branches, interleaved M-RoPE, and routed experts with a sigmoid-gated shared expert on both blocks. The MTP variant
 * appends the NextN blocks with their fusion and gated head, and its MTP-only sidecars hold the NextN block alone,
 * with or without copies of the target's {@code token_embd} and LM head. Byte-level BPE tokenizer. Source: llama.cpp
 * src/models/qwen4exp.cpp {@code load_arch_hparams} / {@code load_arch_tensors}.
 */
public final class SyntheticQwen4ExpModels {

    /** The EOS id of the tokenizer and of {@code ple.eos_token_id}. */
    public static final int EOS = 2;
    /** Hyper-connection streams and their low rank. */
    public static final int HC = 4;
    public static final int HC_LOW_RANK = 8;
    /** N-gram size, heads per n-gram and convolution taps of the PLE module. */
    public static final int PLE_NGRAM = 3;
    public static final int PLE_HEADS_PER_NGRAM = 2;
    public static final int PLE_CONV_KERNEL = 2;
    /** Rows of each PLE head's hash range in {@code per_layer_token_embd}. */
    public static final int PLE_HEAD_ROWS = 61;
    private static final int QSA_RATIO = 4;
    private static final int INDEXER_HEADS = 4;
    private static final int INDEXER_DIM = 32;
    private static final int INDEXER_TOP_K = 8;
    private static final long[] MULTIPLIERS = {0x9E3779B97F4A7C15L, 0xC2B2AE3D27D4EB4FL, 0x165667B19E3779F9L};

    private SyntheticQwen4ExpModels() {
    }

    /**
     * 2 layers of 4 query heads / 2 KV heads of width 16 (16 rotary, sections 2/2/2/2), 4 experts of width 32 with 2
     * used and a shared expert of 32, GDN 2 key heads / 4 value heads of state 16 with a 4-tap convolution.
     */
    public static SyntheticModels.Config config() {
        return new SyntheticModels.Config.Builder().nLayer(2).nHead(4).nHeadKv(2).headDim(16).nRot(16).nFf(128)
                .normEps(1e-6f).nExpert(4).nExpertUsed(2).nFfExp(32).nFfShexp(32).ssmConv(4).ssmState(16)
                .ssmGroups(2).ssmDtRank(4).build();
    }

    /** The fixture with {@code ple.image_token_id} 0: an embedding row hashes the EOS id. */
    public static GgufTestWriter qwen4exp() {
        return qwen4exp(0);
    }

    /** The fixture with the QSA indexer on block 1. */
    public static GgufTestWriter qwen4exp(int imageTokenId) {
        return qwen4exp(imageTokenId, true);
    }

    /**
     * LLM_ARCH_QWEN4EXP with the PLE module on block 0.
     *
     * @param imageTokenId {@code ple.image_token_id}, the stand-in an embedding row hashes; 0 for the EOS id
     * @param qsa          whether block 1 pools its indexer keys (compress ratio {@value #QSA_RATIO}); without it
     *                     the ratios are all 0 and block 1 runs dense gated attention (qwen4exp.cpp:64-75), which
     *                     has a multi-sequence step
     */
    public static GgufTestWriter qwen4exp(int imageTokenId, boolean qsa) {
        return qwen4exp(imageTokenId, qsa, 0);
    }

    /**
     * The fixture with one NextN block after its two trunk blocks: the model whose MTP drafter runs the block
     * {@code blk.2} over the trunk's hyper-connection streams (qwen4exp.cpp:395-410).
     */
    public static GgufTestWriter qwen4expMtp() {
        return qwen4exp(0, true, 1);
    }

    /**
     * An MTP-only sidecar of {@link #qwen4expMtp()}: the same model's metadata and only the NextN block's tensors
     * ({@code blk.2}). With {@code ownEmbedding} it carries a copy of the target's {@code token_embd}, and with
     * {@code ownHead} a copy of the target's LM head; both are the target's values, since the two writers derive
     * them from the same seed and name. Without a tensor the drafter takes the target's
     * ({@code llama_model_share_mtp_io}, qwen4exp.cpp:278-300). The PLE metadata is absent, so the file needs no
     * trunk-level PLE table.
     */
    public static GgufTestWriter qwen4expMtpSidecar(boolean ownEmbedding, boolean ownHead) {
        SyntheticModels.Config c = config().toBuilder().nLayer(config().nLayer() + 1).build();
        GgufTestWriter w = SyntheticText.base("qwen4exp", c);
        SyntheticText.attentionKeys(w, c, true);
        hparams(w, c, 0, QSA_RATIO, 1, false);
        vocab(w, c);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        if (ownEmbedding) {
            t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        }
        int il = c.nLayer() - 1;
        hyperConnections(t, c, blk(il, "hc_attn"));
        hyperConnections(t, c, blk(il, "hc_ffn"));
        attention(t, c, il);
        nextnBlock(t, c, il);
        experts(t, c, il);
        if (ownHead) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    /** The MTP-only sidecar carrying both IO tensors, or neither. */
    public static GgufTestWriter qwen4expMtpSidecar(boolean ownIo) {
        return qwen4expMtpSidecar(ownIo, ownIo);
    }

    /**
     * LLM_ARCH_QWEN4EXP with the PLE module on block 0 and {@code nextn} NextN blocks after the two trunk blocks.
     *
     * @param imageTokenId {@code ple.image_token_id}, the stand-in an embedding row hashes; 0 for the EOS id
     * @param qsa          whether the attended blocks pool their indexer keys (compress ratio {@value #QSA_RATIO})
     * @param nextn        NextN blocks appended after the trunk, each a full-attention QSA block with its fusion and
     *                     gated head
     */
    public static GgufTestWriter qwen4exp(int imageTokenId, boolean qsa, int nextn) {
        SyntheticModels.Config base = config();
        int trunk = base.nLayer();
        SyntheticModels.Config c = nextn == 0 ? base : base.toBuilder().nLayer(trunk + nextn).build();
        GgufTestWriter w = SyntheticText.base("qwen4exp", c);
        SyntheticText.attentionKeys(w, c, true);
        hparams(w, c, imageTokenId, qsa ? QSA_RATIO : 0, nextn, true);
        vocab(w, c);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long hcDim = (long) HC * c.nEmbd();
        int pleHeads = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM;
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.random("per_layer_token_embd.weight", -0.5f, 0.5f, c.nEmbd() / pleHeads, (long) pleHeads * PLE_HEAD_ROWS);
        hyperConnections(t, c, "output_hc");
        for (int il = 0; il < c.nLayer(); il++) {
            hyperConnections(t, c, blk(il, "hc_attn"));
            hyperConnections(t, c, blk(il, "hc_ffn"));
            if (il == 0) {
                gatedDeltaNet(t, c, il);
                ple(t, c, il, hcDim);
            } else {
                attention(t, c, il);
            }
            if (il >= trunk) {
                nextnBlock(t, c, il);
            }
            experts(t, c, il);
        }
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /** The tokenizer of every qwen4exp fixture. */
    private static void vocab(GgufTestWriter w, SyntheticModels.Config c) {
        SyntheticText.bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|im_start|>",
            "<|im_end|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", EOS).bool("tokenizer.ggml.add_bos_token", false);
    }

    private static void hparams(GgufTestWriter w, SyntheticModels.Config c, int imageTokenId, int ratio, int nextn,
                                boolean ple) {
        int pleHeads = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM;
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.u32Array(w.archKey("rope.dimension_sections"), 2, 2, 2, 2);
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        boolean[] recurrent = new boolean[c.nLayer()];
        recurrent[0] = true;
        w.boolArray(w.archKey("attention.recurrent_layers"), recurrent);
        if (nextn > 0) {
            w.u32(w.archKey("nextn_predict_layers"), nextn);
        }
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_feed_forward_length"), c.nFfShexp());
        w.u32(w.archKey("ssm.conv_kernel"), c.ssmConv());
        w.u32(w.archKey("ssm.state_size"), c.ssmState());
        w.u32(w.archKey("ssm.group_count"), c.ssmGroups());
        w.u32(w.archKey("ssm.time_step_rank"), c.ssmDtRank());
        w.u32(w.archKey("ssm.inner_size"), (long) c.ssmState() * c.ssmDtRank());
        w.u32(w.archKey("hyper_connection.count"), HC);
        w.u32(w.archKey("hyper_connection.low_rank"), HC_LOW_RANK);
        w.u32(w.archKey("attention.indexer.head_count"), INDEXER_HEADS);
        w.u32(w.archKey("attention.indexer.key_length"), INDEXER_DIM);
        w.u32(w.archKey("attention.indexer.top_k"), INDEXER_TOP_K);
        long[] ratios = new long[c.nLayer()];
        for (int il = 0; il < ratios.length; il++) {
            ratios[il] = recurrent[il] ? 0 : ratio;
        }
        w.u32Array(w.archKey("attention.compress_ratios"), ratios);
        if (!ple) {
            return;
        }
        w.u32Array(w.archKey("ple.layers"), 0);
        w.u32(w.archKey("ple.ngram_size"), PLE_NGRAM);
        w.u32(w.archKey("ple.heads_per_ngram"), PLE_HEADS_PER_NGRAM);
        w.u32(w.archKey("ple.conv_kernel"), PLE_CONV_KERNEL);
        w.u32(w.archKey("ple.eos_token_id"), EOS);
        w.u32(w.archKey("ple.image_token_id"), imageTokenId);
        w.u32(w.archKey("embedding_length_per_layer_input"), c.nEmbd() / pleHeads);
        w.u64Array(w.archKey("ple.layer_multipliers"), MULTIPLIERS);
        long[] offsets = new long[pleHeads];
        long[] sizes = new long[pleHeads];
        for (int h = 0; h < pleHeads; h++) {
            offsets[h] = (long) h * PLE_HEAD_ROWS;
            sizes[h] = PLE_HEAD_ROWS;
        }
        w.u64Array(w.archKey("ple.head_offsets"), offsets);
        w.u64Array(w.archKey("ple.head_vocab_sizes"), sizes);
    }

    /**
     * The NextN fusion and the gated head of block {@code il}: {@code eh_proj} per hyper-connection stream, the
     * input norms over the embedding and the streamed residual, and the head's stream collapse
     * (qwen4exp.cpp:402-410).
     */
    private static void nextnBlock(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        long hcDim = (long) HC * c.nEmbd();
        t.matrix(blk(il, "nextn.eh_proj.weight"), 2L * c.nEmbd(), c.nEmbd());
        t.ones(blk(il, "nextn.enorm.weight"), c.nEmbd());
        t.ones(blk(il, "nextn.hnorm.weight"), c.nEmbd(), HC);
        t.ones(blk(il, "nextn.hc_head_norm.weight"), c.nEmbd(), HC);
        t.random(blk(il, "nextn.hc_head_down.weight"), -0.1f, 0.1f, hcDim, HC_LOW_RANK);
        t.random(blk(il, "nextn.hc_head_up.weight"), -0.1f, 0.1f, HC_LOW_RANK, hcDim);
    }

    /** One hyper-connection module: the per-stream norm gammas, the low-rank down / up and the inject weights. */
    private static void hyperConnections(SyntheticModels.Tensors t, SyntheticModels.Config c, String prefix) {
        long hcDim = (long) HC * c.nEmbd();
        t.ones(prefix + "_norm.weight", c.nEmbd(), HC);
        t.random(prefix + "_down.weight", -0.1f, 0.1f, hcDim, HC_LOW_RANK);
        t.random(prefix + "_up.weight", -0.1f, 0.1f, HC_LOW_RANK, hcDim);
        if (!prefix.equals("output_hc")) {
            t.random(prefix + "_inject.weight", -0.1f, 0.1f, hcDim, HC);
        }
    }

    /** The GDN mixer with separate qkv / gate and beta / alpha projections. */
    private static void gatedDeltaNet(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        long keyDim = (long) c.ssmState() * c.ssmGroups();
        long valueDim = (long) c.ssmState() * c.ssmDtRank();
        long convDim = keyDim * 2 + valueDim;
        t.matrix(blk(il, "attn_qkv.weight"), c.nEmbd(), convDim);
        t.matrix(blk(il, "attn_gate.weight"), c.nEmbd(), valueDim);
        t.random(blk(il, "ssm_conv1d.weight"), -0.3f, 0.3f, c.ssmConv(), convDim);
        t.random(blk(il, "ssm_dt.bias"), -0.5f, 0.5f, c.ssmDtRank());
        t.random(blk(il, "ssm_a"), -1.5f, -0.5f, c.ssmDtRank());
        t.matrix(blk(il, "ssm_beta.weight"), c.nEmbd(), c.ssmDtRank());
        t.matrix(blk(il, "ssm_alpha.weight"), c.nEmbd(), c.ssmDtRank());
        t.ones(blk(il, "ssm_norm.weight"), c.ssmState());
        t.matrix(blk(il, "ssm_out.weight"), valueDim, c.nEmbd());
    }

    /** The n-gram embedding module of the PLE layer (key / value projections, grouped norms, dilated conv). */
    private static void ple(SyntheticModels.Tensors t, SyntheticModels.Config c, int il, long hcDim) {
        t.random(blk(il, "ple_key.weight"), -0.3f, 0.3f, c.nEmbd(), hcDim);
        t.random(blk(il, "ple_value.weight"), -0.3f, 0.3f, c.nEmbd(), c.nEmbd());
        t.random(blk(il, "ple_norm_key.weight"), 0.5f, 1.5f, c.nEmbd(), HC);
        t.random(blk(il, "ple_norm_query.weight"), 0.5f, 1.5f, c.nEmbd(), HC);
        t.random(blk(il, "ple_norm_conv.weight"), 0.5f, 1.5f, c.nEmbd(), HC);
        t.random(blk(il, "ple_conv1d.weight"), -0.5f, 0.5f, PLE_CONV_KERNEL, hcDim);
    }

    /** The gated full attention ({@code attn_q} holds query and gate per head) with the QSA indexer. */
    private static void attention(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        t.qkvo(il, 2L * c.nHead() * c.headDim());
        t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
        t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
        t.matrix(blk(il, "indexer.q_proj.weight"), c.nEmbd(), (long) INDEXER_HEADS * INDEXER_DIM);
        t.matrix(blk(il, "indexer.k_proj.weight"), c.nEmbd(), INDEXER_DIM);
        t.ones(blk(il, "indexer.q_norm.weight"), INDEXER_DIM);
        t.ones(blk(il, "indexer.k_norm.weight"), INDEXER_DIM);
    }

    /** Routed experts and the sigmoid-gated shared expert. */
    private static void experts(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
        t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
        t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
        t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
        t.random(blk(il, "ffn_gate_inp_shexp.weight"), -0.1f, 0.1f, c.nEmbd());
        t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfShexp());
        t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfShexp());
        t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfShexp(), c.nEmbd());
    }
}
