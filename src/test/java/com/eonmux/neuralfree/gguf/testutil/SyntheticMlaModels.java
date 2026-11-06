package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * Synthetic GGUF writers of the multi-head latent attention families, sized like {@link SyntheticModels}: the
 * absorbed-MLA deepseek2 layout ({@code attn_k_b}/{@code attn_v_b} per head over a latent cache) and deepseek32, the
 * same layout with the DSA lightning indexer, which runs the sparse MLA attention. Block 0 is a dense lead, the
 * later blocks route {@code nExpert} experts plus one shared expert. Byte-level BPE tokenizer.
 *
 * <p>The MLA geometry comes from the config: {@code headDim} is the query/key head ({@code nope + rope}),
 * {@code nRot} the rotary part, and the value head and the latent widths are fixed by {@link #V_HEAD},
 * {@link #KV_LORA} and {@link #Q_LORA}.</p>
 */
public final class SyntheticMlaModels {

    /** Rank of the low-rank query projection. */
    public static final int Q_LORA = 32;
    /** Rank of the shared latent KV row. */
    public static final int KV_LORA = 32;
    /** Value head width after the per-head {@code attn_v_b} expansion. */
    public static final int V_HEAD = 16;
    /** Lightning-indexer heads and head width of deepseek32 and glm5-next. */
    public static final int INDEXER_HEADS = 2;
    public static final int INDEXER_DIM = 64;
    /** The glm5-next k-pool size and the rows its QSA indexer selects. */
    public static final int KPOOL = 4;
    public static final int QSA_TOP_K = 32;

    private SyntheticMlaModels() {
    }

    /** 2 layers, 4 heads of key dim 32 (16 rotary), 4 experts of width 32 (2 used), layer 1 MoE. */
    public static SyntheticModels.Config config() {
        return new SyntheticModels.Config.Builder().nLayer(2).nHead(4).nHeadKv(1).headDim(32).nRot(16).nFf(128)
                .nExpert(4).nExpertUsed(2).nFfExp(32).normEps(1e-6f).build();
    }

    public static GgufTestWriter deepseek2() {
        return deepseek2(config());
    }

    /** LLM_ARCH_DEEPSEEK2 with the absorbed MLA weights of current llama.cpp conversions (models/deepseek2.cpp). */
    public static GgufTestWriter deepseek2(SyntheticModels.Config c) {
        return mla("deepseek2", c, -1);
    }

    public static GgufTestWriter deepseek32() {
        return deepseek32(config(), 64);
    }

    /**
     * LLM_ARCH_DEEPSEEK32 (models/deepseek32.cpp): deepseek2's layout with the lightning indexer selecting
     * {@code topK} cached rows per query.
     */
    public static GgufTestWriter deepseek32(SyntheticModels.Config c, int topK) {
        requirePositive("deepseek32 indexer top-k", topK);
        return mla("deepseek32", c, topK);
    }

    public static GgufTestWriter glm5Next() {
        return glm5Next(new SyntheticModels.Config.Builder().nLayer(2).nHead(4).nHeadKv(1).headDim(32).nRot(0)
                .nFf(128).nExpert(4).nExpertUsed(2).nFfExp(32).normEps(1e-6f).build());
    }

    /**
     * LLM_ARCH_GLM5_NEXT (models/glm5-next.cpp) with nope-only MLA attention ({@code nRot} must be 0, glm5-next.cpp
     * load_hparams) on every layer (no KDA layers): the QSA k-pool
     * indexer of {@link #KPOOL}-token pools selecting {@link #QSA_TOP_K} rows (every pool of a 32-token window),
     * four hyper-connection streams collapsed by their mean before the head, sigmoid-gated experts with a selection
     * bias and one shared expert.
     */
    public static GgufTestWriter glm5Next(SyntheticModels.Config c) {
        if (c.nRot() != 0) {
            throw new IllegalArgumentException("glm5-next attention is nope-only, got rope dimension " + c.nRot());
        }
        GgufTestWriter w = mlaHeader("glm5-next", c);
        int hc = 4;
        int hcMix = (2 + hc) * hc;
        w.u32(w.archKey("ssm.conv_kernel"), 4);
        w.u32(w.archKey("kda.head_dim"), c.headDim());
        w.u32(w.archKey("attention.indexer.head_count"), INDEXER_HEADS);
        w.u32(w.archKey("attention.indexer.key_length"), INDEXER_DIM);
        w.u32(w.archKey("attention.indexer.top_k"), QSA_TOP_K);
        w.u32(w.archKey("attention.indexer.kpool"), KPOOL);
        w.u32(w.archKey("hyper_connection.count"), hc);
        w.u32(w.archKey("hyper_connection.sinkhorn_iterations"), 20);
        w.f32(w.archKey("hyper_connection.epsilon"), 1e-6f);
        tokenizer(w, c);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            for (String site : new String[] {"attn", "ffn"}) {
                t.random(blk(il, "hc_" + site + "_fn.weight"), -0.05f, 0.05f, (long) hc * c.nEmbd(), hcMix);
                t.random(blk(il, "hc_" + site + "_base.weight"), -0.1f, 0.1f, hcMix);
                t.random(blk(il, "hc_" + site + "_scale.weight"), 0.5f, 1.0f, 3);
            }
            mlaAttention(t, c, il);
            t.matrix(blk(il, "indexer.attn_q_b.weight"), Q_LORA, (long) INDEXER_HEADS * INDEXER_DIM);
            t.matrix(blk(il, "indexer.attn_k.weight"), c.nEmbd(), INDEXER_DIM);
            t.ones(blk(il, "indexer.k_norm.weight"), INDEXER_DIM);
            w.tensorFilled(blk(il, "indexer.k_norm.bias"), GGMLType.F32, 0.0f, INDEXER_DIM);
            t.matrix(blk(il, "indexer.proj.weight"), c.nEmbd(), INDEXER_HEADS);
            t.matrix(blk(il, "indexer_compressor_gate.weight"), c.nEmbd(), INDEXER_DIM);
            t.random(blk(il, "indexer_compressor_ape.weight"), -0.1f, 0.1f, INDEXER_DIM, KPOOL);
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (il == 0) {
                t.gatedFfn(il, c.nFf());
            } else {
                routedExperts(t, c, il);
                t.random(blk(il, "exp_probs_b.bias"), -0.02f, 0.02f, c.nExpert());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    private static GgufTestWriter mla(String arch, SyntheticModels.Config c, int indexerTopK) {
        GgufTestWriter w = mlaHeader(arch, c);
        w.u32(w.archKey("expert_gating_func"), 1);
        w.f32(w.archKey("expert_weights_scale"), 1.0f);
        w.bool(w.archKey("expert_weights_norm"), false);
        if (indexerTopK > 0) {
            w.u32(w.archKey("attention.indexer.head_count"), INDEXER_HEADS);
            w.u32(w.archKey("attention.indexer.key_length"), INDEXER_DIM);
            w.u32(w.archKey("attention.indexer.top_k"), indexerTopK);
        }
        tokenizer(w, c);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            mlaAttention(t, c, il);
            if (indexerTopK > 0) {
                t.matrix(blk(il, "indexer.attn_q_b.weight"), Q_LORA, (long) INDEXER_HEADS * INDEXER_DIM);
                t.matrix(blk(il, "indexer.attn_k.weight"), c.nEmbd(), INDEXER_DIM);
                t.ones(blk(il, "indexer.k_norm.weight"), INDEXER_DIM);
                w.tensorFilled(blk(il, "indexer.k_norm.bias"), GGMLType.F32, 0.0f, INDEXER_DIM);
                t.matrix(blk(il, "indexer.proj.weight"), c.nEmbd(), INDEXER_HEADS);
            }
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (il == 0) {
                t.gatedFfn(il, c.nFf());
            } else {
                routedExperts(t, c, il);
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /** The metadata every MLA family shares: trunk sizes, the MLA geometry, the experts and the dense lead. */
    private static GgufTestWriter mlaHeader(String arch, SyntheticModels.Config c) {
        requirePositive(arch + " expert count", c.nExpert());
        requirePositive(arch + " expert feed-forward length", c.nFfExp());
        int rope = c.nRot();
        requirePositive(arch + " non-rotary query width", c.headDim() - rope);
        GgufTestWriter w = SyntheticText.base(arch, c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), KV_LORA + rope);
        w.u32(w.archKey("attention.value_length"), KV_LORA);
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), V_HEAD);
        w.u32(w.archKey("attention.q_lora_rank"), Q_LORA);
        w.u32(w.archKey("attention.kv_lora_rank"), KV_LORA);
        w.u32(w.archKey("rope.dimension_count"), rope);
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("leading_dense_block_count"), 1);
        return w;
    }

    private static void tokenizer(GgufTestWriter w, SyntheticModels.Config c) {
        SyntheticText.bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);
    }

    /** The absorbed MLA attention tensors of block {@code il}. */
    private static void mlaAttention(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        int rope = c.nRot();
        t.matrix(blk(il, "attn_q_a.weight"), c.nEmbd(), Q_LORA);
        t.ones(blk(il, "attn_q_a_norm.weight"), Q_LORA);
        t.matrix(blk(il, "attn_q_b.weight"), Q_LORA, (long) c.nHead() * c.headDim());
        t.matrix(blk(il, "attn_kv_a_mqa.weight"), c.nEmbd(), KV_LORA + rope);
        t.ones(blk(il, "attn_kv_a_norm.weight"), KV_LORA);
        t.matrix(blk(il, "attn_k_b.weight"), c.headDim() - rope, KV_LORA, c.nHead());
        t.matrix(blk(il, "attn_v_b.weight"), KV_LORA, V_HEAD, c.nHead());
        t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * V_HEAD, c.nEmbd());
    }

    /** The routed experts, their router and one shared expert of block {@code il}. */
    private static void routedExperts(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
        t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
        t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
        t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
        t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
        t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
        t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
    }
}
