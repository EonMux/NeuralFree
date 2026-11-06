package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * Multi-head latent attention fixtures of {@link SyntheticModels}, all with the DSA lightning indexer where the
 * architecture has one: deepseek2 (MLA with the q_lora path, absorbed latent cache), deepseek32 (the same layout
 * plus the indexer) and mistral-large4 (MLA without query compression, SSMax, per-layer LeRoPE, a KV source map).
 * Sources: llama.cpp src/models/{deepseek2,deepseek32,mistral-large4}.cpp
 * {@code load_arch_hparams}/{@code load_arch_tensors}.
 */
public final class SyntheticMlaDsaModels {

    /** Rank of the low-rank query projection of deepseek2 and deepseek32. */
    public static final int Q_LORA = 32;
    /** Rank of the shared latent KV row of deepseek2 and deepseek32. */
    public static final int KV_LORA = 32;
    /** Lightning-indexer heads and head width of deepseek32 and mistral-large4. */
    public static final int IDX_HEADS = 2;
    public static final int IDX_DIM = 64;

    private SyntheticMlaDsaModels() {
    }

    // ================================================================== deepseek2 / deepseek32

    public static GgufTestWriter deepseek2() {
        return deepseek2(SyntheticModels.Config.mistralLarge4());
    }

    /**
     * LLM_ARCH_DEEPSEEK2: MLA with the q_lora path ({@code attn_q_a} -> {@code attn_q_a_norm} -> {@code attn_q_b})
     * and the absorbed latent cache of {@code kv_lora + n_rot} per row (models/deepseek2.cpp); layer 0 runs a dense
     * FFN, layer 1 routed experts with a shared expert.
     */
    public static GgufTestWriter deepseek2(SyntheticModels.Config c) {
        return deepseekMla("deepseek2", c, false);
    }

    public static GgufTestWriter deepseek32() {
        return deepseek32(SyntheticModels.Config.mistralLarge4());
    }

    /**
     * LLM_ARCH_DEEPSEEK32: the {@link #deepseek2(SyntheticModels.Config)} layers plus the DSA lightning indexer on
     * every layer, whose queries read the normalized q_lora rank (models/deepseek32.cpp:207-218). Indexer top-k 8 of
     * {@code 2 x 64}-wide heads (one Hadamard block, the always-rotated deepseek32 indexer cache).
     */
    public static GgufTestWriter deepseek32(SyntheticModels.Config c) {
        return deepseekMla("deepseek32", c, true);
    }

    private static GgufTestWriter deepseekMla(String arch, SyntheticModels.Config c, boolean indexer) {
        requireAttention(c, arch);
        requirePositive(arch + " expert feed-forward length", c.nFfExp());
        requirePositive(arch + " expert count", c.nExpert());
        requirePositive(arch + " no-rope head dims", c.headDim() - c.nRot());
        GgufTestWriter w = base(arch, c);
        requirePositive(arch + " kv_lora_rank", c.kvLora());
        long kvLora = c.kvLora();
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), kvLora + c.nRot());
        w.u32(w.archKey("attention.value_length"), kvLora);
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("attention.q_lora_rank"), Q_LORA);
        w.u32(w.archKey("attention.kv_lora_rank"), kvLora);
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), 2);
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("leading_dense_block_count"), 1);
        w.u32(w.archKey("expert_gating_func"), 1);
        if (indexer) {
            w.u32(w.archKey("attention.indexer.head_count"), IDX_HEADS);
            w.u32(w.archKey("attention.indexer.key_length"), IDX_DIM);
            w.u32(w.archKey("attention.indexer.top_k"), 8);
        }
        bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            deepseekAttention(t, c, il);
            if (indexer) {
                deepseekIndexer(w, t, c, il);
            }
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (il == 0) {
                t.gatedFfn(il, c.nFf());
            } else {
                t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
                t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
                t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
                t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
                t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    private static void deepseekAttention(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        long qLora = Q_LORA;
        long kvLora = c.kvLora();
        t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
        t.matrix(blk(il, "attn_q_a.weight"), c.nEmbd(), qLora);
        t.ones(blk(il, "attn_q_a_norm.weight"), qLora);
        t.matrix(blk(il, "attn_q_b.weight"), qLora, (long) c.nHead() * c.headDim());
        t.matrix(blk(il, "attn_kv_a_mqa.weight"), c.nEmbd(), kvLora + c.nRot());
        t.ones(blk(il, "attn_kv_a_norm.weight"), kvLora);
        t.matrix(blk(il, "attn_k_b.weight"), c.headDim() - c.nRot(), kvLora, c.nHead());
        t.matrix(blk(il, "attn_v_b.weight"), kvLora, c.headDim(), c.nHead());
        t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
    }

    private static void deepseekIndexer(GgufTestWriter w, SyntheticModels.Tensors t, SyntheticModels.Config c,
                                        int il) {
        long width = IDX_HEADS * IDX_DIM;
        t.matrix(blk(il, "indexer.attn_q_b.weight"), Q_LORA, width);
        t.matrix(blk(il, "indexer.attn_k.weight"), c.nEmbd(), IDX_DIM);
        t.ones(blk(il, "indexer.k_norm.weight"), IDX_DIM);
        w.tensorFilled(blk(il, "indexer.k_norm.bias"), GGMLType.F32, 0.0f, IDX_DIM);
        t.matrix(blk(il, "indexer.proj.weight"), c.nEmbd(), IDX_HEADS);
    }

    // ================================================================== mistral-large4

    public static GgufTestWriter mistralLarge4() {
        return mistralLarge4(SyntheticModels.Config.mistralLarge4());
    }

    /**
     * A mistral-large4 file whose layer 0 rotates and whose other layers are NoPE ({@code attention.rope_pattern}):
     * the NoPE layers have no rotary part in {@code attn_kv_a_mqa}, absorb the whole query head in
     * {@code attn_k_b} and pad their latent rows with zero rope dims; every layer owns its KV cache.
     */
    public static GgufTestWriter mistralLarge4NoPe(SyntheticModels.Config c) {
        boolean[] pattern = new boolean[c.nLayer()];
        if (pattern.length > 0) {
            pattern[0] = true;
        }
        return mistralLarge4(c, pattern);
    }

    /**
     * LLM_ARCH_MISTRAL_LARGE4: MLA (no Q LoRA) with the DSA lightning indexer on every layer, SSMax scale and
     * bias, per-layer LeRoPE factors and, without {@code ropePattern}, layer 1 reading layer 0's KV cache
     * ({@code attention.kv_source_map}). Layer 0 runs a dense FFN, layer 1 routed experts with a shared expert and
     * softmax gating (models/mistral-large4.cpp:6-148).
     */
    public static GgufTestWriter mistralLarge4(SyntheticModels.Config c) {
        return mistralLarge4(c, null);
    }

    private static GgufTestWriter mistralLarge4(SyntheticModels.Config c, boolean[] ropePattern) {
        requireAttention(c, "mistral-large4");
        requirePositive("mistral-large4 expert feed-forward length", c.nFfExp());
        requirePositive("mistral-large4 expert count", c.nExpert());
        long kvLora = KV_LORA;
        long idxDim = IDX_DIM;
        long idxHeads = IDX_HEADS;
        GgufTestWriter w = base("mistral-large4", c);
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), c.nHeadKv());
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("attention.layer_norm_epsilon"), c.normEps());
        w.u32(w.archKey("attention.kv_lora_rank"), kvLora);
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), 2);
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("leading_dense_block_count"), 1);
        w.u32(w.archKey("expert_gating_func"), 1);
        w.u32(w.archKey("attention.indexer.head_count"), idxHeads);
        w.u32(w.archKey("attention.indexer.key_length"), idxDim);
        w.u32(w.archKey("attention.indexer.top_k"), 8);
        if (ropePattern == null) {
            w.i32Array(w.archKey("attention.kv_source_map"), 0, 0);
        } else {
            int[] pattern = new int[ropePattern.length];
            for (int il = 0; il < pattern.length; il++) {
                pattern[il] = ropePattern[il] ? 1 : 0;
            }
            w.i32Array(w.archKey("attention.rope_pattern"), pattern);
            int[] identity = new int[ropePattern.length];
            for (int il = 0; il < identity.length; il++) {
                identity[il] = il;
            }
            w.i32Array(w.archKey("attention.kv_source_map"), identity);
        }
        w.f32(w.archKey("final_logit_softcapping"), 30.0f);
        bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("rope_freqs.weight", c.nRot() / 2);
        for (int il = 0; il < c.nLayer(); il++) {
            boolean layerRope = ropePattern == null || ropePattern[il];
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.matrix(blk(il, "attn_q.weight"), c.nEmbd(), (long) c.nHead() * c.headDim());
            // a shared-KV layer stores no latent rows and carries no attn_kv_a (mistral-large4.cpp:96-99, 152-153)
            boolean ownsKv = ropePattern != null || il == 0;
            if (ownsKv) {
                t.matrix(blk(il, "attn_kv_a_mqa.weight"), c.nEmbd(), kvLora + (layerRope ? c.nRot() : 0));
                t.ones(blk(il, "attn_kv_a_norm.weight"), kvLora);
            }
            // a NoPE layer absorbs the whole query head, a roped layer only its nope part (mistral-large4.cpp:85-101)
            t.matrix(blk(il, "attn_k_b.weight"), layerRope ? c.headDim() - c.nRot() : c.headDim(), kvLora, c.nHead());
            t.matrix(blk(il, "attn_v_b.weight"), kvLora, c.headDim(), c.nHead());
            t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
            t.random(blk(il, "attn_ssmax.weight"), 0.0f, 0.1f, c.nHead());
            t.random(blk(il, "attn_ssmax.bias"), -0.05f, 0.05f, c.nHead());
            t.random(blk(il, "indexer.q_norm.weight"), 0.5f, 1.5f, c.nEmbd());
            t.random(blk(il, "indexer.q_norm.bias"), -0.1f, 0.1f, c.nEmbd());
            t.matrix(blk(il, "indexer.attn_q_b.weight"), c.nEmbd(), idxHeads * idxDim);
            t.ones(blk(il, "indexer.q_norm.weight"), c.nEmbd());
            w.tensorFilled(blk(il, "indexer.q_norm.bias"), GGMLType.F32, 0.0f, c.nEmbd());
            t.matrix(blk(il, "indexer.attn_k.weight"), c.nEmbd(), idxDim);
            t.ones(blk(il, "indexer.k_norm.weight"), idxDim);
            w.tensorFilled(blk(il, "indexer.k_norm.bias"), GGMLType.F32, 0.0f, idxDim);
            t.matrix(blk(il, "indexer.proj.weight"), c.nEmbd(), idxHeads);
            t.random(blk(il, "indexer.ssmax.weight"), 0.0f, 0.1f, idxHeads);
            t.random(blk(il, "indexer.ssmax.bias"), -0.05f, 0.05f, idxHeads);
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (il == 0) {
                t.gatedFfn(il, c.nFf());
            } else {
                t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
                t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
                t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
                t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
                t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }
}
