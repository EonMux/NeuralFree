package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * Synthetic GGUF writers of the architectures whose NextN / MTP heads this task enables, sized like
 * {@link SyntheticModels}: EXAONE 4.5 ({@code exaone4} with appended NextN blocks), EXAONE-MoE ({@code exaone-moe}),
 * BailingMoE V2 ({@code bailingmoe2}), Dots3-Note ({@code dots3note}) and dense Nemotron-H ({@code nemotron_h}).
 *
 * <p>Each builder returns a fresh {@link GgufTestWriter} holding the metadata and tensors the llama.cpp loaders (and
 * the vLLM MTP heads they extend) read. The blocks use the converters' GGUF tensor names with the loader shapes
 * ({@code ne[0]} first).</p>
 */
public final class SyntheticMtpArchModels {

    /** The EXAONE-MoE sliding window written when a config sets none. */
    public static final int EXAONE_MOE_WINDOW = 32;
    /** The Dots3-Note sliding window written when a config sets none. */
    public static final int DOTS3_WINDOW = 32;
    /** The DSA indexer top-k of the Dots3-Note fixture. */
    public static final int DOTS3_TOP_K = 8;

    private SyntheticMtpArchModels() {
    }

    // ================================================================== exaone4 / EXAONE 4.5

    /**
     * LLM_ARCH_EXAONE4 with {@code nextn} appended NextN blocks (exaone4.cpp:24-71): post-attention and post-FFN
     * layernorms, Q/K normalisation and the {@code nextn.eh_proj}/norm tensors vLLM's {@code Exaone4_5_MTP} reads.
     */
    public static GgufTestWriter exaone4Mtp(SyntheticModels.Config c, int nextn) {
        SyntheticText.requireAttention(c, "exaone4");
        if (c.nRot() != c.headDim()) {
            throw new IllegalArgumentException("exaone4 requires rope.dimension_count == head dim, got " + c.nRot()
                    + " vs " + c.headDim());
        }
        requireNextn(nextn, "exaone4");
        int trunk = c.nLayer();
        GgufTestWriter w = SyntheticText.base("exaone4", c.toBuilder().nLayer(trunk + nextn).build());
        SyntheticText.attentionKeys(w, c, false);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        if (nextn > 0) {
            w.u32(w.archKey("nextn_predict_layers"), nextn);
        }
        SyntheticText.bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|user|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < trunk + nextn; il++) {
            t.ones(blk(il, "post_attention_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.gatedFfn(il, c.nFf());
            t.ones(blk(il, "post_ffw_norm.weight"), c.nEmbd());
            if (il >= trunk) {
                nextnFusion(t, c, il);
                t.ones(blk(il, "nextn.shared_head_norm.weight"), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    // ================================================================== exaone-moe

    /**
     * LLM_ARCH_EXAONE_MOE (exaone-moe.cpp:20-100): a dense lead, then routed plus one shared expert, with the NextN
     * blocks carrying a dense FFN (exaone.py:178-200, {@code is_mtp=True}).
     */
    public static GgufTestWriter exaoneMoeMtp(SyntheticModels.Config c, int nextn) {
        SyntheticText.requireAttention(c, "exaone-moe");
        requireExperts(c, "exaone-moe");
        requireNextn(nextn, "exaone-moe");
        int trunk = c.nLayer();
        int window = c.slidingWindow() > 0 ? c.slidingWindow() : EXAONE_MOE_WINDOW;
        GgufTestWriter w = SyntheticText.base("exaone-moe", c.toBuilder().nLayer(trunk + nextn).build());
        SyntheticText.attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("attention.sliding_window"), window);
        w.boolArray(w.archKey("attention.sliding_window_pattern"), allTrue(trunk + nextn));
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("expert_shared_feed_forward_length"), c.nFfExp());
        w.f32(w.archKey("expert_weights_scale"), 1f);
        w.bool(w.archKey("expert_weights_norm"), true);
        w.u32(w.archKey("expert_gating_func"), 2);
        w.u32(w.archKey("leading_dense_block_count"), 1);
        if (nextn > 0) {
            w.u32(w.archKey("nextn_predict_layers"), nextn);
        }
        SyntheticText.bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|user|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < trunk + nextn; il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (il == 0 || il >= trunk) {
                t.gatedFfn(il, c.nFf());
            } else {
                moeFfn(t, c, il);
            }
            if (il >= trunk) {
                nextnFusion(t, c, il);
                t.ones(blk(il, "nextn.shared_head_norm.weight"), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    // ================================================================== bailingmoe2

    /**
     * LLM_ARCH_BAILINGMOE2 (bailingmoe2.cpp:20-82): fused QKV with Q/K norms, a dense lead and biased routed plus
     * shared experts; the NextN blocks carry the experts and the {@code layer_output_norm} head norm of vLLM's
     * {@code BailingMoeV25MTPModel}.
     */
    public static GgufTestWriter bailingMoe2Mtp(SyntheticModels.Config c, int nextn) {
        SyntheticText.requireAttention(c, "bailingmoe2");
        requireExperts(c, "bailingmoe2");
        requireNextn(nextn, "bailingmoe2");
        int trunk = c.nLayer();
        GgufTestWriter w = SyntheticText.base("bailingmoe2", c.toBuilder().nLayer(trunk + nextn).build());
        SyntheticText.attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("leading_dense_block_count"), 1);
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("expert_shared_feed_forward_length"), c.nFfExp());
        w.f32(w.archKey("expert_weights_scale"), 1f);
        w.bool(w.archKey("expert_weights_norm"), false);
        w.u32(w.archKey("expert_gating_func"), 2);
        if (nextn > 0) {
            w.u32(w.archKey("nextn_predict_layers"), nextn);
        }
        SyntheticText.bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|user|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        long qkv = c.nEmbd() + 2L * c.nHeadKv() * c.headDim();
        for (int il = 0; il < trunk + nextn; il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.matrix(blk(il, "attn_qkv.weight"), c.nEmbd(), qkv);
            t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (il == 0) {
                t.gatedFfn(il, c.nFf());
            } else {
                moeFfn(t, c, il);
            }
            if (il >= trunk) {
                nextnFusion(t, c, il);
                t.ones(blk(il, "layer_output_norm.weight"), c.nEmbd());
                t.matrix(blk(il, "nextn.shared_head_head.weight"), c.nEmbd(), c.nVocab());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    // ================================================================== dots3-note

    /**
     * LLM_ARCH_DOTS3NOTE (dots3note.cpp:48-145): absorbed MLA with the DSA indexer on the full-attention trunk
     * layers, the sliding-window MLA geometry on the NextN block, dense FFNs on the lead and the NextN blocks and
     * routed experts elsewhere.
     */
    public static GgufTestWriter dots3NoteMtp(SyntheticModels.Config c, int nextn) {
        SyntheticText.requireAttention(c, "dots3note");
        requireExperts(c, "dots3note");
        requireNextn(nextn, "dots3note");
        requirePositive("dots3note non-rotary query width", c.headDim() - c.nRot());
        int trunk = c.nLayer();
        int window = c.slidingWindow() > 0 ? c.slidingWindow() : DOTS3_WINDOW;
        int kvLora = SyntheticMlaModels.KV_LORA;
        int qLora = SyntheticMlaModels.Q_LORA;
        int vHead = SyntheticMlaModels.V_HEAD;
        GgufTestWriter w = SyntheticText.base("dots3note", c.toBuilder().nLayer(trunk + nextn).build());
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), kvLora + c.nRot());
        w.u32(w.archKey("attention.value_length"), kvLora);
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), vHead);
        w.u32(w.archKey("attention.q_lora_rank"), qLora);
        w.u32(w.archKey("attention.kv_lora_rank"), kvLora);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("leading_dense_block_count"), 1);
        w.f32(w.archKey("expert_weights_scale"), 1f);
        w.bool(w.archKey("expert_weights_norm"), false);
        w.u32(w.archKey("expert_gating_func"), 2);
        w.u32(w.archKey("attention.kv_lora_rank_swa"), kvLora);
        w.u32(w.archKey("attention.key_length_swa"), kvLora + c.nRot());
        w.u32(w.archKey("attention.value_length_swa"), kvLora);
        w.u32(w.archKey("attention.key_length_mla_swa"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla_swa"), vHead);
        w.u32(w.archKey("attention.sliding_window"), window);
        w.f32(w.archKey("rope.freq_base_swa"), c.ropeFreqBase());
        w.boolArray(w.archKey("attention.sliding_window_pattern"), slidingPattern(trunk + nextn, trunk));
        w.u32(w.archKey("attention.indexer.head_count"), SyntheticMlaModels.INDEXER_HEADS);
        w.u32(w.archKey("attention.indexer.key_length"), SyntheticMlaModels.INDEXER_DIM);
        w.u32(w.archKey("attention.indexer.top_k"), DOTS3_TOP_K);
        w.boolArray(w.archKey("attention.indexer.types"), allTrue(trunk));
        if (nextn > 0) {
            w.u32(w.archKey("nextn_predict_layers"), nextn);
        }
        SyntheticText.bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|user|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < trunk + nextn; il++) {
            mla(t, c, il, qLora, kvLora, vHead);
            if (il < trunk) {
                indexer(t, c, il, qLora);
            }
            if (il == 0 || il >= trunk) {
                t.gatedFfn(il, c.nFf());
            } else {
                moeFfn(t, c, il);
            }
            if (il >= trunk) {
                nextnFusion(t, c, il);
                t.matrix(blk(il, "nextn.embed_tokens.weight"), c.nEmbd(), c.nVocab());
                t.ones(blk(il, "nextn.shared_head_norm.weight"), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /** The MLA projections of block {@code il}; every kind shares the fixture's geometry. */
    private static void mla(SyntheticModels.Tensors t, SyntheticModels.Config c, int il, int qLora, int kvLora,
                            int vHead) {
        int nope = c.headDim() - c.nRot();
        t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
        t.matrix(blk(il, "attn_q_a.weight"), c.nEmbd(), qLora);
        t.ones(blk(il, "attn_q_a_norm.weight"), qLora);
        t.matrix(blk(il, "attn_q_b.weight"), qLora, (long) c.nHead() * c.headDim());
        t.matrix(blk(il, "attn_kv_a_mqa.weight"), c.nEmbd(), kvLora + c.nRot());
        t.ones(blk(il, "attn_kv_a_norm.weight"), kvLora);
        t.matrix(blk(il, "attn_k_b.weight"), nope, kvLora, c.nHead());
        t.matrix(blk(il, "attn_v_b.weight"), kvLora, vHead, c.nHead());
        t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * vHead, c.nEmbd());
        t.ones(blk(il, "attn_k_norm.weight"), c.nRot());
        t.matrix(blk(il, "attn_gate.weight"), c.nEmbd(), c.nHead());
        t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
    }

    /** The DSA lightning indexer of the full-indexer trunk block {@code il}. */
    private static void indexer(SyntheticModels.Tensors t, SyntheticModels.Config c, int il, int qLora) {
        int heads = SyntheticMlaModels.INDEXER_HEADS;
        int dim = SyntheticMlaModels.INDEXER_DIM;
        t.ones(blk(il, "indexer.k_norm.weight"), dim);
        t.bias(blk(il, "indexer.k_norm.bias"), dim);
        t.matrix(blk(il, "indexer.proj.weight"), c.nEmbd(), heads);
        t.matrix(blk(il, "indexer.attn_k.weight"), c.nEmbd(), dim);
        t.matrix(blk(il, "indexer.attn_q_b.weight"), qLora, (long) heads * dim);
    }

    // ================================================================== dense nemotron-h

    /**
     * LLM_ARCH_NEMOTRON_H with a dense trunk (attention then squared-ReLU MLP) and one attention-only NextN block:
     * the {@code mtp_hybrid_override_pattern "*"} head of vLLM's {@code NemotronHMTPModel}, folded by the converter
     * into {@code blk.<trunk>}.
     */
    public static GgufTestWriter nemotronHDenseMtp(SyntheticModels.Config c, int nextn) {
        SyntheticText.requireAttention(c, "nemotron_h");
        requireNextn(nextn, "nemotron_h");
        if (nextn != 1) {
            throw new IllegalArgumentException("dense nemotron_h MTP supports one NextN block, got " + nextn);
        }
        int trunk = c.nLayer();
        int blocks = trunk + nextn;
        GgufTestWriter w = SyntheticText.base("nemotron_h", c.toBuilder().nLayer(blocks).build());
        w.u32Array(w.archKey("feed_forward_length"), layerFf(c, blocks));
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32Array(w.archKey("attention.head_count_kv"), layerKv(c, blocks));
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.f32(w.archKey("attention.layer_norm_epsilon"), 1e-5f);
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), 1e-5f);
        w.u32(w.archKey("ssm.conv_kernel"), 4);
        w.u32(w.archKey("ssm.inner_size"), 2L * c.nEmbd());
        w.u32(w.archKey("ssm.state_size"), 16);
        w.u32(w.archKey("ssm.time_step_rank"), 8);
        w.u32(w.archKey("ssm.group_count"), 2);
        w.u32(w.archKey("nextn_predict_layers"), nextn);
        SyntheticText.bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|user|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < blocks; il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            if (il != trunk - 1) {
                t.qkvo(il, c.nHead() * c.headDim());
            } else {
                t.matrix(blk(il, "ffn_up.weight"), c.nEmbd(), c.nFf());
                t.matrix(blk(il, "ffn_down.weight"), c.nFf(), c.nEmbd());
            }
            if (il >= trunk) {
                t.qkvo(il, c.nHead() * c.headDim());
                nextnFusion(t, c, il);
                t.ones(blk(il, "nextn.shared_head_norm.weight"), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    private static long[] layerFf(SyntheticModels.Config c, int blocks) {
        long[] ff = new long[blocks];
        ff[blocks - 2] = c.nFf();
        return ff;
    }

    private static long[] layerKv(SyntheticModels.Config c, int blocks) {
        long[] kv = new long[blocks];
        for (int il = 0; il < blocks; il++) {
            kv[il] = il == blocks - 2 ? 0 : c.nHeadKv();
        }
        return kv;
    }

    // ================================================================== shared tensors

    /** The {@code nextn.eh_proj}/enorm/hnorm tensors of block {@code il}. */
    private static void nextnFusion(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        t.matrix(blk(il, "nextn.eh_proj.weight"), 2L * c.nEmbd(), c.nEmbd());
        t.ones(blk(il, "nextn.enorm.weight"), c.nEmbd());
        t.ones(blk(il, "nextn.hnorm.weight"), c.nEmbd());
    }

    /** The routed experts, their router and one shared expert of block {@code il}. */
    private static void moeFfn(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
        t.bias(blk(il, "exp_probs_b.bias"), c.nExpert());
        t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
        t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
        t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
        t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
        t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
        t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
    }

    private static boolean[] allTrue(int n) {
        boolean[] flags = new boolean[n];
        java.util.Arrays.fill(flags, true);
        return flags;
    }

    private static boolean[] slidingPattern(int blocks, int trunk) {
        boolean[] flags = new boolean[blocks];
        for (int il = trunk; il < blocks; il++) {
            flags[il] = true;
        }
        return flags;
    }

    private static void requireNextn(int nextn, String arch) {
        if (nextn < 0) {
            throw new IllegalArgumentException(arch + " needs a non-negative NextN block count, got " + nextn);
        }
    }

    private static void requireExperts(SyntheticModels.Config c, String arch) {
        requirePositive(arch + " expert count", c.nExpert());
        requirePositive(arch + " expert feed-forward length", c.nFfExp());
        if (c.nExpertUsed() <= 0 || c.nExpertUsed() > c.nExpert()) {
            throw new IllegalArgumentException(arch + " expert_used_count must be in [1, " + c.nExpert() + "], got "
                    + c.nExpertUsed());
        }
    }
}
