package com.eonmux.neuralfree.gguf.testutil;

import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireSsm;

/**
 * Hybrid decoder fixtures of {@link SyntheticModels}, mixing attention with SSM, gated-delta-net, KDA or short
 * convolution layers: qwen3next (hybrid gated-delta-net + attention + MoE), lfm3 (hybrid short convolution + gated
 * SWA attention), lfm2 (alternating short convolutions and attention), plamo2 (hybrid PLaMo-2 Mamba + fused-QKV
 * attention) and kimi-linear (hybrid KDA + MLA + MoE). Sources: llama.cpp
 * src/models/{qwen3next,lfm3,lfm2,plamo2,kimi-linear}.cpp {@code load_arch_hparams}/{@code load_arch_tensors}.
 */
public final class SyntheticHybridModels {

    /** The {@code expert_gating_func} code of sigmoid gating (GatingFunc.SIGMOID). */
    private static final int GATING_SIGMOID = 2;
    /** Value head width of the kimi-linear MLA layers, standing in for the checkpoints' 128. */
    private static final int MLA_VALUE_HEAD = 16;

    private SyntheticHybridModels() {
    }

    // ================================================================== qwen3next

    public static GgufTestWriter qwen3next() {
        return qwen3next(SyntheticModels.Config.qwen3next());
    }

    /**
     * LLM_ARCH_QWEN3NEXT: layer {@code il} is full attention when
     * {@code (il + 1) % full_attention_interval == 0}, else a gated delta net in the
     * {@code attn_qkv}/{@code attn_gate} layout. Attention layers have a gated Q projection
     * ({@code 2 * n_head * head_dim} rows) and per-head Q/K norms. Every layer has a routed MoE with
     * a sigmoid-gated shared expert. GDN dims: head_k = head_v = {@code ssm.state_size},
     * k heads = {@code ssm.group_count}, v heads = {@code ssm.time_step_rank}.
     */
    public static GgufTestWriter qwen3next(SyntheticModels.Config c) {
        requireAttention(c, "qwen3next");
        requirePositive("qwen3next expert count", c.nExpert());
        if (c.nExpertUsed() <= 0 || c.nExpertUsed() > c.nExpert()) {
            throw new IllegalArgumentException("qwen3next expert_used_count must be in [1, " + c.nExpert() + "], got "
                    + c.nExpertUsed());
        }
        requirePositive("qwen3next expert feed-forward length", c.nFfExp());
        requirePositive("qwen3next shared expert feed-forward length", c.nFfShexp());
        requirePositive("qwen3next full attention interval", c.fullAttentionInterval());
        requireSsm(c, "qwen3next");
        if (c.ssmDtRank() % c.ssmGroups() != 0) {
            throw new IllegalArgumentException("qwen3next value heads (" + c.ssmDtRank()
                    + ") must be a multiple of key heads (" + c.ssmGroups() + ")");
        }
        if (c.nRot() <= 0 || c.nRot() > c.headDim() || c.nRot() % 2 != 0) {
            throw new IllegalArgumentException("qwen3next rope dimension must be even and in (0, head dim], got "
                    + c.nRot());
        }
        long headK = c.ssmState();
        long headV = c.ssmState();
        long keyDim = headK * c.ssmGroups();
        long valueDim = headV * c.ssmDtRank();
        long convDim = keyDim * 2 + valueDim;

        GgufTestWriter w = base("qwen3next", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_feed_forward_length"), c.nFfShexp());
        w.u32(w.archKey("ssm.conv_kernel"), c.ssmConv());
        w.u32(w.archKey("ssm.state_size"), c.ssmState());
        w.u32(w.archKey("ssm.group_count"), c.ssmGroups());
        w.u32(w.archKey("ssm.time_step_rank"), c.ssmDtRank());
        w.u32(w.archKey("ssm.inner_size"), valueDim);
        w.u32(w.archKey("full_attention_interval"), c.fullAttentionInterval());
        bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|im_start|>", "<|im_end|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.ones(blk(il, "post_attention_norm.weight"), c.nEmbd());
            if ((il + 1) % c.fullAttentionInterval() == 0) {
                t.qkvo(il, 2L * c.nHead() * c.headDim());
                t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
                t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            } else {
                t.matrix(blk(il, "attn_qkv.weight"), c.nEmbd(), convDim);
                t.matrix(blk(il, "attn_gate.weight"), c.nEmbd(), valueDim);
                t.random(blk(il, "ssm_conv1d.weight"), -0.3f, 0.3f, c.ssmConv(), convDim);
                t.random(blk(il, "ssm_dt.bias"), -0.5f, 0.5f, c.ssmDtRank());
                t.random(blk(il, "ssm_a"), -1.5f, -0.5f, c.ssmDtRank());
                t.matrix(blk(il, "ssm_ba.weight"), c.nEmbd(), 2L * c.ssmDtRank());
                t.ones(blk(il, "ssm_norm.weight"), headV);
                t.matrix(blk(il, "ssm_out.weight"), valueDim, c.nEmbd());
            }
            t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
            t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
            if (c.expertBiases()) {
                t.random(blk(il, "ffn_gate_exps.bias"), -0.05f, 0.05f, c.nFfExp(), c.nExpert());
                t.random(blk(il, "ffn_up_exps.bias"), -0.05f, 0.05f, c.nFfExp(), c.nExpert());
                t.random(blk(il, "ffn_down_exps.bias"), -0.05f, 0.05f, c.nEmbd(), c.nExpert());
            }
            t.random(blk(il, "ffn_gate_inp_shexp.weight"), -0.1f, 0.1f, c.nEmbd());
            t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfShexp());
            t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfShexp());
            t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfShexp(), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    // ================================================================== lfm3

    public static GgufTestWriter lfm3() {
        return lfm3(SyntheticModels.Config.lfm3(), true);
    }

    /**
     * LLM_ARCH_LFM3: even layers are gated attention (layer 0 sliding-window, layer 2 full with the learned
     * {@code rope_freqs}), odd layers are short convolutions. The output head is tied to the token embedding
     * when {@code tiedOutput} holds (lfm3.cpp:38-41).
     */
    public static GgufTestWriter lfm3(SyntheticModels.Config c, boolean causal) {
        requireAttention(c, "lfm3");
        requirePositive("lfm3 shortconv.l_cache", c.ssmConv());
        requirePositive("lfm3 sliding window", c.slidingWindow());
        if (c.nLayer() != 4) {
            throw new IllegalArgumentException("lfm3 fixture has 4 layers (attn, conv, attn, conv), got " + c.nLayer());
        }
        GgufTestWriter w = base("lfm3", c);
        w.u32Array(w.archKey("attention.head_count"), c.nHead(), 0, c.nHead(), 0);
        w.u32Array(w.archKey("attention.head_count_kv"), c.nHeadKv(), 0, c.nHeadKv(), 0);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("shortconv.l_cache"), c.ssmConv());
        w.u32(w.archKey("attention.sliding_window"), c.slidingWindow());
        w.boolArray(w.archKey("attention.sliding_window_pattern"), true, false, false, false);
        w.f32(w.archKey("rope.freq_base_swa"), c.ropeFreqBase());
        w.bool(w.archKey("attention.causal"), causal);
        bpeVocab(w, c.nVocab(), "lfm2", new String[] {"<|startoftext|>", "<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("rope_freqs.weight", c.nRot() / 2);
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            if (il % 2 == 0) {
                long kvRows = (long) c.nHeadKv() * c.headDim();
                t.matrix(blk(il, "attn_q.weight"), c.nEmbd(), (long) c.nHead() * c.headDim());
                t.matrix(blk(il, "attn_k.weight"), c.nEmbd(), kvRows);
                t.matrix(blk(il, "attn_v.weight"), c.nEmbd(), kvRows);
                t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
                t.matrix(blk(il, "attn_gate.weight"), c.nEmbd(), c.nHead());
                t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
                t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            } else {
                t.matrix(blk(il, "shortconv.conv.weight"), c.ssmConv(), c.nEmbd());
                t.matrix(blk(il, "shortconv.in_proj.weight"), c.nEmbd(), 3L * c.nEmbd());
                t.matrix(blk(il, "shortconv.out_proj.weight"), c.nEmbd(), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    // ================================================================== lfm2

    public static GgufTestWriter lfm2() {
        return lfm2(SyntheticModels.Config.lfm2());
    }

    /**
     * LLM_ARCH_LFM2: even layers are short convolutions, odd layers are attention with per-head Q/K norms (a
     * layer's {@code attention.head_count_kv} entry 0 marks the convolutions, lfm2.cpp:14-16). The output norm is
     * stored as {@code token_embd_norm} and the head ties to the token embedding when {@code tiedOutput} holds.
     */
    public static GgufTestWriter lfm2(SyntheticModels.Config c) {
        requireAttention(c, "lfm2");
        requirePositive("lfm2 shortconv.l_cache", c.ssmConv());
        GgufTestWriter w = base("lfm2", c);
        long[] heads = new long[c.nLayer()];
        long[] headsKv = new long[c.nLayer()];
        for (int il = 0; il < c.nLayer(); il++) {
            boolean attention = il % 2 == 1;
            heads[il] = attention ? c.nHead() : 0;
            headsKv[il] = attention ? c.nHeadKv() : 0;
        }
        w.u32Array(w.archKey("attention.head_count"), heads);
        w.u32Array(w.archKey("attention.head_count_kv"), headsKv);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("shortconv.l_cache"), c.ssmConv());
        bpeVocab(w, c.nVocab(), "lfm2", new String[] {"<|startoftext|>", "<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.ones("token_embd_norm.weight", c.nEmbd());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.gatedFfn(il, c.nFf());
            if (il % 2 == 1) {
                t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
                t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
                t.qkvo(il, (long) c.nHead() * c.headDim());
            } else {
                t.matrix(blk(il, "shortconv.conv.weight"), c.ssmConv(), c.nEmbd());
                t.matrix(blk(il, "shortconv.in_proj.weight"), c.nEmbd(), 3L * c.nEmbd());
                t.matrix(blk(il, "shortconv.out_proj.weight"), c.nEmbd(), c.nEmbd());
            }
        }
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    // ================================================================== plamo2

    public static GgufTestWriter plamo2() {
        return plamo2(SyntheticModels.Config.plamo2());
    }

    /**
     * LLM_ARCH_PLAMO2: alternating PLaMo-2 Mamba layers (no KV heads) and fused-QKV attention layers. The Mamba
     * layer splits {@code d_inner} into {@code ssm.time_step_rank} heads, projects dt through a
     * {@code max(64, n_embd / 16)}-wide weight (plamo2.cpp:41-43) and carries suffixless dt/B/C norms (and no conv
     * bias); the attention layer scales by {@code 1/sqrt(head_dim_v)} and ropes whole heads. Every block holds
     * suffixless post-attention and post-FFW norms and a fused gate-and-up FFN.
     */
    public static GgufTestWriter plamo2(SyntheticModels.Config c) {
        requireAttention(c, "plamo2");
        requirePositive("plamo2 conv kernel", c.ssmConv());
        requirePositive("plamo2 inner size", c.ssmInner());
        requirePositive("plamo2 state size", c.ssmState());
        requirePositive("plamo2 time step rank", c.ssmDtRank());
        if (c.ssmGroups() != 0) {
            throw new IllegalArgumentException("plamo2 needs ssm.group_count 0, got " + c.ssmGroups());
        }
        if (c.ssmInner() % c.ssmDtRank() != 0) {
            throw new IllegalArgumentException("plamo2 needs d_inner (" + c.ssmInner() + ") divisible by heads ("
                    + c.ssmDtRank() + ")");
        }
        if (c.nRot() <= 0 || c.nRot() > c.headDim() || c.nRot() % 2 != 0) {
            throw new IllegalArgumentException("plamo2 rope dimension must be even and in (0, head dim], got "
                    + c.nRot());
        }
        if (c.nLayer() < 2 || c.nLayer() % 2 != 0) {
            throw new IllegalArgumentException("plamo2 alternates mamba and attention layers, needs an even layer"
                    + " count of at least 2, got " + c.nLayer());
        }
        long dtDim = Math.max(64L, c.nEmbd() / 16L);
        int[] heads = new int[c.nLayer()];
        int[] kvHeads = new int[c.nLayer()];
        Arrays.fill(heads, c.nHead());
        for (int il = 0; il < kvHeads.length; il++) {
            kvHeads[il] = il % 2 == 1 ? c.nHeadKv() : 0;
        }
        GgufTestWriter w = base("plamo2", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32Array(w.archKey("attention.head_count"), toLongs(heads));
        w.u32Array(w.archKey("attention.head_count_kv"), toLongs(kvHeads));
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("ssm.conv_kernel"), c.ssmConv());
        w.u32(w.archKey("ssm.inner_size"), c.ssmInner());
        w.u32(w.archKey("ssm.state_size"), c.ssmState());
        w.u32(w.archKey("ssm.time_step_rank"), c.ssmDtRank());
        w.u32(w.archKey("ssm.group_count"), 0);
        bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long qkvWidth = ((long) c.nHead() + 2L * c.nHeadKv()) * c.headDim();
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            if (kvHeads[il] == 0) {
                t.matrix(blk(il, "ssm_in.weight"), c.nEmbd(), 2L * c.ssmInner());
                t.random(blk(il, "ssm_conv1d.weight"), -0.3f, 0.3f, c.ssmConv(), c.ssmInner());
                t.matrix(blk(il, "ssm_x.weight"), c.ssmInner(), dtDim + 2L * c.ssmState());
                t.matrix(blk(il, "ssm_dt.weight"), dtDim, c.ssmDtRank());
                t.random(blk(il, "ssm_dt.bias"), -0.5f, 0.5f, c.ssmDtRank());
                t.random(blk(il, "ssm_a"), -1.5f, -0.5f, c.ssmDtRank());
                t.ones(blk(il, "ssm_d"), c.ssmDtRank());
                t.matrix(blk(il, "ssm_out.weight"), c.ssmInner(), c.nEmbd());
                t.ones(blk(il, "ssm_dt_norm"), dtDim);
                t.ones(blk(il, "ssm_b_norm"), c.ssmState());
                t.ones(blk(il, "ssm_c_norm"), c.ssmState());
            } else {
                t.matrix(blk(il, "attn_qkv.weight"), c.nEmbd(), qkvWidth);
                t.ones(blk(il, "attn_q_norm.weight"), c.headDim(), c.nHead());
                t.ones(blk(il, "attn_k_norm.weight"), c.headDim(), c.nHeadKv());
                t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
            }
            t.ones(blk(il, "post_attention_norm"), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.matrix(blk(il, "ffn_down.weight"), c.nFf(), c.nEmbd());
            t.matrix(blk(il, "ffn_up.weight"), c.nEmbd(), 2L * c.nFf());
            t.ones(blk(il, "post_ffw_norm"), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    // ================================================================== kimi-linear

    public static GgufTestWriter kimiLinear() {
        return kimiLinear(SyntheticModels.Config.kimiLinear());
    }

    /**
     * LLM_ARCH_KIMI_LINEAR: alternating Kimi delta attention layers (no KV heads) and absorbed-MLA attention
     * layers, a dense first block and sigmoid-routed experts after it (kimi-linear.cpp:4-27). The KDA layer
     * depthwise-convolves the Q/K/V streams, gates the decay through the low-rank {@code ssm_f_b(ssm_f_a x)} plus
     * the dt bias and the output through the low-rank {@code ssm_g_b(ssm_g_a x)}, and closes with a gated RMS norm
     * over the KDA head; the attention layer runs the probed MLA without query compression ({@code attn_q}
     * direct, fused {@code attn_kv_b}) and with nope-only heads ({@code rope.dimension_count} 0, the form this
     * engine's kimi-linear MLA runs).
     */
    public static GgufTestWriter kimiLinear(SyntheticModels.Config c) {
        requirePositive("kimi-linear expert count", c.nExpert());
        if (c.nExpertUsed() <= 0 || c.nExpertUsed() > c.nExpert()) {
            throw new IllegalArgumentException("kimi-linear expert_used_count must be in [1, " + c.nExpert()
                    + "], got " + c.nExpertUsed());
        }
        requirePositive("kimi-linear expert feed-forward length", c.nFfExp());
        requirePositive("kimi-linear conv kernel", c.ssmConv());
        requirePositive("kimi-linear KDA head dim", c.kdaHeadDim());
        if (c.nRot() != 0) {
            throw new IllegalArgumentException("kimi-linear MLA is nope-only in this engine, got rope dimension "
                    + c.nRot());
        }
        if (c.nLayer() < 2 || c.nLayer() % 2 != 0) {
            throw new IllegalArgumentException("kimi-linear alternates KDA and MLA layers, needs an even layer count"
                    + " of at least 2, got " + c.nLayer());
        }
        int kdaHead = c.kdaHeadDim();
        long dInner = (long) c.nHead() * kdaHead;
        long nopeV = c.headDim() + MLA_VALUE_HEAD;
        int[] kvHeads = new int[c.nLayer()];
        for (int il = 0; il < kvHeads.length; il++) {
            kvHeads[il] = il % 2 == 1 ? c.nHeadKv() : 0;
        }
        GgufTestWriter w = base("kimi-linear", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32Array(w.archKey("attention.head_count_kv"), toLongs(kvHeads));
        w.u32(w.archKey("attention.key_length"), c.headDim());
        w.u32(w.archKey("attention.value_length"), c.headDim());
        w.u32(w.archKey("attention.key_length_mla"), c.headDim());
        w.u32(w.archKey("attention.value_length_mla"), MLA_VALUE_HEAD);
        w.u32(w.archKey("attention.kv_lora_rank"), c.kvLora());
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("ssm.conv_kernel"), c.ssmConv());
        w.u32(w.archKey("kda.head_dim"), kdaHead);
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("leading_dense_block_count"), 1);
        w.u32(w.archKey("expert_gating_func"), GATING_SIGMOID);
        w.f32(w.archKey("expert_weights_scale"), 1.0f);
        bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            if (kvHeads[il] == 0) {
                for (String stream : new String[] {"q", "k", "v"}) {
                    t.random(blk(il, "ssm_conv1d_" + stream + ".weight"), -0.3f, 0.3f, c.ssmConv(), 1L, dInner, 1L);
                    t.matrix(blk(il, "attn_" + stream + ".weight"), c.nEmbd(), dInner);
                }
                t.matrix(blk(il, "ssm_f_a.weight"), c.nEmbd(), kdaHead);
                t.matrix(blk(il, "ssm_f_b.weight"), kdaHead, dInner);
                t.matrix(blk(il, "ssm_beta.weight"), c.nEmbd(), c.nHead());
                t.random(blk(il, "ssm_a"), -1.5f, -0.5f, 1L, c.nHead());
                t.random(blk(il, "ssm_dt.bias"), -0.5f, 0.5f, dInner);
                t.matrix(blk(il, "ssm_g_a.weight"), c.nEmbd(), kdaHead);
                t.matrix(blk(il, "ssm_g_b.weight"), kdaHead, dInner);
                t.ones(blk(il, "ssm_norm.weight"), kdaHead);
                t.matrix(blk(il, "attn_output.weight"), dInner, c.nEmbd());
            } else {
                t.ones(blk(il, "attn_kv_a_norm.weight"), c.kvLora());
                t.matrix(blk(il, "attn_q.weight"), c.nEmbd(), (long) c.nHead() * c.headDim());
                t.matrix(blk(il, "attn_kv_a_mqa.weight"), c.nEmbd(), c.kvLora());
                t.matrix(blk(il, "attn_kv_b.weight"), c.kvLora(), (long) c.nHead() * nopeV);
                t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * MLA_VALUE_HEAD, c.nEmbd());
            }
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (il == 0) {
                t.gatedFfn(il, c.nFf());
            } else {
                t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
                t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
                t.random(blk(il, "exp_probs_b.bias"), -0.02f, 0.02f, c.nExpert());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    private static long[] toLongs(int[] v) {
        long[] out = new long[v.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = v[i];
        }
        return out;
    }
}
