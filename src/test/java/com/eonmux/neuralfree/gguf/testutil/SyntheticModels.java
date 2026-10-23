package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Tiny, structurally faithful GGUF fixtures for llama, gemma2 (sliding-window attention),
 * qwen3next (hybrid gated-delta-net + attention + MoE), mamba2 and bert.
 *
 * <p>Each builder returns a fresh {@link GgufTestWriter} holding the metadata keys and value types
 * the llama.cpp converters write and every tensor the llama.cpp loader requires, with the loader's
 * shapes ({@code ne[0]} first). Tests may edit the writer before writing it, e.g. to drop a tensor
 * or to replace a key with an array. Sources: llama.cpp src/models/{llama,gemma2,qwen3next,mamba2,bert}.cpp
 * {@code load_arch_hparams}/{@code load_arch_tensors}, llama.cpp src/llama-model.cpp
 * {@code load_hparams}/{@code create_tensor_qkv}/{@code create_tensor_gate_up_exps}, and llama.cpp
 * gguf-py/gguf/gguf_writer.py for value types.
 *
 * <p>Values are deterministic per (config seed, tensor name). Projection matrices use
 * {@link Config#weightType()}: random values when the writer can encode floats into that type,
 * zero blocks otherwise. Norm weights are 1, norm biases 0, and SSM parameters F32 with
 * realistic signs (negative {@code ssm_a}, unit {@code ssm_d}). Every fixture carries a complete
 * tokenizer: SPM with byte fallback (llama, gemma2), byte-level BPE with merges (qwen3next,
 * mamba2) or WordPiece in phantom-space form (bert).
 */
public final class SyntheticModels {

    /** llama.cpp {@code llama_token_type} values. */
    public static final int TOKEN_TYPE_NORMAL = 1;
    public static final int TOKEN_TYPE_UNKNOWN = 2;
    public static final int TOKEN_TYPE_CONTROL = 3;
    public static final int TOKEN_TYPE_BYTE = 6;

    /** llama.cpp {@code LLAMA_POOLING_TYPE_MEAN}, written for the bert fixture. */
    public static final int POOLING_MEAN = 1;
    /** Token type count written for the bert fixture ("sequence A" / "sequence B"). */
    public static final int BERT_TOKEN_TYPES = 2;

    private static final String SPM_SPACE = "▁";
    private static final String BPE_SPACE = "Ġ";

    private SyntheticModels() {
    }

    // ================================================================== llama

    public static GgufTestWriter llama() {
        return llama(Config.llama());
    }

    /** LLM_ARCH_LLAMA: dense GQA attention + SwiGLU FFN, SPM tokenizer. */
    public static GgufTestWriter llama(Config c) {
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
                new int[] {TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL});
        w.u32("tokenizer.ggml.unknown_token_id", 0).u32("tokenizer.ggml.bos_token_id", 1)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", true)
                .bool("tokenizer.ggml.add_eos_token", false);

        Tensors t = new Tensors(w, c);
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

    // ================================================================== gemma2

    public static GgufTestWriter gemma2() {
        return gemma2(Config.gemma2());
    }

    /**
     * LLM_ARCH_GEMMA2: sandwich norms, attention/final logit soft-capping and the default SWA
     * pattern 2 (even layers sliding, odd layers global). The output head is always tied, as in
     * the llama.cpp loader.
     */
    public static GgufTestWriter gemma2(Config c) {
        requireAttention(c, "gemma2");
        requirePositive("gemma2 sliding window", c.slidingWindow());
        GgufTestWriter w = base("gemma2", c);
        attentionKeys(w, c, true);
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.f32(w.archKey("attn_logit_softcapping"), c.attnLogitSoftcap());
        w.f32(w.archKey("final_logit_softcapping"), c.finalLogitSoftcap());
        w.u32(w.archKey("attention.sliding_window"), c.slidingWindow());
        spmVocab(w, c.nVocab(),
                new String[] {"<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"},
                new int[] {TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_UNKNOWN,
                        TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL});
        w.u32("tokenizer.ggml.padding_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .u32("tokenizer.ggml.bos_token_id", 2).u32("tokenizer.ggml.unknown_token_id", 3)
                .bool("tokenizer.ggml.add_bos_token", true).bool("tokenizer.ggml.add_eos_token", false)
                .bool("tokenizer.ggml.add_space_prefix", false);

        Tensors t = new Tensors(w, c);
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

    // ================================================================== qwen3next

    public static GgufTestWriter qwen3next() {
        return qwen3next(Config.qwen3next());
    }

    /**
     * LLM_ARCH_QWEN3NEXT: layer {@code il} is full attention when
     * {@code (il + 1) % full_attention_interval == 0}, else a gated delta net in the
     * {@code attn_qkv}/{@code attn_gate} layout. Attention layers have a gated Q projection
     * ({@code 2 * n_head * head_dim} rows) and per-head Q/K norms. Every layer has a routed MoE with
     * a sigmoid-gated shared expert. GDN dims: head_k = head_v = {@code ssm.state_size},
     * k heads = {@code ssm.group_count}, v heads = {@code ssm.time_step_rank}.
     */
    public static GgufTestWriter qwen3next(Config c) {
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

        Tensors t = new Tensors(w, c);
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

    // ================================================================== mamba2

    public static GgufTestWriter mamba2() {
        return mamba2(Config.mamba2());
    }

    /**
     * LLM_ARCH_MAMBA2: attention-free SSD blocks. {@code conv_dim = d_inner + 2*n_group*d_state},
     * {@code ssm_in} rows {@code = d_inner + conv_dim + n_heads} with n_heads =
     * {@code ssm.time_step_rank}; head dim {@code = d_inner / n_heads}.
     */
    public static GgufTestWriter mamba2(Config c) {
        requireSsm(c, "mamba2");
        requirePositive("mamba2 inner size", c.ssmInner());
        if (c.ssmInner() % c.ssmDtRank() != 0 || c.ssmInner() % c.ssmGroups() != 0
                || c.ssmDtRank() % c.ssmGroups() != 0) {
            throw new IllegalArgumentException("mamba2 needs d_inner (" + c.ssmInner() + ") divisible by heads ("
                    + c.ssmDtRank() + ") and groups (" + c.ssmGroups() + "), and heads divisible by groups");
        }
        long convDim = c.ssmInner() + 2L * c.ssmGroups() * c.ssmState();
        long inProj = c.ssmInner() + convDim + c.ssmDtRank();

        GgufTestWriter w = base("mamba2", c);
        w.u32(w.archKey("feed_forward_length"), 0);
        w.u32(w.archKey("attention.head_count"), 0);
        w.u32(w.archKey("ssm.conv_kernel"), c.ssmConv());
        w.u32(w.archKey("ssm.inner_size"), c.ssmInner());
        w.u32(w.archKey("ssm.state_size"), c.ssmState());
        w.u32(w.archKey("ssm.time_step_rank"), c.ssmDtRank());
        w.u32(w.archKey("ssm.group_count"), c.ssmGroups());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        Tensors t = new Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.matrix(blk(il, "ssm_in.weight"), c.nEmbd(), inProj);
            t.random(blk(il, "ssm_conv1d.weight"), -0.3f, 0.3f, c.ssmConv(), convDim);
            t.random(blk(il, "ssm_conv1d.bias"), -0.02f, 0.02f, convDim);
            t.random(blk(il, "ssm_dt.bias"), -0.5f, 0.5f, c.ssmDtRank());
            t.random(blk(il, "ssm_a"), -1.5f, -0.5f, 1, c.ssmDtRank());
            t.ones(blk(il, "ssm_d"), 1, c.ssmDtRank());
            t.ones(blk(il, "ssm_norm.weight"), c.ssmInner() / c.ssmGroups(), c.ssmGroups());
            t.matrix(blk(il, "ssm_out.weight"), c.ssmInner(), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }

    // ================================================================== bert

    public static GgufTestWriter bert() {
        return bert(Config.bert());
    }

    /**
     * LLM_ARCH_BERT: non-causal encoder with learned positions, token-type embeddings, post-norm
     * LayerNorm blocks with biases and mean pooling. Requires {@code n_head_kv == n_head} and
     * {@code n_head * head_dim == n_embd}; there is no output head.
     */
    public static GgufTestWriter bert(Config c) {
        requireAttention(c, "bert");
        if (c.nHeadKv() != c.nHead() || (long) c.nHead() * c.headDim() != c.nEmbd()) {
            throw new IllegalArgumentException("bert requires n_head_kv == n_head and n_head * head_dim == n_embd");
        }
        GgufTestWriter w = base("bert", c);
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.f32(w.archKey("attention.layer_norm_epsilon"), c.normEps());
        w.bool(w.archKey("attention.causal"), false);
        w.u32(w.archKey("pooling_type"), POOLING_MEAN);
        wpmVocab(w, c.nVocab());
        w.u32("tokenizer.ggml.token_type_count", BERT_TOKEN_TYPES)
                .u32("tokenizer.ggml.padding_token_id", 0).u32("tokenizer.ggml.unknown_token_id", 1)
                .u32("tokenizer.ggml.cls_token_id", 2).u32("tokenizer.ggml.seperator_token_id", 3)
                .u32("tokenizer.ggml.mask_token_id", 4);

        Tensors t = new Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.random("token_types.weight", -0.1f, 0.1f, c.nEmbd(), BERT_TOKEN_TYPES);
        t.random("position_embd.weight", -0.1f, 0.1f, c.nEmbd(), c.nCtxTrain());
        t.layerNorm("token_embd_norm", c.nEmbd());
        for (int il = 0; il < c.nLayer(); il++) {
            for (String p : new String[] {"attn_q", "attn_k", "attn_v", "attn_output"}) {
                t.matrix(blk(il, p + ".weight"), c.nEmbd(), c.nEmbd());
                t.bias(blk(il, p + ".bias"), c.nEmbd());
            }
            t.layerNorm(blk(il, "attn_output_norm"), c.nEmbd());
            t.matrix(blk(il, "ffn_up.weight"), c.nEmbd(), c.nFf());
            t.bias(blk(il, "ffn_up.bias"), c.nFf());
            t.matrix(blk(il, "ffn_down.weight"), c.nFf(), c.nEmbd());
            t.bias(blk(il, "ffn_down.bias"), c.nEmbd());
            t.layerNorm(blk(il, "layer_output_norm"), c.nEmbd());
        }
        return w;
    }

    // ================================================================== shared metadata

    private static GgufTestWriter base(String arch, Config c) {
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        w.str("general.name", "synthetic-" + arch);
        w.u32(w.archKey("vocab_size"), c.nVocab());
        w.u32(w.archKey("context_length"), c.nCtxTrain());
        w.u32(w.archKey("embedding_length"), c.nEmbd());
        w.u32(w.archKey("block_count"), c.nLayer());
        return w;
    }

    /** Head counts and FFN length; key/value lengths when forced or when they differ from n_embd/n_head. */
    private static void attentionKeys(GgufTestWriter w, Config c, boolean alwaysHeadDim) {
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), c.nHeadKv());
        if (alwaysHeadDim || (long) c.nHead() * c.headDim() != c.nEmbd()) {
            w.u32(w.archKey("attention.key_length"), c.headDim());
            w.u32(w.archKey("attention.value_length"), c.headDim());
        }
    }

    private static String blk(int il, String suffix) {
        return "blk." + il + "." + suffix;
    }

    private static void requireAttention(Config c, String arch) {
        if (c.nHead() <= 0 || c.nHeadKv() <= 0 || c.headDim() <= 0 || c.nFf() <= 0) {
            throw new IllegalArgumentException(arch + " needs positive head count, kv head count, head dim and"
                    + " feed-forward length");
        }
        if (c.nHead() % c.nHeadKv() != 0) {
            throw new IllegalArgumentException(arch + ": n_head (" + c.nHead() + ") must be a multiple of n_head_kv ("
                    + c.nHeadKv() + ")");
        }
    }

    private static void requireSsm(Config c, String arch) {
        requirePositive(arch + " ssm conv kernel", c.ssmConv());
        requirePositive(arch + " ssm state size", c.ssmState());
        requirePositive(arch + " ssm group count", c.ssmGroups());
        requirePositive(arch + " ssm time step rank", c.ssmDtRank());
    }

    private static void requirePositive(String what, long v) {
        if (v <= 0) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }

    // ================================================================== tokenizers

    /** SPM vocab: specials, the 256 {@code <0xNN>} byte tokens, then scored pieces. */
    private static void spmVocab(GgufTestWriter w, int nVocab, String[] specials, int[] specialTypes) {
        int fixed = specials.length + 256;
        requireVocab("SPM", nVocab, fixed + 1);
        List<String> tokens = new ArrayList<>(nVocab);
        float[] scores = new float[nVocab];
        int[] types = new int[nVocab];
        for (int i = 0; i < specials.length; i++) {
            tokens.add(specials[i]);
            types[i] = specialTypes[i];
        }
        for (int b = 0; b < 256; b++) {
            types[tokens.size()] = TOKEN_TYPE_BYTE;
            tokens.add(String.format("<0x%02X>", b));
        }
        List<String> pieces = letterPieces(nVocab - fixed, SPM_SPACE, true);
        for (int i = 0; i < pieces.size(); i++) {
            scores[tokens.size()] = -i;
            types[tokens.size()] = TOKEN_TYPE_NORMAL;
            tokens.add(pieces.get(i));
        }
        w.str("tokenizer.ggml.model", "llama").str("tokenizer.ggml.pre", "default")
                .strArray("tokenizer.ggml.tokens", tokens).f32Array("tokenizer.ggml.scores", scores)
                .i32Array("tokenizer.ggml.token_type", types);
    }

    /** Byte-level BPE vocab: specials, the 256 GPT-2 byte symbols, then two-symbol merges. */
    private static void bpeVocab(GgufTestWriter w, int nVocab, String pre, String[] specials) {
        int fixed = specials.length + 256;
        requireVocab("BPE", nVocab, fixed + 1);
        List<String> tokens = new ArrayList<>(nVocab);
        int[] types = new int[nVocab];
        for (String s : specials) {
            types[tokens.size()] = TOKEN_TYPE_CONTROL;
            tokens.add(s);
        }
        for (int cp : gpt2ByteSymbols()) {
            types[tokens.size()] = TOKEN_TYPE_NORMAL;
            tokens.add(new String(Character.toChars(cp)));
        }
        List<String> merges = new ArrayList<>();
        for (String first : bpeFirstSymbols()) {
            for (char second = 'a'; second <= 'z' && tokens.size() < nVocab; second++) {
                types[tokens.size()] = TOKEN_TYPE_NORMAL;
                tokens.add(first + second);
                merges.add(first + " " + second);
            }
        }
        if (tokens.size() != nVocab) {
            throw new IllegalArgumentException("BPE fixture vocab supports at most " + tokens.size() + " tokens, asked "
                    + nVocab);
        }
        w.str("tokenizer.ggml.model", "gpt2").str("tokenizer.ggml.pre", pre)
                .strArray("tokenizer.ggml.tokens", tokens).i32Array("tokenizer.ggml.token_type", types)
                .strArray("tokenizer.ggml.merges", merges);
    }

    /** WordPiece vocab in llama.cpp phantom-space form: words carry U+2581, "##" continuations are bare. */
    private static void wpmVocab(GgufTestWriter w, int nVocab) {
        String[] specials = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
        requireVocab("WPM", nVocab, specials.length + 1);
        List<String> tokens = new ArrayList<>(nVocab);
        int[] types = new int[nVocab];
        for (String s : specials) {
            types[tokens.size()] = TOKEN_TYPE_CONTROL;
            tokens.add(s);
        }
        for (String piece : letterPieces(nVocab - specials.length, SPM_SPACE, false)) {
            types[tokens.size()] = TOKEN_TYPE_NORMAL;
            tokens.add(piece);
        }
        w.str("tokenizer.ggml.model", "bert").str("tokenizer.ggml.pre", "default")
                .strArray("tokenizer.ggml.tokens", tokens).i32Array("tokenizer.ggml.token_type", types);
    }

    private static void requireVocab(String kind, int nVocab, int min) {
        if (nVocab < min) {
            throw new IllegalArgumentException(kind + " fixture vocab needs at least " + min + " tokens, got " + nVocab);
        }
    }

    /**
     * {@code count} distinct pieces over a-z by increasing length, each as "{space}word" and as the
     * bare continuation "word"; the lone space marker comes first when requested.
     */
    private static List<String> letterPieces(int count, String space, boolean includeLoneSpace) {
        Set<String> out = new LinkedHashSet<>();
        if (includeLoneSpace && count > 0) {
            out.add(space);
        }
        for (int len = 1; out.size() < count; len++) {
            int combos = (int) Math.pow(26, len);
            for (int k = 0; k < combos && out.size() < count; k++) {
                StringBuilder word = new StringBuilder();
                for (int p = 0, v = k; p < len; p++, v /= 26) {
                    word.insert(0, (char) ('a' + v % 26));
                }
                out.add(space + word);
                if (out.size() < count) {
                    out.add(word.toString());
                }
            }
        }
        return new ArrayList<>(out);
    }

    /** First symbols of the BPE merges: the GPT-2 space symbol, then a-z. */
    private static List<String> bpeFirstSymbols() {
        List<String> out = new ArrayList<>();
        out.add(BPE_SPACE);
        for (char ch = 'a'; ch <= 'z'; ch++) {
            out.add(String.valueOf(ch));
        }
        return out;
    }

    /** Code points of GPT-2 {@code bytes_to_unicode} in its listing order (llama.cpp gguf-py/gguf/vocab.py). */
    private static int[] gpt2ByteSymbols() {
        List<Integer> bs = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) {
            bs.add(b);
        }
        for (int b = 0xA1; b <= 0xAC; b++) {
            bs.add(b);
        }
        for (int b = 0xAE; b <= 0xFF; b++) {
            bs.add(b);
        }
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        int[] out = new int[cs.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = cs.get(i);
        }
        return out;
    }

    // ================================================================== tensor emission

    /** Adds tensors to one writer with values derived from the config seed and the tensor name. */
    private static final class Tensors {
        private final GgufTestWriter w;
        private final Config c;

        Tensors(GgufTestWriter w, Config c) {
            this.w = w;
            this.c = c;
        }

        /** Projection weight in the configured weight type. */
        void matrix(String name, long... ne) {
            GGMLType type = c.weightType();
            if (GgufTestWriter.canEncodeFloats(type)) {
                w.tensorRandom(name, type, seedFor(name), -0.1f, 0.1f, ne);
            } else {
                w.tensorZeros(name, type, ne);
            }
        }

        void random(String name, float lo, float hi, long... ne) {
            w.tensorRandom(name, GGMLType.F32, seedFor(name), lo, hi, ne);
        }

        void ones(String name, long... ne) {
            w.tensorFilled(name, GGMLType.F32, 1.0f, ne);
        }

        void bias(String name, long n) {
            random(name, -0.02f, 0.02f, n);
        }

        void layerNorm(String prefix, long n) {
            ones(prefix + ".weight", n);
            w.tensorFilled(prefix + ".bias", GGMLType.F32, 0.0f, n);
        }

        /** Separate Q/K/V/O projections; {@code qRows} covers gated-Q layouts. */
        void qkvo(int il, long qRows) {
            long kvRows = (long) c.nHeadKv() * c.headDim();
            matrix(blk(il, "attn_q.weight"), c.nEmbd(), qRows);
            matrix(blk(il, "attn_k.weight"), c.nEmbd(), kvRows);
            matrix(blk(il, "attn_v.weight"), c.nEmbd(), kvRows);
            matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
        }

        void gatedFfn(int il, long nFf) {
            matrix(blk(il, "ffn_gate.weight"), c.nEmbd(), nFf);
            matrix(blk(il, "ffn_up.weight"), c.nEmbd(), nFf);
            matrix(blk(il, "ffn_down.weight"), nFf, c.nEmbd());
        }

        private long seedFor(String name) {
            return c.seed() ^ (name.hashCode() * 0x9E3779B97F4A7C15L);
        }
    }

    // ================================================================== config

    /**
     * Immutable fixture dimensions. Start from an arch default ({@link #llama()}, {@link #gemma2()},
     * {@link #qwen3next()}, {@link #mamba2()}, {@link #bert()}) and adjust with {@link #toBuilder()}.
     * Each arch builder reads only the fields that apply to it; {@code normEps} is the RMS epsilon
     * for RMS-norm archs and the LayerNorm epsilon for bert.
     */
    public static final class Config {
        private final int nVocab;
        private final int nEmbd;
        private final int nLayer;
        private final int nHead;
        private final int nHeadKv;
        private final int headDim;
        private final int nRot;
        private final int nFf;
        private final int nCtxTrain;
        private final float normEps;
        private final float ropeFreqBase;
        private final int slidingWindow;
        private final float attnLogitSoftcap;
        private final float finalLogitSoftcap;
        private final int nExpert;
        private final int nExpertUsed;
        private final int nFfExp;
        private final int nFfShexp;
        private final int fullAttentionInterval;
        private final int ssmConv;
        private final int ssmInner;
        private final int ssmState;
        private final int ssmGroups;
        private final int ssmDtRank;
        private final GGMLType weightType;
        private final long seed;
        private final boolean tiedOutput;

        private Config(Builder b) {
            this.nVocab = b.nVocab;
            this.nEmbd = b.nEmbd;
            this.nLayer = b.nLayer;
            this.nHead = b.nHead;
            this.nHeadKv = b.nHeadKv;
            this.headDim = b.headDim;
            this.nRot = b.nRot;
            this.nFf = b.nFf;
            this.nCtxTrain = b.nCtxTrain;
            this.normEps = b.normEps;
            this.ropeFreqBase = b.ropeFreqBase;
            this.slidingWindow = b.slidingWindow;
            this.attnLogitSoftcap = b.attnLogitSoftcap;
            this.finalLogitSoftcap = b.finalLogitSoftcap;
            this.nExpert = b.nExpert;
            this.nExpertUsed = b.nExpertUsed;
            this.nFfExp = b.nFfExp;
            this.nFfShexp = b.nFfShexp;
            this.fullAttentionInterval = b.fullAttentionInterval;
            this.ssmConv = b.ssmConv;
            this.ssmInner = b.ssmInner;
            this.ssmState = b.ssmState;
            this.ssmGroups = b.ssmGroups;
            this.ssmDtRank = b.ssmDtRank;
            this.weightType = b.weightType;
            this.seed = b.seed;
            this.tiedOutput = b.tiedOutput;
        }

        /** 2 layers, n_embd 64, 4 heads / 2 KV heads of dim 16, FFN 128, vocab 320, F32. */
        public static Config llama() {
            return attentionDefaults().nLayer(2).normEps(1e-5f).build();
        }

        /** 4 layers (SWA, global, SWA, global), window 32, soft-caps 50 / 30. */
        public static Config gemma2() {
            return attentionDefaults().nLayer(4).normEps(1e-6f).slidingWindow(32).attnLogitSoftcap(50.0f)
                    .finalLogitSoftcap(30.0f).tiedOutput(true).build();
        }

        /** 4 layers (GDN, GDN, GDN, attention), 4 experts / 2 used, GDN 2 k-heads / 4 v-heads of dim 16. */
        public static Config qwen3next() {
            return attentionDefaults().nLayer(4).nRot(4).normEps(1e-6f).ropeFreqBase(1.0e7f).nExpert(4)
                    .nExpertUsed(2).nFfExp(32).nFfShexp(32).fullAttentionInterval(4).ssmConv(4).ssmState(16)
                    .ssmGroups(2).ssmDtRank(4).build();
        }

        /** 2 layers, d_inner 128, d_state 16, 1 group, 8 heads of dim 16, conv 4. */
        public static Config mamba2() {
            return new Builder().nLayer(2).nHead(0).nHeadKv(0).headDim(0).nRot(0).nFf(0).normEps(1e-5f)
                    .ssmConv(4).ssmInner(128).ssmState(16).ssmGroups(1).ssmDtRank(8).build();
        }

        /** 2 layers, 4 heads of dim 16 (MHA), FFN 128, 128 positions, LayerNorm eps 1e-12. */
        public static Config bert() {
            return attentionDefaults().nLayer(2).nHeadKv(4).nCtxTrain(128).normEps(1e-12f).build();
        }

        private static Builder attentionDefaults() {
            return new Builder().nHead(4).nHeadKv(2).headDim(16).nRot(16).nFf(128);
        }

        public Builder toBuilder() {
            return new Builder(this);
        }

        public int nVocab() { return nVocab; }
        public int nEmbd() { return nEmbd; }
        public int nLayer() { return nLayer; }
        public int nHead() { return nHead; }
        public int nHeadKv() { return nHeadKv; }
        public int headDim() { return headDim; }
        public int nRot() { return nRot; }
        public int nFf() { return nFf; }
        public int nCtxTrain() { return nCtxTrain; }
        public float normEps() { return normEps; }
        public float ropeFreqBase() { return ropeFreqBase; }
        public int slidingWindow() { return slidingWindow; }
        public float attnLogitSoftcap() { return attnLogitSoftcap; }
        public float finalLogitSoftcap() { return finalLogitSoftcap; }
        public int nExpert() { return nExpert; }
        public int nExpertUsed() { return nExpertUsed; }
        public int nFfExp() { return nFfExp; }
        public int nFfShexp() { return nFfShexp; }
        public int fullAttentionInterval() { return fullAttentionInterval; }
        public int ssmConv() { return ssmConv; }
        public int ssmInner() { return ssmInner; }
        public int ssmState() { return ssmState; }
        public int ssmGroups() { return ssmGroups; }
        public int ssmDtRank() { return ssmDtRank; }
        public GGMLType weightType() { return weightType; }
        public long seed() { return seed; }
        public boolean tiedOutput() { return tiedOutput; }

        /** Mutable builder for {@link Config}; {@link #build()} validates the shared fields. */
        public static final class Builder {
            private int nVocab = 320;
            private int nEmbd = 64;
            private int nLayer = 2;
            private int nHead;
            private int nHeadKv;
            private int headDim;
            private int nRot;
            private int nFf;
            private int nCtxTrain = 256;
            private float normEps = 1e-5f;
            private float ropeFreqBase = 10000.0f;
            private int slidingWindow;
            private float attnLogitSoftcap;
            private float finalLogitSoftcap;
            private int nExpert;
            private int nExpertUsed;
            private int nFfExp;
            private int nFfShexp;
            private int fullAttentionInterval;
            private int ssmConv;
            private int ssmInner;
            private int ssmState;
            private int ssmGroups;
            private int ssmDtRank;
            private GGMLType weightType = GGMLType.F32;
            private long seed = 42L;
            private boolean tiedOutput;

            public Builder() {
            }

            private Builder(Config c) {
                nVocab = c.nVocab;
                nEmbd = c.nEmbd;
                nLayer = c.nLayer;
                nHead = c.nHead;
                nHeadKv = c.nHeadKv;
                headDim = c.headDim;
                nRot = c.nRot;
                nFf = c.nFf;
                nCtxTrain = c.nCtxTrain;
                normEps = c.normEps;
                ropeFreqBase = c.ropeFreqBase;
                slidingWindow = c.slidingWindow;
                attnLogitSoftcap = c.attnLogitSoftcap;
                finalLogitSoftcap = c.finalLogitSoftcap;
                nExpert = c.nExpert;
                nExpertUsed = c.nExpertUsed;
                nFfExp = c.nFfExp;
                nFfShexp = c.nFfShexp;
                fullAttentionInterval = c.fullAttentionInterval;
                ssmConv = c.ssmConv;
                ssmInner = c.ssmInner;
                ssmState = c.ssmState;
                ssmGroups = c.ssmGroups;
                ssmDtRank = c.ssmDtRank;
                weightType = c.weightType;
                seed = c.seed;
                tiedOutput = c.tiedOutput;
            }

            public Builder nVocab(int v) { nVocab = v; return this; }
            public Builder nEmbd(int v) { nEmbd = v; return this; }
            public Builder nLayer(int v) { nLayer = v; return this; }
            public Builder nHead(int v) { nHead = v; return this; }
            public Builder nHeadKv(int v) { nHeadKv = v; return this; }
            public Builder headDim(int v) { headDim = v; return this; }
            public Builder nRot(int v) { nRot = v; return this; }
            public Builder nFf(int v) { nFf = v; return this; }
            public Builder nCtxTrain(int v) { nCtxTrain = v; return this; }
            public Builder normEps(float v) { normEps = v; return this; }
            public Builder ropeFreqBase(float v) { ropeFreqBase = v; return this; }
            public Builder slidingWindow(int v) { slidingWindow = v; return this; }
            public Builder attnLogitSoftcap(float v) { attnLogitSoftcap = v; return this; }
            public Builder finalLogitSoftcap(float v) { finalLogitSoftcap = v; return this; }
            public Builder nExpert(int v) { nExpert = v; return this; }
            public Builder nExpertUsed(int v) { nExpertUsed = v; return this; }
            public Builder nFfExp(int v) { nFfExp = v; return this; }
            public Builder nFfShexp(int v) { nFfShexp = v; return this; }
            public Builder fullAttentionInterval(int v) { fullAttentionInterval = v; return this; }
            public Builder ssmConv(int v) { ssmConv = v; return this; }
            public Builder ssmInner(int v) { ssmInner = v; return this; }
            public Builder ssmState(int v) { ssmState = v; return this; }
            public Builder ssmGroups(int v) { ssmGroups = v; return this; }
            public Builder ssmDtRank(int v) { ssmDtRank = v; return this; }
            public Builder weightType(GGMLType v) { weightType = Objects.requireNonNull(v, "weightType"); return this; }
            public Builder seed(long v) { seed = v; return this; }
            public Builder tiedOutput(boolean v) { tiedOutput = v; return this; }

            public Config build() {
                requirePositive("vocab size", nVocab);
                requirePositive("embedding length", nEmbd);
                requirePositive("layer count", nLayer);
                requirePositive("training context length", nCtxTrain);
                if (nHead < 0 || nHeadKv < 0 || headDim < 0 || nRot < 0 || nFf < 0) {
                    throw new IllegalArgumentException("head counts, head dim, rope dim and FFN length must be >= 0");
                }
                if (!(normEps > 0.0f)) {
                    throw new IllegalArgumentException("norm epsilon must be positive, got " + normEps);
                }
                if (weightType == GGMLType.UNKNOWN || weightType.getBlockSize() <= 0
                        || weightType.getTypeSize() <= 0) {
                    throw new IllegalArgumentException(weightType + " is not a storage type");
                }
                return new Config(this);
            }
        }
    }
}
