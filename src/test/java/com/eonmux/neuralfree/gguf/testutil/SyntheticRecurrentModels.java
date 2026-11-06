package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireSsm;

/**
 * Attention-free SSM fixtures of {@link SyntheticModels}: mamba (selective scan, Mamba-1 layout) and mamba2 (SSD
 * blocks). Sources: llama.cpp src/models/{mamba,mamba2}.cpp {@code load_arch_hparams}/{@code load_arch_tensors}.
 */
public final class SyntheticRecurrentModels {

    private SyntheticRecurrentModels() {
    }

    // ================================================================== mamba2

    public static GgufTestWriter mamba2() {
        return mamba2(SyntheticModels.Config.mamba2());
    }

    /**
     * LLM_ARCH_MAMBA2: attention-free SSD blocks. {@code conv_dim = d_inner + 2*n_group*d_state},
     * {@code ssm_in} rows {@code = d_inner + conv_dim + n_heads} with n_heads =
     * {@code ssm.time_step_rank}; head dim {@code = d_inner / n_heads}.
     */
    public static GgufTestWriter mamba2(SyntheticModels.Config c) {
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

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
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

    // ================================================================== mamba1

    public static GgufTestWriter mamba() {
        return mamba(SyntheticModels.Config.mamba());
    }

    /**
     * LLM_ARCH_MAMBA: attention-free selective-scan blocks in the Mamba-1 layout: {@code ssm.inner_size}
     * {@code = 2 * n_embd} (mamba.cpp:43), no B/C groups ({@code conv_dim = d_inner}), the separate
     * {@code ssm_x}/{@code ssm_dt} projections of the dt, B and C inputs and a conv bias.
     */
    public static GgufTestWriter mamba(SyntheticModels.Config c) {
        requirePositive("mamba conv kernel", c.ssmConv());
        requirePositive("mamba inner size", c.ssmInner());
        requirePositive("mamba state size", c.ssmState());
        requirePositive("mamba time step rank", c.ssmDtRank());
        if (c.ssmInner() != 2L * c.nEmbd()) {
            throw new IllegalArgumentException("mamba needs ssm.inner_size == 2 * n_embd, got " + c.ssmInner()
                    + " for n_embd " + c.nEmbd());
        }
        GgufTestWriter w = base("mamba", c);
        w.u32(w.archKey("feed_forward_length"), 0);
        w.u32(w.archKey("attention.head_count"), 0);
        w.u32(w.archKey("ssm.conv_kernel"), c.ssmConv());
        w.u32(w.archKey("ssm.inner_size"), c.ssmInner());
        w.u32(w.archKey("ssm.state_size"), c.ssmState());
        w.u32(w.archKey("ssm.time_step_rank"), c.ssmDtRank());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.matrix(blk(il, "ssm_in.weight"), c.nEmbd(), 2L * c.ssmInner());
            t.random(blk(il, "ssm_conv1d.weight"), -0.3f, 0.3f, c.ssmConv(), c.ssmInner());
            t.random(blk(il, "ssm_conv1d.bias"), -0.02f, 0.02f, c.ssmInner());
            t.matrix(blk(il, "ssm_x.weight"), c.ssmInner(), c.ssmDtRank() + 2L * c.ssmState());
            t.matrix(blk(il, "ssm_dt.weight"), c.ssmDtRank(), c.ssmInner());
            t.random(blk(il, "ssm_dt.bias"), -0.5f, 0.5f, c.ssmInner());
            t.random(blk(il, "ssm_a"), -1.5f, -0.5f, c.ssmState(), c.ssmInner());
            t.ones(blk(il, "ssm_d"), c.ssmInner());
            t.matrix(blk(il, "ssm_out.weight"), c.ssmInner(), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }
}
