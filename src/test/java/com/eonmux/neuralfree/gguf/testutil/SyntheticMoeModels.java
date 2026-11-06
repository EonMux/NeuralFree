package com.eonmux.neuralfree.gguf.testutil;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.attentionKeys;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.base;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.bpeVocab;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requireAttention;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * Routed-expert fixtures of {@link SyntheticModels}: glm4moe (one leading dense block, sigmoid-gated experts with a
 * selection bias and one shared expert, NextN / MTP blocks) and qwen3moe (softmax-gated, renormalized experts without
 * a shared expert). Sources: llama.cpp src/models/{glm4-moe,qwen3moe}.cpp
 * {@code load_arch_hparams}/{@code load_arch_tensors} and llama.cpp src/llama-model.cpp
 * {@code create_tensor_gate_up_exps}.
 */
public final class SyntheticMoeModels {

    private SyntheticMoeModels() {
    }

    // ================================================================== glm4moe

    /**
     * LLM_ARCH_GLM4_MOE (glm4-moe.cpp, llama-model.cpp glm4moe tensors): one leading dense block, then routed experts
     * with sigmoid gating, a selection bias and one shared expert over {@code c.nLayer()} trunk layers, followed by
     * {@code nextn} NextN / MTP blocks that carry their own attention and MoE FFN beside {@code nextn.eh_proj},
     * {@code nextn.enorm}, {@code nextn.hnorm} and {@code nextn.shared_head_norm}; the blocks read the trunk's token
     * table and output head.
     */
    public static GgufTestWriter glm4moe(SyntheticModels.Config c, int nextn) {
        requireAttention(c, "glm4moe");
        requirePositive("glm4moe expert count", c.nExpert());
        requirePositive("glm4moe expert feed-forward length", c.nFfExp());
        if (c.nExpertUsed() <= 0 || c.nExpertUsed() > c.nExpert()) {
            throw new IllegalArgumentException("glm4moe expert_used_count must be in [1, " + c.nExpert() + "], got "
                    + c.nExpertUsed());
        }
        if (c.nLayer() < 2 || nextn < 0) {
            throw new IllegalArgumentException("glm4moe needs a dense and an MoE trunk layer and a non-negative NextN"
                    + " block count, got " + c.nLayer() + " layers and " + nextn + " NextN blocks");
        }
        GgufTestWriter w = base("glm4moe", c.toBuilder().nLayer(c.nLayer() + nextn).build());
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.u32(w.archKey("leading_dense_block_count"), 1);
        w.bool(w.archKey("expert_weights_norm"), true);
        w.f32(w.archKey("expert_weights_scale"), 1f);
        w.u32(w.archKey("nextn_predict_layers"), nextn);
        bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|user|>", "<|assistant|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .u32("tokenizer.ggml.padding_token_id", 0).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer() + nextn; il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "post_attention_norm.weight"), c.nEmbd());
            if (il == 0) {
                t.gatedFfn(il, c.nFf());
            } else {
                t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
                t.random(blk(il, "exp_probs_b.bias"), -0.05f, 0.05f, c.nExpert());
                t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
                t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
                t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
                t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
                t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
            }
            if (il >= c.nLayer()) {
                t.matrix(blk(il, "nextn.eh_proj.weight"), 2L * c.nEmbd(), c.nEmbd());
                t.ones(blk(il, "nextn.enorm.weight"), c.nEmbd());
                t.ones(blk(il, "nextn.hnorm.weight"), c.nEmbd());
                t.ones(blk(il, "nextn.shared_head_norm.weight"), c.nEmbd());
            }
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    // ================================================================== qwen3moe

    public static GgufTestWriter qwen3moe() {
        return qwen3moe(SyntheticModels.Config.qwen3moe());
    }

    /**
     * LLM_ARCH_QWEN3MOE: plain GQA attention (with per-head Q/K norms) over a routed MoE FFN with softmax gating
     * and renormalized weights, no shared expert (qwen3moe.cpp:14-56, DenseTensorRequirementsLlama.qwen3moe).
     */
    public static GgufTestWriter qwen3moe(SyntheticModels.Config c) {
        requireAttention(c, "qwen3moe");
        requirePositive("qwen3moe expert count", c.nExpert());
        if (c.nExpertUsed() <= 0 || c.nExpertUsed() > c.nExpert()) {
            throw new IllegalArgumentException("qwen3moe expert_used_count must be in [1, " + c.nExpert() + "], got "
                    + c.nExpertUsed());
        }
        requirePositive("qwen3moe expert feed-forward length", c.nFfExp());
        if (c.nRot() <= 0 || c.nRot() > c.headDim() || c.nRot() % 2 != 0) {
            throw new IllegalArgumentException("qwen3moe rope dimension must be even and in (0, head dim], got "
                    + c.nRot());
        }
        GgufTestWriter w = base("qwen3moe", c);
        attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        bpeVocab(w, c.nVocab(), "qwen2", new String[] {"<|endoftext|>", "<|im_start|>", "<|im_end|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.padding_token_id", 0)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", false);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.qkvo(il, c.nHead() * c.headDim());
            t.ones(blk(il, "attn_q_norm.weight"), c.headDim());
            t.ones(blk(il, "attn_k_norm.weight"), c.headDim());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
            t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
        }
        t.ones("output_norm.weight", c.nEmbd());
        if (!c.tiedOutput()) {
            t.matrix("output.weight", c.nEmbd(), c.nVocab());
        }
        return w;
    }
}
