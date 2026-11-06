package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import java.util.Arrays;

/**
 * Hparams hooks of the Mamba-family architectures: mamba, mamba2, jamba, falcon-h1, plamo2, granitehybrid,
 * nemotron_h and nemotron_h_moe. Each hook transcribes the llama.cpp {@code load_arch_hparams} body cited on
 * it (src/models/&lt;arch&gt;.cpp) and adds the dimension checks the arch's tensor loader or graph asserts,
 * so a malformed file fails at load with the offending key named.
 */
public final class HybridHooksSsm {

    private HybridHooksSsm() {
    }

    /** mamba.cpp:3-11. */
    public static void mamba(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readSsm(b, kv, false);                            // mamba.cpp:4-7
        b.ssmDtBcRms(kv.bool(GgufKeys.SSM_DT_B_C_RMS, false));        // mamba.cpp:8
        HybridHooks.readRmsEps(b, kv);                                // mamba.cpp:10
        requireExpansionTwo(b, "mamba.cpp:43");
    }

    /** mamba2.cpp:3-10. */
    public static void mamba2(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readSsm(b, kv, true);                             // mamba2.cpp:4-8
        HybridHooks.readRmsEps(b, kv);                                // mamba2.cpp:10
        requireMamba2Heads(b);
    }

    /**
     * jamba.cpp:3-13. The MoE layers are the layers that carry {@code ffn_gate_inp} (jamba.cpp:88-99); the
     * metadata does not mark them. The routed experts use softmax gating without weight normalization
     * (jamba.cpp:170-172).
     */
    public static void jamba(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readSsm(b, kv, false);                            // jamba.cpp:4-7
        HybridHooks.readRmsEps(b, kv);                                // jamba.cpp:9
        HybridHooks.markRecurrentByKvHeads(b);                        // jamba.cpp:11-13
        b.gating(GatingFunc.SOFTMAX);                                 // jamba.cpp:172
        b.expertWeightsNorm(false);                                   // jamba.cpp:170
        requireExpansionTwo(b, "jamba.cpp:32");
    }

    /**
     * falcon-h1.cpp:3-14. Every block holds both a Mamba-2 mixer and an attention mixer, so every layer,
     * NextN blocks included, is flagged recurrent and also keeps its KV cache (llama-model.cpp:3121-3123).
     */
    public static void falconH1(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                // falcon-h1.cpp:5
        HybridHooks.readSsm(b, kv, true);                             // falcon-h1.cpp:8-12
        boolean[] all = new boolean[b.nLayerAll()];                   // falcon-h1.cpp:14
        Arrays.fill(all, true);
        b.isRecurrentArray(all);
        requireMamba2Heads(b);
    }

    /**
     * plamo2.cpp:4-21. {@code attention.key_length} / {@code value_length} (plamo2.cpp:15-16) are optional
     * reads of the keys {@link com.eonmux.neuralfree.gguf.hparams.HparamsLoader} already loaded. The PLaMo2
     * Mamba layer has no B/C groups and splits {@code d_inner} into {@code ssm_dt_rank} heads
     * (plamo2.cpp:278-279).
     */
    public static void plamo2(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                // plamo2.cpp:5
        HybridHooks.readSsm(b, kv, true);                             // plamo2.cpp:8-12
        HybridHooks.markRecurrentByKvHeads(b);                        // plamo2.cpp:18-20
        if (b.ssmNGroup() != 0) {
            throw HybridHooks.invalid(b, "metadata key '" + kv.key(GgufKeys.SSM_GROUP_COUNT)
                    + "' must be 0 for the PLaMo2 Mamba layer, got " + b.ssmNGroup() + " (plamo2.cpp:279)");
        }
        HybridHooks.requireNonZero(kv, GgufKeys.SSM_TIME_STEP_RANK, b.ssmDtRank());
        if (b.ssmDInner() % b.ssmDtRank() != 0) {
            throw HybridHooks.invalid(b, "ssm.inner_size " + b.ssmDInner() + " is not a multiple of ssm.time_step_rank "
                    + b.ssmDtRank() + " (plamo2.cpp:278)");
        }
    }

    /**
     * granite-hybrid.cpp:3-38. {@code rope.scaling.finetuned} defaults to true and switches RoPE on or off
     * for every layer (granite-hybrid.cpp:16-20). The MoE branch routes with softmax and normalized weights
     * regardless of metadata (granite-hybrid.cpp:266-268); residual branches are scaled by
     * {@code residual_scale} and logits divided by {@code logit_scale} (granite-hybrid.cpp:194, 235-236, 290-291).
     */
    public static void graniteHybrid(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                // granite-hybrid.cpp:4
        b.logitScale(kv.f32(GgufKeys.LOGIT_SCALE, b.logitScale()));   // granite-hybrid.cpp:5
        b.residualScale(kv.f32(GgufKeys.RESIDUAL_SCALE, b.residualScale()));      // granite-hybrid.cpp:6
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));   // granite-hybrid.cpp:7
        b.attentionScale(kv.f32(GgufKeys.ATTENTION_SCALE, b.attentionScale()));   // granite-hybrid.cpp:8
        HybridHooks.readSsm(b, kv, true);                             // granite-hybrid.cpp:10-14
        boolean ropeFinetuned = kv.bool(GgufKeys.ROPE_SCALING_FINETUNED, true);   // granite-hybrid.cpp:17-18
        b.ropeFinetuned(ropeFinetuned);                               // granite-hybrid.cpp:19
        boolean[] rope = new boolean[b.nLayerAll()];                  // granite-hybrid.cpp:20
        Arrays.fill(rope, ropeFinetuned);
        b.ropeEnabledArray(rope);
        HybridHooks.markRecurrentByKvHeads(b);                        // granite-hybrid.cpp:23-25
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));   // granite-hybrid.cpp:38
        b.gating(GatingFunc.SOFTMAX);                                 // granite-hybrid.cpp:268
        b.expertWeightsNorm(true);                                    // granite-hybrid.cpp:266
        requireExpansionTwo(b, "granite-hybrid.cpp:54");
        requireMamba2Heads(b);
        HybridHooks.requireEqualHeadSizes(b, "granite-hybrid.cpp:137");
    }

    /**
     * nemotron-h.cpp:5-29 (nemotron_h_moe reuses it, models.h:1685-1689). A trunk layer is a Mamba-2 layer
     * when it has neither KV heads nor an FFN, an attention layer when it has KV heads and no FFN, and an FFN
     * layer otherwise (nemotron-h.cpp:85-147); FFN layers are MoE when experts exist, and a MoE build's NextN blocks
     * carry the MoE sub-layer while a dense build's NextN block is attention only (nemotron-h.cpp:149-190, vLLM
     * {@code NemotronHMTPModel}). Routing uses sigmoid gating regardless of metadata (nemotron-h.cpp:324).
     */
    public static void nemotronH(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readSsm(b, kv, true);                             // nemotron-h.cpp:6-10
        boolean[] recr = new boolean[b.nLayerAll()];                  // nemotron-h.cpp:14-16
        for (int il = 0; il < recr.length; il++) {
            recr[il] = il < b.nLayer() && b.nHeadKv(il) == 0 && b.nFf(il) == 0;
        }
        b.isRecurrentArray(recr);
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));  // nemotron-h.cpp:18
        b.normRmsEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, b.normEps()));   // nemotron-h.cpp:19-21
        HparamsSupport.readExpertFfLength(b, kv, false);              // nemotron-h.cpp:24
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // nemotron-h.cpp:25
        b.nExpertShared(kv.u32(GgufKeys.EXPERT_SHARED_COUNT, b.nExpertShared()));         // nemotron-h.cpp:26
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm()));  // nemotron-h.cpp:27
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // nemotron-h.cpp:28
        b.moeLatentSize(kv.u32(GgufKeys.MOE_LATENT_SIZE, b.moeLatentSize()));             // nemotron-h.cpp:29
        b.gating(GatingFunc.SIGMOID);                                 // nemotron-h.cpp:324
        boolean[] moe = new boolean[b.nLayerAll()];
        if (b.nExpert() > 0) {
            for (int il = 0; il < moe.length; il++) {
                moe[il] = il >= b.nLayer() || (!recr[il] && b.nFf(il) != 0);
            }
        }
        b.moeLayers(moe);
        if (b.nExpert() > 0) {
            // an MoE build folds an expert FFN into every NextN block (nemotron-h.cpp:157-160); a dense build's
            // NextN block is attention only (vLLM NemotronHMTPModel, mtp_hybrid_override_pattern "*")
            for (int il = b.nLayer(); il < b.nLayerAll(); il++) {
                if (b.nFfExp(il) == 0 && b.nExpertUsed(il) == 0) {
                    throw HybridHooks.invalid(b, "NextN layer " + il + " declares neither expert_feed_forward_length"
                            + " nor expert_used_count, cannot determine the expert FFN size");
                }
            }
        }
        requireMamba2Heads(b);
        HybridHooks.requireEqualHeadSizes(b, "nemotron-h.cpp:205");
    }

    /** {@code ssm.inner_size == 2 * embedding_length}, the only expansion factor the Mamba-1 graph supports. */
    private static void requireExpansionTwo(ModelHparamsBuilder b, String where) {
        if ((long) b.ssmDInner() != 2L * b.nEmbd()) {
            throw HybridHooks.invalid(b, "ssm.inner_size " + b.ssmDInner() + " must be twice the embedding length "
                    + b.nEmbd() + "; only an expansion factor of 2 is supported (" + where + ")");
        }
    }

    /**
     * Zamba2 (vLLM {@code Zamba2ForCausalLM}): every block runs a Mamba-2 mixer, and the blocks the
     * {@code shared_blocks.hybrid_layers} array marks also run a shared attention/MLP block first. The hybrid layers
     * take the shared blocks cyclically ({@code hybrid_ordinal % shared_blocks.count}, vLLM
     * {@code Zamba2Model.__init__}), and each use adds its own adapters. RoPE applies to the hybrid layers only when
     * {@code shared_blocks.memory_rope} is set ({@code use_mem_rope}).
     */
    public static void zamba2(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readSsm(b, kv, true);
        HybridHooks.readRmsEps(b, kv);
        requireMamba2Heads(b);
        int n = b.nLayerAll();
        boolean[] hybrid = HybridHooks.readLayerFlags(kv, GgufKeys.SHARED_BLOCKS_HYBRID_LAYERS, n);
        if (hybrid == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(GgufKeys.SHARED_BLOCKS_HYBRID_LAYERS)
                    + "' is missing");
        }
        int memBlocks = kv.u32Required(GgufKeys.SHARED_BLOCKS_COUNT);
        int rank = kv.u32Required(GgufKeys.SHARED_BLOCKS_ADAPTER_RANK);
        boolean attentionAdapter = kv.bool(GgufKeys.SHARED_BLOCKS_ATTENTION_ADAPTER, false);
        boolean memoryRope = kv.bool(GgufKeys.SHARED_BLOCKS_MEMORY_ROPE, false);
        int hybridCount = 0;
        for (boolean h : hybrid) {
            hybridCount += h ? 1 : 0;
        }
        if (hybridCount == 0) {
            throw HybridHooks.invalid(b, "metadata key '" + kv.key(GgufKeys.SHARED_BLOCKS_HYBRID_LAYERS)
                    + "' marks no hybrid layer; a Zamba2 model needs at least one shared transformer block");
        }
        if (memBlocks < 1 || memBlocks > hybridCount) {
            throw HybridHooks.invalid(b, "shared_blocks.count " + memBlocks + " must be in [1, " + hybridCount
                    + "], the number of hybrid layers (vLLM Zamba2Model cycles the blocks over them)");
        }
        if (rank < 1) {
            throw HybridHooks.invalid(b, "shared_blocks.adapter_rank " + rank + " must be positive; the shared"
                    + " MLP always adds its per-use adapter");
        }
        int nHead = b.nHead(0);
        int headK = b.headKFull();
        if (nHead <= 0 || headK != 2L * b.nEmbd() / nHead || (2L * b.nEmbd()) % nHead != 0) {
            throw HybridHooks.invalid(b, "attention.key_length " + headK + " must be 2 * embedding_length / "
                    + "attention.head_count (" + (2L * b.nEmbd()) + " / " + nHead + "), the shared attention's"
                    + " per-head width (vLLM Zamba2Config.attention_head_dim)");
        }
        if (headK % 2 != 0) {
            throw HybridHooks.invalid(b, "attention.key_length " + headK + " must be even: the shared attention"
                    + " scales by (head_dim / 2) ** -0.5 and rotates whole heads");
        }
        if (b.headVFull() != headK) {
            throw HybridHooks.invalid(b, "attention.value_length " + b.headVFull() + " must equal attention.key_length "
                    + headK + "; the shared attention has one width per head");
        }
        for (int il = 0; il < n; il++) {
            if (hybrid[il] && b.nHeadKv(il) != nHead) {
                throw HybridHooks.invalid(b, "layer " + il + " attention.head_count_kv " + b.nHeadKv(il)
                        + " must equal attention.head_count " + nHead + ": the shared attention's q/k/v adapters"
                        + " and projections all have 2 * embedding_length outputs (vLLM Zamba2Attention)");
            }
        }
        if (memoryRope && b.nRotFull() != headK) {
            throw HybridHooks.invalid(b, "rope.dimension_count " + b.nRotFull() + " must be attention.key_length "
                    + headK + "; the shared attention rotates the whole head (use_mem_rope)");
        }
        boolean[] recr = new boolean[n];
        Arrays.fill(recr, true);
        b.isRecurrentArray(recr);
        boolean[] rope = new boolean[n];
        Arrays.fill(rope, memoryRope);
        b.ropeEnabledArray(rope);
        int[] blocks = new int[n];
        Arrays.fill(blocks, -1);
        int ordinal = 0;
        for (int il = 0; il < n; il++) {
            if (hybrid[il]) {
                blocks[il] = ordinal++ % memBlocks;
            }
        }
        b.isHybridLayerArray(hybrid);
        b.zamba2SharedBlockArray(blocks);
        b.zamba2MemBlocks(memBlocks);
        b.zamba2AdapterRank(rank);
        b.zamba2AttentionAdapter(attentionAdapter);
        b.zamba2MemoryRope(memoryRope);
    }

    /**
     * The Mamba-2 layer splits {@code d_inner} into {@code ssm_dt_rank} heads and {@code ssm_n_group} B/C
     * groups (mamba-base.cpp:173-178).
     */
    private static void requireMamba2Heads(ModelHparamsBuilder b) {
        if (b.ssmDtRank() <= 0 || b.ssmNGroup() <= 0) {
            throw HybridHooks.invalid(b, "ssm.time_step_rank " + b.ssmDtRank() + " and ssm.group_count "
                    + b.ssmNGroup() + " must be positive (mamba-base.cpp:177-178)");
        }
        if (b.ssmDInner() % b.ssmDtRank() != 0 || b.ssmDInner() % b.ssmNGroup() != 0) {
            throw HybridHooks.invalid(b, "ssm.inner_size " + b.ssmDInner() + " must be a multiple of ssm.time_step_rank "
                    + b.ssmDtRank() + " and ssm.group_count " + b.ssmNGroup() + " (mamba-base.cpp:177-178)");
        }
    }
}
