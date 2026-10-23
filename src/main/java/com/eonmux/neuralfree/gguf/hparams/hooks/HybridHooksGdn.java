package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hparams hooks of the Gated DeltaNet hybrids: qwen3next, qwen35, qwen35moe and qwen4exp. Linear-attention
 * (recurrent) layers alternate with gated full-attention layers per {@code attention.recurrent_layers} or
 * {@code full_attention_interval} (default 4). The GDN mixer has {@code ssm.group_count} K heads of size
 * {@code ssm.state_size} and {@code ssm.time_step_rank} V heads of size {@code ssm.inner_size / V heads}.
 *
 * <p>When V heads outnumber K heads the K heads are broadcast: qwen3next repeats each K head over a
 * contiguous group of V heads ({@link GdnHeadMap#GROUPED}, qwen3next.cpp:513-530); qwen35, qwen35moe and
 * qwen4exp tile the K heads ({@link GdnHeadMap#TILED}, {@code ggml_repeat_4d}: qwen35.cpp:450-453,
 * qwen35moe.cpp:460-463, qwen4exp.cpp:1126-1129).
 *
 * <p>Whether a qwen3next recurrent layer uses the fused {@code ssm_ba} / {@code attn_qkv} projections or
 * the legacy {@code ssm_in} projection is a tensor fact, resolved into {@code RecurrentSpec.legacySsmIn()} and
 * {@code RecurrentSpec.fusedBetaAlpha()} by {@code RecurrentResolver}.
 */
public final class HybridHooksGdn {

    /** Default {@code full_attention_interval} (qwen3next.cpp:18, qwen35.cpp:18, qwen4exp.cpp:150). */
    static final int DEFAULT_FULL_ATTENTION_INTERVAL = 4;

    private HybridHooksGdn() {
    }

    /**
     * qwen3next.cpp:4-23. The model must have experts (qwen3next.cpp:34-36); every block, NextN included,
     * routes with softmax gating and normalized weights (qwen3next.cpp:568-570, 761-763).
     */
    public static void qwen3next(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, false);                      // qwen3next.cpp:5
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));   // qwen3next.cpp:6
        HybridHooks.readRmsEps(b, kv);                                        // qwen3next.cpp:7
        HybridHooks.readSsm(b, kv, true);                                     // qwen3next.cpp:10-14
        HybridHooks.readRecurrentLayers(b, kv, DEFAULT_FULL_ATTENTION_INTERVAL);   // qwen3next.cpp:17-23
        if (b.nExpert() == 0) {                                               // qwen3next.cpp:34-36
            throw HybridHooks.invalid(b, "model cannot have zero experts");
        }
        routedSoftmaxNormalized(b);
        b.gdnHeadMap(GdnHeadMap.GROUPED);
        requireGdnHeads(b);
        HybridHooks.requireEqualHeadSizes(b, "qwen3next.cpp:240");
    }

    /** qwen35.cpp:4-23 (dense FFN, IMROPE sections required). */
    public static void qwen35(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                        // qwen35.cpp:5
        HparamsSupport.readRopeSections(b, kv, true);                         // qwen35.cpp:6
        HybridHooks.readSsm(b, kv, true);                                     // qwen35.cpp:9-13
        HybridHooks.readRecurrentLayers(b, kv, DEFAULT_FULL_ATTENTION_INTERVAL);   // qwen35.cpp:17-23
        b.gdnHeadMap(GdnHeadMap.TILED);
        requireGdnHeads(b);
        HybridHooks.requireEqualHeadSizes(b, "qwen35.cpp:141");
    }

    /** qwen35moe.cpp:4-26; MoE routing as qwen3next (qwen35moe.cpp:500-502, 681-683). */
    public static void qwen35moe(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, false);                      // qwen35moe.cpp:5
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));   // qwen35moe.cpp:6
        HybridHooks.readRmsEps(b, kv);                                        // qwen35moe.cpp:7
        HparamsSupport.readRopeSections(b, kv, true);                         // qwen35moe.cpp:9
        HybridHooks.readSsm(b, kv, true);                                     // qwen35moe.cpp:12-16
        HybridHooks.readRecurrentLayers(b, kv, DEFAULT_FULL_ATTENTION_INTERVAL);   // qwen35moe.cpp:20-26
        routedSoftmaxNormalized(b);
        b.gdnHeadMap(GdnHeadMap.TILED);
        requireGdnHeads(b);
        HybridHooks.requireEqualHeadSizes(b, "qwen35moe.cpp:160");
    }

    /**
     * qwen4exp.cpp:26-162: the qwen35moe trunk plus hyper-connections, the QSA indexer and the PLE n-gram
     * embedding module ({@link HybridHooksQwen4Exp}). Routing as qwen3next (qwen4exp.cpp:1164-1166).
     */
    public static void qwen4exp(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, false);                      // qwen4exp.cpp:27
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));   // qwen4exp.cpp:28
        HybridHooks.readRmsEps(b, kv);                                        // qwen4exp.cpp:29
        HparamsSupport.readRopeSections(b, kv, true);                         // qwen4exp.cpp:31
        HybridHooks.readSsm(b, kv, true);                                     // qwen4exp.cpp:33-37
        HybridHooks.requireNonZero(kv, GgufKeys.SSM_CONV_KERNEL, b.ssmDConv());        // qwen4exp.cpp:38
        HybridHooks.requireNonZero(kv, GgufKeys.SSM_INNER_SIZE, b.ssmDInner());        // qwen4exp.cpp:39
        HybridHooks.requireNonZero(kv, GgufKeys.SSM_STATE_SIZE, b.ssmDState());        // qwen4exp.cpp:40
        HybridHooks.requireNonZero(kv, GgufKeys.SSM_TIME_STEP_RANK, b.ssmDtRank());    // qwen4exp.cpp:41
        HybridHooks.requireNonZero(kv, GgufKeys.SSM_GROUP_COUNT, b.ssmNGroup());       // qwen4exp.cpp:42
        HybridHooksQwen4Exp.readHyperConnections(b, kv);                      // qwen4exp.cpp:45-54
        HybridHooksQwen4Exp.readIndexer(b, kv);                               // qwen4exp.cpp:56-83
        HybridHooksQwen4Exp.readPle(b, kv);                                   // qwen4exp.cpp:86-146
        HybridHooks.readRecurrentLayers(b, kv, DEFAULT_FULL_ATTENTION_INTERVAL);   // qwen4exp.cpp:149-157
        HybridHooksQwen4Exp.requirePleRecurrent(b);                           // qwen4exp.cpp:160-164
        routedSoftmaxNormalized(b);
        b.gdnHeadMap(GdnHeadMap.TILED);
        requireGdnHeads(b);
        HybridHooks.requireEqualHeadSizes(b, "qwen4exp.cpp:403");
    }

    /** The Qwen MoE blocks route with softmax over all experts and renormalize the top-k weights. */
    private static void routedSoftmaxNormalized(ModelHparamsBuilder b) {
        b.gating(GatingFunc.SOFTMAX);
        b.expertWeightsNorm(true);
    }

    /**
     * The GDN mixer needs positive K/V head counts, V heads that split {@code ssm.inner_size} evenly
     * ({@code head_v_dim = d_inner / num_v_heads}, qwen35.cpp:358) and a whole number of V heads per K head
     * (qwen3next.cpp:513, qwen35.cpp:451, qwen35moe.cpp:461, qwen4exp.cpp:1127).
     */
    private static void requireGdnHeads(ModelHparamsBuilder b) {
        int nK = b.ssmNGroup();
        int nV = b.ssmDtRank();
        if (nK <= 0 || nV <= 0) {
            throw HybridHooks.invalid(b, "ssm.group_count " + nK + " (K heads) and ssm.time_step_rank " + nV
                    + " (V heads) must be positive");
        }
        if (b.ssmDInner() % nV != 0) {
            throw HybridHooks.invalid(b, "ssm.inner_size " + b.ssmDInner() + " is not a multiple of the " + nV
                    + " V heads");
        }
        if (nV % nK != 0) {
            throw HybridHooks.invalid(b, nV + " V heads are not a multiple of the " + nK + " K heads");
        }
    }
}
