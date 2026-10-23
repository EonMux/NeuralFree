package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hparams hooks of the LFM2 short-convolution hybrids: lfm2 and lfm2moe. A layer without KV heads is a
 * short-convolution layer whose rolling state keeps {@code shortconv.l_cache - 1} rows
 * ({@code n_embd_r}); the others are attention layers with per-head Q/K norms.
 */
public final class HybridHooksLfm {

    /** Decision-model output types: choice, score, null (lfm2.cpp:7-8). */
    private static final int N_DECISION_TYPES = 3;

    private HybridHooksLfm() {
    }

    /**
     * lfm2.cpp:10-44. With {@code decision.block_count > 0} the model is a non-causal decision encoder: the
     * last {@code n_layer_decision} trunk blocks form the decision head, the output has three channels, the
     * memory is disabled (llama-model.cpp:2782-2784) and the short convolutions use a centred 3-tap window
     * (lfm2.cpp:363-369). An {@code attention.sliding_window > 0} makes the attention layers
     * {@link SwaType#STANDARD} sliding-window layers.
     */
    public static void lfm2(ModelHparamsBuilder b, GgufKv kv) {
        b.shortconvLCache(kv.u32Required(GgufKeys.SHORTCONV_L_CACHE));           // lfm2.cpp:11
        HybridHooks.readRmsEps(b, kv);                                         // lfm2.cpp:12
        HybridHooks.markRecurrentByKvHeads(b);                                 // lfm2.cpp:14-16
        b.nLayerDenseLead(b.nLayer());                                         // lfm2.cpp:18
        b.nLayerDecision(kv.u32(GgufKeys.DECISION_BLOCK_COUNT, b.nLayerDecision()));   // lfm2.cpp:29
        if (b.nLayerDecision() > 0) {                                          // lfm2.cpp:30-36
            if (b.nLayerDecision() >= b.nLayer() || b.causal()) {
                throw HybridHooks.invalid(b, "invalid decision head: decision.block_count " + b.nLayerDecision()
                        + " must be below the block count " + b.nLayer() + " and attention must be non-causal");
            }
            b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));
            b.nEmbdOutImpl(N_DECISION_TYPES);
            if (b.shortconvLCache() != 3) {
                throw HybridHooks.invalid(b, "the decision model's centred short convolution needs shortconv.l_cache 3,"
                        + " got " + b.shortconvLCache() + " (lfm2.cpp:369)");
            }
            int tokenTypes = kv.u32(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, 0);
            if (tokenTypes != N_DECISION_TYPES) {                              // lfm2.cpp:75-77
                throw HybridHooks.invalid(b, "decision model must have one token type per question type: metadata key '"
                        + kv.key(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT) + "' must be " + N_DECISION_TYPES + ", got "
                        + tokenTypes);
            }
        }
        if (HparamsSupport.readSlidingWindow(b, kv, false) && b.nSwa() > 0) {   // lfm2.cpp:38-43
            b.swaType(SwaType.STANDARD);
            for (int il = 0; il < b.nLayer(); il++) {
                b.isSwa(il, !b.isRecurrent(il));
            }
        }
        requireLCache(b, kv);
        requireMatchingKvHeads(b);
    }

    /**
     * lfm2moe.cpp:5-14. Unlike lfm2 it reads no sliding window. The routed experts normalize their weights
     * regardless of metadata (lfm-base.cpp:110).
     */
    public static void lfm2moe(ModelHparamsBuilder b, GgufKv kv) {
        b.shortconvLCache(kv.u32Required(GgufKeys.SHORTCONV_L_CACHE));           // lfm2moe.cpp:6
        HybridHooks.readRmsEps(b, kv);                                         // lfm2moe.cpp:7
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead()));   // lfm2moe.cpp:8
        HparamsSupport.readExpertFfLength(b, kv, true);                        // lfm2moe.cpp:9
        b.gating(HybridHooks.requiredGating(kv));                              // lfm2moe.cpp:10
        HybridHooks.markRecurrentByKvHeads(b);                                 // lfm2moe.cpp:12-14
        b.expertWeightsNorm(true);                                             // lfm-base.cpp:110
        if (b.nLayerDenseLead() < b.nLayer() && b.nExpert() == 0) {            // lfm2moe.cpp:41-43
            throw HybridHooks.invalid(b, "blocks from leading_dense_block_count " + b.nLayerDenseLead()
                    + " on route experts, but expert_count is 0");
        }
        requireLCache(b, kv);
        requireMatchingKvHeads(b);
    }

    /**
     * The short-convolution state keeps {@code l_cache - 1} rows, so the window must exceed 1
     * (lfm-base.cpp:21-22).
     */
    private static void requireLCache(ModelHparamsBuilder b, GgufKv kv) {
        if (b.shortconvLCache() <= 1) {
            throw HybridHooks.invalid(b, "metadata key '" + kv.key(GgufKeys.SHORTCONV_L_CACHE)
                    + "' must be greater than 1, got " + b.shortconvLCache() + " (lfm-base.cpp:21)");
        }
    }

    /**
     * Attention layers before the decision head need equal K and V widths per layer,
     * {@code n_embd_k_gqa(il) == n_embd_v_gqa(il)} (lfm2.cpp:159, lfm2moe.cpp:61); sliding-window layers use
     * the SWA head sizes.
     */
    private static void requireMatchingKvHeads(ModelHparamsBuilder b) {
        for (int il = 0; il < b.nLayer() - b.nLayerDecision(); il++) {
            if (b.isRecurrent(il)) {
                continue;
            }
            boolean swa = b.isSwa(il);
            long k = (long) (swa ? b.headKSwa() : b.headKFull()) * b.nHeadKv(il);
            long v = (long) (swa ? b.headVSwa() : b.headVFull()) * b.nHeadKv(il);
            if (k != v) {
                throw HybridHooks.invalid(b, "attention layer " + il + " has K width " + k + " and V width " + v
                        + "; they must be equal (lfm2.cpp:159)");
            }
        }
    }
}
