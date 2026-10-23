package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import java.util.Arrays;

/**
 * Hparams hooks of the Kimi delta attention (KDA) hybrids: kimi-linear, kimi-k3, bailingmoe3 and glm5-next.
 * A layer without KV heads is a KDA (recurrent) layer with {@code kda.head_dim} heads; the others are MLA
 * attention layers. The {@link KdaGateMode} records the form of the KDA gates:
 * <ul>
 *   <li>kimi-linear, glm5-next: decay gate {@code ssm_f_b(ssm_f_a x) + dt_b}, low-rank output gate
 *       {@code ssm_g_b(ssm_g_a x)} ({@link KdaGateMode#LOWRANK}; kimi-linear.cpp:314-317, 367-368,
 *       glm5-next.cpp:835-837, 874-875);</li>
 *   <li>kimi-k3: the same decay gate, full-rank output gate {@code ssm_g x} ({@link KdaGateMode#FULL_SSM_G};
 *       kimi-k3.cpp:422-424, 463);</li>
 *   <li>bailingmoe3: full-rank decay gate {@code ssm_f_a x + dt_b} squashed by a sigmoid and scaled by the
 *       lower bound, full-rank output gate {@code ssm_g_a x} ({@link KdaGateMode#FULL_SSM_G_A};
 *       bailingmoe3.cpp:271-275, 292).</li>
 * </ul>
 */
public final class HybridHooksKda {

    /** The only hyper-connection stream count the glm5-next graph supports (glm5-next.cpp:53). */
    static final int GLM5_NEXT_HC_MULT = 4;

    private HybridHooksKda() {
    }

    /** kimi-linear.cpp:4-27; routed weights are normalized regardless of metadata (kimi-linear.cpp:520). */
    public static void kimiLinear(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                        // kimi-linear.cpp:5
        HybridHooks.readMla(b, kv);                                           // kimi-linear.cpp:6-8
        readKdaShape(b, kv);                                                  // kimi-linear.cpp:9-10
        HybridHooks.markRecurrentByKvHeads(b);                                // kimi-linear.cpp:17-19
        HparamsSupport.readExpertFfLength(b, kv, true);                       // kimi-linear.cpp:22
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));        // kimi-linear.cpp:23
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead()));       // kimi-linear.cpp:24
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale()));     // kimi-linear.cpp:25
        b.gating(HybridHooks.requiredGating(kv));                             // kimi-linear.cpp:26
        b.expertWeightsNorm(true);                                            // kimi-linear.cpp:520
        b.kdaGateMode(KdaGateMode.LOWRANK);
    }

    /**
     * kimi-k3.cpp:16-46. The MLA cache stores the compressed latent, so the full value head is
     * {@code kv_lora_rank} (kimi-k3.cpp:26-28). Cross-layer attention residuals over
     * {@code attn_res.block_size} layers and the SiTU activation parameters are required.
     */
    public static void kimiK3(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                        // kimi-k3.cpp:17
        HybridHooks.readMla(b, kv);                                           // kimi-k3.cpp:18-19, 21
        b.nLoraQ(kv.u32(GgufKeys.ATTENTION_Q_LORA_RANK, b.nLoraQ()));         // kimi-k3.cpp:20
        readKdaShape(b, kv);                                                  // kimi-k3.cpp:22-23
        b.kdaGateLowerBound(kv.f32(GgufKeys.KDA_GATE_LOWER_BOUND, b.kdaGateLowerBound()));     // kimi-k3.cpp:24
        b.headVFull(b.nLoraKv());                                             // kimi-k3.cpp:28
        HybridHooks.markRecurrentByKvHeads(b);                                // kimi-k3.cpp:31-33
        HparamsSupport.readExpertFfLength(b, kv, true);                       // kimi-k3.cpp:35
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));        // kimi-k3.cpp:36
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead()));       // kimi-k3.cpp:37
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale()));     // kimi-k3.cpp:38
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm()));       // kimi-k3.cpp:39
        b.gating(HybridHooks.requiredGating(kv));                             // kimi-k3.cpp:40
        b.nExpertLatent(kv.u32(GgufKeys.EXPERT_LATENT_LENGTH, b.nExpertLatent()));   // kimi-k3.cpp:41
        b.attnResBlockSize(kv.u32Required(GgufKeys.ATTN_RES_BLOCK_SIZE));     // kimi-k3.cpp:43
        b.situBeta(kv.f32Required(GgufKeys.ACTIVATION_SITU_BETA));            // kimi-k3.cpp:44
        b.situLinearBeta(kv.f32Required(GgufKeys.ACTIVATION_SITU_LINEAR_BETA));   // kimi-k3.cpp:45
        b.kdaGateMode(KdaGateMode.FULL_SSM_G);
    }

    /**
     * bailingmoe3.cpp:6-38. {@code kda.safe_gate} defaults to true and must be true, and the required
     * {@code kda.gate_lower_bound} must be negative (bailingmoe3.cpp:33-34). Without a shared-expert size
     * the shared FFN is {@code n_ff_exp(0) * max(1, n_expert_shared)} wide (bailingmoe3.cpp:29-31). With
     * {@code rope.dimension_sections} present the attention layers use M-RoPE (llama-model.cpp:3550-3555).
     */
    public static void bailingmoe3(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                        // bailingmoe3.cpp:7
        HybridHooks.readMla(b, kv);                                           // bailingmoe3.cpp:8-10
        b.nLoraQ(kv.u32(GgufKeys.ATTENTION_Q_LORA_RANK, b.nLoraQ()));         // bailingmoe3.cpp:11
        readKdaShape(b, kv);                                                  // bailingmoe3.cpp:12-13
        b.kdaSafeGate(kv.bool(GgufKeys.KDA_SAFE_GATE, true));                 // bailingmoe3.cpp:14-16
        b.kdaGateLowerBound(kv.f32Required(GgufKeys.KDA_GATE_LOWER_BOUND));   // bailingmoe3.cpp:17
        HparamsSupport.readRopeSections(b, kv, false);                        // bailingmoe3.cpp:18
        HparamsSupport.readExpertFfLength(b, kv, true);                       // bailingmoe3.cpp:19
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));    // bailingmoe3.cpp:20
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));        // bailingmoe3.cpp:21
        b.nLayerDenseLead(kv.u32Required(GgufKeys.LEADING_DENSE_BLOCK_COUNT));    // bailingmoe3.cpp:22
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale()));     // bailingmoe3.cpp:23
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm()));       // bailingmoe3.cpp:24
        b.gating(HybridHooks.requiredGating(kv));                             // bailingmoe3.cpp:25
        readSwigluClamps(b, kv);                                              // bailingmoe3.cpp:26-27
        if (b.nFfShexp() == 0) {                                              // bailingmoe3.cpp:29-31
            long shexp = (long) b.nFfExp(0) * Math.max(1, b.nExpertShared());
            if (shexp > Integer.MAX_VALUE) {
                throw HybridHooks.invalid(b, "shared expert FFN size " + shexp + " overflows");
            }
            b.nFfShexp((int) shexp);
        }
        if (!b.kdaSafeGate()) {                                               // bailingmoe3.cpp:33
            throw HybridHooks.invalid(b, "metadata key '" + kv.key(GgufKeys.KDA_SAFE_GATE)
                    + "' must be true; only the safe KDA gate is supported");
        }
        if (!(b.kdaGateLowerBound() < 0.0f)) {                                // bailingmoe3.cpp:34
            throw HybridHooks.invalid(b, "metadata key '" + kv.key(GgufKeys.KDA_GATE_LOWER_BOUND)
                    + "' must be negative, got " + b.kdaGateLowerBound());
        }
        HybridHooks.markRecurrentByKvHeads(b);                                // bailingmoe3.cpp:36-38
        b.kdaGateMode(KdaGateMode.FULL_SSM_G_A);
    }

    /**
     * glm5-next.cpp:7-54. Recurrence is marked over every block, NextN included (glm5-next.cpp:21-23).
     * Gating defaults to sigmoid when absent or NONE (glm5-next.cpp:30-33). The k-pool indexer requires
     * {@code kpool > 1} dividing {@code top_k} (glm5-next.cpp:45); every layer runs the full indexer unless
     * {@code attention.indexer.types} (one entry per trunk layer) says otherwise (glm5-next.cpp:46-47).
     */
    public static void glm5Next(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                        // glm5-next.cpp:8
        b.normEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_EPS, b.normEps()));     // glm5-next.cpp:9
        HybridHooks.readMla(b, kv);                                           // glm5-next.cpp:10-11, 13
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));             // glm5-next.cpp:12
        readKdaShape(b, kv);                                                  // glm5-next.cpp:14-15
        b.kdaGateLowerBound(kv.f32(GgufKeys.KDA_GATE_LOWER_BOUND, b.kdaGateLowerBound()));     // glm5-next.cpp:16
        b.headVFull(b.nLoraKv());                                             // glm5-next.cpp:19
        for (int il = 0; il < b.nLayerAll(); il++) {                          // glm5-next.cpp:21-23
            b.isRecurrent(il, b.nHeadKv(il) == 0);
        }
        HparamsSupport.readExpertFfLength(b, kv, true);                       // glm5-next.cpp:25
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));        // glm5-next.cpp:26
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead()));       // glm5-next.cpp:27
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale()));     // glm5-next.cpp:28
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm()));       // glm5-next.cpp:29
        b.gating(HparamsSupport.readGating(kv, GatingFunc.SIGMOID));          // glm5-next.cpp:30-33
        b.swigluClampExpArray(kv.keyOrArrF32(GgufKeys.SWIGLU_CLAMP_EXP, b.nLayerAll(), 0f));   // glm5-next.cpp:34
        b.swigluClampShexpArray(kv.has(GgufKeys.SWIGLU_CLAMP_SHEXP)          // glm5-next.cpp:35-37
                ? kv.keyOrArrF32(GgufKeys.SWIGLU_CLAMP_SHEXP, b.nLayerAll(), 0f)
                : b.swigluClampExpArray());
        readKpoolIndexer(b, kv);                                              // glm5-next.cpp:40-47
        b.dsv4HcMult(kv.u32Required(GgufKeys.HYPER_CONNECTION_COUNT));       // glm5-next.cpp:50
        b.dsv4HcSinkhornIters(kv.u32Required(GgufKeys.HYPER_CONNECTION_SINKHORN_ITERATIONS));   // glm5-next.cpp:51
        b.dsv4HcEps(kv.f32Required(GgufKeys.HYPER_CONNECTION_EPSILON));      // glm5-next.cpp:52
        if (b.dsv4HcMult() != GLM5_NEXT_HC_MULT) {                            // glm5-next.cpp:53
            throw HybridHooks.invalid(b, "metadata key '" + kv.key(GgufKeys.HYPER_CONNECTION_COUNT) + "' must be "
                    + GLM5_NEXT_HC_MULT + ", got " + b.dsv4HcMult());
        }
        requireGlm5NextAttention(b, kv);
        b.kdaGateMode(KdaGateMode.LOWRANK);
    }

    /**
     * The glm5-next MLA attention is nope-only, {@code n_rot == 0} (glm5-next.cpp:1059), and a trunk attention
     * layer without the full indexer reuses the selection of the previous full-indexer layer, so the first
     * trunk attention layer must run the full indexer (glm5-next.cpp:1081-1087).
     */
    private static void requireGlm5NextAttention(ModelHparamsBuilder b, GgufKv kv) {
        boolean hasAttention = false;
        boolean fullSeen = false;
        for (int il = 0; il < b.nLayerAll(); il++) {
            if (b.isRecurrent(il)) {
                continue;
            }
            hasAttention = true;
            if (il >= b.nLayer() || b.isIndexerFull(il)) {
                fullSeen = true;
            } else if (!fullSeen) {
                throw HybridHooks.invalid(b, "attention layer " + il + " shares the indexer selection of an earlier"
                        + " full-indexer layer, but none precedes it (metadata key '"
                        + kv.key(GgufKeys.ATTENTION_INDEXER_TYPES) + "')");
            }
        }
        if (hasAttention && b.nRotFull() != 0) {
            throw HybridHooks.invalid(b, "the MLA attention is nope-only: metadata key '"
                    + kv.key(GgufKeys.ROPE_DIMENSION_COUNT) + "' must be 0, got " + b.nRotFull());
        }
    }

    /** Required {@code ssm.conv_kernel} and {@code kda.head_dim}. */
    private static void readKdaShape(ModelHparamsBuilder b, GgufKv kv) {
        b.ssmDConv(kv.u32Required(GgufKeys.SSM_CONV_KERNEL));
        b.kdaHeadDim(kv.u32Required(GgufKeys.KDA_HEAD_DIM));
    }

    /** Optional per-layer SwiGLU clamps {@code swiglu_clamp_exp} / {@code swiglu_clamp_shexp}. */
    private static void readSwigluClamps(ModelHparamsBuilder b, GgufKv kv) {
        b.swigluClampExpArray(kv.keyOrArrF32(GgufKeys.SWIGLU_CLAMP_EXP, b.nLayerAll(), 0f));
        b.swigluClampShexpArray(kv.keyOrArrF32(GgufKeys.SWIGLU_CLAMP_SHEXP, b.nLayerAll(), 0f));
    }

    /** glm5-next.cpp:40-47. */
    private static void readKpoolIndexer(ModelHparamsBuilder b, GgufKv kv) {
        b.indexerNHead(kv.u32Required(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT));
        b.indexerHeadSize(kv.u32Required(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH));
        b.indexerTopK(kv.u32Required(GgufKeys.ATTENTION_INDEXER_TOP_K));
        b.indexerKpool(kv.u32Required(GgufKeys.ATTENTION_INDEXER_KPOOL));
        b.indexerKpoolSelectTail(kv.bool(GgufKeys.ATTENTION_INDEXER_KPOOL_SELECT_TAIL, b.indexerKpoolSelectTail()));
        if (b.indexerKpool() <= 1 || b.indexerTopK() % b.indexerKpool() != 0) {
            throw HybridHooks.invalid(b, "the k-pool indexer needs attention.indexer.kpool above 1 dividing"
                    + " attention.indexer.top_k, got " + b.indexerKpool() + " and " + b.indexerTopK());
        }
        boolean[] full = new boolean[b.nLayerAll()];
        Arrays.fill(full, true);
        boolean[] types = HybridHooks.readLayerFlags(kv, GgufKeys.ATTENTION_INDEXER_TYPES, b.nLayer());
        if (types != null) {
            System.arraycopy(types, 0, full, 0, types.length);
        }
        b.isIndexerFullArray(full);
    }
}
