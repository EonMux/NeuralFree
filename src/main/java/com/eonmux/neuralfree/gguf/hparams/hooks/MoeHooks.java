package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.gating;
import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.requiredGating;
import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.rmsEps;

/**
 * {@link com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook} implementations for the MoE architectures with
 * plain multi-head attention, referenced from {@link com.eonmux.neuralfree.gguf.arch.ArchTableMoeClassic}.
 * The MLA/DSA hooks are in {@link MoeHooksMla}, the sliding-window MoE hooks in {@link MoeHooksSwa}.
 *
 * <p>Each hook transcribes {@code llama_model_<arch>::load_arch_hparams} in llama.cpp src/models/&lt;file&gt;.cpp
 * (cited as {@code models/<file>.cpp:line}). Where the llama.cpp graph passes a constant instead of the loaded
 * value to {@code build_moe_ffn} (router gating, weight normalization, weight scale), the hook stores that
 * constant so {@link ModelHparams#gating()}, {@link ModelHparams#expertWeightsNorm()} and
 * {@link ModelHparams#expertWeightsScale()} hold the values the graph uses. An absent or {@code NONE}
 * gating key with no architecture default aborts the llama.cpp graph (llama-graph.cpp:2256), so such keys
 * are required here.
 */
public final class MoeHooks {

    private MoeHooks() {
    }

    /** models/llama.cpp:3-31; graph models/llama.cpp:203-212 routes with softmax and normalized weights. */
    public static void llama(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :7
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :210-212
    }

    /**
     * models/llama4.cpp:3-42. A present zero {@code sliding_window} disables SWA and RoPE skipping; otherwise
     * chunked attention with an 8192 window is forced regardless of the key, with the temperature tuning
     * of the 3-chunked/1-full pattern. MoE layers follow the tensor loader (models/llama4.cpp:62), RoPE layers
     * the graph rule {@code n_no_rope_layer_step > 0 && (il + 1) % n_no_rope_layer_step != 0}
     * (models/llama4.cpp:143-144). The graph routes with sigmoid and unnormalized weights (models/llama4.cpp:226-228).
     */
    public static void llama4(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        b.nMoeLayerStep(kv.u32Required(GgufKeys.INTERLEAVE_MOE_LAYER_STEP));              // :6
        boolean foundSwa = HparamsSupport.readSlidingWindow(b, kv, false);                // :8
        if (foundSwa && b.nSwa() == 0) {                                                  // :9
            b.swaType(SwaType.NONE);                                                      // :10
            b.noRopeLayerStep(b.nLayer());                                                // :11
        } else {
            b.swaType(SwaType.CHUNKED);                                                   // :13
            b.nSwa(8192);                                                                 // :14
            b.attnTempFloorScale(8192);                                                   // :15
            b.attnTempScale(0.1f);                                                        // :16
            b.attnTempOffset(1.0f);                                                       // :17
            HparamsSupport.loadSwaPattern(b, kv, 4, false);                               // :19
            MoeHooksSupport.swaRopeFromFull(b, kv);                                       // :21-23
        }
        b.useKqNorm(b.nExpert() != 128);                                                  // :25-41 (17B_128E has no QK norm)
        MoeHooksSupport.requireExperts(b, "llama4");                                      // tensors :47-49
        final int moeStep = b.nMoeLayerStep();
        final int nLayer = b.nLayer();
        b.moeLayers(MoeHooksSupport.layersWhere(b,
                il -> il < nLayer && moeStep > 0 && (il + 1) % moeStep == 0));             // tensors :62
        final int ropeStep = b.noRopeLayerStep();
        b.ropeEnabledArray(MoeHooksSupport.layersWhere(b,
                il -> ropeStep > 0 && (il + 1) % ropeStep != 0));                         // graph :143-144
        b.gating(GatingFunc.SIGMOID).expertWeightsNorm(false);                            // graph :226-228
    }

    /**
     * models/grok.cpp:3-33. The router logit soft-cap is read as llama.cpp does, but no llama.cpp graph applies it
     * ({@code f_router_logit_softcapping}, llama-hparams.h:148, is otherwise only written by the model saver).
     * Graph routing: softmax, normalized (models/grok.cpp:165-167).
     */
    public static void grok(ModelHparamsBuilder b, GgufKv kv) {
        b.yarnBetaFast(8.0f);                                                             // :5
        b.logitScale(0.5773502691896257f);                                                // :6
        b.embeddingScale(78.38367176906169f);                                             // :7
        b.attnOutScale(0.08838834764831845f);                                             // :8
        b.attnSoftcap(30.0f);                                                             // :9
        b.routerSoftcap(30.0f);                                                           // :10
        b.finalSoftcap(0.0f);                                                             // :12
        rmsEps(b, kv);                                                                    // :14
        HparamsSupport.readExpertFfLength(b, kv, false);                                  // :15
        b.logitScale(kv.f32(GgufKeys.LOGIT_SCALE, b.logitScale()));                       // :16
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));           // :17
        b.attnOutScale(kv.f32(GgufKeys.ATTENTION_OUTPUT_SCALE, b.attnOutScale()));        // :18
        b.attnSoftcap(kv.f32(GgufKeys.ATTN_LOGIT_SOFTCAPPING, b.attnSoftcap()));          // :19
        b.routerSoftcap(kv.f32(GgufKeys.ROUTER_LOGIT_SOFTCAPPING, b.routerSoftcap()));    // :20
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, b.finalSoftcap()));       // :21
        b.attnTempLength(kv.u32(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, b.attnTempLength())); // :23
        b.yarnExtFactor(kv.f32(GgufKeys.ROPE_SCALING_YARN_EXT_FACTOR, b.yarnExtFactor()));   // :24
        b.yarnAttnFactor(kv.f32(GgufKeys.ROPE_SCALING_YARN_ATTN_FACTOR, b.yarnAttnFactor())); // :25
        b.yarnBetaFast(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, b.yarnBetaFast()));   // :26
        b.yarnBetaSlow(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, b.yarnBetaSlow()));   // :27
        MoeHooksSupport.requireExperts(b, "grok");                                        // tensors :38-40
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :165-167
    }

    /** models/qwen2moe.cpp:3-7; graph models/qwen2moe.cpp:139-141 routes with softmax, unnormalized. */
    public static void qwen2moe(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, false);                                  // :4
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :5
        rmsEps(b, kv);                                                                    // :7
        MoeHooksSupport.requireExperts(b, "qwen2moe");                                    // tensors :37-42
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(false);                            // graph :139-141
    }

    /** models/qwen3vlmoe.cpp:3-7; graph models/qwen3vlmoe.cpp:152-154 routes with softmax, normalized. */
    public static void qwen3vlmoe(ModelHparamsBuilder b, GgufKv kv) {
        b.nDeepstackLayers(kv.u32(GgufKeys.NUM_DEEPSTACK_LAYERS, b.nDeepstackLayers()));  // :4
        HparamsSupport.readRopeSections(b, kv, true);                                     // :5
        HparamsSupport.readExpertFfLength(b, kv, false);                                  // :6
        rmsEps(b, kv);                                                                    // :7
        MoeHooksSupport.requireExperts(b, "qwen3vlmoe");                                  // tensors :44-49
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :152-154
    }

    /**
     * models/phimoe.cpp:3-4. No sliding window is configured (the phi3 SWA branch is not part of the phimoe
     * loader). Graph (phi3 template, models/phi3.cpp:160-162): softmax, normalized.
     */
    public static void phimoe(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // models/phi3.cpp:160-162
    }

    /** models/dbrx.cpp:3-5; graph models/dbrx.cpp:122-124 routes with softmax, normalized. */
    public static void dbrx(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));                      // :4
        b.clampKqv(kv.f32Required(GgufKeys.ATTENTION_CLAMP_KQV));                         // :5
        MoeHooksSupport.requireExperts(b, "dbrx");                                        // tensors :16-18
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :122-124
    }

    /** models/olmoe.cpp:3-4; graph models/olmoe.cpp:143-145 routes with softmax, unnormalized. */
    public static void olmoe(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        MoeHooksSupport.requireExperts(b, "olmoe");                                       // tensors :35-40
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(false);                            // graph :143-145
    }

    /** models/arctic.cpp:3-4; graph models/arctic.cpp:148-150 routes with softmax, normalized. */
    public static void arctic(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :148-150
    }

    /** models/deepseek.cpp:3-8; graph models/deepseek.cpp:152-154 routes with softmax, unnormalized. */
    public static void deepseek(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :5
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :6
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :7
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :8
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(false);                            // graph :152-154
        MoeHooksSupport.requireExpertsAfterLead(b, "deepseek", b.nLayer());               // tensors :50-55
    }

    /** models/granite-moe.cpp:3-18; graph (granite template, models/granite.cpp:284-286): softmax, normalized. */
    public static void graniteMoe(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));                               // :5
        b.residualScale(kv.f32(GgufKeys.RESIDUAL_SCALE, b.residualScale()));              // :6
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));           // :7
        b.attentionScale(kv.f32(GgufKeys.ATTENTION_SCALE, b.attentionScale()));           // :8
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :18
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // models/granite.cpp:284-286
    }

    /** models/bailingmoe.cpp:3-9; graph models/bailingmoe.cpp:135-137 routes with softmax and the loaded norm flag. */
    public static void bailingmoe(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :5
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :6
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :7
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :8
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :9
        b.gating(GatingFunc.SOFTMAX);                                                     // graph :137
        MoeHooksSupport.requireExperts(b, "bailingmoe");                                  // tensors :41-46
    }

    /** models/bailingmoe2.cpp:3-11. MoE layers are {@code il >= n_layer_dense_lead}, NextN blocks included (:54). */
    public static void bailingmoe2(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :5
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :6
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :7
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :8
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :9
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :10
        b.gating(requiredGating(kv));                                                     // :11
        MoeHooksSupport.requireExperts(b, "bailingmoe2");                                 // tensors :32-33
    }

    /** models/dots1.cpp:3-10; the graph passes the loaded gating (models/dots1.cpp:154), which has no default. */
    public static void dots1(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :5
        MoeHooksSupport.requireExpertsAfterLead(b, "dots1", b.nLayer());                  // tensors :50-55
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :6
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :7
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :8
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :9
        b.gating(requiredGating(kv));                                                     // :10
    }

    /**
     * ERNIE 4.5 MoE: llama_model_ernie4_5::load_arch_hparams with the MoE branch (models/ernie4-5.cpp:3-13).
     * The graph's MoE rule {@code il >= n_layer_dense_lead && (il + 1) % n_moe_layer_step == 0}
     * (models/ernie4-5-moe.cpp:63-64) divides by the step, so it must be positive. Routing: softmax, normalized
     * (models/ernie4-5-moe.cpp:88-90).
     */
    public static void ernie45Moe(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readRopeSections(b, kv, false);                                    // :5
        rmsEps(b, kv);                                                                    // :7
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :9
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :10
        b.nMoeLayerStep(MoeHooksSupport.requirePositive(kv, GgufKeys.INTERLEAVE_MOE_LAYER_STEP,
                kv.u32Required(GgufKeys.INTERLEAVE_MOE_LAYER_STEP)));                     // :11
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :12
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :88-90
    }

    /** models/hunyuan-moe.cpp:3-6; graph models/hunyuan-moe.cpp:154-157 routes with softmax, normalized. */
    public static void hunyuanMoe(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :6
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :155-157
    }

    /**
     * models/hy-v3.cpp:3-14. Dense and MoE blocks are told apart by {@code ffn_gate_inp} presence in the graph
     * (no dense-lead key exists for this architecture).
     */
    public static void hyV3(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :6
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :7, :12-13
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :8
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :9
    }

    /** models/grovemoe.cpp:3-8; graph models/grovemoe.cpp:144-146 routes with softmax, normalized. */
    public static void grovemoe(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :4
        b.nFfChexp(kv.u32(GgufKeys.EXPERT_CHUNK_FEED_FORWARD_LENGTH, b.nFfChexp()));      // :5
        b.expertGroupScale(kv.f32Required(GgufKeys.EXPERT_GROUP_SCALE));                  // :6
        b.nGroupExperts(kv.u32Required(GgufKeys.EXPERTS_PER_GROUP));                      // :7
        rmsEps(b, kv);                                                                    // :8
        if (b.nGroupExperts() <= 0) {                                                     // tensors :31
            throw new GgufFormatException("grovemoe: experts_per_group must be positive");
        }
        MoeHooksSupport.requireExperts(b, "grovemoe");                                    // tensors :29-30
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :144-146
    }

    /** models/minimax-m2.cpp:3-6; graph models/minimax-m2.cpp:137-139: normalized weights, loaded gating. */
    public static void minimaxM2(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        b.gating(requiredGating(kv));                                                     // :6
        b.expertWeightsNorm(true);                                                        // graph :137
    }

    /**
     * models/k2-horizon.cpp:3-35. MoE layers are {@code n_expert > 0 && il >= n_layer_dense_lead}
     * (models/k2-horizon.cpp:67); {@code moe_every_n_layers} is read but not used by the graph. Value-expert
     * (MoVA) counts are checked as the llama.cpp asserts do (:29-35).
     */
    public static void k2Horizon(ModelHparamsBuilder b, GgufKv kv) {
        b.yarnBetaFast(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, b.yarnBetaFast()));   // :4
        b.yarnBetaSlow(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, b.yarnBetaSlow()));   // :5
        rmsEps(b, kv);                                                                    // :6
        b.nNormGroups(kv.u32(GgufKeys.ATTENTION_GROUPNORM_GROUPS, b.nNormGroups()));      // :7
        if (b.nNormGroups() == 0) {                                                       // :8
            b.nNormGroups(1);                                                             // :9
        }
        if (b.nExpert() > 0) {                                                            // :12
            HparamsSupport.readExpertFfLength(b, kv, true);                               // :13
            b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :14
            b.moeEveryN(kv.u32(GgufKeys.MOE_EVERY_N_LAYERS, b.moeEveryN()));              // :15
            b.nExpertShared(kv.u32(GgufKeys.EXPERT_SHARED_COUNT, b.nExpertShared()));     // :16
            b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp())); // :17
            b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :18
            b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :19
            b.gating(gating(kv, GatingFunc.SIGMOID));                                     // :20-23
            for (int il = b.nLayerDenseLead(); il < b.nLayer(); il++) {                   // tensors :91-94
                if (b.nFfExp(il) == 0) {
                    throw new GgufFormatException("K2 Horizon MoE layer requires expert_feed_forward_length");
                }
            }
        }
        b.nValueExpert(kv.u32(GgufKeys.ATTENTION_VALUE_EXPERT_COUNT, b.nValueExpert()));  // :27
        b.nValueExpertUsed(kv.u32(GgufKeys.ATTENTION_VALUE_EXPERT_USED_COUNT, b.nValueExpertUsed())); // :28
        int nv = b.nValueExpert();
        int nvUsed = b.nValueExpertUsed();
        if (nv > 0) {                                                                     // :29-32
            if (nv > ModelHparams.MAX_EXPERTS || nvUsed <= 0 || nvUsed > nv) {
                throw new GgufFormatException("k2-horizon: invalid value experts: " + nvUsed + " used of " + nv
                        + " (maximum " + ModelHparams.MAX_EXPERTS + ")");
            }
        } else if (nvUsed != 0) {                                                         // :33-34
            throw new GgufFormatException("k2-horizon: value_expert_used_count set without value experts");
        }
    }
}
