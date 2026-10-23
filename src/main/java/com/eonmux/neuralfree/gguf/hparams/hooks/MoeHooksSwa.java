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
 * Hparams hooks for the sliding-window MoE architectures, referenced from
 * {@link com.eonmux.neuralfree.gguf.arch.ArchTableMoeSwa}. Each hook transcribes {@code load_arch_hparams}
 * of llama.cpp src/models/&lt;file&gt;.cpp (cited {@code models/<file>.cpp:line}); conventions as in {@link MoeHooks}.
 * Where the graph decides per layer whether RoPE is applied from the SWA flags or a NoPE step, the hook
 * stores that decision in {@link ModelHparamsBuilder#ropeEnabledArray(boolean[])}.
 */
public final class MoeHooksSwa {

    /** Window forced by smallthinker when the file enables SWA (models/smallthinker.cpp:8). */
    private static final int SMALLTHINKER_SWA = 4096;
    /** Window preset by exaone-moe before the required key is read (models/exaone-moe.cpp:5). */
    private static final int EXAONE_MOE_SWA = 128;

    private MoeHooksSwa() {
    }

    /** models/flint.cpp:3-18; RoPE per {@code attention.rope_pattern} (models/flint.cpp:112). */
    public static void flint(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLayerDenseLead(kv.u32Required(GgufKeys.LEADING_DENSE_BLOCK_COUNT));            // :5
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :6
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :7
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :8
        b.expertWeightsNorm(MoeHooksSupport.boolRequired(kv, GgufKeys.EXPERT_WEIGHTS_NORM)); // :9
        b.gating(requiredGating(kv));                                                     // :10
        b.nFfShexp(kv.u32Required(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH));           // :12
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :13
        b.ropeEnabledArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_ROPE_PATTERN,
                b.ropeEnabledArray(), true));                                             // :14
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :15
        b.swaType(SwaType.STANDARD);                                                      // :16
        MoeHooksSupport.requireExpertsAfterLead(b, "flint", b.nLayer());                  // tensors :53-58
    }

    /**
     * models/gemma4.cpp:4-27. The last {@code shared_kv_layers} blocks reuse earlier KV caches; SWA layers attend
     * bidirectionally when non-causal attention is requested. Graph MoE: softmax, normalized, scale 1
     * (models/gemma4.cpp:338-340).
     */
    public static void gemma4(ModelHparamsBuilder b, GgufKv kv) {
        b.swaType(SwaType.STANDARD);                                                      // :5
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :6
        int sharedKv = kv.u32(GgufKeys.ATTENTION_SHARED_KV_LAYERS, 0);                    // :8-9
        b.sharedKvLayers(sharedKv);
        b.nLayerKvFromStart(b.nLayerAll() - sharedKv);                                    // :11
        b.attentionScale(1.0f);                                                           // :12
        b.ropeFreqBaseSwa(kv.f32(GgufKeys.ROPE_FREQ_BASE_SWA, b.ropeFreqBaseSwa()));      // :14
        HparamsSupport.readExpertFfLength(b, kv, false);                                  // :15
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :16
        rmsEps(b, kv);                                                                    // :17
        b.nEmbdPerLayer(kv.u32Required(GgufKeys.EMBEDDING_LENGTH_PER_LAYER));             // :18
        b.headKSwa(kv.u32Required(GgufKeys.ATTENTION_KEY_LENGTH_SWA));                    // :19
        b.headVSwa(kv.u32Required(GgufKeys.ATTENTION_VALUE_LENGTH_SWA));                  // :20
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, b.finalSoftcap()));       // :21
        b.nonCausalType(ModelHparams.NON_CAUSAL_SWA_ONLY);                                // :26
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true).expertWeightsScale(1.0f);    // graph :338-340
        if (b.headKFull() != b.headVFull() || b.headKSwa() != b.headVSwa()) {             // tensors :43-48
            throw new GgufFormatException("Gemma 4 requires equal key and value head sizes (full "
                    + b.headKFull() + "/" + b.headVFull() + ", SWA " + b.headKSwa() + "/" + b.headVSwa() + ")");
        }
    }

    /**
     * models/maple.cpp:3-22. RoPE only on SWA layers (models/maple.cpp:90). Graph MoE: softmax, normalized, scale 1
     * (models/maple.cpp:128-130).
     */
    public static void maple(ModelHparamsBuilder b, GgufKv kv) {
        b.swaType(SwaType.STANDARD);                                                      // :4
        rmsEps(b, kv);                                                                    // :6
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :7
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :8
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :10
        MoeHooksSupport.swaRopeFromFull(b, kv);                                           // :12-14
        b.swigluClampExpArray(MoeHooksSupport.keyOrArrF32(b, kv, GgufKeys.SWIGLU_CLAMP_EXP, true)); // :16
        b.ropeEnabledArray(b.isSwaArray());                                               // graph :90
        MoeHooksSupport.requireExperts(b, "maple");                                       // tensors :35-40
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true).expertWeightsScale(1.0f);    // graph :128-130
    }

    /**
     * models/cohere2moe.cpp:3-38. At least one of the layer-norm epsilons must be present; an absent RMS epsilon
     * is stored as 0, which selects LayerNorm in the graph (models/cohere2moe.cpp:149). RoPE applies to SWA layers
     * and to the dense lead (models/cohere2moe.cpp:160-175).
     */
    public static void cohere2moe(ModelHparamsBuilder b, GgufKv kv) {
        boolean foundNorm = kv.has(GgufKeys.ATTENTION_LAYERNORM_EPS);                     // :4
        boolean foundRms = kv.has(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS);                  // :5
        if (!foundNorm && !foundRms) {                                                    // :6-8
            throw new GgufFormatException("missing Cohere2 MoE norm epsilon");
        }
        if (foundNorm) {
            b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));
        }
        b.normRmsEps(foundRms ? kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS) : 0.0f); // :9-11
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :13
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));                               // :14
        b.nLayerDenseLead(kv.u32Required(GgufKeys.LEADING_DENSE_BLOCK_COUNT));            // :15
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :16
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :17
        b.nExpertShared(kv.u32(GgufKeys.EXPERT_SHARED_COUNT, b.nExpertShared()));         // :18
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :19
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :20
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :21-25
        b.swaType(SwaType.STANDARD);                                                      // :27
        HparamsSupport.loadSwaPattern(b, kv, 4, true);                                    // :28
        MoeHooksSupport.swaRopeFromFull(b, kv);                                           // :30-32
        final boolean[] swa = b.isSwaArray();
        final int lead = b.nLayerDenseLead();
        b.ropeEnabledArray(MoeHooksSupport.layersWhere(b, il -> swa[il] || il < lead));  // graph :160-175
        MoeHooksSupport.requireExperts(b, "cohere2moe");                                  // tensors :58-63
    }

    /**
     * models/muse-spark.cpp:3-19. RoPE only on SWA layers (models/muse-spark.cpp:105); the graph passes the loaded
     * gating (models/muse-spark.cpp:193).
     */
    public static void museSpark(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.postNormEps(kv.f32Required(GgufKeys.POST_NORM_EPS));                            // :5
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :6
        b.ropeFreqBaseSwa(kv.f32Required(GgufKeys.ROPE_FREQ_BASE_SWA));                   // :7
        b.finalSoftcap(kv.f32Required(GgufKeys.FINAL_LOGIT_SOFTCAPPING));                 // :8
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));                               // :9
        b.moeLatentSize(kv.u32Required(GgufKeys.MOE_LATENT_SIZE));                        // :10
        b.expertWeightsScale(kv.f32Required(GgufKeys.EXPERT_WEIGHTS_SCALE));              // :11
        b.gating(requiredGating(kv));                                                     // :12
        b.expertWeightsNorm(MoeHooksSupport.boolRequired(kv, GgufKeys.EXPERT_WEIGHTS_NORM)); // :13
        HparamsSupport.readExpertFfLength(b, kv, false);                                  // :14
        b.swaType(SwaType.STANDARD);                                                      // :16
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :17
        b.ropeEnabledArray(b.isSwaArray());                                               // graph :105
    }

    /**
     * models/exaone-moe.cpp:3-20. RoPE only on SWA (local) layers (models/exaone-moe.cpp:131,150). The leading dense
     * blocks and the NextN blocks use a dense FFN (tensors models/exaone-moe.cpp:67), so MoE layers are the trunk
     * layers at or after the dense lead.
     */
    public static void exaoneMoe(ModelHparamsBuilder b, GgufKv kv) {
        b.swaType(SwaType.STANDARD);                                                      // :4
        b.nSwa(EXAONE_MOE_SWA);                                                           // :5
        HparamsSupport.loadSwaPattern(b, kv, 4, false);                                   // :6
        MoeHooksSupport.swaRopeFromFull(b, kv);                                           // :7-10
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :11
        rmsEps(b, kv);                                                                    // :12
        b.nExpertShared(kv.u32(GgufKeys.EXPERT_SHARED_COUNT, b.nExpertShared()));         // :13
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :14
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :15
        b.gating(requiredGating(kv));                                                     // :16
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :17
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :18
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :19
        b.ropeEnabledArray(b.isSwaArray());                                               // graph :131,150
        final int nLayer = b.nLayer();
        final int lead = b.nLayerDenseLead();
        final boolean experts = b.nExpert() > 0;
        b.moeLayers(MoeHooksSupport.layersWhere(b, il -> experts && il >= lead && il < nLayer)); // tensors :67
        MoeHooksSupport.requireExpertsAfterLead(b, "exaone-moe", b.nLayer());             // tensors :75-80
    }

    /**
     * models/granite-swa.cpp:5-54. Expert count and per-layer used count are already read by the generic loader
     * (llama-model.cpp:1402-1404) with the same optional semantics. RoPE per the optional
     * {@code attention.rope_pattern} (models/granite-swa.cpp:209). Graph MoE: softmax, normalized
     * (models/granite-swa.cpp:282-284).
     */
    public static void graniteSwa(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :6
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));                               // :7
        b.residualScale(kv.f32(GgufKeys.RESIDUAL_SCALE, b.residualScale()));              // :8
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));           // :9
        b.attentionScale(kv.f32(GgufKeys.ATTENTION_SCALE, b.attentionScale()));           // :10
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :17
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :18
        b.swaType(SwaType.STANDARD);                                                      // :19
        MoeHooksSupport.readDeepstackMapping(b, kv);                                      // :22-40
        b.ropeEnabledArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_ROPE_PATTERN,
                b.ropeEnabledArray(), false));                                            // :43
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :53
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // graph :282-284
    }

    /**
     * models/afmoe.cpp:3-31. SWA (3 sliding, 1 full) only when the file sets a positive window. RoPE follows the
     * NoPE step rule {@code n_no_rope_layer_step > 0 && (il + 1) % n_no_rope_layer_step != 0}
     * (models/afmoe.cpp:135-136).
     */
    public static void afmoe(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :5
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :6
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :7
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :9
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :10
        HparamsSupport.readSlidingWindow(b, kv, false);                                   // :11
        if (b.nSwa() > 0) {                                                               // :15
            b.swaType(SwaType.STANDARD);                                                  // :16
            HparamsSupport.loadSwaPattern(b, kv, 4, false);                               // :17
            MoeHooksSupport.swaRopeFromFull(b, kv);                                       // :19-21
        } else {
            b.swaType(SwaType.NONE);                                                      // :23
        }
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :8, :27-29
        final int step = b.noRopeLayerStep();
        b.ropeEnabledArray(MoeHooksSupport.layersWhere(b,
                il -> step > 0 && (il + 1) % step != 0));                                 // graph :135-136
    }

    /**
     * models/laguna.cpp:9-59. One shared expert by default; a missing shared-expert size is derived from the
     * routed size. SWA (full first, period 4) only with a positive window; SWA layers use unscaled RoPE.
     */
    public static void laguna(ModelHparamsBuilder b, GgufKv kv) {
        b.yarnBetaFast(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, b.yarnBetaFast()));   // :10
        b.yarnBetaSlow(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, b.yarnBetaSlow()));   // :11
        rmsEps(b, kv);                                                                    // :13
        b.nLayerDenseLead(kv.u32Required(GgufKeys.LEADING_DENSE_BLOCK_COUNT));            // :14
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :15
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :17
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :18
        b.nExpertShared(kv.u32(GgufKeys.EXPERT_SHARED_COUNT, 1));                         // :23-24
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :25
        if (b.nFfShexp() == 0) {                                                          // :26
            b.nFfShexp(b.nFfExp(0) * b.nExpertShared());                                  // :30
        }
        b.nSwa(kv.u32(GgufKeys.ATTENTION_SLIDING_WINDOW, 0));                             // :37-38
        if (b.nSwa() > 0) {                                                               // :39
            b.swaType(SwaType.STANDARD);                                                  // :40
            HparamsSupport.loadSwaPattern(b, kv, 4, true);                                // :42
            b.ropeFreqBaseSwa(b.ropeFreqBase());                                          // :48
            b.ropeFreqScaleSwa(1.0f);                                                     // :49
            b.ropeFreqBaseSwa(kv.f32(GgufKeys.ROPE_FREQ_BASE_SWA, b.ropeFreqBaseSwa()));  // :50
            b.nRotSwa(kv.u32(GgufKeys.ROPE_DIMENSION_COUNT_SWA, b.nRotSwa()));            // :51
        }
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :16, :56-58
    }

    /**
     * gpt-oss, models/openai-moe.cpp:3-15: alternating sliding/full layers. Graph MoE: softmax over the selected
     * logits, unnormalized (models/openai-moe.cpp:139-141).
     */
    public static void gptOss(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :6
        b.swaType(SwaType.STANDARD);                                                      // :8
        HparamsSupport.loadSwaPattern(b, kv, 2, false);                                   // :9
        MoeHooksSupport.swaRopeFromFull(b, kv);                                           // :11-13
        b.gating(GatingFunc.SOFTMAX_WEIGHT).expertWeightsNorm(false);                     // graph :139-141
    }

    /**
     * models/smallthinker.cpp:3-23. A present positive window enables SWA with a forced 4096 window; otherwise
     * RoPE skipping is disabled. RoPE rule {@code n_no_rope_layer_step == n_layer || il % step != 0}
     * (models/smallthinker.cpp:106-107). Graph MoE: normalized, loaded gating (models/smallthinker.cpp:156-158).
     */
    public static void smallthinker(ModelHparamsBuilder b, GgufKv kv) {
        boolean foundSwa = HparamsSupport.readSlidingWindow(b, kv, false);                // :4
        if (foundSwa && b.nSwa() > 0) {                                                   // :6
            b.swaType(SwaType.STANDARD);                                                  // :7
            b.nSwa(SMALLTHINKER_SWA);                                                     // :8
            HparamsSupport.loadSwaPattern(b, kv, 4, true);                                // :9
            MoeHooksSupport.swaRopeFromFull(b, kv);                                       // :11-13
        } else {
            b.swaType(SwaType.NONE);                                                      // :15
            b.noRopeLayerStep(b.nLayer());                                                // :16
        }
        HparamsSupport.readExpertFfLength(b, kv, false);                                  // :19
        rmsEps(b, kv);                                                                    // :20
        b.gating(requiredGating(kv));                                                     // :21
        b.expertWeightsNorm(true);                                                        // graph :156
        MoeHooksSupport.requireExperts(b, "smallthinker");                                // tensors :54-55
        final int step = b.noRopeLayerStep();
        final int nLayer = b.nLayer();
        if (step <= 0) {
            throw new GgufFormatException("smallthinker: no-RoPE layer step must be positive");
        }
        b.ropeEnabledArray(MoeHooksSupport.layersWhere(b,
                il -> step == nLayer || il % step != 0));                                 // graph :106-107
    }

    /**
     * models/mimo2.cpp:3-19. The value scale applies only when present and different from 1. MoE layers are those
     * with an {@code ffn_gate_inp} tensor. Graph MoE: sigmoid, normalized (models/mimo2.cpp:212-214).
     */
    public static void mimo2(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.swaType(SwaType.STANDARD);                                                      // :6
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :8
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :9
        b.ropeFreqBaseSwa(kv.f32(GgufKeys.ROPE_FREQ_BASE_SWA, b.ropeFreqBaseSwa()));      // :10
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :12
        if (kv.has(GgufKeys.ATTENTION_VALUE_SCALE)) {                                     // :14-17
            float valueScale = kv.f32Required(GgufKeys.ATTENTION_VALUE_SCALE);
            if (valueScale != 1.0f) {
                b.attnValueScale(valueScale);
            }
        }
        b.gating(GatingFunc.SIGMOID).expertWeightsNorm(true);                             // graph :212-214
    }

    /**
     * models/step35.cpp:3-31. Full-attention layers rotate half of the head dimensions. MoE layers are those with
     * an {@code ffn_gate_inp} tensor.
     */
    public static void step35(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.swaType(SwaType.STANDARD);                                                      // :6
        b.nRotFull(b.nRotFull() / 2);                                                     // :9
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :12
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));     // :13
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :15
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :16
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :14, :19-21
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :23
        b.ropeFreqBaseSwa(kv.f32(GgufKeys.ROPE_FREQ_BASE_SWA, b.ropeFreqBaseSwa()));      // :24
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :26
        float[] clampExp = MoeHooksSupport.keyOrArrF32(b, kv, GgufKeys.SWIGLU_CLAMP_EXP, false); // :28
        if (clampExp != null) {
            b.swigluClampExpArray(clampExp);
        }
        float[] clampShexp = MoeHooksSupport.keyOrArrF32(b, kv, GgufKeys.SWIGLU_CLAMP_SHEXP, false); // :29
        if (clampShexp != null) {
            b.swigluClampShexpArray(clampShexp);
        }
    }

    /**
     * models/maion-coder.cpp:3-25. Every {@code n_moe_layer_step}-th layer is a latent MoE layer
     * ({@code (il + 1) % step == 0}, tensors models/maion-coder.cpp:38); RoPE only on SWA layers. Graph MoE:
     * softmax, normalized, no weight scale (models/maion-coder.cpp:131-132).
     */
    public static void maionCoder(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        b.nMoeLayerStep(kv.u32Required(GgufKeys.INTERLEAVE_MOE_LAYER_STEP));              // :6
        b.moeLatentSize(kv.u32Required(GgufKeys.MOE_LATENT_SIZE));                        // :7
        if (b.nMoeLayerStep() == 0 || b.moeLatentSize() == 0 || b.nExpert() == 0) {       // :9-11
            throw new GgufFormatException(
                    "maion-coder requires a positive MoE layer step, latent size and expert count");
        }
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :13
        requireFullSwaPattern(b, kv,
                "maion-coder requires a positive sliding window and a sliding window pattern for each layer"); // :14-18
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :19
        b.ropeEnabledArray(b.isSwaArray());                                               // :20
        b.swaType(SwaType.STANDARD);                                                      // :21
        b.ropeFreqBaseSwa(b.ropeFreqBase());                                              // :22
        b.ropeFreqScaleSwa(b.ropeFreqScale());                                            // :23
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true).expertWeightsScale(0.0f);    // graph :131-132
    }

    /** models/mellum.cpp:3-26: softmax gating and weight normalization by default; SWA only with a positive window. */
    public static void mellum(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        HparamsSupport.readSlidingWindow(b, kv, false);                                   // :6
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, true));                 // :8, :10
        b.gating(gating(kv, GatingFunc.SOFTMAX));                                         // :9, :11
        if (b.nSwa() > 0) {                                                               // :13
            b.swaType(SwaType.STANDARD);                                                  // :14
            HparamsSupport.loadSwaPattern(b, kv, 4, false);                               // :16
            b.ropeFreqBaseSwa(b.ropeFreqBase());                                          // :18
            b.ropeFreqScaleSwa(1.0f);                                                     // :19
            b.ropeFreqBaseSwa(kv.f32(GgufKeys.ROPE_FREQ_BASE_SWA, b.ropeFreqBaseSwa()));  // :21
        } else {
            b.swaType(SwaType.NONE);                                                      // :23
        }
        MoeHooksSupport.requireExperts(b, "mellum");                                      // tensors :45-50
    }

    /**
     * models/beam.cpp:3-53. Full-attention layers are NoPE and scale queries by a power-form temperature over
     * {@code pos + 1}; at least one SWA layer is required.
     */
    public static void beam(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLayerDenseLead(kv.u32Required(GgufKeys.LEADING_DENSE_BLOCK_COUNT));            // :5
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :6
        b.expertWeightsScale(kv.f32Required(GgufKeys.EXPERT_WEIGHTS_SCALE));              // :8
        b.expertWeightsNorm(MoeHooksSupport.boolRequired(kv, GgufKeys.EXPERT_WEIGHTS_NORM)); // :9
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, 0.0f));                   // :10-11
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :7, :13-15
        if (b.nLayerDenseLead() > b.nLayer()) {                                           // :16-18
            throw new GgufFormatException("beam: leading_dense_block_count exceeds the layer count");
        }
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :20
        b.swaType(SwaType.STANDARD);                                                      // :21
        requireFullSwaPattern(b, kv,
                "beam: requires a positive sliding window and a sliding window pattern for each layer"); // :22-26
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :27
        if (!b.isSwaAny()) {                                                              // :28-30
            throw new GgufFormatException("beam: sliding window pattern has no sliding window layer");
        }
        b.ropeEnabledArray(b.isSwaArray());                                               // :33
        MoeHooksSupport.swaRopeFromFull(b, kv);                                           // :34-36
        b.attnTempScale(kv.f32(GgufKeys.ATTENTION_TEMPERATURE_SCALE, 0.1f));              // :39, :41
        b.attnTempLength(kv.u32(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, 2048));            // :40, :42
        if (b.attnTempScale() == 0.0f) {                                                  // :43-45
            throw new GgufFormatException("beam: attention.temperature_scale must be nonzero");
        }
        if (b.attnTempLength() == 0) {                                                    // :46-48
            throw new GgufFormatException("beam: attention.temperature_length must be positive");
        }
        b.attnTempFloorScale(b.attnTempLength());                                         // :49
        b.attnTempOffset(1.0f);                                                           // :50
        b.attnTempPow(true);                                                              // :51
    }

    /**
     * models/kolibri1.cpp:3-22: one shared expert, RoPE only on SWA layers (models/kolibri1.cpp:98). Graph MoE: loaded
     * gating and norm flag, weight scale 1 (models/kolibri1.cpp:151-152).
     */
    public static void kolibri1(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :5
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :6
        b.nFfShexp(kv.u32Required(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH));           // :7
        b.expertWeightsNorm(MoeHooksSupport.boolRequired(kv, GgufKeys.EXPERT_WEIGHTS_NORM)); // :8
        b.gating(requiredGating(kv));                                                     // :9
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :10
        if (b.nSwa() == 0 || b.nExpertShared() != 1 || b.nFfShexp() == 0) {               // :12-14
            throw new GgufFormatException("Kolibri-1: invalid sliding window or shared expert configuration");
        }
        b.swaType(SwaType.STANDARD);                                                      // :16
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :17
        b.ropeFreqBaseSwa(b.ropeFreqBase());                                              // :19
        b.ropeFreqScaleSwa(b.ropeFreqScale());                                            // :20
        b.ropeEnabledArray(b.isSwaArray());                                               // graph :98
        b.expertWeightsScale(1.0f);                                                       // graph :151
    }

    /** {@code get_arr_n(sliding_window_pattern)} must equal the block count and the window must be positive. */
    private static void requireFullSwaPattern(ModelHparamsBuilder b, GgufKv kv, String message) {
        String key = GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN;
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
        }
        if (b.nSwa() == 0 || kv.arrayLength(key) != b.nLayerAll()) {
            throw new GgufFormatException(message);
        }
    }
}
