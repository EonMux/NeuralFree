package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import java.util.Arrays;

/**
 * Hooks of the speculative-decoding drafters eagle3, gemma4-assistant and dflash. Each transcribes
 * {@code llama_model_<arch>::load_arch_hparams}; line references are to llama.cpp src/models/.
 */
public final class SpecialHooksDraft {

    /** EAGLE3 fuses exactly three target layers (eagle3.cpp:9-11). */
    private static final int EAGLE3_TARGET_LAYERS = 3;

    public static final ArchHparamsHook EAGLE3 = SpecialHooksDraft::eagle3;
    public static final ArchHparamsHook GEMMA4_ASSISTANT = SpecialHooksDraft::gemma4Assistant;
    public static final ArchHparamsHook DFLASH = SpecialHooksDraft::dflash;

    private SpecialHooksDraft() {
    }

    /** eagle3.cpp:3-36: three target layers of width {@code target_hidden_size} feed the fusion projection. */
    private static void eagle3(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // eagle3.cpp:4
        int[] targets = kv.i32Array(GgufKeys.TARGET_LAYERS);                       // eagle3.cpp:6-8
        if (targets == null) {
            throw SpecialHooks.invalid(b, "EAGLE3 model requires '" + kv.key(GgufKeys.TARGET_LAYERS)
                    + "' (extract_layers) in GGUF metadata");
        }
        if (targets.length != EAGLE3_TARGET_LAYERS) {                              // eagle3.cpp:9-11
            throw SpecialHooks.invalid(b, "EAGLE3 requires exactly " + EAGLE3_TARGET_LAYERS + " entries in '"
                    + kv.key(GgufKeys.TARGET_LAYERS) + "', got " + targets.length);
        }
        b.targetLayers(targets);
        int nEmbdTarget = kv.u32Required(GgufKeys.TARGET_HIDDEN_SIZE);              // eagle3.cpp:19
        b.targetHiddenSize(nEmbdTarget);
        b.nEmbdInpEncImpl(targets.length * nEmbdTarget);                           // eagle3.cpp:22
        b.normBeforeResidual(kv.bool(GgufKeys.NORM_BEFORE_RESIDUAL, false));       // eagle3.cpp:26
        b.normBeforeFc(kv.bool(GgufKeys.NORM_BEFORE_FC, false));                   // eagle3.cpp:33
    }

    /**
     * gemma4-assistant.cpp:3-19: the input width is the target's hidden size carried by
     * {@code embedding_length_out}; SWA layers come from a required bool array; the head-size and width checks
     * are those of gemma4-assistant.cpp:24-32.
     */
    private static void gemma4Assistant(ModelHparamsBuilder b, GgufKv kv) {
        int nEmbdOut = b.nEmbdOutImpl() > 0 ? b.nEmbdOutImpl() : b.nEmbd();
        b.nEmbdInpImpl(nEmbdOut);                                                  // gemma4-assistant.cpp:4
        b.swaType(SwaType.STANDARD);                                               // gemma4-assistant.cpp:6
        requireSwaPatternArray(b, kv);                                             // gemma4-assistant.cpp:7
        kv.u32(GgufKeys.ATTENTION_SHARED_KV_LAYERS, 0);                            // gemma4-assistant.cpp:9-10, unused
        b.attentionScale(1.0f);                                                    // gemma4-assistant.cpp:12
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());           // gemma4-assistant.cpp:14
        HparamsSupport.readSlidingWindow(b, kv, true);                             // gemma4-assistant.cpp:15
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // gemma4-assistant.cpp:16
        b.headKSwa(kv.u32Required(GgufKeys.ATTENTION_KEY_LENGTH_SWA));            // gemma4-assistant.cpp:17
        b.headVSwa(kv.u32Required(GgufKeys.ATTENTION_VALUE_LENGTH_SWA));          // gemma4-assistant.cpp:18
        SpecialHooks.requireEqualHeadSizes(b);                                     // gemma4-assistant.cpp:24-29
        if (nEmbdOut == b.nEmbd()) {                                               // gemma4-assistant.cpp:30-32
            throw SpecialHooks.invalid(b, "Gemma 4 assistant requires embedding_length_out to carry the target"
                    + " hidden size");
        }
    }

    /** Required bool array {@code attention.sliding_window_pattern} ({@code get_arr} without the optional flag). */
    private static void requireSwaPatternArray(ModelHparamsBuilder b, GgufKv kv) {
        String key = GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN;
        if (!kv.has(key)) {
            throw SpecialHooks.missingKey(kv, key);
        }
        if (!kv.isArray(key)) {
            throw new GgufFormatException("metadata key '" + kv.key(key) + "' must be a per-layer bool array");
        }
        HparamsSupport.loadSwaPattern(b, kv, 0, false);
    }

    /**
     * dflash.cpp:7-119: block-diffusion drafter conditioned on target hidden states. Optional LiLiCorr
     * correction head, DFlash2 convolution/selector metadata, and a DeepSeek-V4 (DSpark) backbone when
     * {@code hyper_connection.count > 0}; otherwise an optional interleaved sliding window. The metadata-only
     * checks of the tensor loader (dflash.cpp:153-156 and 184-190) run here as well.
     */
    private static void dflash(ModelHparamsBuilder b, GgufKv kv) {
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));   // dflash.cpp:9
        b.attentionScale(kv.f32(GgufKeys.ATTENTION_SCALE, b.attentionScale()));   // dflash.cpp:10
        b.attnValueScale(kv.f32(GgufKeys.ATTENTION_VALUE_SCALE, b.attnValueScale())); // dflash.cpp:11
        String hiddenAct = kv.str(GgufKeys.HIDDEN_ACT, null);                      // dflash.cpp:13-21
        if (hiddenAct != null) {
            if (!"gelu".equals(hiddenAct) && !"gelu_pytorch_tanh".equals(hiddenAct) && !"silu".equals(hiddenAct)) {
                throw SpecialHooks.invalid(b, "unsupported DFlash hidden activation: " + hiddenAct);
            }
            b.hiddenActivation(hiddenAct);
        }
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // dflash.cpp:23
        b.logitScale(kv.f32(GgufKeys.LOGIT_SCALE, b.logitScale()));               // dflash.cpp:24
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, 0.0f));          // dflash.cpp:25-26
        HparamsSupport.readRopeSections(b, kv, false);                             // dflash.cpp:29
        b.dflashBlockSize(kv.u32(GgufKeys.DFLASH_BLOCK_SIZE, b.dflashBlockSize()));             // dflash.cpp:31
        b.dflashConvKernelSize(kv.u32(GgufKeys.DFLASH_CONV_KERNEL_SIZE, b.dflashConvKernelSize())); // dflash.cpp:32
        b.dflashConvGroupSize(kv.u32(GgufKeys.DFLASH_CONV_GROUP_SIZE, b.dflashConvGroupSize()));    // dflash.cpp:33
        b.dflashSelectorRank(kv.u32(GgufKeys.DFLASH_SELECTOR_RANK, b.dflashSelectorRank()));       // dflash.cpp:34
        b.dflashSelectorTopK(kv.u32(GgufKeys.DFLASH_SELECTOR_TOP_K, b.dflashSelectorTopK()));      // dflash.cpp:35
        dflashLilicorr(b, kv);                                                     // dflash.cpp:36-48
        dflashTargets(b, kv);                                                      // dflash.cpp:50-54
        boolean conv = b.dflashConvKernelSize() > 0 && b.dflashConvGroupSize() > 0;   // dflash.cpp:153-156
        if (conv && b.nEmbd() % b.dflashConvGroupSize() != 0) {
            throw SpecialHooks.invalid(b, "DFlash hidden size " + b.nEmbd() + " must be divisible by conv_group_size "
                    + b.dflashConvGroupSize());
        }
        b.dsv4HcMult(kv.u32(GgufKeys.HYPER_CONNECTION_COUNT, 0));                 // dflash.cpp:66
        if (b.dsv4HcMult() > 0) {
            dflashDsv4(b, kv);                                                     // dflash.cpp:67-107
            return;
        }
        if (HparamsSupport.readSlidingWindow(b, kv, false) && b.nSwa() > 0) {    // dflash.cpp:111
            b.swaType(SwaType.STANDARD);                                           // dflash.cpp:112
            requireSwaPatternArray(b, kv);                                         // dflash.cpp:113
            b.ropeFreqBaseSwa(b.ropeFreqBase());                                   // dflash.cpp:114
            b.ropeFreqScaleSwa(b.ropeFreqScale());                                 // dflash.cpp:115
        }
    }

    /**
     * LiLiCorr keys (dflash.cpp:36-48): all required once {@code lilicorr.hidden_size} is present, then
     * validated; the head dimensions read by the tensor loader (dflash.cpp:184-190) are checked as well. Without
     * {@code lilicorr.hidden_size} a positive {@code lilicorr.num_layers} is rejected: llama.cpp still reads it
     * and would build a zero-width head (dflash.cpp:38, 179).
     */
    private static void dflashLilicorr(ModelHparamsBuilder b, GgufKv kv) {
        if (!kv.has(GgufKeys.LILICORR_HIDDEN_SIZE)) {
            if (kv.u32(GgufKeys.LILICORR_NUM_LAYERS, 0) > 0) {
                throw SpecialHooks.invalid(b, "'" + kv.key(GgufKeys.LILICORR_NUM_LAYERS) + "' is set without '"
                        + kv.key(GgufKeys.LILICORR_HIDDEN_SIZE) + "'");
            }
            return;
        }
        b.lilicorrHiddenSize(kv.u32Required(GgufKeys.LILICORR_HIDDEN_SIZE));
        b.lilicorrCandidateTopK(kv.u32Required(GgufKeys.LILICORR_CANDIDATE_TOPK));
        b.lilicorrNumHeads(kv.u32Required(GgufKeys.LILICORR_NUM_HEADS));
        b.lilicorrLayers(kv.u32Required(GgufKeys.LILICORR_NUM_LAYERS));
        b.lilicorrVectorEps(kv.f32Required(GgufKeys.LILICORR_VECTOR_EPS));
        b.lilicorrLogitScale(kv.f32Required(GgufKeys.LILICORR_LOGIT_SCALE));
        long topK = b.lilicorrCandidateTopK();
        if (b.dflashBlockSize() < 2 || b.lilicorrHiddenSize() == 0 || b.lilicorrNumHeads() == 0
                || b.lilicorrHiddenSize() % b.lilicorrNumHeads() != 0 || topK == 0 || b.lilicorrLayers() == 0
                || b.dflashSelectorTopK() > 0 || topK * (topK + 1) > b.nEmbd()) {
            throw SpecialHooks.invalid(b, "invalid LiLiCorr DFlash metadata");
        }
        b.lilicorrFactorDim(kv.u32Required(GgufKeys.LILICORR_FACTOR_DIM));
        b.lilicorrMlpHiddenSize(kv.u32Required(GgufKeys.LILICORR_MLP_HIDDEN_SIZE));
        if (b.lilicorrFactorDim() == 0 || b.lilicorrMlpHiddenSize() == 0 || b.lilicorrLayers() > b.nLayerAll()) {
            throw SpecialHooks.invalid(b, "invalid LiLiCorr head dimensions");
        }
    }

    /** Target layers (dflash.cpp:50-54): required; their hidden states are concatenated as the fc input. */
    private static void dflashTargets(ModelHparamsBuilder b, GgufKv kv) {
        int[] targets = kv.i32Array(GgufKeys.TARGET_LAYERS);
        if (targets == null) {
            throw SpecialHooks.invalid(b, "DFlash model requires '" + kv.key(GgufKeys.TARGET_LAYERS)
                    + "' in GGUF metadata");
        }
        b.targetLayers(targets);
        b.nEmbdInpEncImpl(targets.length * b.nEmbd());
    }

    /**
     * DeepSeek-V4 DSpark backbone (dflash.cpp:67-107): MLA-style stages with sqrt-softplus MoE,
     * uncompressed attention and a uniform sliding window on every block.
     */
    private static void dflashDsv4(ModelHparamsBuilder b, GgufKv kv) {
        int n = b.nLayerAll();
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                 // dflash.cpp:68
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                // dflash.cpp:69
        HparamsSupport.readExpertFfLength(b, kv, true);                            // dflash.cpp:70
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));            // dflash.cpp:71
        b.expertWeightsScale(kv.f32Required(GgufKeys.EXPERT_WEIGHTS_SCALE));      // dflash.cpp:72
        b.expertWeightsNorm(SpecialHooks.boolRequired(kv, GgufKeys.EXPERT_WEIGHTS_NORM)); // dflash.cpp:73
        if (!kv.has(GgufKeys.EXPERT_GATING_FUNC)) {                                // dflash.cpp:74
            throw SpecialHooks.missingKey(kv, GgufKeys.EXPERT_GATING_FUNC);
        }
        GatingFunc gating = HparamsSupport.readGating(kv, null);
        if (gating != GatingFunc.SQRT_SOFTPLUS) {                                  // dflash.cpp:87-89
            throw SpecialHooks.invalid(b, "DSpark DSV4 draft expects sqrtsoftplus MoE scoring");
        }
        b.gating(gating);
        if (!kv.has(GgufKeys.SWIGLU_CLAMP_EXP)) {                                  // dflash.cpp:75
            throw SpecialHooks.missingKey(kv, GgufKeys.SWIGLU_CLAMP_EXP);
        }
        float[] clampExp = kv.keyOrArrF32(GgufKeys.SWIGLU_CLAMP_EXP, n, 0.0f);
        b.swigluClampExpArray(clampExp);
        b.swigluClampShexpArray(kv.has(GgufKeys.SWIGLU_CLAMP_SHEXP)             // dflash.cpp:76-78
                ? kv.keyOrArrF32(GgufKeys.SWIGLU_CLAMP_SHEXP, n, 0.0f) : clampExp);
        b.dsv4OGroupCount(kv.u32Required(GgufKeys.ATTENTION_OUTPUT_GROUP_COUNT)); // dflash.cpp:79
        b.dsv4OLoraRank(kv.u32Required(GgufKeys.ATTENTION_OUTPUT_LORA_RANK));    // dflash.cpp:80
        b.dsv4HcSinkhornIters(kv.u32Required(GgufKeys.HYPER_CONNECTION_SINKHORN_ITERATIONS)); // dflash.cpp:81
        b.dsv4HcEps(kv.f32Required(GgufKeys.HYPER_CONNECTION_EPSILON));          // dflash.cpp:82
        requireUncompressed(b, kv);                                                // dflash.cpp:83, 90-94
        if (b.dsv4OGroupCount() <= 0) {                                            // dflash.cpp:85
            throw SpecialHooks.invalid(b, "attention.output_group_count must be positive");
        }
        if (b.nSwa() <= 0) {                                                       // dflash.cpp:96
            throw SpecialHooks.invalid(b, "DSpark DSV4 draft needs a positive sliding window");
        }
        b.swaType(SwaType.STANDARD);                                               // dflash.cpp:97
        boolean[] swa = new boolean[n];                                            // dflash.cpp:98-101
        Arrays.fill(swa, true);
        b.isSwaArray(swa);
        b.ropeFreqBaseSwa(b.ropeFreqBase());                                       // dflash.cpp:102
        b.ropeFreqScaleSwa(b.ropeFreqScale());                                     // dflash.cpp:103
    }

    /** {@code attention.compress_ratios} (optional) must be zero on every block (dflash.cpp:83, 90-94). */
    private static void requireUncompressed(ModelHparamsBuilder b, GgufKv kv) {
        int[] ratios = kv.u32Array(GgufKeys.ATTENTION_COMPRESS_RATIOS);
        if (ratios == null) {
            return;
        }
        for (int il = 0; il < Math.min(ratios.length, b.nLayerAll()); il++) {
            if (ratios[il] != 0) {
                throw SpecialHooks.invalid(b, "DSpark DSV4 draft expects uncompressed attention on all stages,"
                        + " block " + il + " has compress ratio " + ratios[il]);
            }
        }
    }
}
