package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hooks of the masked-diffusion (dream, llada, llada-moe, rnd1), looped (hrm_text, nanbeige) and audio
 * (pockettts, wavtokenizer-dec) architectures. Each transcribes {@code llama_model_<arch>::load_arch_hparams};
 * line references are to llama.cpp src/models/.
 *
 * <p>The diffusion mask token is a vocabulary key ({@code tokenizer.ggml.mask_token_id}) and
 * {@code diffusion.shift_logits} is read by {@link com.eonmux.neuralfree.gguf.hparams.HparamsLoader} for every
 * diffusion architecture.
 */
public final class SpecialHooksGen {

    /** Posnet blocks wavtokenizer-dec knows: resnet 0, 1, 3, 4, attention 2, norm 5 (wavtokenizer-dec.cpp:34-73). */
    private static final int WAVTOKENIZER_MAX_POSNET_LAYERS = 6;

    public static final ArchHparamsHook DREAM = SpecialHooksGen::denseDiffusion;
    public static final ArchHparamsHook LLADA = SpecialHooksGen::denseDiffusion;
    public static final ArchHparamsHook LLADA_MOE = SpecialHooksGen::lladaMoe;
    public static final ArchHparamsHook RND1 = SpecialHooksGen::rnd1;
    public static final ArchHparamsHook HRM_TEXT = SpecialHooksGen::hrmText;
    public static final ArchHparamsHook NANBEIGE = SpecialHooksGen::nanbeige;
    public static final ArchHparamsHook POCKETTTS = SpecialHooksGen::pocketTts;
    public static final ArchHparamsHook WAVTOKENIZER_DEC = SpecialHooksGen::wavTokenizerDec;

    private SpecialHooksGen() {
    }

    /** dream.cpp:3-16 and llada.cpp:3-17: RMS epsilon and bidirectional attention. */
    private static void denseDiffusion(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // dream.cpp:4, llada.cpp:4
        b.causal(false);                                                           // dream.cpp:15, llada.cpp:16
    }

    /** llada-moe.cpp:3-14; softmax routing with unnormalized weights (llada-moe.cpp:126-136). */
    private static void lladaMoe(ModelHparamsBuilder b, GgufKv kv) {
        moeDiffusion(b, kv);
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(false);                    // llada-moe.cpp:133-135
    }

    /** rnd1.cpp:3-14; softmax routing with normalized weights (rnd1.cpp:139-149). */
    private static void rnd1(ModelHparamsBuilder b, GgufKv kv) {
        moeDiffusion(b, kv);
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                     // rnd1.cpp:146-148
    }

    /**
     * Keys shared by llada-moe.cpp:3-14 and rnd1.cpp:3-14, plus the expert checks of llada-moe.cpp:25-26 and
     * rnd1.cpp:44-49. When the layer-0 expert width {@code n_ff_exp()} is zero (key absent or zero) the experts
     * are {@code n_ff / n_expert_used} wide, the width their tensors are created with (llada-moe.cpp:42,
     * rnd1.cpp:52).
     */
    private static void moeDiffusion(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, false);                           // llada-moe.cpp:4, rnd1.cpp:4
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // llada-moe.cpp:5, rnd1.cpp:5
        b.causal(false);                                                           // llada-moe.cpp:8, rnd1.cpp:13
        if (b.nExpert() <= 0 || b.nExpertUsed(0) <= 0) {    // llada-moe.cpp:25-26, rnd1.cpp:44-49
            throw SpecialHooks.invalid(b, "expert_count and expert_used_count must be positive, got "
                    + b.nExpert() + " and " + b.nExpertUsed(0));
        }
        if (b.nFfExp(0) == 0) {
            int nFfExp = b.nFf(0) / b.nExpertUsed(0);
            for (int il = 0; il < b.nLayerAll(); il++) {
                b.nFfExp(il, nFfExp);
            }
        }
    }

    /**
     * hrm-text.cpp:6-32: low/high stacks of {@code layers_per_stack} blocks; the block count is the expanded
     * cache-slot count {@code layers_per_stack * h_cycles * (l_cycles + 1)}.
     */
    private static void hrmText(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // hrm-text.cpp:7
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));   // hrm-text.cpp:8
        b.hrmLayersPerStack(kv.u32Required(GgufKeys.HRM_LAYERS_PER_STACK));       // hrm-text.cpp:10
        b.hrmHCycles(kv.u32Required(GgufKeys.HRM_H_CYCLES));                       // hrm-text.cpp:11
        b.hrmLCycles(kv.u32Required(GgufKeys.HRM_L_CYCLES));                       // hrm-text.cpp:12
        b.hrmPrefixLm(kv.bool(GgufKeys.HRM_PREFIX_LM, false));                     // hrm-text.cpp:15
        if (b.hrmLayersPerStack() <= 0 || b.hrmHCycles() <= 0 || b.hrmLCycles() <= 0) {  // hrm-text.cpp:17-19
            throw SpecialHooks.invalid(b, "hrm layers_per_stack, h_cycles and l_cycles must be positive, got "
                    + b.hrmLayersPerStack() + ", " + b.hrmHCycles() + ", " + b.hrmLCycles());
        }
        long slots = (long) b.hrmLayersPerStack() * b.hrmHCycles() * (b.hrmLCycles() + 1L);  // hrm-text.cpp:22
        if (b.nLayer() != slots) {                                                 // hrm-text.cpp:23
            throw SpecialHooks.invalid(b, "block count " + b.nLayer() + " must equal layers_per_stack * h_cycles"
                    + " * (l_cycles + 1) = " + slots);
        }
    }

    /**
     * nanbeige.cpp:3-35: {@code num_loops} passes over the physical blocks. With more than one loop the block
     * count becomes {@code n_phys * num_loops} and each looped slot copies the head, FFN, SWA and recurrent
     * layout of its physical block; the slots alias the physical weights.
     */
    private static void nanbeige(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // nanbeige.cpp:4
        long loops = kv.u32Long(GgufKeys.NUM_LOOPS, 1);                            // nanbeige.cpp:6-7
        if (loops < 1) {                                                           // nanbeige.cpp:8
            throw SpecialHooks.invalid(b, "num_loops must be at least 1, got " + loops);
        }
        b.skipLoopFinalNorm(kv.bool(GgufKeys.SKIP_LOOP_FINAL_NORM, false));       // nanbeige.cpp:10-11
        int phys = b.nLayer();                                                     // nanbeige.cpp:13
        if (phys * loops > ModelHparams.MAX_LAYERS) {                             // nanbeige.cpp:16
            throw SpecialHooks.invalid(b, phys + " blocks * " + loops + " loops exceed the maximum of "
                    + ModelHparams.MAX_LAYERS + " layers");
        }
        int nLoops = (int) loops;
        b.numLoops(nLoops);                                                        // nanbeige.cpp:17
        if (nLoops > 1) {                                                          // nanbeige.cpp:20-32
            int[] nHead = b.nHeadArray();
            int[] nHeadKv = b.nHeadKvArray();
            int[] nFf = b.nFfArray();
            boolean[] swa = b.isSwaArray();
            boolean[] recurrent = b.isRecurrentArray();
            b.layerCounts(phys * nLoops, b.nextnLayers());
            for (int j = 1; j < nLoops; j++) {
                for (int i = 0; i < phys; i++) {
                    int dst = i + j * phys;
                    b.nHead(dst, nHead[i]);
                    b.nHeadKv(dst, nHeadKv[i]);
                    b.nFf(dst, nFf[i]);
                    b.isSwa(dst, swa[i]);
                    b.isRecurrent(dst, recurrent[i]);
                }
            }
        }
    }

    /** pockettts.cpp:6-14. */
    private static void pocketTts(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // pockettts.cpp:7
    }

    /**
     * wavtokenizer-dec.cpp:3-7. The posnet/ConvNeXt dimensions are read by the loader; the posnet layout and
     * width checks come from wavtokenizer-dec.cpp:73 and 78.
     */
    private static void wavTokenizerDec(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // wavtokenizer-dec.cpp:4
        b.normGroupEps(kv.f32Required(GgufKeys.ATTENTION_GROUPNORM_EPS));        // wavtokenizer-dec.cpp:5
        b.nNormGroups(kv.u32Required(GgufKeys.ATTENTION_GROUPNORM_GROUPS));      // wavtokenizer-dec.cpp:6
        if (b.posnetLayers() > WAVTOKENIZER_MAX_POSNET_LAYERS) {                  // wavtokenizer-dec.cpp:73
            throw SpecialHooks.invalid(b, "unknown posnet layer: " + b.posnetLayers() + " posnet blocks, at most "
                    + WAVTOKENIZER_MAX_POSNET_LAYERS + " are defined");
        }
        if (b.posnetEmbd() != b.convnextEmbd()) {                                  // wavtokenizer-dec.cpp:78
            throw SpecialHooks.invalid(b, "posnet width " + b.posnetEmbd() + " must equal convnext width "
                    + b.convnextEmbd());
        }
    }
}
