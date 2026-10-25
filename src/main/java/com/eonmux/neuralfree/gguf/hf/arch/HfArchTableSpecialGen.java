package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekV4Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDraftHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDraftHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDiffusionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGemma4Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGenHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenMoeHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries of Appendix A table HK-SG: the draft checkpoints (LC/conversion/qwen.py
 * {@code DFlashModel} and {@code DSparkModel}, LC/conversion/nemotron.py {@code LiLiCorrDraftModel},
 * LC/conversion/gemma.py {@code Gemma4DSparkModel} and {@code Gemma4AssistantModel},
 * LC/conversion/muse_glimmer.py {@code MuseGlimmerAssistantModel}, LC/conversion/deepseek.py
 * {@code DeepseekV4DSparkModel}), the masked-diffusion and recurrent-depth models (LC/conversion/dream.py,
 * llada.py, qwen.py {@code RND1Model}, hrm_text.py, nanbeige.py) and the audio generators (LC/conversion/
 * pockettts.py, qwen3tts.py, wavtokenizer.py). {@link HfArchRegistry} merges this table with the other tables; a
 * Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableSpecialGen {

    private HfArchTableSpecialGen() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                dflash("DFlashDraftModel", "DFlash2DraftModel"),
                dspark("Qwen3DSparkModel", "DSparkDraftModel", "DSparkSpeculator", "Lfm2DSparkDraftModel",
                        "LingDSparkModel"),
                HfArchEntry.of(Collections.singletonList("DeepseekV4DSparkModel"), HfArchResolver.of(Arch.DFLASH),
                        HfDraftHooksB::dsv4DsparkMeta, HfDraftHooksB::dsv4DsparkTensors,
                        HfDraftHooksB::dsv4DsparkVocab)
                        .withConfigFixup(HfDraftHooksB::dsv4DsparkFixup)
                        .withIndexedBlockCount(HfDraftHooksB::dsv4DsparkBlockCount)
                        .withForceType(HfDraftHooksB::dsv4DsparkForceType)
                        .withLoadNameFilter(HfDraftHooksB::dsv4DsparkFilter),
                lilicorr(),
                gemma4Dspark(),
                museGlimmerAssistant(),
                gemma4Assistant(),
                diffusion(),
                llada(),
                lladaMoe(),
                rnd1(),
                hrmText(),
                nanbeige(),
                pocketTts(),
                qwen3Tts(),
                wavTokenizerDec()));
    }

    private static HfArchEntry dflash(String... hfNames) {
        return HfArchEntry.of(Arrays.asList(hfNames), HfArchResolver.of(Arch.DFLASH), HfDraftHooksA::dflashMeta,
                HfDraftHooksA::dflashTensors, HfDraftHooksA::dflashVocab)
                .withLoadNameFilter(HfDraftHooksA::dflashFilter)
                .withExtra(HfDraftHooksA::dflashExtra);
    }

    private static HfArchEntry dspark(String... hfNames) {
        return dflash(hfNames).withLoadNameFilter(HfDraftHooksA::dsparkFilter)
                .withExtra(HfDraftHooksA::dsparkExtra);
    }

    private static HfArchEntry lilicorr() {
        return HfArchEntry.of(Collections.singletonList("LiLiCorrDraftModel"), HfArchResolver.of(Arch.DFLASH),
                HfDraftHooksB::lilicorrMeta, HfDraftHooksB::lilicorrTensors, HfDraftHooksB::lilicorrVocab)
                .withConfigFixup(HfDraftHooksB::lilicorrFixup)
                .withLoadNameFilter(HfDraftHooksB::lilicorrFilter)
                .withExtra(HfDraftHooksB::lilicorrExtra)
                .withForceType(HfDraftHooksB::lilicorrForceType)
                .withBlockCount(HfDraftHooksB::lilicorrBlockCount);
    }

    private static HfArchEntry gemma4Dspark() {
        return HfArchEntry.of(Collections.singletonList("Gemma4DSparkModel"), HfArchResolver.of(Arch.DFLASH),
                HfDraftHooksB::gemma4DsparkMeta, HfDraftHooksB::gemma4DsparkTensors,
                HfDraftHooksB::gemma4DsparkVocab)
                .withConfigFixup(HfDraftHooksB::gemma4DsparkFixup)
                .withLoadNameFilter(HfDraftHooksB::gemma4DsparkFilter)
                .withExtra(HfDraftHooksB::gemma4DsparkExtra);
    }

    private static HfArchEntry museGlimmerAssistant() {
        return HfArchEntry.of(Collections.singletonList("MuseGlimmerAssistantModel"), HfArchResolver.of(Arch.DFLASH),
                HfDraftHooksA::museGlimmerAssistantMeta, HfDraftHooksA::museGlimmerAssistantTensors,
                HfDraftHooksA::museGlimmerAssistantVocab);
    }

    /** {@code Gemma4AssistantModel} on the HK-B2 Gemma 4 hooks (LC/conversion/gemma.py:936-961). */
    private static HfArchEntry gemma4Assistant() {
        return HfArchEntry.of(Arrays.asList("Gemma4AssistantForCausalLM", "Gemma4UnifiedAssistantForCausalLM"),
                HfArchResolver.of(Arch.GEMMA4_ASSISTANT), HfDraftHooksB::gemma4AssistantMeta,
                HfDraftHooksB::gemma4AssistantTensors, HfGemma4Hooks::vocab)
                .withNameFilter(HfDraftHooksB::gemma4AssistantFilter)
                .withIndex(HfGemma4Hooks::index)
                .withExtra(HfGemma4Hooks::extra);
    }

    private static HfArchEntry diffusion() {
        return HfArchEntry.of(Collections.singletonList("DreamModel"), HfArchResolver.of(Arch.DREAM),
                HfDiffusionHooks::dreamMeta, HfDiffusionHooks::passThrough, HfDiffusionHooks::dreamVocab);
    }

    private static HfArchEntry llada() {
        return HfArchEntry.of(Collections.singletonList("LLaDAModelLM"), HfArchResolver.of(Arch.LLADA),
                HfDiffusionHooks::lladaMeta, HfDiffusionHooks::lladaTensors, HfDiffusionHooks::lladaVocab);
    }

    /** {@code LLaDAMoEModel} (llada.py:119): the MoE tensors are the copied Qwen2Moe expert stack (llada.py:152-190). */
    private static HfArchEntry lladaMoe() {
        return HfArchEntry.of(Arrays.asList("LLaDAMoEModel", "LLaDAMoEModelLM"), HfArchResolver.of(Arch.LLADA_MOE),
                HfDiffusionHooks::lladaMoeMeta, HfQwenMoeHooks::qwen2MoeTensors,
                HfQwenMoeHooks::qwen2MoeVocab);
    }

    private static HfArchEntry rnd1() {
        return HfArchEntry.of(Collections.singletonList("RND1"), HfArchResolver.of(Arch.RND1),
                HfDiffusionHooks::rnd1Meta, HfQwenMoeHooks::qwen2MoeTensors, HfQwenMoeHooks::qwen2MoeVocab)
                .withExtra(HfQwenMoeHooks::qwen2MoeExtra);
    }

    private static HfArchEntry hrmText() {
        return HfArchEntry.of(Collections.singletonList("HrmTextForCausalLM"), HfArchResolver.of(Arch.HRM_TEXT),
                HfDiffusionHooks::hrmTextMeta, HfDiffusionHooks::hrmTextTensors, HfDiffusionHooks::hrmTextVocab)
                .withBlockCount(HfDiffusionHooks::hrmTextBlockCount);
    }

    private static HfArchEntry nanbeige() {
        return HfArchEntry.of(Collections.singletonList("NanbeigeForCausalLM"), HfArchResolver.of(Arch.NANBEIGE),
                HfDiffusionHooks::nanbeigeMeta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                .withNameFilter(HfLlamaHooks.FILTER)
                .withExtra(HfLlamaHooks::extra);
    }

    private static HfArchEntry pocketTts() {
        return HfArchEntry.of(Collections.singletonList("PocketTTSModel"), HfArchResolver.of(Arch.POCKETTTS),
                HfGenHooks::pocketTtsMeta, HfGenHooks::pocketTtsTensors, HfGenHooks::pocketTtsVocab);
    }

    private static HfArchEntry qwen3Tts() {
        return HfArchEntry.of(Collections.singletonList("Qwen3TTSForConditionalGeneration"),
                HfArchResolver.of(Arch.QWEN3TTS), HfGenHooks::qwen3TtsMeta, HfGenHooks::qwen3TtsTensors,
                HfGenHooks::qwen3TtsVocab)
                .withLoadNameFilter(HfGenHooks::qwen3TtsFilter)
                .withConfigFixup(HfGenHooks::qwen3TtsFixup);
    }

    private static HfArchEntry wavTokenizerDec() {
        return HfArchEntry.of(Collections.singletonList("WavTokenizerDec"), HfArchResolver.of(Arch.WAVTOKENIZER_DEC),
                HfGenHooks::wavTokenizerDecMeta, HfDiffusionHooks::passThrough, HfGenHooks::wavTokenizerDecVocab)
                .withLoadNameFilter(HfGenHooks::wavTokenizerDecFilter);
    }
}
