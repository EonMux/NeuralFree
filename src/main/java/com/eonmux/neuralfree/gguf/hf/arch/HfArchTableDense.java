package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseHooksC;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGemmaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLimiteHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMistral3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenMoeHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the modern dense decoder families (the converter classes of mistral3.py,
 * gemma.py, olmo.py, llama.py's SmolLM3/Arcee/Apertus, hunyuan.py, qwen.py, qwenvl.py, qwen3vl.py, step3.py,
 * ernie.py, phi.py, sensenova_u1.py and limite.py). {@link HfArchRegistry} merges this table with the others; a
 * Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}. The hook classes of the Qwen rows
 * (Qwen2, Qwen3, Qwen3Moe) and of the Gemma3, Ernie4_5 and Phi3Mini rows compose the rows their converters inherit;
 * the classes of this table only extend them.
 */
final class HfArchTableDense {

    private HfArchTableDense() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                HfArchEntry.of(Collections.singletonList("NEOChatModel"), HfArchResolver.of(Arch.SENSENOVA_U1),
                                HfDenseHooksB::sensenovaMeta, HfDenseHooksB::sensenovaTensors,
                                HfQwenHooks::qwen2Vocab)
                        .withNameFilter(HfDenseHooksB.SENSENOVA_FILTER),
                HfArchEntry.of(Collections.singletonList("ApertusForCausalLM"), HfArchResolver.of(Arch.APERTUS),
                                HfLlamaHooks::meta, HfDenseHooksC::apertusTensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra)
                        .withUndoPermute(false),
                HfArchEntry.of(Collections.singletonList("ArceeForCausalLM"), HfArchResolver.of(Arch.ARCEE),
                                HfDenseHooksC::arceeMeta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.of(Arrays.asList("AudioFlamingo3ForConditionalGeneration", "DotsOCRForCausalLM",
                                "KORMoForCausalLM", "Qwen2AudioForConditionalGeneration", "Qwen2ForCausalLM",
                                "Qwen2Model"), HfArchResolver.of(Arch.QWEN2),
                                HfQwenHooks::qwen2Meta, HfQwenHooks::qwen2Tensors, HfQwenHooks::qwen2Vocab),
                HfArchEntry.of(Arrays.asList("Ernie4_5ForCausalLM", "Ernie4_5_ForCausalLM"),
                                HfArchResolver.of(Arch.ERNIE4_5),
                                HfMetadataHook.GENERIC, HfBaseDenseHooks::ernie45Tensors,
                                HfBaseDenseHooks::ernie45Vocab)
                        .withNameFilter(HfBaseDenseHooks::ernie45Filter),
                HfArchEntry.of(Collections.singletonList("Gemma2ForCausalLM"), HfArchResolver.of(Arch.GEMMA2),
                                HfDenseHooksA::gemma2Meta, HfDenseHooksA::gemmaTensors,
                                HfDenseHooksA::gemma2Vocab)
                        .withNameFilter(HfDenseHooksA.GEMMA_FILTER),
                HfArchEntry.of(Arrays.asList("Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"),
                                HfArchResolver.of(Arch.GEMMA3),
                                HfGemmaHooks::meta, HfGemmaHooks::tensors, HfGemmaHooks::vocab),
                HfArchEntry.of(Arrays.asList("Gemma3nForCausalLM", "Gemma3nForConditionalGeneration"),
                                HfArchResolver.of(Arch.GEMMA3N),
                                HfDenseHooksA::gemma3nMeta, HfDenseHooksA::gemma3nTensors,
                                HfDenseHooksA::gemma3nVocab)
                        .withNameFilter(HfDenseHooksA.GEMMA3N_FILTER),
                HfArchEntry.of(Collections.singletonList("GemmaForCausalLM"), HfArchResolver.of(Arch.GEMMA),
                                HfDenseHooksA::gemmaMeta, HfDenseHooksA::gemmaTensors, HfDenseHooksA::gemmaVocab)
                        .withNameFilter(HfDenseHooksA.GEMMA_FILTER),
                HfArchEntry.of(Collections.singletonList("HunYuanDenseV1ForCausalLM"),
                                HfArchResolver.of(Arch.HUNYUAN_DENSE),
                                HfDenseHooksC::hunyuanMeta, HfDenseHooksC::hunyuanTensors,
                                HfDenseHooksC::hunyuanVocab),
                HfArchEntry.of(Collections.singletonList("HunYuanVLForConditionalGeneration"),
                                HfArchResolver.of(Arch.HUNYUAN_VL),
                                HfDenseHooksC::hunyuanVlMeta, HfDenseHooksC::hunyuanTensors,
                                HfDenseHooksC::hunyuanVocab)
                        .withConfigFixup(HfDenseHooksC.HUNYUAN_VL_FIXUP),
                HfArchEntry.of(Arrays.asList("Ministral3ForCausalLM", "Ministral3Model",
                                "Mistral3ForConditionalGeneration", "Mistral3Model"),
                                HfMistral3Hooks::arch, HfMistral3Hooks::meta, HfMistral3Hooks::tensors,
                                HfMistral3Hooks::vocab),
                HfArchEntry.of(Collections.singletonList("VoxtralRealtimeForConditionalGeneration"),
                                HfArchResolver.of(Arch.MISTRAL3),
                                HfMistral3Hooks::voxtralMeta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.of(Arrays.asList("OLMoForCausalLM", "OlmoForCausalLM"), HfArchResolver.of(Arch.OLMO),
                                HfDenseHooksC::olmoMeta, HfDenseHooksC::olmoTensors,
                                HfDenseHooksC::gpt2Vocab),
                HfArchEntry.of(Arrays.asList("Olmo2ForCausalLM", "Olmo3ForCausalLM"), HfArchResolver.of(Arch.OLMO2),
                                HfDenseHooksC::olmo2Meta, HfTensorHook.DEFAULT, HfDenseHooksC::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("PaddleOCRVLForConditionalGeneration"),
                                HfArchResolver.of(Arch.PADDLEOCR),
                                HfMetadataHook.GENERIC, HfBaseDenseHooks::ernie45Tensors,
                                HfBaseDenseHooks::ernie45Vocab)
                        .withNameFilter(HfBaseDenseHooks::ernie45Filter),
                HfArchEntry.of(Arrays.asList("Phi3ForCausalLM", "Phi4ForCausalLMV"), HfArchResolver.of(Arch.PHI3),
                                HfBaseDenseHooks::phi3Meta, HfTensorHook.DEFAULT, HfBaseDenseHooks::phi3Vocab)
                        .withExtra(HfBaseDenseHooks::phi3Extra),
                HfArchEntry.of(Collections.singletonList("QWenLMHeadModel"), HfArchResolver.of(Arch.QWEN),
                                HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, HfDenseHooksB::qwenVocab),
                HfArchEntry.of(Arrays.asList("Qwen2VLForConditionalGeneration", "Qwen2VLModel",
                                "Qwen2_5OmniModel", "Qwen2_5_VLForConditionalGeneration"),
                                HfArchResolver.of(Arch.QWEN2VL),
                                HfQwenHooks::qwen2Meta, HfQwenHooks::qwen2Tensors,
                                HfDenseHooksB::qwen2VlVocab)
                        .withNameFilter(HfDenseHooksB.THINKER_FILTER),
                HfArchEntry.of(Collections.singletonList("Qwen3ASRForConditionalGeneration"),
                                HfArchResolver.of(Arch.QWEN3VL),
                                HfDenseHooksB::asrMeta, HfQwenHooks::qwen3Tensors, HfDenseHooksB::asrVocab)
                        .withNameFilter(HfDenseHooksB.THINKER_FILTER),
                // LimiteModel (limite.py:15): value embeddings, the exclusive self attention, the learned residual
                // scales and the MUDD connections.
                HfArchEntry.of(Collections.singletonList("LimiteForCausalLM"), HfLimiteHooks::arch,
                                HfLimiteHooks::meta, HfLimiteHooks::tensors, HfLimiteHooks::vocab)
                        .withConfigFixup(HfLimiteHooks::fixup)
                        .withForceType(HfLimiteHooks::forceType),
                HfArchEntry.of(Arrays.asList("Qwen3ForCausalLM", "Qwen3Model"), HfArchResolver.of(Arch.QWEN3),
                                HfQwenHooks::qwen3Meta, HfQwenHooks::qwen3Tensors, HfQwenHooks::qwen3Vocab),
                // SopranoTextModel (soprano.py:37): the Qwen3 backbone of the Soprano TTS checkpoints, named by the
                // Soprano checkpoint detector; the decoder.pth Vocos weights belong to the projector file.
                HfArchEntry.of(Collections.singletonList("SopranoTextModel"), HfArchResolver.of(Arch.QWEN3),
                                HfQwenHooks::qwen3Meta, HfQwenHooks::qwen3Tensors, HfQwenHooks::sopranoVocab),
                HfArchEntry.of(Collections.singletonList("Qwen3MoeForCausalLM"), HfArchResolver.of(Arch.QWEN3MOE),
                                HfQwenMoeHooks::qwen2MoeMeta, HfQwenMoeHooks::qwen2MoeTensors,
                                HfQwenMoeHooks::qwen3MoeVocab)
                        .withExtra(HfQwenMoeHooks::qwen2MoeExtra),
                HfArchEntry.of(Collections.singletonList("Qwen3VLForConditionalGeneration"),
                                HfArchResolver.of(Arch.QWEN3VL),
                                HfDenseHooksB::qwen3VlMeta, HfQwenHooks::qwen3Tensors, HfQwenHooks::qwen3Vocab)
                        .withNameFilter(HfDenseHooksB.THINKER_FILTER),
                HfArchEntry.of(Collections.singletonList("SeedOssForCausalLM"), HfArchResolver.of(Arch.SEED_OSS),
                                HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, HfDenseHooksC::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("SmolLM3ForCausalLM"), HfArchResolver.of(Arch.SMOLLM3),
                                HfLlamaHooks::meta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.of(Collections.singletonList("StepVLForConditionalGeneration"),
                                HfArchResolver.of(Arch.QWEN3),
                                HfQwenHooks::qwen3Meta, HfQwenHooks::qwen3Tensors, HfQwenHooks::qwen3Vocab)));
    }
}
