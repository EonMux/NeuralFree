package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEagle3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEagle3MlaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEagle3Qwen3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMistralNativeHooks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the Llama family (llama.cpp's LlamaModel, GraniteModel and GraniteMoeModel),
 * Mistral native checkpoints and EAGLE3 drafts. {@link HfArchRegistry} merges this table with the other tables; a
 * Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableLlama {

    /** The names llama.cpp registers for {@code LlamaModel} (LC/conversion/llama.py:17-30). */
    static final List<String> LLAMA_NAMES = Collections.unmodifiableList(Arrays.asList(
            "IQuestCoderForCausalLM", "LLaMAForCausalLM", "LlamaForCausalLM", "LlamaModel", "Eagle3DraftModel",
            "Eagle3Speculator", "Eagle3LlamaForCausalLM", "LlamaForCausalLMEagle3", "LlavaForConditionalGeneration",
            "MistralForCausalLM", "MixtralForCausalLM", "VLlama3ForCausalLM", "VoxtralForConditionalGeneration"));

    /** The Qwen3 EAGLE-3 names vLLM's {@code qwen3_eagle3} module serves (registry.py:715-717). */
    static final List<String> EAGLE3_QWEN3_NAMES = Collections.unmodifiableList(Arrays.asList(
            "Eagle3Qwen3ForCausalLM", "PeagleQwen3ForCausalLM"));

    private HfArchTableLlama() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        List<String> llama = new ArrayList<>(LLAMA_NAMES);
        llama.addAll(HfEagle3Hooks.VLLM_ALIASES);
        return Collections.unmodifiableList(Arrays.asList(
                HfArchEntry.of(llama, HfLlamaHooks::arch, HfLlamaHooks::meta, HfLlamaHooks::tensors,
                                HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                // the Qwen3 EAGLE-3 layer: the same eagle3 row with Qwen3's projection norms and per-tap encoder
                // norms; the Llama extras add the llama3 rope factors of a draft that declares them
                HfArchEntry.of(EAGLE3_QWEN3_NAMES, HfArchResolver.of(Arch.EAGLE3), HfEagle3Qwen3Hooks::meta,
                                HfEagle3Qwen3Hooks::tensors, HfEagle3Qwen3Hooks::vocab)
                        .withConfigFixup(HfEagle3Qwen3Hooks::fixup)
                        .withLoadNameFilter(HfEagle3Qwen3Hooks::filter)
                        .withExtra(HfLlamaHooks::extra),
                // the DeepSeek EAGLE-3 layer: the same eagle3 row with multi-head latent attention and the
                // DeepSeek MLA geometry; its tensor set replaces the GQA projections with the MLA ones
                HfArchEntry.of(HfEagle3MlaHooks.NAMES, HfArchResolver.of(Arch.EAGLE3), HfEagle3MlaHooks::meta,
                                HfEagle3MlaHooks::tensors, HfEagle3MlaHooks::vocab)
                        .withConfigFixup(HfEagle3MlaHooks::fixup)
                        .withLoadNameFilter(HfEagle3MlaHooks::filter),
                HfArchEntry.of(Collections.singletonList(HfConfigLoader.MISTRAL_NATIVE), HfMistralNativeHooks::arch,
                                HfMistralNativeHooks::meta, HfMistralNativeHooks::tensors, HfMistralNativeHooks::vocab)
                        .withIndex(HfMistralNativeHooks::index)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfMistralNativeHooks::extra)
                        .withUndoPermute(false)));
    }
}
