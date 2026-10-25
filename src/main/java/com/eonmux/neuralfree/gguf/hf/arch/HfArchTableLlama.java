package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEagle3Hooks;
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
                HfArchEntry.of(Collections.singletonList(HfConfigLoader.MISTRAL_NATIVE), HfMistralNativeHooks::arch,
                                HfMistralNativeHooks::meta, HfMistralNativeHooks::tensors, HfMistralNativeHooks::vocab)
                        .withIndex(HfMistralNativeHooks::index)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfMistralNativeHooks::extra)
                        .withUndoPermute(false)));
    }
}
