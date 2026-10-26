package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDraftHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEagle1Hooks;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the standalone draft checkpoints the converter does not register (EAGLE-1,
 * standalone MTP and further DFlash and DSpark variants). {@link HfArchRegistry} merges this table with the other
 * tables; a Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 *
 * <p>SP-4a registers the EAGLE-1 classes with the fused {@code llama_eagle} layout of {@link Arch#EAGLE}
 * ({@code ArchTableDraft}, the GGUF-side row): {@code Eagle*ForCausalLM} covers vLLM's {@code EagleModel} and
 * {@code EagleLlamaModel} plus SGLang's {@code LlamaEagleForCausalLM}, and {@code EagleDeepSeekMTPModel} is the
 * MTP-shaped alias of the same single-drafter form. The remaining vLLM draft classes stay refused with the named
 * reason GAPS.md records: the {@code Eagle3*}/{@code Peagle*} fusions of further target families are
 * EAGLE-3 layouts ({@code HfArchTableLlama} registers the EAGLE-3 ones llama.cpp converts), and the multi-beam and
 * DSpark variants draft through engines NF has no drafter for.</p>
 */
final class HfArchTableDrafts {

    /** The EAGLE-1 classes of vLLM's {@code llama_eagle} registration and SGLang's port. */
    static final List<String> EAGLE1_NAMES = Arrays.asList("EagleForCausalLM", "EagleModel", "EagleLlamaModel",
            "LlamaEagleForCausalLM", "EagleDeepSeekMTPModel");

    private HfArchTableDrafts() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.singletonList(eagle1());
    }

    /** The EAGLE-1 entry: the {@code Arch.EAGLE} row, its metadata, tensors and target-shared vocabulary. */
    private static HfArchEntry eagle1() {
        return HfArchEntry.of(EAGLE1_NAMES, HfArchResolver.of(Arch.EAGLE), HfEagle1Hooks::meta,
                        HfEagle1Hooks::tensors, HfEagle1Hooks::vocab)
                .withConfigFixup(HfEagle1Hooks::fixup)
                .withLoadNameFilter(HfEagle1Hooks::filter);
    }
}
