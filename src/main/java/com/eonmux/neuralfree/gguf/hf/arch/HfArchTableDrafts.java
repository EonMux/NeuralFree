package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDraftHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEagle1Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEagleFcHooks;
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
 * <p>The table registers the EAGLE-1 classes with the fused {@code llama_eagle} layout of {@link Arch#EAGLE}
 * ({@code ArchTableDraft}, the GGUF-side row): {@code Eagle*ForCausalLM} covers vLLM's {@code EagleModel},
 * {@code EagleLlamaModel} ({@code EagleLlamaForCausalLM}, the llama target fused drafter) and
 * {@code EagleMistralForCausalLM} (the mistral target fused drafter, whose checkpoint adds the input norm and the
 * final norm) plus SGLang's {@code LlamaEagleForCausalLM}, and {@code EagleDeepSeekMTPModel} is the MTP-shaped
 * alias of the same single-drafter form: its DeepSeek checkpoints declare the MLA geometry ({@code kv_lora_rank},
 * {@code qk_nope_head_dim}, {@code qk_rope_head_dim}, {@code v_head_dim}) and convert through
 * {@link com.eonmux.neuralfree.gguf.hf.hooks.HfEagleMlaHooks} onto the same row. The concat-fc drafters vLLM serves
 * next to it register as their own architectures ({@link HfEagleFcHooks}, {@code ArchTableDraft}):
 * {@code EagleMistralLarge3ForCausalLM} over the DeepSeek MLA geometry, {@code EagleLlama4ForCausalLM} over chunked
 * local attention, {@code EagleMiniCPMForCausalLM} with its {@code scale_emb} / {@code scale_depth} /
 * {@code scale_width} scales, and {@code EagleCohereForCausalLM} with its parallel LayerNorm block. The remaining
 * vLLM draft classes stay refused with a named reason: the {@code Eagle3*}/{@code Peagle*} fusions of further target
 * families are EAGLE-3 layouts ({@code HfArchTableLlama} registers the EAGLE-3 ones the engine runs, including the
 * Qwen3 layout); the multi-beam and DSpark variants draft through engines NF has no drafter for.</p>
 */
final class HfArchTableDrafts {

    /** The EAGLE-1 classes of vLLM's {@code llama_eagle} and {@code mistral_eagle} registration and SGLang's port. */
    static final List<String> EAGLE1_NAMES = Arrays.asList("EagleForCausalLM", "EagleModel", "EagleLlamaModel",
            "LlamaEagleForCausalLM", "EagleLlamaForCausalLM", "EagleMistralForCausalLM", "EagleDeepSeekMTPModel");

    private HfArchTableDrafts() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(eagle1(), concatFc(
                        "EagleMistralLarge3ForCausalLM", Arch.EAGLE_MISTRAL_LARGE3,
                        HfEagleFcHooks::mistralLarge3Meta, HfEagleFcHooks::mistralLarge3Tensors),
                concatFc("EagleLlama4ForCausalLM", Arch.EAGLE_LLAMA4,
                        HfEagleFcHooks::llama4Meta, HfEagleFcHooks::llama4Tensors),
                concatFc("EagleMiniCPMForCausalLM", Arch.EAGLE_MINICPM,
                        HfEagleFcHooks::miniCpmMeta, HfEagleFcHooks::miniCpmTensors),
                concatFc("EagleCohereForCausalLM", Arch.EAGLE_COHERE,
                        HfEagleFcHooks::cohereMeta, HfEagleFcHooks::cohereTensors)));
    }

    /** The EAGLE-1 entry: the {@code Arch.EAGLE} row, its metadata, tensors and target-shared vocabulary. */
    private static HfArchEntry eagle1() {
        return HfArchEntry.of(EAGLE1_NAMES, HfArchResolver.of(Arch.EAGLE), HfEagle1Hooks::meta,
                        HfEagle1Hooks::tensors, HfEagle1Hooks::vocab)
                .withConfigFixup(HfEagle1Hooks::fixup)
                .withLoadNameFilter(HfEagle1Hooks::filter)
                .withExtra(HfEagle1Hooks::extra);
    }

    /**
     * A vLLM concat-fc draft: the architecture's {@code ArchTableDraft} row, the family's metadata and tensor
     * hooks, the target's tokenizer, and no Q/K permutation (each family's layout is the one its GGUF row reads; the
     * MiniCPM tensors carry their own).
     */
    private static HfArchEntry concatFc(String name, Arch arch, HfMetadataHook meta, HfTensorHook tensors) {
        return HfArchEntry.of(Collections.singletonList(name), HfArchResolver.of(arch), meta, tensors,
                        HfEagleFcHooks::vocab)
                .withUndoPermute(false);
    }
}
