package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.hooks.SpecialHooksDraft;
import com.eonmux.neuralfree.gguf.hparams.hooks.SpecialHooksGen;
import com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirementsDraft;
import com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirementsGen;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Masked-diffusion, looped, audio and drafter rows of {@link ArchTableSpecial}. Rope types follow llama.cpp
 * {@code llama_model_rope_type} (llama-model.cpp:3476-3667); the diffusion models run without memory
 * (llama-model.cpp:2800-2803) and the others with an attention KV cache.
 */
final class ArchTableSpecialGen {

    private ArchTableSpecialGen() {
    }

    /** The rows; {@code bases} holds the other tables' rows for qwen3tts. */
    static List<ArchEntry> rows(Map<Arch, ArchEntry> bases) {
        return Arrays.asList(
                ArchEntry.of(diffusion(Arch.DREAM, RopeType.NEOX).build(),
                        SpecialHooksGen.DREAM, SpecialTensorRequirementsGen.dream()),
                ArchEntry.of(diffusion(Arch.LLADA, RopeType.NORM).build(),
                        SpecialHooksGen.LLADA, SpecialTensorRequirementsGen.llada()),
                // llada-moe.cpp:126-136, rnd1.cpp:139-149: softmax routing
                ArchEntry.of(diffusion(Arch.LLADA_MOE, RopeType.NEOX).defaultGating(GatingFunc.SOFTMAX).build(),
                        SpecialHooksGen.LLADA_MOE, SpecialTensorRequirementsGen.lladaMoe()),
                ArchEntry.of(diffusion(Arch.RND1, RopeType.NEOX).defaultGating(GatingFunc.SOFTMAX).build(),
                        SpecialHooksGen.RND1, SpecialTensorRequirementsGen.rnd1()),
                // hrm-text.cpp:107-162: weightless RMS norms around each sub-block
                ArchEntry.of(kv(Arch.HRM_TEXT, RopeType.NEOX).support(SupportClass.GEN).build(),
                        SpecialHooksGen.HRM_TEXT, SpecialTensorRequirementsGen.hrmText()),
                ArchEntry.of(kv(Arch.NANBEIGE, RopeType.NORM).support(SupportClass.GEN).build(),
                        SpecialHooksGen.NANBEIGE, SpecialTensorRequirementsGen.nanbeige()),
                // pockettts.cpp:69-119: LayerNorm, ungated GELU FFN
                ArchEntry.of(kv(Arch.POCKETTTS, RopeType.NORM).normKind(NormKind.LAYER).ffnAct(ActOp.GELU_TANH)
                                .ffnGating(FfnGating.SEQ).support(SupportClass.AUDIO).build(),
                        SpecialHooksGen.POCKETTTS, SpecialTensorRequirementsGen.pocketTts()),
                // wavtokenizer-dec.cpp:144-253: group norms in the posnet, LayerNorm and GELU in the ConvNeXt blocks
                ArchEntry.of(ArchTraits.builder(Arch.WAVTOKENIZER_DEC).ropeType(RopeType.NONE)
                                .memoryKind(MemoryKind.NONE).normKind(NormKind.LAYER).ffnAct(ActOp.GELU_TANH)
                                .ffnGating(FfnGating.SEQ).support(SupportClass.AUDIO).build(),
                        SpecialHooksGen.WAVTOKENIZER_DEC, SpecialTensorRequirementsGen.wavTokenizerDec()),
                // models.h:735-737: qwen3vl hparams, tensors and graph
                ArchTableSpecial.derived(bases, kv(Arch.QWEN3TTS, RopeType.IMROPE).support(SupportClass.AUDIO)
                                .graphFamily(Arch.QWEN3VL).build(),
                        Arch.QWEN3VL, ArchHparamsHook.NONE, Collections.<TensorRequirement>emptyList()),
                ArchEntry.of(kv(Arch.EAGLE3, RopeType.NORM).support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.EAGLE3, SpecialTensorRequirementsDraft.eagle3()),
                // gemma4-assistant.cpp:143-181: GELU-gated FFN with sandwich norms
                ArchEntry.of(kv(Arch.GEMMA4_ASSISTANT, RopeType.NEOX).ffnAct(ActOp.GELU_TANH)
                                .postNormRole(PostNormRole.SANDWICH).support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.GEMMA4_ASSISTANT, SpecialTensorRequirementsDraft.gemma4Assistant()),
                // HparamsLoader resolves the dflash rope type from rope.dimension_sections and
                // hyper_connection.count (llama-model.cpp:3640-3646); DSpark stages route with sqrt-softplus
                ArchEntry.of(kv(Arch.DFLASH, RopeType.NEOX).defaultGating(GatingFunc.SQRT_SOFTPLUS)
                                .support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.DFLASH, SpecialTensorRequirementsDraft.dflash()));
    }

    /** Masked-diffusion model: bidirectional regardless of metadata, no memory. */
    private static ArchTraits.Builder diffusion(Arch arch, RopeType rope) {
        return ArchTraits.builder(arch).ropeType(rope).memoryKind(MemoryKind.NONE).forcedNonCausal(true)
                .support(SupportClass.DIFF);
    }

    /** Causal model with an attention KV cache. */
    private static ArchTraits.Builder kv(Arch arch, RopeType rope) {
        return ArchTraits.builder(arch).ropeType(rope).memoryKind(MemoryKind.KV);
    }
}
