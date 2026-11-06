package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksMla;
import com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsMla;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.arch.ArchTableMoe.decoder;

/**
 * Rows of {@link ArchTableMoe} for the multi-head latent attention (MLA), DeepSeek sparse attention (DSA),
 * DeepSeek-V4 and MiniMax sparse attention architectures. Conventions as in {@link ArchTableMoeClassic};
 * hooks in {@link MoeHooksMla}, tensor lists in {@link MoeTensorRequirementsMla}. Whether a file uses MLA
 * is a hparam ({@code key_length_mla}/{@code value_length_mla}), not a memory kind.
 */
public final class ArchTableMoeMla {

    private ArchTableMoeMla() {
    }

    /** The rows. */
    public static List<ArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // llama-model.cpp:3606. Non-absorbed MLA with a dense FFN.
                ArchEntry.of(decoder(Arch.MINICPM3, RopeType.NEOX).residualKind(ResidualKind.SCALED).build(),
                        MoeHooksMla::minicpm3, MoeTensorRequirementsMla.minicpm3()),
                // llama-model.cpp:3520. MLA (absorbed when k_b/v_b are split), YaRN, grouped and biased routing.
                ArchEntry.of(decoder(Arch.DEEPSEEK2, RopeType.NORM).build(),
                        MoeHooksMla::deepseek2, MoeTensorRequirementsMla.deepseek2()),
                // vLLM's AXK1ForCausalLM (AXK1.py): the deepseek2 MLA/MoE graph with a post-MLP norm on the
                // sparse layers and the llama 4 query temperature. NF-defined because no llama.cpp converter or
                // graph carries that arrangement.
                ArchEntry.of(decoder(Arch.AXK1, RopeType.NORM).graphFamily(Arch.DEEPSEEK2)
                                .defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksMla::axk1, MoeTensorRequirementsMla.axk1()),
                // llama-model.cpp:3521. deepseek2 graph (models.h) with plain multi-head attention.
                ArchEntry.of(decoder(Arch.DEEPSEEK2OCR, RopeType.NORM).graphFamily(Arch.DEEPSEEK2).build(),
                        MoeHooksMla::deepseek2ocr, MoeTensorRequirementsMla.deepseek2ocr()),
                // llama-model.cpp:3522. MLA plus the lightning indexer selecting top-k keys.
                ArchEntry.of(decoder(Arch.DEEPSEEK32, RopeType.NORM, MemoryKind.DSA).build(),
                        MoeHooksMla::deepseek32, MoeTensorRequirementsMla.deepseek32()),
                // llama-model.cpp:3523. Compressed KV caches with indexer, hyper-connections, hash routing on the first
                // layers and sqrt(softplus) scoring.
                ArchEntry.of(decoder(Arch.DEEPSEEK4, RopeType.NORM, MemoryKind.DSV4)
                                .residualKind(ResidualKind.HYPER_CONNECTIONS)
                                .defaultGating(GatingFunc.SQRT_SOFTPLUS).build(),
                        MoeHooksMla::deepseek4, MoeTensorRequirementsMla.deepseek4()),
                // models/deepseek41.cpp: the DeepSeek-V4 attention reads other layers' compressed caches
                // (attention.kv_source_layers / indexer.source_layers), attends over the window plus the indexer's
                // top-k compressed rows, picks candidate blocks in two levels and adds the Engram n-gram layers.
                ArchEntry.of(decoder(Arch.DEEPSEEK41, RopeType.NORM, MemoryKind.DSV4)
                                .residualKind(ResidualKind.HYPER_CONNECTIONS)
                                .defaultGating(GatingFunc.SQRT_SOFTPLUS).build(),
                        MoeHooksMla::deepseek41, MoeTensorRequirementsMla.deepseek41()),
                // llama-model.cpp:3663-3664: MROPE when the file carries rope sections, NEOX otherwise.
                // post_attention_norm is the pre-FFN norm.
                ArchEntry.of(decoder(Arch.GLM4_MOE, RopeType.NEOX).mropeWhenSections(true)
                                .postNormRole(PostNormRole.PRE_FFN).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksMla::glm4Moe, MoeTensorRequirementsMla.glm4Moe()),
                // llama-model.cpp:3545. MLA plus a DSA indexer on full-indexer layers; shared layers reuse the top-k.
                ArchEntry.of(decoder(Arch.GLM_DSA, RopeType.NORM, MemoryKind.DSA)
                                .defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksMla::glmDsa, MoeTensorRequirementsMla.glmDsa()),
                // llama-model.cpp:3526. MLA without Q LoRA; ungated squared-ReLU FFN (models/plm.cpp:180).
                ArchEntry.of(decoder(Arch.PLM, RopeType.NORM).ffnAct(ActOp.RELU_SQR).ffnGating(FfnGating.SEQ).build(),
                        MoeHooksMla::plm, MoeTensorRequirementsMla.plm()),
                // llama-model.cpp:3600. MLA with the SSMax temperature, LeRoPE factors, per-layer rope patterns and
                // KV source maps; the DSA indexer runs on the full-indexer layers only.
                ArchEntry.of(decoder(Arch.MISTRAL_LARGE4, RopeType.NORM, MemoryKind.DSA).build(),
                        MoeHooksMla::mistralLarge4, MoeTensorRequirementsMla.mistralLarge4()),
                // llama-model.cpp:3542. Loads and runs as deepseek2 (models.h).
                ArchEntry.of(decoder(Arch.MISTRAL4, RopeType.NORM).graphFamily(Arch.DEEPSEEK2).build(),
                        MoeHooksMla::deepseek2, MoeTensorRequirementsMla.deepseek2()),
                // llama-model.cpp:3554-3555. MLA with sinks and output gate, hyper-connections, optional DSA indexer.
                ArchEntry.of(decoder(Arch.HY_V4, RopeType.NORM, MemoryKind.DSA)
                                .residualKind(ResidualKind.HYPER_CONNECTIONS).build(),
                        MoeHooksMla::hyV4, MoeTensorRequirementsMla.hyV4()),
                // llama-model.cpp:3546. MLA with separate sliding-window ranks, interleaved SWA, DSA indexer.
                ArchEntry.of(decoder(Arch.DOTS3NOTE, RopeType.NORM, MemoryKind.DSA_ISWA).build(),
                        MoeHooksMla::dots3note, MoeTensorRequirementsMla.dots3note()),
                // llama-model.cpp:3622. Block-sparse MiniMax attention; experts use the clamped OpenAI SwiGLU variant
                // (LLM_FFN_SWIGLU_OAI_MOE, models/minimax-m3.cpp:568), resolved by the MoE graph.
                ArchEntry.of(decoder(Arch.MINIMAX_M3, RopeType.NEOX, MemoryKind.MSA).build(),
                        MoeHooksMla::minimaxM3, MoeTensorRequirementsMla.minimaxM3()),
                // vLLM's longcat_flash.py: one logical block is two physical blocks of MLA attention and a dense MLP,
                // the first also carrying the shortcut-connected MoE with its identity (zero-compute) experts.
                ArchEntry.of(decoder(Arch.LONGCAT_FLASH, RopeType.NORM).build(),
                        MoeHooksMla::longcatFlash, MoeTensorRequirementsMla.longcatFlash()),
                // vLLM's longcat_flash_ngram.py: the same model, with an n-gram input embedding fused with the token
                // embedding instead of the plain token lookup.
                ArchEntry.of(decoder(Arch.LONGCAT_FLASH_NGRAM, RopeType.NORM).build(),
                        MoeHooksMla::longcatFlash, MoeTensorRequirementsMla.longcatNgram())));
    }
}
