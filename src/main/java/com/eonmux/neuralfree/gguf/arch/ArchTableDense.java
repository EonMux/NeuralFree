package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.hparams.hooks.DenseHooks;
import com.eonmux.neuralfree.gguf.hparams.hooks.DenseHooksModern;
import com.eonmux.neuralfree.gguf.hparams.hooks.DenseHooksSwa;
import com.eonmux.neuralfree.gguf.spec.req.DenseTensorRequirementsClassic;
import com.eonmux.neuralfree.gguf.spec.req.DenseTensorRequirementsLlama;
import com.eonmux.neuralfree.gguf.spec.req.DenseTensorRequirementsSandwich;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Architecture table for the plain decoder attention architectures: 62 generative
 * decoders with a standard KV cache, in llama.cpp {@code llm_arch} order. {@link ArchRegistry} merges this
 * table with the other tables; an architecture appears in exactly one table.
 *
 * <p>Traits record the static facts of each llama.cpp graph (src/models/&lt;arch&gt;.cpp): RoPE type
 * ({@code llama_model_rope_type}, llama-model.cpp:3476-3674), norm kind, FFN activation and gating from the
 * {@code build_ffn} call, post-norm role and residual wiring. File-dependent values (eps, SWA, scales) are
 * resolved by the hooks in {@link DenseHooks}, {@link DenseHooksSwa} and {@link DenseHooksModern}; the tensor
 * lists come from {@link DenseTensorRequirementsClassic}, {@link DenseTensorRequirementsLlama} and
 * {@link DenseTensorRequirementsSandwich}.
 */
public final class ArchTableDense {

    private ArchTableDense() {
    }

    /** The table rows. */
    public static List<ArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // Per-layer attention-free and linear-replacement blocks are resolved from n_head / n_head_kv.
                ArchEntry.of(decoder(Arch.DECI, RopeType.NORM).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.deci()),
                // Attention and MLP read the same normalized input and are summed (falcon.cpp:121-135).
                ArchEntry.of(layerNormGeluDecoder(Arch.FALCON, RopeType.NEOX).residualKind(ResidualKind.PARALLEL).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.falcon()),
                ArchEntry.of(layerNormGeluDecoder(Arch.GPT2, RopeType.NONE).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.gpt2()),
                // Parallel residual is per file: ModelHparams.useParallelResidual() (gptneox.cpp:143).
                ArchEntry.of(layerNormGeluDecoder(Arch.GPTNEOX, RopeType.NEOX).build(),
                        DenseHooks::gptneox, DenseTensorRequirementsClassic.gptneox()),
                ArchEntry.of(layerNormGeluDecoder(Arch.MPT, RopeType.NONE).build(),
                        DenseHooks::mpt, DenseTensorRequirementsClassic.mpt()),
                ArchEntry.of(decoder(Arch.BAICHUAN, RopeType.NORM).build(),
                        DenseHooks::baichuan, DenseTensorRequirementsClassic.llamaUntied()),
                ArchEntry.of(layerNormGeluDecoder(Arch.STARCODER, RopeType.NORM).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.starcoder()),
                ArchEntry.of(decoder(Arch.REFACT, RopeType.NONE).build(),
                        DenseHooks::refact, DenseTensorRequirementsLlama.llamaWithExperts()),
                ArchEntry.of(layerNormGeluDecoder(Arch.BLOOM, RopeType.NONE).build(),
                        DenseHooks::bloom, DenseTensorRequirementsClassic.bloom()),
                // Without ffn_norm (StableLM 2 12B) the MLP reads the attention norm output, a parallel
                // residual decided by tensor presence (stablelm.cpp:129-138).
                ArchEntry.of(decoder(Arch.STABLELM, RopeType.NEOX).normKind(NormKind.LAYER).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.stablelm()),
                ArchEntry.of(decoder(Arch.QWEN, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsClassic.qwen()),
                ArchEntry.of(decoder(Arch.QWEN2, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.qwen2()),
                ArchEntry.of(decoder(Arch.QWEN2VL, RopeType.MROPE).build(),
                        DenseHooksModern::qwen2vl, DenseTensorRequirementsLlama.qwen2()),
                // llama.cpp reports MROPE, but the graph rotates text with NEOX over head/2 and
                // feeds text positions [t, 0, 0, 0] (sensenova-u1.cpp:12-18).
                ArchEntry.of(decoder(Arch.SENSENOVA_U1, RopeType.NEOX).mropeTextPositions(MropeTextPositions.P000).build(),
                        DenseHooksModern::sensenovaU1, DenseTensorRequirementsLlama.llamaQkNorm()),
                ArchEntry.of(decoder(Arch.QWEN3, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.qwen3()),
                ArchEntry.of(decoder(Arch.QWEN3MOE, RopeType.NEOX).build(),
                        DenseHooksModern::qwen3moe, DenseTensorRequirementsLlama.qwen3moe()),
                ArchEntry.of(decoder(Arch.QWEN3VL, RopeType.IMROPE).build(),
                        DenseHooksModern::qwen3vl, DenseTensorRequirementsLlama.qwen3vl()),
                // Attention and MLP read the same normalized input and are summed (phi2.cpp:116-117).
                ArchEntry.of(layerNormGeluDecoder(Arch.PHI2, RopeType.NEOX).residualKind(ResidualKind.PARALLEL).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.phi2()),
                ArchEntry.of(decoder(Arch.PHI3, RopeType.NEOX).ffnGating(FfnGating.FUSED_UP).build(),
                        DenseHooks::phi3, DenseTensorRequirementsLlama.phi3()),
                // The MLP reuses the attention norm output; both branches are summed (plamo.cpp:111-112).
                ArchEntry.of(decoder(Arch.PLAMO, RopeType.NEOX).residualKind(ResidualKind.PARALLEL).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsClassic.plamo()),
                ArchEntry.of(decoder(Arch.PLAMO3, RopeType.NEOX).ffnGating(FfnGating.FUSED_UP)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        DenseHooks::plamo3, DenseTensorRequirementsSandwich.plamo3()),
                ArchEntry.of(layerNormGeluDecoder(Arch.CODESHELL, RopeType.NEOX).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.codeshell()),
                ArchEntry.of(decoder(Arch.ORION, RopeType.NEOX).normKind(NormKind.LAYER).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.orion()),
                ArchEntry.of(decoder(Arch.INTERNLM2, RopeType.NORM).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsClassic.llamaUntied()),
                // Runs the Granite graph (models.h:1885): scaled residual branches and divided logits.
                ArchEntry.of(decoder(Arch.MINICPM, RopeType.NORM).residualKind(ResidualKind.SCALED)
                                .graphFamily(Arch.GRANITE).build(),
                        DenseHooksModern::minicpm, DenseTensorRequirementsLlama.llamaWithExperts()),
                ArchEntry.of(decoder(Arch.GEMMA, RopeType.NEOX).ffnAct(ActOp.GELU_TANH).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsSandwich.gemma()),
                ArchEntry.of(decoder(Arch.GEMMA2, RopeType.NEOX).ffnAct(ActOp.GELU_TANH)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        DenseHooksSwa::gemma2, DenseTensorRequirementsSandwich.gemma2()),
                ArchEntry.of(decoder(Arch.GEMMA3, RopeType.NEOX).ffnAct(ActOp.GELU_TANH)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        DenseHooksSwa::gemma3, DenseTensorRequirementsSandwich.gemma3()),
                ArchEntry.of(decoder(Arch.GEMMA3N, RopeType.NEOX).ffnAct(ActOp.GELU_TANH)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        DenseHooksSwa::gemma3n, DenseTensorRequirementsSandwich.gemma3n()),
                ArchEntry.of(layerNormGeluDecoder(Arch.STARCODER2, RopeType.NEOX).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.starcoder2()),
                ArchEntry.of(decoder(Arch.XVERSE, RopeType.NORM).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsClassic.llamaUntied()),
                // The MLP reuses the attention norm output; both branches are summed (command-r.cpp:106-119).
                ArchEntry.of(decoder(Arch.COMMAND_R, RopeType.NORM).normKind(NormKind.LAYER)
                                .residualKind(ResidualKind.PARALLEL).build(),
                        DenseHooks::commandR, DenseTensorRequirementsClassic.commandR()),
                ArchEntry.of(decoder(Arch.COHERE2, RopeType.NORM).normKind(NormKind.LAYER)
                                .residualKind(ResidualKind.PARALLEL).build(),
                        DenseHooksSwa::cohere2, DenseTensorRequirementsClassic.cohere2()),
                // Parameter-free LayerNorm: no norm tensors.
                ArchEntry.of(decoder(Arch.OLMO, RopeType.NORM).normKind(NormKind.LAYER).build(),
                        DenseHooks::olmo, DenseTensorRequirementsClassic.olmo()),
                // No pre-norms: norms apply to the branch outputs only.
                ArchEntry.of(decoder(Arch.OLMO2, RopeType.NEOX).postNormRole(PostNormRole.POST_ONLY).build(),
                        DenseHooksSwa::olmo2, DenseTensorRequirementsSandwich.olmo2()),
                ArchEntry.of(decoder(Arch.MUSE_GLIMMER, RopeType.NORM).postNormRole(PostNormRole.SANDWICH).build(),
                        DenseHooksSwa::museGlimmer, DenseTensorRequirementsSandwich.museGlimmer()),
                ArchEntry.of(decoder(Arch.OPENELM, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.openelm()),
                ArchEntry.of(decoder(Arch.CHATGLM, RopeType.NORM).ffnGating(FfnGating.FUSED_UP).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.chatglm()),
                // NORM, or MROPE when the file carries M-RoPE sections (llama-model.cpp:3661-3662).
                ArchEntry.of(decoder(Arch.GLM4, RopeType.NORM).mropeWhenSections(true).ffnGating(FfnGating.FUSED_UP)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        DenseHooksModern::glm4, DenseTensorRequirementsSandwich.glm4()),
                // The FFN is a gated SiLU MLP (bitnet.cpp:132).
                ArchEntry.of(decoder(Arch.BITNET, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsClassic.bitnet()),
                ArchEntry.of(decoder(Arch.JAIS, RopeType.NONE).normKind(NormKind.LAYER).build(),
                        DenseHooks::jais, DenseTensorRequirementsClassic.jais()),
                ArchEntry.of(decoder(Arch.JAIS2, RopeType.NEOX).normKind(NormKind.LAYER).ffnAct(ActOp.RELU_SQR)
                                .ffnGating(FfnGating.SEQ).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.jais2()),
                ArchEntry.of(decoder(Arch.NEMOTRON, RopeType.NEOX).normKind(NormKind.LAYER).ffnAct(ActOp.RELU_SQR)
                                .ffnGating(FfnGating.SEQ).build(),
                        DenseHooks::layerNormEps, DenseTensorRequirementsClassic.nemotron()),
                ArchEntry.of(decoder(Arch.EXAONE, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.exaone()),
                // No pre-norms: norms apply to the branch outputs only.
                ArchEntry.of(decoder(Arch.EXAONE4, RopeType.NEOX).postNormRole(PostNormRole.POST_ONLY).build(),
                        DenseHooksSwa::exaone4, DenseTensorRequirementsSandwich.exaone4()),
                // Scaled residual branches and divided logits (granite.cpp:248-249, 308-309).
                ArchEntry.of(decoder(Arch.GRANITE, RopeType.NORM).residualKind(ResidualKind.SCALED).build(),
                        DenseHooksModern::granite, DenseTensorRequirementsLlama.llamaWithExperts()),
                ArchEntry.of(decoder(Arch.GRANITE_SWITCH, RopeType.NORM).residualKind(ResidualKind.SCALED).build(),
                        DenseHooksModern::graniteSwitch, DenseTensorRequirementsLlama.graniteSwitch()),
                ArchEntry.of(decoder(Arch.CHAMELEON, RopeType.NORM).build(),
                        DenseHooks::chameleon, DenseTensorRequirementsClassic.chameleon()),
                ArchEntry.of(decoder(Arch.ARCEE, RopeType.NORM).ffnAct(ActOp.RELU_SQR).ffnGating(FfnGating.SEQ).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.arcee()),
                ArchEntry.of(decoder(Arch.ERNIE4_5, RopeType.NORM).build(),
                        DenseHooksModern::ernie45, DenseTensorRequirementsLlama.ernie45()),
                // Reuses the hunyuan_vl loader and graph (models.h:2260-2264); always NEOX, M-RoPE sections are ignored.
                ArchEntry.of(decoder(Arch.HUNYUAN_DENSE, RopeType.NEOX).graphFamily(Arch.HUNYUAN_VL).build(),
                        DenseHooksModern::hunyuanVl, DenseTensorRequirementsLlama.llamaQkNorm()),
                // NEOX, or MROPE when the file carries M-RoPE sections (llama-model.cpp:3665-3666).
                ArchEntry.of(decoder(Arch.HUNYUAN_VL, RopeType.NEOX).mropeWhenSections(true).build(),
                        DenseHooksModern::hunyuanVl, DenseTensorRequirementsLlama.llamaQkNorm()),
                ArchEntry.of(decoder(Arch.SMOLLM3, RopeType.NORM).build(),
                        DenseHooksModern::smollm3, DenseTensorRequirementsLlama.llamaTied()),
                // post_attention_norm is the pre-FFN norm; there is no ffn_norm.
                ArchEntry.of(decoder(Arch.SEED_OSS, RopeType.NEOX).postNormRole(PostNormRole.PRE_FFN).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.seedOss()),
                ArchEntry.of(decoder(Arch.APERTUS, RopeType.NEOX).ffnAct(ActOp.XIELU).ffnGating(FfnGating.SEQ).build(),
                        DenseHooksModern::apertus, DenseTensorRequirementsLlama.apertus()),
                // The vis_* expert weights serve image tokens only.
                ArchEntry.of(decoder(Arch.COGVLM, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.cogvlm()),
                ArchEntry.of(decoder(Arch.PANGU_EMBED, RopeType.NEOX).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.panguEmbedded()),
                ArchEntry.of(decoder(Arch.MISTRAL3, RopeType.NORM).build(),
                        DenseHooksModern::mistral3, DenseTensorRequirementsLlama.mistral3()),
                // Reuses the ernie4_5 loader (models.h:2143-2145) with its own MROPE graph.
                ArchEntry.of(decoder(Arch.PADDLEOCR, RopeType.MROPE).build(),
                        DenseHooksModern::ernie45, DenseTensorRequirementsLlama.ernie45()),
                ArchEntry.of(decoder(Arch.SPARK2_5, RopeType.NEOX).ffnAct(ActOp.GELU_TANH).build(),
                        DenseHooksSwa::spark25, DenseTensorRequirementsLlama.spark25()),
                ArchEntry.of(decoder(Arch.MAINCODER, RopeType.NORM).build(),
                        DenseHooks::rmsEps, DenseTensorRequirementsLlama.llamaQkNorm()),
                // Weightless norms, embedding skip scaled by layer_output_scale, multiplied logits.
                ArchEntry.of(decoder(Arch.TALKIE, RopeType.NEOX).build(),
                        DenseHooks::talkie, DenseTensorRequirementsLlama.talkie())));
    }

    /**
     * Traits builder of a supported causal decoder with a standard KV cache; defaults RMSNorm, SiLU, parallel
     * gate/up FFN, no post-norm, standard residual.
     */
    private static ArchTraits.Builder decoder(Arch arch, RopeType ropeType) {
        return ArchTraits.builder(arch)
                .ropeType(ropeType)
                .memoryKind(MemoryKind.KV)
                .support(SupportClass.GEN);
    }

    /** {@link #decoder} with LayerNorm blocks and an ungated tanh-GELU MLP (GPT-2 style). */
    private static ArchTraits.Builder layerNormGeluDecoder(Arch arch, RopeType ropeType) {
        return decoder(arch, ropeType)
                .normKind(NormKind.LAYER)
                .ffnAct(ActOp.GELU_TANH)
                .ffnGating(FfnGating.SEQ);
    }
}
