package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooks;
import com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirements;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.arch.ArchTableMoe.decoder;

/**
 * Rows of {@link ArchTableMoe} for the MoE architectures with plain multi-head attention and no sliding
 * window.
 *
 * <p>RoPE types follow {@code llama_model_rope_type} (llama-model.cpp:3476-3674, each entry citing its line);
 * norm kind, FFN activation, post-norm role and residual wiring follow each graph (src/models/&lt;file&gt;.cpp).
 * The router function, weight normalization and weight scale the graph uses are resolved per file by
 * {@link MoeHooks}; {@code defaultGating} records the function llama.cpp assumes when the file does not name
 * one. Tensor lists come from {@link MoeTensorRequirements}.
 */
public final class ArchTableMoeClassic {

    private ArchTableMoeClassic() {
    }

    /** The rows. */
    public static List<ArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // llama-model.cpp:3505. Dense FFN without experts, Mixtral-style MoE with them.
                ArchEntry.of(decoder(Arch.LLAMA, RopeType.NORM).build(),
                        MoeHooks::llama, MoeTensorRequirements.llama()),
                // llama-model.cpp:3507. Chunked attention (8192) on 3 of 4 layers, NoPE + temperature on the 4th; MoE
                // every interleave step with a shared expert; weightless QK norm except on 128-expert files.
                ArchEntry.of(decoder(Arch.LLAMA4, RopeType.NORM).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooks::llama4, MoeTensorRequirements.llama4()),
                // llama-model.cpp:3560. GELU experts; attn_output_norm and the block output norm wrap both branches.
                ArchEntry.of(decoder(Arch.GROK, RopeType.NEOX).ffnAct(ActOp.GELU_TANH)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        MoeHooks::grok, MoeTensorRequirements.grok()),
                // llama-model.cpp:3561. LayerNorm without bias; attn_output_norm is the pre-FFN norm.
                ArchEntry.of(decoder(Arch.DBRX, RopeType.NEOX).normKind(NormKind.LAYER)
                                .postNormRole(PostNormRole.PRE_FFN).build(),
                        MoeHooks::dbrx, MoeTensorRequirements.dbrx()),
                // llama-model.cpp:3573. Shared expert gated by sigmoid(ffn_gate_inp_shexp).
                ArchEntry.of(decoder(Arch.QWEN2MOE, RopeType.NEOX).build(),
                        MoeHooks::qwen2moe, MoeTensorRequirements.qwen2moe()),
                // llama-model.cpp:3653-3659. Interleaved M-RoPE over the four position sections.
                ArchEntry.of(decoder(Arch.QWEN3VLMOE, RopeType.IMROPE).build(),
                        MoeHooks::qwen3vlmoe, MoeTensorRequirements.qwen3vlmoe()),
                // llama-model.cpp:3584. phi3 graph (models.h): RMS norms with bias tensors, LongRoPE factors.
                ArchEntry.of(decoder(Arch.PHIMOE, RopeType.NEOX).graphFamily(Arch.PHI3).build(),
                        MoeHooks::phimoe, MoeTensorRequirements.phimoe()),
                // llama-model.cpp:3581. Full-width QK norm; unnormalized softmax routing.
                ArchEntry.of(decoder(Arch.OLMOE, RopeType.NEOX).build(),
                        MoeHooks::olmoe, MoeTensorRequirements.olmoe()),
                // llama-model.cpp:3518. Dense FFN residual in parallel with the expert branch (own norm ffn_norm_exps).
                ArchEntry.of(decoder(Arch.ARCTIC, RopeType.NORM).build(),
                        MoeHooks::arctic, MoeTensorRequirements.arctic()),
                // llama-model.cpp:3519. Dense lead, routed plus shared experts.
                ArchEntry.of(decoder(Arch.DEEPSEEK, RopeType.NORM).build(),
                        MoeHooks::deepseek, MoeTensorRequirements.deepseek()),
                // llama-model.cpp:3529. granite graph (models.h): scaled residuals, logits divided by logit_scale.
                ArchEntry.of(decoder(Arch.GRANITE_MOE, RopeType.NORM).residualKind(ResidualKind.SCALED)
                                .graphFamily(Arch.GRANITE).build(),
                        MoeHooks::graniteMoe, MoeTensorRequirements.graniteMoe()),
                // llama-model.cpp:3534. Softmax routing, weight normalization from the file.
                ArchEntry.of(decoder(Arch.BAILINGMOE, RopeType.NORM).build(),
                        MoeHooks::bailingmoe, MoeTensorRequirements.bailingmoe()),
                // llama-model.cpp:3607. Grouped, biased routing (gating named by the file); partial NEOX rotation.
                ArchEntry.of(decoder(Arch.BAILINGMOE2, RopeType.NEOX).build(),
                        MoeHooks::bailingmoe2, MoeTensorRequirements.bailingmoe2()),
                // llama-model.cpp:3608. Dense lead, biased routed plus shared experts (gating named by the file).
                ArchEntry.of(decoder(Arch.DOTS1, RopeType.NEOX).build(),
                        MoeHooks::dots1, MoeTensorRequirements.dots1()),
                // llama-model.cpp:3539. ernie4-5-moe graph; MoE on layers il >= dense lead with (il + 1) % step == 0.
                ArchEntry.of(decoder(Arch.ERNIE4_5_MOE, RopeType.NORM).build(),
                        MoeHooks::ernie45Moe, MoeTensorRequirements.ernie45Moe()),
                // llama-model.cpp:3609. QK norms; shared expert on every layer.
                ArchEntry.of(decoder(Arch.HUNYUAN_MOE, RopeType.NEOX).build(),
                        MoeHooks::hunyuanMoe, MoeTensorRequirements.hunyuanMoe()),
                // llama-model.cpp:3613. A block is MoE when its ffn_gate_inp is present; sigmoid routing by default.
                ArchEntry.of(decoder(Arch.HY_V3, RopeType.NEOX).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooks::hyV3, MoeTensorRequirements.hyV3()),
                // llama-model.cpp:3618. Routed experts plus scaled chunk experts per expert group.
                ArchEntry.of(decoder(Arch.GROVEMOE, RopeType.NEOX).build(),
                        MoeHooks::grovemoe, MoeTensorRequirements.grovemoe()),
                // llama-model.cpp:3621. Full-width QK norm; biased routing (gating named by the file).
                ArchEntry.of(decoder(Arch.MINIMAX_M2, RopeType.NEOX).build(),
                        MoeHooks::minimaxM2, MoeTensorRequirements.minimaxM2()),
                // llama-model.cpp:3632. Block norms are RMS norms over attention.groupnorm_groups channel groups (1 =
                // plain RMS); MoE from the dense lead, value experts (MoVA) on MoE layers; sigmoid routing by default.
                ArchEntry.of(decoder(Arch.K2_HORIZON, RopeType.NEOX).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooks::k2Horizon, MoeTensorRequirements.k2Horizon())));
    }
}
