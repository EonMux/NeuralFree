package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSwa;
import com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSwa;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.arch.ArchTableMoe.decoder;

/**
 * Rows of {@link ArchTableMoe} for the sliding-window MoE architectures. Conventions as in
 * {@link ArchTableMoeClassic}; the SWA type, window and layer pattern are resolved per file by
 * {@link MoeHooksSwa} (including archs whose SWA depends on a positive {@code sliding_window}), tensor lists
 * come from {@link MoeTensorRequirementsSwa}.
 */
public final class ArchTableMoeSwa {

    private ArchTableMoeSwa() {
    }

    /** The rows. */
    public static List<ArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // llama-model.cpp:3576. Explicit SWA and RoPE layer arrays, attention sinks, biased routing.
                ArchEntry.of(decoder(Arch.FLINT, RopeType.NEOX).build(),
                        MoeHooksSwa::flint, MoeTensorRequirementsSwa.flint()),
                // llama-model.cpp:3592. GELU FFN and experts, sandwich norms, per-layer embeddings, KV sharing.
                ArchEntry.of(decoder(Arch.GEMMA4, RopeType.NEOX).ffnAct(ActOp.GELU_TANH)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        MoeHooksSwa::gemma4, MoeTensorRequirementsSwa.gemma4()),
                // llama-model.cpp:3634. RoPE only on SWA layers; clamped routed-expert SwiGLU.
                ArchEntry.of(decoder(Arch.MAPLE, RopeType.NEOX).build(),
                        MoeHooksSwa::maple, MoeTensorRequirementsSwa.maple()),
                // llama-model.cpp:3516. Attention and MoE read the same norm output (parallel residual). The block norm
                // is LayerNorm when the file has no RMS epsilon (stored as 0), RMS otherwise
                // (models/cohere2moe.cpp:149).
                ArchEntry.of(decoder(Arch.COHERE2MOE, RopeType.NORM).normKind(NormKind.LAYER)
                                .residualKind(ResidualKind.PARALLEL).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksSwa::cohere2moe, MoeTensorRequirementsSwa.cohere2moe()),
                // llama-model.cpp:3525. Gated attention, learned residual alpha/beta mixing, latent MoE; RoPE on SWA
                // layers.
                ArchEntry.of(decoder(Arch.MUSE_SPARK, RopeType.NORM).postNormRole(PostNormRole.SANDWICH).build(),
                        MoeHooksSwa::museSpark, MoeTensorRequirementsSwa.museSpark()),
                // llama-model.cpp:3605. RoPE only on local (SWA) layers; dense lead and NextN blocks are dense.
                ArchEntry.of(decoder(Arch.EXAONE_MOE, RopeType.NEOX).build(),
                        MoeHooksSwa::exaoneMoe, MoeTensorRequirementsSwa.exaoneMoe()),
                // llama-model.cpp:3532. granite-style scaled residuals and divided logits; sinks; RoPE per
                // rope_pattern.
                ArchEntry.of(decoder(Arch.GRANITE_SWA, RopeType.NORM).residualKind(ResidualKind.SCALED).build(),
                        MoeHooksSwa::graniteSwa, MoeTensorRequirementsSwa.graniteSwa()),
                // llama-model.cpp:3625. Sandwich norms, gated attention, NoPE every 4th layer.
                ArchEntry.of(decoder(Arch.AFMOE, RopeType.NEOX).postNormRole(PostNormRole.SANDWICH)
                                .defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksSwa::afmoe, MoeTensorRequirementsSwa.afmoe()),
                // llama-model.cpp:3626. Gated attention; YaRN on full layers, plain RoPE on SWA layers.
                ArchEntry.of(decoder(Arch.LAGUNA, RopeType.NEOX).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksSwa::laguna, MoeTensorRequirementsSwa.laguna()),
                // llama-model.cpp:3611. gpt-oss: sinks, post_attention_norm as the pre-FFN norm, softmax over the
                // selected router logits; experts use LLM_FFN_SWIGLU_OAI_MOE (models/openai-moe.cpp:139), resolved by
                // the MoE graph.
                ArchEntry.of(decoder(Arch.OPENAI_MOE, RopeType.NEOX).postNormRole(PostNormRole.PRE_FFN)
                                .defaultGating(GatingFunc.SOFTMAX_WEIGHT).build(),
                        MoeHooksSwa::gptOss, MoeTensorRequirementsSwa.gptOss()),
                // llama-model.cpp:3616. ReLU experts; the router reads the block input before the attention norm.
                ArchEntry.of(decoder(Arch.SMALLTHINKER, RopeType.NEOX).ffnAct(ActOp.RELU).build(),
                        MoeHooksSwa::smallthinker, MoeTensorRequirementsSwa.smallthinker()),
                // llama-model.cpp:3628. A block is MoE when its ffn_gate_inp is present; sigmoid routing.
                ArchEntry.of(decoder(Arch.MIMO2, RopeType.NEOX).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksSwa::mimo2, MoeTensorRequirementsSwa.mimo2()),
                // llama-model.cpp:3629. Half-rotary full layers, optional head-wise attention gate, clamped SwiGLU; a
                // block is MoE when its ffn_gate_inp is present.
                ArchEntry.of(decoder(Arch.STEP35, RopeType.NEOX).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksSwa::step35, MoeTensorRequirementsSwa.step35()),
                // llama-model.cpp:3593. Sandwich norms; latent MoE every interleave step; RoPE on SWA layers.
                ArchEntry.of(decoder(Arch.MAION_CODER, RopeType.NEOX).postNormRole(PostNormRole.SANDWICH).build(),
                        MoeHooksSwa::maionCoder, MoeTensorRequirementsSwa.maionCoder()),
                // llama-model.cpp:3633. Softmax routing by default; SWA layers use plain RoPE.
                ArchEntry.of(decoder(Arch.MELLUM, RopeType.NEOX).build(),
                        MoeHooksSwa::mellum, MoeTensorRequirementsSwa.mellum()),
                // llama-model.cpp:3636. Sandwich norms, biased per-head attention gate, NoPE full layers with power
                // temperature, depth-scaled residual adds.
                ArchEntry.of(decoder(Arch.BEAM, RopeType.NEOX).postNormRole(PostNormRole.SANDWICH)
                                .residualKind(ResidualKind.DEPTH_SCALED).defaultGating(GatingFunc.SIGMOID).build(),
                        MoeHooksSwa::beam, MoeTensorRequirementsSwa.beam()),
                // llama-model.cpp:3637. Sandwich norms; biased routing with one shared expert; RoPE on SWA layers.
                ArchEntry.of(decoder(Arch.KOLIBRI1, RopeType.NEOX).postNormRole(PostNormRole.SANDWICH).build(),
                        MoeHooksSwa::kolibri1, MoeTensorRequirementsSwa.kolibri1())));
    }
}
