package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.hooks.HybridHooksGdn;
import com.eonmux.neuralfree.gguf.hparams.hooks.HybridHooksKda;
import com.eonmux.neuralfree.gguf.hparams.hooks.HybridHooksLfm;
import com.eonmux.neuralfree.gguf.hparams.hooks.HybridHooksLightning;
import com.eonmux.neuralfree.gguf.hparams.hooks.HybridHooksRwkv;
import com.eonmux.neuralfree.gguf.hparams.hooks.HybridHooksSsm;
import com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirementsGdn;
import com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirementsKda;
import com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirementsRwkv;
import com.eonmux.neuralfree.gguf.spec.req.HybridTensorRequirementsSsm;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Architecture table for the recurrent and hybrid architectures: 23 generative decoders whose memory holds
 * recurrent states (Mamba SSM, RWKV, Gated DeltaNet, KDA, lightning attention, short convolution), alone or
 * next to an attention KV cache, in llama.cpp {@code llm_arch} order. {@link ArchRegistry} merges this table
 * with the other tables; an architecture appears in exactly one table.
 *
 * <p>Traits record the static facts of each llama.cpp graph (src/models/&lt;arch&gt;.cpp): RoPE type
 * ({@code llama_model_rope_type}, llama-model.cpp:3476-3674), memory ({@code llama_model::create_memory},
 * llama-model.cpp:2778-3160), norm kind, FFN activation and gating from the {@code build_ffn} call, post-norm
 * role and residual wiring. File-dependent values (SSM sizes, recurrent layer flags, GDN head map, KDA gate
 * form, token shift, short-convolution window) are resolved by the hooks in {@link HybridHooksSsm},
 * {@link HybridHooksRwkv}, {@link HybridHooksLfm}, {@link HybridHooksGdn}, {@link HybridHooksKda} and
 * {@link HybridHooksLightning}; the tensor lists come from {@link HybridTensorRequirementsSsm},
 * {@link HybridTensorRequirementsRwkv}, {@link HybridTensorRequirementsGdn} and
 * {@link HybridTensorRequirementsKda}.
 */
public final class ArchTableHybrid {

    private ArchTableHybrid() {
    }

    /** The table rows. */
    public static List<ArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // post_attention_norm is the pre-FFN norm; attention output is sigmoid-gated by the Q half.
                ArchEntry.of(qwenGdn(Arch.QWEN3NEXT, RopeType.NEOX).build(),
                        HybridHooksGdn::qwen3next, HybridTensorRequirementsGdn.qwen3next()),
                ArchEntry.of(qwenGdn(Arch.QWEN35, RopeType.IMROPE).build(),
                        HybridHooksGdn::qwen35, HybridTensorRequirementsGdn.qwen35()),
                ArchEntry.of(qwenGdn(Arch.QWEN35MOE, RopeType.IMROPE).build(),
                        HybridHooksGdn::qwen35moe, HybridTensorRequirementsGdn.qwen35moe()),
                // Hyper-connection mixers replace the residual adds and the block norms; the QSA indexer of the
                // attention layers needs the indexed hybrid memory (llama-model.cpp:3120, 3139-3143).
                ArchEntry.of(hybrid(Arch.QWEN4EXP, RopeType.IMROPE).memoryKind(MemoryKind.HYBRID_IDX)
                                .residualKind(ResidualKind.HYPER_CONNECTIONS).build(),
                        HybridHooksGdn::qwen4exp, HybridTensorRequirementsGdn.qwen4exp()),
                // Sandwich norms around both branches; ffn_up holds the fused SwiGLU gate and up halves
                // (plamo2.cpp:101, 168).
                ArchEntry.of(hybrid(Arch.PLAMO2, RopeType.NEOX).ffnGating(FfnGating.FUSED_UP)
                                .postNormRole(PostNormRole.SANDWICH).build(),
                        HybridHooksSsm::plamo2, HybridTensorRequirementsSsm.plamo2()),
                ArchEntry.of(recurrent(Arch.MAMBA).build(),
                        HybridHooksSsm::mamba, HybridTensorRequirementsSsm.mamba()),
                ArchEntry.of(recurrent(Arch.MAMBA2).graphFamily(Arch.MAMBA).build(),
                        HybridHooksSsm::mamba2, HybridTensorRequirementsSsm.mamba2()),
                ArchEntry.of(hybrid(Arch.JAMBA, RopeType.NONE).build(),
                        HybridHooksSsm::jamba, HybridTensorRequirementsSsm.jamba()),
                // Every block runs the Mamba-2 and attention mixers in parallel and keeps both memories
                // (llama-model.cpp:3121-3123).
                ArchEntry.of(hybrid(Arch.FALCON_H1, RopeType.NEOX).build(),
                        HybridHooksSsm::falconH1, HybridTensorRequirementsSsm.falconH1()),
                // Single-kind layers; the FFN layers run an ungated squared-ReLU MLP, routed experts use sigmoid
                // gating with a selection bias (nemotron-h.cpp:315-324).
                ArchEntry.of(nemotronH(Arch.NEMOTRON_H).build(),
                        HybridHooksSsm::nemotronH, HybridTensorRequirementsSsm.nemotronH()),
                ArchEntry.of(nemotronH(Arch.NEMOTRON_H_MOE).graphFamily(Arch.NEMOTRON_H).build(),
                        HybridHooksSsm::nemotronH, HybridTensorRequirementsSsm.nemotronH()),
                // LayerNorm throughout; the channel mix is an ungated squared-ReLU MLP behind a receptance gate.
                ArchEntry.of(rwkvChannelMix(Arch.RWKV6).build(),
                        HybridHooksRwkv::rwkv6, HybridTensorRequirementsRwkv.rwkv6()),
                ArchEntry.of(recurrent(Arch.RWKV6QWEN2).build(),
                        HybridHooksRwkv::rwkv6qwen2, HybridTensorRequirementsRwkv.rwkv6qwen2()),
                ArchEntry.of(rwkvChannelMix(Arch.RWKV7).build(),
                        HybridHooksRwkv::rwkv7, HybridTensorRequirementsRwkv.rwkv7()),
                ArchEntry.of(recurrent(Arch.ARWKV7).build(),
                        HybridHooksRwkv::arwkv7, HybridTensorRequirementsRwkv.arwkv7()),
                // Residual branches are scaled by residual_scale and logits divided by logit_scale
                // (granite-hybrid.cpp:194, 235-236, 290-291).
                ArchEntry.of(hybrid(Arch.GRANITE_HYBRID, RopeType.NORM).residualKind(ResidualKind.SCALED).build(),
                        HybridHooksSsm::graniteHybrid, HybridTensorRequirementsSsm.graniteHybrid()),
                // M-RoPE when the file has rope sections, NORM otherwise (llama-model.cpp:3550-3555).
                ArchEntry.of(hybrid(Arch.BAILINGMOE3, RopeType.NORM).mropeWhenSections(true)
                                .defaultGating(GatingFunc.SIGMOID).build(),
                        HybridHooksKda::bailingmoe3, HybridTensorRequirementsKda.bailingmoe3()),
                ArchEntry.of(hybrid(Arch.LFM2, RopeType.NEOX).build(),
                        HybridHooksLfm::lfm2, HybridTensorRequirementsRwkv.lfm2()),
                ArchEntry.of(hybrid(Arch.LFM2MOE, RopeType.NEOX).graphFamily(Arch.LFM2)
                                .defaultGating(GatingFunc.SIGMOID).build(),
                        HybridHooksLfm::lfm2moe, HybridTensorRequirementsRwkv.lfm2moe()),
                // Both residual branches add the normed block input scaled by residual_scale
                // (minimax-01.cpp:246-249, 428, 440, 455).
                ArchEntry.of(hybrid(Arch.MINIMAX_01, RopeType.NEOX).residualKind(ResidualKind.NORMED_INPUT_SCALED).build(),
                        HybridHooksLightning::minimax01, HybridTensorRequirementsGdn.minimax01()),
                ArchEntry.of(hybrid(Arch.KIMI_LINEAR, RopeType.NONE).defaultGating(GatingFunc.SIGMOID).build(),
                        HybridHooksKda::kimiLinear, HybridTensorRequirementsKda.kimiLinear()),
                // Cross-layer attention residuals (attn_res_block_size) and the SiTU GLU (situ_beta,
                // situ_linear_beta) are file-resolved hparams (kimi-k3.cpp:43-45, 590).
                ArchEntry.of(hybrid(Arch.KIMI_K3, RopeType.NONE).defaultGating(GatingFunc.SIGMOID).build(),
                        HybridHooksKda::kimiK3, HybridTensorRequirementsKda.kimiK3()),
                // KDA + k-pool DSA hybrid with mHC residual streams (llama-model.cpp:2883-2900).
                ArchEntry.of(hybrid(Arch.GLM5_NEXT, RopeType.NONE).memoryKind(MemoryKind.HYBRID_IDX)
                                .residualKind(ResidualKind.HYPER_CONNECTIONS).defaultGating(GatingFunc.SIGMOID).build(),
                        HybridHooksKda::glm5Next, HybridTensorRequirementsKda.glm5Next())));
    }

    /** RMS-normed SwiGLU decoder with recurrent and attention layers ({@link MemoryKind#HYBRID}). */
    private static ArchTraits.Builder hybrid(Arch arch, RopeType rope) {
        return ArchTraits.builder(arch)
                .support(SupportClass.GEN)
                .ropeType(rope)
                .memoryKind(MemoryKind.HYBRID)
                .normKind(NormKind.RMS)
                .ffnAct(ActOp.SILU)
                .ffnGating(FfnGating.PAR)
                .defaultGating(GatingFunc.SOFTMAX);
    }

    /** RMS-normed decoder whose every layer is recurrent ({@link MemoryKind#RECURRENT}, no RoPE). */
    private static ArchTraits.Builder recurrent(Arch arch) {
        return hybrid(arch, RopeType.NONE).memoryKind(MemoryKind.RECURRENT);
    }

    /** Qwen GDN hybrid: {@code post_attention_norm} is the pre-FFN norm. */
    private static ArchTraits.Builder qwenGdn(Arch arch, RopeType rope) {
        return hybrid(arch, rope).postNormRole(PostNormRole.PRE_FFN);
    }

    /** Nemotron-H: ungated squared-ReLU MLP layers and sigmoid-routed experts. */
    private static ArchTraits.Builder nemotronH(Arch arch) {
        return hybrid(arch, RopeType.NONE)
                .ffnAct(ActOp.RELU_SQR)
                .ffnGating(FfnGating.SEQ)
                .defaultGating(GatingFunc.SIGMOID);
    }

    /** RWKV with LayerNorm and the squared-ReLU channel-mix FFN. */
    private static ArchTraits.Builder rwkvChannelMix(Arch arch) {
        return recurrent(arch)
                .normKind(NormKind.LAYER)
                .ffnAct(ActOp.RELU_SQR)
                .ffnGating(FfnGating.SEQ);
    }
}
