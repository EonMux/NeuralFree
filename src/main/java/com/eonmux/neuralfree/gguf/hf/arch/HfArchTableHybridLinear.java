package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfAliceAiHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBailingV25Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBailingV3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDecisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfKdaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfKdaHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLinearHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLinearHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfOlmoHybridHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenLinearHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenMoeHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfRwkvHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries: the linear-attention hybrids and their
 * neighbours — KDA (Kimi-Linear, Kimi-K3 and AliceAI), Bailing-MoE V3 and V2.5, GLM-5 Next, MiniMax-Text-01, Qwen4Exp, the
 * MiniCPM-V 4.6/4.7 text models, the RWKV world formats, the Qwen 3 Next / 3.5 / 3.5 MoE rows and the
 * Qwen3.5 decision models (LC conversion classes bailingmoe3.py:19/:200, kimi_linear.py:17, kimi_k3.py:19,
 * glm.py:409, minimax.py:16, qwen4exp.py:18, minicpm.py:123/:211, rwkv.py:15/:88/:142/:268, pplx_decider.py:39,
 * vllm_d3.py:31, qwen.py:707, h2o_lightning.py:27, lev.py:96/:167/:227). The Qwen 3 Next / 3.5 / 3.5 MoE rows
 * build through the {@link HfQwenLinearHooks} statics read-only. {@link HfArchRegistry} merges this table
 * with the other tables; a Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableHybridLinear {

    private HfArchTableHybridLinear() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // AliceAIForCausalLM (vLLM models/alice_ai.py): KDA layers with the low-rank decay and output
                // gates alternating with Qwen3-Next gated attention, a sigmoid-routed MoE with a gated shared
                // expert and the cross-layer attention residuals grouped by block_attn_res_block_size.
                HfArchEntry.of(Collections.singletonList("AliceAIForCausalLM"), HfAliceAiHooks.arch(),
                                HfAliceAiHooks::meta, HfAliceAiHooks::tensors, HfAliceAiHooks::vocab)
                        .withExtra(HfAliceAiHooks::extra),
                // BailingMoeV3Model (bailingmoe3.py:19): KDA attention per layer, the MLA kv_b split and the
                // MTP export modes.
                HfArchEntry.of(Collections.singletonList("BailingMoeV3ForCausalLM"), HfBailingV3Hooks.arch(),
                                HfBailingV3Hooks::meta, HfBailingV3Hooks::tensors, HfBailingV3Hooks::vocab)
                        .withIndexedBlockCount(HfBailingV3Hooks::blockCount)
                        .withLoadNameFilter(HfBailingV3Hooks::filter)
                        .withExtra(HfBailingV3Hooks::extra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.BAILING),
                // BailingMoeV3VLModel (bailingmoe3.py:200): the same format behind the VL conditional-generation
                // wrapper, whose metadata reads the text config.
                HfArchEntry.of(Collections.singletonList("BailingMoeV3VLForConditionalGeneration"),
                                HfBailingV3Hooks.arch(), (ctx, kv, plan) -> HfBailingV3Hooks.vlMeta(ctx, kv),
                                HfBailingV3Hooks::tensors,
                                HfBailingV3Hooks::vocab)
                        .withIndexedBlockCount(HfBailingV3Hooks::blockCount)
                        .withLoadNameFilter(HfBailingV3Hooks::filter)
                        .withExtra(HfBailingV3Hooks::extra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.BAILING),
                // PplxDeciderModel (pplx_decider.py:39): the readout side file becomes the output layer and the
                // base Qwen3.5 MTP tensors are dropped.
                HfArchEntry.of(Collections.singletonList("PplxDeciderModel"), HfDecisionHooks.arch(),
                                HfDecisionHooks::pplxMeta, HfDecisionHooks::readoutTensors,
                                HfDecisionHooks::pplxVocab)
                        .withNameFilter(HfDecisionHooks.noMtpFilter())
                        .withExtra(HfDecisionHooks::pplxExtra),
                // VllmD3Model (vllm_d3.py:31): the same readout with the non-causal full attention of the
                // Decision 3.0 format, also under vLLM's name for it (d3.py D3ForSequenceClassification).
                HfArchEntry.of(Arrays.asList("VllmD3Model", HfDecisionHooks.D3_SEQUENCE_CLASSIFICATION),
                                HfDecisionHooks.arch(), HfDecisionHooks::vllmD3Meta, HfDecisionHooks::vllmD3Tensors,
                                HfDecisionHooks::vllmD3Vocab)
                        .withConfigFixup(HfDecisionHooks::vllmD3Fixup)
                        .withNameFilter(HfDecisionHooks.noMtpFilter())
                        .withExtra(HfDecisionHooks::vllmD3Extra),
                // OpenJevModel (qwen.py:707): the fixed openjev temperatures on the Qwen3.5 chain.
                HfArchEntry.of(Collections.singletonList("OpenJevModel"), HfDecisionHooks.arch(),
                                HfDecisionHooks::openjevMeta, HfDecisionHooks::openjevTensors,
                                HfDecisionHooks::openjevVocab)
                        .withNameFilter(HfDecisionHooks.noMtpFilter()),
                // H2OLightningModel (h2o_lightning.py:27): the served temperatures and labels of
                // serve_config.json on the Qwen3.5 chain.
                HfArchEntry.of(Collections.singletonList("H2OLightningModel"), HfDecisionHooks.arch(),
                                HfDecisionHooks::h2oMeta, HfDecisionHooks::h2oTensors,
                                HfDecisionHooks::h2oVocab)
                        .withNameFilter(HfDecisionHooks.noMtpFilter()),
                // MiniCPMOCRSeqClsModel (qwen.py:713): the Qwen3.5 trunk with the score.weight value head read at the
                // last token as the two class scores, and the fixed OCR consistency prompt.
                HfArchEntry.of(Collections.singletonList("MiniCPMOCRForSequenceClassification"),
                                HfDecisionHooks.arch(), HfDecisionHooks::miniCpmOcrJevMeta,
                                HfDecisionHooks::miniCpmOcrJevTensors, HfDecisionHooks::miniCpmOcrJevVocab)
                        .withNameFilter(HfDecisionHooks.noMtpFilter()),
                // LevModel (lev.py:96): the plain LoRA merges into the local base checkpoint at load.
                HfArchEntry.of(Collections.singletonList("LevModel"), HfDecisionHooks.arch(),
                                HfDecisionHooks::levMeta, HfDecisionHooks::loraTensors,
                                HfDecisionHooks::levVocab)
                        .withIndex(HfDecisionHooks::decisionLora)
                        .withNameFilter(HfDecisionHooks.noMtpFilter())
                        .withExtra(HfDecisionHooks::decisionLoraExtra),
                // KevModel (lev.py:167): the pointer head needs head.pt, a pickle NF refuses with the named
                // error; the entry still carries the merge hooks for the safetensors layout.
                HfArchEntry.of(Collections.singletonList("KevModel"), HfDecisionHooks.arch(),
                                HfDecisionHooks::kevMeta, HfDecisionHooks::loraTensors,
                                HfDecisionHooks::kevVocab)
                        .withIndex(HfDecisionHooks::decisionLora)
                        .withNameFilter(HfDecisionHooks.noMtpFilter())
                        .withExtra(HfDecisionHooks::decisionLoraExtra),
                // NimbleModel (lev.py:227): the plain LoRA merge without its own metadata.
                HfArchEntry.of(Collections.singletonList("NimbleModel"), HfDecisionHooks.arch(),
                                HfDecisionHooks::nimbleMeta, HfDecisionHooks::loraTensors,
                                HfDecisionHooks::nimbleVocab)
                        .withIndex(HfDecisionHooks::decisionLora)
                        .withNameFilter(HfDecisionHooks.noMtpFilter())
                        .withExtra(HfDecisionHooks::decisionLoraExtra),
                // Glm5NextModel (glm.py:409): the KDA attention with the nope-MLA, the DSA kpool and mHC,
                // with the small tensors forced F32; vLLM's text-only Glm5NextForCausalLM (models/glm5next) is the
                // same text model with a flat configuration and model.layers names.
                HfArchEntry.of(Arrays.asList("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"),
                                HfLinearHooksA.glmArch(), HfLinearHooksA::glmMeta, HfLinearHooksA::glmTensors,
                                HfLinearHooksA::glmVocab)
                        .withForceType(HfLinearHooksA::glmForceType)
                        .withExtra(HfLinearHooksA::glmExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // KimiK3Model (kimi_k3.py:19): the KDA geometry with the residual-score fusions and the bundled
                // Kimi-K3 chat template.
                HfArchEntry.of(Collections.singletonList("KimiK3ForConditionalGeneration"),
                                HfKdaHooksB.k3Arch(), HfKdaHooksB::k3Meta, HfKdaHooksB::k3Tensors,
                                HfKdaHooksB::k3Vocab)
                        .withExtra(HfKdaHooksB::k3Extra),
                // KimiLinearModel (kimi_linear.py:17): the base KDA geometry over full and partial attention.
                HfArchEntry.of(Arrays.asList("KimiLinearForCausalLM", "KimiLinearModel"), HfKdaHooks.linearArch(),
                                HfKdaHooks::linearMeta, HfKdaHooks::linearTensors, HfKdaHooks::linearVocab)
                        .withExtra(HfKdaHooks::linearExtra),
                // MiniCPMV4_6TextModel (minicpm.py:123): the Qwen3.5 text trunk of the MiniCPM-V wrapper.
                HfArchEntry.of(Collections.singletonList("MiniCPMV4_6ForConditionalGeneration"),
                                HfLinearHooksB.miniCpmV46Arch(), HfLinearHooksB::miniCpm4_6Meta,
                                HfQwenLinearHooks::vReorderTensors, HfQwenMoeHooks::qwen2MoeVocab),
                // MiniCPMOCR1_0TextModel (minicpm.py:281): the MiniCPM-V 4.6 text trunk without MTP (no_mtp).
                HfArchEntry.of(Collections.singletonList("MiniCPMOcrForConditionalGeneration"),
                                HfLinearHooksB.miniCpmV46Arch(), HfLinearHooksB::miniCpm4_6Meta,
                                HfQwenLinearHooks::vReorderTensors, HfQwenMoeHooks::qwen2MoeVocab)
                        .withNameFilter(HfDecisionHooks.noMtpFilter()),
                // MiniCPMV4_7TextModel (minicpm.py:211): ZYXT rope section order; the arch is QWEN35 or QWEN35MOE
                // by the configuration.
                HfArchEntry.of(Collections.singletonList("MiniCPMV4_7ForConditionalGeneration"),
                                HfLinearHooksB::miniCpmV47Arch, HfLinearHooksB::miniCpm4_7Meta,
                                HfQwenLinearHooks::vReorderTensors, HfQwenMoeHooks::qwen2MoeVocab),
                // MiniMaxText01Model (minimax.py:16): the linear-attention stack with the zero-embedding
                // suppression list and the residual scale.
                HfArchEntry.of(Arrays.asList("MiniMaxText01ForCausalLM", "MiniMaxM1ForCausalLM"),
                                HfLinearHooksA.minimaxArch(), HfLinearHooksA::minimaxMeta,
                                HfLinearHooksA::minimaxTensors, HfLinearHooksA::minimaxVocab)
                        .withIndexedBlockCount(HfLinearHooksA::blockCount)
                        .withLoadNameFilter(HfLinearHooksA::filter)
                        .withExtra(HfLinearHooksA::minimaxExtra),
                // Qwen3NextModel (qwen.py:382): the gated-delta-net hybrid, read-only.
                HfArchEntry.of(Collections.singletonList("Qwen3NextForCausalLM"),
                                HfArchResolver.of(Arch.QWEN3NEXT), HfQwenLinearHooks::qwen3NextMeta,
                                HfQwenLinearHooks::qwen3NextTensors, HfQwenMoeHooks::qwen2MoeVocab)
                        .withIndexedBlockCount(HfQwenHooks::mtpBlockCount)
                        .withLoadNameFilter(HfQwenHooks::mtpFilter)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.QWEN),
                // BailingMoE V2.5 (vLLM models/bailing_moe_linear.py): lightning-attention layers alternating with
                // MLA layers by layer_group_size, the fused QKV regrouped and the MLA latent split; MTP layers are
                // dropped. vLLM registers the class name below.
                HfArchEntry.of(Arrays.asList("BailingMoeV2_5ForCausalLM", "BailingMoeV25ForCausalLM"),
                                HfBailingV25Hooks.arch(), HfBailingV25Hooks::meta, HfBailingV25Hooks::tensors,
                                HfBailingV25Hooks::vocab)
                        .withLoadNameFilter(HfBailingV25Hooks::filter)
                        .withExtra(HfBailingV25Hooks::extra)
                        .withForceType(HfBailingV25Hooks::forceType),
                // OlmoHybridForCausalLM (olmo_hybrid.py): Gated DeltaNet layers alternating with OLMo2 post-norm
                // attention; the checkpoint keeps the linear attention's Q/K/V projections and convolutions separate.
                HfArchEntry.of(Collections.singletonList("OlmoHybridForCausalLM"),
                                HfArchResolver.of(Arch.OLMO_HYBRID), HfOlmoHybridHooks::meta,
                                HfOlmoHybridHooks::tensors, HfOlmoHybridHooks::vocab)
                        .withExtra(HfOlmoHybridHooks::requireComplete),
                // Qwen3_5TextModel (qwen.py:676): the Qwen3.5 trunk with the V-head reorder.
                HfArchEntry.of(Arrays.asList("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
                                "Qwen3_5TextModel"), HfArchResolver.of(Arch.QWEN35),
                                HfQwenLinearHooks::qwen35Meta, HfQwenLinearHooks::vReorderTensors,
                                HfQwenMoeHooks::qwen2MoeVocab)
                        .withIndexedBlockCount(HfQwenHooks::mtpBlockCount)
                        .withLoadNameFilter(HfQwenHooks::mtpFilter)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.QWEN),
                // Qwen3_5MoeTextModel (qwen.py:751): the Qwen3.5 MoE trunk.
                HfArchEntry.of(Arrays.asList("Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"),
                                HfArchResolver.of(Arch.QWEN35MOE), HfQwenLinearHooks::qwen35MoeMeta,
                                HfQwenLinearHooks::vReorderTensors, HfQwenMoeHooks::qwen2MoeVocab)
                        .withIndexedBlockCount(HfQwenHooks::mtpBlockCount)
                        .withLoadNameFilter(HfQwenHooks::mtpFilter)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.QWEN),
                // Qwen4ExpTextModel (qwen4exp.py:18): the hyper-connection stack with the QSA indexer, the PLE
                // n-gram embeddings and the MTP fc concat.
                HfArchEntry.of(Arrays.asList("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"),
                                HfLinearHooksB.qwen4ExpArch(), HfLinearHooksB::qwen4ExpMeta,
                                HfLinearHooksB::qwen4ExpTensors, HfQwenMoeHooks::qwen2MoeVocab)
                        .withLoadNameFilter(HfLinearHooksB::qwen4ExpFilter)
                        .withIndexedBlockCount(HfLinearHooksB::qwen4ExpBlockCount)
                        .withExtra(HfLinearHooksB::qwen4ExpExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.QWEN),
                // RWKV6Model (rwkv.py:15): the world format.
                HfArchEntry.of(Collections.singletonList("Rwkv6ForCausalLM"), HfRwkvHooks.rwkv6Arch(),
                                HfRwkvHooks::rwkv6Meta, HfRwkvHooks::rwkv6Tensors, HfRwkvHooks::worldVocab),
                // RWKV6Qwen2Model (rwkv.py:88): SentencePiece falling back to GPT-2 (rwkv.py:89-91).
                HfArchEntry.of(Collections.singletonList("RWKV6Qwen2ForCausalLM"), HfRwkvHooks.rwkv6Qwen2Arch(),
                                HfRwkvHooks::rwkv6Qwen2Meta, HfRwkvHooks::rwkv6Qwen2Tensors,
                                HfRwkvHooks::sentencePieceOrGpt2),
                // Rwkv7Model (rwkv.py:142): the RWKV-7 world format.
                HfArchEntry.of(Arrays.asList("RWKV7ForCausalLM", "Rwkv7ForCausalLM"), HfRwkvHooks.rwkv7Arch(),
                                HfRwkvHooks::rwkv7Meta, HfRwkvHooks::rwkv7Tensors, HfRwkvHooks::worldVocab)
                        .withNameFilter(HfRwkvHooks.rwkv7Filter()),
                // ARwkv7Model (rwkv.py:268): the ARWKV-7 hybrid world format.
                HfArchEntry.of(Collections.singletonList("RwkvHybridForCausalLM"), HfRwkvHooks.arwkv7Arch(),
                                HfRwkvHooks::arwkv7Meta, HfRwkvHooks::arwkv7Tensors, HfRwkvHooks::worldVocab)
                        .withNameFilter(HfRwkvHooks.rwkv7Filter())));
    }
}
