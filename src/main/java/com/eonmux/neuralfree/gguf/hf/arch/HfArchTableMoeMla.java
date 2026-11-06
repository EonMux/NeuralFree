package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekV4Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekV41Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMiniMaxHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMistralLarge4Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMistralMoeHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeMlaHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeMlaHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeMlaHooksC;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries: the multi-head latent attention, sparse
 * attention and indexer converter classes — DeepseekV2 (with the OCR, Kimi and Youtu variants), DeepseekV32,
 * DeepseekV4, Dots3-Note, the GLM rows (GLM-4 MoE, GLM-4 MoE Lite, GLM MoE DSA), HY-V4, MiniCPM3, MiniMaxM3, PLM,
 * Solar-Open and the synthesized native Mistral MoE format (LC conversion classes deepseek.py:236/:466/:527,
 * dots3.py:19, glm.py:113/:249/:318/:387, hy_v4.py:26, minicpm.py:66, minimax.py:169, plm.py:8, mistral.py:113).
 * The DeepseekV2, DeepseekV4 and MiniMaxM3 rows delegate to the {@link HfDeepseekHooks},
 * {@link HfDeepseekV4Hooks} and {@link HfMiniMaxHooks} statics. {@link HfArchRegistry} merges this table with the
 * other tables; a Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableMoeMla {

    private HfArchTableMoeMla() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // DeepseekV2Model (deepseek.py:236): the DeepSeek-V2/V3, the DeepSeek-OCR variants, the Kimi
                // and the Youtu rows.
                HfArchEntry.of(Arrays.asList("DeepseekOCRForCausalLM", "DeepseekV2ForCausalLM",
                                "DeepseekV3ForCausalLM", "KimiK25ForConditionalGeneration",
                                "KimiVLForConditionalGeneration", "UnlimitedOCRForCausalLM", "YoutuForCausalLM",
                                "YoutuVLForConditionalGeneration"), HfDeepseekHooks::arch, HfDeepseekHooks::meta,
                                HfDeepseekHooks::tensors, HfDeepseekHooks::vocab)
                        .withLoadNameFilter(ctx -> HfDeepseekHooks::filter)
                        .withExtra(HfDeepseekHooks::extra),
                // DeepseekV32Model (deepseek.py:466): the DSA indexer keys and the appended NextN block.
                HfArchEntry.of(Collections.singletonList("DeepseekV32ForCausalLM"), HfArchResolver.of(Arch.DEEPSEEK32),
                                HfMoeMlaHooksA::deepseek32Meta, HfMoeMlaHooksA::deepseek32Tensors,
                                HfMoeMlaHooksA::deepseek32Vocab)
                        .withIndexedBlockCount(HfMoeMlaHooksA::deepseek32BlockCount)
                        .withLoadNameFilter(HfMoeMlaHooksA.mtpKeepFilter())
                        .withExtra(HfMoeMlaHooksA::deepseekExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // DeepseekV4Model (deepseek.py:527): the DeepSeek-V4 hooks, read-only.
                HfArchEntry.of(Collections.singletonList("DeepseekV4ForCausalLM"), HfArchResolver.of(Arch.DEEPSEEK4),
                                HfDeepseekV4Hooks::meta, HfDeepseekV4Hooks::tensors, HfDeepseekV4Hooks::vocab)
                        .withConfigFixup(HfDeepseekV4Hooks::configFixup)
                        .withIndexedBlockCount(HfDeepseekV4Hooks::blockCount)
                        .withLoadNameFilter(HfDeepseekV4Hooks::filter)
                        .withForceType(HfDeepseekV4Hooks::forceType)
                        .withExtra(HfDeepseekV4Hooks::extra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.DEEPSEEK_V4),
                // DeepseekV41Model (deepseek41.py:117): the V4 hooks with the compressed-cache source layers, the
                // two-level candidate selection, the indexer key names, no MTP export and the F32 first-layer gate.
                HfArchEntry.of(Collections.singletonList("DeepseekV41ForCausalLM"), HfDeepseekV41Hooks::arch,
                                HfDeepseekV41Hooks::meta, HfDeepseekV41Hooks::tensors, HfDeepseekV41Hooks::vocab)
                        .withConfigFixup(HfDeepseekV41Hooks::configFixup)
                        .withBlockCount(HfDeepseekV41Hooks::blockCount)
                        .withLoadNameFilter(ctx -> HfDeepseekV41Hooks.filter())
                        .withForceType(HfDeepseekV41Hooks::forceType)
                        .withExtra(HfDeepseekV41Hooks::extra),
                // Dots3NoteModel (dots3.py:19): the dual-kind (full / sliding-window) MLA with the DSA indexer,
                // the MTP block detected from the weights and the F32 routing load-bearing forcing.
                HfArchEntry.of(Arrays.asList("Dots3NoteForCausalLM", "Dots3NoteForConditionalGeneration",
                                "Dots3NoteTextForCausalLM"), HfArchResolver.of(Arch.DOTS3NOTE),
                                HfMoeMlaHooksA::dots3Meta, HfMoeMlaHooksA::dots3Tensors,
                                HfMoeMlaHooksA::dots3Vocab)
                        .withConfigFixup(HfMoeMlaHooksA::dots3Fixup)
                        .withBlockCount(HfMoeMlaHooksA::dots3ConfigBlockCount)
                        .withIndexedBlockCount(HfMoeMlaHooksA::dots3BlockCount)
                        .withLoadNameFilter(HfMoeMlaHooksA.dots3Filter())
                        .withForceType(HfMoeMlaHooksA::dots3ForceType)
                        .withExtra(HfMoeMlaHooksA::deepseekExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // Glm4MoeModel (glm.py:113): the standalone MLA geometry, the routed-expert keys and the GLM
                // vocabulary.
                HfArchEntry.of(Arrays.asList("Glm4MoeForCausalLM", "Glm4vMoeForConditionalGeneration"),
                                HfArchResolver.of(Arch.GLM4_MOE), HfMoeMlaHooksB::glm4MoeMeta,
                                HfMoeMlaHooksB::glm4MoeTensors, HfMoeMlaHooksA::glmVocab)
                        .withIndexedBlockCount(HfMoeMlaHooksB::glm4MoeBlockCount)
                        .withExtra(HfMoeMlaHooksB::glm4MoeExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // Glm4MoeLiteModel (glm.py:249): the DeepseekV2-deriving MLA with the MTP layers kept.
                HfArchEntry.of(Collections.singletonList("Glm4MoeLiteForCausalLM"),
                                HfArchResolver.of(Arch.DEEPSEEK2), HfMoeMlaHooksA::glm4MoeLiteMeta,
                                HfMoeMlaHooksA::glm4MoeLiteTensors, HfMoeMlaHooksA::glmVocab)
                        .withIndexedBlockCount(HfMoeMlaHooksA::glm4MoeLiteBlockCount)
                        .withLoadNameFilter(HfMoeMlaHooksA.mtpKeepFilter())
                        .withExtra(HfMoeMlaHooksA::deepseekExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // GlmMoeDsaModel (glm.py:318): the DSA indexer over the base MLA, with the per-layer
                // indexer_types.
                HfArchEntry.of(Collections.singletonList("GlmMoeDsaForCausalLM"), HfArchResolver.of(Arch.GLM_DSA),
                                HfMoeMlaHooksA::glmMoeDsaMeta, HfMoeMlaHooksA::glmMoeDsaTensors,
                                HfMoeMlaHooksA::glmVocab)
                        .withIndexedBlockCount(HfMoeMlaHooksA::glmMoeDsaBlockCount)
                        .withLoadNameFilter(HfMoeMlaHooksA.mtpKeepFilter())
                        .withExtra(HfMoeMlaHooksA::deepseekExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // HYV4Model (hy_v4.py:26): the DeepseekV2-composing MLA with the fused stacked experts split,
                // the iHC metadata, the per-layer indexer keys and the F32 forcing of the iHC *_fn matrices.
                HfArchEntry.of(Collections.singletonList("HYV4ForCausalLM"), HfArchResolver.of(Arch.HY_V4),
                                HfMoeMlaHooksC::hyV4Meta, HfMoeMlaHooksC::hyV4Tensors, HfDeepseekHooks::vocab)
                        .withConfigFixup(HfMoeMlaHooksC::hyV4Fixup)
                        .withBlockCount(HfMoeMlaHooksC::hyV4BlockCount)
                        .withNameFilter(HfMoeMlaHooksC.HYV4_FILTER)
                        .withForceType(HfMoeMlaHooksC::hyV4ForceType)
                        .withExtra(HfMoeMlaHooksC::hyV4Extra),
                // MiniCPM3Model (minicpm.py:66): the standalone MLA geometry with the long and short
                // LongRoPE factors.
                HfArchEntry.of(Collections.singletonList("MiniCPM3ForCausalLM"), HfArchResolver.of(Arch.MINICPM3),
                                HfMoeMlaHooksB::miniCpm3Meta, HfMiniMaxHooks::tensors,
                                HfMoeMlaHooksB::miniCpm3Vocab)
                        .withExtra(HfMoeMlaHooksB::miniCpm3Extra),
                // MiniMaxM3Model (minimax.py:169): the MiniMax M2 base with the block-sparse indexer, the
                // Gemma-style 1 + w norms and the F32 indexer forcing.
                HfArchEntry.of(Arrays.asList("MiniMaxM3SparseForCausalLM",
                                "MiniMaxM3SparseForConditionalGeneration"), HfArchResolver.of(Arch.MINIMAX_M3),
                                HfMoeMlaHooksB::miniMaxM3Meta, HfMoeMlaHooksB::miniMaxM3Tensors,
                                HfMiniMaxHooks::vocab)
                        .withForceType(HfMoeMlaHooksB::miniMaxM3ForceType),
                // PLMModel (plm.py:8): the MLA-lite geometry with the GPT-2 vocabulary.
                HfArchEntry.of(Collections.singletonList("PLMForCausalLM"), HfArchResolver.of(Arch.PLM),
                                HfMoeMlaHooksB::plmMeta, HfMiniMaxHooks::tensors, HfMoeMlaHooksB::plmVocab),
                // SolarOpenModel (glm.py:387): the GLM-4 MoE geometry with the Solar-Open vocabulary specials.
                HfArchEntry.of(Collections.singletonList("SolarOpenForCausalLM"), HfArchResolver.of(Arch.GLM4_MOE),
                                HfMoeMlaHooksB::glm4MoeMeta, HfMoeMlaHooksB::glm4MoeTensors,
                                HfMoeMlaHooksB::solarOpenVocab)
                        .withIndexedBlockCount(HfMoeMlaHooksB::glm4MoeBlockCount)
                        .withExtra(HfMoeMlaHooksB::glm4MoeExtra),
                // MistralMoeModel (mistral.py:113, unregistered): the synthesized native Mistral MoE format,
                // always DEEPSEEK2, registered under the name HfConfigLoader synthesizes from params.json.
                HfArchEntry.of(Collections.singletonList(HfConfigLoader.MISTRAL_NATIVE_MOE), HfMistralMoeHooks::arch,
                                HfMistralMoeHooks::meta, HfMistralMoeHooks::tensors, HfMistralMoeHooks::vocab)
                        .withConfigFixup(HfMistralMoeHooks::fixup)
                        .withIndex(HfMistralMoeHooks::index)
                        .withNameFilter(HfMistralMoeHooks::filter)
                        .withExtra(HfMistralMoeHooks::extra)
                        .withUndoPermute(false),
                // MistralLarge4Model (mistral4.py:15): the Mistral MoE remap over MISTRAL_LARGE4, with the
                // indexer metadata, the LeRoPE factors, the per-layer wkv_b split and the tanh logit cap.
                HfArchEntry.of(Collections.singletonList("MistralLarge4ForConditionalGeneration"),
                                HfMistralLarge4Hooks::arch, HfMistralLarge4Hooks::meta,
                                HfMistralLarge4Hooks::tensors, HfMistralLarge4Hooks::vocab)
                        .withConfigFixup(HfMistralLarge4Hooks::fixup)
                        .withNameFilter(HfMistralLarge4Hooks.FILTER)
                        .withExtra(HfMistralLarge4Hooks::extra)));
    }
}
