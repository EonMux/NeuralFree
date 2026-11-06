package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGemma4Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGptOssHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGraniteHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeSwaHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeSwaHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeSwaHooksC;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeSwaHooksD;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenMoeHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries: the sliding-window and attention-sink MoE
 * converter classes — Afmoe, Kolibri-1, Laguna, Flint, Beam, Cohere-2 MoE, ExaoneMoE, the Gemma 4 rows, GPT-OSS, the
 * Granite SWA rows, Maple, Mellum, MiMo-V2, Muse-Spark, SmallThinker and Step3.5 (LC conversion classes
 * afmoe.py:17, laguna.py:17, flint.py:9, beam.py:14, command_r.py:67, exaone.py:132, gemma.py:634/:936, gpt_oss.py:15,
 * granite.py:78/:125, kolibri1.py:9, maple.py:15, mellum.py:15, mimo.py:18, muse_spark.py:39, smallthinker.py:15,
 * step3.py:102). {@link HfArchRegistry} merges this table with the other tables; a Hugging Face name appears in
 * exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableMoeSwa {

    private HfArchTableMoeSwa() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                HfArchEntry.of(Collections.singletonList("AfmoeForCausalLM"), HfArchResolver.of(Arch.AFMOE),
                                HfMoeSwaHooksA::afmoeMeta, HfMoeSwaHooksA::afmoeTensors, HfLlamaHooks::vocab)
                        .withNameFilter(HfMoeSwaHooksA.AFMOE_FILTER)
                        .withExtra(HfMoeSwaHooksA::afmoeExtra),
                HfArchEntry.of(Collections.singletonList("LagunaForCausalLM"), HfArchResolver.of(Arch.LAGUNA),
                                HfMoeSwaHooksC::lagunaMeta, HfMoeSwaHooksC::lagunaTensors, HfMoeSwaHooksC::lagunaVocab)
                        .withExtra(HfMoeSwaHooksC::lagunaExtra),
                HfArchEntry.of(Collections.singletonList("FlintForCausalLM"), HfArchResolver.of(Arch.FLINT),
                                HfMoeSwaHooksA::flintMeta, HfMoeSwaHooksA::flintTensors,
                                HfQwenMoeHooks::qwen2MoeVocab)
                        .withExtra(HfQwenMoeHooks::qwen2MoeExtra),
                HfArchEntry.of(Collections.singletonList("BeamForCausalLM"), HfArchResolver.of(Arch.BEAM),
                                HfMoeSwaHooksB::beamMeta, HfMoeSwaHooksB::beamTensors, HfMoeSwaHooksB::beamVocab)
                        .withConfigFixup(HfMoeSwaHooksB::beamFixup)
                        .withNameFilter(HfMoeSwaHooksB.BEAM_FILTER)
                        .withExtra(HfMoeSwaHooksB::beamExtra),
                HfArchEntry.of(Collections.singletonList("Cohere2MoeForCausalLM"),
                                HfArchResolver.of(Arch.COHERE2MOE), HfMoeSwaHooksA::cohere2Meta,
                                HfMoeSwaHooksA::cohere2Tensors, HfMoeSwaHooksA::gpt2Vocab)
                        .withBlockCount(HfMoeSwaHooksA::cohere2BlockCount)
                        .withExtra(HfMoeSwaHooksA::cohere2Extra),
                HfArchEntry.of(Arrays.asList("ExaoneMoEForCausalLM", "ExaoneMoeForCausalLM"),
                                HfArchResolver.of(Arch.EXAONE_MOE), HfMoeSwaHooksC::exaoneMoeMeta,
                                HfMoeSwaHooksC::exaoneMoeTensors, HfBaseDenseHooks::exaone4Vocab)
                        .withIndexedBlockCount(HfMoeSwaHooksC::exaoneMoeBlocks)
                        .withExtra(HfMoeSwaHooksC::exaoneMoeExtra),
                HfArchEntry.of(Arrays.asList("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"),
                                HfArchResolver.of(Arch.GEMMA4), HfGemma4Hooks::meta, HfGemma4Hooks::tensors,
                                HfGemma4Hooks::vocab)
                        .withNameFilter(HfGemma4Hooks::filter)
                        .withIndex(HfGemma4Hooks::index)
                        .withExtra(HfGemma4Hooks::extra),
                HfArchEntry.of(Collections.singletonList("Gemma4UnifiedForConditionalGeneration"),
                                HfArchResolver.of(Arch.GEMMA4), HfMoeSwaHooksA::gemma4UnifiedMeta,
                                HfGemma4Hooks::tensors, HfGemma4Hooks::vocab)
                        .withNameFilter(HfGemma4Hooks::filter)
                        .withIndex(HfGemma4Hooks::index)
                        .withExtra(HfGemma4Hooks::extra),
                HfArchEntry.of(Collections.singletonList("GptOssForCausalLM"), HfArchResolver.of(Arch.OPENAI_MOE),
                                HfGptOssHooks::meta, HfGptOssHooks::tensors, HfGptOssHooks::vocab),
                HfArchEntry.of(Collections.singletonList("GraniteSWAForCausalLM"),
                                HfArchResolver.of(Arch.GRANITE_SWA), HfMoeSwaHooksA::graniteSwaMeta,
                                HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfMoeSwaHooksA.GRANITE_SWA_FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.of(Collections.singletonList("GraniteMoeSWAForCausalLM"),
                                HfArchResolver.of(Arch.GRANITE_SWA), HfMoeSwaHooksA::graniteMoeSwaMeta,
                                HfGraniteHooks::moeTensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfMoeSwaHooksA.GRANITE_SWA_FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.of(Collections.singletonList("Kolibri1ForCausalLM"), HfArchResolver.of(Arch.KOLIBRI1),
                                HfMoeSwaHooksA::kolibri1Meta, HfMoeSwaHooksA::afmoeTensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfMoeSwaHooksA::kolibri1Fixup)
                        .withNameFilter(HfMoeSwaHooksA.AFMOE_FILTER)
                        .withExtra(HfMoeSwaHooksA::afmoeExtra),
                HfArchEntry.of(Collections.singletonList("MapleForCausalLM"), HfArchResolver.of(Arch.MAPLE),
                                HfMoeSwaHooksC::mapleMeta, HfMoeSwaHooksC::mapleTensors,
                                HfMoeSwaHooksC::mapleVocab)
                        .withExtra(HfMoeSwaHooksC::mapleExtra),
                HfArchEntry.of(Collections.singletonList("MellumForCausalLM"), HfArchResolver.of(Arch.MELLUM),
                                HfMoeSwaHooksC::mellumMeta, HfMoeSwaHooksC::mellumTensors, HfMoeSwaHooksA::gpt2Vocab)
                        .withIndexedBlockCount(HfMoeSwaHooksC::mellumBlocks)
                        .withExtra(HfMoeSwaHooksC::mellumExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON)
                        .withMtpOnlyGuard(cfg -> cfg.findInt(0, "num_nextn_predict_layers",
                                "num_mtp_layers") > 0),
                HfArchEntry.of(Arrays.asList("MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM"),
                                HfArchResolver.of(Arch.MIMO2), HfMoeSwaHooksB::mimoMeta, HfMoeSwaHooksB::mimoTensors,
                                HfMoeSwaHooksA::gpt2Vocab)
                        .withIndex(HfMoeSwaHooksB::mimoIndex)
                        .withIndexedBlockCount(HfMoeSwaHooksB::mimoBlocks)
                        .withNameFilter(HfMoeSwaHooksB.MIMO_FILTER)
                        .withExtra(HfMoeSwaHooksB::mimoExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                HfArchEntry.of(Collections.singletonList("MuseSparkForConditionalGeneration"),
                                HfArchResolver.of(Arch.MUSE_SPARK), HfMoeSwaHooksD::museSparkMeta,
                                HfMoeSwaHooksD::museSparkTensors, HfMoeSwaHooksD::museSparkVocab),
                HfArchEntry.of(Collections.singletonList("SmallThinkerForCausalLM"),
                                HfArchResolver.of(Arch.SMALLTHINKER), HfMoeSwaHooksC::smallthinkerMeta,
                                HfMoeSwaHooksC::smallthinkerTensors, HfMoeSwaHooksA::gpt2Vocab)
                        .withExtra(HfMoeSwaHooksC::smallthinkerExtra),
                // Step3p5MTP is the draft-model view of the same checkpoint: its MTP layers are the appended NextN
                // blocks this entry keeps (the num_nextn_predict_layers of the text model).
                HfArchEntry.of(Arrays.asList("Step3p5ForCausalLM", "Step3p7ForConditionalGeneration", "Step3p5MTP"),
                                HfArchResolver.of(Arch.STEP35), HfMoeSwaHooksD::step35Meta,
                                HfMoeSwaHooksD::step35Tensors, HfMoeSwaHooksA::gpt2Vocab)
                        .withIndexedBlockCount(HfMoeSwaHooksD::step35Blocks)
                        .withNameFilter(HfMoeSwaHooksD.STEP35_FILTER)
                        .withExtra(HfMoeSwaHooksD::step35Extra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON)));
    }
}
