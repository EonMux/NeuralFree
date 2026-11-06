package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.hooks.HfGraniteHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLfm3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSsmHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSsmHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSsmHybridHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSsmHybridHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfZamba2Hooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the state-space-model hybrids (llama.cpp's {@code MambaModel},
 * {@code Mamba2Model}, {@code FalconH1Model}, {@code JambaModel}, {@code GraniteHybridModel},
 * {@code NemotronHModel}, {@code NemotronHPuzzleModel}, {@code Plamo2Model}, the LFM2/LFM3 families and vLLM's
 * {@code Zamba2ForCausalLM}).
 * {@link HfArchRegistry} merges this table with the other tables; a Hugging Face name appears in exactly one
 * table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableHybridSsm {

    private HfArchTableHybridSsm() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // MambaModel, FalconMamba (LC/conversion/mamba.py:18-100)
                HfArchEntry.of(Arrays.asList("MambaForCausalLM", "MambaLMHeadModel", "FalconMambaForCausalLM"),
                                HfSsmHooks::mambaArch, HfSsmHooks::mambaMeta, HfSsmHooks::mambaTensor,
                                HfSsmHooks::mambaVocab)
                        .withConfigFixup(HfSsmHooks::mambaFixup),
                // Mamba2Model (mamba.py:105-200)
                HfArchEntry.of(Collections.singletonList("Mamba2ForCausalLM"), HfSsmHooks::mamba2Arch,
                                HfSsmHooks::mamba2Meta, HfSsmHooks::mamba2Tensor, HfSsmHooks::mamba2Vocab)
                        .withConfigFixup(HfSsmHooks::mamba2Fixup)
                        .withNameFilter(HfSsmHooks.MAMBA2_FILTER),
                // Zamba2Model (transformers/models/zamba2/modeling_zamba2.py)
                HfArchEntry.of(Collections.singletonList("Zamba2ForCausalLM"), HfZamba2Hooks::arch,
                                HfZamba2Hooks::meta, HfZamba2Hooks::tensors, HfZamba2Hooks::vocab)
                        .withConfigFixup(HfZamba2Hooks::fixup),
                // FalconH1Model (falcon_h1.py:16-119)
                HfArchEntry.of(Collections.singletonList("FalconH1ForCausalLM"), HfSsmHooksA::falconH1Arch,
                                HfSsmHooksA::falconH1Meta, HfSsmHooksA::falconH1Tensor, HfSsmHooksA::falconH1Vocab)
                        .withNameFilter(HfSsmHooks.MAMBA2_FILTER),
                // JambaModel (jamba.py:15-111)
                HfArchEntry.of(Collections.singletonList("JambaForCausalLM"), HfSsmHooksA::jambaArch,
                                HfSsmHooksA::jambaMeta, HfSsmHooksA::jambaTensor, HfSsmHooksA::jambaVocab)
                        .withExtra(HfSsmHooksA::jambaExtra),
                // Plamo2Model (plamo.py:63-148)
                HfArchEntry.of(Arrays.asList("PLaMo2ForCausalLM", "Plamo2ForCausalLM"), HfSsmHooksA::plamo2Arch,
                                HfSsmHooksA::plamo2Meta, HfSsmHooksA::plamo2Tensor, HfSsmHooksA::plamo2Vocab),
                // GraniteHybridModel (granite.py:436-576): the no-SSM layers fall back to granite/granitemoe
                HfArchEntry.of(Arrays.asList("BambaForCausalLM", "GraniteMoeHybridForCausalLM"),
                                HfSsmHybridHooksA::graniteHybridArch, HfSsmHybridHooksA::graniteHybridMeta,
                                HfSsmHybridHooksA::graniteHybridTensor, HfSsmHybridHooksA::graniteHybridVocab)
                        .withConfigFixup(HfSsmHybridHooksA::graniteHybridFixup)
                        .withNameFilter(HfSsmHybridHooksA.GRANITE_HYBRID_FILTER),
                // NemotronHModel (nemotron.py:405-731): the MoE MTP block folds as the extra trailing block
                HfArchEntry.of(Collections.singletonList("NemotronHForCausalLM"), HfSsmHybridHooksA::nemotronHArch,
                                HfSsmHybridHooksA::nemotronHMeta, HfSsmHybridHooksA::nemotronHTensor,
                                HfSsmHybridHooksA::nemotronHVocab)
                        .withConfigFixup(HfSsmHybridHooksA::nemotronHFixup)
                        .withIndexedBlockCount(HfSsmHybridHooksA::nemotronHBlockCount)
                        .withNameFilter(HfSsmHybridHooksA.NEMOTRON_H_FILTER)
                        .withExtra(HfSsmHybridHooksA::nemotronHExtra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.NEMOTRON)
                        .withMtpOnlyGuard(HfSsmHybridHooksA::nemotronMtpFolded),
                // NemotronHPuzzleModel (nemotron.py:733-928): the MTP head drops (no per-block MTP path)
                HfArchEntry.of(Collections.singletonList("NemotronHPuzzleForCausalLM"),
                                HfSsmHybridHooksA::nemotronHPuzzleArch, HfSsmHybridHooksA::nemotronHPuzzleMeta,
                                HfSsmHybridHooksA::nemotronHPuzzleTensor, HfSsmHybridHooksA::nemotronHPuzzleVocab)
                        .withConfigFixup(HfSsmHybridHooksA::nemotronHPuzzleFixup)
                        .withNameFilter(HfSsmHybridHooksA.PUZZLE_FILTER)
                        .withExtra(HfSsmHybridHooksA::nemotronHExtra),
                // LFM2Model and D1Model (lfm2.py:20-131, 87-131)
                HfArchEntry.of(Arrays.asList("LFM2ForCausalLM", "Lfm2ForCausalLM"), HfSsmHybridHooksB::lfm2Arch,
                                HfSsmHybridHooksB::lfm2Meta, HfSsmHybridHooksB::lfm2Tensor,
                                HfSsmHybridHooksB::lfm2Vocab)
                        .withConfigFixup(HfSsmHybridHooksB::lfm2Fixup)
                        .withNameFilter(HfSsmHybridHooksB.LFM2_FILTER),
                HfArchEntry.of(Collections.singletonList("D1Model"), HfSsmHybridHooksB::lfm2Arch,
                                HfSsmHybridHooksB::d1Meta, HfSsmHybridHooksB::lfm2Tensor,
                                HfSsmHybridHooksB::d1Vocab)
                        .withConfigFixup(HfSsmHybridHooksB::lfm2Fixup)
                        .withNameFilter(HfSsmHybridHooksB.LFM2_FILTER),
                // LFM2ColBertModel (lfm2.py:135-161)
                HfArchEntry.of(Arrays.asList("Lfm2BidirectionalForMaskedLM", "Lfm2BidirectionalModel",
                                "Lfm2Model"),
                                HfSsmHybridHooksB::lfm2Arch, HfSsmHybridHooksB::colBertMeta,
                                HfSsmHybridHooksB::colBertTensor, HfSsmHybridHooksB::colBertVocab)
                        .withConfigFixup(HfSsmHybridHooksB::lfm2Fixup)
                        .withNameFilter(HfSsmHybridHooksB.LFM2_DENSE_FILTER)
                        .withExtra(HfSsmHybridHooksB::colBertExtra),
                // D1OmniModel (lfm2.py:164-276) via the d1_omni hparams loader (lfm2.py:164-183)
                HfArchEntry.of(Collections.singletonList("D1OmniModel"), HfSsmHybridHooksB::lfm2Arch,
                                HfSsmHybridHooksB::d1OmniMeta, HfSsmHybridHooksB::d1OmniTensor,
                                HfSsmHybridHooksB::d1OmniVocab)
                        .withConfigFixup(HfSsmHybridHooksB::d1OmniFixup)
                        .withNameFilter(HfSsmHybridHooksB.D1_OMNI_FILTER),
                // LFM2MoeModel (lfm2.py:279-345)
                HfArchEntry.of(Collections.singletonList("Lfm2MoeForCausalLM"), HfSsmHybridHooksB::lfm2MoeArch,
                                HfSsmHybridHooksB::lfm2MoeMeta, HfSsmHybridHooksB::lfm2MoeTensor,
                                HfSsmHybridHooksB::lfm2Vocab)
                        .withConfigFixup(HfSsmHybridHooksB::lfm2MoeFixup)
                        .withNameFilter(HfSsmHybridHooksB.LFM2_MOE_FILTER)
                        .withExtra(HfSsmHybridHooksB::lfm2MoeExtra),
                // Lfm3ForCausalLM, Lfm3VlForConditionalGeneration and Lfm3OmniForConditionalGeneration
                // (lfm3.py:18-119): the LFM3 hybrid trunk; the media tensors of the VL/Omni checkpoints drop.
                HfArchEntry.of(Arrays.asList("Lfm3ForCausalLM", "Lfm3VlForConditionalGeneration",
                                "Lfm3OmniForConditionalGeneration"),
                                HfLfm3Hooks::arch, HfLfm3Hooks::meta, HfLfm3Hooks::tensors, HfLfm3Hooks::vocab)
                        .withConfigFixup(HfLfm3Hooks::fixup)
                        .withNameFilter(HfLfm3Hooks.FILTER)
                        .withExtra(HfLfm3Hooks::extra),
                // Lfm3D1Model (lfm3.py:145-176): the d1 decision head on the LFM3-VL/Omni trunk.
                HfArchEntry.of(Collections.singletonList("Lfm3D1Model"), HfLfm3Hooks::arch,
                                HfLfm3Hooks::d1Meta, HfLfm3Hooks::tensors, HfLfm3Hooks::d1Vocab)
                        .withConfigFixup(HfLfm3Hooks::fixup)
                        .withNameFilter(HfLfm3Hooks.FILTER)
                        .withExtra(HfLfm3Hooks::extra),
                // LFM25AudioTokenizer (lfm2.py:495-519)
                HfArchEntry.of(Collections.singletonList("Lfm25AudioTokenizer"), HfSsmHybridHooksB::lfm2Arch,
                                HfSsmHybridHooksB::audioTokenizerMeta, HfSsmHybridHooksB::audioTokenizerTensor,
                                HfSsmHybridHooksB::audioTokenizerVocab)
                        .withConfigFixup(HfSsmHybridHooksB::lfm2Fixup)
                        .withNameFilter(HfSsmHybridHooksB.AUDIO_TOKENIZER_FILTER)));
    }
}
