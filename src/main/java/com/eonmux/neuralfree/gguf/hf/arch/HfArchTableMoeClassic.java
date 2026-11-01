package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGraniteHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMiniMaxHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeClassicHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeClassicHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeClassicHooksC;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMoeClassicHooksD;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenMoeHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.ToIntFunction;

/**
 * Hugging Face architecture entries for the classic mixture-of-experts converter classes (Appendix A table HK-MC).
 * {@link HfArchRegistry} merges this table with the other tables; a Hugging Face name appears in exactly one table per
 * {@link HfArchEntry.Kind}. {@code UltravoxModel} is a named rejection: the converter raises for it as a text model
 * (LC/conversion/ultravox.py:17-19) because the class only builds the audio encoder plus projector.
 */
final class HfArchTableMoeClassic {

    /** The block count of BailingMoeV2Model / SarvamMoEModel: the NextN layers are appended (bailingmoe.py:116-120). */
    private static final ToIntFunction<HfConfig> BAILING_V2_BLOCKS = cfg ->
            cfg.blockCount() + HfMoeClassicHooksB.nextnPredictLayers(cfg);

    private HfArchTableMoeClassic() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                HfArchEntry.of(Collections.singletonList("ArcticForCausalLM"), HfArchResolver.of(Arch.ARCTIC),
                                HfMoeClassicHooksA::arcticMeta, HfMoeClassicHooksA::arcticTensors,
                                HfMoeClassicHooksA::arcticVocab)
                        .withExtra(HfMoeClassicHooksA::arcticExtra),
                HfArchEntry.of(Collections.singletonList("BailingMoeForCausalLM"), HfArchResolver.of(Arch.BAILINGMOE),
                                HfMoeClassicHooksB::bailingMoeMeta, HfMoeClassicHooksB::bailingMoeTensors,
                                HfMoeClassicHooksB::bailingMoeVocab)
                        .withExtra(HfMoeClassicHooksB::bailingMoeExtra),
                HfArchEntry.of(Collections.singletonList("BailingMoeV2ForCausalLM"),
                                HfArchResolver.of(Arch.BAILINGMOE2),
                                HfMoeClassicHooksB::bailingMoeV2Meta, HfMoeClassicHooksB::bailingMoeV2Tensors,
                                HfMoeClassicHooksB::bailingMoeVocab)
                        .withNameFilter(HfMoeClassicHooksB.BAILING_V2_FILTER)
                        .withBlockCount(BAILING_V2_BLOCKS)
                        .withExtra(HfMoeClassicHooksB::bailingMoeV2Extra),
                HfArchEntry.of(Arrays.asList("SarvamMoEForCausalLM", "modeling_sarvam_moe.SarvamMoEForCausalLM"),
                                HfArchResolver.of(Arch.BAILINGMOE2),
                                HfMoeClassicHooksB::sarvamMoeMeta, HfMoeClassicHooksB::sarvamMoeTensors,
                                HfMoeClassicHooksB::bailingMoeVocab)
                        .withNameFilter(HfMoeClassicHooksB.BAILING_V2_FILTER)
                        .withBlockCount(BAILING_V2_BLOCKS)
                        .withExtra(HfMoeClassicHooksB::bailingMoeV2Extra),
                HfArchEntry.of(Collections.singletonList("DbrxForCausalLM"), HfArchResolver.of(Arch.DBRX),
                                HfMoeClassicHooksA::dbrxMeta, HfMoeClassicHooksA::dbrxTensors,
                                HfMoeClassicHooksA::gpt2Vocab)
                        .withForceType(HfMoeClassicHooksA::dbrxForceType),
                HfArchEntry.of(Collections.singletonList("DeepseekForCausalLM"), HfArchResolver.of(Arch.DEEPSEEK),
                                HfMoeClassicHooksB::deepseekMeta, HfMoeClassicHooksB::deepseekTensors,
                                HfMoeClassicHooksB::deepseekVocab)
                        .withExtra(HfMoeClassicHooksB::deepseekExtra),
                HfArchEntry.of(Collections.singletonList("Dots1ForCausalLM"), HfArchResolver.of(Arch.DOTS1),
                                HfMoeClassicHooksB::dots1Meta, HfMoeClassicHooksB::dots1Tensors,
                                HfQwenMoeHooks::qwen2MoeVocab)
                        .withConfigFixup(HfMoeClassicHooksB::dots1Fixup)
                        .withExtra(HfQwenMoeHooks::qwen2MoeExtra),
                HfArchEntry.of(Collections.singletonList("Ernie4_5_MoeForCausalLM"),
                                HfArchResolver.of(Arch.ERNIE4_5_MOE),
                                HfMoeClassicHooksB::ernie45MoeMeta, HfMoeClassicHooksB::ernie45MoeTensors,
                                HfMoeClassicHooksB::ernie45MoeVocab)
                        .withNameFilter(HfMoeClassicHooksB::ernie45MoeFilter)
                        .withExtra(HfMoeClassicHooksB::ernie45MoeExtra),
                HfArchEntry.of(Arrays.asList("GraniteMoeForCausalLM", "GraniteMoeSharedForCausalLM"),
                                HfArchResolver.of(Arch.GRANITE_MOE),
                                HfGraniteHooks::moeMeta, HfGraniteHooks::moeTensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfGraniteHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.of(Arrays.asList("Grok1ForCausalLM", "GrokForCausalLM"), HfArchResolver.of(Arch.GROK),
                                HfMoeClassicHooksA::grokMeta, HfMoeClassicHooksA::grokTensors,
                                HfMoeClassicHooksA::grokVocab)
                        .withExtra(HfMoeClassicHooksA::grokExtra),
                HfArchEntry.of(Arrays.asList("GroveMoeForCausalLM", "modeling_grove_moe.GroveMoeForCausalLM"),
                                HfArchResolver.of(Arch.GROVEMOE),
                                HfMoeClassicHooksC::groveMoeMeta, HfMoeClassicHooksC::groveMoeTensors,
                                HfMoeClassicHooksA::gpt2Vocab)
                        .withExtra(HfMoeClassicHooksC::groveMoeExtra),
                HfArchEntry.of(Collections.singletonList("HunYuanMoEV1ForCausalLM"),
                                HfArchResolver.of(Arch.HUNYUAN_MOE),
                                HfMoeClassicHooksC::hunYuanMoeMeta, HfMoeClassicHooksC::hunYuanMoeTensors,
                                HfMoeClassicHooksC::hunYuanMoeVocab)
                        .withExtra(HfMoeClassicHooksC::hunYuanMoeExtra),
                HfArchEntry.of(Collections.singletonList("HYV3ForCausalLM"), HfArchResolver.of(Arch.HY_V3),
                                HfMoeClassicHooksC::hyV3Meta, HfMoeClassicHooksC::hyV3Tensors,
                                HfMoeClassicHooksA::gpt2Vocab)
                        .withLoadNameFilter(HfMoeClassicHooksC::hyV3Filter)
                        .withIndexedBlockCount(HfMoeClassicHooksC::hyV3BlockCount)
                        .withExtra(HfMoeClassicHooksC::hyV3Extra),
                HfArchEntry.of(Collections.singletonList("InternLM3ForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                                HfMoeClassicHooksD::internLm3Meta, HfMoeClassicHooksD::internLm3Tensors,
                                HfMoeClassicHooksD::internLm3Vocab)
                        .withNameFilter(HfMoeClassicHooksD.INTERN_LM3_FILTER),
                HfArchEntry.of(Collections.singletonList("JanusForConditionalGeneration"),
                                HfArchResolver.of(Arch.LLAMA),
                                HfLlamaHooks::meta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfMoeClassicHooksD.JANUS_FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.of(Collections.singletonList("K2HorizonForCausalLM"), HfArchResolver.of(Arch.K2_HORIZON),
                                HfMoeClassicHooksC::k2HorizonMeta, HfMoeClassicHooksC::k2HorizonTensors,
                                HfMoeClassicHooksA::gpt2Vocab)
                        .withExtra(HfMoeClassicHooksC::k2HorizonExtra),
                HfArchEntry.of(Arrays.asList("Llama4ForCausalLM", "Llama4ForConditionalGeneration"),
                                HfArchResolver.of(Arch.LLAMA4),
                                HfMoeClassicHooksA::llama4Meta, HfMoeClassicHooksA::llama4Tensors,
                                HfMoeClassicHooksA::llama4Vocab)
                        .withConfigFixup(HfMoeClassicHooksA::llama4Fixup)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra)
                        .withUndoPermute(false),
                HfArchEntry.of(Collections.singletonList("MiniMaxM2ForCausalLM"), HfArchResolver.of(Arch.MINIMAX_M2),
                                HfMiniMaxHooks::meta, HfMiniMaxHooks::tensors, HfMiniMaxHooks::vocab),
                HfArchEntry.of(Collections.singletonList("OlmoeForCausalLM"), HfArchResolver.of(Arch.OLMOE),
                                HfMoeClassicHooksC::olmoeMeta, HfMoeClassicHooksC::olmoeTensors,
                                HfMoeClassicHooksA::gpt2Vocab)
                        .withExtra(HfMoeClassicHooksC::olmoeExtra),
                HfArchEntry.of(Collections.singletonList("PhiMoEForCausalLM"), HfArchResolver.of(Arch.PHIMOE),
                                HfMoeClassicHooksC::phiMoeMeta, HfMoeClassicHooksC::phiMoeTensors,
                                HfBaseDenseHooks::phi3Vocab)
                        .withExtra(HfBaseDenseHooks::phi3Extra),
                HfArchEntry.of(Collections.singletonList("Qwen2MoeForCausalLM"), HfArchResolver.of(Arch.QWEN2MOE),
                                HfQwenMoeHooks::qwen2MoeMeta, HfQwenMoeHooks::qwen2MoeTensors,
                                HfQwenMoeHooks::qwen2MoeVocab)
                        .withExtra(HfQwenMoeHooks::qwen2MoeExtra),
                HfArchEntry.of(Collections.singletonList("Qwen3OmniMoeForConditionalGeneration"),
                                HfArchResolver.of(Arch.QWEN3VLMOE),
                                HfMoeClassicHooksD::qwen3OmniMoeMeta, HfMoeClassicHooksD::qwen3VlMoeTensors,
                                HfMoeClassicHooksD::qwen3OmniMoeVocab)
                        .withNameFilter(HfMoeClassicHooksD.THINKER_FILTER)
                        .withExtra(HfQwenMoeHooks::qwen2MoeExtra),
                HfArchEntry.of(Collections.singletonList("Qwen3VLMoeForConditionalGeneration"),
                                HfArchResolver.of(Arch.QWEN3VLMOE),
                                HfMoeClassicHooksD::qwen3VlMoeMeta, HfMoeClassicHooksD::qwen3VlMoeTensors,
                                HfQwenMoeHooks::qwen3MoeVocab)
                        .withNameFilter(HfMoeClassicHooksD.THINKER_FILTER)
                        .withExtra(HfQwenMoeHooks::qwen2MoeExtra),
                HfArchEntry.of(Collections.singletonList("Sarashina2VisionForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                                HfLlamaHooks::meta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfMoeClassicHooksD.SARASHINA_FILTER)
                        .withExtra(HfLlamaHooks::extra),
                HfArchEntry.rejected(Collections.singletonList("UltravoxModel"),
                        "UltravoxModel has no text model; it is a projector checkpoint")));
    }
}
