package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksC;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksD;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksE;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksF;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGraniteHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfHookSupport;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the classic dense decoder families of Appendix A table HK-D1 (the
 * {@code TextModel} classes of {@code BaichuanModel} … {@code XverseModel}) and of HK-0's {@code GraniteModel},
 * registered through {@link HfArchRegistry} with the compatible hook parts A to E. {@code GraniteMoeModel} is
 * registered on HK-MC's table. The Hugging Face names are exactly the Appendix A rows.
 */
final class HfArchTableDenseClassic {

    private HfArchTableDenseClassic() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // ---- part A: Baichuan … CodeShell
                HfArchEntry.of(Arrays.asList("BaiChuanForCausalLM", "BaichuanForCausalLM"),
                                HfArchResolver.of(Arch.BAICHUAN), HfDenseClassicHooksA::baichuanMeta,
                                HfDenseClassicHooksA::baichuanTensors, HfDenseClassicHooksE::sentencePieceVocab),
                HfArchEntry.of(Arrays.asList("BitnetForCausalLM", "BitNetForCausalLM"),
                                HfArchResolver.of(Arch.BITNET), HfDenseClassicHooksA::bitnetMeta,
                                HfDenseClassicHooksA::bitnetTensors,
                                HfDenseClassicHooksE::sentencePieceVocab),
                HfArchEntry.of(Arrays.asList("BloomForCausalLM", "BloomModel"), HfArchResolver.of(Arch.BLOOM),
                                HfArchTableDenseClassic::bloomMeta, HfArchTableDenseClassic::bloomTensors,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Arrays.asList("ChameleonForCausalLM", "ChameleonForConditionalGeneration"),
                                HfArchResolver.of(Arch.CHAMELEON), HfDenseClassicHooksA::chameleonMeta,
                                HfDenseClassicHooksA::chameleonTensors, HfDenseClassicHooksA::gpt2Vocab)
                        .withNameFilter(HfDenseClassicHooksA::chameleonFilter),
                HfArchEntry.of(Arrays.asList("ChatGLMForConditionalGeneration", "ChatGLMModel", "GlmForCausalLM"),
                                HfArchResolver.of(Arch.CHATGLM), HfDenseClassicHooksA::chatglmMeta,
                                HfTensorHook.DEFAULT, HfDenseClassicHooksA::chatglm3Vocab)
                        .withNameFilter(HfDenseClassicHooksA.CHATGLM_FILTER),
                HfArchEntry.of(Collections.singletonList("CodeShellForCausalLM"),
                                HfArchResolver.of(Arch.CODESHELL), HfDenseClassicHooksA::codeshellMeta,
                                HfTensorHook.DEFAULT, HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("CogVLMForCausalLM"), HfArchResolver.of(Arch.COGVLM),
                                HfLlamaHooks::meta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withExtra(HfLlamaHooks::extra),
                // ---- part B: Cohere … Exaone
                HfArchEntry.of(Collections.singletonList("Cohere2ForCausalLM"), HfArchResolver.of(Arch.COHERE2),
                                HfDenseClassicHooksB::cohere2Meta, HfDenseClassicHooksB::cohere2Tensors,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("CohereForCausalLM"), HfArchResolver.of(Arch.COMMAND_R),
                                HfDenseClassicHooksB::commandR2Meta, HfTensorHook.DEFAULT,
                                HfDenseClassicHooksA::gpt2Vocab)
                        .withConfigFixup(HfDenseClassicHooksB::commandR2Fixup),
                HfArchEntry.of(Collections.singletonList("DeciLMForCausalLM"), HfArchResolver.of(Arch.DECI),
                                HfDenseClassicHooksB::deciMeta, HfDenseClassicHooksB::deciTensors,
                                HfLlamaHooks::vocab)
                        .withConfigFixup(HfDenseClassicHooksB::deciFixup)
                        .withExtra(HfDenseClassicHooksB::deciExtra),
                HfArchEntry.of(Collections.singletonList("Exaone4_5_ForConditionalGeneration"),
                                HfArchResolver.of(Arch.EXAONE4), HfDenseClassicHooksB::exaone4Meta,
                                HfDenseClassicHooksB::exaone45Tensors, HfDenseClassicHooksB::exaone4Vocab)
                        .withExtra(HfDenseClassicHooksB::exaone4Extra),
                HfArchEntry.of(Collections.singletonList("Exaone4ForCausalLM"), HfArchResolver.of(Arch.EXAONE4),
                                HfDenseClassicHooksB::exaone4Meta, HfTensorHook.DEFAULT,
                                HfDenseClassicHooksB::exaone4Vocab)
                        .withExtra(HfDenseClassicHooksB::exaone4Extra),
                HfArchEntry.of(Collections.singletonList("ExaoneForCausalLM"), HfArchResolver.of(Arch.EXAONE),
                                HfDenseClassicHooksB::exaoneMeta, HfTensorHook.DEFAULT,
                                HfDenseClassicHooksA::gpt2Vocab),
                // ---- part C: Falcon … InternLM2
                HfArchEntry.of(Arrays.asList("FalconForCausalLM", "RWForCausalLM"), HfArchResolver.of(Arch.FALCON),
                                HfDenseClassicHooksC::falconMeta, HfDenseClassicHooksC::falconTensors,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("GPT2LMHeadModel"), HfArchResolver.of(Arch.GPT2),
                                HfDenseClassicHooksC::gpt2Meta, HfDenseClassicHooksC::gpt2Tensors,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("GPTBigCodeForCausalLM"), HfArchResolver.of(Arch.STARCODER),
                                HfDenseClassicHooksC::starcoderMeta, HfTensorHook.DEFAULT,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("GPTNeoXForCausalLM"), HfArchResolver.of(Arch.GPTNEOX),
                                HfDenseClassicHooksC::gptneoxMeta, HfDenseClassicHooksC::gptneoxTensors,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("GPTRefactForCausalLM"), HfArchResolver.of(Arch.REFACT),
                                HfDenseClassicHooksC::refactMeta, HfDenseClassicHooksC::refactTensors,
                                HfDenseClassicHooksC::refactVocab),
                HfArchEntry.of(Arrays.asList("Glm4ForCausalLM", "Glm4vForConditionalGeneration"),
                                HfArchResolver.of(Arch.GLM4), HfDenseClassicHooksC::glm4Meta,
                                HfDenseClassicHooksC::glm4Tensors, HfArchTableDenseClassic::glm4Vocab),
                HfArchEntry.of(Collections.singletonList("GlmOcrForConditionalGeneration"),
                                HfArchResolver.of(Arch.GLM4), HfDenseClassicHooksC::glmOcrMeta,
                                HfDenseClassicHooksC::glm4Tensors, HfArchTableDenseClassic::glm4Vocab),
                // ---- HK-0: GraniteModel
                HfArchEntry.of(Collections.singletonList("GraniteForCausalLM"), HfArchResolver.of(Arch.GRANITE),
                                HfGraniteHooks::meta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfGraniteHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                // GraniteSwitchModel composes the HK-0 GraniteMoe hooks over the switch adapters (granite.py:238).
                HfArchEntry.of(Collections.singletonList("GraniteSwitchForCausalLM"),
                                HfArchResolver.of(Arch.GRANITE_SWITCH), HfDenseClassicHooksF::graniteSwitchMeta,
                                HfDenseClassicHooksF::graniteSwitchTensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfGraniteHooks.FILTER)
                        .withIndexedBlockCount(HfDenseClassicHooksF::graniteSwitchBlocks)
                        .withExtra(HfLlamaHooks::extra)
                        .withUndoPermute(false),
                HfArchEntry.of(Collections.singletonList("InternLM2ForCausalLM"), HfArchResolver.of(Arch.INTERNLM2),
                                HfDenseClassicHooksE::internlm2Meta, HfDenseClassicHooksE::internlm2Tensors,
                                HfDenseClassicHooksE::internlm2Vocab),
                // ---- part D: Jais … MiniCPM
                HfArchEntry.of(Collections.singletonList("JAISLMHeadModel"), HfArchResolver.of(Arch.JAIS),
                                HfDenseClassicHooksD::jaisMeta, HfDenseClassicHooksD::jaisTensors,
                                HfDenseClassicHooksD::jaisVocab)
                        .withNameFilter(HfDenseClassicHooksD.JAIS_FILTER),
                HfArchEntry.of(Collections.singletonList("Jais2ForCausalLM"), HfArchResolver.of(Arch.JAIS2),
                                HfDenseClassicHooksD::jais2Meta, HfTensorHook.DEFAULT,
                                HfDenseClassicHooksA::gpt2Vocab)
                        .withNameFilter(HfDenseClassicHooksD.JAIS_FILTER),
                HfArchEntry.of(Arrays.asList("LlavaStableLMEpochForCausalLM", "StableLMEpochForCausalLM",
                                "StableLmForCausalLM"), HfArchResolver.of(Arch.STABLELM),
                                HfDenseClassicHooksD::stablelmMeta, HfDenseClassicHooksD::stablelmTensors,
                                HfDenseClassicHooksD::stablelmVocab)
                        .withExtra(HfDenseClassicHooksD::stablelmExtraChecks),
                HfArchEntry.of(Collections.singletonList("MPTForCausalLM"), HfArchResolver.of(Arch.MPT),
                                HfDenseClassicHooksD::mptMeta, HfDenseClassicHooksD::mptTensors,
                                HfDenseClassicHooksD::mptVocab),
                HfArchEntry.of(Collections.singletonList("MaincoderForCausalLM"),
                                HfArchResolver.of(Arch.MAINCODER), HfDenseClassicHooksD::maincoderMeta,
                                HfTensorHook.DEFAULT, HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("MiniCPMForCausalLM"), HfArchResolver.of(Arch.MINICPM),
                                HfDenseClassicHooksD::miniCPMMeta, HfDenseClassicHooksD::miniCPMTensors,
                                HfDenseClassicHooksD::miniCPMVocab)
                        .withExtra(HfDenseClassicHooksD::miniCPMExtra),
                HfArchEntry.of(Collections.singletonList("NemotronForCausalLM"), HfArchResolver.of(Arch.NEMOTRON),
                                HfDenseClassicHooksD::nemotronMeta, HfDenseClassicHooksD::nemotronTensors,
                                HfDenseClassicHooksD::nemotronVocab),
                HfArchEntry.of(Collections.singletonList("MuseGlimmerForConditionalGeneration"),
                                HfArchResolver.of(Arch.MUSE_GLIMMER), HfArchTableDenseClassic::museGlimmerMeta,
                                HfDenseClassicHooksD::museGlimmerTensors, HfDenseClassicHooksD::museGlimmerVocab),
                // ---- part E: Mpt … Xverse
                HfArchEntry.of(Collections.singletonList("OpenELMForCausalLM"), HfArchResolver.of(Arch.OPENELM),
                                HfDenseClassicHooksE::openelmMeta, HfDenseClassicHooksE::openelmTensors,
                                HfDenseClassicHooksE::openelmVocab)
                        .withConfigFixup(HfDenseClassicHooksE::openelmFixup)
                        .withBlockCount(HfArchTableDenseClassic::openelmBlocks),
                HfArchEntry.of(Collections.singletonList("OrionForCausalLM"), HfArchResolver.of(Arch.ORION),
                                HfDenseClassicHooksE::orionMeta, HfTensorHook.DEFAULT,
                                HfDenseClassicHooksE::sentencePieceVocab),
                HfArchEntry.of(Collections.singletonList("PlamoForCausalLM"), HfArchResolver.of(Arch.PLAMO),
                                HfDenseClassicHooksE::plamoMeta, HfDenseClassicHooksE::plamoTensors,
                                HfDenseClassicHooksE::sentencePieceVocab),
                HfArchEntry.of(Arrays.asList("PLaMo3ForCausalLM", "Plamo3ForCausalLM"),
                                HfArchResolver.of(Arch.PLAMO3), HfDenseClassicHooksE::plamo3Meta,
                                HfDenseClassicHooksE::plamo3Tensors, HfDenseClassicHooksE::plamo3Vocab),
                HfArchEntry.of(Collections.singletonList("PanguEmbeddedForCausalLM"),
                                HfArchResolver.of(Arch.PANGU_EMBED), HfDenseClassicHooksE::panguMeta,
                                HfDenseClassicHooksE::panguTensors, HfDenseClassicHooksE::panguVocab),
                HfArchEntry.of(Collections.singletonList("PhiForCausalLM"), HfArchResolver.of(Arch.PHI2),
                                HfDenseClassicHooksE::phi2Meta, HfTensorHook.DEFAULT,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("RuGPT3XLForCausalLM"), HfArchResolver.of(Arch.GPT2),
                                HfDenseClassicHooksC::gpt2Meta, HfDenseClassicHooksC::rugptTensors,
                                HfDenseClassicHooksA::gpt2Vocab)
                        .withExtra(HfDenseClassicHooksC::rugptExtra),
                HfArchEntry.of(Collections.singletonList("Spark2_5ForCausalLM"), HfArchResolver.of(Arch.SPARK2_5),
                                HfDenseClassicHooksF::spark25Meta, HfDenseClassicHooksF::spark25Tensors,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("Starcoder2ForCausalLM"),
                                HfArchResolver.of(Arch.STARCODER2), HfDenseClassicHooksF::starcoder2Meta,
                                HfTensorHook.DEFAULT, HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("TalkieForCausalLM"), HfArchResolver.of(Arch.TALKIE),
                                HfDenseClassicHooksF::talkieMeta, HfDenseClassicHooksF::talkieTensors,
                                HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("XverseForCausalLM"), HfArchResolver.of(Arch.XVERSE),
                                HfDenseClassicHooksF::xverseMeta, HfDenseClassicHooksF::xverseTensors,
                                HfDenseClassicHooksF::xverseVocab)));
    }

    // ------------------------------------------------------------------ class wiring helpers

    /**
     * {@code BloomModel.set_gguf_parameters} (bloom.py:18-32), hand-set: the {@code seq_length} context with the
     * width as fallback, the {@code hidden_size | n_embed} width and 4x FFN, the heads over the same count as
     * key/value heads, and the {@code layer_norm_epsilon} epsilon.
     *
     * @throws GgufFormatException when a head or width count is missing
     */
    private static void bloomMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int embed = cfg.findInt("hidden_size", "n_embed").orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'hidden_size' or 'n_embed' is missing"));
        int heads = cfg.findInt("n_head", "num_attention_heads").orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'n_head' or 'num_attention_heads' is"
                        + " missing"));
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.findInt(embed, "seq_length"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, embed);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * embed);
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, heads);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, cfg.findFloat("layer_norm_epsilon").orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'layer_norm_epsilon' is missing")));
    }

    /**
     * The Bloom fused {@code query_key_value} layout (bloom.py:51-71, the {@code [nHead, 3, headDim, ...]}
     * regroup of {@link HfDenseClassicHooksA#interleavedQkvTensors}), then the inherited mapping for every other
     * tensor.
     */
    private static void bloomTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfDenseClassicHooksA.interleavedQkvTensors(ctx, hfName, src, out);
        if (!out.isConsumed(hfName)) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
        }
    }

    /**
     * The {@code Glm4Model.set_vocab} vocabulary (glm.py:24-48): the GLM one.
     */
    private static java.util.Map<String, Object> glm4Vocab(HfLoadContext ctx, com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver pre)
            throws IOException {
        return HfVocabWriters.glm(HfHookSupport.vocabInput(ctx), pre);
    }

    /** The muse eot id of {@code MuseGlimmerModel.set_vocab} (muse_glimmer.py:39-45) over the parameters. */
    private static void museGlimmerMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfDenseClassicHooksD.museGlimmerMeta(ctx, kv, plan);
        HfDenseClassicHooksD.museGlimmerEot(ctx, kv, plan);
    }

    /**
     * {@code OpenELMModel.find_hparam}'s layer count (openelm.py:61-64): the block count is
     * {@code num_transformer_layers}.
     */
    private static int openelmBlocks(HfConfig cfg) {
        return cfg.requireInt("num_transformer_layers");
    }
}
