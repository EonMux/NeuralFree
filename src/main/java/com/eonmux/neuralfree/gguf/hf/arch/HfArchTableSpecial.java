package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBertHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBertVariantHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfEncoderHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGemmaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGemma4Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfT5Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenLinearHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries of Appendix A table HK-SP, the encoders and embedding models
 * (LC/conversion/{bert,clef,gemma,llama,t5,vllm_decision2}.py): the BERT family with its RoBERTa/XLM-R
 * variant rows, the ModernBert/Nomic/Neo/Euro/Jina classes, the EmbeddingGemma embedders on the HK-B2 hooks,
 * the Qwen3 decision heads and the T5 classes. {@code LlamaEmbedNemotronModel} composes the HK-0 Llama hooks
 * on the LLAMA_EMBED arch. {@link HfArchRegistry} merges this table with the other tables; a Hugging Face
 * name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableSpecial {

    private HfArchTableSpecial() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // BertModel (bert.py:19): BERT
                HfArchEntry.of(Arrays.asList("BertForMaskedLM", "BertForSequenceClassification", "BertModel",
                                "CamembertModel"), HfArchResolver.of(Arch.BERT),
                        HfBertHooks::meta, HfBertHooks::tensors, HfBertHooks::vocab)
                        .withNameFilter(HfBertHooks.FILTER),
                // DistilBertModel (bert.py:245): BERT
                HfArchEntry.of(Arrays.asList("DistilBertForMaskedLM", "DistilBertForSequenceClassification",
                                "DistilBertModel"), HfArchResolver.of(Arch.BERT),
                        HfBertVariantHooks::distilMeta, HfBertHooks::tensors, HfBertVariantHooks::distilVocab)
                        .withNameFilter(HfBertVariantHooks::distilFilter),
                // EmbeddingGemma2Model (gemma.py:815): GEMMA_EMBEDDING2 on the HK-B2 Gemma 4 hooks
                HfArchEntry.of(Collections.singletonList("EmbeddingGemma2Model"),
                                HfArchResolver.of(Arch.GEMMA_EMBEDDING2), HfEncoderHooks::embeddingGemma2Meta,
                                HfGemma4Hooks::tensors, HfGemmaHooks::vocab)
                        .withConfigFixup(HfEncoderHooks::embeddingGemma2Fixup)
                        .withIndex(HfEncoderHooks::index)
                        .withNameFilter(HfGemma4Hooks::filter),
                // EuroBertModel (bert.py:440): EUROBERT
                HfArchEntry.of(Arrays.asList("EuroBertModel", "JinaEmbeddingsV5Model"),
                                HfArchResolver.of(Arch.EUROBERT), HfBertVariantHooks::euroMeta,
                                HfBertHooks::tensors, HfBertVariantHooks::euroVocab)
                        .withNameFilter(HfBertVariantHooks::euroFilter),
                // EmbeddingGemma (gemma.py:186): GEMMA_EMBEDDING on the HK-B2 Gemma hooks
                HfArchEntry.of(Collections.singletonList("Gemma3TextModel"),
                                HfArchResolver.of(Arch.GEMMA_EMBEDDING), HfEncoderHooks::embeddingGemmaMeta,
                                HfGemmaHooks::tensors, HfGemmaHooks::vocab)
                        .withIndex(HfEncoderHooks::index),
                // JinaBertV2Model (bert.py:572): JINA_BERT_V2
                HfArchEntry.of(Arrays.asList("JinaBertForMaskedLM", "JinaBertModel"),
                                HfArchResolver.of(Arch.JINA_BERT_V2), HfBertHooks::meta, HfBertHooks::tensors,
                                HfBertVariantHooks::jinaV2Vocab)
                        .withNameFilter(HfBertHooks.FILTER),
                // LlamaEmbedNemotronModel (llama.py:422): LLAMA_EMBED on the HK-0 Llama hooks
                HfArchEntry.of(Collections.singletonList("LlamaBidirectionalModel"),
                                HfArchResolver.of(Arch.LLAMA_EMBED), HfLlamaHooks::meta, HfLlamaHooks::tensors,
                                HfLlamaHooks::vocab)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                // T5Model (t5.py:20): T5
                HfArchEntry.of(Arrays.asList("MT5ForConditionalGeneration", "T5ForConditionalGeneration",
                                "T5WithLMHeadModel", "UMT5ForConditionalGeneration", "UMT5Model"),
                                HfArchResolver.of(Arch.T5), HfT5Hooks::meta, HfT5Hooks::tensors,
                                HfT5Hooks::vocab),
                // ModernBertDecisionModel (bert.py:680): MODERN_BERT
                HfArchEntry.of(Collections.singletonList("ModernBertDecisionModel"),
                                HfArchResolver.of(Arch.MODERN_BERT), HfBertVariantHooks::decisionMeta,
                                HfBertVariantHooks::decisionTensors, HfBertVariantHooks::decisionVocab)
                        .withConfigFixup(HfBertVariantHooks::decisionFixup)
                        .withNameFilter(HfBertVariantHooks::decisionFilter),
                // ModernBertModel (bert.py:600): MODERN_BERT
                HfArchEntry.of(Arrays.asList("ModernBertForMaskedLM",
                                "ModernBertForSequenceClassification", "ModernBertModel"),
                                HfArchResolver.of(Arch.MODERN_BERT), HfBertVariantHooks::modernMeta,
                                HfBertVariantHooks::modernTensors, HfBertVariantHooks::modernVocab)
                        .withNameFilter(HfBertVariantHooks::modernFilter),
                // NeoBert (bert.py:408): NEO_BERT
                HfArchEntry.of(Arrays.asList("NeoBERT", "NeoBERTForSequenceClassification", "NeoBERTLMHead"),
                                HfArchResolver.of(Arch.NEO_BERT), HfBertVariantHooks::neoMeta,
                                HfBertHooks::tensors, HfBertHooks::vocab)
                        .withNameFilter(HfBertVariantHooks::neoFilter),
                // NomicBertModel (bert.py:319): NOMIC_BERT or NOMIC_BERT_MOE (bert.py:327-328)
                HfArchEntry.of(Collections.singletonList("NomicBertModel"),
                                HfBertVariantHooks::nomicArch, HfBertVariantHooks::nomicMeta,
                                HfBertVariantHooks::nomicTensors, HfBertVariantHooks::nomicVocab)
                        .withConfigFixup(HfBertVariantHooks::nomicFixup)
                        .withNameFilter(HfBertVariantHooks::nomicFilter),
                // RobertaModel (bert.py:269): BERT
                HfArchEntry.of(Arrays.asList("RobertaForSequenceClassification", "RobertaModel"),
                                HfArchResolver.of(Arch.BERT), HfBertHooks::meta, HfBertHooks::robertaTensor,
                                HfBertHooks::robertaVocab)
                        .withConfigFixup(HfBertHooks::fixup)
                        .withNameFilter(HfBertHooks::robertaFilter),
                // T5EncoderModel (t5.py:158): T5ENCODER
                HfArchEntry.of(Collections.singletonList("T5EncoderModel"), HfArchResolver.of(Arch.T5ENCODER),
                                HfT5Hooks::encoderMeta, HfT5Hooks::encoderTensors, HfT5Hooks::vocab),
                // VllmDecision2Model (vllm_decision2.py:38): VLLM_DECISION2 on the HK-B1 Qwen3 hooks
                HfArchEntry.of(Collections.singletonList("VllmDecision2Model"),
                                HfArchResolver.of(Arch.VLLM_DECISION2), HfEncoderHooks::vllmDecision2Meta,
                                HfEncoderHooks::vllmDecision2Tensors, HfQwenHooks::qwen3Vocab)
                        .withConfigFixup(HfEncoderHooks::vllmDecision2Fixup)
                        .withIndex(HfEncoderHooks::vllmDecision2Index)
                        .withForceType(HfEncoderHooks::decisionHeadForceType),
                // ClefModel (clef.py:34): CLEF on the Qwen3.5 metadata hooks; a Clef checkpoint keeps the
                // tokenizer of the Qwen family, whose set_vocab is HfQwenHooks::qwen3Vocab
                HfArchEntry.of(Collections.singletonList("ClefModel"), HfArchResolver.of(Arch.CLEF),
                                HfEncoderHooks::clefMeta, HfEncoderHooks::clefTensors, HfQwenHooks::qwen3Vocab)
                        .withConfigFixup(HfEncoderHooks::clefFixup)
                        .withIndex(HfEncoderHooks::clefIndex)
                        .withForceType(HfEncoderHooks::decisionHeadForceType),
                // XLMRobertaModel (bert.py:469): BERT, or JINA_BERT_V3 with lora_adaptations
                HfArchEntry.of(Arrays.asList("XLMRobertaForSequenceClassification", "XLMRobertaModel"),
                                HfBertHooks::xlmRobertaArch, HfBertHooks::meta, HfBertHooks::xlmRobertaTensor,
                                HfBertHooks::xlmRobertaVocabEntry)
                        .withConfigFixup(HfBertHooks::fixup)
                        .withNameFilter(HfBertHooks.FILTER_XLMR)));
    }
}
