package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfAsrForcedAlignerHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBertHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBertVariantHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBgeM3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfClipHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfColBertHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksC;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksE;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfAxk1Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDomynEdgeHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDoryHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfIQuestHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGptjHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfHyperclovaxHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfInklingHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfJinaVlHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfPanguMoeHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSarvamHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfStep1Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfStep3TextHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfStep5Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLongcatFlashHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMistralLarge3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfOptHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfParam2MoeHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfPoolingHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfRnj1Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSolarHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSpladeHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSsmHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSsmHybridHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfVllmAliasHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the names vLLM registers over model families NeuralFree already converts
 * (V/model_executor/models/registry.py), each composed from the family's hooks and {@link HfVllmAliasHooks}, plus the
 * vLLM-only text families llama.cpp's converter does not list (GPT-J, gpt_j.py and OPT, opt.py). Names that are plain
 * aliases of an entry with nothing to add are listed on that entry instead. {@link HfArchRegistry} merges this table
 * with the other tables; a Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableVllm {

    private HfArchTableVllm() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // MistralModel -> LlamaForCausalLM under vLLM's embedding adapter: headless Llama names.
                HfArchEntry.of(Collections.singletonList("MistralModel"), HfLlamaHooks::arch,
                                HfVllmAliasHooks.withEmbeddingPooling(HfLlamaHooks::meta), HfLlamaHooks::tensors,
                                HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfVllmAliasHooks.headless(HfLlamaHooks.FILTER))
                        .withExtra(HfLlamaHooks::extra),
                // TeleChat3ForCausalLM -> LlamaForCausalLM with the telechat3-yarn attention factor.
                HfArchEntry.of(Collections.singletonList("TeleChat3ForCausalLM"), HfLlamaHooks::arch,
                                HfVllmAliasHooks::teleChat3Meta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                // TeleFLMForCausalLM (teleflm.py): Llama with the muP multipliers on the Granite architecture,
                // which has no rope_freqs tensor, so the Llama extra tensors are not emitted.
                HfArchEntry.of(Collections.singletonList("TeleFLMForCausalLM"), HfArchResolver.of(Arch.GRANITE),
                                HfVllmAliasHooks::teleFlmMeta, HfLlamaHooks::tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfLlamaHooks::fixup)
                        .withNameFilter(HfLlamaHooks.FILTER),
                // TeleChat2ForCausalLM (telechat2.py): Llama under the transformer.h names.
                HfArchEntry.of(Collections.singletonList("TeleChat2ForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                                HfLlamaHooks::meta, HfVllmAliasHooks::teleChat2Tensors, HfLlamaHooks::vocab)
                        .withConfigFixup(HfVllmAliasHooks::teleChat2Fixup)
                        .withNameFilter(HfVllmAliasHooks::teleChat2Filter)
                        .withExtra(HfLlamaHooks::extra),
                // Gemma2Model -> Gemma2ForCausalLM under vLLM's embedding adapter: headless Gemma2 names.
                HfArchEntry.of(Collections.singletonList("Gemma2Model"), HfArchResolver.of(Arch.GEMMA2),
                                HfVllmAliasHooks.withEmbeddingPooling(HfDenseHooksA::gemma2Meta),
                                HfDenseHooksA::gemmaTensors, HfDenseHooksA::gemma2Vocab)
                        .withNameFilter(HfVllmAliasHooks.headless(HfDenseHooksA.GEMMA_FILTER)),
                // MiMoForCausalLM (mimo.py) and MiMoMTPModel, the draft-model view of the same checkpoint (mimo_mtp.py
                // MiMoMTP): Qwen2 whose model.mtp_layers become the appended NextN blocks of the draft head. vLLM
                // serves the checkpoint itself as its MTP draft (MiMoMTP.load_weights keeps the MTP layers, the
                // shared embedding and the head), so the trunk keeps serving the model and the blocks draft for it
                // (speculative.cpp:2655-2690).
                HfArchEntry.of(Arrays.asList("MiMoForCausalLM", "MiMoMTPModel"), HfArchResolver.of(Arch.QWEN2),
                                HfVllmAliasHooks::mimoMeta, HfVllmAliasHooks::mimoTensors, HfQwenHooks::qwen2Vocab)
                        .withIndexedBlockCount(HfVllmAliasHooks::mimoBlockCount)
                        .withLoadNameFilter(HfVllmAliasHooks::mimoFilter)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // DeepseekV3BidirectionalModel -> DeepseekV3ForCausalLM under vLLM's embedding adapter: headless
                // DeepSeek names; the configuration's is_causal false reaches the attention.causal key.
                HfArchEntry.of(Collections.singletonList("DeepseekV3BidirectionalModel"), HfDeepseekHooks::arch,
                                HfVllmAliasHooks.withEmbeddingPooling(HfDeepseekHooks::meta),
                                HfDeepseekHooks::tensors, HfDeepseekHooks::vocab)
                        .withNameFilter(HfVllmAliasHooks.headless(HfDeepseekHooks::filter))
                        .withExtra(HfDeepseekHooks::extra),
                // MistralLarge3ForCausalLM (mistral_large_3.py): DeepSeek-V3 with the native Mistral names and the
                // llama_4_scaling query temperature.
                HfArchEntry.of(Collections.singletonList("MistralLarge3ForCausalLM"), HfDeepseekHooks::arch,
                                HfMistralLarge3Hooks::meta, HfDeepseekHooks::tensors, HfMistralLarge3Hooks::vocab)
                        .withIndex(HfMistralLarge3Hooks::index)
                        .withLoadNameFilter(ctx -> HfDeepseekHooks::filter)
                        .withExtra(HfDeepseekHooks::extra),
                // JinaVLForRanking (jina_vl.py JinaVLForSequenceClassification): the Qwen2-VL text model with the
                // ReLU score head, the document-first score template and the appended score token.
                HfArchEntry.of(Arrays.asList(HfJinaVlHooks.JINA_VL_RANKING,
                                HfJinaVlHooks.JINA_VL_SEQUENCE_CLASSIFICATION), HfArchResolver.of(Arch.QWEN2VL),
                                HfJinaVlHooks::jinaVlMeta, HfJinaVlHooks::jinaVlTensors, HfDenseHooksB::qwen2VlVocab)
                        .withNameFilter(HfDenseHooksB.THINKER_FILTER),
                // BertForTokenClassification (bert.py): the single classifier projection over every token.
                HfArchEntry.of(Collections.singletonList("BertForTokenClassification"),
                                HfArchResolver.of(Arch.BERT), HfBertHooks::meta, HfBertHooks::tokenTensor,
                                HfBertHooks::vocab)
                        .withNameFilter(HfBertHooks.FILTER),
                // RobertaForTokenClassification / XLMRobertaForTokenClassification (roberta.py): the same head over
                // the RoBERTa and XLM-R backbones, including the position-embedding chop.
                HfArchEntry.of(Collections.singletonList("RobertaForTokenClassification"),
                                HfArchResolver.of(Arch.BERT), HfBertHooks::meta, HfBertHooks::robertaTokenTensor,
                                HfBertHooks::robertaVocab)
                        .withConfigFixup(HfBertHooks::fixup)
                        .withNameFilter(HfBertHooks::robertaFilter),
                HfArchEntry.of(Collections.singletonList("XLMRobertaForTokenClassification"),
                                HfBertHooks::xlmRobertaArch, HfBertHooks::meta,
                                HfBertHooks::xlmRobertaTokenTensor, HfBertHooks::xlmRobertaVocabEntry)
                        .withConfigFixup(HfBertHooks::fixup)
                        .withNameFilter(HfBertHooks.FILTER_XLMR),
                // ModernBertForTokenClassification (modernbert.py): the prediction head (dense, activation, norm)
                // followed by the classifier, applied to every token.
                HfArchEntry.of(Collections.singletonList("ModernBertForTokenClassification"),
                                HfArchResolver.of(Arch.MODERN_BERT), HfBertVariantHooks::modernMeta,
                                HfBertVariantHooks::modernTokenTensor, HfBertVariantHooks::modernVocab)
                        .withNameFilter(HfBertVariantHooks::modernFilter),
                // Step1ForCausalLM (step1.py): no rotary position, vLLM's square-root ALiBi over the full head,
                // RMSNorm and a gated SiLU FFN with optional projection biases.
                HfArchEntry.of(Collections.singletonList("Step1ForCausalLM"), HfStep1Hooks::arch,
                                HfStep1Hooks::meta, HfStep1Hooks::tensors, HfDenseClassicHooksA::gpt2Vocab)
                        .withConfigFixup(HfStep1Hooks::fixup),
                // PanguProMoEV2ForCausalLM / PanguUltraMoEForCausalLM (openpangu.py): the DeepSeek-V2 MLA layout when
                // the configuration carries the MLA geometry, the pangu-embedded attention otherwise; sink
                // attention, sandwich norms and interleaved sliding windows are refused by name.
                HfArchEntry.of(Arrays.asList("PanguProMoEV2ForCausalLM", "PanguUltraMoEForCausalLM"),
                                HfPanguMoeHooks::arch, HfPanguMoeHooks::meta, HfPanguMoeHooks::tensors,
                                HfPanguMoeHooks::vocab)
                        .withConfigFixup(HfPanguMoeHooks::fixup)
                        .withExtra(HfDeepseekHooks::extra),
                // SarvamMLAForCausalLM (sarvam.py): a DeepSeek-V2-style MLA trunk with routed and one shared expert,
                // sigmoid routing and a zero-mean router bias; the released checkpoints omit the routing defaults.
                HfArchEntry.of(Collections.singletonList("SarvamMLAForCausalLM"), HfArchResolver.of(Arch.DEEPSEEK2),
                                HfSarvamHooks::meta, HfSarvamHooks::tensors, HfDeepseekHooks::vocab)
                        .withConfigFixup(HfSarvamHooks::fixup)
                        .withNameFilter(HfDeepseekHooks::filter)
                        .withExtra(HfDeepseekHooks::extra),
                // GPTJForCausalLM (gpt_j.py): a vLLM-only text family with no llama.cpp converter. Separate,
                // bias-free q/k/v projections, adjacent-pair rotary over the first rotary_dim dims of each head,
                // biased GELU MLP and output head, and a parallel residual over the shared normed input.
                HfArchEntry.of(Collections.singletonList("GPTJForCausalLM"), HfArchResolver.of(Arch.GPTJ),
                                HfGptjHooks::meta, HfGptjHooks::tensors, HfDenseClassicHooksA::gpt2Vocab)
                        .withNameFilter(HfGptjHooks::filter),
                // SolarForCausalLM (solar.py): the Llama graph with the block-skip connections and a logit
                // multiplier refused by name.
                HfArchEntry.of(Collections.singletonList("SolarForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                                HfSolarHooks::meta, HfSolarHooks::tensors, HfSolarHooks::vocab)
                        .withConfigFixup(HfSolarHooks::fixup)
                        .withNameFilter(HfSolarHooks::filter)
                        .withExtra(HfLlamaHooks::extra),
                // HyperCLOVAXForCausalLM (hyperclovax.py): a vLLM-only text family with no llama.cpp converter. Llama
                // attention and SwiGLU with the MuP embedding, residual and logit multipliers and a post norm after
                // each sub-layer output.
                HfArchEntry.of(Collections.singletonList("HyperCLOVAXForCausalLM"),
                                HfArchResolver.of(Arch.HYPERCLOVAX), HfHyperclovaxHooks::meta,
                                HfHyperclovaxHooks::tensors, HfLlamaHooks::vocab)
                        .withNameFilter(HfHyperclovaxHooks::filter)
                        .withUndoPermute(false),
                // DomynEdgeForCausalLM (domyn_edge.py): a vLLM-only text family with no llama.cpp converter. Qwen2
                // with sandwich norms, a weightless per-head Q norm, a per-layer RoPE switch and per-layer head
                // widths over one shared sliding window.
                HfArchEntry.of(Collections.singletonList("DomynEdgeForCausalLM"),
                                HfArchResolver.of(Arch.DOMYN_EDGE), HfDomynEdgeHooks::meta,
                                HfDomynEdgeHooks::tensors, HfQwenHooks::qwen2Vocab)
                        .withNameFilter(HfDomynEdgeHooks::filter)
                        .withUndoPermute(false),
                // Rnj1ForCausalLM (rnj1.py): the Gemma 3 block (Q/K head norms, sandwich norms, a gated
                // tanh-GELU MLP, the sqrt-wide token scale and the final logit soft cap) with chunked local
                // attention, whose layers see their aligned chunk and the one before it.
                HfArchEntry.of(Collections.singletonList("Rnj1ForCausalLM"),
                                HfRnj1Hooks::arch, HfRnj1Hooks::meta, HfRnj1Hooks::tensors, HfRnj1Hooks::vocab)
                        .withUndoPermute(false),
                // DoryForCausalLM (dory.py): nGPT blocks with justnorm residual helpers, alternating attention-only
                // and MLP-only layers, and the block [n_input, n_input + n_recurrent) repeated n_recurrent_loops
                // times with one KV cache per pass. The learned nGPT scales are folded into the projections.
                HfArchEntry.of(Collections.singletonList("DoryForCausalLM"),
                                HfDoryHooks::arch, HfDoryHooks::meta, HfDoryHooks::tensors, HfDoryHooks::vocab)
                        .withUndoPermute(false),
                // IQuestLoopCoderForCausalLM (iquest_loopcoder.py): a Llama block repeated loop_num times, later
                // passes mixing a read-only attention over the first pass's cache with a sliding-window local
                // attention over their own cache, gated per head by gate_projections.<l>.gate_proj.
                HfArchEntry.of(Collections.singletonList("IQuestLoopCoderForCausalLM"),
                                HfIQuestHooks::arch, HfIQuestHooks::meta, HfIQuestHooks::tensors,
                                HfIQuestHooks::vocab)
                        .withUndoPermute(false),
                // AXK1ForCausalLM (AXK1.py): the DeepSeek-V2 MLA and routed experts with the llama 4 query
                // temperature and a post-MLP norm on the sparse layers only.
                HfArchEntry.of(Collections.singletonList("AXK1ForCausalLM"),
                                HfAxk1Hooks::arch, HfAxk1Hooks::meta, HfAxk1Hooks::tensors, HfAxk1Hooks::vocab)
                        .withNameFilter(HfAxk1Hooks::filter)
                        .withExtra(HfAxk1Hooks::extra),
                // InklingForCausalLM (inkling): a vLLM-only text family with no llama.cpp converter. NoPE attention
                // with per-query relative logits (a query-side attn_rel projection widened by one shared
                // [d_rel, rel_extent] table), per-head Q/K RMS norms and depthwise short convolutions on the K/V,
                // attention-output and MLP-output streams. The fused attn.qkvr rows split into q/k/v/rel and the
                // interleaved gate_up rows into gate/up; the multimodal towers are dropped.
                HfArchEntry.of(Arrays.asList("InklingForCausalLM", "InklingForConditionalGeneration"),
                                HfInklingHooks::arch, HfInklingHooks::meta, HfInklingHooks::tensors,
                                HfInklingHooks::vocab)
                        .withNameFilter(HfInklingHooks::filter)
                        .withUndoPermute(false),
                // Param2MoEForCausalLM (param2moe.py): the dots1 layout, whose dense lead, biased sigmoid routing,
                // shared expert and fused QKV match. Grouped routing and NextN heads are refused by name.
                HfArchEntry.of(Collections.singletonList("Param2MoEForCausalLM"), HfArchResolver.of(Arch.DOTS1),
                                HfParam2MoeHooks::meta, HfParam2MoeHooks::tensors, HfParam2MoeHooks::vocab)
                        .withConfigFixup(HfParam2MoeHooks::fixup)
                        .withNameFilter(HfParam2MoeHooks::filter)
                        .withExtra(HfParam2MoeHooks::extra)
                        .withUndoPermute(false),
                // OPTForCausalLM (opt.py): another vLLM-only text family with no llama.cpp converter. Learned
                // positions offset by two padding rows, separate biased q/k/v projections, an ungated ReLU MLP,
                // optional project_in/project_out, and pre- or post-residual block norms (the 350M variant).
                HfArchEntry.of(Collections.singletonList("OPTForCausalLM"), HfArchResolver.of(Arch.OPT),
                                HfOptHooks::meta, HfOptHooks::tensors, HfDenseClassicHooksA::gpt2Vocab)
                        .withNameFilter(HfOptHooks::filter),
                // Qwen2ForRewardModel / Qwen2ForProcessRewardModel (qwen2_rm.py): the two score projections with
                // their ReLU, over every token (vLLM's default token pooling ALL).
                HfArchEntry.of(Collections.singletonList("Qwen2ForRewardModel"), HfArchResolver.of(Arch.QWEN2),
                                HfPoolingHooks::qwen2RewardMeta, HfPoolingHooks::qwen2RewardTensor,
                                HfQwenHooks::qwen2Vocab),
                HfArchEntry.of(Collections.singletonList("Qwen2ForProcessRewardModel"),
                                HfArchResolver.of(Arch.QWEN2), HfPoolingHooks::qwen2ProcessMeta,
                                HfPoolingHooks::qwen2RewardTensor, HfQwenHooks::qwen2Vocab),
                // The ColBERT late-interaction classes (colbert.py): every token projected through the
                // linear / colbert_linear weights and L2-normalized (HfColBertHooks), one entry per backbone.
                HfArchEntry.of(Collections.singletonList("HF_ColBERT"), HfArchResolver.of(Arch.BERT),
                                HfColBertHooks::bertMeta, HfColBertHooks::bertTensor, HfBertHooks::vocab)
                        .withNameFilter(HfBertHooks.FILTER),
                HfArchEntry.of(Collections.singletonList("ColBERTModernBertModel"),
                                HfArchResolver.of(Arch.MODERN_BERT), HfColBertHooks::modernMeta,
                                HfColBertHooks::modernTensor, HfBertVariantHooks::modernVocab)
                        .withNameFilter(HfBertVariantHooks::modernFilter),
                HfArchEntry.of(Collections.singletonList("ColBERTJinaRobertaModel"),
                                HfBertHooks::xlmRobertaArch, HfColBertHooks::bertMeta,
                                HfColBertHooks::xlmRobertaTensor, HfBertHooks::xlmRobertaVocabEntry)
                        .withConfigFixup(HfBertHooks::fixup)
                        .withNameFilter(HfBertHooks.FILTER_XLMR),
                HfArchEntry.of(Collections.singletonList("ColBERTLfm2Model"), HfSsmHybridHooksB::lfm2Arch,
                                HfColBertHooks::lfm2Meta, HfColBertHooks::lfm2Tensor,
                                HfSsmHybridHooksB::colBertVocab)
                        .withConfigFixup(HfSsmHybridHooksB::lfm2Fixup)
                        .withNameFilter(HfSsmHybridHooksB.LFM2_DENSE_FILTER),
                // BertSpladeSparseEmbeddingModel (bert.py:684): the masked-LM head as the classifier whose
                // per-token logits reduce into the sparse embedding (HfSpladeHooks).
                HfArchEntry.of(Collections.singletonList("BertSpladeSparseEmbeddingModel"),
                                HfArchResolver.of(Arch.BERT), HfSpladeHooks::meta, HfSpladeHooks::tensors,
                                HfSpladeHooks::vocab)
                        .withNameFilter(HfSpladeHooks.FILTER),
                // InternLM2ForRewardModel (internlm2.py): the v_head reward projection over every token.
                HfArchEntry.of(Collections.singletonList("InternLM2ForRewardModel"),
                                HfArchResolver.of(Arch.INTERNLM2), HfPoolingHooks::internlm2RewardMeta,
                                HfPoolingHooks::internlm2RewardTensor, HfDenseClassicHooksE::internlm2Vocab),
                // GPT2ForSequenceClassification / JambaForSequenceClassification: the score projection over the
                // last row (vLLM's default LAST sequence pooling).
                HfArchEntry.of(Collections.singletonList("GPT2ForSequenceClassification"),
                                HfArchResolver.of(Arch.GPT2), HfPoolingHooks::gpt2SeqMeta,
                                HfPoolingHooks::gpt2SeqTensor, HfDenseClassicHooksA::gpt2Vocab),
                HfArchEntry.of(Collections.singletonList("JambaForSequenceClassification"),
                                HfSsmHooksA::jambaArch, HfPoolingHooks::jambaSeqMeta,
                                HfPoolingHooks::jambaSeqTensor, HfSsmHooksA::jambaVocab),
                // LlamaBidirectionalForSequenceClassification (llama.py:574): vLLM's as_seq_cls_model over the
                // bidirectional Llama, the score projection over the last row.
                HfArchEntry.of(Collections.singletonList("LlamaBidirectionalForSequenceClassification"),
                                HfArchResolver.of(Arch.LLAMA_EMBED), HfPoolingHooks::llamaBidirectionalSeqMeta,
                                HfPoolingHooks::llamaSeqTensor, HfLlamaHooks::vocab)
                        .withNameFilter(HfLlamaHooks.FILTER)
                        .withExtra(HfLlamaHooks::extra),
                // CLIPModel / SiglipModel (clip.py CLIPEmbeddingModel, siglip.py SiglipEmbeddingModel): both towers as
                // the two-tower clip file of the legacy convert_image_encoder_to_gguf.py.
                HfArchEntry.of(Arrays.asList(HfClipHooks.CLIP_MODEL, HfClipHooks.SIGLIP_MODEL),
                                HfArchResolver.of(Arch.CLIP), HfClipHooks::clipMeta, HfClipHooks::clipTensors,
                                HfClipHooks::clipVocab)
                        .withConfigFixup(HfClipHooks::fixup)
                        .withNameFilter(HfClipHooks.FILTER),
                // Step3TextForCausalLM (step3_text.py): the shared query projection is RMS-normalized and widened
                // per head (inter_norm and wq) before RoPE, and every MoE layer adds a plain shared expert to the
                // routed sum.
                HfArchEntry.of(Collections.singletonList("Step3TextForCausalLM"), HfStep3TextHooks::arch,
                                HfStep3TextHooks::meta, HfStep3TextHooks::tensors,
                                HfDenseClassicHooksA::gpt2Vocab)
                        .withConfigFixup(HfStep3TextHooks::fixup),
                // Step5ForCausalLM / Step5MTP (vllm/models/step5): GQA attention with a per-head output gate and
                // zero-centered RMS norms, the weighted-ReLU sparse indexer over 8-token region summaries, dense and
                // routed-plus-shared-expert FFNs, and the appended NextN/MTP block with its fusion projections and
                // shared head. Step5MTP is the draft-model view of the same checkpoint (mtp.py Step5MultiTokenPredictor).
                HfArchEntry.of(Arrays.asList("Step5ForCausalLM", "Step5MTP"), HfStep5Hooks::arch,
                                HfStep5Hooks::meta, HfStep5Hooks::tensors, HfQwenHooks::qwen2Vocab)
                        .withConfigFixup(HfStep5Hooks::fixup)
                        .withIndexedBlockCount(HfStep5Hooks::blocks)
                        .withNameFilter(HfStep5Hooks.STEP5_FILTER)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep("model.embed_tokens.weight", "lm_head.weight"),
                // The remaining vLLM classifier names are recognized so a checkpoint names the missing backbone
                // instead of falling through to the converter's "unknown architecture" error.
                HfArchEntry.rejected(Collections.singletonList("GteNewForSequenceClassification"),
                        "'GteNewForSequenceClassification' needs the GTE-new BERT-with-RoPE backbone"
                                + " (bert_with_rope.py:559-598: packed qkv, a rotary encoder with an interleaved"
                                + " GPT-J-style rotation, a gated up/gate MLP split from up_gate_proj); NeuralFree has"
                                + " no rotary-encoder BERT graph (its BERT, Jina-BERT-v2 and Nomic-BERT graphs use"
                                + " absolute positions)"),
                HfArchEntry.rejected(Collections.singletonList("OpenAIPrivacyFilterForTokenClassification"),
                        "'OpenAIPrivacyFilterForTokenClassification' needs the gpt-oss backbone as an encoder-only"
                                + " model (openai_privacy_filter.py:28-60): every layer attends both ways inside a"
                                + " symmetric +-sliding_window band, with GPT-J interleaved RoPE. NeuralFree's gpt-oss"
                                + " graph (OPENAI_MOE) is causal, alternates a causal sliding window per layer and"
                                + " rotates with NeoX pairs, and its non-causal mask is full, not banded"),
                HfArchEntry.rejected(Collections.singletonList("LlamaNemotronVLForSequenceClassification"),
                        "'LlamaNemotronVLForSequenceClassification' needs the Llama-Nemotron-VL projector"
                                + " (nemotron_vl.py:122-260): a SigLIP vision tower behind the InternVL pixel-shuffle"
                                + " and an mlp1 LayerNorm/Linear/GELU/Linear merger, with the bidirectional Llama text"
                                + " weights nested under language_model.model; NeuralFree converts CLIP/SigLIP"
                                + " two-tower files and LLAMA_EMBED text files, not that merged tower layout"),
                HfArchEntry.rejected(Collections.singletonList("ClefOmniForJointSchema"),
                        "'ClefOmniForJointSchema' needs the joint-schema head (clef_joint_schema.py:100-358): a"
                                + " schema-spanned TransformerDecoder over question and option embeddings, shipped in"
                                + " joint_head.safetensors and driven by the request's joint_schema spans. NeuralFree's"
                                + " clef conversion is the ClefModel/ClefFlashForJointSchema decision head, a"
                                + " different graph; the Qwen3-Omni-MoE thinker alone is not that head"),
                // BgeM3EmbeddingModel (roberta.py:187): the XLM-RoBERTa backbone with the colbert_linear and
                // sparse_linear side projections read from their own .pt files.
                HfArchEntry.of(Collections.singletonList(HfBgeM3Hooks.ARCH), HfBertHooks::xlmRobertaArch,
                                HfBgeM3Hooks::meta, HfBertHooks::xlmRobertaTensor, HfBgeM3Hooks::vocab)
                        .withConfigFixup(HfBertHooks::fixup)
                        .withNameFilter(HfBgeM3Hooks.FILTER)
                        .withExtra(HfBgeM3Hooks::sideTensors),
                // Qwen3ASRForcedAlignerForTokenClassification (qwen3_asr_forced_aligner.py:30): the Qwen3-ASR text
                // backbone with thinker.lm_head.weight as a per-token time-bin classifier.
                HfArchEntry.of(Collections.singletonList(HfAsrForcedAlignerHooks.ARCH),
                                HfArchResolver.of(Arch.QWEN3VL), HfAsrForcedAlignerHooks::meta,
                                HfAsrForcedAlignerHooks::tensors, HfDenseHooksB::asrVocab)
                        .withNameFilter(HfDenseHooksB.THINKER_FILTER),
                // LongcatFlashForCausalLM (longcat_flash.py) and LongCatFlashMTPModel (longcat_flash_mtp.py): a
                // vLLM-only text family with no llama.cpp converter. Every logical layer is two physical blocks of
                // MLA attention and a dense SwiGLU MLP; the first block also holds the shortcut MoE with its
                // identity (zero-compute) experts, and the MTP head becomes the appended NextN block.
                HfArchEntry.of(Arrays.asList("LongcatFlashForCausalLM", "LongCatFlashMTPModel"),
                                HfLongcatFlashHooks::arch, HfLongcatFlashHooks::meta, HfLongcatFlashHooks::tensors,
                                HfLongcatFlashHooks::vocab)
                        .withBlockCount(HfLongcatFlashHooks::blockCount)
                        .withIndexedBlockCount(HfLongcatFlashHooks::blockCount)
                        .withLoadNameFilter(HfLongcatFlashHooks::loadFilter)
                        .withExtra(HfLongcatFlashHooks::extra)
                        .withMtpExport(true)
                        .withMtpOnlyRootKeep(HfMtpRoots.COMMON),
                // LongcatFlashNgramForCausalLM (longcat_flash_ngram.py): the same model with the fused n-gram
                // input embedding; its checkpoints carry no MTP head (vLLM maps model.mtp.* to None).
                HfArchEntry.of(Collections.singletonList("LongcatFlashNgramForCausalLM"),
                                HfLongcatFlashHooks::arch, HfLongcatFlashHooks::meta, HfLongcatFlashHooks::tensors,
                                HfLongcatFlashHooks::vocab)
                        .withBlockCount(HfLongcatFlashHooks::blockCount)
                        .withIndexedBlockCount(HfLongcatFlashHooks::blockCount)
                        .withLoadNameFilter(HfLongcatFlashHooks::loadFilter)
                        .withExtra(HfLongcatFlashHooks::extra)));
    }
}
