package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfAsrForcedAlignerHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfAudioTowerHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekOcrVisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekVisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfGemmaOmniHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLfm3VisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLfmOmniHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlavaVisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMimoNemotronHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMmprojHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfPocketTtsGenHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwen3TtsGenHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenDerivedVisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenOmniHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfQwenVlVisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSopranoHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfTowerVisionHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfVisionTowerHooksA;
import com.eonmux.neuralfree.gguf.hf.hooks.HfVisionTowerHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfVisionTowerHooksC;
import com.eonmux.neuralfree.gguf.hf.hooks.HfWhisperAudioHooks;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Hugging Face architecture entries for the vision and audio projector checkpoints (the {@code --mmproj} exports of
 * llama.cpp's converter, the {@code MmprojModel} subclasses of llava.py, pixtral.py, qwenvl.py, qwen3vl.py, gemma.py,
 * granite.py, lfm2.py, lfm3.py, mimo.py, nemotron.py, soprano.py, pockettts.py, qwen3tts.py, smolvlm.py, internvl.py,
 * llama4.py, kimivl.py, sarashina2.py, lighton_ocr.py and ultravox.py); every entry has kind {@link
 * HfArchEntry.Kind#MMPROJ} and resolves to {@link Arch#CLIP}, whose file the projector loaders serve. The remaining
 * converter classes the tables list for this kind refuse with their class named. {@link HfArchRegistry} merges this
 * table with the other tables; a Hugging Face name appears in exactly one table per {@link HfArchEntry.Kind}.
 */
final class HfArchTableMmproj {

    private HfArchTableMmproj() {
    }

    /** The entries of this table, in registration order. */
    static List<HfArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // LlavaVisionModel (llava.py:17): the Pixtral tower and its projector, with the [IMG_BREAK] row.
                HfArchEntry.of(Arrays.asList(HfLlavaVisionHooks.LLAVA, HfLlavaVisionHooks.MISTRAL3),
                                HfArchResolver.of(Arch.CLIP), HfLlavaVisionHooks::meta,
                                HfLlavaVisionHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Mistral3EmbeddingVisionModel (llava.py:134): the embedding model's headless text model.
                HfArchEntry.of(Collections.singletonList(HfLlavaVisionHooks.MISTRAL3_EMBEDDING),
                                HfArchResolver.of(Arch.CLIP), HfLlavaVisionHooks::embeddingMeta,
                                HfLlavaVisionHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withConfigFixup(HfLlavaVisionHooks.EMBEDDING_FIXUP)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // LightOnOCRVisionModel (lighton_ocr.py:13): the Pixtral tower under the newer checkpoint names.
                HfArchEntry.of(Collections.singletonList(HfLlavaVisionHooks.LIGHTONOCR),
                                HfArchResolver.of(Arch.CLIP), HfLlavaVisionHooks::lightOnOcrMeta,
                                HfLlavaVisionHooks::lightOnOcrTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfLlavaVisionHooks.LIGHTONOCR_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Qwen2VLVisionModel (qwenvl.py:45) with Sarashina2VLVisionModel (sarashina2.py:24).
                HfArchEntry.of(Arrays.asList("Qwen2VLForConditionalGeneration", "Qwen2VLModel",
                                "Qwen2_5_VLForConditionalGeneration", "Sarashina2VisionForCausalLM"),
                                HfArchResolver.of(Arch.CLIP), HfQwenVlVisionHooks::qwen2vlMeta,
                                HfQwenVlVisionHooks::qwen2vlTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenVlVisionHooks.QWEN2VL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfQwenVlVisionHooks.FORCE),
                // Qwen3VLVisionModel (qwen3vl.py:17).
                HfArchEntry.of(Arrays.asList("Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration",
                                "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
                                "OpenJevModel", "InternS1ProForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwenVlVisionHooks::qwen3vlMeta,
                                HfQwenVlVisionHooks::qwen3vlTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenVlVisionHooks.QWEN3VL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Glm4VVisionModel (qwen3vl.py:240): the GLM-4V tower maps through the converter's name map.
                HfArchEntry.of(Arrays.asList("Glm4vForConditionalGeneration", "Glm4vMoeForConditionalGeneration",
                                "GlmOcrForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwenVlVisionHooks::glm4vMeta,
                                HfQwenVlVisionHooks::glm4vTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenVlVisionHooks.QWEN3VL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Glm5NextVisionModel (qwen3vl.py:266): the GLM-5.3-Flash tower, glm4v layout with the clamped FFN.
                HfArchEntry.of(Collections.singletonList("Glm5NextForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwenVlVisionHooks::glm5NextMeta,
                                HfQwenVlVisionHooks::glm4vTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenVlVisionHooks.QWEN3VL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // The Qwen3-VL subclasses that keep the tower as is: ClefVisionModel (clef.py:149),
                // H2OLightningVisionModel (h2o_lightning.py:95) and Qwen4ExpVisionModel (qwen4exp.py:265).
                HfArchEntry.of(Arrays.asList("ClefModel", "H2OLightningModel", "Qwen4ExpForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwenDerivedVisionHooks::qwen3vlMeta,
                                HfQwenVlVisionHooks::qwen3vlTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenVlVisionHooks.QWEN3VL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // PplxDeciderVisionModel (pplx_decider.py:96) and VllmD3VisionModel (vllm_d3.py:58): the Qwen3-VL
                // tower with the decision models' image pixel bounds.
                HfArchEntry.of(Arrays.asList("PplxDeciderModel", "VllmD3Model"),
                                HfArchResolver.of(Arch.CLIP), HfQwenDerivedVisionHooks::decisionMeta,
                                HfQwenVlVisionHooks::qwen3vlTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenVlVisionHooks.QWEN3VL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // MiniCPMOCRSeqClsVisionModel (qwen3vl.py:148): the Qwen3-VL tower with the classifier's pixel budget.
                HfArchEntry.of(Collections.singletonList("MiniCPMOCRForSequenceClassification"),
                                HfArchResolver.of(Arch.CLIP), HfQwenDerivedVisionHooks::seqClsMeta,
                                HfQwenVlVisionHooks::qwen3vlTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenVlVisionHooks.QWEN3VL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Exaone4_5VisionModel (exaone.py:275): the Qwen2-VL styled EXAONE 4.5 tower.
                HfArchEntry.of(Collections.singletonList("Exaone4_5_ForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwenDerivedVisionHooks::exaoneMeta,
                                HfQwenDerivedVisionHooks::exaoneTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenDerivedVisionHooks.EXAONE_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfQwenVlVisionHooks.FORCE),
                // BailingMoeV3VLVisionModel (bailingmoe3.py:237): the Ling-3.0-Flash tower and its top-level MLP.
                HfArchEntry.of(Collections.singletonList("BailingMoeV3VLForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwenDerivedVisionHooks::bailingMeta,
                                HfQwenDerivedVisionHooks::bailingTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenDerivedVisionHooks.BAILING_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Gemma3VisionModel (gemma.py:262): the SigLIP tower and the pool projector.
                HfArchEntry.of(Collections.singletonList("Gemma3ForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfTowerVisionHooks::gemma3Meta,
                                HfTowerVisionHooks::gemma3Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfTowerVisionHooks.GEMMA3_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfTowerVisionHooks.GEMMA3_FORCE),
                // SmolVLMModel (smolvlm.py:13): the Idefics3 tower and its connector.
                HfArchEntry.of(Arrays.asList("Idefics3ForConditionalGeneration", "SmolVLMForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfTowerVisionHooks::smolvlmMeta,
                                HfTowerVisionHooks::smolvlmTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfTowerVisionHooks.SMOLVLM_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfTowerVisionHooks.SMOLVLM_FORCE),
                // CogVLMVisionModel (cogvlm.py:15).
                HfArchEntry.of(Collections.singletonList("CogVLMForCausalLM"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::cogvlmMeta,
                                HfTensorHook.DEFAULT, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.COGVLM_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Cohere2VisionModel (command_r.py:188).
                HfArchEntry.of(Collections.singletonList("Cohere2VisionForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::cohere2Meta,
                                HfTensorHook.DEFAULT, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.COHERE2_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.VISION_EMBEDDINGS_F32),
                // DotsOCRVisionModel (dotsocr.py:13).
                HfArchEntry.of(Collections.singletonList("DotsOCRForCausalLM"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::dotsOcrMeta,
                                HfTensorHook.DEFAULT, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.DOTS_OCR_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // PaddleOCRVisionModel (ernie.py:168).
                HfArchEntry.of(Collections.singletonList("PaddleOCRVisionModel"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::paddleMeta,
                                HfTensorHook.DEFAULT, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.PADDLE_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // JanusProVisionModel (januspro.py:39).
                HfArchEntry.of(Collections.singletonList("JanusForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::janusMeta,
                                HfVisionTowerHooksA::janusTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.JANUS_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // YoutuVLVisionModel (youtuvl.py:13).
                HfArchEntry.of(Collections.singletonList("YoutuVLForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::youtuMeta,
                                HfVisionTowerHooksA::youtuTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.YOUTU_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Step3VLVisionModel (step3.py:20), shared by Step 3.7.
                HfArchEntry.of(Arrays.asList("StepVLForConditionalGeneration", "Step3p7ForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::step3Meta,
                                HfVisionTowerHooksA::step3Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.STEP3_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfVisionTowerHooksA.STEP3_FORCE),
                // HunyuanVLVisionModel (hunyuan.py:288).
                HfArchEntry.of(Collections.singletonList("HunYuanVLForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::hunyuanMeta,
                                HfVisionTowerHooksA::hunyuanTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.HUNYUAN_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfVisionTowerHooksA.HUNYUAN_FORCE),
                // SenseNovaU1VisionModel (sensenova_u1.py:49): a blockless patch-and-dense embedder.
                HfArchEntry.of(Collections.singletonList("NEOVisionModel"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::senseNovaMeta,
                                HfVisionTowerHooksA::senseNovaTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.SENSENOVA_FILTER)
                        .withBlockCount(cfg -> 0)
                        .withForceType(HfVisionTowerHooksA.SENSENOVA_FORCE),
                // LFM2VLModel (lfm2.py:362).
                HfArchEntry.of(Collections.singletonList("Lfm2VlForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksA::lfm2Meta,
                                HfVisionTowerHooksA::lfm2Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksA.LFM2_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // InternVisionModel (internvl.py:13).
                HfArchEntry.of(Collections.singletonList("InternVisionModel"),
                                HfArchResolver.of(Arch.CLIP), HfTowerVisionHooks::internvlMeta,
                                HfTowerVisionHooks::internvlTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfTowerVisionHooks.INTERNVL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfTowerVisionHooks.INTERNVL_FORCE),
                // Llama4VisionModel (llama4.py:14).
                HfArchEntry.of(Collections.singletonList("Llama4ForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfTowerVisionHooks::llama4Meta,
                                HfTowerVisionHooks::llama4Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfTowerVisionHooks.LLAMA4_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfTowerVisionHooks.LLAMA4_FORCE),
                // DeepseekOCRVisionModel (deepseek.py:22) with OCR2 and Unlimited (deepseek.py:104-110).
                HfArchEntry.of(Arrays.asList("DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM",
                                "UnlimitedOCRForCausalLM"),
                                HfArchResolver.of(Arch.CLIP), HfDeepseekOcrVisionHooks::meta,
                                HfDeepseekOcrVisionHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfDeepseekOcrVisionHooks.FILTER)
                        .withBlockCount(HfDeepseekOcrVisionHooks::blocks)
                        .withForceType(HfDeepseekOcrVisionHooks.FORCE),
                // DeepseekV4FlashVisionModel (deepseek.py:1042): the V4-Flash ViT and its aligner.
                HfArchEntry.of(Collections.singletonList("DeepseekV4ForCausalLM"),
                                HfArchResolver.of(Arch.CLIP), HfDeepseekVisionHooks::v4Meta,
                                HfDeepseekVisionHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfDeepseekVisionHooks.FILTER)
                        .withBlockCount(HfDeepseekVisionHooks::v4Blocks)
                        .withForceType(HfDeepseekVisionHooks.FORCE),
                // DeepseekV41VisionModel (deepseek41.py:427): the V4.1 token block over the same tower.
                HfArchEntry.of(Collections.singletonList("DeepseekV41ForCausalLM"),
                                HfArchResolver.of(Arch.CLIP), HfDeepseekVisionHooks::v41Meta,
                                HfDeepseekVisionHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfDeepseekVisionHooks.FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfDeepseekVisionHooks.FORCE),
                // LFM3VLMmprojModel (lfm3.py:271): the vision half of LFM3VLVisionMixin (lfm3.py:179-266). The Omni
                // checkpoint, which also carries the audio tower and its codec, has its own entry below.
                HfArchEntry.of(Collections.singletonList("Lfm3VlForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfLfm3VisionHooks::meta,
                                HfLfm3VisionHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfLfm3VisionHooks.FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfLfm3VisionHooks.FORCE),
                // MiniCPMV4_6VisionModel (minicpm.py:141), MiniCPMV4_7VisionModel (minicpm.py:242) and
                // MiniCPMOCR1_0VisionModel (minicpm.py:288): the SigLIP tower with its insertion-point ViT merger.
                HfArchEntry.of(Arrays.asList("MiniCPMV4_6ForConditionalGeneration",
                                "MiniCPMV4_7ForConditionalGeneration", "MiniCPMOcrForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksC::minicpmMeta,
                                HfVisionTowerHooksC::minicpmTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withLoadNameFilter(HfVisionTowerHooksC.MINICPM_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Dots3NoteMmprojModel (dots3.py:201): the pyramid MoE vision tower and the speech encoder.
                HfArchEntry.of(Arrays.asList("Dots3NoteForCausalLM", "Dots3NoteForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksC::dots3Meta,
                                HfVisionTowerHooksC::dots3Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksC.DOTS3_FILTER)
                        .withBlockCount(HfVisionTowerHooksC::dots3Blocks)
                        .withForceType(HfVisionTowerHooksC.DOTS3_FORCE),
                // KimiK25Model (kimivl.py:57) and its Glm5vModel subclass (kimivl.py:161): the MoonViT tower.
                HfArchEntry.of(Arrays.asList("KimiK25ForConditionalGeneration", "Glm5vForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksB::kimiMeta,
                                HfVisionTowerHooksB::kimiTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksB.KIMI_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // MiniMaxM3VisionModel (minimax.py:210).
                HfArchEntry.of(Collections.singletonList("MiniMaxM3SparseForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksB::minimaxMeta,
                                HfVisionTowerHooksB::minimaxTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksB.MINIMAX_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // MuseGlimmerVisionModel (muse_glimmer.py:83).
                HfArchEntry.of(Collections.singletonList("MuseGlimmerForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksB::museGlimmerMeta,
                                HfVisionTowerHooksB::museTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksB.MUSE_GLIMMER_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // MuseSparkVisionModel (muse_spark.py:146): the Muse-Glimmer tower with an audio stage; two encoders.
                HfArchEntry.of(Collections.singletonList("MuseSparkForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksB::museSparkMeta,
                                HfVisionTowerHooksB::museSparkTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksB.MUSE_SPARK_FILTER)
                        .withBlockCount(HfVisionTowerHooksB::museSparkBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Phi4VisionMmprojModel (phi.py:216): every vision block but the last.
                HfArchEntry.of(Collections.singletonList("Phi4ForCausalLMV"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksB::phi4Meta,
                                HfVisionTowerHooksB::phi4Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksB.PHI4_FILTER)
                        .withBlockCount(HfVisionTowerHooksB::phi4Blocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // Granite4VisionMmprojModel (granite.py:692).
                HfArchEntry.of(Collections.singletonList("Granite4VisionForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksB::graniteMeta,
                                HfVisionTowerHooksB::graniteTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksB.GRANITE_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS),
                // NemotronNanoV2VLModel (nemotron.py:124) and the RADIOModel entry of the same converter.
                HfArchEntry.of(Arrays.asList("NemotronH_Nano_VL_V2", "RADIOModel"),
                                HfArchResolver.of(Arch.CLIP), HfVisionTowerHooksB::nemotronMeta,
                                HfVisionTowerHooksB::nemotronTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfVisionTowerHooksB.NEMOTRON_FILTER)
                        .withBlockCount(HfVisionTowerHooksB::nemotronBlocks)
                        .withForceType(HfVisionTowerHooksB.NEMOTRON_FORCE),
                // KimiVLModel (kimivl.py:15).
                HfArchEntry.of(Collections.singletonList("KimiVLForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfTowerVisionHooks::kimivlMeta,
                                HfTowerVisionHooks::kimivlTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfTowerVisionHooks.KIMIVL_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfTowerVisionHooks.KIMIVL_FORCE),
                // WhisperEncoderModel (ultravox.py:89): the Qwen2-Audio tower and its fc projector.
                HfArchEntry.of(Collections.singletonList("Qwen2AudioForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfWhisperAudioHooks::qwen2aMeta,
                                HfWhisperAudioHooks::qwen2aTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfWhisperAudioHooks.WHISPER_FILTER)
                        .withBlockCount(HfMmprojHooks::audioBlocks)
                        .withForceType(HfWhisperAudioHooks.CONV_F16),
                // UltravoxWhisperEncoderModel (ultravox.py:131).
                HfArchEntry.of(Collections.singletonList("UltravoxModel"),
                                HfArchResolver.of(Arch.CLIP), HfWhisperAudioHooks::ultravoxMeta,
                                HfWhisperAudioHooks::qwen2aTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfWhisperAudioHooks.WHISPER_FILTER)
                        .withBlockCount(HfMmprojHooks::audioBlocks)
                        .withForceType(HfWhisperAudioHooks.CONV_F16),
                // VoxtralWhisperEncoderModel (ultravox.py:216).
                HfArchEntry.of(Collections.singletonList("VoxtralForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfWhisperAudioHooks::voxtralMeta,
                                HfWhisperAudioHooks::qwen2aTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfWhisperAudioHooks.WHISPER_FILTER)
                        .withBlockCount(HfMmprojHooks::audioBlocks)
                        .withForceType(HfWhisperAudioHooks.CONV_F16),
                // AudioFlamingo3WhisperEncoderModel (ultravox.py:232).
                HfArchEntry.of(Collections.singletonList("AudioFlamingo3ForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfWhisperAudioHooks::musicFlamingoMeta,
                                HfWhisperAudioHooks::qwen2aTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfWhisperAudioHooks.WHISPER_FILTER)
                        .withBlockCount(HfMmprojHooks::audioBlocks)
                        .withForceType(HfWhisperAudioHooks.CONV_F32),
                // MERaLiONWhisperEncoderModel (ultravox.py:166): the speech tower and its adapter.
                HfArchEntry.of(Collections.singletonList("MERaLiON2ForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfWhisperAudioHooks::meralionMeta,
                                HfWhisperAudioHooks::meralionTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfWhisperAudioHooks.MERALION_FILTER)
                        .withBlockCount(HfWhisperAudioHooks::meralionBlocks)
                        .withForceType(HfWhisperAudioHooks.CONV_F16),
                // Qwen3TTSSpeakerEncoderModel (qwen3tts.py:190): the code predictor and the code2wav decoder of the
                // speech_tokenizer, the generator file Qwen3TtsClip serves.
                HfArchEntry.of(Collections.singletonList("Qwen3TTSForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwen3TtsGenHooks::meta,
                                HfQwen3TtsGenHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withLoadNameFilter(HfQwen3TtsGenHooks::filter)
                        .withBlockCount(HfQwen3TtsGenHooks::blockCount)
                        .withConfigFixup(HfQwen3TtsGenHooks::fixup)
                        .withExtra(HfQwen3TtsGenHooks::extra)
                        .withForceType(HfQwen3TtsGenHooks.FORCE),
                // PocketTTSMmprojModel (pockettts.py:177): the flow network and the mimi decoder; the same
                // checkpoint's text entry converts the backbone.
                HfArchEntry.of(Collections.singletonList(HfPocketTtsGenHooks.ARCH),
                                HfArchResolver.of(Arch.CLIP), HfPocketTtsGenHooks::meta,
                                HfPocketTtsGenHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withIndexedBlockCount(HfPocketTtsGenHooks::decoderBlocks)
                        .withForceType(HfPocketTtsGenHooks.SEANET_F16),
                // GlmASRWhisperEncoderModel (ultravox.py:24): the Whisper tower, the GLM adapter and its boundary
                // rows, under the common clip.projector_type the GLM runtime reads.
                HfArchEntry.of(Collections.singletonList("GlmasrModel"),
                                HfArchResolver.of(Arch.CLIP), HfAudioTowerHooks::glmaMeta,
                                HfAudioTowerHooks::glmaTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfAudioTowerHooks.GLMA_FILTER)
                        .withBlockCount(HfMmprojHooks::audioBlocks)
                        .withForceType(HfAudioTowerHooks.GLMA_FORCE),
                // LFM2AudioModel (lfm2.py:465): the conformer encoder and its projector.
                HfArchEntry.of(Collections.singletonList("Lfm2AudioForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfAudioTowerHooks::lfm2aMeta,
                                HfAudioTowerHooks::lfm2aTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfAudioTowerHooks.LFM2A_FILTER)
                        .withBlockCount(HfAudioTowerHooks::lfm2aBlocks)
                        .withForceType(HfAudioTowerHooks.CONFORMER_FORCE),
                // LFM3AudioMmprojModel (lfm3.py:309): the encoder, its projector and the liquid-codec generator.
                HfArchEntry.of(Collections.singletonList("Lfm3AudioForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfLfmOmniHooks::lfm3AudioMeta,
                                HfLfmOmniHooks::lfm3AudioTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfLfmOmniHooks.LFM3A_FILTER)
                        .withBlockCount(HfLfmOmniHooks::lfm3AudioBlocks)
                        .withExtra(HfLfmOmniHooks::lfm3Codec)
                        .withForceType(HfLfmOmniHooks.LFM3A_FORCE),
                // GraniteSpeechMmprojModel and GraniteSpeechPlusMmprojModel (granite.py:579, :663).
                HfArchEntry.of(Arrays.asList("GraniteSpeechForConditionalGeneration",
                                "GraniteSpeechPlusForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfAudioTowerHooks::graniteSpeechMeta,
                                HfAudioTowerHooks::graniteSpeechTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfAudioTowerHooks.GRANITE_SPEECH_FILTER)
                        .withBlockCount(HfAudioTowerHooks::graniteSpeechBlocks)
                        .withForceType(HfAudioTowerHooks.GRANITE_SPEECH_FORCE),
                // VoxtralRealtimeEncoderModel (ultravox.py:216): the realtime Voxtral speech tower.
                HfArchEntry.of(Collections.singletonList("VoxtralRealtimeForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfAudioTowerHooks::voxtralRtMeta,
                                HfAudioTowerHooks::voxtralRtTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfAudioTowerHooks.VOXTRAL_RT_FILTER)
                        .withBlockCount(HfMmprojHooks::audioBlocks)
                        .withForceType(HfAudioTowerHooks.VOXTRAL_RT_FORCE),
                // Qwen3ASRMmprojModel (qwen3vl.py:233): the Qwen3-Omni audio half alone; the forced aligner shares
                // the same audio tower (qwen3_asr_forced_aligner.py).
                HfArchEntry.of(Arrays.asList("Qwen3ASRForConditionalGeneration", HfAsrForcedAlignerHooks.ARCH),
                                HfArchResolver.of(Arch.CLIP), HfQwenOmniHooks::qwen3AsrMeta,
                                HfQwenOmniHooks::qwen3OmniTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenOmniHooks.QWEN3_OMNI_FILTER)
                        .withBlockCount(HfMmprojHooks::audioBlocks)
                        .withExtra(HfQwenOmniHooks::positionTable)
                        .withForceType(HfQwenOmniHooks.FORCE),
                // SopranoModel (soprano.py:51): the Vocos decoder of decoder.pth, a generator file only.
                HfArchEntry.of(Collections.singletonList(HfSopranoHooks.ARCH),
                                HfArchResolver.of(Arch.CLIP), HfSopranoHooks::meta,
                                HfSopranoHooks::drop, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfSopranoHooks.FILTER)
                        .withBlockCount(cfg -> 0)
                        .withIndexedBlockCount(HfSopranoHooks::decoderBlocks)
                        .withExtra(HfSopranoHooks::tensors)
                        .withForceType(HfSopranoHooks.FORCE),
                // Qwen25OmniModel (qwenvl.py:167): the Qwen2.5-VL tower and the Qwen2.5 audio tower in one file,
                // under the legacy mixed-modality projector that resolves per modality.
                HfArchEntry.of(Collections.singletonList("Qwen2_5OmniModel"),
                                HfArchResolver.of(Arch.CLIP), HfQwenOmniHooks::qwen25OmniMeta,
                                HfQwenOmniHooks::qwen25OmniTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenOmniHooks.QWEN25_OMNI_FILTER)
                        .withMultiTower()
                        .withExtra(HfQwenOmniHooks::positionTable)
                        .withForceType(HfQwenOmniHooks.FORCE),
                // Qwen3OmniMmprojModel (qwen3vl.py:159): the Qwen3-VL tower and the Qwen3 audio tower.
                HfArchEntry.of(Collections.singletonList("Qwen3OmniMoeForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfQwenOmniHooks::qwen3OmniMeta,
                                HfQwenOmniHooks::qwen3OmniTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfQwenOmniHooks.QWEN3_OMNI_FILTER)
                        .withMultiTower()
                        .withExtra(HfQwenOmniHooks::positionTable)
                        .withForceType(HfQwenOmniHooks.FORCE),
                // Gemma3nVisionAudioModel (gemma.py:368): the MobileNet-V5 tower and the conformer audio tower.
                HfArchEntry.of(Collections.singletonList("Gemma3nForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfGemmaOmniHooks::gemma3nMeta,
                                HfGemmaOmniHooks::gemma3nTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfGemmaOmniHooks.GEMMA3N_FILTER)
                        .withMultiTower()
                        .withForceType(HfGemmaOmniHooks.GEMMA3N_FORCE),
                // Gemma4VisionAudioModel (gemma.py:978) and its embedding variant (gemma.py:1059).
                HfArchEntry.of(Collections.singletonList("Gemma4ForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfGemmaOmniHooks::gemma4Meta,
                                HfGemmaOmniHooks::gemma4Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfGemmaOmniHooks.GEMMA4_FILTER)
                        .withMultiTower()
                        .withForceType(HfGemmaOmniHooks.GEMMA4_FORCE),
                HfArchEntry.of(Collections.singletonList("EmbeddingGemma2Model"),
                                HfArchResolver.of(Arch.CLIP), HfGemmaOmniHooks::gemma4Meta,
                                HfGemmaOmniHooks::embeddingGemma2Tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfGemmaOmniHooks.EMBEDDING_GEMMA2_FILTER)
                        .withMultiTower()
                        .withForceType(HfGemmaOmniHooks.GEMMA4_FORCE),
                // Gemma4UnifiedVisionAudioModel (gemma.py:1067): the transformer-less tower pair.
                HfArchEntry.of(Collections.singletonList("Gemma4UnifiedForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfGemmaOmniHooks::gemma4UnifiedMeta,
                                HfGemmaOmniHooks::gemma4UnifiedTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfGemmaOmniHooks.GEMMA4_FILTER)
                        .withMultiTower()
                        .withForceType(HfGemmaOmniHooks.GEMMA4_FORCE),
                // MiMoV2VisionAudioModel (mimo.py:345): the Qwen2-VL-style vision tower and the dots audio tower.
                HfArchEntry.of(Collections.singletonList("MiMoV2ForCausalLM"),
                                HfArchResolver.of(Arch.CLIP), HfMimoNemotronHooks::mimoMeta,
                                HfMimoNemotronHooks::mimoTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfMimoNemotronHooks.MIMO_FILTER)
                        .withMultiTower()
                        .withExtra(HfMimoNemotronHooks::mimoExtra)
                        .withForceType(HfMimoNemotronHooks.MIMO_FORCE),
                // D1OmniMmprojModel (lfm2.py:413): the LFM2-VL tower and the d1 conformer audio tower.
                HfArchEntry.of(Collections.singletonList("D1OmniModel"),
                                HfArchResolver.of(Arch.CLIP), HfLfmOmniHooks::d1OmniMeta,
                                HfLfmOmniHooks::d1OmniTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfAudioTowerHooks.D1OMNI_FILTER)
                        .withMultiTower()
                        .withForceType(HfAudioTowerHooks.CONFORMER_FORCE),
                // LFM3OmniMmprojModel (lfm3.py:575): the LFM3-VL tower and the LFM3 audio encoder and generator.
                HfArchEntry.of(Collections.singletonList("Lfm3OmniForConditionalGeneration"),
                                HfArchResolver.of(Arch.CLIP), HfLfmOmniHooks::lfm3OmniMeta,
                                HfLfmOmniHooks::lfm3OmniTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withMultiTower()
                        .withExtra(HfLfmOmniHooks::lfm3Codec)
                        .withForceType(HfLfmOmniHooks.LFM3_OMNI_FORCE),
                // NemotronOmniVLModel (nemotron.py:259): the RADIO v2.5 tower and its projector. The converter
                // leaves the Parakeet audio tower out of the clip keys (`MmprojModel.set_gguf_parameters` is called
                // with the class's vision-only flags), so the file declares the vision encoder only.
                HfArchEntry.of(Collections.singletonList("NemotronH_Omni_Reasoning_V3"),
                                HfArchResolver.of(Arch.CLIP), HfMimoNemotronHooks::nemotronMeta,
                                HfMimoNemotronHooks::nemotronTensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withNameFilter(HfMimoNemotronHooks.NEMOTRON_FILTER)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMimoNemotronHooks.NEMOTRON_FORCE),
                // PixtralModel (pixtral.py:12): Mistral native params.json checkpoints of the Pixtral tower.
                HfArchEntry.of(Arrays.asList(HfConfigLoader.MISTRAL_NATIVE, HfConfigLoader.MISTRAL_NATIVE_MOE),
                                HfArchResolver.of(Arch.CLIP), HfLlavaVisionHooks::meta,
                                HfLlavaVisionHooks::tensors, HfMmprojHooks::noVocab)
                        .withKind(HfArchEntry.Kind.MMPROJ)
                        .withBlockCount(HfMmprojHooks::visionBlocks)
                        .withForceType(HfMmprojHooks.PATCH_KERNELS)));
    }
}
