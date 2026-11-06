package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * Hyperparameters of one mmproj encoder, from the {@code clip.*} metadata of its GGUF file: the counterpart of
 * llama.cpp {@code clip_hparams} (tools/mtmd/clip-model.h) filled by {@code load_hparams} (clip.cpp). Per-encoder
 * keys carry the modality prefix ({@code clip.vision.embedding_length} &hellip;); the encoder the hparams describe
 * is the one {@link #from} was called for, so one file with two encoders yields one {@code ClipHparams} per
 * modality. Immutable.
 *
 * <p>The projector type comes from {@code clip.projector_type}, or the modality-specific
 * {@code clip.<modality>.projector_type} when the common key is absent (mixed-modality files); the legacy
 * {@code qwen2.5o} resolves to {@code qwen2.5vl_merger} for vision and {@code qwen2a} otherwise. An unknown type
 * or a missing required key fails with {@link GgufFormatException} naming the key.
 */
public final class ClipHparams {

    /** Patch-merge layout of the vision input (llama.cpp {@code mm_patch_merge_type}). */
    public enum PatchMerge {
        /** Patches are plain flattened patch vectors. */
        FLAT,
        /** Patches come from an unpad-cropped image ({@code spatial_unpad}); the projector sees a square grid. */
        SPATIAL_UNPAD
    }

    /** The patch side clip.cpp sets for DeepSeek-OCR and DeepSeek-OCR 2 over the file's value. */
    static final int DEEPSEEK_OCR_PATCH = 16;

    /** The global-view side clip.cpp sets for DeepSeek-OCR and DeepSeek-OCR 2 over the file's value. */
    static final int DEEPSEEK_OCR_IMAGE = 1024;

    private final ProjectorModality modality;
    private final ProjectorType projectorType;
    private final int nEmbd;
    private final int nFf;
    private final int nLayer;
    private final int projectionDim;
    private final int nHead;
    private final int nHeadKv;
    private final int nEmbdHead;
    private final float eps;
    private final ClipFfnOp ffnOp;
    private final PatchMerge patchMerge;
    private final int[] featureLayers;
    private final int imageSize;
    private final int patchSize;
    private final int nMerge;
    private final int nWaPattern;
    private final int attnWindowSize;
    private final float[] imageMean;
    private final float[] imageStd;
    private final int minicpmvVersion;
    private final int minicpmvQueryNum;
    private final int nExpertUsed;
    private final float swigluClamp;
    private final ClipAudioHparams audio;
    private final VisionFamilyHparams visionFamily;

    private ClipHparams(ProjectorModality modality, ProjectorType projectorType, int nEmbd, int nFf, int nLayer,
                        int projectionDim, int nHead, int nHeadKv, int nEmbdHead, float eps, ClipFfnOp ffnOp,
                        PatchMerge patchMerge, int[] featureLayers, int imageSize, int patchSize, int nMerge,
                        int nWaPattern, int attnWindowSize, float[] imageMean, float[] imageStd,
                        int minicpmvVersion, int minicpmvQueryNum, int nExpertUsed, float swigluClamp,
                        ClipAudioHparams audio, VisionFamilyHparams visionFamily) {
        this.modality = modality;
        this.projectorType = projectorType;
        this.nEmbd = nEmbd;
        this.nFf = nFf;
        this.nLayer = nLayer;
        this.projectionDim = projectionDim;
        this.nHead = nHead;
        this.nHeadKv = nHeadKv;
        this.nEmbdHead = nEmbdHead;
        this.eps = eps;
        this.ffnOp = ffnOp;
        this.patchMerge = patchMerge;
        this.featureLayers = featureLayers;
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.nMerge = nMerge;
        this.nWaPattern = nWaPattern;
        this.attnWindowSize = attnWindowSize;
        this.imageMean = imageMean;
        this.imageStd = imageStd;
        this.minicpmvVersion = minicpmvVersion;
        this.minicpmvQueryNum = minicpmvQueryNum;
        this.nExpertUsed = nExpertUsed;
        this.swigluClamp = swigluClamp;
        this.audio = audio;
        this.visionFamily = visionFamily;
    }

    /**
     * The encoder hparams of {@code kv} for {@code modality}.
     *
     * @throws IllegalArgumentException for a null argument, or a modality the file does not carry
     *                                  ({@code clip.has_vision_encoder} and friends)
     * @throws GgufFormatException      for an unknown projector type or a missing, malformed or out-of-range key
     */
    public static ClipHparams from(GgufKv kv, ProjectorModality modality) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(modality, "modality");
        requireModality(kv, modality);
        ProjectorType type = projectorType(kv, modality);
        String prefix = "clip." + modality.keyPrefix() + ".";
        int nEmbd = requiredU32(kv, prefix + "embedding_length");
        int nHead = requiredU32(kv, prefix + "attention.head_count");
        int nEmbdHead = kv.u32(prefix + "attention.head_dim", 0);
        int nFf = requiredU32(kv, prefix + "feed_forward_length");
        int nLayer = requiredU32(kv, prefix + "block_count");
        int projectionDim = requiredU32(kv, prefix + "projection_dim");
        float eps = requiredF32(kv, prefix + "attention.layer_norm_epsilon");
        checkPositive(prefix, "embedding_length", nEmbd);
        // the gemma4ua and gemma4uv embedders and the SenseNova-U1 patch embedder have no encoder blocks: their
        // converters write 0 FFN width and blocks (and 0 heads for gemma4ua)
        boolean blockless = type == ProjectorType.GEMMA4UA || type == ProjectorType.SENSENOVA_U1
                || type == ProjectorType.GEMMA4UV;
        if (!blockless) {
            checkPositive(prefix, "attention.head_count", nHead);
        }
        if (type == ProjectorType.YASA2) {
            requireNoBlocks(prefix, type, nLayer, nFf);
        } else if (!blockless) {
            checkPositive(prefix, "feed_forward_length", nFf);
            checkPositive(prefix, "block_count", nLayer);
        }
        // the MiniCPM-V converter writes 0: the resampler's width comes from resampler.proj.weight
        if (type != ProjectorType.MINICPMV || projectionDim != 0) {
            checkPositive(prefix, "projection_dim", projectionDim);
        }
        if (!(eps > 0f) || Float.isInfinite(eps)) {
            throw new GgufFormatException("metadata key '" + prefix + "attention.layer_norm_epsilon' must be"
                    + " positive, got " + eps);
        }
        if (requiresHeadCountKv(type) && !kv.has(prefix + "attention.head_count_kv")) {
            throw new GgufFormatException("required metadata key '" + prefix + "attention.head_count_kv' is missing");
        }
        int nHeadKv = kv.u32(prefix + "attention.head_count_kv", nHead);
        if (!(blockless && nHead == 0 && nHeadKv == 0) && (nHeadKv <= 0 || nHead % nHeadKv != 0)) {
            throw new GgufFormatException("metadata key '" + prefix + "attention.head_count_kv' must divide"
                    + " attention.head_count, got " + nHeadKv + " / " + nHead);
        }
        // the voxtral realtime and mimovl encoders project to n_head * head_dim, independent of the embedding width
        boolean freeHeadDim = type == ProjectorType.VOXTRAL_RT || type == ProjectorType.MIMOVL;
        if (nEmbdHead != 0 && (nEmbdHead <= 0
                || !freeHeadDim && (nEmbd % nEmbdHead != 0 || nEmbd / nEmbdHead != nHead))) {
            throw new GgufFormatException("metadata key '" + prefix + "attention.head_dim' must be 0 or"
                    + " embedding_length / attention.head_count (" + nEmbd + " / " + nHead + "), got " + nEmbdHead);
        }
        ClipFfnOp ffnOp = visionFfnOverride(type, ClipAudioHparams.ffnOverride(type,
                ClipFfnOp.fromUseGeluSilu(kv.bool("clip.use_gelu", false), kv.bool("clip.use_silu", false))));
        eps = ClipAudioHparams.epsOverride(type, eps);
        if (type == ProjectorType.DEEPSEEKOCR) {
            // clip.cpp load_hparams: the CLIP body runs its layer norms at 1e-5 (the SAM stage at 1e-6)
            eps = 1e-5f;
        }
        ClipAudioHparams audio = modality == ProjectorModality.AUDIO ? ClipAudioHparams.from(kv, type, nLayer) : null;
        PatchMerge patchMerge = PatchMerge.FLAT;
        if ("spatial_unpad".equals(kv.str("clip.vision.mm_patch_merge_type", null))) {
            patchMerge = PatchMerge.SPATIAL_UNPAD;
        }
        int[] featureLayers = readFeatureLayers(kv, prefix, nLayer);
        int imageSize = 0;
        int patchSize = 1;
        int nMerge = 1;
        int nWaPattern = 0;
        int attnWindowSize = 0;
        float[] imageMean = null;
        float[] imageStd = null;
        int minicpmvVersion = 0;
        int minicpmvQueryNum = 0;
        int nExpertUsed = 0;
        float swigluClamp = 0f;
        VisionFamilyHparams visionFamily = VisionFamilyHparams.NONE;
        if (modality == ProjectorModality.VISION) {
            imageSize = requiredU32(kv, "clip.vision.image_size");
            patchSize = requiredU32(kv, "clip.vision.patch_size");
            if (type == ProjectorType.DEEPSEEKOCR || type == ProjectorType.DEEPSEEKOCR2) {
                // clip.cpp load_hparams: the SAM backbone runs 16-pixel patches of the 1024 global view
                patchSize = DEEPSEEK_OCR_PATCH;
                imageSize = DEEPSEEK_OCR_IMAGE;
            }
            // the SenseNova-U1, dots.ocr and dots3-note converters write image_size 0: their images take any size
            // within the pixel bounds
            if (!allowsDynamicImageSize(type) || imageSize != 0) {
                checkPositive("clip.vision.", "image_size", imageSize);
            }
            checkPositive("clip.vision.", "patch_size", patchSize);
            boolean fixedGrid = usesImageSizeGrid(type);
            if (fixedGrid && imageSize % patchSize != 0) {
                throw new GgufFormatException("metadata key 'clip.vision.image_size' must be a multiple of"
                        + " clip.vision.patch_size (" + patchSize + "), got " + imageSize);
            }
            PatchMergeRule mergeRule = type.patchMerge();
            nMerge = mergeRule.read(kv);
            if (fixedGrid && (imageSize / patchSize) % nMerge != 0) {
                throw new GgufFormatException("the patch merge " + nMerge + " (" + mergeRule + ") of projector type '"
                        + type + "' must divide the patch grid " + (imageSize / patchSize));
            }
            if (type == ProjectorType.DOTS3NOTE_V) {
                nExpertUsed = requiredU32(kv, "clip.vision.expert_used_count");
                checkPositive("clip.vision.", "expert_used_count", nExpertUsed);
            }
            if (type == ProjectorType.GLM5V) {
                swigluClamp = requiredF32(kv, "clip.vision.swiglu_clamp");
                if (!(swigluClamp >= 0f) || Float.isInfinite(swigluClamp)) {
                    throw new GgufFormatException("metadata key 'clip.vision.swiglu_clamp' must be a finite"
                            + " non-negative limit, got " + swigluClamp);
                }
            }
            nWaPattern = kv.u32("clip.vision.n_wa_pattern", 0);
            if (nWaPattern < 0) {
                throw new GgufFormatException("metadata key 'clip.vision.n_wa_pattern' must not be negative, got "
                        + nWaPattern);
            }
            attnWindowSize = kv.u32("clip.vision.window_size", 0);
            imageMean = requiredColorArray(kv, "clip.vision.image_mean");
            imageStd = requiredColorArray(kv, "clip.vision.image_std");
            minicpmvVersion = kv.i32("clip.minicpmv_version", 0);
            minicpmvQueryNum = minicpmvQueryNum(kv, type);
            if (type == ProjectorType.MINICPMV && minicpmvVersion == 0) {
                minicpmvVersion = 2;
            }
            visionFamily = VisionFamilyHparams.from(kv, type, nLayer, nMerge, featureLayers);
        }
        return new ClipHparams(modality, type, nEmbd, nFf, nLayer, projectionDim, nHead, nHeadKv, nEmbdHead, eps,
                ffnOp, patchMerge, featureLayers, imageSize, patchSize, nMerge, nWaPattern, attnWindowSize,
                imageMean, imageStd, minicpmvVersion, minicpmvQueryNum, nExpertUsed, swigluClamp, audio,
                visionFamily);
    }

    /**
     * The block keys of an encoder without transformer blocks (the yasa2 ConvNeXt backbone): llama.cpp loads
     * {@code block_count} transformer blocks for every vision type but this graph reads none, so the count must
     * be 0; the FFN width is unread and may be 0.
     */
    private static void requireNoBlocks(String prefix, ProjectorType type, int nLayer, int nFf) {
        if (nLayer != 0) {
            throw new GgufFormatException("metadata key '" + prefix + "block_count' must be 0 for projector type '"
                    + type + "' (a ConvNeXt encoder without transformer blocks), got " + nLayer);
        }
        if (nFf < 0) {
            throw new GgufFormatException("metadata key '" + prefix + "feed_forward_length' must not be negative,"
                    + " got " + nFf);
        }
    }

    /**
     * The FFN activation clip.cpp forces for a vision encoder: GELU for HunyuanVL and the Gemma 4 tower and
     * embedder ({@code load_hparams}), exact GELU for Nemotron Omni ({@code load_hparams}) and for the MiniMax-M3
     * and Muse Glimmer towers (their graphs pass {@code FFN_GELU_ERF} to {@code build_vit}), quick GELU for the
     * DeepSeek-OCR CLIP tower and SiLU for the DeepSeek-OCR 2 Qwen2 encoder (whose SAM backbones keep
     * {@code fromKeys}); {@code fromKeys} otherwise.
     */
    /** The GQA vision encoders whose {@code load_hparams} requires {@code clip.vision.attention.head_count_kv}. */
    private static boolean requiresHeadCountKv(ProjectorType type) {
        return type == ProjectorType.LFM3VL || type == ProjectorType.EXAONE4_5 || type == ProjectorType.DEEPSEEKOCR2
                || type == ProjectorType.MIMOVL;
    }

    private static ClipFfnOp visionFfnOverride(ProjectorType type, ClipFfnOp fromKeys) {
        switch (type) {
            case HUNYUANVL:
            case GEMMA4V:
            case GEMMA4UV:
                return ClipFfnOp.GELU;
            case NEMOTRON_OMNI:
            case MINIMAX_M3:
            case MUSE_GLIMMER:
                return ClipFfnOp.GELU_ERF;
            case DEEPSEEKOCR:
                return ClipFfnOp.GELU_QUICK;
            case DEEPSEEKOCR2:
                return ClipFfnOp.SILU;
            default:
                return fromKeys;
        }
    }

    /**
     * True for the families whose converters write a nominal {@code clip.vision.image_size} of 0 because their
     * encoder reads the grid from each image and the pixel bounds carry the input budget (SenseNova-U1, dots.ocr and
     * dots3-note).
     */
    private static boolean allowsDynamicImageSize(ProjectorType type) {
        return type == ProjectorType.SENSENOVA_U1 || type == ProjectorType.DOTS_OCR
                || type == ProjectorType.DOTS3NOTE_V;
    }

    /**
     * True for the families whose encoder runs the square {@code clip.vision.image_size} grid (the CLIP / SigLIP
     * tiles, the LLaVA-derived MobileVLM / GLM-Edge / CogVLM / Granite Vision 4.1 tiles, InternVL and Llama 4);
     * the dynamic-resolution families read the grid from each image, so their nominal image size need not be a
     * patch multiple.
     */
    private static boolean usesImageSizeGrid(ProjectorType type) {
        switch (type) {
            case MLP:
            case MLP_NORM:
            case LDP:
            case LDPV2:
            case GLM_EDGE:
            case COGVLM:
            case GRANITE4_VISION:
            case GEMMA3:
            case IDEFICS3:
            case INTERNVL:
            case LLAMA4:
            case JANUS_PRO:
            case COHERE2V:
            case NEMOTRON_V2_VL:
                return true;
            default:
                return false;
        }
    }

    /**
     * The resampler query count: the {@link ImageTokenFormula#resamplerQueryNum} of a MiniCPM-V file, and
     * {@code clip.minicpmv_query_num} (0 when absent) for the other families.
     */
    private static int minicpmvQueryNum(GgufKv kv, ProjectorType type) {
        if (type == ProjectorType.MINICPMV) {
            return ImageTokenFormula.resamplerQueryNum(kv);
        }
        int n = kv.u32("clip.minicpmv_query_num", 0);
        if (n < 0) {
            throw new GgufFormatException("metadata key 'clip.minicpmv_query_num' must not be negative, got " + n);
        }
        return n;
    }

    private static void requireModality(GgufKv kv, ProjectorModality modality) {
        String key;
        switch (modality) {
            case VISION:
                key = "clip.has_vision_encoder";
                break;
            case AUDIO:
                key = "clip.has_audio_encoder";
                break;
            case TEXT:
                key = "clip.has_text_encoder";
                break;
            default:
                key = "clip.has_gen_audio_encoder";
        }
        if (!kv.bool(key, false)) {
            throw new IllegalArgumentException("the file has no " + modality.name().toLowerCase(Locale.ROOT)
                    + " encoder (" + key + " is absent or false)");
        }
    }

    /**
     * The hparams of one tower of a CLIP two-tower embedding file (llama.cpp's legacy
     * {@code convert_image_encoder_to_gguf.py} without a projector): the {@code clip.text.*} keys of the text tower
     * or the {@code clip.vision.*} keys of the vision tower, which carries the square {@code image_size} tile of
     * {@code patch_size} patches and the normalization colors. The file declares no projector, so
     * {@link #projectorType()} is {@link ProjectorType#UNKNOWN} and {@link #twoTower()} is true; the blocks follow
     * the converter's naming, whose {@code ffn_up} / {@code ffn_down} names are swapped.
     *
     * @throws IllegalArgumentException for a null argument, an audio modality, or a tower the file does not carry
     *                                  ({@code clip.has_text_encoder}, {@code clip.has_vision_encoder})
     * @throws GgufFormatException      for a missing, malformed or out-of-range key
     */
    public static ClipHparams twoTower(GgufKv kv, ProjectorModality modality) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(modality, "modality");
        if (modality != ProjectorModality.TEXT && modality != ProjectorModality.VISION) {
            throw new IllegalArgumentException("a CLIP two-tower file holds a text and a vision tower, not "
                    + modality.name().toLowerCase(Locale.ROOT));
        }
        requireModality(kv, modality);
        String prefix = "clip." + modality.keyPrefix() + ".";
        int nEmbd = requiredU32(kv, prefix + "embedding_length");
        int nHead = requiredU32(kv, prefix + "attention.head_count");
        int nFf = requiredU32(kv, prefix + "feed_forward_length");
        int nLayer = requiredU32(kv, prefix + "block_count");
        int projectionDim = requiredU32(kv, prefix + "projection_dim");
        float eps = requiredF32(kv, prefix + "attention.layer_norm_epsilon");
        checkPositive(prefix, "embedding_length", nEmbd);
        checkPositive(prefix, "attention.head_count", nHead);
        checkPositive(prefix, "feed_forward_length", nFf);
        checkPositive(prefix, "block_count", nLayer);
        if (projectionDim < 0) {
            throw new GgufFormatException("metadata key '" + prefix + "projection_dim' must not be negative, got "
                    + projectionDim);
        }
        if (nEmbd % nHead != 0) {
            throw new GgufFormatException("metadata key '" + prefix + "attention.head_count' must divide "
                    + prefix + "embedding_length (" + nEmbd + "), got " + nHead);
        }
        if (!(eps > 0f) || Float.isInfinite(eps)) {
            throw new GgufFormatException("metadata key '" + prefix + "attention.layer_norm_epsilon' must be"
                    + " positive, got " + eps);
        }
        ClipFfnOp ffnOp = ClipFfnOp.fromUseGeluSilu(kv.bool("clip.use_gelu", false), kv.bool("clip.use_silu", false));
        int imageSize = 0;
        int patchSize = 1;
        float[] imageMean = null;
        float[] imageStd = null;
        if (modality == ProjectorModality.VISION) {
            imageSize = requiredU32(kv, "clip.vision.image_size");
            patchSize = requiredU32(kv, "clip.vision.patch_size");
            checkPositive("clip.vision.", "image_size", imageSize);
            checkPositive("clip.vision.", "patch_size", patchSize);
            if (imageSize % patchSize != 0) {
                throw new GgufFormatException("metadata key 'clip.vision.image_size' must be a multiple of"
                        + " clip.vision.patch_size (" + patchSize + "), got " + imageSize);
            }
            imageMean = requiredColorArray(kv, "clip.vision.image_mean");
            imageStd = requiredColorArray(kv, "clip.vision.image_std");
        }
        return new ClipHparams(modality, ProjectorType.UNKNOWN, nEmbd, nFf, nLayer, projectionDim, nHead, nHead, 0,
                eps, ffnOp, PatchMerge.FLAT, new int[0], imageSize, patchSize, 1, 0, 0, imageMean, imageStd, 0, 0, 0,
                0f, null, VisionFamilyHparams.NONE);
    }

    private static ProjectorType projectorType(GgufKv kv, ProjectorModality modality) {
        String name = ProjectorType.declaredName(kv, modality);
        ProjectorType type = ProjectorType.fromName(name);
        if (type == ProjectorType.UNKNOWN) {
            throw new GgufFormatException("unknown projector type: " + name);
        }
        return type;
    }

    private static int requiredU32(GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        return kv.u32(key, 0);
    }

    private static float requiredF32(GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        return kv.f32(key, 0f);
    }

    private static int[] readFeatureLayers(GgufKv kv, String prefix, int nLayer) {
        int[] layers = kv.i32Array(prefix + "feature_layer");
        if (layers == null) {
            return new int[0];
        }
        for (int v : layers) {
            if (v > nLayer) {
                throw new GgufFormatException("metadata key '" + prefix + "feature_layer' index " + v
                        + " is out of range (block_count " + nLayer + ")");
            }
        }
        return layers;
    }

    private static float[] requiredColorArray(GgufKv kv, String key) {
        float[] v = kv.f32Array(key);
        if (v == null || v.length < 3) {
            throw new GgufFormatException("metadata key '" + key + "' must hold at least 3 values, got "
                    + (v == null ? "nothing" : v.length));
        }
        return Arrays.copyOf(v, 3);
    }

    private static void checkPositive(String prefix, String key, int v) {
        if (v <= 0) {
            throw new GgufFormatException("metadata key '" + prefix + key + "' must be positive, got " + v);
        }
    }

    /** The modality these hparams describe. */
    public ProjectorModality modality() {
        return modality;
    }

    /** The resolved projector type; {@link ProjectorType#UNKNOWN} for a tower of a two-tower file. */
    public ProjectorType projectorType() {
        return projectorType;
    }

    /** True for the hparams of a CLIP two-tower file's tower ({@link #twoTower(GgufKv, ProjectorModality)}). */
    public boolean twoTower() {
        return projectorType == ProjectorType.UNKNOWN;
    }

    /** Encoder width ({@code clip.<m>.embedding_length}). */
    public int nEmbd() {
        return nEmbd;
    }

    /** FFN width ({@code clip.<m>.feed_forward_length}). */
    public int nFf() {
        return nFf;
    }

    /** Encoder block count ({@code clip.<m>.block_count}). */
    public int nLayer() {
        return nLayer;
    }

    /**
     * Projector output width ({@code clip.<m>.projection_dim}); the text model's embedding width. 0 for MiniCPM-V
     * files, whose width is the resampler projection's.
     */
    public int projectionDim() {
        return projectionDim;
    }

    /** Attention head count ({@code clip.<m>.attention.head_count}). */
    public int nHead() {
        return nHead;
    }

    /** KV head count, defaults to {@link #nHead()} ({@code clip.<m>.attention.head_count_kv}). */
    public int nHeadKv() {
        return nHeadKv;
    }

    /** Head width, 0 = {@code nEmbd / nHead} ({@code clip.<m>.attention.head_dim}). */
    public int nEmbdHead() {
        return nEmbdHead;
    }

    /** {@link #nEmbdHead()} resolved: {@code nEmbd / nHead} when the key is 0; 0 for a tower without heads. */
    public int dHead() {
        if (nEmbdHead > 0) {
            return nEmbdHead;
        }
        return nHead > 0 ? nEmbd / nHead : 0;
    }

    /** Layer-norm epsilon ({@code clip.<m>.attention.layer_norm_epsilon}). */
    public float eps() {
        return eps;
    }

    /** FFN activation, from {@code clip.use_gelu} / {@code clip.use_silu}. */
    public ClipFfnOp ffnOp() {
        return ffnOp;
    }

    /**
     * The SwiGLU clamp limit of the GLM-5V FFNs ({@code clip.vision.swiglu_clamp}, required for that type); 0
     * (unclamped) for every other encoder.
     */
    public float swigluClamp() {
        return swigluClamp;
    }

    /**
     * The op of a gated FFN ({@code build_ffn} with a gate): {@link GluOp#SWIGLU_CLAMP} for a SiLU FFN with a
     * positive {@link #swigluClamp()} ({@code ggml_swiglu_clamp}), {@link ClipFfnOp#glu()} otherwise.
     */
    public GluOp ffnGlu() {
        return clampsSwiglu() ? GluOp.SWIGLU_CLAMP : ffnOp.glu();
    }

    /** The parameters of {@link #ffnGlu()}: the clamp limit of a clamped SwiGLU, none otherwise. */
    public ActParams ffnGluParams() {
        return clampsSwiglu() ? ActParams.builder().limit(swigluClamp).build() : ActParams.NONE;
    }

    private boolean clampsSwiglu() {
        return ffnOp == ClipFfnOp.SILU && swigluClamp > 0f;
    }

    /** Vision patch-merge layout, {@link PatchMerge#FLAT} unless the file says {@code spatial_unpad}. */
    public PatchMerge patchMerge() {
        return patchMerge;
    }

    /**
     * Vision feature-layer indexes tapped before the projector ({@code clip.<m>.feature_layer}), ascending as
     * stored; empty when absent.
     */
    public int[] featureLayers() {
        return featureLayers.clone();
    }

    /** Vision input size in pixels ({@code clip.vision.image_size}); 0 for audio. */
    public int imageSize() {
        return imageSize;
    }

    /** Vision patch size in pixels ({@code clip.vision.patch_size}); 1 for audio. */
    public int patchSize() {
        return patchSize;
    }

    /**
     * Patches merged per side after the encoder, at least 1, read with the {@link ProjectorType#patchMerge()} rule
     * of the type: {@code clip.vision.spatial_merge_size} (qwen, pixtral, lightonocr, hunyuanvl, dots3note) or
     * {@code clip.vision.projector.scale_factor} (gemma3, idefics3, internvl, llama4, kimivl, lfm2, d1-omni,
     * cohere2v, dots.ocr), or fixed by the family (2 for paddleocr). Only the families that run the
     * {@code image_size} grid require it to divide that grid.
     */
    public int nMerge() {
        return nMerge;
    }

    /**
     * Window-attention period ({@code clip.vision.n_wa_pattern}): every n-th layer is full attention, 0 = all
     * layers full attention.
     */
    public int nWaPattern() {
        return nWaPattern;
    }

    /**
     * Window-attention window in pixels ({@code clip.vision.window_size}); 0 = the qwen2.5-vl default of 112.
     */
    public int attnWindowSize() {
        return attnWindowSize;
    }

    /** MiniCPM-V generation ({@code clip.minicpmv_version}, 2 when absent); 0 for the other families. */
    public int minicpmvVersion() {
        return minicpmvVersion;
    }

    /** Resampler query count, the MiniCPM-V image token count per tile; 0 for the other families. */
    public int minicpmvQueryNum() {
        return minicpmvQueryNum;
    }

    /** Routed experts per token of the MoE vision blocks ({@code clip.vision.expert_used_count}); 0 without MoE. */
    public int nExpertUsed() {
        return nExpertUsed;
    }

    /** Vision normalization mean per channel ({@code clip.vision.image_mean}). */
    public float[] imageMean() {
        return imageMean.clone();
    }

    /** Vision normalization standard deviation per channel ({@code clip.vision.image_std}). */
    public float[] imageStd() {
        return imageStd.clone();
    }

    /** The family-specific vision keys (empty for the families that read none, and for audio). */
    public VisionFamilyHparams visionFamily() {
        return visionFamily;
    }

    /**
     * The audio keys of an {@link ProjectorModality#AUDIO} encoder: {@code clip.audio.*} beyond the shared ones
     * and the frontend framing the projector fixes.
     *
     * @throws IllegalStateException for a vision or audio-generation encoder
     */
    public ClipAudioHparams audio() {
        if (audio == null) {
            throw new IllegalStateException("the " + modality.name().toLowerCase(Locale.ROOT)
                    + " encoder has no audio hparams");
        }
        return audio;
    }

    @Override
    public String toString() {
        return "ClipHparams{" + modality.name().toLowerCase(Locale.ROOT) + ", projector=" + projectorType
                + ", n_embd=" + nEmbd + ", n_ff=" + nFf + ", n_layer=" + nLayer + ", projection_dim="
                + projectionDim + ", heads=" + nHead + "/" + nHeadKv + ", eps=" + eps + ", ffn=" + ffnOp
                + (audio != null ? ", " + audio : "") + "}";
    }
}
