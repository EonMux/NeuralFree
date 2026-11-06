package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * The vision projectors of the CogVLM, Command-R (Cohere2), DotsOCR, PaddleOCR, Janus-Pro, Youtu-VL, Step 3-VL,
 * Hunyuan-VL, SenseNova-U1 and LFM2-VL checkpoints, ports of the llama.cpp converter classes
 * {@code CogVLMVisionModel} (cogvlm.py:15), {@code Cohere2VisionModel} (command_r.py:188), {@code
 * DotsOCRVisionModel} (dotsocr.py:13), {@code PaddleOCRVisionModel} (ernie.py:168), {@code JanusProVisionModel}
 * (januspro.py:39), {@code YoutuVLVisionModel} (youtuvl.py:13), {@code Step3VLVisionModel} (step3.py:20), {@code
 * HunyuanVLVisionModel} (hunyuan.py:288), {@code SenseNovaU1VisionModel} (sensenova_u1.py:49) and {@code
 * LFM2VLModel} (lfm2.py:362). Each class keeps its tower's names, writes its projector type and the merge,
 * activation, epsilon, pixel and tiling keys the clip graph reads, and applies the name or layout changes the
 * converter makes: the Step 3 downsampler projectors, the Janus aligner, the Hunyuan CLS row, the SenseNova
 * patch/dense embedding pair and the LFM2 patchifier.
 */
public final class HfVisionTowerHooksA {

    /** {@code VisionProjectorType.COGVLM}. */
    public static final String COGVLM = "cogvlm";
    /** {@code VisionProjectorType.COHERE2V}. */
    public static final String COHERE2V = "cohere2v";
    /** {@code VisionProjectorType.DOTSOCR}. */
    public static final String DOTS_OCR = "dots_ocr";
    /** {@code VisionProjectorType.PADDLEOCR}. */
    public static final String PADDLEOCR = "paddleocr";
    /** {@code VisionProjectorType.JANUS_PRO}. */
    public static final String JANUS_PRO = "janus_pro";
    /** {@code VisionProjectorType.YOUTUVL}. */
    public static final String YOUTUVL = "youtuvl";
    /** {@code VisionProjectorType.STEP3VL}. */
    public static final String STEP3VL = "step3vl";
    /** {@code VisionProjectorType.HUNYUANVL}. */
    public static final String HUNYUANVL = "hunyuanvl";
    /** {@code VisionProjectorType.SENSENOVA_U1}. */
    public static final String SENSENOVA_U1 = "sensenova_u1";
    /** {@code VisionProjectorType.LFM2}. */
    public static final String LFM2 = "lfm2";

    /** The default layer-norm epsilon of the CogVLM, Janus-Pro and Youtu-VL towers (cogvlm.py:19, januspro.py:53). */
    private static final double DEFAULT_EPS = 1e-6;
    /** The Youtu-VL image size written when the vision config has none (youtuvl.py:18). */
    private static final int YOUTU_IMAGE_SIZE = 560;
    /** The Hunyuan-VL image size written when the vision config has none (hunyuan.py:294). */
    private static final int HUNYUAN_IMAGE_SIZE = 2048;
    /** The Step 3-VL preprocessor image size (step3.py:64). */
    private static final int STEP3_PREPROC_IMAGE_SIZE = 3024;
    /** The SenseNova-U1 defaults for an older vision config (sensenova_u1.py:88-91). */
    private static final int SENSENOVA_HIDDEN = 1024;
    private static final int SENSENOVA_MIN_PIXELS = 65_536;
    private static final int SENSENOVA_MAX_PIXELS = 16_777_216;
    private static final double SENSENOVA_DOWNSAMPLE = 0.5;
    /** The dataset colors some classes write when the preprocessor config has none. */
    private static final List<Float> IMAGENET_MEAN = unmodifiableFloats(0.485f, 0.456f, 0.406f);
    private static final List<Float> IMAGENET_STD = unmodifiableFloats(0.229f, 0.224f, 0.225f);
    private static final List<Float> MISTRAL_MEAN = unmodifiableFloats(0.48145466f, 0.4578275f, 0.40821073f);
    private static final List<Float> MISTRAL_STD = unmodifiableFloats(0.26862954f, 0.26130258f, 0.27577711f);
    /** The 16-pixel patchifier LFM2-VL reshapes its patch embedding with (lfm2.py:391-392). */
    private static final int LFM2_PATCH_AREA = 16;

    private HfVisionTowerHooksA() {
    }

    private static List<Float> unmodifiableFloats(float... values) {
        List<Float> out = new java.util.ArrayList<>(values.length);
        for (float v : values) {
            out.add(v);
        }
        return Collections.unmodifiableList(out);
    }

    private static JsonView vision(HfConfig cfg) {
        return cfg.vision().orElseThrow(() -> new GgufFormatException(cfg.source()
                + ": the checkpoint has no 'vision_config'"));
    }

    // ------------------------------------------------------------------ CogVLM

    /** {@code CogVLMVisionModel.filter_tensors} (cogvlm.py:25-31): the {@code model.vision.} tower survives. */
    public static final HfNameFilter COGVLM_FILTER = (name, base) ->
            name.startsWith("model.vision.") ? base.apply(name) : null;

    /** {@code CogVLMVisionModel.set_gguf_parameters} (cogvlm.py:17-21): the base keys, {@code cogvlm}, epsilon. */
    public static void cogvlmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfMmprojHooks.visionMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, COGVLM);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(ctx.config(), "layer_norm_eps").orElse(DEFAULT_EPS));
    }

    // ------------------------------------------------------------------ Cohere2 (Command-R vision)

    /** {@code Cohere2VisionModel.filter_tensors} (command_r.py:203-208). */
    public static final HfNameFilter COHERE2_FILTER = (name, base) ->
            name.startsWith("model.vision_tower.") || name.startsWith("model.multi_modal_projector.")
                    ? base.apply(name) : null;

    /**
     * {@code Cohere2VisionModel.set_gguf_parameters} (command_r.py:189-196): the base keys, {@code cohere2v}, the
     * required epsilon, the root downsample factor, the processor's tile limit and GELU.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void cohere2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, COHERE2V);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64(cfg, "layer_norm_eps"));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, rootInt(cfg, "downsample_factor"));
        kv.set(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, HfMmprojHooks.preInt(cfg, "max_patches"));
        kv.set(GgufKeys.CLIP_USE_GELU, true);
    }

    // ------------------------------------------------------------------ DotsOCR

    /** The DotsOCR block renames: the FFN is stored as {@code fc1} / {@code fc2} / {@code fc3} (dotsocr.py:35-45). */
    public static final HfNameFilter DOTS_OCR_FILTER = (name, base) -> {
        if (!name.startsWith("vision_tower.")) {
            return null;
        }
        String renamed = name;
        if (name.contains("vision_tower.blocks.") && name.contains(".mlp.")) {
            renamed = name.replace("vision_tower.blocks.", "visual.blocks.")
                    .replace(".fc1", ".gate_proj").replace(".fc2", ".down_proj").replace(".fc3", ".up_proj");
        }
        return base.apply(renamed);
    };

    /**
     * {@code DotsOCRVisionModel.set_gguf_parameters} (dotsocr.py:20-28): the base keys of the dynamic-resolution
     * tower ({@code image_size} 0), {@code dots_ocr}, the processor's pixel bounds, the required RMS epsilon, the
     * spatial-merge scale factor and SiLU.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void dotsOcrMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        HfMmprojHooks.visionMeta(ctx, kv, 0, HfMmprojHooks.visionInt(cfg, "patch_size"),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.HEAD_KEYS),
                HfMmprojHooks.blocksOf(cfg, vision, "vision_config"));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, DOTS_OCR);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, HfMmprojHooks.preInt(cfg, "min_pixels"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, HfMmprojHooks.preInt(cfg, "max_pixels"));
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, HfMmprojHooks.visionF64(cfg, "rms_norm_eps"));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, HfMmprojHooks.visionInt(cfg, "spatial_merge_size"));
        kv.set(GgufKeys.CLIP_USE_SILU, true);
    }

    // ------------------------------------------------------------------ PaddleOCR-VL

    /** {@code PaddleOCRVisionModel.filter_tensors} (ernie.py:185-205). */
    public static final HfNameFilter PADDLE_FILTER = (name, base) -> {
        if (!name.contains("vision_model") && !name.contains("mlp_AR")) {
            return null;
        }
        String renamed = name.replace("visual.", "model.");
        if (renamed.contains("packing_position_embedding") || renamed.contains("vision_model.head")) {
            return null;
        }
        return base.apply(renamed);
    };

    /**
     * {@code PaddleOCRVisionModel.set_gguf_parameters} (ernie.py:174-183): the base keys of the SigLIP-derived tower
     * at {@code sqrt(max_pixels)}, {@code paddleocr}, the processor's pixel bounds, GELU and the RMS epsilon.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void paddleMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        int maxPixels = HfMmprojHooks.preInt(cfg, "max_pixels");
        int minPixels = HfMmprojHooks.preInt(cfg, "min_pixels");
        int imageSize = (int) Math.sqrt(maxPixels);
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, HfMmprojHooks.visionInt(cfg, "patch_size"),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.HEAD_KEYS), HfMmprojHooks.visionBlocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, PADDLEOCR);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, maxPixels);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, minPixels);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "rms_norm_eps").orElse(DEFAULT_EPS));
    }

    // ------------------------------------------------------------------ Janus-Pro

    /** {@code JanusProVisionModel.filter_tensors} (januspro.py:92-110): the generation branches drop. */
    public static final HfNameFilter JANUS_FILTER = (name, base) -> {
        if (name.startsWith("model.vqmodel.") || name.startsWith("vqmodel.")
                || name.startsWith("model.generation_embeddings.") || name.startsWith("generation_embeddings.")
                || name.startsWith("model.generation_aligner.") || name.startsWith("generation_aligner.")
                || name.startsWith("model.generation_head.") || name.startsWith("generation_head.")) {
            return null;
        }
        return base.apply(name);
    };

    /**
     * {@code JanusProVisionModel.set_gguf_parameters} (januspro.py:47-58): the base keys of the SigLIP tower with its
     * FFN width derived from {@code mlp_ratio}, {@code janus_pro}, the epsilon and the activation.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void janusMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        int hidden = HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS);
        int ffn;
        if (HfMmprojHooks.visionFind(cfg, HfMmprojHooks.FFN_KEYS).isPresent()) {
            ffn = HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS);
        } else {
            Object ratio = HfMmprojHooks.visionFind(cfg, "mlp_ratio").orElse(null);
            if (!(ratio instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": the Janus-Pro vision config has no"
                        + " intermediate_size and no mlp_ratio to derive it (januspro.py:41-46)");
            }
            ffn = (int) Math.round(hidden * ((Number) ratio).doubleValue());
        }
        HfMmprojHooks.visionMeta(ctx, kv, HfMmprojHooks.visionInt(cfg, "image_size"),
                HfMmprojHooks.visionInt(cfg, "patch_size"), hidden, ffn,
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.HEAD_KEYS),
                HfMmprojHooks.blocksOf(cfg, vision, "vision_config"));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, JANUS_PRO);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(DEFAULT_EPS));
        Object act = HfMmprojHooks.visionFind(cfg, "hidden_act").orElse(null);
        String hiddenAct = act instanceof String ? ((String) act).toLowerCase(Locale.ROOT) : "";
        if ("gelu".equals(hiddenAct)) {
            kv.set(GgufKeys.CLIP_USE_GELU, true);
        } else if ("silu".equals(hiddenAct)) {
            kv.set(GgufKeys.CLIP_USE_SILU, true);
        }
    }

    /**
     * {@code JanusProVisionModel.modify_tensors} (januspro.py:60-90, :112-125): the aligner's {@code fc1} lands at
     * {@code mm.0} and its {@code hidden_layers.<i>} at {@code mm.<i + 1>}; the vision tower maps through the clip
     * name table; anything else is dropped.
     *
     * @throws GgufFormatException when an aligner tensor has an unknown name
     */
    public static void janusTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String local;
        if (hfName.startsWith("model.aligner.")) {
            local = hfName.substring("model.aligner.".length());
        } else if (hfName.startsWith("aligner.")) {
            local = hfName.substring("aligner.".length());
        } else if (hfName.startsWith("model.vision_model.") || hfName.startsWith("vision_model.")) {
            out.add(ctx.names().mapOrThrow(hfName), src);
            return;
        } else {
            return;
        }
        String suffix = hfName.endsWith(".bias") ? ".bias" : ".weight";
        int index;
        if (local.startsWith("fc1.")) {
            index = 0;
        } else if (local.startsWith("hidden_layers.")) {
            String[] parts = local.split("\\.", 3);
            if (parts.length < 3) {
                throw new GgufFormatException("Unexpected Janus aligner tensor name: " + hfName
                        + " (januspro.py:74-78)");
            }
            index = Integer.parseInt(parts[1]) + 1;
        } else {
            throw new GgufFormatException("Unsupported Janus aligner tensor: " + hfName + " (januspro.py:80)");
        }
        out.add(ctx.names().ggufName("V_MMPROJ", index, suffix), src);
    }

    // ------------------------------------------------------------------ Youtu-VL

    /** {@code YoutuVLVisionModel.filter_tensors} (youtuvl.py:51-60). */
    public static final HfNameFilter YOUTU_FILTER = (name, base) -> {
        if (name.startsWith("lm_head.") || name.startsWith("model.layers.")
                || name.startsWith("model.embed_tokens.") || name.startsWith("model.norm.")) {
            return null;
        }
        return base.apply(name);
    };

    /**
     * {@code YoutuVLVisionModel.set_gguf_parameters} (youtuvl.py:20-49): the base keys at 560 pixels by default,
     * {@code youtuvl}, the epsilon, the activation, the spatial merge, the window and the explicit full-attention
     * layers.
     *
     * @throws GgufFormatException naming the missing or unsupported key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void youtuMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv,
                HfMmprojHooks.visionIntOpt(cfg, "image_size").orElse(YOUTU_IMAGE_SIZE));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, YOUTUVL);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(DEFAULT_EPS));
        Object act = HfMmprojHooks.visionFind(cfg, "hidden_act").orElse("gelu_pytorch_tanh");
        String hiddenAct = String.valueOf(act).toLowerCase(Locale.ROOT);
        if ("gelu".equals(hiddenAct) || "gelu_pytorch_tanh".equals(hiddenAct) || "gelu_fast".equals(hiddenAct)
                || "gelu_new".equals(hiddenAct) || "gelu_accurate".equals(hiddenAct)) {
            kv.set(GgufKeys.CLIP_USE_GELU, true);
        } else if ("silu".equals(hiddenAct)) {
            kv.set(GgufKeys.CLIP_USE_SILU, true);
        } else {
            throw new GgufFormatException(cfg.source() + ": Unsupported activation function for YOUTUVL: "
                    + hiddenAct + " (youtuvl.py:30-36)");
        }
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE,
                HfMmprojHooks.visionIntOpt(cfg, "spatial_merge_size").orElse(2));
        HfMmprojHooks.visionIntOpt(cfg, "window_size")
                .ifPresent(v -> kv.set(GgufKeys.CLIP_VISION_WINDOW_SIZE, v));
        List<Integer> full = HfMmprojHooks.intList(cfg, vision(cfg), "fullatt_block_indexes");
        if (full.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": fullatt_block_indexes is required for youtuvl"
                    + " (youtuvl.py:44)");
        }
        kv.set(GgufKeys.CLIP_VISION_WA_LAYER_INDEXES, full);
    }

    /** {@code YoutuVLVisionModel.modify_tensors} (youtuvl.py:62-70): unmappable names are logged and skipped. */
    public static void youtuTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        java.util.Optional<com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap.Mapped> mapped =
                ctx.names().map(hfName, ".weight", ".bias");
        if (mapped.isPresent()) {
            out.add(mapped.get().ggufName(), src);
        } else {
            ctx.log("Cannot map tensor: " + hfName);
        }
    }

    // ------------------------------------------------------------------ Step 3-VL

    /** {@code Step3VLVisionModel.filter_tensors} (step3.py:68-75): the text tower and the output head drop. */
    public static final HfNameFilter STEP3_FILTER = (name, base) ->
            name.startsWith("model.") || name.startsWith("lm_head.") ? null : base.apply(name);

    /**
     * {@code Step3VLVisionModel.set_gguf_parameters} (step3.py:42-65): the base keys of the only validated checkpoint
     * (Step3-VL-10B), {@code step3vl}, the epsilon, the 2x2 projector stride and the preprocessor image size. Any
     * other geometry is refused by name.
     *
     * @throws GgufFormatException naming the unsupported geometry
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void step3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        int hidden = HfMmprojHooks.visionFind(cfg, "hidden_size").isPresent()
                ? HfMmprojHooks.visionInt(cfg, "hidden_size") : HfMmprojHooks.visionInt(cfg, "width");
        Object configured = HfMmprojHooks.visionFind(cfg, "intermediate_size").orElse(null);
        boolean absent = !(configured instanceof Number) || ((Number) configured).doubleValue() == 0.0;
        int ffn;
        if (absent) {
            Object ratio = HfMmprojHooks.visionFind(cfg, "mlp_ratio").orElse(null);
            double mlpRatio = ratio instanceof Number ? ((Number) ratio).doubleValue() : 8960.0 / 1536.0;
            ffn = (int) Math.round(hidden * mlpRatio);
        } else {
            ffn = ((Number) configured).intValue();
        }
        int stride = rootInt(cfg, "understand_projector_stride");
        int imageSize = HfMmprojHooks.visionInt(cfg, "image_size");
        int blocks = HfMmprojHooks.blocksOf(cfg, vision, "vision_config");
        if (stride != 2 || imageSize != 728 || hidden != 1536 || blocks != 47) {
            throw new GgufFormatException(cfg.source() + ": current Step3-VL conversion path is only validated for"
                    + " Step3-VL-10B: got projector stride " + stride + ", image size " + imageSize + ", width "
                    + hidden + " and " + blocks + " blocks (step3.py:52-55)");
        }
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, HfMmprojHooks.visionInt(cfg, "patch_size"), hidden, ffn,
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.HEAD_KEYS), blocks, MISTRAL_MEAN, MISTRAL_STD);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, STEP3VL);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(1e-5));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, stride * stride);
        kv.set(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, STEP3_PREPROC_IMAGE_SIZE);
    }

    /**
     * {@code Step3VLVisionModel.modify_tensors} (step3.py:77-95): the downsamplers become {@code mm.<n>}, the large
     * projector {@code mm.model.fc}, and the tower names lose their gamma and fused-projection spellings.
     *
     * @throws GgufFormatException when a downsampler name is malformed
     */
    public static void step3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (hfName.startsWith("vision_model.vit_downsampler")) {
            String rest = hfName.substring("vision_model.vit_downsampler".length());
            int dot = rest.indexOf('.');
            if (dot <= 0) {
                throw new GgufFormatException("Unexpected Step3-VL projector tensor '" + hfName + "'"
                        + " (step3.py:80-83)");
            }
            int index;
            try {
                index = Integer.parseInt(rest.substring(0, dot)) - 1;
            } catch (NumberFormatException e) {
                throw new GgufFormatException("Unexpected Step3-VL projector tensor '" + hfName + "'", e);
            }
            out.add(ctx.names().ggufName("V_MMPROJ", index, rest.substring(dot)), src);
            return;
        }
        if ("vit_large_projector.weight".equals(hfName)) {
            out.add(ctx.names().ggufName("V_MMPROJ_FC", -1, ".weight"), src);
            return;
        }
        if (hfName.startsWith("vision_model.")) {
            String name = hfName;
            if ("vision_model.positional_embedding".equals(name)) {
                name += ".weight";
            } else if (name.endsWith(".gamma") && name.contains(".ls_")) {
                name = name.substring(0, name.length() - ".gamma".length()) + ".weight";
            }
            name = name.replace("attn.in_proj_weight", "attn.in_proj.weight")
                    .replace("attn.in_proj_bias", "attn.in_proj.bias");
            out.add(ctx.names().mapOrThrow(name), src);
        }
    }

    /** The position table F32, the projector linears F16 at {@code --outtype f16} (step3.py:67-72). */
    public static final HfForceType STEP3_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains(".position_embd.")) {
            return GGMLType.F32;
        }
        if ((ggufName.contains("mm.0.") || ggufName.contains("mm.1.")) && ggufName.endsWith(".weight")) {
            return outType == com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType.F16
                    ? GGMLType.F16 : GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    // ------------------------------------------------------------------ Hunyuan-VL

    /** {@code HunyuanVLVisionModel.filter_tensors} (hunyuan.py:308-314): only the {@code vit.} tower survives. */
    public static final HfNameFilter HUNYUAN_FILTER = (name, base) ->
            name.startsWith("vit.") ? base.apply(name) : null;

    /**
     * {@code HunyuanVLVisionModel.set_gguf_parameters} (hunyuan.py:296-306): the base keys at the config's image size
     * or {@code max_image_size}, {@code hunyuanvl}, GELU, the RMS epsilon, the spatial merge and the pixel bounds.
     *
     * @throws GgufFormatException naming the missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void hunyuanMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        int imageSize = HfMmprojHooks.visionIntOpt(cfg, "image_size")
                .orElseGet(() -> HfMmprojHooks.visionIntOpt(cfg, "max_image_size").orElse(HUNYUAN_IMAGE_SIZE));
        HfMmprojHooks.visionMeta(ctx, kv, imageSize);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, HUNYUANVL);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "rms_norm_eps").orElse(1e-5));
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE,
                HfMmprojHooks.visionIntOpt(cfg, "spatial_merge_size").orElse(2));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, HfMmprojHooks.preInt(cfg, "min_pixels"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, HfMmprojHooks.preInt(cfg, "max_pixels"));
    }

    /**
     * {@code HunyuanVLVisionModel.modify_tensors} (hunyuan.py:316-321): the CLS row (row 0) of the position table is
     * stripped before mapping.
     *
     * @throws GgufFormatException when the position table has no rows to strip
     */
    public static void hunyuanTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (hfName.contains("position_embedding")) {
            long[] shape = src.shape();
            if (shape.length == 0 || shape[0] < 2) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the shape "
                        + Arrays.toString(shape) + "; the Hunyuan-VL position table holds a CLS row and patches");
            }
            out.add(ctx.names().mapOrThrow(hfName), RowTransforms.sliceRows(src, 1, shape[0] - 1));
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The projector linears F16 at {@code --outtype f16} (hunyuan.py:323-328); the rest follows the base rule. */
    public static final HfForceType HUNYUAN_FORCE = (ggufName, nDims, source, outType) -> {
        if ((ggufName.contains("mm.0.") || ggufName.contains("mm.2.")) && ggufName.endsWith(".weight")) {
            return outType == com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType.F16
                    ? GGMLType.F16 : GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    // ------------------------------------------------------------------ SenseNova-U1

    /** {@code SenseNovaU1VisionModel.filter_tensors} (sensenova_u1.py:77-81): only the {@code vision_model.} tower. */
    public static final HfNameFilter SENSENOVA_FILTER = (name, base) ->
            name.startsWith("vision_model.") ? base.apply(name) : null;

    /**
     * {@code SenseNovaU1VisionModel.set_gguf_parameters} (sensenova_u1.py:83-106): the base keys of the blockless
     * patch-and-dense embedder, {@code sensenova_u1}, the head width, the epsilon, the pixel bounds and the
     * downsample merge.
     *
     * @throws GgufFormatException naming an unusable downsample ratio or a missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void senseNovaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        int hidden = HfMmprojHooks.intOr(cfg, vision, SENSENOVA_HIDDEN, "hidden_size");
        int ffn = HfMmprojHooks.intOr(cfg, vision, 0, HfMmprojHooks.FFN_KEYS);
        int heads = HfMmprojHooks.visionIntOpt(cfg, "num_attention_heads", "num_heads", "heads").orElse(1);
        int blocks = HfMmprojHooks.intOr(cfg, vision, 0, "num_hidden_layers", "n_layers", "n_layer",
                "num_layers", "depth", "layers", "encoder_layers", "vt_num_hidden_layers");
        HfMmprojHooks.visionMeta(ctx, kv, HfMmprojHooks.visionIntOpt(cfg, "image_size").orElse(0),
                HfMmprojHooks.visionInt(cfg, "patch_size"), hidden, ffn, heads, blocks, IMAGENET_MEAN, IMAGENET_STD);
        double ratio = HfMmprojHooks.f64Or(cfg, vision, SENSENOVA_DOWNSAMPLE, "downsample_ratio");
        if (!(ratio > 0.0)) {
            throw new GgufFormatException(cfg.source() + ": SenseNova U1 downsample_ratio must be positive, got "
                    + ratio + " (sensenova_u1.py:94-95)");
        }
        int merge = (int) Math.round(1.0 / ratio);
        if (Math.abs(1.0 / ratio - merge) > 1e-6) {
            throw new GgufFormatException(cfg.source() + ": SenseNova U1 downsample_ratio must have an integer"
                    + " reciprocal, got " + ratio + " (sensenova_u1.py:98-99)");
        }
        double eps = HfMmprojHooks.f64Or(cfg, vision, DEFAULT_EPS, "layer_norm_eps", "rms_norm_eps");
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, SENSENOVA_U1);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_DIM, hidden);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, eps);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS,
                HfMmprojHooks.intOr(cfg, vision, SENSENOVA_MIN_PIXELS, "min_pixels"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS,
                HfMmprojHooks.intOr(cfg, vision, SENSENOVA_MAX_PIXELS, "max_pixels"));
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, merge);
    }

    /**
     * {@code SenseNovaU1VisionModel.modify_tensors} (sensenova_u1.py:108-124): the four embedding tensors take their
     * patch and projector names; anything else is refused.
     *
     * @throws GgufFormatException naming the unexpected tensor
     */
    public static void senseNovaTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if ("vision_model.embeddings.patch_embedding.weight".equals(hfName)) {
            out.add("v.patch_embd.weight", src);
            return;
        }
        if ("vision_model.embeddings.patch_embedding.bias".equals(hfName)) {
            out.add("v.patch_embd.bias", src);
            return;
        }
        if ("vision_model.embeddings.dense_embedding.weight".equals(hfName)) {
            out.add(ctx.names().ggufName("V_MMPROJ", 0, ".weight"), src);
            return;
        }
        if ("vision_model.embeddings.dense_embedding.bias".equals(hfName)) {
            out.add(ctx.names().ggufName("V_MMPROJ", 0, ".bias"), src);
            return;
        }
        throw new GgufFormatException("Unexpected SenseNova U1 vision tensor: '" + hfName + "' (sensenova_u1.py:122)");
    }

    /** The dense embedding keeps F16 at {@code --outtype f16} (sensenova_u1.py:126-131). */
    public static final HfForceType SENSENOVA_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.equals("mm.0.weight")) {
            return outType == com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType.F16
                    ? GGMLType.F16 : GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    // ------------------------------------------------------------------ LFM2-VL

    /** {@code LFM2VLModel.filter_tensors} (lfm2.py:384-390): the tower and projector prefixes lose {@code model.}. */
    public static final HfNameFilter LFM2_FILTER = (name, base) -> {
        String renamed = name.replace("model.vision_tower.", "vision_tower.")
                .replace("model.multi_modal_projector.", "multi_modal_projector.");
        return base.apply(renamed);
    };

    /**
     * {@code LFM2VLModel.set_gguf_parameters} (lfm2.py:372-383): the base keys at 256 pixels, {@code lfm2}, the
     * required epsilon, the root downsample factor, GELU, the truncated block count, the resize algorithm and the
     * processor's tiling parameters.
     *
     * @throws GgufFormatException naming the missing or unsupported key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void lfm2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, 256);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, LFM2);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, HfMmprojHooks.visionF64(cfg, "layer_norm_eps"));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, rootIntOr(cfg, 2, "downsample_factor"));
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        int blocks = HfMmprojHooks.visionBlocks(cfg);
        int featureLayer = rootIntOr(cfg, -1, "vision_feature_layer");
        int drop = Math.max(0, -(featureLayer + 1));
        if (drop >= blocks) {
            throw new GgufFormatException(cfg.source() + ": vision_feature_layer " + featureLayer + " drops every"
                    + " one of the " + blocks + " vision blocks");
        }
        kv.set(GgufKeys.CLIP_VISION_BLOCK_COUNT, blocks - drop);
        Object raw = HfMmprojHooks.preprocessor(cfg).raw("resample");
        if (raw != null) {
            int code = rootIntValue(cfg, raw, "preprocessor resample");
            String algorithm;
            switch (code) {
                case 1: algorithm = "lanczos"; break;
                case 2: algorithm = "bilinear"; break;
                case 3: algorithm = "bicubic"; break;
                default:
                    throw new GgufFormatException(cfg.source() + ": unsupported resample: " + raw
                            + " (lfm2.py:377-380)");
            }
            kv.set(GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO, algorithm);
        }
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        if (HfMmprojHooks.boolOr(pre, true, "do_image_splitting")) {
            kv.set(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, HfMmprojHooks.intOr(cfg, pre, 2, "min_tiles"));
            kv.set(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, HfMmprojHooks.intOr(cfg, pre, 10, "max_tiles"));
        } else {
            kv.set(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, 1);
            kv.set(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, 1);
        }
        kv.set(GgufKeys.CLIP_VISION_PREPROC_TILE_SIZE, HfMmprojHooks.intOr(cfg, pre, 512, "tile_size"));
        kv.set(GgufKeys.CLIP_VISION_PREPROC_MAX_PIXELS_TOLERANCE,
                HfMmprojHooks.f64Or(cfg, pre, 2.0, "max_pixels_tolerance"));
    }

    /**
     * {@code LFM2VLModel.modify_tensors} (lfm2.py:392-397): the flat patch embedding is viewed as
     * {@code [n_embd, 16, 16, 3]} and permuted to the convolution kernel {@code [n_embd, 3, 16, 16]}.
     *
     * @throws GgufFormatException when the embedding does not hold whole 16-pixel RGB patches
     */
    public static void lfm2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.contains("patch_embedding.weight")) {
            long[] shape = src.shape();
            long numel = 1;
            for (long d : shape) {
                numel = Math.multiplyExact(numel, d);
            }
            if (shape.length < 1 || shape[0] <= 0
                    || numel != shape[0] * 3L * LFM2_PATCH_AREA * LFM2_PATCH_AREA) {
                throw new GgufFormatException("LFM2-VL patch embedding has shape " + Arrays.toString(shape)
                        + "; expected [n_embd, " + (3 * LFM2_PATCH_AREA * LFM2_PATCH_AREA) + "] (lfm2.py:391-392)");
            }
            RowSource viewed = ElementTransforms.reshape(src, shape[0], LFM2_PATCH_AREA, LFM2_PATCH_AREA, 3);
            out.add(ctx.names().mapOrThrow(hfName),
                    ElementTransforms.permuteDims(viewed, 0, 3, 1, 2));
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    // ------------------------------------------------------------------ root configuration helpers

    /** The required integer {@code key} of the root configuration. */
    private static int rootInt(HfConfig cfg, String key) {
        return rootIntValue(cfg, cfg.raw().raw(key), key);
    }

    /** The integer {@code key} of the root configuration, or {@code dflt} when absent. */
    private static int rootIntOr(HfConfig cfg, int dflt, String key) {
        Object raw = cfg.raw().raw(key);
        return raw instanceof Number && !(raw instanceof Boolean) ? ((Number) raw).intValue() : dflt;
    }

    private static int rootIntValue(HfConfig cfg, Object raw, String key) {
        if (!(raw instanceof Number) || raw instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": the configuration has no integer '" + key + "'");
        }
        return ((Number) raw).intValue();
    }
}
