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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The LFM3-VL vision tower and projector (LC/conversion/lfm3.py:179-311) of
 * {@code Lfm3VlForConditionalGeneration}: the two-slice temporal patchifier split into {@code v.patch_embd.weight}
 * and {@code v.patch_embd.weight.1}, the embedding norm, the depthwise spatial conv and the
 * LayerNorm-GELU-linear projector, with the {@code lfm3vl} projector type the clip graph reads. The checkpoint must
 * carry the fixed GELU projector, the 1e6 vision rope base, three RGB input channels and the 3x3 patch convolution
 * the runtime hardcodes; anything else is refused by name. The Omni export's audio tower and codec are
 * {@link HfLfmOmniHooks}.
 */
public final class HfLfm3VisionHooks {

    /** {@code VisionProjectorType.LFM3VL}. */
    public static final String PROJECTOR = "lfm3vl";

    /** The dynamic image size the reference writes for the patchifier (lfm3.py:302). */
    private static final int IMAGE_SIZE = 512;

    /** The vision RMS-norm epsilon of the release checkpoints (lfm3.py:190). */
    private static final double VISION_NORM_EPS = 1e-6;

    /** The projector's input LayerNorm epsilon the clip graph hardcodes (lfm3.py:186-189). */
    private static final double PROJECTOR_NORM_EPS = 1e-6;

    /** The fixed video configuration of the runtime (lfm3.py:204-205), without the class's frame bound. */
    private static final Map<String, Object> VIDEO = videoConfig();

    /** The video frame bound of the vision-only class (lfm3.py:209). */
    public static final int VIDEO_MAX_FRAMES = 768;

    private HfLfm3VisionHooks() {
    }

    /**
     * {@code LFM3VLMmprojModel.modify_tensors} (lfm3.py:307-311): the tower and the multimodal projector survive,
     * every other checkpoint tensor (the text model, the audio branches) drops.
     */
    public static final HfNameFilter FILTER = (name, base) ->
            name.startsWith("model.vision_tower.") || name.startsWith("model.multi_modal_projector.")
                    ? base.apply(name) : null;

    /**
     * {@code _set_lfm3vl_gguf_parameters} (lfm3.py:182-208): the base vision keys, the {@code lfm3vl} projector, the
     * KV head count, the 2x2 merge factor, the attention and projector epsilons, SiLU, the pixel bounds and the
     * resize algorithm, after checking every value the clip graph hardcodes.
     *
     * @throws GgufFormatException naming the unsupported or missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        meta(ctx, kv, plan, GgufKeys.CLIP_PROJECTOR_TYPE);
    }

    /**
     * {@link #meta(HfLoadContext, HfMetadataBuilder, HfPlanView)} writing the projector type under
     * {@code projectorKey}: the common key for a vision-only file, {@code clip.vision.projector_type} when the file
     * also carries the Omni audio tower.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, String projectorKey)
            throws IOException {
        meta(ctx, kv, plan, projectorKey, VIDEO_MAX_FRAMES);
    }

    /**
     * {@link #meta(HfLoadContext, HfMetadataBuilder, HfPlanView, String)} with the video frame bound the class
     * hardcodes: 768 for the vision-only checkpoint, 384 for the Omni one ({@code LFM3OmniMmprojModel._video_max_frames},
     * lfm3.py:581).
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, String projectorKey,
                            int videoMaxFrames) throws IOException {
        HfConfig cfg = ctx.config();
        if (!"gelu".equals(string(cfg, "projector_hidden_act"))) {
            throw new GgufFormatException(cfg.source() + ": unsupported projector_hidden_act "
                    + cfg.flat().get("projector_hidden_act") + "; the clip LFM3-VL projector applies GELU");
        }
        if (!Boolean.TRUE.equals(cfg.flat().get("projector_bias"))) {
            throw new GgufFormatException(cfg.source() + ": unsupported projector_bias "
                    + cfg.flat().get("projector_bias") + "; the clip LFM3-VL projector has biases");
        }
        if (number(cfg, "projector_layer_norm_eps") != PROJECTOR_NORM_EPS) {
            throw new GgufFormatException(cfg.source() + ": unsupported projector_layer_norm_eps "
                    + cfg.flat().get("projector_layer_norm_eps") + "; the clip LFM3-VL projector normalizes with "
                    + PROJECTOR_NORM_EPS);
        }
        JsonView vision = vision(cfg);
        if (ropeTheta(vision, cfg) != 1e6) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision rope_parameters.rope_theta;"
                    + " the clip LFM3-VL tower uses 1e6");
        }
        if (intOf(vision, cfg, "in_channels") != 3) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision in_channels "
                    + vision.raw("in_channels") + "; the clip LFM3-VL patchifier takes RGB");
        }
        if (intOf(vision, cfg, "conv_kernel_size") != 3) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision conv_kernel_size "
                    + vision.raw("conv_kernel_size") + "; the clip LFM3-VL spatial conv is 3x3");
        }
        HfMmprojHooks.visionMeta(ctx, kv, IMAGE_SIZE, intOf(vision, cfg, "patch_size"));
        kv.set(projectorKey, PROJECTOR);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT_KV,
                HfMmprojHooks.visionInt(cfg, "num_key_value_heads"));
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, intOf(cfg, cfg.raw(), "downsample_factor"));
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "norm_eps").orElse(VISION_NORM_EPS));
        kv.set(GgufKeys.CLIP_USE_SILU, true);
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        JsonView size = object(cfg, pre.raw("size"), "preprocessor size");
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, integer(cfg, size, "shortest_edge"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, integer(cfg, size, "longest_edge"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO, resizeAlgorithm(cfg, pre.raw("resample")));
        checkVideo(cfg, pre, videoMaxFrames);
    }

    /**
     * {@code LFM3VLVisionMixin._lfm3vl_tensors} (lfm3.py:225-266): the patchifier splits along its temporal axis, the
     * embedding norm, conv and projector tensors take their clip names, and the tower blocks map through the
     * converter's name table.
     *
     * @throws GgufFormatException when the patchifier shape does not match the vision configuration or a projector
     *                             tensor is unknown
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if ("model.vision_tower.patchifier.weight".equals(hfName)) {
            patchifier(ctx, src, out);
            return;
        }
        if ("model.vision_tower.embedding_norm.weight".equals(hfName)) {
            out.add("v.embedding_norm.weight", src);
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= 0 && hfName.endsWith("conv.in_proj.weight")) {
            out.add("v.blk." + bid + ".conv2d_in_proj.weight", src);
            return;
        }
        if (bid >= 0 && hfName.endsWith("conv.spatial_conv.weight")) {
            out.add("v.blk." + bid + ".conv2d_spatial.weight", src);
            return;
        }
        if (bid >= 0 && hfName.endsWith("conv.out_proj.weight")) {
            out.add("v.blk." + bid + ".conv2d_out_proj.weight", src);
            return;
        }
        if (hfName.startsWith("model.multi_modal_projector.")) {
            String rest = hfName.substring("model.multi_modal_projector.".length());
            int dot = rest.indexOf('.');
            if (dot < 0) {
                throw new GgufFormatException("Unsupported LFM3-VL projector tensor '" + hfName + "'");
            }
            String layer = rest.substring(0, dot);
            String target;
            if ("layer_norm".equals(layer)) {
                target = "mm.input_norm";
            } else if ("linear_1".equals(layer)) {
                target = "mm.0";
            } else if ("linear_2".equals(layer)) {
                target = "mm.2";
            } else {
                throw new GgufFormatException("Unsupported LFM3-VL projector tensor '" + hfName + "'");
            }
            out.add(target + rest.substring(dot), src);
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The patchifier {@code [n_embd, 3, temporal, p, p]} split into one kernel per temporal slice (lfm3.py:227-238). */
    private static void patchifier(HfLoadContext ctx, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        int temporal = intOf(vision, cfg, "temporal_patch_size");
        int patch = intOf(vision, cfg, "patch_size");
        long[] shape = src.shape();
        if (shape.length != 5 || shape[1] != 3 || shape[2] != temporal || shape[3] != patch
                || shape[4] != patch) {
            throw new GgufFormatException("LFM3-VL patchifier has shape " + Arrays.toString(shape)
                    + "; expected [n_embd, 3, " + temporal + ", " + patch + ", " + patch + "] (lfm3.py:230-233)");
        }
        out.add("v.patch_embd.weight", ElementTransforms.squeezeConv(RowTransforms.sliceDim(src, 2, 0, 1)));
        if (temporal == 2) {
            out.add("v.patch_embd.weight.1", ElementTransforms.squeezeConv(RowTransforms.sliceDim(src, 2, 1, 1)));
        } else if (temporal != 1) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision temporal_patch_size " + temporal
                    + "; the clip patchifier has one or two temporal slices");
        }
    }

    /** The patch kernels stay F16, the tiny depthwise spatial conv F32 (lfm3.py:216-224). */
    public static final HfForceType FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains("patch_embd")) {
            return GGMLType.F16;
        }
        if (ggufName.contains("conv2d_spatial")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.VISION_EMBEDDINGS_F32.force(ggufName, nDims, source, outType);
    };

    /** The {@code video_processor} block of the preprocessor config, else the side file (lfm3.py:200-208). */
    private static void checkVideo(HfConfig cfg, JsonView pre, int maxFrames) throws IOException {
        Object video = pre.raw("video_processor");
        JsonView v;
        if (video instanceof Map) {
            v = object(cfg, video, "preprocessor video_processor");
        } else {
            java.nio.file.Path path = cfg.sideFile("video_preprocessor_config.json");
            if (!java.nio.file.Files.isRegularFile(path)) {
                return;
            }
            v = HfBertVariantHooks.readJsonView(path);
        }
        for (Map.Entry<String, Object> e : VIDEO.entrySet()) {
            Object actual = v.raw(e.getKey());
            if (!valuesEqual(e.getValue(), actual)) {
                throw new GgufFormatException(cfg.source() + ": unsupported video " + e.getKey() + "=" + actual
                        + ", expected " + e.getValue() + " (lfm3.py:204-208)");
            }
        }
        Object actualMax = v.raw("max_frames");
        if (!valuesEqual(maxFrames, actualMax)) {
            throw new GgufFormatException(cfg.source() + ": unsupported video max_frames=" + actualMax
                    + ", expected " + maxFrames + " (lfm3.py:205, :581)");
        }
    }

    /** JSON equality with numbers by value and integer lists element by element. */
    private static boolean valuesEqual(Object expected, Object actual) {
        if (expected instanceof Number && actual instanceof Number) {
            return ((Number) expected).doubleValue() == ((Number) actual).doubleValue();
        }
        if (expected instanceof Map && actual instanceof Map) {
            Map<?, ?> e = (Map<?, ?>) expected;
            Map<?, ?> a = (Map<?, ?>) actual;
            if (e.size() != a.size()) {
                return false;
            }
            for (Map.Entry<?, ?> entry : e.entrySet()) {
                if (!a.containsKey(entry.getKey()) || !valuesEqual(entry.getValue(), a.get(entry.getKey()))) {
                    return false;
                }
            }
            return true;
        }
        return expected == null ? actual == null : expected.equals(actual);
    }

    private static Map<String, Object> videoConfig() {
        Map<String, Object> size = new LinkedHashMap<>();
        size.put("shortest_edge", 4096);
        size.put("longest_edge", 25165824);
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("fps", 2.0);
        m.put("min_frames", 4);
        m.put("temporal_patch_size", 2);
        m.put("size", size);
        return java.util.Collections.unmodifiableMap(m);
    }

    /** The {@code PIL} resample enum the torchvision backend supports (lfm3.py:193-196). */
    private static String resizeAlgorithm(HfConfig cfg, Object resample) {
        if (resample instanceof Number && ((Number) resample).intValue() == 2) {
            return "bilinear";
        }
        if (resample instanceof Number && ((Number) resample).intValue() == 3) {
            return "bicubic";
        }
        throw new GgufFormatException(cfg.source() + ": unsupported preprocessor resample " + resample
                + "; the clip resampler is bilinear or bicubic");
    }

    private static JsonView vision(HfConfig cfg) {
        return cfg.vision().orElseThrow(() -> new GgufFormatException(cfg.source() + ": the checkpoint has no"
                + " 'vision_config'; this projector converts the LFM3-VL vision tower"));
    }

    private static double ropeTheta(JsonView vision, HfConfig cfg) {
        Object rope = vision.raw("rope_parameters");
        if (!(rope instanceof Map) || !(((Map<?, ?>) rope).get("rope_theta") instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": the vision configuration has no numeric"
                    + " rope_parameters.rope_theta");
        }
        return ((Number) ((Map<?, ?>) rope).get("rope_theta")).doubleValue();
    }

    private static String string(HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof String)) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no '" + key + "'");
        }
        return (String) v;
    }

    private static double number(HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no number '" + key + "'");
        }
        return ((Number) v).doubleValue();
    }

    private static int intOf(JsonView view, HfConfig cfg, String key) {
        return integer(cfg, view, key);
    }

    private static int intOf(HfConfig cfg, JsonView view, String key) {
        return integer(cfg, view, key);
    }

    private static int integer(HfConfig cfg, JsonView view, String key) {
        Object v = view.raw(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no integer '" + key + "'");
        }
        return ((Number) v).intValue();
    }

    private static JsonView object(HfConfig cfg, Object value, String what) {
        if (!(value instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": the " + what + " is not an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) value;
        return JsonView.of(map, what);
    }
}
