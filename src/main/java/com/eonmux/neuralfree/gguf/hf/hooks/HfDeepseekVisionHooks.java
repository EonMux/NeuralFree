package com.eonmux.neuralfree.gguf.hf.hooks;

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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The DeepSeek-V4-Flash vision tower and its DeepSeek-V4.1 variant (LC/conversion/deepseek.py:1041-1088 and
 * LC/conversion/deepseek41.py:427-459): a patchifier over flattened {@code (3, p, p)} patches, SwiGLU blocks whose
 * fused {@code mlp.w1} splits into the gate and up projections, the {@code aligner} projector, the image sentinels and
 * the {@code deepseek4v} / {@code deepseek41v} projector types. The V4.1 checkpoint nests its tower under
 * {@code vision_config}; the V4 checkpoint keeps the {@code vision_*} keys at the top level.
 */
public final class HfDeepseekVisionHooks {

    /** {@code VisionProjectorType.DEEPSEEK4V}. */
    public static final String DEEPSEEK4V = "deepseek4v";
    /** {@code VisionProjectorType.DEEPSEEK41V}. */
    public static final String DEEPSEEK41V = "deepseek41v";

    /** The vision RMS-norm epsilon, the PyTorch default rather than the text model's (deepseek.py:1075). */
    private static final double VISION_NORM_EPS = 1e-6;

    /** The per-pixel normalization of a checkpoint without a preprocessor config (deepseek.py:1048-1051). */
    private static final List<Float> DEFAULT_COLORS = Arrays.asList(0.5f, 0.5f, 0.5f);

    private HfDeepseekVisionHooks() {
    }

    /** {@code vision_n_layers}: the V4 tower's block count. */
    public static int v4Blocks(HfConfig cfg) {
        return topInt(cfg, "vision_n_layers");
    }

    /**
     * {@code DeepseekV4FlashVisionModel.filter_tensors} (deepseek.py:1085-1089): the vision tower, the aligner and the
     * image sentinels survive; the language model drops.
     */
    public static final HfNameFilter FILTER = (name, base) ->
            name.startsWith("vision.") || name.startsWith("aligner.") || name.startsWith("image_")
                    ? base.apply(name) : null;

    /**
     * {@code DeepseekV4FlashVisionModel.set_gguf_parameters} (deepseek.py:1070-1083): the base vision keys from the
     * top-level {@code vision_*} configuration, the {@code deepseek4v} projector, the downsample scale factor and the
     * minimum pixel count, after checking the token and ratio limits clip.cpp hardcodes.
     *
     * @throws GgufFormatException naming the unsupported or missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void v4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        int layers = topInt(cfg, "vision_n_layers");
        if (layers <= 0) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no 'vision_n_layers'; this"
                    + " projector converts the DeepSeek-V4-Flash vision tower");
        }
        int intDim = topInt(cfg, "vision_inter_dim");
        int patch = topInt(cfg, "vision_patch_size");
        int downsample = topInt(cfg, "vision_downsample_ratio");
        int imageSize = patch * downsample * 16;
        base(ctx, kv, imageSize, patch, topInt(cfg, "vision_dim"), intDim, topInt(cfg, "vision_n_heads"), layers);
        projector(kv, DEEPSEEK4V, downsample, topInt(cfg, "vision_min_pixels"));
        if (topInt(cfg, "vision_max_n_token") != 384) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision_max_n_token "
                    + cfg.flat().get("vision_max_n_token") + "; clip.cpp fixes it at 384");
        }
        if (topInt(cfg, "vision_max_wh_ratio") != 8) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision_max_wh_ratio "
                    + cfg.flat().get("vision_max_wh_ratio") + "; clip.cpp fixes it at 8");
        }
    }

    /**
     * {@code DeepseekV41VisionModel.set_gguf_parameters} (deepseek41.py:448-459): the base vision keys from the nested
     * {@code vision_config}, the {@code deepseek41v} projector, the downsample scale factor and the minimum pixels,
     * after checking the token, ratio and rope values clip.cpp hardcodes.
     *
     * @throws GgufFormatException naming the unsupported or missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void v41Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = cfg.vision().orElseThrow(() -> new GgufFormatException(cfg.source()
                + ": the checkpoint has no 'vision_config'"));
        int layers = intOf(cfg, vision, "num_hidden_layers");
        if (layers <= 0) {
            throw new GgufFormatException(cfg.source() + ": DeepseekV41VisionModel requires"
                    + " vision_config.num_hidden_layers > 0");
        }
        int patch = intOf(cfg, vision, "patch_size");
        int downsample = intOf(cfg, vision, "downsample_ratio");
        base(ctx, kv, patch * downsample * 16, patch, intOf(cfg, vision, "hidden_size"),
                intOf(cfg, vision, "intermediate_size"), intOf(cfg, vision, "num_attention_heads"), layers);
        projector(kv, DEEPSEEK41V, downsample, intOf(cfg, vision, "min_pixels"));
        if (intOf(cfg, vision, "max_image_tokens") != 1024) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision_config.max_image_tokens;"
                    + " clip.cpp fixes it at 1024");
        }
        if (vision.has("max_wh_ratio") && vision.raw("max_wh_ratio") != null) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision_config.max_wh_ratio "
                    + vision.raw("max_wh_ratio") + "; clip.cpp expects no limit");
        }
        Object theta = vision.raw("rope_theta");
        if (theta != null && number(cfg, theta, "vision_config.rope_theta") != 10000.0) {
            throw new GgufFormatException(cfg.source() + ": unsupported vision_config.rope_theta " + theta
                    + "; clip.cpp fixes it at 10000");
        }
    }

    /**
     * {@code DeepseekV4FlashVisionModel.modify_tensors} (deepseek.py:1091-1105): the patchifier reshaped to a
     * convolution kernel and the fused {@code mlp.w1} split into gate and up; everything else maps through the
     * converter's clip name table.
     *
     * @throws GgufFormatException when the patchifier shape does not match the configuration
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if ("vision.patch_embed.proj.weight".equals(hfName)) {
            int patch = patchSize(ctx.config());
            long[] shape = src.shape();
            if (shape.length != 2 || shape[1] != 3L * patch * patch) {
                throw new GgufFormatException("DeepSeek vision patch embedding has shape "
                        + Arrays.toString(shape) + "; expected [n_embd, " + (3L * patch * patch) + "]"
                        + " (deepseek.py:1094-1097)");
            }
            out.add(ctx.names().mapOrThrow(hfName), ElementTransforms.reshape(src, shape[0], 3, patch, patch));
            return;
        }
        if (hfName.contains(".mlp.w1.")) {
            long[] shape = src.shape();
            if (shape.length < 1 || shape[0] % 2 != 0) {
                throw new GgufFormatException("DeepSeek vision fused mlp.w1 has shape " + Arrays.toString(shape)
                        + "; its rows are the even gate and up halves (deepseek.py:1099-1103)");
            }
            long half = shape[0] / 2;
            out.add(ctx.names().mapOrThrow(hfName.replace("w1", "w1_gate")),
                    RowTransforms.sliceRows(src, 0, half));
            out.add(ctx.names().mapOrThrow(hfName.replace("w1", "w1_up")),
                    RowTransforms.sliceRows(src, half, half));
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The patch kernel keeps F16 at {@code --outtype f16} and F32 otherwise (base.py:3229-3232). */
    public static final HfForceType FORCE = HfMmprojHooks.PATCH_KERNELS;

    private static void projector(HfMetadataBuilder kv, String type, int scaleFactor, long minPixels) {
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, type);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, VISION_NORM_EPS);
        kv.set(GgufKeys.CLIP_USE_SILU, true);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, scaleFactor);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, minPixels);
    }

    /** The base {@code MmprojModel} vision keys with the 0.5 normalization defaults (base.py:3200-3220). */
    private static void base(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize, int patchSize, int hidden,
                             int ffn, int heads, int blocks) throws IOException {
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.CLIP_HAS_VISION_ENCODER, true);
        kv.set(GgufKeys.CLIP_VISION_PROJECTION_DIM, HfMmprojHooks.textWidth(cfg));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_SIZE, imageSize);
        kv.set(GgufKeys.CLIP_VISION_PATCH_SIZE, patchSize);
        kv.set(GgufKeys.CLIP_VISION_EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.CLIP_VISION_FEED_FORWARD_LENGTH, ffn);
        kv.set(GgufKeys.CLIP_VISION_BLOCK_COUNT, blocks);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT, heads);
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MEAN, colors(cfg, pre, "image_mean"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_STD, colors(cfg, pre, "image_std"));
    }

    private static List<Float> colors(HfConfig cfg, JsonView pre, String key) {
        if (!pre.has(key)) {
            return DEFAULT_COLORS;
        }
        Object raw = pre.raw(key);
        if (!(raw instanceof List) || ((List<?>) raw).size() < 3) {
            throw new GgufFormatException(cfg.source() + ": the preprocessor configuration's '" + key
                    + "' needs three channels");
        }
        List<Float> out = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            out.add((float) number(cfg, ((List<?>) raw).get(i), "preprocessor " + key));
        }
        return out;
    }

    private static int patchSize(HfConfig cfg) {
        return cfg.vision().isPresent() ? intOf(cfg, cfg.vision().get(), "patch_size")
                : topInt(cfg, "vision_patch_size");
    }

    private static int topInt(HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no integer '" + key + "'");
        }
        return ((Number) v).intValue();
    }

    private static int intOf(HfConfig cfg, JsonView view, String key) {
        Object v = view.raw(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no integer '" + key + "'");
        }
        return ((Number) v).intValue();
    }

    private static double number(HfConfig cfg, Object value, String where) {
        if (!(value instanceof Number) || value instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' is not a number");
        }
        return ((Number) value).doubleValue();
    }
}
