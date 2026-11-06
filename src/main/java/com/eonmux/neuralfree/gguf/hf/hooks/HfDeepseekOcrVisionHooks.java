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
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The DeepSeek-OCR vision projector (LC/conversion/deepseek.py:22-102), shared by {@code DeepseekOCRForCausalLM},
 * {@code DeepseekOCR2ForCausalLM} and {@code UnlimitedOCRVisionModel}: the SAM ViT and its neck, the CLIP tower, the
 * single-layer projector, the image sentinels and the {@code deepseekocr} projector type, plus the per-model tile
 * limit the reference class overrides.
 */
public final class HfDeepseekOcrVisionHooks {

    /** {@code VisionProjectorType.DEEPSEEKOCR}. */
    public static final String PROJECTOR = "deepseekocr";
    /** {@code VisionProjectorType.DEEPSEEKOCR2}. */
    public static final String PROJECTOR_OCR2 = "deepseekocr2";

    /** The vision layer-norm epsilon (deepseek.py:35). */
    private static final double DEFAULT_NORM_EPS = 1e-6;
    /** The CLIP window size the checkpoint configuration omits (deepseek.py:46). */
    private static final int DEFAULT_WINDOW = 14;
    /** The default image sequence length (deepseek.py:40). */
    private static final int DEFAULT_SEQ_LENGTH = 256;
    /** The DeepSeek-OCR 2 Qwen2 encoder defaults (deepseek.py:120-126). */
    private static final int OCR2_PATCH_SIZE = 16;
    private static final int OCR2_INTERMEDIATE_SIZE = 4864;
    private static final int OCR2_HEADS = 14;
    private static final int OCR2_LAYERS = 24;
    /** The DeepSeek-OCR 2 key/value head count (deepseek.py:129). */
    private static final int OCR2_KV_HEADS = 2;

    private HfDeepseekOcrVisionHooks() {
    }

    /**
     * {@code DeepseekOCRVisionModel.filter_tensors} (deepseek.py:87-102): the language model drops, the SAM and CLIP
     * position tables gain their missing {@code .weight}, and the rest goes through the inherited filter.
     */
    public static final HfNameFilter FILTER = (name, base) -> {
        if (name.startsWith("lm_head.") || name.startsWith("model.embed_tokens.")
                || name.startsWith("model.layers.") || name.startsWith("model.norm.")) {
            return null;
        }
        String renamed = name;
        if (name.endsWith("pos_embed") || name.endsWith("rel_pos_h") || name.endsWith("rel_pos_w")) {
            renamed = name + ".weight";
        }
        return base.apply(renamed);
    };

    /**
     * {@code DeepseekOCRVisionModel.set_gguf_parameters} (deepseek.py:32-59): the base vision keys from the merged
     * vision configuration, the {@code deepseekocr} projector, the SAM geometry and the preprocessor tile bounds.
     *
     * @throws GgufFormatException naming the missing or unsupported key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        boolean ocr2 = "DeepseekOCR2ForCausalLM".equals(cfg.architecture());
        JsonView vision = mergedVision(cfg);
        int imageSize = intOf(cfg, vision, "image_size");
        int patch = intOr(cfg, vision, "patch_size", ocr2 ? OCR2_PATCH_SIZE : -1);
        if (patch < 0) {
            throw new GgufFormatException(cfg.source() + ": the vision configuration has no 'patch_size'");
        }
        int ffn = intOr(cfg, vision, HfMmprojHooks.FFN_KEYS, ocr2 ? OCR2_INTERMEDIATE_SIZE : -1);
        if (ffn < 0) {
            ffn = intOf(cfg, vision, HfMmprojHooks.FFN_KEYS);
        }
        int heads = intOr(cfg, vision, "num_heads", ocr2 ? OCR2_HEADS : -1);
        if (heads < 0) {
            heads = intOf(cfg, vision, "num_heads", "heads");
        }
        int blocks = blocksOr(cfg, vision, ocr2 ? OCR2_LAYERS : -1);
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, patch, intOf(cfg, vision, HfMmprojHooks.EMBEDDING_KEYS),
                ffn, heads, blocks);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, ocr2 ? PROJECTOR_OCR2 : PROJECTOR);
        if (ocr2) {
            kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT_KV, OCR2_KV_HEADS);
        }
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                numberOr(cfg, vision, "layer_norm_eps", DEFAULT_NORM_EPS));
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        long seq = seqLength(cfg);
        int perSide = (int) Math.sqrt(seq);
        int factor = perSide > 0 ? (imageSize / patch) / perSide : 0;
        if (factor > 0 && factor != 4) {
            kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, factor);
        }
        kv.set(GgufKeys.CLIP_VISION_WINDOW_SIZE, intOr(cfg, vision, "window_size", DEFAULT_WINDOW));
        kv.set(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, 2);
        JsonView sam = object(cfg, vision.raw("sam"), "vision_config.sam");
        kv.set(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, maxTiles(cfg));
        kv.set(GgufKeys.CLIP_VISION_SAM_BLOCK_COUNT, intOf(cfg, sam, "layers"));
        kv.set(GgufKeys.CLIP_VISION_SAM_EMBEDDING_LENGTH, intOf(cfg, sam, "width"));
        kv.set(GgufKeys.CLIP_VISION_SAM_HEAD_COUNT, intOf(cfg, sam, "heads"));
    }

    /**
     * {@code DeepseekOCRVisionModel.modify_tensors} (deepseek.py:77-84): the separator is stored as a row, everything
     * else maps through the converter's name table.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (hfName.endsWith("view_seperator")) {
            long[] shape = src.shape();
            if (shape.length != 1) {
                throw new GgufFormatException("DeepSeek-OCR view separator has shape "
                        + java.util.Arrays.toString(shape) + "; expected one row of values");
            }
            out.add(ctx.names().mapOrThrow(hfName), ElementTransforms.reshape(src, 1, shape[0]));
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The SAM and CLIP position tables, necks and nets keep F32; the patch kernel follows the base rule. */
    public static final HfForceType FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains("v.sam.") || ggufName.contains("v.class_embd")
                || ggufName.contains("v.position_embd") || ggufName.contains("v.patch_embd")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    /**
     * The projector's block count: the merged vision configuration's count, or the Qwen2 encoder's fixed 24 layers
     * for DeepSeek-OCR 2, whose configuration leaves them out (deepseek.py:133-137).
     */
    public static int blocks(HfConfig cfg) {
        JsonView vision = mergedVision(cfg);
        if ("DeepseekOCR2ForCausalLM".equals(cfg.architecture())
                && HfMmprojHooks.intOr(cfg, vision, -1, HfMmprojHooks.BLOCK_COUNT_KEYS
                        .toArray(new String[0])) < 0) {
            return OCR2_LAYERS;
        }
        return HfMmprojHooks.blocksOf(cfg, vision, "vision_config");
    }

    /** The per-class tile limit: 9 in the base class, overridden by OCR2 and Unlimited. */
    private static int maxTiles(HfConfig cfg) {
        String arch = cfg.architecture();
        if ("UnlimitedOCRForCausalLM".equals(arch)) {
            return 32;
        }
        if ("DeepseekOCR2ForCausalLM".equals(arch)) {
            return 6;
        }
        return 9;
    }

    /** The processor's {@code image_seq_length}, 256 without one (deepseek.py:40). */
    private static long seqLength(HfConfig cfg) throws IOException {
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        Object v = pre.raw("image_seq_length");
        return v instanceof Number ? ((Number) v).longValue() : DEFAULT_SEQ_LENGTH;
    }

    /**
     * {@code DeepseekOCRVisionModel.get_vision_config} (deepseek.py:61-75): the SAM ViT-B block injected under
     * {@code sam}, the CLIP-L keys merged over the vision configuration, and {@code hidden_size} / {@code num_heads}
     * / {@code intermediate_size} derived from {@code width} and {@code heads}.
     */
    private static JsonView mergedVision(HfConfig cfg) {
        JsonView raw = cfg.vision().orElseThrow(() -> new GgufFormatException(cfg.source()
                + ": DeepseekOCR model requires 'vision_config' in the model configuration"));
        Map<String, Object> merged = new LinkedHashMap<>();
        for (String key : raw.keys()) {
            merged.put(key, raw.raw(key));
        }
        Object width = merged.get("width");
        if (width instanceof Map) {
            Map<?, ?> widths = (Map<?, ?>) width;
            Object sam = widths.get("sam_vit_b");
            if (sam instanceof Map) {
                merged.put("sam", sam);
            }
            Object clip = widths.get("clip-l-14-224");
            if (clip instanceof Map) {
                for (Map.Entry<?, ?> e : ((Map<?, ?>) clip).entrySet()) {
                    merged.put(String.valueOf(e.getKey()), e.getValue());
                }
            }
            Object qwen2 = widths.get("qwen2-0_5b");
            if (qwen2 instanceof Map) {
                Object dim = ((Map<?, ?>) qwen2).get("dim");
                if (!(dim instanceof Number)) {
                    throw new GgufFormatException(cfg.source() + ": vision_config.width['qwen2-0_5b'] has no 'dim'");
                }
                merged.put("hidden_size", dim);
            }
            if (!(sam instanceof Map) && !(clip instanceof Map) && !(qwen2 instanceof Map)) {
                throw new GgufFormatException(cfg.source() + ": vision_config.width names no known tower");
            }
        } else if (width instanceof Number) {
            merged.put("hidden_size", width);
        }
        Object heads = merged.get("heads");
        if (heads instanceof Number && !(heads instanceof Boolean)) {
            merged.put("num_heads", heads);
            merged.put("intermediate_size", ((Number) heads).intValue() * 4);
        }
        return JsonView.of(merged, "the merged vision configuration");
    }

    private static int intOf(HfConfig cfg, JsonView view, String... aliases) {
        for (String alias : aliases) {
            Object v = view.raw(alias);
            if (v instanceof Number && !(v instanceof Boolean)) {
                return ((Number) v).intValue();
            }
        }
        throw new GgufFormatException(cfg.source() + ": the vision configuration has none of "
                + java.util.Arrays.toString(aliases));
    }

    private static int intOf(HfConfig cfg, JsonView view, String key) {
        return intOf(cfg, view, new String[]{key});
    }

    private static int intOr(HfConfig cfg, JsonView view, String[] keys, int dflt) {
        for (String key : keys) {
            Object v = view.raw(key);
            if (v instanceof Number && !(v instanceof Boolean)) {
                return ((Number) v).intValue();
            }
        }
        return dflt;
    }

    private static int blocksOr(HfConfig cfg, JsonView view, int dflt) {
        for (String key : HfMmprojHooks.BLOCK_COUNT_KEYS) {
            Object v = view.raw(key);
            if (v instanceof Number && !(v instanceof Boolean)) {
                return ((Number) v).intValue();
            }
        }
        return dflt;
    }

    private static int intOr(HfConfig cfg, JsonView view, String key, int dflt) {
        Object v = view.raw(key);
        return v instanceof Number && !(v instanceof Boolean) ? ((Number) v).intValue() : dflt;
    }

    private static double numberOr(HfConfig cfg, JsonView view, String key, double dflt) {
        Object v = view.raw(key);
        return v instanceof Number && !(v instanceof Boolean) ? ((Number) v).doubleValue() : dflt;
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
