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
import java.util.Arrays;
import java.util.List;

/**
 * The SigLIP / CLIP vision towers with their own projectors through the Hugging Face loader, ports of the llama.cpp
 * converter classes {@code Gemma3VisionModel} (LC/conversion/gemma.py:262-310), {@code SmolVLMModel}
 * (LC/conversion/smolvlm.py), {@code InternVisionModel} (LC/conversion/internvl.py), {@code Llama4VisionModel}
 * (LC/conversion/llama4.py) and {@code KimiVLModel} (LC/conversion/kimivl.py:15-68): each family's name filter, its
 * projector keys and scale factor, the Gemma 3 norm-plus-one correction, the InternVL fused Q/K/V split, the Llama 4
 * single-layer projector and the Kimi-VL position table reshape. Every method is a public static hook reference.
 */
public final class HfTowerVisionHooks {

    /** The projector of Gemma 3 ({@code VisionProjectorType.GEMMA3}). */
    public static final String GEMMA3 = "gemma3";
    /** The projector of Idefics3 / SmolVLM ({@code VisionProjectorType.IDEFICS3}). */
    public static final String IDEFICS3 = "idefics3";
    /** The projector of InternVL ({@code VisionProjectorType.INTERNVL}). */
    public static final String INTERNVL = "internvl";
    /** The projector of Llama 4 ({@code VisionProjectorType.LLAMA4}). */
    public static final String LLAMA4 = "llama4";
    /** The projector of Kimi-VL ({@code VisionProjectorType.KIMIVL}). */
    public static final String KIMIVL = "kimivl";
    /** The default RMS epsilon of the SigLIP tower (gemma.py:269). */
    private static final double GEMMA3_EPS = 1e-6;
    /** The default layer-norm epsilon of the SmolVLM and Kimi-VL towers (smolvlm.py:32, kimivl.py:41). */
    private static final double DEFAULT_EPS = 1e-5;
    /** The compatibility image size of the Kimi-VL tower, 64 patches of 14 pixels per side (kimivl.py:24). */
    private static final int KIMIVL_IMAGE_SIZE = 64 * 14;

    private HfTowerVisionHooks() {
    }

    // ------------------------------------------------------------------ Gemma 3

    /** {@code vision_model.head.} drops, the tower and projector prefixes stay, {@code _weight} renames (gemma.py:287-299). */
    public static final HfNameFilter GEMMA3_FILTER = (name, base) -> {
        if (name.contains("vision_model.head.")) {
            return null;
        }
        if (!name.startsWith("multi_modal_projector.") && !name.startsWith("vision_tower.")
                && !name.startsWith("multimodal_projector.") && !name.startsWith("vision_model.")) {
            return null;
        }
        return base.apply(name.replace("_weight", ".weight"));
    };

    /**
     * The base vision keys, the {@code gemma3} projector, its epsilon and GELU, and the patch-merge scale factor
     * from the processor's {@code image_seq_length} (gemma.py:264-279).
     */
    public static void gemma3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, GEMMA3);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(GEMMA3_EPS));
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        long seq = seqLength(cfg);
        int nPerSide = (int) Math.sqrt(seq);
        int factor = nPerSide > 0 ? (HfMmprojHooks.visionInt(cfg, "image_size")
                / HfMmprojHooks.visionInt(cfg, "patch_size")) / nPerSide : 0;
        if (factor > 0 && factor != 4) {
            kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, factor);
        }
    }

    /** The processor's {@code image_seq_length}, 256 without one (gemma.py:272). */
    private static long seqLength(HfConfig cfg) throws IOException {
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        Object v = pre.raw("image_seq_length");
        return v instanceof Number ? ((Number) v).longValue() : 256;
    }

    /**
     * The Gemma 3 tower and pool projector (gemma.py:302-310): the projector norm restored from its stored-minus-one
     * form ({@code Gemma3RMSNorm}), everything else under its mapped name.
     */
    public static void gemma3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (hfName.contains("soft_emb_norm.weight")) {
            out.add(ctx.names().mapOrThrow(hfName), HfHookSupport.normPlusOne(src));
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The projector linears keep F16, the tower embeddings and position tables F32, the patch kernels the base rule
     * (gemma.py:280-286). */
    public static final HfForceType GEMMA3_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains("input_projection")) {
            return GGMLType.F16;
        }
        return HfMmprojHooks.VISION_EMBEDDINGS_F32.force(ggufName, nDims, source, outType);
    };

    // ------------------------------------------------------------------ SmolVLM / Idefics3

    /** The tower and connector names ({@code vision_tower} / {@code vision_model} / {@code model.connector}). */
    public static final HfNameFilter SMOLVLM_FILTER = (name, base) ->
            name.contains("vision_tower") || name.contains("vision_model") || name.contains("model.connector")
                    ? base.apply(name) : null;

    /**
     * The base vision keys with the SmolVLM2 config defaults (smolvlm.py:22-26), the {@code idefics3} projector,
     * its epsilon and GELU, the pixel-shuffle scale factor of the root {@code scale_factor} and the preprocessor's
     * longest edge (smolvlm.py:28-42).
     */
    public static void smolvlmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        // SmolVLM2 omits tower geometry the transformers defaults fill in (smolvlm.py:19-26)
        boolean smolvlm2 = "smolvlm_vision".equals(HfMmprojHooks.visionStr(cfg, "model_type"));
        int hidden = smolvlm2 ? HfMmprojHooks.visionIntOpt(cfg, "hidden_size").orElse(1152)
                : HfMmprojHooks.visionInt(cfg, "hidden_size");
        int heads = smolvlm2 ? HfMmprojHooks.visionIntOpt(cfg, "num_attention_heads").orElse(16)
                : HfMmprojHooks.visionInt(cfg, "num_attention_heads");
        int ffn = smolvlm2 ? HfMmprojHooks.visionIntOpt(cfg, "intermediate_size").orElse(3072)
                : HfMmprojHooks.visionInt(cfg, "intermediate_size");
        int imageSize = HfMmprojHooks.visionInt(cfg, "image_size");
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, HfMmprojHooks.visionInt(cfg, "patch_size"), hidden, ffn, heads,
                HfMmprojHooks.visionBlocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, IDEFICS3);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(DEFAULT_EPS));
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, cfg.raw().has("scale_factor")
                ? cfg.raw().i32("scale_factor") : 2);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, longestEdge(ctx, cfg, imageSize));
    }

    /** The preprocessor's {@code size.longest_edge}, the tower's image size without one (smolvlm.py:39). */
    private static int longestEdge(HfLoadContext ctx, HfConfig cfg, int imageSize) throws IOException {
        JsonView size = HfMmprojHooks.preprocessor(cfg).optObj("size");
        Object v = size.raw("longest_edge");
        return v instanceof Number ? ((Number) v).intValue() : imageSize;
    }

    /** The mapped names (smolvlm.py:48-60). */
    public static void smolvlmTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The tower embeddings keep F32, the patch kernels the base rule (smolvlm.py:43-46). */
    public static final HfForceType SMOLVLM_FORCE = HfMmprojHooks.VISION_EMBEDDINGS_F32;

    // ------------------------------------------------------------------ InternVL

    /** The tower and projector prefixes, the Intern-S1 renames and the scale-suffix repair (internvl.py:57-80). */
    public static final HfNameFilter INTERNVL_FILTER = (name, base) -> {
        if (!name.startsWith("vision_model") && !name.startsWith("mlp") && !name.startsWith("model.vision_tower")
                && !name.startsWith("model.multi_modal_projector")) {
            return null;
        }
        name = renameInternS1(name);
        if (name.startsWith("vision_model")) {
            name = "vision_tower." + name;
        }
        if ((name.contains(".ls") || name.contains(".lambda_") || name.contains("position_embedding"))
                && !name.endsWith(".weight")) {
            name += ".weight";
        }
        return base.apply(name);
    };

    /** The Intern-S1 projector names of the older layout (internvl.py:63-70). */
    private static String renameInternS1(String name) {
        int at = name.indexOf("model.multi_modal_projector.");
        if (at < 0) {
            return name;
        }
        String tail = name.substring(at + "model.multi_modal_projector.".length());
        String renamed;
        switch (tail) {
            case "layer_norm.bias": renamed = "mlp1.0.bias"; break;
            case "layer_norm.weight": renamed = "mlp1.0.weight"; break;
            case "linear_1.bias": renamed = "mlp1.1.bias"; break;
            case "linear_1.weight": renamed = "mlp1.1.weight"; break;
            case "linear_2.bias": renamed = "mlp1.3.bias"; break;
            case "linear_2.weight": renamed = "mlp1.3.weight"; break;
            default: renamed = null;
        }
        return renamed == null ? name : name.substring(0, at) + renamed;
    }

    /**
     * The base vision keys with the list forms of {@code image_size} / {@code patch_size} narrowed to their first
     * entry (internvl.py:33-36), the {@code internvl} projector with its activation, epsilon, pixel-shuffle scale
     * factor of the root {@code downsample_ratio} and the tile budget (internvl.py:37-52).
     */
    public static void internvlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, firstInt(cfg, "image_size"), firstInt(cfg, "patch_size"));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, INTERNVL);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64(cfg, "layer_norm_eps"));
        activation(ctx, cfg, kv);
        JsonView root = cfg.raw();
        if (!root.has("downsample_ratio")) {
            throw new GgufFormatException(cfg.source() + ": 'downsample_ratio' of the checkpoint is missing;"
                    + " clip.vision.projector.scale_factor follows it (internvl.py:45-48)");
        }
        double ratio = root.f64("downsample_ratio");
        if (!(ratio > 0.0)) {
            throw new GgufFormatException(cfg.source() + ": 'downsample_ratio' is " + ratio
                    + "; the pixel-shuffle scale factor follows its reciprocal");
        }
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, (int) (1.0 / ratio));
        int minTiles = cfg.raw().has("min_dynamic_patch") ? cfg.raw().i32("min_dynamic_patch") : 0;
        int maxTiles = cfg.raw().has("max_dynamic_patch") ? cfg.raw().i32("max_dynamic_patch") : 0;
        if (minTiles > 0) {
            kv.set(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, minTiles);
        }
        if (maxTiles > 0) {
            kv.set(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, maxTiles);
        }
    }

    /** The first element of the number list under {@code key}, an integer itself taken directly (internvl.py:33-36). */
    private static int firstInt(HfConfig cfg, String key) {
        Object v = HfMmprojHooks.visionFind(cfg, key).orElse(null);
        if (v instanceof List && !((List<?>) v).isEmpty()) {
            v = ((List<?>) v).get(0);
        }
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": 'vision_config." + key + "' is "
                    + (v == null ? "nothing" : v.getClass().getSimpleName()) + ", not a size");
        }
        return ((Number) v).intValue();
    }

    /** {@code use_silu} for {@code silu}, {@code use_gelu} for {@code gelu}; anything else is refused by name. */
    private static void activation(HfLoadContext ctx, HfConfig cfg, HfMetadataBuilder kv) {
        String act = HfMmprojHooks.visionStr(cfg, "hidden_act");
        switch (act) {
            case "silu":
                kv.set(GgufKeys.CLIP_USE_SILU, true);
                break;
            case "gelu":
                kv.set(GgufKeys.CLIP_USE_GELU, true);
                break;
            default:
                throw new GgufFormatException(cfg.source() + ": hidden_act '" + act
                        + "' is unsupported: the InternViT tower runs silu or gelu (internvl.py:41-45)");
        }
    }

    /** The fused Q/K/V split (internvl.py:83-96), everything else under its mapped name. */
    public static void internvlTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (HfMmprojHooks.splitFused(ctx, hfName, src, out, "attn.qkv", "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj")) {
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The position tables keep F32, the patch kernels the base rule (internvl.py:53-56). */
    public static final HfForceType INTERNVL_FORCE = HfMmprojHooks.POSITION_EMBD_F32;

    // ------------------------------------------------------------------ Llama 4

    /** The projector and tower names ({@code multi_modal_projector} / {@code vision_model}), the suffix repair. */
    public static final HfNameFilter LLAMA4_FILTER = (name, base) -> {
        if (!name.contains("multi_modal_projector") && !name.contains("vision_model")) {
            return null;
        }
        if (name.contains("positional_embedding_vlm") && !name.contains(".weight")) {
            name += ".weight";
        }
        return base.apply(name);
    };

    /**
     * The base vision keys, the {@code llama4} projector, the tower's {@code norm_eps}, the pixel-shuffle scale
     * factor of {@code pixel_shuffle_ratio} and GELU (llama4.py:11-19).
     */
    public static void llama4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, LLAMA4);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, HfMmprojHooks.visionF64(cfg, "norm_eps"));
        double ratio = HfMmprojHooks.visionF64(cfg, "pixel_shuffle_ratio");
        if (!(ratio > 0.0)) {
            throw new GgufFormatException(cfg.source() + ": 'pixel_shuffle_ratio' is " + ratio
                    + "; the pixel-shuffle scale factor follows its reciprocal");
        }
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, (int) (1.0 / ratio));
        if (!"gelu".equals(HfMmprojHooks.visionStr(cfg, "hidden_act"))) {
            throw new GgufFormatException(cfg.source() + ": hidden_act '" + HfMmprojHooks.visionStr(cfg, "hidden_act")
                    + "' is unsupported: the Llama 4 tower runs gelu (llama4.py:17)");
        }
        kv.set(GgufKeys.CLIP_USE_GELU, true);
    }

    /**
     * The tower, with the single fully connected layer of the projector ({@code multi_modal_projector.linear_1}) as
     * {@code mm.model.fc} despite its numbered name (llama4.py:24-27).
     */
    public static void llama4Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.contains("multi_modal_projector.linear_1")) {
            String suffix = hfName.endsWith(".bias") ? ".bias" : ".weight";
            out.add(ctx.names().ggufName("V_MMPROJ_FC", -1, suffix), src);
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The patch kernels of the base rule (llama4.py:11-19 writes none of its own). */
    public static final HfForceType LLAMA4_FORCE = HfMmprojHooks.PATCH_KERNELS;

    // ------------------------------------------------------------------ Kimi-VL

    /** The tower and projector names ({@code vision_tower} / {@code multi_modal_projector}) (kimivl.py:44-50). */
    public static final HfNameFilter KIMIVL_FILTER = (name, base) ->
            name.contains("vision_tower") || name.contains("multi_modal_projector") ? base.apply(name) : null;

    /**
     * The base vision keys with the compatibility image size of 64 patches per side (kimivl.py:24), the
     * {@code kimivl} projector, GELU, the pixel-shuffle scale factor 2 and the epsilon (kimivl.py:30-41).
     */
    public static void kimivlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, KIMIVL_IMAGE_SIZE);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, KIMIVL);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, 2);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(DEFAULT_EPS));
    }

    /**
     * The tower (kimivl.py:52-65): the position table {@code [h, w, c]} flattened to {@code [h * w, c]}, the fused
     * {@code wqkv} split along the output dimension, everything else under its mapped name.
     */
    public static void kimivlTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.contains("pos_emb.weight")) {
            long[] shape = src.shape();
            if (shape.length != 3) {
                throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source()
                        + " has the shape " + Arrays.toString(shape) + "; the Kimi-VL position table is [h, w, c]");
            }
            out.add(ctx.names().mapOrThrow(hfName),
                    ElementTransforms.reshape(src, shape[0] * shape[1], shape[2]));
            return;
        }
        if (HfMmprojHooks.splitFused(ctx, hfName, src, out, "wqkv", "wq", "wk", "wv")) {
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The patch kernels of the base rule (kimivl.py:30-41 forces none of its own). */
    public static final HfForceType KIMIVL_FORCE = HfMmprojHooks.PATCH_KERNELS;
}
