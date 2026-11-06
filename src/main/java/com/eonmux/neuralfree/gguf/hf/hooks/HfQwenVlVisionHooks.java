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
import java.util.Locale;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * The Qwen2-VL, Qwen2.5-VL and Qwen3-VL vision towers through the Hugging Face loader, ports of llama.cpp {@code
 * Qwen2VLVisionModel} (LC/conversion/qwenvl.py:45-141) and {@code Qwen3VLVisionModel} (LC/conversion/qwen3vl.py:17-146)
 * with their Sarashina 2 and GLM-4V subclasses: the tower kept from the {@code visual.} names, the fused Q/K/V
 * projections split, the two-frame patch kernel split into one Conv2d kernel per frame, the Qwen2-VL merger under
 * {@code mm.0} / {@code mm.2}, the Qwen3-VL deepstack taps at their absolute block indexes, and the window-attention
 * pattern of Qwen2.5-VL. Every method is a public static hook reference.
 */
public final class HfQwenVlVisionHooks {

    /** The projector of Qwen2-VL ({@code VisionProjectorType.QWEN2VL}). */
    public static final String QWEN2VL = "qwen2vl_merger";
    /** The projector of Qwen2.5-VL ({@code VisionProjectorType.QWEN25VL}). */
    public static final String QWEN25VL = "qwen2.5vl_merger";
    /** The projector of Qwen3-VL ({@code VisionProjectorType.QWEN3VL}). */
    public static final String QWEN3VL = "qwen3vl_merger";
    /** The projector of GLM-4V ({@code VisionProjectorType.GLM4V}). */
    public static final String GLM4V = "glm4v";
    /** The projector of GLM-5.3-Flash ({@code VisionProjectorType.GLM5V}). */
    public static final String GLM5V = "glm5v";
    /** The default image size of the Qwen2-VL tower (qwenvl.py:53). */
    private static final int QWEN2VL_IMAGE_SIZE = 560;
    /** The default RMS epsilon of the Qwen towers. */
    private static final double DEFAULT_EPS = 1e-6;
    /** The only temporal patch depth the two-frame split supports (qwenvl.py:118, qwen3vl.py:118). */
    private static final int TEMPORAL_PATCH_SIZE = 2;

    private HfQwenVlVisionHooks() {
    }

    // ------------------------------------------------------------------ filters

    /** Only the {@code visual.} names belong to the projector file (qwenvl.py:101-106). */
    public static final HfNameFilter QWEN2VL_FILTER = (name, base) ->
            name.startsWith("visual.") ? base.apply(name) : null;

    /** {@code lm_head.} and {@code mtp.} drop, {@code model.visual.} renamed to {@code visual.} (qwen3vl.py:74-90). */
    public static final HfNameFilter QWEN3VL_FILTER = (name, base) -> {
        if (name.startsWith("lm_head.") || name.startsWith("mtp.")) {
            return null;
        }
        if (name.startsWith("model.visual.")) {
            name = name.substring("model.".length());
        }
        return name.startsWith("visual.") ? base.apply(name) : null;
    };

    // ------------------------------------------------------------------ forced storage types

    /** The learned position tables keep F32, the patch kernels follow the base rule (qwenvl.py:107-110). */
    public static final HfForceType FORCE = HfMmprojHooks.POSITION_EMBD_F32;

    // ------------------------------------------------------------------ Qwen2-VL / Qwen2.5-VL

    /**
     * The base vision keys and the family projector (qwenvl.py:55-76): {@code qwen2vl_merger} for a {@code
     * qwen2_vl} checkpoint, {@code qwen2.5vl_merger} with {@code clip.use_silu} and {@code clip.vision.n_wa_pattern}
     * from the full-attention block indexes for a {@code qwen2_5_vl} one; the root {@code rms_norm_eps} as the
     * vision layer-norm epsilon. A Sarashina 2 vision checkpoint converts as {@code qwen2_vl} whatever its root
     * {@code model_type} says (sarashina2.py:27).
     */
    public static void qwen2vlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, qwen2vlImageSize(cfg), HfMmprojHooks.visionInt(cfg, "patch_size"),
                qwen2vlHidden(cfg), qwen2vlFfn(cfg), HfMmprojHooks.visionInt(cfg, "num_heads"),
                HfMmprojHooks.visionBlocks(cfg));
        String modelType = rootStr(cfg, "model_type");
        boolean sarashina = "Sarashina2VisionForCausalLM".equals(cfg.architecture());
        if (sarashina || "qwen2_vl".equals(modelType)) {
            kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, QWEN2VL);
        } else if ("qwen2_5_vl".equals(modelType)) {
            kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, QWEN25VL);
            kv.set(GgufKeys.CLIP_USE_SILU, true);
            kv.set(GgufKeys.CLIP_VISION_N_WA_PATTERN, windowPattern(cfg));
        } else {
            throw new GgufFormatException(cfg.source() + ": model_type '" + modelType
                    + "' is no Qwen-VL model type (qwen2_vl, qwen2_5_vl) (qwenvl.py:76)");
        }
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                cfg.raw().has("rms_norm_eps") ? cfg.raw().f64("rms_norm_eps") : DEFAULT_EPS);
    }

    /** {@code find_vparam(['image_size'])} with the default the class fills (qwenvl.py:53). */
    private static int qwen2vlImageSize(HfConfig cfg) {
        return HfMmprojHooks.visionIntOpt(cfg, "image_size").orElse(QWEN2VL_IMAGE_SIZE);
    }

    /** The tower width: {@code embed_dim} of a Qwen2-VL config, else the base aliases (qwenvl.py:57-61). */
    private static int qwen2vlHidden(HfConfig cfg) {
        return HfMmprojHooks.visionFind(cfg, "embed_dim").isPresent()
                ? HfMmprojHooks.visionInt(cfg, "embed_dim")
                : HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS);
    }

    /** The merger width: the {@code hidden_size} of a config with {@code embed_dim}, else the FFN aliases
     * (qwenvl.py:58-61 renames the keys only for the Qwen2-VL layout). */
    private static int qwen2vlFfn(HfConfig cfg) {
        return HfMmprojHooks.visionFind(cfg, "embed_dim").isPresent()
                ? HfMmprojHooks.visionInt(cfg, "hidden_size")
                : HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS);
    }

    /**
     * The tower image size: {@code image_size} when present, else
     * {@code sqrt(num_position_embeddings) * patch_size} (qwen3vl.py:24-31).
     */
    public static int imageSize(HfConfig cfg) {
        int patchSize = HfMmprojHooks.visionIntOpt(cfg, "patch_size").orElse(16);
        return HfMmprojHooks.visionIntOpt(cfg, "image_size").orElseGet(
                () -> (int) (Math.sqrt(HfMmprojHooks.visionIntOpt(cfg, "num_position_embeddings").orElse(2304))
                        * patchSize));
    }

    /** The string {@code key} of the root configuration, required. */
    static String rootStr(HfConfig cfg, String key) {
        Object v = cfg.raw().raw(key);
        if (!(v instanceof String) || ((String) v).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a non-empty string");
        }
        return (String) v;
    }

    /**
     * The window-attention pattern of {@code fullatt_block_indexes}: one more than the first index, with every
     * later index exactly one period after the previous (qwenvl.py:68-75).
     */
    private static int windowPattern(HfConfig cfg) {
        Object v = HfMmprojHooks.visionFind(cfg, "fullatt_block_indexes").orElse(null);
        if (!(v instanceof List) || ((List<?>) v).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": 'fullatt_block_indexes' is required for qwen2_5_vl"
                    + " (qwenvl.py:69)");
        }
        long[] indexes = new long[((List<?>) v).size()];
        for (int i = 0; i < indexes.length; i++) {
            Object e = ((List<?>) v).get(i);
            if (!(e instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": 'fullatt_block_indexes[" + i + "]' is "
                        + (e == null ? "null" : e.getClass().getSimpleName()) + ", not a block index");
            }
            indexes[i] = ((Number) e).longValue();
        }
        long period = indexes[0] + 1;
        for (int i = 1; i < indexes.length; i++) {
            if (indexes[i] - indexes[i - 1] != period) {
                throw new GgufFormatException(cfg.source() + ": Invalid fullatt_block_indexes: "
                        + Arrays.toString(indexes) + " (qwenvl.py:73)");
            }
        }
        return Math.toIntExact(period);
    }

    // ------------------------------------------------------------------ Qwen3-VL

    /**
     * The base vision keys and the {@code qwen3vl_merger} projector (qwen3vl.py:47-71): the image size computed
     * from {@code num_position_embeddings} when the configuration has none, the tower heads of {@code num_heads},
     * the text model's {@code rms_norm_eps} as the vision epsilon, GELU, the spatial merge and the deepstack block
     * flags.
     */
    public static void qwen3vlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        qwen3vlMeta(ctx, kv, plan, GgufKeys.CLIP_PROJECTOR_TYPE);
    }

    /**
     * {@link #qwen3vlMeta} writing the projector type under {@code projectorKey}: the common
     * {@code clip.projector_type} for a vision-only checkpoint, {@code clip.vision.projector_type} for a projector
     * file that also carries an audio tower.
     */
    public static void qwen3vlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, String projectorKey)
            throws IOException {
        HfConfig cfg = ctx.config();
        int blocks = HfMmprojHooks.visionInt(cfg, "depth");
        HfMmprojHooks.visionMeta(ctx, kv, imageSize(cfg));
        kv.set(projectorKey, QWEN3VL);
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        OptionalInt merge = HfMmprojHooks.visionIntOpt(cfg, "spatial_merge_size");
        if (merge.isPresent()) {
            kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, merge.getAsInt());
        }
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                cfg.text().has("rms_norm_eps") ? cfg.text().f64("rms_norm_eps") : DEFAULT_EPS);
        kv.set(GgufKeys.CLIP_VISION_IS_DEEPSTACK_LAYERS, deepstackFlags(cfg, blocks));
    }

    /**
     * The Qwen2.5-Omni vision half ({@code Qwen25OmniModel}, qwenvl.py:167-208): the Qwen2.5-VL tower keys and the
     * {@code qwen2.5o} mixed-modality projector, whose encoders resolve per modality in the loader.
     */
    public static void qwen25OmniVisionMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, qwen2vlImageSize(cfg), HfMmprojHooks.visionInt(cfg, "patch_size"),
                qwen2vlHidden(cfg), qwen2vlFfn(cfg), HfMmprojHooks.visionInt(cfg, "num_heads"),
                HfMmprojHooks.visionBlocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, "qwen2.5o");
        kv.set(GgufKeys.CLIP_USE_SILU, true);
        kv.set(GgufKeys.CLIP_VISION_N_WA_PATTERN, windowPattern(cfg));
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                cfg.raw().has("rms_norm_eps") ? cfg.raw().f64("rms_norm_eps") : DEFAULT_EPS);
    }

    /** One flag per block, true at the {@code deepstack_visual_indexes} taps (qwen3vl.py:65-68, :69). */
    private static List<Boolean> deepstackFlags(HfConfig cfg, int blocks) {
        Object v = HfMmprojHooks.visionFind(cfg, "deepstack_visual_indexes").orElse(null);
        boolean[] taps = new boolean[blocks];
        if (v instanceof List) {
            for (Object e : (List<?>) v) {
                if (!(e instanceof Number)) {
                    throw new GgufFormatException(cfg.source() + ": 'deepstack_visual_indexes' must hold block"
                            + " indexes, got " + (e == null ? "null" : e.getClass().getSimpleName()));
                }
                int idx = ((Number) e).intValue();
                if (idx < 0 || idx >= blocks) {
                    throw new GgufFormatException(cfg.source() + ": 'deepstack_visual_indexes' entry " + idx
                            + " lies outside the " + blocks + " vision blocks");
                }
                taps[idx] = true;
            }
        }
        List<Boolean> out = new ArrayList<>(blocks);
        for (boolean tap : taps) {
            out.add(tap);
        }
        return out;
    }

    // ------------------------------------------------------------------ GLM-4V / GLM-5.3-Flash

    /**
     * The base vision keys and the {@code glm4v} projector (qwen3vl.py:240-257): the tower activation as
     * {@code clip.use_gelu} / {@code clip.use_silu} and the tower's {@code rms_norm_eps} as the vision layer-norm
     * epsilon; unlike the Qwen3-VL keys, no deepstack or window keys are written.
     */
    public static void glm4vMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, imageSize(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, GLM4V);
        String act = HfMmprojHooks.visionFind(cfg, "hidden_act").orElse("").toString().toLowerCase(Locale.ROOT);
        if ("gelu".equals(act)) {
            kv.set(GgufKeys.CLIP_USE_GELU, true);
        } else if ("silu".equals(act)) {
            kv.set(GgufKeys.CLIP_USE_SILU, true);
        }
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "rms_norm_eps").orElse(1e-5));
    }

    /**
     * {@link #glm4vMeta} with the {@code glm5v} projector, the spatial merge, the SwiGLU clamp of the tower's
     * {@code swiglu_limit} and the image token budget of the processor (qwen3vl.py:262-280).
     */
    public static void glm5NextMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        glm4vMeta(ctx, kv, plan);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, GLM5V);
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE,
                HfMmprojHooks.visionIntOpt(cfg, "spatial_merge_size").orElse(2));
        Optional<Double> clamp = HfMmprojHooks.visionF64Opt(cfg, "swiglu_limit");
        if (clamp.isPresent()) {
            kv.set(GgufKeys.CLIP_VISION_SWIGLU_CLAMP, clamp.get());
        }
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        int patch = intOf(pre, "patch_size", 14);
        int merge = intOf(pre, "merge_size", 2);
        long pixelsPerToken = (long) patch * patch * merge * merge;
        writePixels(ctx, kv, pre, pixelsPerToken);
    }

    /** The integer member {@code key} of the preprocessor configuration, or {@code dflt}. */
    private static int intOf(JsonView pre, String key, int dflt) {
        Object v = pre.raw(key);
        return v instanceof Number ? ((Number) v).intValue() : dflt;
    }

    /** {@code clip.vision.image_min_pixels} / {@code max_pixels} from the preprocessor token budgets. */
    private static void writePixels(HfLoadContext ctx, HfMetadataBuilder kv, JsonView pre, long pixelsPerToken) {
        Object min = pre.raw("min_image_tokens");
        if (min instanceof Number) {
            kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, Math.multiplyExact(((Number) min).longValue(),
                    pixelsPerToken));
        }
        Object max = pre.raw("max_image_tokens");
        if (max instanceof Number) {
            kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, Math.multiplyExact(((Number) max).longValue(),
                    pixelsPerToken));
        }
    }

    /**
     * The GLM-4V tower (qwen3vl.py:259-262): the merger names map through the converter's name map, everything
     * else runs {@link #qwen3vlTensors}, whose two-frame patch split the GLM tower shares.
     */
    public static void glm4vTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("visual.merger.")) {
            out.add(ctx.names().mapOrThrow(hfName), src);
            return;
        }
        qwen3vlTensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The Qwen2-VL tower: the fused Q/K/V projections split (qwenvl.py:112-121), the two-frame patch kernel split
     * into {@code v.patch_embd.weight} and {@code v.patch_embd.weight.1} (qwenvl.py:122-128), everything else under
     * its mapped name.
     */
    public static void qwen2vlTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (HfMmprojHooks.splitFused(ctx, hfName, src, out, ".qkv.", ".q.", ".k.", ".v.")) {
            return;
        }
        if (hfName.endsWith("patch_embed.proj.weight")) {
            splitPatchFrames(ctx, hfName, src, out);
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /**
     * The Qwen3-VL tower (qwen3vl.py:97-145): the deepstack mergers at their absolute block indexes, the merger
     * MLP as {@code mm.0} / {@code mm.2} and its norm as {@code v.post_ln}, the two-frame patch kernel split with
     * its bias, everything else under its mapped name.
     */
    public static void qwen3vlTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("visual.deepstack_merger_list.")) {
            deepstack(ctx, hfName, src, out);
            return;
        }
        if (hfName.startsWith("visual.merger.")) {
            merger(ctx, hfName, src, out);
            return;
        }
        if (hfName.equals("visual.patch_embed.proj.weight")) {
            splitPatchFrames(ctx, hfName, src, out);
            return;
        }
        if (hfName.equals("visual.patch_embed.proj.bias")) {
            out.add(ctx.names().ggufName("V_ENC_EMBD_PATCH", -1, ".bias"), src);
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /**
     * The deepstack merger of list index {@code i} (qwen3vl.py:99-123): its norm and linear layers land under
     * {@code v.deepstack.<idx>.…}, where {@code idx} is the absolute block index {@code deepstack_visual_indexes[i]}.
     */
    private static void deepstack(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        String tail = hfName.substring("visual.deepstack_merger_list.".length());
        int dot = tail.indexOf('.');
        if (dot <= 0) {
            throw new GgufFormatException("tensor '" + hfName + "' is not a deepstack merger"
                    + " (visual.deepstack_merger_list.<i>.<part>.<suffix>)");
        }
        int i;
        try {
            i = Integer.parseInt(tail.substring(0, dot));
        } catch (NumberFormatException e) {
            throw new GgufFormatException("tensor '" + hfName + "' has deepstack list index '"
                    + tail.substring(0, dot) + "', not a number");
        }
        String rest = tail.substring(dot + 1);
        String tensor;
        if (rest.startsWith("norm.")) {
            tensor = "V_DS_NORM";
        } else if (rest.startsWith("linear_fc1.")) {
            tensor = "V_DS_FC1";
        } else if (rest.startsWith("linear_fc2.")) {
            tensor = "V_DS_FC2";
        } else {
            throw new GgufFormatException("Unexpected deepstack tensor: " + hfName + " (qwen3vl.py:120)");
        }
        int idx = deepstackIndexes(cfg)[i];
        int blocks = HfMmprojHooks.visionInt(cfg, "depth");
        if (idx < 0 || idx >= blocks) {
            throw new GgufFormatException(cfg.source() + ": deepstack index " + idx + " of '"
                    + hfName + "' lies outside the " + blocks + " vision blocks");
        }
        String suffix = rest.substring(rest.indexOf('.') + 1);
        out.add(ctx.names().ggufName(tensor, idx, "." + suffix), src);
    }

    /** The absolute block indexes of the deepstack taps, required. */
    private static int[] deepstackIndexes(HfConfig cfg) {
        Object v = HfMmprojHooks.visionFind(cfg, "deepstack_visual_indexes").orElse(null);
        if (!(v instanceof List) || ((List<?>) v).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": 'deepstack_visual_indexes' is required for the"
                    + " deepstack merger tensors (qwen3vl.py:107)");
        }
        int[] out = new int[((List<?>) v).size()];
        for (int k = 0; k < out.length; k++) {
            Object e = ((List<?>) v).get(k);
            if (!(e instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": 'deepstack_visual_indexes[" + k + "]' is "
                        + (e == null ? "null" : e.getClass().getSimpleName()) + ", not a block index");
            }
            out[k] = ((Number) e).intValue();
        }
        return out;
    }

    /**
     * The Qwen3-VL merger (qwen3vl.py:125-143): {@code linear_fc1} / {@code linear_fc2} as {@code mm.0} /
     * {@code mm.2} with their tail, {@code norm} as {@code v.post_ln}; anything else is refused by name.
     */
    private static void merger(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String suffix = hfName.substring("visual.merger.".length());
        if (suffix.startsWith("linear_fc1.") || suffix.startsWith("linear_fc2.")) {
            int fc = suffix.startsWith("linear_fc1.") ? 0 : 2;
            String tail = suffix.substring("linear_fcN.".length());
            out.add(ctx.names().ggufName("V_MMPROJ", fc, "." + tail), src);
            return;
        }
        if (suffix.startsWith("norm.")) {
            out.add(ctx.names().ggufName("V_POST_NORM", -1, "." + suffix.substring("norm.".length())), src);
            return;
        }
        throw new GgufFormatException("Unexpected merger tensor: " + hfName + " (qwen3vl.py:142)");
    }

    /**
     * The two-frame patch kernel split (qwenvl.py:122-128, qwen3vl.py:115-121): the Conv3d weight of shape
     * {@code [c1, c2, 2, kh, kw]} becomes the Conv2d kernels {@code v.patch_embd.weight} and
     * {@code v.patch_embd.weight.1}.
     */
    static void splitPatchFrames(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        long[] shape = src.shape();
        if (shape.length != 5 || shape[2] != TEMPORAL_PATCH_SIZE) {
            throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the shape "
                    + Arrays.toString(shape) + "; the patch kernel is a Conv3d weight of temporal depth "
                    + TEMPORAL_PATCH_SIZE + " (qwenvl.py:118)");
        }
        String base = ctx.names().ggufName("V_ENC_EMBD_PATCH", -1);
        out.add(base + ".weight", ElementTransforms.squeezeConv(RowTransforms.sliceDim(src, 2, 0, 1)));
        out.add(base + ".weight.1", ElementTransforms.squeezeConv(RowTransforms.sliceDim(src, 2, 1, 1)));
    }
}
