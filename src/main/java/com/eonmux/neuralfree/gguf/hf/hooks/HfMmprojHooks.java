package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * The {@code MmprojModel} base class (LC/conversion/base.py:3056-3262) as the hooks the projector entries share: the
 * vision and audio configuration lookups ({@code find_vparam} / {@code find_aparam}, first alias present, a present
 * null ends the search), the block count of the encoder, the merged {@code preprocessor_config.json} and
 * {@code processor_config.json}, the base {@code set_gguf_parameters} ({@code clip.has_vision_encoder} /
 * {@code clip.has_audio_encoder}, {@code clip.<modality>.projection_dim} from the text width, the tower geometry and
 * the normalization colors), the fused projection split, the empty vocabulary of a projector file, and the forced
 * storage types of the patch kernels and position tables. The per-family hooks call {@link #visionMeta} or
 * {@link #audioMeta} first, then write their own keys. Every method is a public static hook reference.
 */
public final class HfMmprojHooks {

    /** The block-count keys of {@code MmprojModel} (base.py:3059). */
    public static final List<String> BLOCK_COUNT_KEYS = Collections.unmodifiableList(Arrays.asList(
            "n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth", "layers", "encoder_layers",
            "vt_num_hidden_layers"));

    /** The width aliases of the base vision hyperparameters (base.py:3203). */
    public static final String[] EMBEDDING_KEYS = {"hidden_size", "width", "vt_hidden_size"};
    /** The FFN aliases of the base vision hyperparameters (base.py:3204). */
    public static final String[] FFN_KEYS = {"intermediate_size", "vt_intermediate_size"};
    /** The head aliases of the base vision hyperparameters (base.py:3206). */
    public static final String[] HEAD_KEYS = {"num_attention_heads", "num_heads", "heads", "vt_num_attention_heads"};

    private HfMmprojHooks() {
    }

    // ------------------------------------------------------------------ configuration lookups

    /**
     * The first alias present in {@code view}, even when its value is JSON null ({@code find_hparam}, base.py:1170,
     * and {@code _find_param}, base.py:3234-3241).
     */
    public static Optional<Object> find(JsonView view, String... aliases) {
        Objects.requireNonNull(view, "view");
        if (aliases.length == 0) {
            throw new IllegalArgumentException("a hyperparameter lookup needs at least one alias");
        }
        for (String alias : aliases) {
            if (view.has(alias)) {
                return Optional.ofNullable(view.raw(alias));
            }
        }
        return Optional.empty();
    }

    private static Optional<Object> find(HfConfig cfg, Optional<JsonView> view, String name, String... aliases) {
        if (!view.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no '" + name + "'; this projector"
                    + " converts the " + name.substring(0, name.indexOf('_')) + " tower");
        }
        return find(view.get(), aliases);
    }

    /** {@link #find} over the vision config. */
    public static Optional<Object> visionFind(HfConfig cfg, String... aliases) {
        return find(cfg, cfg.vision(), "vision_config", aliases);
    }

    /**
     * The audio encoder configuration of {@code cfg}: the root {@code whisper_config} when present, else
     * {@code audio_config} (base.py:3146-3148, {@code MmprojModel.get_audio_config}).
     */
    public static Optional<JsonView> audioConfig(HfConfig cfg) {
        JsonView root = cfg.raw();
        return root.has("whisper_config") ? Optional.of(root.obj("whisper_config")) : cfg.audio();
    }

    /** {@link #find} over the audio config. */
    public static Optional<Object> audioFind(HfConfig cfg, String... aliases) {
        return find(cfg, audioConfig(cfg), "audio_config", aliases);
    }

    /** The first present integer of the vision config, required. */
    public static int visionInt(HfConfig cfg, String... aliases) {
        return requireInt(cfg, "vision_config", visionFind(cfg, aliases), aliases);
    }

    /** The first present integer of the vision config, or empty. */
    public static OptionalInt visionIntOpt(HfConfig cfg, String... aliases) {
        return optInt(cfg, "vision_config", visionFind(cfg, aliases), aliases);
    }

    /** The first present number of the vision config, required. */
    public static double visionF64(HfConfig cfg, String... aliases) {
        return requireF64(cfg, "vision_config", visionFind(cfg, aliases), aliases);
    }

    /** The first present number of the vision config, or empty. */
    public static Optional<Double> visionF64Opt(HfConfig cfg, String... aliases) {
        Optional<Object> v = visionFind(cfg, aliases);
        return v.isPresent() ? Optional.of(requireF64(cfg, "vision_config", v, aliases)) : Optional.empty();
    }

    /** The first present string of the vision config, required. */
    public static String visionStr(HfConfig cfg, String... aliases) {
        Object v = visionFind(cfg, aliases).orElse(null);
        if (!(v instanceof String) || ((String) v).isEmpty()) {
            throw missing(cfg, "vision_config", aliases, v);
        }
        return (String) v;
    }

    /** The first present integer of the audio config, required. */
    public static int audioInt(HfConfig cfg, String... aliases) {
        return requireInt(cfg, "audio_config", audioFind(cfg, aliases), aliases);
    }

    /** The first present integer of {@code view} (named {@code name} in messages), required. */
    public static int intOf(HfConfig cfg, JsonView view, String name, String... aliases) {
        return requireInt(cfg, name, find(view, aliases), aliases);
    }

    /** The first present block-count key of {@code view} (named {@code name} in messages), required. */
    public static int blocksOf(HfConfig cfg, JsonView view, String name) {
        String[] keys = BLOCK_COUNT_KEYS.toArray(new String[0]);
        return requireInt(cfg, name, find(view, keys), keys);
    }

    /** The first present integer of the audio config, or empty. */
    public static OptionalInt audioIntOpt(HfConfig cfg, String... aliases) {
        return optInt(cfg, "audio_config", audioFind(cfg, aliases), aliases);
    }

    /**
     * The text model's width ({@code n_embd_text}, base.py:3065-3081): the flat configuration's {@code hidden_size},
     * else {@code n_embd}, positive.
     */
    public static int textWidth(HfConfig cfg) {
        Object v = cfg.flat().containsKey("hidden_size") ? cfg.flat().get("hidden_size")
                : cfg.flat().get("n_embd");
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": the text width 'hidden_size' of the checkpoint is"
                    + " missing; an mmproj file projects into the text model's rows");
        }
        int width = ((Number) v).intValue();
        if (width <= 0) {
            throw new GgufFormatException(cfg.source() + ": the text width 'hidden_size' is " + width);
        }
        return width;
    }

    /** The vision tower's block count ({@code find_vparam(n_block_keys)}, required). */
    public static int visionBlocks(HfConfig cfg) {
        return blocks(cfg, cfg.vision(), "vision_config");
    }

    /** The audio tower's block count ({@code find_aparam(n_block_keys)}, required). */
    public static int audioBlocks(HfConfig cfg) {
        return blocks(cfg, audioConfig(cfg), "audio_config");
    }

    private static int blocks(HfConfig cfg, Optional<JsonView> view, String name) {
        String[] keys = BLOCK_COUNT_KEYS.toArray(new String[0]);
        return requireInt(cfg, name, find(cfg, view, name, keys), keys);
    }

    private static int requireInt(HfConfig cfg, String view, Optional<Object> v, String[] aliases) {
        Object raw = v.orElse(null);
        if (!(isIntegral(raw))) {
            throw missing(cfg, view, aliases, raw);
        }
        return ((Number) raw).intValue();
    }

    private static double requireF64(HfConfig cfg, String view, Optional<Object> v, String[] aliases) {
        Object raw = v.orElse(null);
        if (!(raw instanceof Number)) {
            throw missing(cfg, view, aliases, raw);
        }
        return ((Number) raw).doubleValue();
    }

    private static OptionalInt optInt(HfConfig cfg, String view, Optional<Object> v, String[] aliases) {
        if (!v.isPresent()) {
            return OptionalInt.empty();
        }
        return OptionalInt.of(requireInt(cfg, view, v, aliases));
    }

    private static boolean isIntegral(Object raw) {
        if (!(raw instanceof Number)) {
            return false;
        }
        double d = ((Number) raw).doubleValue();
        return d == Math.rint(d);
    }

    private static GgufFormatException missing(HfConfig cfg, String view, String[] aliases, Object raw) {
        return new GgufFormatException(cfg.source() + ": '" + view + "' has none of " + Arrays.toString(aliases)
                + " with a usable value, got " + describe(raw));
    }

    /** The first present integer of {@code view} under {@code aliases}, or {@code dflt} when none is usable. */
    public static int intOr(HfConfig cfg, JsonView view, int dflt, String... aliases) {
        Object raw = find(view, aliases).orElse(null);
        return isIntegral(raw) ? ((Number) raw).intValue() : dflt;
    }

    /** The first present number of {@code view} under {@code aliases}, or {@code dflt} when none is usable. */
    public static double f64Or(HfConfig cfg, JsonView view, double dflt, String... aliases) {
        Object raw = find(view, aliases).orElse(null);
        return raw instanceof Number ? ((Number) raw).doubleValue() : dflt;
    }

    /** The first present boolean of {@code view} under {@code aliases}, or {@code dflt} when none is usable. */
    public static boolean boolOr(JsonView view, boolean dflt, String... aliases) {
        Object raw = find(view, aliases).orElse(null);
        return raw instanceof Boolean ? (Boolean) raw : dflt;
    }

    /**
     * The integer list under the first present alias of {@code view}, empty when none is present.
     *
     * @throws GgufFormatException when a present value is not a list of integers
     */
    public static List<Integer> intList(HfConfig cfg, JsonView view, String... aliases) {
        Object raw = find(view, aliases).orElse(null);
        if (raw == null) {
            return Collections.emptyList();
        }
        if (!(raw instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": '" + Arrays.toString(aliases) + "' is "
                    + describe(raw) + ", not a list of integers");
        }
        List<?> list = (List<?>) raw;
        List<Integer> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            if (!isIntegral(list.get(i))) {
                throw new GgufFormatException(cfg.source() + ": '" + Arrays.toString(aliases) + "[" + i + "]' is "
                        + describe(list.get(i)) + ", not an integer");
            }
            out.add(((Number) list.get(i)).intValue());
        }
        return out;
    }

    /** {@link #intList} as a primitive array. */
    public static int[] intArray(HfConfig cfg, JsonView view, String... aliases) {
        List<Integer> list = intList(cfg, view, aliases);
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    /** The object under {@code key} of {@code view}, required. */
    public static JsonView object(HfConfig cfg, JsonView view, String key) {
        Object raw = view.raw(key);
        if (!(raw instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' is " + describe(raw) + ", not an object");
        }
        return JsonView.of(asMap(raw), key);
    }

    private static String describe(Object raw) {
        return raw == null ? "nothing" : raw.getClass().getSimpleName() + " " + raw;
    }

    // ------------------------------------------------------------------ preprocessor configuration

    /**
     * The merged preprocessor configuration (base.py:3195-3212): {@code preprocessor_config.json} with its
     * {@code media_proc_cfg} hoisted, then {@code processor_config.json} with its {@code image_processor} hoisted
     * over it; either file may be missing, and the empty view refuses the first required key.
     */
    public static JsonView preprocessor(HfConfig cfg) throws IOException {
        Map<String, Object> merged = new LinkedHashMap<>();
        mergeHoisted(merged, HfBertVariantHooks.readJsonView(cfg.sideFile("preprocessor_config.json")),
                "media_proc_cfg");
        mergeHoisted(merged, HfBertVariantHooks.readJsonView(cfg.sideFile("processor_config.json")),
                "image_processor");
        return JsonView.of(merged, "the preprocessor configuration");
    }

    private static void mergeHoisted(Map<String, Object> merged, JsonView view, String hoist) {
        for (String key : view.keys()) {
            merged.put(key, view.raw(key));
        }
        Object nested = view.raw(hoist);
        if (nested instanceof Map) {
            for (Map.Entry<String, Object> e : asMap(nested).entrySet()) {
                merged.put(e.getKey(), e.getValue());
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object v) {
        return (Map<String, Object>) v;
    }

    /** The integer {@code key} of the merged preprocessor configuration, required. */
    public static int preInt(HfConfig cfg, String key) throws IOException {
        return intOf(cfg, preprocessor(cfg), "preprocessor_config", key);
    }

    /** The integer {@code key} of the merged preprocessor configuration, or {@code dflt}. */
    public static int preIntOr(HfConfig cfg, String key, int dflt) throws IOException {
        return intOr(cfg, preprocessor(cfg), dflt, key);
    }

    /** The number {@code key} of the merged preprocessor configuration, or {@code dflt}. */
    public static double preF64Or(HfConfig cfg, String key, double dflt) throws IOException {
        return f64Or(cfg, preprocessor(cfg), dflt, key);
    }

    /** The boolean {@code key} of the merged preprocessor configuration, or {@code dflt}. */
    public static boolean preBoolOr(HfConfig cfg, String key, boolean dflt) throws IOException {
        return boolOr(preprocessor(cfg), dflt, key);
    }

    /** The string {@code key} of the merged preprocessor configuration, or null. */
    public static String preStr(HfConfig cfg, String key) throws IOException {
        Object raw = preprocessor(cfg).raw(key);
        return raw instanceof String ? (String) raw : null;
    }

    // ------------------------------------------------------------------ metadata (base set_gguf_parameters)

    /**
     * The base vision keys for a tower of the given geometry (base.py:3186-3209): {@code clip.has_vision_encoder},
     * {@code clip.vision.projection_dim} from the text width, {@code clip.vision.image_size} / {@code patch_size},
     * {@code embedding_length} / {@code feed_forward_length} / {@code block_count} / {@code attention.head_count},
     * and the normalization colors of the preprocessor configuration.
     */
    public static void visionMeta(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize, int patchSize, int hidden,
                                  int ffn, int heads, int blocks) throws IOException {
        visionMetaConfig(ctx, kv, imageSize, patchSize, hidden, ffn, heads, blocks, "vision_config", null, null);
    }

    /**
     * {@link #visionMeta(HfLoadContext, HfMetadataBuilder, int, int, int, int, int, int)} with the normalization
     * colors a class whose preprocessor configuration omits them writes instead (Step 3-VL, SenseNova-U1 and
     * Nemotron Nano V2 VL fill in their dataset means and standard deviations, step3.py:36-37,
     * sensenova_u1.py:74-75, nemotron.py:156-159). A color present in the preprocessor configuration wins over the
     * given default.
     */
    public static void visionMeta(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize, int patchSize, int hidden,
                                  int ffn, int heads, int blocks, List<Float> defaultMean, List<Float> defaultStd)
            throws IOException {
        HfConfig cfg = ctx.config();
        visionKeys(ctx, kv, imageSize, patchSize, hidden, ffn, heads, blocks);
        JsonView pre = preprocessor(cfg);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MEAN, colorArray(cfg, pre, "image_mean", defaultMean));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_STD, colorArray(cfg, pre, "image_std", defaultStd));
    }

    /**
     * {@link #visionMeta(HfLoadContext, HfMetadataBuilder, int, int, int, int, int, int)} with fixed normalization
     * colors written instead of the preprocessor configuration's (a tower the converter always normalizes with its
     * own colors, or one whose preprocessor file has none). Both arrays are given, or neither.
     */
    public static void visionMeta(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize, int patchSize, int hidden,
                                  int ffn, int heads, int blocks, float[] mean, float[] std) throws IOException {
        visionMetaConfig(ctx, kv, imageSize, patchSize, hidden, ffn, heads, blocks, "vision_config", mean, std);
    }

    /**
     * {@link #visionMeta(HfLoadContext, HfMetadataBuilder, int, int, int, int, int, int)} naming the configuration
     * object the geometry came from, for a tower read from another member than {@code vision_config}, with the
     * normalization colors given (null reads the preprocessor configuration).
     */
    public static void visionMetaConfig(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize, int patchSize,
                                        int hidden, int ffn, int heads, int blocks, String viewName, float[] mean,
                                        float[] std) throws IOException {
        HfConfig cfg = ctx.config();
        visionKeys(ctx, kv, imageSize, patchSize, hidden, ffn, heads, blocks);
        if (mean != null && std != null) {
            kv.set(GgufKeys.CLIP_VISION_IMAGE_MEAN, floats(mean));
            kv.set(GgufKeys.CLIP_VISION_IMAGE_STD, floats(std));
            return;
        }
        JsonView pre = preprocessor(cfg);
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MEAN, colorArray(cfg, pre, "image_mean", null));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_STD, colorArray(cfg, pre, "image_std", null));
    }

    /** The geometry and encoder keys of the vision metadata, without the normalization colors. */
    private static void visionKeys(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize, int patchSize, int hidden,
                                   int ffn, int heads, int blocks) {
        kv.set(GgufKeys.CLIP_HAS_VISION_ENCODER, true);
        kv.set(GgufKeys.CLIP_VISION_PROJECTION_DIM, textWidth(ctx.config()));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_SIZE, imageSize);
        kv.set(GgufKeys.CLIP_VISION_PATCH_SIZE, patchSize);
        kv.set(GgufKeys.CLIP_VISION_EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.CLIP_VISION_FEED_FORWARD_LENGTH, ffn);
        kv.set(GgufKeys.CLIP_VISION_BLOCK_COUNT, blocks);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT, heads);
    }

    private static List<Float> floats(float[] values) {
        List<Float> out = new ArrayList<>(values.length);
        for (float v : values) {
            out.add(v);
        }
        return out;
    }

    /** The base vision keys with the converter's alias lookups of the vision config. */
    public static void visionMeta(HfLoadContext ctx, HfMetadataBuilder kv) throws IOException {
        HfConfig cfg = ctx.config();
        visionMeta(ctx, kv, visionInt(cfg, "image_size"), visionInt(cfg, "patch_size"),
                visionInt(cfg, EMBEDDING_KEYS), visionInt(cfg, FFN_KEYS), visionInt(cfg, HEAD_KEYS),
                visionBlocks(cfg));
    }

    /** The base vision keys with the converter's alias lookups, computing an absent image size as {@code imageSize}. */
    public static void visionMeta(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize) throws IOException {
        HfConfig cfg = ctx.config();
        visionMeta(ctx, kv, imageSize, visionInt(cfg, "patch_size"), visionInt(cfg, EMBEDDING_KEYS),
                visionInt(cfg, FFN_KEYS), visionInt(cfg, HEAD_KEYS), visionBlocks(cfg));
    }

    /**
     * The base vision keys with the converter's alias lookups for every value except the given image and patch sizes
     * (a checkpoint whose sizes are lists, or a class that fixes the image size).
     */
    public static void visionMeta(HfLoadContext ctx, HfMetadataBuilder kv, int imageSize, int patchSize)
            throws IOException {
        HfConfig cfg = ctx.config();
        visionMeta(ctx, kv, imageSize, patchSize, visionInt(cfg, EMBEDDING_KEYS), visionInt(cfg, FFN_KEYS),
                visionInt(cfg, HEAD_KEYS), visionBlocks(cfg));
    }

    /**
     * The base audio keys (base.py:3211-3218): {@code clip.has_audio_encoder}, {@code clip.audio.projection_dim}
     * from the text width, and {@code embedding_length} / {@code feed_forward_length} / {@code block_count} /
     * {@code attention.head_count}.
     */
    public static void audioMeta(HfLoadContext ctx, HfMetadataBuilder kv, int hidden, int ffn, int heads, int blocks) {
        kv.set(GgufKeys.CLIP_HAS_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTION_DIM, textWidth(ctx.config()));
        kv.set(GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, ffn);
        kv.set(GgufKeys.CLIP_AUDIO_BLOCK_COUNT, blocks);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT, heads);
    }

    /** The base audio keys with the converter's alias lookups of the audio config. */
    public static void audioMeta(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        audioMeta(ctx, kv, audioInt(cfg, "hidden_size"), audioInt(cfg, "intermediate_size"),
                audioInt(cfg, "num_attention_heads"), audioBlocks(cfg));
    }

    /**
     * The base audio keys with geometry the caller read, naming the configuration object it came from: the
     * counterpart of {@link #visionMetaConfig} for an audio tower.
     */
    public static void audioMetaConfig(HfConfig cfg, HfMetadataBuilder kv, int hidden, int ffn, int heads,
                                       int blocks, String viewName) {
        kv.set(GgufKeys.CLIP_HAS_AUDIO_ENCODER, true);
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTION_DIM, textWidth(cfg));
        kv.set(GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, ffn);
        kv.set(GgufKeys.CLIP_AUDIO_BLOCK_COUNT, blocks);
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT, heads);
    }

    private static List<Float> colorArray(HfConfig cfg, JsonView pre, String key, List<Float> defaults) {
        if (!pre.has(key)) {
            if (defaults != null) {
                return defaults;
            }
            throw new GgufFormatException(cfg.source() + ": the preprocessor configuration has no '" + key
                    + "'; clip.vision.image_mean and clip.vision.image_std carry the normalization colors");
        }
        List<Object> raw = pre.list(key);
        if (raw.size() < 3) {
            throw new GgufFormatException(cfg.source() + ": the preprocessor configuration's '" + key + "' holds "
                    + raw.size() + " values; the normalization colors have three channels");
        }
        List<Float> out = new ArrayList<>(raw.size());
        for (int i = 0; i < raw.size(); i++) {
            if (!(raw.get(i) instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": the preprocessor configuration's '" + key + "["
                        + i + "]' is not a number");
            }
            out.add(((Number) raw.get(i)).floatValue());
        }
        return out;
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * No vocabulary: {@code MmprojModel.write_vocab} raises (base.py:3210) and the main path of the converter writes
     * none, so a projector view carries no {@code tokenizer.*} keys.
     */
    public static Map<String, Object> noVocab(HfLoadContext ctx, PreNameResolver pre) {
        return Collections.emptyMap();
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The fused Q/K/V projection {@code hfName}, split along the output dimension into the projections whose names
     * replace {@code marker} with {@code q} / {@code k} / {@code v}: rows of a weight (rank 2), columns of a bias.
     * Every renamed name maps through the converter's name map. False when the name has no {@code marker}.
     */
    public static boolean splitFused(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, String marker,
                                     String q, String k, String v) throws IOException {
        if (!hfName.contains(marker)) {
            return false;
        }
        long[] shape = src.shape();
        if (shape.length < 1 || shape.length > 2) {
            throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " has the rank "
                    + shape.length + " shape " + Arrays.toString(shape) + "; a fused projection is a weight or a"
                    + " bias");
        }
        long c3 = shape[0];
        if (c3 <= 0 || c3 % 3 != 0 || c3 > Integer.MAX_VALUE) {
            throw new GgufFormatException("tensor '" + hfName + "' of " + ctx.config().source() + " holds " + c3
                    + " fused outputs; the width must divide evenly into Q, K and V");
        }
        int third = (int) (c3 / 3);
        RowSource[] parts = shape.length == 2
                ? RowTransforms.splitRows(src, third, third, third)
                : new RowSource[]{ColumnTransforms.sliceCols(src, 0, third),
                        ColumnTransforms.sliceCols(src, third, third),
                        ColumnTransforms.sliceCols(src, 2L * third, third)};
        out.add(ctx.names().mapOrThrow(hfName.replace(marker, q)), parts[0]);
        out.add(ctx.names().mapOrThrow(hfName.replace(marker, k)), parts[1]);
        out.add(ctx.names().mapOrThrow(hfName.replace(marker, v)), parts[2]);
        return true;
    }

    // ------------------------------------------------------------------ forced storage types

    /**
     * The patch kernels and patch mergers keep F16 at {@code --outtype f16} and F32 otherwise
     * ({@code MmprojModel.tensor_force_quant}, base.py:3247-3251), null for every other tensor.
     */
    public static GGMLType patchKernelForce(String ggufName, ModelSourceOptions.OutType outType) {
        if (ggufName.contains(".patch_embd.weight") || ggufName.contains(".patch_merger.weight")) {
            return outType == ModelSourceOptions.OutType.F16 ? GGMLType.F16 : GGMLType.F32;
        }
        return null;
    }

    /**
     * The tower embeddings keep F32 ({@code ".embeddings." in name} of the classes that force them, gemma.py:280,
     * smolvlm.py:43), the patch kernels the base rule: for {@code v.patch_embd} and {@code v.position_embd}, whose
     * checkpoint names are the vision model's {@code .embeddings.} tensors.
     */
    public static final HfForceType VISION_EMBEDDINGS_F32 = (ggufName, nDims, source, outType) ->
            ggufName.contains("v.patch_embd") || ggufName.contains("v.position_embd")
                    ? GGMLType.F32 : patchKernelForce(ggufName, outType);

    /** The learned position tables keep F32, the patch kernels follow {@link #patchKernelForce}. */
    public static final HfForceType POSITION_EMBD_F32 = (ggufName, nDims, source, outType) ->
            ggufName.contains(".position_embd.") ? GGMLType.F32 : patchKernelForce(ggufName, outType);

    /** The patch kernels of {@link #patchKernelForce} alone. */
    public static final HfForceType PATCH_KERNELS = (ggufName, nDims, source, outType) ->
            patchKernelForce(ggufName, outType);
}
