package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The MiniCPM-V 4.6 / 4.7 / OCR 1.0 projectors and the Dots3-Note vision-and-audio projector, ports of the llama.cpp
 * converter classes {@code MiniCPMV4_6VisionModel} (minicpm.py:141), {@code MiniCPMV4_7VisionModel} (minicpm.py:242),
 * {@code MiniCPMOCR1_0VisionModel} (minicpm.py:288) and {@code Dots3NoteMmprojModel} (dots3.py:201). The MiniCPM
 * classes export the SigLIP tower with its ViT merger inserted before a chosen block and select the downsample mode
 * from the processor; 4.7 additionally carries the tile separators taken from the tokenizer. Dots3-Note exports a
 * pyramid mixture-of-experts vision tower and a speech encoder with its own projector.
 */
public final class HfVisionTowerHooksC {

    /** {@code VisionProjectorType.MINICPMV4_6}. */
    public static final String MINICPMV4_6 = "minicpmv4_6";
    /** {@code VisionProjectorType.MINICPMV4_7}. */
    public static final String MINICPMV4_7 = "minicpmv4_7";
    /** {@code VisionProjectorType.MINICPM_OCR_1_0}. */
    public static final String MINICPM_OCR_1_0 = "minicpm_ocr_1_0";
    /** {@code VisionProjectorType.DOTS3NOTE_V}. */
    public static final String DOTS3NOTE_V = "dots3note_v";
    /** {@code VisionProjectorType.DOTS3NOTE_A}. */
    public static final String DOTS3NOTE_A = "dots3note_a";

    /** The MiniCPM-V 4.7 processor's default slice resolution (minicpm.py:248). */
    private static final int MINICPMV4_7_SCALE_RESOLUTION = 448;
    /** The default ViT-merger insertion block (minicpm.py:186). */
    private static final int DEFAULT_INSERT_LAYER = 6;
    /** The default vision layer-norm epsilon of the MiniCPM-V tower (minicpm.py:196). */
    private static final double MINICPM_EPS = 1e-6;
    /** The tile separators of MiniCPM-V 4.7, their order matching {@code clip_suffix_rows} (minicpm.py:249). */
    private static final String[] MINICPMV4_7_SEPARATORS = {"</image>", "<slice>", "</slice>", "\n"};
    /** The Dots3-Note audio epsilon (dots3.py:317). */
    private static final double DOTS3_AUDIO_EPS = 1e-6;

    private HfVisionTowerHooksC() {
    }

    /**
     * The tensor-map block count of dots3-note: the vision pyramid's layers, or the speech encoder's layers when it is
     * deeper (both encoders share one name map).
     */
    public static int dots3Blocks(HfConfig cfg) {
        int vision = HfMmprojHooks.blocksOf(cfg, vision(cfg), "vision_config");
        JsonView audio = HfMmprojHooks.object(cfg, cfg.raw(), "audio_config");
        if (!audio.has("whisper_config")) {
            return vision;
        }
        JsonView whisper = HfMmprojHooks.object(cfg, audio, "whisper_config");
        return Math.max(vision, HfMmprojHooks.intOr(cfg, whisper, 0, "encoder_layers"));
    }

    private static JsonView vision(HfConfig cfg) {
        return cfg.vision().orElseThrow(() -> new GgufFormatException(cfg.source()
                + ": the checkpoint has no 'vision_config'"));
    }

    // ------------------------------------------------------------------ MiniCPM-V 4.6 / 4.7 / OCR 1.0

    /** The downsampling mode of a MiniCPM-V checkpoint (minicpm.py:152-158, :254-258). */
    private static String downsampleMode(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        JsonView pre;
        try {
            pre = HfMmprojHooks.preprocessor(cfg);
        } catch (IOException e) {
            throw new GgufFormatException(cfg.source() + ": cannot read the preprocessor configuration", e);
        }
        Object raw = pre.raw("downsample_mode");
        if (raw == null) {
            raw = cfg.raw().raw("downsample_mode");
        }
        String mode = raw instanceof String ? (String) raw : "16x";
        if (!"4x".equals(mode) && !"16x".equals(mode)) {
            throw new GgufFormatException(cfg.source() + ": Unsupported downsample mode: " + mode
                    + " (minicpm.py:156-158)");
        }
        return mode;
    }

    /**
     * {@code MiniCPMV4_6VisionModel.filter_tensors} (minicpm.py:199-207) with the 4.7 separator tensor and the 4x
     * mode's dropped ViT merger.
     */
    public static final java.util.function.Function<HfLoadContext, HfNameFilter> MINICPM_FILTER = ctx -> {
        boolean keepSeparators = "MiniCPMV4_7ForConditionalGeneration".equals(ctx.config().architecture());
        boolean dropMerger = "4x".equals(downsampleMode(ctx));
        return (name, base) -> {
            if (name.startsWith("lm_head.") || name.startsWith("mtp")) {
                return null;
            }
            if (keepSeparators && "model.language_model.embed_tokens.weight".equals(name)) {
                return name;
            }
            if (dropMerger && name.contains(".vit_merger.")) {
                return null;
            }
            return base.apply(name);
        };
    };

    /**
     * {@code MiniCPMV4_6VisionModel.set_gguf_parameters} (minicpm.py:167-196): the base vision keys with the
     * processor's slice resolution, the class's projector type, the downsample scale factor, the slice count, the
     * ViT-merger insertion layer, GELU and the epsilon. The 4.7 class and the OCR class differ in type and default
     * slice resolution only.
     *
     * @throws GgufFormatException naming the unsupported mode or missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void minicpmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        String arch = cfg.architecture();
        String type;
        Integer defaultResolution;
        if ("MiniCPMV4_7ForConditionalGeneration".equals(arch)) {
            type = MINICPMV4_7;
            defaultResolution = MINICPMV4_7_SCALE_RESOLUTION;
        } else if ("MiniCPMOcrForConditionalGeneration".equals(arch)) {
            type = MINICPM_OCR_1_0;
            defaultResolution = null;
        } else {
            type = MINICPMV4_6;
            defaultResolution = null;
        }
        String mode = downsampleMode(ctx);
        JsonView pre = HfMmprojHooks.preprocessor(cfg);
        Object rawResolution = pre.raw("scale_resolution");
        int imageSize;
        if (rawResolution instanceof Number) {
            imageSize = ((Number) rawResolution).intValue();
        } else if (defaultResolution != null) {
            imageSize = defaultResolution;
        } else {
            imageSize = HfMmprojHooks.visionInt(cfg, "image_size");
        }
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, HfMmprojHooks.visionInt(cfg, "patch_size"),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.HEAD_KEYS), HfMmprojHooks.visionBlocks(cfg));
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, type);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, "4x".equals(mode) ? 2 : 4);
        Object slices = pre.raw("max_slice_nums");
        if (slices instanceof Number) {
            kv.set(GgufKeys.CLIP_VISION_MAX_SLICE_NUMS, ((Number) slices).intValue());
        }
        int insert = insertLayer(ctx);
        kv.set(GgufKeys.CLIP_VISION_WA_LAYER_INDEXES, Collections.singletonList(insert));
        kv.set(GgufKeys.CLIP_USE_GELU, true);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(MINICPM_EPS));
    }

    /** The ViT-merger insertion block: the root then the vision config's {@code insert_layer_id}, 6 otherwise. */
    private static int insertLayer(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        Object root = cfg.raw().raw("insert_layer_id");
        if (root instanceof Number && !(root instanceof Boolean)) {
            return ((Number) root).intValue();
        }
        return HfMmprojHooks.visionIntOpt(cfg, "insert_layer_id").orElse(DEFAULT_INSERT_LAYER);
    }

    /**
     * {@code MiniCPMV4_7VisionModel.modify_tensors} (minicpm.py:267-280): the tile separators are rows of the text
     * embedding, and every other tensor maps through the clip name table.
     *
     * @throws GgufFormatException when a separator is not a single token of the checkpoint's tokenizer
     * @throws IOException         when the tokenizer files cannot be read
     */
    public static void minicpmTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if ("model.language_model.embed_tokens.weight".equals(hfName)) {
            int[] ids = separatorIds(ctx);
            out.add("v.tok_embd_sep", RowTransforms.gatherRows(src, 1, ids));
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** The token id of each tile separator, in {@link #MINICPMV4_7_SEPARATORS} order. */
    private static int[] separatorIds(HfLoadContext ctx) throws IOException {
        Path file = ctx.tokenizerFiles().tokenizerJson();
        if (file == null) {
            throw new GgufFormatException(ctx.config().source() + ": the MiniCPM-V 4.7 separator rows need"
                    + " tokenizer.json");
        }
        HfTokenizerJson tok = HfTokenizerJson.parse(file);
        Map<String, Integer> byText = new LinkedHashMap<>();
        for (HfTokenizerJson.AddedToken t : tok.addedTokens()) {
            byText.putIfAbsent(t.content(), t.id());
        }
        String[] vocab = tok.model().vocab();
        for (int i = 0; i < vocab.length; i++) {
            byText.putIfAbsent(vocab[i], i);
        }
        int[] ids = new int[MINICPMV4_7_SEPARATORS.length];
        for (int i = 0; i < ids.length; i++) {
            String separator = MINICPMV4_7_SEPARATORS[i];
            Integer id = byText.get(separator);
            if (id == null) {
                id = byText.get(byteLevel(separator));
            }
            if (id == null) {
                throw new GgufFormatException(ctx.config().source() + ": the MiniCPM-V 4.7 separator "
                        + quote(separator) + " is not a single token of the tokenizer");
            }
            ids[i] = id;
        }
        return ids;
    }

    /** GPT-2's byte-level alphabet: the string a byte-level BPE vocabulary stores the UTF-8 text as. */
    private static String byteLevel(String text) {
        StringBuilder b = new StringBuilder();
        for (byte raw : text.getBytes(StandardCharsets.UTF_8)) {
            int c = raw & 0xFF;
            if (c >= 33 && c <= 126 || c >= 161 && c <= 172 || c >= 174 && c <= 255) {
                b.append((char) c);
            } else {
                b.append((char) (256 + c));
            }
        }
        return b.toString();
    }

    private static String quote(String text) {
        return "'" + text.replace("\\", "\\\\").replace("\n", "\\n") + "'";
    }

    // ------------------------------------------------------------------ Dots3-Note

    /** {@code Dots3NoteMmprojModel.filter_tensors} (dots3.py:273-280): the vision and audio encoders. */
    public static final HfNameFilter DOTS3_FILTER = (name, base) ->
            name.startsWith("vision_encoder.") || name.startsWith("audio_encoder.") ? base.apply(name) : null;

    /**
     * {@code Dots3NoteMmprojModel} (dots3.py:201-272): the pyramid mixture-of-experts vision tower with its
     * {@code dots3note_v} projector and the speech encoder with {@code dots3note_a}. Every configuration variant the
     * clip graphs do not implement is refused by name.
     *
     * @throws GgufFormatException naming the unsupported or missing key
     * @throws IOException         when the preprocessor files cannot be read
     */
    public static void dots3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        JsonView vision = vision(cfg);
        String adapter = stringOr(vision, "adapter_type", "");
        if (!"patch_merger".equals(adapter) || !truthy(vision, "pre_pixel_shuffle")) {
            throw new GgufFormatException(cfg.source() + ": dots3-note vision conversion requires"
                    + " adapter_type=patch_merger and pre_pixel_shuffle (dots3.py:224-225)");
        }
        if (!"sigmoid".equals(stringOr(vision, "router_scoring_func", "sigmoid"))
                || f64Or(vision, "router_scale", 1.0) != 1.0) {
            throw new GgufFormatException(cfg.source() + ": dots3-note vision conversion only supports sigmoid"
                    + " routing with router_scale=1.0 (dots3.py:226-228)");
        }
        if (HfMmprojHooks.intOr(cfg, vision, 1, "temporal_patch_size") != 1 || truthy(vision, "use_bias")
                || !truthy(vision, "use_qk_norm")) {
            throw new GgufFormatException(cfg.source() + ": unsupported dots3-note vision config variant"
                    + " (dots3.py:229-230)");
        }
        JsonView pre = mergedPreprocessor(cfg);
        int hidden = HfMmprojHooks.intOf(cfg, vision, "vision_config", "embed_dim");
        int imageSize = 0;
        HfMmprojHooks.visionMeta(ctx, kv, imageSize, HfMmprojHooks.intOf(cfg, vision, "vision_config", "patch_size"),
                hidden, HfMmprojHooks.intOf(cfg, vision, "vision_config", HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.intOf(cfg, vision, "vision_config", HfMmprojHooks.HEAD_KEYS),
                HfMmprojHooks.blocksOf(cfg, vision, "vision_config"),
                pre.has("image_mean") ? numbers(cfg, pre, "image_mean") : null,
                pre.has("image_std") ? numbers(cfg, pre, "image_std") : null);
        kv.set(GgufKeys.CLIP_VISION_PROJECTOR_TYPE, DOTS3NOTE_V);
        kv.set(GgufKeys.CLIP_USE_SILU, true);
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, f64Or(vision, "rms_norm_eps", 1e-6));
        kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE,
                HfMmprojHooks.intOf(cfg, vision, "vision_config", "spatial_merge_size"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, HfMmprojHooks.intOf(cfg, pre, "preprocessor", "min_pixels"));
        kv.set(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, HfMmprojHooks.intOf(cfg, pre, "preprocessor", "max_pixels"));
        List<Integer> pyramid = pyramid(cfg);
        kv.set(GgufKeys.CLIP_VISION_EXPERT_COUNT_PER_LAYER, pyramid);
        kv.set(GgufKeys.CLIP_VISION_EXPERT_USED_COUNT,
                (int) f64Or(vision, "capacity_factor", 1.0));

        Object audioRaw = cfg.raw().raw("audio_config");
        if (!(audioRaw instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": the Dots3-Note checkpoint has no 'audio_config'");
        }
        JsonView audio = JsonView.of(asMap(audioRaw), "audio_config");
        JsonView whisper = HfMmprojHooks.object(cfg, audio, "whisper_config");
        if (!"swiglu".equals(stringOr(whisper, "activation_function", ""))) {
            throw new GgufFormatException(cfg.source() + ": dots3-note audio conversion requires the swiglu"
                    + " activation (dots3.py:236-237)");
        }
        checkAudioVariant(cfg, audio, whisper);
        int audioHidden = intOfAny(cfg, audio, whisper, "hidden_size", "d_model");
        int audioFfn = intOfAny(cfg, audio, whisper, "intermediate_size", "encoder_ffn_dim");
        int audioHeads = intOfAny(cfg, audio, whisper, "num_attention_heads", "encoder_attention_heads");
        int audioBlocks = blocksOfAudio(cfg, audio, whisper);
        HfMmprojHooks.audioMeta(ctx, kv, audioHidden, audioFfn, audioHeads, audioBlocks);
        kv.set(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, DOTS3NOTE_A);
        kv.set(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, HfMmprojHooks.intOf(cfg, whisper, "whisper_config",
                "num_mel_bins"));
        kv.set(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON, DOTS3_AUDIO_EPS);
    }

    /** The audio variant checks of {@code Dots3NoteMmprojModel.__init__} (dots3.py:231-241). */
    private static void checkAudioVariant(HfConfig cfg, JsonView audio, JsonView whisper) {
        if (!truthy(audio, "use_conv2d_stem") || !truthy(audio, "use_rope") || !truthy(audio, "use_rms_norm")
                || truthy(audio, "use_causal")) {
            throw new GgufFormatException(cfg.source() + ": unsupported dots3-note audio config variant"
                    + " (dots3.py:232-234)");
        }
        if (HfMmprojHooks.intOr(cfg, audio, 1, "merge_factor") != 1
                || HfMmprojHooks.intOr(cfg, audio, 0, "chunk_seconds") != 60) {
            throw new GgufFormatException(cfg.source() + ": unsupported dots3-note audio chunking config"
                    + " (dots3.py:240-241)");
        }
        Object rope = audio.raw("rope_parameters");
        JsonView ropeView = rope instanceof Map ? JsonView.of(asMap(rope), "audio_config.rope_parameters")
                : JsonView.empty(cfg.source(), "audio_config.rope_parameters");
        if (f64Or(ropeView, "partial_rotary_factor", -1.0) != 0.5
                || f64Or(ropeView, "rope_theta", -1.0) != 10_000.0) {
            throw new GgufFormatException(cfg.source() + ": unsupported dots3-note audio rope config"
                    + " (dots3.py:242-245)");
        }
    }

    /**
     * {@code Dots3NoteMmprojModel.modify_tensors} (dots3.py:282-304): the router tensors gain their {@code .weight},
     * the fused audio {@code fc1} splits into the gate and up halves, and the vision experts of a block stack into one
     * tensor per projection.
     *
     * @throws GgufFormatException when a router name is unexpected, an expert block is dense, or the fused audio
     *                             projection has an odd width
     */
    public static void dots3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String name = hfName;
        if (name.endsWith(".gate_weight") || name.endsWith(".router_bias")) {
            name += ".weight";
        }
        if (name.contains(".speech_encoder.layers.") && name.contains(".fc1.")) {
            long[] shape = src.shape();
            if (shape.length < 1 || shape[0] % 2 != 0) {
                throw new GgufFormatException("Dots3-Note fused audio fc1 has shape " + Arrays.toString(shape)
                        + "; its rows are the even gate and up halves (dots3.py:286-289)");
            }
            long half = shape[0] / 2;
            out.add(ctx.names().mapOrThrow(name.replace(".fc1.", ".fc1_gate.")),
                    RowTransforms.sliceRows(src, 0, half));
            out.add(ctx.names().mapOrThrow(name.replace(".fc1.", ".fc1_up.")),
                    RowTransforms.sliceRows(src, half, half));
            return;
        }
        if (name.contains(".mlp.experts.")) {
            int bid = HfLoadContext.bidOf(name);
            List<Integer> pyramid = pyramid(ctx.config());
            if (bid < 0 || bid >= pyramid.size() || pyramid.get(bid) <= 0) {
                throw new GgufFormatException("Dots3-Note vision expert tensor '" + hfName + "' has no routed"
                        + " experts in its block");
            }
            collector(ctx, pyramid.get(bid)).offer(ctx, name, out);
            return;
        }
        out.add(ctx.names().mapOrThrow(name), src);
    }

    /** The per-block expert collector, kept per load. */
    private static HfHookSupport.ExpertCollector collector(HfLoadContext ctx, int experts) {
        @SuppressWarnings("unchecked")
        Map<Integer, HfHookSupport.ExpertCollector> map =
                ctx.state("dots3.experts", Map.class, LinkedHashMap::new);
        return map.computeIfAbsent(experts, n -> new HfHookSupport.ExpertCollector(n,
                Arrays.asList("fc1", "fc2", "fc3"), HfHookSupport.MergedName.SAME_PREFIX));
    }

    /** The routed expert count per vision block (dots3.py:222). */
    private static List<Integer> pyramid(HfConfig cfg) {
        List<Integer> out = new ArrayList<>();
        for (int n : HfMmprojHooks.intArray(cfg, vision(cfg), "pyramid_num_routed")) {
            out.add(Math.max(0, n));
        }
        return out;
    }

    /** The F32 routing tables and convolutions (dots3.py:306-313). */
    public static final HfForceType DOTS3_FORCE = (ggufName, nDims, source, outType) -> {
        if (ggufName.contains(".ffn_gate_inp.") || ggufName.contains(".exp_probs_b.")
                || ggufName.contains(".conv2d") || ggufName.contains("a.conv_out")) {
            return GGMLType.F32;
        }
        return HfMmprojHooks.patchKernelForce(ggufName, outType);
    };

    /** The preprocessor configuration with a nested {@code vision_config} object hoisted over it (dots3.py:209-210). */
    private static JsonView mergedPreprocessor(HfConfig cfg) throws IOException {
        JsonView root = HfMmprojHooks.preprocessor(cfg);
        Object nested = root.raw("vision_config");
        if (!(nested instanceof Map)) {
            return root;
        }
        Map<String, Object> merged = new LinkedHashMap<>();
        for (String key : root.keys()) {
            merged.put(key, root.raw(key));
        }
        merged.putAll(asMap(nested));
        return JsonView.of(merged, "the merged preprocessor configuration");
    }

    private static int intOfAny(HfConfig cfg, JsonView primary, JsonView fallback, String... aliases) {
        for (String alias : aliases) {
            Object raw = primary.raw(alias);
            if (raw instanceof Number && !(raw instanceof Boolean)) {
                return ((Number) raw).intValue();
            }
            raw = fallback.raw(alias);
            if (raw instanceof Number && !(raw instanceof Boolean)) {
                return ((Number) raw).intValue();
            }
        }
        throw new GgufFormatException(cfg.source() + ": the audio configuration has none of "
                + Arrays.toString(aliases));
    }

    private static int blocksOfAudio(HfConfig cfg, JsonView audio, JsonView whisper) {
        for (String key : HfMmprojHooks.BLOCK_COUNT_KEYS) {
            Object raw = audio.raw(key);
            if (raw instanceof Number && !(raw instanceof Boolean)) {
                return ((Number) raw).intValue();
            }
            raw = whisper.raw(key);
            if (raw instanceof Number && !(raw instanceof Boolean)) {
                return ((Number) raw).intValue();
            }
        }
        throw new GgufFormatException(cfg.source() + ": the audio configuration has no block count");
    }

    private static boolean truthy(JsonView view, String key) {
        return Boolean.TRUE.equals(view.raw(key));
    }

    private static String stringOr(JsonView view, String key, String dflt) {
        Object raw = view.raw(key);
        return raw instanceof String ? (String) raw : dflt;
    }

    private static double f64Or(JsonView view, String key, double dflt) {
        Object raw = view.raw(key);
        return raw instanceof Number && !(raw instanceof Boolean) ? ((Number) raw).doubleValue() : dflt;
    }

    private static List<Float> numbers(HfConfig cfg, JsonView view, String key) {
        Object raw = view.raw(key);
        if (!(raw instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' is not a list of colors");
        }
        List<Float> out = new ArrayList<>();
        for (Object v : (List<?>) raw) {
            if (!(v instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": '" + key + "' holds a non-number");
            }
            out.add(((Number) v).floatValue());
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }
}
