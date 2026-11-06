package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfConfigFixup;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The Pixtral vision tower and its projector through the Hugging Face loader, a port of llama.cpp {@code
 * LlavaVisionModel} (LC/conversion/llava.py:17-131) and its {@code Mistral3EmbeddingVisionModel} subclass
 * (llava.py:134-148): the vision tower under the {@code multi_modal_projector.} / {@code vision_tower.} /
 * {@code vision_encoder.} / {@code vision_language_adapter.} / {@code patch_merger.} /
 * {@code pre_mm_projector_norm} prefixes with the rotate-half Q/K rows un-permuted, the {@code pixtral} projector
 * with its activation and layer-norm epsilon, the spatial merge of the root configuration, and the
 * {@code [IMG_BREAK]} token row of the text embedding as {@code v.token_embd.img_break}. Mistral native
 * {@code params.json} checkpoints convert too ({@code PixtralModel}, pixtral.py): their adapter projections are
 * {@code mm.1} / {@code mm.2}, their Q/K rows already sit in the rotate-half layout, and the break row comes from
 * {@code tok_embeddings.weight}. Every method is a public static hook reference.
 */
public final class HfLlavaVisionHooks {

    /** The Hugging Face architecture name of the LLaVA / Pixtral multimodal model. */
    public static final String LLAVA = "LlavaForConditionalGeneration";
    /** The Hugging Face architecture name of Mistral Small 3.1. */
    public static final String MISTRAL3 = "Mistral3ForConditionalGeneration";
    /** The Hugging Face architecture name of the Nemotron embedding model with the Pixtral tower. */
    public static final String MISTRAL3_EMBEDDING = "Mistral3Model";
    /** The row the projector separates image lines with. */
    public static final String IMG_BREAK_TOKEN = "[IMG_BREAK]";

    /** The tower and projector prefixes {@code LlavaVisionModel.modify_tensors} keeps (llava.py:87-94). */
    private static final String[] PREFIXES = {"multi_modal_projector.", "vision_tower.", "vision_encoder.",
            "vision_language_adapter.", "patch_merger.", "pre_mm_projector_norm"};
    /** The default of the hard-coded vision layer-norm epsilon (llava.py:28). */
    private static final double LAYER_NORM_EPS = 1e-5;
    /** The fixed Mistral-native normalization colors (base.py:44-45). */
    private static final float[] MISTRAL_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] MISTRAL_STD = {0.26862954f, 0.26130258f, 0.27577711f};

    private HfLlavaVisionHooks() {
    }

    /**
     * The projector view of the embedding model needs the headless text model ({@code is_causal} false,
     * llava.py:137-139); anything else is refused naming the configuration.
     */
    public static final HfConfigFixup EMBEDDING_FIXUP = cfg -> {
        Object causal = cfg.flat().get("is_causal");
        if (!Boolean.FALSE.equals(causal)) {
            throw new GgufFormatException(cfg.source() + ": '" + MISTRAL3_EMBEDDING + "' as a projector expects"
                    + " the Nemotron Embed text model with 'is_causal' false, got " + (causal == null ? "nothing"
                    : causal));
        }
        return cfg;
    };

    /**
     * The base vision keys and the {@code pixtral} projector. For a Hugging Face checkpoint the activation, the
     * vision layer-norm epsilon and the root {@code spatial_merge_size} come from the vision and root configurations
     * (llava.py:100-113); for a Mistral native {@code params.json} checkpoint the projector is {@code pixtral} with
     * SiLU, the root {@code norm_eps}, the vision tower's {@code rope_theta} and the spatial merge read from
     * {@code mm_projector_id == "patch_merge"}, and the preprocessor's fixed normalization colors
     * (LC/conversion/pixtral.py:16-32).
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        if (cfg.mistralNative()) {
            mistralNativeMeta(ctx, kv);
            return;
        }
        String modelType = HfMmprojHooks.visionStr(cfg, "model_type");
        if (!"pixtral".equals(modelType)) {
            throw new GgufFormatException(cfg.source() + ": vision model_type '" + modelType
                    + "' is unsupported: LlavaVisionModel converts the Pixtral tower (llava.py:51)");
        }
        HfMmprojHooks.visionMeta(ctx, kv);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, "pixtral");
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON,
                HfMmprojHooks.visionF64Opt(cfg, "layer_norm_eps").orElse(LAYER_NORM_EPS));
        activation(cfg, kv);
        JsonView root = cfg.raw();
        if (root.has("spatial_merge_size")) {
            kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, root.i32("spatial_merge_size"));
        }
    }

    /** The Mistral native Pixtral keys ({@code PixtralModel.set_gguf_parameters}, pixtral.py:16-32). */
    private static void mistralNativeMeta(HfLoadContext ctx, HfMetadataBuilder kv) throws IOException {
        HfConfig cfg = ctx.config();
        HfMmprojHooks.visionMeta(ctx, kv, HfMmprojHooks.visionInt(cfg, "image_size"),
                HfMmprojHooks.visionInt(cfg, "patch_size"), HfMmprojHooks.visionInt(cfg, HfMmprojHooks.EMBEDDING_KEYS),
                HfMmprojHooks.visionInt(cfg, HfMmprojHooks.FFN_KEYS),
                HfMmprojHooks.visionInt(cfg, "num_attention_heads"), HfMmprojHooks.visionBlocks(cfg),
                MISTRAL_MEAN, MISTRAL_STD);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, "pixtral");
        Object eps = cfg.flat().get("norm_eps");
        if (!(eps instanceof Number) || eps instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": 'norm_eps' of params.json is required for the"
                    + " Pixtral tower (pixtral.py:20-22)");
        }
        kv.set(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, ((Number) eps).doubleValue());
        Object theta = HfMmprojHooks.visionFind(cfg, "rope_theta").orElse(null);
        if (!(theta instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": the vision encoder has no numeric 'rope_theta'"
                    + " (pixtral.py:23)");
        }
        kv.set(GgufKeys.ROPE_FREQ_BASE, ((Number) theta).doubleValue());
        kv.set(GgufKeys.CLIP_USE_SILU, true);
        if ("patch_merge".equals(HfMmprojHooks.visionFind(cfg, "mm_projector_id").orElse(null))) {
            kv.set(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE,
                    HfMmprojHooks.visionInt(cfg, "spatial_merge_size"));
        }
    }

    /** {@link #meta} plus the preprocessor's longest edge as {@code clip.vision.preproc_image_size} (llava.py:147). */
    public static void embeddingMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        meta(ctx, kv, plan);
        JsonView size = HfMmprojHooks.preprocessor(ctx.config()).obj("size");
        if (!size.has("longest_edge")) {
            throw new GgufFormatException(ctx.config().source() + ": the preprocessor configuration's 'size' has no"
                    + " 'longest_edge'; clip.vision.preproc_image_size carries the longest edge");
        }
        kv.set(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, size.i32("longest_edge"));
    }

    /**
     * The tower and projector tensors, the Q/K rows un-permuted from the rotate-half layout (llava.py:103-106), and
     * the {@code [IMG_BREAK]} embedding row of {@code embed_tokens.weight} as {@code v.token_embd.img_break}
     * (llava.py:123-127). Names under no prefix and the rest of the embedding are dropped.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        towerTensors(ctx, hfName, src, out, true);
    }

    /** The tower and projector names, mapped as {@link #tensors} but without the break row. */
    public static void lightOnOcrTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        towerTensors(ctx, hfName, src, out, false);
    }

    /**
     * The tower and projector tensors, the Q/K rows un-permuted from the rotate-half layout (llava.py:103-106), and
     * with {@code breakRow} the {@code [IMG_BREAK]} embedding row of {@code embed_tokens.weight} as
     * {@code v.token_embd.img_break} (llava.py:123-127). Names under no prefix and the rest of the embedding are
     * dropped.
     */
    private static void towerTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out,
                                     boolean breakRow) throws IOException {
        if (startsWithAny(hfName)) {
            RowSource data = src;
            if (!ctx.config().mistralNative() && (hfName.endsWith("q_proj.weight")
                    || hfName.endsWith("q_proj.bias") || hfName.endsWith("k_proj.weight")
                    || hfName.endsWith("k_proj.bias"))) {
                int heads = HfMmprojHooks.visionInt(ctx.config(), "num_attention_heads");
                data = RowTransforms.permuteQk(data, heads, heads);
            }
            if (ctx.config().mistralNative() && adapter(ctx, hfName, src, out)) {
                return;
            }
            out.add(ctx.names().mapOrThrow(hfName), data);
            return;
        }
        String embedKey = ctx.config().mistralNative() ? "tok_embeddings.weight" : "embed_tokens.weight";
        if (breakRow && hfName.contains(embedKey)) {
            int id = ctx.config().mistralNative() ? mistralBreakId(ctx) : tokenId(ctx, IMG_BREAK_TOKEN);
            if (id > 0) {
                long[] shape = src.shape();
                long[] rowShape = Arrays.copyOfRange(shape, 1, shape.length);
                out.add(ctx.names().ggufName("V_TOK_EMBD_IMG_BREAK", -1),
                        ElementTransforms.reshape(RowTransforms.sliceRows(src, id, 1), rowShape));
            }
        }
    }

    /** The Mistral-native adapter projections as {@code mm.1} / {@code mm.2} (pixtral.py:35-44); false otherwise. */
    private static boolean adapter(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String suffix;
        int tensor;
        if (hfName.startsWith("vision_language_adapter.w_in.")) {
            tensor = 1;
            suffix = hfName.substring("vision_language_adapter.w_in".length());
        } else if (hfName.startsWith("vision_language_adapter.w_out.")) {
            tensor = 2;
            suffix = hfName.substring("vision_language_adapter.w_out".length());
        } else {
            return false;
        }
        out.add(ctx.names().ggufName("V_MMPROJ", tensor, suffix), src);
        return true;
    }

    /**
     * The Mistral-native {@code image_break_token_id} of the vision configuration, resolving the {@code -1}
     * placeholder from {@code tekken.json} or {@code tokenizer.json} (llava.py:34-44).
     *
     * @throws GgufFormatException when no tokenizer file names the token
     */
    private static int mistralBreakId(HfLoadContext ctx) throws IOException {
        Object raw = HfMmprojHooks.visionFind(ctx.config(), "image_break_token_id").orElse(null);
        if (raw instanceof Number && ((Number) raw).intValue() > 0) {
            return ((Number) raw).intValue();
        }
        Map<String, Object> tekken = jsonObject(ctx.config().dir().resolve("tekken.json"));
        if (tekken != null && tekken.get("special_tokens") instanceof List) {
            for (Object e : (List<?>) tekken.get("special_tokens")) {
                if (e instanceof Map && IMG_BREAK_TOKEN.equals(((Map<?, ?>) e).get("token_str"))) {
                    Object rank = ((Map<?, ?>) e).get("rank");
                    if (rank instanceof Number) {
                        return ((Number) rank).intValue();
                    }
                }
            }
        }
        Map<String, Object> tokenizer = jsonObject(ctx.tokenizerFiles().tokenizerJson());
        if (tokenizer != null && tokenizer.get("added_tokens") instanceof List) {
            for (Object t : (List<?>) tokenizer.get("added_tokens")) {
                if (t instanceof Map && IMG_BREAK_TOKEN.equals(((Map<?, ?>) t).get("content"))) {
                    Object id = ((Map<?, ?>) t).get("id");
                    if (id instanceof Number) {
                        return ((Number) id).intValue();
                    }
                }
            }
        }
        throw new GgufFormatException("Token '" + IMG_BREAK_TOKEN + "' not found in the Mistral tokenizer files"
                + " (llava.py:44); the Pixtral projector writes its row as v.token_embd.img_break");
    }

    /** {@code use_silu} for {@code silu}, {@code use_gelu} for {@code gelu}; anything else is refused by name. */
    private static void activation(HfConfig cfg, HfMetadataBuilder kv) {
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
                        + "' is unsupported: the Pixtral tower runs silu or gelu (llava.py:107-111)");
        }
    }

    private static boolean startsWithAny(String hfName) {
        for (String prefix : PREFIXES) {
            if (hfName.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ LightOnOCR

    /**
     * {@link #meta} with the {@code lightonocr} projector (lighton_ocr.py:17-21): the same Pixtral tower, without
     * the {@code [IMG_BREAK]} row ({@code use_break_tok} false, lighton_ocr.py:15).
     */
    public static void lightOnOcrMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        meta(ctx, kv, plan);
        kv.set(GgufKeys.CLIP_PROJECTOR_TYPE, "lightonocr");
    }

    /**
     * The newer checkpoint names (lighton_ocr.py:23-30): {@code model.vision_encoder.} renamed to
     * {@code vision_tower.} and {@code model.vision_projection.} to {@code multi_modal_projector.} before the
     * prefix check, then {@link #tensors} without the break row.
     */
    public static final HfNameFilter LIGHTONOCR_FILTER = (name, base) -> base.apply(name
            .replace("model.vision_encoder.", "vision_tower.")
            .replace("model.vision_projection.", "multi_modal_projector."));

    /** The Hugging Face architecture name of the LightOnOCR multimodal model. */
    public static final String LIGHTONOCR = "LightOnOCRForConditionalGeneration";

    /**
     * The id of {@code token}: the {@code added_tokens_decoder} of {@code tokenizer_config.json}, else the
     * {@code added_tokens} of {@code tokenizer.json} (llava.py:57-71).
     *
     * @throws GgufFormatException when the tokenizer files name no such token
     */
    static int tokenId(HfLoadContext ctx, String token) throws IOException {
        Map<String, Object> config = jsonObject(ctx.tokenizerFiles().tokenizerConfigJson());
        if (config != null) {
            Object decoder = config.get("added_tokens_decoder");
            if (decoder instanceof Map) {
                for (Map.Entry<?, ?> e : asMap(decoder).entrySet()) {
                    Object data = e.getValue();
                    if (data instanceof Map && token.equals(asMap(data).get("content"))) {
                        return Integer.parseInt(String.valueOf(e.getKey()));
                    }
                }
            }
        }
        Map<String, Object> tokenizer = jsonObject(ctx.tokenizerFiles().tokenizerJson());
        if (tokenizer != null) {
            Object added = tokenizer.get("added_tokens");
            if (added instanceof List) {
                for (Object t : (List<?>) added) {
                    if (t instanceof Map && token.equals(((Map<?, ?>) t).get("content"))) {
                        Object id = ((Map<?, ?>) t).get("id");
                        if (id instanceof Number) {
                            return ((Number) id).intValue();
                        }
                    }
                }
            }
        }
        throw new GgufFormatException("Token '" + token + "' not found in tokenizer config. (llava.py:71): the"
                + " Pixtral projector writes the " + token + " row as v.token_embd.img_break");
    }

    private static Map<String, Object> jsonObject(java.nio.file.Path file) throws IOException {
        if (file == null || !Files.isRegularFile(file)) {
            return null;
        }
        try {
            Object parsed = Json.parse(file, HfConfigLoader.MAX_CONFIG_BYTES, Json.Limits.LENIENT);
            return parsed instanceof Map ? asMap(parsed) : null;
        } catch (com.eonmux.neuralfree.utils.json.JsonException e) {
            throw new GgufFormatException(e.getMessage(), e);
        }
    }

    private static Map<String, Object> asMap(Object v) {
        return (Map<String, Object>) v;
    }
}
