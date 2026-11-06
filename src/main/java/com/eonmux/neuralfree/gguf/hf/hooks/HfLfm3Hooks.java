package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Hooks of the LFM3 text converter (LC/conversion/lfm3.py): {@code LFM3Model} (lfm3.py:18-119) and the decision
 * {@code LFM3D1Model} (lfm3.py:145-176). The text model of the VL/Omni/Audio checkpoints is the same class; its
 * encoders and output heads are dropped by the class filter.
 *
 * <p>The model mixes causal short convolutions with full-attention and sliding-window layers. The per-layer key/value
 * head vector marks the convolution layers with 0 (lfm3.py:86-92), the full-attention layers rotate with the learned
 * {@code rope_freqs} factors of the partial rotary factor (lfm3.py:101-112), and the post-attention and post-FFN norms
 * are optional. A d1 checkpoint reads its full-attention layers bidirectionally and records the decision head
 * temperatures (lfm3.py:159-176).</p>
 */
public final class HfLfm3Hooks {

    /**
     * {@code LFM3Model._media_prefixes} (lfm3.py:24-25): the encoder, projector and audio-output tensors that live
     * next to the text model and do not become part of its file.
     */
    private static final List<String> MEDIA_PREFIXES = List.of("encoder.", "mlp.", "codebook_embedding.",
            "depth_embeddings.", "depth_linear.", "depthformer.", "mask_embedding", "model.codebook_embedding.",
            "model.depth_decoder.");

    /** The LFM3 full-attention and sliding-window layer types (lfm3.py:83-97). */
    private static final String SLIDING_ATTENTION = "sliding_attention";
    private static final String CONV = "conv";

    /** The d1 served configuration the converter refuses to change (lfm3.py:151). */
    private static final long D1_CODES = 2;
    private static final String D1_ATTENTION = "set";

    /** The decision temperature names of the d1 calibration (lfm3.py:175). */
    private static final List<String> D1_CALIBRATION_TEMPERATURES = List.of("noul", "choice", "score");

    /** The languages of the TTS prompt (lfm3.py:46-51), in the converter's order. */
    private static final Map<String, String> TTS_LANGS = ttsLangs();

    private HfLfm3Hooks() {
    }

    // ------------------------------------------------------------------ configuration and filter

    /** {@link Arch#LFM3} for every LFM3 checkpoint, the d1 and the VL/Omni/Audio text sides included. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.LFM3;
    }

    /**
     * {@code LFM3Model.set_gguf_parameters}' key/value patch (lfm3.py:86-92): {@code num_key_value_heads} becomes the
     * per-layer vector, 0 on the convolution layers, so the generic metadata writes the vector.
     *
     * @throws GgufFormatException when {@code layer_types} is missing or {@code num_key_value_heads} is not a scalar
     */
    public static HfConfig fixup(HfConfig cfg) {
        List<?> types = layerTypes(cfg);
        Object heads = cfg.flat().get("num_key_value_heads");
        if (!(heads instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": 'num_key_value_heads' must be a scalar for the per-layer"
                    + " key/value vector of lfm3 (lfm3.py:85-92), got " + heads);
        }
        long value = ((Number) heads).longValue();
        List<Number> out = new ArrayList<>(types.size());
        for (Object type : types) {
            out.add(CONV.equals(type) ? 0L : value);
        }
        return cfg.withFlat(Collections.singletonMap("num_key_value_heads", out));
    }

    /**
     * {@code LFM3Model.filter_tensors} (lfm3.py:27-36): the encoder, projector and audio-output tensors skip and every
     * {@code lfm.} prefix strips before the inherited filter.
     */
    public static final HfNameFilter FILTER = (name, base) -> {
        if (startsWithMedia(name)) {
            return null;
        }
        return base.apply(HfHookSupport.stripPrefixes(name, "lfm."));
    };

    private static boolean startsWithMedia(String name) {
        for (String prefix : MEDIA_PREFIXES) {
            if (name.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code LFM3Model.set_gguf_parameters} (lfm3.py:82-99): the generic text-model metadata over the fixed-up
     * configuration, then the head width as the rotary width, the per-layer sliding-window pattern, the window and the
     * short-convolution cache.
     *
     * @throws GgufFormatException when a required key is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (long) cfg.requireInt("head_dim"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, slidingPattern(cfg));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, (long) cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.SHORTCONV_L_CACHE, (long) cfg.requireInt("conv_L_cache"));
    }

    /**
     * {@code LFM3D1Model.set_gguf_parameters} (lfm3.py:159-176): the LFM3 parameters after the served d1 codes and
     * attention layout check, the bidirectional attention, the {@code lfm3-d1}/{@code lfm3-d1-omni} decision type from
     * the checkpoint's own {@code model_type}, and the calibration's shared temperature for the three option kinds.
     *
     * @throws GgufFormatException when the checkpoint records another d1 layout or per-bucket temperatures
     * @throws IOException         when {@code calibration.json} cannot be read
     */
    public static void d1Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        if (cfg.findInt(-1, "d1_codes") != (int) D1_CODES) {
            throw new GgufFormatException(cfg.source() + ": unsupported d1 checkpoint: d1_codes = "
                    + cfg.flat().get("d1_codes") + ", expected " + D1_CODES + " (lfm3.py:151)");
        }
        if (!D1_ATTENTION.equals(cfg.findStringOr(null, "d1_attention"))) {
            throw new GgufFormatException(cfg.source() + ": unsupported d1 checkpoint: d1_attention = "
                    + cfg.flat().get("d1_attention") + ", expected '" + D1_ATTENTION + "' (lfm3.py:151)");
        }
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        kv.set(GgufKeys.DECISION_TYPE, d1Type(cfg));

        Map<String, Object> calibration = HfDecisionSupport.JsonObjects.readJsonObject(
                cfg.sideFile("calibration.json"));
        Object perBucket = calibration.get("temperatures");
        if (truthy(perBucket)) {
            throw new GgufFormatException("calibration.json: unsupported d1 calibration: per-bucket temperatures");
        }
        Object temperature = calibration.get("default");
        if (!(temperature instanceof Number)) {
            throw new GgufFormatException("calibration.json: 'default' must be a number, got " + temperature);
        }
        for (String name : D1_CALIBRATION_TEMPERATURES) {
            kv.setAt(GgufKeys.DECISION_TEMPERATURE, ((Number) temperature).doubleValue(), name);
        }
    }

    /** {@code _D1_TYPES} (lfm3.py:123): the decision type of the checkpoint's {@code d1_family}. */
    private static String d1Type(HfConfig cfg) {
        String family = cfg.findStringOr(cfg.modelType(), "d1_family");
        if ("lfm3_vl".equals(family)) {
            return "lfm3-d1";
        }
        if ("lfm3_omni".equals(family)) {
            return "lfm3-d1-omni";
        }
        throw new GgufFormatException(cfg.source() + ": a d1 checkpoint must record model_type lfm3_vl or lfm3_omni,"
                + " got '" + family + "' (lfm3.py:123)");
    }

    private static List<?> layerTypes(HfConfig cfg) {
        Object types = cfg.flat().get("layer_types");
        if (!(types instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": 'layer_types' must be an array (lfm3.py:83), got "
                    + types);
        }
        return (List<?>) types;
    }

    /** The sliding-window flags of the layers: {@code sliding_attention} layers are the windowed ones. */
    private static List<Boolean> slidingPattern(HfConfig cfg) {
        List<?> types = layerTypes(cfg);
        List<Boolean> out = new ArrayList<>(types.size());
        for (Object type : types) {
            out.add(SLIDING_ATTENTION.equals(type));
        }
        return out;
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code LFM3Model.modify_tensors} (lfm3.py:114-119): the convolution weights squeeze to two dimensions, the rest
     * map through the generated pattern tables.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource s = hfName.contains("conv.conv") ? ElementTransforms.squeezeConv(src) : src;
        out.add(ctx.names().mapOrThrow(hfName), s);
    }

    /**
     * {@code LFM3Model.generate_extra_tensors} (lfm3.py:101-112): the learned {@code rope_freqs} factors of the
     * full-attention layers, 1.0 for the {@code int(partial * head_dim) / 2} rotated pairs and 1e30 for the rest.
     *
     * @throws GgufFormatException when the partial rotary factor or the head width is missing
     */
    public static void extra(HfLoadContext ctx, TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        int headDim = cfg.requireInt("head_dim");
        double partial = cfg.rope().global().partialRotaryFactor().orElse(1.0);
        RowSource freqs = RopeTensors.proportionalFreqs(headDim, partial);
        out.add(ctx.names().ggufName("ROPE_FREQS", -1, ".weight"), freqs);
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * {@code LFM3Model.set_vocab} (lfm3.py:66-80): the plain GPT-2 vocabulary plus the speech templates of the
     * Audio and Omni checkpoints, selected by the checkpoint's own {@code model_type} (the ASR prompt for audio, the
     * TTS prompt for audio and omni, its think block empty on omni).
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(gpt2(ctx, pre));
        String modelType = ctx.config().modelType();
        boolean audio = "lfm3_audio".equals(modelType);
        boolean omni = "lfm3_omni".equals(modelType);
        List<Map<String, Object>> templates = new ArrayList<>();
        if (audio) {
            templates.add(template("asr", ASR_TEMPLATE));
        }
        if (audio || omni) {
            templates.add(template("tts", ttsTemplate(omni)));
        }
        if (!templates.isEmpty()) {
            md.putAll(HfChatTemplates.toGguf(templates));
        }
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code LFM3D1Model.set_vocab} (lfm3.py:153-157): the plain GPT-2 vocabulary plus the d1 systemone prompt whose
     * answer opens after an empty think block.
     */
    public static Map<String, Object> d1Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(gpt2(ctx, pre));
        md.putAll(HfChatTemplates.toGguf(Collections.singletonList(
                template("systemone", HfSsmHybridHooksB.d1Systemone() + "<think></think>"))));
        return Collections.unmodifiableMap(md);
    }

    private static Map<String, Object> gpt2(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    private static Map<String, Object> template(String name, String body) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("name", name);
        out.put("template", body);
        return out;
    }

    /** {@code LFM3Model._asr_template} (lfm3.py:39-43). */
    private static final String ASR_TEMPLATE = "{{- bos_token -}}"
            + "{{- '<|im_start|>system\\nPerform live ASR.<|im_end|>\\n' -}}"
            + "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}";

    /**
     * {@code LFM3Model._tts_template} (lfm3.py:54-64): the language code is looked up in the model card's table,
     * anything else is taken as the language name; the voice is female unless the prompt says male.
     */
    private static String ttsTemplate(boolean think) {
        return "{%- set names = " + json(TTS_LANGS) + " -%}"
                + "{%- set code = (language | lower) if language is defined and language else 'en' -%}"
                + "{%- set lang = names[code] if code in names else language -%}"
                + "{%- set v = (voice | lower) if voice is defined and voice and (voice | lower) in ['female', 'male']"
                + " else 'female' -%}"
                + "{{- bos_token -}}"
                + "{{- '<|im_start|>system\\nPerform TTS in ' + lang + ' using a ' + v + ' voice.<|im_end|>\\n' -}}"
                + "{{- '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}"
                + "{{- '<|im_start|>assistant\\n" + (think ? "<think></think>" : "") + "<|audio_start|>' -}}";
    }

    /** The language table of the TTS prompt, in the converter's order (lfm3.py:46-51). */
    private static Map<String, String> ttsLangs() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("ar", "Arabic");
        m.put("da", "Danish");
        m.put("de", "German");
        m.put("el", "Greek");
        m.put("en", "American English");
        m.put("en-us", "American English");
        m.put("en-gb", "British English");
        m.put("fi", "Finnish");
        m.put("fr", "French");
        m.put("he", "Hebrew");
        m.put("it", "Italian");
        m.put("ja", "Japanese");
        m.put("ko", "Korean");
        m.put("nl", "Dutch");
        m.put("pl", "Polish");
        m.put("pt", "Portuguese");
        m.put("es", "Spanish");
        m.put("sv", "Swedish");
        m.put("tr", "Turkish");
        m.put("zh", "Chinese");
        return Collections.unmodifiableMap(m);
    }

    /** {@code json.dumps} of the language table: {@code {"ar": "Arabic", ...}}. */
    private static String json(Map<String, String> m) {
        StringBuilder sb = new StringBuilder("{");
        for (Map.Entry<String, String> e : m.entrySet()) {
            if (sb.length() > 1) {
                sb.append(", ");
            }
            sb.append('"').append(e.getKey()).append("\": \"").append(e.getValue()).append('"');
        }
        return sb.append('}').toString();
    }

    /** Python truthiness of a configuration value. */
    private static boolean truthy(Object value) {
        if (value == null) {
            return false;
        }
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue() != 0;
        }
        if (value instanceof String) {
            return !((String) value).isEmpty();
        }
        if (value instanceof List) {
            return !((List<?>) value).isEmpty();
        }
        return !(value instanceof Map) || !((Map<?, ?>) value).isEmpty();
    }
}
