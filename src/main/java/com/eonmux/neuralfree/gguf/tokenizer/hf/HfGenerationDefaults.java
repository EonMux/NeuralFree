package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The model's default generation settings from {@code generation_config.json} as GGUF metadata.
 *
 * <p>The sampling keys are those llama.cpp's {@code Metadata.load} copies (gguf-py/gguf/metadata.py:76-89) with the
 * writer's types (gguf_writer.py:530-564): {@code sequence}, {@code top_k}, {@code top_p}, {@code min_p},
 * {@code xtc_probability}, {@code xtc_threshold}, {@code temperature} as {@code temp}, {@code penalty_last_n},
 * {@code penalty_repeat} (else {@code repetition_penalty}), {@code mirostat}, {@code mirostat_tau} and
 * {@code mirostat_eta}, each written when present and not null. NeuralFree adds {@code presence_penalty},
 * {@code frequency_penalty} and {@code max_new_tokens}, which vLLM applies as model defaults, the
 * {@code eos_token_id} ids of generation_config.json and config.json (or its {@code text_config}) as extra
 * end-of-generation ids, as SGLang unions them, and {@code suppress_tokens}. {@code do_sample} is ignored, as vLLM
 * and llama.cpp ignore it. A value of the wrong type is logged and skipped.</p>
 */
public final class HfGenerationDefaults {

    private static final Logger LOGGER = LogManager.getLogger(HfGenerationDefaults.class);

    private static final String EOS_TOKEN_ID = "eos_token_id";

    private HfGenerationDefaults() {}

    /**
     * The generation defaults of {@code config}'s checkpoint.
     *
     * @param config     the model configuration, with its generation_config.json when present
     * @param tokenCount the token count of the vocabulary; ids outside it are dropped, and without a vocabulary no
     *                   token ids are written
     */
    public static Map<String, Object> of(HfConfig config, OptionalInt tokenCount) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(tokenCount, "tokenCount");
        JsonView generation = config.generation().orElse(null);
        return of(generation, config.raw(), tokenCount);
    }

    /**
     * The generation defaults of a generation_config.json object (null when absent) and a config.json root.
     *
     * @param tokenCount the token count of the vocabulary; ids outside it are dropped
     */
    public static Map<String, Object> of(JsonView generation, JsonView config, OptionalInt tokenCount) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(tokenCount, "tokenCount");
        Map<String, Object> out = new LinkedHashMap<>();
        if (generation != null) sampling(generation, out);
        if (tokenCount.isPresent()) {
            int n = tokenCount.getAsInt();
            Set<Integer> eog = new LinkedHashSet<>();
            if (generation != null) addIds(generation, EOS_TOKEN_ID, n, eog);
            addIds(config, EOS_TOKEN_ID, n, eog);
            Object textConfig = config.raw("text_config");
            if (config.raw(EOS_TOKEN_ID) == null && textConfig instanceof Map) {
                addIds(JsonView.of(textConfig, config.source(), "text_config"), EOS_TOKEN_ID, n, eog);
            }
            if (!eog.isEmpty()) {
                out.put(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS, Collections.unmodifiableList(new ArrayList<>(eog)));
            }
            if (generation != null && generation.raw("suppress_tokens") != null) {
                Set<Integer> suppress = new LinkedHashSet<>();
                addIds(generation, "suppress_tokens", n, suppress);
                out.put(GgufKeys.TOKENIZER_SUPPRESS_TOKENS, Collections.unmodifiableList(new ArrayList<>(suppress)));
            }
        }
        return Collections.unmodifiableMap(out);
    }

    private static void sampling(JsonView g, Map<String, Object> out) {
        string(g, "sequence", GgufKeys.GENERAL_SAMPLING_SEQUENCE, out);
        int32(g, "top_k", GgufKeys.GENERAL_SAMPLING_TOP_K, out);
        float32(g, "top_p", GgufKeys.GENERAL_SAMPLING_TOP_P, out);
        float32(g, "min_p", GgufKeys.GENERAL_SAMPLING_MIN_P, out);
        float32(g, "xtc_probability", GgufKeys.GENERAL_SAMPLING_XTC_PROBABILITY, out);
        float32(g, "xtc_threshold", GgufKeys.GENERAL_SAMPLING_XTC_THRESHOLD, out);
        float32(g, "temperature", GgufKeys.GENERAL_SAMPLING_TEMP, out);
        int32(g, "penalty_last_n", GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N, out);
        float32(g, g.has("penalty_repeat") ? "penalty_repeat" : "repetition_penalty",
                GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT, out);
        int32(g, "mirostat", GgufKeys.GENERAL_SAMPLING_MIROSTAT, out);
        float32(g, "mirostat_tau", GgufKeys.GENERAL_SAMPLING_MIROSTAT_TAU, out);
        float32(g, "mirostat_eta", GgufKeys.GENERAL_SAMPLING_MIROSTAT_ETA, out);
        float32(g, "presence_penalty", GgufKeys.NF_SAMPLING_PRESENCE_PENALTY, out);
        float32(g, "frequency_penalty", GgufKeys.NF_SAMPLING_FREQUENCY_PENALTY, out);
        int32(g, "max_new_tokens", GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, out);
        if (g.raw("do_sample") != null) LOGGER.debug("{} is ignored", g.where("do_sample"));
    }

    private static void string(JsonView g, String field, String key, Map<String, Object> out) {
        Object v = g.raw(field);
        if (v == null) return;
        if (v instanceof String) {
            out.put(key, v);
        } else {
            skip(g, field, v, "a string");
        }
    }

    private static void int32(JsonView g, String field, String key, Map<String, Object> out) {
        Object v = g.raw(field);
        if (v == null) return;
        Long value = integral(v);
        if (value != null && value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE) {
            out.put(key, value.intValue());
        } else {
            skip(g, field, v, "a 32-bit integer");
        }
    }

    private static void float32(JsonView g, String field, String key, Map<String, Object> out) {
        Object v = g.raw(field);
        if (v == null) return;
        if (v instanceof Number && !(v instanceof BigInteger) && Double.isFinite(((Number) v).doubleValue())) {
            out.put(key, ((Number) v).floatValue());
        } else {
            skip(g, field, v, "a finite number");
        }
    }

    /** A JSON integer, or a float with an integral value, as a long; else null. */
    private static Long integral(Object v) {
        if (v instanceof Long || v instanceof Integer) return ((Number) v).longValue();
        if (v instanceof Double) {
            double d = (Double) v;
            if (d == Math.rint(d) && Math.abs(d) <= Integer.MAX_VALUE) return (long) d;
        }
        return null;
    }

    private static void skip(JsonView g, String field, Object v, String expected) {
        LOGGER.warn("{} is {} ({}), not {}; it is ignored", g.where(field), v, v.getClass().getSimpleName(), expected);
    }

    /** Adds the ids of {@code field} (an integer or a list of integers) that are in {@code [0, n)}. */
    private static void addIds(JsonView v, String field, int n, Set<Integer> out) {
        Object raw = v.raw(field);
        if (raw == null) return;
        List<?> items = raw instanceof List ? (List<?>) raw : Collections.singletonList(raw);
        for (Object item : items) {
            Long id = integral(item);
            if (id == null) {
                LOGGER.warn("{} holds {}, which is not a token id; it is ignored", v.where(field), item);
            } else if (id < 0 || id >= n) {
                LOGGER.warn("{} id {} is outside the vocabulary [0, {}); it is ignored", v.where(field), id, n);
            } else {
                out.add(id.intValue());
            }
        }
    }
}
