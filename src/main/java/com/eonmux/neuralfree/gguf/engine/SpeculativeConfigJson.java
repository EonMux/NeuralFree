package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.SuffixDrafter;
import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * vLLM's {@code --speculative-config} JSON object (vllm/config/speculative.py {@code SpeculativeConfig}) restated as
 * the speculative decoding flags the launchers parse, so the JSON sets the same {@link EngineConfig} and
 * {@link SpeculativeOptions} fields the flags do: {@code LLMDemo} applies the settings as its own flags and the
 * server as its launch flags, each refusing what it does not run, naming the key. Stateless.
 *
 * <p>Keys (a JSON {@code null} is an absent key, as vLLM's {@code None}; absent keys keep the configuration):
 * <ul>
 *   <li>{@code method}: {@code ngram} (also {@code [ngram]}) is {@code --spec-type ngram-simple}, {@code suffix}
 *       is {@code suffix}, {@code draft_model} is {@code draft-simple}, {@code mtp} and vLLM's
 *       model-specific MTP names ({@code deepseek_mtp}, {@code qwen3_next_mtp}, ...) are {@code draft-mtp}, and
 *       {@code eagle}, {@code eagle3} and {@code dflash} are {@code auto}, the draft file deciding its route.
 *       Without a method, a {@code model} of {@code ngram} means {@code ngram}, any other model {@code auto}.</li>
 *   <li>{@code model}: the draft file ({@code --spec-draft-model}); needed by {@code draft_model}, {@code eagle},
 *       {@code eagle3} and {@code dflash}, refused beside {@code ngram} and {@code suffix}.</li>
 *   <li>{@code num_speculative_tokens}: {@code --spec-draft-n-max}. For {@code suffix} it defaults to the tree
 *       depth, as vLLM's does.</li>
 *   <li>{@code prompt_lookup_max} / {@code prompt_lookup_min}: {@code --spec-ngram-simple-size-n} /
 *       {@code --spec-ngram-simple-size-n-min}; one of them alone sets both, as vLLM resolves it.</li>
 *   <li>{@code suffix_decoding_max_tree_depth}, {@code _max_cached_requests}, {@code _max_spec_factor},
 *       {@code _min_token_prob}: the {@code --spec-suffix-*} limits.</li>
 *   <li>{@code rejection_sample_method} ({@code standard} is {@code rejection}), {@code synthetic_acceptance_rates},
 *       {@code synthetic_acceptance_length}, {@code fly_window_size}, {@code fly_entropy_threshold},
 *       {@code fly_entropy_top_k}: the verification flags.</li>
 *   <li>{@code use_heterogeneous_vocab}: {@code --spec-draft-heterogeneous-vocab} ({@code --no-...} for false),
 *       which vLLM accepts only with {@code method='draft_model'}.</li>
 *   <li>{@code draft_lm_head_quantization}: the derived candidate head of a {@code dflash} drafter at a lower
 *       precision ({@code --draft-head-prec}); vLLM's {@code fp8} and {@code nvfp4} and NeuralFree's {@code q8_0}
 *       translate.</li>
 * </ul>
 * {@code draft_tensor_parallel_size} and {@code tensor_parallel_size} are refused with their reason, the methods
 * NeuralFree does not run ({@code ngram_gpu}, {@code custom_class}, {@code dspark}, {@code extract_hidden_states})
 * by name, and every other key as unsupported.
 */
public final class SpeculativeConfigJson {

    /** The launch flag that carries the JSON, vLLM's spelling. */
    public static final String FLAG = "--speculative-config";

    /**
     * One translated setting: the JSON key and value it comes from, the flag and the flag's value. Immutable.
     */
    public static final class Setting {
        private final String key;
        private final String source;
        private final String flag;
        private final String value;

        Setting(String key, String source, String flag, String value) {
            this.key = key;
            this.source = source;
            this.flag = flag;
            this.value = value;
        }

        Setting(String key, String flag, String value) {
            this(key, value, flag, value);
        }

        /** The JSON key the setting comes from. */
        public String key() { return key; }

        /** The JSON value the setting comes from, as text. */
        public String source() { return source; }

        /** The flag that applies the setting. */
        public String flag() { return flag; }

        /** The flag's value. */
        public String value() { return value; }

        /**
         * {@code failure} restated naming the JSON key: {@code --speculative-config key=source: failure}.
         */
        public IllegalArgumentException refusal(String failure) {
            return new IllegalArgumentException(FLAG + " " + key + "=" + source + ": " + failure);
        }

        @Override
        public String toString() {
            return key + "=" + source + " -> " + flag + " " + value;
        }
    }

    private static final List<String> KEYS = Collections.unmodifiableList(Arrays.asList("method", "model",
            "num_speculative_tokens", "prompt_lookup_max", "prompt_lookup_min", "suffix_decoding_max_tree_depth",
            "suffix_decoding_max_cached_requests", "suffix_decoding_max_spec_factor", "suffix_decoding_min_token_prob",
            "rejection_sample_method", "synthetic_acceptance_rates", "synthetic_acceptance_length",
            "fly_window_size", "fly_entropy_threshold", "fly_entropy_top_k", "use_heterogeneous_vocab",
            "draft_lm_head_quantization"));

    /** vLLM's model-specific MTP method names, which it maps to {@code mtp} (vllm/config/speculative.py). */
    private static final Set<String> MTP_METHODS = new HashSet<>(Arrays.asList("mtp", "deepseek_mtp",
            "dots3_note_mtp", "mimo_mtp", "mimo_v2_mtp", "glm4_moe_mtp", "glm4_moe_lite_mtp", "glm_ocr_mtp",
            "ernie_mtp", "nemotron_h_mtp", "nemotron3_5_transcribe_mtp", "exaone_moe_mtp", "exaone4_5_mtp",
            "qwen3_next_mtp", "qwen4_exp_mtp", "qwen3_5_mtp", "longcat_flash_mtp", "bailing_hybrid_v3_mtp",
            "minimax_m3_mtp", "bailing_hybrid_mtp", "kimi_k3_mtp", "pangu_ultra_moe_mtp", "step3p5_mtp",
            "step5_mtp", "hy_v3_mtp", "hy_v4_mtp", "gemma4_mtp", "inkling_mtp", "glm5_next_mtp"));

    /** The methods that draft from a draft file whose architecture decides the route. */
    private static final Set<String> FILE_METHODS = new HashSet<>(Arrays.asList("eagle", "eagle3", "dflash"));

    private SpeculativeConfigJson() {
    }

    /**
     * The settings of a {@code --speculative-config} JSON object, in the order the flags apply.
     *
     * @throws IllegalArgumentException naming {@value #FLAG} and the key for text that is not a JSON object, an
     *                                  unsupported key or method, a value of the wrong type, or a method without
     *                                  the model it needs
     */
    public static List<Setting> parse(String json) {
        Map<String, Object> config = object(Objects.requireNonNull(json, "json"));
        for (String key : config.keySet()) {
            refuseKey(key);
        }
        List<Setting> out = new ArrayList<>();
        String method = method(config);
        String type = draftType(method, config);
        out.add(new Setting("method", method == null ? type : method, "--spec-type", type));
        String model = string(config, "model");
        if (model != null && !isNgramModel(method, model)) {
            out.add(new Setting("model", "--spec-draft-model", model));
        }
        draftMax(config, type, out);
        promptLookup(config, out);
        add(out, config, "suffix_decoding_max_tree_depth", "--spec-suffix-max-tree-depth", true);
        add(out, config, "suffix_decoding_max_cached_requests", "--spec-suffix-max-cached-requests", true);
        add(out, config, "suffix_decoding_max_spec_factor", "--spec-suffix-max-spec-factor", false);
        add(out, config, "suffix_decoding_min_token_prob", "--spec-suffix-min-token-prob", false);
        heterogeneousVocab(config, method, out);
        draftHeadQuantization(config, out);
        verification(config, out);
        return Collections.unmodifiableList(out);
    }

    /**
     * {@code draft_lm_head_quantization}: the derived candidate head of a {@code dflash} drafter at a lower
     * precision, stored with {@code --draft-head-prec}.
     */
    private static void draftHeadQuantization(Map<String, Object> config, List<Setting> out) {
        Object value = config.get("draft_lm_head_quantization");
        if (value == null) {
            return;
        }
        DraftHeadOptions.Prec prec = DraftHeadOptions.Prec.of(FLAG + " draft_lm_head_quantization",
                value.toString());
        out.add(new Setting("draft_lm_head_quantization", "--draft-head-prec",
                prec.name().toLowerCase(Locale.ROOT)));
    }

    /**
     * {@code use_heterogeneous_vocab}: the enable (or disable) flag, refused beside a method other than
     * {@code draft_model} as vLLM refuses it.
     */
    private static void heterogeneousVocab(Map<String, Object> config, String method, List<Setting> out) {
        Object value = config.get("use_heterogeneous_vocab");
        if (value == null) {
            return;
        }
        if (!(value instanceof Boolean)) {
            throw new IllegalArgumentException(FLAG + " use_heterogeneous_vocab must be a boolean, got " + value);
        }
        boolean enabled = (Boolean) value;
        if (enabled && method != null && !method.equals("draft_model")) {
            throw new IllegalArgumentException(FLAG + " use_heterogeneous_vocab requires method='draft_model', but"
                    + " the method is " + method);
        }
        out.add(new Setting("use_heterogeneous_vocab", value.toString(),
                enabled ? "--spec-draft-heterogeneous-vocab" : "--no-spec-draft-heterogeneous-vocab",
                value.toString()));
    }

    /** The JSON object of {@code json}, its keys as strings. */
    private static Map<String, Object> object(String json) {
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(FLAG + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException(FLAG + " must be a JSON object, got "
                    + (parsed == null ? "null" : parsed.getClass().getSimpleName()));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) parsed).entrySet()) {
            if (e.getValue() != null) {
                out.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        return out;
    }

    private static void refuseKey(String key) {
        if (KEYS.contains(key)) {
            return;
        }
        if (key.equals("draft_tensor_parallel_size")) {
            throw new IllegalArgumentException(FLAG + " key 'draft_tensor_parallel_size' is not supported: the"
                    + " draft model runs in the target's engine without tensor parallelism");
        }
        if (key.equals("tensor_parallel_size")) {
            throw new IllegalArgumentException(FLAG + " key 'tensor_parallel_size' is not a speculative config"
                    + " key (vLLM refuses it too; it names draft_tensor_parallel_size, which is not supported)");
        }
        throw new IllegalArgumentException(FLAG + " key '" + key + "' is not supported; the supported keys are "
                + String.join(", ", KEYS));
    }

    /**
     * The method, lower-cased; absent, {@code ngram} for vLLM's {@code model: "ngram"} and null for a draft file,
     * whose architecture decides the route.
     */
    private static String method(Map<String, Object> config) {
        String method = string(config, "method");
        if (method != null) {
            return method.trim().toLowerCase(Locale.ROOT);
        }
        String model = string(config, "model");
        if (model == null) {
            throw new IllegalArgumentException(FLAG + " needs a method or a model");
        }
        return isNgramModel(null, model) ? "ngram" : null;
    }

    /** vLLM's {@code model: "ngram"} (or {@code "[ngram]"}) that names the n-gram method, not a draft file. */
    private static boolean isNgramModel(String method, String model) {
        String m = model.trim();
        return (method == null || method.equals("ngram") || method.equals("[ngram]"))
                && (m.equals("ngram") || m.equals("[ngram]"));
    }

    /**
     * The {@code --spec-type} of {@code method} (null: a draft file of any route), refusing the methods NeuralFree
     * does not run and a model where the method needs none or lacks one.
     */
    private static String draftType(String method, Map<String, Object> config) {
        String model = string(config, "model");
        boolean hasFile = model != null && !isNgramModel(method, model);
        if (method == null || FILE_METHODS.contains(method)) {
            return requireFile(method == null ? "auto" : method, hasFile, "auto");
        }
        if (method.equals("draft_model")) {
            return requireFile(method, hasFile, "draft-simple");
        }
        if (MTP_METHODS.contains(method)) {
            // a draft file holds the MTP blocks; without one the target's own NextN / MTP blocks draft
            return "draft-mtp";
        }
        String type;
        if (method.equals("ngram") || method.equals("[ngram]")) {
            type = "ngram-simple";
        } else if (method.equals("suffix")) {
            type = "suffix";
        } else {
            throw unsupportedMethod(method);
        }
        if (hasFile) {
            throw new IllegalArgumentException(FLAG + " method " + method + " drafts without a model; remove"
                    + " model=" + model);
        }
        return type;
    }

    private static String requireFile(String method, boolean hasFile, String type) {
        if (!hasFile) {
            throw new IllegalArgumentException(FLAG + " method " + method + " needs a model (the draft file)");
        }
        return type;
    }

    private static IllegalArgumentException unsupportedMethod(String method) {
        if (method.equals("ngram_gpu")) {
            return new IllegalArgumentException(FLAG + " method ngram_gpu is not supported: n-gram lookup runs on"
                    + " the host; use method ngram");
        }
        if (method.equals("custom_class") || method.equals("dspark") || method.equals("extract_hidden_states")) {
            return new IllegalArgumentException(FLAG + " method " + method + " is not supported");
        }
        return new IllegalArgumentException(FLAG + " method '" + method + "' is unknown; use ngram, suffix,"
                + " draft_model, mtp (or a model's MTP name), eagle, eagle3 or dflash");
    }

    /** {@code num_speculative_tokens}, or for {@code suffix} the tree depth vLLM defaults it to. */
    private static void draftMax(Map<String, Object> config, String type, List<Setting> out) {
        if (config.containsKey("num_speculative_tokens")) {
            add(out, config, "num_speculative_tokens", "--spec-draft-n-max", true);
        } else if (type.equals("suffix")) {
            String depth = config.containsKey("suffix_decoding_max_tree_depth")
                    ? integer(config, "suffix_decoding_max_tree_depth")
                    : Integer.toString(SuffixDrafter.DEFAULT_MAX_TREE_DEPTH);
            out.add(new Setting("num_speculative_tokens", "--spec-draft-n-max", depth));
        }
    }

    /** {@code prompt_lookup_max} / {@code _min}; one alone sets both (vLLM's {@code NgramSpeculatorConfig}). */
    private static void promptLookup(Map<String, Object> config, List<Setting> out) {
        boolean hasMax = config.containsKey("prompt_lookup_max");
        boolean hasMin = config.containsKey("prompt_lookup_min");
        if (!hasMax && !hasMin) {
            return;
        }
        String max = integer(config, hasMax ? "prompt_lookup_max" : "prompt_lookup_min");
        String min = integer(config, hasMin ? "prompt_lookup_min" : "prompt_lookup_max");
        if (Long.parseLong(min) > Long.parseLong(max)) {
            throw new IllegalArgumentException(FLAG + " prompt_lookup_min=" + min + " must be <= prompt_lookup_max="
                    + max);
        }
        out.add(new Setting(hasMax ? "prompt_lookup_max" : "prompt_lookup_min", "--spec-ngram-simple-size-n", max));
        out.add(new Setting(hasMin ? "prompt_lookup_min" : "prompt_lookup_max", "--spec-ngram-simple-size-n-min",
                min));
    }

    private static void verification(Map<String, Object> config, List<Setting> out) {
        String method = string(config, "rejection_sample_method");
        if (method != null) {
            String m = method.trim().toLowerCase(Locale.ROOT);
            out.add(new Setting("rejection_sample_method", method, "--rejection-sample-method",
                    m.equals("standard") ? "rejection" : m));
        }
        if (config.containsKey("synthetic_acceptance_rates")) {
            out.add(new Setting("synthetic_acceptance_rates", "--spec-synth-rates", rates(config)));
        }
        add(out, config, "synthetic_acceptance_length", "--spec-synth-len", false);
        add(out, config, "fly_window_size", "--spec-fly-window", true);
        add(out, config, "fly_entropy_threshold", "--spec-fly-entropy-threshold", false);
        add(out, config, "fly_entropy_top_k", "--spec-fly-entropy-top-k", true);
    }

    private static void add(List<Setting> out, Map<String, Object> config, String key, String flag,
                            boolean integral) {
        if (config.containsKey(key)) {
            out.add(new Setting(key, flag, integral ? integer(config, key) : number(config, key)));
        }
    }

    private static String string(Map<String, Object> config, String key) {
        Object v = config.get(key);
        if (v == null) {
            return null;
        }
        if (!(v instanceof String)) {
            throw new IllegalArgumentException(FLAG + " " + key + " must be a string, got " + v);
        }
        return (String) v;
    }

    private static String integer(Map<String, Object> config, String key) {
        Object v = config.get(key);
        if (!(v instanceof Long)) {
            throw new IllegalArgumentException(FLAG + " " + key + " must be an integer, got " + v);
        }
        return v.toString();
    }

    private static String number(Map<String, Object> config, String key) {
        Object v = config.get(key);
        if (!(v instanceof Long) && !(v instanceof Double)) {
            throw new IllegalArgumentException(FLAG + " " + key + " must be a number, got " + v);
        }
        return v.toString();
    }

    /** {@code synthetic_acceptance_rates}: a non-empty array of numbers, joined as {@code --spec-synth-rates}. */
    private static String rates(Map<String, Object> config) {
        Object v = config.get("synthetic_acceptance_rates");
        if (!(v instanceof List) || ((List<?>) v).isEmpty()) {
            throw new IllegalArgumentException(FLAG + " synthetic_acceptance_rates must be a non-empty array of"
                    + " numbers, got " + v);
        }
        StringBuilder sb = new StringBuilder();
        for (Object rate : (List<?>) v) {
            if (!(rate instanceof Long) && !(rate instanceof Double)) {
                throw new IllegalArgumentException(FLAG + " synthetic_acceptance_rates holds " + rate
                        + ", not a number");
            }
            sb.append(sb.length() == 0 ? "" : ",").append(rate);
        }
        return sb.toString();
    }
}
