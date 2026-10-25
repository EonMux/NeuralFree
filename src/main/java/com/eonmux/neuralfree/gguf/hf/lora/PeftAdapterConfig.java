package com.eonmux.neuralfree.gguf.hf.lora;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The {@code adapter_config.json} of a PEFT LoRA adapter directory, parsed leniently and validated against what
 * {@code convert_lora_to_gguf.py} loads (LC/convert_lora_to_gguf.py:357, :376-377 read the file; conversion constants
 * from the converter's parameter use). Immutable.
 *
 * <p>Every adapter below is a plain LoRA of two factor tensors, which is all an adapter GGUF can hold
 * (LC/convert_lora_to_gguf.py:499-502): the config therefore refuses every PEFT method or option that adds trainable
 * parameters beyond {@code lora_A} / {@code lora_B}. {@code rank_pattern} is accepted without validation, because the
 * ranks come from the factor shapes and not from the configuration; {@code alpha_pattern} is accepted only when every
 * value equals {@code lora_alpha}, so one plain alpha covers the adapter (LC/convert_lora_to_gguf.py:531 reads a
 * single {@code lparams["lora_alpha"]}).</p>
 */
public final class PeftAdapterConfig {

    /** The adapter configuration file of a PEFT directory. */
    public static final String ADAPTER_CONFIG_JSON = "adapter_config.json";

    /** The only {@code peft_type} an adapter GGUF holds (the {@code LoraModel} of the converter). */
    private static final String PEFT_TYPE_LORA = "lora";

    /** {@code init_lora_weights} values that only seed {@code lora_A} / {@code lora_B} and are accepted as-is. */
    private static final List<String> ACCEPTED_INIT_LORA_WEIGHTS = List.of("gaussian", "loftq", "olora");

    private final Path dir;
    private final Map<String, Object> values;
    private final int rank;
    private final double loraAlpha;
    private final boolean useRslora;
    private final String baseModelNameOrPath;
    private final List<Integer> aloraInvocationTokens;
    private final String invocationString;

    private PeftAdapterConfig(Path dir, Map<String, Object> values, int rank, double loraAlpha, boolean useRslora,
                              String baseModelNameOrPath, List<Integer> aloraInvocationTokens,
                              String invocationString) {
        this.dir = dir;
        this.values = values;
        this.rank = rank;
        this.loraAlpha = loraAlpha;
        this.useRslora = useRslora;
        this.baseModelNameOrPath = baseModelNameOrPath;
        this.aloraInvocationTokens = aloraInvocationTokens;
        this.invocationString = invocationString;
    }

    /**
     * Reads and validates {@code dir/adapter_config.json}.
     *
     * @throws IOException         when the file cannot be read or is invalid JSON
     * @throws GgufFormatException for a method or option an adapter GGUF cannot represent
     */
    public static PeftAdapterConfig read(Path dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Path file = dir.resolve(ADAPTER_CONFIG_JSON);
        if (!Files.isRegularFile(file)) {
            throw new IOException("Invalid PEFT adapter '" + dir + "': " + ADAPTER_CONFIG_JSON + " is missing");
        }
        Map<String, Object> values;
        Object parsed;
        try {
            parsed = Json.parse(file, Json.MAX_FILE_BYTES, Json.Limits.LENIENT);
        } catch (com.eonmux.neuralfree.utils.json.JsonException e) {
            throw new IOException("Invalid PEFT adapter '" + dir + "': " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IOException("Invalid PEFT adapter '" + dir + "': " + ADAPTER_CONFIG_JSON
                    + ": expected a JSON object");
        }
        // Json produces maps keyed by String; the copy keeps Json's mutability out of this config.
        values = new java.util.LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) parsed).entrySet()) {
            values.put(String.valueOf(e.getKey()), e.getValue());
        }
        requirePeftTypeLora(dir, values);
        requireNotRejected(dir, values);
        int rank = rankOf(dir, values);
        double alpha = alphaOf(dir, values);
        boolean rslora = boolOf(dir, values, "use_rslora", false);
        checkAlphaPattern(dir, values, alpha);
        return new PeftAdapterConfig(dir, Collections.unmodifiableMap(values), rank, alpha, rslora,
                stringOf(values, "base_model_name_or_path"), invocationTokensOf(dir, values),
                stringOf(values, "invocation_string"));
    }

    /** The adapter directory the configuration was read from (for messages). */
    public Path dir() {
        return dir;
    }

    /** The parsed configuration, unchanged (unmodifiable). */
    public Map<String, Object> values() {
        return values;
    }

    /** The rank {@code r} (converter {@code lparams["lora_alpha"]} counterpart; ranks are checked per tensor). */
    public int rank() {
        return rank;
    }

    /** {@code lora_alpha} as stored, without the rank-stabilized adjustment. */
    public double loraAlpha() {
        return loraAlpha;
    }

    /** True for a rank-stabilized adapter ({@code use_rslora}, scaled {@code lora_alpha / sqrt(r)}). */
    public boolean usesRslora() {
        return useRslora;
    }

    /**
     * The NeuralFree alpha: {@code lora_alpha} with {@code use_rslora}, {@code lora_alpha * sqrt(r)}, so that
     * NeuralFree's {@code alpha / r} equals the rank-stabilized scale {@code lora_alpha / sqrt(r)}.
     */
    public float alpha() {
        return (float) (useRslora ? loraAlpha * Math.sqrt(rank) : loraAlpha);
    }

    /** {@code base_model_name_or_path}, or null when the configuration does not name a base. */
    public String baseModelNameOrPath() {
        return baseModelNameOrPath;
    }

    /** {@code alora_invocation_tokens}, or null when absent (unmodifiable). */
    public List<Integer> aloraInvocationTokens() {
        return aloraInvocationTokens == null ? null : Collections.unmodifiableList(aloraInvocationTokens);
    }

    /** {@code invocation_string}, or null when absent. */
    public String invocationString() {
        return invocationString;
    }

    @Override
    public String toString() {
        return "PeftAdapterConfig{" + dir + ", " + ADAPTER_CONFIG_JSON + "}";
    }

    // ------------------------------------------------------------------ validation

    private static void requirePeftTypeLora(Path dir, Map<String, Object> values) {
        Object type = values.get("peft_type");
        if (type == null) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': " + ADAPTER_CONFIG_JSON
                    + " has no peft_type; convert_lora_to_gguf.py loads LoRA adapters only");
        }
        if (!(type instanceof String) || !PEFT_TYPE_LORA.equalsIgnoreCase(String.valueOf(type))) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': peft_type '" + type
                    + "' is not supported; convert_lora_to_gguf.py loads LoRA adapters only");
        }
    }

    /** The options an adapter GGUF cannot represent; each refusal names the value. */
    private static void requireNotRejected(Path dir, Map<String, Object> values) {
        String config = ADAPTER_CONFIG_JSON;
        if (boolOf(dir, values, "use_dora", false)) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': " + config + " asks use_dora;"
                    + " a DoRA delta is a magnitude scaling, not a B @ A LoRA delta");
        }
        Object save = values.get("modules_to_save");
        if (nonEmpty(save)) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': " + config + " has"
                    + " " + describe(save) + " modules_to_save; whole-module copies are not LoRA factors");
        }
        Object replication = values.get("layer_replication");
        if (nonEmpty(replication)) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': " + config + " has"
                    + " " + describe(replication) + " layer_replication; the factor shapes do not follow it");
        }
        if (boolOf(dir, values, "lora_bias", false)) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': " + config + " sets lora_bias true;"
                    + " trained LoRA biases are parameters a B @ A delta does not carry");
        }
        requireInit(dir, values);
    }

    /** {@code init_lora_weights}: values that imply parameters besides {@code lora_A} / {@code lora_B} are refused. */
    private static void requireInit(Path dir, Map<String, Object> values) {
        Object init = values.get("init_lora_weights");
        if (init == null || init instanceof Boolean) {
            return;
        }
        String name = String.valueOf(init);
        String lower = name.toLowerCase(java.util.Locale.ROOT);
        if (ACCEPTED_INIT_LORA_WEIGHTS.stream().anyMatch(lower::startsWith)) {
            return;
        }
        throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': init_lora_weights '" + init
                + "' implies parameters besides LoRA's lora_A and lora_B; convert_lora_to_gguf.py loads"
                + " plain LoRA adapters only");
    }

    /** {@code alpha_pattern} values must all equal {@code lora_alpha}, so a single {@code adapter.lora.alpha} holds. */
    private static void checkAlphaPattern(Path dir, Map<String, Object> values, double alpha) {
        Object pattern = values.get("alpha_pattern");
        if (!nonEmpty(pattern)) {
            return;
        }
        if (!(pattern instanceof Map)) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': alpha_pattern must be an object"
                    + " of number values, got " + pattern.getClass().getSimpleName());
        }
        for (Map.Entry<?, ?> e : ((Map<?, ?>) pattern).entrySet()) {
            if (!(e.getValue() instanceof Number)
                    || !nearly(((Number) e.getValue()).doubleValue(), alpha)) {
                throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': alpha_pattern["
                        + e.getKey() + "] = " + e.getValue() + " differs from lora_alpha " + alpha
                        + "; the converter writes a single adapter.lora.alpha");
            }
        }
    }

    private static int rankOf(Path dir, Map<String, Object> values) {
        int rank = intOf(dir, values, "r", "lora_alpha needs a numeric r");
        if (rank <= 0) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': r = " + rank
                    + " is not a positive rank");
        }
        return rank;
    }

    private static double alphaOf(Path dir, Map<String, Object> values) {
        Object v = values.get("lora_alpha");
        if (!(v instanceof Number) || !Double.isFinite(((Number) v).doubleValue())) {
            throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': lora_alpha needs a finite number,"
                    + " got " + (v == null ? "nothing" : describe(v)));
        }
        return ((Number) v).doubleValue();
    }

    /** {@code alora_invocation_tokens}: every entry fits an unsigned 32-bit token id. */
    private static List<Integer> invocationTokensOf(Path dir, Map<String, Object> values) {
        Object v = values.get("alora_invocation_tokens");
        if (nonEmpty(v)) {
            List<Integer> out;
            if (v instanceof List) {
                out = new ArrayList<>();
                for (Object o : (List<?>) v) {
                    if (!(o instanceof Number) || ((Number) o).doubleValue() < 0
                            || ((Number) o).longValue() > 0xFFFF_FFFFL) {
                        throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': alora_invocation_tokens"
                                + " needs unsigned 32-bit token ids, got " + describe(o));
                    }
                    out.add(((Number) o).intValue());
                }
            } else {
                throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': alora_invocation_tokens"
                        + " must be an array of token ids, got " + describe(v));
            }
            return out;
        }
        return null;
    }

    // ------------------------------------------------------------------ scalar reading

    private static String stringOf(Map<String, Object> values, String key) {
        Object v = values.get(key);
        return v instanceof String ? (String) v : null;
    }

    private static int intOf(Path dir, Map<String, Object> values, String key, String message) {
        Object v = values.get(key);
        if (v instanceof Number && Double.isFinite(((Number) v).doubleValue())
                && ((Number) v).doubleValue() == Math.floor(((Number) v).doubleValue())) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': " + message + ", got "
                + (v == null ? "nothing" : describe(v)));
    }

    private static boolean boolOf(Path dir, Map<String, Object> values, String key, boolean dflt) {
        Object v = values.get(key);
        if (v == null || !(v instanceof Boolean)) {
            if (v != null) {
                throw new GgufFormatException("Invalid PEFT adapter '" + dir + "': " + key
                        + " must be a boolean, got " + describe(v));
            }
            return dflt;
        }
        return (Boolean) v;
    }

    private static boolean nonEmpty(Object v) {
        if (v == null) {
            return false;
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        if (v instanceof Map) {
            return !((Map<?, ?>) v).isEmpty();
        }
        return true;
    }

    private static String describe(Object v) {
        return v == null ? "nothing" : (v instanceof List ? "an array"
                : v instanceof Map ? "an object" : "'" + v + "'");
    }

    private static boolean nearly(double a, double b) {
        return a == b || Math.abs(a - b) <= Math.max(Math.abs(a), Math.abs(b)) * 1e-6;
    }
}
