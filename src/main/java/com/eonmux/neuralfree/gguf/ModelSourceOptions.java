package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.load.MetadataOverrides;
import com.eonmux.neuralfree.gguf.load.RopeOverrides;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.math.BigInteger;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**
 * How a model is opened: how a checkpoint that is not stored as GGUF is turned into its GGUF view
 * ({@link ModelSources}), and the loading extras RX applies to every source format and engine as metadata
 * overrides of the view, tensor checks, dummy weights, rope overrides and the warmup decode. GGUF files are read
 * as stored except for those extras. Read from system properties (absent keys keep their defaults):
 * <ul>
 *   <li>{@value #PROP_MATERIALIZE}: auto | memory | disk | none, where derived tensors live; default auto.</li>
 *   <li>{@value #PROP_MEMORY_MAX}: {@code <n>[K|M|G]} bytes of derived tensors auto keeps in direct memory; default
 *       2G.</li>
 *   <li>{@value #PROP_CACHE_DIR}: parent of the per-process directories of derived tensors written to disk; default
 *       {@code ${user.home}/.cache/neuralfree/hf}.</li>
 *   <li>{@value #PROP_OUTTYPE}: source | auto | f32 | f16 | bf16 | q8_0, the type of float tensors (the converter's
 *       {@code --outtype}, with source keeping each tensor's stored type); default source.</li>
 *   <li>{@value #PROP_ONLINE_QUANT}: none | q8_0 | q4_0 | q4_1 | q5_0 | q5_1 | iq4_nl, online quantization of float
 *       matmul weights; default none.</li>
 *   <li>{@value #PROP_EXACT_QUANT}: prefer bit-exact weight types over close repacks; default false.</li>
 *   <li>{@value #PROP_MTP}: keep MTP / NextN layers (the converter's {@code --no-mtp} sets false); default true.</li>
 *   <li>{@value #PROP_ST_DENSE}: include sentence-transformers dense heads; default true.</li>
 *   <li>{@value #PROP_OVERRIDES}: a JSON object merged into {@code config.json} before conversion; default none.</li>
 *   <li>{@value #PROP_TOKENIZER}: a directory whose tokenizer files replace the model's, or a GGUF file whose
 *       {@code tokenizer.*} keys do; default none.</li>
 *   <li>{@value #PROP_LORA_BASE_CONFIG}: the Hugging Face configuration of a LoRA adapter's base when the base is a
 *       GGUF file; default none.</li>
 *   <li>{@value #PROP_BASE_DIR}: a local copy of the base model a decision-LoRA checkpoint merges at load; default
 *       none.</li>
 *   <li>{@value #PROP_OVERRIDE_KV}: the {@code --override-kv} metadata overrides as {@code key=type:value} entries
 *       in one string separated by {@code ;}; default none.</li>
 *   <li>{@code neuralfree.rope.*} and {@code neuralfree.yarn.*} (see {@link RopeOverrides}): the rope and YaRN
 *       context extension overrides; default none.</li>
 *   <li>{@value #PROP_CHECK_TENSORS}: check every tensor value for NaN and infinity at load; default false.</li>
 *   <li>{@value #PROP_LOAD_FORMAT}: full | dummy, seeded random weights instead of the file's (dummy loads have
 *       no content inputs, so nothing under one persists); default full.</li>
 *   <li>{@value #PROP_WARMUP}: run the load's warmup decode and reset; default true.</li>
 * </ul>
 * The tensor read mode is not here: it stays with {@link TensorReadMode}, whose property set every reader of
 * tensor bytes consults. Enum values ignore case and treat {@code -} and {@code _} alike. Immutable.
 */
public final class ModelSourceOptions {

    public static final String PROP_MATERIALIZE = "neuralfree.hf.materialize";
    public static final String PROP_MEMORY_MAX = "neuralfree.hf.materialize.memory.max";
    public static final String PROP_CACHE_DIR = "neuralfree.hf.cache.dir";
    public static final String PROP_OUTTYPE = "neuralfree.hf.outtype";
    public static final String PROP_ONLINE_QUANT = "neuralfree.quant.online";
    public static final String PROP_EXACT_QUANT = "neuralfree.hf.quant.exact";
    public static final String PROP_MTP = "neuralfree.hf.mtp";
    public static final String PROP_ST_DENSE = "neuralfree.hf.st.dense";
    public static final String PROP_OVERRIDES = "neuralfree.hf.overrides";
    public static final String PROP_TOKENIZER = "neuralfree.hf.tokenizer";
    public static final String PROP_LORA_BASE_CONFIG = "neuralfree.lora.base.config";
    public static final String PROP_BASE_DIR = "neuralfree.hf.base.dir";
    public static final String PROP_OVERRIDE_KV = "neuralfree.override.kv";
    public static final String PROP_CHECK_TENSORS = "neuralfree.check.tensors";
    public static final String PROP_LOAD_FORMAT = "neuralfree.load.format";
    public static final String PROP_WARMUP = "neuralfree.warmup";

    /** Default {@link #memoryBudget()}: 2 GiB. */
    public static final long DEFAULT_MEMORY_BUDGET = 2L << 30;

    /** Where tensors derived from the stored ones (permuted, repacked, stacked) are kept. */
    public enum Materialize {
        /** Chosen per reader and per tensor from the producer and {@link #memoryBudget()}. */
        AUTO,
        /** Produced once per requested region into direct memory. */
        MEMORY,
        /** Produced once per tensor into a private sidecar file and mapped read-only. */
        DISK,
        /** Produced on every read. */
        NONE;

        /** @throws IllegalArgumentException for an unknown name, naming {@value ModelSourceOptions#PROP_MATERIALIZE} */
        public static Materialize parse(String value) {
            return parseEnum(Materialize.class, PROP_MATERIALIZE, value);
        }
    }

    /** Type of float tensors in the view (the converter's {@code --outtype}). */
    public enum OutType {
        /** Each tensor keeps its stored float type. */
        SOURCE,
        /**
         * The converter's {@code --outtype auto}: BF16 or F16 after the first stored tensor of rank 2 or more that has
         * one of those types, F16 when none has.
         */
        AUTO,
        F32,
        F16,
        BF16,
        /** Q8_0, quantized at load. */
        Q8_0;

        /** @throws IllegalArgumentException for an unknown name, naming {@value ModelSourceOptions#PROP_OUTTYPE} */
        public static OutType parse(String value) {
            return parseEnum(OutType.class, PROP_OUTTYPE, value);
        }
    }

    /** Online quantization of float matmul weights at load. */
    public enum OnlineQuant {
        NONE,
        Q8_0,
        Q4_0,
        Q4_1,
        Q5_0,
        Q5_1,
        IQ4_NL;

        /**
         * @throws IllegalArgumentException for an unknown name, naming {@value ModelSourceOptions#PROP_ONLINE_QUANT}
         */
        public static OnlineQuant parse(String value) {
            return parseEnum(OnlineQuant.class, PROP_ONLINE_QUANT, value);
        }
    }

    /**
     * The load's weights ({@code --load-format}, the dummy loaders of the benchmark setups).
     */
    public enum LoadFormat {
        /** The checkpoint's own weights (and a Hugging Face conversion's). */
        FULL,
        /** Seeded random weights of the declared tensor types; nothing of the load is a content input. */
        DUMMY;

        /**
         * @throws IllegalArgumentException for an unknown name, naming {@value #PROP_LOAD_FORMAT}
         */
        public static LoadFormat parse(String value) {
            return parseEnum(LoadFormat.class, PROP_LOAD_FORMAT, value);
        }
    }

    private final Materialize materialize;
    private final long memoryBudget;
    private final Path cacheDir;
    private final OutType outType;
    private final OnlineQuant onlineQuant;
    private final boolean exactQuant;
    private final boolean includeMtp;
    private final boolean stDenseModules;
    private final Map<String, Object> configOverrides;
    private final Path tokenizerDir;
    private final Path loraBaseConfig;
    private final Path baseDir;
    private final List<MetadataOverrides.Override> metadataOverrides;
    private final RopeOverrides ropeOverrides;
    private final boolean checkTensors;
    private final LoadFormat loadFormat;
    private final boolean warmup;

    private ModelSourceOptions(Builder b) {
        this.materialize = b.materialize;
        this.memoryBudget = b.memoryBudget;
        this.cacheDir = b.cacheDir;
        this.outType = b.outType;
        this.onlineQuant = b.onlineQuant;
        this.exactQuant = b.exactQuant;
        this.includeMtp = b.includeMtp;
        this.stDenseModules = b.stDenseModules;
        this.configOverrides = b.configOverrides;
        this.tokenizerDir = b.tokenizerDir;
        this.loraBaseConfig = b.loraBaseConfig;
        this.baseDir = b.baseDir;
        this.metadataOverrides = b.metadataOverrides;
        this.ropeOverrides = b.ropeOverrides;
        this.checkTensors = b.checkTensors;
        this.loadFormat = b.loadFormat;
        this.warmup = b.warmup;
    }

    /** Defaults: auto materialization within 2 GiB, the user cache directory, stored types, no overrides. */
    public static ModelSourceOptions defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Options from {@link System#getProperties()}. */
    public static ModelSourceOptions fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    /**
     * Options from a property set; absent or blank keys keep their defaults.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static ModelSourceOptions fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        Builder b = builder();
        String v = value(p, PROP_MATERIALIZE);
        if (v != null) {
            b.materialize(Materialize.parse(v));
        }
        v = value(p, PROP_MEMORY_MAX);
        if (v != null) {
            b.memoryBudget(parseSize(PROP_MEMORY_MAX, v));
        }
        v = value(p, PROP_CACHE_DIR);
        if (v != null) {
            b.cacheDir(parsePath(PROP_CACHE_DIR, v));
        }
        v = value(p, PROP_OUTTYPE);
        if (v != null) {
            b.outType(OutType.parse(v));
        }
        v = value(p, PROP_ONLINE_QUANT);
        if (v != null) {
            b.onlineQuant(OnlineQuant.parse(v));
        }
        v = value(p, PROP_EXACT_QUANT);
        if (v != null) {
            b.exactQuant(parseBoolean(PROP_EXACT_QUANT, v));
        }
        v = value(p, PROP_MTP);
        if (v != null) {
            b.includeMtp(parseBoolean(PROP_MTP, v));
        }
        v = value(p, PROP_ST_DENSE);
        if (v != null) {
            b.stDenseModules(parseBoolean(PROP_ST_DENSE, v));
        }
        v = value(p, PROP_OVERRIDES);
        if (v != null) {
            b.configOverridesJson(v);
        }
        v = value(p, PROP_TOKENIZER);
        if (v != null) {
            b.tokenizerDir(parsePath(PROP_TOKENIZER, v));
        }
        v = value(p, PROP_LORA_BASE_CONFIG);
        if (v != null) {
            b.loraBaseConfig(parsePath(PROP_LORA_BASE_CONFIG, v));
        }
        v = value(p, PROP_BASE_DIR);
        if (v != null) {
            b.baseDir(parsePath(PROP_BASE_DIR, v));
        }
        v = value(p, PROP_OVERRIDE_KV);
        if (v != null) {
            b.metadataOverrideList(v);
        }
        RopeOverrides rope = RopeOverrides.fromProperties(p);
        if (!rope.isEmpty()) {
            b.ropeOverrides(rope);
        }
        v = value(p, PROP_CHECK_TENSORS);
        if (v != null) {
            b.checkTensors(parseBoolean(PROP_CHECK_TENSORS, v));
        }
        v = value(p, PROP_LOAD_FORMAT);
        if (v != null) {
            b.loadFormat(LoadFormat.parse(v));
        }
        v = value(p, PROP_WARMUP);
        if (v != null) {
            b.warmup(parseBoolean(PROP_WARMUP, v));
        }
        return b.build();
    }

    /** Where derived tensors live. */
    public Materialize materialize() { return materialize; }
    /** Largest total of derived bytes {@link Materialize#AUTO} keeps in direct memory. */
    public long memoryBudget() { return memoryBudget; }
    /** Parent of the per-process directories of derived tensors written to disk. */
    public Path cacheDir() { return cacheDir; }
    /** Type of float tensors in the view. */
    public OutType outType() { return outType; }
    /** Online quantization of float matmul weights. */
    public OnlineQuant onlineQuant() { return onlineQuant; }
    /** Whether bit-exact weight types are preferred over close repacks. */
    public boolean exactQuant() { return exactQuant; }
    /** Whether MTP / NextN layers are kept. */
    public boolean includeMtp() { return includeMtp; }
    /** Whether sentence-transformers dense heads are included. */
    public boolean stDenseModules() { return stDenseModules; }

    /**
     * The JSON object merged into {@code config.json} before conversion; empty for none. Unmodifiable at every level,
     * in the {@link Json} value model (String keys in order; String, Boolean, Long, BigInteger, Double, null, List,
     * Map).
     */
    public Map<String, Object> configOverrides() { return configOverrides; }

    /** Directory or GGUF file supplying the tokenizer, or null for the model's own. */
    public Path tokenizerDir() { return tokenizerDir; }
    /** Hugging Face configuration of a LoRA adapter's GGUF base, or null. */
    public Path loraBaseConfig() { return loraBaseConfig; }
    /** Local copy of the base model a decision-LoRA checkpoint merges, or null. */
    public Path baseDir() { return baseDir; }

    /**
     * The {@code --override-kv} metadata overrides applied to every opened source's GGUF view (post-conversion,
     * as {@link MetadataOverrides}), in their {@code --override-kv} order; empty for none.
     */
    public List<MetadataOverrides.Override> metadataOverrides() { return metadataOverrides; }

    /** The rope and YaRN context extension overrides ({@code ModelSources} applies them as metadata overrides). */
    public RopeOverrides ropeOverrides() { return ropeOverrides; }

    /** Whether every tensor value is checked for NaN and infinity at load ({@code --check-tensors}). */
    public boolean checkTensors() { return checkTensors; }

    /** The load's weights: the checkpoint's own ({@link LoadFormat#FULL}) or seeded random ({@link LoadFormat#DUMMY}). */
    public LoadFormat loadFormat() { return loadFormat; }

    /** Whether the load runs its warmup decode and reset ({@code --no-warmup} turns it off). */
    public boolean warmup() { return warmup; }

    public Builder toBuilder() {
        Builder b = builder();
        b.materialize = materialize;
        b.memoryBudget = memoryBudget;
        b.cacheDir = cacheDir;
        b.outType = outType;
        b.onlineQuant = onlineQuant;
        b.exactQuant = exactQuant;
        b.includeMtp = includeMtp;
        b.stDenseModules = stDenseModules;
        b.configOverrides = configOverrides;
        b.tokenizerDir = tokenizerDir;
        b.loraBaseConfig = loraBaseConfig;
        b.baseDir = baseDir;
        b.metadataOverrides = metadataOverrides;
        b.ropeOverrides = ropeOverrides;
        b.checkTensors = checkTensors;
        b.loadFormat = loadFormat;
        b.warmup = warmup;
        return b;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ModelSourceOptions)) {
            return false;
        }
        ModelSourceOptions other = (ModelSourceOptions) o;
        return materialize == other.materialize && memoryBudget == other.memoryBudget
                && cacheDir.equals(other.cacheDir) && outType == other.outType && onlineQuant == other.onlineQuant
                && exactQuant == other.exactQuant && includeMtp == other.includeMtp
                && stDenseModules == other.stDenseModules
                && configOverrides.equals(other.configOverrides)
                && Objects.equals(tokenizerDir, other.tokenizerDir)
                && Objects.equals(loraBaseConfig, other.loraBaseConfig) && Objects.equals(baseDir, other.baseDir)
                && metadataOverrides.equals(other.metadataOverrides) && ropeOverrides.equals(other.ropeOverrides)
                && checkTensors == other.checkTensors && loadFormat == other.loadFormat
                && warmup == other.warmup;
    }

    @Override
    public int hashCode() {
        return Objects.hash(materialize, memoryBudget, cacheDir, outType, onlineQuant, exactQuant, includeMtp,
                stDenseModules, configOverrides, tokenizerDir, loraBaseConfig, baseDir, metadataOverrides,
                ropeOverrides, checkTensors, loadFormat, warmup);
    }

    @Override
    public String toString() {
        return "ModelSourceOptions{materialize=" + materialize + ", memoryBudget=" + memoryBudget + ", cacheDir="
                + cacheDir + ", outType=" + outType + ", onlineQuant=" + onlineQuant + ", exactQuant=" + exactQuant
                + ", includeMtp=" + includeMtp + ", stDenseModules=" + stDenseModules
                + (configOverrides.isEmpty() ? "" : ", overrides=" + Json.toString(configOverrides, Json.Style.CANONICAL))
                + (metadataOverrides.isEmpty() ? "" : ", kv=" + MetadataOverrides.toString(metadataOverrides))
                + (ropeOverrides.isEmpty() ? "" : ", rope=" + ropeOverrides)
                + (checkTensors ? ", checkTensors" : "")
                + (loadFormat == LoadFormat.FULL ? "" : ", loadFormat=" + loadFormat.name().toLowerCase(Locale.ROOT))
                + (warmup ? "" : ", no warmup")
                + (tokenizerDir != null ? ", tokenizer=" + tokenizerDir : "")
                + (loraBaseConfig != null ? ", loraBaseConfig=" + loraBaseConfig : "")
                + (baseDir != null ? ", baseDir=" + baseDir : "") + '}';
    }

    // ------------------------------------------------------------------ parsing

    private static String value(Properties p, String key) {
        String v = p.getProperty(key);
        if (v == null) {
            return null;
        }
        String s = v.trim();
        return s.isEmpty() ? null : s;
    }

    /** An enum constant by name, ignoring case, with {@code -} and {@code _} equivalent. */
    static <E extends Enum<E>> E parseEnum(Class<E> type, String property, String value) {
        Objects.requireNonNull(value, property);
        String s = value.trim().toUpperCase(Locale.ROOT).replace('-', '_');
        StringBuilder valid = new StringBuilder();
        for (E e : type.getEnumConstants()) {
            if (e.name().equals(s)) {
                return e;
            }
            valid.append(valid.length() == 0 ? "" : ", ").append(e.name().toLowerCase(Locale.ROOT));
        }
        throw new IllegalArgumentException(property + "=" + value + " is invalid; use " + valid);
    }

    /**
     * A byte count {@code <n>[K|M|G]}: a non-negative decimal integer with an optional binary suffix (case-insensitive,
     * {@code K} = 1024).
     *
     * @throws IllegalArgumentException naming {@code property} for a malformed, negative or overflowing value
     */
    static long parseSize(String property, String value) {
        String s = value.trim();
        long unit = 1L;
        if (!s.isEmpty()) {
            char last = Character.toUpperCase(s.charAt(s.length() - 1));
            int shift = last == 'K' ? 10 : last == 'M' ? 20 : last == 'G' ? 30 : 0;
            if (shift != 0) {
                unit = 1L << shift;
                s = s.substring(0, s.length() - 1).trim();
            }
        }
        long n;
        try {
            n = Long.parseLong(s);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + "=" + value + " is not a size; use <n>[K|M|G]", e);
        }
        if (n < 0) {
            throw new IllegalArgumentException(property + "=" + value + " must not be negative");
        }
        try {
            return Math.multiplyExact(n, unit);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(property + "=" + value + " overflows a 64-bit byte count", e);
        }
    }

    private static boolean parseBoolean(String property, String value) {
        String s = value.trim().toLowerCase(Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(property + "=" + value + " is not a boolean; use true or false");
    }

    private static Path parsePath(String property, String value) {
        try {
            return Paths.get(value.trim());
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException(property + "=" + value + " is not a valid path: " + e.getReason(), e);
        }
    }

    /**
     * The overrides object in {@code text}, parsed with {@link Json.Limits#STRICT}.
     *
     * @throws IllegalArgumentException naming {@value #PROP_OVERRIDES} when the text is not one JSON object
     */
    private static Map<String, Object> parseOverrides(String text) {
        Object parsed;
        try {
            parsed = Json.parse(text);
        } catch (JsonException e) {
            throw new IllegalArgumentException(PROP_OVERRIDES + "=" + text + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException(PROP_OVERRIDES + "=" + text + " is not a JSON object; use {\"key\": value, ...}");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> object = (Map<String, Object>) parsed;
        return immutableObject(object, "");
    }

    /** A deep unmodifiable copy of a JSON object; integers widen to Long and Float to Double. */
    private static Map<String, Object> immutableObject(Map<?, ?> object, String path) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : object.entrySet()) {
            if (!(e.getKey() instanceof String)) {
                throw new IllegalArgumentException(PROP_OVERRIDES + ": key " + e.getKey() + " at '" + path
                        + "' is not a string");
            }
            String key = (String) e.getKey();
            out.put(key, immutableValue(e.getValue(), path.isEmpty() ? key : path + "." + key));
        }
        return Collections.unmodifiableMap(out);
    }

    private static Object immutableValue(Object v, String path) {
        if (v == null || v instanceof String || v instanceof Boolean || v instanceof Long || v instanceof BigInteger
                || v instanceof Double) {
            return v;
        }
        if (v instanceof Integer || v instanceof Short || v instanceof Byte) {
            return ((Number) v).longValue();
        }
        if (v instanceof Float) {
            return ((Float) v).doubleValue();
        }
        if (v instanceof Map) {
            return immutableObject((Map<?, ?>) v, path);
        }
        if (v instanceof List) {
            List<?> items = (List<?>) v;
            List<Object> out = new ArrayList<>(items.size());
            for (int i = 0; i < items.size(); i++) {
                out.add(immutableValue(items.get(i), path + "[" + i + "]"));
            }
            return Collections.unmodifiableList(out);
        }
        throw new IllegalArgumentException(PROP_OVERRIDES + ": value at '" + path + "' has type "
                + v.getClass().getName() + ", which is not a JSON value");
    }

    private static Path defaultCacheDir() {
        return Paths.get(System.getProperty("user.home", ""), ".cache", "neuralfree", "hf");
    }

    public static final class Builder {
        private Materialize materialize = Materialize.AUTO;
        private long memoryBudget = DEFAULT_MEMORY_BUDGET;
        private Path cacheDir = defaultCacheDir();
        private OutType outType = OutType.SOURCE;
        private OnlineQuant onlineQuant = OnlineQuant.NONE;
        private boolean exactQuant;
        private boolean includeMtp = true;
        private boolean stDenseModules = true;
        private Map<String, Object> configOverrides = Collections.emptyMap();
        private Path tokenizerDir;
        private Path loraBaseConfig;
        private Path baseDir;
        private List<MetadataOverrides.Override> metadataOverrides = Collections.emptyList();
        private RopeOverrides ropeOverrides = RopeOverrides.defaults();
        private boolean checkTensors;
        private LoadFormat loadFormat = LoadFormat.FULL;
        private boolean warmup = true;

        private Builder() {
        }

        public Builder materialize(Materialize v) {
            this.materialize = Objects.requireNonNull(v, "materialize");
            return this;
        }

        /** Bytes; checked by {@link #build()}. */
        public Builder memoryBudget(long v) { this.memoryBudget = v; return this; }
        public Builder cacheDir(Path v) { this.cacheDir = Objects.requireNonNull(v, "cacheDir"); return this; }
        public Builder outType(OutType v) { this.outType = Objects.requireNonNull(v, "outType"); return this; }

        public Builder onlineQuant(OnlineQuant v) {
            this.onlineQuant = Objects.requireNonNull(v, "onlineQuant");
            return this;
        }

        public Builder exactQuant(boolean v) { this.exactQuant = v; return this; }
        public Builder includeMtp(boolean v) { this.includeMtp = v; return this; }
        public Builder stDenseModules(boolean v) { this.stDenseModules = v; return this; }

        /**
         * The text of a JSON object (the {@value #PROP_OVERRIDES} form); null or blank removes the overrides.
         *
         * @throws IllegalArgumentException naming {@value #PROP_OVERRIDES} when the text is not one JSON object
         */
        public Builder configOverridesJson(String v) {
            this.configOverrides = v == null || v.trim().isEmpty() ? Collections.emptyMap() : parseOverrides(v.trim());
            return this;
        }

        /**
         * A JSON object in the {@link Json} value model, copied deeply; null removes the overrides.
         *
         * @throws IllegalArgumentException naming {@value #PROP_OVERRIDES} for a key that is not a string or a value
         *                                  outside the JSON value model
         */
        public Builder configOverrides(Map<String, ?> v) {
            this.configOverrides = v == null ? Collections.emptyMap() : immutableObject(v, "");
            return this;
        }

        /** A tokenizer directory or GGUF file; null removes it. */
        public Builder tokenizerDir(Path v) { this.tokenizerDir = v; return this; }
        /** The LoRA base configuration; null removes it. */
        public Builder loraBaseConfig(Path v) { this.loraBaseConfig = v; return this; }
        /** The decision-LoRA base directory; null removes it. */
        public Builder baseDir(Path v) { this.baseDir = v; return this; }

        /**
         * Replaces the {@code --override-kv} overrides with the entries of {@code v} (the
         * {@value #PROP_OVERRIDE_KV} list form, {@code ;} separated); null or blank removes them all.
         *
         * @throws IllegalArgumentException naming the entry when one of them cannot be parsed
         */
        public Builder metadataOverrideList(String v) {
            this.metadataOverrides = v == null || v.trim().isEmpty() ? Collections.emptyList()
                    : MetadataOverrides.parseList(v, PROP_OVERRIDE_KV);
            return this;
        }

        /**
         * Appends one {@code key=type:value} entry ({@code --override-kv}); repeated flags accumulate in order.
         *
         * @throws IllegalArgumentException naming the entry when it cannot be parsed
         */
        public Builder metadataOverride(String kv) {
            List<MetadataOverrides.Override> l = new ArrayList<>(metadataOverrides.size() + 1);
            l.addAll(metadataOverrides);
            l.add(MetadataOverrides.parse(kv, PROP_OVERRIDE_KV));
            this.metadataOverrides = Collections.unmodifiableList(l);
            return this;
        }

        /**
         * Replaces the rope and YaRN context extension overrides; null removes them.
         */
        public Builder ropeOverrides(RopeOverrides v) {
            this.ropeOverrides = v == null ? RopeOverrides.defaults() : v;
            return this;
        }

        /** Builds the rope overrides on a {@link RopeOverrides.Builder} and replaces them. */
        public Builder ropeOverrides(RopeOverrides.Builder v) {
            return ropeOverrides(v == null ? null : v.build());
        }

        /** Changes one rope or YaRN option, keeping the ones already set ({@code R} e.g. {@code rope.freqBase}). */
        public Builder ropeScaling(RopeOverrides.ScalingType v) {
            RopeOverrides.Builder b = ropeOverrides.toBuilder();
            b.scalingType(v);
            return ropeOverrides(b);
        }

        /** {@code --rope-freq-scale} / its alias {@code --rope-scale}. */
        public Builder ropeFreqScale(float v) {
            RopeOverrides.Builder b = ropeOverrides.toBuilder();
            b.freqScale(v);
            return ropeOverrides(b);
        }

        public Builder ropeFreqBase(float v) {
            RopeOverrides.Builder b = ropeOverrides.toBuilder();
            b.freqBase(v);
            return ropeOverrides(b);
        }

        /** The {@code --yarn-*} values: {@code --yarn-ext-factor}, {@code --yarn-attn-factor},
         *  {@code --yarn-beta-fast} and {@code --yarn-beta-slow}. */
        public Builder yarnExtFactor(float v) { return yarn((rb, f) -> rb.yarnExtFactor(f), v); }
        public Builder yarnAttnFactor(float v) { return yarn((rb, f) -> rb.yarnAttnFactor(f), v); }
        public Builder yarnBetaFast(float v) { return yarn((rb, f) -> rb.yarnBetaFast(f), v); }
        public Builder yarnBetaSlow(float v) { return yarn((rb, f) -> rb.yarnBetaSlow(f), v); }

        /** Applies one {@code --yarn-*} value to the rope overrides built so far. */
        private Builder yarn(java.util.function.BiConsumer<RopeOverrides.Builder, Float> set, float v) {
            RopeOverrides.Builder b = ropeOverrides.toBuilder();
            set.accept(b, v);
            return ropeOverrides(b);
        }

        /** {@code --yarn-orig-ctx}: 0 keeps the model's training context. */
        public Builder yarnOrigCtx(int v) {
            RopeOverrides.Builder b = ropeOverrides.toBuilder();
            b.yarnOrigCtx(v);
            return ropeOverrides(b);
        }

        public Builder checkTensors(boolean v) { this.checkTensors = v; return this; }

        public Builder loadFormat(LoadFormat v) {
            this.loadFormat = Objects.requireNonNull(v, PROP_LOAD_FORMAT);
            return this;
        }

        public Builder warmup(boolean v) { this.warmup = v; return this; }

        /** @throws IllegalArgumentException for a negative memory budget */
        public ModelSourceOptions build() {
            if (memoryBudget < 0) {
                throw new IllegalArgumentException(PROP_MEMORY_MAX + " must not be negative, got " + memoryBudget);
            }
            return new ModelSourceOptions(this);
        }
    }
}
