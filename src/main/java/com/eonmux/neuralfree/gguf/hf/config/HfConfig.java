package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * The model configuration of a Hugging Face checkpoint as the llama.cpp converter sees it (LC/conversion/base.py
 * {@code load_hparams}, {@code TextModel.__init__}): the root object after overrides and sub-config renames, and the
 * flattened view {@code {**root, **text_config}} in which {@code text_config} members win.
 *
 * <p>Lookups follow {@code find_hparam}: the first alias present in {@link #flat()} wins, even when its value is
 * JSON {@code null} (the lookup then reports absence). Integer accessors enforce the caps of the key's family
 * (vocabulary, layers, experts, heads, widths) and throw {@link GgufFormatException} naming the source file and the
 * key. Instances are immutable; every container handed out is unmodifiable at every depth.</p>
 */
public final class HfConfig {

    /** The Hugging Face configuration file name. */
    public static final String CONFIG_JSON = "config.json";
    /** The Mistral native configuration file name. */
    public static final String PARAMS_JSON = "params.json";
    /** The generation defaults file name. */
    public static final String GENERATION_CONFIG_JSON = "generation_config.json";

    /** Largest accepted vocabulary size. */
    public static final long MAX_VOCAB = 16_777_216L;
    /** Largest accepted layer count. */
    public static final long MAX_LAYERS = 4096L;
    /** Largest accepted expert count. */
    public static final long MAX_EXPERTS = 65_536L;
    /** Largest accepted head count. */
    public static final long MAX_HEADS = 65_536L;
    /** Largest accepted width (hidden, intermediate, head and rank sizes). */
    public static final long MAX_WIDTH = 1_048_576L;

    /** Architecture names whose root wins over {@code text_config.architectures} (base.py get_model_architecture). */
    static final List<String> TEXT_PASSTHROUGH_ARCHITECTURES = List.of("NEOChatModel",
            "StepVLForConditionalGeneration", "Sarashina2VisionForCausalLM", "Exaone4_5_ForConditionalGeneration",
            "Step3p7ForConditionalGeneration", "KimiK3ForConditionalGeneration");

    private static final List<String> BLOCK_COUNT_ALIASES = List.of("n_layers", "num_hidden_layers", "n_layer",
            "num_layers");
    private static final List<String> WIDTH_NAMES = List.of("dim", "n_embd", "n_embed", "d_model", "n_inner",
            "d_ff", "d_inner", "d_state", "ffn_dim", "hidden_dim", "width");
    private static final List<String> NON_WIDTH_PARTS = List.of("window", "position", "max_", "seq", "ctx",
            "chunk", "context", "block", "cache", "length", "token", "batch", "bucket", "stride", "period");

    private final Path dir;
    private final String source;
    private final Map<String, Object> root;
    private final Map<String, Object> text;
    private final Map<String, Object> flat;
    private final Map<String, Object> generation;
    private final String architecture;
    private final boolean mistralNative;
    private final Path baseModelDir;
    private final HfRope rope;

    private HfConfig(Path dir, String source, Map<String, Object> root, Map<String, Object> flat,
                     Map<String, Object> generation, String architecture, boolean mistralNative, Path baseModelDir) {
        this.dir = dir;
        this.source = source;
        this.root = root;
        Object t = root.get("text_config");
        this.text = t instanceof Map ? asObject(t) : Collections.emptyMap();
        this.flat = flat;
        this.generation = generation;
        this.architecture = architecture;
        this.mistralNative = mistralNative;
        this.baseModelDir = baseModelDir;
        this.rope = HfRope.of(flat, source);
    }

    /**
     * A configuration over {@code root}, a JSON object in the {@link Json} value model that has already had its
     * overrides and sub-config renames applied (see {@link HfConfigLoader}).
     *
     * @param dir           the model directory side files resolve against
     * @param source        the file the root came from, named in messages (for example {@code config.json})
     * @param generation    the {@code generation_config.json} object, or null
     * @param mistralNative whether the root was translated from Mistral native {@code params.json}
     * @param baseModelDir  the local base model of a LoRA-style checkpoint, or null
     * @throws GgufFormatException when the root has no usable architecture or a malformed RoPE block
     */
    static HfConfig create(Path dir, String source, Map<String, Object> root, Map<String, Object> generation,
                           boolean mistralNative, Path baseModelDir) {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(source, "source");
        Map<String, Object> frozenRoot = freeze(root);
        Object textConfig = frozenRoot.get("text_config");
        if (textConfig != null && !(textConfig instanceof Map)) {
            throw new GgufFormatException(source + ": 'text_config' must be an object, got "
                    + typeName(textConfig));
        }
        Map<String, Object> flat = new LinkedHashMap<>(frozenRoot);
        if (textConfig != null) {
            flat.putAll(asObject(textConfig));
        }
        String arch = detectArchitecture(frozenRoot, source);
        return new HfConfig(dir, source, frozenRoot, Collections.unmodifiableMap(flat),
                generation == null ? null : freeze(generation), arch, mistralNative, baseModelDir);
    }

    /** The model directory. */
    public Path dir() {
        return dir;
    }

    /** The file the configuration came from, as named in messages (for example {@code config.json}). */
    public String source() {
        return source;
    }

    /** The root object. */
    public JsonView raw() {
        return JsonView.of(root, source);
    }

    /** The {@code text_config} object, or an empty view when the root has none. */
    public JsonView text() {
        return text.isEmpty() && !root.containsKey("text_config")
                ? JsonView.empty(source, "text_config") : JsonView.of(text, source, "text_config");
    }

    /** The flattened {@code {**root, **text_config}} map; unmodifiable at every depth. */
    public Map<String, Object> flat() {
        return flat;
    }

    /**
     * The vision encoder configuration: {@code vision_config}, {@code thinker_config.vision_config}, or for Mistral
     * native checkpoints {@code vision_encoder} / {@code multimodal.vision_encoder_args}.
     */
    public Optional<JsonView> vision() {
        Optional<JsonView> v = member(root, "vision_config", "vision_config");
        if (v.isPresent()) {
            return v;
        }
        v = nested("thinker_config", "vision_config");
        if (v.isPresent()) {
            return v;
        }
        v = member(root, "vision_encoder", "vision_encoder");
        return v.isPresent() ? v : nested("multimodal", "vision_encoder_args");
    }

    /** The audio encoder configuration: {@code audio_config} or {@code thinker_config.audio_config}. */
    public Optional<JsonView> audio() {
        Optional<JsonView> a = member(root, "audio_config", "audio_config");
        return a.isPresent() ? a : nested("thinker_config", "audio_config");
    }

    /** The {@code generation_config.json} object when the checkpoint has one. */
    public Optional<JsonView> generation() {
        return generation == null ? Optional.empty()
                : Optional.of(JsonView.of(generation, GENERATION_CONFIG_JSON));
    }

    /** The Hugging Face architecture name the converter dispatches on (base.py get_model_architecture). */
    public String architecture() {
        return architecture;
    }

    /** The root {@code model_type}, or the empty string when absent. */
    public String modelType() {
        Object v = root.get("model_type");
        return v instanceof String ? (String) v : "";
    }

    /** Whether the configuration was translated from Mistral native {@code params.json}. */
    public boolean mistralNative() {
        return mistralNative;
    }

    /** The local base model directory of a LoRA-style decision checkpoint (Lev, Kev, Nimble). */
    public Optional<Path> baseModelDir() {
        return Optional.ofNullable(baseModelDir);
    }

    /** The normalized RoPE parameters of the flattened configuration. */
    public HfRope rope() {
        return rope;
    }

    /**
     * A copy whose {@link #flat()} has {@code replacements} applied on top (a {@code null} value stores JSON null).
     * The root, text, vision, audio and generation views are unchanged; {@link #rope()} is recomputed.
     */
    public HfConfig withFlat(Map<String, Object> replacements) {
        Objects.requireNonNull(replacements, "replacements");
        Map<String, Object> merged = new LinkedHashMap<>(flat);
        for (Map.Entry<String, Object> e : replacements.entrySet()) {
            merged.put(Objects.requireNonNull(e.getKey(), "replacement key"), freezeValue(e.getValue()));
        }
        return new HfConfig(dir, source, root, Collections.unmodifiableMap(merged), generation, architecture,
                mistralNative, baseModelDir);
    }

    /**
     * The model directory path of side file {@code name}, a name taken from file contents, validated by the
     * content-name rule: relative, no {@code ..} segment, no NUL, no drive or root, and lexically under the
     * model directory after normalization.
     *
     * @throws IOException naming the directory and the file name when the name breaks the rule
     */
    public Path sideFile(String name) throws IOException {
        return contentPath(dir, name);
    }

    /** As {@link #sideFile}, for a file name taken from the contents of a checkpoint in {@code dir}. */
    public static Path contentPath(Path dir, String name) throws IOException {
        Objects.requireNonNull(dir, "dir");
        String reason = contentNameProblem(name);
        if (reason == null) {
            Path base = dir.normalize();
            Path resolved = base.resolve(name).normalize();
            if (resolved.startsWith(base) && !resolved.equals(base)) {
                return resolved;
            }
            reason = "it resolves outside the model directory";
        }
        throw new IOException("Invalid Hugging Face checkpoint '" + dir + "': file name '" + name
                + "' taken from the checkpoint is rejected: " + reason);
    }

    private static String contentNameProblem(String name) {
        if (name == null || name.isEmpty()) {
            return "it is empty";
        }
        if (name.indexOf('\0') >= 0) {
            return "it contains NUL";
        }
        if (name.startsWith("/") || name.startsWith("\\") || (name.length() > 1 && name.charAt(1) == ':')) {
            return "it is absolute";
        }
        for (String segment : name.split("[/\\\\]")) {
            if (segment.equals("..")) {
                return "it contains a '..' segment";
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ lookups

    /**
     * The value of the first alias present in {@link #flat()}. A present alias whose value is JSON {@code null}
     * ends the search and yields empty, as {@code find_hparam} returns {@code None}.
     */
    public Optional<Object> find(String... aliases) {
        String key = firstPresent(aliases);
        return key == null ? Optional.empty() : Optional.ofNullable(flat.get(key));
    }

    /** As {@link #find}, as a capped {@code int}. */
    public OptionalInt findInt(String... aliases) {
        String key = firstPresent(aliases);
        Object v = key == null ? null : flat.get(key);
        return v == null ? OptionalInt.empty() : OptionalInt.of(toCappedInt(key, v));
    }

    /** As {@link #findInt(String...)}, or {@code dflt} when no alias has a value. */
    public int findInt(int dflt, String... aliases) {
        OptionalInt v = findInt(aliases);
        return v.isPresent() ? v.getAsInt() : dflt;
    }

    /**
     * As {@link #findInt(String...)}, required.
     *
     * @throws GgufFormatException naming the source and the aliases when none has a value
     */
    public int requireInt(String... aliases) {
        OptionalInt v = findInt(aliases);
        if (!v.isPresent()) {
            throw new GgufFormatException(source + ": required key " + describe(aliases) + " is missing");
        }
        return v.getAsInt();
    }

    /** As {@link #find}, as a capped {@code long}. */
    public OptionalLong findLong(String... aliases) {
        String key = firstPresent(aliases);
        Object v = key == null ? null : flat.get(key);
        return v == null ? OptionalLong.empty() : OptionalLong.of(toCappedLong(key, v));
    }

    /** As {@link #findLong(String...)}, or {@code dflt} when no alias has a value. */
    public long findLong(long dflt, String... aliases) {
        OptionalLong v = findLong(aliases);
        return v.isPresent() ? v.getAsLong() : dflt;
    }

    /** As {@link #find}, as a number (integers convert exactly where representable). */
    public OptionalDouble findFloat(String... aliases) {
        String key = firstPresent(aliases);
        Object v = key == null ? null : flat.get(key);
        return v == null ? OptionalDouble.empty() : OptionalDouble.of(toDouble(source, where(key), v));
    }

    /** As {@link #findFloat(String...)}, or {@code dflt} when no alias has a value. */
    public double findFloat(double dflt, String... aliases) {
        OptionalDouble v = findFloat(aliases);
        return v.isPresent() ? v.getAsDouble() : dflt;
    }

    /** As {@link #find}, as a boolean. */
    public Optional<Boolean> findBool(String... aliases) {
        String key = firstPresent(aliases);
        Object v = key == null ? null : flat.get(key);
        if (v == null) {
            return Optional.empty();
        }
        if (!(v instanceof Boolean)) {
            throw mismatch(key, "a boolean", v);
        }
        return Optional.of((Boolean) v);
    }

    /** As {@link #findBool(String...)}, or {@code dflt} when no alias has a value. */
    public boolean findBool(boolean dflt, String... aliases) {
        return findBool(aliases).orElse(dflt);
    }

    /** As {@link #find}, as a string. */
    public Optional<String> findString(String... aliases) {
        String key = firstPresent(aliases);
        Object v = key == null ? null : flat.get(key);
        if (v == null) {
            return Optional.empty();
        }
        if (!(v instanceof String)) {
            throw mismatch(key, "a string", v);
        }
        return Optional.of((String) v);
    }

    /** As {@link #findString(String...)}, or {@code dflt} when no alias has a value. */
    public String findStringOr(String dflt, String... aliases) {
        return findString(aliases).orElse(dflt);
    }

    /** As {@link #find}, as a list of capped integers (a scalar is not a list). */
    public Optional<List<Integer>> findIntList(String... aliases) {
        String key = firstPresent(aliases);
        Object v = key == null ? null : flat.get(key);
        if (v == null) {
            return Optional.empty();
        }
        if (!(v instanceof List)) {
            throw mismatch(key, "an array of integers", v);
        }
        List<?> items = (List<?>) v;
        List<Integer> out = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Object item = items.get(i);
            if (item == null) {
                throw mismatch(key + "[" + i + "]", "an integer", null);
            }
            out.add(toCappedInt(key, item));
        }
        return Optional.of(Collections.unmodifiableList(out));
    }

    /** As {@link #findIntList(String...)}, or {@code dflt} when no alias has a value. */
    public List<Integer> findIntList(List<Integer> dflt, String... aliases) {
        return findIntList(aliases).orElse(dflt);
    }

    /**
     * The block count ({@code n_layers, num_hidden_layers, n_layer, num_layers}).
     *
     * @throws GgufFormatException when none is present
     */
    public int blockCount() {
        return requireInt(BLOCK_COUNT_ALIASES.toArray(new String[0]));
    }

    private String firstPresent(String... aliases) {
        if (aliases.length == 0) {
            throw new IllegalArgumentException("at least one alias is required");
        }
        for (String a : aliases) {
            if (flat.containsKey(a)) {
                return a;
            }
        }
        return null;
    }

    /** The key as named in messages: {@code text_config.<key>} when the text config supplied it. */
    private String where(String key) {
        return text.containsKey(key) ? "text_config." + key : key;
    }

    private static String describe(String... aliases) {
        return aliases.length == 1 ? "'" + aliases[0] + "'" : "(one of " + String.join(", ", aliases) + ")";
    }

    private GgufFormatException mismatch(String key, String expected, Object got) {
        return new GgufFormatException(source + ": key '" + where(key) + "' must be " + expected + ", got "
                + typeName(got));
    }

    private int toCappedInt(String key, Object v) {
        long l = toCappedLong(key, v);
        if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
            throw new GgufFormatException(source + ": key '" + where(key) + "' value " + l
                    + " is out of the 32-bit integer range");
        }
        return (int) l;
    }

    private long toCappedLong(String key, Object v) {
        long l = toLong(source, where(key), v);
        long cap = capOf(key);
        if (cap > 0 && (l < 0 || l > cap)) {
            throw new GgufFormatException(source + ": key '" + where(key) + "' value " + l + " is outside [0, "
                    + cap + "] for " + familyOf(key));
        }
        return l;
    }

    /** The configuration cap of the key's family, or 0 when the key belongs to no capped family. */
    static long capOf(String key) {
        switch (familyOf(key)) {
            case "vocabulary sizes": return MAX_VOCAB;
            case "layer counts": return MAX_LAYERS;
            case "head counts": return MAX_HEADS;
            case "widths": return MAX_WIDTH;
            case "expert counts": return MAX_EXPERTS;
            default: return 0;
        }
    }

    private static String familyOf(String key) {
        String k = key.toLowerCase(Locale.ROOT);
        if (k.contains("vocab")) {
            return "vocabulary sizes";
        }
        if (k.contains("layer") && !k.contains("norm") && !k.contains("scale") && !k.contains("idx")
                && !k.contains("index") && !k.contains("types")) {
            return "layer counts";
        }
        if (isWidth(k)) {
            return "widths";
        }
        if (k.contains("head")) {
            return "head counts";
        }
        if (k.contains("expert")) {
            return "expert counts";
        }
        return "uncapped keys";
    }

    private static boolean isWidth(String k) {
        for (String part : NON_WIDTH_PARTS) {
            if (k.contains(part)) {
                return false;
            }
        }
        return WIDTH_NAMES.contains(k) || k.endsWith("_size") || k.endsWith("_dim") || k.endsWith("_rank");
    }

    // ------------------------------------------------------------------ architecture

    private static String detectArchitecture(Map<String, Object> root, String source) {
        String arch = null;
        Object arches = root.get("architectures");
        if (arches instanceof List && !((List<?>) arches).isEmpty()) {
            arch = firstName(((List<?>) arches).get(0), source, "architectures[0]");
        } else if (arches != null && !(arches instanceof List)) {
            throw new GgufFormatException(source + ": 'architectures' must be an array, got " + typeName(arches));
        } else if (root.containsKey("ssm_cfg")) {
            Object ssm = root.get("ssm_cfg");
            Object layer = ssm instanceof Map ? ((Map<?, ?>) ssm).get("layer") : null;
            arch = (layer == null ? "Mamba" : firstName(layer, source, "ssm_cfg.layer")) + "ForCausalLM";
        }
        if (arch != null && TEXT_PASSTHROUGH_ARCHITECTURES.contains(arch)) {
            return arch;
        }
        Object textConfig = root.get("text_config");
        if (textConfig instanceof Map) {
            Object textArches = ((Map<?, ?>) textConfig).get("architectures");
            if (textArches instanceof List && !((List<?>) textArches).isEmpty()) {
                arch = firstName(((List<?>) textArches).get(0), source, "text_config.architectures[0]");
            } else if (textArches != null && !(textArches instanceof List)) {
                throw new GgufFormatException(source + ": 'text_config.architectures' must be an array, got "
                        + typeName(textArches));
            }
        }
        if (arch == null) {
            throw new GgufFormatException(source + ": failed to detect the model architecture: no 'architectures', "
                    + "'ssm_cfg' or 'text_config.architectures'");
        }
        return arch;
    }

    private static String firstName(Object v, String source, String key) {
        if (!(v instanceof String) || ((String) v).isEmpty()) {
            throw new GgufFormatException(source + ": '" + key + "' must be a non-empty string, got " + typeName(v));
        }
        return (String) v;
    }

    private Optional<JsonView> member(Map<String, Object> map, String key, String path) {
        Object v = map.get(key);
        if (v == null) {
            return Optional.empty();
        }
        if (!(v instanceof Map)) {
            throw new GgufFormatException(source + ": '" + path + "' must be an object, got " + typeName(v));
        }
        return Optional.of(JsonView.of(v, source, path));
    }

    private Optional<JsonView> nested(String outer, String inner) {
        Object o = root.get(outer);
        return o instanceof Map ? member(asObject(o), inner, outer + "." + inner) : Optional.empty();
    }

    // ------------------------------------------------------------------ value helpers

    /** {@code v} as an exact {@code long}; integral doubles convert, fractions, booleans and overflow fail. */
    static long toLong(String source, String key, Object v) {
        return toLong(source + ": key '" + key + "'", v);
    }

    /** As {@link #toLong(String, String, Object)}; {@code where} names the value in messages. */
    static long toLong(String where, Object v) {
        if (v instanceof Long) {
            return (Long) v;
        }
        if (v instanceof BigInteger && ((BigInteger) v).bitLength() < Long.SIZE) {
            return ((BigInteger) v).longValue();
        }
        if (v instanceof Double) {
            double d = (Double) v;
            if (d == Math.rint(d) && !Double.isInfinite(d) && d >= -0x1p63 && d < 0x1p63) {
                return (long) d;
            }
        }
        if (v instanceof Integer) {
            return (Integer) v;
        }
        throw new GgufFormatException(where + " must be an integer, got " + typeName(v)
                + (v instanceof Number ? " " + v : ""));
    }

    /** {@code v} as a {@code double}; integers convert, other types fail. */
    static double toDouble(String source, String key, Object v) {
        return toDouble(source + ": key '" + key + "'", v);
    }

    /** As {@link #toDouble(String, String, Object)}; {@code where} names the value in messages. */
    static double toDouble(String where, Object v) {
        if (v instanceof Number && !(v instanceof BigInteger && ((BigInteger) v).bitLength() > 1023)) {
            return ((Number) v).doubleValue();
        }
        throw new GgufFormatException(where + " must be a number, got " + typeName(v));
    }

    /** JSON type name of a value of the {@link Json} value model. */
    static String typeName(Object v) {
        if (v == null) return "null";
        if (v instanceof Boolean) return "boolean";
        if (v instanceof Long || v instanceof BigInteger || v instanceof Integer) return "integer";
        if (v instanceof Number) return "number";
        if (v instanceof String) return "string";
        if (v instanceof List) return "array";
        if (v instanceof Map) return "object";
        return v.getClass().getSimpleName();
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asObject(Object v) {
        return (Map<String, Object>) v;
    }

    /** A deep, unmodifiable copy of a JSON object. */
    static Map<String, Object> freeze(Map<String, ?> map) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, ?> e : map.entrySet()) {
            out.put(e.getKey(), freezeValue(e.getValue()));
        }
        return Collections.unmodifiableMap(out);
    }

    @SuppressWarnings("unchecked")
    private static Object freezeValue(Object v) {
        if (v instanceof Map) {
            return freeze((Map<String, ?>) v);
        }
        if (v instanceof List) {
            List<Object> out = new ArrayList<>(((List<?>) v).size());
            for (Object item : (List<?>) v) {
                out.add(freezeValue(item));
            }
            return Collections.unmodifiableList(out);
        }
        return v;
    }

    /** A deep, mutable copy of a JSON value (objects become {@link LinkedHashMap}, arrays {@link ArrayList}). */
    @SuppressWarnings("unchecked")
    static Object mutableCopy(Object v) {
        if (v instanceof Map) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<String, ?> e : ((Map<String, ?>) v).entrySet()) {
                out.put(e.getKey(), mutableCopy(e.getValue()));
            }
            return out;
        }
        if (v instanceof List) {
            List<Object> out = new ArrayList<>(((List<?>) v).size());
            for (Object item : (List<?>) v) {
                out.add(mutableCopy(item));
            }
            return out;
        }
        return v;
    }

    @Override
    public String toString() {
        return "HfConfig{" + dir + ", " + source + ", architecture=" + architecture
                + (mistralNative ? ", mistralNative" : "") + "}";
    }
}
