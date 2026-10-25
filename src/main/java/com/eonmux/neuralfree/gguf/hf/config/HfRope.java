package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The RoPE parameters of a flattened configuration, normalized as the llama.cpp converter and vLLM do
 * (LC/conversion/base.py TextModel.__init__ and set_gguf_parameters; V/transformers_utils/config.py
 * patch_legacy_rope_type):
 * <ul>
 *   <li>the input is {@code rope_parameters}, else {@code rope_scaling} (a present {@code rope_parameters: null}
 *       selects nothing, as {@code dict.get} does), possibly nested per layer type;</li>
 *   <li>legacy forms: {@code type} becomes {@code rope_type} (a conflicting pair is an error, except
 *       {@code su/longrope} and {@code mrope/default}); {@code su} becomes {@code longrope}; {@code mrope} becomes
 *       {@code default} and requires {@code mrope_section}; {@code telechat3-yarn} becomes {@code yarn} with
 *       {@code attention_factor = 0.07 * ln(factor) + 1};</li>
 *   <li>when not nested by {@code full_attention}/{@code sliding_attention}, the global theta, local theta (as
 *       {@code sliding_attention.rope_theta}), partial rotary factor and {@code original_max_position_embeddings}
 *       found among the configuration aliases are mirrored into the parameters when absent there.</li>
 * </ul>
 * {@link #global()} is {@code full_attention} when present, else the whole parameter object, as the converter
 * reads it.
 */
public final class HfRope {

    /** Global theta aliases, first present wins. */
    public static final List<String> THETA_ALIASES = List.of("global_rope_theta", "rope_global_theta",
            "rope_theta_global", "rope_theta", "rotary_emb_base");
    /** Sliding-window (local) theta aliases, first present wins. */
    public static final List<String> LOCAL_THETA_ALIASES = List.of("local_rope_theta", "rope_local_theta",
            "rope_theta_local", "swa_rope_theta", "rope_local_base_freq");
    /** Partial rotary factor aliases, first present wins. */
    public static final List<String> PARTIAL_ROTARY_ALIASES = List.of("partial_rotary_factor", "rope_pct",
            "rope_percent", "rotary_pct", "rotary_emb_fraction");

    /** Keys of a parameter object nested by layer type (transformers ALLOWED_LAYER_TYPES plus DeepSeek-V4). */
    static final Set<String> LAYER_TYPES = Set.of("full_attention", "sliding_attention", "chunked_attention",
            "window_attention", "indexed_attention", "compressed_sparse_attention", "heavily_compressed_attention",
            "deepseek_sparse_attention", "minimax_m3_sparse", "conv", "moe", "hybrid", "hybrid_sliding",
            "reference_sliding_attention", "linear_attention", "sparse", "dense", "main", "compress");

    private final Map<String, Object> params;
    private final Params global;
    private final boolean nested;
    private final String source;

    private HfRope(Map<String, Object> params, boolean nested, String source) {
        this.params = params;
        this.source = source;
        this.nested = nested;
        Object full = params.get("full_attention");
        this.global = new Params(full instanceof Map ? HfConfig.asObject(full) : params, source,
                full instanceof Map ? "rope_parameters.full_attention" : "rope_parameters");
    }

    /**
     * Normalizes the RoPE block of {@code flat}, the flattened configuration.
     *
     * @throws GgufFormatException naming {@code source} and the key for malformed or conflicting parameters
     */
    static HfRope of(Map<String, Object> flat, String source) {
        Object in = flat.containsKey("rope_parameters") ? flat.get("rope_parameters") : flat.get("rope_scaling");
        if (in != null && !(in instanceof Map)) {
            throw new GgufFormatException(source + ": 'rope_parameters'/'rope_scaling' must be an object, got "
                    + HfConfig.typeName(in));
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> params = in == null ? new LinkedHashMap<>()
                : (Map<String, Object>) HfConfig.mutableCopy(in);
        boolean nested = isNested(params);
        patchLegacy(params, nested, source);
        if (!params.containsKey("full_attention") && !params.containsKey("sliding_attention")) {
            Object local = first(flat, LOCAL_THETA_ALIASES);
            if (local != null) {
                Map<String, Object> sliding = new LinkedHashMap<>();
                sliding.put("rope_theta", local);
                params.put("sliding_attention", sliding);
            }
            mirror(params, "rope_theta", first(flat, THETA_ALIASES));
            mirror(params, "partial_rotary_factor", first(flat, PARTIAL_ROTARY_ALIASES));
            mirror(params, "original_max_position_embeddings", flat.get("original_max_position_embeddings"));
        }
        return new HfRope(HfConfig.freeze(params), nested, source);
    }

    private static void mirror(Map<String, Object> params, String key, Object value) {
        if (!params.containsKey(key) && value != null) {
            params.put(key, value);
        }
    }

    /** The value of the first alias present (a present null ends the search, as find_hparam does). */
    private static Object first(Map<String, Object> flat, List<String> aliases) {
        for (String a : aliases) {
            if (flat.containsKey(a)) {
                return flat.get(a);
            }
        }
        return null;
    }

    private static boolean isNested(Map<String, Object> params) {
        return !params.isEmpty() && LAYER_TYPES.containsAll(params.keySet());
    }

    private static void patchLegacy(Map<String, Object> params, boolean nested, String source) {
        if (nested) {
            for (Map.Entry<String, Object> e : params.entrySet()) {
                if (e.getValue() instanceof Map) {
                    patchOne(HfConfig.asObject(e.getValue()), source, "rope_parameters." + e.getKey());
                } else if (e.getValue() != null) {
                    throw new GgufFormatException(source + ": 'rope_parameters." + e.getKey()
                            + "' must be an object, got " + HfConfig.typeName(e.getValue()));
                }
            }
        } else {
            patchOne(params, source, "rope_parameters");
        }
    }

    private static void patchOne(Map<String, Object> p, String source, String path) {
        if (p.containsKey("rope_type") && p.containsKey("type")) {
            Object modern = p.get("rope_type");
            Object legacy = p.get("type");
            boolean compatible = ("su".equals(legacy) && "longrope".equals(modern))
                    || ("mrope".equals(legacy) && "default".equals(modern));
            if (!compatible && !Objects.equals(modern, legacy)) {
                throw new GgufFormatException(source + ": '" + path + "' has conflicting rope_type '" + modern
                        + "' and legacy type '" + legacy + "'");
            }
        }
        if (!p.containsKey("rope_type") && p.containsKey("type")) {
            p.put("rope_type", p.get("type"));
        }
        Object type = p.get("rope_type");
        if (type != null && !(type instanceof String)) {
            throw new GgufFormatException(source + ": '" + path + ".rope_type' must be a string, got "
                    + HfConfig.typeName(type));
        }
        if ("su".equals(type)) {
            p.put("rope_type", "longrope");
        } else if ("mrope".equals(type)) {
            if (!p.containsKey("mrope_section")) {
                throw new GgufFormatException(source + ": '" + path
                        + "': legacy rope_type 'mrope' requires 'mrope_section'");
            }
            p.put("rope_type", "default");
        } else if ("telechat3-yarn".equals(type)) {
            if (p.get("factor") == null) {
                throw new GgufFormatException(source + ": '" + path + "': rope_type 'telechat3-yarn' requires 'factor'");
            }
            double factor = HfConfig.toDouble(source, path + ".factor", p.get("factor"));
            if (!(factor > 0) || Double.isInfinite(factor)) {
                throw new GgufFormatException(source + ": '" + path + ".factor' must be a positive finite number for "
                        + "rope_type 'telechat3-yarn', got " + factor);
            }
            p.put("rope_type", "yarn");
            p.put("attention_factor", 0.07 * Math.log(factor) + 1.0);
            p.remove("type");
        }
    }

    /** The parameters of full attention layers: {@code full_attention} when nested, else the whole object. */
    public Params global() {
        return global;
    }

    /** The parameters of sliding-window layers ({@code sliding_attention}), when present. */
    public Optional<Params> sliding() {
        return layer("sliding_attention");
    }

    /** The parameters nested under layer type {@code layerType}, when present. */
    public Optional<Params> layer(String layerType) {
        Object v = params.get(layerType);
        return v instanceof Map
                ? Optional.of(new Params(HfConfig.asObject(v), source, "rope_parameters." + layerType))
                : Optional.empty();
    }

    /** Whether the configured parameter object is nested by layer type (decided before the aliases are mirrored). */
    public boolean nested() {
        return nested;
    }

    /** The whole normalized parameter object; unmodifiable at every depth. */
    public Map<String, Object> raw() {
        return params;
    }

    /** The global theta ({@code rope_theta} of {@link #global()}). */
    public OptionalDouble theta() {
        return global.theta();
    }

    /** The sliding-window theta ({@code sliding_attention.rope_theta}). */
    public OptionalDouble localTheta() {
        Optional<Params> s = sliding();
        return s.isPresent() ? s.get().theta() : OptionalDouble.empty();
    }

    /**
     * The M-RoPE sections: {@code mrope_section} (or the legacy {@code xdrope_section}) of the top-level parameter
     * object, as the converter reads it; the converter pads to 4 entries.
     */
    public Optional<List<Integer>> mropeSection() {
        Params top = new Params(params, source, "rope_parameters");
        Optional<List<Integer>> s = top.intList("mrope_section");
        return s.isPresent() ? s : top.intList("xdrope_section");
    }

    @Override
    public String toString() {
        return "HfRope" + params;
    }

    /** One RoPE parameter object with typed accessors; absent and null members are empty. */
    public static final class Params {
        private final Map<String, Object> map;
        private final String source;
        private final String path;

        Params(Map<String, Object> map, String source, String path) {
            this.map = map;
            this.source = source;
            this.path = path;
        }

        /** The members; unmodifiable at every depth. */
        public Map<String, Object> raw() {
            return map;
        }

        /** The member {@code key}, empty when absent or null. */
        public Optional<Object> get(String key) {
            return Optional.ofNullable(map.get(key));
        }

        /** {@code rope_type} after legacy normalization. */
        public Optional<String> type() {
            Object v = map.get("rope_type");
            if (v != null && !(v instanceof String)) {
                throw new GgufFormatException(source + ": '" + path + ".rope_type' must be a string");
            }
            return Optional.ofNullable((String) v);
        }

        public OptionalDouble factor() { return number("factor"); }

        public OptionalInt originalMaxPositionEmbeddings() { return integer("original_max_position_embeddings"); }

        public OptionalDouble theta() { return number("rope_theta"); }

        public OptionalDouble partialRotaryFactor() { return number("partial_rotary_factor"); }

        /** YaRN {@code truncate}. */
        public Optional<Boolean> truncate() {
            Object v = map.get("truncate");
            if (v != null && !(v instanceof Boolean)) {
                throw new GgufFormatException(source + ": '" + path + ".truncate' must be a boolean");
            }
            return Optional.ofNullable((Boolean) v);
        }

        public OptionalDouble extrapolationFactor() { return number("extrapolation_factor"); }

        /** YaRN attention factor: {@code attention_factor}, else {@code attn_factor}. */
        public OptionalDouble attentionFactor() {
            return map.containsKey("attention_factor") ? number("attention_factor") : number("attn_factor");
        }

        public OptionalDouble betaFast() { return number("beta_fast"); }

        public OptionalDouble betaSlow() { return number("beta_slow"); }

        /** Llama 3 {@code low_freq_factor}. */
        public OptionalDouble lowFreqFactor() { return number("low_freq_factor"); }

        /** Llama 3 {@code high_freq_factor}. */
        public OptionalDouble highFreqFactor() { return number("high_freq_factor"); }

        /** LongRoPE {@code long_factor}. */
        public Optional<List<Double>> longFactor() { return numberList("long_factor"); }

        /** LongRoPE {@code short_factor}. */
        public Optional<List<Double>> shortFactor() { return numberList("short_factor"); }

        public OptionalDouble mscale() { return number("mscale"); }

        public OptionalDouble mscaleAllDim() { return number("mscale_all_dim"); }

        /** Dynamic NTK {@code alpha} (HunYuan). */
        public OptionalDouble alpha() { return number("alpha"); }

        /** {@code mrope_section}, else the legacy {@code xdrope_section}, of this object. */
        public Optional<List<Integer>> mropeSection() {
            Optional<List<Integer>> s = intList("mrope_section");
            return s.isPresent() ? s : intList("xdrope_section");
        }

        private OptionalDouble number(String key) {
            Object v = map.get(key);
            return v == null ? OptionalDouble.empty()
                    : OptionalDouble.of(HfConfig.toDouble(source, path + "." + key, v));
        }

        private OptionalInt integer(String key) {
            Object v = map.get(key);
            if (v == null) {
                return OptionalInt.empty();
            }
            long l = HfConfig.toLong(source, path + "." + key, v);
            if (l < 0 || l > Integer.MAX_VALUE) {
                throw new GgufFormatException(source + ": '" + path + "." + key + "' value " + l + " is out of range");
            }
            return OptionalInt.of((int) l);
        }

        private Optional<List<Double>> numberList(String key) {
            Object v = map.get(key);
            if (v == null) {
                return Optional.empty();
            }
            if (!(v instanceof List)) {
                throw new GgufFormatException(source + ": '" + path + "." + key + "' must be an array of numbers");
            }
            List<?> items = (List<?>) v;
            List<Double> out = new ArrayList<>(items.size());
            for (int i = 0; i < items.size(); i++) {
                out.add(HfConfig.toDouble(source, path + "." + key + "[" + i + "]", items.get(i)));
            }
            return Optional.of(Collections.unmodifiableList(out));
        }

        private Optional<List<Integer>> intList(String key) {
            Object v = map.get(key);
            if (v == null) {
                return Optional.empty();
            }
            if (!(v instanceof List)) {
                throw new GgufFormatException(source + ": '" + path + "." + key + "' must be an array of integers");
            }
            List<?> items = (List<?>) v;
            List<Integer> out = new ArrayList<>(items.size());
            for (int i = 0; i < items.size(); i++) {
                long l = HfConfig.toLong(source, path + "." + key + "[" + i + "]", items.get(i));
                if (l < 0 || l > HfConfig.MAX_WIDTH) {
                    throw new GgufFormatException(source + ": '" + path + "." + key + "[" + i + "]' value " + l
                            + " is outside [0, " + HfConfig.MAX_WIDTH + "]");
                }
                out.add((int) l);
            }
            return Optional.of(Collections.unmodifiableList(out));
        }

        @Override
        public String toString() {
            return path + map;
        }
    }
}
