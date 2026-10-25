package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The compressed-tensors {@code transform_config} of a quantization config (compressed-tensors >= 0.10:
 * transform_config.py, transform_scheme.py, transform_args.py): named schemes, each a transform type applied at
 * module locations. Parsing is as strict as the pydantic models (unknown members are errors); whether a scheme can be
 * honored is decided by its consumer. Immutable.
 */
public final class HfTransformConfig {

    /** Registered transform types. */
    public static final Set<String> TYPES = Set.of("hadamard", "random-hadamard", "random-matrix", "matrix");

    private static final Set<String> CONFIG_KEYS = Set.of("config_groups");
    private static final Set<String> SCHEME_KEYS = Set.of("type", "apply", "randomize", "requires_grad", "head_dim",
            "precision");
    private static final Set<String> APPLY_KEYS = Set.of("targets", "location", "inverse", "ignore");

    /** Where a transform is applied (TransformLocation). */
    public enum Location {
        /** Online, on the module's input activations. */
        INPUT,
        /** Offline, fused into the weight's input side. */
        WEIGHT_INPUT,
        /** Offline, fused into the weight's output side. */
        WEIGHT_OUTPUT,
        /** Online, on the module's output activations. */
        OUTPUT,
        /** Online, on the key cache values. */
        K_CACHE,
        /** Online, on the attention query values. */
        Q_ATTN;

        /** The location named {@code value} in the config. */
        static Location parse(String value, String where) {
            for (Location l : values()) {
                if (l.name().toLowerCase(Locale.ROOT).equals(value)) {
                    return l;
                }
            }
            throw new GgufFormatException(where + ": unknown transform location '" + value + "'");
        }

        /** Whether the transform runs on activations at inference time. */
        public boolean isOnline() {
            return this != WEIGHT_INPUT && this != WEIGHT_OUTPUT;
        }
    }

    /** One application of a scheme (TransformArgs). */
    public static final class Apply {
        private final Location location;
        private final List<String> targets;
        private final List<String> ignore;
        private final boolean inverse;

        Apply(Location location, List<String> targets, List<String> ignore, boolean inverse) {
            this.location = location;
            this.targets = Collections.unmodifiableList(new ArrayList<>(targets));
            this.ignore = Collections.unmodifiableList(new ArrayList<>(ignore));
            this.inverse = inverse;
        }

        public Location location() { return location; }

        /** Target modules: names, {@code re:} regexes or class names. */
        public List<String> targets() { return targets; }

        /** Modules excluded from the targets. */
        public List<String> ignore() { return ignore; }

        /** Whether the inverse transform is applied. */
        public boolean inverse() { return inverse; }

        @Override
        public String toString() {
            return location + (inverse ? " inverse" : "") + " targets=" + targets
                    + (ignore.isEmpty() ? "" : " ignore=" + ignore);
        }
    }

    /** One named transform scheme (TransformScheme). */
    public static final class Scheme {
        private final String name;
        private final String type;
        private final Integer headDim;
        private final boolean randomize;
        private final String precision;
        private final List<Apply> applies;

        Scheme(String name, String type, Integer headDim, boolean randomize, String precision, List<Apply> applies) {
            this.name = name;
            this.type = type;
            this.headDim = headDim;
            this.randomize = randomize;
            this.precision = precision;
            this.applies = Collections.unmodifiableList(new ArrayList<>(applies));
        }

        /** The config group name. */
        public String name() { return name; }

        /** One of {@link #TYPES}. */
        public String type() { return type; }

        /** The block size of a block-diagonal transform, when set. */
        public OptionalInt headDim() { return headDim == null ? OptionalInt.empty() : OptionalInt.of(headDim); }

        /** Whether each application has its own randomized weights. */
        public boolean randomize() { return randomize; }

        /** The configured online precision ({@code precision}), or null for the default float32. */
        public String precision() { return precision; }

        public List<Apply> applies() { return applies; }

        @Override
        public String toString() {
            return name + ":" + type + (headDim == null ? "" : " head_dim=" + headDim)
                    + (randomize ? " randomized" : "") + " " + applies;
        }
    }

    private final List<Scheme> schemes;

    private HfTransformConfig(List<Scheme> schemes) {
        this.schemes = Collections.unmodifiableList(new ArrayList<>(schemes));
    }

    /**
     * Parses {@code transformConfig}, the value of {@code quantization_config.transform_config}.
     *
     * @param where names the member in messages
     * @throws GgufFormatException for a malformed config, an unknown member, type or location
     */
    public static HfTransformConfig parse(Object transformConfig, String where) {
        Map<String, Object> root = object(transformConfig, where);
        checkKeys(root, CONFIG_KEYS, where);
        Map<String, Object> groups = object(root.get("config_groups"), where + ".config_groups");
        List<Scheme> schemes = new ArrayList<>();
        for (Map.Entry<String, Object> g : groups.entrySet()) {
            schemes.add(scheme(g.getKey(), object(g.getValue(), where + ".config_groups." + g.getKey()),
                    where + ".config_groups." + g.getKey()));
        }
        return new HfTransformConfig(schemes);
    }

    private static Scheme scheme(String name, Map<String, Object> s, String where) {
        checkKeys(s, SCHEME_KEYS, where);
        String type = QuantArgs.strOr(s, "type", null, where);
        if (type == null || !TYPES.contains(type)) {
            throw new GgufFormatException(where + ": unsupported transform type '" + type + "'; known types are "
                    + TYPES);
        }
        Integer headDim = s.get("head_dim") == null ? null : QuantArgs.intOr(s, "head_dim", 0, where);
        if (headDim != null && (headDim <= 0 || headDim > HfConfig.MAX_WIDTH)) {
            throw new GgufFormatException(where + ".head_dim " + headDim + " is outside [1, " + HfConfig.MAX_WIDTH + "]");
        }
        Object applyValue = s.get("apply");
        List<Apply> applies = new ArrayList<>();
        if (applyValue != null) {
            if (!(applyValue instanceof List)) {
                throw new GgufFormatException(where + ".apply must be an array, got " + HfConfig.typeName(applyValue));
            }
            List<?> list = (List<?>) applyValue;
            for (int i = 0; i < list.size(); i++) {
                applies.add(apply(object(list.get(i), where + ".apply[" + i + "]"), where + ".apply[" + i + "]"));
            }
        }
        QuantArgs.boolOr(s, "requires_grad", false, where);
        return new Scheme(name, type, headDim, QuantArgs.boolOr(s, "randomize", false, where),
                QuantArgs.strOr(s, "precision", null, where), applies);
    }

    private static Apply apply(Map<String, Object> a, String where) {
        checkKeys(a, APPLY_KEYS, where);
        if (!a.containsKey("targets")) {
            throw new GgufFormatException(where + ": required member 'targets' is missing");
        }
        String location = QuantArgs.strOr(a, "location", null, where);
        if (location == null) {
            throw new GgufFormatException(where + ": required member 'location' is missing");
        }
        return new Apply(Location.parse(location, where), strings(a.get("targets"), where + ".targets"),
                a.get("ignore") == null ? List.of() : strings(a.get("ignore"), where + ".ignore"),
                QuantArgs.boolOr(a, "inverse", false, where));
    }

    /** A string or a list of strings (TransformArgs wrap_singleton). */
    private static List<String> strings(Object v, String where) {
        if (v instanceof String) {
            return List.of((String) v);
        }
        if (!(v instanceof List)) {
            throw new GgufFormatException(where + " must be a string or an array of strings, got "
                    + HfConfig.typeName(v));
        }
        List<String> out = new ArrayList<>();
        for (Object item : (List<?>) v) {
            if (!(item instanceof String)) {
                throw new GgufFormatException(where + " entries must be strings, got " + HfConfig.typeName(item));
            }
            out.add((String) item);
        }
        return out;
    }

    private static Map<String, Object> object(Object v, String where) {
        if (!(v instanceof Map)) {
            throw new GgufFormatException(where + " must be an object, got " + HfConfig.typeName(v));
        }
        return HfConfig.asObject(v);
    }

    private static void checkKeys(Map<String, Object> m, Set<String> allowed, String where) {
        for (String k : m.keySet()) {
            if (!allowed.contains(k)) {
                throw new GgufFormatException(where + ": unknown member '" + k + "'; allowed members are " + allowed);
            }
        }
    }

    /** The schemes in config order. */
    public List<Scheme> schemes() {
        return schemes;
    }

    /** Whether the config declares no scheme. */
    public boolean isEmpty() {
        return schemes.isEmpty();
    }

    @Override
    public String toString() {
        return "HfTransformConfig" + schemes;
    }
}
