package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.hf.config.QuantRule.ModuleMatcher;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * vLLM's online quantization configuration ({@code --quantization-config}, V/config/quantization.py
 * {@code QuantizationConfigArgs}) and the per-layer choice its {@code OnlineQuantizationConfig} makes
 * (V/model_executor/layers/quantization/online/base.py):
 * <ul>
 *   <li>{@code linear} and {@code moe}: the weight spec of the linear layers ({@code LinearBase}) and of the routed
 *       experts ({@code RoutedExperts}). Each is an online shorthand name (its spec of that layer kind), a
 *       {@code QUANT_KEY_NAMES} weight name, or an object {@code {"weight": ..., "activation": ...}}; a null weight
 *       leaves the kind unquantized.</li>
 *   <li>{@code ignore}: layers left unquantized, each an exact name, a {@code re:} regex ({@code re.match}) or an
 *       {@code fnmatch} pattern.</li>
 *   <li>{@code targets}: per-layer shorthands keyed by the same patterns, exclusive with {@code linear} and
 *       {@code moe}; a layer no pattern matches stays unquantized.</li>
 * </ul>
 * A fused layer ({@code qkv_proj}, {@code gate_up_proj}) matches a pattern by its own name or by all of its shard
 * names, as vLLM's {@code find_matching_patterns}; shards that disagree are an error. Activation overrides are
 * refused: NeuralFree quantizes weights only. Errors are {@link IllegalArgumentException}s naming the member.
 * Immutable.
 */
public final class OnlineQuantConfig {

    /** The layer classes vLLM consults online quantization for. */
    public enum LayerKind {
        /** {@code LinearBase}: the linear layers, the shared experts included. */
        LINEAR,
        /** {@code RoutedExperts}: the stacked routed experts of a mixture-of-experts layer. */
        MOE,
        /** {@code VocabParallelEmbedding} and {@code ParallelLMHead}: online quantization never applies. */
        EMBEDDING
    }

    /** The weight quantization keys a spec names: {@code QUANT_KEY_NAMES}, plus the shorthands' static FP4 keys. */
    private enum WeightKey {
        FP8_STATIC_TENSOR("fp8_per_tensor_static"),
        FP8_DYNAMIC_TENSOR("fp8_per_tensor_dynamic"),
        FP8_DYNAMIC_TOKEN("fp8_per_token"),
        FP8_STATIC_CHANNEL("fp8_per_channel_static"),
        FP8_STATIC_BLOCK("fp8_per_block_static"),
        FP8_DYNAMIC_BLOCK("fp8_per_block_dynamic"),
        MXFP8_DYNAMIC("mxfp8"),
        MXFP4_DYNAMIC("mxfp4"),
        NVFP4_DYNAMIC_TOKEN("nvfp4_per_token"),
        INT8_STATIC_CHANNEL("int8_per_channel_static"),
        INT8_DYNAMIC_GROUP32("int8_per_group_32"),
        INT4_STATIC_GROUP32("int4_per_group_32"),
        /** {@code kMxfp4Static}, the weight of the {@code mxfp4} shorthand. */
        MXFP4_STATIC(null),
        /** {@code kNvfp4Static}, the weight of the {@code nvfp4_per_token} shorthand. */
        NVFP4_STATIC(null);

        /** The {@code QUANT_KEY_NAMES} name, or null for a key only a shorthand reaches. */
        final String keyName;

        WeightKey(String keyName) {
            this.keyName = keyName;
        }

        String label() {
            return keyName != null ? keyName : name().toLowerCase(Locale.ROOT);
        }

        static WeightKey named(String name) {
            for (WeightKey k : values()) {
                if (name.equals(k.keyName)) {
                    return k;
                }
            }
            return null;
        }
    }

    /** One online shorthand of {@code _ONLINE_SHORTHANDS}: the weight of each layer kind, null where it has none. */
    private static final class Shorthand {
        final WeightKey linear;
        final WeightKey moe;

        Shorthand(WeightKey linear, WeightKey moe) {
            this.linear = linear;
            this.moe = moe;
        }
    }

    private static final Map<String, Shorthand> SHORTHANDS;

    static {
        Map<String, Shorthand> m = new LinkedHashMap<>();
        m.put("fp8_per_tensor", new Shorthand(WeightKey.FP8_STATIC_TENSOR, WeightKey.FP8_STATIC_TENSOR));
        m.put("fp8_per_block", new Shorthand(WeightKey.FP8_STATIC_BLOCK, WeightKey.FP8_STATIC_BLOCK));
        m.put("fp8_per_channel", new Shorthand(WeightKey.FP8_STATIC_CHANNEL, WeightKey.FP8_STATIC_CHANNEL));
        m.put("mxfp8", new Shorthand(WeightKey.MXFP8_DYNAMIC, WeightKey.MXFP8_DYNAMIC));
        m.put("mxfp4", new Shorthand(WeightKey.MXFP4_STATIC, WeightKey.MXFP4_STATIC));
        m.put("int8_per_channel_weight_only", new Shorthand(null, WeightKey.INT8_STATIC_CHANNEL));
        m.put("nvfp4_per_token", new Shorthand(null, WeightKey.NVFP4_STATIC));
        SHORTHANDS = Collections.unmodifiableMap(m);
    }

    /**
     * The fused layers of vLLM's {@code packed_modules_mapping} tables, each with its shard modules in fusion order:
     * the models that fuse them name the fused layer, and patterns may match either name.
     */
    private static final String[][] FUSED = {
        {"qkv_proj", "q_proj", "k_proj", "v_proj"},
        {"qkv_proj", "query", "key", "value"},
        {"gate_up_proj", "gate_proj", "up_proj"},
        {"gate_up_proj", "w1", "w3"},
        {"fused_qkv_a_proj", "q_a_proj", "kv_a_proj_with_mqa"},
        {"in_proj_qkvz", "in_proj_qkv", "in_proj_z"},
        {"in_proj_ba", "in_proj_b", "in_proj_a"},
    };

    private static final String WEIGHT_SUFFIX = ".weight";

    private final WeightSpec linear;
    private final WeightSpec moe;
    private final List<String> ignore;
    private final Map<String, String> targets;
    private final Map<String, NameMatcher> ignoreMatchers;
    private final Map<String, NameMatcher> targetMatchers;

    private OnlineQuantConfig(WeightSpec linear, WeightSpec moe, List<String> ignore, Map<String, String> targets) {
        this.linear = linear;
        this.moe = moe;
        this.ignore = Collections.unmodifiableList(new ArrayList<>(ignore));
        this.targets = targets == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(targets));
        if (this.targets != null && (linear != null || moe != null)) {
            throw new IllegalArgumentException("quantization_config.targets is exclusive with quantization_config"
                    + ".linear and .moe; got targets " + this.targets + ", linear " + linear + ", moe " + moe);
        }
        this.ignoreMatchers = matchers(this.ignore);
        this.targetMatchers = this.targets == null ? Collections.emptyMap() : matchers(this.targets.keySet());
    }

    private static Map<String, NameMatcher> matchers(Iterable<String> patterns) {
        Map<String, NameMatcher> out = new LinkedHashMap<>();
        for (String p : patterns) {
            out.put(p, NameMatcher.of(p));
        }
        return Collections.unmodifiableMap(out);
    }

    // ------------------------------------------------------------------ construction

    /**
     * The configuration a shorthand target stands for (vLLM's {@code _ONLINE_SHORTHANDS}; {@code experts_int8} is
     * the routed-expert int8 of {@code int8_per_channel_weight_only}); null for {@link OnlineQuant#NONE},
     * {@link OnlineQuant#ONLINE} and the llama-quantize block types, which are no vLLM shorthand.
     */
    public static OnlineQuantConfig shorthand(OnlineQuant q) {
        Objects.requireNonNull(q, "q");
        String name = q == OnlineQuant.EXPERTS_INT8 ? "int8_per_channel_weight_only"
                : q.name().toLowerCase(Locale.ROOT);
        Shorthand s = SHORTHANDS.get(name);
        if (s == null) {
            return null;
        }
        return new OnlineQuantConfig(s.linear == null ? null : new WeightSpec(s.linear),
                s.moe == null ? null : new WeightSpec(s.moe), Collections.emptyList(), null);
    }

    /**
     * The configuration in the JSON object {@code json} (the {@link com.eonmux.neuralfree.utils.json.Json} value
     * model). With {@code checkpoint}, {@code json} is a checkpoint's quantization_config, whose
     * {@code quant_method} must be {@code online}; otherwise any member but {@code linear}, {@code moe},
     * {@code ignore} and {@code targets} is refused.
     *
     * @param where the source named in errors, for example {@code neuralfree.quant.config}
     * @throws IllegalArgumentException naming {@code where} and the member for an invalid configuration
     */
    public static OnlineQuantConfig parse(Map<String, ?> json, String where, boolean checkpoint) {
        Objects.requireNonNull(json, "json");
        Objects.requireNonNull(where, "where");
        try {
            for (String key : json.keySet()) {
                boolean known = key.equals("linear") || key.equals("moe") || key.equals("ignore")
                        || key.equals("targets") || (checkpoint && key.equals("quant_method"));
                if (!known) {
                    throw new IllegalArgumentException("unknown member '" + key + "'; the members are linear, moe,"
                            + " ignore and targets");
                }
            }
            Object method = json.get("quant_method");
            if (checkpoint && !(method instanceof String && ((String) method).equalsIgnoreCase("online"))) {
                throw new IllegalArgumentException("quant_method must be 'online', got " + method);
            }
            return new OnlineQuantConfig(spec(json.get("linear"), "linear"), spec(json.get("moe"), "moe"),
                    ignoreList(json.get("ignore")), targetMap(json.get("targets")));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(where + ": " + e.getMessage(), e);
        }
    }

    /**
     * This configuration over {@code base}, as vLLM's {@code resolve_quantization_config} composes
     * {@code --quantization-config} over the {@code --quantization} shorthand: every member this one sets wins (an
     * empty {@code ignore} or {@code targets} counts as unset), the others come from {@code base}.
     *
     * @throws IllegalArgumentException when the result sets {@code targets} together with {@code linear} or
     *                                  {@code moe}
     */
    public OnlineQuantConfig over(OnlineQuantConfig base) {
        if (base == null) {
            return this;
        }
        return new OnlineQuantConfig(linear != null ? linear : base.linear, moe != null ? moe : base.moe,
                !ignore.isEmpty() ? ignore : base.ignore,
                targets != null && !targets.isEmpty() ? targets : base.targets);
    }

    /**
     * The configuration a load quantizes with: {@code config} over the shorthand of {@code q}, and {@code config}
     * alone for {@link OnlineQuant#ONLINE} or {@link OnlineQuant#NONE}; null for {@link OnlineQuant#NONE} without
     * a configuration and for the llama-quantize block types.
     *
     * @throws IllegalArgumentException when {@code q} is {@link OnlineQuant#ONLINE} without a configuration, a
     *                                  llama-quantize type with one, or the result sets none of {@code linear},
     *                                  {@code moe} and {@code targets}
     */
    public static OnlineQuantConfig effective(OnlineQuant q, OnlineQuantConfig config) {
        Objects.requireNonNull(q, "q");
        if (q == OnlineQuant.NONE && config == null) {
            return null;
        }
        if (q != OnlineQuant.NONE && !q.isVllm()) {
            if (config != null) {
                throw new IllegalArgumentException("a quantization config configures vLLM's online methods; "
                        + q.name().toLowerCase(Locale.ROOT) + " is a llama-quantize type");
            }
            return null;
        }
        OnlineQuantConfig base = shorthand(q);
        if (config == null && base == null) {
            throw new IllegalArgumentException("the online method quantizes as a quantization config says, and"
                    + " none is given");
        }
        OnlineQuantConfig r = config == null ? base : config.over(base);
        if (r.linear == null && r.moe == null && r.targets == null) {
            throw new IllegalArgumentException("online quantization needs at least one of quantization_config"
                    + ".linear, .moe and .targets");
        }
        return r;
    }

    // ------------------------------------------------------------------ members

    /** Whether the {@code linear} or {@code moe} spec of {@code kind} names a weight (not for {@code targets}). */
    public boolean quantizes(LayerKind kind) {
        WeightSpec s = kind == LayerKind.LINEAR ? linear : kind == LayerKind.MOE ? moe : null;
        return s != null && s.weight != null;
    }

    /** The ignore entries, as written. */
    public List<String> ignore() {
        return ignore;
    }

    /** Whether layer names decide anything: {@code targets} is set or {@code ignore} holds a pattern. */
    public boolean matchesNames() {
        return targets != null || !ignore.isEmpty();
    }

    /** The {@code targets} patterns with their shorthand names, in order; null when unset. */
    public Map<String, String> targets() {
        return targets;
    }

    // ------------------------------------------------------------------ the per-layer choice

    /** The method a selected layer quantizes with and the configuration member that chose it. */
    public static final class Selection {
        private final OnlineQuant target;
        private final String origin;

        Selection(OnlineQuant target, String origin) {
            this.target = target;
            this.origin = origin;
        }

        /** The online method. */
        public OnlineQuant target() {
            return target;
        }

        /** The member that chose it, for example {@code linear: fp8_per_block_static}. */
        public String origin() {
            return origin;
        }

        @Override
        public String toString() {
            return target.name().toLowerCase(Locale.ROOT) + " (from " + origin + ")";
        }
    }

    /**
     * The online method of layer {@code layer} of {@code kind}, as vLLM's {@code resolve_quant_method_cls}; null
     * when the layer stays unquantized. {@code targets} decides alone when set; otherwise the spec of the layer's
     * kind applies unless {@code ignore} matches the layer.
     *
     * @throws IllegalArgumentException for the configurations vLLM refuses at that layer: shards of a fused layer
     *                                  matched differently, a layer in both {@code ignore} and {@code targets},
     *                                  several {@code targets} patterns for one layer, a {@code targets} shorthand
     *                                  without a spec for the kind, an embedding or LM head matched by
     *                                  {@code targets}, and weights with no online method for the kind
     */
    public Selection select(LayerKind kind, Layer layer) {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(layer, "layer");
        if (targets != null) {
            return selectTarget(kind, layer);
        }
        if (kind == LayerKind.EMBEDDING) {
            return null;
        }
        WeightSpec spec = kind == LayerKind.LINEAR ? linear : moe;
        if (ignored(layer)) {
            return null;
        }
        OnlineQuant method = spec == null || spec.weight == null ? null : method(spec.weight, kind, layer);
        return method == null ? null : new Selection(method, (kind == LayerKind.LINEAR ? "linear: " : "moe: ") + spec);
    }

    private Selection selectTarget(LayerKind kind, Layer layer) {
        boolean ignored = ignored(layer);
        String pattern = matchedTarget(layer);
        if (ignored && pattern != null) {
            throw new IllegalArgumentException("layer " + layer.name() + " matches both quantization_config.ignore"
                    + " and quantization_config.targets ('" + pattern + "'); a layer may be in only one");
        }
        if (ignored || pattern == null) {
            return null;
        }
        String shorthand = targets.get(pattern);
        if (kind == LayerKind.EMBEDDING) {
            throw new IllegalArgumentException("layer " + layer.name() + " is matched by quantization_config.targets"
                    + " ('" + pattern + "'), but online quantization does not apply to embeddings and LM heads");
        }
        Shorthand s = SHORTHANDS.get(shorthand);
        WeightKey weight = kind == LayerKind.LINEAR ? s.linear : s.moe;
        if (weight == null) {
            throw new IllegalArgumentException("quantization_config.targets '" + pattern + "' = " + shorthand
                    + " has no spec for " + (kind == LayerKind.LINEAR ? "linear layers" : "routed experts")
                    + " (matched at " + layer.name() + ")");
        }
        return new Selection(method(weight, kind, layer), "targets: " + pattern + " = " + shorthand);
    }

    /**
     * The online method of {@code weight} on a layer of {@code kind} (vLLM's {@code _ONLINE_LINEAR_METHODS} and
     * {@code _ONLINE_MOE_METHODS}); null for {@code int4_per_group_32}, vLLM's load-time requantization of MXFP4
     * checkpoints, which online quantization leaves to the checkpoint method.
     */
    private static OnlineQuant method(WeightKey weight, LayerKind kind, Layer layer) {
        switch (weight) {
            case FP8_STATIC_TENSOR: return OnlineQuant.FP8_PER_TENSOR;
            case FP8_STATIC_BLOCK: return OnlineQuant.FP8_PER_BLOCK;
            case FP8_STATIC_CHANNEL: return OnlineQuant.FP8_PER_CHANNEL;
            case MXFP8_DYNAMIC: return OnlineQuant.MXFP8;
            case MXFP4_STATIC: return OnlineQuant.MXFP4;
            case INT8_STATIC_CHANNEL:
                if (kind == LayerKind.MOE) {
                    return OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY;
                }
                break;
            case NVFP4_STATIC:
                if (kind == LayerKind.MOE) {
                    return OnlineQuant.NVFP4_PER_TOKEN;
                }
                break;
            case INT4_STATIC_GROUP32:
                return null;
            default:
                break;
        }
        throw new IllegalArgumentException("online quantization of " + (kind == LayerKind.LINEAR ? "linear layers"
                : "routed experts") + " with weight " + weight.label() + " is not supported (layer " + layer.name()
                + "); the supported weights are " + (kind == LayerKind.LINEAR
                ? "fp8_per_tensor_static, fp8_per_block_static, fp8_per_channel_static, mxfp8 and the mxfp4 shorthand"
                : "fp8_per_tensor_static, fp8_per_block_static, fp8_per_channel_static, mxfp8,"
                + " int8_per_channel_static and the mxfp4 and nvfp4_per_token shorthands"));
    }

    /** {@code should_ignore_layer}: every shard of the layer matched by {@code ignore}, or none. */
    private boolean ignored(Layer layer) {
        if (ignore.isEmpty()) {
            return false;
        }
        List<Set<String>> per = perShard(layer, ignoreMatchers);
        int hit = 0;
        for (Set<String> s : per) {
            hit += s.isEmpty() ? 0 : 1;
        }
        if (hit > 0 && hit < per.size()) {
            throw new IllegalArgumentException("quantization_config.ignore matches only some shards of the fused"
                    + " layer " + layer.name() + " " + layer.shards() + "; all shards of a fused layer quantize alike");
        }
        return hit > 0;
    }

    /** {@code _find_matching_targets}: the one pattern the layer, or every shard alike, matches; null for none. */
    private String matchedTarget(Layer layer) {
        List<Set<String>> per = perShard(layer, targetMatchers);
        int hit = 0;
        for (Set<String> s : per) {
            hit += s.isEmpty() ? 0 : 1;
        }
        if (hit == 0) {
            return null;
        }
        if (hit < per.size()) {
            throw new IllegalArgumentException("quantization_config.targets matches only some shards of the fused"
                    + " layer " + layer.name() + ": " + per + "; every shard of a fused layer must match a target");
        }
        Set<String> values = new LinkedHashSet<>();
        for (Set<String> s : per) {
            if (s.size() > 1) {
                throw new IllegalArgumentException("several quantization_config.targets patterns match "
                        + layer.name() + ": " + per + "; a layer may match at most one");
            }
            values.add(targets.get(s.iterator().next()));
        }
        if (values.size() > 1) {
            throw new IllegalArgumentException("the shards of the fused layer " + layer.name() + " match"
                    + " quantization_config.targets of different methods: " + per);
        }
        return per.get(0).iterator().next();
    }

    /**
     * {@code find_matching_patterns(..., use_fnmatch=True)}: the patterns matching the layer's own name, or, when
     * none does and the layer is fused, those matching each shard name.
     */
    private static List<Set<String>> perShard(Layer layer, Map<String, NameMatcher> patterns) {
        Set<String> direct = matching(layer.name(), patterns);
        if (!direct.isEmpty() || layer.shards().isEmpty()) {
            return Collections.singletonList(direct);
        }
        List<Set<String>> out = new ArrayList<>();
        for (String shard : layer.shards()) {
            out.add(matching(shard, patterns));
        }
        return out;
    }

    private static Set<String> matching(String name, Map<String, NameMatcher> patterns) {
        Set<String> out = new LinkedHashSet<>();
        for (Map.Entry<String, NameMatcher> p : patterns.entrySet()) {
            if (p.getValue().matches(name)) {
                out.add(p.getKey());
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ layer names

    /**
     * A layer as vLLM names it: its module name and, for a fused layer, the names of its shards. Immutable.
     */
    public static final class Layer {
        private final String name;
        private final List<String> shards;

        private Layer(String name, List<String> shards) {
            this.name = name;
            this.shards = Collections.unmodifiableList(new ArrayList<>(shards));
        }

        /** An unfused layer named {@code name}. */
        public static Layer named(String name) {
            return new Layer(Objects.requireNonNull(name, "name"), Collections.emptyList());
        }

        /**
         * The layer of checkpoint tensor {@code hfTensor} of {@code kind}, with checkpoint names as vLLM's names
         * ({@link #of(String, LayerKind, java.util.function.Predicate, VllmWeightsMapper)} with
         * {@link VllmWeightsMapper#IDENTITY}).
         */
        public static Layer of(String hfTensor, LayerKind kind, java.util.function.Predicate<String> hasTensor) {
            return of(hfTensor, kind, hasTensor, VllmWeightsMapper.IDENTITY);
        }

        /**
         * The layer of checkpoint tensor {@code hfTensor} of {@code kind}, named as the vLLM module it loads into:
         * the name {@code mapper} gives the tensor (the checkpoint name when vLLM's model drops it, a weight another
         * vLLM model loads); the routed experts are the module up to the {@code experts} component; a module the
         * mapper fuses ({@code stacked}) is its fused layer with the mapper's shards; a shard whose sibling shards the
         * checkpoint stores too ({@code hasTensor} of their checkpoint {@code .weight}) is its fused layer, and a
         * fused checkpoint module keeps its name with its shard names; anything else is its module.
         *
         * @throws IllegalArgumentException when {@code mapper} is {@link VllmWeightsMapper#unresolved()}
         */
        public static Layer of(String hfTensor, LayerKind kind, java.util.function.Predicate<String> hasTensor,
                               VllmWeightsMapper mapper) {
            Objects.requireNonNull(hfTensor, "hfTensor");
            Objects.requireNonNull(mapper, "mapper");
            if (mapper.unresolved()) {
                throw new IllegalArgumentException("vLLM names the layers of " + mapper.architecture()
                        + " with code (" + mapper.unresolvedReason() + "), so quantization_config patterns cannot"
                        + " be matched to them");
            }
            String mapped = mapper.map(hfTensor);
            String module = stripWeight(mapped == null ? hfTensor : mapped);
            if (kind == LayerKind.MOE) {
                int at = module.lastIndexOf(".experts.");
                if (at >= 0) {
                    return named(module.substring(0, at + ".experts".length()));
                }
                return named(module);
            }
            int dot = module.lastIndexOf('.');
            String parent = dot < 0 ? "" : module.substring(0, dot + 1);
            String proj = module.substring(dot + 1);
            List<String> stacked = mapper.stackedShards(module);
            if (!stacked.isEmpty()) {
                return new Layer(module, stacked);
            }
            for (String[] f : FUSED) {
                if (proj.equals(f[0])) {
                    return new Layer(module, shardNames(parent, f));
                }
            }
            String hfModule = stripWeight(hfTensor);
            int hfDot = hfModule.lastIndexOf('.');
            String hfParent = hfDot < 0 ? "" : hfModule.substring(0, hfDot + 1);
            for (String[] f : FUSED) {
                if (indexOf(f, proj) > 0 && allPresent(hfParent, f, hasTensor)) {
                    return new Layer(parent + f[0], shardNames(parent, f));
                }
            }
            return named(module);
        }

        private static String stripWeight(String tensor) {
            return tensor.endsWith(WEIGHT_SUFFIX) ? tensor.substring(0, tensor.length() - WEIGHT_SUFFIX.length())
                    : tensor;
        }

        private static int indexOf(String[] f, String proj) {
            for (int i = 1; i < f.length; i++) {
                if (f[i].equals(proj)) {
                    return i;
                }
            }
            return -1;
        }

        private static boolean allPresent(String parent, String[] f, java.util.function.Predicate<String> has) {
            for (int i = 1; i < f.length; i++) {
                if (!has.test(parent + f[i] + WEIGHT_SUFFIX)) {
                    return false;
                }
            }
            return true;
        }

        private static List<String> shardNames(String parent, String[] f) {
            List<String> out = new ArrayList<>();
            for (int i = 1; i < f.length; i++) {
                out.add(parent + f[i]);
            }
            return out;
        }

        /** The layer name ({@code model.layers.0.self_attn.qkv_proj}). */
        public String name() {
            return name;
        }

        /** The shard names of a fused layer, in fusion order; empty for an unfused layer. */
        public List<String> shards() {
            return shards;
        }

        @Override
        public String toString() {
            return shards.isEmpty() ? name : name + " " + shards;
        }
    }

    // ------------------------------------------------------------------ parsing

    /** A {@code linear} or {@code moe} spec: its weight, null for a kind left unquantized. */
    private static final class WeightSpec {
        final WeightKey weight;

        WeightSpec(WeightKey weight) {
            this.weight = weight;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof WeightSpec && weight == ((WeightSpec) o).weight;
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(weight);
        }

        @Override
        public String toString() {
            return weight == null ? "None" : weight.label();
        }
    }

    /** {@code _coerce_spec}: a shorthand's spec of {@code field}, a weight name, or a weight/activation object. */
    private static WeightSpec spec(Object v, String field) {
        if (v == null) {
            return null;
        }
        if (v instanceof String) {
            String s = (String) v;
            Shorthand sh = SHORTHANDS.get(s);
            if (sh != null) {
                WeightKey k = field.equals("linear") ? sh.linear : sh.moe;
                if (k == null) {
                    throw new IllegalArgumentException(field + ": online shorthand '" + s + "' has no " + field
                            + " spec");
                }
                return new WeightSpec(k);
            }
            return new WeightSpec(weightKey(s, field));
        }
        if (!(v instanceof Map)) {
            throw new IllegalArgumentException(field + " must be a string or an object, got " + HfConfig.typeName(v));
        }
        Map<?, ?> m = (Map<?, ?>) v;
        for (Object key : m.keySet()) {
            if (!"weight".equals(key) && !"activation".equals(key)) {
                throw new IllegalArgumentException(field + ": unknown member '" + key + "'; a spec has weight and"
                        + " activation");
            }
        }
        if (m.get("activation") != null) {
            throw new IllegalArgumentException(field + ".activation=" + m.get("activation") + " is not supported:"
                    + " online quantization here quantizes weights only");
        }
        Object w = m.get("weight");
        if (w != null && !(w instanceof String)) {
            throw new IllegalArgumentException(field + ".weight must be a string, got " + HfConfig.typeName(w));
        }
        return new WeightSpec(w == null ? null : weightKey((String) w, field + ".weight"));
    }

    private static WeightKey weightKey(String name, String where) {
        WeightKey k = WeightKey.named(name);
        if (k == null) {
            List<String> names = new ArrayList<>();
            for (WeightKey key : WeightKey.values()) {
                if (key.keyName != null) {
                    names.add(key.keyName);
                }
            }
            throw new IllegalArgumentException(where + ": unknown quantization name '" + name + "'; use an online"
                    + " shorthand " + SHORTHANDS.keySet() + " or one of " + names);
        }
        return k;
    }

    private static List<String> ignoreList(Object v) {
        if (v == null) {
            return Collections.emptyList();
        }
        if (!(v instanceof List)) {
            throw new IllegalArgumentException("ignore must be an array of strings, got " + HfConfig.typeName(v));
        }
        List<String> out = new ArrayList<>();
        for (Object o : (List<?>) v) {
            if (!(o instanceof String)) {
                throw new IllegalArgumentException("ignore entries must be strings, got " + HfConfig.typeName(o));
            }
            out.add((String) o);
        }
        return out;
    }

    private static Map<String, String> targetMap(Object v) {
        if (v == null) {
            return null;
        }
        if (!(v instanceof Map)) {
            throw new IllegalArgumentException("targets must be an object, got " + HfConfig.typeName(v));
        }
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
            Object value = e.getValue();
            if (!(value instanceof String) || !SHORTHANDS.containsKey(value)) {
                throw new IllegalArgumentException("targets['" + e.getKey() + "'] = " + value + " is not an online"
                        + " shorthand; use one of " + SHORTHANDS.keySet());
            }
            out.put(String.valueOf(e.getKey()), (String) value);
        }
        return out;
    }

    /** {@code is_equal_or_regex_match(..., use_fnmatch=True)}: {@code re:} regex, else equal or fnmatch. */
    private static final class NameMatcher {
        private final ModuleMatcher regex;
        private final ModuleMatcher glob;
        private final String exact;

        private NameMatcher(String pattern) {
            if (pattern.startsWith("re:")) {
                this.regex = ModuleMatcher.regex(pattern.substring(3));
                this.glob = null;
                this.exact = null;
            } else {
                this.regex = null;
                this.glob = ModuleMatcher.glob(pattern);
                this.exact = pattern;
            }
        }

        static NameMatcher of(String pattern) {
            try {
                return new NameMatcher(pattern);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("pattern '" + pattern + "' is invalid: " + e.getMessage(), e);
            }
        }

        boolean matches(String name) {
            return regex != null ? regex.matches(name) : exact.equals(name) || glob.matches(name);
        }
    }

    // ------------------------------------------------------------------ value semantics

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof OnlineQuantConfig)) {
            return false;
        }
        OnlineQuantConfig c = (OnlineQuantConfig) o;
        return Objects.equals(linear, c.linear) && Objects.equals(moe, c.moe) && ignore.equals(c.ignore)
                && Objects.equals(targets, c.targets);
    }

    @Override
    public int hashCode() {
        return Objects.hash(linear, moe, ignore, targets);
    }

    @Override
    public String toString() {
        return "OnlineQuantConfig{linear=" + linear + ", moe=" + moe + ", ignore=" + ignore + ", targets=" + targets
                + "}";
    }
}
