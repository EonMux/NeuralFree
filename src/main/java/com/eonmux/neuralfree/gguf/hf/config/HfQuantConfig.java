package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.QuantRule.ModuleMatcher;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;

/**
 * The quantization of a checkpoint, found and normalized as vLLM does
 * (V/transformers_utils/model_arch_config_convertor.py, V/config/model.py _verify_quantization), with each method's
 * configuration reduced to an ignore list and ordered {@link QuantRule}s.
 *
 * <p>Sources, first present wins: {@code quantization_config}; {@code compression_config};
 * {@code text_config.quantization_config}; {@code hf_quant_config.json} (ModelOpt &lt;= 0.29); an MLX
 * {@code quantization} (only {@code bits}, {@code group_size}, {@code mode} and per-module entries, no
 * {@code quant_method}); the {@code quantization} block of a Mistral checkpoint, in either its native
 * {@code params.json} or its Hugging Face layout. A
 * {@code quantization_config} of MLX shape is MLX as well, at every bit width. A ModelOpt MIXED_PRECISION config
 * without {@code quantized_layers} defers to {@code hf_quant_config.json} when that file is ModelOpt's.</p>
 *
 * <p>For a module, {@link #argsFor} applies {@link #isIgnored} and then the first matching rule. Errors are
 * {@link GgufFormatException}s naming the file and member, except unreadable files ({@link IOException}). Immutable.</p>
 */
public final class HfQuantConfig {

    private static final Logger LOG = LogManager.getLogger(HfQuantConfig.class);

    /** The ModelOpt &lt;= 0.29 side file. */
    public static final String HF_QUANT_CONFIG_JSON = "hf_quant_config.json";

    private static final Set<String> DEEPSEEK_V4_TYPES = Set.of("deepseek_v4", "deepseek_v4_text", "deepseek_v41",
            "deepseek_v41_text");

    private final HfQuantMethod method;
    private final String methodName;
    private final JsonView view;
    private final String modelOptAlgo;
    private final List<String> ignoreList;
    private final List<ModuleMatcher> ignore;
    private final List<QuantRule> rules;
    private final HfTransformConfig transforms;
    private final OnlineQuantConfig onlineConfig;

    private HfQuantConfig(Assembly a) {
        this.method = a.method;
        this.methodName = a.name;
        this.view = a.view;
        this.modelOptAlgo = a.algo;
        this.ignoreList = Collections.unmodifiableList(new ArrayList<>(a.ignoreList));
        this.ignore = Collections.unmodifiableList(new ArrayList<>(a.ignore));
        this.rules = Collections.unmodifiableList(new ArrayList<>(a.rules));
        this.transforms = a.transforms;
        this.onlineConfig = a.onlineConfig;
    }

    /** An unquantized checkpoint. */
    public static HfQuantConfig none() {
        return new HfQuantConfig(new Assembly(HfQuantMethod.NONE, "", JsonView.empty(HfConfig.CONFIG_JSON,
                "quantization_config")));
    }

    /**
     * The quantization of {@code cfg}'s checkpoint.
     *
     * @throws IOException         when {@code hf_quant_config.json} exists but cannot be read
     * @throws GgufFormatException for unknown, rejected or malformed quantization configurations
     */
    public static HfQuantConfig of(HfConfig cfg) throws IOException {
        Found f = discover(cfg);
        if (f == null) {
            return none();
        }
        try {
            return new HfQuantConfig(assemble(cfg, f));
        } catch (JsonException e) {
            throw new GgufFormatException(f.view.source() + ": " + e.getMessage(), e);
        }
    }

    /** The normalized method. */
    public HfQuantMethod method() { return method; }

    /** The normalized on-disk method name (for example {@code gptq_marlin}); empty for {@link HfQuantMethod#NONE}. */
    public String methodName() { return methodName; }

    /** The configuration the method was read from. */
    public JsonView view() { return view; }

    /** The ModelOpt {@code quant_algo}, upper-cased, for ModelOpt checkpoints. */
    public Optional<String> modelOptAlgo() { return Optional.ofNullable(modelOptAlgo); }

    /** Whether weights are stored quantized (not {@link HfQuantMethod#NONE} or {@link HfQuantMethod#ONLINE}). */
    public boolean isQuantized() { return method != HfQuantMethod.NONE && method != HfQuantMethod.ONLINE; }

    /**
     * The online-quantization target a configuration naming one of vLLM's online shorthands quantizes float
     * weights at load with ({@link HfQuantAliases#onlineTarget(String)}), or null. A serialized method keeps its
     * float modules as stored ({@code fp8} and {@code mxfp8} name FP8 and ModelOpt MXFP8 checkpoints, as in vLLM),
     * and an explicit {@code --quant-online} wins over this default.
     */
    public ModelSourceOptions.OnlineQuant onlineTarget() {
        return method == HfQuantMethod.ONLINE ? HfQuantAliases.onlineTarget(methodName) : null;
    }

    /**
     * The layer selection of a configuration whose quant_method is vLLM's bare {@code online}: its
     * {@code linear}, {@code moe}, {@code ignore} and {@code targets} ({@link OnlineQuantConfig}), which quantize
     * float weights at load unless {@code --quant-online} names another method; null for every other configuration.
     */
    public OnlineQuantConfig onlineConfig() { return onlineConfig; }

    /** The configured ignore entries, as written. */
    public List<String> ignoreList() { return ignoreList; }

    /** Whether HF module {@code module} is excluded by the ignore list, with the method's matching semantics. */
    public boolean isIgnored(String module) {
        for (ModuleMatcher m : ignore) {
            if (m.matches(module)) {
                return true;
            }
        }
        return false;
    }

    /** The rules in priority order; the first matching rule decides. */
    public List<QuantRule> rules() { return rules; }

    /** The storage arguments of {@code module}: empty when it is ignored, excluded by a rule, or matched by none. */
    public Optional<QuantArgs> argsFor(String module) {
        if (isIgnored(module)) {
            return Optional.empty();
        }
        for (QuantRule r : rules) {
            if (r.matches(module)) {
                return Optional.ofNullable(r.args());
            }
        }
        return Optional.empty();
    }

    /** The compressed-tensors {@code transform_config}, when present. */
    public Optional<HfTransformConfig> transforms() { return Optional.ofNullable(transforms); }

    @Override
    public String toString() {
        return "HfQuantConfig{" + method + (methodName.isEmpty() ? "" : " '" + methodName + "'")
                + (modelOptAlgo == null ? "" : " " + modelOptAlgo) + ", ignore=" + ignoreList + ", rules=" + rules
                + (onlineConfig == null ? "" : ", online=" + onlineConfig) + "}";
    }

    // ------------------------------------------------------------------ discovery

    private static final class Found {
        final JsonView view;
        final String forcedName;
        final HfQuantMethod forcedMethod;

        Found(JsonView view, String forcedName) {
            this(view, forcedName, null);
        }

        Found(JsonView view, String forcedName, HfQuantMethod forcedMethod) {
            this.view = view;
            this.forcedName = forcedName;
            this.forcedMethod = forcedMethod;
        }
    }

    private static Found discover(HfConfig cfg) throws IOException {
        Found inConfig = null;
        if (cfg.raw().raw("quantization_config") != null) {
            inConfig = candidate(cfg, map(cfg, "quantization_config"), "quantization_config");
        } else if (cfg.raw().raw("compression_config") != null) {
            inConfig = candidate(cfg, map(cfg, "compression_config"), "compression_config");
        } else if (cfg.text().raw("quantization_config") != null) {
            inConfig = candidate(cfg, object(cfg.text().raw("quantization_config"),
                    cfg.source() + ": text_config.quantization_config"), "text_config.quantization_config");
        } else {
            Found side = modelOptSideFile(cfg);
            return side != null ? side : quantizationBlock(cfg);
        }
        if (inConfig != null && isMixedWithoutLayers(inConfig.view)) {
            Found side = modelOptSideFile(cfg);
            if (side != null) {
                return side;
            }
        }
        return inConfig;
    }

    /** {@code hf_quant_config.json} when it exists and was written by ModelOpt, else null. */
    private static Found modelOptSideFile(HfConfig cfg) throws IOException {
        Path side = cfg.dir().resolve(HF_QUANT_CONFIG_JSON);
        if (!Files.isRegularFile(side)) {
            return null;
        }
        Map<String, Object> m = HfConfigLoader.readObject(cfg.dir(), side);
        if (isModelOpt(m)) {
            return new Found(JsonView.of(m, HF_QUANT_CONFIG_JSON), null);
        }
        LOG.info("{} in {} is ignored: its producer is not modelopt", HF_QUANT_CONFIG_JSON, cfg.dir());
        return null;
    }

    /**
     * V weight_utils.get_quant_config: a ModelOpt MIXED_PRECISION config without {@code quantized_layers} keeps the
     * per-layer map only in {@code hf_quant_config.json}.
     */
    private static boolean isMixedWithoutLayers(JsonView view) {
        Map<String, Object> m = toMap(view);
        Object inner = m.get("quantization");
        Map<?, ?> in = inner instanceof Map ? (Map<?, ?>) inner : Collections.emptyMap();
        Object algo = m.get("quant_algo") != null ? m.get("quant_algo") : in.get("quant_algo");
        boolean mixed = "modelopt_mixed".equals(m.get("quant_method"))
                || (algo instanceof String && ((String) algo).toUpperCase(Locale.ROOT).equals("MIXED_PRECISION"));
        return mixed && m.get("quantized_layers") == null && in.get("quantized_layers") == null;
    }

    /** The top-level {@code quantization} block: Mistral (native or HF layout) or MLX; null when it is neither. */
    private static Found quantizationBlock(HfConfig cfg) {
        Object q = cfg.raw().raw("quantization");
        if (q == null) {
            return null;
        }
        Map<String, Object> qm = object(q, cfg.source() + ": quantization");
        if (cfg.mistralNative() || qm.containsKey("qformat_weight")) {
            return mistralNative(cfg, qm);
        }
        String mlx = mlxName(qm);
        if (mlx != null) {
            return new Found(JsonView.of(qm, cfg.source(), "quantization"), mlx, HfQuantMethod.MLX);
        }
        LOG.info("{}: 'quantization' is not an MLX quantization block; the checkpoint is treated as unquantized",
                cfg.source());
        return null;
    }

    private static Found candidate(HfConfig cfg, Map<String, Object> m, String path) {
        JsonView view = JsonView.of(m, cfg.source(), path);
        if (m.isEmpty()) {
            return null;
        }
        String mlx = mlxName(m);
        if (mlx != null) {
            return new Found(view, mlx, HfQuantMethod.MLX);
        }
        if (isModelOpt(m)) {
            return new Found(view, null);
        }
        Object method = m.get("quant_method");
        if (method != null && !(method instanceof String)) {
            throw new GgufFormatException(cfg.source() + ": " + path + ".quant_method must be a string, got "
                    + HfConfig.typeName(method));
        }
        if (method == null || ((String) method).isEmpty()) {
            LOG.warn("{}: '{}' has no quant_method; the checkpoint is treated as unquantized", cfg.source(), path);
            return null;
        }
        return new Found(view, null);
    }

    /** V model_arch_config_convertor: a {@code producer.name == modelopt} or legacy ModelOpt layout. */
    private static boolean isModelOpt(Map<String, Object> m) {
        Object producer = m.get("producer");
        Object inner = m.get("quantization");
        return (producer instanceof Map && "modelopt".equals(((Map<?, ?>) producer).get("name")))
                || (inner instanceof Map && ((Map<?, ?>) inner).containsKey("modelopt_quant_config"));
    }

    /**
     * SG mlx.py override: {@code mlx_q<bits>} for a bare {@code bits}/{@code group_size} block, else null. Every bit
     * width is MLX; only 4 and 8 have SGLang aliases.
     */
    private static String mlxName(Map<String, Object> m) {
        if (m.containsKey("quant_method") || !(m.get("bits") instanceof Long) || !(m.get("group_size") instanceof Long)) {
            return null;
        }
        for (Map.Entry<String, Object> e : m.entrySet()) {
            String k = e.getKey();
            Object v = e.getValue();
            boolean known = k.equals("bits") || k.equals("group_size") || k.equals("mode")
                    || v instanceof Map || v instanceof Boolean;
            if (!known) {
                return null;
            }
        }
        return "mlx_q" + m.get("bits");
    }

    /** V configs/mistral.py _remap_mistral_quantization_args. */
    private static Found mistralNative(HfConfig cfg, Map<String, Object> q) {
        if (q.isEmpty()) {
            return null;
        }
        String where = cfg.source() + ": quantization";
        if ("fp8_e4m3".equals(q.get("qformat_weight"))) {
            Map<String, Object> fp8 = new LinkedHashMap<>();
            fp8.put("quant_method", "fp8");
            Object scheme = q.get("qscheme_weight");
            if ("BLOCK".equals(scheme)) {
                fp8.put("weight_block_size", blockShape(q, where));
                fp8.put("activation_scheme", "dynamic");
            } else {
                if (scheme != null && !"TENSOR".equals(scheme)) {
                    throw new GgufFormatException(where + ": qscheme_weight '" + scheme
                            + "' is not TENSOR or BLOCK");
                }
                Object act = q.get("qscheme_act");
                if (act != null && !"NO_SCALES".equals(act) && !"TENSOR".equals(act)) {
                    throw new GgufFormatException(where + ": qscheme_act '" + act + "' is not NO_SCALES or TENSOR");
                }
                fp8.put("activation_scheme", "NO_SCALES".equals(act) ? "dynamic" : "static");
            }
            return new Found(JsonView.of(fp8, cfg.source(), "quantization"), null);
        }
        Object qm = q.get("quant_method");
        if (qm != null && String.valueOf(qm).toLowerCase(Locale.ROOT).replace('_', '-').equals("compressed-tensors")) {
            return new Found(JsonView.of(q, cfg.source(), "quantization"), "compressed-tensors");
        }
        throw new GgufFormatException(where + ": unsupported Mistral quantization " + q);
    }

    /** The {@code [block_size_out, block_size_in]} of a block-scaled Mistral weight quantization. */
    private static List<Object> blockShape(Map<String, Object> q, String where) {
        Object out = q.get("block_size_out");
        Object in = q.get("block_size_in");
        if (!(out instanceof Number) || !(in instanceof Number)) {
            throw new GgufFormatException(where + ": qscheme_weight 'BLOCK' needs numeric 'block_size_out' and"
                    + " 'block_size_in', got " + out + " and " + in);
        }
        return List.of(out, in);
    }

    // ------------------------------------------------------------------ assembly

    /** The mutable state while one configuration is reduced to rules. */
    static final class Assembly {
        HfQuantMethod method;
        final String name;
        final JsonView view;
        final Map<String, Object> q;
        final String where;
        String algo;
        final List<String> ignoreList = new ArrayList<>();
        final List<ModuleMatcher> ignore = new ArrayList<>();
        final List<QuantRule> rules = new ArrayList<>();
        HfTransformConfig transforms;
        OnlineQuantConfig onlineConfig;

        Assembly(HfQuantMethod method, String name, JsonView view) {
            this.method = method;
            this.name = name;
            this.view = view;
            this.q = toMap(view);
            this.where = view.source() + ": " + (view.path().isEmpty() ? "<root>" : view.path());
        }

        /** Adds the string entries of {@code entries} (null for none), each matched as {@code kind} makes it. */
        void ignore(Object entries, String key, Function<String, ModuleMatcher> kind) {
            if (entries == null) {
                return;
            }
            for (String e : strings(entries, where + "." + key)) {
                ignoreList.add(e);
                ignore.add(kind.apply(e));
            }
        }

        void all(QuantArgs args) {
            rules.add(QuantRule.of(ModuleMatcher.anyLinear(), args));
        }
    }

    private static Assembly assemble(HfConfig cfg, Found f) {
        if (f.forcedMethod != null) {
            Assembly a = new Assembly(f.forcedMethod, f.forcedName, f.view);
            a.method.read(cfg, a);
            return a;
        }
        Map<String, Object> m = toMap(f.view);
        String raw = f.forcedName != null ? f.forcedName : isModelOpt(m) ? "modelopt"
                : QuantArgs.strOr(m, "quant_method", "", f.view.source() + ": " + f.view.path());
        HfQuantAliases.Resolution r = HfQuantAliases.resolve(raw);
        Assembly a = new Assembly(r.method(), r.name(), f.view);
        if (r.method() == HfQuantMethod.REJECTED) {
            throw new GgufFormatException("unsupported quant_method '" + raw + "': " + r.reason());
        }
        if (r.method() == HfQuantMethod.ONLINE) {
            if (r.name().equals(HfQuantAliases.ONLINE)) {
                a.onlineConfig = onlineConfig(a);
            } else if (HfQuantAliases.onlineTarget(r.name()) == null) {
                logOnline(raw);
            }
        } else if (r.method() == HfQuantMethod.FP8 && DEEPSEEK_V4_TYPES.contains(cfg.modelType())) {
            a.method = HfQuantMethod.DEEPSEEK_V4_FP8;
        }
        a.method.read(cfg, a);
        return a;
    }

    /** The {@link OnlineQuantConfig} of a quant_method {@code online} configuration. */
    private static OnlineQuantConfig onlineConfig(Assembly a) {
        try {
            return OnlineQuantConfig.parse(a.q, a.where, true);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException(e.getMessage(), e);
        }
    }

    static void logOnline(String raw) {
        LOG.info("quant_method '{}' quantizes at load; weights load as stored (use --quant-online to quantize)", raw);
    }

    // ------------------------------------------------------------------ helpers

    static Object firstOf(Map<String, Object> m, String... keys) {
        for (String k : keys) {
            if (m.get(k) != null) {
                return m.get(k);
            }
        }
        return null;
    }

    private static Map<String, Object> map(HfConfig cfg, String key) {
        return object(cfg.raw().raw(key), cfg.source() + ": " + key);
    }

    private static Map<String, Object> toMap(JsonView view) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String k : view.keys()) {
            out.put(k, view.raw(k));
        }
        return Collections.unmodifiableMap(out);
    }

    static Map<String, Object> object(Object v, String where) {
        if (!(v instanceof Map)) {
            throw new GgufFormatException(where + " must be an object, got " + HfConfig.typeName(v));
        }
        return HfConfig.asObject(v);
    }

    static Map<String, Object> optObject(Object v, String where) {
        return v == null ? Collections.emptyMap() : object(v, where);
    }

    static Map<String, Object> nullableObject(Object v, String where) {
        return v == null ? null : object(v, where);
    }

    static List<String> strings(Object v, String where) {
        if (v instanceof String) {
            return List.of((String) v);
        }
        if (!(v instanceof List)) {
            throw new GgufFormatException(where + " must be an array of strings, got " + HfConfig.typeName(v));
        }
        List<String> out = new ArrayList<>();
        for (Object o : (List<?>) v) {
            if (!(o instanceof String)) {
                throw new GgufFormatException(where + " entries must be strings, got " + HfConfig.typeName(o));
            }
            out.add((String) o);
        }
        return out;
    }
}
