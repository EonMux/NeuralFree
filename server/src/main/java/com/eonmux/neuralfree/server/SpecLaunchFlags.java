package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.DraftHeadOptions;
import com.eonmux.neuralfree.gguf.engine.DraftTypes;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.SpeculativeConfigJson;
import com.eonmux.neuralfree.gguf.engine.TensorOverride;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

/**
 * The speculative decoding flags of {@link ServerLaunchArgs} in llama-server's spellings (and the CLI's
 * {@code --spec-verify} / fly / synthetic flags), mapped to the {@code neuralfree.draft.*} properties the batch
 * engine reads, and vLLM's {@code --speculative-config} JSON, whose keys apply as those flags
 * ({@link SpeculativeConfigJson}): the drafter types ({@code --spec-type}, repeated flags append as llama.cpp's
 * do), the draft model and its KV types and ubatch, the draft lengths and minimum probability, every n-gram type's
 * sizes (a lookup size range included, vLLM's {@code prompt_lookup_min} / {@code prompt_lookup_max}), the lookup
 * cache files, the suffix limits, the verification mode and its settings, vLLM's {@code draft_sample_method},
 * the derived candidate head of a {@code dflash} drafter ({@code --draft-head-vocab}, {@code --draft-head-freq},
 * {@code --draft-head-keep}, {@code --draft-head-prec}, {@code --draft-prec}), and
 * the separate draft model's own placement ({@code -devd}, {@code -ngld}, {@code -otd}, {@code -ncmoed};
 * {@code -cmoed} is a boolean, which {@link ServerLaunchArgs} takes directly). llama.cpp's backend-sampling
 * switches ({@code -bs}, {@code --backend-sampling} and their {@code --no-} spelling, plus the draft-only pair)
 * take no value and map to the device-placement properties the launch reads; so does
 * {@code --[no-]spec-draft-heterogeneous-vocab}, the draft model's token-level-intersection drafting (vLLM
 * {@code use_heterogeneous_vocab}). The drafters the batch engine does
 * not run per sequence ({@code draft-mtp}, the target's own NextN head) and a Hugging Face draft download
 * ({@code -hfd}) are refused by name at launch, as is every unusable value. Every value is checked here and named
 * by its flag; the combined speculative settings are checked once the command line is read ({@link #validate}).
 */
final class SpecLaunchFlags {

    /** Turns a flag's raw value into its property value, refusing an unusable one by the flag's name. */
    @FunctionalInterface
    private interface Value {
        String parse(String flag, String value);
    }

    /** The property a flag sets and how its value is read. */
    private record Taken(String property, Value value) {
    }

    /** The flags the server takes, mapped to their properties. */
    private static final Map<String, Taken> TAKEN = taken();

    /**
     * llama.cpp's backend-sampling switches, which take no value: {@code -bs} / {@code --backend-sampling} run
     * every sampling on the backend the tokens come from (generation sampling, verification and the draft), their
     * {@code --no-} spelling turns all three off, and {@code --[no-]spec-draft-backend-sampling} only moves the
     * draft's sampling. The {@code --[no-]spec-draft-tree} pair turns the dflash drafter's single-pass draft-tree
     * verification ({@code neuralfree.draft.tree}) on and off, {@code --spec-draft-tree-nodes} /
     * {@code --spec-tree-nodes} sets its fixed verify width ({@code neuralfree.draft.tree.nodes}), and the
     * {@code --[no-]w12[.-head]} pairs turn the W12 weight streaming ({@code neuralfree.weights.w12[.head]}) on
     * and off.
     */
    private static final Set<String> SWITCHES = Set.of("-bs", "--backend-sampling", "--no-backend-sampling",
            "--spec-draft-backend-sampling", "--no-spec-draft-backend-sampling",
            "--spec-draft-heterogeneous-vocab", "--no-spec-draft-heterogeneous-vocab",
            "--spec-draft-tree", "--no-spec-draft-tree",
            "--w12", "--no-w12", "--w12-head", "--no-w12-head");

    /** The draft flags the server does not take: the Hugging Face draft downloads. */
    private static final Set<String> REFUSED = Set.of("-hfd", "-hfrd", "--hf-repo-draft", "--spec-draft-hf");

    private SpecLaunchFlags() {
    }

    private static Map<String, Taken> taken() {
        Map<String, Taken> m = new HashMap<>();
        Value draftMax = whole(1, Integer.MAX_VALUE);
        Value draftMin = whole(0, Integer.MAX_VALUE);
        Value sizeN = whole(1, NgramDrafter.MAX_SIZE_N);
        Value sizeM = whole(1, NgramDrafter.MAX_SIZE_M);
        Value hits = whole(1, Integer.MAX_VALUE);
        put(m, new Taken(EngineConfig.PROP_DRAFT_MAX, draftMax), "--spec-draft-n-max", "--draft-max", "--draft-n-max");
        put(m, new Taken(EngineConfig.PROP_DRAFT_MIN, draftMin), "--spec-draft-n-min", "--draft-min", "--draft-n-min");
        put(m, new Taken(EngineConfig.PROP_DRAFT, SpecLaunchFlags::regularFile), "-md", "--model-draft",
                "--spec-draft-model");
        put(m, new Taken(EngineConfig.PROP_DRAFT_P_MIN, SpecLaunchFlags::probability), "--spec-draft-p-min",
                "--draft-p-min");
        put(m, new Taken(MemoryConfig.PROP_DRAFT_KV_TYPE_K, SpecLaunchFlags::kvType), "-ctkd", "--cache-type-k-draft",
                "--spec-draft-type-k");
        put(m, new Taken(MemoryConfig.PROP_DRAFT_KV_TYPE_V, SpecLaunchFlags::kvType), "-ctvd", "--cache-type-v-draft",
                "--spec-draft-type-v");
        put(m, new Taken(MemoryConfig.PROP_DRAFT_UBATCH, whole(0, Integer.MAX_VALUE)), "-ubd", "--spec-draft-n-ubatch",
                "--spec-draft-ubatch-size", "--ubatch-size-draft");
        put(m, new Taken(EngineConfig.PROP_DRAFT_N_GPU_LAYERS, SpecLaunchFlags::gpuLayers), "-ngld",
                "--spec-draft-ngl", "--gpu-layers-draft", "--n-gpu-layers-draft");
        put(m, new Taken(EngineConfig.PROP_DRAFT_GPU_DEVICES, SpecLaunchFlags::deviceList), "-devd",
                "--spec-draft-device", "--device-draft");
        put(m, new Taken(EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES, SpecLaunchFlags::overrideTensors), "-otd",
                "--spec-draft-override-tensor", "--override-tensor-draft");
        put(m, new Taken(EngineConfig.PROP_DRAFT_N_CPU_MOE, whole(0, Integer.MAX_VALUE)), "-ncmoed",
                "--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "--n-cpu-moe-draft");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n", whole(2, NgramDrafter.MAX_SIZE_N)),
                "--spec-ngram-simple-size-n", "--ngram-prompt-lookup-max");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n.min", whole(1, NgramDrafter.MAX_SIZE_N)),
                "--spec-ngram-simple-size-n-min", "--ngram-prompt-lookup-min");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.m", sizeM), "--spec-ngram-simple-size-m");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MAP_K + ".size.n", sizeN), "--spec-ngram-map-k-size-n");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MAP_K + ".size.m", sizeM), "--spec-ngram-map-k-size-m");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MAP_K + ".min.hits", hits), "--spec-ngram-map-k-min-hits");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MAP_K4V + ".size.n", sizeN), "--spec-ngram-map-k4v-size-n");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MAP_K4V + ".size.m", sizeM), "--spec-ngram-map-k4v-size-m");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MAP_K4V + ".min.hits", hits), "--spec-ngram-map-k4v-min-hits");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MOD + ".n.match", sizeN), "--spec-ngram-mod-n-match");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MOD + ".n.max", sizeM), "--spec-ngram-mod-n-max");
        put(m, new Taken(SpeculativeOptions.PROP_NGRAM_MOD + ".n.min", draftMin), "--spec-ngram-mod-n-min");
        put(m, new Taken(SpeculativeOptions.PROP_LOOKUP_CACHE_STATIC, SpecLaunchFlags::regularFile), "-lcs",
                "--lookup-cache-static");
        put(m, new Taken(SpeculativeOptions.PROP_LOOKUP_CACHE_DYNAMIC, SpecLaunchFlags::writableFile), "-lcd",
                "--lookup-cache-dynamic");
        put(m, new Taken(SpeculativeOptions.PROP_SUFFIX_MAX_TREE_DEPTH, whole(1, Integer.MAX_VALUE)),
                "--spec-suffix-max-tree-depth");
        put(m, new Taken(SpeculativeOptions.PROP_SUFFIX_MAX_SPEC_FACTOR, SpecLaunchFlags::nonNegative),
                "--spec-suffix-max-spec-factor");
        put(m, new Taken(SpeculativeOptions.PROP_SUFFIX_MAX_SPEC_OFFSET, SpecLaunchFlags::nonNegative),
                "--spec-suffix-max-spec-offset");
        put(m, new Taken(SpeculativeOptions.PROP_SUFFIX_MIN_TOKEN_PROB, SpecLaunchFlags::probability),
                "--spec-suffix-min-token-prob");
        put(m, new Taken(SpeculativeOptions.PROP_SUFFIX_MAX_CACHED_REQUESTS, whole(0, Integer.MAX_VALUE)),
                "--spec-suffix-max-cached-requests");
        put(m, new Taken(SpeculativeOptions.PROP_VERIFY, SpecLaunchFlags::verifyMode), "--spec-verify",
                "--rejection-sample-method");
        put(m, new Taken(SpeculativeOptions.PROP_VERIFY, SpecLaunchFlags::draftSampling), "--spec-draft-sampling");
        put(m, new Taken(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD, SpecLaunchFlags::sampleMethod),
                "--spec-draft-sample-method");
        put(m, new Taken(SpeculativeOptions.PROP_FLY_WINDOW, whole(0, Integer.MAX_VALUE)), "--spec-fly-window");
        put(m, new Taken(SpeculativeOptions.PROP_FLY_ENTROPY_THRESHOLD, SpecLaunchFlags::nonNegative),
                "--spec-fly-entropy-threshold");
        put(m, new Taken(SpeculativeOptions.PROP_FLY_ENTROPY_TOP_K, whole(1, Integer.MAX_VALUE)),
                "--spec-fly-entropy-top-k");
        put(m, new Taken(SpeculativeOptions.PROP_SYNTH_LEN, SpecLaunchFlags::nonNegative), "--spec-synth-len");
        put(m, new Taken(SpeculativeOptions.PROP_SYNTH_RATES, SpecLaunchFlags::rates), "--spec-synth-rates");
        put(m, new Taken(SpeculativeOptions.PROP_TREE_TEMP, SpecLaunchFlags::positive), "--spec-draft-tree-temp",
                "--spec-tree-temp");
        put(m, new Taken(SpeculativeOptions.PROP_TREE_NODES, whole(1, SpeculativeOptions.MAX_TREE_NODES)),
                "--spec-draft-tree-nodes", "--spec-tree-nodes");
        put(m, new Taken(SpeculativeOptions.PROP_W12_F16, SpecLaunchFlags::w12F16Mode), "--w12-f16");
        put(m, new Taken(SpeculativeOptions.PROP_DRAFT_HEAD_VOCAB, whole(1, Integer.MAX_VALUE)),
                "--draft-head-vocab");
        put(m, new Taken(SpeculativeOptions.PROP_DRAFT_HEAD_FREQ, SpecLaunchFlags::regularFile),
                "--draft-head-freq");
        put(m, new Taken(SpeculativeOptions.PROP_DRAFT_HEAD_KEEP, SpecLaunchFlags::ids), "--draft-head-keep");
        put(m, new Taken(SpeculativeOptions.PROP_DRAFT_HEAD_PREC, SpecLaunchFlags::prec), "--draft-head-prec");
        put(m, new Taken(SpeculativeOptions.PROP_DRAFT_PREC, SpecLaunchFlags::prec), "--draft-prec");
        return Map.copyOf(m);
    }

    private static void put(Map<String, Taken> m, Taken taken, String... flags) {
        for (String flag : flags) {
            m.put(flag, taken);
        }
    }

    /** Whether {@code flag} is a speculative decoding flag, taken, refused or a {@link #SWITCHES} switch. */
    static boolean handles(String flag) {
        return flag.equals("--spec-type") || flag.equals(SpeculativeConfigJson.FLAG)
                || flag.equals("--spec-ngram-simple-min-hits") || TAKEN.containsKey(flag)
                || REFUSED.contains(flag) || SWITCHES.contains(flag);
    }

    /** Whether {@code flag} is a valueless backend-sampling switch ({@link #SWITCHES}). */
    static boolean isSwitch(String flag) {
        return SWITCHES.contains(flag);
    }

    /**
     * Refuses {@code flag} when the server does not take it.
     *
     * @throws IllegalArgumentException naming the flag
     */
    static void refuseUnsupported(String flag) {
        if (REFUSED.contains(flag)) {
            throw new IllegalArgumentException(flag + " is not supported by the server: its draft model (-md) loads"
                    + " from a local file, so a Hugging Face draft download runs in the single-sequence CLI");
        }
        if (flag.equals("--spec-ngram-simple-min-hits")) {
            throw new IllegalArgumentException(flag + " is not a setting: ngram-simple has no hit threshold"
                    + " (llama.cpp reads min hits for the map types only: --spec-ngram-map-k-min-hits,"
                    + " --spec-ngram-map-k4v-min-hits)");
        }
    }

    /**
     * Writes the property of the taken {@code flag} with its {@code value}.
     *
     * @throws IllegalArgumentException naming the flag for an unusable value
     */
    static void apply(Map<String, String> properties, String flag, String value) {
        if (isSwitch(flag)) {
            applySwitch(properties, flag, value);
            return;
        }
        if (flag.equals(SpeculativeConfigJson.FLAG)) {
            speculativeConfig(properties, value);
            return;
        }
        if (flag.equals("--spec-type")) {
            specType(properties, value);
            return;
        }
        Taken target = TAKEN.get(flag);
        properties.put(target.property(), target.value().parse(flag, value));
    }

    /**
     * Writes the properties of the backend-sampling switch {@code flag}: every device placement on for {@code -bs}
     * and {@code --backend-sampling}, all three off for {@code --no-backend-sampling}, and only the draft's for the
     * {@code --[no-]spec-draft-backend-sampling} pair. The switch takes no value; {@code value} is the argument that
     * wrongly follows it, refused by name.
     *
     * @throws IllegalArgumentException naming the flag and the unwanted value
     */
    static void applySwitch(Map<String, String> properties, String flag, String value) {
        if (value != null) {
            throw new IllegalArgumentException(flag + " takes no value, got '" + value + "'");
        }
        switch (flag) {
            case "-bs", "--backend-sampling" -> backendSampling(properties, "true");
            case "--no-backend-sampling" -> backendSampling(properties, "false");
            case "--spec-draft-backend-sampling" ->
                    properties.put(SpeculativeOptions.PROP_DFLASH_DEVICE, "true");
            case "--no-spec-draft-backend-sampling" ->
                    properties.put(SpeculativeOptions.PROP_DFLASH_DEVICE, "false");
            case "--spec-draft-heterogeneous-vocab" ->
                    properties.put(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB, "true");
            case "--spec-draft-tree" -> properties.put(SpeculativeOptions.PROP_TREE, "true");
            case "--w12" -> properties.put(SpeculativeOptions.PROP_W12, "true");
            case "--no-w12" -> properties.put(SpeculativeOptions.PROP_W12, "false");
            case "--w12-head" -> properties.put(SpeculativeOptions.PROP_W12_HEAD, "true");
            case "--no-w12-head" -> properties.put(SpeculativeOptions.PROP_W12_HEAD, "false");
            default -> {
                if (flag.equals("--no-spec-draft-tree")) {
                    properties.put(SpeculativeOptions.PROP_TREE, "false");
                } else {
                    properties.put(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB, "false");
                }
            }
        }
    }

    /** The three device-placement properties llama.cpp's {@code -bs} moves together. */
    private static void backendSampling(Map<String, String> properties, String on) {
        properties.put(GenerationLoop.PROP_SAMPLING_DEVICE, on);
        properties.put(SpeculativeOptions.PROP_VERIFY_DEVICE, on);
        properties.put(SpeculativeOptions.PROP_DFLASH_DEVICE, on);
    }

    /**
     * vLLM's {@code --speculative-config}: each key applies as the launch flag {@link SpeculativeConfigJson} maps it
     * to, and a key whose flag the server refuses is refused naming the key.
     */
    private static void speculativeConfig(Map<String, String> properties, String json) {
        for (SpeculativeConfigJson.Setting s : SpeculativeConfigJson.parse(json)) {
            try {
                if (!handles(s.flag())) {
                    throw new IllegalArgumentException(s.flag() + " is not a server flag");
                }
                refuseUnsupported(s.flag());
                apply(properties, s.flag(), isSwitch(s.flag()) ? null : s.value());
            } catch (IllegalArgumentException e) {
                throw s.refusal(e.getMessage());
            }
        }
    }

    /**
     * Checks the combined speculative settings of the parsed command line (a prompt-lookup minimum above its
     * maximum, an n-gram lookup too short for its kind), as the engine would read them.
     *
     * @throws IllegalArgumentException naming the property of the contradicting settings
     */
    static void validate(Map<String, String> properties) {
        Properties p = new Properties();
        p.putAll(properties);
        try {
            SpeculativeOptions.fromProperties(p);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("the speculative decoding flags contradict: " + e.getMessage(), e);
        }
    }

    /**
     * {@code --spec-type}: appended to the earlier ones; {@code draft-mtp} is refused, since the target's NextN
     * head and MTP draft files draft over one sequence's hidden rows and run single-sequence.
     */
    private static void specType(Map<String, String> properties, String value) {
        String previous = properties.get(EngineConfig.PROP_DRAFT_TYPE);
        String joined = previous == null ? value : previous + "," + value;
        List<EngineConfig.DraftType> types;
        try {
            types = DraftTypes.ordered(DraftTypes.parse(joined));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("--spec-type: " + e.getMessage(), e);
        }
        if (types.contains(EngineConfig.DraftType.MTP)) {
            throw new IllegalArgumentException("--spec-type " + DraftTypes.names(types) + " is not supported by the"
                    + " server: its batch engine drafts with the lookup types, the suffix tree and a separate draft"
                    + " model (draft-simple); draft-mtp runs in the single-sequence CLI");
        }
        properties.put(EngineConfig.PROP_DRAFT_TYPE, joined);
    }

    // ------------------------------------------------------------------ values

    private static Value whole(long min, long max) {
        return (flag, value) -> ServerLaunchArgs.whole(flag, value, min, max).toString();
    }

    private static double decimal(String flag, String value) {
        double d;
        try {
            d = Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " must be a number, got '" + value + "'");
        }
        if (!Double.isFinite(d)) {
            throw new IllegalArgumentException(flag + " must be finite, got " + value);
        }
        return d;
    }

    private static String nonNegative(String flag, String value) {
        double d = decimal(flag, value);
        if (d < 0) {
            throw new IllegalArgumentException(flag + " must not be negative, got " + value);
        }
        return value.trim();
    }

    private static String positive(String flag, String value) {
        double d = decimal(flag, value);
        if (d <= 0) {
            throw new IllegalArgumentException(flag + " must be positive, got " + value);
        }
        return value.trim();
    }

    private static String probability(String flag, String value) {
        double d = decimal(flag, value);
        if (d < 0 || d > 1) {
            throw new IllegalArgumentException(flag + " must be between 0 and 1, got " + value);
        }
        return value.trim();
    }

    private static String regularFile(String flag, String value) {
        if (!Files.isRegularFile(Path.of(value))) {
            throw new IllegalArgumentException(flag + " is not a regular file: " + value);
        }
        return value;
    }

    /** A file read when present and written on exit: an existing directory or a non-directory path. */
    private static String writableFile(String flag, String value) {
        Path path = Path.of(value);
        if (Files.isDirectory(path)) {
            throw new IllegalArgumentException(flag + " is a directory, not a file: " + value);
        }
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null && !Files.isDirectory(parent)) {
            throw new IllegalArgumentException(flag + " names a file in a missing directory: " + value);
        }
        return value;
    }

    private static String kvType(String flag, String value) {
        return KvDType.parse(value, flag).name().toLowerCase(Locale.ROOT);
    }

    /** {@code -ngld}: {@code auto}, {@code all}, llama.cpp's {@code -1} / {@code -2} markers, or a layer count. */
    private static String gpuLayers(String flag, String value) {
        String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.equals("auto") || v.equals("-1")) {
            return "auto";
        }
        if (v.equals("all") || v.equals("-2")) {
            return "all";
        }
        return whole(0, Integer.MAX_VALUE).parse(flag, value);
    }

    /** {@code -devd}: a comma-separated device list of selectors, or {@code none} alone. */
    private static String deviceList(String flag, String value) {
        List<String> out = new ArrayList<>();
        for (String entry : value.split(",", -1)) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new IllegalArgumentException(flag + " " + value + " has an empty device entry");
            }
            out.add(s);
        }
        boolean none = out.size() == 1 && out.get(0).equalsIgnoreCase(EngineConfig.DRAFT_NO_DEVICE);
        for (String s : out) {
            if (!none && s.equalsIgnoreCase(EngineConfig.DRAFT_NO_DEVICE)) {
                throw new IllegalArgumentException(flag + " " + value + ": " + EngineConfig.DRAFT_NO_DEVICE
                        + " keeps the draft model off every GPU and cannot be listed with devices");
            }
        }
        return String.join(",", out);
    }

    /** {@code -otd}: comma-separated {@code pattern=target} entries kept as the property's value. */
    private static String overrideTensors(String flag, String value) {
        for (String entry : value.split(",")) {
            String e = entry.trim();
            if (e.isEmpty()) {
                continue;
            }
            int eq = e.indexOf('=');
            if (eq <= 0 || eq == e.length() - 1) {
                throw new IllegalArgumentException(flag + " entry '" + e + "' is <tensor name pattern>=<target>");
            }
            try {
                TensorOverride.parse(e.substring(0, eq), e.substring(eq + 1));
            } catch (IllegalArgumentException ex) {
                throw new IllegalArgumentException(flag + " entry '" + e + "': " + ex.getMessage(), ex);
            }
        }
        return value.trim();
    }

    private static String verifyMode(String flag, String value) {
        try {
            return VerifyMode.parse(value).toString();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
    }

    /** llama.cpp {@code --spec-draft-sampling}: greedy verifies by match, probabilistic by rejection sampling. */
    private static String draftSampling(String flag, String value) {
        String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.equals("greedy")) {
            return "match";
        }
        if (v.equals("probabilistic")) {
            return "rejection";
        }
        throw new IllegalArgumentException(flag + " '" + value + "' is invalid; use greedy or probabilistic");
    }

    /** vLLM {@code draft_sample_method}: how the draft model picks its tokens, greedy or probabilistic. */
    private static String sampleMethod(String flag, String value) {
        return SpeculativeOptions.parseDraftSampleMethod(flag, value) ? "probabilistic" : "greedy";
    }

    private static String rates(String flag, String value) {
        SpeculativeOptions.parseRates(flag, value);
        return value.trim();
    }

    /** Comma-separated token ids the derived draft head keeps ({@code neuralfree.draft.head.keep}). */
    private static String ids(String flag, String value) {
        SpeculativeOptions.parseIds(flag, value);
        return value.trim();
    }

    /** A draft precision: {@code fp8} or {@code q8_0}; vLLM's other value {@code nvfp4} is refused by name. */
    private static String prec(String flag, String value) {
        return DraftHeadOptions.Prec.of(flag, value).name().toLowerCase(Locale.ROOT);
    }

    /** {@code --w12-f16}: the W12 streaming's treatment of an F16 tensor's values, {@code refuse} or {@code round}. */
    private static String w12F16Mode(String flag, String value) {
        return W12F16Mode.parse(flag, value).propertyNameValue();
    }
}
