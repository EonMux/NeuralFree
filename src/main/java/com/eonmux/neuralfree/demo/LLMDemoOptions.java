package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Command-line options of {@link LLMDemo}: parsing, validation, the usage text and the engine, sampler and diffusion
 * configurations they select. Engine and sampler settings start from the {@code -Dneuralfree.*} system properties
 * ({@link EngineConfig#fromSystemProperties()}, whose sampler defaults are {@link SamplerConfig#fromProperties});
 * the demo's own sampling defaults ({@link #DEFAULT_TEMPERATURE} and the others) apply only where no property is
 * set, and flags override both. Flag names follow llama.cpp where it has an equivalent ({@code common/arg.cpp});
 * no short alias is given a meaning that differs from llama.cpp's. Immutable.
 */
final class LLMDemoOptions {

    /** What the demo runs. */
    enum Mode {
        /**
         * Chosen from the model: classify for a decision head, rerank (with {@code --document}) or classify for RANK
         * pooling, embed for any other pooling type, else from the support class.
         */
        AUTO,
        GENERATE,
        EMBED,
        CLASSIFY,
        RERANK,
        DIFFUSION,
        AUDIO;

        static Mode parse(String value) throws UsageException {
            String s = value.trim().toUpperCase(Locale.ROOT);
            for (Mode m : values()) {
                if (m.name().equals(s)) {
                    return m;
                }
            }
            throw new UsageException("--mode " + value + " is invalid; use generate, embed, classify, rerank,"
                    + " diffusion or audio");
        }

        /** Whether the mode evaluates whole inputs in one non-causal or memoryless pass. */
        boolean pooled() {
            return this == EMBED || this == CLASSIFY || this == RERANK;
        }
    }

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {
        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    static final float DEFAULT_TEMPERATURE = 0.6f;
    static final float DEFAULT_TOP_P = 0.95f;
    static final int DEFAULT_TOP_K = 40;
    static final float DEFAULT_MIN_P = 0.05f;
    static final float DEFAULT_REPEAT_PENALTY = 1.0f;
    static final int DEFAULT_REPEAT_WINDOW = 256;
    static final int DEFAULT_MAX_TOKENS = 2560;
    static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
    /**
     * Ubatch of the pooled modes when none is given: llama.cpp {@code llama-embedding} evaluates each input in one
     * batch of {@code max(n_batch = 2048, n_ctx)} tokens (examples/embedding/embedding.cpp:122-131).
     */
    static final int POOLED_UBATCH = 2048;

    /** The model path given on the command line, or null. */
    final String modelPath;
    /** The file the session statistics are appended to ({@code --stats-file}), or null for none. */
    final String statsFile;
    final String prompt;
    final String systemPrompt;
    final boolean raw;
    final boolean thinking;
    final boolean debugPrompt;
    final boolean nonInteractive;
    final Mode mode;
    final boolean showAllChannels;
    final boolean benchmark;
    final boolean benchStreaming;
    final boolean memoryReportOnly;
    final boolean noMmap;
    final int embedPreview;
    final int[] tokenTypes;
    final String document;
    final int[] decisionOrder;
    final int[] audioCodes;
    final String audioOut;
    final boolean help;
    /** Engine settings before the mode-dependent ubatch default ({@link #engineConfig(Mode)}). */
    final EngineConfig engine;
    final boolean ubatchExplicit;
    /** The request of every autoregressive generation: token budget, sampler, stop strings and ignore-eos. */
    final GenerationRequest generation;
    final DiffusionConfig diffusion;

    private LLMDemoOptions(Parser p) {
        this.modelPath = p.modelPath;
        this.statsFile = p.statsFile;
        this.prompt = p.prompt;
        this.systemPrompt = p.systemPrompt;
        this.raw = p.raw;
        this.thinking = p.thinking;
        this.debugPrompt = p.debugPrompt;
        this.nonInteractive = p.nonInteractive;
        this.mode = p.mode;
        this.showAllChannels = p.showAllChannels;
        this.benchmark = p.benchmark;
        this.benchStreaming = p.benchStreaming;
        this.memoryReportOnly = p.memoryReportOnly;
        this.noMmap = p.noMmap;
        this.embedPreview = p.embedPreview;
        this.tokenTypes = p.tokenTypes;
        this.document = p.document;
        this.decisionOrder = p.decisionOrder;
        this.audioCodes = p.audioCodes;
        this.audioOut = p.audioOut;
        this.help = p.help;
        this.engine = p.engine;
        this.ubatchExplicit = p.ubatchExplicit;
        this.generation = p.generation;
        this.diffusion = p.diffusion;
    }

    /**
     * Parses {@code args}; a first argument that is not a flag is the model path.
     *
     * @throws UsageException for an unknown flag, a missing value or a value out of range
     */
    static LLMDemoOptions parse(String[] args) throws UsageException {
        Parser p = new Parser(args);
        p.run();
        return new LLMDemoOptions(p);
    }

    /**
     * The engine configuration for {@code resolved}: pooled modes (embed, classify, rerank) take a ubatch of
     * {@code max(}{@value #POOLED_UBATCH}{@code , ctx)} unless one was given, so long inputs fit one non-causal pass.
     */
    EngineConfig engineConfig(Mode resolved) {
        if (ubatchExplicit || !resolved.pooled()) {
            return engine;
        }
        MemoryConfig m = engine.memory();
        int ubatch = Math.max(POOLED_UBATCH, m.nCtx());
        return engine.toBuilder().memory(m.toBuilder().ubatch(ubatch).build()).build();
    }

    /** The mutable state of one parse. */
    private static final class Parser {
        private final String[] args;
        private int i;

        String modelPath;
        String statsFile;
        String prompt;
        String systemPrompt = DEFAULT_SYSTEM_PROMPT;
        boolean raw;
        boolean thinking;
        boolean debugPrompt;
        boolean nonInteractive;
        Mode mode = Mode.AUTO;
        int maxTokens = DEFAULT_MAX_TOKENS;
        boolean showAllChannels = true;
        boolean benchmark;
        boolean benchStreaming;
        boolean memoryReportOnly;
        boolean noMmap;
        int embedPreview = 8;
        int[] tokenTypes;
        String document;
        int[] decisionOrder;
        int[] audioCodes;
        String audioOut;
        boolean help;
        EngineConfig engine;
        boolean ubatchExplicit;
        final List<String> stopStrings = new ArrayList<>();
        boolean ignoreEos;
        GenerationRequest generation;
        DiffusionConfig diffusion;

        private final EngineConfig.Builder eb;
        private final MemoryConfig.Builder mb;
        private final SamplerConfig.Builder sb;
        private final DiffusionConfig.Builder db = DiffusionConfig.builder();
        private final List<EngineConfig.LoraSpec> loras = new ArrayList<>();
        private boolean diffusionEps;
        private boolean diffusionBlock;

        Parser(String[] args) throws UsageException {
            this.args = args.clone();
            EngineConfig base;
            try {
                base = EngineConfig.fromSystemProperties();
            } catch (IllegalArgumentException e) {
                throw new UsageException(e.getMessage());
            }
            this.eb = base.toBuilder();
            this.mb = base.memory().toBuilder();
            this.loras.addAll(base.loras());
            this.ubatchExplicit = System.getProperty(MemoryConfig.PROP_UBATCH) != null;
            this.sb = samplerBase(base.samplerDefaults());
            diffusionBase(base.samplerDefaults());
        }

        /**
         * The property sampler with the demo defaults in place of every setting no property gives (temperature,
         * top-k, top-p, min-p, repeat penalty and window); typical-p, presence and frequency penalties, the seed
         * and greedy keep their property or llama.cpp default values.
         */
        private static SamplerConfig.Builder samplerBase(SamplerConfig fromProperties) {
            SamplerConfig.Builder b = fromProperties.toBuilder();
            if (!propertySet(SamplerConfig.PROP_TEMPERATURE)) {
                b.temperature(DEFAULT_TEMPERATURE);
            }
            if (!propertySet(SamplerConfig.PROP_TOP_K)) {
                b.topK(DEFAULT_TOP_K);
            }
            if (!propertySet(SamplerConfig.PROP_TOP_P)) {
                b.topP(DEFAULT_TOP_P);
            }
            if (!propertySet(SamplerConfig.PROP_MIN_P)) {
                b.minP(DEFAULT_MIN_P);
            }
            if (!propertySet(SamplerConfig.PROP_REPEAT_PENALTY)) {
                b.repeatPenalty(DEFAULT_REPEAT_PENALTY);
            }
            if (!propertySet(SamplerConfig.PROP_PENALTY_LAST_N)) {
                b.penaltyLastN(DEFAULT_REPEAT_WINDOW);
            }
            return b;
        }

        /** Diffusion sampling takes the temperature, top-k, top-p and seed properties that are set. */
        private void diffusionBase(SamplerConfig fromProperties) {
            if (propertySet(SamplerConfig.PROP_TEMPERATURE)) {
                db.temperature(fromProperties.temperature());
            }
            if (propertySet(SamplerConfig.PROP_TOP_K)) {
                db.topK(fromProperties.topK());
            }
            if (propertySet(SamplerConfig.PROP_TOP_P)) {
                db.topP(fromProperties.topP());
            }
            if (propertySet(SamplerConfig.PROP_SEED)) {
                db.seed(fromProperties.seed());
            }
        }

        /** Whether a system property has a non-blank value, the rule {@link EngineConfig#fromProperties} applies. */
        private static boolean propertySet(String key) {
            String v = System.getProperty(key);
            return v != null && !v.trim().isEmpty();
        }

        void run() throws UsageException {
            boolean cliLoras = false;
            for (i = 0; i < args.length; i++) {
                String a = args[i];
                if (i == 0 && !a.startsWith("-")) {
                    modelPath = a;
                    continue;
                }
                if (a.equals("--lora") || a.equals("--lora-scaled")) {
                    if (!cliLoras) {
                        loras.clear();
                        cliLoras = true;
                    }
                    if (a.equals("--lora")) {
                        lora(value(a));
                    } else {
                        loraScaled(a);
                    }
                    continue;
                }
                if (!general(a) && !backend(a) && !memory(a) && !sampling(a) && !speculative(a) && !diffusion(a)
                        && !heads(a)) {
                    throw new UsageException("unknown option " + a + " (see --help)");
                }
            }
            finish();
        }

        private boolean general(String a) throws UsageException {
            switch (a) {
                case "--model": case "-m": modelPath = value(a); return true;
                case "--prompt": case "-p": prompt = value(a); return true;
                case "--system": systemPrompt = value(a); return true;
                case "--max-tokens": case "--n-predict": case "-n":
                    maxTokens = intValue(a, -1, Integer.MAX_VALUE);
                    if (maxTokens == 0) {
                        throw new UsageException(a + " must be positive, or -1 for no limit");
                    }
                    return true;
                case "--mode": mode = Mode.parse(value(a)); return true;
                case "--embed": mode = Mode.EMBED; return true;
                case "--raw": raw = true; return true;
                case "--thinking": thinking = true; return true;
                case "--debug-prompt": debugPrompt = true; return true;
                case "--non-interactive": nonInteractive = true; return true;
                case "--show-all-channels": showAllChannels = true; return true;
                case "--content-only": showAllChannels = false; return true;
                case "--benchmark": benchmark = true; return true;
                case "--bench-streaming": benchStreaming = true; return true;
                case "--vram-budget": case "--memory-report": memoryReportOnly = true; return true;
                case "--no-mmap": noMmap = true; return true;
                case "--stats-file": statsFile = value(a); return true;
                case "--help": case "-h": help = true; return true;
                default: return false;
            }
        }

        private boolean backend(String a) throws UsageException {
            switch (a) {
                case "--cpu": eb.backend(EngineConfig.BackendMode.CPU); return true;
                case "--gpu": eb.backend(EngineConfig.BackendMode.AUTO); return true;
                case "--backend":
                    try {
                        eb.backend(EngineConfig.BackendMode.parse(value(a)));
                    } catch (IllegalArgumentException e) {
                        throw new UsageException(a + ": " + e.getMessage());
                    }
                    return true;
                case "--device": case "-device": case "-dev": eb.deviceFilter(value(a)); return true;
                case "--ram-cache-mb": eb.ramCacheMb(longValue(a, EngineConfig.RAM_CACHE_AUTO, Long.MAX_VALUE)); return true;
                case "--vram-depth": eb.vramDepth(intValue(a, 0, Integer.MAX_VALUE)); return true;
                default: return false;
            }
        }

        private boolean memory(String a) throws UsageException {
            switch (a) {
                case "--ctx": case "-c": case "--ctx-size": case "--max-context":
                    mb.nCtx(intValue(a, 0, Integer.MAX_VALUE));
                    return true;
                case "--kv-type": {
                    KvDType t = kvType(a);
                    mb.typeK(t).typeV(t);
                    return true;
                }
                case "--kv-type-k": case "--cache-type-k": case "-ctk": mb.typeK(kvType(a)); return true;
                case "--kv-type-v": case "--cache-type-v": case "-ctv": mb.typeV(kvType(a)); return true;
                case "--ubatch": case "-ub": case "--ubatch-size":
                    mb.ubatch(intValue(a, 1, Integer.MAX_VALUE));
                    ubatchExplicit = true;
                    return true;
                case "--swa-full": mb.swaFull(true); return true;
                case "--mla-cache": mb.mlaCache(mlaCache(a)); return true;
                case "--no-prefix-reuse": mb.prefixReuse(false); return true;
                case "--context-shift": mb.contextShift(true); return true;
                case "--no-context-shift": mb.contextShift(false); return true;
                case "--no-kv-rotation": mb.kvRotation(false); return true;
                default: return false;
            }
        }

        private boolean sampling(String a) throws UsageException {
            switch (a) {
                case "--seed": case "-s": {
                    long seed = longValue(a, SamplerConfig.RANDOM_SEED, Long.MAX_VALUE);
                    sb.seed(seed);
                    db.seed(seed);
                    eb.seed(seed);
                    return true;
                }
                case "--temp": case "--temperature": {
                    float t = floatValue(a, 0f, Float.MAX_VALUE);
                    sb.temperature(t);
                    db.temperature(t);
                    return true;
                }
                case "--top-k": {
                    int k = intValue(a, 0, Integer.MAX_VALUE);
                    sb.topK(k);
                    db.topK(k);
                    return true;
                }
                case "--top-p": {
                    float p = floatValue(a, 0f, 1f);
                    sb.topP(p);
                    db.topP(p);
                    return true;
                }
                case "--min-p": sb.minP(floatValue(a, 0f, 1f)); return true;
                case "--typical": case "--typical-p": sb.typicalP(floatValue(a, 0f, 1f)); return true;
                case "--presence-penalty":
                    sb.presencePenalty(floatValue(a, -Float.MAX_VALUE, Float.MAX_VALUE));
                    return true;
                case "--frequency-penalty":
                    sb.frequencyPenalty(floatValue(a, -Float.MAX_VALUE, Float.MAX_VALUE));
                    return true;
                case "--ignore-eos": ignoreEos = true; return true;
                case "--reverse-prompt": case "-r": {
                    String stop = value(a);
                    if (stop.isEmpty()) {
                        throw new UsageException(a + " needs non-empty text");
                    }
                    stopStrings.add(stop);
                    return true;
                }
                case "--repeat-penalty": sb.repeatPenalty(floatValue(a, 0f, Float.MAX_VALUE)); return true;
                case "--repeat-last-n":
                    sb.penaltyLastN(intValue(a, SamplerConfig.PENALTY_WHOLE_CONTEXT, Integer.MAX_VALUE));
                    return true;
                case "--repeat-window": {
                    int n = intValue(a, 0, Integer.MAX_VALUE);
                    sb.penaltyLastN(n == 0 ? SamplerConfig.PENALTY_WHOLE_CONTEXT : n);
                    return true;
                }
                default: return false;
            }
        }

        private boolean speculative(String a) throws UsageException {
            switch (a) {
                case "--draft": case "-md": case "--model-draft": eb.draftModelPath(value(a)); return true;
                case "--draft-max": case "--draft-n-max": case "--spec-draft-n-max":
                    eb.draftMax(intValue(a, 1, Integer.MAX_VALUE));
                    return true;
                case "--draft-min": case "--draft-n-min": case "--spec-draft-n-min":
                    eb.draftMin(intValue(a, 0, Integer.MAX_VALUE));
                    return true;
                case "--draft-p-min": case "--spec-draft-p-min": eb.draftPMin(floatValue(a, 0f, 1f)); return true;
                case "--mtp": eb.mtp(true); return true;
                case "--spec-type": {
                    String v = value(a);
                    try {
                        eb.draftType(EngineConfig.DraftType.parse(v));
                    } catch (IllegalArgumentException e) {
                        throw new UsageException(a + ": " + e.getMessage());
                    }
                    return true;
                }
                default: return false;
            }
        }

        private boolean diffusion(String a) throws UsageException {
            switch (a) {
                case "--diffusion-steps": db.steps(intValue(a, 1, Integer.MAX_VALUE)); return true;
                case "--diffusion-eps": {
                    float eps = floatValue(a, 0f, 1f);
                    db.timestepSchedule(eps);
                    diffusionEps = true;
                    return true;
                }
                case "--block-length": case "--diffusion-block-length":
                    db.blockSchedule(intValue(a, 1, Integer.MAX_VALUE));
                    diffusionBlock = true;
                    return true;
                case "--algorithm": case "--diffusion-algorithm":
                    try {
                        db.algorithm(DiffusionConfig.Algorithm.fromCode(intValue(a, 0, 4)));
                    } catch (IllegalArgumentException e) {
                        throw new UsageException(a + ": " + e.getMessage());
                    }
                    return true;
                case "--diffusion-alg-temp": db.algTemp(floatValue(a, 0f, Float.MAX_VALUE)); return true;
                case "--cfg": case "--diffusion-cfg-scale": db.cfgScale(floatValue(a, 0f, Float.MAX_VALUE)); return true;
                case "--diffusion-add-gumbel-noise": db.addGumbelNoise(true); return true;
                default: return false;
            }
        }

        private boolean heads(String a) throws UsageException {
            switch (a) {
                case "--embed-preview": embedPreview = intValue(a, 0, Integer.MAX_VALUE); return true;
                case "--token-types": tokenTypes = intList(a, 0); return true;
                case "--document": document = value(a); return true;
                case "--decision-order": decisionOrder = intList(a, Integer.MIN_VALUE); return true;
                case "--audio-codes": audioCodes = intList(a, 0); return true;
                case "--audio-out": audioOut = value(a); return true;
                default: return false;
            }
        }

        /** {@code --lora FNAME[:SCALE][,...]}: each entry at scale 1 unless it carries one. */
        private void lora(String value) throws UsageException {
            try {
                List<EngineConfig.LoraSpec> parsed = EngineConfig.LoraSpec.parseList(value);
                if (parsed.isEmpty()) {
                    throw new UsageException("--lora needs at least one adapter path");
                }
                loras.addAll(parsed);
            } catch (IllegalArgumentException e) {
                throw new UsageException("--lora: " + e.getMessage());
            }
        }

        /**
         * {@code --lora-scaled FNAME:SCALE[,...]} (llama.cpp common/arg.cpp:2978-2990, every entry needs a scale) or
         * the two-argument form {@code --lora-scaled FNAME SCALE}.
         */
        private void loraScaled(String flag) throws UsageException {
            String first = value(flag);
            if (i + 1 < args.length && isFloat(args[i + 1]) && !hasScale(first)) {
                float scale = Float.parseFloat(args[++i]);
                addLora(first, scale, flag);
                return;
            }
            for (String entry : first.split(",")) {
                String e = entry.trim();
                if (e.isEmpty()) {
                    continue;
                }
                int colon = e.lastIndexOf(':');
                if (colon <= 0 || !isFloat(e.substring(colon + 1))) {
                    throw new UsageException(flag + " entries are FNAME:SCALE, got '" + e + "'");
                }
                addLora(e.substring(0, colon), Float.parseFloat(e.substring(colon + 1)), flag);
            }
        }

        private void addLora(String path, float scale, String flag) throws UsageException {
            try {
                loras.add(new EngineConfig.LoraSpec(path, scale));
            } catch (IllegalArgumentException e) {
                throw new UsageException(flag + ": " + e.getMessage());
            }
        }

        private static boolean hasScale(String entry) {
            int colon = entry.lastIndexOf(':');
            return colon > 0 && isFloat(entry.substring(colon + 1));
        }

        private static boolean isFloat(String s) {
            try {
                return Float.isFinite(Float.parseFloat(s.trim()));
            } catch (NumberFormatException e) {
                return false;
            }
        }

        private void finish() throws UsageException {
            if (diffusionEps && diffusionBlock) {
                throw new UsageException("--diffusion-eps (timestep schedule) and --block-length (block schedule)"
                        + " exclude each other");
            }
            if (benchmark && benchStreaming) {
                throw new UsageException("--benchmark and --bench-streaming exclude each other");
            }
            try {
                this.engine = eb.memory(mb.build()).loras(loras).build();
                this.generation = GenerationRequest.builder().maxNewTokens(maxTokens).sampler(sb.build())
                        .stopStrings(stopStrings).ignoreEos(ignoreEos).build();
                this.diffusion = db.build();
            } catch (IllegalArgumentException e) {
                throw new UsageException(e.getMessage());
            }
            if (engine.draftModelPath() != null && engine.mtp()) {
                throw new UsageException("--draft and --mtp exclude each other: a target binds one drafter");
            }
        }

        private String value(String flag) throws UsageException {
            if (i + 1 >= args.length) {
                throw new UsageException(flag + " needs a value");
            }
            return args[++i];
        }

        private int intValue(String flag, int min, int max) throws UsageException {
            String v = value(flag);
            try {
                int n = Integer.parseInt(v.trim());
                if (n < min || n > max) {
                    throw new UsageException(flag + " " + v + " is outside [" + min + ", " + max + "]");
                }
                return n;
            } catch (NumberFormatException e) {
                throw new UsageException(flag + " needs an integer, got '" + v + "'");
            }
        }

        private long longValue(String flag, long min, long max) throws UsageException {
            String v = value(flag);
            try {
                long n = Long.parseLong(v.trim());
                if (n < min || n > max) {
                    throw new UsageException(flag + " " + v + " is outside [" + min + ", " + max + "]");
                }
                return n;
            } catch (NumberFormatException e) {
                throw new UsageException(flag + " needs an integer, got '" + v + "'");
            }
        }

        private float floatValue(String flag, float min, float max) throws UsageException {
            String v = value(flag);
            try {
                float f = Float.parseFloat(v.trim());
                if (!(f >= min && f <= max)) {
                    throw new UsageException(flag + " " + v + " is outside [" + min + ", " + max + "]");
                }
                return f;
            } catch (NumberFormatException e) {
                throw new UsageException(flag + " needs a number, got '" + v + "'");
            }
        }

        private int[] intList(String flag, int min) throws UsageException {
            String v = value(flag);
            List<Integer> out = new ArrayList<>();
            for (String part : v.split("[,\\s]+")) {
                if (part.isEmpty()) {
                    continue;
                }
                try {
                    int n = Integer.parseInt(part);
                    if (n < min) {
                        throw new UsageException(flag + " value " + n + " is below " + min);
                    }
                    out.add(n);
                } catch (NumberFormatException e) {
                    throw new UsageException(flag + " needs comma-separated integers, got '" + part + "'");
                }
            }
            if (out.isEmpty()) {
                throw new UsageException(flag + " needs at least one integer");
            }
            int[] a = new int[out.size()];
            for (int k = 0; k < a.length; k++) {
                a[k] = out.get(k);
            }
            return a;
        }

        private KvDType kvType(String flag) throws UsageException {
            try {
                return KvDType.parse(value(flag), flag);
            } catch (IllegalArgumentException e) {
                throw new UsageException(e.getMessage());
            }
        }

        private MemoryConfig.MlaCacheMode mlaCache(String flag) throws UsageException {
            String v = value(flag).trim().toUpperCase(Locale.ROOT);
            for (MemoryConfig.MlaCacheMode m : MemoryConfig.MlaCacheMode.values()) {
                if (m.name().equals(v)) {
                    return m;
                }
            }
            throw new UsageException(flag + " " + v.toLowerCase(Locale.ROOT)
                    + " is invalid; use auto, latent or decompressed");
        }
    }

    /** The usage text. */
    static List<String> usage() {
        List<String> u = new ArrayList<>();
        Collections.addAll(u,
                "Usage: LLMDemo [model.gguf] [options]",
                "",
                "Model and prompt:",
                "  --model, -m <path>           GGUF model (or NEURALFREE_MODEL / -Dneuralfree.model)",
                "  --prompt, -p <text>          Run one prompt instead of the console chat",
                "  --system <text>              System prompt (default: '" + DEFAULT_SYSTEM_PROMPT + "')",
                "  --raw                        Use the prompt verbatim (no chat template)",
                "  --thinking                   Enable the chat template's thinking mode",
                "  --max-tokens, -n <n>         Max new tokens, -1 = until EOS (default: " + DEFAULT_MAX_TOKENS + ")",
                "  --non-interactive            Without --prompt, read the prompt from standard input",
                "  --mode <m>                   generate | embed | classify | rerank | diffusion | audio",
                "                               (default: from the model's head: decision -> classify, rank ->",
                "                               rerank with --document else classify, pooled -> embed;",
                "                               otherwise from its support class)",
                "  --embed                      Same as --mode embed",
                "",
                "Backend:",
                "  --gpu | --cpu                GPU (best backend, default) or the CPU engine",
                "  --backend <type>             auto | opencl | cuda | vulkan | cpu",
                "  --device, -device <substr>   Use the GPU whose name contains <substr>",
                "  --ram-cache-mb <n>           Host cache of streamed layers in MB (-1 = automatic)",
                "  --vram-depth <n>             Streamed-layer slots in VRAM (0 = automatic)",
                "  --no-mmap                    Read tensors into memory instead of mapping them",
                "",
                "Memory:",
                "  --ctx, -c <n>                Context length (0 = fit to VRAM / model); alias --max-context",
                "  --kv-type <t>                KV cache type for K and V (default f16): f32 | f16 | bf16 | q8_0 |",
                "                               q4_0 | q4_1 | iq4_nl | q5_0 | q5_1",
                "  --kv-type-k / --kv-type-v <t> KV cache type of K / V only (MLA and DeepSeek-V4 need K = V)",
                "  --ubatch, -ub <n>            Tokens per forward step (default 512; pooled modes "
                        + POOLED_UBATCH + " or ctx)",
                "  --swa-full                   Full-size caches for sliding-window layers",
                "  --mla-cache <m>              auto | latent | decompressed",
                "  --no-prefix-reuse            Re-evaluate the whole prompt every turn",
                "  --context-shift              Drop old positions when the context is full",
                "  --no-kv-rotation             Store quantized KV rows without rotation",
                "",
                "Sampling (defaults apply where no -Dneuralfree.* sampler property is set):",
                "  --temp <f>                   Temperature, 0 = greedy (default " + DEFAULT_TEMPERATURE + ")",
                "  --top-k <n>                  Top-k, 0 = off (default " + DEFAULT_TOP_K + ")",
                "  --top-p <f>                  Top-p (default " + DEFAULT_TOP_P + ")",
                "  --min-p <f>                  Min-p (default " + DEFAULT_MIN_P + ")",
                "  --typical, --typical-p <f>   Locally typical sampling, 1 = off (default 1)",
                "  --repeat-penalty <f>         Repetition penalty, 1 = off (default " + DEFAULT_REPEAT_PENALTY + ")",
                "  --presence-penalty <f>       Presence penalty, 0 = off (default 0)",
                "  --frequency-penalty <f>      Frequency penalty, 0 = off (default 0)",
                "  --repeat-last-n <n>          Penalty window, 0 = off, -1 = whole context (default "
                        + DEFAULT_REPEAT_WINDOW + ")",
                "  --repeat-window <n>          Penalty window, 0 = whole context",
                "  --seed, -s <n>               Sampler seed (-1 = random)",
                "  --ignore-eos                 Keep generating past end-of-generation tokens",
                "  --reverse-prompt, -r <text>  Stop when the output contains <text> (repeatable)",
                "",
                "Speculative decoding:",
                "  --draft, -md <gguf>          Draft model: dflash, eagle3, gemma4-assistant, a GGUF of the target's",
                "                               model with NextN/MTP blocks (MTP-only or full; only its MTP blocks",
                "                               draft) or a small LM",
                "  --mtp                        Draft with the target's own NextN/MTP head",
                "  --spec-type <t>              How the draft file drafts (default auto): auto | draft-mtp (its",
                "                               NextN/MTP blocks; without --draft, the target's own) | draft-simple",
                "                               (a separate draft model, NextN/MTP blocks or not)",
                "  --draft-max <n>              Max drafted tokens per round (default 3)",
                "  --draft-min <n>              Min draft length that is verified (default 0)",
                "  --draft-p-min <p>            Stop drafting below this probability (default 0)",
                "",
                "LoRA adapters (repeatable):",
                "  --lora <gguf>[:scale][,...]  Apply adapters (scale 1 unless given)",
                "  --lora-scaled <gguf>:<scale>[,...] | --lora-scaled <gguf> <scale>",
                "",
                "Diffusion (--mode diffusion):",
                "  --diffusion-steps <n>        Denoising steps (default 128)",
                "  --diffusion-eps <f>          Timestep schedule epsilon (Dream; default 0.001)",
                "  --block-length <n>           Block schedule with blocks of <n> tokens (LLaDA)",
                "  --algorithm <0-4>            origin | entropy | margin | random | confidence (default 4)",
                "  --diffusion-alg-temp <f>     Confidence temperature",
                "  --cfg <f>                    Classifier-free guidance scale (0 = off)",
                "  --diffusion-add-gumbel-noise Gumbel noise on the logits",
                "",
                "Embeddings, classifiers, audio:",
                "  --embed-preview <n>          Leading dimensions to print (default 8)",
                "  --token-types <csv>          Segment id per token (BERT token types)",
                "  --document <text>            Document scored against the prompt (--mode rerank)",
                "  --decision-order <csv>       Decision code per token (clef / vllm-decision2 heads)",
                "  --audio-codes <csv>          WavTokenizer codes (default: parsed from the prompt)",
                "  --audio-out <file>           Write audio features / hidden rows as float32 little-endian",
                "",
                "Output and diagnostics:",
                "  --show-all-channels          Print reasoning and tool-call text with headers (default)",
                "  --content-only               Print only the answer channel",
                "  --debug-prompt               Print the formatted prompt and its tokens",
                "  --vram-budget                Print the memory plan and device memory, then exit",
                "  --benchmark                  GPU decode profiling benchmark",
                "  --bench-streaming            GPU compute / transfer overlap benchmark",
                "  --stats-file <path>          Append the session's statistics to <path>",
                "  --help, -h                   Show this help");
        return u;
    }
}
