package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateDetector;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateKwargs;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.engine.ControlVectorSpec;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LoraSpec;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import com.eonmux.neuralfree.gguf.load.RopeOverrides;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_MAX_TOKENS;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_MIN_P;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_REPEAT_PENALTY;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_REPEAT_WINDOW;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_SYSTEM_PROMPT;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_TEMPERATURE;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_TOP_K;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.DEFAULT_TOP_P;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.Mode;
import static com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;

/**
 * The parser behind {@link LLMDemoOptions#parse(String[])}: walks the command line into the mutable state of one
 * parse, validating flag values and translating them into the engine, sampler, generation and diffusion builders.
 */
final class LLMDemoOptionsParser {
    private final String[] args;
    private int i;

    String modelPath;
    String statsFile;
    String prompt;
    String systemPrompt = DEFAULT_SYSTEM_PROMPT;
    boolean raw;
    Boolean thinking = Boolean.FALSE;
    /** The template variables of {@link #templateVariable} and {@code --reasoning}, in flag order. */
    final Map<String, Object> templateKwargs = new LinkedHashMap<>();
    boolean debugPrompt;
    boolean nonInteractive;
    Mode mode = Mode.AUTO;
    int maxTokens = DEFAULT_MAX_TOKENS;
    boolean maxTokensSet;
    boolean showAllChannels = true;
    boolean benchmark;
    boolean benchStreaming;
    boolean memoryReportOnly;
    boolean noMmap;
    /** {@code --no-jinja}: the model's template is replaced by the built-in its text detects. */
    boolean noJinja;
    /** The template text the chat-template overrides contribute ({@code ChatTemplateSource#overrideValueText}), or null. */
    String templateRaw;
    int embedPreview = 8;
    int[] tokenTypes;
    String document;
    int[] decisionOrder;
    int[] audioCodes;
    String audioOut;
    String mmproj;
    /** The {@code --mmproj-url} projector URL, or null. */
    String mmprojUrl;
    /** {@code --no-mmproj} / {@code --no-mmproj-auto}: no projector, and none is looked up. */
    boolean noMmproj;
    /** The last of {@code --mmproj-offload} (TRUE) / {@code --no-mmproj-offload} (FALSE) / {@code -mmdev}, or null. */
    Boolean mmprojOffload;
    /** The {@code --mmproj-device}/{@code -mmdev} selector, or null. */
    String mmprojDevice;
    int imageMinTokens;
    int imageMaxTokens;
    String audioLang;
    String voice;
    String audioSpeaker;
    int audioFft;
    int audioHop;
    int audioRate;
    boolean help;
    boolean version;
    boolean listDevices;
    boolean cacheList;
    String serverBase;
    String apiKey;
    String outFile;
    boolean multilineInput;
    boolean interactive;
    boolean interactiveFirst;
    String inPrefix;
    String inSuffix;
    boolean inPrefixBos;
    boolean escape = true;
    /** The first flag given that configures a locally loaded model, or null. */
    String localFlag;
    boolean singleTurn;
    final List<LLMDemoOptions.MediaFile> media = new ArrayList<>();
    /** {@code -cnv} (TRUE), {@code -no-cnv} (FALSE) or neither (null). */
    Boolean conversation;
    EngineConfig engine;
    boolean ubatchExplicit;
    final List<String> stopStrings = new ArrayList<>();
    boolean ignoreEos;
    GenerationRequest generation;
    DiffusionConfig diffusion;
    String exportGguf;
    GgufWriter.Mode exportMode = GgufWriter.Mode.NATIVE;
    boolean exportOverwrite;
    String hfRepo;
    String hfFile;
    /** The {@code --prompt-cache} session file, or null. */
    String promptCache;
    boolean promptCacheAll;
    boolean promptCacheRo;
    /** The {@code --grammar}/{@code --json-schema} constraint of the generation request, or null. */
    ConstraintSpec constraint;
    /** The {@code --logit-bias} entries, token id to bias, in flag order. */
    final Map<Integer, Float> logitBias = new LinkedHashMap<>();
    /** The {@code --dry-sequence-breaker} list, null until the first flag clears llama.cpp's defaults. */
    List<String> dryBreakers;
    /** The {@code --reasoning-budget} token count, {@link GenerationRequest#REASONING_BUDGET_DISABLED} for none. */
    int reasoningBudget = GenerationRequest.REASONING_BUDGET_DISABLED;
    /** The {@code --reasoning-budget-message} text, or null. */
    String reasoningBudgetMessage;
    /** The {@code --grp-attn-n} self-extend factor, {@link GenerationRequest#GRP_ATTN_OFF} for off. */
    int grpAttnN = GenerationRequest.GRP_ATTN_OFF;
    /** The {@code --grp-attn-w} self-extend width, {@link GenerationRequest#DEFAULT_GRP_ATTN_W}. */
    int grpAttnW = GenerationRequest.DEFAULT_GRP_ATTN_W;

    private final EngineConfig.Builder eb;
    private final MemoryConfig.Builder mb;
    /** The model-load flags shared with the batched tools, writing {@link #eb} and {@link #mb}. */
    private final EngineLoadFlags load;
    private final SpeculativeFlags spec;
    private final SamplerConfig.Builder sb;
    private final DiffusionConfig.Builder db = DiffusionConfig.builder();
    private final List<LoraSpec> loras = new ArrayList<>();
    private final List<ControlVectorSpec> controlVectors = new ArrayList<>();
    private int controlVectorLayerStart;
    private int controlVectorLayerEnd;
    private final ModelSourceOptions.Builder hb;
    private boolean diffusionEps;
    private boolean diffusionBlock;

    LLMDemoOptionsParser(String[] args) throws UsageException {
        this.args = args.clone();
        EngineConfig base;
        try {
            base = EngineConfig.fromSystemProperties();
            templateKwargs.putAll(ChatTemplateKwargs.fromProperties(System.getProperties()));
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        this.eb = base.toBuilder();
        this.mb = base.memory().toBuilder();
        this.load = new EngineLoadFlags(eb, mb, base.moeCache(), base.eplb());
        this.spec = new SpeculativeFlags(eb, mb, base);
        this.loras.addAll(base.loras());
        this.controlVectors.addAll(base.controlVectors());
        this.controlVectorLayerStart = base.controlVectorLayerStart();
        this.controlVectorLayerEnd = base.controlVectorLayerEnd();
        this.ubatchExplicit = System.getProperty(MemoryConfig.PROP_UBATCH) != null;
        this.sb = samplerBase(base.samplerDefaults());
        diffusionBase(base.samplerDefaults());
        this.hb = base.sourceOptions().toBuilder();
    }

    /**
     * The property sampler with the demo defaults, as fallbacks, in place of every setting no property gives
     * (temperature, top-k, top-p, min-p, repeat penalty and window); typical-p, presence and frequency
     * penalties, the seed and greedy keep their property or llama.cpp default values. Property values stay
     * explicit, and flags make their settings explicit.
     */
    private static SamplerConfig.Builder samplerBase(SamplerConfig fromProperties) {
        SamplerConfig.Builder b = fromProperties.toBuilder();
        if (!fromProperties.isExplicit(SamplerConfig.Setting.TEMPERATURE)) {
            b.fallbackTemperature(DEFAULT_TEMPERATURE);
        }
        if (!fromProperties.isExplicit(SamplerConfig.Setting.TOP_K)) {
            b.fallbackTopK(DEFAULT_TOP_K);
        }
        if (!fromProperties.isExplicit(SamplerConfig.Setting.TOP_P)) {
            b.fallbackTopP(DEFAULT_TOP_P);
        }
        if (!fromProperties.isExplicit(SamplerConfig.Setting.MIN_P)) {
            b.fallbackMinP(DEFAULT_MIN_P);
        }
        if (!fromProperties.isExplicit(SamplerConfig.Setting.REPEAT_PENALTY)) {
            b.fallbackRepeatPenalty(DEFAULT_REPEAT_PENALTY);
        }
        if (!fromProperties.isExplicit(SamplerConfig.Setting.PENALTY_LAST_N)) {
            b.fallbackPenaltyLastN(DEFAULT_REPEAT_WINDOW);
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
        boolean cliCvecs = false;
        for (i = 0; i < args.length; i++) {
            String a = args[i];
            if (i == 0 && !a.startsWith("-")) {
                modelPath = a;
                continue;
            }
            if (a.equals("--lora") || a.equals("--lora-scaled")) {
                localOnly(a);
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
            if (a.equals("--control-vector") || a.equals("--control-vector-scaled")) {
                localOnly(a);
                if (!cliCvecs) {
                    controlVectors.clear();
                    cliCvecs = true;
                }
                if (a.equals("--control-vector")) {
                    controlVector(value(a));
                } else {
                    controlVectorScaled(a);
                }
                continue;
            }
            if (a.equals("--control-vector-layer-range")) {
                localOnly(a);
                controlVectorLayerStart = intValue(a, Integer.MIN_VALUE, Integer.MAX_VALUE);
                controlVectorLayerEnd = intValue(a, Integer.MIN_VALUE, Integer.MAX_VALUE);
                continue;
            }
            if (general(a) || templateVariable(a)) {
                continue;
            }
            if (chatTemplate(a) || loadFlag(a) || memory(a)) {
                localOnly(a);
                continue;
            }
            if (sampling(a)) {
                continue;
            }
            if (speculative(a) || diffusion(a) || heads(a) || loadingExtras(a) || huggingFace(a)
                    || exporting(a)) {
                localOnly(a);
                continue;
            }
            throw new UsageException("unknown option " + a + " (see --help)");
        }
        finish();
    }

    /** Records a flag that only a locally loaded model reads, for the client-mode check. */
    private void localOnly(String flag) {
        if (localFlag == null) {
            localFlag = flag;
        }
    }

    private boolean general(String a) throws UsageException {
        switch (a) {
            case "--model": case "-m": modelPath = value(a); return true;
            case "--prompt": case "-p": prompt = value(a); return true;
            case "--system": case "--system-prompt": case "-sys": systemPrompt = value(a); return true;
            case "-f": case "--file": {
                // llama.cpp -f: the file's text is the prompt, one trailing newline dropped
                prompt = readTextFile(a, value(a));
                return true;
            }
            case "-sysf": case "--system-prompt-file":
                systemPrompt = readTextFile(a, value(a));
                return true;
            case "--jinja": noJinja = false; return true;
            case "--no-jinja": noJinja = true; return true;
            case "--max-tokens": case "--n-predict": case "-n":
                maxTokens = intValue(a, -1, Integer.MAX_VALUE);
                if (maxTokens == 0) {
                    throw new UsageException(a + " must be positive, or -1 for no limit");
                }
                maxTokensSet = true;
                return true;
            case "--mode": mode = Mode.parse(value(a)); return true;
            case "--embed": mode = Mode.EMBED; return true;
            case "--raw": raw = true; return true;
            case "--thinking": thinking = Boolean.TRUE; return true;
            case "-rea": case "--reasoning":
                thinking = ReasoningFlags.reasoning(a, value(a), templateKwargs);
                return true;
            case "--debug-prompt": debugPrompt = true; return true;
            case "--non-interactive": nonInteractive = true; return true;
            case "--show-all-channels": showAllChannels = true; return true;
            case "--content-only": showAllChannels = false; return true;
            case "--benchmark": benchmark = true; return true;
            case "--bench-streaming": benchStreaming = true; return true;
            case "--vram-budget": case "--memory-report": memoryReportOnly = true; return true;
            case "--no-mmap": noMmap = true; return true;
            case "--stats-file": statsFile = value(a); return true;
            case "--server-base": serverBase = value(a); return true;
            case "--api-key": apiKey = value(a); return true;
            case "--output": case "--output-file": case "--out-file": case "-o": outFile = value(a); return true;
            case "--multiline-input": case "-mli": multilineInput = true; return true;
            case "--interactive": case "-i": interactive = true; return true;
            case "--interactive-first": case "-if": interactiveFirst = true; return true;
            case "--in-prefix": inPrefix = value(a); return true;
            case "--in-suffix": inSuffix = value(a); return true;
            case "--in-prefix-bos": inPrefixBos = true; return true;
            case "--escape": case "-e": escape = true; return true;
            case "--no-escape": escape = false; return true;
            case "--single-turn": case "-st": singleTurn = true; return true;
            // llama.cpp common/arg.cpp: each flag takes one comma-separated list, the kind from the flag
            case "--image": mediaFiles(a, "image"); return true;
            case "--audio": mediaFiles(a, "audio"); return true;
            case "--video": mediaFiles(a, "video"); return true;
            // llama.cpp common/arg.cpp: the prompt eval state file of the completion loop
            case "--prompt-cache": localOnly(a); promptCache = value(a); return true;
            case "--prompt-cache-all": localOnly(a); promptCacheAll = true; return true;
            case "--prompt-cache-ro": localOnly(a); promptCacheRo = true; return true;
            case "--conversation": case "-cnv": conversation = Boolean.TRUE; return true;
            case "--no-conversation": case "-no-cnv": conversation = Boolean.FALSE; return true;
            case "--help": case "-h": help = true; return true;
            case "--version": version = true; return true;
            case "--list-devices": listDevices = true; return true;
            case "-cl": case "--cache-list": cacheList = true; return true;
            default: return false;
        }
    }

    /**
     * The chat template variables of every rendered chat, in llama.cpp's form ({@code default_template_kwargs}):
     * {@code --chat-template-kwargs JSON} sets each member of a JSON object ({@link ChatTemplateKwargs#parse}),
     * {@code --reasoning-effort LEVEL} sets {@code reasoning_effort}, {@code default} removing it so the template's
     * own default holds, and {@code --reasoning-preserve}/{@code --no-reasoning-preserve} set
     * {@code preserve_reasoning} ({@link ReasoningFlags}); a later flag overrides an earlier one. Without either
     * preserve flag, {@code preserve_reasoning} keeps llama.cpp's default, true ({@link ChatFormatter}).
     */
    private boolean templateVariable(String a) throws UsageException {
        switch (a) {
            case "--chat-template-kwargs": {
                String json = value(a);
                try {
                    templateKwargs.putAll(ChatTemplateKwargs.parse(json, a));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(e.getMessage());
                }
                return true;
            }
            case "--reasoning-effort": ReasoningFlags.effort(a, value(a), templateKwargs); return true;
            case "--reasoning-preserve": ReasoningFlags.preserve(true, templateKwargs); return true;
            case "--no-reasoning-preserve": ReasoningFlags.preserve(false, templateKwargs); return true;
            default: return false;
        }
    }

    /**
     * The chat-template overrides: each writes the template text it contributes, as llama.cpp's flags keep
     * it, into the local model's {@code tokenizer.chat_template} metadata — a built-in name resolves to its
     * source when the model loads, and the text is what the non-Jinja route detects from.
     */
    private boolean chatTemplate(String a) throws UsageException {
        switch (a) {
            case "--chat-template": {
                String template = value(a);
                try {
                    templateRaw = ChatTemplateSource.overrideValueText(template);
                } catch (IllegalArgumentException | UncheckedIOException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                hb.metadataOverride(ChatTemplateSource.KEY_DEFAULT + "=str:" + templateRaw);
                return true;
            }
            case "--chat-template-file": {
                Path file = pathValue(a);
                try {
                    templateRaw = ChatTemplateSource.overrideText(file);
                } catch (IllegalArgumentException | UncheckedIOException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                hb.metadataOverride(ChatTemplateSource.KEY_DEFAULT + "=str:" + templateRaw);
                return true;
            }
            default: return false;
        }
    }

    /** The shared model-load flags ({@link EngineLoadFlags}): backend, placement, flash attention, KV types. */
    private boolean loadFlag(String a) throws UsageException {
        try {
            int last = load.parse(args, i);
            if (last < 0) {
                return false;
            }
            i = last;
            return true;
        } catch (EngineLoadFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    private boolean memory(String a) throws UsageException {
        switch (a) {
            case "--ctx": case "-c": case "--ctx-size": case "--max-context":
                mb.nCtx(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            case "--ubatch": case "-ub": case "--ubatch-size":
                mb.ubatch(intValue(a, 1, Integer.MAX_VALUE));
                ubatchExplicit = true;
                return true;
            case "--swa-full": mb.swaFull(true); return true;
            case "-np": case "--parallel": mb.parallel(intValue(a, 1, MemoryConfig.MAX_PARALLEL)); return true;
            case "--mla-cache": mb.mlaCache(mlaCache(a)); return true;
            case "--no-prefix-reuse": mb.prefixReuse(false); return true;
            case "--ctx-checkpoints": case "--swa-checkpoints": case "-ctxcp":
                mb.ctxCheckpoints(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            case "--checkpoint-min-step": case "-cms":
                mb.checkpointMinStep(intValue(a, 0, Integer.MAX_VALUE));
                return true;
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
            case "--min-keep": sb.minKeep(intValue(a, 0, Integer.MAX_VALUE)); return true;
            case "--samplers": {
                // llama.cpp --samplers: ';'-separated sampler names, the chain order
                List<StageKind> order = new ArrayList<>();
                String joined = value(a);
                for (String part : joined.split(";", -1)) {
                    String name = part.trim();
                    if (name.isEmpty()) {
                        throw new UsageException(a + " has an empty stage name in '" + joined + "'");
                    }
                    try {
                        order.add(StageKind.ofName(name));
                    } catch (IllegalArgumentException e) {
                        throw new UsageException(a + ": " + e.getMessage());
                    }
                }
                try {
                    sb.samplers(order);
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            }
            case "--sampler-seq": case "--sampling-seq": {
                // llama.cpp --sampler-seq: one character per sampler, the chain order
                String chars = value(a);
                try {
                    sb.samplers(StageKind.ofChars(chars));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            }
            case "--adaptive-target":
                // llama.cpp: negative disables adaptive-p
                sb.adaptiveTarget(floatValue(a, -Float.MAX_VALUE, 1f));
                return true;
            case "--adaptive-decay": sb.adaptiveDecay(floatValue(a, 0f, 0.99f)); return true;
            case "--reasoning-budget":
                reasoningBudget = intValue(a, GenerationRequest.REASONING_BUDGET_DISABLED, Integer.MAX_VALUE);
                return true;
            case "--reasoning-budget-message": reasoningBudgetMessage = value(a); return true;
            case "--grp-attn-n":
            case "-gan":
                // llama.cpp --grp-attn-n: the self-extend group-attention factor
                grpAttnN = intValue(a, GenerationRequest.GRP_ATTN_OFF, Integer.MAX_VALUE);
                return true;
            case "--grp-attn-w":
            case "-gaw":
                // llama.cpp --grp-attn-w: the positions one self-extend group spans
                grpAttnW = intValue(a, 1, Integer.MAX_VALUE);
                return true;
            case "--mirostat": sb.mirostat(intValue(a, SamplerConfig.MIROSTAT_OFF, 2)); return true;
            case "--mirostat-ent": {
                float tau = floatValue(a, 0f, Float.MAX_VALUE);
                if (tau <= 0f) {
                    throw new UsageException(a + " " + tau + " must be > 0");
                }
                sb.mirostatTau(tau);
                return true;
            }
            case "--mirostat-lr": {
                float eta = floatValue(a, 0f, Float.MAX_VALUE);
                if (eta <= 0f) {
                    throw new UsageException(a + " " + eta + " must be > 0");
                }
                sb.mirostatEta(eta);
                return true;
            }
            case "--top-n-sigma": case "--top-nsigma":
                // llama.cpp: negative disables the stage; the sampler applies the same gate
                sb.topNSigma(floatValue(a, -Float.MAX_VALUE, Float.MAX_VALUE));
                return true;
            case "--xtc-probability": sb.xtcProbability(floatValue(a, 0f, 1f)); return true;
            case "--xtc-threshold": sb.xtcThreshold(floatValue(a, 0f, 1f)); return true;
            case "--dry-multiplier": sb.dryMultiplier(floatValue(a, 0f, Float.MAX_VALUE)); return true;
            case "--dry-base": sb.dryBase(floatValue(a, 0f, Float.MAX_VALUE)); return true;
            case "--dry-allowed-length": sb.dryAllowedLength(intValue(a, 0, Integer.MAX_VALUE)); return true;
            case "--dry-penalty-last-n": sb.dryPenaltyLastN(intValue(a, 0, Integer.MAX_VALUE)); return true;
            case "--dry-sequence-breaker": {
                // llama.cpp: the first flag drops the default breakers, 'none' leaves none
                String breaker = value(a);
                if (dryBreakers == null) {
                    dryBreakers = new ArrayList<>();
                }
                if (breaker.equals("none")) {
                    dryBreakers.clear();
                } else {
                    dryBreakers.add(breaker);
                }
                return true;
            }
            case "--dynatemp-range": sb.dynatempRange(floatValue(a, 0f, Float.MAX_VALUE)); return true;
            case "--dynatemp-exp": case "--dynatemp-exponent":
                sb.dynatempExponent(floatValue(a, 0f, Float.MAX_VALUE));
                return true;
            case "-l": case "--logit-bias": {
                int id = 0;
                float bias = 0f;
                String spec = value(a);
                int sign = -1;
                for (int k = 1; k < spec.length(); k++) {
                    char c = spec.charAt(k);
                    if (c == '+' || c == '-') {
                        sign = k;
                        break;
                    }
                }
                if (sign <= 0) {
                    throw new UsageException(a + " " + spec + " is not TOKEN_ID(+/-)BIAS, e.g. 15043+1");
                }
                try {
                    id = Integer.parseInt(spec.substring(0, sign).trim());
                    bias = Float.parseFloat(spec.substring(sign).trim());
                } catch (NumberFormatException e) {
                    throw new UsageException(a + " " + spec + " is not TOKEN_ID(+/-)BIAS, e.g. 15043+1");
                }
                if (id < 0) {
                    throw new UsageException(a + " token id " + id + " is negative");
                }
                if (!Float.isFinite(bias)) {
                    throw new UsageException(a + " bias of token " + id + " is not finite");
                }
                logitBias.put(id, bias);
                return true;
            }
            case "--grammar": constraint = ConstraintSpec.grammar(value(a)); return true;
            case "--grammar-file": constraint = ConstraintSpec.grammar(readTextFile(a, value(a))); return true;
            case "-j": case "--json-schema": constraint = ConstraintSpec.jsonSchema(value(a), null); return true;
            case "-jf": case "--json-schema-file":
                constraint = ConstraintSpec.jsonSchema(readTextFile(a, value(a)), null);
                return true;
            default: return false;
        }
    }

    /** The speculative decoding flags ({@link SpeculativeFlags}). */
    private boolean speculative(String a) throws UsageException {
        try {
            int last = spec.parse(args, i);
            if (last < 0) {
                return false;
            }
            i = last;
            return true;
        } catch (EngineLoadFlags.FlagException e) {
            throw new UsageException(e.getMessage());
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

    /** The loading extras of every source format: metadata, rope, dummy and check flags, and the lazy mode. */
    private boolean loadingExtras(String a) throws UsageException {
        switch (a) {
            case "--override-kv":
                try {
                    hb.metadataOverride(value(a));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--check-tensors": hb.checkTensors(true); return true;
            case "--load-format": {
                String v = value(a).trim().toLowerCase(Locale.ROOT).replace('-', '_');
                if (v.equals("dummy")) {
                    hb.loadFormat(ModelSourceOptions.LoadFormat.DUMMY);
                    return true;
                }
                if (v.equals("auto") || v.equals("full")) {
                    hb.loadFormat(ModelSourceOptions.LoadFormat.FULL);
                    return true;
                }
                throw new UsageException(a + " " + v + " is invalid; use auto, full or dummy");
            }
            case "--no-warmup": hb.warmup(false); return true;
            case "--lazy-mode": {
                TensorReadMode current;
                try {
                    current = TensorReadMode.current();
                } catch (IllegalArgumentException e) {
                    throw new UsageException(e.getMessage());
                }
                if (current.locksMemory()) {
                    throw new UsageException("--lazy-mode conflicts with --mlock: lazily read tensors have no"
                            + " resident bytes to pin");
                }
                System.setProperty(TensorReadMode.PROP_READ_MODE, "lazy");
                return true;
            }
            case "--rope-scaling":
                try {
                    hb.ropeScaling(RopeOverrides.ScalingType.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--rope-scale":
            case "--rope-freq-scale": {
                float v = floatValue(a, 0f, Float.MAX_VALUE);
                try {
                    if (a.equals("--rope-scale")) {
                        hb.ropeScale(v);
                    } else {
                        hb.ropeFreqScale(v);
                    }
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            }
            case "--rope-freq-base":
                hb.ropeFreqBase(floatValue(a, 0f, Float.MAX_VALUE));
                return true;
            case "--yarn-ext-factor":
                hb.yarnExtFactor(floatValue(a, -Float.MAX_VALUE, Float.MAX_VALUE));
                return true;
            case "--yarn-attn-factor":
                hb.yarnAttnFactor(floatValue(a, 0f, Float.MAX_VALUE));
                return true;
            case "--yarn-beta-fast":
                hb.yarnBetaFast(floatValue(a, 0f, Float.MAX_VALUE));
                return true;
            case "--yarn-beta-slow":
                hb.yarnBetaSlow(floatValue(a, 0f, Float.MAX_VALUE));
                return true;
            case "--yarn-orig-ctx":
                hb.yarnOrigCtx(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            default: return false;
        }
    }

    /**
     * The Hugging Face checkpoint flags: each sets the matching {@link ModelSourceOptions} or
     * {@link EngineConfig} field on its builder, on top of the {@code -Dneuralfree.*} property base.
     */
    private boolean huggingFace(String a) throws UsageException {
        switch (a) {
            case "-hf": case "-hfr": case "--hf-repo": {
                String spec = value(a);
                try {
                    // as llama.cpp's -hf, parsed by the same repository expression grammar
                    hfRepo = HubRef.parse(spec).toString();
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            }
            case "-hff": case "--hf-file": hfFile = value(a); return true;
            case "--hf-materialize":
                try {
                    hb.materialize(ModelSourceOptions.Materialize.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--hf-memory-max": hb.memoryBudget(sizeValue(a)); return true;
            case "--hf-cache-dir": hb.cacheDir(pathValue(a)); return true;
            case "--hf-outtype":
                try {
                    hb.outType(ModelSourceOptions.OutType.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--quant-online":
                try {
                    hb.onlineQuant(ModelSourceOptions.OnlineQuant.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--leave-output-tensor": hb.quantizeOutputTensor(false); return true;
            case "--quantization-config":
                try {
                    hb.quantConfigJson(value(a));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--hf-quant-exact": hb.exactQuant(true); return true;
            case "--hf-no-mtp": hb.includeMtp(false); return true;
            case "--hf-draft":
                try {
                    hb.draftMode(ModelSourceOptions.DraftMode.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--hf-st-dense": hb.stDenseModules(boolValue(a)); return true;
            case "--hf-overrides":
                try {
                    hb.configOverridesJson(value(a));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--hf-tokenizer": hb.tokenizerDir(pathValue(a)); return true;
            case "--lora-base-config": hb.loraBaseConfig(pathValue(a)); return true;
            case "--hf-base-dir": hb.baseDir(pathValue(a)); return true;
            case "--act-quant":
                try {
                    eb.activationQuant(ActivationQuant.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            case "--model-sampling-defaults": eb.modelSamplingDefaults(boolValue(a)); return true;
            case "--reasoning-parser": eb.reasoningParser(value(a)); return true;
            default: return false;
        }
    }

    /** The {@code --export-gguf} flags; LLMDemo writes the export and exits instead of generating. */
    private boolean exporting(String a) throws UsageException {
        switch (a) {
            case "--export-gguf": exportGguf = value(a); return true;
            case "--export-mode": {
                String v = value(a).trim().toUpperCase(Locale.ROOT).replace('-', '_');
                for (GgufWriter.Mode m : GgufWriter.Mode.values()) {
                    if (m.name().equals(v)) {
                        exportMode = m;
                        return true;
                    }
                }
                throw new UsageException(a + " is invalid; use native, compat or compat-exact");
            }
            case "--export-overwrite": exportOverwrite = true; return true;
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
            case "--mmproj": mmproj = value(a); return true;
            case "-mmu": case "--mmproj-url": mmprojUrl = url(a); return true;
            case "--mmproj-auto": noMmproj = false; return true;
            case "--no-mmproj": case "--no-mmproj-auto": noMmproj = true; return true;
            case "--mmproj-offload": mmprojOffload = true; return true;
            case "--no-mmproj-offload": mmprojOffload = false; return true;
            case "-mmdev": case "--mmproj-device": {
                mmprojDevice = value(a);
                mmprojOffload = !MmprojOptions.NO_DEVICE.equalsIgnoreCase(mmprojDevice.trim());
                return true;
            }
            case "--image-min-tokens": imageMinTokens = intValue(a, 0, Integer.MAX_VALUE); return true;
            case "--image-max-tokens": imageMaxTokens = intValue(a, 0, Integer.MAX_VALUE); return true;
            case "--audio-lang": audioLang = value(a); return true;
            case "--voice": voice = value(a); return true;
            case "--audio-speaker": audioSpeaker = value(a); return true;
            case "--audio-fft": audioFft = intValue(a, 1, 1 << 20); return true;
            case "--audio-hop": audioHop = intValue(a, 1, 1 << 20); return true;
            case "--audio-rate": audioRate = intValue(a, 1, 1 << 24); return true;
            case "--pooling": {
                try {
                    eb.pooling(PoolingKind.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new UsageException(a + ": " + e.getMessage());
                }
                return true;
            }
            default: return false;
        }
    }

    /** {@code --lora FNAME[:SCALE][,...]}: each entry at scale 1 unless it carries one. */
    private void lora(String value) throws UsageException {
        try {
            List<LoraSpec> parsed = LoraSpec.parseList(value);
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
            loras.add(new LoraSpec(path, scale));
        } catch (IllegalArgumentException e) {
            throw new UsageException(flag + ": " + e.getMessage());
        }
    }

    /** {@code --control-vector FNAME[:SCALE][,...]}: each entry at scale 1 unless it carries one. */
    private void controlVector(String value) throws UsageException {
        try {
            List<ControlVectorSpec> parsed = ControlVectorSpec.parseList(value);
            if (parsed.isEmpty()) {
                throw new UsageException("--control-vector needs at least one vector path");
            }
            controlVectors.addAll(parsed);
        } catch (IllegalArgumentException e) {
            throw new UsageException("--control-vector: " + e.getMessage());
        }
    }

    /**
     * {@code --control-vector-scaled FNAME:SCALE[,...]} (llama.cpp common/arg.cpp:2952-2961, every entry needs a
     * scale) or the two-argument form {@code --control-vector-scaled FNAME SCALE}.
     */
    private void controlVectorScaled(String flag) throws UsageException {
        String first = value(flag);
        if (i + 1 < args.length && isFloat(args[i + 1]) && !hasScale(first)) {
            float scale = Float.parseFloat(args[++i]);
            addControlVector(first, scale, flag);
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
            addControlVector(e.substring(0, colon), Float.parseFloat(e.substring(colon + 1)), flag);
        }
    }

    private void addControlVector(String path, float scale, String flag) throws UsageException {
        try {
            controlVectors.add(new ControlVectorSpec(path, scale));
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
        if (promptCacheRo && promptCache == null) {
            throw new UsageException("--prompt-cache-ro needs --prompt-cache, the file it reads");
        }
        if (promptCacheAll && promptCache == null) {
            throw new UsageException("--prompt-cache-all needs --prompt-cache, the file it writes");
        }
        if (promptCache != null && promptCache.trim().isEmpty()) {
            throw new UsageException("--prompt-cache needs a session file path");
        }
        if (promptCache != null && mode != Mode.AUTO && mode != Mode.GENERATE) {
            throw new UsageException("--prompt-cache runs the generate mode; --mode "
                    + mode.name().toLowerCase(Locale.ROOT) + " holds no sequence to cache");
        }
        if (promptCache != null && (benchmark || benchStreaming)) {
            throw new UsageException("--prompt-cache and --benchmark/--bench-streaming exclude each other:"
                    + " the benchmarks run no generation loop");
        }
        if (promptCache != null && memoryReportOnly) {
            throw new UsageException("--prompt-cache and --vram-budget exclude each other: the memory report"
                    + " runs no generation loop");
        }
        if (promptCache != null && exportGguf != null) {
            throw new UsageException("--prompt-cache and --export-gguf exclude each other: the export runs no"
                    + " generation loop");
        }
        if (diffusionEps && diffusionBlock) {
            throw new UsageException("--diffusion-eps (timestep schedule) and --block-length (block schedule)"
                    + " exclude each other");
        }
        if (benchmark && benchStreaming) {
            throw new UsageException("--benchmark and --bench-streaming exclude each other");
        }
        if (hfRepo != null && modelPath != null) {
            throw new UsageException("--model and -hf exclude each other: the Hub download supplies"
                    + " the model file, a local path names its own");
        }
        if (interactiveFirst) {
            interactive = true;
        }
        validateClientMode();
        validateConversation();
        validateMultimodal();
        if (noJinja && templateRaw != null && ChatTemplateDetector.detect(templateRaw) == null) {
            throw new UsageException("--no-jinja: " + ChatTemplateDetector.refusal(templateRaw));
        }
        String consoleFlag = outFile != null ? "-o/--output" : multilineInput ? "-mli" : singleTurn ? "-st" : null;
        if (consoleFlag != null && mode != Mode.AUTO && mode != Mode.GENERATE) {
            throw new UsageException(consoleFlag + " belongs to the interactive console; --mode "
                    + mode.name().toLowerCase(Locale.ROOT) + " runs no conversation");
        }
        if (interactiveCompletion() && mode != Mode.AUTO && mode != Mode.GENERATE) {
            throw new UsageException("--interactive and --in-prefix/--in-suffix run the completion mode;"
                    + " --mode " + mode.name().toLowerCase(Locale.ROOT) + " has no interactive form");
        }
        try {
            eb.sourceOptions(hb.build()).loras(loras).controlVectors(controlVectors)
                    .controlVectorLayerRange(controlVectorLayerStart, controlVectorLayerEnd);
            spec.apply();
            this.engine = load.build();
            GenerationRequest.Builder gb = GenerationRequest.builder();
            if (maxTokensSet) {
                gb.maxNewTokens(maxTokens);
            } else {
                gb.fallbackMaxNewTokens(maxTokens);
            }
            for (Map.Entry<Integer, Float> e : logitBias.entrySet()) {
                gb.logitBias(e.getKey(), e.getValue());
            }
            gb.constraint(constraint)
                    .reasoningBudget(reasoningBudget)
                    .reasoningBudgetMessage(reasoningBudgetMessage)
                    .grpAttnN(grpAttnN)
                    .grpAttnW(grpAttnW);
            if (dryBreakers != null) {
                sb.drySequenceBreakers(dryBreakers);
            }
            this.generation = gb.sampler(sb.build()).stopStrings(stopStrings).ignoreEos(ignoreEos).build();
            this.diffusion = db.build();
        } catch (EngineLoadFlags.FlagException | IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        if (engine.draftModelPath() != null && engine.mtp()) {
            throw new UsageException("--draft and --mtp exclude each other: a target binds one drafter");
        }
        // --no-mmproj with no media and no projector leaves a text-only run, which keeps both
        boolean mediaRun = (multimodalInputGiven() && !noMmproj) || (mmproj != null && mode != Mode.AUDIO);
        if (serverBase == null && (promptCache != null || engine.speculative() || engine.lookaheadOptions().enabled())
                && mediaRun) {
            String what = promptCache != null ? "--prompt-cache"
                    : engine.lookaheadOptions().enabled() ? "lookahead decoding" : "speculative decoding";
            throw new UsageException(what + " rides the text generator; the media turns of a projector run"
                    + " the batch engine and take neither");
        }
        if (promptCache != null && engine.lookaheadOptions().enabled()) {
            throw new UsageException("--prompt-cache saves the model's single-sequence session, which lookahead"
                    + " decoding runs without (its paged sequences hold the steps); drop one of the two");
        }
    }

    /**
     * The multimodal-input combinations: a media file reaches the prompt as the chat template's media marker
     * ({@link com.eonmux.neuralfree.gguf.chat.ChatMessages#MEDIA_MARKER}), so the local media run needs a
     * projector (an explicit {@code --mmproj} or {@code --mmproj-url}, or the automatic sibling of
     * {@code --mmproj-auto}, resolved when the model loads) and refuses the raw completion; every file attaches
     * to the {@code -p} turn or the session's staged media; a mode without text generation refuses the whole
     * set by name. {@code --mmproj} keeps its speech-generator meaning in the audio mode, which takes no
     * multimodal input.
     */
    private void validateMultimodal() throws UsageException {
        if (mode == Mode.AUDIO && multimodalInputGiven()) {
            throw new UsageException("--mode audio speaks through the --mmproj speech generator; "
                    + "--mmproj-url, --no-mmproj, --mmproj-offload, --mmproj-device, --image-min-tokens,"
                    + " --image-max-tokens and --image/--audio/--video configure a text model's multimodal"
                    + " input");
        }
        boolean mediaFiles = !media.isEmpty();
        boolean projector = mmproj != null || mmprojUrl != null;
        if ((multimodalInputGiven() || (projector && mode != Mode.AUDIO))
                && mode != Mode.AUTO && mode != Mode.GENERATE) {
            throw new UsageException("--mode " + mode.name().toLowerCase(Locale.ROOT) + " reads text only;"
                    + " --mmproj, --mmproj-url, --no-mmproj, --mmproj-offload, --mmproj-device,"
                    + " --image-min-tokens/--image-max-tokens and --image/--audio/--video belong to the"
                    + " generate mode");
        }
        if ((multimodalInputGiven() || (projector && mode != Mode.AUDIO)) && interactiveCompletion()) {
            throw new UsageException("--image/--audio/--video and --mmproj attach to the chat template's media"
                    + " markers; the completion REPL (-i, -if, --in-prefix/--in-suffix) completes plain text");
        }
        if (mediaFiles && serverBase == null && !projector && noMmproj) {
            throw new UsageException("--image/--audio/--video need a projector to encode their files;"
                    + " --no-mmproj turns the projector off");
        }
        if (mediaFiles && serverBase == null && (raw || conversation == Boolean.FALSE)) {
            throw new UsageException("--image/--audio/--video reach the prompt as the chat template's media"
                    + " markers; --raw and -no-cnv complete plain text");
        }
        if (mediaFiles && prompt == null && !nonInteractive) {
            throw new UsageException("--image/--audio/--video attach to the -p prompt's turn; give -p, or"
                    + " attach files in the chat with /image, /audio and /video");
        }
        if (mmprojDevice != null) {
            try {
                MmprojOptions.onDevice(mmprojDevice);
            } catch (IllegalArgumentException e) {
                throw new UsageException("--mmproj-device: " + e.getMessage());
            }
        }
        if (imageMinTokens > 0 && imageMaxTokens > 0 && imageMinTokens > imageMaxTokens) {
            throw new UsageException("--image-min-tokens " + imageMinTokens + " exceeds --image-max-tokens "
                    + imageMaxTokens);
        }
    }

    /**
     * The UTF-8 text of {@code path} ({@code -f}, {@code --grammar-file} and their siblings), one trailing
     * newline dropped as llama.cpp's {@code read_file} does; failures are refused naming the flag.
     */
    private static String readTextFile(String flag, String path) throws UsageException {
        try {
            String text = Files.readString(Paths.get(path));
            return text.endsWith("\n") ? text.substring(0, text.length() - 1) : text;
        } catch (FileNotFoundException | java.nio.file.NoSuchFileException e) {
            throw new UsageException(flag + ": no such file '" + path + "'");
        } catch (IOException e) {
            throw new UsageException(flag + ": cannot read '" + path + "': " + e.getMessage());
        }
    }

    /**
     * Client mode ({@code --server-base}) refuses every setting that belongs to a locally loaded model,
     * naming both flags; it runs the chat only.
     */
    private void validateClientMode() throws UsageException {
        if (serverBase == null) {
            return;
        }
        if (modelPath != null) {
            throw new UsageException("--server-base and --model exclude each other: client mode asks the"
                    + " server for its models");
        }
        if (hfRepo != null) {
            throw new UsageException("--server-base and -hf exclude each other: client mode asks the server"
                    + " for its models");
        }
        if (interactiveCompletion()) {
            throw new UsageException("--server-base chats over the server; completion mode (-i, -if,"
                    + " --in-prefix, --in-suffix) runs a local model");
        }
        if (localFlag != null) {
            throw new UsageException("--server-base and " + localFlag + " exclude each other: the server"
                    + " loads and configures the model; " + localFlag + " configures a local one");
        }
        refuseInClientMode(raw, "--raw", "the server renders the chat template");
        refuseInClientMode(debugPrompt, "--debug-prompt", "the server renders and tokenizes the prompt");
        refuseInClientMode(nonInteractive, "--non-interactive", "client mode runs the interactive chat; give"
                + " the first turn with -p");
        refuseInClientMode(noMmap, "--no-mmap", "the server loads the model");
        refuseInClientMode(noJinja, "--no-jinja", "the server loads the model and picks its chat template");
        refuseInClientMode(statsFile != null, "--stats-file", "the statistics file records local runs");
        if (benchmark || benchStreaming) {
            throw new UsageException("--server-base and --benchmark/--bench-streaming exclude each other:"
                    + " the benchmarks profile a local GPU model");
        }
        if (memoryReportOnly) {
            throw new UsageException("--server-base and --vram-budget exclude each other: the memory plan"
                    + " describes a local model");
        }
        if (exportGguf != null) {
            throw new UsageException("--server-base and --export-gguf exclude each other: the export writes"
                    + " a local model");
        }
        if (mode != Mode.AUTO && mode != Mode.GENERATE) {
            throw new UsageException("--server-base serves the chat mode; --mode "
                    + mode.name().toLowerCase(Locale.ROOT) + " runs a local model");
        }
        requireJsonSchema();
    }

    /**
     * The {@code --json-schema}/{@code -jf} text must be JSON in client mode, where it travels as the request's
     * {@code json_schema} value instead of compiling locally.
     */
    private void requireJsonSchema() throws UsageException {
        if (constraint == null || constraint.kind() != ConstraintSpec.Kind.JSON_SCHEMA) {
            return;
        }
        try {
            Json.parse(constraint.text());
        } catch (JsonException e) {
            throw new UsageException("--json-schema is not valid JSON: " + e.getMessage());
        }
    }

    /**
     * {@code -cnv} runs the chat session and so excludes the completion flags; {@code -no-cnv} turns the chat
     * template off, so it completes the {@code -p} prompt once or, with {@code -i}, runs the completion REPL.
     */
    private void validateConversation() throws UsageException {
        if (conversation == null) {
            return;
        }
        if (conversation && interactiveCompletion()) {
            throw new UsageException("-cnv runs the chat; -i, -if, --in-prefix, --in-suffix and --in-prefix-bos"
                    + " run the completion mode (use -no-cnv with them)");
        }
        if (!conversation) {
            if (serverBase != null) {
                throw new UsageException("--server-base chats over the server's chat template; -no-cnv runs"
                        + " a local raw completion");
            }
            if (prompt == null && !interactiveCompletion()) {
                throw new UsageException("-no-cnv completes the -p prompt without the chat template; give -p,"
                        + " or -i for the interactive completion mode");
            }
            raw = true;
        }
    }

    private void refuseInClientMode(boolean given, String flag, String reason) throws UsageException {
        if (given) {
            throw new UsageException("--server-base and " + flag + " exclude each other: " + reason);
        }
    }

    /** Whether any completion-REPL flag was given ({@link LLMDemoOptions#completionInteractive()}). */
    private boolean interactiveCompletion() {
        return interactive || interactiveFirst || inPrefix != null || inSuffix != null || inPrefixBos;
    }

    private String value(String flag) throws UsageException {
        if (i + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[++i];
    }

    /**
     * {@code --image}/{@code --audio}/{@code --video} of one {@code kind}: a comma-separated list of files
     * (llama.cpp's {@code parse_csv_row}), each attached to the {@code -p} turn or staged into the chat; an
     * empty entry is refused naming the flag.
     */
    private void mediaFiles(String flag, String kind) throws UsageException {
        for (String entry : value(flag).split(",", -1)) {
            String file = entry.trim();
            if (file.isEmpty()) {
                throw new UsageException(flag + " has an empty file in its comma-separated list");
            }
            media.add(new LLMDemoOptions.MediaFile(kind, file));
        }
    }

    /** Whether any multimodal input flag besides {@code --mmproj} was given. */
    private boolean multimodalInputGiven() {
        return !media.isEmpty() || mmprojUrl != null || noMmproj || mmprojOffload != null
                || mmprojDevice != null || imageMinTokens > 0 || imageMaxTokens > 0;
    }

    /** An http or https URL value; anything else is rejected naming the flag. */
    private String url(String flag) throws UsageException {
        String v = value(flag);
        String lower = v.toLowerCase(Locale.ROOT);
        if (!lower.startsWith("http://") && !lower.startsWith("https://")) {
            throw new UsageException(flag + " must be an http or https URL, got '" + v + "'");
        }
        return v;
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

    /** A {@code true|false} value; anything else is rejected naming the flag. */
    private boolean boolValue(String flag) throws UsageException {
        String v = value(flag).trim().toLowerCase(Locale.ROOT);
        if (v.equals("true")) {
            return true;
        }
        if (v.equals("false")) {
            return false;
        }
        throw new UsageException(flag + " " + v + " is not a boolean; use true or false");
    }

    /**
     * A byte count {@code <n>[K|M|G]} (the {@code neuralfree.hf.materialize.memory.max} form); negative and
     * overflowing values are rejected naming the flag.
     */
    private long sizeValue(String flag) throws UsageException {
        String v = value(flag);
        String s = v.trim();
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
            throw new UsageException(flag + " " + v + " is not a size; use <n>[K|M|G]");
        }
        if (n < 0) {
            throw new UsageException(flag + " " + v + " must not be negative");
        }
        try {
            return Math.multiplyExact(n, unit);
        } catch (ArithmeticException e) {
            throw new UsageException(flag + " " + v + " overflows a 64-bit byte count");
        }
    }

    /** A path value; malformed paths are rejected naming the flag. */
    private Path pathValue(String flag) throws UsageException {
        try {
            return Paths.get(value(flag).trim());
        } catch (InvalidPathException e) {
            throw new UsageException(flag + " is not a valid path: " + e.getReason());
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
