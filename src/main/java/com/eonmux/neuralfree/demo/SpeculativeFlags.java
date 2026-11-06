package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.EngineLoadFlags.FlagException;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.DraftHeadOptions;
import com.eonmux.neuralfree.gguf.engine.DraftTypes;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import com.eonmux.neuralfree.gguf.engine.SpeculativeConfigJson;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SuffixDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * The speculative decoding flags of {@link LLMDemo} in llama.cpp's spellings (common/arg.cpp) and vLLM's
 * speculative-config fields: the draft file and MTP head, the type list ({@code --spec-type a,b}, repeated flags
 * append as llama.cpp's do), the draft lengths, the draft context's KV types and ubatch, where a dflash draft drafts, each n-gram type's sizes,
 * the lookup cache files, the suffix decoding limits and the verification mode, the dflash draft-tree verification
 * ({@code --[no-]spec-draft-tree} and its temperature and node width), the W12 weight streaming
 * ({@code --[no-]w12[.-head]}), the separate draft model's own
 * placement ({@code -devd}, {@code -ngld}, {@code -otd}, {@code -cmoed}, {@code -ncmoed}) and its thread flags
 * ({@code -td}, {@code -tbd}), and vLLM's
 * {@code --speculative-config} JSON, whose keys apply as those flags ({@link SpeculativeConfigJson}). They write the
 * {@link EngineConfig} and {@link MemoryConfig} builders they are given and the {@link SpeculativeOptions} of the
 * base configuration ({@link #apply}); every invalid value is refused naming its flag or JSON key. One instance
 * parses one command line; it is not thread-safe.
 */
final class SpeculativeFlags {

    private final EngineConfig.Builder engine;
    private final MemoryConfig.Builder memory;
    private final SpeculativeOptions base;
    private final LookaheadOptions.Builder lookahead;
    private final Map<NgramDrafter.Kind, NgramDrafter.Builder> ngram = new EnumMap<>(NgramDrafter.Kind.class);
    private SuffixDrafter.Builder suffix;
    private DraftHeadOptions.Builder draftHead;
    private SpeculativeOptions.Builder options;
    private List<EngineConfig.DraftType> types;
    private final List<String> draftDevices = new ArrayList<>();
    private boolean draftDevicesSet;
    private Integer threads;
    private Integer threadsBatch;
    private String threadsFlag;
    private String threadsBatchFlag;
    private final String ambientThreads;
    private String[] args = new String[0];
    private int i;

    /** Flags writing {@code engine} and {@code memory} over the speculative options of {@code base}. */
    SpeculativeFlags(EngineConfig.Builder engine, MemoryConfig.Builder memory, EngineConfig base) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.memory = Objects.requireNonNull(memory, "memory");
        this.base = Objects.requireNonNull(base, "base").speculativeOptions();
        this.lookahead = base.lookaheadOptions().toBuilder();
        this.options = this.base.toBuilder();
        this.ambientThreads = System.getProperty(CpuParallel.THREADS_PROPERTY);
    }

    /**
     * Parses the speculative flag at {@code args[index]} with its value.
     *
     * @return the index of the flag's last argument, or -1 when {@code args[index]} is not a speculative flag
     * @throws FlagException for a missing or invalid value, naming the flag
     */
    int parse(String[] args, int index) throws FlagException {
        this.args = args;
        this.i = index;
        String a = args[index];
        if (a.equals(SpeculativeConfigJson.FLAG)) {
            speculativeConfig(value(a));
            return i;
        }
        return draft(a) || lookup(a) || verify(a) || lookahead(a) ? i : -1;
    }

    /**
     * The lookahead decoding flags: {@code --lookahead} turns it on with the llama.cpp example's sizes,
     * {@code --no-lookahead} turns it off, and a size flag sets its value and turns it on. Lookahead decoding needs
     * the memory to hold {@code W + G + 1} sequences.
     */
    private boolean lookahead(String a) throws FlagException {
        switch (a) {
            case "--lookahead": lookahead.enabled(true); return true;
            case "--no-lookahead": lookahead.enabled(false); return true;
            case "--lookahead-window":
                lookahead.enabled(true).window(intValue(a, 1, LookaheadOptions.MAX_WINDOW));
                return true;
            case "--lookahead-ngram-size":
                lookahead.enabled(true).ngramSize(intValue(a, 3, LookaheadOptions.MAX_NGRAM_SIZE));
                return true;
            case "--lookahead-max-ngrams":
                lookahead.enabled(true).maxNgrams(intValue(a, 1, LookaheadOptions.MAX_NGRAMS));
                return true;
            default: return false;
        }
    }

    /**
     * vLLM's {@code --speculative-config} JSON: each key applies as the flag {@link SpeculativeConfigJson} maps it
     * to, so it combines with the flags as they do (a later flag overrides it, types append).
     */
    private void speculativeConfig(String json) throws FlagException {
        List<SpeculativeConfigJson.Setting> settings;
        try {
            settings = SpeculativeConfigJson.parse(json);
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
        String[] outerArgs = args;
        int outerIndex = i;
        try {
            for (SpeculativeConfigJson.Setting s : settings) {
                try {
                    if (parse(new String[]{s.flag(), s.value()}, 0) < 0) {
                        throw new IllegalStateException(s.flag() + " is no speculative flag");
                    }
                } catch (FlagException e) {
                    throw new FlagException(s.refusal(e.getMessage()).getMessage());
                }
            }
        } finally {
            args = outerArgs;
            i = outerIndex;
        }
    }

    /**
     * Writes the type list and the speculative options into the engine builder.
     *
     * @throws FlagException when the combined options are invalid, naming the setting
     */
    void apply() throws FlagException {
        try {
            if (types != null) {
                engine.draftTypes(types);
            }
            for (Map.Entry<NgramDrafter.Kind, NgramDrafter.Builder> e : ngram.entrySet()) {
                try {
                    options.ngram(e.getValue().build());
                } catch (IllegalArgumentException refused) {
                    throw new FlagException(flagPrefix(e.getKey()) + "-*: " + refused.getMessage());
                }
            }
            if (suffix != null) {
                options.suffix(suffix.build());
            }
            if (draftHead != null) {
                options.draftHead(draftHead.build());
            }
            if (draftDevicesSet) {
                engine.draftDevices(draftDevices);
            }
            engine.speculativeOptions(options.build());
            engine.lookaheadOptions(lookahead.build());
            applyThreads();
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    /**
     * Sizes the CPU thread pool the draft runs on ({@code -td}, {@code -tbd}): the engine runs the target's and the
     * draft model's host tensors on the one pool {@code neuralfree.cpu.threads} sizes, so a draft thread flag sizes
     * it as {@code -t} does, and draft thread flags that disagree — with each other, or with a thread count an
     * earlier flag set — are refused. llama.cpp keeps two pools for the two models; the engine runs one, and the
     * usage text says so.
     */
    private void applyThreads() throws FlagException {
        if (threads != null && threadsBatch != null && !threads.equals(threadsBatch)) {
            throw new FlagException(threadsFlag + " " + threads + " and " + threadsBatchFlag + " " + threadsBatch
                    + " disagree: the engine runs the target's and the draft model's host tensors on one CPU thread"
                    + " pool, so the draft generation and batch threads are the same; give both flags the same value");
        }
        Integer n = threads != null ? threads : threadsBatch;
        if (n == null) {
            return;
        }
        String current = System.getProperty(CpuParallel.THREADS_PROPERTY);
        String flag = threads != null ? threadsFlag : threadsBatchFlag;
        if (current != null && !current.equals(ambientThreads) && !current.equals(Integer.toString(n))) {
            throw new FlagException(flag + " " + n + " conflicts with " + CpuParallel.THREADS_PROPERTY + "="
                    + current + ", which an earlier flag set: the draft thread flags size the pool --threads (-t)"
                    + " sizes; give them the same value");
        }
        System.setProperty(CpuParallel.THREADS_PROPERTY, Integer.toString(n));
    }

    /** The flag prefix of an n-gram kind's sizes ({@code --spec-ngram-map-k}). */
    private static String flagPrefix(NgramDrafter.Kind kind) {
        switch (kind) {
            case SIMPLE: return "--spec-ngram-simple";
            case MAP_K: return "--spec-ngram-map-k";
            case MAP_K4V: return "--spec-ngram-map-k4v";
            case MOD: return "--spec-ngram-mod";
            default: return "--spec-ngram-" + kind.name().toLowerCase(Locale.ROOT);
        }
    }

    /** The derived candidate head a {@code --draft-head-*} or {@code --draft-prec} flag builds. */
    private DraftHeadOptions.Builder draftHead() {
        if (draftHead == null) {
            draftHead = DraftHeadOptions.builder();
        }
        return draftHead;
    }

    private boolean draft(String a) throws FlagException {
        switch (a) {
            case "--draft": case "-md": case "--model-draft": case "--spec-draft-model":
                engine.draftModelPath(value(a));
                return true;
            case "--draft-max": case "--draft-n-max": case "--spec-draft-n-max":
                engine.draftMax(intValue(a, 1, Integer.MAX_VALUE));
                return true;
            case "--draft-min": case "--draft-n-min": case "--spec-draft-n-min":
                engine.draftMin(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            case "--draft-p-min": case "--spec-draft-p-min": engine.draftPMin(floatValue(a)); return true;
            case "--spec-draft-p-split": case "--draft-p-split": {
                double v = doubleValue(a);
                try {
                    options.pSplit(v);
                } catch (IllegalArgumentException e) {
                    throw new FlagException(a + ": " + e.getMessage());
                }
                return true;
            }
            case "--mtp": engine.mtp(true); return true;
            case "--spec-type": specType(a); return true;
            case "--kv-type-draft": {
                KvDType t = kvType(a);
                memory.draftTypeK(t).draftTypeV(t);
                return true;
            }
            case "--kv-type-k-draft": case "--cache-type-k-draft": case "-ctkd": case "--spec-draft-type-k":
                memory.draftTypeK(kvType(a));
                return true;
            case "--kv-type-v-draft": case "--cache-type-v-draft": case "-ctvd": case "--spec-draft-type-v":
                memory.draftTypeV(kvType(a));
                return true;
            case "--spec-draft-ubatch-size": case "-ubd": case "--ubatch-size-draft": case "--spec-draft-n-ubatch":
                memory.draftUbatch(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            case "--spec-draft-device": case "-devd": case "--device-draft":
                draftDevices(a);
                return true;
            case "--spec-draft-ngl": case "-ngld": case "--gpu-layers-draft": case "--n-gpu-layers-draft":
                engine.draftGpuLayers(gpuLayers(a));
                return true;
            case "--spec-draft-override-tensor": case "-otd": case "--override-tensor-draft":
                addDraftTensorOverrides(a);
                return true;
            case "--spec-draft-cpu-moe": case "-cmoed": case "--cpu-moe-draft":
                engine.draftCpuMoe(true);
                return true;
            case "--spec-draft-n-cpu-moe": case "--spec-draft-ncmoe": case "-ncmoed": case "--n-cpu-moe-draft":
                engine.draftNCpuMoe(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            case "--spec-draft-threads": case "-td": case "--threads-draft":
                threads = intValue(a, 1, Integer.MAX_VALUE);
                threadsFlag = a;
                return true;
            case "--spec-draft-threads-batch": case "-tbd": case "--threads-batch-draft":
                threadsBatch = intValue(a, 1, Integer.MAX_VALUE);
                threadsBatchFlag = a;
                return true;
            case "--spec-draft-backend-sampling": options.dflashOnDevice(true); return true;
            case "--no-spec-draft-backend-sampling": options.dflashOnDevice(false); return true;
            // No llama.cpp or vLLM spelling exists (SpecPilot drives the format through environment
            // variables), so the flags carry the property names.
            case "--w12": options.w12(true); return true;
            case "--no-w12": options.w12(false); return true;
            case "--w12-head": options.w12Head(true); return true;
            case "--no-w12-head": options.w12Head(false); return true;
            case "--w12-f16":
                try {
                    options.w12F16(W12F16Mode.parse("--w12-f16", value(a)));
                } catch (IllegalArgumentException e) {
                    throw new FlagException(e.getMessage());
                }
                return true;
            case "--spec-draft-heterogeneous-vocab": case "--heterogeneous-vocab":
                options.heterogeneousVocab(true);
                return true;
            case "--no-spec-draft-heterogeneous-vocab": options.heterogeneousVocab(false); return true;
            case "--draft-head-vocab":
                draftHead().vocabSize(intValue(a, 1, Integer.MAX_VALUE));
                return true;
            case "--draft-head-freq":
                draftHead().frequencyFile(value(a));
                return true;
            case "--draft-head-keep":
                draftHead().keepIds(SpeculativeOptions.parseIds(a, value(a)));
                return true;
            case "--draft-head-prec":
                draftHead().headPrec(DraftHeadOptions.Prec.of(a, value(a)));
                return true;
            case "--draft-prec":
                draftHead().matricesPrec(DraftHeadOptions.Prec.of(a, value(a)));
                return true;
            case "--spec-draft-tree": options.treeVerify(true); return true;
            case "--no-spec-draft-tree": options.treeVerify(false); return true;
            case "--spec-draft-tree-temp": case "--spec-tree-temp": {
                double v = doubleValue(a);
                try {
                    options.treeTemp(v);
                } catch (IllegalArgumentException e) {
                    throw new FlagException(a + ": " + e.getMessage());
                }
                return true;
            }
            case "--spec-draft-tree-nodes": case "--spec-tree-nodes":
                options.treeNodes(intValue(a, 1, SpeculativeOptions.MAX_TREE_NODES));
                return true;
            case "-bs": case "--backend-sampling": backendSampling(true); return true;
            case "--no-backend-sampling": backendSampling(false); return true;
            default: return false;
        }
    }

    /**
     * llama.cpp {@code -bs}: the draft and verification selections run on the backend — a dflash draft drafts on
     * the target's device and verification selects on it. Generation sampling keeps following the
     * {@code neuralfree.sampling.device} launch property; the later granular flags override each part.
     */
    private void backendSampling(boolean on) {
        options.dflashOnDevice(on);
        options.verifyOnDevice(on);
    }

    /**
     * llama.cpp {@code -devd}: a comma-separated device list of the draft model's own load, indices in enumeration
     * order or name substrings, or {@code none} alone to keep the draft off every GPU (llama.cpp: no offload).
     */
    private void draftDevices(String flag) throws FlagException {
        String v = value(flag);
        List<String> out = new ArrayList<>();
        for (String entry : v.split(",", -1)) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new FlagException(flag + " " + v + " has an empty device entry");
            }
            out.add(s);
        }
        boolean none = out.size() == 1 && out.get(0).equalsIgnoreCase(EngineConfig.DRAFT_NO_DEVICE);
        for (String s : out) {
            if (!none && s.equalsIgnoreCase(EngineConfig.DRAFT_NO_DEVICE)) {
                throw new FlagException(flag + " " + String.join(",", out) + ": " + EngineConfig.DRAFT_NO_DEVICE
                        + " keeps the draft model off every GPU and cannot be listed with devices");
            }
        }
        draftDevices.clear();
        draftDevices.addAll(out);
        draftDevicesSet = true;
    }

    /** Comma-separated {@code -otd} entries of {@code pattern=target}; errors carry the flag. */
    private void addDraftTensorOverrides(String flag) throws FlagException {
        String v = value(flag);
        for (String entry : v.split(",")) {
            String e = entry.trim();
            if (e.isEmpty()) {
                continue;
            }
            int eq = e.indexOf('=');
            if (eq <= 0 || eq == e.length() - 1) {
                throw new FlagException(flag + " entry '" + e + "' is <tensor name pattern>=<target>");
            }
            try {
                engine.addDraftTensorOverride(e.substring(0, eq), e.substring(eq + 1));
            } catch (IllegalArgumentException ex) {
                throw new FlagException(flag + " entry '" + e + "': " + ex.getMessage());
            }
        }
    }

    /** The {@code -ngld} value: {@code auto} or llama.cpp's {@code -1} auto marker, {@code all}, or a count. */
    private int gpuLayers(String flag) throws FlagException {
        String v = value(flag).trim().toLowerCase(Locale.ROOT);
        if (v.equals("auto") || v.equals("-1")) {
            return EngineConfig.GPU_LAYERS_AUTO;
        }
        if (v.equals("all")) {
            return EngineConfig.GPU_LAYERS_ALL;
        }
        try {
            int n = Integer.parseInt(v);
            if (n >= 0) {
                return n;
            }
        } catch (NumberFormatException e) {
            // reported below
        }
        throw new FlagException(flag + " is auto, all or a layer count >= 0, got '" + v + "'");
    }

    /** {@code --spec-type a,b}: the first flag replaces the base list, later ones append (llama.cpp). */
    private void specType(String a) throws FlagException {
        List<EngineConfig.DraftType> next = types == null ? new ArrayList<>() : new ArrayList<>(types);
        try {
            next.addAll(DraftTypes.parse(value(a)));
            types = DraftTypes.ordered(next);
        } catch (IllegalArgumentException e) {
            throw new FlagException(a + ": " + e.getMessage());
        }
    }

    private boolean lookup(String a) throws FlagException {
        try {
            return lookupFlag(a);
        } catch (IllegalArgumentException e) {
            throw new FlagException(a + ": " + e.getMessage());
        }
    }

    private boolean lookupFlag(String a) throws FlagException {
        switch (a) {
            case "--spec-ngram-simple-size-n": case "--ngram-prompt-lookup-max":
                ngram(NgramDrafter.Kind.SIMPLE).sizeN(size(a));
                return true;
            case "--spec-ngram-simple-size-n-min": case "--ngram-prompt-lookup-min":
                ngram(NgramDrafter.Kind.SIMPLE).sizeNMin(size(a));
                return true;
            case "--spec-ngram-simple-size-m": ngram(NgramDrafter.Kind.SIMPLE).sizeM(size(a)); return true;
            case "--spec-ngram-simple-min-hits":
                throw new FlagException(a + ": ngram-simple has no hit threshold (llama.cpp reads min hits for the"
                        + " map types only: --spec-ngram-map-k-min-hits, --spec-ngram-map-k4v-min-hits)");
            case "--spec-ngram-map-k-size-n": ngram(NgramDrafter.Kind.MAP_K).sizeN(size(a)); return true;
            case "--spec-ngram-map-k-size-m": ngram(NgramDrafter.Kind.MAP_K).sizeM(size(a)); return true;
            case "--spec-ngram-map-k-min-hits": ngram(NgramDrafter.Kind.MAP_K).minHits(size(a)); return true;
            case "--spec-ngram-map-k4v-size-n": ngram(NgramDrafter.Kind.MAP_K4V).sizeN(size(a)); return true;
            case "--spec-ngram-map-k4v-size-m": ngram(NgramDrafter.Kind.MAP_K4V).sizeM(size(a)); return true;
            case "--spec-ngram-map-k4v-min-hits": ngram(NgramDrafter.Kind.MAP_K4V).minHits(size(a)); return true;
            case "--spec-ngram-mod-n-match": ngram(NgramDrafter.Kind.MOD).sizeN(size(a)); return true;
            case "--spec-ngram-mod-n-max": ngram(NgramDrafter.Kind.MOD).sizeM(size(a)); return true;
            case "--spec-ngram-mod-n-min":
                ngram(NgramDrafter.Kind.MOD).draftMin(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            case "-lcs": case "--lookup-cache-static": options.lookupCacheStatic(value(a)); return true;
            case "-lcd": case "--lookup-cache-dynamic": options.lookupCacheDynamic(value(a)); return true;
            case "--spec-suffix-max-tree-depth": suffix().maxTreeDepth(intValue(a, 1, Integer.MAX_VALUE)); return true;
            case "--spec-suffix-max-spec-factor": suffix().maxSpecFactor(doubleValue(a)); return true;
            case "--spec-suffix-max-spec-offset": suffix().maxSpecOffset(doubleValue(a)); return true;
            case "--spec-suffix-min-token-prob": suffix().minTokenProb(doubleValue(a)); return true;
            case "--spec-suffix-max-cached-requests":
                suffix().maxCachedRequests(intValue(a, 0, Integer.MAX_VALUE));
                return true;
            default: return false;
        }
    }

    private boolean verify(String a) throws FlagException {
        try {
            switch (a) {
                case "--spec-verify": case "--rejection-sample-method":
                    options.verifyMode(VerifyMode.parse(value(a)));
                    return true;
                case "--spec-draft-sampling": options.verifyMode(draftSampling(a)); return true;
                case "--spec-fly-window": options.flyWindow(intValue(a, 0, Integer.MAX_VALUE)); return true;
                case "--spec-fly-entropy-threshold": options.flyEntropyThreshold(doubleValue(a)); return true;
                case "--spec-fly-entropy-top-k": options.flyEntropyTopK(intValue(a, 1, Integer.MAX_VALUE)); return true;
                case "--spec-synth-len": options.synthLength(doubleValue(a)); return true;
                case "--spec-synth-rates": options.synthRates(SpeculativeOptions.parseRates(a, value(a))); return true;
                default: return false;
            }
        } catch (IllegalArgumentException e) {
            throw new FlagException(a + ": " + e.getMessage());
        }
    }

    /** llama.cpp {@code --spec-draft-sampling}: greedy verifies by match, probabilistic by rejection sampling. */
    private VerifyMode draftSampling(String a) throws FlagException {
        String v = value(a).trim().toLowerCase(Locale.ROOT);
        if (v.equals("greedy")) {
            return VerifyMode.MATCH;
        }
        if (v.equals("probabilistic")) {
            return VerifyMode.REJECTION;
        }
        throw new FlagException(a + " '" + v + "' is invalid; use greedy or probabilistic");
    }

    /** The n-gram sizes of {@code kind}, seeded from the base options. */
    private NgramDrafter.Builder ngram(NgramDrafter.Kind kind) {
        return ngram.computeIfAbsent(kind, k -> base.ngramOptions(k).toBuilder());
    }

    private SuffixDrafter.Builder suffix() {
        if (suffix == null) {
            suffix = base.suffixOptions().toBuilder();
        }
        return suffix;
    }

    // ------------------------------------------------------------------ values

    private String value(String flag) throws FlagException {
        if (i + 1 >= args.length) {
            throw new FlagException(flag + " needs a value");
        }
        return args[++i];
    }

    /** An n-gram size: a positive integer; the drafter's own range is checked when the options build. */
    private int size(String flag) throws FlagException {
        return intValue(flag, 1, Integer.MAX_VALUE);
    }

    private int intValue(String flag, int min, int max) throws FlagException {
        String v = value(flag);
        try {
            int n = Integer.parseInt(v.trim());
            if (n < min || n > max) {
                throw new FlagException(flag + " " + v + " is outside [" + min + ", " + max + "]");
            }
            return n;
        } catch (NumberFormatException e) {
            throw new FlagException(flag + " needs an integer, got '" + v + "'");
        }
    }

    private float floatValue(String flag) throws FlagException {
        double d = doubleValue(flag);
        if (!(d >= 0.0 && d <= 1.0)) {
            throw new FlagException(flag + " " + d + " is outside [0, 1]");
        }
        return (float) d;
    }

    private double doubleValue(String flag) throws FlagException {
        String v = value(flag);
        try {
            return Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            throw new FlagException(flag + " needs a number, got '" + v + "'");
        }
    }

    private KvDType kvType(String flag) throws FlagException {
        try {
            return KvDType.parse(value(flag), flag);
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    // ------------------------------------------------------------------ usage

    /** The usage lines of the speculative decoding flags. */
    static List<String> usage() {
        return Arrays.asList(
                "  --draft, -md <gguf>          Draft model: dflash, eagle, eagle3, gemma4-assistant, a GGUF of the",
                "                               target's model with NextN/MTP blocks (MTP-only or full; only its MTP",
                "                               blocks draft) or a small LM",
                "  --mtp                        Draft with the target's own NextN/MTP head",
                "  --spec-type <t[,t..]>        Drafter types (repeatable, appends; default auto). The draft file:",
                "                               auto | draft-mtp (its NextN/MTP blocks; without --draft, the",
                "                               target's own) | draft-simple (a separate draft model). Without a",
                "                               draft model: ngram-simple | ngram-map-k | ngram-map-k4v | ngram-mod",
                "                               | ngram-cache (n-gram lookup over the history) | suffix (suffix",
                "                               tree). Several chain: the lookups first, the draft file last",
                "  --draft-max <n>              Max drafted tokens per round (default 3)",
                "  --draft-min <n>              Min draft length that is verified (default 0)",
                "  --draft-p-min <p>            Stop drafting below this probability (default 0)",
                "  --spec-draft-p-split <p>     Split a draft branch when a secondary candidate exceeds this",
                "                               probability, over -np > 1 sequences (default 0.1; alias",
                "                               --draft-p-split)",
                "  --kv-type-draft <t>          KV cache type of the draft contexts for K and V (default f16):",
                "                               f32 | f16 | bf16 | q8_0 | q4_0 | q4_1 | iq4_nl | q5_0 | q5_1",
                "  --kv-type-k-draft / --kv-type-v-draft <t>  Draft-cache type of K / V only (MLA drafts need",
                "                               K = V); aliases -ctkd / -ctvd, --spec-draft-type-k/-v,",
                "                               --cache-type-k-draft / --cache-type-v-draft",
                "  --spec-draft-ubatch-size, -ubd <n>   Tokens per forward step of a draft context (0 = the",
                "                               target's ubatch); aliases --ubatch-size-draft,",
                "                               --spec-draft-n-ubatch",
                "  --spec-draft-device, -devd <d1,d2,..>  GPUs of the draft model's own load: indices in",
                "                               enumeration order or name substrings (default: the target's",
                "                               devices); none = the draft on the host; two or more split it by",
                "                               layer; alias --device-draft",
                "  --spec-draft-ngl, -ngld <n|auto|all>  Layers of the draft model's load kept on its device",
                "                               (auto; a count covers the output head, like -ngl); aliases",
                "                               --gpu-layers-draft, --n-gpu-layers-draft",
                "  --spec-draft-override-tensor, -otd <p=t,..>  Tensor placement overrides of the draft model's",
                "                               load, like -ot; alias --override-tensor-draft",
                "  --spec-draft-cpu-moe, -cmoed        MoE expert weights of the draft model on the host",
                "  --spec-draft-n-cpu-moe, -ncmoed <n> The first n MoE layers of the draft model keep their",
                "                               expert weights on the host; aliases --spec-draft-ncmoe,",
                "                               --n-cpu-moe-draft",
                "  --spec-draft-threads, -td <n>  CPU thread pool size the draft model's host tensors run on",
                "                               (default: --threads): the engine runs the target's and the",
                "                               draft's host tensors on one pool, so -tbd (default: -td)",
                "                               sizes the same pool and must agree; aliases",
                "                               --threads-draft, --spec-draft-threads-batch,",
                "                               --threads-batch-draft",
                "  --[no-]spec-draft-backend-sampling  Draft a dflash draft on the GPU target's device (default)",
                "                               or on the host",
                "  --spec-draft-tree / --no-spec-draft-tree  A dflash drafter proposes a draft tree the target",
                "                               verifies in one forward: N tree nodes run as N rows of one target",
                "                               sequence under a tree attention mask (SpecPilot TREE_MODE=ddtree);",
                "                               the accepted root path commits",
                "  --spec-draft-tree-temp, --spec-tree-temp <t>  Temperature the tree builder scales its path",
                "                               scores at (default 1.0)",
                "  --spec-draft-tree-nodes, --spec-tree-nodes <n>  Fixed verify width in tree nodes: every tree",
                "                               verify runs one forward of n + 1 rows (the pending token plus the",
                "                               nodes, shorter trees padded; default 16, at most 63)",
                "  --[no-]w12                   Stream the target's BF16/F16 matmul weights as lossless W12 rows",
                "                               (1.5 bytes per weight), decoded in the GPU projection kernels",
                "                               (property neuralfree.weights.w12; bit-exact at load, BF16 copies",
                "                               freed). A prefill wider than 64 tokens decodes into a BF16 mirror",
                "                               and runs the BF16 GEMM. No llama.cpp/vLLM spelling exists",
                "  --[no-]w12-head              Include the output head in the W12 streaming (default off; needs",
                "                               --w12; property neuralfree.weights.w12.head)",
                "  --w12-f16 <refuse|round>     Treatment of an F16 tensor's values a W12 record cannot store",
                "                               exactly (default refuse): refuse refuses the GPU load naming",
                "                               the tensor and the off-grid weight (lossless), round rounds",
                "                               every weight onto the bfloat16 grid at load (lossy: each",
                "                               off-grid weight loses its 3 low mantissa bits; the load",
                "                               reports how many weights moved); property",
                "                               neuralfree.weights.w12.f16",
                "  --spec-draft-heterogeneous-vocab  Draft across a draft model of a different vocabulary by the",
                "                               token-level intersection of the two tokenizers (vLLM",
                "                               use_heterogeneous_vocab; greedy drafting only); its tokens map into",
                "                               the target's, tokens outside the intersection become unknown tokens",
                "  --draft-head-vocab <n>       Derived candidate head of a dflash drafter: keep n rows of the head",
                "                               the drafter would use (its own output head, else the target's, else",
                "                               the tied embeddings), mapped back by a synthesized d2t; the target",
                "                               verifies with its own head, so tokens are unchanged. Properties only",
                "                               in llama.cpp and vLLM (-Dneuralfree.draft.head.vocab)",
                "  --draft-head-freq <file>     Token-frequency ranking of the kept rows, most frequent first (JSON",
                "                               {\"ranked_ids\": [..]} or lines of \"<id> <count>\"); without it,",
                "                               ascending token id ranks (the BPE merge-rank prior). The draft's",
                "                               special ids and --draft-head-keep stay kept",
                "  --draft-head-keep <id,id,..> Token ids the derived head always keeps beside the draft's special ids",
                "  --draft-head-prec <t>        Storage of the derived head's rows: fp8 (E4M3, per-row scales),",
                "                               nvfp4 (E2M1, per-16 scales, per-tensor global scale in",
                "                               output.scale) or q8_0 (vLLM draft_lm_head_quantization); without",
                "                               it the kept rows are copied in the head's own precision, which is",
                "                               exact",
                "  --draft-prec <t>             Storage of the drafter's own projection matrices (encoder, per-block",
                "                               attention and FFN, DFlash2 convolutions, selector hidden): fp8,",
                "                               nvfp4 or q8_0; already quantized tensors stay as they are",
                "  --backend-sampling, -bs      Every selection on the backend: the draft, the verification and",
                "                               the generation sampling over the device's top candidates;",
                "                               --no-backend-sampling keeps them on the host. The granular flags",
                "                               (--spec-draft-backend-sampling and the neuralfree.* properties)",
                "                               override each part",
                "  --spec-ngram-simple-size-n/-size-m <n>  ngram-simple lookup n-gram / draft m-gram (12 / 48)",
                "  --spec-ngram-simple-size-n-min <n>   smallest lookup n-gram: below -size-n, vLLM's search of",
                "                               the longest match in the range (prompt_lookup_min; default",
                "                               -size-n); --ngram-prompt-lookup-min / -max are the same flags",
                "  --spec-ngram-map-k-size-n/-size-m/-min-hits <n>    ngram-map-k key n-gram, value m-gram, key",
                "                               hits before a draft (12 / 48 / 1); the same for -map-k4v-",
                "  --spec-ngram-mod-n-match/-n-max/-n-min <n>  ngram-mod hashed n-gram, longest and shortest",
                "                               draft (24 / 64 / 48; a chain of the full -n-max is kept)",
                "  --lookup-cache-static, -lcs <file>   ngram-cache static n-gram cache (read only)",
                "  --lookup-cache-dynamic, -lcd <file>  ngram-cache dynamic n-gram cache (read when present,",
                "                               written with this run's n-grams added on exit)",
                "  --spec-suffix-max-tree-depth <n>, --spec-suffix-max-spec-factor <f>,",
                "  --spec-suffix-max-spec-offset <f>, --spec-suffix-min-token-prob <p>",
                "                               suffix decoding limits (64 / 1.0 / 0.0 / 0.1; vLLM suffix_decoding_*)",
                "  --spec-suffix-max-cached-requests <n>  past requests the suffix drafter's request tree keeps,",
                "                               oldest evicted first (10000; 0 = off)",
                "  --speculative-config <json>  vLLM's speculative config: method (ngram | suffix | draft_model |",
                "                               mtp | eagle | eagle3 | dflash), model, num_speculative_tokens,",
                "                               prompt_lookup_min/max, suffix_decoding_*, rejection_sample_method,",
                "                               synthetic_acceptance_*, fly_*, draft_lm_head_quantization; each key",
                "                               applies as its flag",
                "  --spec-verify <m>            match (default) | rejection | block | fly (lossy) | synthetic;",
                "                               alias --rejection-sample-method; --spec-draft-sampling",
                "                               greedy|probabilistic = match|rejection",
                "  --spec-fly-window <n>, --spec-fly-entropy-threshold <f>, --spec-fly-entropy-top-k <n>",
                "                               fly verification's window (0 = min(6, k-1)), entropy gate, top-k",
                "  --spec-synth-len <l> | --spec-synth-rates <p0,p1,..>  synthetic acceptance for benchmarking",
                "                               (needs -Dneuralfree.spec.synthetic=true)",
                "  --lookahead                  Lookahead (Jacobi) decoding instead of speculative decoding: the",
                "                               target verifies observed n-grams against parallel Jacobi guesses.",
                "                               Needs -np <W+G+1>; refuses --draft/--mtp/--spec-type",
                "  --lookahead-window <n>       Jacobi window W, the guesses kept per step (default 15; turns it on)",
                "  --lookahead-ngram-size <n>   Observed n-gram size N (default 5, at least 3; turns it on)",
                "  --lookahead-max-ngrams <n>   Verification n-grams G kept per first token (default 15; turns it on)",
                "  --no-lookahead               Turns lookahead decoding off");
    }
}
