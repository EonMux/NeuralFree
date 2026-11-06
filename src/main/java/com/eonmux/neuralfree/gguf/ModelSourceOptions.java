package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig;
import com.eonmux.neuralfree.gguf.load.MetadataOverrides;
import com.eonmux.neuralfree.gguf.load.RopeOverrides;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
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
 *   <li>{@value #PROP_OUTTYPE}: source | auto | f32 | f16 | bf16 | q8_0 | tq1_0 | tq2_0, the type of float
 *       tensors (the converter's {@code --outtype}, with source keeping each tensor's stored type); default
 *       source.</li>
 *   <li>{@value #PROP_ONLINE_QUANT}: none | q8_0 | q4_0 | q4_1 | q5_0 | q5_1 | iq4_nl | fp8_per_tensor |
 *       fp8_per_block | fp8_per_channel | int8_per_channel_weight_only | experts_int8 | nvfp4_per_token | mxfp8 |
 *       mxfp4 | online (with the aliases fp8, ptpc_fp8 and nvfp4_online), online quantization of float matmul
 *       weights; default none.</li>
 *   <li>{@value #PROP_QUANT_CONFIG}: vLLM's {@code --quantization-config}, a JSON object of {@code linear},
 *       {@code moe}, {@code ignore} and {@code targets} ({@link OnlineQuantConfig}) composed over the
 *       {@value #PROP_ONLINE_QUANT} shorthand, or alone the bare {@code online} method; default none.</li>
 *   <li>{@value #PROP_ONLINE_OUTPUT}: whether online quantization also quantizes {@code output.weight}
 *       (llama-quantize's {@code quantize_output_tensor}); default true, and false is
 *       {@code --leave-output-tensor}.</li>
 *   <li>{@value #PROP_EXACT_QUANT}: prefer bit-exact weight types over close repacks; default false.</li>
 *   <li>{@value #PROP_MTP}: keep MTP / NextN layers (the converter's {@code --no-mtp} sets false); default true.</li>
 *   <li>{@value #PROP_DRAFT}: the converter's draft selection, none | mtp | no-nextn | dspark
 *       ({@link DraftMode}); default none. {@code mtp} exports only the NextN / MTP head and {@code no-nextn} the
 *       target without it; {@code dspark} converts the checkpoint's DSpark draft as a DFLASH model.</li>
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
 *   <li>{@value #PROP_REMOTE_CONCURRENCY}: the objects of an object-store checkpoint ({@code s3://}, {@code gs://},
 *       {@code az://}) downloaded at once when it is mirrored into the cache; default 8.</li>
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
    public static final String PROP_ONLINE_OUTPUT = "neuralfree.quant.online.output";
    public static final String PROP_QUANT_CONFIG = "neuralfree.quant.config";
    public static final String PROP_EXACT_QUANT = "neuralfree.hf.quant.exact";
    public static final String PROP_MTP = "neuralfree.hf.mtp";
    public static final String PROP_DRAFT = "neuralfree.hf.draft";
    public static final String PROP_ST_DENSE = "neuralfree.hf.st.dense";
    public static final String PROP_OVERRIDES = "neuralfree.hf.overrides";
    public static final String PROP_TOKENIZER = "neuralfree.hf.tokenizer";
    public static final String PROP_LORA_BASE_CONFIG = "neuralfree.lora.base.config";
    public static final String PROP_BASE_DIR = "neuralfree.hf.base.dir";
    public static final String PROP_OVERRIDE_KV = "neuralfree.override.kv";
    public static final String PROP_CHECK_TENSORS = "neuralfree.check.tensors";
    public static final String PROP_LOAD_FORMAT = "neuralfree.load.format";
    public static final String PROP_WARMUP = "neuralfree.warmup";
    public static final String PROP_REMOTE_CONCURRENCY = "neuralfree.remote.concurrency";

    /** Default {@link #remoteConcurrency()}: 8 objects at once. */
    public static final int DEFAULT_REMOTE_CONCURRENCY = 8;
    /** Highest accepted {@link #remoteConcurrency()}. */
    public static final int MAX_REMOTE_CONCURRENCY = 256;

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
        Q8_0,
        /** TQ1_0, the BitNet 1.58-bit ternary type, quantized at load. */
        TQ1_0,
        /** TQ2_0, the ternary type, quantized at load. */
        TQ2_0;

        /** @throws IllegalArgumentException for an unknown name, naming {@value ModelSourceOptions#PROP_OUTTYPE} */
        public static OutType parse(String value) {
            return parseEnum(OutType.class, PROP_OUTTYPE, value);
        }
    }

    /**
     * Online quantization of float matmul weights at load. The legacy block types carry llama-quantize's targets
     * and eligibility; the rest are vLLM's online quantization methods, which quantize the linear layers and routed
     * experts but never the embeddings or LM heads: fp8 per tensor, per 128x128 block or per output channel into
     * F8_E4M3_B128 (F8_E4M3_B32 for rows of no whole 128-value block), mxfp8 into MXFP8 and mxfp4 into MXFP4 on
     * both, int8 per output channel into I8_B128 and
     * nvfp4_per_token into NVFP4 on the routed experts only, and the bare online method, whose quantization config
     * ({@value ModelSourceOptions#PROP_QUANT_CONFIG}) chooses per layer.
     */
    public enum OnlineQuant {
        NONE,
        Q8_0,
        Q4_0,
        Q4_1,
        Q5_0,
        Q5_1,
        IQ4_NL,
        /** E4M3 weights with one scale per tensor (per expert of a stacked tensor). */
        FP8_PER_TENSOR,
        /** E4M3 weights with one scale per 128x128 block (per expert of a stacked tensor). */
        FP8_PER_BLOCK,
        /** E4M3 weights with one scale per output channel. */
        FP8_PER_CHANNEL,
        /** Int8 weights with one scale per output channel; routed experts only. */
        INT8_PER_CHANNEL_WEIGHT_ONLY,
        /** vLLM's {@code experts_int8} method: the routed-expert int8 of {@link #INT8_PER_CHANNEL_WEIGHT_ONLY}. */
        EXPERTS_INT8,
        /** NVFP4 weights with a per-expert global scale and an E4M3 scale per 16 elements; routed experts only. */
        NVFP4_PER_TOKEN,
        /** E4M3 weights with one power-of-two scale per 32 elements. */
        MXFP8,
        /** E2M1 weights with one power-of-two scale per 32 elements. */
        MXFP4,
        /** vLLM's bare {@code online} method: the quantization config chooses each layer's method. */
        ONLINE;

        /** The accepted aliases of the vLLM method names ({@code --quantization}). */
        private static final Map<String, OnlineQuant> ALIASES;

        static {
            Map<String, OnlineQuant> m = new LinkedHashMap<>();
            m.put("FP8", FP8_PER_TENSOR);
            m.put("PTPC_FP8", FP8_PER_CHANNEL);
            m.put("NVFP4_ONLINE", NVFP4_PER_TOKEN);
            ALIASES = Collections.unmodifiableMap(m);
        }

        /** Whether this is one of vLLM's online methods rather than a llama-quantize block type or NONE. */
        public boolean isVllm() {
            switch (this) {
                case NONE: case Q8_0: case Q4_0: case Q4_1: case Q5_0: case Q5_1: case IQ4_NL:
                    return false;
                default:
                    return true;
            }
        }

        /**
         * A constant by name, case-insensitive, with {@code -} and {@code _} alike, or one of {@link #ALIASES}.
         *
         * @throws IllegalArgumentException for an unknown name, naming {@value ModelSourceOptions#PROP_ONLINE_QUANT}
         */
        public static OnlineQuant parse(String value) {
            Objects.requireNonNull(value, PROP_ONLINE_QUANT);
            String s = value.trim().toUpperCase(Locale.ROOT).replace('-', '_');
            for (OnlineQuant q : OnlineQuant.values()) {
                if (q.name().equals(s)) {
                    return q;
                }
            }
            OnlineQuant alias = ALIASES.get(s);
            if (alias != null) {
                return alias;
            }
            StringBuilder valid = new StringBuilder("none");
            for (OnlineQuant q : values()) {
                if (q != NONE) {
                    valid.append(" | ").append(q.name().toLowerCase(Locale.ROOT));
                }
            }
            for (String a : ALIASES.keySet()) {
                valid.append(" | ").append(a.toLowerCase(Locale.ROOT));
            }
            throw new IllegalArgumentException(PROP_ONLINE_QUANT + "=" + value + " is invalid; use " + valid);
        }
    }

    /**
     * Which part of a checkpoint with multi-token prediction tensors a conversion exports (the converter's
     * {@code --mtp}, {@code --no-nextn} and {@code --dspark}). The three are mutually exclusive.
     */
    public enum DraftMode {
        /** The checkpoint's own view, MTP / NextN blocks included. */
        NONE,
        /** {@code --mtp}: only the NextN / MTP head, with the tensors it shares with the target. */
        MTP_ONLY,
        /** {@code --no-nextn} ({@code --no-mtp}): the target without its NextN / MTP tensors. */
        NO_NEXTN,
        /** {@code --dspark}: the checkpoint's DSpark draft, as the draft converter's DFLASH model. */
        DSPARK;

        /**
         * A constant by name, case-insensitive, with {@code -} and {@code _} alike; {@code no-mtp} is the alias of
         * {@code no-nextn}.
         *
         * @throws IllegalArgumentException for an unknown name, naming {@value ModelSourceOptions#PROP_DRAFT}
         */
        public static DraftMode parse(String value) {
            Objects.requireNonNull(value, PROP_DRAFT);
            String s = value.trim().toLowerCase(Locale.ROOT).replace('_', '-');
            if (s.equals("none")) {
                return NONE;
            }
            if (s.equals("mtp")) {
                return MTP_ONLY;
            }
            if (s.equals("no-nextn") || s.equals("no-mtp")) {
                return NO_NEXTN;
            }
            if (s.equals("dspark")) {
                return DSPARK;
            }
            throw new IllegalArgumentException(PROP_DRAFT + "=" + value + " is invalid; use none, mtp, no-nextn"
                    + " or dspark");
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
    private final boolean quantizeOutputTensor;
    private final OnlineQuantConfig quantConfig;
    private final boolean exactQuant;
    private final DraftMode draftMode;
    private final boolean stDenseModules;
    private final Map<String, Object> configOverrides;
    private final Path tokenizerDir;
    private final Path loraBaseConfig;
    private final TensorCatalog loraVisionCatalog;
    private final Path baseDir;
    private final List<MetadataOverrides.Override> metadataOverrides;
    private final RopeOverrides ropeOverrides;
    private final boolean checkTensors;
    private final LoadFormat loadFormat;
    private final boolean warmup;
    private final int remoteConcurrency;

    private ModelSourceOptions(Builder b) {
        this.materialize = b.materialize;
        this.memoryBudget = b.memoryBudget;
        this.cacheDir = b.cacheDir;
        this.outType = b.outType;
        this.onlineQuant = b.onlineQuant;
        this.quantizeOutputTensor = b.quantizeOutputTensor;
        this.quantConfig = b.quantConfig;
        this.exactQuant = b.exactQuant;
        this.draftMode = b.draftMode;
        this.stDenseModules = b.stDenseModules;
        this.configOverrides = b.configOverrides;
        this.tokenizerDir = b.tokenizerDir;
        this.loraBaseConfig = b.loraBaseConfig;
        this.loraVisionCatalog = b.loraVisionCatalog;
        this.baseDir = b.baseDir;
        this.metadataOverrides = b.metadataOverrides;
        this.ropeOverrides = b.ropeOverrides;
        this.checkTensors = b.checkTensors;
        this.loadFormat = b.loadFormat;
        this.warmup = b.warmup;
        this.remoteConcurrency = b.remoteConcurrency;
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
        v = value(p, PROP_ONLINE_OUTPUT);
        if (v != null) {
            b.quantizeOutputTensor(parseBoolean(PROP_ONLINE_OUTPUT, v));
        }
        v = value(p, PROP_QUANT_CONFIG);
        if (v != null) {
            b.quantConfigJson(v);
        }
        v = value(p, PROP_EXACT_QUANT);
        if (v != null) {
            b.exactQuant(parseBoolean(PROP_EXACT_QUANT, v));
        }
        DraftMode draft = null;
        v = value(p, PROP_DRAFT);
        if (v != null) {
            draft = DraftMode.parse(v);
        }
        v = value(p, PROP_MTP);
        if (v != null) {
            boolean include = parseBoolean(PROP_MTP, v);
            if (!include && (draft == DraftMode.MTP_ONLY || draft == DraftMode.DSPARK)) {
                throw new IllegalArgumentException(PROP_MTP + "=false (no NextN) conflicts with "
                        + PROP_DRAFT + "=" + (draft == DraftMode.MTP_ONLY ? "mtp" : "dspark"));
            }
            if (!include) {
                draft = DraftMode.NO_NEXTN;
            }
        }
        if (draft != null) {
            b.draftMode(draft);
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
        v = value(p, PROP_REMOTE_CONCURRENCY);
        if (v != null) {
            b.remoteConcurrency(parseConcurrency(v));
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
    /** Whether online quantization also quantizes {@code output.weight} (llama-quantize's choice; default true). */
    public boolean quantizeOutputTensor() { return quantizeOutputTensor; }
    /** vLLM's online quantization config ({@value #PROP_QUANT_CONFIG}), or null for none. */
    public OnlineQuantConfig quantConfig() { return quantConfig; }
    /** Whether bit-exact weight types are preferred over close repacks. */
    public boolean exactQuant() { return exactQuant; }
    /** The converter's draft selection ({@value #PROP_DRAFT}). */
    public DraftMode draftMode() { return draftMode; }

    /** Whether MTP / NextN layers are kept: false only for {@link DraftMode#NO_NEXTN}. */
    public boolean includeMtp() { return draftMode != DraftMode.NO_NEXTN; }

    /** True for {@link DraftMode#MTP_ONLY}: export only the NextN / MTP head. */
    public boolean mtpOnly() { return draftMode == DraftMode.MTP_ONLY; }

    /** True for {@link DraftMode#DSPARK}: export the checkpoint's DSpark draft as a DFLASH model. */
    public boolean dspark() { return draftMode == DraftMode.DSPARK; }
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

    /**
     * The tensor table of the mmproj file opened beside the model, or null: a LoRA adapter target absent from the
     * base model may then resolve against it and becomes a
     * {@link com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter.Side#VISION} pair of the vision encoder and its
     * connector.
     */
    public TensorCatalog loraVisionCatalog() { return loraVisionCatalog; }

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

    /** How many objects of an object-store checkpoint are downloaded at once ({@value #PROP_REMOTE_CONCURRENCY}). */
    public int remoteConcurrency() { return remoteConcurrency; }

    public Builder toBuilder() {
        Builder b = builder();
        b.materialize = materialize;
        b.memoryBudget = memoryBudget;
        b.cacheDir = cacheDir;
        b.outType = outType;
        b.onlineQuant = onlineQuant;
        b.quantizeOutputTensor = quantizeOutputTensor;
        b.quantConfig = quantConfig;
        b.exactQuant = exactQuant;
        b.draftMode = draftMode;
        b.stDenseModules = stDenseModules;
        b.configOverrides = configOverrides;
        b.tokenizerDir = tokenizerDir;
        b.loraBaseConfig = loraBaseConfig;
        b.loraVisionCatalog = loraVisionCatalog;
        b.baseDir = baseDir;
        b.metadataOverrides = metadataOverrides;
        b.ropeOverrides = ropeOverrides;
        b.checkTensors = checkTensors;
        b.loadFormat = loadFormat;
        b.warmup = warmup;
        b.remoteConcurrency = remoteConcurrency;
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
                && quantizeOutputTensor == other.quantizeOutputTensor
                && Objects.equals(quantConfig, other.quantConfig) && exactQuant == other.exactQuant
                && draftMode == other.draftMode && stDenseModules == other.stDenseModules
                && configOverrides.equals(other.configOverrides)
                && Objects.equals(tokenizerDir, other.tokenizerDir)
                && Objects.equals(loraBaseConfig, other.loraBaseConfig)
                && Objects.equals(loraVisionCatalog, other.loraVisionCatalog) && Objects.equals(baseDir, other.baseDir)
                && metadataOverrides.equals(other.metadataOverrides) && ropeOverrides.equals(other.ropeOverrides)
                && checkTensors == other.checkTensors && loadFormat == other.loadFormat
                && warmup == other.warmup && remoteConcurrency == other.remoteConcurrency;
    }

    @Override
    public int hashCode() {
        return Objects.hash(materialize, memoryBudget, cacheDir, outType, onlineQuant, quantizeOutputTensor,
                quantConfig, exactQuant, draftMode, stDenseModules, configOverrides, tokenizerDir, loraBaseConfig,
                loraVisionCatalog, baseDir, metadataOverrides, ropeOverrides, checkTensors, loadFormat, warmup,
                remoteConcurrency);
    }

    @Override
    public String toString() {
        return "ModelSourceOptions{materialize=" + materialize + ", memoryBudget=" + memoryBudget + ", cacheDir="
                + cacheDir + ", outType=" + outType + ", onlineQuant=" + onlineQuant
                + (quantizeOutputTensor ? "" : ", leave output tensor")
                + (quantConfig == null ? "" : ", quantConfig=" + quantConfig) + ", exactQuant=" + exactQuant
                + ", draft=" + draftMode.name().toLowerCase(Locale.ROOT) + ", stDenseModules=" + stDenseModules
                + (configOverrides.isEmpty() ? "" : ", overrides=" + Json.toString(configOverrides, Json.Style.CANONICAL))
                + (metadataOverrides.isEmpty() ? "" : ", kv=" + MetadataOverrides.toString(metadataOverrides))
                + (ropeOverrides.isEmpty() ? "" : ", rope=" + ropeOverrides)
                + (checkTensors ? ", checkTensors" : "")
                + (loadFormat == LoadFormat.FULL ? "" : ", loadFormat=" + loadFormat.name().toLowerCase(Locale.ROOT))
                + (warmup ? "" : ", no warmup")
                + (remoteConcurrency == DEFAULT_REMOTE_CONCURRENCY ? "" : ", remoteConcurrency=" + remoteConcurrency)
                + (tokenizerDir != null ? ", tokenizer=" + tokenizerDir : "")
                + (loraBaseConfig != null ? ", loraBaseConfig=" + loraBaseConfig : "")
                + (loraVisionCatalog != null ? ", loraVisionCatalog=" + loraVisionCatalog.archName() + "/"
                        + loraVisionCatalog.size() + " tensors" : "")
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

    /**
     * A download concurrency in {@code [1, MAX_REMOTE_CONCURRENCY]}.
     *
     * @throws IllegalArgumentException naming {@value #PROP_REMOTE_CONCURRENCY} when the value is not an integer or is
     *                                  out of range
     */
    private static int parseConcurrency(String value) {
        int parsed;
        try {
            parsed = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(PROP_REMOTE_CONCURRENCY + "=" + value + " is not an integer", e);
        }
        if (parsed < 1 || parsed > MAX_REMOTE_CONCURRENCY) {
            throw new IllegalArgumentException(PROP_REMOTE_CONCURRENCY + "=" + value + " is outside [1, "
                    + MAX_REMOTE_CONCURRENCY + "]");
        }
        return parsed;
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

    /**
     * The online quantization config in {@code text}, parsed with {@link Json.Limits#STRICT}.
     *
     * @throws IllegalArgumentException naming {@value #PROP_QUANT_CONFIG} when the text is not one JSON object or
     *                                  not a valid configuration
     */
    private static OnlineQuantConfig parseQuantConfig(String text) {
        Object parsed;
        try {
            parsed = Json.parse(text);
        } catch (JsonException e) {
            throw new IllegalArgumentException(PROP_QUANT_CONFIG + "=" + text + " is not valid JSON: "
                    + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException(PROP_QUANT_CONFIG + "=" + text + " is not a JSON object; use"
                    + " {\"linear\": ..., \"moe\": ..., \"ignore\": [...], \"targets\": {...}}");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> object = (Map<String, Object>) parsed;
        return OnlineQuantConfig.parse(object, PROP_QUANT_CONFIG, false);
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
        private boolean quantizeOutputTensor = true;
        private OnlineQuantConfig quantConfig;
        private boolean exactQuant;
        private DraftMode draftMode = DraftMode.NONE;
        private boolean stDenseModules = true;
        private Map<String, Object> configOverrides = Collections.emptyMap();
        private Path tokenizerDir;
        private Path loraBaseConfig;
        private TensorCatalog loraVisionCatalog;
        private Path baseDir;
        private List<MetadataOverrides.Override> metadataOverrides = Collections.emptyList();
        private RopeOverrides ropeOverrides = RopeOverrides.defaults();
        private boolean checkTensors;
        private LoadFormat loadFormat = LoadFormat.FULL;
        private boolean warmup = true;
        private int remoteConcurrency = DEFAULT_REMOTE_CONCURRENCY;

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

        /** Whether online quantization also quantizes {@code output.weight} (default true). */
        public Builder quantizeOutputTensor(boolean v) { this.quantizeOutputTensor = v; return this; }

        /** vLLM's online quantization config; null removes it. */
        public Builder quantConfig(OnlineQuantConfig v) {
            this.quantConfig = v;
            return this;
        }

        /**
         * vLLM's online quantization config as the text of a JSON object (the {@value #PROP_QUANT_CONFIG} form);
         * null or blank removes it.
         *
         * @throws IllegalArgumentException naming {@value #PROP_QUANT_CONFIG} when the text is not one JSON object
         *                                  or not a valid configuration
         */
        public Builder quantConfigJson(String v) {
            this.quantConfig = v == null || v.trim().isEmpty() ? null : parseQuantConfig(v.trim());
            return this;
        }

        public Builder exactQuant(boolean v) { this.exactQuant = v; return this; }

        /** The converter's draft selection; null keeps {@link DraftMode#NONE}. */
        public Builder draftMode(DraftMode v) {
            this.draftMode = v == null ? DraftMode.NONE : v;
            return this;
        }

        /** Whether MTP / NextN layers are kept: false selects {@link DraftMode#NO_NEXTN}, true {@link DraftMode#NONE}. */
        public Builder includeMtp(boolean v) {
            this.draftMode = v ? DraftMode.NONE : DraftMode.NO_NEXTN;
            return this;
        }
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

        /**
         * The tensor table of the mmproj file beside the model, against which a LoRA adapter target absent from the
         * base model resolves as a vision-encoder pair; null removes it.
         */
        public Builder loraVisionCatalog(TensorCatalog v) { this.loraVisionCatalog = v; return this; }
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

        /** {@code --rope-scale N}: the context scaling factor ({@code rope.scaling.factor = N}). */
        public Builder ropeScale(float v) {
            RopeOverrides.Builder b = ropeOverrides.toBuilder();
            b.scalingFactor(v);
            return ropeOverrides(b);
        }

        /** {@code --rope-freq-scale N}: the frequency scale, a context scaling factor of {@code 1 / N}. */
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

        /** How many objects of an object-store checkpoint are downloaded at once; checked by {@link #build()}. */
        public Builder remoteConcurrency(int v) { this.remoteConcurrency = v; return this; }

        /**
         * @throws IllegalArgumentException for a negative memory budget, a download concurrency outside
         *                                  {@code [1, MAX_REMOTE_CONCURRENCY]}, or a quantization config with a
         *                                  llama-quantize {@value #PROP_ONLINE_QUANT} type
         */
        public ModelSourceOptions build() {
            if (memoryBudget < 0) {
                throw new IllegalArgumentException(PROP_MEMORY_MAX + " must not be negative, got " + memoryBudget);
            }
            if (remoteConcurrency < 1 || remoteConcurrency > MAX_REMOTE_CONCURRENCY) {
                throw new IllegalArgumentException(PROP_REMOTE_CONCURRENCY + " must be in [1, "
                        + MAX_REMOTE_CONCURRENCY + "], got " + remoteConcurrency);
            }
            if (quantConfig != null && onlineQuant != OnlineQuant.NONE && !onlineQuant.isVllm()) {
                throw new IllegalArgumentException(PROP_QUANT_CONFIG + " configures vLLM's online quantization; "
                        + PROP_ONLINE_QUANT + "=" + onlineQuant.name().toLowerCase(Locale.ROOT)
                        + " is a llama-quantize type");
            }
            return new ModelSourceOptions(this);
        }
    }
}
