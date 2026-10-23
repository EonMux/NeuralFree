package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * Everything a caller chooses when opening a model: backend and device, memory, sampling defaults, weight
 * streaming, speculative decoding and LoRA adapters. Immutable; built with {@link #builder()} or read from system
 * properties with {@link #fromSystemProperties()}.
 *
 * <p>Properties read by {@link #fromProperties(Properties)} (absent keys keep their defaults):
 * <ul>
 *   <li>{@code neuralfree.backend}: auto | opencl | cuda | vulkan | cpu, default auto.</li>
 *   <li>{@code neuralfree.device}: device-name substring the backend selection must match, default none.</li>
 *   <li>The {@link MemoryConfig} keys, including {@code neuralfree.context.shift}.</li>
 *   <li>The {@link SamplerConfig} keys, including {@code neuralfree.seed}.</li>
 *   <li>{@code neuralfree.ram.cache.mb}: host cache for streamed weights in MiB; -1 (default) = half the maximum
 *       heap, 0 = no cache.</li>
 *   <li>{@code neuralfree.vram.depth}: layers kept resident in VRAM when streaming; 0 (default) = from the VRAM
 *       budget.</li>
 *   <li>{@code neuralfree.draft}: path of a draft GGUF for speculative decoding, default none: a separate draft model,
 *       an eagle3, gemma4-assistant or dflash drafter, or a GGUF whose last block holds NextN / MTP tensors (with or
 *       without its trunk), which drafts with those blocks only ({@link LLMModelFactory}).</li>
 *   <li>{@code neuralfree.draft.type}: how the draft file drafts, llama.cpp {@code --spec-type}
 *       ({@link DraftType}): auto (default; detected from the file), draft-mtp (alias mtp; its NextN / MTP blocks,
 *       or the model's own without a draft file) or draft-simple (aliases simple, model; a separate draft
 *       model).</li>
 *   <li>{@code neuralfree.draft.max} (default 3), {@code neuralfree.draft.min} (default 0),
 *       {@code neuralfree.draft.p.min} (default 0): llama.cpp {@code common_params_speculative_draft}.</li>
 *   <li>{@code neuralfree.mtp}: self-speculation with the model's own NextN / MTP layers, default false. A GPU
 *       load places the NextN / MTP blocks on the device and in its context fit only when it is set (or
 *       {@code neuralfree.nextn.memory} is), as llama.cpp loads them only with {@code load_mtp}.</li>
 *   <li>{@code neuralfree.lora}: LoRA adapters as {@code path[:scale][,path[:scale]...]} (llama.cpp {@code --lora} /
 *       {@code --lora-scaled}), scale 1 when omitted, default none; see {@link LoraSpec#parseList}.</li>
 * </ul>
 */
public final class EngineConfig {

    public static final String PROP_BACKEND = "neuralfree.backend";
    public static final String PROP_DEVICE = "neuralfree.device";
    public static final String PROP_RAM_CACHE_MB = "neuralfree.ram.cache.mb";
    public static final String PROP_VRAM_DEPTH = "neuralfree.vram.depth";
    public static final String PROP_DRAFT = "neuralfree.draft";
    public static final String PROP_DRAFT_TYPE = "neuralfree.draft.type";
    public static final String PROP_DRAFT_MAX = "neuralfree.draft.max";
    public static final String PROP_DRAFT_MIN = "neuralfree.draft.min";
    public static final String PROP_DRAFT_P_MIN = "neuralfree.draft.p.min";
    public static final String PROP_MTP = "neuralfree.mtp";
    public static final String PROP_LORA = "neuralfree.lora";

    /** {@link #ramCacheMb()} value meaning "half the maximum heap". */
    public static final long RAM_CACHE_AUTO = -1L;
    /** {@link #vramDepth()} value meaning "derive from the VRAM budget". */
    public static final int VRAM_DEPTH_AUTO = 0;

    /**
     * One LoRA adapter to apply: the adapter GGUF file and the user scale (llama.cpp {@code common_adapter_lora_info}
     * path and scale; the effective scale of a tensor is {@code scale * alpha / rank}, llama-adapter.h:53-57).
     * Immutable.
     */
    public static final class LoraSpec {
        private final String path;
        private final float scale;

        /**
         * @throws IllegalArgumentException for a blank path or a non-finite scale
         */
        public LoraSpec(String path, float scale) {
            if (path == null || path.trim().isEmpty()) {
                throw new IllegalArgumentException("LoRA adapter path is empty");
            }
            if (!Float.isFinite(scale)) {
                throw new IllegalArgumentException("LoRA adapter " + path + " has a non-finite scale " + scale);
            }
            this.path = path.trim();
            this.scale = scale;
        }

        /** An adapter at scale 1 ({@code --lora FNAME}). */
        public static LoraSpec of(String path) {
            return new LoraSpec(path, 1.0f);
        }

        /**
         * One {@code path[:scale]} entry: the text after the last {@code ':'} is the scale when it parses as a finite
         * number, otherwise the whole entry is the path (so a path containing {@code ':'} needs an explicit scale).
         *
         * @throws IllegalArgumentException for an empty entry
         */
        public static LoraSpec parse(String entry) {
            String e = Objects.requireNonNull(entry, "entry").trim();
            int colon = e.lastIndexOf(':');
            if (colon > 0 && colon < e.length() - 1) {
                Float scale = finiteOrNull(e.substring(colon + 1).trim());
                if (scale != null) {
                    return new LoraSpec(e.substring(0, colon), scale);
                }
            }
            return of(e);
        }

        /**
         * A comma-separated list of {@link #parse} entries; empty entries are skipped.
         *
         * @throws IllegalArgumentException for an invalid entry
         */
        public static List<LoraSpec> parseList(String value) {
            List<LoraSpec> out = new ArrayList<>();
            if (value == null) {
                return out;
            }
            for (String entry : value.split(",")) {
                if (!entry.trim().isEmpty()) {
                    out.add(parse(entry));
                }
            }
            return out;
        }

        private static Float finiteOrNull(String s) {
            try {
                float f = Float.parseFloat(s);
                return Float.isFinite(f) ? f : null;
            } catch (NumberFormatException ex) {
                return null;
            }
        }

        /** The adapter GGUF file. */
        public String path() { return path; }
        /** The user scale. */
        public float scale() { return scale; }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof LoraSpec)) {
                return false;
            }
            LoraSpec other = (LoraSpec) o;
            return path.equals(other.path) && Float.compare(scale, other.scale) == 0;
        }

        @Override
        public int hashCode() {
            return 31 * path.hashCode() + Float.hashCode(scale);
        }

        @Override
        public String toString() {
            return path + ":" + scale;
        }
    }

    /** Compute backend choice. */
    public enum BackendMode {
        /** The fastest GPU backend that can run the LLM kernels, else the CPU. */
        AUTO,
        OPENCL,
        CUDA,
        VULKAN,
        /** The CPU engine only. */
        CPU;

        /** Whether a GPU backend is requested or allowed. */
        public boolean usesGpu() {
            return this != CPU;
        }

        /**
         * The acceleration mode used for GPU backend selection.
         *
         * @throws IllegalStateException for {@link #CPU}
         */
        public Config.ACCELERATION_MODE toAccelerationMode() {
            switch (this) {
                case AUTO:
                    return Config.ACCELERATION_MODE.AUTO;
                case OPENCL:
                    return Config.ACCELERATION_MODE.OPENCL;
                case CUDA:
                    return Config.ACCELERATION_MODE.CUDA;
                case VULKAN:
                    return Config.ACCELERATION_MODE.VULKAN;
                default:
                    throw new IllegalStateException("backend mode CPU has no GPU acceleration mode");
            }
        }

        /**
         * @throws IllegalArgumentException for an unknown name
         */
        public static BackendMode parse(String value) {
            String s = Objects.requireNonNull(value, "backend mode").trim().toUpperCase(Locale.ROOT);
            for (BackendMode m : values()) {
                if (m.name().equals(s)) {
                    return m;
                }
            }
            throw new IllegalArgumentException(PROP_BACKEND + "=" + value + " is invalid; use auto, opencl, cuda, vulkan or cpu");
        }
    }

    /**
     * How a draft file drafts for its target: llama.cpp's speculative type ({@code --spec-type},
     * common/speculative.cpp:75-77). llama.cpp detects the type from the draft file only when none is given
     * (common/arg.cpp:566-572); an explicit type overrides the detection.
     */
    public enum DraftType {
        /**
         * Detected from the draft file as llama.cpp detects it (common/speculative.cpp:2398-2433): dflash, eagle3 and
         * gemma4-assistant files draft as their architecture; a file whose last block holds {@code nextn.eh_proj}
         * drafts with its NextN / MTP blocks when it is an MTP draft of the target's model, and otherwise, when it
         * holds a model of its own, as a separate draft model; any other file is a separate draft model.
         */
        AUTO("auto"),
        /**
         * {@code draft-mtp}: the draft file's NextN / MTP blocks over the target's hidden states, refused unless the
         * file is an MTP draft of the target's model; without a draft file, the target's own NextN / MTP blocks
         * (common/speculative.cpp:2691-2700), as {@link EngineConfig#mtp()}.
         */
        MTP("draft-mtp"),
        /** {@code draft-simple}: the draft file as a separate draft model, NextN / MTP blocks or not. */
        MODEL("draft-simple");

        private final String llamaName;

        DraftType(String llamaName) {
            this.llamaName = llamaName;
        }

        /** llama.cpp's name of the type ({@code auto} for automatic detection). */
        public String llamaName() {
            return llamaName;
        }

        /**
         * The type of a llama.cpp name ({@code auto}, {@code draft-mtp}, {@code draft-simple}) or an alias
         * ({@code mtp}; {@code simple}, {@code model}), ignoring case.
         *
         * @throws IllegalArgumentException for any other value, naming the valid ones
         */
        public static DraftType parse(String value) {
            String s = Objects.requireNonNull(value, "draft type").trim().toLowerCase(Locale.ROOT);
            switch (s) {
                case "auto":
                    return AUTO;
                case "draft-mtp":
                case "mtp":
                    return MTP;
                case "draft-simple":
                case "simple":
                case "model":
                    return MODEL;
                case "draft-eagle3":
                case "draft-dflash":
                    throw new IllegalArgumentException("draft type " + value + " is detected from the draft file's"
                            + " architecture; use auto");
                default:
                    throw new IllegalArgumentException("draft type " + value + " is invalid; use auto, draft-mtp or"
                            + " draft-simple");
            }
        }
    }

    private final BackendMode backend;
    private final String deviceFilter;
    private final MemoryConfig memory;
    private final SamplerConfig samplerDefaults;
    private final long ramCacheMb;
    private final int vramDepth;
    private final String draftModelPath;
    private final DraftType draftType;
    private final int draftMax;
    private final int draftMin;
    private final float draftPMin;
    private final boolean mtp;
    /** {@link Builder#mtp(boolean)} as set, without the self-speculation {@link DraftType#MTP} implies. */
    private final boolean mtpSet;
    private final List<LoraSpec> loras;

    private EngineConfig(Builder b, MemoryConfig memory) {
        this.backend = b.backend;
        this.deviceFilter = b.deviceFilter;
        this.memory = memory;
        this.samplerDefaults = b.samplerDefaults;
        this.ramCacheMb = b.ramCacheMb;
        this.vramDepth = b.vramDepth;
        this.draftModelPath = b.draftModelPath;
        this.draftType = b.draftType;
        this.draftMax = b.draftMax;
        this.draftMin = b.draftMin;
        this.draftPMin = b.draftPMin;
        this.mtp = b.mtp || selfMtp(b);
        this.mtpSet = b.mtp;
        this.loras = Collections.unmodifiableList(new ArrayList<>(b.loras));
    }

    /**
     * Defaults: AUTO backend, no device filter, default memory and sampling, automatic streaming, no drafting, no
     * LoRA adapters.
     */
    public static EngineConfig defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Configuration from {@link System#getProperties()}. */
    public static EngineConfig fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    /**
     * Configuration from a property set; absent keys keep their defaults.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static EngineConfig fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        Builder d = builder();
        Builder b = builder()
                .memory(MemoryConfig.fromProperties(p))
                .samplerDefaults(SamplerConfig.fromProperties(p))
                .ramCacheMb(longProperty(p, PROP_RAM_CACHE_MB, d.ramCacheMb))
                .vramDepth(intProperty(p, PROP_VRAM_DEPTH, d.vramDepth))
                .draftModelPath(stringProperty(p, PROP_DRAFT))
                .draftType(draftTypeProperty(p))
                .draftMax(intProperty(p, PROP_DRAFT_MAX, d.draftMax))
                .draftMin(intProperty(p, PROP_DRAFT_MIN, d.draftMin))
                .draftPMin(floatProperty(p, PROP_DRAFT_P_MIN, d.draftPMin))
                .mtp(booleanProperty(p, PROP_MTP, d.mtp))
                .deviceFilter(stringProperty(p, PROP_DEVICE))
                .loras(loraProperty(p));
        String backend = stringProperty(p, PROP_BACKEND);
        if (backend != null) {
            b.backend(BackendMode.parse(backend));
        }
        return b.build();
    }

    public BackendMode backend() { return backend; }
    /** Device-name substring for GPU backend selection, or null for none; unused by the CPU backend. */
    public String deviceFilter() { return deviceFilter; }
    public MemoryConfig memory() { return memory; }
    /** Sampling used when a request does not bring its own. */
    public SamplerConfig samplerDefaults() { return samplerDefaults; }
    /** The default sampling seed ({@link SamplerConfig#RANDOM_SEED} for random). */
    public long seed() { return samplerDefaults.seed(); }
    /** Whether a full context shifts instead of stopping ({@link MemoryConfig#contextShift()}). */
    public boolean contextShift() { return memory.contextShift(); }
    /** Host cache for streamed weights in MiB, or {@link #RAM_CACHE_AUTO}. */
    public long ramCacheMb() { return ramCacheMb; }
    /** Layers kept resident in VRAM when streaming, or {@link #VRAM_DEPTH_AUTO}. */
    public int vramDepth() { return vramDepth; }
    /** Draft model path, or null. */
    public String draftModelPath() { return draftModelPath; }
    /** How the draft file drafts for its target ({@link DraftType#AUTO} by default). */
    public DraftType draftType() { return draftType; }
    /** Whether speculative decoding is configured: a draft model or NextN / MTP self-speculation. */
    public boolean speculative() { return draftModelPath != null || mtp; }
    /** Maximum tokens drafted per step. */
    public int draftMax() { return draftMax; }
    /** Minimum draft length worth verifying; shorter drafts are dropped. */
    public int draftMin() { return draftMin; }
    /** Minimum drafter probability of a greedy draft token; drafting stops below it. */
    public float draftPMin() { return draftPMin; }
    /**
     * Whether the model's own NextN / MTP layers draft for it: set explicitly, or {@link DraftType#MTP} without a
     * draft file.
     */
    public boolean mtp() { return mtp; }
    /** The LoRA adapters applied to the model, in load order (unmodifiable, empty for none). */
    public List<LoraSpec> loras() { return loras; }

    /**
     * The sampler of the legacy {@code LLMModel.generate} parameters, the one mapping every orchestrator uses: the
     * configured {@link #samplerDefaults()} with the given temperature, top-p, min-p, top-k and repetition penalty
     * in place of theirs, so typical-p, the presence and frequency penalties, greedy decoding and the seed stay as
     * configured. Values pass through unchanged, as llama.cpp's sampler parameters do, except
     * {@code repeatWindow == 0}, which penalizes the whole context ({@link SamplerConfig#PENALTY_WHOLE_CONTEXT},
     * llama.cpp {@code penalty_last_n = -1}); {@code -1} also means the whole context and {@code n > 0} the last
     * {@code n} tokens. A repetition penalty of 1 disables it.
     *
     * @throws IllegalArgumentException for a value {@link SamplerConfig.Builder#build()} rejects: a non-finite
     *                                  temperature or penalty, top-p or min-p outside [0, 1], a penalty <= 0 or a
     *                                  window below -1
     */
    public SamplerConfig legacySampler(float temperature, float topP, float repetitionPenalty, int repeatWindow,
                                       float minP, int topK) {
        return samplerDefaults.toBuilder()
                .temperature(temperature)
                .topP(topP)
                .minP(minP)
                .topK(topK)
                .repeatPenalty(repetitionPenalty)
                .penaltyLastN(repeatWindow == 0 ? SamplerConfig.PENALTY_WHOLE_CONTEXT : repeatWindow)
                .build();
    }

    public Builder toBuilder() {
        return builder().backend(backend).deviceFilter(deviceFilter).memory(memory).samplerDefaults(samplerDefaults)
                .ramCacheMb(ramCacheMb).vramDepth(vramDepth).draftModelPath(draftModelPath).draftType(draftType)
                .draftMax(draftMax).draftMin(draftMin).draftPMin(draftPMin).mtp(mtpSet).loras(loras);
    }

    @Override
    public String toString() {
        return "EngineConfig{backend=" + backend + (deviceFilter != null ? ", device~'" + deviceFilter + "'" : "")
                + ", " + memory + ", " + samplerDefaults + ", ramCacheMb="
                + (ramCacheMb == RAM_CACHE_AUTO ? "auto" : String.valueOf(ramCacheMb)) + ", vramDepth="
                + (vramDepth == VRAM_DEPTH_AUTO ? "auto" : String.valueOf(vramDepth))
                + (draftModelPath != null ? ", draft=" + draftModelPath : "")
                + (draftType != DraftType.AUTO ? ", draftType=" + draftType.llamaName() : "") + (mtp ? ", mtp" : "")
                + (speculative() ? ", draftMax=" + draftMax + ", draftMin=" + draftMin + ", draftPMin=" + draftPMin : "")
                + (loras.isEmpty() ? "" : ", lora=" + loras)
                + '}';
    }

    static String stringProperty(Properties p, String key) {
        String v = p.getProperty(key);
        if (v == null) {
            return null;
        }
        String s = v.trim();
        return s.isEmpty() ? null : s;
    }

    static DraftType draftTypeProperty(Properties p) {
        String v = stringProperty(p, PROP_DRAFT_TYPE);
        if (v == null) {
            return DraftType.AUTO;
        }
        try {
            return DraftType.parse(v);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_DRAFT_TYPE + "=" + v + ": " + e.getMessage(), e);
        }
    }

    /** {@link DraftType#MTP} without a draft file drafts with the model's own NextN / MTP blocks. */
    private static boolean selfMtp(Builder b) {
        return b.draftType == DraftType.MTP && b.draftModelPath == null;
    }

    static List<LoraSpec> loraProperty(Properties p) {
        try {
            return LoraSpec.parseList(stringProperty(p, PROP_LORA));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_LORA + "=" + p.getProperty(PROP_LORA) + ": " + e.getMessage(), e);
        }
    }

    static int intProperty(Properties p, String key, int dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        try {
            return Integer.parseInt(v);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is not an integer", e);
        }
    }

    static long longProperty(Properties p, String key, long dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        try {
            return Long.parseLong(v);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is not an integer", e);
        }
    }

    static float floatProperty(Properties p, String key, float dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        try {
            return Float.parseFloat(v);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is not a number", e);
        }
    }

    static boolean booleanProperty(Properties p, String key, boolean dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        String s = v.toLowerCase(Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(key + "=" + v + " is not a boolean; use true or false");
    }

    public static final class Builder {
        private BackendMode backend = BackendMode.AUTO;
        private String deviceFilter;
        private MemoryConfig memory = MemoryConfig.defaults();
        private SamplerConfig samplerDefaults = SamplerConfig.defaults();
        private long ramCacheMb = RAM_CACHE_AUTO;
        private int vramDepth = VRAM_DEPTH_AUTO;
        private String draftModelPath;
        private DraftType draftType = DraftType.AUTO;
        private int draftMax = 3;
        private int draftMin;
        private float draftPMin;
        private boolean mtp;
        private final List<LoraSpec> loras = new ArrayList<>();

        private Builder() {
        }

        public Builder backend(BackendMode v) { this.backend = Objects.requireNonNull(v, "backend"); return this; }

        /** Device-name substring; null or blank removes the filter. */
        public Builder deviceFilter(String v) {
            this.deviceFilter = v == null || v.trim().isEmpty() ? null : v.trim();
            return this;
        }

        public Builder memory(MemoryConfig v) { this.memory = Objects.requireNonNull(v, "memory"); return this; }
        public Builder samplerDefaults(SamplerConfig v) { this.samplerDefaults = Objects.requireNonNull(v, "samplerDefaults"); return this; }

        /** Sets the seed of the default sampler. */
        public Builder seed(long v) { this.samplerDefaults = samplerDefaults.withSeed(v); return this; }

        /** Sets {@link MemoryConfig#contextShift()}. */
        public Builder contextShift(boolean v) { this.memory = memory.toBuilder().contextShift(v).build(); return this; }

        public Builder ramCacheMb(long v) { this.ramCacheMb = v; return this; }
        public Builder vramDepth(int v) { this.vramDepth = v; return this; }

        /** Draft model path; null or blank removes it. */
        public Builder draftModelPath(String v) {
            this.draftModelPath = v == null || v.trim().isEmpty() ? null : v.trim();
            return this;
        }

        /** How the draft file drafts ({@link DraftType}). */
        public Builder draftType(DraftType v) { this.draftType = Objects.requireNonNull(v, "draftType"); return this; }

        public Builder draftMax(int v) { this.draftMax = v; return this; }
        public Builder draftMin(int v) { this.draftMin = v; return this; }
        public Builder draftPMin(float v) { this.draftPMin = v; return this; }
        public Builder mtp(boolean v) { this.mtp = v; return this; }

        /** Replaces the LoRA adapter list ({@code --lora} / {@code --lora-scaled} entries in order). */
        public Builder loras(List<LoraSpec> v) {
            Objects.requireNonNull(v, "loras");
            List<LoraSpec> copy = new ArrayList<>(v.size());
            for (LoraSpec s : v) {
                copy.add(Objects.requireNonNull(s, "lora entry"));
            }
            loras.clear();
            loras.addAll(copy);
            return this;
        }

        /** Appends one LoRA adapter at {@code scale}. */
        public Builder addLora(String path, float scale) {
            loras.add(new LoraSpec(path, scale));
            return this;
        }

        /**
         * Builds the configuration. The memory settings receive the speculative defaults of the draft model and MTP
         * settings ({@link MemoryConfig#forSpeculation}) and the compressor rollback snapshots its drafter needs
         * ({@link MemoryConfig#forRollback}, llama.cpp's {@code need_n_rs_seq}, common/common.h:401-407): the draft
         * max for the model's own NextN / MTP blocks and for a draft file under {@link DraftType#MTP} or
         * {@link DraftType#AUTO}, 0 under {@link DraftType#MODEL} or without a drafter. A file under auto may still
         * draft as a separate draft model; {@link LLMModelFactory} resolves and pins that value from the file before
         * loading. Memory values set explicitly are kept.
         *
         * @throws IllegalArgumentException naming the setting when a value is out of range
         */
        public EngineConfig build() {
            if (ramCacheMb < RAM_CACHE_AUTO) {
                throw new IllegalArgumentException(PROP_RAM_CACHE_MB + " must be >= 0, or -1 for automatic, got " + ramCacheMb);
            }
            if (vramDepth < 0) {
                throw new IllegalArgumentException(PROP_VRAM_DEPTH + " must be >= 0 (0 = automatic), got " + vramDepth);
            }
            if (draftMax <= 0) {
                throw new IllegalArgumentException(PROP_DRAFT_MAX + " must be positive, got " + draftMax);
            }
            if (draftMin < 0 || draftMin > draftMax) {
                throw new IllegalArgumentException(PROP_DRAFT_MIN + " must be in [0, " + PROP_DRAFT_MAX + "=" + draftMax
                        + "], got " + draftMin);
            }
            if (!(draftPMin >= 0f && draftPMin <= 1f)) {
                throw new IllegalArgumentException(PROP_DRAFT_P_MIN + " must be in [0, 1], got " + draftPMin);
            }
            if (draftType == DraftType.MODEL && draftModelPath == null && mtp) {
                throw new IllegalArgumentException(PROP_DRAFT_TYPE + "=" + draftType.llamaName() + " runs a separate"
                        + " draft model (" + PROP_DRAFT + "), but none is set and " + PROP_MTP + " drafts with the"
                        + " model's own NextN / MTP blocks");
            }
            boolean ownHead = mtp || selfMtp(this);
            int rollback = ownHead || draftModelPath != null && draftType != DraftType.MODEL ? draftMax : 0;
            return new EngineConfig(this, memory.forSpeculation(draftModelPath != null || ownHead, ownHead)
                    .forRollback(rollback));
        }
    }
}
