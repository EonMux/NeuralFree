package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * User memory settings, read from system properties:
 * <ul>
 *   <li>{@code neuralfree.ctx}: context length; unset (0) lets {@code ContextFitter} pick
 *       {@code min(n_ctx_train, fit(budget))}. {@code neuralfree.max.context} is accepted as an alias.</li>
 *   <li>{@code neuralfree.kv.type}, {@code neuralfree.kv.typeK}, {@code neuralfree.kv.typeV}: a {@link KvDType}
 *       name, case-insensitive (f32 | f16 | bf16 | q8_0 | q4_0 | q4_1 | iq4_nl | q5_0 | q5_1, the llama.cpp
 *       {@code -ctk} / {@code -ctv} set), default F16; the K/V keys override the shared one.</li>
 *   <li>{@code neuralfree.ubatch}: tokens per forward step, default 512.</li>
 *   <li>{@code neuralfree.swa.full}: full-size caches for sliding-window layers, default false.</li>
 *   <li>{@code neuralfree.mla.cache}: auto | latent | decompressed, default auto (latent for MLA archs).</li>
 *   <li>{@code neuralfree.prefix.reuse}: keep the common token prefix between generations, default true.</li>
 *   <li>{@code neuralfree.kv.rotation}: Hadamard rotation of quantized KV, default true.</li>
 *   <li>{@code neuralfree.context.shift}: shift the context when full instead of stopping, default false.</li>
 *   <li>{@code neuralfree.state.snapshots}: allocate a shadow copy of every recurrent state with the memory, so
 *       speculative decoding can snapshot and restore it; default true when speculative decoding is configured
 *       ({@code neuralfree.draft} set or {@code neuralfree.mtp=true}, or the engine configuration's draft model or
 *       MTP setting, see {@link #forSpeculation}), as llama.cpp sizes its recurrent rollback rows from the
 *       speculative settings (common/common.cpp:1720).</li>
 *   <li>{@code neuralfree.nextn.memory}: count the caches of the model's own NextN / MTP draft context in the
 *       context budget; default true when MTP drafting is configured. A GPU load also places the NextN / MTP blocks
 *       on the device when it is set, as llama.cpp loads them and counts them in its fit only with
 *       {@code load_mtp} (llama-model.cpp:3915-3917, common/fit.cpp:141-143).</li>
 * </ul>
 * One setting has no property: {@link #rollbackSnapshots()}, llama.cpp's {@code n_rs_seq}, which the engine derives
 * from the speculative settings ({@code EngineConfig}, {@code LLMModelFactory}) as llama.cpp derives it from the
 * speculative types (common/common.h:401-407, common/common.cpp:1720).
 * {@link #forModel} applies the rules a model imposes on these settings before its memory and scratch are sized.
 * Immutable.
 */
public final class MemoryConfig {

    public static final String PROP_CTX = "neuralfree.ctx";
    public static final String PROP_CTX_ALIAS = "neuralfree.max.context";
    public static final String PROP_KV_TYPE = "neuralfree.kv.type";
    public static final String PROP_KV_TYPE_K = "neuralfree.kv.typeK";
    public static final String PROP_KV_TYPE_V = "neuralfree.kv.typeV";
    public static final String PROP_UBATCH = "neuralfree.ubatch";
    public static final String PROP_SWA_FULL = "neuralfree.swa.full";
    public static final String PROP_MLA_CACHE = "neuralfree.mla.cache";
    public static final String PROP_PREFIX_REUSE = "neuralfree.prefix.reuse";
    public static final String PROP_KV_ROTATION = "neuralfree.kv.rotation";
    public static final String PROP_CONTEXT_SHIFT = "neuralfree.context.shift";
    public static final String PROP_STATE_SNAPSHOTS = "neuralfree.state.snapshots";
    public static final String PROP_NEXTN_MEMORY = "neuralfree.nextn.memory";

    /** Engine key naming a draft model; when set, speculative decoding is configured. */
    private static final String ENGINE_DRAFT = "neuralfree.draft";
    /** Engine key enabling self-speculation with the model's own NextN / MTP layers. */
    private static final String ENGINE_MTP = "neuralfree.mtp";

    /** Default tokens per forward step. */
    public static final int DEFAULT_UBATCH = 512;

    /**
     * Row type of the K and V caches of every draft context ({@link #forDraftContext()}): F16, llama.cpp's
     * {@code --spec-draft-type-k/-v} default ({@code common_params_speculative_draft.cache_type_k/v},
     * common/common.h:348-349).
     */
    public static final KvDType DRAFT_KV_TYPE = KvDType.F16;

    /** MLA cache form. */
    public enum MlaCacheMode {
        /** Latent for MLA archs with absorbed weights, decompressed otherwise. */
        AUTO,
        /** Store {@code [kv_lora | rope]} latent rows. */
        LATENT,
        /** Store per-head K and V. */
        DECOMPRESSED;

        static MlaCacheMode parse(String v) {
            String s = v.trim().toUpperCase(Locale.ROOT);
            for (MlaCacheMode m : values()) {
                if (m.name().equals(s)) {
                    return m;
                }
            }
            throw new IllegalArgumentException(PROP_MLA_CACHE + "=" + v + " is invalid; use auto, latent or decompressed");
        }
    }

    private final int nCtx;
    private final KvDType typeK;
    private final KvDType typeV;
    private final int ubatch;
    private final boolean swaFull;
    private final MlaCacheMode mlaCache;
    private final boolean prefixReuse;
    private final boolean kvRotation;
    private final boolean contextShift;
    private final boolean stateSnapshots;
    private final boolean nextnMemory;
    private final int rollbackSnapshots;
    private final boolean stateSnapshotsSet;
    private final boolean nextnMemorySet;
    private final boolean rollbackSnapshotsSet;

    private MemoryConfig(Builder b) {
        this.nCtx = b.nCtx;
        this.typeK = b.typeK;
        this.typeV = b.typeV;
        this.ubatch = b.ubatch;
        this.swaFull = b.swaFull;
        this.mlaCache = b.mlaCache;
        this.prefixReuse = b.prefixReuse;
        this.kvRotation = b.kvRotation;
        this.contextShift = b.contextShift;
        this.stateSnapshots = b.stateSnapshots;
        this.nextnMemory = b.nextnMemory;
        this.rollbackSnapshots = b.rollbackSnapshots;
        this.stateSnapshotsSet = b.stateSnapshotsSet;
        this.nextnMemorySet = b.nextnMemorySet;
        this.rollbackSnapshotsSet = b.rollbackSnapshotsSet;
    }

    /**
     * Defaults: auto context, F16 KV, ubatch 512, ring SWA caches, auto MLA cache, prefix reuse and rotation on, no
     * recurrent-state snapshots, no NextN draft memory and no compressor rollback snapshots.
     */
    public static MemoryConfig defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Configuration from {@link System#getProperties()}. */
    public static MemoryConfig fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    /**
     * Configuration from a property set; absent keys keep their defaults.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static MemoryConfig fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        Builder b = builder();
        String ctx = p.getProperty(PROP_CTX);
        String ctxKey = PROP_CTX;
        if (ctx == null) {
            ctx = p.getProperty(PROP_CTX_ALIAS);
            ctxKey = PROP_CTX_ALIAS;
        }
        if (ctx != null) {
            b.nCtx(parseInt(ctxKey, ctx));
        }
        String kv = p.getProperty(PROP_KV_TYPE);
        if (kv != null) {
            KvDType t = KvDType.parse(kv, PROP_KV_TYPE);
            b.typeK(t).typeV(t);
        }
        String k = p.getProperty(PROP_KV_TYPE_K);
        if (k != null) {
            b.typeK(KvDType.parse(k, PROP_KV_TYPE_K));
        }
        String v = p.getProperty(PROP_KV_TYPE_V);
        if (v != null) {
            b.typeV(KvDType.parse(v, PROP_KV_TYPE_V));
        }
        String ub = p.getProperty(PROP_UBATCH);
        if (ub != null) {
            b.ubatch(parseInt(PROP_UBATCH, ub));
        }
        String mla = p.getProperty(PROP_MLA_CACHE);
        if (mla != null) {
            b.mlaCache(MlaCacheMode.parse(mla));
        }
        b.swaFull(parseBool(p, PROP_SWA_FULL, false));
        b.prefixReuse(parseBool(p, PROP_PREFIX_REUSE, true));
        b.kvRotation(parseBool(p, PROP_KV_ROTATION, true));
        b.contextShift(parseBool(p, PROP_CONTEXT_SHIFT, false));
        if (p.getProperty(PROP_STATE_SNAPSHOTS) != null) {
            b.stateSnapshots(parseBool(p, PROP_STATE_SNAPSHOTS, false));
        }
        if (p.getProperty(PROP_NEXTN_MEMORY) != null) {
            b.nextnMemory(parseBool(p, PROP_NEXTN_MEMORY, false));
        }
        boolean mtp = engineFlag(p, ENGINE_MTP);
        return b.build().forSpeculation(mtp || nonBlank(p.getProperty(ENGINE_DRAFT)), mtp);
    }

    /**
     * This configuration with the speculative-decoding defaults applied: {@link #stateSnapshots()} follows
     * {@code speculative} and {@link #nextnMemory()} follows {@code mtp}, except where a value was set explicitly
     * (a builder call or its system property), which is kept.
     *
     * @param speculative whether a drafter (a draft model or the model's own NextN / MTP layers) verifies against
     *                    this memory
     * @param mtp         whether the model's own NextN / MTP layers draft, so their caches share the budget
     */
    public MemoryConfig forSpeculation(boolean speculative, boolean mtp) {
        boolean snapshots = stateSnapshotsSet ? stateSnapshots : speculative;
        boolean nextn = nextnMemorySet ? nextnMemory : mtp;
        if (snapshots == stateSnapshots && nextn == nextnMemory) {
            return this;
        }
        Builder b = toBuilder();
        b.stateSnapshots = snapshots;
        b.nextnMemory = nextn;
        return b.build();
    }

    /**
     * This configuration with {@code n} compressor rollback snapshots ({@link #rollbackSnapshots()}), unless a value
     * was set explicitly with {@link Builder#rollbackSnapshots(int)}, which is kept.
     *
     * @throws IllegalArgumentException for a negative {@code n}
     */
    public MemoryConfig forRollback(int n) {
        checkRollbackSnapshots(n);
        if (rollbackSnapshotsSet || n == rollbackSnapshots) {
            return this;
        }
        Builder b = toBuilder();
        b.rollbackSnapshots = n;
        return b.build();
    }

    /**
     * This configuration with {@code n} compressor rollback snapshots fixed as if set with
     * {@link Builder#rollbackSnapshots(int)}, so later {@link #forRollback} calls keep it, unless a value was already
     * set explicitly, which is kept. The engine pins the value it resolves from the draft file before loading.
     *
     * @throws IllegalArgumentException for a negative {@code n}
     */
    public MemoryConfig pinRollback(int n) {
        checkRollbackSnapshots(n);
        if (rollbackSnapshotsSet) {
            return this;
        }
        return toBuilder().rollbackSnapshots(n).build();
    }

    /**
     * These settings as a draft context of speculative decoding runs with them: the caches of a separate draft model
     * or eagle3 drafter, and the NextN / MTP draft context of the target's own NextN blocks or of a draft file with
     * NextN blocks. llama.cpp creates every draft context from {@code common_base_params_to_speculative}, which takes
     * the draft cache types ({@link #DRAFT_KV_TYPE}, common/speculative.cpp:2606-2607), and sets
     * {@code n_rs_seq = 0} (common/speculative.cpp:2669, and the same parameters in the fit,
     * common/common.cpp:1258-1265). Hence:
     * <ul>
     *   <li>K and V rows of {@link #DRAFT_KV_TYPE}, so a draft cache is quantized and Hadamard-rotated only when that
     *       type is;</li>
     *   <li>no recurrent-state snapshots and no compressor rollback snapshots: the target is verified, never the
     *       draft context;</li>
     *   <li>no context shift and no NextN draft context of its own.</li>
     * </ul>
     * The context and ubatch stay these settings' own; a caller applies the target plan's. Idempotent.
     */
    public MemoryConfig forDraftContext() {
        return toBuilder()
                .typeK(DRAFT_KV_TYPE)
                .typeV(DRAFT_KV_TYPE)
                .contextShift(false)
                .stateSnapshots(false)
                .nextnMemory(false)
                .rollbackSnapshots(0)
                .build();
    }

    /**
     * These settings as {@code spec} runs with them, the rule both the CPU and the GPU orchestrators apply before
     * sizing the context, the memory and the scratch:
     * <ul>
     *   <li>a model that evaluates its whole input in one step ({@link #singleStepInput}) gets a ubatch of its whole
     *       context, the training context when the context is automatic or the user's context otherwise, as
     *       llama.cpp requires {@code n_ubatch >= n_tokens} for a non-causal input (llama-context.cpp:1628 encode,
     *       :1914 decode);</li>
     *   <li>a model without memory ({@link MemoryKind#NONE}) takes at most its context ({@link #memorylessContext})
     *       in one step, the bound the memory plan puts on the ubatch of every other model ({@link MemoryPlanner}
     *       clamps it to the fitted context).</li>
     * </ul>
     * Every other setting is kept.
     */
    public MemoryConfig forModel(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        int u = ubatch;
        if (singleStepInput(spec)) {
            int target = autoContext() ? spec.hparams().nCtxTrain() : nCtx;
            u = Math.max(u, target);
        }
        if (spec.memoryKind() == MemoryKind.NONE) {
            u = Math.min(u, memorylessContext(spec));
        }
        return u == ubatch ? this : toBuilder().ubatch(u).build();
    }

    /**
     * The context of {@code spec} when it runs without memory: the user's context, else the training context, else
     * (no training context) this ubatch.
     */
    public int memorylessContext(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        if (!autoContext()) {
            return nCtx;
        }
        int train = spec.hparams().nCtxTrain();
        return train > 0 ? train : ubatch;
    }

    /**
     * Whether {@code spec} evaluates its whole input in one step: the source of an encoder-decoder model and the
     * codes of the WavTokenizer decoder.
     */
    public static boolean singleStepInput(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.supportClass() == SupportClass.ENC_DEC || spec.arch() == Arch.WAVTOKENIZER_DEC;
    }

    private static void checkRollbackSnapshots(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("compressor rollback snapshots (n_rs_seq) must be >= 0, got " + n);
        }
    }

    private static int parseInt(String key, String value) {
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + value + " is not an integer", e);
        }
    }

    private static boolean nonBlank(String v) {
        return v != null && !v.trim().isEmpty();
    }

    /** An engine boolean read as the engine reads it: absent or blank means false. */
    private static boolean engineFlag(Properties p, String key) {
        return nonBlank(p.getProperty(key)) && parseBool(p, key, false);
    }

    private static boolean parseBool(Properties p, String key, boolean dflt) {
        String v = p.getProperty(key);
        if (v == null) {
            return dflt;
        }
        String s = v.trim().toLowerCase(Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(key + "=" + v + " is not a boolean; use true or false");
    }

    /** Requested context length; 0 means "fit to the budget". */
    public int nCtx() { return nCtx; }
    public boolean autoContext() { return nCtx == 0; }
    public KvDType typeK() { return typeK; }
    public KvDType typeV() { return typeV; }
    public int ubatch() { return ubatch; }
    public boolean swaFull() { return swaFull; }
    public MlaCacheMode mlaCache() { return mlaCache; }
    public boolean prefixReuse() { return prefixReuse; }
    public boolean kvRotation() { return kvRotation; }
    public boolean contextShift() { return contextShift; }
    /** Whether the memory keeps a shadow copy of every recurrent state for snapshot / restore. */
    public boolean stateSnapshots() { return stateSnapshots; }
    /**
     * Whether the context budget also holds the caches of the model's own NextN / MTP draft context; a GPU load then
     * also places the NextN / MTP blocks on the device.
     */
    public boolean nextnMemory() { return nextnMemory; }

    /**
     * Rollback snapshots of the DeepSeek-V4 compressor state (llama.cpp {@code n_rs_seq}, llama-cparams.h:16): the
     * most trailing positions a speculative verify may drop from a model whose compressors accumulate state across
     * positions. The planner allocates this many extra planes per compressor tensor
     * ({@link LayerMemoryPlan.CompressedKvPlan#snapshots()}); 0 disables partial rollback of such memory. The engine
     * sets it to the draft length for an MTP, eagle3, gemma4-assistant or dflash drafter and to 0 for a separate draft
     * model or no drafter (common/common.h:401-407).
     */
    public int rollbackSnapshots() { return rollbackSnapshots; }

    /** A builder with every setting of this configuration, including whether each speculative default was set. */
    public Builder toBuilder() {
        Builder b = builder().nCtx(nCtx).typeK(typeK).typeV(typeV).ubatch(ubatch).swaFull(swaFull).mlaCache(mlaCache)
                .prefixReuse(prefixReuse).kvRotation(kvRotation).contextShift(contextShift);
        b.stateSnapshots = stateSnapshots;
        b.stateSnapshotsSet = stateSnapshotsSet;
        b.nextnMemory = nextnMemory;
        b.nextnMemorySet = nextnMemorySet;
        b.rollbackSnapshots = rollbackSnapshots;
        b.rollbackSnapshotsSet = rollbackSnapshotsSet;
        return b;
    }

    @Override
    public String toString() {
        return "MemoryConfig{ctx=" + (nCtx == 0 ? "auto" : String.valueOf(nCtx)) + ", K=" + typeK + ", V=" + typeV
                + ", ubatch=" + ubatch + ", swaFull=" + swaFull + ", mla=" + mlaCache + ", prefixReuse=" + prefixReuse
                + ", rotation=" + kvRotation + ", contextShift=" + contextShift + ", stateSnapshots=" + stateSnapshots
                + ", nextnMemory=" + nextnMemory
                + (rollbackSnapshots > 0 ? ", rollbackSnapshots=" + rollbackSnapshots : "") + '}';
    }

    public static final class Builder {
        private int nCtx;
        private KvDType typeK = KvDType.F16;
        private KvDType typeV = KvDType.F16;
        private int ubatch = DEFAULT_UBATCH;
        private boolean swaFull;
        private MlaCacheMode mlaCache = MlaCacheMode.AUTO;
        private boolean prefixReuse = true;
        private boolean kvRotation = true;
        private boolean contextShift;
        private boolean stateSnapshots;
        private boolean nextnMemory;
        private int rollbackSnapshots;
        private boolean stateSnapshotsSet;
        private boolean nextnMemorySet;
        private boolean rollbackSnapshotsSet;

        private Builder() {
        }

        public Builder nCtx(int v) { this.nCtx = v; return this; }
        public Builder typeK(KvDType v) { this.typeK = Objects.requireNonNull(v, "typeK"); return this; }
        public Builder typeV(KvDType v) { this.typeV = Objects.requireNonNull(v, "typeV"); return this; }
        public Builder ubatch(int v) { this.ubatch = v; return this; }
        public Builder swaFull(boolean v) { this.swaFull = v; return this; }
        public Builder mlaCache(MlaCacheMode v) { this.mlaCache = Objects.requireNonNull(v, "mlaCache"); return this; }
        public Builder prefixReuse(boolean v) { this.prefixReuse = v; return this; }
        public Builder kvRotation(boolean v) { this.kvRotation = v; return this; }
        public Builder contextShift(boolean v) { this.contextShift = v; return this; }
        /** Sets {@link MemoryConfig#stateSnapshots()} explicitly; {@link #forSpeculation} then keeps it. */
        public Builder stateSnapshots(boolean v) {
            this.stateSnapshots = v;
            this.stateSnapshotsSet = true;
            return this;
        }

        /** Sets {@link MemoryConfig#nextnMemory()} explicitly; {@link #forSpeculation} then keeps it. */
        public Builder nextnMemory(boolean v) {
            this.nextnMemory = v;
            this.nextnMemorySet = true;
            return this;
        }

        /**
         * Sets {@link MemoryConfig#rollbackSnapshots()} explicitly; {@link MemoryConfig#forRollback} then keeps it.
         * A negative value is rejected by {@link #build()}.
         */
        public Builder rollbackSnapshots(int v) {
            this.rollbackSnapshots = v;
            this.rollbackSnapshotsSet = true;
            return this;
        }

        /**
         * @throws IllegalArgumentException for a negative context, a non-positive ubatch or negative rollback
         *                                  snapshots
         */
        public MemoryConfig build() {
            if (nCtx < 0) {
                throw new IllegalArgumentException(PROP_CTX + " must be >= 0 (0 = auto), got " + nCtx);
            }
            if (ubatch <= 0) {
                throw new IllegalArgumentException(PROP_UBATCH + " must be positive, got " + ubatch);
            }
            checkRollbackSnapshots(rollbackSnapshots);
            return new MemoryConfig(this);
        }
    }
}
