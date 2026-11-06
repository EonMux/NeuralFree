package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.batch.PrefixHashAlgo;
import com.eonmux.neuralfree.gguf.engine.batch.SchedulingPolicy;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventsConfig;
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
 *   <li>{@code neuralfree.chunked.prefill}: split a prompt prefill into per-step chunks beside the decodes of
 *       the batch engine (vLLM {@code enable_chunked_prefill}), default true.</li>
 *   <li>{@code neuralfree.max.batched.tokens}: the token budget one batch-engine step shares between the
 *       prefill chunk rows and the decode rows (vLLM {@code max_num_batched_tokens}); 0 (default) sizes the
 *       budget as the ubatch.</li>
 *   <li>{@code neuralfree.long.prefill.tokens}: the rows one request's prefill chunk may take per step while
 *       other requests run or wait for work (vLLM {@code long_prefill_token_threshold}); 0 (default) sets no
 *       cap.</li>
 *   <li>{@code neuralfree.max.encoder.tokens}: the media item rows one batch-engine step schedules over all its
 *       media prefills (vLLM {@code max_num_encoder_input_tokens}); text and prompt-embedding rows take none. 0
 *       (default) sizes the budget as the step's token budget, and an item larger than the whole budget always
 *       schedules (vLLM raises the budget to the largest multimodal item).</li>
 *   <li>{@code neuralfree.encoder.cache.rows}: rows of encoder output the batch engine's encoder cache holds —
 *       the embedding rows of identified media chunks stored once per content digest and shared by every request
 *       that carries the same medium, the least recently used unreferenced entries evicted when a new one needs
 *       the space (vLLM {@code encoder_cache_size} and its {@code EncoderCacheManager}); 0 (default) sizes the
 *       cache as the step's token budget, a negative value turns it off.</li>
 *   <li>{@code neuralfree.scheduling.policy}: the order the batch engine admits and preempts its requests in
 *       (vLLM {@code --scheduling-policy}, {@link SchedulingPolicy}): {@code fcfs} (default) by arrival, taking
 *       {@code priority} 0 only; {@code priority} by the lowest {@code priority} value, ties by arrival.</li>
 *   <li>{@code neuralfree.draft.kv.type}, {@code neuralfree.draft.kv.typeK}, {@code neuralfree.draft.kv.typeV}: the
 *       row types of the K and V caches of every draft context ({@link #forDraftContext()}), llama.cpp's
 *       {@code --spec-draft-type-k} / {@code --spec-draft-type-v} ({@code -ctkd} / {@code -ctvd}), default F16;
 *       the K/V keys override the shared one.</li>
 *   <li>{@code neuralfree.draft.ubatch}: tokens per forward step of a draft context, llama.cpp's
 *       {@code --spec-draft-ubatch-size} ({@code -ubd}); 0 (default) keeps the target plan's ubatch.</li>
 *   <li>{@code neuralfree.swa.full}: full-size caches for sliding-window layers, default false.</li>
 *   <li>{@code neuralfree.swa.checkpoint.reserve}: pages of every sliding-window group's paged pool reserved for
 *       the window pages context checkpoints pin (llama.cpp keeps its SWA checkpoints as host-memory copies, which
 *       need no pool room); default 0. The reserve makes the mid-prefill checkpoints of a sliding-window model
 *       survive beside running requests: without it the pool is sized for the running windows alone and a
 *       checkpoint's window yields to the next step's writes.</li>
 *   <li>{@code neuralfree.mla.cache}: auto | latent | decompressed, default auto (latent for MLA archs).</li>
 *   <li>{@code neuralfree.prefix.reuse}: keep the common token prefix between generations, default true. A model
 *       larger than the physical RAM turns the cache off unless the key is set (Avalon's server does the same for
 *       its prompt cache, server-context.cpp:1550-1566); set it to {@code true} to force the cache on.</li>
 *   <li>{@code neuralfree.prefix.cache.budget}: tokens the prefix cache's blocks may hold beside the running
 *       requests (the blocks no request uses) before the least recently used ones are evicted; 0 (default) leaves
 *       the cache bounded by the KV pools alone.</li>
 *   <li>{@code neuralfree.prefix.hash.algo}: the hash algorithm of the prefix cache's block hashes (vLLM
 *       {@code --prefix-caching-hash-algo}, {@link PrefixHashAlgo}): {@code sha256} (default, the vLLM default,
 *       over Python's pickle serialization), {@code sha256_cbor} (over canonical CBOR, the cross-language form a
 *       router recomputes), {@code xxhash} and {@code xxhash_cbor} (XXH3's 128-bit digest over the two
 *       serializations). The chain's first block hashes vLLM's {@code NONE_HASH} seed: the {@code PYTHONHASHSEED}
 *       environment variable when set, vLLM's fixed {@code vllm-none-hash} for the SHA-256 forms, and fresh random
 *       bytes for the XXH3 ones.</li>
 *   <li>{@code neuralfree.ctx.checkpoints}: context checkpoints the batch engine's prefix cache keeps per sequence
 *       slot for a memory whose prefix cannot be reused from KV pages alone — recurrent state (Mamba, RWKV,
 *       hybrid models) or sliding-window groups (Gemma 2/3, gpt-oss, Mistral SWA); default 32 (llama.cpp
 *       {@code --ctx-checkpoints} / {@code --swa-checkpoints}). 0 turns prefix reuse off for such a memory.</li>
 *   <li>{@code neuralfree.checkpoint.min.step}: the spacing in tokens of the checkpoints a long prompt's prefill
 *       takes on its way (llama.cpp {@code --checkpoint-min-step}), default 8192; 0 sets no minimum, so the
 *       prefill takes one at the end of every chunk. The checkpoint before the prompt's last page and the one at a
 *       sequence's end are taken either way.</li>
 *   <li>{@code neuralfree.cache.ram}: MiB of host RAM the prefix cache's host tier may hold (llama.cpp's
 *       {@code --cache-ram}, vLLM's CPU KV offloading): blocks the KV pools evict for room move there and come back
 *       into fresh pages when a later prompt walks them. 0 (default) keeps no host tier, -1 leaves it unbounded.
 *       It needs the batch engine's prefix cache over a memory whose cached blocks are a sequence's whole state
 *       (no recurrent, DSV4 compressed, sparse side or cross-attention caches); otherwise the engine refuses it by
 *       name.</li>
 *   <li>{@code neuralfree.cache.disk.mib}: MiB of disk the prefix cache's disk tier may hold (vLLM's tiering
 *       filesystem tier, llama.cpp's {@code --prompt-cache} file): blocks the host tier drops for its budget move
 *       there as one file per block and come back through the host tier when a later prompt walks them, so the
 *       cache outlives the process. 0 (default) keeps no disk tier, -1 leaves it unbounded. It runs under the
 *       host-RAM tier, which must be on, and refuses beside it for every reason the host tier refuses.</li>
 *   <li>{@code neuralfree.cache.disk.dir}: the directory the disk tier's block files live in; required when
 *       {@code neuralfree.cache.disk.mib} is set and refused beside it while the tier is off. Files of another
 *       block layout (another model, cache type or page size) in the directory are left alone and never served.</li>
 *   <li>{@code neuralfree.kv.events}: the publisher of the prefix cache's KV cache events (vLLM's
 *       {@code kv_events_config}): unset, {@code null} or {@code none} (default) publish nothing, {@code log} logs
 *       every event batch, {@code zmq} publishes on vLLM's ZMQ transport ({@code tcp://*:5557}), and a class
 *       name loads that {@code KvEventPublisher} implementation. The value may also be vLLM's
 *       {@code KVEventsConfig} JSON itself, with the ZMQ transport fields ({@code endpoint},
 *       {@code replay_endpoint}, {@code buffer_steps}, {@code hwm}, {@code max_queue_size}, {@code topic}).</li>
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
 *   <li>{@code neuralfree.parallel}: how many sequences the memory holds at once, llama.cpp's {@code -np}
 *       ({@code [1, 256]}, default 1). Values above 1 switch the memory to the paged form, as does
 *       {@code neuralfree.kv.paged}.</li>
 *   <li>{@code neuralfree.kv.paged}: force the paged memory form for a single sequence; default false. Paged memory
 *       keeps the rows of a KV group in reference-counted pages of {@code neuralfree.kv.page.size} rows instead of
 *       one contiguous row range.</li>
 *   <li>{@code neuralfree.kv.page.size}: page size P of the paged KV pools, a power of two in {@code [1, 256]},
 *       default 16 (vLLM {@code DEFAULT_BLOCK_SIZE}, vllm/config/cache.py:182).</li>
 *   <li>{@code neuralfree.kv.unified}: share the whole context between the {@code neuralfree.parallel} sequences
 *       of a paged memory; default true (llama.cpp {@code --kv-unified}). False gives each sequence a fixed
 *       {@code nCtx / parallel} cap, enforced by {@link PagedSequences}.</li>
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
    public static final String PROP_CHUNKED_PREFILL = "neuralfree.chunked.prefill";
    public static final String PROP_MAX_BATCHED_TOKENS = "neuralfree.max.batched.tokens";
    public static final String PROP_LONG_PREFILL_TOKENS = "neuralfree.long.prefill.tokens";
    public static final String PROP_MAX_ENCODER_TOKENS = "neuralfree.max.encoder.tokens";
    public static final String PROP_ENCODER_CACHE_ROWS = "neuralfree.encoder.cache.rows";
    public static final String PROP_SCHEDULING_POLICY = "neuralfree.scheduling.policy";
    public static final String PROP_DRAFT_KV_TYPE = "neuralfree.draft.kv.type";
    public static final String PROP_DRAFT_KV_TYPE_K = "neuralfree.draft.kv.typeK";
    public static final String PROP_DRAFT_KV_TYPE_V = "neuralfree.draft.kv.typeV";
    public static final String PROP_DRAFT_UBATCH = "neuralfree.draft.ubatch";
    public static final String PROP_SWA_FULL = "neuralfree.swa.full";
    public static final String PROP_SWA_CHECKPOINT_RESERVE = "neuralfree.swa.checkpoint.reserve";
    public static final String PROP_MLA_CACHE = "neuralfree.mla.cache";
    public static final String PROP_PREFIX_REUSE = "neuralfree.prefix.reuse";
    public static final String PROP_PREFIX_CACHE_BUDGET = "neuralfree.prefix.cache.budget";
    public static final String PROP_PREFIX_HASH_ALGO = "neuralfree.prefix.hash.algo";
    public static final String PROP_CTX_CHECKPOINTS = "neuralfree.ctx.checkpoints";
    public static final String PROP_CHECKPOINT_MIN_STEP = "neuralfree.checkpoint.min.step";
    public static final String PROP_CACHE_RAM = "neuralfree.cache.ram";
    public static final String PROP_CACHE_DISK_MIB = "neuralfree.cache.disk.mib";
    public static final String PROP_CACHE_DISK_DIR = "neuralfree.cache.disk.dir";
    public static final String PROP_KV_EVENTS = "neuralfree.kv.events";
    public static final String PROP_KV_ROTATION = "neuralfree.kv.rotation";
    public static final String PROP_CONTEXT_SHIFT = "neuralfree.context.shift";
    public static final String PROP_STATE_SNAPSHOTS = "neuralfree.state.snapshots";
    public static final String PROP_NEXTN_MEMORY = "neuralfree.nextn.memory";
    public static final String PROP_PARALLEL = "neuralfree.parallel";
    public static final String PROP_KV_PAGED = "neuralfree.kv.paged";
    public static final String PROP_KV_PAGE_SIZE = "neuralfree.kv.page.size";
    public static final String PROP_KV_UNIFIED = "neuralfree.kv.unified";

    /** Engine key naming a draft model; when set, speculative decoding is configured. */
    private static final String ENGINE_DRAFT = "neuralfree.draft";
    /** Engine key enabling self-speculation with the model's own NextN / MTP layers. */
    private static final String ENGINE_MTP = "neuralfree.mtp";

    /** Default tokens per forward step. */
    public static final int DEFAULT_UBATCH = 512;

    /** Bounds of {@code neuralfree.parallel} and of {@code neuralfree.kv.page.size} (llama.cpp {@code -np}). */
    public static final int MAX_PARALLEL = 256;
    /** Bounds of the page size: a power of two in {@code [1, 256]} (vLLM block sizes). */
    public static final int MAX_PAGE_SIZE = 256;
    /** Default page size of the paged KV pools (vLLM {@code DEFAULT_BLOCK_SIZE}). */
    public static final int DEFAULT_PAGE_SIZE = 16;
    /** Default context checkpoints per sequence slot (llama.cpp {@code n_ctx_checkpoints}). */
    public static final int DEFAULT_CTX_CHECKPOINTS = 32;
    /** Default spacing of the checkpoints on a long prompt's way (llama.cpp {@code checkpoint_min_step}). */
    public static final int DEFAULT_CHECKPOINT_MIN_STEP = 8192;

    /**
     * Row type of the K and V caches of every draft context ({@link #forDraftContext()}): F16, llama.cpp's
     * {@code --spec-draft-type-k/-v} default ({@code common_params_speculative_draft.cache_type_k/v},
     * common/common.h:348-349). A configuration overrides it per side with {@code neuralfree.draft.kv.typeK/V}
     * ({@link #draftTypeK()} / {@link #draftTypeV()}).
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
    private final boolean chunkedPrefill;
    private final int maxBatchedTokens;
    private final int longPrefillTokens;
    private final int maxEncoderTokens;
    private final int encoderCacheRows;
    private final SchedulingPolicy schedulingPolicy;
    private final KvDType draftTypeK;
    private final KvDType draftTypeV;
    private final int draftUbatch;
    private final boolean swaFull;
    private final int swaCheckpointReserve;
    private final MlaCacheMode mlaCache;
    private final boolean prefixReuse;
    private final int prefixCacheBudget;
    private final PrefixHashAlgo prefixHashAlgo;
    private final int ctxCheckpoints;
    private final int checkpointMinStep;
    private final int cacheRamMib;
    private final int cacheDiskMib;
    private final String cacheDiskDir;
    private final String kvEvents;
    private final boolean prefixReuseSet;
    private final boolean kvRotation;
    private final boolean contextShift;
    private final boolean stateSnapshots;
    private final boolean nextnMemory;
    private final int rollbackSnapshots;
    private final int parallel;
    private final boolean kvPaged;
    private final int kvPageSize;
    private final boolean kvUnified;
    private final boolean stateSnapshotsSet;
    private final boolean nextnMemorySet;
    private final boolean rollbackSnapshotsSet;

    private MemoryConfig(Builder b) {
        this.nCtx = b.nCtx;
        this.typeK = b.typeK;
        this.typeV = b.typeV;
        this.ubatch = b.ubatch;
        this.chunkedPrefill = b.chunkedPrefill;
        this.maxBatchedTokens = b.maxBatchedTokens;
        this.longPrefillTokens = b.longPrefillTokens;
        this.maxEncoderTokens = b.maxEncoderTokens;
        this.encoderCacheRows = b.encoderCacheRows;
        this.schedulingPolicy = b.schedulingPolicy;
        this.draftTypeK = b.draftTypeK;
        this.draftTypeV = b.draftTypeV;
        this.draftUbatch = b.draftUbatch;
        this.swaFull = b.swaFull;
        this.swaCheckpointReserve = b.swaCheckpointReserve;
        this.mlaCache = b.mlaCache;
        this.prefixReuse = b.prefixReuse;
        this.prefixCacheBudget = b.prefixCacheBudget;
        this.prefixHashAlgo = b.prefixHashAlgo;
        this.ctxCheckpoints = b.ctxCheckpoints;
        this.checkpointMinStep = b.checkpointMinStep;
        this.cacheRamMib = b.cacheRamMib;
        this.cacheDiskMib = b.cacheDiskMib;
        this.cacheDiskDir = b.cacheDiskDir;
        this.kvEvents = b.kvEvents;
        this.prefixReuseSet = b.prefixReuseSet;
        this.kvRotation = b.kvRotation;
        this.contextShift = b.contextShift;
        this.stateSnapshots = b.stateSnapshots;
        this.nextnMemory = b.nextnMemory;
        this.rollbackSnapshots = b.rollbackSnapshots;
        this.parallel = b.parallel;
        this.kvPaged = b.kvPaged;
        this.kvPageSize = b.kvPageSize;
        this.kvUnified = b.kvUnified;
        this.stateSnapshotsSet = b.stateSnapshotsSet;
        this.nextnMemorySet = b.nextnMemorySet;
        this.rollbackSnapshotsSet = b.rollbackSnapshotsSet;
    }

    /**
     * Defaults: auto context, F16 KV, ubatch 512, chunked prefill on beside the decodes with the ubatch as the
     * per-step token budget and no long-prefill cap, first-come first-served scheduling, ring SWA caches, auto MLA cache, prefix reuse and rotation on,
     * no recurrent-state snapshots, no NextN draft memory and no compressor rollback snapshots.
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
            KvCacheTypes t = KvCacheTypes.parse(kv, PROP_KV_TYPE);
            b.typeK(t.typeK()).typeV(t.typeV());
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
        b.chunkedPrefill(parseBool(p, PROP_CHUNKED_PREFILL, true));
        String budget = p.getProperty(PROP_MAX_BATCHED_TOKENS);
        if (budget != null) {
            b.maxBatchedTokens(parseInt(PROP_MAX_BATCHED_TOKENS, budget));
        }
        String cap = p.getProperty(PROP_LONG_PREFILL_TOKENS);
        if (cap != null) {
            b.longPrefillTokens(parseInt(PROP_LONG_PREFILL_TOKENS, cap));
        }
        String encoderBudget = p.getProperty(PROP_MAX_ENCODER_TOKENS);
        if (encoderBudget != null) {
            b.maxEncoderTokens(parseInt(PROP_MAX_ENCODER_TOKENS, encoderBudget));
        }
        String encoderCache = p.getProperty(PROP_ENCODER_CACHE_ROWS);
        if (encoderCache != null) {
            b.encoderCacheRows(parseInt(PROP_ENCODER_CACHE_ROWS, encoderCache));
        }
        String policy = p.getProperty(PROP_SCHEDULING_POLICY);
        if (policy != null) {
            b.schedulingPolicy(SchedulingPolicy.parse(policy, PROP_SCHEDULING_POLICY));
        }
        String draftKv = p.getProperty(PROP_DRAFT_KV_TYPE);
        if (draftKv != null) {
            KvCacheTypes t = KvCacheTypes.parse(draftKv, PROP_DRAFT_KV_TYPE);
            b.draftTypeK(t.typeK()).draftTypeV(t.typeV());
        }
        String draftK = p.getProperty(PROP_DRAFT_KV_TYPE_K);
        if (draftK != null) {
            b.draftTypeK(KvDType.parse(draftK, PROP_DRAFT_KV_TYPE_K));
        }
        String draftV = p.getProperty(PROP_DRAFT_KV_TYPE_V);
        if (draftV != null) {
            b.draftTypeV(KvDType.parse(draftV, PROP_DRAFT_KV_TYPE_V));
        }
        String draftUb = p.getProperty(PROP_DRAFT_UBATCH);
        if (draftUb != null) {
            b.draftUbatch(parseInt(PROP_DRAFT_UBATCH, draftUb));
        }
        String mla = p.getProperty(PROP_MLA_CACHE);
        if (mla != null) {
            b.mlaCache(MlaCacheMode.parse(mla));
        }
        b.swaFull(parseBool(p, PROP_SWA_FULL, false));
        String swaReserve = p.getProperty(PROP_SWA_CHECKPOINT_RESERVE);
        if (swaReserve != null) {
            b.swaCheckpointReserve(parseInt(PROP_SWA_CHECKPOINT_RESERVE, swaReserve));
        }
        if (p.getProperty(PROP_PREFIX_REUSE) != null) {
            b.prefixReuse(parseBool(p, PROP_PREFIX_REUSE, true));
        }
        String cacheBudget = p.getProperty(PROP_PREFIX_CACHE_BUDGET);
        if (cacheBudget != null) {
            b.prefixCacheBudget(parseInt(PROP_PREFIX_CACHE_BUDGET, cacheBudget));
        }
        String hashAlgo = p.getProperty(PROP_PREFIX_HASH_ALGO);
        if (hashAlgo != null) {
            b.prefixHashAlgo(PrefixHashAlgo.parse(hashAlgo, PROP_PREFIX_HASH_ALGO));
        }
        String checkpoints = p.getProperty(PROP_CTX_CHECKPOINTS);
        if (checkpoints != null) {
            b.ctxCheckpoints(parseInt(PROP_CTX_CHECKPOINTS, checkpoints));
        }
        String minStep = p.getProperty(PROP_CHECKPOINT_MIN_STEP);
        if (minStep != null) {
            b.checkpointMinStep(parseInt(PROP_CHECKPOINT_MIN_STEP, minStep));
        }
        String cacheRam = p.getProperty(PROP_CACHE_RAM);
        if (cacheRam != null) {
            b.cacheRamMib(parseInt(PROP_CACHE_RAM, cacheRam));
        }
        String cacheDisk = p.getProperty(PROP_CACHE_DISK_MIB);
        if (cacheDisk != null) {
            b.cacheDiskMib(parseInt(PROP_CACHE_DISK_MIB, cacheDisk));
        }
        String cacheDiskDir = p.getProperty(PROP_CACHE_DISK_DIR);
        if (cacheDiskDir != null) {
            b.cacheDiskDir(cacheDiskDir);
        }
        b.kvEvents(p.getProperty(PROP_KV_EVENTS));
        b.kvRotation(parseBool(p, PROP_KV_ROTATION, true));
        b.contextShift(parseBool(p, PROP_CONTEXT_SHIFT, false));
        String par = p.getProperty(PROP_PARALLEL);
        if (par != null) {
            b.parallel(parseInt(PROP_PARALLEL, par));
        }
        b.kvPaged(parseBool(p, PROP_KV_PAGED, false));
        String page = p.getProperty(PROP_KV_PAGE_SIZE);
        if (page != null) {
            b.kvPageSize(parseInt(PROP_KV_PAGE_SIZE, page));
        }
        b.kvUnified(parseBool(p, PROP_KV_UNIFIED, true));
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
     * the draft cache types ({@link #draftTypeK()} / {@link #draftTypeV()}, common/speculative.cpp:2714-2715), and
     * sets {@code n_rs_seq = 0} (common/speculative.cpp:2669, and the same parameters in the fit,
     * common/common.cpp:1258-1265). Hence:
     * <ul>
     *   <li>K and V rows of {@link #draftTypeK()} / {@link #draftTypeV()} (F16 unless configured with
     *       {@code neuralfree.draft.kv.typeK/V}, llama.cpp {@code -ctkd} / {@code -ctvd}), so a draft cache is
     *       quantized and Hadamard-rotated only when that type is;</li>
     *   <li>no recurrent-state snapshots and no compressor rollback snapshots: the target is verified, never the
     *       draft context;</li>
     *   <li>no context shift and no NextN draft context of its own;</li>
     *   <li>contiguous single-sequence memory ({@link #parallel()} 1, {@link #paged()} false): the draft context of
     *       a drafter holds one sequence, as llama.cpp drafts run on a single-seq context.</li>
     * </ul>
     * The context and ubatch stay these settings' own; a caller applies the target plan's context and, through
     * {@link #draftUbatchOr}, the draft ubatch (llama.cpp replaces the draft context's {@code n_ubatch} with the
     * speculative setting when it is above 0, common/speculative.cpp:2695-2697). Idempotent.
     */
    public MemoryConfig forDraftContext() {
        return toBuilder()
                .typeK(draftTypeK)
                .typeV(draftTypeV)
                .contextShift(false)
                .stateSnapshots(false)
                .nextnMemory(false)
                .rollbackSnapshots(0)
                .parallel(1)
                .kvPaged(false)
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
    /** Whether the batch engine splits prompt prefills into per-step chunks beside the decodes. */
    public boolean chunkedPrefill() { return chunkedPrefill; }
    /** The token budget one batch-engine step shares between prefill chunk rows and decode rows; 0 = the ubatch. */
    public int maxBatchedTokens() { return maxBatchedTokens; }
    /** The per-step cap on one request's prefill chunk while other requests run or wait; 0 = no cap. */
    public int longPrefillTokens() { return longPrefillTokens; }
    /**
     * The media item rows one batch-engine step schedules over all its media prefills (vLLM's
     * {@code max_num_encoder_input_tokens}); 0 (default) sizes the budget as the step's token budget.
     */
    public int maxEncoderTokens() { return maxEncoderTokens; }
    /**
     * Rows of encoder output the batch engine's encoder cache holds (vLLM's {@code encoder_cache_size}); 0
     * (default) sizes it as the step's token budget, a negative value turns the cache off.
     */
    public int encoderCacheRows() { return encoderCacheRows; }
    /**
     * The order the batch engine admits and preempts its requests in (vLLM's {@code --scheduling-policy}):
     * {@link SchedulingPolicy#FCFS} unless configured.
     */
    public SchedulingPolicy schedulingPolicy() { return schedulingPolicy; }
    /** Row type of the K cache of every draft context ({@link #forDraftContext()}); F16 unless configured. */
    public KvDType draftTypeK() { return draftTypeK; }
    /** Row type of the V cache of every draft context ({@link #forDraftContext()}); F16 unless configured. */
    public KvDType draftTypeV() { return draftTypeV; }
    /** Tokens per forward step of a draft context; 0 keeps the target plan's ubatch (llama.cpp {@code -ubd}). */
    public int draftUbatch() { return draftUbatch; }
    /** The draft ubatch when one is set, else {@code fallback} (llama.cpp {@code -ubd} 0 = the target's {@code -ub}). */
    public int draftUbatchOr(int fallback) { return draftUbatch > 0 ? draftUbatch : fallback; }
    public boolean swaFull() { return swaFull; }
    /**
     * Pages of every sliding-window group's paged pool reserved for the windows context checkpoints pin; 0 (the
     * default) adds none.
     */
    public int swaCheckpointReserve() { return swaCheckpointReserve; }
    public MlaCacheMode mlaCache() { return mlaCache; }
    public boolean prefixReuse() { return prefixReuse; }
    /**
     * Tokens the prefix cache's blocks no running request uses may hold before the least recently used ones are
     * evicted; 0 leaves the cache bounded by the KV pools alone.
     */
    public int prefixCacheBudget() { return prefixCacheBudget; }

    /** The block-hash algorithm of the prefix cache (vLLM {@code prefix_caching_hash_algo}, default sha256). */
    public PrefixHashAlgo prefixHashAlgo() { return prefixHashAlgo; }
    /**
     * Context checkpoints the prefix cache keeps per sequence slot for a memory with recurrent state or
     * sliding-window groups; 0 turns prefix reuse off for such a memory.
     */
    public int ctxCheckpoints() { return ctxCheckpoints; }
    /**
     * The spacing in tokens of the checkpoints a long prompt's prefill takes on its way; 0 sets no minimum, so the
     * prefill takes one at the end of every chunk.
     */
    public int checkpointMinStep() { return checkpointMinStep; }
    /**
     * MiB of host RAM the prefix cache's host tier may hold ({@code --cache-ram}): 0 keeps no tier, -1 leaves it
     * unbounded.
     */
    public int cacheRamMib() { return cacheRamMib; }
    /**
     * MiB of disk the prefix cache's disk tier may hold: 0 keeps no tier, -1 leaves it unbounded. It runs under the
     * host-RAM tier, which must be on.
     */
    public int cacheDiskMib() { return cacheDiskMib; }
    /** The directory the disk tier's block files live in, or null without a disk tier. */
    public String cacheDiskDir() { return cacheDiskDir; }
    /**
     * The KV cache event publisher ({@code neuralfree.kv.events}): {@code log}, {@code zmq}, a
     * {@code KvEventPublisher} class name, vLLM's {@code KVEventsConfig} JSON object (ZMQ transport fields
     * included), or null when the engine publishes no events.
     */
    public String kvEvents() { return kvEvents; }
    /** Whether {@link #prefixReuse()} was set with the property or a builder call, so no default may change it. */
    public boolean prefixReuseSet() { return prefixReuseSet; }
    public boolean kvRotation() { return kvRotation; }
    public boolean contextShift() { return contextShift; }
    /** Whether the memory keeps a shadow copy of every recurrent state for snapshot / restore. */
    public boolean stateSnapshots() { return stateSnapshots; }
    /**
     * Whether the context budget also holds the caches of the model's own NextN / MTP draft context; a GPU load then
     * also places the NextN / MTP blocks on the device.
     */
    public boolean nextnMemory() { return nextnMemory; }

    /** Sequence slots the memory holds at once (llama.cpp {@code -np}): 1 keeps today's contiguous form. */
    public int parallel() { return parallel; }
    /** True when the memory uses the paged KV form: {@link #parallel()} over 1 or {@link #kvPaged()} set. */
    public boolean paged() { return parallel > 1 || kvPaged; }
    public boolean kvPaged() { return kvPaged; }
    /** Page size P of the paged KV pools; a power of two in {@code [1, 256]}, irrelevant when not paged. */
    public int kvPageSize() { return kvPageSize; }
    /** True when the paged memory shares the whole context between its sequences (llama.cpp {@code --kv-unified}). */
    public boolean kvUnified() { return kvUnified; }

    /**
     * Rollback snapshots of the DeepSeek-V4 compressor state (llama.cpp {@code n_rs_seq}, llama-cparams.h:16): the
     * most trailing positions a speculative verify may drop from a model whose compressors accumulate state across
     * positions. The planner allocates this many extra planes per compressor tensor
     * ({@link LayerMemoryPlan.CompressedKvPlan#snapshots()}); 0 disables partial rollback of such memory. The engine
     * sets it to the draft length for an MTP, eagle, eagle3, gemma4-assistant or dflash drafter and to 0 for a
     * separate draft model or no drafter (common/common.h:401-407).
     */
    public int rollbackSnapshots() { return rollbackSnapshots; }

    /** A builder with every setting of this configuration, including whether the prefix-reuse and speculative defaults were set. */
    public Builder toBuilder() {
        Builder b = builder().nCtx(nCtx).typeK(typeK).typeV(typeV).ubatch(ubatch)
                .draftTypeK(draftTypeK).draftTypeV(draftTypeV).draftUbatch(draftUbatch)
                .swaFull(swaFull).swaCheckpointReserve(swaCheckpointReserve).mlaCache(mlaCache)
                .chunkedPrefill(chunkedPrefill).maxBatchedTokens(maxBatchedTokens).longPrefillTokens(longPrefillTokens)
                .maxEncoderTokens(maxEncoderTokens).encoderCacheRows(encoderCacheRows)
                .schedulingPolicy(schedulingPolicy)
                .prefixCacheBudget(prefixCacheBudget).prefixHashAlgo(prefixHashAlgo).ctxCheckpoints(ctxCheckpoints)
                .checkpointMinStep(checkpointMinStep).cacheRamMib(cacheRamMib)
                .cacheDiskMib(cacheDiskMib).cacheDiskDir(cacheDiskDir).kvEvents(kvEvents)
                .kvRotation(kvRotation)
                .contextShift(contextShift)
                .parallel(parallel).kvPaged(kvPaged).kvPageSize(kvPageSize).kvUnified(kvUnified);
        b.prefixReuse = prefixReuse;
        b.prefixReuseSet = prefixReuseSet;
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
        boolean paged = paged();
        boolean draftTypes = draftTypeK != DRAFT_KV_TYPE || draftTypeV != DRAFT_KV_TYPE;
        return "MemoryConfig{ctx=" + (nCtx == 0 ? "auto" : String.valueOf(nCtx)) + ", K=" + typeK + ", V=" + typeV
                + ", ubatch=" + ubatch
                + (chunkedPrefill ? "" : ", chunkedPrefill=false")
                + (maxBatchedTokens > 0 ? ", maxBatchedTokens=" + maxBatchedTokens : "")
                + (longPrefillTokens > 0 ? ", longPrefillTokens=" + longPrefillTokens : "")
                + (maxEncoderTokens > 0 ? ", maxEncoderTokens=" + maxEncoderTokens : "")
                + (encoderCacheRows != 0 ? ", encoderCacheRows=" + encoderCacheRows : "")
                + (schedulingPolicy != SchedulingPolicy.FCFS
                        ? ", schedulingPolicy=" + schedulingPolicy.wireName() : "")
                + (draftTypes ? ", draftK=" + draftTypeK + ", draftV=" + draftTypeV : "")
                + (draftUbatch > 0 ? ", draftUbatch=" + draftUbatch : "")
                + ", swaFull=" + swaFull + ", mla=" + mlaCache + ", prefixReuse=" + prefixReuse
                + (prefixHashAlgo != PrefixHashAlgo.SHA256 ? ", prefixHashAlgo=" + prefixHashAlgo : "")
                + (prefixCacheBudget > 0 ? ", prefixCacheBudget=" + prefixCacheBudget : "")
                + (ctxCheckpoints != DEFAULT_CTX_CHECKPOINTS ? ", ctxCheckpoints=" + ctxCheckpoints : "")
                + (checkpointMinStep != DEFAULT_CHECKPOINT_MIN_STEP ? ", checkpointMinStep=" + checkpointMinStep : "")
                + (cacheRamMib != 0 ? ", cacheRamMib=" + cacheRamMib : "")
                + (cacheDiskMib != 0 ? ", cacheDiskMib=" + cacheDiskMib + " under " + cacheDiskDir : "")
                + (kvEvents != null ? ", kvEvents=" + kvEvents : "")
                + ", rotation=" + kvRotation + ", contextShift=" + contextShift + ", stateSnapshots=" + stateSnapshots
                + ", nextnMemory=" + nextnMemory
                + (rollbackSnapshots > 0 ? ", rollbackSnapshots=" + rollbackSnapshots : "")
                + (paged ? ", parallel=" + parallel + ", pageSize=" + kvPageSize + ", unified=" + kvUnified : "") + '}';
    }

    public static final class Builder {
        private int nCtx;
        private KvDType typeK = KvDType.F16;
        private KvDType typeV = KvDType.F16;
        private int ubatch = DEFAULT_UBATCH;
        private boolean chunkedPrefill = true;
        private int maxBatchedTokens;
        private int longPrefillTokens;
        private int maxEncoderTokens;
        private int encoderCacheRows;
        private SchedulingPolicy schedulingPolicy = SchedulingPolicy.FCFS;
        private KvDType draftTypeK = DRAFT_KV_TYPE;
        private KvDType draftTypeV = DRAFT_KV_TYPE;
        private int draftUbatch;
        private boolean swaFull;
        private int swaCheckpointReserve;
        private MlaCacheMode mlaCache = MlaCacheMode.AUTO;
        private boolean prefixReuse = true;
        private int prefixCacheBudget;
        private PrefixHashAlgo prefixHashAlgo = PrefixHashAlgo.SHA256;
        private int ctxCheckpoints = DEFAULT_CTX_CHECKPOINTS;
        private int checkpointMinStep = DEFAULT_CHECKPOINT_MIN_STEP;
        private int cacheRamMib;
        private int cacheDiskMib;
        private String cacheDiskDir;
        private String kvEvents;
        private boolean prefixReuseSet;
        private boolean kvRotation = true;
        private boolean contextShift;
        private boolean stateSnapshots;
        private boolean nextnMemory;
        private int rollbackSnapshots;
        private int parallel = 1;
        private boolean kvPaged;
        private int kvPageSize = DEFAULT_PAGE_SIZE;
        private boolean kvUnified = true;
        private boolean stateSnapshotsSet;
        private boolean nextnMemorySet;
        private boolean rollbackSnapshotsSet;

        private Builder() {
        }

        public Builder nCtx(int v) { this.nCtx = v; return this; }
        public Builder typeK(KvDType v) { this.typeK = Objects.requireNonNull(v, "typeK"); return this; }
        public Builder typeV(KvDType v) { this.typeV = Objects.requireNonNull(v, "typeV"); return this; }
        public Builder ubatch(int v) { this.ubatch = v; return this; }
        /** Sets whether prompt prefills chunk per step beside the decodes (default true). */
        public Builder chunkedPrefill(boolean v) { this.chunkedPrefill = v; return this; }
        /** Sets the per-step token budget over prefill chunk rows and decode rows; 0 sizes it as the ubatch. */
        public Builder maxBatchedTokens(int v) { this.maxBatchedTokens = v; return this; }
        /** Sets the per-step cap on one request's prefill chunk; 0 sets no cap. */
        public Builder longPrefillTokens(int v) { this.longPrefillTokens = v; return this; }
        /**
         * Sets the media rows one batch-engine step schedules over its media prefills (vLLM's
         * {@code max_num_encoder_input_tokens}); 0 sizes the budget as the step's token budget.
         */
        public Builder maxEncoderTokens(int v) { this.maxEncoderTokens = v; return this; }
        /**
         * Sets the encoder output cache's row budget (vLLM's {@code encoder_cache_size}); 0 sizes it as the
         * step's token budget and a negative value turns the cache off.
         */
        public Builder encoderCacheRows(int v) { this.encoderCacheRows = v; return this; }
        /** Sets the batch engine's scheduling policy (default {@link SchedulingPolicy#FCFS}). */
        public Builder schedulingPolicy(SchedulingPolicy v) {
            this.schedulingPolicy = Objects.requireNonNull(v, "schedulingPolicy");
            return this;
        }
        /** Sets the K row type of every draft context (default {@link #DRAFT_KV_TYPE}). */
        public Builder draftTypeK(KvDType v) { this.draftTypeK = Objects.requireNonNull(v, "draftTypeK"); return this; }
        /** Sets the V row type of every draft context (default {@link #DRAFT_KV_TYPE}). */
        public Builder draftTypeV(KvDType v) { this.draftTypeV = Objects.requireNonNull(v, "draftTypeV"); return this; }
        /** Sets the tokens per forward step of a draft context; 0 keeps the target plan's ubatch. */
        public Builder draftUbatch(int v) { this.draftUbatch = v; return this; }
        /** Sets the sequence slots of the memory ({@code -np}); the default 1 keeps today's contiguous form. */
        public Builder parallel(int v) { this.parallel = v; return this; }
        /** Forces the paged KV form for a single sequence. */
        public Builder kvPaged(boolean v) { this.kvPaged = v; return this; }
        /** Sets the page size P of the paged KV pools (default {@link #DEFAULT_PAGE_SIZE}). */
        public Builder kvPageSize(int v) { this.kvPageSize = v; return this; }
        /** Shares the whole context between the paged sequences (default true, llama.cpp {@code --kv-unified}). */
        public Builder kvUnified(boolean v) { this.kvUnified = v; return this; }
        public Builder swaFull(boolean v) { this.swaFull = v; return this; }
        /** Sets the sliding-window pool pages reserved for checkpoint windows (default 0). */
        public Builder swaCheckpointReserve(int v) { this.swaCheckpointReserve = v; return this; }
        public Builder mlaCache(MlaCacheMode v) { this.mlaCache = Objects.requireNonNull(v, "mlaCache"); return this; }
        /**
         * Sets {@link MemoryConfig#prefixReuse()} explicitly; a model larger than RAM then keeps it
         * ({@code BatchEngine} turns an unset cache off beside such a model).
         */
        public Builder prefixReuse(boolean v) {
            this.prefixReuse = v;
            this.prefixReuseSet = true;
            return this;
        }
        /** Sets the prefix cache's token budget; 0 (default) leaves it bounded by the KV pools alone. */
        public Builder prefixCacheBudget(int v) { this.prefixCacheBudget = v; return this; }
        /** Sets the block-hash algorithm of the prefix cache (vLLM {@code --prefix-caching-hash-algo}). */
        public Builder prefixHashAlgo(PrefixHashAlgo v) { this.prefixHashAlgo = Objects.requireNonNull(v, "v"); return this; }
        /** Sets the context checkpoints kept per sequence slot (default {@link #DEFAULT_CTX_CHECKPOINTS}). */
        public Builder ctxCheckpoints(int v) { this.ctxCheckpoints = v; return this; }
        /** Sets the checkpoint spacing on a long prompt's way (default {@link #DEFAULT_CHECKPOINT_MIN_STEP}). */
        public Builder checkpointMinStep(int v) { this.checkpointMinStep = v; return this; }
        /** Sets the host-RAM tier's MiB ({@code --cache-ram}); 0 (default) keeps none, -1 leaves it unbounded. */
        public Builder cacheRamMib(int v) { this.cacheRamMib = v; return this; }
        /**
         * Sets the disk tier's MiB (vLLM's tiering filesystem tier); 0 (default) keeps none, -1 leaves it
         * unbounded. It runs under the host-RAM tier, which must be on, and needs {@link #cacheDiskDir(String)}.
         */
        public Builder cacheDiskMib(int v) { this.cacheDiskMib = v; return this; }
        /** Sets the directory the disk tier's block files live in; required when the disk tier is on. */
        public Builder cacheDiskDir(String v) {
            this.cacheDiskDir = v == null || v.trim().isEmpty() ? null : v.trim();
            return this;
        }
        /**
         * Sets the KV cache event publisher: {@code log}, {@code zmq}, a {@code KvEventPublisher} class name, or
         * vLLM's {@code KVEventsConfig} JSON object (ZMQ transport fields included); null, blank, {@code null} or
         * {@code none} publish nothing.
         */
        public Builder kvEvents(String v) {
            String spec = v == null ? "" : v.trim();
            this.kvEvents = spec.isEmpty() || spec.equalsIgnoreCase("null") || spec.equalsIgnoreCase("none")
                    ? null : spec;
            return this;
        }
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
         * @throws IllegalArgumentException for a negative context, a non-positive ubatch, a negative token budget,
         *                                  long-prefill cap, prefix cache budget, checkpoint count or checkpoint
         *                                  spacing, sliding-window checkpoint reserve, a host-RAM cache size below
         *                                  -1, a disk cache size below -1, a disk cache without its directory or
         *                                  beside an off host-RAM tier, negative rollback snapshots, a negative
         *                                  draft ubatch, a parallel count outside {@code [1, 256]}, a page size that
         *                                  is not a power of two in {@code [1, 256]}, or a KV cache event JSON that
         *                                  is not a usable {@code KVEventsConfig}
         */
        public MemoryConfig build() {
            if (nCtx < 0) {
                throw new IllegalArgumentException(PROP_CTX + " must be >= 0 (0 = auto), got " + nCtx);
            }
            if (kvEvents != null && kvEvents.startsWith("{")) {
                KvEventsConfig.parse(kvEvents, PROP_KV_EVENTS);
            }
            if (prefixCacheBudget < 0) {
                throw new IllegalArgumentException(PROP_PREFIX_CACHE_BUDGET + " must be >= 0 (0 = unbounded), got "
                        + prefixCacheBudget);
            }
            if (ctxCheckpoints < 0) {
                throw new IllegalArgumentException(PROP_CTX_CHECKPOINTS + " must be >= 0 (0 = no checkpoints), got "
                        + ctxCheckpoints);
            }
            if (checkpointMinStep < 0) {
                throw new IllegalArgumentException(PROP_CHECKPOINT_MIN_STEP
                        + " must be >= 0 (0 = no minimum spacing), got " + checkpointMinStep);
            }
            if (cacheRamMib < -1) {
                throw new IllegalArgumentException(PROP_CACHE_RAM + " must be -1 (no limit), 0 (off) or a MiB count,"
                        + " got " + cacheRamMib);
            }
            if (swaCheckpointReserve < 0) {
                throw new IllegalArgumentException(PROP_SWA_CHECKPOINT_RESERVE
                        + " must be >= 0 (0 = no reserve), got " + swaCheckpointReserve);
            }
            if (cacheDiskMib < -1) {
                throw new IllegalArgumentException(PROP_CACHE_DISK_MIB + " must be -1 (no limit), 0 (off) or a MiB"
                        + " count, got " + cacheDiskMib);
            }
            if (cacheDiskMib != 0 && cacheDiskDir == null) {
                throw new IllegalArgumentException(PROP_CACHE_DISK_MIB + "=" + cacheDiskMib + " needs "
                        + PROP_CACHE_DISK_DIR + ", the directory its block files live in");
            }
            if (cacheDiskMib == 0 && cacheDiskDir != null) {
                throw new IllegalArgumentException(PROP_CACHE_DISK_DIR + "=" + cacheDiskDir + " needs "
                        + PROP_CACHE_DISK_MIB + " set; 0 (default) keeps no disk tier");
            }
            if (cacheDiskMib != 0 && cacheRamMib == 0) {
                throw new IllegalArgumentException(PROP_CACHE_DISK_MIB + "=" + cacheDiskMib + " runs under the"
                        + " host-RAM tier, which " + PROP_CACHE_RAM + "=0 keeps off; set a positive "
                        + PROP_CACHE_RAM + " (or -1) beside it");
            }
            if (ubatch <= 0) {
                throw new IllegalArgumentException(PROP_UBATCH + " must be positive, got " + ubatch);
            }
            if (maxBatchedTokens < 0) {
                throw new IllegalArgumentException(PROP_MAX_BATCHED_TOKENS
                        + " must be >= 0 (0 = the ubatch), got " + maxBatchedTokens);
            }
            if (longPrefillTokens < 0) {
                throw new IllegalArgumentException(PROP_LONG_PREFILL_TOKENS
                        + " must be >= 0 (0 = no cap), got " + longPrefillTokens);
            }
            if (maxEncoderTokens < 0) {
                throw new IllegalArgumentException(PROP_MAX_ENCODER_TOKENS
                        + " must be >= 0 (0 = the step's token budget), got " + maxEncoderTokens);
            }
            if (encoderCacheRows < -1) {
                throw new IllegalArgumentException(PROP_ENCODER_CACHE_ROWS + " must be >= -1 (-1 = the cache off,"
                        + " 0 = the step's token budget), got " + encoderCacheRows);
            }
            if (draftUbatch < 0) {
                throw new IllegalArgumentException(PROP_DRAFT_UBATCH + " must be >= 0 (0 = the target's ubatch), got "
                        + draftUbatch);
            }
            checkRollbackSnapshots(rollbackSnapshots);
            if (parallel < 1 || parallel > MAX_PARALLEL) {
                throw new IllegalArgumentException(PROP_PARALLEL + " must be in [1, " + MAX_PARALLEL + "], got "
                        + parallel);
            }
            if (kvPageSize < 1 || kvPageSize > MAX_PAGE_SIZE || Integer.bitCount(kvPageSize) != 1) {
                throw new IllegalArgumentException(PROP_KV_PAGE_SIZE + " must be a power of two in [1, "
                        + MAX_PAGE_SIZE + "], got " + kvPageSize);
            }
            KvCacheTypes.checkPair(typeK, typeV, PROP_KV_TYPE);
            KvCacheTypes.checkPair(draftTypeK, draftTypeV, PROP_DRAFT_KV_TYPE);
            return new MemoryConfig(this);
        }
    }
}
