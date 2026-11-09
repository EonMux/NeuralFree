package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.HostMemory;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EmbeddingRowSupport;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import com.eonmux.neuralfree.gguf.engine.LookupDrafterFactory;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublisher;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.utils.LoggingUtil;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Opens a {@link BatchEngine} over a model and its configuration: resolves the drafter the configuration
 * selects (the lookup types, a separate draft model), refuses what the shared batch cannot carry (a
 * tree-masked step over a multi-section rope or over attention layers that take no tree mask, recurrent state
 * the step cannot rewind, a memory that is not paged), sizes the step's budgets and assembles the cache
 * machinery (context checkpoints, the host-RAM and disk tiers of the prefix cache). {@link BatchEngine#forModel}
 * delegates here; the factory also shares the engine's construction helpers, which the engine's constructor
 * calls to assemble its collaborators.
 */
final class BatchEngineFactory {

    /**
     * The engine's logger: the factory's messages report on the engine it opens.
     */
    private static final Logger LOGGER = Logger.getLogger(BatchEngine.class.getName());

    private static final double GIB = 1024.0 * 1024.0 * 1024.0;

    private BatchEngineFactory() {
    }

    /**
     * The host-RAM tier of {@code cacheRamMib} MiB ({@code neuralfree.cache.ram}, -1 unbounded), or null for 0.
     *
     * @param cacheRefusal why the engine runs no prefix cache, or null when it runs one
     * @throws UnsupportedOperationException naming the property when the engine runs no prefix cache, its memory
     *                                       restores a cached prefix only at a context checkpoint (the tier moves
     *                                       plain blocks only), or its cached blocks cannot leave it
     *                                       ({@code layoutRefusal})
     */
    static HostBlockCache hostTier(int cacheRamMib, String cacheRefusal, String layoutRefusal) {
        if (cacheRamMib == 0) {
            return null;
        }
        if (cacheRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_RAM + " refused: " + cacheRefusal);
        }
        if (layoutRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_RAM + " refused: " + layoutRefusal);
        }
        return new HostBlockCache(cacheRamMib < 0 ? -1L : cacheRamMib * (1L << 20));
    }

    /**
     * The disk tier of {@code cacheDiskMib} MiB under {@code cacheDiskDir} ({@code neuralfree.cache.disk.mib} /
     * {@code neuralfree.cache.disk.dir}, -1 unbounded), or null for 0. Its files are named by the chain hashes
     * {@code hasher} computes and kept in the namespace of its algorithm and seed, so a directory shared with
     * another seed or another engine's blocks only misses. It runs under the host-RAM tier, which
     * {@link MemoryConfig} requires to be on, and refuses for every reason the host tier refuses, naming its own
     * property.
     *
     * @throws UnsupportedOperationException naming {@code neuralfree.cache.disk.mib} when the engine runs no prefix
     *                                       cache, its memory restores a cached prefix only at a context
     *                                       checkpoint, or its cached blocks cannot leave it
     * @throws UncheckedIOException           naming {@code neuralfree.cache.disk.dir} when the directory cannot be
     *                                        created or read
     */
    static DiskBlockCache diskTier(MemoryConfig memoryConfig, BlockLayout layout, BlockHasher hasher,
                                   String cacheRefusal, String layoutRefusal, DiskBlockCache.Io diskIo) {
        if (memoryConfig.cacheDiskMib() == 0) {
            return null;
        }
        if (cacheRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_DISK_MIB + " refused: " + cacheRefusal);
        }
        if (layoutRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_DISK_MIB + " refused: " + layoutRefusal);
        }
        try {
            return new DiskBlockCache(Path.of(memoryConfig.cacheDiskDir()),
                    memoryConfig.cacheDiskMib() < 0 ? -1L : memoryConfig.cacheDiskMib() * (1L << 20), layout,
                    hasher, diskIo);
        } catch (IOException e) {
            throw new UncheckedIOException(MemoryConfig.PROP_CACHE_DISK_DIR + "=" + memoryConfig.cacheDiskDir()
                    + " is not usable: " + e.getMessage(), e);
        }
    }

    /**
     * The step's encoder compute budget (vLLM's {@code encoder_compute_budget}, raised to the largest multimodal
     * item): {@code neuralfree.max.encoder.tokens} when set, the step's token budget otherwise.
     */
    static int encoderComputeBudget(int configured, int tokenBudget) {
        return configured > 0 ? configured : tokenBudget;
    }

    /**
     * The encoder cache's row budget (vLLM's {@code encoder_cache_size}): {@code neuralfree.encoder.cache.rows}
     * when positive, the step's token budget when 0, and 0 (the cache off) when negative.
     */
    static int encoderCacheBudget(int configured, int tokenBudget) {
        if (configured < 0) {
            return 0;
        }
        return configured > 0 ? configured : tokenBudget;
    }

    /**
     * Opens the engine {@link BatchEngine#forModel(EngineModel, EngineConfig)} documents; that entry keeps the
     * public contract and delegates here.
     */
    static BatchEngine forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(config, "config");
        return forModel(model, config, HostMemory.totalBytes(), KvEventPublishers.forConfig(config.memory()),
                DiskBlockCache.FILES);
    }

    /**
     * Opens the engine {@link BatchEngine#forModel(EngineModel, EngineConfig, KvEventPublisher)} documents; that
     * entry keeps the public contract and delegates here.
     */
    static BatchEngine forModel(EngineModel model, EngineConfig config, KvEventPublisher events) {
        return forModel(model, config, HostMemory.totalBytes(), events, DiskBlockCache.FILES);
    }

    /**
     * {@link #forModel(EngineModel, EngineConfig, KvEventPublisher)} over {@code diskIo}, so a test can run the
     * engine beside a slow filesystem and observe that no step waits on it.
     */
    static BatchEngine forModel(EngineModel model, EngineConfig config, KvEventPublisher events,
                                DiskBlockCache.Io diskIo) {
        return forModel(model, config, HostMemory.totalBytes(), events, diskIo);
    }

    /** {@link BatchEngine#forModel(EngineModel, EngineConfig)} against a host whose physical RAM is
     *  {@code totalRamBytes} (&lt;= 0 = unknown, the default then stands). */
    static BatchEngine forModel(EngineModel model, EngineConfig config, long totalRamBytes) {
        Objects.requireNonNull(config, "config");
        return forModel(model, config, totalRamBytes, KvEventPublishers.forConfig(config.memory()),
                DiskBlockCache.FILES);
    }

    private static BatchEngine forModel(EngineModel model, EngineConfig config, long totalRamBytes,
                                        KvEventPublisher events, DiskBlockCache.Io diskIo) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(events, "events");
        PagedSequences paged = pagedMemory(model);
        BatchSpec spec = configSpec(model, config, paged);
        BatchDrafter drafter = spec == null ? null : configDrafter(model, config, paged);
        if (drafter != null) {
            if (spec.treeVerify() && !drafter.proposesTrees()) {
                throw new UnsupportedOperationException(SpeculativeOptions.PROP_TREE + "=true verifies tree"
                        + " proposals, and the batch drafter " + drafter + " proposes chains only; the tree verify"
                        + " runs over " + EngineConfig.PROP_DRAFT + " (a draft model), which splits each request's"
                        + " tree over " + SpeculativeOptions.PROP_TREE_BRANCHES + "=" + spec.treeBranches()
                        + " draft sequences");
            }
            LOGGER.info("speculative decoding on " + model.name() + ": " + drafter + ", draft max "
                    + spec.draftMax() + " (requested " + config.draftMax() + "), min " + spec.draftMin()
                    + ", verify " + spec.verifyMode()
                    + (spec.treeVerify() ? ", tree verify (width " + spec.treeWidth() + ", budget "
                    + spec.treeBudgetPolicyName() + ")" : ""));
        } else {
            spec = null;
        }
        try {
            return new BatchEngine(model.forwardBackend(), model.tokenizer(),
                    ReasoningParsers.forModel(config.reasoningParser(), config.toolParser(), model.tokenizer()),
                    SamplingDefaults.forModel(model, config), paged, paged.maxSequences(),
                    prefixReuseForHostRam(config.memory(), model.spec().tensors().totalBytes(), totalRamBytes),
                    config.memory(), chunkedPrefill(config), tokenBudget(config), drafter, spec, drafter != null,
                    events, model.loraRegistry(), EmbeddingRowSupport.recordToken(model.spec()),
                    model.poolingOperator(), model.supportClass(), model.spec().hparams().causal(), diskIo,
                    LogitsProcessors.forConfig(config), config.lookaheadOptions());
        } catch (RuntimeException | Error e) {
            if (drafter != null) {
                closeAll(Collections.singletonList(drafter), e);
            }
            throw e;
        }
    }

    /**
     * Opens the engine {@link BatchEngine#forModel(EngineModel, EngineConfig, BatchDrafter, BatchSpec)}
     * documents; that entry keeps the public contract and delegates here.
     */
    static BatchEngine forModel(EngineModel model, EngineConfig config, BatchDrafter drafter, BatchSpec spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(drafter, "drafter");
        Objects.requireNonNull(spec, "spec");
        PagedSequences paged = pagedMemory(model);
        if (spec.draftMax() > rollbackBound(paged)) {
            throw new UnsupportedOperationException(model.architecture() + ": a round of " + spec.draftMax()
                    + " drafted tokens cannot roll back: the DSV4 compressor keeps "
                    + paged.plan().rollbackSnapshots() + " rollback snapshots per sequence");
        }
        if (spec.treeVerify()) {
            checkBatchTreeRope(model, spec.treeVerify());
            if (!drafter.proposesTrees()) {
                throw new UnsupportedOperationException(SpeculativeOptions.PROP_TREE + "=true verifies tree"
                        + " proposals, and the caller's batch drafter " + drafter + " proposes chains only");
            }
        }
        return new BatchEngine(model.forwardBackend(), model.tokenizer(),
                ReasoningParsers.forModel(config.reasoningParser(), config.toolParser(), model.tokenizer()),
                SamplingDefaults.forModel(model, config), paged, paged.maxSequences(),
                prefixReuseForHostRam(config.memory(), model.spec().tensors().totalBytes(), HostMemory.totalBytes()),
                config.memory(), chunkedPrefill(config), tokenBudget(config), drafter, spec, false,
                KvEventPublishers.forConfig(config.memory()), model.loraRegistry(),
                EmbeddingRowSupport.recordToken(model.spec()), model.poolingOperator(), model.supportClass(),
                model.spec().hparams().causal(), DiskBlockCache.FILES, LogitsProcessors.forConfig(config),
                config.lookaheadOptions());
    }

    /** Whether a prompt prefill splits into budget-bounded chunks beside the decodes ({@code config.memory()}). */
    private static boolean chunkedPrefill(EngineConfig config) {
        return config.memory().chunkedPrefill();
    }

    /** The step's token budget over prefill chunk rows and decode rows: {@code neuralfree.max.batched.tokens}, the ubatch by default. */
    private static int tokenBudget(EngineConfig config) {
        MemoryConfig mem = config.memory();
        return mem.maxBatchedTokens() > 0 ? mem.maxBatchedTokens() : mem.ubatch();
    }

    /**
     * The paged memory the batch engine runs over, or the refusal by name: the model must generate text
     * autoregressively or pool (an embedding or classifier with pooling heads) on a backend whose memory is paged
     * ({@code neuralfree.parallel > 1} or {@code neuralfree.kv.paged=true}), on either compute backend; a
     * masked-diffusion or unsupported model refuses, and a pooling-only model without pooling heads refuses too —
     * with them it serves its pooling requests here beside any generation. An encoder-decoder also needs a
     * backend that keeps one encoding per sequence ({@code ForwardBackend.sequenceEncoder()}, the CPU and GPU
     * encoder-decoders).
     */
    private static PagedSequences pagedMemory(EngineModel model) {
        SupportClass support = model.supportClass();
        switch (support) {
            case DIFF:
                throw new UnsupportedOperationException(model.architecture()
                        + " is a masked-diffusion model; generate with DiffusionGenerator");
            case NA:
                throw new UnsupportedOperationException(model.architecture() + " (" + support
                        + ") does not generate text autoregressively");
            case EMBED:
            case CLS:
                if (model.poolingOperator() == null) {
                    throw new UnsupportedOperationException(model.architecture() + " (" + support
                            + ") does not generate text autoregressively");
                }
                break;
            default:
                break;
        }
        ForwardBackend backend = model.forwardBackend();
        ModelMemory memory = backend.memory();
        PagedSequences paged = memory == null ? null : memory.pagedSequences();
        if (paged == null) {
            throw new UnsupportedOperationException(model.architecture() + ": batched generation needs paged memory"
                    + " with a batched forward (neuralfree.parallel > 1 or neuralfree.kv.paged=true), got "
                    + (memory == null ? "no memory" : memory.getClass().getSimpleName()));
        }
        if (support == SupportClass.ENC_DEC && backend.sequenceEncoder() == null) {
            throw new UnsupportedOperationException(model.architecture() + " is an encoder-decoder and "
                    + backend.getClass().getSimpleName() + " keeps one encoding for all its sequences; batched"
                    + " encoder-decoder generation needs a backend whose sequences keep their own (a SequenceEncoder:"
                    + " the CPU and GPU encoder-decoder backends over paged memory)");
        }
        return paged;
    }

    /**
     * The most drafted tokens a verify round may drop: unbounded, except that DSV4 compressed caches drop at most
     * their compressor rollback snapshots per sequence ({@code MemoryPlan.rollbackSnapshots()}, as
     * {@code SpeculativeDecoder} bounds its rounds).
     */
    private static int rollbackBound(PagedSequences paged) {
        return paged.plan().hasCompressed() ? paged.plan().rollbackSnapshots() : Integer.MAX_VALUE;
    }

    /**
     * Whether cached prefix blocks can restore a sequence of {@code memory}: not the decoder rows of an
     * encoder-decoder ({@code encoder} set), which depend on the sequence's source that no cached block records;
     * and not the rows of a memory with DSV4 compressed caches whose streams the memory cannot copy for a context
     * checkpoint (the device memory — the heap memory copies them).
     */
    static boolean cacheable(ModelMemory memory, SequenceEncoder encoder) {
        return encoder == null && memory.compressedStateRefusal() == null;
    }

    /**
     * Refuses a memory whose recurrent state the batch engine cannot keep per sequence: the state of every
     * sequence must live in its own row the engine can zero when the slot frees, and copy for a context
     * checkpoint. Refuses a drafter over recurrent state too: a rejected draft leaves the paged rows by
     * {@code remove}, which cannot rewind a recurrent state to the last accepted token.
     *
     * @throws UnsupportedOperationException naming what the memory or the drafter lacks
     */
    static void checkSequenceState(ModelMemory memory, BatchDrafter drafter) {
        if (!memory.hasRecurrentState()) {
            return;
        }
        String refusal = memory.sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("batched generation over recurrent state needs a recurrent"
                    + " state row per sequence: " + refusal);
        }
        if (drafter != null) {
            throw new UnsupportedOperationException("speculative decoding (" + EngineConfig.PROP_DRAFT_TYPE
                    + ") over recurrent state is not supported by the batch engine: a rejected draft cannot rewind"
                    + " the recurrent state");
        }
    }

    /**
     * The context checkpoints of a memory whose cached prefixes restore only at a checkpoint ({@code needs} names
     * why), or null when {@code neuralfree.ctx.checkpoints} is 0: the prefix cache is then off for this memory.
     */
    static ContextCheckpoints contextCheckpoints(PagedSequences sequences, ModelMemory memory,
                                                 MemoryConfig config, String needs) {
        if (config.ctxCheckpoints() == 0) {
            LoggingUtil.getLogger(BatchEngine.class).info(
                    "the prefix cache is off: a memory with {} restores a cached prefix only at a context"
                            + " checkpoint, and {}=0", needs, MemoryConfig.PROP_CTX_CHECKPOINTS);
            return null;
        }
        return new ContextCheckpoints(sequences, memory, config.ctxCheckpoints(), config.checkpointMinStep());
    }

    /**
     * The speculative settings {@code config} selects ({@code neuralfree.draft.max}, {@code .min}, the verify mode),
     * or null when it configures no drafter, or when the memory cannot drop a round's rejected tokens (DSV4
     * compressed caches planned without rollback snapshots), which turns speculation off, logged.
     *
     * @throws IllegalStateException    for synthetic verification without its opt-in, naming the property
     * @throws IllegalArgumentException for invalid synthetic settings
     */
    private static BatchSpec configSpec(EngineModel model, EngineConfig config, PagedSequences paged) {
        if (config.draftModelPath() == null && !config.mtp() && config.lookupTypes().isEmpty()) {
            if (config.speculativeOptions().probabilisticDraft()) {
                throw new IllegalArgumentException(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD + "=probabilistic"
                        + " applies to a draft model, but " + EngineConfig.PROP_DRAFT + " names none");
            }
            return null;
        }
        checkBatchTreeRope(model, config.speculativeOptions().treeVerify());
        int draftMax = Math.min(config.draftMax(), rollbackBound(paged));
        if (draftMax < 1) {
            LOGGER.info("speculative decoding is off on " + model.name() + ": the DSV4 compressor keeps no rollback"
                    + " snapshots to drop a round's rejected tokens");
            return null;
        }
        return BatchSpec.of(draftMax, Math.min(config.draftMin(), draftMax), config.speculativeOptions());
    }

    /**
     * Refuses the batch engine's tree verify over a multi-section rope (M-RoPE): the tree rows run at their depth
     * positions, which the shared batch carries as the one text section the batch engine ropes every text row from;
     * a multi-section model's per-section pattern per row has no batch tree form. The single-sequence engine
     * ({@code SpeculativeDecoder}) verifies trees over a multi-section rope.
     *
     * @throws UnsupportedOperationException naming the rope type when {@code neuralfree.draft.tree} is set
     */
    private static void checkBatchTreeRope(EngineModel model, boolean treeVerify) {
        if (!treeVerify) {
            return;
        }
        com.eonmux.neuralfree.gguf.arch.RopeType ropeType = model.spec().hparams().ropeType();
        if (com.eonmux.neuralfree.gguf.ops.rope.RopeMath.positionsPerToken(ropeType) > 1) {
            throw new UnsupportedOperationException("the batch engine verifies tree proposals over a one-section"
                    + " rope; " + model.name() + " ropes a " + ropeType + " from several positions per token"
                    + " (M-RoPE), whose per-row section pattern a batch tree step does not carry. Turn "
                    + SpeculativeOptions.PROP_TREE + " off for this model");
        }
        checkBatchTreeAttention("model " + model.name(), model.spec());
    }

    /**
     * Refuses a tree-masked batch over attention whose mask rule the shared masked batch has no form for: the
     * standard paged attention consumes the per-row ancestor mask (the CPU core and the GPU paged flash kernel),
     * and the MLA layers ride the same masked paged launches (the mixer uploads each segment's relativized mask
     * rows and packs the segment's fresh-row count into the launch flags, as the single-sequence tree verify on
     * MLA does), while DSA, MSA, DSV4, recurrent and Step5 layers run their own attention whose kernels take no
     * tree mask — a step that silently ignored the mask would verify tokens no target sampled. The
     * single-sequence engine ({@code SpeculativeDecoder}) carries the same limits.
     *
     * @throws UnsupportedOperationException naming the layer's attention when the model runs a masked batch
     */
    static void checkBatchTreeAttention(String role, com.eonmux.neuralfree.gguf.spec.ModelSpec spec) {
        List<com.eonmux.neuralfree.gguf.spec.LayerSpec> layers = spec.layers();
        for (int i = 0; i < layers.size(); i++) {
            com.eonmux.neuralfree.gguf.spec.LayerSpec layer = layers.get(i);
            String takes = layer.dsa() != null ? "DSA" : layer.msa() != null ? "MSA"
                    : layer.step5() != null ? "Step5" : layer.dsv4() != null ? "DSV4" : layer.recurrent() != null
                    ? "recurrent" : null;
            if (takes != null) {
                throw new UnsupportedOperationException("a tree-masked batch runs the standard paged attention,"
                        + " whose per-row ancestor mask the " + takes + " layers of the " + role + " (layer " + i
                        + ") take no part in; turn " + SpeculativeOptions.PROP_TREE + " off for this model");
            }
        }
    }

    /**
     * The engine drafter {@code config} selects (vLLM's batched proposers, llama.cpp server's per-slot speculative
     * state): the lookup types ({@code neuralfree.draft.type}: {@code ngram-simple}, {@code ngram-map-k},
     * {@code ngram-map-k4v}, {@code ngram-mod}, {@code ngram-cache}, {@code suffix}) as a {@link BatchLookupDrafter}
     * with one drafter set per sequence slot, and a separate draft model ({@code neuralfree.draft}) as a
     * {@link BatchModelDrafter} over paged draft memory, chained in llama.cpp's ask order (the lookups first) when
     * both are set. The MTP head ({@code neuralfree.mtp}) and the hidden-fed draft files (eagle3,
     * gemma4-assistant, dflash, MTP draft files) refuse by name: they read the target's hidden rows of one
     * sequence ({@code LLMModelFactory.openSpeculative} runs them).
     *
     * @throws UnsupportedOperationException for the MTP head or a hidden-fed draft file
     * @throws UncheckedIOException          when a lookup cache file or the draft model cannot be read, naming it
     * @throws IllegalArgumentException      when the draft model cannot draft for the target, or probabilistic
     *                                       drafting is set without a draft model
     */
    private static BatchDrafter configDrafter(EngineModel model, EngineConfig config, PagedSequences paged) {
        if (config.mtp()) {
            throw new UnsupportedOperationException("the batch engine drafts with a separate draft model ("
                    + EngineConfig.PROP_DRAFT + "), the n-gram and the suffix drafters; " + EngineConfig.PROP_MTP
                    + " drafts with the target's NextN heads over one sequence's hidden rows, which runs"
                    + " single-sequence (LLMModelFactory.openSpeculative)");
        }
        boolean probabilistic = config.speculativeOptions().probabilisticDraft();
        if (probabilistic && config.draftModelPath() == null) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD + "=probabilistic"
                    + " applies to a draft model, but " + EngineConfig.PROP_DRAFT + " names none");
        }
        List<BatchDrafter> members = new ArrayList<>(2);
        try {
            LookupDrafterFactory lookups = LookupDrafterFactory.open(config);
            if (!lookups.isEmpty()) {
                members.add(new BatchLookupDrafter(lookups, paged));
            }
            if (config.draftModelPath() != null) {
                members.add(modelDrafter(model, config, probabilistic));
            }
        } catch (IOException e) {
            closeAll(members, e);
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (RuntimeException | Error e) {
            closeAll(members, e);
            throw e;
        }
        return members.size() == 1 ? members.get(0) : new BatchDrafterChain(members, paged.maxSequences());
    }

    /** The batch draft model of {@code config.draftModelPath()} over the target {@code model}. */
    private static BatchDrafter modelDrafter(EngineModel model, EngineConfig config, boolean probabilistic)
            throws IOException {
        EngineModel draft = LLMModelFactory.openBatchDraftModel(model, config);
        try {
            return new BatchModelDrafter(draft, model, config.draftPMin(), probabilistic,
                    config.speculativeOptions().heterogeneousVocab(), config.speculativeOptions().treeBranches());
        } catch (RuntimeException | Error e) {
            try {
                draft.close();
            } catch (IOException | RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    private static void closeAll(List<BatchDrafter> members, Throwable failure) {
        for (BatchDrafter m : members) {
            try {
                m.close();
            } catch (RuntimeException e) {
                failure.addSuppressed(e);
            }
        }
    }

    /**
     * Whether the prompt cache ({@link PrefixCache}) runs for a model of {@code modelBytes} on a host of
     * {@code totalRamBytes}: an explicit setting stands, and an unset cache is turned off beside a model larger
     * than the physical RAM — the model lives in the page cache of its mapping, and the retired sequences the
     * cache keeps hold host pages the model needs (Avalon tools/server/server-context.cpp:1550-1566).
     */
    static boolean prefixReuseForHostRam(MemoryConfig memory, long modelBytes, long totalRamBytes) {
        Objects.requireNonNull(memory, "memory");
        if (!memory.prefixReuse()) {
            return false;
        }
        if (memory.prefixReuseSet() || totalRamBytes <= 0 || modelBytes <= totalRamBytes) {
            return true;
        }
        LoggingUtil.getLogger(BatchEngine.class).info(
                "model ({} GiB) is larger than RAM ({} GiB): the prefix cache is disabled, set {}=true to force it",
                modelBytes / GIB, totalRamBytes / GIB, MemoryConfig.PROP_PREFIX_REUSE);
        return false;
    }
}
