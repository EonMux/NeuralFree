package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.VRAMBudget;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuAllReduce;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuLayerExecutor;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuPrefillDriver;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SessionFile;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.moe.EplbLoadWindow;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertHeat;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertRemap;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The device group of a tensor-parallel load: rank 0 runs on the model's device beside the rest of the model, and
 * every other rank runs a full GPU stack of its own (kernels, placeholders, scratch, dispatch helpers) over the
 * shard weights its slice of every split projection needs. Two splits share the group:
 *
 * <ul>
 *   <li>a feed-forward row split ({@code neuralfree.split.mode=row}) shards the trunk's dense FFN on the hidden
 *       dimension over rank 0 and its peers ({@link FfnRowSplitPlan});</li>
 *   <li>a tensor-parallel split ({@code neuralfree.split.mode=tensor}) shards the attention heads, their KV caches
 *       and the dense FFN over the same ranks ({@link TpAttentionPlan} with the FFN plan);</li>
 *   <li>a context split ({@code neuralfree.split.mode=context}) with a prefill query split over 1
 *       ({@code neuralfree.prefill_context_parallel_size}, vLLM's prefill context parallelism) runs every prefill's
 *       query rows as the DualChunkSwap partition over the ranks' context blocks ({@link PcpPrefillCoordinator},
 *       {@link PcpAttentionMixer}).</li>
 * </ul>
 *
 * <p>Rank 0 computes its shard straight from the main device's slots; every peer receives a replay of the layer
 * input in its own buffer, computes there and an all-reduce ({@link GpuAllReduce#sum}) sums the partials into rank
 * 0's output slot. A tensor split gives every rank its own {@link DeviceModelMemory} with the shard KV caches, and
 * mirrors the main memory's steps (positions, sequence appends, resets and rollbacks) onto them, so the rank caches
 * hold exactly the positions the main sequence holds.</p>
 *
 * <p>{@link #select} opens the peers ({@link GpuDevices#selectPeers}) and returns null when the machine offers no
 * second device, so the caller loads the model on one device. The selection's acceleration releases the peer
 * backends it opened; the group owns the rank infrastructure, shard buffers, memories and the collective temporary.
 * Not thread-safe; the forward path drives it from the model's step.</p>
 */
public final class TensorParallelGroup implements AutoCloseable {

    /** One rank's GPU stack, shard weights and run-time state. Rank 0 borrows the model device's pieces. */
    private static final class Rank {
        final int index;
        final String name;
        final ComputeBackend backend;
        final ComputeQueue queue;
        final LlmKernels kernels;
        final GpuPlaceholders placeholders;
        final GpuScratch scratch;
        final GpuOps ops;
        final ModelSpec spec;
        GpuWeights weights;
        GpuForwardContext context;
        DeviceModelMemory memory;
        MemoryPlan memoryPlan;
        ScratchSlot input;
        ScratchSlot partial;

        Rank(int index, String name, ComputeBackend backend, ComputeQueue queue, LlmKernels kernels,
             GpuPlaceholders placeholders, GpuScratch scratch, GpuOps ops, ModelSpec spec) {
            this.index = index;
            this.name = name;
            this.backend = backend;
            this.queue = queue;
            this.kernels = kernels;
            this.placeholders = placeholders;
            this.scratch = scratch;
            this.ops = ops;
            this.spec = spec;
        }
    }

    private final ComputeBackend mainBackend;
    private final HWAcceleration ownedAcceleration;
    private final List<GpuDevices.Selected> peers;
    private final List<Rank> ranks = new ArrayList<>();
    private final List<GpuBuffer> rankBuffers = new ArrayList<>();
    private ModelSpec spec;
    private int ubatch;
    private int nCtx;
    /** The prefill query-split rank count ({@code EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL}); 1 runs whole prefills. */
    private int prefillSplit = 1;
    private String archName;
    private FfnRowSplitPlan ffnPlan;
    private TpAttentionPlan attentionPlan;
    private TpVocabPlan vocabPlan;
    private TpMoePlan moePlan;
    private GpuBuffer allReduceTemporary;
    private FfnRowSplitUploader ffnUploader;
    private TpAttentionUploader attentionUploader;
    private TpVocabUploader vocabUploader;
    private TpMoeUploader moeUploader;
    private TpVocab vocab;
    /** Routed-row recorder of rank 0 (the routing is replicated on every rank), or null for none. */
    private MoeExpertHeat expertHeat;
    /** Live routed-row window of rank 0 for the runtime expert rearrangement, or null for none. */
    private EplbLoadWindow moeLoadWindow;
    /** Replicator of the whole trunk tensors to the peers of a prefill query split, or null for none. */
    private PcpWeightReplicator pcpReplicator;
    /** The DualChunkSwap prefill trunk runner of a prefill query split, or null until the model binds it. */
    private PcpPrefillCoordinator pcpCoordinator;
    /** The prefill context-parallel attention mixers by layer il (the peer finish path reads their parts). */
    private final Map<Integer, CpAttentionMixer> pcpMixes = new LinkedHashMap<>();

    /** The rank-memory mirror: the step uploads, appends, paged steps, shifts and resets of the rank caches. */
    private final TpMemoryMirror mirror = new TpMemoryMirror(this);

    private TensorParallelGroup(ComputeBackend mainBackend, HWAcceleration ownedAcceleration,
                                List<GpuDevices.Selected> peers) {
        this.mainBackend = Objects.requireNonNull(mainBackend, "mainBackend");
        this.ownedAcceleration = ownedAcceleration;
        this.peers = List.copyOf(peers);
    }

    /**
     * Selects and opens the peer devices ({@link GpuDevices#selectPeers}).
     *
     * @return the group holding the selection, or null when the machine offers no second device
     * @throws IllegalArgumentException when a device selector names no device
     */
    public static TensorParallelGroup select(EngineConfig cfg, ComputeBackend mainBackend) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(mainBackend, "mainBackend");
        HWAcceleration acceleration = HWAcceleration.forMode(cfg.backend().toAccelerationMode());
        List<GpuDevices.Selected> peers;
        try {
            peers = GpuDevices.selectPeers(acceleration, cfg, mainBackend);
        } catch (RuntimeException | Error failure) {
            try {
                acceleration.shutdown();
            } catch (RuntimeException shutdownFailure) {
                failure.addSuppressed(shutdownFailure);
            }
            throw failure;
        }
        if (peers.isEmpty()) {
            try {
                acceleration.shutdown();
            } catch (RuntimeException shutdownFailure) {
                // no device opened; the enumeration release failed with it
            }
            return null;
        }
        return new TensorParallelGroup(mainBackend, acceleration, peers);
    }

    /** A group over ranks the caller opened itself (a test over two backends of one device). */
    static TensorParallelGroup forPeers(ComputeBackend mainBackend, List<GpuDevices.Selected> peers) {
        return new TensorParallelGroup(mainBackend, null, peers);
    }

    /**
     * The routed-row recorder rank 0's forward context counts its routers' selections into, or null for none. The
     * routing is replicated on every rank, so rank 0's ids are the whole step's; the other ranks record nothing.
     *
     * @throws IllegalStateException when the group is already initialized
     */
    public void expertHeat(MoeExpertHeat heat) {
        if (spec != null) {
            throw new IllegalStateException("the tensor-parallel group is already initialized");
        }
        this.expertHeat = heat;
    }

    /**
     * The live routed-row load window rank 0's forward context feeds the runtime expert rearrangement from, or null
     * for none. The routing is replicated on every rank, so rank 0's routed rows are the whole step's.
     *
     * @throws IllegalStateException when the group is already initialized
     */
    public void moeLoadWindow(EplbLoadWindow window) {
        if (spec != null) {
            throw new IllegalStateException("the tensor-parallel group is already initialized");
        }
        this.moeLoadWindow = window;
    }

    /**
     * Migrates the expert banks to new physical layouts in place across every rank
     * ({@link TpMoeUploader#rearrange}): the runtime expert rearrangement's step boundary calls this with the
     * rebalanced layouts.
     *
     * @throws IllegalStateException when the group holds no expert split
     */
    public void rearrangeExperts(Map<Integer, int[]> layouts) {
        if (moeUploader == null) {
            throw new IllegalStateException("the tensor-parallel group holds no expert split to rearrange;"
                    + " enable " + EngineConfig.PROP_EXPERT_PARALLEL + "=true with an EPLB rebalancing placement");
        }
        moeUploader.rearrange(layouts, this);
    }

    /**
     * Sets the prefill query-split rank count (vLLM's {@code prefill_context_parallel_size}, before {@link #init}):
     * over 1, every prefill's query rows cut into the DualChunkSwap chunk pairs and every rank projects, stores and
     * attends its own pair while the cache scatter fills the other ranks' blocks. Needs the context split's own
     * rank count, which {@link #init} checks.
     *
     * @throws IllegalStateException   when the group is already initialized
     * @throws IllegalArgumentException for a count below 1
     */
    public void prefillContextParallelSize(int size) {
        if (spec != null) {
            throw new IllegalStateException("the tensor-parallel group is already initialized");
        }
        if (size < 1) {
            throw new IllegalArgumentException("the prefill context-parallel size must be >= 1, got " + size);
        }
        this.prefillSplit = size;
    }

    /**
     * True when every prefill's query rows split over the ranks through the contiguous context split's block
     * windows (vLLM's prefill context parallelism over the contiguous cache): the block base defines the rows a
     * rank packs, so a paged split (whose pages assign by position instead) routes the prefills through the
     * coordinator's batched trunk rather than this block-window split.
     */
    boolean prefillSplitEnabled() {
        return pcpEnabled() && !mirror.hasPageShard();
    }

    /** True when the split runs the DualChunkSwap prefill (vLLM's prefill context parallelism). */
    boolean pcpEnabled() {
        return attentionPlan != null && attentionPlan.contextParallel() && prefillSplit > 1;
    }

    /** True while the prefill context-parallel coordinator runs a step: the mixers route their phases through
     * its exchange, whatever row count a rank's packed window carries. */
    boolean pcpStepRunning() {
        return pcpCoordinator != null && pcpCoordinator.running();
    }

    /** The prefill query-split rank count ({@code 1} runs whole prefills). */
    public int pcpSize() {
        requireInit();
        return prefillSplit;
    }

    /** First context position of {@code rank}'s block (the planner's own base formula). */
    int contextBlockStart(int rank) {
        return (int) ((long) rank * nCtx / ranks.size());
    }

    /** Position one past the last row of {@code rank}'s context block. */
    int contextBlockEnd(int rank) {
        return (int) ((long) (rank + 1) * nCtx / ranks.size());
    }

    /** A rank's shard KV cache and step inputs, or null when the split holds no per-rank caches. */
    DeviceModelMemory rankMemory(int rank) {
        return ranks.get(rank).memory;
    }

    /** The plan a rank's shard memory was built from, or null before {@link #init} pinned it. */
    MemoryPlan rankMemoryPlan(int rank) {
        return ranks.get(rank).memoryPlan;
    }

    /**
     * Notes the coordinator's packed step window: the appends split the step's rows from it, and the next
     * non-trunk step upload re-runs (the trunk uploads packed windows the plain upload would otherwise see
     * as already applied).
     */
    void noteStepWindow(int pos0, int nTokens) {
        mirror.noteStepWindow(pos0, nTokens);
    }

    /** The rows of the step at {@code [pos0, pos0 + n)} this rank computes under the prefill query split. */
    int prefillSplitOffset(int rank, int pos0, int n) {
        if (!prefillSplitEnabled() || n <= 1) {
            return 0;
        }
        return Math.min(n, Math.max(0, contextBlockStart(rank) - pos0));
    }

    /**
     * Initializes a feed-forward-only group ({@code split.mode=row}): rank 0 runs on the model device with the
     * model's queue, kernels and placeholders, every peer gets own kernels, placeholders, scratch and dispatch
     * helpers, and every peer gets buffers for the replayed FFN input and its partial output.
     */
    public void init(ModelSpec spec, int ubatch, LlmKernels mainKernels, ComputeQueue mainQueue,
                     GpuPlaceholders mainPlaceholders, FfnRowSplitPlan ffnPlan) {
        init(spec, ubatch, mainKernels, mainQueue, mainPlaceholders, ffnPlan, null, null, 0);
    }

    /**
     * Initializes a group. A tensor split ({@code attentionPlan} non-null) gives every rank a shard-local spec, a
     * scratch sized from the shard spec and a memory plan for its shard KV caches; the per-rank VRAM fit check runs
     * in {@link #upload} after the model's own weights are resident and before any shard is allocated.
     *
     * @param memory the model's memory settings, for the per-rank memory plans (tensor split only)
     * @param nCtx   the main plan's context length (tensor split only)
     * @throws IllegalStateException    when the group is already initialized
     * @throws IllegalArgumentException when the plan does not split over the group's ranks
     */
    public void init(ModelSpec spec, int ubatch, LlmKernels mainKernels, ComputeQueue mainQueue,
                     GpuPlaceholders mainPlaceholders, FfnRowSplitPlan ffnPlan, TpAttentionPlan attentionPlan,
                     MemoryConfig memory, int nCtx) {
        init(spec, ubatch, mainKernels, mainQueue, mainPlaceholders, ffnPlan, attentionPlan, null, null, memory,
                nCtx);
    }

    /**
     * Initializes a group with a vocabulary split and an expert split ({@code vocabPlan} / {@code moePlan} non-null):
     * the token table, the LM head, the routed experts and the shared experts are sharded too.
     *
     * @throws IllegalArgumentException when a plan does not split over the group's ranks
     */
    public void init(ModelSpec spec, int ubatch, LlmKernels mainKernels, ComputeQueue mainQueue,
                     GpuPlaceholders mainPlaceholders, FfnRowSplitPlan ffnPlan, TpAttentionPlan attentionPlan,
                     TpVocabPlan vocabPlan, TpMoePlan moePlan, MemoryConfig memory, int nCtx) {
        if (this.ffnPlan != null || this.attentionPlan != null) {
            throw new IllegalStateException("the tensor-parallel group is already initialized");
        }
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(mainKernels, "mainKernels");
        Objects.requireNonNull(mainQueue, "mainQueue");
        Objects.requireNonNull(mainPlaceholders, "mainPlaceholders");
        if (ffnPlan != null && ffnPlan.ranks() != peers.size() + 1) {
            throw new IllegalArgumentException("the feed-forward plan splits over " + ffnPlan.ranks() + " rank(s),"
                    + " but the group holds " + (peers.size() + 1) + " (the model device and " + peers.size()
                    + " peer(s))");
        }
        if (attentionPlan != null && attentionPlan.ranks() != peers.size() + 1) {
            throw new IllegalArgumentException("the attention plan splits over " + attentionPlan.ranks() + " rank(s),"
                    + " but the group holds " + (peers.size() + 1) + " (the model device and " + peers.size()
                    + " peer(s))");
        }
        if (vocabPlan != null && vocabPlan.ranks() != peers.size() + 1) {
            throw new IllegalArgumentException("the vocabulary plan splits over " + vocabPlan.ranks() + " rank(s),"
                    + " but the group holds " + (peers.size() + 1) + " (the model device and " + peers.size()
                    + " peer(s))");
        }
        if (moePlan != null && moePlan.ranks() != peers.size() + 1) {
            throw new IllegalArgumentException("the expert plan splits over " + moePlan.ranks() + " rank(s),"
                    + " but the group holds " + (peers.size() + 1) + " (the model device and " + peers.size()
                    + " peer(s))");
        }
        if (attentionPlan != null && (memory == null || nCtx <= 0)) {
            throw new IllegalArgumentException("a tensor split needs the memory settings and the context length"
                    + " for its per-rank memory plans");
        }
        if (attentionPlan != null && attentionPlan.contextParallel() && nCtx < peers.size() + 1) {
            throw new IllegalArgumentException("context parallelism over " + (peers.size() + 1) + " ranks needs a"
                    + " context of at least that many positions, got " + nCtx);
        }
        if (prefillSplit > 1 && (attentionPlan == null || !attentionPlan.contextParallel()
                || prefillSplit != peers.size() + 1)) {
            throw new IllegalArgumentException("a prefill context-parallel size of " + prefillSplit
                    + " splits every prefill's query rows over the context split's own ranks, got "
                    + (peers.size() + 1) + " rank(s) on a " + (attentionPlan == null ? "tensor"
                    : attentionPlan.contextParallel() ? "context" : "row") + " split");
        }
        this.spec = spec;
        this.ubatch = ubatch;
        this.nCtx = nCtx;
        this.archName = spec.arch().ggufName();
        this.ffnPlan = ffnPlan;
        this.attentionPlan = attentionPlan;
        this.vocabPlan = vocabPlan;
        this.moePlan = moePlan;
        if (prefillSplit > 1) {
            refusePcpModel(spec);
        }
        List<ModelSpec> rankSpecs = new ArrayList<>();
        List<MemoryPlan> memoryPlans = new ArrayList<>();
        long maxInput = 1;
        long maxOutput = 1;
        long maxGateUp = 1;
        for (int rank = 0; rank < peers.size() + 1; rank++) {
            ModelSpec rankSpec = TpShardSpec.of(spec, ffnPlan, attentionPlan, rank);
            rankSpecs.add(rankSpec);
            maxInput = Math.max(maxInput, streamWidth(rankSpec));
            maxOutput = Math.max(maxOutput, streamWidth(rankSpec));
            if (ffnPlan != null) {
                for (FfnRowSplitPlan.Layer layer : ffnPlan.layers()) {
                    maxInput = Math.max(maxInput, layer.nIn());
                    maxOutput = Math.max(maxOutput, layer.nOut());
                    maxGateUp = Math.max(maxGateUp, 2L * layer.maxShard());
                }
            }
            if (attentionPlan != null) {
                java.util.function.IntPredicate split = il -> attentionPlan.layerOf(spec.layer(il)) != null;
                MemoryPlan rankPlan = attentionPlan.contextParallel()
                        ? MemoryPlanner.planForContextShard(rankSpec, memory, nCtx, rank, peers.size() + 1, split)
                        : MemoryPlanner.planForContext(rankSpec, memory, nCtx, split);
                memoryPlans.add(rankPlan);
            }
        }
        this.allReduceTemporary = mainBackend.allocateBuffer(
                Math.min(GpuAllReduce.CHUNK_FLOATS, ubatch * maxOutput) * Float.BYTES, true);
        try {
            for (int rank = 0; rank < peers.size() + 1; rank++) {
                ModelSpec rankSpec = rankSpecs.get(rank);
                GpuScratch rankScratch = attentionPlan != null || vocabPlan != null || moePlan != null
                        ? GpuScratch.forModel(rank == 0 ? mainBackend : peers.get(rank - 1).backend(), rankSpec,
                        ubatch)
                        : rowScratch(rank, ubatch, maxGateUp);
                addRank(rank, rankSpec, rankScratch, mainKernels, mainQueue, mainPlaceholders);
                if (attentionPlan != null) {
                    ranks.get(rank).memoryPlan = memoryPlans.get(rank);
                }
                if (rank > 0) {
                    openRankSlots(ranks.get(rank), ubatch * maxInput, ubatch * maxOutput);
                }
            }
        } catch (Throwable failure) {
            try {
                close();
            } catch (IOException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
    }

    /**
     * Refuses a model a prefill query split cannot partition. Every rank runs every trunk layer whole over its own
     * DualChunkSwap packed rows, so nothing may read token ids mid-stack — the media and embedding segments (per-layer
     * and n-gram embeddings, the embedding-skip add, token-switched LoRA) — and no trunk layer may run a recurrent
     * scan or a cross-token pass whose sequence state a rank cannot cut, or routed experts the replication would
     * duplicate whole.
     *
     * @throws UnsupportedOperationException naming the first construct the split cannot partition
     */
    private void refusePcpModel(ModelSpec spec) {
        if (spec.schedule().kind() != LayerSchedule.Kind.LINEAR) {
            throw new UnsupportedOperationException("a prefill context-parallel size of " + prefillSplit
                    + " runs every trunk layer whole on every rank, which a " + spec.schedule().kind()
                    + " schedule's passes between layers cannot; keep the prefill context-parallel size at 1");
        }
        if (attentionPlan != null && !attentionPlan.skipped().isEmpty()) {
            throw new UnsupportedOperationException("a prefill context-parallel size of " + prefillSplit
                    + " runs every trunk layer on every rank, but the context split keeps whole-device attention"
                    + " on " + attentionPlan.skipped().get(0) + "; keep the prefill context-parallel size at 1");
        }
        if (spec.embedding().perLayerEmbeddings()) {
            throw new UnsupportedOperationException("a prefill context-parallel size of " + prefillSplit
                    + " cannot serve per-layer token embeddings (media and embedding segments read token ids"
                    + " mid-stack); keep the prefill context-parallel size at 1");
        }
        for (LayerSpec l : spec.layers()) {
            if (l.mixerKind() != MixerKind.ATTENTION && l.mixerKind() != MixerKind.NONE) {
                throw new UnsupportedOperationException("a prefill context-parallel size of " + prefillSplit
                        + " cannot partition layer " + l.il() + "'s " + l.mixerKind()
                        + " mixer: its scan or cross-token sequence state does not cut per rank; keep the"
                        + " prefill context-parallel size at 1");
            }
            if (l.ffnKind() == FfnKind.MOE || l.ffnKind() == FfnKind.MOE_LATENT
                    || l.ffnKind() == FfnKind.MOE_CHUNKED || l.ffnKind() == FfnKind.MOE_PARALLEL_DENSE
                    || l.ffnKind() == FfnKind.MOE_SHORTCUT) {
                throw new UnsupportedOperationException("a prefill context-parallel size of " + prefillSplit
                        + " runs every trunk layer whole on every rank, which layer " + l.il() + "'s routed"
                        + " experts (" + l.ffnKind() + ") would duplicate whole; keep the prefill"
                        + " context-parallel size at 1");
            }
            String extra = pcpExtraRefusal(l);
            if (extra != null) {
                throw new UnsupportedOperationException("a prefill context-parallel size of " + prefillSplit
                        + " cannot serve layer " + l.il() + "'s " + extra + " extra: it reads token ids"
                        + " mid-stack; keep the prefill context-parallel size at 1");
            }
        }
    }

    /** The first token-id-reading extra of a layer a prefill query split cannot serve, or null. */
    private static String pcpExtraRefusal(LayerSpec l) {
        if (l.hasExtra(LayerSpec.Extras.PLE) || l.hasExtra(LayerSpec.Extras.NGRAM_PLE)) {
            return LayerSpec.Extras.PLE;
        }
        if (l.hasExtra(LayerSpec.Extras.NGRAM_EMBD) || l.hasExtra(LayerSpec.Extras.ENGRAM)) {
            return LayerSpec.Extras.NGRAM_EMBD;
        }
        if (l.hasExtra(LayerSpec.Extras.EMBD_SKIP)) {
            return LayerSpec.Extras.EMBD_SKIP;
        }
        if (l.hasExtra(LayerSpec.Extras.SWITCHED_LORA)) {
            return LayerSpec.Extras.SWITCHED_LORA;
        }
        return null;
    }

    /** The scratch of a feed-forward-only rank: the gate-and-up rows of the widest split FFN. */
    private GpuScratch rowScratch(int rank, int ubatch, long gateUpMax) {
        ComputeBackend backend = rank == 0 ? mainBackend : peers.get(rank - 1).backend();
        return GpuScratch.builder(backend, ubatch)
                .slot(ScratchSlots.FFN_GATE_UP, Math.max(1, ubatch * gateUpMax))
                .build();
    }

    private void addRank(int index, ModelSpec rankSpec, GpuScratch scratch, LlmKernels mainKernels,
                         ComputeQueue mainQueue, GpuPlaceholders mainPlaceholders) {
        try {
            if (index == 0) {
                GpuOps ops = new GpuOps(mainQueue, mainKernels, scratch, mainPlaceholders);
                ranks.add(new Rank(index, "rank0 (the model device)", mainBackend, mainQueue, mainKernels,
                        mainPlaceholders, scratch, ops, rankSpec));
                return;
            }
            GpuDevices.Selected peer = peers.get(index - 1);
            ComputeQueue queue = peer.backend().computeQueue();
            LlmKernels kernels = LlmKernels.open(peer.backend());
            GpuPlaceholders placeholders = GpuPlaceholders.allocate(peer.backend(), queue);
            GpuOps ops = new GpuOps(queue, kernels, scratch, placeholders);
            ranks.add(new Rank(index, "rank" + index + " (" + peer.name() + ")", peer.backend(), queue, kernels,
                    placeholders, scratch, ops, rankSpec));
        } catch (Throwable failure) {
            try {
                scratch.close();
            } catch (RuntimeException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
    }

    /** The buffers a peer rank exchanges with the main device: the replayed input and its output partial. */
    private void openRankSlots(Rank rank, long inputFloats, long partialFloats) {
        GpuBuffer input = rank.backend.allocateBuffer(inputFloats * Float.BYTES, true);
        rankBuffers.add(input);
        GpuBuffer partial = rank.backend.allocateBuffer(partialFloats * Float.BYTES, true);
        rankBuffers.add(partial);
        rank.input = ScratchSlot.of("tp.input.rank" + rank.index, input, 0, inputFloats);
        rank.partial = ScratchSlot.of("tp.partial.rank" + rank.index, partial, 0, partialFloats);
    }

    /** The widest residual stream of a spec. */
    private static long streamWidth(ModelSpec spec) {
        return Math.max(1, spec.maxStreamWidth());
    }

    /**
     * Runs the per-rank fit check, allocates every rank's shard KV cache, uploads every rank's shards and binds each
     * rank's forward context over them. Nothing is uploaded or allocated for a rank that does not fit.
     *
     * @throws IllegalArgumentException when the model has no data for a split tensor
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when a rank's shards or cache do not fit
     */
    public void upload(WeightStore store, GpuWeightUploader mainUploader) {
        requireInit();
        if (ffnUploader != null || attentionUploader != null) {
            throw new IllegalStateException("the tensor-parallel group is already uploaded");
        }
        Objects.requireNonNull(store, "store");
        Objects.requireNonNull(mainUploader, "mainUploader");
        if (attentionPlan != null) {
            if (pcpEnabled()) {
                this.pcpReplicator = new PcpWeightReplicator(attentionPlan, mainUploader, ranks.get(0).queue);
            }
            checkFit(mainUploader);
            for (Rank rank : ranks) {
                rank.memory = DeviceModelMemory.allocate(rank.backend, rank.queue, rank.memoryPlan,
                        spec.hparams().mropeTextPositions(), null,
                        il -> attentionPlan.layerOf(spec.layer(il)) != null);
            }
        }
        ffnUploader = ffnPlan == null ? null : new FfnRowSplitUploader(ffnPlan, mainUploader, store);
        attentionUploader = attentionPlan == null ? null : new TpAttentionUploader(attentionPlan, mainUploader, store);
        vocabUploader = vocabPlan == null ? null : new TpVocabUploader(vocabPlan, mainUploader, store);
        moeUploader = moePlan == null ? null : new TpMoeUploader(moePlan, mainUploader, store);
        GpuWeight[] vocabTables = vocabPlan == null ? null : new GpuWeight[ranks.size()];
        GpuWeight[] vocabHeads = vocabPlan == null ? null : new GpuWeight[ranks.size()];
        GpuWeight[] vocabBiases = vocabPlan == null ? null : new GpuWeight[ranks.size()];
        for (Rank rank : ranks) {
            Map<String, GpuWeight> shard = new LinkedHashMap<>();
            if (ffnUploader != null) {
                shard.putAll(ffnUploader.uploadRank(rank.index, rank.backend, rank.queue));
            }
            if (attentionUploader != null) {
                shard.putAll(attentionUploader.uploadRank(rank.index, rank.backend, rank.queue));
            }
            Map<Integer, MoeExpertRemap.Layer> remap = null;
            if (moeUploader != null) {
                shard.putAll(moeUploader.uploadRank(rank.index, rank.backend, rank.queue, rank.index > 0));
                remap = moeUploader.uploadRemap(rank.index, rank.backend, rank.queue);
            }
            if (vocabUploader != null) {
                Map<String, GpuWeight> vocabShard = vocabUploader.uploadRank(rank.index, rank.backend, rank.queue);
                vocabTables[rank.index] = vocabShard.get(vocabPlan.tokenTensor());
                vocabHeads[rank.index] = vocabShard.get(vocabPlan.headTensor());
                vocabBiases[rank.index] = vocabPlan.headBias() == null ? null
                        : vocabShard.get(vocabPlan.headBias());
                if (rank.index == 0) {
                    for (Map.Entry<String, GpuWeight> entry : vocabShard.entrySet()) {
                        mainUploader.adopt(entry.getKey(), entry.getValue());
                    }
                } else {
                    shard.putAll(vocabShard);
                }
            }
            if (pcpReplicator != null && rank.index > 0) {
                // the DualChunkSwap prefill runs every trunk layer whole on the peer: it needs the whole
                // blk.* tensors the plans leave unsplit, resident on the peer's own device
                pcpReplicator.replicate(rank.index, rank.backend, rank.queue, shard);
            }
            rank.weights = weightsOf(archName, shard, rank.index == 0 ? mainUploader : null);
            GpuForwardContext.Builder b = GpuForwardContext.builder(rank.kernels, rank.ops, rank.weights, rank.spec);
            if (rank.memory != null) {
                b.memory(rank.memory);
            }
            if (rank.index == 0) {
                b.heat(expertHeat);
                b.moeLoadWindow(moeLoadWindow);
            }
            if (remap != null && !remap.isEmpty()) {
                b.moeExpertRemap(MoeExpertRemap.of(remap));
            }
            rank.context = b.build();
            if (rank.index > 0) {
                WeightKernelPrewarm.prewarm(mainUploader.plans(this::splits), rank.ops, rank.kernels);
            }
        }
        if (vocabUploader != null) {
            this.vocab = new TpVocab(vocabPlan, this, vocabTables, vocabHeads, vocabBiases, ubatch);
        }
    }

    private boolean splits(GGUFTensorInfo tensor) {
        return (ffnPlan != null && ffnPlan.splits(tensor.name()))
                || (attentionPlan != null && attentionPlan.splits(tensor.name()))
                || (vocabPlan != null && vocabPlan.splits(tensor.name()))
                || (moePlan != null && moePlan.splits(tensor.name()));
    }

    /**
     * A read-only view of one rank's shard weights under the model's tensor names: rank 0 falls back to the model
     * device's resident weights for the tensors the split leaves whole (a MoE router, a scale), which live on the same
     * backend.
     */
    private static GpuWeights weightsOf(String archName, Map<String, GpuWeight> shard, GpuWeights fallback) {
        return new GpuWeights() {
            @Override
            public GpuWeight find(String name) {
                GpuWeight w = shard.get(name);
                return w != null ? w : fallback == null ? null : fallback.find(name);
            }

            @Override
            public String archName() {
                return archName;
            }
        };
    }

    /** The FFN decorator the model registry takes: split layers bind a {@link FfnRowSplitBlock} or a MoE block. */
    public GpuBlockRegistry.FfnDecorator decorator() {
        requireInit();
        FfnRowSplitPlan ffn = ffnPlan;
        TpMoePlan moe = moePlan;
        return (layer, deviceBlock) -> {
            if (moe != null) {
                TpMoePlan.Layer split = moe.layerOf(layer);
                if (split != null) {
                    return new TpMoeBlock(this, layer, split);
                }
            }
            FfnRowSplitPlan.Layer split = ffn == null ? null : ffn.layerOf(layer);
            return split == null ? deviceBlock : new FfnRowSplitBlock(this, layer, split);
        };
    }

    /** The mixer decorator a tensor split installs: split attention binds a {@link TpAttentionMixer}. */
    public GpuBlockRegistry.MixerDecorator attentionDecorator() {
        requireInit();
        TpAttentionPlan plan = attentionPlan;
        return (layer, deviceBlock) -> {
            TpAttentionPlan.Layer split = plan == null ? null : plan.layerOf(layer);
            if (split == null) {
                return deviceBlock;
            }
            if (plan.contextParallel()) {
                CpAttentionMixer mixer = pcpEnabled()
                        ? new PcpAttentionMixer(this, layer)
                        : new CpAttentionMixer(this, layer);
                pcpMixes.put(layer.il(), mixer);
                return mixer;
            }
            return new TpAttentionMixer(this, layer, split);
        };
    }

    /** Number of ranks the split covers (the model device and its peers). */
    public int rankCount() {
        requireInit();
        return ranks.size();
    }

    /** Number of peer devices the selection opened (rank 0 is the model device; valid before {@link #init}). */
    public int peerCount() {
        return peers.size();
    }

    /** The feed-forward plan, or null for an attention-only group. */
    public FfnRowSplitPlan plan() {
        requireInit();
        return ffnPlan;
    }

    /** The attention plan, or null in a feed-forward-only (row) split. */
    public TpAttentionPlan attentionPlan() {
        requireInit();
        return attentionPlan;
    }

    /** The model spec the rank contexts bind against. */
    public ModelSpec spec() {
        requireInit();
        return spec;
    }

    GpuForwardContext rankContext(int rank) {
        return ranks.get(rank).context;
    }

    /** The queue of a rank's device (rank 0's is the model's own queue). */
    ComputeQueue rankQueue(int rank) {
        return ranks.get(rank).queue;
    }

    /** The backend of a rank's device (rank 0's is the model's own backend). */
    ComputeBackend rankBackend(int rank) {
        return ranks.get(rank).backend;
    }

    /** The dispatch helpers of a rank's device. */
    GpuOps rankOps(int rank) {
        return ranks.get(rank).ops;
    }

    /** The scratch of a rank's device. */
    GpuScratch rankScratch(int rank) {
        return ranks.get(rank).scratch;
    }

    /** The prefill context-parallel attention mixer a layer's model executor bound. */
    CpAttentionMixer pcpAttention(LayerSpec layer) {
        CpAttentionMixer mixer = pcpMixes.get(layer.il());
        if (mixer == null) {
            throw new IllegalStateException("layer " + layer.il() + " binds no prefill context-parallel"
                    + " attention mixer");
        }
        return mixer;
    }

    private void refusePcpAttentionLayersAndRelease() {
        try {
            refusePcpAttentionLayers();
        } catch (UnsupportedOperationException refusal) {
            try {
                close();
            } catch (IOException cleanupFailure) {
                refusal.addSuppressed(cleanupFailure);
            }
            throw refusal;
        }
    }

    private void refusePcpAttentionLayers() {
        for (Map.Entry<Integer, CpAttentionMixer> entry : pcpMixes.entrySet()) {
            CpAttentionMixer mixer = entry.getValue();
            for (int rank = 0; rank < ranks.size(); rank++) {
                String reason = mixer.part(rank).pcpRefusal();
                if (reason != null) {
                    throw new UnsupportedOperationException("layer " + entry.getKey() + " cannot run with "
                            + EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL + "=" + prefillSplit + ": " + reason);
                }
            }
        }
    }

    /**
     * The DualChunkSwap prefill trunk runner the model installs on its prefill driver for the contiguous cache:
     * rank 0's executor drives the phases, the coordinator interlaces the ranks layer by layer. Built once, after
     * {@link #upload} (the rank contexts it builds peer trunks over must exist) and after the model executor bound
     * this decorator's mixers.
     *
     * @throws IllegalStateException when the split runs no contiguous prefill query split, the runner is already
     *                               built, or no layer bound a mixer yet
     */
    public GpuPrefillDriver.TrunkRunner pcpTrunkRunner(GpuForwardContext modelContext, GpuLayerExecutor mainTrunk) {
        requireInit();
        if (!prefillSplitEnabled()) {
            throw new IllegalStateException("the split does not run a contiguous prefill query split"
                    + " (neuralfree.prefill_context_parallel_size over 1 on split.mode=context over a contiguous"
                    + " cache)");
        }
        if (pcpCoordinator != null) {
            throw new IllegalStateException("the prefill context-parallel coordinator is already built");
        }
        refusePcpAttentionLayersAndRelease();
        PcpPrefillCoordinator coordinator = new PcpPrefillCoordinator(this, modelContext, mainTrunk);
        this.pcpCoordinator = coordinator;
        return coordinator;
    }

    /**
     * The DualChunkSwap prefill batched-trunk runner the model installs on its prefill driver over a paged cache:
     * the coordinator carries the whole batched step (its segments' chunk pairs over the ranks' translated page
     * tables) while the plain executor keeps the decode steps. Built once, after {@link #upload} and after the
     * model executor bound this decorator's mixers.
     *
     * @throws IllegalStateException when the split runs no prefill query split, the runner is already built, or no
     *                               layer bound a mixer yet
     */
    public GpuPrefillDriver.BatchTrunkRunner pcpBatchTrunkRunner(GpuForwardContext modelContext,
                                                                GpuLayerExecutor mainTrunk) {
        requireInit();
        if (!pcpEnabled()) {
            throw new IllegalStateException("the split does not run a prefill query split"
                    + " (neuralfree.prefill_context_parallel_size over 1 on split.mode=context)");
        }
        if (pcpCoordinator != null) {
            throw new IllegalStateException("the prefill context-parallel coordinator is already built");
        }
        refusePcpAttentionLayersAndRelease();
        PcpPrefillCoordinator coordinator = new PcpPrefillCoordinator(this, modelContext, mainTrunk);
        this.pcpCoordinator = coordinator;
        return coordinator;
    }

    /** The built DualChunkSwap prefill coordinator. */
    PcpPrefillCoordinator pcpCoordinator() {
        if (pcpCoordinator == null) {
            throw new IllegalStateException("the prefill context-parallel coordinator is not built");
        }
        return pcpCoordinator;
    }

    /** The vocabulary-parallel embedding lookup, or null when the vocabulary is not split. */
    public GpuEmbeddingStage.VocabParallel vocabEmbedding() {
        requireInit();
        return vocab;
    }

    /** The vocabulary-parallel LM head, or null when the vocabulary is not split. */
    public GpuLogitsHead.VocabSplit vocabHead() {
        requireInit();
        return vocab;
    }

    /** The replayed input slot of a peer rank (rank 0 reads the model's input slot directly). */
    ScratchSlot inputSlot(int rank) {
        return ranks.get(rank).input;
    }

    /** The partial output slot of a peer rank. */
    ScratchSlot partialSlot(int rank) {
        return ranks.get(rank).partial;
    }

    /** The collective temporary on the model device, holding at least one all-reduce chunk. */
    GpuBuffer allReduceTemporary() {
        return allReduceTemporary;
    }

    // ------------------------------------------------------------------ memory mirroring
    // The mirroring itself is TpMemoryMirror; these members keep its operations on the group's API.

    /** Uploads the current step's positions to every rank's step inputs ({@link TpMemoryMirror#uploadStepInputs}). */
    void uploadStepInputs(int pos0, int nTokens) {
        mirror.uploadStepInputs(pos0, nTokens);
    }

    /** Records a step's tokens on every rank's sequence record ({@link TpMemoryMirror#appendStep}). */
    void appendStep(int[] tokens, int off, int n) {
        mirror.appendStep(tokens, off, n);
    }

    /** Records a step of embedding rows, carrying {@code token}, on every rank's sequence record. */
    void appendEmbeddingStep(int n, int token) {
        mirror.appendEmbeddingStep(n, token);
    }

    /** True when every rank can drop the last {@code nTokens} positions. */
    boolean canRollback(int nTokens) {
        return mirror.canRollback(nTokens);
    }

    /** Empties every rank's cache ({@link TpMemoryMirror#reset}). */
    void reset() {
        mirror.reset();
    }

    /** Drops the last {@code nTokens} positions from every rank's cache. */
    void rollback(int nTokens) {
        mirror.rollback(nTokens);
    }

    /** Uploads a paged step to every rank cache ({@link TpMemoryMirror#beginPagedStep}). */
    void beginPagedStep(PagedStep step, int[] positions, int[][] ropeSections, boolean nonCausal,
                        PagedSequences mainPaged, int[][] mainMirror) {
        mirror.beginPagedStep(step, positions, ropeSections, nonCausal, mainPaged, mainMirror);
    }

    /** Clears the paged step from every rank cache. */
    void endPagedStep() {
        mirror.endPagedStep();
    }

    /** Runs one page copy of the copy-on-write protocol on every rank cache. */
    void copyPage(PageCopy copy, PagedSequences mainPaged) {
        mirror.copyPage(copy, mainPaged);
    }

    /** Why a per-sequence context shift over the rank caches would refuse, or null. */
    String sequenceShiftRefusal() {
        return mirror.sequenceShiftRefusal();
    }

    /** Applies a shift the model device planned to every rank cache ({@link TpMemoryMirror#applySequenceShift}). */
    void applySequenceShift(int[][] deltas, List<com.eonmux.neuralfree.gguf.memory.paged.RowMove> moves,
                            List<com.eonmux.neuralfree.gguf.memory.paged.PageCopy> copies, int[][] posMirror) {
        mirror.applySequenceShift(deltas, moves, copies, posMirror);
    }

    /** Completes the rank shifts after the mixers re-rotated the moved keys. */
    void moveShiftedRows(List<com.eonmux.neuralfree.gguf.memory.paged.RowMove> moves) {
        mirror.moveShiftedRows(moves);
    }

    /**
     * Refuses a paged main memory whose step geometry the rank caches cannot address, and builds the context
     * split's page shard ({@link TpMemoryMirror#verifyPagedMemory}).
     *
     * @throws UnsupportedOperationException naming the rank and the differing geometry
     */
    public void verifyPagedMemory(ModelMemory main) {
        requireInit();
        mirror.verifyPagedMemory(main);
    }

    /** Why a contiguous context shift over the rank caches would refuse, or null. */
    String contextShiftRefusal() {
        return mirror.contextShiftRefusal();
    }

    /**
     * Runs the contiguous context shift (llama.cpp K-shift) on every rank cache
     * ({@link TpMemoryMirror#shiftPositions}).
     *
     * @throws UnsupportedOperationException when a rank cache cannot shift
     * @throws IllegalArgumentException      for a range beyond the stored positions
     */
    void shiftPositions(int nKeep, int nDiscard) {
        mirror.shiftPositions(nKeep, nDiscard);
    }

    /** The per-rank memories, in rank order (unmodifiable). */
    List<DeviceModelMemory> rankMemories() {
        List<DeviceModelMemory> out = new ArrayList<>();
        for (Rank rank : ranks) {
            if (rank.memory != null) {
                out.add(rank.memory);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Writes the session of the main memory and of every rank's shard caches to {@code out}: the container holds the
     * main sequence first, then rank 0's shard memory and every peer's, each in the {@link SessionFile} format. A
     * load restores every rank, so the next decode continues the saved sequence with the same rows on every device.
     * A context-parallel paged split refuses by name: the rank page images follow the main page tables at run time
     * and have no session form of their own.
     */
    public void saveSession(ModelMemory main, Path out) throws IOException {
        requireInit();
        mirror.saveSession(main, out);
    }

    /**
     * Reads a tensor-parallel session written by {@link #saveSession} into the main memory and every rank's shard
     * caches, resetting them first.
     */
    public void loadSession(ModelMemory main, Path in) throws IOException {
        requireInit();
        mirror.loadSession(main, in);
    }


    // ------------------------------------------------------------------ fit check

    /**
     * Refuses a rank whose shard weights and KV cache do not fit its device, before any shard is uploaded or any
     * rank cache allocated. The rank scratch, exchange slots and collective temporary are already allocated.
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException naming the rank and the shortfall in MiB
     */
    private void checkFit(GpuWeightUploader uploader) {
        for (Rank rank : ranks) {
            checkFit(rank.index, rank.memoryPlan, uploader);
        }
    }

    private void checkFit(int rank, MemoryPlan rankPlan, GpuWeightUploader uploader) {
        long weights = plannedShardBytes(rank, uploader);
        if (pcpReplicator != null && rank > 0) {
            weights += pcpReplicator.replicatedBytes();
        }
        // the rank scratch, exchange slots and the collective temporary are allocated by init, so the free figure
        // already excludes them; only the shard weights and the rank's KV cache are still to come
        long need = weights + rankPlan.totalBytes();
        ComputeBackend backend = rank == 0 ? mainBackend : peers.get(rank - 1).backend();
        long free = VRAMBudget.queryFreeVram(backend).bytes();
        if (free <= 0) {
            return;
        }
        long available = free - 64L * 1024 * 1024;
        if (need > available) {
            throw new com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException(rankName(rank) + " needs about "
                    + mib(need) + " MiB for its tensor-parallel shard (" + mib(weights) + " MiB of weights and "
                    + mib(rankPlan.totalBytes()) + " MiB of KV cache), but the"
                    + " device has " + mib(Math.max(0, available)) + " MiB usable: the shard is short by "
                    + mib(need - available) + " MiB; use fewer devices, a smaller context or"
                    + " neuralfree.gpu.vram.limit");
        }
    }

    String rankName(int rank) {
        return rank == 0 ? "rank0 (the model device)" : "rank" + rank + " (" + peers.get(rank - 1).name() + ")";
    }

    /** Device bytes one rank's shard weights take, planned from the uploader's tensor plans. */
    private long plannedShardBytes(int rank, GpuWeightUploader uploader) {
        long bytes = 0;
        if (ffnPlan != null) {
            bytes += ShardBytes.ffn(ffnPlan, rank, uploader);
        }
        if (attentionPlan != null) {
            bytes += ShardBytes.attention(attentionPlan, rank, uploader);
        }
        if (vocabPlan != null) {
            bytes += vocabPlan.tokenBytes(rank) + vocabPlan.headBytes(rank) + vocabPlan.headBiasBytes(rank);
        }
        if (moePlan != null) {
            bytes += ShardBytes.moe(moePlan, rank, uploader);
            if (rank > 0) {
                // rank 0 reads the whole MoE support tensors from the model device's resident weights
                bytes += ShardBytes.moeSupport(moePlan, rank, uploader);
            }
        }
        return bytes;
    }

    private static long mib(long bytes) {
        return (bytes + 1024L * 1024 - 1) / (1024L * 1024);
    }

    // ------------------------------------------------------------------ lifecycle

    @Override
    public void close() throws IOException {
        ffnPlan = null;
        attentionPlan = null;
        mirror.discard();
        Throwable first = null;
        first = closeStep(pcpCoordinator, first);
        pcpCoordinator = null;
        first = closeStep(pcpReplicator, first);
        pcpReplicator = null;
        pcpMixes.clear();
        first = closeStep(ffnUploader, first);
        ffnUploader = null;
        first = closeStep(attentionUploader, first);
        attentionUploader = null;
        first = closeStep(vocabUploader, first);
        vocabUploader = null;
        first = closeStep(moeUploader, first);
        moeUploader = null;
        moePlan = null;
        expertHeat = null;
        moeLoadWindow = null;
        if (vocab != null) {
            first = closeStep(vocab::close, first);
            vocab = null;
        }
        vocabPlan = null;
        first = closeStep(allReduceTemporary, first);
        allReduceTemporary = null;
        List<GpuBuffer> buffers = new ArrayList<>(rankBuffers);
        rankBuffers.clear();
        for (GpuBuffer b : buffers) {
            first = closeStep(b, first);
        }
        List<Rank> closed = new ArrayList<>(ranks);
        ranks.clear();
        for (Rank rank : closed) {
            first = closeStep(rank.ops, first);
            first = closeStep(rank.scratch, first);
            first = closeStep(rank.memory, first);
            if (rank.index > 0) {
                first = closeStep(rank.placeholders, first);
            }
        }
        if (ownedAcceleration != null) {
            try {
                ownedAcceleration.shutdown();
            } catch (Throwable t) {
                if (first == null) {
                    first = t;
                } else if (t != first) {
                    first.addSuppressed(t);
                }
            }
        }
        rethrow(first);
    }

    private void requireInit() {
        if (ffnPlan == null && attentionPlan == null && vocabPlan == null && moePlan == null) {
            throw new IllegalStateException("the tensor-parallel group is not initialized");
        }
    }

    private static Throwable closeStep(AutoCloseable c, Throwable first) {
        if (c == null) {
            return first;
        }
        try {
            c.close();
            return first;
        } catch (Throwable t) {
            if (first == null) {
                return t;
            }
            if (t != first) {
                first.addSuppressed(t);
            }
            return first;
        }
    }

    private static void rethrow(Throwable first) throws IOException {
        if (first instanceof RuntimeException) {
            throw (RuntimeException) first;
        }
        if (first instanceof Error) {
            throw (Error) first;
        }
        if (first != null) {
            throw new IOException("closing the tensor-parallel group failed", first);
        }
    }

    @Override
    public String toString() {
        return "TensorParallelGroup{" + ranks.size() + " ranks, " + (ffnPlan == null ? 0 : ffnPlan.layers().size())
                + " FFN / " + (attentionPlan == null ? 0 : attentionPlan.layers().size()) + " attention layer(s)}";
    }
}
