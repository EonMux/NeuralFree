package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.VRAMBudget;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.gpu.diag.ObserverSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertPrefetch;
import com.eonmux.neuralfree.gguf.gpu.weights.PlacementFitter;
import com.eonmux.neuralfree.gguf.gpu.weights.StreamingLayerWeightSource;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The device resources of one stage of a layer split, acquired in load order by {@link GpuSplitSession}: the LLM
 * kernels, the free VRAM measured before the stage allocated anything, the placeholders and scratch, the weight
 * store and uploader of the stage's tensors, the dispatch helpers, the memory of the span's layers and, when the
 * stage's {@link PlacementFitter} fit streams part of its span, the streaming source over a store of its own, and,
 * when the routed experts of its layers stay in RAM, the MoE expert cache of those layers. Each stage owns an
 * {@link ObserverSlot} its forward context reports to. {@link #close()} releases everything in reverse order; the
 * backend itself belongs to the acceleration of the session. Mutable during assembly, not thread-safe.
 */
final class SplitStage implements AutoCloseable {

    private final int index;
    private final GpuDevices.Selected device;
    private final ComputeQueue queue;
    private final LlmKernels kernels;
    private final VRAMBudget.FreeVram freeAtLoad;
    private final long allocatedAtLoad;
    private final long limitBytes;
    private final ObserverSlot observer = new ObserverSlot();
    private GpuPlaceholders placeholders;
    private GpuScratch scratch;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private GpuOps ops;
    private DeviceModelMemory memory;
    private WeightStore streamingStore;
    private StreamingLayerWeightSource streaming;
    private List<LayerWeightSet> streamed = Collections.emptyList();
    private PlacementFitter.Fit fit;
    private long reservedBytes;
    private MoeExpertCache moeCache;
    private MoeExpertPrefetch moePrefetch;

    private SplitStage(int index, GpuDevices.Selected device, LlmKernels kernels, long limitBytes) {
        this.index = index;
        this.device = device;
        this.queue = device.backend().computeQueue();
        this.kernels = kernels;
        this.limitBytes = limitBytes;
        this.freeAtLoad = VRAMBudget.queryFreeVram(device.backend());
        this.allocatedAtLoad = device.backend().getAllocatedBytes();
    }

    /**
     * Opens stage {@code index} on {@code device}: kernels, placeholders, scratch of {@code ubatch} tokens, the weight
     * store and the uploader. A failure releases what the stage acquired.
     *
     * @param limitBytes the cap on the device bytes the stage uses ({@code neuralfree.gpu.vram.limit}), 0 for none
     * @param w12        whether the stage streams the target's BF16/F16 matmul weights as lossless W12
     *                   ({@code neuralfree.weights.w12})
     * @param w12Head    whether the W12 streaming includes the output head ({@code neuralfree.weights.w12.head})
     * @param w12F16     how the W12 streaming holds an F16 tensor to the bfloat16 grid
     *                   ({@code neuralfree.weights.w12.f16})
     * @throws IllegalStateException when the device cannot run the LLM kernels
     * @throws IOException           when the model store cannot be mapped
     */
    static SplitStage open(int index, GpuDevices.Selected device, ModelSource source,
                           InputTransformLoader.Result transforms, ModelSpec initial, int ubatch, long limitBytes,
                           boolean w12, boolean w12Head, W12F16Mode w12F16)
            throws IOException {
        LlmKernels kernels;
        try {
            kernels = LlmKernels.open(device.backend());
        } catch (RuntimeException e) {
            throw new IllegalStateException("device " + device.name() + " cannot run the LLM kernels: "
                    + e.getMessage(), e);
        }
        SplitStage stage = new SplitStage(index, device, kernels, limitBytes);
        try {
            ComputeBackend backend = device.backend();
            stage.placeholders = GpuPlaceholders.allocate(backend, stage.queue);
            stage.scratch = GpuScratch.forModel(backend, initial, ubatch);
            stage.store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD);
            stage.uploader = new GpuWeightUploader(backend, stage.queue, source.getTensors(), stage.store,
                    initial.arch().ggufName(),
                    GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels))
                            .transforms(transforms).w12(w12).w12Head(w12Head).w12F16(w12F16).build());
            return stage;
        } catch (IOException | RuntimeException e) {
            try {
                stage.close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    int index() { return index; }
    String name() { return device.name(); }
    ComputeBackend backend() { return device.backend(); }

    /**
     * Closes the device's connection when the stage runs on a remote device; a local device's backend belongs to the
     * acceleration of the session and is not touched. Idempotent.
     */
    void closeDevice() {
        device.close();
    }
    ComputeQueue queue() { return queue; }
    LlmKernels kernels() { return kernels; }
    GpuWeightUploader uploader() { return uploader; }
    WeightStore store() { return store; }
    GpuOps ops() { return ops; }
    DeviceModelMemory memory() { return memory; }
    ObserverSlot observer() { return observer; }
    /** The streaming source of the stage, or null when its span is resident. */
    StreamingLayerWeightSource streaming() { return streaming; }
    /** The streamed sets of the stage, empty when its span is resident (unmodifiable). */
    List<LayerWeightSet> streamed() { return streamed; }
    /** The placement fit of the stage, or null for a model that keeps every weight resident. */
    PlacementFitter.Fit fit() { return fit; }
    /** The MoE expert cache of the stage's layers, or null. */
    MoeExpertCache moeCache() { return moeCache; }
    /** The prefill expert prefetch over the stage's cache, or null. */
    MoeExpertPrefetch moePrefetch() { return moePrefetch; }

    /**
     * Device bytes left for this stage: the free VRAM measured before the stage allocated anything, less everything
     * its backend allocated since, the bytes its context fit reserved for later ({@link #reserve}) and
     * {@code pending}, the allocations still to come (the adapter uploads), capped by
     * {@code neuralfree.gpu.vram.limit} ({@link VRAMBudget#limitedBytes}).
     */
    long usableBytes(long pending) {
        ComputeBackend backend = device.backend();
        long allocatedSince = Math.max(0L, backend.getAllocatedBytes() - allocatedAtLoad);
        long held = Math.addExact(Math.max(0L, pending), reservedBytes);
        long budget = VRAMBudget.memoryBudgetBytes(backend.capabilities().getTotalVramBytes(), freeAtLoad, 0L,
                allocatedSince, held);
        return VRAMBudget.limitedBytes(budget, limitBytes, allocatedSince, held);
    }

    /**
     * Holds {@code bytes} the context fit set aside beside the stage's caches for memory allocated later (the NextN
     * draft caches, a drafter) out of every later budget of the stage (streaming slots, the MoE expert cache).
     */
    void reserve(long bytes) {
        if (bytes < 0) {
            throw new IllegalArgumentException("reserved bytes must not be negative, got " + bytes);
        }
        this.reservedBytes = bytes;
    }

    /** Releases the reservation once the memory it held for is allocated. */
    void releaseReserve() {
        this.reservedBytes = 0L;
    }

    /** Keeps the MoE expert cache of the stage's layers and its prefetch pool (either may be null). */
    void moeCache(MoeExpertCache cache, MoeExpertPrefetch prefetch) {
        this.moeCache = cache;
        this.moePrefetch = prefetch;
    }

    /** Upload planning of the stage's streamed sets: its uploader's device types, shard limits and transforms. */
    LayerWeightSet.Options uploadOptions(InputTransformLoader.Result transforms) {
        return LayerWeightSet.Options.builder(t -> uploader.deviceType(t.name()))
                .maxShardBytes(LayerWeightSet.Options.maxShardBytes(device.backend().capabilities()))
                .transforms(transforms)
                .build();
    }

    /** Keeps the placement fit the stage loads with; its streamed sets stream from the host. */
    void placed(PlacementFitter.Fit chosen) {
        this.fit = Objects.requireNonNull(chosen, "fit");
        this.streamed = chosen.streamedSets();
    }

    /** Drops the placement fit of a layout the split did not keep: the span is resident until placed again. */
    void unplace() {
        this.fit = null;
        this.streamed = Collections.emptyList();
    }

    /** Replaces the scratch with one of the planned ubatch when the plan lowered it. */
    void fitScratch(ModelSpec spec, MemoryPlan plan) {
        if (plan.ubatch() < scratch.ubatch()) {
            GpuScratch larger = scratch;
            scratch = null;
            larger.close();
            scratch = GpuScratch.forModel(device.backend(), spec, plan);
        }
        ops = new GpuOps(queue, kernels, scratch, placeholders);
    }

    /** Allocates the device memory of the span's layers, as {@code shardPlan} plans them. */
    void allocateMemory(MemoryPlan shardPlan, MropeTextPositions mropeTextPositions) {
        memory = DeviceModelMemory.allocate(device.backend(), queue, shardPlan, mropeTextPositions, null);
    }

    /** Uploads the tensors {@code owned} accepts, but the streamed sets, as resident weights. */
    void uploadResident(Predicate<GGUFTensorInfo> owned) {
        Set<String> streamedNames = new HashSet<>(LayerWeightSet.allNames(streamed));
        uploader.uploadResident(t -> !owned.test(t) || streamedNames.contains(t.name()));
    }

    /**
     * Opens the streaming source over the streamed sets, its slots fitted to {@code budget} device bytes and its
     * host tier to {@code ramCacheBytes}; nothing when the span is resident.
     *
     * @throws IOException when the streaming store cannot be mapped
     */
    void openStreaming(ModelSource source, ModelSpec forwardSpec, EngineConfig cfg, long budget, long ramCacheBytes)
            throws IOException {
        if (streamed.isEmpty()) {
            return;
        }
        streamingStore = source.createMappedStore(StoreUse.STREAMING);
        streaming = StreamingLayerWeightSource.open(device.backend(), queue, uploader, streamingStore,
                source.getTensors(), forwardSpec, streamed, cfg, budget, ramCacheBytes);
    }

    /** Device bytes of the streamed sets read from the host per pass, 0 for a resident span. */
    long streamedBytes() {
        long sum = 0L;
        for (LayerWeightSet s : streamed) {
            sum += s.uploadBytes();
        }
        return sum;
    }

    /**
     * Releases the streaming source and its store, the memory, the dispatch helpers, scratch, placeholders, the
     * uploaded weights and their store. Idempotent.
     */
    @Override
    public void close() {
        RuntimeException first = null;
        first = Closeables.close(moePrefetch, first);
        moePrefetch = null;
        first = Closeables.close(moeCache, first);
        moeCache = null;
        first = Closeables.close(streaming, first);
        streaming = null;
        first = Closeables.close(streamingStore, first);
        streamingStore = null;
        first = Closeables.close(memory, first);
        memory = null;
        first = Closeables.close(ops, first);
        ops = null;
        first = Closeables.close(scratch, first);
        scratch = null;
        first = Closeables.close(placeholders, first);
        placeholders = null;
        first = Closeables.close(uploader, first);
        uploader = null;
        first = Closeables.close(store, first);
        store = null;
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "SplitStage{" + index + ", " + device.name() + (streaming != null ? ", " + streaming : "") + '}';
    }
}
