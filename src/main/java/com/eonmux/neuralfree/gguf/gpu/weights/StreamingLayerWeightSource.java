package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * {@link LayerWeightSource} over streamed layers: the trunk's per-layer tensors ({@link LayerWeightSet}) live in
 * {@code depth} device slots ({@link LayerSlots}), everything else in the resident weights.
 *
 * <p><b>Order.</b> The execution sequence is the weight layer of every trunk layer of the {@link ModelSpec}, repeated
 * cyclically (loops and hierarchical schedules revisit weight layers). Streamed layers take slots in order of first
 * use ({@code slot = rank % depth}), so consecutive layers use different slots whenever {@code depth >= 2}.</p>
 *
 * <p><b>Event chain.</b> {@link #await(int) await(il)} first retires the previously awaited layer (a compute marker
 * after its last dispatch guards its slot), stages {@code il} if it is not resident, makes the compute queue wait for
 * its uploads, and then stages ahead: the next {@code depth - 1} streamed layers of the sequence, each waiting on the
 * compute marker of the layer that last used its slot. With {@code depth = 2} the upload of layer {@code il + 1}
 * therefore runs on the transfer queue while layer {@code il} computes: it waits only for layer {@code il - 1}, the
 * slot's previous user. A stage never evicts the layer being computed or a layer needed sooner than the one it
 * loads. {@link #prefetch(int)} stages a layer early when its slot is free of the layer being computed, and advances
 * the host RAM tier.</p>
 *
 * <p>Blocks may bind to {@link #weightsFor(int)} once: every layer keeps its device tensors in the same slot buffers
 * for the life of the source. Not thread-safe.</p>
 */
public final class StreamingLayerWeightSource implements LayerWeightSource {

    private static final Logger LOGGER = LogManager.getLogger(StreamingLayerWeightSource.class);

    private final GpuWeights resident;
    private final LayerSlots slots;
    private final WeightDataProvider data;
    private final int depth;
    private final int[] setOfLayer;
    private final int[] layerOfSet;
    private final int[] sequence;
    private final GpuWeights[] views;
    private final boolean[] seen;
    private int current = -1;
    private int cursor = -1;
    private boolean closed;

    private StreamingLayerWeightSource(GpuWeights resident, LayerSlots slots, WeightDataProvider data, int depth,
                                       Index index) {
        this.resident = resident;
        this.slots = slots;
        this.data = data;
        this.depth = depth;
        this.setOfLayer = index.setOfLayer;
        this.layerOfSet = index.layerOfSet;
        this.sequence = index.sequence;
        this.seen = new boolean[setOfLayer.length];
        this.views = new GpuWeights[layerOfSet.length];
        for (int i = 0; i < views.length; i++) {
            views[i] = new LayerView(slots.weights(i), resident);
        }
    }

    /**
     * Opens a streaming source configured by an {@link EngineConfig}: the depth comes from
     * {@link #resolveDepth} ({@code neuralfree.vram.depth}, else the budget), the host tier caches
     * {@link WeightDataProvider#ramCacheBytes} ({@code neuralfree.ram.cache.mb}) and prefetches beyond that depth.
     *
     * @param store                tensor data of the model file (not owned)
     * @param tensors              all tensor infos of the file
     * @param streamingBudgetBytes device bytes available for the slots
     * @throws IllegalArgumentException as {@link #open(ComputeBackend, ComputeQueue, GpuWeights, WeightDataProvider,
     *                                  ModelSpec, List, int)}
     * @throws GpuOutOfMemoryException  when not even one slot fits the budget
     */
    public static StreamingLayerWeightSource open(ComputeBackend backend, ComputeQueue computeQueue,
                                                  GpuWeights resident, WeightStore store,
                                                  List<GGUFTensorInfo> tensors, ModelSpec spec,
                                                  List<LayerWeightSet> sets, EngineConfig config,
                                                  long streamingBudgetBytes) {
        int depth = resolveDepth(config, streamingBudgetBytes, spec, sets);
        int nWeightLayers = Index.of(spec, sets).setOfLayer.length;
        WeightDataProvider data = new WeightDataProvider(store, tensors,
                LayerWeightSet.namesByLayer(sets, nWeightLayers), config);
        return open(backend, computeQueue, resident, data, spec, sets, depth);
    }

    /**
     * Opens a streaming source.
     *
     * @param backend      backend allocating the slots
     * @param computeQueue queue of every dispatch reading the weights
     * @param resident     model-level tensors and every tensor not streamed
     * @param data         host tier covering every streamed weight layer; owned by the source (closed on failure too)
     * @param spec         model spec giving the execution sequence
     * @param sets         streamed layers ({@link LayerWeightSet#forTrunk})
     * @param depth        device slots; clamped to the number of streamed layers
     * @throws IllegalArgumentException when a set's weight layer is not in the trunk, repeats, or is not covered by the
     *                                  provider, or {@code depth < 1}
     * @throws RuntimeException         from the backend when the slots cannot be allocated
     */
    public static StreamingLayerWeightSource open(ComputeBackend backend, ComputeQueue computeQueue,
                                                  GpuWeights resident, WeightDataProvider data, ModelSpec spec,
                                                  List<LayerWeightSet> sets, int depth) {
        Objects.requireNonNull(data, "data");
        try {
            Objects.requireNonNull(resident, "resident");
            Index index = Index.of(spec, sets);
            if (sets.isEmpty()) {
                throw new IllegalArgumentException(spec.arch().ggufName() + ": no streamed weight layers");
            }
            for (int l : index.layerOfSet) {
                if (l >= data.layerCount()) {
                    throw new IllegalArgumentException("the weight data provider covers " + data.layerCount()
                            + " weight layers; streamed layer " + l + " is outside");
                }
            }
            int d = clampDepth(depth, sets.size());
            data.setVramDepth(d);
            LayerSlots slots = LayerSlots.allocate(backend, computeQueue, data, sets, index.slotAssignment(d), d);
            LOGGER.info(
                    "{}: streaming {} weight layers through {} slots ({} MB), RAM cache {} MB",
                    spec.arch().ggufName(), sets.size(), d, slots.deviceBytes() / (1024 * 1024),
                    data.maxCacheBytes() / (1024 * 1024));
            return new StreamingLayerWeightSource(resident, slots, data, d, index);
        } catch (RuntimeException e) {
            data.close();
            throw e;
        }
    }

    /**
     * Device bytes of the slots for a depth.
     *
     * @throws IllegalArgumentException as {@link #open}
     */
    public static long plannedBytes(ModelSpec spec, List<LayerWeightSet> sets, int depth) {
        Index index = Index.of(spec, sets);
        int d = clampDepth(depth, sets.size());
        return LayerSlots.plannedBytes(sets, index.slotAssignment(d), d);
    }

    /**
     * Slot count for a configuration: {@link EngineConfig#vramDepth()} when set (clamped to the streamed layers),
     * otherwise the largest depth whose slots fit {@code streamingBudgetBytes}.
     *
     * @throws IllegalArgumentException when there are no streamed layers
     * @throws GpuOutOfMemoryException  when not even one slot fits the budget
     */
    public static int resolveDepth(EngineConfig config, long streamingBudgetBytes, ModelSpec spec,
                                   List<LayerWeightSet> sets) {
        Objects.requireNonNull(config, "config");
        Index index = Index.of(spec, sets);
        int n = sets.size();
        if (n == 0) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": no streamed weight layers");
        }
        if (config.vramDepth() != EngineConfig.VRAM_DEPTH_AUTO) {
            int d = Math.min(config.vramDepth(), n);
            long need = LayerSlots.plannedBytes(sets, index.slotAssignment(d), d);
            if (need > streamingBudgetBytes) {
                LOGGER.warn(
                        "{}={} needs {} MB of streaming slots, more than the {} MB budget",
                        EngineConfig.PROP_VRAM_DEPTH, d, need / (1024 * 1024), streamingBudgetBytes / (1024 * 1024));
            }
            return d;
        }
        int best = 0;
        long one = -1;
        for (int d = 1; d <= n; d++) {
            long need = LayerSlots.plannedBytes(sets, index.slotAssignment(d), d);
            if (d == 1) {
                one = need;
            }
            if (need <= streamingBudgetBytes) {
                best = d;
            }
        }
        if (best == 0) {
            throw new GpuOutOfMemoryException(spec.arch().ggufName() + ": streaming one layer needs " + one
                    + " bytes of device memory but only " + Math.max(0, streamingBudgetBytes)
                    + " bytes are left for streamed weights; lower the context or the batch size");
        }
        return best;
    }

    private static int clampDepth(int depth, int nSets) {
        if (depth < 1) {
            throw new IllegalArgumentException("streaming depth must be at least 1, got " + depth);
        }
        return Math.min(depth, Math.max(1, nSets));
    }

    @Override
    public void prefetch(int il) {
        ensureOpen();
        int set = setOf(il);
        if (set < 0) {
            return;
        }
        advanceHostTier(il);
        if (slots.isResident(set)) {
            return;
        }
        int occupant = slots.occupant(slots.slotOf(set));
        if (occupant >= 0 && layerOfSet[occupant] == current) {
            return;
        }
        slots.stage(set);
    }

    @Override
    public void await(int il) {
        ensureOpen();
        int set = setOf(il);
        if (il == current && (set < 0 || slots.isResident(set))) {
            if (set >= 0) {
                slots.gate(set);
            }
            return;
        }
        retireCurrent();
        if (set >= 0) {
            if (!slots.isResident(set)) {
                slots.stage(set);
            }
            slots.gate(set);
        }
        current = il;
        int pos = positionOf(il);
        if (pos >= 0) {
            cursor = pos;
        }
        if (set >= 0) {
            advanceHostTier(il);
        }
        stageAhead();
    }

    @Override
    public GpuWeights weightsFor(int il) {
        ensureOpen();
        int set = setOf(il);
        return set < 0 ? resident : views[set];
    }

    /**
     * Stages the first {@code depth} streamed layers of the sequence and loads the host RAM window, so the next
     * forward pass finds its leading layers in flight. The slot of the layer last awaited is left alone: dispatches
     * reading it may still be pending, and it has no release marker until the next {@link #await}.
     */
    public void warmup() {
        ensureOpen();
        data.warmup();
        if (current >= 0 && current < seen.length) {
            seen[current] = true;
        }
        int staged = 0;
        for (int pos = 0; pos < sequence.length && staged < depth; pos++) {
            int wl = sequence[pos];
            int set = setOfLayer[wl];
            if (seen[wl] || set < 0) {
                continue;
            }
            seen[wl] = true;
            staged++;
            int occupant = slots.occupant(slots.slotOf(set));
            if (!slots.isResident(set) && (occupant < 0 || !seen[layerOfSet[occupant]])) {
                slots.stage(set);
            }
        }
        Arrays.fill(seen, false);
    }

    /** Number of device slots. */
    public int depth() {
        return depth;
    }

    /** Number of streamed weight layers. */
    public int streamedLayers() {
        return layerOfSet.length;
    }

    /** Device bytes of the slots. */
    public long deviceBytes() {
        return slots.deviceBytes();
    }

    /** True when uploads overlap compute on a separate transfer queue. */
    public boolean asyncTransfers() {
        return slots.asyncTransfers();
    }

    /** Waits for the queues and releases the slots and the host tier. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        slots.close();
    }

    /** Guards the slot of the layer just computed: its last dispatch has been enqueued. */
    private void retireCurrent() {
        if (current >= 0) {
            int set = setOf(current);
            if (set >= 0) {
                slots.release(set);
            }
        }
        current = -1;
    }

    /**
     * Stages the next {@code depth - 1} streamed layers after the cursor, stopping at the first whose slot holds the
     * current layer or a layer needed sooner.
     */
    private void stageAhead() {
        if (cursor < 0 || depth < 2) {
            return;
        }
        if (current >= 0 && current < seen.length) {
            seen[current] = true;
        }
        int counted = 0;
        for (int step = 1; step < sequence.length && counted < depth - 1; step++) {
            int wl = sequence[(cursor + step) % sequence.length];
            int set = setOfLayer[wl];
            if (seen[wl]) {
                continue;
            }
            seen[wl] = true;
            if (set < 0) {
                continue;
            }
            counted++;
            if (slots.isResident(set)) {
                continue;
            }
            int occupant = slots.occupant(slots.slotOf(set));
            if (occupant >= 0 && seen[layerOfSet[occupant]]) {
                break;
            }
            slots.stage(set);
        }
        Arrays.fill(seen, false);
    }

    /** Next position of weight layer {@code il} in the sequence after the cursor (cyclic), or -1. */
    private int positionOf(int il) {
        if (il >= seen.length) {
            return -1;
        }
        int start = cursor < 0 ? 0 : cursor + 1;
        for (int k = 0; k < sequence.length; k++) {
            int pos = (start + k) % sequence.length;
            if (sequence[pos] == il) {
                return pos;
            }
        }
        return -1;
    }

    private void advanceHostTier(int il) {
        if (il < data.layerCount()) {
            data.advanceExecution(il);
        }
    }

    private int setOf(int il) {
        if (il < 0) {
            throw new IllegalArgumentException("weight layer index " + il + " is negative");
        }
        return il < setOfLayer.length ? setOfLayer[il] : -1;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("streaming layer weight source is closed");
        }
    }

    @Override
    public String toString() {
        return "StreamingLayerWeightSource{" + layerOfSet.length + " layers, depth=" + depth + ", " + slots + '}';
    }

    /** Execution sequence and the weight layer / set mapping of one spec. Immutable. */
    private static final class Index {
        final int[] sequence;
        final int[] setOfLayer;
        final int[] layerOfSet;
        /** Streamed sets in order of first use. */
        final int[] firstUse;

        private Index(int[] sequence, int[] setOfLayer, int[] layerOfSet, int[] firstUse) {
            this.sequence = sequence;
            this.setOfLayer = setOfLayer;
            this.layerOfSet = layerOfSet;
            this.firstUse = firstUse;
        }

        static Index of(ModelSpec spec, List<LayerWeightSet> sets) {
            Objects.requireNonNull(spec, "spec");
            Objects.requireNonNull(sets, "sets");
            List<LayerSpec> layers = spec.layers();
            if (layers.isEmpty()) {
                throw new IllegalArgumentException(spec.arch().ggufName() + ": the trunk has no layers");
            }
            int[] sequence = new int[layers.size()];
            int maxLayer = 0;
            for (int i = 0; i < sequence.length; i++) {
                sequence[i] = layers.get(i).weightLayer();
                maxLayer = Math.max(maxLayer, sequence[i]);
            }
            int[] setOfLayer = new int[maxLayer + 1];
            Arrays.fill(setOfLayer, -1);
            int[] layerOfSet = new int[sets.size()];
            for (int i = 0; i < sets.size(); i++) {
                int wl = Objects.requireNonNull(sets.get(i), "set " + i).weightLayer();
                if (wl < 0 || wl > maxLayer || !contains(sequence, wl)) {
                    throw new IllegalArgumentException(spec.arch().ggufName() + ": streamed weight layer " + wl
                            + " is not used by the trunk");
                }
                if (setOfLayer[wl] >= 0) {
                    throw new IllegalArgumentException(spec.arch().ggufName() + ": two streamed sets for weight layer "
                            + wl);
                }
                setOfLayer[wl] = i;
                layerOfSet[i] = wl;
            }
            int[] firstUse = new int[sets.size()];
            boolean[] placed = new boolean[sets.size()];
            int n = 0;
            for (int wl : sequence) {
                int set = setOfLayer[wl];
                if (set >= 0 && !placed[set]) {
                    placed[set] = true;
                    firstUse[n++] = set;
                }
            }
            return new Index(sequence, setOfLayer, layerOfSet, firstUse);
        }

        private static boolean contains(int[] a, int v) {
            for (int x : a) {
                if (x == v) {
                    return true;
                }
            }
            return false;
        }

        /** Slot of every set: its rank in order of first use, modulo the depth. */
        int[] slotAssignment(int depth) {
            int[] slotOf = new int[layerOfSet.length];
            for (int rank = 0; rank < firstUse.length; rank++) {
                slotOf[firstUse[rank]] = rank % depth;
            }
            return slotOf;
        }
    }

    /** The tensors of one streamed layer, falling back to the resident weights. */
    private static final class LayerView implements GpuWeights {
        private final Map<String, GpuWeight> own;
        private final GpuWeights resident;

        LayerView(Map<String, GpuWeight> own, GpuWeights resident) {
            this.own = own;
            this.resident = resident;
        }

        @Override
        public GpuWeight find(String name) {
            GpuWeight w = own.get(name);
            return w != null ? w : resident.find(name);
        }

        @Override
        public String archName() {
            return resident.archName();
        }

        @Override
        public String toString() {
            return "LayerView{" + own.size() + " streamed tensors over " + resident + '}';
        }
    }
}
