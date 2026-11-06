package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.ops.moe.MoeSplitDispatcher;
import com.eonmux.neuralfree.gguf.ops.moe.MoeTailSkip;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The GPU expert cache of host-resident MoE experts (llama.cpp {@code llama-moe-cache}, Avalon's CUDA MoE cache):
 * hot experts of layers whose expert stacks live in host RAM are copied into spare VRAM banks, and the routed
 * experts read them through a per-layer indirection map. One slot holds one expert of one layer — all of its
 * expert tensors at one offset of the group's banks — and the layers with the same expert layout share one bank
 * set and one {@link MoeCacheSlru}.
 *
 * <p><b>Step protocol</b> (the block runs it between its router and its expert dispatches; the routed ids must
 * already be read back to the host, which synchronizes the queue as llama.cpp's cache copy callback does):
 * <ol>
 *   <li>{@link #beginStep}: the pending fills of earlier steps are gated (the compute queue waits on their
 *       transfer markers) and become visible in the layer's map; the map is re-uploaded when it changed; then the
 *       step's demand is planned. Steps of at most {@link #MAX_BATCH} tokens plan every demand (hits promote,
 *       misses take slots); larger steps only accumulate the routing histogram, so a long prefill never evicts
 *       the experts a later generation needs; the first small step after prefill admits the hottest experts.
 *       With a {@link MoeSplitDispatcher}, the step's remaining misses whose PCIe copy ends the MoE window sooner
 *       than their host rows upload right away on the compute queue and run on the device this step
 *       (Avalon {@code moe_hosted_split}); the rest stay misses the host computes.</li>
 *   <li>the block dispatches its expert kernels over the banks and the map; the pairs whose map entry is negative
 *       are computed on the host, and their outputs are uploaded before the weighted sum.</li>
 *   <li>{@link #endStep}: a compute marker after the dispatches guards the banks, and the step's remaining fills
 *       are enqueued on the transfer queue behind it, so a fill never overwrites an expert being read. A fill
 *       becomes visible at the layer's next {@link #beginStep}; until then its expert stays a miss, which the host
 *       computes while the copy runs (Avalon's hosted MoE overlap).</li>
 * </ol>
 *
 * <p>A fill that takes a victim slot clears the victim's map entry at once; the victim's layer re-uploads its map at
 * its next {@link #beginStep}, before its dispatches could read the overwritten slot. During a prefill the tail
 * slots of a group can be lent to a {@link MoeExpertPrefetch} ({@link #lendSlots} / {@link #restoreSlots}, Avalon's
 * prefetch zones): their experts are evicted and the slots leave the SLRU until they are given back.</p>
 *
 * <p>With no separate transfer queue every order is queue order and the markers are omitted. The banks of a
 * disabled group (its budget share holds fewer slots than one expert takes) are never allocated: every pair of
 * its layers is a miss and the host computes all of them.</p>
 *
 * <p>Owns its buffers and the {@link WeightDataProvider} it reads the expert bytes from. Not thread-safe.</p>
 */
public final class MoeExpertCache implements AutoCloseable {

    /** Steps with more tokens bypass the per-step planning and only accumulate the histogram
     * (llama-moe-cache.cpp {@code max_batch}). */
    public static final int MAX_BATCH = 32;

    private static final Logger LOGGER = LogManager.getLogger(MoeExpertCache.class);

    /** One expert tensor of a cached set: the stored name and its {@code [cols, rowsPerExpert, nExpert]} shape. */
    public static final class TensorPlan {
        private final String name;
        private final long cols;
        private final long rowsPerExpert;
        private final int nExpert;
        private final GGMLType type;

        /**
         * @throws IllegalArgumentException for non-positive sizes or a blank name
         */
        public TensorPlan(String name, long cols, long rowsPerExpert, int nExpert, GGMLType type) {
            if (name == null || name.trim().isEmpty()) {
                throw new IllegalArgumentException("expert tensor name is empty");
            }
            if (cols <= 0 || rowsPerExpert <= 0 || nExpert <= 0) {
                throw new IllegalArgumentException("expert tensor " + name + " needs positive sizes, got cols "
                        + cols + ", rows " + rowsPerExpert + ", n_expert " + nExpert);
            }
            this.name = name;
            this.cols = cols;
            this.rowsPerExpert = rowsPerExpert;
            this.nExpert = nExpert;
            this.type = Objects.requireNonNull(type, "type");
        }

        public String name() {
            return name;
        }

        public long cols() {
            return cols;
        }

        public long rowsPerExpert() {
            return rowsPerExpert;
        }

        /** The experts of the stack ({@code ne[2]}). */
        public int nExpert() {
            return nExpert;
        }

        public GGMLType type() {
            return type;
        }

        long expertBytes() {
            return type.getBytesForElements(cols * rowsPerExpert);
        }

        @Override
        public String toString() {
            return name + "[" + cols + ", " + rowsPerExpert + ", " + nExpert + "]:" + type;
        }
    }

    /** One cached layer: its weight layer, top-k and the ordered expert tensors (equal orders share a group). */
    public static final class LayerPlan {
        private final int weightLayer;
        private final int k;
        private final List<TensorPlan> tensors;

        /**
         * @throws IllegalArgumentException for a negative weight layer, a top-k below one, or no tensors
         */
        public LayerPlan(int weightLayer, int k, List<TensorPlan> tensors) {
            if (weightLayer < 0) {
                throw new IllegalArgumentException("weight layer " + weightLayer + " is negative");
            }
            if (k < 1) {
                throw new IllegalArgumentException("layer " + weightLayer + ": top-k must be positive, got " + k);
            }
            if (tensors == null || tensors.isEmpty()) {
                throw new IllegalArgumentException("layer " + weightLayer + " has no expert tensors");
            }
            this.weightLayer = weightLayer;
            this.k = k;
            this.tensors = Collections.unmodifiableList(new ArrayList<>(tensors));
        }

        public int weightLayer() {
            return weightLayer;
        }

        public int k() {
            return k;
        }

        /** The ordered expert tensors; equal lists share a group. */
        public List<TensorPlan> tensors() {
            return tensors;
        }

        int nExpert() {
            return tensors.get(0).nExpert();
        }

        long expertBytes() {
            long sum = 0;
            for (TensorPlan t : tensors) {
                sum = Math.addExact(sum, t.expertBytes());
            }
            return sum;
        }

        boolean sameLayout(LayerPlan o) {
            if (tensors.size() != o.tensors.size() || k != o.k || nExpert() != o.nExpert()) {
                return false;
            }
            for (int i = 0; i < tensors.size(); i++) {
                TensorPlan a = tensors.get(i);
                TensorPlan b = o.tensors.get(i);
                if (a.cols != b.cols || a.rowsPerExpert != b.rowsPerExpert || a.nExpert != b.nExpert
                        || a.type != b.type) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public String toString() {
            return "layer " + weightLayer + " k=" + k + " " + tensors;
        }
    }

    /** What one forward step of one layer does with its routed pairs. */
    public static final class Step {
        private final int[] missPairs;
        private final int missCount;
        private final int hits;
        private final int[] skippedPairs;
        private final int skippedCount;

        Step(int[] missPairs, int missCount, int hits) {
            this(missPairs, missCount, hits, new int[0], 0);
        }

        Step(int[] missPairs, int missCount, int hits, int[] skippedPairs, int skippedCount) {
            this.missPairs = missPairs;
            this.missCount = missCount;
            this.hits = hits;
            this.skippedPairs = skippedPairs;
            this.skippedCount = skippedCount;
        }

        /**
         * A step whose every pair is a miss: the pass of a layer the cache does not serve.
         *
         * @throws IllegalArgumentException for a negative pair count
         */
        public static Step allMiss(int pairs) {
            if (pairs < 0) {
                throw new IllegalArgumentException("pair count must be >= 0, got " + pairs);
            }
            int[] out = new int[pairs];
            for (int i = 0; i < pairs; i++) {
                out[i] = i;
            }
            return new Step(out, pairs, 0);
        }

        /** Pair indices ({@code t * k + j}) of the misses, valid up to {@link #missCount()}. */
        public int[] missPairs() {
            return missPairs;
        }

        public int missCount() {
            return missCount;
        }

        /** Pairs whose expert was visible in the cache (the GPU computed them). */
        public int hitCount() {
            return hits;
        }

        /**
         * Pair indices ({@code t * k + j}) of the misses the tail-skip rule drops, valid up to
         * {@link #skippedCount()}: neither the host nor the GPU computes them, their expert output rows stay zero.
         */
        public int[] skippedPairs() {
            return skippedPairs;
        }

        /** Misses the tail-skip rule drops ({@link MoeTailSkip}); 0 when it is off. */
        public int skippedCount() {
            return skippedCount;
        }
    }

    /** Banks, slot pool and SLRU of the layers with the same expert layout. */
    private static final class Group {
        final List<TensorPlan> tensors;
        final List<Integer> plans = new ArrayList<>();
        final long[] bankBytes;
        final GpuBuffer[] banks;
        final long bytesPerSlot;
        final long hostBytes;
        /** Slots per layer of the group; 0 disables the group. */
        int nSlots;
        MoeCacheSlru slru;
        /** Per slot, the layer whose expert it was last given to (null when never), and that expert. */
        Layer[] owner;
        int[] ownerExpert;
        /** Slots lent out at the tail of the banks, {@code [nSlots - lentSlots, nSlots)}. */
        int lentSlots;
        int hits;
        int misses;
        long filledBytes;

        Group(List<TensorPlan> tensors) {
            this.tensors = tensors;
            this.bankBytes = new long[tensors.size()];
            long perSlot = 0;
            for (int i = 0; i < bankBytes.length; i++) {
                bankBytes[i] = tensors.get(i).expertBytes();
                perSlot = Math.addExact(perSlot, bankBytes[i]);
            }
            this.bytesPerSlot = perSlot;
            long host = 0;
            for (TensorPlan t : tensors) {
                host = Math.addExact(host, t.expertBytes());
            }
            this.hostBytes = host;
            this.banks = new GpuBuffer[tensors.size()];
        }

        int nExpert() {
            return tensors.get(0).nExpert();
        }
    }

    /** One layer's plan, map, histogram and pending fills. */
    private static final class Layer {
        final LayerPlan plan;
        final int weightLayer;
        final int group;
        /** Position of the layer inside its group (the SLRU's layer index). */
        final int groupPos;
        final int[] map;
        final GpuBuffer mapBuffer;
        final long[] heat;
        final List<MoeCacheSlru.Fill> fills = new ArrayList<>();
        /** Transfer marker of every pending fill; fills of one step share their marker. */
        final List<GpuEvent> fillMarkers = new ArrayList<>();
        boolean sawPrefill;
        /** True when the host map changed since the last upload (an eviction cleared an entry). */
        boolean mapDirty;
        int k;

        Layer(LayerPlan plan, int group, int groupPos, GpuBuffer mapBuffer) {
            this.plan = plan;
            this.weightLayer = plan.weightLayer();
            this.group = group;
            this.groupPos = groupPos;
            this.map = new int[plan.nExpert()];
            Arrays.fill(map, -1);
            this.mapBuffer = mapBuffer;
            this.heat = new long[plan.nExpert()];
            this.k = plan.k();
        }
    }

    private final ComputeQueue computeQueue;
    private final ComputeQueue transferQueue;
    private final boolean sameQueue;
    private final WeightDataProvider data;
    private final MoeSplitDispatcher dispatcher;
    private final List<LayerPlan> plans;
    private final List<Group> groups;
    private final Group[] groupOfPlan;
    private final Map<Integer, Layer> layerOfWeightLayer;
    private final List<Layer> layers;
    private final ByteBuffer staging;
    private final ByteBuffer mapStaging;
    private final int[] pairScratch;
    /** Pair indices of a step's misses the tail-skip rule drops ({@link Step#skippedPairs()}). */
    private final int[] skipPairScratch;
    private final List<MoeCacheSlru.Fill> stepFills = MoeCacheSlru.newFillList();
    /** Scratch of one {@link MoeCacheSlru#plan} call, merged into {@link #stepFills}. */
    private final List<MoeCacheSlru.Fill> planFills = MoeCacheSlru.newFillList();
    private long bankBytes;
    private boolean closed;

    private MoeExpertCache(ComputeQueue computeQueue, ComputeQueue transferQueue, WeightDataProvider data,
                           MoeSplitDispatcher dispatcher, List<LayerPlan> plans, List<Group> groups,
                           Group[] groupOfPlan, List<Layer> layers, int maxPairs) {
        this.computeQueue = computeQueue;
        this.transferQueue = transferQueue;
        this.sameQueue = transferQueue == computeQueue;
        this.data = data;
        this.dispatcher = dispatcher;
        this.plans = plans;
        this.groups = groups;
        this.groupOfPlan = groupOfPlan;
        this.layers = layers;
        this.layerOfWeightLayer = new HashMap<>();
        for (Layer l : layers) {
            layerOfWeightLayer.put(l.weightLayer, l);
        }
        this.staging = ByteBuffer.allocateDirect((int) largestPlanBytes()).order(ByteOrder.LITTLE_ENDIAN);
        long mapBytes = 0;
        for (Layer l : layers) {
            mapBytes = Math.addExact(mapBytes, mapBytes(l.map.length));
        }
        this.mapStaging = ByteBuffer.allocateDirect((int) mapBytes).order(ByteOrder.LITTLE_ENDIAN);
        this.pairScratch = new int[maxPairs];
        this.skipPairScratch = new int[maxPairs];
    }

    private long largestPlanBytes() {
        long max = 0;
        for (LayerPlan p : plans) {
            max = Math.max(max, p.expertBytes());
        }
        return max;
    }

    private static long mapBytes(int nExpert) {
        return 4L * nExpert;
    }

    /**
     * Builds the cache: groups the plans by layout, sizes each group's slot pool from its share of the budget
     * (llama-moe-cache.cpp:345-364), allocates the banks and the per-layer maps, and preloads the slots from
     * {@code ranking} (Avalon {@code moe-cache-ranking.bin}, hottest first) when one is given.
     *
     * @param backend     backend owning the buffers and the queues
     * @param compute     queue every expert dispatch runs on
     * @param data        host tier of the expert bytes (owned by the cache, closed on failure too); every expert
     *                    tensor must be addressable by name
     * @param plans       one plan per cached layer
     * @param budgetBytes device bytes the banks may take
     * @param maxPairs    largest routed-pair count of one step ({@code ubatch * k}); sizes the miss list
     * @param ranking     the model's ranking session, whose preloaded ranking fills the slots; null or one without a
     *                    preload leaves the cache empty
     * @param modelName   model name for the log line
     * @throws IllegalArgumentException for an empty plan list, a budget below one slot, a plan tensor unknown to
     *                                  the provider, plans that disagree inside a group about the expert count, or
     *                                  an expert tensor above 2 GiB
     * @throws UncheckedIOException     when an expert tensor's bytes cannot be read at preload
     */
    public static MoeExpertCache open(ComputeBackend backend, ComputeQueue compute, WeightDataProvider data,
                                      List<LayerPlan> plans, long budgetBytes, int maxPairs,
                                      MoeCacheRanking ranking, String modelName) {
        return open(backend, compute, data, plans, budgetBytes, maxPairs, ranking, modelName, null);
    }

    /**
     * Builds the cache with a split dispatcher: on a step of two or more tokens the dispatcher decides which
     * demanded misses upload into their slots now, on the compute queue, so their pairs run on the device this step
     * (Avalon {@code moe_hosted_split}); null computes every miss on the host until its deferred fill lands.
     *
     * @param dispatcher the split dispatcher over the host team's cost model, or null for none
     */
    public static MoeExpertCache open(ComputeBackend backend, ComputeQueue compute, WeightDataProvider data,
                                      List<LayerPlan> plans, long budgetBytes, int maxPairs,
                                      MoeCacheRanking ranking, String modelName,
                                      MoeSplitDispatcher dispatcher) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(compute, "compute");
        Objects.requireNonNull(data, "data");
        if (plans == null || plans.isEmpty()) {
            throw new IllegalArgumentException("a MoE expert cache needs at least one layer plan");
        }
        if (budgetBytes <= 0) {
            throw new IllegalArgumentException("MoE expert cache budget must be positive, got " + budgetBytes);
        }
        if (maxPairs < 1) {
            throw new IllegalArgumentException("max pair count must be positive, got " + maxPairs);
        }
        List<LayerPlan> planList = Collections.unmodifiableList(new ArrayList<>(plans));
        for (LayerPlan p : planList) {
            for (TensorPlan t : p.tensors()) {
                if (!data.containsKey(t.name())) {
                    throw new IllegalArgumentException("cached expert tensor '" + t.name()
                            + "' is not known to the weight data provider");
                }
                if (t.expertBytes() > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("cached expert tensor '" + t.name() + "' holds "
                            + t.expertBytes() + " bytes per expert, above one host transfer");
                }
            }
        }
        List<Group> groups = buildGroups(planList);
        Group[] groupOfPlan = new Group[planList.size()];
        for (int g = 0; g < groups.size(); g++) {
            for (int pi : groups.get(g).plans) {
                groupOfPlan[pi] = groups.get(g);
            }
        }
        sizeGroups(groups, budgetBytes);

        ComputeQueue tq = backend.transferQueue();
        ComputeQueue transfer = backend.capabilities().supportsAsyncTransfer() && tq != null ? tq : compute;
        List<Layer> layers = new ArrayList<>(planList.size());
        long bankTotal = 0;
        boolean allocated = false;
        try {
            for (Group g : groups) {
                if (g.nSlots <= 0) {
                    continue;
                }
                for (int i = 0; i < g.banks.length; i++) {
                    g.banks[i] = backend.allocateBuffer(Math.multiplyExact(g.bankBytes[i], g.nSlots));
                    if (g.bankBytes[i] >= Integer.BYTES) {
                        compute.enqueueFillBuffer(g.banks[i], 0, Integer.BYTES, 0);
                    }
                    bankTotal = Math.addExact(bankTotal, Math.multiplyExact(g.bankBytes[i], g.nSlots));
                }
                g.slru = new MoeCacheSlru(g.plans.size(), g.nExpert(), g.nSlots);
                g.owner = new Layer[g.nSlots];
                g.ownerExpert = new int[g.nSlots];
            }
            for (int i = 0; i < planList.size(); i++) {
                LayerPlan p = planList.get(i);
                long bytes = mapBytes(p.nExpert());
                GpuBuffer mapBuffer = backend.allocateBuffer(bytes);
                compute.enqueueFillBuffer(mapBuffer, 0, bytes, 0xFFFFFFFF);
                layers.add(new Layer(p, groupOfPlan[i] == null ? -1 : indexOf(groups, groupOfPlan[i]),
                        groups.get(indexOf(groups, groupOfPlan[i])).plans.indexOf(i), mapBuffer));
            }
            compute.finish();
            allocated = true;
            MoeExpertCache cache = new MoeExpertCache(compute, transfer, data, dispatcher, planList, groups,
                    groupOfPlan, layers, maxPairs);
            cache.bankBytes = bankTotal;
            boolean preload = ranking != null && ranking.preloaded();
            if (preload) {
                cache.preload(ranking);
            }
            LOGGER.info("{}: MoE expert cache: {} layer(s) in {} group(s), {} slot(s), {} MiB of banks{}",
                    modelName, planList.size(), groups.size(), cache.totalSlots(), bankTotal / (1024 * 1024),
                    preload ? ", preloaded from " + ranking.preloadedFrom() : "");
            return cache;
        } catch (RuntimeException e) {
            if (!allocated) {
                RuntimeException failure = closePartial(groups, layers, e);
                throw failure;
            }
            throw e;
        }
    }

    private static int indexOf(List<Group> groups, Group g) {
        for (int i = 0; i < groups.size(); i++) {
            if (groups.get(i) == g) {
                return i;
            }
        }
        throw new AssertionError("group is not in the list");
    }

    /** Closes the buffers allocated before a failure, returning the first close error, or {@code failure}. */
    private static RuntimeException closePartial(List<Group> groups, List<Layer> layers, RuntimeException failure) {
        RuntimeException first = null;
        for (Group g : groups) {
            for (GpuBuffer b : g.banks) {
                if (b != null) {
                    try {
                        b.close();
                    } catch (RuntimeException e) {
                        first = first != null ? first : e;
                    }
                }
            }
        }
        for (Layer l : layers) {
            try {
                l.mapBuffer.close();
            } catch (RuntimeException e) {
                first = first != null ? first : e;
            }
        }
        return first != null ? first : failure;
    }

    /** Groups the plans by equal tensor layout, in plan order. */
    private static List<Group> buildGroups(List<LayerPlan> plans) {
        List<Group> out = new ArrayList<>();
        List<LayerPlan> refs = new ArrayList<>();
        for (int i = 0; i < plans.size(); i++) {
            LayerPlan p = plans.get(i);
            int g = 0;
            while (g < refs.size() && !refs.get(g).sameLayout(p)) {
                g++;
            }
            if (g == refs.size()) {
                out.add(new Group(p.tensors()));
                refs.add(p);
            }
            out.get(g).plans.add(i);
        }
        return out;
    }

    /**
     * The slot pool of each group: its budget share divided by one expert's bytes (llama-moe-cache.cpp:345-364),
     * capped at the distinct experts of its layers. A group that cannot hold one demand's experts is disabled.
     */
    private static void sizeGroups(List<Group> groups, long budgetBytes) {
        long total = 0;
        for (Group g : groups) {
            total = Math.addExact(total, g.hostBytes * g.plans.size());
        }
        for (Group g : groups) {
            double share = (double) budgetBytes * (g.hostBytes * g.plans.size()) / total;
            int nSlots = (int) Math.min(Integer.MAX_VALUE, (long) Math.floor(share / g.bytesPerSlot));
            int distinct = Math.multiplyExact(g.nExpert(), g.plans.size());
            g.nSlots = Math.min(nSlots, distinct);
        }
    }

    // ================================================================== step protocol

    /**
     * Starts one forward step of layer {@code il} over the routed {@code ids} ({@code nTokens * k} of them, in
     * pair order): completes the pending fills of earlier steps, uploads the map when it changed, plans the
     * demand or accumulates the histogram, and reports the pairs whose experts the host must compute.
     *
     * @throws IllegalArgumentException for an array shorter than the pairs, an id outside the experts, a pair
     *                                  count above the open-time maximum, or a layer the cache does not hold
     */
    public Step beginStep(int il, int[] ids, int nIds, int nTokens) {
        return beginStep(il, ids, nIds, nTokens, true);
    }

    /**
     * {@link #beginStep(int, int[], int, int)} for a step whose misses the host computes ({@code hostedMisses}) or
     * one whose misses a {@link MoeExpertPrefetch} uploads into its zones: the latter never runs the split
     * dispatcher, since none of its misses is left for the host.
     *
     * @throws IllegalArgumentException as {@link #beginStep(int, int[], int, int)}
     */
    public Step beginStep(int il, int[] ids, int nIds, int nTokens, boolean hostedMisses) {
        return beginStep(il, ids, nIds, nTokens, hostedMisses, MoeTailSkip.OFF);
    }

    /**
     * {@link #beginStep(int, int[], int, int, boolean)} with the hosted tail-skip rule
     * (Avalon {@code GGML_CUDA_MOE_TAIL_SKIP}, {@link MoeTailSkip}): misses of an uncached expert that every token
     * routes only at rank {@code tailSkipRank} or lower leave {@link Step#missPairs()} for
     * {@link Step#skippedPairs()} before the split dispatcher sees them, so a dropped expert is neither moved to the
     * GPU nor computed on the host. {@link MoeTailSkip#OFF} keeps every miss, and a step whose misses a
     * {@link MoeExpertPrefetch} uploads ({@code hostedMisses = false}) is never pruned.
     *
     * @throws IllegalArgumentException as {@link #beginStep(int, int[], int, int)}, and for a rank above
     *                                  {@link MoeTailSkip#MAX_RANK}
     */
    public Step beginStep(int il, int[] ids, int nIds, int nTokens, boolean hostedMisses, int tailSkipRank) {
        ensureOpen();
        Layer l = layer(il);
        if (nIds < 0 || nIds > ids.length) {
            throw new IllegalArgumentException("pair count " + nIds + " outside [0, " + ids.length + "]");
        }
        if (nIds > pairScratch.length) {
            throw new IllegalArgumentException("pair count " + nIds + " exceeds the open-time maximum "
                    + pairScratch.length);
        }
        if (nIds % l.k != 0) {
            throw new IllegalArgumentException("layer " + il + ": " + nIds + " routed ids are not " + l.k
                    + " per token");
        }
        if (tailSkipRank < MoeTailSkip.OFF || tailSkipRank > MoeTailSkip.MAX_RANK) {
            throw new IllegalArgumentException("tail-skip rank must be " + MoeTailSkip.OFF + " (off) or 1-"
                    + MoeTailSkip.MAX_RANK + ", got " + tailSkipRank);
        }
        Group g = groups.get(l.group);
        completeFills(l, g);
        if (l.mapDirty) {
            uploadMap(l);
        }
        if (nTokens <= MAX_BATCH && l.sawPrefill) {
            flushHeat(l, g);
        }
        Step step = planStep(l, g, ids, nIds, nTokens, hostedMisses && dispatcher != null,
                hostedMisses ? tailSkipRank : MoeTailSkip.OFF);
        l.sawPrefill = nTokens > MAX_BATCH;
        return step;
    }

    private Step planStep(Layer l, Group g, int[] ids, int nIds, int nTokens, boolean split, int tailSkipRank) {
        int hits = 0;
        int missCount = 0;
        if (g.nSlots <= 0) {
            for (int p = 0; p < nIds; p++) {
                pairScratch[p] = p;
            }
            g.misses += nIds;
            return tailSkippedStep(l, ids, nIds, nIds, 0, tailSkipRank);
        }
        if (nTokens > MAX_BATCH) {
            for (int p = 0; p < nIds; p++) {
                int e = checkId(l, ids[p]);
                l.heat[e]++;
                if (l.map[e] >= 0) {
                    hits++;
                } else {
                    pairScratch[missCount++] = p;
                }
            }
            g.hits += hits;
            g.misses += missCount;
            return tailSkippedStep(l, ids, nIds, missCount, hits, tailSkipRank);
        }
        int planned = g.slru.plan(layerPos(g, l), ids, nIds, planFills);
        if (planned < 0) {
            // the demand is wider than the pool: the dispatcher picks the misses worth an immediate copy, the
            // rest stay misses the host computes
            if (split) {
                migrateForThisStep(l, g, ids, nIds, nTokens, 0, tailSkipMask(l, ids, nIds, tailSkipRank));
            }
            for (int p = 0; p < nIds; p++) {
                int e = checkId(l, ids[p]);
                if (l.map[e] >= 0) {
                    hits++;
                } else {
                    pairScratch[missCount++] = p;
                }
            }
            g.hits += hits;
            g.misses += missCount;
            return tailSkippedStep(l, ids, nIds, missCount, hits, tailSkipRank);
        }
        // a slot is reserved the moment its fill is planned, so no later plan can double-assign it
        for (MoeCacheSlru.Fill f : planFills) {
            g.slru.fillStarted(f.slot);
            assign(g, l, f.expert, f.slot);
        }
        stepFills.addAll(planFills);
        // the rule sees the cache as it is before the split, so a dropped expert is never a split candidate
        boolean[] skip = tailSkipMask(l, ids, nIds, tailSkipRank);
        if (split) {
            migrateForThisStep(l, g, ids, nIds, nTokens, planFills.size(), skip);
        }
        for (int p = 0; p < nIds; p++) {
            int e = checkId(l, ids[p]);
            if (l.map[e] >= 0) {
                hits++;
            } else {
                pairScratch[missCount++] = p;
            }
        }
        g.hits += hits;
        g.misses += missCount;
        return partitionMisses(ids, missCount, hits, skip);
    }

    /** The tail-skip mask of the step over the cache's current residency; all false when the rule is off. */
    private boolean[] tailSkipMask(Layer l, int[] ids, int nIds, int tailSkipRank) {
        if (tailSkipRank == MoeTailSkip.OFF) {
            return new boolean[l.map.length];
        }
        boolean[] cached = new boolean[l.map.length];
        for (int p = 0; p < nIds; p++) {
            int e = checkId(l, ids[p]);
            cached[e] = l.map[e] >= 0;
        }
        return MoeTailSkip.skippedExperts(ids, nIds, l.k, tailSkipRank, cached, l.map.length);
    }

    /** A step whose misses in {@link #pairScratch} the tail-skip rule of {@code l} may prune. */
    private Step tailSkippedStep(Layer l, int[] ids, int nIds, int missCount, int hits, int tailSkipRank) {
        if (tailSkipRank == MoeTailSkip.OFF || missCount == 0) {
            return new Step(pairScratch, missCount, hits);
        }
        return partitionMisses(ids, missCount, hits, tailSkipMask(l, ids, nIds, tailSkipRank));
    }

    /**
     * Splits the {@code missCount} misses in {@link #pairScratch} into the host rows and the pairs the tail-skip
     * mask drops; the host rows stay at the front of {@link #pairScratch}, the dropped pairs go to
     * {@link #skipPairScratch}.
     */
    private Step partitionMisses(int[] ids, int missCount, int hits, boolean[] skip) {
        int kept = 0;
        int dropped = 0;
        for (int q = 0; q < missCount; q++) {
            int p = pairScratch[q];
            if (skip[ids[p]]) {
                skipPairScratch[dropped++] = p;
            } else {
                pairScratch[kept++] = p;
            }
        }
        return new Step(pairScratch, kept, hits, skipPairScratch, dropped);
    }

    /**
     * The dispatcher's split of the step's remaining misses (Avalon {@code moe_hosted_split}): the experts whose
     * PCIe copy ends the MoE window sooner than their host rows upload into their planned or a victim slot right
     * away, on the compute queue ahead of the pass, and their pairs run on the device this step; their deferred
     * fills leave {@link #stepFills}, so the transfer queue never rewrites them. Steps with fewer than two tokens
     * never split ({@link MoeSplitDispatcher}'s rule); their misses stay with the host and their planned fills
     * land at {@link #endStep}. Experts the tail-skip rule drops ({@code skip}) are not candidates: their rows count
     * neither for the host nor for the device, as Avalon prunes the CPU rows before its dispatcher runs.
     */
    private void migrateForThisStep(Layer l, Group g, int[] ids, int nIds, int nTokens, int plannedFills,
                                    boolean[] skip) {
        int[] rows = new int[l.map.length];
        boolean[] cached = new boolean[l.map.length];
        for (int p = 0; p < nIds; p++) {
            int e = checkId(l, ids[p]);
            if (skip[e]) {
                continue;
            }
            rows[e]++;
            cached[e] = l.map[e] >= 0;
        }
        MoeSplitDispatcher.Plan move = dispatcher.decide(nTokens, rows, cached,
                plannedFills + g.slru.emptySlots(), g.bytesPerSlot, g.tensors.size());
        if (move.isEmpty()) {
            return;
        }
        boolean mapChanged = false;
        for (int e : move.migrate()) {
            if (l.map[e] >= 0) {
                continue;
            }
            int slot = takePlannedOrAdmit(l, g, e);
            if (slot < 0) {
                continue;
            }
            try {
                uploadSync(l, g, e, slot);
            } catch (IOException ex) {
                g.slru.fillDone(slot);
                throw new UncheckedIOException("MoE expert cache split fill of layer " + l.weightLayer
                        + " failed: " + ex.getMessage(), ex);
            }
            g.slru.fillDone(slot);
            l.map[e] = slot;
            mapChanged = true;
        }
        if (mapChanged) {
            uploadMap(l);
        }
    }

    /** The planned fill of expert {@code e} of this step, removing it from the deferred set, or -1's slot. */
    private int takePlannedOrAdmit(Layer l, Group g, int expert) {
        for (MoeCacheSlru.Fill f : stepFills) {
            if (f.layer == l.groupPos && f.expert == expert) {
                stepFills.remove(f);
                return f.slot;
            }
        }
        int slot = g.slru.admit(l.groupPos, expert);
        if (slot >= 0) {
            assign(g, l, expert, slot);
        }
        return slot;
    }

    /** Uploads one expert of every bank of the group onto the compute queue, ahead of the pass. */
    private void uploadSync(Layer l, Group g, int expert, int slot) throws IOException {
        for (int i = 0; i < g.tensors.size(); i++) {
            TensorPlan t = l.plan.tensors().get(i);
            long off = Math.multiplyExact((long) slot, g.bankBytes[i]);
            staging.clear();
            data.readRange(t.name(), (long) expert * g.bankBytes[i], (int) g.bankBytes[i], staging);
            GpuEvent evt = computeQueue.enqueueWriteBuffer(g.banks[i], off, g.bankBytes[i], staging);
            if (evt != null) {
                evt.close();
            }
            g.filledBytes += g.bankBytes[i];
        }
    }

    private int checkId(Layer l, int e) {
        if (e < 0 || e >= l.map.length) {
            throw new IllegalArgumentException("layer " + l.weightLayer + ": routed expert " + e + " outside [0, "
                    + l.map.length + ")");
        }
        return e;
    }

    /** The position of a layer inside its group (the SLRU's layer index). */
    private int layerPos(Group g, Layer l) {
        return l.groupPos;
    }

    /**
     * Ends one forward step of layer {@code il}: records the compute marker its dispatches are behind and
     * enqueues the step's fill bytes onto the transfer queue behind it. Called after the last dispatch that
     * reads the layer's banks; the fills become visible at the layer's next {@link #beginStep}.
     *
     * @throws UncheckedIOException when an expert tensor's bytes cannot be read (the slots of the fills that
     *                              were enqueued stay reserved)
     */
    public void endStep(int il) {
        ensureOpen();
        Layer l = layer(il);
        if (stepFills.isEmpty()) {
            return;
        }
        Group g = groups.get(l.group);
        List<MoeCacheSlru.Fill> fills = new ArrayList<>(stepFills);
        stepFills.clear();
        GpuEvent wait = null;
        try {
            if (!sameQueue) {
                computeQueue.flush();
                wait = computeQueue.enqueueMarker();
            }
            for (MoeCacheSlru.Fill f : fills) {
                uploadFill(l, g, f, wait);
                l.fills.add(f);
                l.fillMarkers.add(wait);
            }
            transferQueue.flush();
        } catch (IOException e) {
            LOGGER.warn("MoE expert cache: fill of layer {} failed, {} slot(s) stay reserved: {}", il, fills.size(),
                    e.getMessage());
            throw new UncheckedIOException("MoE expert cache fill of layer " + il + " failed: " + e.getMessage(), e);
        }
    }

    /** Uploads one expert of every bank of the group, from byte 0 of the slot. */
    private void uploadFill(Layer l, Group g, MoeCacheSlru.Fill f, GpuEvent wait) throws IOException {
        GpuEvent[] waits = wait != null ? new GpuEvent[]{wait} : new GpuEvent[0];
        for (int i = 0; i < g.tensors.size(); i++) {
            TensorPlan t = l.plan.tensors().get(i);
            long off = Math.multiplyExact((long) f.slot, g.bankBytes[i]);
            staging.clear();
            data.readRange(t.name(), (long) f.expert * g.bankBytes[i], (int) g.bankBytes[i], staging);
            GpuEvent evt = transferQueue.enqueueWriteBuffer(g.banks[i], off, g.bankBytes[i], staging, waits);
            if (evt != null) {
                evt.close();
            }
            g.filledBytes += g.bankBytes[i];
        }
    }

    /** Applies the layer's completed fills to its map and uploads the map when it changed. */
    private void completeFills(Layer l, Group g) {
        if (l.fills.isEmpty()) {
            return;
        }
        gateMarkers(l);
        for (int i = 0; i < l.fills.size(); i++) {
            MoeCacheSlru.Fill f = l.fills.get(i);
            l.map[f.expert] = f.slot;
            g.slru.fillDone(f.slot);
        }
        l.fills.clear();
        l.fillMarkers.clear();
        uploadMap(l);
    }

    /** Waits for every pending fill marker of the layer, then closes the markers nothing else references. */
    private void gateMarkers(Layer l) {
        if (sameQueue) {
            return;
        }
        List<GpuEvent> distinct = new ArrayList<>();
        for (GpuEvent m : l.fillMarkers) {
            if (m != null && indexOf(distinct, m) < 0) {
                distinct.add(m);
            }
        }
        if (!distinct.isEmpty()) {
            GpuEvent gate = computeQueue.enqueueMarker(distinct.toArray(new GpuEvent[0]));
            if (gate != null) {
                gate.close();
            }
        }
        for (GpuEvent m : distinct) {
            if (indexOf(l.fillMarkers, m) == l.fillMarkers.lastIndexOf(m)) {
                m.close();
            }
        }
    }

    private static int indexOf(List<GpuEvent> list, GpuEvent e) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == e) {
                return i;
            }
        }
        return -1;
    }

    private void uploadMap(Layer l) {
        l.mapDirty = false;
        mapStaging.clear();
        for (int e : l.map) {
            mapStaging.putInt(e);
        }
        GpuEvent evt = computeQueue.enqueueWriteBuffer(l.mapBuffer, 0, mapBytes(l.map.length), mapStaging);
        if (evt != null) {
            evt.close();
        }
    }

    /** Admits the histogram's hottest not-cached experts while victims exist, then clears the histogram. */
    private void flushHeat(Layer l, Group g) {
        List<Integer> order = new ArrayList<>(l.heat.length);
        for (int e = 0; e < l.heat.length; e++) {
            if (l.heat[e] > 0 && l.map[e] < 0) {
                order.add(e);
            }
        }
        order.sort((a, b) -> {
            long ha = l.heat[a];
            long hb = l.heat[b];
            return ha != hb ? Long.compare(hb, ha) : Integer.compare(a, b);
        });
        for (int e : order) {
            int slot = g.slru.admit(layerPos(g, l), e);
            if (slot < 0) {
                break;
            }
            g.slru.fillStarted(slot);
            assign(g, l, e, slot);
            stepFills.add(new MoeCacheSlru.Fill(layerPos(g, l), e, slot));
        }
        Arrays.fill(l.heat, 0L);
    }

    /**
     * Records that expert {@code e} of layer {@code l} takes {@code slot}: the slot's previous expert loses its map
     * entry, and its layer's map is re-uploaded at that layer's next {@link #beginStep}, before any of its dispatches
     * could read the slot the fill overwrites.
     */
    private void assign(Group g, Layer l, int e, int slot) {
        evict(g, slot);
        g.owner[slot] = l;
        g.ownerExpert[slot] = e;
    }

    /** Clears the map entry of the expert that holds {@code slot}, if any; the slot keeps no owner. */
    private static void evict(Group g, int slot) {
        Layer prev = g.owner[slot];
        if (prev != null && prev.map[g.ownerExpert[slot]] == slot) {
            prev.map[g.ownerExpert[slot]] = -1;
            prev.mapDirty = true;
        }
        g.owner[slot] = null;
    }

    // ================================================================== slot lending (prefill prefetch)

    /**
     * Lends the last {@code count} slots of layer {@code il}'s group out of the cache (Avalon's prefetch zones,
     * {@code moe_cache_lend}): the pending fills of the group's layers complete first (the compute queue waits for
     * them, as their next {@link #beginStep} would), then the experts in the lent slots are evicted, so no map
     * points into them and no admission takes them until {@link #restoreSlots}. The caller writes the lent slots
     * only behind a compute marker enqueued after this call, which orders its writes after every earlier read.
     *
     * @throws IllegalArgumentException when {@code count} is outside {@code [1, slotCount]}
     * @throws IllegalStateException    when the group already lends slots or a step is between its
     *                                  {@link #beginStep} and {@link #endStep}
     */
    void lendSlots(int il, int count) {
        ensureOpen();
        Group g = groups.get(layer(il).group);
        if (count < 1 || count > g.nSlots) {
            throw new IllegalArgumentException("lent slot count " + count + " outside [1, " + g.nSlots + "]");
        }
        if (g.lentSlots > 0) {
            throw new IllegalStateException("layer " + il + ": the group already lends " + g.lentSlots + " slot(s)");
        }
        if (!stepFills.isEmpty()) {
            throw new IllegalStateException("slots cannot be lent while a step's fills are being planned");
        }
        for (Layer member : layers) {
            if (member.group == layer(il).group) {
                completeFills(member, g);
            }
        }
        for (int s = g.nSlots - count; s < g.nSlots; s++) {
            evict(g, s);
            g.slru.lend(s);
        }
        g.lentSlots = count;
    }

    /** Returns the slots layer {@code il}'s group lends, empty, to the cache; a no-op when it lends none. */
    void restoreSlots(int il) {
        ensureOpen();
        Group g = groups.get(layer(il).group);
        for (int s = g.nSlots - g.lentSlots; s < g.nSlots; s++) {
            g.slru.restore(s);
        }
        g.lentSlots = 0;
    }

    /** Slots layer {@code il}'s group lends out (0 when none). */
    int lentSlots(int il) {
        return groups.get(layer(il).group).lentSlots;
    }

    /** The group index of layer {@code il}; layers of one group share banks and slots. */
    int groupOf(int il) {
        return layer(il).group;
    }

    /** The experts of layer {@code il}. */
    int expertCount(int il) {
        return layer(il).map.length;
    }

    /**
     * The bank slot whose expert {@code e} of layer {@code il} the GPU reads in place, or -1: a fill still in
     * flight does not count until the layer's next {@link #beginStep} makes it visible.
     */
    int residentSlot(int il, int e) {
        Layer l = layer(il);
        return l.map[checkId(l, e)];
    }

    /** Reads {@code count} consecutive experts from {@code e0} of tensor {@code tensorIdx} of layer {@code il}. */
    void readExperts(int il, int tensorIdx, int e0, int count, ByteBuffer dst) throws IOException {
        Layer l = layer(il);
        long stride = groups.get(l.group).bankBytes[tensorIdx];
        dst.clear();
        data.readRange(l.plan.tensors().get(tensorIdx).name(), Math.multiplyExact((long) e0, stride),
                Math.toIntExact(Math.multiplyExact((long) count, stride)), dst);
    }

    /** Bytes of one expert of tensor {@code tensorIdx} of layer {@code il} (one slot of its bank). */
    long slotBytes(int il, int tensorIdx) {
        return groups.get(layer(il).group).bankBytes[tensorIdx];
    }

    /** The queue the expert dispatches run on. */
    ComputeQueue computeQueue() {
        return computeQueue;
    }

    /** The queue the fills run on (the compute queue when the backend has no asynchronous transfers). */
    ComputeQueue transferQueue() {
        return transferQueue;
    }

    // ================================================================== accessors for the block

    /** The layer plans the cache was opened over, in execution order. */
    public List<LayerPlan> layerPlans() {
        return plans;
    }

    /** True when layer {@code il} holds at least one slot. */
    public boolean serves(int il) {
        ensureOpen();
        return groups.get(layer(il).group).nSlots > 0;
    }

    /** The device map of layer {@code il}: per expert, the slot holding it, or -1. */
    public GpuBuffer mapBuffer(int il) {
        return layer(il).mapBuffer;
    }

    /** Bank of tensor {@code tensorIdx} of layer {@code il}'s group. */
    public GpuBuffer bank(int il, int tensorIdx) {
        Group g = groups.get(layer(il).group);
        if (tensorIdx < 0 || tensorIdx >= g.banks.length) {
            throw new IllegalArgumentException("tensor index " + tensorIdx + " outside [0, " + g.banks.length + ")");
        }
        return g.banks[tensorIdx];
    }

    /**
     * Tensor plan {@code tensorIdx} of layer {@code il}: the layer's own plan (the group's members share the
     * layout and the banks, and the names are the layer's).
     */
    public TensorPlan tensor(int il, int tensorIdx) {
        Layer l = layer(il);
        List<TensorPlan> tensors = l.plan.tensors();
        if (tensorIdx < 0 || tensorIdx >= tensors.size()) {
            throw new IllegalArgumentException("tensor index " + tensorIdx + " outside [0, " + tensors.size()
                    + ")");
        }
        return tensors.get(tensorIdx);
    }

    /** Tensor count of layer {@code il}'s expert set. */
    public int tensorCount(int il) {
        return groups.get(layer(il).group).tensors.size();
    }

    /** Slot count of layer {@code il}'s group (0 when the group is disabled). */
    public int slotCount(int il) {
        return groups.get(layer(il).group).nSlots;
    }

    /** The stored type of tensor {@code tensorIdx} of layer {@code il} (the banks upload bytes unchanged). */
    public GGMLType tensorType(int il, int tensorIdx) {
        return tensor(il, tensorIdx).type();
    }

    /** Hits / misses / filled bytes since open, for the close log and diagnostics. */
    public String stats() {
        long hits = 0;
        long misses = 0;
        long bytes = 0;
        for (Group g : groups) {
            hits += g.hits;
            misses += g.misses;
            bytes += g.filledBytes;
        }
        long probed = hits + misses;
        return "hits " + hits + ", misses " + misses + " (" + (probed == 0 ? 0.0 : 100.0 * hits / probed)
                + "%), " + bytes / (1024 * 1024) + " MiB filled";
    }

    private int totalSlots() {
        int sum = 0;
        for (Group g : groups) {
            sum += g.nSlots * g.plans.size();
        }
        return sum;
    }

    private Layer layer(int il) {
        Layer l = layerOfWeightLayer.get(il);
        if (l == null) {
            throw new IllegalArgumentException("weight layer " + il + " is not cached");
        }
        return l;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the MoE expert cache is closed");
        }
    }

    /** Fills every enabled layer's slots from the ranking, hottest first; uploads start immediately and the
     * experts become visible at the layer's first {@link #beginStep}. */
    private void preload(MoeCacheRanking ranking) {
        int nExpert = groups.get(0).nExpert();
        int maxLayer = -1;
        for (Layer l : layers) {
            maxLayer = Math.max(maxLayer, l.weightLayer);
        }
        if (ranking.rankingExperts() != nExpert || ranking.rankingLayers() < maxLayer + 1) {
            LOGGER.warn("MoE expert cache: ranking holds {} layers x {} experts, the cache serves {} layers x {}"
                    + " experts; ignored", ranking.rankingLayers(), ranking.rankingExperts(), maxLayer + 1, nExpert);
            return;
        }
        for (Layer l : layers) {
            Group g = groups.get(l.group);
            if (g.nSlots <= 0) {
                continue;
            }
            boolean[] admitted = new boolean[l.map.length];
            int assigned = 0;
            for (RankedExpert e : ranking.within(maxLayer + 1, l.map.length)) {
                if (assigned >= g.nSlots) {
                    break;
                }
                if (e.layer() != l.weightLayer || admitted[e.expert()]) {
                    continue;
                }
                int slot = g.slru.admit(layerPos(g, l), e.expert());
                if (slot < 0) {
                    break;
                }
                admitted[e.expert()] = true;
                assign(g, l, e.expert(), slot);
                MoeCacheSlru.Fill f = new MoeCacheSlru.Fill(layerPos(g, l), e.expert(), slot);
                try {
                    uploadFill(l, g, f, null);
                } catch (IOException ex) {
                    g.slru.fillStarted(slot);
                    throw new UncheckedIOException("MoE expert cache preload of layer " + l.weightLayer
                            + " failed: " + ex.getMessage(), ex);
                }
                g.slru.fillStarted(slot);
                l.fills.add(f);
                l.fillMarkers.add(sameQueue ? null : transferQueue.enqueueMarker());
                assigned++;
            }
        }
        transferQueue.flush();
        LOGGER.info("MoE expert cache: preloaded {} slot(s)", totalSlots());
    }

    /** Waits for both queues, then releases the events, the buffers and the provider, and logs the stats. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        LOGGER.info("MoE expert cache closed: {}", stats());
        RuntimeException failure = null;
        try {
            transferQueue.finish();
            computeQueue.finish();
        } catch (RuntimeException e) {
            failure = e;
        }
        for (Layer l : layers) {
            List<GpuEvent> distinct = new ArrayList<>();
            for (GpuEvent m : l.fillMarkers) {
                if (m != null && indexOf(distinct, m) < 0) {
                    distinct.add(m);
                }
            }
            for (GpuEvent m : distinct) {
                try {
                    m.close();
                } catch (RuntimeException e) {
                    failure = failure != null ? failure : e;
                }
            }
            try {
                l.mapBuffer.close();
            } catch (RuntimeException e) {
                failure = failure != null ? failure : e;
            }
        }
        for (Group g : groups) {
            for (GpuBuffer b : g.banks) {
                if (b != null) {
                    try {
                        b.close();
                    } catch (RuntimeException e) {
                        failure = failure != null ? failure : e;
                    }
                }
            }
        }
        try {
            data.close();
        } catch (RuntimeException e) {
            failure = failure != null ? failure : e;
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "MoeExpertCache{" + layers.size() + " layers, " + groups.size() + " groups, " + totalSlots()
                + " slots, " + bankBytes / (1024 * 1024) + " MiB}";
    }
}
