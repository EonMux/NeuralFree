package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Prefill expert prefetch over a {@link MoeExpertCache} (Avalon {@code GGML_SCHED_PREFETCH_EXPERTS} with the cache's
 * lent zones): a prefill-sized step demands virtually every expert of a layer, so instead of computing the cache's
 * misses on the host, the experts the cache does not hold are uploaded into slots borrowed from the cache, and the
 * routed products read every expert from VRAM in one dispatch: the cached experts in place in their bank slots, the
 * others in the borrowed zone, through one prefill map per layer (Avalon's table of expert addresses).
 *
 * <p>A step is prefetched when it is prefill-sized ({@link #prefillSized}) and routes more than the pool's hosted
 * row bound (Avalon {@code GGML_MOE_HOSTED_MAX_ROWS}: smaller blocks keep the host misses, which copy no weight).</p>
 *
 * <p><b>Zones.</b> A zone is {@code n_expert} consecutive slots at the tail of a cache group's banks, one whole
 * layer stack: expert {@code e} of the layer it holds sits at slot {@code first + e} of every bank. The pool asks
 * for {@code zones} zones per group, fewer when the group's slots hold fewer layer stacks, none when they hold not
 * even one (that group's layers keep the host misses). The zones are lent at the first prefill-sized step of a
 * group ({@link MoeExpertCache#lendSlots}, which evicts their experts) and given back at its first smaller step
 * ({@link MoeExpertCache#restoreSlots}), so decode keeps the cache's whole capacity.</p>
 *
 * <p><b>Step protocol</b> (the cached pass runs it around the cache's own):</p>
 * <ol>
 *   <li>{@link #begin}, before {@link MoeExpertCache#beginStep}: says whether the step is prefetched, lending the
 *       group's zones or giving them back.</li>
 *   <li>{@link #cover}, after {@link MoeExpertCache#beginStep}: the layer takes the zone of its position in the
 *       group's execution order (round robin over the zones); the step's missing experts the zone does not hold yet
 *       are uploaded on the transfer queue; the next layers of the group (one per other zone) are staged: every
 *       expert the cache does not hold is uploaded ahead, behind the compute that last read the zone, so the copy
 *       overlaps this layer's compute; the compute queue waits for the layer's uploads, and the layer's prefill map
 *       is written. The caller dispatches the products once over the banks and that map.</li>
 *   <li>{@link #end}, after the dispatch: a compute marker guards the zone against the next upload into it.</li>
 * </ol>
 *
 * <p>Uploads copy runs of consecutive experts, one read and one write per run and tensor (Avalon's runs of copied
 * experts). The staging skips what the cache holds at staging time; an expert evicted before its step is uploaded at
 * the step, and one admitted meanwhile is read in place, so a skipped copy is never missing. With no separate
 * transfer queue every order is queue order and the markers are omitted. Owns its prefill maps; the zones and the
 * expert bytes are the cache's. Not thread-safe.</p>
 */
public final class MoeExpertPrefetch implements AutoCloseable {

    /** A step prefetches when its routed pairs reach this many per expert (Avalon {@code ggml-backend.cpp:2504}). */
    public static final int PAIRS_PER_EXPERT = 2;
    /** Largest host read of one upload. */
    static final long CHUNK_BYTES = 16L << 20;

    private static final Logger LOGGER = LogManager.getLogger(MoeExpertPrefetch.class);

    /** One zone: its first bank slot, the layer whose experts it holds and the markers guarding it. */
    private static final class Zone {
        final int firstSlot;
        /** Weight layer whose experts the zone holds, or -1. */
        int holder = -1;
        /** Experts of {@link #holder} uploaded into the zone. */
        boolean[] staged;
        /** Transfer marker after the zone's latest uploads; null when none is pending. */
        GpuEvent uploadMarker;
        /** Compute marker after the latest read of the zone (or the lend); null when none is pending. */
        GpuEvent freeMarker;

        Zone(int firstSlot) {
            this.firstSlot = firstSlot;
        }
    }

    /** The zones of one cache group and the group's layers in execution order. */
    private static final class Group {
        final int nExpert;
        final List<Integer> order = new ArrayList<>();
        final Zone[] zones;
        boolean lent;

        Group(int nExpert, int nSlots, int wanted) {
            this.nExpert = nExpert;
            int n = Math.min(wanted, nSlots / nExpert);
            this.zones = new Zone[n];
            for (int j = 0; j < n; j++) {
                zones[j] = new Zone(nSlots - (n - j) * nExpert);
            }
        }
    }

    private final MoeExpertCache cache;
    private final ComputeQueue computeQueue;
    private final ComputeQueue transferQueue;
    private final boolean sameQueue;
    private final int hostedMaxRows;
    private final List<Group> groups;
    private final Map<Integer, Group> groupOfLayer;
    private final Map<Integer, Integer> positionOfLayer;
    private final Map<Integer, Integer> tensorsOfLayer;
    private final Map<Integer, GpuBuffer> maps;
    private final ByteBuffer staging;
    private final ByteBuffer mapStaging;
    private long prefillSteps;
    private long lends;
    private long stagedExperts;
    private long stepExperts;
    private long residentReads;
    private long copiedBytes;
    private boolean closed;

    private MoeExpertPrefetch(MoeExpertCache cache, int hostedMaxRows, List<Group> groups,
                              Map<Integer, Group> groupOfLayer,
                              Map<Integer, Integer> positionOfLayer, Map<Integer, Integer> tensorsOfLayer,
                              Map<Integer, GpuBuffer> maps, long stagingBytes, int largestNExpert) {
        this.cache = cache;
        this.computeQueue = cache.computeQueue();
        this.transferQueue = cache.transferQueue();
        this.sameQueue = computeQueue == transferQueue;
        this.hostedMaxRows = hostedMaxRows;
        this.groups = groups;
        this.groupOfLayer = groupOfLayer;
        this.positionOfLayer = positionOfLayer;
        this.tensorsOfLayer = tensorsOfLayer;
        this.maps = maps;
        this.staging = ByteBuffer.allocateDirect(Math.toIntExact(stagingBytes)).order(ByteOrder.LITTLE_ENDIAN);
        this.mapStaging = ByteBuffer.allocateDirect(4 * largestNExpert).order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Builds the pool over the cache's layers: per group, {@code zones} zones of {@code n_expert} slots each (fewer
     * when the group's slots hold fewer layer stacks), and one prefill map per layer of a group with zones.
     *
     * @param backend   backend owning the prefill maps
     * @param cache     cache whose tail slots the zones borrow and whose host tier the uploads read
     * @param plans     the cache's layer plans, in execution order
     * @param zones     zones per group (layer stacks in flight; Avalon's slot count)
     * @param hostedMaxRows routed rows up to which a step keeps the host misses (0: every prefill-sized step)
     * @param modelName model name for the log line
     * @throws IllegalArgumentException for a zone count below one, a negative row bound or an empty plan list
     */
    public static MoeExpertPrefetch open(ComputeBackend backend, MoeExpertCache cache,
                                         List<MoeExpertCache.LayerPlan> plans, int zones, int hostedMaxRows,
                                         String modelName) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(cache, "cache");
        if (plans == null || plans.isEmpty()) {
            throw new IllegalArgumentException("a MoE expert prefetch pool needs at least one layer plan");
        }
        if (zones < 1) {
            throw new IllegalArgumentException("prefetch zone count must be at least 1, got " + zones);
        }
        if (hostedMaxRows < 0) {
            throw new IllegalArgumentException("hosted row bound must be >= 0, got " + hostedMaxRows);
        }
        Map<Integer, Group> byIndex = new HashMap<>();
        List<Group> groups = new ArrayList<>();
        Map<Integer, Group> groupOfLayer = new HashMap<>();
        Map<Integer, Integer> positionOfLayer = new HashMap<>();
        Map<Integer, Integer> tensorsOfLayer = new HashMap<>();
        long stagingBytes = 4;
        int largestNExpert = 1;
        for (MoeExpertCache.LayerPlan p : plans) {
            int wl = p.weightLayer();
            Group g = byIndex.get(cache.groupOf(wl));
            if (g == null) {
                g = new Group(cache.expertCount(wl), cache.slotCount(wl), zones);
                byIndex.put(cache.groupOf(wl), g);
                groups.add(g);
            }
            positionOfLayer.put(wl, g.order.size());
            g.order.add(wl);
            groupOfLayer.put(wl, g);
            tensorsOfLayer.put(wl, cache.tensorCount(wl));
            largestNExpert = Math.max(largestNExpert, g.nExpert);
            for (int i = 0; i < cache.tensorCount(wl); i++) {
                long stride = cache.slotBytes(wl, i);
                stagingBytes = Math.max(stagingBytes, chunkExperts(stride) * stride);
            }
        }
        Map<Integer, GpuBuffer> maps = allocateMaps(backend, groupOfLayer);
        MoeExpertPrefetch pool = new MoeExpertPrefetch(cache, hostedMaxRows, groups, groupOfLayer, positionOfLayer,
                tensorsOfLayer, maps, stagingBytes, largestNExpert);
        LOGGER.info("{}: MoE expert prefetch: {}", modelName, pool.describe());
        return pool;
    }

    /** One prefill map per layer of a group with zones; every buffer is closed again on failure. */
    private static Map<Integer, GpuBuffer> allocateMaps(ComputeBackend backend, Map<Integer, Group> groupOfLayer) {
        Map<Integer, GpuBuffer> maps = new HashMap<>();
        try {
            for (Map.Entry<Integer, Group> e : groupOfLayer.entrySet()) {
                if (e.getValue().zones.length > 0) {
                    maps.put(e.getKey(), backend.allocateBuffer(4L * e.getValue().nExpert));
                }
            }
            return maps;
        } catch (RuntimeException failure) {
            for (GpuBuffer b : maps.values()) {
                try {
                    b.close();
                } catch (RuntimeException suppressed) {
                    failure.addSuppressed(suppressed);
                }
            }
            throw failure;
        }
    }

    /** Experts per upload of a tensor whose experts take {@code stride} bytes each: at least one. */
    static long chunkExperts(long stride) {
        return Math.max(1L, CHUNK_BYTES / stride);
    }

    /**
     * True when a step of {@code nTokens} tokens and {@code pairs} routed pairs over {@code nExpert} experts is
     * prefill-sized: more than one token and at least {@link #PAIRS_PER_EXPERT} pairs per expert.
     */
    public static boolean prefillSized(int nTokens, int pairs, int nExpert) {
        return nTokens > 1 && pairs >= (long) PAIRS_PER_EXPERT * nExpert;
    }

    // ================================================================== step protocol

    /**
     * Starts one step of layer {@code il}, before the cache's {@link MoeExpertCache#beginStep}: a prefill-sized
     * step above the hosted row bound, of a layer whose group has zones, is prefetched (the group's zones are lent
     * when they are not yet); any other step gives the group's lent zones back.
     *
     * @return true when the step is prefetched: the caller runs {@link #cover} and {@link #end} around its dispatch
     * @throws IllegalArgumentException for a layer the pool does not know
     */
    public boolean begin(int il, int nTokens, int pairs) {
        ensureOpen();
        Group g = group(il);
        if (g.zones.length == 0) {
            return false;
        }
        if (!prefillSized(nTokens, pairs, g.nExpert) || pairs <= hostedMaxRows) {
            release(g, il);
            return false;
        }
        if (!g.lent) {
            lend(g, il);
        }
        return true;
    }

    /**
     * Readies the prefetched step of layer {@code il} after the cache's {@link MoeExpertCache#beginStep}: uploads
     * the step's missing experts the layer's zone does not hold, stages the group's next layers into the other
     * zones, makes the compute queue wait for the layer's uploads, and writes the layer's prefill map: per expert,
     * its bank slot when the cache holds it, else its zone slot when the zone holds it, else -1. {@code ids} holds
     * the routed ids in pair order, {@code missPairs} the pair indices of the step's misses.
     *
     * @return the prefill map the products read instead of the cache's map
     * @throws IllegalStateException    when {@link #begin} did not prefetch the step
     * @throws IllegalArgumentException for a routed id outside the experts
     * @throws UncheckedIOException     when an expert's bytes cannot be read
     */
    public GpuBuffer cover(int il, int[] ids, int[] missPairs, int missCount) {
        ensureOpen();
        Group g = group(il);
        if (!g.lent) {
            throw new IllegalStateException("layer " + il + ": the step is not prefetched; call begin first");
        }
        Zone z = claim(g, il);
        boolean[] missing = new boolean[g.nExpert];
        for (int q = 0; q < missCount; q++) {
            missing[checkExpert(il, g, ids[missPairs[q]])] = true;
        }
        stepExperts += upload(il, z, missing);
        stageAhead(g, il);
        gate(z);
        residentReads += writeMap(il, g, z, ids);
        prefillSteps++;
        return maps.get(il);
    }

    /** Ends the prefetched step of layer {@code il} after its dispatch: guards the zone the dispatch read. */
    public void end(int il) {
        ensureOpen();
        Group g = group(il);
        if (!g.lent || sameQueue) {
            return;
        }
        Zone z = zoneOf(g, il);
        computeQueue.flush();
        closeEvent(z.freeMarker);
        z.freeMarker = computeQueue.enqueueMarker();
    }

    /** True when layer {@code il}'s group currently borrows its zones from the cache. */
    public boolean lent(int il) {
        ensureOpen();
        return group(il).lent;
    }

    /** Zones of layer {@code il}'s group (0 when the group's slots hold no layer stack). */
    public int zoneCount(int il) {
        ensureOpen();
        return group(il).zones.length;
    }

    /** Prefetched steps, lends, experts copied ahead and at step time, resident reads and bytes copied. */
    public String stats() {
        return "prefill steps " + prefillSteps + ", lends " + lends + ", experts copied " + (stagedExperts
                + stepExperts) + " (" + stagedExperts + " staged ahead, " + stepExperts + " at step time), "
                + residentReads + " read in place, " + copiedBytes / (1024 * 1024) + " MiB copied";
    }

    /** Copies staged ahead so far. */
    public long stagedExperts() {
        return stagedExperts;
    }

    /** Copies made at step time (the step's misses its zone did not hold yet) so far. */
    public long stepExperts() {
        return stepExperts;
    }

    /** Demanded experts the prefetched steps read in place from the cache so far. */
    public long residentReads() {
        return residentReads;
    }

    // ================================================================== lending

    /** Lends the group's zones: the cache evicts their experts, and every zone waits for the compute so far. */
    private void lend(Group g, int il) {
        cache.lendSlots(il, g.zones.length * g.nExpert);
        for (Zone z : g.zones) {
            z.holder = -1;
            z.staged = null;
            if (!sameQueue) {
                computeQueue.flush();
                closeEvent(z.freeMarker);
                z.freeMarker = computeQueue.enqueueMarker();
            }
        }
        g.lent = true;
        lends++;
    }

    /**
     * Gives the group's zones back to the cache. A later fill into a zone slot is enqueued behind the compute
     * marker of its own step, after every read of the zone, and behind the zone's uploads on the transfer queue.
     */
    private void release(Group g, int il) {
        if (!g.lent) {
            return;
        }
        cache.restoreSlots(il);
        for (Zone z : g.zones) {
            z.holder = -1;
            z.staged = null;
            closeEvent(z.uploadMarker);
            z.uploadMarker = null;
            closeEvent(z.freeMarker);
            z.freeMarker = null;
        }
        g.lent = false;
    }

    // ================================================================== zones and staging

    private Zone zoneOf(Group g, int il) {
        return g.zones[positionOfLayer.get(il) % g.zones.length];
    }

    /** The zone of layer {@code il}, taken over (its staged set cleared) when it holds another layer. */
    private Zone claim(Group g, int il) {
        Zone z = zoneOf(g, il);
        if (z.holder != il) {
            z.holder = il;
            z.staged = new boolean[g.nExpert];
        }
        return z;
    }

    /**
     * Stages the group's next layers after {@code il}, one per other zone: every expert the cache does not hold.
     * A layer within {@code zones - 1} positions ahead never shares a zone with {@code il} or with another staged
     * layer, so staging never displaces a layer whose step has not run.
     */
    private void stageAhead(Group g, int il) {
        int pos = positionOfLayer.get(il);
        for (int d = 1; d < g.zones.length && pos + d < g.order.size(); d++) {
            int wl = g.order.get(pos + d);
            Zone z = claim(g, wl);
            boolean[] wanted = new boolean[g.nExpert];
            for (int e = 0; e < g.nExpert; e++) {
                wanted[e] = cache.residentSlot(wl, e) < 0;
            }
            stagedExperts += upload(wl, z, wanted);
        }
    }

    /**
     * Uploads the experts {@code wanted} that zone {@code z} does not hold yet, as runs of consecutive experts,
     * behind the zone's free marker; marks them staged.
     *
     * @return the experts copied
     */
    private int upload(int il, Zone z, boolean[] wanted) {
        int copied = 0;
        GpuEvent wait = z.freeMarker;
        GpuEvent[] waits = wait != null ? new GpuEvent[]{wait} : new GpuEvent[0];
        try {
            int e = 0;
            while (e < wanted.length) {
                if (!wanted[e] || z.staged[e]) {
                    e++;
                    continue;
                }
                int e0 = e;
                while (e < wanted.length && wanted[e] && !z.staged[e]) {
                    e++;
                }
                uploadRun(il, z, e0, e, waits);
                for (int x = e0; x < e; x++) {
                    z.staged[x] = true;
                }
                copied += e - e0;
            }
        } catch (IOException ex) {
            throw new UncheckedIOException("MoE expert prefetch of layer " + il + " failed: " + ex.getMessage(), ex);
        }
        if (copied > 0) {
            z.freeMarker = null;
            closeEvent(wait);
            if (!sameQueue) {
                transferQueue.flush();
                closeEvent(z.uploadMarker);
                z.uploadMarker = transferQueue.enqueueMarker();
            }
        }
        return copied;
    }

    /** Uploads experts {@code [e0, e1)} of every expert tensor of layer {@code il} into the zone's slots. */
    private void uploadRun(int il, Zone z, int e0, int e1, GpuEvent[] waits) throws IOException {
        int tensors = tensorsOfLayer.get(il);
        for (int i = 0; i < tensors; i++) {
            long stride = cache.slotBytes(il, i);
            int chunk = (int) Math.min(chunkExperts(stride), Integer.MAX_VALUE);
            GpuBuffer bank = cache.bank(il, i);
            for (int c0 = e0; c0 < e1; c0 += chunk) {
                int count = Math.min(chunk, e1 - c0);
                long bytes = count * stride;
                cache.readExperts(il, i, c0, count, staging);
                GpuEvent evt = transferQueue.enqueueWriteBuffer(bank, (long) (z.firstSlot + c0) * stride, bytes,
                        staging, waits);
                closeEvent(evt);
                copiedBytes += bytes;
            }
        }
    }

    /** Makes the compute queue wait for the zone's pending uploads. */
    private void gate(Zone z) {
        if (sameQueue || z.uploadMarker == null) {
            return;
        }
        GpuEvent gate = computeQueue.enqueueMarker(z.uploadMarker);
        closeEvent(gate);
    }

    /**
     * Writes the prefill map of layer {@code il}: the bank slot of each expert the cache holds, else its zone slot
     * when the zone holds it, else -1.
     *
     * @return the distinct demanded experts read in place from the cache
     */
    private int writeMap(int il, Group g, Zone z, int[] ids) {
        int[] entries = new int[g.nExpert];
        for (int e = 0; e < g.nExpert; e++) {
            int slot = cache.residentSlot(il, e);
            entries[e] = slot >= 0 ? slot : z.staged[e] ? z.firstSlot + e : -1;
        }
        boolean[] seen = new boolean[g.nExpert];
        int resident = 0;
        for (int id : ids) {
            int e = checkExpert(il, g, id);
            if (!seen[e]) {
                seen[e] = true;
                resident += cache.residentSlot(il, e) >= 0 ? 1 : 0;
            }
        }
        mapStaging.clear();
        for (int v : entries) {
            mapStaging.putInt(v);
        }
        closeEvent(computeQueue.enqueueWriteBuffer(maps.get(il), 0, 4L * g.nExpert, mapStaging));
        return resident;
    }

    // ================================================================== helpers

    private Group group(int il) {
        Group g = groupOfLayer.get(il);
        if (g == null) {
            throw new IllegalArgumentException("weight layer " + il + " is not prefetched");
        }
        return g;
    }

    private static int checkExpert(int il, Group g, int e) {
        if (e < 0 || e >= g.nExpert) {
            throw new IllegalArgumentException("layer " + il + ": routed expert " + e + " outside [0, " + g.nExpert
                    + ")");
        }
        return e;
    }

    private static void closeEvent(GpuEvent evt) {
        if (evt != null) {
            evt.close();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the MoE expert prefetch pool is closed");
        }
    }

    /** Waits for both queues, then releases the markers and the prefill maps and logs the stats. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        LOGGER.info("MoE expert prefetch closed: {}", stats());
        RuntimeException failure = null;
        try {
            transferQueue.finish();
            computeQueue.finish();
        } catch (RuntimeException e) {
            failure = e;
        }
        List<Runnable> releases = new ArrayList<>();
        for (Group g : groups) {
            for (Zone z : g.zones) {
                GpuEvent upload = z.uploadMarker;
                GpuEvent free = z.freeMarker;
                z.uploadMarker = null;
                z.freeMarker = null;
                releases.add(() -> closeEvent(upload));
                releases.add(() -> closeEvent(free));
            }
        }
        for (GpuBuffer map : maps.values()) {
            releases.add(map::close);
        }
        for (Runnable r : releases) {
            try {
                r.run();
            } catch (RuntimeException e) {
                failure = failure != null ? failure : e;
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private String describe() {
        StringBuilder b = new StringBuilder();
        for (Group g : groups) {
            if (b.length() > 0) {
                b.append("; ");
            }
            b.append(g.order.size()).append(" layer(s) x ").append(g.nExpert).append(" experts: ")
                    .append(g.zones.length).append(" zone(s)");
            if (g.zones.length == 0) {
                b.append(" (the cache holds less than one layer stack; misses stay on the host)");
            }
        }
        return b.append(sameQueue ? ", uploads on the compute queue" : "").toString();
    }

    @Override
    public String toString() {
        return "MoeExpertPrefetch{" + describe() + "}";
    }
}
