package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet.Entry;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Device slots for streamed layers ({@link LayerWeightSet}) of any architecture: the transfer pipeline behind
 * {@link StreamingLayerWeightSource}.
 *
 * <p><b>Layout.</b> Each set has a fixed slot ({@code slotOfSet}); a slot holds one set at a time. A slot owns a
 * pool of buffers; a set's tensor shards, ordered by size, take the pool's buffers in the same order, and pool
 * buffer {@code j} is as large as the {@code j}-th largest shard of any set assigned to the slot. Every set
 * therefore has its own {@link GpuWeight}s, built once over its slot's buffers, which stay valid for binding
 * whenever the set is resident.</p>
 *
 * <p><b>Event chain.</b> {@link #stage} enqueues the set's uploads on the transfer queue; the first upload waits on
 * the compute marker {@link #release} recorded after the last dispatch that read the slot's previous set, and a
 * transfer marker after the last upload completes the staging. {@link #gate} makes the compute queue wait on that
 * marker, so dispatches enqueued afterwards read the new weights while the host never blocks. With one queue
 * (no asynchronous transfer, or {@code -Dneuralfree.stream.sync=true}) queue order alone orders everything.</p>
 *
 * <p><b>Host side.</b> Tensors are read through the {@link WeightDataProvider} (RAM cache, else the mapped file) in
 * chunks of at most {@value #CHUNK_BYTES} bytes into one reusable staging buffer: writes copy their source at
 * enqueue. Tensors whose device type is F32 but whose stored type is not are decoded on the host per chunk.</p>
 *
 * <p>Owns its buffers and the provider. Not thread-safe.</p>
 */
final class LayerSlots implements AutoCloseable {

    /** Largest host chunk of one transfer. */
    public static final int CHUNK_BYTES = 64 << 20;
    /**
     * Bytes written to the transfer queue after which the next {@link #stage} first finishes it. A queue may keep
     * the host copy of every write until it is finished (OpenCL does), so this bounds the host memory those copies
     * hold when the queue is otherwise never finished.
     */
    public static final long MAX_UNFINISHED_WRITE_BYTES = 1L << 30;
    private static final Logger LOGGER = LogManager.getLogger(LayerSlots.class);
    private static final GpuEvent[] NO_EVENTS = new GpuEvent[0];

    private final ComputeQueue computeQueue;
    private final ComputeQueue transferQueue;
    private final boolean sameQueue;
    private final WeightDataProvider data;
    private final List<LayerWeightSet> sets;
    private final int[] slotOf;
    private final GpuBuffer[][] pools;
    private final List<Map<String, GpuWeight>> weights;
    private final int[] occupant;
    private final GpuEvent[] staged;
    private final GpuEvent[] released;
    private final long deviceBytes;
    private final ByteBuffer staging;
    private final ByteBuffer raw;
    private final float[] decoded;
    private long unfinishedWriteBytes;
    private boolean closed;

    /** Pool sizes per slot and the pool buffer of every shard of every set. */
    private static final class Plan {
        final long[][] poolBytes;
        final int[][] bufferOf;

        Plan(long[][] poolBytes, int[][] bufferOf) {
            this.poolBytes = poolBytes;
            this.bufferOf = bufferOf;
        }

        long totalBytes() {
            long total = 0;
            for (long[] slot : poolBytes) {
                for (long b : slot) {
                    total += b;
                }
            }
            return total;
        }
    }

    private LayerSlots(ComputeQueue computeQueue, ComputeQueue transferQueue, WeightDataProvider data,
                       List<LayerWeightSet> sets, int[] slotOf,
                       GpuBuffer[][] pools, Plan plan) {
        this.computeQueue = computeQueue;
        this.transferQueue = transferQueue;
        this.sameQueue = transferQueue == computeQueue;
        this.data = data;
        this.sets = sets;
        this.slotOf = slotOf;
        this.pools = pools;
        this.deviceBytes = plan.totalBytes();
        this.occupant = new int[pools.length];
        Arrays.fill(occupant, -1);
        this.staged = new GpuEvent[pools.length];
        this.released = new GpuEvent[pools.length];

        List<Map<String, GpuWeight>> ws = new ArrayList<>(sets.size());
        long stagingBytes = 0;
        long rawBytes = 0;
        long floats = 0;
        for (int i = 0; i < sets.size(); i++) {
            Map<String, GpuWeight> map = new LinkedHashMap<>();
            int flat = 0;
            for (Entry e : sets.get(i).entries()) {
                List<GpuBuffer> bufs = new ArrayList<>(e.shardCount());
                for (int k = 0; k < e.shardCount(); k++) {
                    bufs.add(pools[slotOf[i]][plan.bufferOf[i][flat++]]);
                }
                GpuWeight w = GpuWeight.sharded(e.name(), e.uploadType(), e.ne(), bufs, e.shardRowCounts());
                if (e.hadamard() != null) {
                    w = w.withHadamard(e.hadamard());
                }
                if (e.scale() != null) {
                    w = w.withScale(readScale(data, e));
                }
                map.put(e.name(), w);
                long rows = chunkRows(e);
                if (e.dequantize()) {
                    rawBytes = Math.max(rawBytes, rows * e.storedRowBytes());
                    floats = Math.max(floats, rows * e.cols());
                    stagingBytes = Math.max(stagingBytes, rows * e.uploadRowBytes());
                } else {
                    stagingBytes = Math.max(stagingBytes, rows * e.storedRowBytes());
                }
            }
            ws.add(Collections.unmodifiableMap(map));
        }
        this.weights = Collections.unmodifiableList(ws);
        this.staging = ByteBuffer.allocateDirect((int) stagingBytes).order(ByteOrder.LITTLE_ENDIAN);
        this.raw = rawBytes > 0 ? ByteBuffer.allocateDirect((int) rawBytes).order(ByteOrder.LITTLE_ENDIAN) : null;
        this.decoded = floats > 0 ? new float[(int) floats] : null;
    }

    /**
     * Allocates the slots of {@code sets}.
     *
     * @param backend      backend allocating the buffers and providing the transfer queue
     * @param computeQueue queue every dispatch reading the weights runs on
     * @param data         host tier reading the tensors; owned by the slots once allocation succeeds
     * @param sets         streamed layers; set {@code i} lives in slot {@code slotOfSet[i]}
     * @param slotOfSet    slot of every set, in {@code [0, slotCount)}
     * @param slotCount    number of slots
     * @throws IllegalArgumentException when the arguments disagree or a set is empty
     * @throws RuntimeException         from the backend when a buffer cannot be allocated (nothing stays allocated)
     */
    public static LayerSlots allocate(ComputeBackend backend, ComputeQueue computeQueue, WeightDataProvider data,
                                      List<LayerWeightSet> sets,
                                      int[] slotOfSet, int slotCount) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(computeQueue, "computeQueue");
        Objects.requireNonNull(data, "data");
        List<LayerWeightSet> setList =
                Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(sets, "sets")));
        int[] slotOf = slotOfSet.clone();
        Plan plan = plan(setList, slotOf, slotCount);
        for (LayerWeightSet set : setList) {
            for (String name : set.names()) {
                if (!data.containsKey(name)) {
                    throw new IllegalArgumentException("streamed tensor '" + name
                            + "' is not known to the weight data provider");
                }
            }
        }
        boolean forceSync = Boolean.getBoolean("neuralfree.stream.sync");
        boolean async = !forceSync && backend.capabilities().supportsAsyncTransfer();
        ComputeQueue tq = backend.transferQueue();
        ComputeQueue transfer = async && tq != null ? tq : computeQueue;

        GpuBuffer[][] pools = new GpuBuffer[slotCount][];
        try {
            for (int s = 0; s < slotCount; s++) {
                pools[s] = new GpuBuffer[plan.poolBytes[s].length];
                for (int j = 0; j < pools[s].length; j++) {
                    pools[s][j] = backend.allocateBuffer(plan.poolBytes[s][j]);
                    // Touch each buffer so a driver that commits device memory lazily fails here, at load.
                    if (plan.poolBytes[s][j] >= Integer.BYTES) {
                        computeQueue.enqueueFillBuffer(pools[s][j], 0, Integer.BYTES, 0);
                    }
                }
            }
            computeQueue.finish();
            LayerSlots slots = new LayerSlots(computeQueue, transfer, data, setList, slotOf, pools, plan);
            LOGGER.info(
                    "Layer weight streaming: {} layers in {} slots, {} MB on the device, async={}",
                    setList.size(), slotCount, plan.totalBytes() / (1024 * 1024), transfer != computeQueue);
            return slots;
        } catch (RuntimeException e) {
            for (GpuBuffer[] pool : pools) {
                if (pool == null) {
                    continue;
                }
                for (GpuBuffer b : pool) {
                    if (b != null) {
                        b.close();
                    }
                }
            }
            throw e;
        }
    }

    /**
     * Device bytes {@link #allocate} would take for this assignment.
     *
     * @throws IllegalArgumentException when the arguments disagree or a set is empty
     */
    public static long plannedBytes(List<LayerWeightSet> sets,
                                    int[] slotOfSet, int slotCount) {
        return plan(sets, slotOfSet, slotCount).totalBytes();
    }

    private static Plan plan(List<LayerWeightSet> sets, int[] slotOf,
                             int slotCount) {
        Objects.requireNonNull(sets, "sets");
        Objects.requireNonNull(slotOf, "slotOfSet");
        if (slotCount <= 0) {
            throw new IllegalArgumentException("slot count must be positive, got " + slotCount);
        }
        if (slotOf.length != sets.size()) {
            throw new IllegalArgumentException(sets.size() + " sets but " + slotOf.length + " slot assignments");
        }
        List<List<Long>> pool = new ArrayList<>(slotCount);
        for (int s = 0; s < slotCount; s++) {
            pool.add(new ArrayList<>());
        }
        int[][] bufferOf = new int[sets.size()][];
        for (int i = 0; i < sets.size(); i++) {
            LayerWeightSet set =
                    Objects.requireNonNull(sets.get(i), "set " + i);
            int s = slotOf[i];
            if (s < 0 || s >= slotCount) {
                throw new IllegalArgumentException("set " + set + " assigned to slot " + s + " outside [0, "
                        + slotCount + ")");
            }
            if (set.entries().isEmpty()) {
                throw new IllegalArgumentException("set " + set + " has no tensors");
            }
            List<Long> sizes = new ArrayList<>();
            for (Entry e : set.entries()) {
                for (int k = 0; k < e.shardCount(); k++) {
                    sizes.add(e.shardBytes(k));
                }
            }
            Integer[] order = new Integer[sizes.size()];
            for (int k = 0; k < order.length; k++) {
                order[k] = k;
            }
            Arrays.sort(order, (a, b) -> {
                int c = Long.compare(sizes.get(b), sizes.get(a));
                return c != 0 ? c : Integer.compare(a, b);
            });
            bufferOf[i] = new int[order.length];
            List<Long> slotPool = pool.get(s);
            for (int r = 0; r < order.length; r++) {
                bufferOf[i][order[r]] = r;
                long need = sizes.get(order[r]);
                if (r == slotPool.size()) {
                    slotPool.add(need);
                } else if (slotPool.get(r) < need) {
                    slotPool.set(r, need);
                }
            }
        }
        long[][] poolBytes = new long[slotCount][];
        for (int s = 0; s < slotCount; s++) {
            List<Long> p = pool.get(s);
            poolBytes[s] = new long[p.size()];
            for (int j = 0; j < poolBytes[s].length; j++) {
                poolBytes[s][j] = p.get(j);
            }
        }
        return new Plan(poolBytes, bufferOf);
    }

    /**
     * The output scale {@code w_s} of a streamed weight, read from its scalar {@code <base>.scale} tensor.
     *
     * @throws UncheckedIOException  when the scale cannot be read
     * @throws IllegalStateException when the scale is not finite
     */
    private static float readScale(WeightDataProvider data, Entry e) {
        String name = e.scale().name();
        int bytes = Math.toIntExact(e.scale().sizeInBytes());
        ByteBuffer buf = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        try {
            data.readRange(name, 0, bytes, buf);
        } catch (IOException ex) {
            throw new UncheckedIOException("cannot read output scale " + name + ": " + ex.getMessage(), ex);
        }
        float[] v = new float[1];
        CpuBlockDecoders.decodeElements(e.scale().type(), buf, 0L, v, 0, 1);
        if (!Float.isFinite(v[0])) {
            throw new IllegalStateException("tensor " + name + " holds a non-finite output scale " + v[0]);
        }
        return v[0];
    }

    /** Rows per host chunk of one tensor: at most {@value #CHUNK_BYTES} bytes, at least one row. */
    private static int chunkRows(Entry e) {
        long rowBytes = Math.max(e.storedRowBytes(), e.uploadRowBytes());
        if (rowBytes > Integer.MAX_VALUE) {
            throw new IllegalStateException("tensor " + e.name() + " has rows of " + rowBytes
                    + " bytes, more than one host transfer holds");
        }
        long rows = Math.max(1, CHUNK_BYTES / rowBytes);
        long maxShard = 0;
        for (int k = 0; k < e.shardCount(); k++) {
            maxShard = Math.max(maxShard, e.shardRows(k));
        }
        return (int) Math.min(rows, maxShard);
    }

    public int setCount() { return sets.size(); }
    public int slotCount() { return pools.length; }
    /** The streamed layer of set {@code i}. */
    public LayerWeightSet set(int i) { return sets.get(checkSet(i)); }
    /** Slot of set {@code i}. */
    public int slotOf(int i) { return slotOf[checkSet(i)]; }
    /** Set held (or being staged) by a slot, or -1 when the slot is empty. */
    public int occupant(int slot) { return occupant[slot]; }
    /** True when set {@code i} is in its slot, possibly with its uploads still in flight. */
    public boolean isResident(int i) { return occupant[slotOf[checkSet(i)]] == i; }
    /** Bytes of all slot buffers. */
    public long deviceBytes() { return deviceBytes; }
    /** True when uploads run on their own transfer queue. */
    public boolean asyncTransfers() { return !sameQueue; }

    /**
     * Device tensors of set {@code i} by stored name (unmodifiable); valid whenever the set is resident and
     * gated.
     */
    public Map<String, GpuWeight> weights(int i) {
        return weights.get(checkSet(i));
    }

    /**
     * Uploads set {@code i} into its slot, evicting the slot's previous set; a no-op when it is resident. The first
     * upload waits on the slot's release marker; the uploads are flushed to the device before returning. When more
     * than {@link #MAX_UNFINISHED_WRITE_BYTES} were written since the transfer queue was last finished, it is
     * finished first: its pending uploads wait only on compute markers that were flushed when they were enqueued.
     *
     * @throws UncheckedIOException when a tensor cannot be read (the slot is then empty)
     */
    public void stage(int i) {
        ensureOpen();
        int s = slotOf[checkSet(i)];
        if (occupant[s] == i) {
            return;
        }
        if (unfinishedWriteBytes >= MAX_UNFINISHED_WRITE_BYTES) {
            transferQueue.finish();
            unfinishedWriteBytes = 0;
        }
        occupant[s] = -1;
        if (staged[s] != null) {
            staged[s].close();
            staged[s] = null;
        }
        GpuEvent wait = released[s];
        released[s] = null;
        GpuEvent[] waits = NO_EVENTS;
        if (wait != null) {
            computeQueue.flush();
            waits = new GpuEvent[]{wait};
        }
        LayerWeightSet set = sets.get(i);
        Map<String, GpuWeight> dev = weights.get(i);
        try {
            for (Entry e : set.entries()) {
                GpuWeight w = dev.get(e.name());
                for (int k = 0; k < e.shardCount(); k++) {
                    waits = upload(e, w.shard(k), w.shardRowStart(k), e.shardRows(k), waits);
                }
            }
        } catch (IOException ex) {
            throw new UncheckedIOException("streaming weight layer " + set.stack() + "." + set.weightLayer()
                    + " failed: " + ex.getMessage(), ex);
        } finally {
            if (wait != null) {
                if (waits.length == 0) {
                    wait.close();
                } else {
                    released[s] = wait;
                }
            }
        }
        if (!sameQueue) {
            staged[s] = transferQueue.enqueueMarker();
        }
        transferQueue.flush();
        occupant[s] = i;
    }

    private GpuEvent[] upload(Entry e, GpuBuffer buf, long rowStart, long rows, GpuEvent[] waits)
            throws IOException {
        long storedRow = e.storedRowBytes();
        long deviceRow = e.uploadRowBytes();
        int chunk = chunkRows(e);
        GpuEvent[] w = waits;
        for (long done = 0; done < rows; ) {
            int n = (int) Math.min(chunk, rows - done);
            long srcOff = (rowStart + done) * storedRow;
            int srcBytes = (int) (n * storedRow);
            int bytes;
            if (e.dequantize()) {
                raw.clear();
                data.readRange(e.name(), srcOff, srcBytes, raw);
                int elems = (int) (n * e.cols());
                CpuBlockDecoders.decodeElements(e.source().type(), raw, 0, decoded, 0, elems);
                staging.clear();
                staging.asFloatBuffer().put(decoded, 0, elems);
                bytes = elems * Float.BYTES;
            } else {
                staging.clear();
                data.readRange(e.name(), srcOff, srcBytes, staging);
                bytes = srcBytes;
            }
            GpuEvent evt = transferQueue.enqueueWriteBuffer(buf, done * deviceRow, bytes, staging, w);
            if (evt != null) {
                evt.close();
            }
            unfinishedWriteBytes += bytes;
            w = NO_EVENTS;
            done += n;
        }
        return w;
    }

    /**
     * Makes the compute queue wait for set {@code i}'s uploads: dispatches enqueued afterwards read its weights.
     *
     * @throws IllegalStateException when the set is not in its slot
     */
    public void gate(int i) {
        ensureOpen();
        int s = slotOf[checkSet(i)];
        if (occupant[s] != i) {
            throw new IllegalStateException("streamed layer " + sets.get(i) + " is not staged; slot " + s
                    + " holds "
                    + (occupant[s] < 0 ? "nothing" : sets.get(occupant[s]).toString()));
        }
        GpuEvent st = staged[s];
        if (st == null) {
            return;
        }
        staged[s] = null;
        try {
            GpuEvent gate = computeQueue.enqueueMarker(st);
            if (gate != null) {
                gate.close();
            }
        } finally {
            st.close();
        }
    }

    /**
     * Records that every dispatch reading set {@code i}'s weights has been enqueued: the next upload into its slot
     * waits for them.
     */
    public void release(int i) {
        ensureOpen();
        int s = slotOf[checkSet(i)];
        if (sameQueue) {
            return;
        }
        GpuEvent marker = computeQueue.enqueueMarker();
        if (released[s] != null) {
            released[s].close();
        }
        released[s] = marker;
    }

    private int checkSet(int i) {
        if (i < 0 || i >= sets.size()) {
            throw new IndexOutOfBoundsException("streamed set " + i + " outside [0, " + sets.size() + ")");
        }
        return i;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("layer weight slots are closed");
        }
    }

    /** Waits for both queues, then releases the events, the slot buffers and the provider. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        try {
            transferQueue.finish();
            computeQueue.finish();
        } catch (RuntimeException e) {
            failure = e;
        }
        for (int s = 0; s < pools.length; s++) {
            if (staged[s] != null) {
                staged[s].close();
                staged[s] = null;
            }
            if (released[s] != null) {
                released[s].close();
                released[s] = null;
            }
            occupant[s] = -1;
            for (GpuBuffer b : pools[s]) {
                b.close();
            }
        }
        data.close();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "LayerSlots{" + sets.size() + " layers, " + pools.length + " slots, " + deviceBytes + " B}";
    }
}
