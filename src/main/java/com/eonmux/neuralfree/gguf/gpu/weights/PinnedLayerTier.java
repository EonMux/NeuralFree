package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.gguf.HostMemory;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet.Entry;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Page-locked host copies of the streamed layers, from which {@link LayerSlots} stages a layer with one device copy
 * per shard ({@link ComputeQueue#enqueueCopyBuffer}): the device's copy engine reads the pinned memory directly, so a
 * stage costs no host copy and no host time beyond enqueueing the copies. Every set gets one host-mapped buffer
 * ({@link ComputeBackend#allocateHostStagingBuffer}, system memory) holding its shards in upload form, each at a
 * {@value #ALIGNMENT}-byte boundary, filled once at load from the host tier.
 *
 * <p>The page-locked bytes of every open tier of the process are bounded by {@value #PROPERTY}: the default
 * ({@code -1}) is half of the RAM the model file leaves free and at most half of the RAM the host has available when
 * the tier opens ({@link HostMemory#availableBytes()}: page-locked memory cannot be swapped, so the tier must not
 * crowd out the other processes), nothing when the model is as large as RAM or the RAM is unknown; {@code 0} turns
 * the tier off; a positive value is the bound in MiB. A tier is opened only when the backend allocates host-mapped
 * memory, the streamed layers outnumber the device slots (otherwise each layer is staged once) and every set fits the
 * bound. A device that shares the host's memory gets the tier as well: its slots live in the same RAM, so the copies
 * hold the streamed layers a second time there, but a stage is then one copy the device runs instead of a host read
 * and write on the forward thread. A backend that cannot allocate or fill the memory leaves the layers streaming from
 * the host tier. Owns its buffers. Not thread-safe.</p>
 */
final class PinnedLayerTier implements AutoCloseable {

    /** Bound of the page-locked copies of streamed layers, in MiB; -1 = automatic, 0 = off. */
    static final String PROPERTY = "neuralfree.stream.pinned.mb";
    /** The automatic bound. */
    static final long AUTO = -1L;
    /** Alignment of every shard within its set's buffer. */
    static final long ALIGNMENT = 256L;

    private static final long MIB = 1024L * 1024L;
    private static final Logger LOGGER = LogManager.getLogger(PinnedLayerTier.class);
    /** Page-locked bytes held by the open tiers of the process. */
    private static final AtomicLong PINNED = new AtomicLong();

    private final List<HostMappedBuffer> buffers;
    private final long[][] offsets;
    private final long bytes;
    private boolean closed;

    private PinnedLayerTier(List<HostMappedBuffer> buffers, long[][] offsets, long bytes) {
        this.buffers = buffers;
        this.offsets = offsets;
        this.bytes = bytes;
    }

    /**
     * The bound {@value #PROPERTY} sets, in MiB, or {@link #AUTO} when it is unset.
     *
     * @throws IllegalArgumentException when the value is not an integer or is below -1
     */
    static long configuredMb() {
        String v = System.getProperty(PROPERTY);
        if (v == null || v.trim().isEmpty()) {
            return AUTO;
        }
        long mb;
        try {
            mb = Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(PROPERTY + "=" + v + " is not a size in MiB (-1 = automatic,"
                    + " 0 = off)", e);
        }
        if (mb < AUTO) {
            throw new IllegalArgumentException(PROPERTY + "=" + v + " is below -1 (-1 = automatic, 0 = off)");
        }
        return mb;
    }

    /**
     * The page-locked bytes a new tier may take.
     *
     * @param configuredMb      {@link #AUTO}, 0 (off) or a bound in MiB for every tier of the process
     * @param modelBytes        storage bytes of the model file
     * @param totalRamBytes     physical RAM, or &lt;= 0 when unknown
     * @param availableRamBytes RAM the host can hand out now ({@link HostMemory#availableBytes()}), which already
     *                          excludes what the other tiers pinned; &lt;= 0 when unknown, then only the total bounds
     *                          the automatic budget
     * @param pinnedElsewhere   page-locked bytes the process's other tiers hold
     */
    static long budgetBytes(long configuredMb, long modelBytes, long totalRamBytes, long availableRamBytes,
                            long pinnedElsewhere) {
        long elsewhere = Math.max(0L, pinnedElsewhere);
        if (configuredMb == AUTO) {
            if (totalRamBytes <= 0 || modelBytes >= totalRamBytes) {
                return 0L;
            }
            long processBound = (totalRamBytes - modelBytes) / 2 - elsewhere;
            return Math.max(0L, availableRamBytes > 0 ? Math.min(processBound, availableRamBytes / 2) : processBound);
        }
        if (configuredMb < 0) {
            throw new IllegalArgumentException(PROPERTY + " must be -1, 0 or a size in MiB, got " + configuredMb);
        }
        long bound = configuredMb > Long.MAX_VALUE / MIB ? Long.MAX_VALUE : configuredMb * MIB;
        return Math.max(0L, bound - elsewhere);
    }

    /** Bytes of every set's buffer: its shards in upload form, each at an aligned offset. */
    static long hostBytes(List<LayerWeightSet> sets) {
        long total = 0L;
        for (LayerWeightSet set : sets) {
            total = Math.addExact(total, setBytes(set));
        }
        return total;
    }

    private static long setBytes(LayerWeightSet set) {
        long at = 0L;
        for (Entry e : set.entries()) {
            for (int k = 0; k < e.shardCount(); k++) {
                at = Math.addExact(align(at), e.shardBytes(k));
            }
        }
        return at;
    }

    private static long align(long at) {
        return (at + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }

    /**
     * The pinned copies of {@code sets}, or null when the tier does not apply: the backend has no host-mapped memory,
     * every set has a slot of its own, the sets exceed the bound, or the backend fails to allocate, map or flush the
     * memory (then nothing stays allocated and the failure is logged).
     *
     * @throws java.io.UncheckedIOException when the host tier cannot read a tensor (nothing stays allocated): the
     *                                       host tier would fail the same read
     */
    static PinnedLayerTier open(ComputeBackend backend, WeightDataProvider data, List<LayerWeightSet> sets,
                                int slotCount) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(data, "data");
        if (sets.isEmpty() || slotCount >= sets.size() || !backend.supportsHostMappedBuffers()) {
            return null;
        }
        long need = hostBytes(sets);
        long budget = budgetBytes(configuredMb(), data.modelBytes(), HostMemory.totalBytes(),
                HostMemory.availableBytes(), PINNED.get());
        if (need > budget) {
            LOGGER.info("Streamed layers stage from the host tier: their {} MiB of page-locked copies exceed the"
                    + " {} MiB {} allows", need / MIB, budget / MIB, PROPERTY);
            return null;
        }
        PINNED.addAndGet(need);
        List<HostMappedBuffer> buffers = new ArrayList<>(sets.size());
        boolean opened = false;
        try {
            long[][] offsets = new long[sets.size()][];
            ShardFiller filler = new ShardFiller(data);
            for (int i = 0; i < sets.size(); i++) {
                LayerWeightSet set = sets.get(i);
                try {
                    HostMappedBuffer buffer = backend.allocateHostStagingBuffer(setBytes(set));
                    buffers.add(buffer);
                    offsets[i] = fill(filler, set, buffer);
                    buffer.flush();
                } catch (UncheckedIOException e) {
                    throw e;
                } catch (RuntimeException e) {
                    LOGGER.warn("Streamed layers stage from the host tier: page-locked copies of {} failed", set, e);
                    return null;
                }
            }
            PinnedLayerTier tier = new PinnedLayerTier(buffers, offsets, need);
            opened = true;
            LOGGER.info("Streamed layers stage from {} MiB of page-locked host copies ({})", need / MIB, PROPERTY);
            return tier;
        } finally {
            if (!opened) {
                PINNED.addAndGet(-need);
                RuntimeException failure = Closeables.closeAll(buffers, null);
                if (failure != null) {
                    LOGGER.warn("Releasing page-locked copies of streamed layers failed: {}", failure.toString());
                }
            }
        }
    }

    private static long[] fill(ShardFiller filler, LayerWeightSet set, HostMappedBuffer buffer) {
        int shards = 0;
        for (Entry e : set.entries()) {
            shards += e.shardCount();
        }
        long[] at = new long[shards];
        long next = 0L;
        int flat = 0;
        for (Entry e : set.entries()) {
            for (int k = 0; k < e.shardCount(); k++) {
                at[flat] = align(next);
                filler.fill(e, k, buffer, at[flat], "pinned copy of streamed tensor");
                next = at[flat] + e.shardBytes(k);
                flat++;
            }
        }
        return at;
    }

    /** Page-locked bytes of the tier. */
    long bytes() {
        return bytes;
    }

    /**
     * Enqueues the device copy of shard {@code flat} (the entries' shards in set order) of set {@code set} into
     * {@code dst} from byte 0, waiting on {@code waits} first.
     */
    void copy(ComputeQueue queue, int set, int flat, long shardBytes, GpuBuffer dst, GpuEvent... waits) {
        if (closed) {
            throw new IllegalStateException("the pinned copies of the streamed layers are closed");
        }
        queue.enqueueCopyBuffer(buffers.get(set), offsets[set][flat], dst, 0L, shardBytes, waits);
    }

    /** Releases the buffers; the queues that copied from them must be finished. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        PINNED.addAndGet(-bytes);
        RuntimeException failure = Closeables.closeAll(buffers, null);
        if (failure != null) {
            throw failure;
        }
    }
}
