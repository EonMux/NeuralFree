package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The page-locked staging of a resident upload's raw chunks: a ring of {@value #SLOTS} host-mapped buffers
 * ({@link ComputeBackend#allocateHostStagingBuffer}) that a chunk is copied into by several threads at once and that
 * the queue's copy engine reads with one device copy ({@link ComputeQueue#enqueueCopyBuffer}).
 *
 * <p>A resident load of mapped tensor data is bound by the host copy out of the page cache: one thread copies about
 * 8 GB/s, well below what a PCIe 4 or 5 link takes from page-locked memory, and a backend's own write path makes that
 * copy on the calling thread. The ring spreads each chunk's copy over the calling thread and up to
 * {@link #fillThreads()} pool threads, and the device copy of one slot runs while the next slots fill. A slot is
 * refilled only after the copy that read it completed (its marker).</p>
 *
 * <p>Only a {@link DeviceClass#DISCRETE} device whose backend prefers this staging
 * ({@link ComputeBackend#prefersParallelUploadStaging}) gets a ring. A device that shares the host's memory reads the
 * slot over the same memory bus its buffer lives on, so the device copy would be a second pass over the bytes the
 * backend's own write already places; a device of unknown class keeps the backend's write path, and so does a backend
 * whose write path is as fast (Vulkan stages its writes through host-visible memory at about the ring's rate, so the
 * ring would only add its slots' allocation).</p>
 *
 * <p>Owned by one uploader; not thread-safe. {@link #close()} waits for the copies still reading the slots, then
 * releases the slots and the pool.</p>
 */
final class PinnedUploadRing implements AutoCloseable {

    /** Slots of the ring: one in the copy engine, the others filling. */
    static final int SLOTS = 3;
    /** Least bytes one fill thread copies; smaller chunks are copied by fewer threads. */
    static final int MIN_PART_BYTES = 1 << 20;
    private static final int MAX_FILL_THREADS = 4;
    private static final AtomicInteger POOLS = new AtomicInteger();
    private static final Logger LOGGER = LogManager.getLogger(PinnedUploadRing.class);

    private final HostMappedBuffer[] slots;
    private final GpuEvent[] released;
    private final int slotBytes;
    private final int threads;
    private final ExecutorService pool;
    private int next;
    private boolean closed;

    private PinnedUploadRing(HostMappedBuffer[] slots, int slotBytes, int threads) {
        this.slots = slots;
        this.released = new GpuEvent[slots.length];
        this.slotBytes = slotBytes;
        this.threads = threads;
        int id = POOLS.incrementAndGet();
        AtomicInteger n = new AtomicInteger();
        this.pool = threads > 1 ? Executors.newFixedThreadPool(threads - 1, r -> {
            Thread t = new Thread(r, "nf-upload-fill-" + id + "-" + n.incrementAndGet());
            t.setDaemon(true);
            return t;
        }) : null;
    }

    /** Threads that copy one chunk: the caller and the pool, a quarter of the logical processors, one to four. */
    static int fillThreads() {
        return Math.max(1, Math.min(MAX_FILL_THREADS, Runtime.getRuntime().availableProcessors() / 4));
    }

    /**
     * A ring of {@code slotBytes}-byte slots on {@code backend}, or null when the device is not
     * {@link DeviceClass#DISCRETE}, the backend does not prefer this staging or has no host-mapped memory, or it cannot
     * allocate the slots (nothing then stays allocated).
     *
     * @throws IllegalArgumentException when {@code slotBytes} is not positive
     */
    static PinnedUploadRing open(ComputeBackend backend, int slotBytes) {
        Objects.requireNonNull(backend, "backend");
        if (slotBytes <= 0) {
            throw new IllegalArgumentException("slot size must be positive, got " + slotBytes);
        }
        if (!backend.supportsHostMappedBuffers() || !backend.prefersParallelUploadStaging()
                || backend.capabilities().getDeviceClass() != DeviceClass.DISCRETE) {
            return null;
        }
        HostMappedBuffer[] slots = new HostMappedBuffer[SLOTS];
        try {
            for (int i = 0; i < SLOTS; i++) {
                slots[i] = backend.allocateHostStagingBuffer(slotBytes);
            }
        } catch (RuntimeException e) {
            List<AutoCloseable> made = new ArrayList<>();
            for (HostMappedBuffer s : slots) {
                if (s != null) {
                    made.add(s);
                }
            }
            RuntimeException failure = Closeables.closeAll(made, null);
            if (failure != null) {
                e.addSuppressed(failure);
            }
            LOGGER.warn("Weight uploads write through the queue: page-locked upload slots of {} bytes failed",
                    slotBytes, e);
            return null;
        }
        return new PinnedUploadRing(slots, slotBytes, fillThreads());
    }

    /** Bytes one chunk may hold. */
    int slotBytes() {
        return slotBytes;
    }

    /**
     * Copies {@code src}'s remaining bytes (at most {@link #slotBytes()}) into the next slot and enqueues the device
     * copy of the slot to {@code dst} at byte {@code dstOffset} on {@code queue}. {@code src} is not moved. The slots
     * belong to the ring's backend, so {@code queue} and {@code dst} must be that backend's: another backend's device
     * refuses the copy.
     *
     * @throws IllegalArgumentException when the chunk exceeds a slot
     * @throws IllegalStateException    when the ring is closed, or a wait for the slot is interrupted
     */
    void upload(ByteBuffer src, GpuBuffer dst, long dstOffset, ComputeQueue queue) {
        if (closed) {
            throw new IllegalStateException("the upload ring is closed");
        }
        int n = src.remaining();
        if (n > slotBytes) {
            throw new IllegalArgumentException("a chunk of " + n + " bytes exceeds the " + slotBytes + "-byte slot");
        }
        if (n == 0) {
            return;
        }
        int k = next;
        next = (next + 1) % slots.length;
        awaitReleased(k);
        fill(src, slots[k].hostView(0, n));
        slots[k].flush();
        queue.enqueueCopyBuffer(slots[k], 0L, dst, dstOffset, n);
        released[k] = queue.enqueueMarker();
    }

    /**
     * Copies {@code src} into {@code dst} over the caller and the pool, in parts of at least
     * {@link #MIN_PART_BYTES}.
     */
    private void fill(ByteBuffer src, ByteBuffer dst) {
        int n = src.remaining();
        int parts = pool == null ? 1 : Math.max(1, Math.min(threads, n / MIN_PART_BYTES));
        if (parts == 1) {
            dst.put(src.duplicate());
            return;
        }
        int base = src.position();
        int part = (n + parts - 1) / parts;
        List<Future<?>> others = new ArrayList<>(parts - 1);
        for (int p = 1; p < parts; p++) {
            int from = p * part;
            int len = Math.min(part, n - from);
            if (len <= 0) {
                break;
            }
            others.add(pool.submit(() -> dst.duplicate().position(from).put(range(src, base + from, len))));
        }
        dst.duplicate().position(0).put(range(src, base, Math.min(part, n)));
        RuntimeException failure = null;
        for (Future<?> f : others) {
            try {
                f.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                failure = new IllegalStateException("interrupted while staging a weight chunk", e);
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                RuntimeException r = cause instanceof RuntimeException ? (RuntimeException) cause
                        : new IllegalStateException("staging a weight chunk failed: " + cause, cause);
                if (failure == null) {
                    failure = r;
                } else {
                    failure.addSuppressed(r);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** Bytes {@code [from, from + len)} of {@code src} as a buffer of their own. */
    private static ByteBuffer range(ByteBuffer src, int from, int len) {
        ByteBuffer d = src.duplicate();
        d.limit(from + len);
        d.position(from);
        return d;
    }

    /** Waits until the copy that last read slot {@code k} completed. */
    private void awaitReleased(int k) {
        GpuEvent e = released[k];
        if (e == null) {
            return;
        }
        released[k] = null;
        try {
            e.awaitCompletion();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for a weight copy", ex);
        } finally {
            e.close();
        }
    }

    /** Waits for the copies reading the slots, then releases the slots and the pool. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (int k = 0; k < slots.length; k++) {
            try {
                awaitReleased(k);
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        if (pool != null) {
            pool.shutdownNow();
        }
        List<AutoCloseable> owned = new ArrayList<>(slots.length);
        for (HostMappedBuffer s : slots) {
            owned.add(s);
        }
        failure = Closeables.closeAll(owned, failure);
        if (failure != null) {
            throw failure;
        }
    }
}
