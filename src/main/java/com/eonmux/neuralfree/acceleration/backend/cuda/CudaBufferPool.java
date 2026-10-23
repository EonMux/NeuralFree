package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.utils.Closeables;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.driver.CUdeviceptr;
import org.apache.logging.log4j.Logger;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Stream-ordered caching allocator of one {@link CudaComputeBackend}, so releasing a buffer never
 * waits for the device ({@code cuMemFree} synchronizes it).
 *
 * <ul>
 *   <li><b>Release.</b> {@link #release} records a fence on every stream of the backend
 *       ({@link CudaMemoryDriver#recordFence()}) and queues the block as pending. When the fence
 *       cannot be recorded, the block is freed at once with {@code cuMemFree}.</li>
 *   <li><b>Reuse.</b> {@link #acquire} first moves pending blocks whose fences have completed to
 *       the ready buckets (one bucket per block size), oldest first, stopping at the first fence
 *       still in flight; it never waits. It then takes the smallest ready block of at least the
 *       requested size and at most twice it, and at most {@link #MAX_REUSE_SLACK_BYTES} larger,
 *       or allocates a fresh block of exactly the requested size. The absolute bound keeps a
 *       long-lived large buffer from pinning a much larger pooled block.</li>
 *   <li><b>Out of memory.</b> When a fresh allocation fails with out-of-memory and blocks are
 *       pooled, the pool synchronizes the context, frees every pooled block and retries once.</li>
 *   <li><b>Trim and close.</b> {@link #trim()} and {@link #close()} free every pooled block,
 *       synchronizing the context first when a block may still be in use. {@link #close()} also
 *       frees every block still handed out, because the primary context the blocks live in is
 *       shared with the other backends of the device and outlives this pool; releasing such a
 *       block afterwards does nothing.</li>
 *   <li><b>Accounting.</b> {@link #inUseBytes()} counts blocks handed out and not yet released;
 *       {@link #pooledBytes()} counts blocks held for reuse. Their sum is the device memory the
 *       pool holds from the driver. A block whose {@code cuMemFree} fails is dropped from both.</li>
 * </ul>
 *
 * <p>The stream-ordered driver allocator ({@code cuMemAllocAsync}/{@code cuMemFreeAsync}) is not
 * used: its free is ordered against one stream only, while a backend buffer may be used on the
 * compute, transfer and caller-owned streams. The fences cover all of them.</p>
 *
 * <p>Safe for use from several threads. The pool lock is held only for bookkeeping and the
 * non-blocking fence queries ({@code cuEventQuery}); allocation, free, fence recording and
 * synchronization run outside it.</p>
 */
public final class CudaBufferPool {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaBufferPool.class);

    /** Largest excess of a reused block over the request, in bytes (20 MiB). */
    public static final long MAX_REUSE_SLACK_BYTES = 20L << 20;

    private final CudaMemoryDriver driver;
    private final Object lock = new Object();
    private final ArrayDeque<Block> pending = new ArrayDeque<>();
    private final TreeMap<Long, ArrayDeque<CUdeviceptr>> ready = new TreeMap<>();
    /** Blocks handed out and not yet released, by pointer identity, with their sizes. */
    private final Map<CUdeviceptr, Long> handedOut = new IdentityHashMap<>();
    private int readyBlocks;
    private long inUseBytes;
    private long pooledBytes;
    private long freshAllocations;
    private long reusedAllocations;
    private long oomRetries;
    private boolean closed;

    /**
     * @throws IllegalArgumentException when {@code driver} is null
     */
    CudaBufferPool(CudaMemoryDriver driver) {
        if (driver == null) {
            throw new IllegalArgumentException("CUDA memory driver must not be null");
        }
        this.driver = driver;
    }

    /**
     * A device block of at least {@code bytes}: a ready pooled block of at most twice the size and
     * at most {@link #MAX_REUSE_SLACK_BYTES} larger, otherwise a fresh allocation of exactly
     * {@code bytes}. Never waits for the device, except for the synchronization of the
     * out-of-memory retry.
     *
     * @param bytes the size to allocate, already rounded to the backend's allocation granularity
     * @throws IllegalArgumentException when {@code bytes} is not positive
     * @throws IllegalStateException    when the pool is closed
     * @throws GpuOutOfMemoryException  when the device is out of memory after the retry
     * @throws RuntimeException         any other translated driver failure
     */
    Allocation acquire(long bytes) {
        if (bytes <= 0) {
            throw new IllegalArgumentException("allocation size must be positive, got " + bytes);
        }
        Allocation pooled = takeReady(bytes);
        if (pooled != null) {
            return pooled;
        }
        CUdeviceptr pointer = allocateFresh(bytes);
        synchronized (lock) {
            if (!closed) {
                handedOut.put(pointer, bytes);
                inUseBytes += bytes;
                freshAllocations++;
                return new Allocation(pointer, bytes);
            }
        }
        IllegalStateException e = new IllegalStateException("CUDA buffer pool was closed during an allocation");
        try {
            driver.free(pointer);
        } catch (RuntimeException free) {
            e.addSuppressed(free);
        }
        throw e;
    }

    /**
     * Returns a block of {@link #acquire} for reuse once the work enqueued so far on the backend's
     * streams has finished. After {@link #close()} it does nothing: the close freed the block.
     *
     * @throws IllegalArgumentException when {@code pointer} is null, {@code bytes} is not positive,
     *                                  or the pool did not hand out this block with this size (or
     *                                  it was already released)
     * @throws RuntimeException         the translated failure of recording the fence; the block was
     *                                  then freed with {@code cuMemFree} (failures of that free are
     *                                  suppressed on it)
     */
    void release(CUdeviceptr pointer, long bytes) {
        if (pointer == null) {
            throw new IllegalArgumentException("released CUDA allocation must not be null");
        }
        if (bytes <= 0) {
            throw new IllegalArgumentException("released allocation size must be positive, got " + bytes);
        }
        synchronized (lock) {
            if (closed) {
                return;
            }
            Long held = handedOut.get(pointer);
            if (held == null || held != bytes) {
                throw new IllegalArgumentException("CUDA block of " + bytes + " bytes was not handed out by this pool"
                        + (held == null ? " or was already released" : " (it holds " + held + " bytes)"));
            }
        }
        CudaMemoryDriver.Fence fence;
        try {
            fence = driver.recordFence();
        } catch (RuntimeException e) {
            freeUnfenced(pointer, bytes, e);
            throw e;
        }
        boolean pooled;
        synchronized (lock) {
            pooled = !closed && handedOut.remove(pointer) != null;
            if (pooled) {
                inUseBytes -= bytes;
                pending.addLast(new Block(pointer, bytes, fence));
                pooledBytes += bytes;
            }
        }
        if (!pooled) {
            fence.release();
        }
    }

    /**
     * Frees every pooled block, synchronizing the context first when one may still be in use.
     * Does nothing once closed.
     *
     * @return the bytes freed
     * @throws RuntimeException the first translated driver failure, after every block was freed
     */
    public long trim() {
        synchronized (lock) {
            if (closed) {
                return 0;
            }
        }
        return releaseAll(false, Collections.emptyList());
    }

    /**
     * Frees every pooled block and every block still handed out, whose buffers become invalid, and
     * refuses further acquisitions; releasing a block afterwards does nothing. Idempotent. The
     * context is synchronized first when a freed block may still be in use. The backend calls it
     * while it still holds its retain of the primary context.
     *
     * @throws RuntimeException the first translated driver failure, after every block was freed
     */
    void close() {
        List<Block> stillHandedOut = new ArrayList<>();
        synchronized (lock) {
            if (closed) {
                return;
            }
            closed = true;
            for (Map.Entry<CUdeviceptr, Long> block : handedOut.entrySet()) {
                stillHandedOut.add(new Block(block.getKey(), block.getValue(), null));
            }
            handedOut.clear();
            inUseBytes = 0;
        }
        if (!stillHandedOut.isEmpty()) {
            LOGGER.debug("CUDA buffer pool closing with {} blocks of open buffers; freeing them", stillHandedOut.size());
        }
        releaseAll(!stillHandedOut.isEmpty(), stillHandedOut);
    }

    /** Bytes of the blocks handed out and not yet released. */
    public long inUseBytes() {
        synchronized (lock) {
            return inUseBytes;
        }
    }

    /** Bytes of the blocks held for reuse, pending or ready. */
    public long pooledBytes() {
        synchronized (lock) {
            return pooledBytes;
        }
    }

    /** A consistent snapshot of the pool's counters. */
    public Stats stats() {
        synchronized (lock) {
            return new Stats(inUseBytes, pooledBytes, pending.size(), readyBlocks, freshAllocations,
                    reusedAllocations, oomRetries);
        }
    }

    private Allocation takeReady(long bytes) {
        List<CudaMemoryDriver.Fence> completed = new ArrayList<>();
        try {
            synchronized (lock) {
                if (closed) {
                    throw new IllegalStateException("CUDA buffer pool is closed (its backend was closed)");
                }
                collectCompletedLocked(completed);
                Map.Entry<Long, ArrayDeque<CUdeviceptr>> bucket = ready.ceilingEntry(bytes);
                if (bucket == null || bucket.getKey() - bytes > Math.min(bytes, MAX_REUSE_SLACK_BYTES)) {
                    return null;
                }
                long size = bucket.getKey();
                CUdeviceptr pointer = bucket.getValue().pollFirst();
                if (bucket.getValue().isEmpty()) {
                    ready.remove(size);
                }
                readyBlocks--;
                pooledBytes -= size;
                handedOut.put(pointer, size);
                inUseBytes += size;
                reusedAllocations++;
                return new Allocation(pointer, size);
            }
        } finally {
            releaseFences(completed);
        }
    }

    /** Moves pending blocks whose fences completed to the ready buckets, oldest first. */
    private void collectCompletedLocked(List<CudaMemoryDriver.Fence> completed) {
        while (!pending.isEmpty()) {
            Block head = pending.peekFirst();
            if (!head.fence.isComplete()) {
                return;
            }
            pending.pollFirst();
            ready.computeIfAbsent(head.bytes, size -> new ArrayDeque<>()).addFirst(head.pointer);
            readyBlocks++;
            completed.add(head.fence);
        }
    }

    private CUdeviceptr allocateFresh(long bytes) {
        try {
            return driver.allocate(bytes);
        } catch (GpuOutOfMemoryException first) {
            if (pooledBytes() == 0) {
                throw first;
            }
            long released;
            try {
                released = releaseAll(true, Collections.emptyList());
            } catch (RuntimeException e) {
                e.addSuppressed(first);
                throw e;
            }
            synchronized (lock) {
                oomRetries++;
            }
            LOGGER.debug("CUDA allocation of {} bytes ran out of memory; freed {} pooled bytes and retrying", bytes,
                    released);
            try {
                return driver.allocate(bytes);
            } catch (RuntimeException second) {
                second.addSuppressed(first);
                throw second;
            }
        }
    }

    /**
     * Empties the pool and frees every pooled block and every block of {@code alsoFree}, which the
     * caller has already removed from the accounting. The context is synchronized first when
     * {@code synchronize} is set or a block's fence has not been seen complete.
     */
    private long releaseAll(boolean synchronize, List<Block> alsoFree) {
        List<CudaMemoryDriver.Fence> completed = new ArrayList<>();
        List<Block> drained = new ArrayList<>(alsoFree);
        RuntimeException failure = null;
        synchronized (lock) {
            try {
                collectCompletedLocked(completed);
            } catch (RuntimeException e) {
                failure = e;
            }
            drained.addAll(pending);
            pending.clear();
            for (Map.Entry<Long, ArrayDeque<CUdeviceptr>> bucket : ready.entrySet()) {
                for (CUdeviceptr pointer : bucket.getValue()) {
                    drained.add(new Block(pointer, bucket.getKey(), null));
                }
            }
            ready.clear();
            readyBlocks = 0;
            pooledBytes = 0;
        }
        releaseFences(completed);
        boolean inFlight = synchronize || failure != null;
        for (Block block : drained) {
            inFlight |= block.fence != null;
        }
        if (inFlight && !drained.isEmpty()) {
            try {
                driver.synchronize();
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
        }
        long freed = 0;
        for (Block block : drained) {
            try {
                driver.free(block.pointer);
                freed += block.bytes;
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
            if (block.fence != null) {
                block.fence.release();
            }
        }
        if (failure != null) {
            throw failure;
        }
        return freed;
    }

    /** Frees a block whose fence could not be recorded, unless a concurrent {@link #close()} already freed it. */
    private void freeUnfenced(CUdeviceptr pointer, long bytes, RuntimeException cause) {
        synchronized (lock) {
            if (handedOut.remove(pointer) == null) {
                return;
            }
            inUseBytes -= bytes;
        }
        try {
            driver.free(pointer);
        } catch (RuntimeException e) {
            cause.addSuppressed(e);
        }
    }

    private static void releaseFences(List<CudaMemoryDriver.Fence> fences) {
        for (CudaMemoryDriver.Fence fence : fences) {
            fence.release();
        }
    }

    /** A device block handed out by {@link #acquire}. */
    static final class Allocation {
        private final CUdeviceptr pointer;
        private final long bytes;

        Allocation(CUdeviceptr pointer, long bytes) {
            this.pointer = pointer;
            this.bytes = bytes;
        }

        /** The block's device pointer. */
        CUdeviceptr pointer() {
            return pointer;
        }

        /** The block's size in bytes, within {@link #acquire}'s reuse bounds of the request. */
        long bytes() {
            return bytes;
        }
    }

    /** A pooled block; {@code fence} is null once the block is known to be idle. */
    private static final class Block {
        private final CUdeviceptr pointer;
        private final long bytes;
        private final CudaMemoryDriver.Fence fence;

        Block(CUdeviceptr pointer, long bytes, CudaMemoryDriver.Fence fence) {
            this.pointer = pointer;
            this.bytes = bytes;
            this.fence = fence;
        }
    }

    /** Immutable snapshot of a pool's counters, for diagnostics. */
    public static final class Stats {
        private final long inUseBytes;
        private final long pooledBytes;
        private final int pendingBlocks;
        private final int readyBlocks;
        private final long freshAllocations;
        private final long reusedAllocations;
        private final long oomRetries;

        Stats(long inUseBytes, long pooledBytes, int pendingBlocks, int readyBlocks, long freshAllocations,
              long reusedAllocations, long oomRetries) {
            this.inUseBytes = inUseBytes;
            this.pooledBytes = pooledBytes;
            this.pendingBlocks = pendingBlocks;
            this.readyBlocks = readyBlocks;
            this.freshAllocations = freshAllocations;
            this.reusedAllocations = reusedAllocations;
            this.oomRetries = oomRetries;
        }

        /** Bytes of the blocks handed out and not yet released. */
        public long inUseBytes() {
            return inUseBytes;
        }

        /** Bytes held for reuse, pending or ready. */
        public long pooledBytes() {
            return pooledBytes;
        }

        /** Released blocks whose fences have not been seen complete. */
        public int pendingBlocks() {
            return pendingBlocks;
        }

        /** Released blocks ready for reuse. */
        public int readyBlocks() {
            return readyBlocks;
        }

        /** Acquisitions served by {@code cuMemAlloc}. */
        public long freshAllocations() {
            return freshAllocations;
        }

        /** Acquisitions served by a pooled block. */
        public long reusedAllocations() {
            return reusedAllocations;
        }

        /** Out-of-memory failures answered by freeing the pool and retrying the allocation once. */
        public long oomRetries() {
            return oomRetries;
        }

        @Override
        public String toString() {
            return "CudaBufferPool.Stats{inUse=" + inUseBytes + " B, pooled=" + pooledBytes + " B, pending="
                    + pendingBlocks + ", ready=" + readyBlocks + ", fresh=" + freshAllocations + ", reused="
                    + reusedAllocations + ", oomRetries=" + oomRetries + "}";
        }
    }
}
