package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.utils.Closeables;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.Pointer;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;

import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;
import static jcuda.driver.JCudaDriver.cuMemHostAlloc;

/**
 * Ring of page-locked host memory for asynchronous host-device copies. Every
 * host write goes through an upload ring, and every read through a readback ring.
 *
 * <ul>
 *   <li>The block is allocated lazily ({@code cuMemHostAlloc}, portable) at
 *       {@value #INITIAL_CAPACITY_BYTES} bytes, or less when the maximum is smaller, and
 *       doubles, up to the maximum, whenever a request exceeds half of it. A grow waits for
 *       the ring to drain, then replaces the block.</li>
 *   <li>Slices are 256-byte aligned and handed out in FIFO order. A slice is reclaimed
 *       after {@link Slice#release()} or once the event given to
 *       {@link Slice#releaseAfter(CudaGpuEvent)} completes. When the ring is full,
 *       {@link #acquire(long)} waits on the oldest slice's event.</li>
 *   <li>A request is at most {@link #maxSliceBytes()} (half the maximum), so two
 *       slices of the largest size can be in flight; callers split larger transfers.</li>
 * </ul>
 * Not thread-safe: the owning queue serialises every call, including slice releases.
 */
@NotThreadSafe("The owning CudaComputeQueue serialises all ring and slice calls under its transfer lock.")
public final class CudaPinnedRing implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaPinnedRing.class);

    /** Maximum ring size in MiB, shared by the upload and readback rings. */
    public static final String STAGING_MB_PROPERTY = "neuralfree.gpu.stagingMb";
    /** Default maximum upload ring size in MiB. */
    public static final int DEFAULT_UPLOAD_MB = 64;
    /** Default maximum readback ring size in MiB. */
    public static final int DEFAULT_READBACK_MB = 16;
    /** Alignment of every slice in bytes. */
    public static final long SLICE_ALIGNMENT = 256;
    /** Capacity of the first block, unless the maximum is smaller. */
    public static final long INITIAL_CAPACITY_BYTES = 4L << 20;

    private static final long MIB = 1L << 20;
    /** Largest ring size: a direct ByteBuffer view is int-indexed. */
    private static final int MAX_STAGING_MB = 1024;

    private final CudaContext context;
    private final String name;
    private final long maxCapacity;
    private final long maxSlice;
    private final ArrayDeque<Slice> inFlight = new ArrayDeque<>();
    private Pointer base;
    private long capacity;
    private long tail;
    private boolean closed;

    /**
     * @param context          context the pinned memory belongs to
     * @param name             ring name for diagnostics, for example {@code "upload"}
     * @param maxCapacityBytes upper bound the ring may grow to
     * @throws IllegalArgumentException when an argument is null or the bound is outside
     *                                  [2 * {@value #SLICE_ALIGNMENT}, {@value #MAX_STAGING_MB} MiB]
     */
    public CudaPinnedRing(CudaContext context, String name, long maxCapacityBytes) {
        if (context == null) {
            throw new IllegalArgumentException("CUDA context must not be null");
        }
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("ring name must not be null or empty");
        }
        if (maxCapacityBytes < 2 * SLICE_ALIGNMENT || maxCapacityBytes > MAX_STAGING_MB * MIB) {
            throw new IllegalArgumentException("pinned " + name + " ring bound must be between " + (2 * SLICE_ALIGNMENT)
                    + " bytes and " + MAX_STAGING_MB + " MiB, got " + maxCapacityBytes + " bytes");
        }
        this.context = context;
        this.name = name;
        this.maxCapacity = maxCapacityBytes;
        this.maxSlice = (maxCapacityBytes / 2) / SLICE_ALIGNMENT * SLICE_ALIGNMENT;
    }

    /**
     * The configured ring bound: {@code -D}{@value #STAGING_MB_PROPERTY} in MiB when set,
     * otherwise {@code defaultMb}.
     *
     * @throws IllegalArgumentException when the property is not an integer in [1, 1024]
     */
    public static long configuredMaxBytes(int defaultMb) {
        String value = System.getProperty(STAGING_MB_PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return defaultMb * MIB;
        }
        int mb;
        try {
            mb = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-D" + STAGING_MB_PROPERTY + " must be an integer number of MiB, got '"
                    + value + "'", e);
        }
        if (mb < 1 || mb > MAX_STAGING_MB) {
            throw new IllegalArgumentException("-D" + STAGING_MB_PROPERTY + " must be between 1 and " + MAX_STAGING_MB
                    + " MiB, got " + mb);
        }
        return mb * MIB;
    }

    /**
     * Acquires a 256-byte-aligned span of {@code n} bytes, waiting on the oldest in-flight
     * slice's event while the ring is full.
     *
     * @throws IllegalArgumentException when {@code n} is not in [1, {@link #maxSliceBytes()}]
     * @throws IllegalStateException    when the ring is closed, or full of slices awaiting an
     *                                  explicit {@link Slice#release()}
     * @throws RuntimeException         the translated driver failure of an allocation or wait
     */
    public Slice acquire(long n) {
        Slice slice = tryAcquire(n);
        if (slice == null) {
            throw new IllegalStateException("pinned " + name + " ring has no room for " + n
                    + " bytes: every in-flight slice awaits an explicit release");
        }
        return slice;
    }

    /**
     * Like {@link #acquire(long)}, but returns null instead of waiting when the oldest
     * in-flight slice has no release event (it awaits {@link Slice#release()}).
     */
    public Slice tryAcquire(long n) {
        ensureOpen();
        if (n <= 0 || n > maxSlice) {
            throw new IllegalArgumentException("pinned " + name + " ring request of " + n
                    + " bytes must be between 1 and " + maxSlice + " bytes");
        }
        long need = align(n);
        if (base == null || (2 * need > capacity && capacity < maxCapacity)) {
            if (!resize(need)) {
                return null;
            }
        }
        while (true) {
            long offset = place(need);
            if (offset < 0) {
                reclaim();
                offset = place(need);
            }
            if (offset >= 0) {
                Slice slice = new Slice(this, offset, need, n);
                inFlight.addLast(slice);
                tail = offset + need;
                return slice;
            }
            Slice oldest = inFlight.peekFirst();
            if (oldest == null || oldest.event == null) {
                return null;
            }
            oldest.event.synchronize();
            oldest.finish();
        }
    }

    /** Largest request {@link #acquire(long)} accepts, in bytes. */
    public long maxSliceBytes() {
        return maxSlice;
    }

    /** Current ring capacity in bytes (0 before first use). */
    public long capacity() {
        return capacity;
    }

    /** Number of slices not yet reclaimed. */
    public int inFlightCount() {
        return inFlight.size();
    }

    /**
     * Waits for every in-flight slice with a release event and frees the pinned memory;
     * idempotent. Slices still held become unusable. The owning queue closes the ring while its
     * backend still holds its retain of the context.
     *
     * @throws IllegalStateException when the context was already released; the memory is then
     *                               not freed
     * @throws RuntimeException      the first translated driver failure, after the memory was freed
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (Slice slice : inFlight) {
            try {
                if (slice.event != null) {
                    slice.event.synchronize();
                }
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
            slice.finish();
        }
        inFlight.clear();
        try {
            freeBlock();
        } catch (RuntimeException e) {
            failure = Closeables.chain(failure, e);
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "CudaPinnedRing{" + name + ", capacity=" + capacity + ", max=" + maxCapacity + ", inFlight="
                + inFlight.size() + (closed ? ", closed" : "") + "}";
    }

    /** Offset where {@code need} bytes fit after the newest slice, or -1. */
    private long place(long need) {
        if (inFlight.isEmpty()) {
            tail = 0;
            return need <= capacity ? 0 : -1;
        }
        long head = inFlight.peekFirst().offset;
        if (tail > head) {
            if (capacity - tail >= need) {
                return tail;
            }
            return head >= need ? 0 : -1;
        }
        return head - tail >= need ? tail : -1;
    }

    /** Pops finished slices from the front: explicitly released, or with a completed event. */
    private void reclaim() {
        while (!inFlight.isEmpty()) {
            Slice oldest = inFlight.peekFirst();
            if (oldest.state == Slice.EVENT && oldest.event.isComplete()) {
                oldest.finish();
            }
            if (oldest.state != Slice.DONE) {
                return;
            }
            inFlight.pollFirst();
        }
    }

    /**
     * Drains the ring and replaces the block with one of at least {@code 2 * need} bytes
     * (bounded by the maximum). Returns false when a held slice blocks the drain.
     */
    private boolean resize(long need) {
        while (!inFlight.isEmpty()) {
            reclaim();
            Slice oldest = inFlight.peekFirst();
            if (oldest == null) {
                break;
            }
            if (oldest.event == null) {
                return false;
            }
            oldest.event.synchronize();
            oldest.finish();
        }
        long target = capacity == 0 ? Math.min(INITIAL_CAPACITY_BYTES, maxCapacity) : capacity;
        while (target < 2 * need && target < maxCapacity) {
            target *= 2;
        }
        target = Math.min(target, maxCapacity);
        freeBlock();
        context.makeCurrent();
        Pointer block = new Pointer();
        long bytes = target;
        CudaErrors.check("cuMemHostAlloc(" + name + " staging)", bytes,
                () -> cuMemHostAlloc(block, bytes, CU_MEMHOSTALLOC_PORTABLE));
        if (block.getByteBuffer(0, bytes) == null) {
            IllegalStateException missingView = new IllegalStateException("pinned " + name
                    + " ring: JCuda returned page-locked memory without a ByteBuffer view");
            try {
                CudaErrors.check("cuMemFreeHost(" + name + " staging)", () -> cuMemFreeHost(block));
            } catch (RuntimeException e) {
                missingView.addSuppressed(e);
            }
            throw missingView;
        }
        base = block;
        capacity = bytes;
        tail = 0;
        LOGGER.debug("Pinned {} ring now {} KiB (max {} KiB)", name, bytes >> 10, maxCapacity >> 10);
        return true;
    }

    private void freeBlock() {
        if (base == null) {
            return;
        }
        context.makeCurrent();
        Pointer block = base;
        base = null;
        capacity = 0;
        tail = 0;
        CudaErrors.check("cuMemFreeHost(" + name + " staging)", () -> cuMemFreeHost(block));
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("pinned " + name + " ring is closed");
        }
    }

    private static long align(long n) {
        return (n + SLICE_ALIGNMENT - 1) / SLICE_ALIGNMENT * SLICE_ALIGNMENT;
    }

    /** One acquired span of pinned memory. */
    public static final class Slice {

        static final int HELD = 0;
        static final int EVENT = 1;
        static final int DONE = 2;

        private final CudaPinnedRing ring;
        private final long offset;
        private final long alignedSize;
        private final long size;
        private int state = HELD;
        private CudaGpuEvent event;

        private Slice(CudaPinnedRing ring, long offset, long alignedSize, long size) {
            this.ring = ring;
            this.offset = offset;
            this.alignedSize = alignedSize;
            this.size = size;
        }

        /**
         * Device-visible pointer to the start of the span.
         *
         * @throws IllegalStateException when the slice was already handed back
         */
        public Pointer pointer() {
            ensureHeld();
            return ring.base.withByteOffset(offset);
        }

        /**
         * Host view of the span in native byte order, positioned at 0 with limit {@link #size()}.
         *
         * @throws IllegalStateException when the slice was already handed back
         */
        public ByteBuffer byteBuffer() {
            ensureHeld();
            return ring.base.getByteBuffer(offset, size);
        }

        /** Requested size in bytes. */
        public long size() {
            return size;
        }

        /** Bytes the slice occupies in the ring (the size rounded up to 256). */
        public long alignedSize() {
            return alignedSize;
        }

        /**
         * Returns the span to the ring once {@code event} completes; the ring takes ownership
         * of the handle and closes it.
         *
         * @throws IllegalArgumentException when {@code event} is null
         * @throws IllegalStateException    when the slice was already handed back
         */
        public void releaseAfter(CudaGpuEvent event) {
            if (event == null) {
                throw new IllegalArgumentException("release event must not be null");
            }
            ensureHeld();
            this.event = event;
            this.state = EVENT;
        }

        /**
         * Returns the span to the ring immediately; only valid when no copy is in flight.
         *
         * @throws IllegalStateException when the slice was already handed back
         */
        public void release() {
            ensureHeld();
            state = DONE;
        }

        private void finish() {
            if (event != null) {
                event.close();
                event = null;
            }
            state = DONE;
        }

        private void ensureHeld() {
            if (state != HELD) {
                throw new IllegalStateException("pinned " + ring.name + " ring slice at offset " + offset
                        + " was already handed back");
            }
            if (ring.closed) {
                throw new IllegalStateException("pinned " + ring.name + " ring is closed");
            }
        }
    }
}
