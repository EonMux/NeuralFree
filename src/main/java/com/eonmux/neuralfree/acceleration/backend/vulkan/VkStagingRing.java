package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

/**
 * Ring over a persistently mapped host-visible buffer for copy-at-enqueue uploads
 * (coherent memory) or batched readbacks (cached memory). Slices are tagged with
 * the timeline value of the batch that uses them and reclaimed once that value
 * completes; a full ring waits for its oldest value.
 *
 * <p>Positions are virtual byte counters that only grow: {@code head} is where the next slice
 * starts and {@code tail} is the start of the oldest slice still in flight, so
 * {@code head - tail} bytes are in use and the physical offset is {@code position % capacity}.
 * A slice never wraps: when it would cross the end of the buffer the remainder of the buffer is
 * skipped and charged to that slice. Consecutive slices of one batch share one in-flight span,
 * so reclaiming is a pop per retired batch. The ring buffer gets its own dedicated memory on
 * first use; a request larger than the ring gets a one-off dedicated buffer that is destroyed
 * when its batch retires.</p>
 *
 * <p>Upload memory is required to be host-coherent, so uploads need no flush. Readback memory
 * prefers cached and coherent memory; when it is not coherent the consumer calls
 * {@link Slice#invalidate()} before reading.</p>
 *
 * <p>Thread-safe. The ring never calls its {@link Timeline} while holding its own lock, so a
 * queue that retires batches (and calls {@link #releaseUpTo}) from another thread cannot deadlock
 * against {@link #acquire}. The value of a slice is read from the timeline on every attempt,
 * so a slice acquired after the ring had to submit the batch being recorded is tagged with the
 * next batch. The queue keeps its pending value stable while it acquires a slice and records
 * the command that uses it.</p>
 */
public final class VkStagingRing implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkStagingRing.class);

    /** Ring size in MiB, applied to both the upload and the readback ring when set. */
    public static final String STAGING_MB_PROPERTY = "neuralfree.gpu.stagingMb";
    /** Default upload ring size in MiB. */
    public static final int DEFAULT_UPLOAD_MB = 64;
    /** Default readback ring size in MiB. */
    public static final int DEFAULT_READBACK_MB = 16;
    /** Alignment of every slice offset in bytes. */
    public static final long ALIGNMENT = 256;

    private static final long MIB = 1024L * 1024L;

    /** The queue timeline a ring synchronises with. */
    public interface Timeline {
        /** The value the batch currently being recorded will signal. */
        long pendingValue();

        /** Submits pending work if needed, waits until {@code value} completes, and retires up to it. */
        void waitForValue(long value);
    }

    private final VkMemoryAllocator allocator;
    private final boolean upload;
    private final long capacity;

    private final ArrayDeque<Span> inFlight = new ArrayDeque<>();
    private final ArrayDeque<OneOff> oneOffs = new ArrayDeque<>();
    private VkGpuBuffer ring;
    private long head;
    private long tail;
    private long lastValue;
    private boolean closed;

    /**
     * @param upload        true for an upload ring, false for a readback ring
     * @param capacityBytes ring size, rounded up to {@link #ALIGNMENT}; memory is allocated on first use
     */
    public VkStagingRing(VkMemoryAllocator allocator, boolean upload, long capacityBytes) {
        if (allocator == null) {
            throw new IllegalArgumentException("allocator must not be null");
        }
        if (capacityBytes <= 0) {
            throw new IllegalArgumentException("staging ring capacity must be positive, got " + capacityBytes);
        }
        long rounded = VkMemoryBlock.alignUp(capacityBytes, ALIGNMENT);
        if (rounded > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("staging ring capacity must be below 2 GiB, got " + capacityBytes);
        }
        this.allocator = allocator;
        this.upload = upload;
        this.capacity = rounded;
    }

    /**
     * Ring capacity in bytes from {@code -Dneuralfree.gpu.stagingMb}, or the default for the ring kind
     * ({@value #DEFAULT_UPLOAD_MB} MiB upload, {@value #DEFAULT_READBACK_MB} MiB readback).
     *
     * @throws IllegalArgumentException when the property is not a positive integer below 2048
     */
    public static long configuredCapacity(boolean upload) {
        String value = System.getProperty(STAGING_MB_PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return (upload ? DEFAULT_UPLOAD_MB : DEFAULT_READBACK_MB) * MIB;
        }
        long mb;
        try {
            mb = Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-D" + STAGING_MB_PROPERTY
                    + " must be a positive integer below 2048 (MiB), got '" + value + "'", e);
        }
        if (mb <= 0 || mb >= 2048) {
            throw new IllegalArgumentException("-D" + STAGING_MB_PROPERTY
                    + " must be a positive integer below 2048 (MiB), got '" + value + "'");
        }
        return mb * MIB;
    }

    /**
     * Acquires a 256-byte-aligned slice of {@code n} bytes tagged with {@code timeline.pendingValue()}.
     * When the ring is full it waits for the oldest batch holding ring space, which may submit the
     * batch being recorded. A request larger than the ring gets a one-off buffer released when it retires.
     *
     * @throws IllegalArgumentException when {@code n} is not positive or does not fit a Java buffer (2 GiB)
     * @throws IllegalStateException    when the ring is closed or the timeline value decreases
     */
    public Slice acquire(long n, Timeline timeline) {
        if (n <= 0) {
            throw new IllegalArgumentException("staging slice size must be positive, got " + n);
        }
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("staging slice of " + n
                    + " bytes exceeds the 2 GiB host view limit; split the transfer");
        }
        if (timeline == null) {
            throw new IllegalArgumentException("timeline must not be null");
        }
        long padded = VkMemoryBlock.alignUp(n, ALIGNMENT);
        if (padded > capacity) {
            return acquireOneOff(n, timeline.pendingValue());
        }
        while (true) {
            long value = timeline.pendingValue();
            long oldest;
            synchronized (this) {
                ensureOpen();
                checkMonotonic(value);
                ensureRing();
                Slice slice = tryAllocate(n, padded, value);
                if (slice != null) {
                    return slice;
                }
                oldest = inFlight.peekFirst().value;
            }
            timeline.waitForValue(oldest);
            releaseUpTo(oldest);
        }
    }

    /** Reclaims every slice whose batch value is at most {@code completedValue}. */
    public synchronized void releaseUpTo(long completedValue) {
        while (!inFlight.isEmpty() && inFlight.peekFirst().value <= completedValue) {
            tail = inFlight.pollFirst().end;
        }
        if (inFlight.isEmpty()) {
            head = 0;
            tail = 0;
        }
        while (!oneOffs.isEmpty() && oneOffs.peekFirst().value <= completedValue) {
            oneOffs.pollFirst().buffer.closeRetired();
        }
    }

    /** Ring size in bytes. */
    public long capacity() {
        return capacity;
    }

    /** Bytes of the ring held by in-flight slices, including skipped tail space. */
    public synchronized long usedBytes() {
        return head - tail;
    }

    /** True for an upload ring, false for a readback ring. */
    public boolean isUpload() {
        return upload;
    }

    /**
     * Frees the ring buffer and every one-off buffer; idempotent. The caller ensures no batch
     * that uses a slice is still executing.
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        inFlight.clear();
        while (!oneOffs.isEmpty()) {
            oneOffs.pollFirst().buffer.closeRetired();
        }
        if (ring != null) {
            ring.closeRetired();
            ring = null;
        }
        head = 0;
        tail = 0;
    }

    private Slice tryAllocate(long n, long padded, long value) {
        long physical = head % capacity;
        long skip = physical + padded > capacity ? capacity - physical : 0;
        if (head - tail + skip + padded > capacity) {
            return null;
        }
        long start = head + skip;
        long offset = start % capacity;
        head = start + padded;
        Span last = inFlight.peekLast();
        if (last != null && last.value == value) {
            last.end = head;
        } else {
            inFlight.addLast(new Span(head, value));
        }
        return new Slice(allocator, ring, offset, n, value);
    }

    private Slice acquireOneOff(long n, long value) {
        synchronized (this) {
            ensureOpen();
            checkMonotonic(value);
        }
        VkGpuBuffer buffer = allocator.createBuffer(n, usageFlags(), requiredFlags(), preferredFlags(), true);
        synchronized (this) {
            if (closed) {
                buffer.closeRetired();
                throw new IllegalStateException("staging ring closed");
            }
            oneOffs.addLast(new OneOff(buffer, value));
        }
        LOGGER.debug("Vulkan {} staging request of {} bytes exceeds the {}-byte ring; using a one-off buffer",
                upload ? "upload" : "readback", n, capacity);
        return new Slice(allocator, buffer, 0, n, value);
    }

    private void ensureRing() {
        if (ring == null) {
            ring = allocator.createBuffer(capacity, usageFlags(), requiredFlags(), preferredFlags(), true);
            LOGGER.debug("Vulkan {} staging ring: {} bytes, memory type {}", upload ? "upload" : "readback",
                    capacity, ring.memoryTypeIndex());
        }
    }

    private void checkMonotonic(long value) {
        if (value < lastValue) {
            throw new IllegalStateException("timeline value " + value + " is below the previous value " + lastValue);
        }
        lastValue = value;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("staging ring closed");
        }
    }

    private int usageFlags() {
        return upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    private int requiredFlags() {
        return upload
                ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }

    private int preferredFlags() {
        return upload ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    /** In-flight ring bytes of one batch, ending at virtual position {@code end}. */
    private static final class Span {
        long end;
        final long value;

        Span(long end, long value) {
            this.end = end;
            this.value = value;
        }
    }

    /** A dedicated buffer for one oversize request. */
    private static final class OneOff {
        final VkGpuBuffer buffer;
        final long value;

        OneOff(VkGpuBuffer buffer, long value) {
            this.buffer = buffer;
            this.value = value;
        }
    }

    /** One acquired span of the ring. */
    public static final class Slice {

        private final VkMemoryAllocator allocator;
        private final VkGpuBuffer buffer;
        private final long offset;
        private final long size;
        private final ByteBuffer mapped;
        private final long batchValue;

        private Slice(VkMemoryAllocator allocator, VkGpuBuffer buffer, long offset, long size, long batchValue) {
            this.allocator = allocator;
            this.buffer = buffer;
            this.offset = offset;
            this.size = size;
            this.mapped = MemoryUtil.memByteBuffer(buffer.hostPtr() + offset, (int) size)
                    .order(ByteOrder.nativeOrder());
            this.batchValue = batchValue;
        }

        /** The {@code VkBuffer} the span lives in. */
        public long vkBuffer() {
            return buffer.vkBuffer();
        }

        /** The staging buffer the span lives in (for hazard tracking and invalidation). */
        public VkGpuBuffer buffer() {
            return buffer;
        }

        /** Byte offset of the span in {@link #vkBuffer()}. */
        public long offset() {
            return offset;
        }

        public long size() {
            return size;
        }

        /** Host view of the span, positioned at 0 with limit {@link #size()}, in native byte order. */
        public ByteBuffer mapped() {
            return mapped;
        }

        /** Timeline value of the batch that uses the span. */
        public long batchValue() {
            return batchValue;
        }

        /** Makes device writes to the span visible to the host when the memory is not coherent. */
        public void invalidate() {
            allocator.invalidate(buffer, offset, size);
        }

        @Override
        public String toString() {
            return "Slice{offset=" + offset + ", size=" + size + ", value=" + batchValue + ", buffer=" + buffer + '}';
        }
    }
}
