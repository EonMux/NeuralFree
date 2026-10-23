package com.eonmux.neuralfree.acceleration.backend.ocl;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Direct host staging buffers of one {@link OclComputeQueue}.
 *
 * <p>A non-blocking {@code clEnqueueWriteBuffer}/{@code clEnqueueReadBuffer} needs its host memory to
 * stay valid until the command completes, so every staging buffer handed to the driver is retained
 * in flight until the queue's {@code finish()} ({@link #recycleInFlight()}) and then moves to the
 * free list. A finish therefore closes one reuse cycle.</p>
 *
 * <p>Two classes of buffers are kept differently:</p>
 * <ul>
 *   <li><b>Small</b> (capacity up to {@link #MAX_SMALL_BUFFER_BYTES}): the per-step transfers
 *       (token ids, positions, logits). They are kept under a count bound and a byte budget; above
 *       either, the largest are dropped first.</li>
 *   <li><b>Large</b>: bulk transfers such as weight-upload chunks. A large buffer stays in the free
 *       list only while it is reused: one that was not handed to the driver during the cycle that
 *       just finished is dropped, so a streaming upload keeps recycling its chunk buffers, and they
 *       become garbage-collectable at the first finish after the upload instead of pinning direct
 *       memory for the queue's lifetime. Small requests are never served from large buffers.</li>
 * </ul>
 *
 * <p>Not thread-safe; used by the queue's single enqueuing thread.</p>
 */
final class OclStagingPool {

    /** Allocation granularity of staging buffers. */
    static final int ALIGNMENT = 256;
    /** Most small free buffers kept. */
    static final int MAX_FREE_BUFFERS = 32;
    /** Largest small buffer; covers a 1M-entry float logits read. */
    static final long MAX_SMALL_BUFFER_BYTES = 4L << 20;
    /** Most bytes kept in small free buffers. */
    static final long MAX_FREE_BYTES = 16L << 20;

    private final int maxFreeBuffers;
    private final long maxSmallBufferBytes;
    private final long maxFreeBytes;
    private final List<ByteBuffer> inFlight = new ArrayList<>();
    private final List<ByteBuffer> smallFree = new ArrayList<>();
    private final List<LargeBuffer> largeFree = new ArrayList<>();
    private long smallFreeBytes;
    private long largeFreeBytes;
    /** Number of completed reuse cycles. */
    private long cycle;

    OclStagingPool() {
        this(MAX_FREE_BUFFERS, MAX_SMALL_BUFFER_BYTES, MAX_FREE_BYTES);
    }

    /**
     * @param maxFreeBuffers      most small free buffers kept
     * @param maxSmallBufferBytes largest capacity treated as small
     * @param maxFreeBytes        most total capacity of small free buffers
     */
    OclStagingPool(int maxFreeBuffers, long maxSmallBufferBytes, long maxFreeBytes) {
        if (maxFreeBuffers < 0 || maxSmallBufferBytes < 0 || maxFreeBytes < 0) {
            throw new IllegalArgumentException("staging pool bounds must not be negative: buffers=" + maxFreeBuffers
                    + ", small buffer bytes=" + maxSmallBufferBytes + ", total bytes=" + maxFreeBytes);
        }
        this.maxFreeBuffers = maxFreeBuffers;
        this.maxSmallBufferBytes = maxSmallBufferBytes;
        this.maxFreeBytes = maxFreeBytes;
    }

    /**
     * A cleared native-order direct buffer of at least {@code sizeInBytes}: the smallest free buffer
     * of the request's class that fits, or a new one rounded up to {@link #ALIGNMENT}. The buffer is
     * not yet in flight.
     *
     * @throws IllegalArgumentException when {@code sizeInBytes} is negative
     */
    ByteBuffer acquire(int sizeInBytes) {
        if (sizeInBytes < 0) {
            throw new IllegalArgumentException("staging size must not be negative: " + sizeInBytes);
        }
        long aligned = ((sizeInBytes + (long) ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        int capacity = (int) Math.min(aligned, Integer.MAX_VALUE);
        ByteBuffer reused = capacity <= maxSmallBufferBytes ? takeSmall(sizeInBytes) : takeLarge(sizeInBytes);
        if (reused != null) {
            reused.clear();
            return reused;
        }
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** Keeps {@code buf} alive until {@link #recycleInFlight()}: the driver may still access it. */
    void retain(ByteBuffer buf) {
        inFlight.add(buf);
    }

    /** Returns a buffer that was acquired but never handed to the driver. */
    void release(ByteBuffer buf) {
        offer(buf);
        trimSmall();
    }

    /**
     * Ends a reuse cycle: moves every in-flight buffer to the free list, drops the large buffers the
     * cycle did not use and trims the small ones to their bounds. Call only after the queue has
     * finished, when the driver no longer accesses any of them.
     */
    void recycleInFlight() {
        cycle++;
        Iterator<LargeBuffer> it = largeFree.iterator();
        while (it.hasNext()) {
            LargeBuffer idle = it.next();
            if (idle.cycle < cycle) {
                largeFreeBytes -= idle.buffer.capacity();
                it.remove();
            }
        }
        for (ByteBuffer buf : inFlight) {
            offer(buf);
        }
        inFlight.clear();
        trimSmall();
    }

    /** Drops every buffer, in flight or free. */
    void clear() {
        inFlight.clear();
        smallFree.clear();
        largeFree.clear();
        smallFreeBytes = 0;
        largeFreeBytes = 0;
    }

    /** Free buffers of both classes. */
    int freeCount() {
        return smallFree.size() + largeFree.size();
    }

    /** Capacity of the free buffers of both classes. */
    long freeBytes() {
        return smallFreeBytes + largeFreeBytes;
    }

    int inFlightCount() {
        return inFlight.size();
    }

    private ByteBuffer takeSmall(int sizeInBytes) {
        int best = -1;
        for (int i = smallFree.size() - 1; i >= 0; i--) {
            int cap = smallFree.get(i).capacity();
            if (cap >= sizeInBytes && (best < 0 || cap < smallFree.get(best).capacity())) {
                best = i;
            }
        }
        if (best < 0) {
            return null;
        }
        ByteBuffer buf = smallFree.remove(best);
        smallFreeBytes -= buf.capacity();
        return buf;
    }

    private ByteBuffer takeLarge(int sizeInBytes) {
        int best = -1;
        for (int i = largeFree.size() - 1; i >= 0; i--) {
            int cap = largeFree.get(i).buffer.capacity();
            if (cap >= sizeInBytes && (best < 0 || cap < largeFree.get(best).buffer.capacity())) {
                best = i;
            }
        }
        if (best < 0) {
            return null;
        }
        ByteBuffer buf = largeFree.remove(best).buffer;
        largeFreeBytes -= buf.capacity();
        return buf;
    }

    /** Adds a buffer to the free list of its class, stamped with the current cycle. */
    private void offer(ByteBuffer buf) {
        if (buf.capacity() <= maxSmallBufferBytes) {
            smallFree.add(buf);
            smallFreeBytes += buf.capacity();
        } else {
            largeFree.add(new LargeBuffer(buf, cycle));
            largeFreeBytes += buf.capacity();
        }
    }

    private void trimSmall() {
        while (!smallFree.isEmpty() && (smallFree.size() > maxFreeBuffers || smallFreeBytes > maxFreeBytes)) {
            int largest = 0;
            for (int i = 1; i < smallFree.size(); i++) {
                if (smallFree.get(i).capacity() > smallFree.get(largest).capacity()) {
                    largest = i;
                }
            }
            smallFreeBytes -= smallFree.remove(largest).capacity();
        }
    }

    /** A free large buffer and the reuse cycle in which it was last handed to the driver. */
    private static final class LargeBuffer {
        final ByteBuffer buffer;
        final long cycle;

        LargeBuffer(ByteBuffer buffer, long cycle) {
            this.buffer = buffer;
            this.cycle = cycle;
        }
    }
}
