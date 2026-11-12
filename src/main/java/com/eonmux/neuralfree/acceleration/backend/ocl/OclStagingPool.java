package com.eonmux.neuralfree.acceleration.backend.ocl;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * Direct host staging buffers of one {@link OclComputeQueue}.
 *
 * <p>A non-blocking {@code clEnqueueWriteBuffer}/{@code clEnqueueReadBuffer} needs its host memory to
 * stay valid until the command completes, so every staging buffer handed to the driver is retained
 * in flight, under a ticket in retention order, until the queue knows the command completed: at the
 * queue's {@code finish()} ({@link #recycleInFlight()}), which closes one reuse cycle, or when a
 * later read on the in-order queue completed ({@link #recycleThrough}), which frees every buffer
 * retained up to that read's ticket without closing the cycle.</p>
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
 *       memory for the queue's lifetime. The free large buffers other than the largest are kept
 *       under a byte budget, dropping the longest-free first, so buffers recycled before a finish
 *       (by a completed read or a write the queue waited for) that no later request fits do not
 *       accumulate. The largest free buffer is exempt, because it serves every request another
 *       free buffer could: a transfer above the budget repeated every cycle keeps reusing its
 *       buffer, which the idle-cycle rule still drops once the transfers stop. Small requests are
 *       never served from large buffers.</li>
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
    /** Most bytes kept in large free buffers besides the largest one. */
    static final long MAX_LARGE_FREE_BYTES = 128L << 20;

    private final int maxFreeBuffers;
    private final long maxSmallBufferBytes;
    private final long maxFreeBytes;
    private final long maxLargeFreeBytes;
    private final ArrayDeque<InFlight> inFlight = new ArrayDeque<>();
    private final List<ByteBuffer> smallFree = new ArrayList<>();
    private final List<LargeBuffer> largeFree = new ArrayList<>();
    private long smallFreeBytes;
    private long largeFreeBytes;
    /** Number of completed reuse cycles. */
    private long cycle;
    /** Tickets handed out by {@link #retain}; the latest one. */
    private long tickets;

    OclStagingPool() {
        this(MAX_FREE_BUFFERS, MAX_SMALL_BUFFER_BYTES, MAX_FREE_BYTES, MAX_LARGE_FREE_BYTES);
    }

    /**
     * @param maxFreeBuffers      most small free buffers kept
     * @param maxSmallBufferBytes largest capacity treated as small
     * @param maxFreeBytes        most total capacity of small free buffers
     */
    OclStagingPool(int maxFreeBuffers, long maxSmallBufferBytes, long maxFreeBytes) {
        this(maxFreeBuffers, maxSmallBufferBytes, maxFreeBytes, MAX_LARGE_FREE_BYTES);
    }

    /**
     * @param maxFreeBuffers      most small free buffers kept
     * @param maxSmallBufferBytes largest capacity treated as small
     * @param maxFreeBytes        most total capacity of small free buffers
     * @param maxLargeFreeBytes   most total capacity of large free buffers besides the largest
     */
    OclStagingPool(int maxFreeBuffers, long maxSmallBufferBytes, long maxFreeBytes, long maxLargeFreeBytes) {
        if (maxFreeBuffers < 0 || maxSmallBufferBytes < 0 || maxFreeBytes < 0 || maxLargeFreeBytes < 0) {
            throw new IllegalArgumentException("staging pool bounds must not be negative: buffers=" + maxFreeBuffers
                    + ", small buffer bytes=" + maxSmallBufferBytes + ", total bytes=" + maxFreeBytes
                    + ", large total bytes=" + maxLargeFreeBytes);
        }
        this.maxFreeBuffers = maxFreeBuffers;
        this.maxSmallBufferBytes = maxSmallBufferBytes;
        this.maxFreeBytes = maxFreeBytes;
        this.maxLargeFreeBytes = maxLargeFreeBytes;
    }

    /**
     * The capacity of a new staging buffer for {@code sizeInBytes}: the size rounded up to {@link #ALIGNMENT}.
     *
     * @throws IllegalArgumentException when {@code sizeInBytes} is negative
     */
    static int capacityFor(int sizeInBytes) {
        if (sizeInBytes < 0) {
            throw new IllegalArgumentException("staging size must not be negative: " + sizeInBytes);
        }
        long aligned = ((sizeInBytes + (long) ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        return (int) Math.min(aligned, Integer.MAX_VALUE);
    }

    /**
     * A cleared native-order direct buffer of at least {@code sizeInBytes}: the smallest free buffer
     * of the request's class that fits, or a new one rounded up to {@link #ALIGNMENT}. The buffer is
     * not yet in flight.
     *
     * @throws IllegalArgumentException when {@code sizeInBytes} is negative
     */
    ByteBuffer acquire(int sizeInBytes) {
        int capacity = capacityFor(sizeInBytes);
        ByteBuffer reused = capacity <= maxSmallBufferBytes ? takeSmall(sizeInBytes) : takeLarge(sizeInBytes);
        if (reused != null) {
            reused.clear();
            return reused;
        }
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /**
     * Keeps {@code buf} alive until {@link #recycleInFlight()} or a {@link #recycleThrough} of its
     * ticket: the driver may still access it.
     *
     * @return the buffer's ticket, greater than every ticket handed out before
     */
    long retain(ByteBuffer buf) {
        tickets++;
        inFlight.addLast(new InFlight(buf, tickets));
        return tickets;
    }

    /**
     * Moves the in-flight buffers retained up to and including {@code ticket} to the free list and
     * trims the free buffers to their bounds; the reuse cycle stays open. Call only once the driver no
     * longer accesses any of them: a command the in-order queue ran after all of them completed.
     */
    void recycleThrough(long ticket) {
        boolean moved = false;
        while (!inFlight.isEmpty() && inFlight.peekFirst().ticket <= ticket) {
            offer(inFlight.pollFirst().buffer);
            moved = true;
        }
        if (moved) {
            trimSmall();
            trimLarge();
        }
    }

    /** Returns a buffer that was acquired but never handed to the driver. */
    void release(ByteBuffer buf) {
        offer(buf);
        trimSmall();
        trimLarge();
    }

    /**
     * Ends a reuse cycle: drops the large free buffers the cycle did not hand to the driver (those
     * freed mid-cycle were handed to it in this cycle and stay), moves every in-flight buffer to the
     * free list and trims the free buffers to their bounds. Call only after the queue has
     * finished, when the driver no longer accesses any of them.
     */
    void recycleInFlight() {
        Iterator<LargeBuffer> it = largeFree.iterator();
        while (it.hasNext()) {
            LargeBuffer idle = it.next();
            if (idle.cycle < cycle) {
                largeFreeBytes -= idle.buffer.capacity();
                it.remove();
            }
        }
        for (InFlight held : inFlight) {
            offer(held.buffer);
        }
        inFlight.clear();
        cycle++;
        trimSmall();
        trimLarge();
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

    /**
     * Drops the largest small free buffers until both bounds hold: one sort by capacity, then
     * removals from the end, so trimming any number of buffers costs {@code O(n log n)}.
     */
    private void trimSmall() {
        if (smallFree.size() <= maxFreeBuffers && smallFreeBytes <= maxFreeBytes) {
            return;
        }
        smallFree.sort(Comparator.comparingInt(ByteBuffer::capacity));
        while (!smallFree.isEmpty() && (smallFree.size() > maxFreeBuffers || smallFreeBytes > maxFreeBytes)) {
            smallFreeBytes -= smallFree.remove(smallFree.size() - 1).capacity();
        }
    }

    /**
     * Drops the large free buffers that have been free longest, except the largest one (the most recently freed
     * among equals), until the others' capacity is within its bound: one pass over the free list.
     */
    private void trimLarge() {
        LargeBuffer largest = null;
        for (LargeBuffer free : largeFree) {
            if (largest == null || free.buffer.capacity() >= largest.buffer.capacity()) {
                largest = free;
            }
        }
        if (largest == null) {
            return;
        }
        long excess = largeFreeBytes - largest.buffer.capacity() - maxLargeFreeBytes;
        if (excess <= 0) {
            return;
        }
        List<LargeBuffer> kept = new ArrayList<>(largeFree.size());
        for (LargeBuffer free : largeFree) {
            if (excess > 0 && free != largest) {
                excess -= free.buffer.capacity();
                largeFreeBytes -= free.buffer.capacity();
            } else {
                kept.add(free);
            }
        }
        largeFree.clear();
        largeFree.addAll(kept);
    }

    /** An in-flight buffer and the ticket {@link #retain} gave it. */
    private static final class InFlight {
        final ByteBuffer buffer;
        final long ticket;

        InFlight(ByteBuffer buffer, long ticket) {
            this.buffer = buffer;
            this.ticket = ticket;
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
