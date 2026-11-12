package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.jocl.CLException;
import org.jocl.cl_event;

import java.util.ArrayDeque;

import static org.jocl.CL.clReleaseEvent;

/**
 * The non-blocking writes of one {@link OclComputeQueue} whose staging buffers may still be in flight, oldest first,
 * each with a retained reference to its event, its staging ticket ({@link OclStagingPool#retain}) and the capacity of
 * its staging buffer.
 *
 * <p>The queue keeps the staging of these writes under {@link #MAX_BYTES}: before a write that would take it above
 * the budget, the queue waits for the oldest writes, so a run of writes that never reads or finishes (parameter
 * uploads, a streamed weight upload) holds a bounded amount of direct memory instead of every byte uploaded since
 * the last synchronization. A write completed by the in-order queue (a later read completed, or the queue finished)
 * is dropped with {@link #releaseThrough} or {@link #releaseAll}.</p>
 *
 * <p>Not thread-safe; used by the queue's single enqueuing thread.</p>
 */
final class OclPendingWrites {
    private static final Logger LOGGER = LoggingUtil.getLogger(OclPendingWrites.class);

    /**
     * Most staging bytes of writes in flight before a new write waits for the oldest ones: two default weight-upload
     * chunks ({@code GpuWeightUploader.DEFAULT_STAGING_BYTES}), so the host copy of one chunk overlaps the transfer
     * of the one before it.
     */
    static final long MAX_BYTES = 128L << 20;

    private final long maxBytes;
    private final ArrayDeque<Write> writes = new ArrayDeque<>();
    private long bytes;

    OclPendingWrites() {
        this(MAX_BYTES);
    }

    /** @param maxBytes most staging bytes in flight before a new write waits */
    OclPendingWrites(long maxBytes) {
        if (maxBytes < 0) {
            throw new IllegalArgumentException("write staging budget must not be negative: " + maxBytes);
        }
        this.maxBytes = maxBytes;
    }

    /** Tracks a write; {@code event} is a reference this object releases. */
    void add(cl_event event, long ticket, long stagingBytes) {
        writes.addLast(new Write(event, ticket, stagingBytes));
        bytes += stagingBytes;
    }

    /** Whether a write staging {@code incoming} more bytes must first wait for the oldest write. */
    boolean overBudgetWith(long incoming) {
        return !writes.isEmpty() && bytes + incoming > maxBytes;
    }

    /** Removes the oldest write; its event becomes the caller's to release. Null when no write is tracked. */
    Write pollOldest() {
        Write w = writes.pollFirst();
        if (w != null) {
            bytes -= w.stagingBytes;
        }
        return w;
    }

    /** Drops every write whose staging ticket is at most {@code ticket}: the in-order queue completed them. */
    void releaseThrough(long ticket) {
        while (!writes.isEmpty() && writes.peekFirst().ticket <= ticket) {
            release(pollOldest().event);
        }
    }

    /** Drops every write. */
    void releaseAll() {
        while (!writes.isEmpty()) {
            release(pollOldest().event);
        }
    }

    /** Staging bytes of the tracked writes. */
    long bytes() {
        return bytes;
    }

    int count() {
        return writes.size();
    }

    /**
     * Releases one event reference. A failed release leaks only that reference, so it is logged rather than thrown
     * from the write or finish that dropped the event.
     */
    static void release(cl_event event) {
        try {
            clReleaseEvent(event);
        } catch (CLException e) {
            LOGGER.warn("clReleaseEvent of a completed OpenCL write failed: {}", OclErrors.nameOf(e.getStatus()));
        }
    }

    /** One tracked write. */
    static final class Write {
        final cl_event event;
        final long ticket;
        final long stagingBytes;

        Write(cl_event event, long ticket, long stagingBytes) {
            this.event = event;
            this.ticket = ticket;
            this.stagingBytes = stagingBytes;
        }
    }
}
