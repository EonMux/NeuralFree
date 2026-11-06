package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

/**
 * A point in one HIP stream, backed by a pooled timing-disabled event ({@code hipEventDisableTiming}).
 *
 * <ul>
 *   <li>{@link #awaitCompletion()} blocks until the stream reaches the point and then completes
 *       every read its queue issued up to that point, so read destinations are valid afterwards.</li>
 *   <li>Several handles may share one recorded event ({@code share()}): the backend keeps its own
 *       handle for staging memory while the caller holds another. The event returns to the
 *       {@link HipEventPool} once every handle is closed.</li>
 *   <li>Any queue of the same backend may wait on it; a queue waiting on its own event skips the
 *       wait because streams execute in order.</li>
 *   <li>Timing is disabled, so {@link #getElapsedNanos()} reports -1.</li>
 * </ul>
 * Handles may be awaited and closed from any thread.
 */
public final class HipGpuEvent implements GpuEvent {

    private final HipEventPool pool;
    private final Lease lease;
    private final HipComputeQueue owner;
    private final long sequence;
    private volatile boolean closed;

    /**
     * Wraps an event just recorded on {@code owner}'s stream.
     *
     * @param pool     pool the event returns to
     * @param event    the recorded event
     * @param owner    queue whose stream recorded it, or null
     * @param sequence the owner's sequence number of this point (0 without owner)
     */
    HipGpuEvent(HipEventPool pool, long event, HipComputeQueue owner, long sequence) {
        if (pool == null) {
            throw new IllegalArgumentException("HIP event pool must not be null");
        }
        this.pool = pool;
        this.lease = new Lease(event);
        this.owner = owner;
        this.sequence = sequence;
    }

    private HipGpuEvent(HipGpuEvent template) {
        this.pool = template.pool;
        this.lease = template.lease;
        this.owner = template.owner;
        this.sequence = template.sequence;
    }

    /**
     * Blocks until the stream reaches this point, then completes the owning queue's reads
     * issued up to it.
     *
     * @throws IllegalStateException when this handle or the backend was closed
     * @throws RuntimeException      the translated driver failure
     */
    @Override
    public void awaitCompletion() throws InterruptedException {
        synchronize();
        if (owner != null) {
            owner.retireReadsUpTo(sequence);
        }
    }

    /**
     * True when the stream has passed this point. Read destinations become valid only through
     * {@link #awaitCompletion()} or the queue's {@code finish()}. A closed handle reports true.
     *
     * @throws IllegalStateException when the backend was closed
     * @throws RuntimeException      the translated driver failure
     */
    @Override
    public boolean isComplete() {
        if (closed) {
            return true;
        }
        long event = liveEvent();
        return pool.driver().eventQuery(event);
    }

    /**
     * Releases this handle; the event returns to the pool once every shared handle is closed.
     * Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
        }
        if (lease.release()) {
            pool.recycle(lease.event);
        }
    }

    /**
     * Timing is disabled on pooled events.
     *
     * @return -1
     * @throws IllegalStateException when the handle was closed
     */
    @Override
    public long getElapsedNanos() {
        if (closed) {
            throw new IllegalStateException("HIP event already closed");
        }
        return -1;
    }

    /** True once this handle was closed. */
    public boolean isClosed() {
        return closed;
    }

    /**
     * The driver event, for waits by queues of the same backend.
     *
     * @throws IllegalStateException when this handle or the backend was closed
     */
    public long getHipEvent() {
        return liveEvent();
    }

    /** The queue whose stream recorded this event, or null. */
    public HipComputeQueue owner() {
        return owner;
    }

    /** The owning queue's sequence number of this point (0 without owner). */
    public long sequence() {
        return sequence;
    }

    /** True when the event comes from {@code eventPool}, i.e. from the same backend. */
    boolean isFrom(HipEventPool eventPool) {
        return pool == eventPool;
    }

    /**
     * Another handle on the same recorded point, closed independently of this one.
     *
     * @throws IllegalStateException when this handle was closed
     */
    HipGpuEvent share() {
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("cannot share a closed HIP event");
            }
            lease.retain();
        }
        return new HipGpuEvent(this);
    }

    /**
     * Blocks until the stream reaches this point ({@code hipEventSynchronize}) without
     * completing reads.
     *
     * @throws IllegalStateException when this handle or the backend was closed
     * @throws RuntimeException      the translated driver failure
     */
    void synchronize() {
        long event = liveEvent();
        pool.driver().eventSynchronize(event);
    }

    @Override
    public String toString() {
        return "HipGpuEvent{seq=" + sequence + (owner == null ? "" : ", " + owner.getQueueType())
                + (closed ? ", closed" : "") + "}";
    }

    private long liveEvent() {
        if (closed) {
            throw new IllegalStateException("HIP event already closed");
        }
        if (pool.isClosed()) {
            throw new IllegalStateException("HIP event used after its backend was closed");
        }
        pool.context().makeCurrent(pool.driver());
        return lease.event;
    }

    /** Reference count of the handles sharing one recorded event. */
    private static final class Lease {
        private final long event;
        private int refs = 1;

        Lease(long event) {
            this.event = event;
        }

        synchronized void retain() {
            if (refs <= 0) {
                throw new IllegalStateException("HIP event already returned to its pool");
            }
            refs++;
        }

        /** Drops one reference; true when it was the last. */
        synchronized boolean release() {
            refs--;
            return refs == 0;
        }
    }
}
