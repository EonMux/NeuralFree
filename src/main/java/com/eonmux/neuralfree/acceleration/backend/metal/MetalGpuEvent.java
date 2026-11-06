package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

/**
 * A point in one Metal command queue, identified by the value the backend's shared
 * {@code MTLSharedEvent} is signaled at when the batch holding the point completes.
 *
 * <ul>
 *   <li>{@link #awaitCompletion()} commits the queue's open batch and blocks until the point is
 *       reached, then completes every read the queue issued up to it, so read destinations are
 *       valid afterwards.</li>
 *   <li>Events are caller-owned but hold no framework resource: closing one only invalidates the
 *       handle, and the queue's shared event is released when the queue closes.</li>
 *   <li>Any queue of the same backend may wait on it; a queue waiting on its own event skips the
 *       wait because its batches execute in order.</li>
 *   <li>Timing is not measured, so {@link #getElapsedNanos()} reports -1.</li>
 * </ul>
 */
public final class MetalGpuEvent implements GpuEvent {

    private final MetalComputeQueue owner;
    private final MetalComputeQueue.Batch batch;
    private volatile boolean closed;

    MetalGpuEvent(MetalComputeQueue owner, MetalComputeQueue.Batch batch) {
        if (owner == null || batch == null) {
            throw new IllegalArgumentException("Metal event needs a queue and a batch");
        }
        this.owner = owner;
        this.batch = batch;
    }

    /**
     * Blocks until the queue reaches this point.
     *
     * @throws IllegalStateException when this handle or the backend was closed
     * @throws RuntimeException      the framework failure of a command buffer
     */
    @Override
    public void awaitCompletion() throws InterruptedException {
        if (closed) {
            throw new IllegalStateException("Metal event already closed");
        }
        owner.awaitEvent(this);
    }

    /**
     * True when the batch has completed. Read destinations become valid only through
     * {@link #awaitCompletion()} or the queue's {@code finish()}. A closed handle reports true.
     */
    @Override
    public boolean isComplete() {
        if (closed) {
            return true;
        }
        owner.backend().ensureOpen();
        long value = batch.value;
        return value > 0 && owner.signaledValue() >= value;
    }

    /** Releases this handle; the shared event stays with the backend. Idempotent. */
    @Override
    public void close() {
        closed = true;
    }

    /**
     * Timing is not measured on Metal batches.
     *
     * @return -1
     * @throws IllegalStateException when the handle was closed
     */
    @Override
    public long getElapsedNanos() {
        if (closed) {
            throw new IllegalStateException("Metal event already closed");
        }
        return -1;
    }

    /** True once this handle was closed. */
    public boolean isClosed() {
        return closed;
    }

    /** The shared-event value this point completes at; 0 until the batch commits. */
    public long value() {
        return batch.value;
    }

    /** The queue whose batch records this point. */
    public MetalComputeQueue owner() {
        return owner;
    }

    /** Commits the producer queue so another queue can wait on this point. */
    void submit() {
        owner.flush();
    }

    @Override
    public String toString() {
        return "MetalGpuEvent{value=" + batch.value + ", " + owner.getQueueType()
                + (closed ? ", closed" : "") + "}";
    }
}
