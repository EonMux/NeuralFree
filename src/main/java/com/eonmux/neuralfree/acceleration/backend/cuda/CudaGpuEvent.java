package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import jcuda.CudaException;
import jcuda.driver.CUevent;

import static jcuda.driver.JCudaDriver.cuEventQuery;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;

/**
 * A point in one CUDA stream, backed by a pooled timing-disabled {@link CUevent}.
 *
 * <ul>
 *   <li>{@link #awaitCompletion()} blocks until the stream reaches the point and then
 *       completes every read its queue issued up to that point, so read destinations
 *       are valid afterwards.</li>
 *   <li>Several handles may share one recorded event ({@link #share()}): the backend
 *       keeps its own handle for staging memory while the caller holds another. The
 *       event returns to the {@link CudaEventPool} once every handle is closed.</li>
 *   <li>Any queue of the same backend may wait on it; a queue waiting on its own event
 *       skips the wait because streams execute in order.</li>
 *   <li>Timing is disabled, so {@link #getElapsedNanos()} reports -1.</li>
 * </ul>
 * Handles may be awaited and closed from any thread.
 */
public final class CudaGpuEvent implements GpuEvent {

    private final CudaEventPool pool;
    private final Lease lease;
    private final CudaComputeQueue owner;
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
    CudaGpuEvent(CudaEventPool pool, CUevent event, CudaComputeQueue owner, long sequence) {
        if (pool == null) {
            throw new IllegalArgumentException("CUDA event pool must not be null");
        }
        if (event == null) {
            throw new IllegalArgumentException("CUevent must not be null");
        }
        this.pool = pool;
        this.lease = new Lease(event);
        this.owner = owner;
        this.sequence = sequence;
    }

    private CudaGpuEvent(CudaGpuEvent template) {
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
        CUevent event = liveEvent();
        int result;
        try {
            result = cuEventQuery(event);
        } catch (CudaException e) {
            if (CudaErrors.isNotReady(e)) {
                return false;
            }
            throw CudaErrors.translate(e, "cuEventQuery", -1);
        }
        if (result == CudaErrors.SUCCESS) {
            return true;
        }
        if (result == CudaErrors.NOT_READY) {
            return false;
        }
        throw CudaErrors.translate(result, "cuEventQuery", -1);
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
            throw new IllegalStateException("CUDA event already closed");
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
    public CUevent getCUEvent() {
        return liveEvent();
    }

    /** The queue whose stream recorded this event, or null. */
    public CudaComputeQueue owner() {
        return owner;
    }

    /** The owning queue's sequence number of this point (0 without owner). */
    public long sequence() {
        return sequence;
    }

    /** True when the event comes from {@code eventPool}, i.e. from the same backend. */
    boolean isFrom(CudaEventPool eventPool) {
        return pool == eventPool;
    }

    /**
     * Another handle on the same recorded point, closed independently of this one.
     *
     * @throws IllegalStateException when this handle was closed
     */
    CudaGpuEvent share() {
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("cannot share a closed CUDA event");
            }
            lease.retain();
        }
        return new CudaGpuEvent(this);
    }

    /**
     * Blocks until the stream reaches this point ({@code cuEventSynchronize}) without
     * completing reads.
     *
     * @throws IllegalStateException when this handle or the backend was closed
     * @throws RuntimeException      the translated driver failure
     */
    void synchronize() {
        CUevent event = liveEvent();
        CudaErrors.check("cuEventSynchronize", () -> cuEventSynchronize(event));
    }

    @Override
    public String toString() {
        return "CudaGpuEvent{seq=" + sequence + (owner == null ? "" : ", " + owner.getQueueType())
                + (closed ? ", closed" : "") + "}";
    }

    private CUevent liveEvent() {
        if (closed) {
            throw new IllegalStateException("CUDA event already closed");
        }
        if (pool.isClosed()) {
            throw new IllegalStateException("CUDA event used after its backend was closed");
        }
        pool.context().makeCurrent();
        return lease.event;
    }

    /** Reference count of the handles sharing one recorded event. */
    private static final class Lease {
        private final CUevent event;
        private int refs = 1;

        Lease(CUevent event) {
            this.event = event;
        }

        synchronized void retain() {
            if (refs <= 0) {
                throw new IllegalStateException("CUDA event already returned to its pool");
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
