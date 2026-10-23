package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

/**
 * A point on a compute queue's timeline semaphore: the event completes when the queue's timeline
 * reaches {@link #value()}, the value the batch holding the command signals.
 *
 * <p>Waiting submits that batch first when it is still being recorded, so an event never waits on
 * work that was not submitted. Completing an event retires the queue's batches up to its value, which
 * copies finished readbacks into their destination arrays. Any queue of the same backend can wait on
 * the event on the GPU (a timeline-semaphore wait in its next submission). The event holds no Vulkan
 * object; {@link #close()} only marks it unusable as a wait.</p>
 *
 * <p>Thread-safe: completion goes through the owning queue, which serialises it.</p>
 */
public final class VkGpuEvent implements GpuEvent {

    private final VkComputeQueue owner;
    private final long value;
    private volatile boolean closed;

    /**
     * @param owner the queue whose timeline the value belongs to
     * @param value the timeline value that completes the event; 0 is complete from the start
     */
    public VkGpuEvent(VkComputeQueue owner, long value) {
        if (owner == null) {
            throw new IllegalArgumentException("event owner queue must not be null");
        }
        if (value < 0) {
            throw new IllegalArgumentException("timeline value must not be negative, got " + value);
        }
        this.owner = owner;
        this.value = value;
    }

    /** The queue whose timeline this event belongs to. */
    public VkComputeQueue owner() {
        return owner;
    }

    /** The timeline value that completes this event. */
    public long value() {
        return value;
    }

    public boolean isClosed() {
        return closed;
    }

    /**
     * Submits the owning batch if needed, blocks until the timeline reaches the value, and retires
     * the owner's batches up to it (readback destinations are then valid).
     *
     * @throws InterruptedException when the waiting thread is interrupted
     * @throws IllegalStateException when the event is closed or its queue was closed before completing it
     */
    @Override
    public void awaitCompletion() throws InterruptedException {
        if (closed) {
            throw new IllegalStateException("Vulkan event (value " + value + ") already closed");
        }
        owner.awaitValue(value);
    }

    /**
     * Submits the owning batch if needed and reports whether the timeline reached the value,
     * retiring the owner's batches up to it when it did. A closed event reports true.
     */
    @Override
    public boolean isComplete() {
        return closed || owner.pollValue(value);
    }

    /** Marks the event closed; it holds no Vulkan object. Idempotent. */
    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "VkGpuEvent{value=" + value + ", queue=" + owner + (closed ? ", closed" : "") + '}';
    }
}
