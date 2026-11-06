package com.eonmux.neuralfree.acceleration.backend;

/**
 * A caller-owned completion point on a GPU queue. Every returned event must be closed, preferably
 * with try-with-resources: waiting or finishing the queue does not release it. Closing an event
 * releases its handle without cancelling the command or waiting for completion. Backend shutdown
 * releases outstanding driver events as a safety net and warns about events callers left open.
 */
public interface GpuEvent extends AutoCloseable {
    void awaitCompletion() throws InterruptedException;
    boolean isComplete();

    /** Releases this handle; idempotent. A closed handle must not be used as a queue dependency. */
    @Override
    void close();

    /**
     * Returns GPU-side elapsed time in nanoseconds.
     * Requires profiling-enabled command queue. Must be called after event completes.
     * Default: returns -1 (not supported).
     */
    default long getElapsedNanos() { return -1; }
}
