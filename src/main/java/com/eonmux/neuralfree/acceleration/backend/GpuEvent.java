package com.eonmux.neuralfree.acceleration.backend;

public interface GpuEvent extends AutoCloseable {
    void awaitCompletion() throws InterruptedException;
    boolean isComplete();
    void close();

    /**
     * Returns GPU-side elapsed time in nanoseconds.
     * Requires profiling-enabled command queue. Must be called after event completes.
     * Default: returns -1 (not supported).
     */
    default long getElapsedNanos() { return -1; }
}
