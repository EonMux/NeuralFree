package com.eonmux.neuralfree.gguf.hub.slice;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Waits for the tasks of the slicer's thread pools, rethrowing a task's failure in the caller's terms: its
 * {@link IOException} or {@link InterruptedException} as itself, a runtime exception as itself, anything else
 * wrapped in an {@link IOException}.
 */
final class SliceFutures {

    private SliceFutures() {
    }

    /** Waits for {@code future} and returns its value, rethrowing its failure. */
    static <T> T await(Future<T> future) throws IOException, InterruptedException {
        try {
            return future.get();
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException) {
                throw (IOException) cause;
            }
            if (cause instanceof InterruptedException) {
                throw (InterruptedException) cause;
            }
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            throw new IOException(cause);
        }
    }
}
