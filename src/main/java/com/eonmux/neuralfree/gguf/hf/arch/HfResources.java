package com.eonmux.neuralfree.gguf.hf.arch;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;

/**
 * The resources a Hugging Face model source opens while it is assembled and keeps until it is closed: base
 * checkpoints, side files, a draft's target model. {@link #close()} closes them in reverse registration order, closes
 * every one even when some fail, and rethrows the first failure with the others suppressed. Thread-safe.
 */
public final class HfResources implements AutoCloseable {

    private final Deque<AutoCloseable> open = new ArrayDeque<>();
    private boolean closed;

    /**
     * Registers {@code resource} and returns it.
     *
     * @throws IllegalStateException when the registry is closed; the resource is closed first
     */
    public <T extends AutoCloseable> T register(T resource) {
        Objects.requireNonNull(resource, "resource");
        synchronized (open) {
            if (!closed) {
                open.push(resource);
                return resource;
            }
        }
        IllegalStateException e = new IllegalStateException("the model source is closed");
        try {
            resource.close();
        } catch (Exception suppressed) {
            e.addSuppressed(suppressed);
        }
        throw e;
    }

    /** The number of registered resources not yet closed. */
    public int size() {
        synchronized (open) {
            return open.size();
        }
    }

    @Override
    public void close() throws Exception {
        Exception first = null;
        while (true) {
            AutoCloseable next;
            synchronized (open) {
                closed = true;
                next = open.poll();
            }
            if (next == null) {
                break;
            }
            try {
                next.close();
            } catch (Exception e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }
}
