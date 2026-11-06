package com.eonmux.neuralfree.acceleration.backend;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * One captured {@link GpuGraph} per shape, in least-recently-used order: the first step of a
 * shape is captured, every later step of the same shape replays the stored graph. The shape key
 * is caller-defined (for example token count and batch size); a key collision only costs a
 * wrong reuse, which the caller prevents by deriving the key from everything the captured
 * commands depend on.
 *
 * <p>{@link #getOrCapture} returns the stored graph without re-running its step, so the step's
 * body runs once per key. The cache is bounded: admitting one more graph than
 * {@link #maxGraphs()} closes the least recently used one. {@link #close()} closes every stored
 * graph. Not thread-safe.</p>
 */
public final class GpuGraphCache implements AutoCloseable {

    /** Enqueues the commands of one capture on a queue and returns the graph they were captured into. */
    public interface CaptureStep {

        /**
         * Runs once per shape key: begins the capture on the queue, enqueues the step's
         * commands and ends the capture.
         *
         * @throws RuntimeException the step's failure; the queue's capture is then aborted by
         *                          the caller of {@link #getOrCapture} when the step went
         *                          through {@link #captured}
         */
        GpuGraph capture();
    }

    /**
     * A {@link CaptureStep} that captures {@code body} on {@code queue}: begins the queue's
     * capture, runs the body, ends the capture and returns the graph. When the body or the end
     * of the capture fails, the capture is aborted before the failure propagates, so the queue
     * is never left capturing.
     *
     * @throws IllegalArgumentException when an argument is null
     */
    public static CaptureStep captured(ComputeQueue queue, Runnable body) {
        Objects.requireNonNull(queue, "queue must not be null");
        Objects.requireNonNull(body, "body must not be null");
        return () -> {
            queue.beginCapture();
            try {
                body.run();
                return queue.endCapture();
            } catch (RuntimeException e) {
                try {
                    queue.abortCapture();
                } catch (RuntimeException abort) {
                    e.addSuppressed(abort);
                }
                throw e;
            }
        };
    }

    private final int maxGraphs;
    private final Map<Object, GpuGraph> graphs = new HashMap<>();
    /** Insertion and access order of the keys, for the least recently used eviction. */
    private final Set<Object> order = new LinkedHashSet<>();
    private boolean closed;

    /**
     * @param maxGraphs the graphs kept at once, at least 1
     * @throws IllegalArgumentException when {@code maxGraphs < 1}
     */
    public GpuGraphCache(int maxGraphs) {
        if (maxGraphs < 1) {
            throw new IllegalArgumentException("maxGraphs must be at least 1, got " + maxGraphs);
        }
        this.maxGraphs = maxGraphs;
    }

    /** The graphs kept at once. */
    public int maxGraphs() {
        return maxGraphs;
    }

    /** The graphs currently stored. */
    public int size() {
        return graphs.size();
    }

    /** True while {@code shapeKey} has a stored graph. */
    public boolean contains(Object shapeKey) {
        Objects.requireNonNull(shapeKey, "shapeKey must not be null");
        return graphs.containsKey(shapeKey);
    }

    /**
     * The graph of {@code shapeKey}, capturing it through {@code step} on the first call.
     * The key is marked most recently used; storing one graph beyond {@link #maxGraphs()} closes
     * the least recently used one.
     *
     * @throws IllegalArgumentException when an argument is null or the step returned null
     * @throws IllegalStateException    when the cache is closed
     * @throws RuntimeException         the step's failure; no graph is then stored for the key
     */
    public GpuGraph getOrCapture(Object shapeKey, CaptureStep step) {
        Objects.requireNonNull(shapeKey, "shapeKey must not be null");
        Objects.requireNonNull(step, "step must not be null");
        if (closed) {
            throw new IllegalStateException("graph cache already closed");
        }
        GpuGraph graph = graphs.get(shapeKey);
        if (graph != null) {
            touch(shapeKey);
            return graph;
        }
        GpuGraph captured = step.capture();
        if (captured == null) {
            throw new IllegalArgumentException("capture step of " + shapeKey + " returned null");
        }
        graphs.put(shapeKey, captured);
        order.add(shapeKey);
        evictOverCapacity();
        return captured;
    }

    /** Marks {@code shapeKey} most recently used. */
    private void touch(Object shapeKey) {
        order.remove(shapeKey);
        order.add(shapeKey);
    }

    /** Closes the least recently used stored graphs until the cache holds at most {@link #maxGraphs()}. */
    private void evictOverCapacity() {
        while (graphs.size() > maxGraphs) {
            Iterator<Object> oldest = order.iterator();
            Object key = oldest.next();
            oldest.remove();
            GpuGraph evicted = graphs.remove(key);
            try {
                evicted.close();
            } catch (RuntimeException ignored) {
                // closing a pooled graph whose backend is gone is not a cache failure
            }
        }
    }

    /**
     * Closes every stored graph. Idempotent.
     *
     * @throws RuntimeException the first failure, after every graph was closed
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (GpuGraph graph : graphs.values()) {
            try {
                graph.close();
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        graphs.clear();
        order.clear();
        if (failure != null) {
            throw failure;
        }
    }
}
