package com.eonmux.neuralfree.gguf.mm;

import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;

/**
 * Single-flight encodes per key ({@link MediaContext}): while one caller encodes an item, a concurrent caller with
 * the same key waits for that encode and takes its result (or its failure) instead of encoding the item again, as
 * vLLM's encoder cache joins requests on the same media hash. A key leaves the table when its encode ends, so a
 * later caller encodes afresh (and normally hits the cache the encode filled). Thread-safe.
 *
 * @param <K> the item key
 * @param <V> the encoding
 */
final class EncodeFlights<K, V> {

    private final ConcurrentHashMap<K, CompletableFuture<V>> running = new ConcurrentHashMap<>();

    /**
     * The encoding of {@code key}: the running encode's when another caller is encoding it, else
     * {@code encode}'s, run on this thread.
     *
     * @throws RuntimeException the encode's own failure, rethrown to every caller that waited on it
     * @throws IllegalStateException when the caller is interrupted while it waits
     */
    V run(K key, Supplier<V> encode) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(encode, "encode");
        CompletableFuture<V> mine = new CompletableFuture<>();
        CompletableFuture<V> other = running.putIfAbsent(key, mine);
        if (other != null) {
            return await(other);
        }
        try {
            V out = encode.get();
            mine.complete(out);
            return out;
        } catch (RuntimeException | Error e) {
            mine.completeExceptionally(e);
            throw e;
        } finally {
            running.remove(key, mine);
        }
    }

    /** How many keys are being encoded now. */
    int inFlight() {
        return running.size();
    }

    private static <V> V await(CompletableFuture<V> other) {
        try {
            return other.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for another request's encode of the same"
                    + " media item", e);
        } catch (CancellationException e) {
            throw new IllegalStateException("another request's encode of the same media item was cancelled", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            if (cause instanceof Error) {
                throw (Error) cause;
            }
            throw new IllegalStateException(String.valueOf(cause), cause);
        }
    }
}
