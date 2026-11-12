package com.eonmux.neuralfree.gguf.mm;

import java.io.InterruptedIOException;
import java.nio.channels.ClosedByInterruptException;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;

/**
 * Single-flight encodes per key ({@link MediaContext}): while one caller encodes an item, a concurrent caller with
 * the same key waits for that encode and takes its result (or its failure) instead of encoding the item again, as
 * vLLM's encoder cache joins requests on the same media hash. A failure that belongs to the encoding caller rather
 * than to the item (its thread interrupted, its request cancelled) is not handed on: the waiters run the flight
 * again, one of them encoding. A key leaves the table when its encode ends, so a later caller encodes afresh (and
 * normally hits the cache the encode filled). Thread-safe.
 *
 * @param <K> the item key
 * @param <V> the encoding
 */
final class EncodeFlights<K, V> {

    /** The completion of a flight whose encoding caller gave up for reasons of its own; its waiters retry. */
    private static final class Abandoned extends RuntimeException {
        Abandoned(Throwable cause) {
            super("the encoding request gave up", cause, false, false);
        }
    }

    private final ConcurrentHashMap<K, CompletableFuture<V>> running = new ConcurrentHashMap<>();

    /**
     * The encoding of {@code key}: the running encode's when another caller is encoding it, else
     * {@code encode}'s, run on this thread. When the running encode's caller is interrupted or cancelled, this
     * caller runs the flight again instead of taking that failure.
     *
     * @throws RuntimeException the encode's own failure, rethrown to every caller that waited on it
     * @throws IllegalStateException when the caller is interrupted while it waits
     */
    V run(K key, Supplier<V> encode) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(encode, "encode");
        while (true) {
            CompletableFuture<V> mine = new CompletableFuture<>();
            CompletableFuture<V> other = running.putIfAbsent(key, mine);
            if (other == null) {
                return lead(key, mine, encode);
            }
            try {
                return await(other);
            } catch (Abandoned retry) {
                running.remove(key, other);
            }
        }
    }

    private V lead(K key, CompletableFuture<V> mine, Supplier<V> encode) {
        try {
            V out = encode.get();
            mine.complete(out);
            return out;
        } catch (RuntimeException | Error e) {
            mine.completeExceptionally(callersOwn(e) ? new Abandoned(e) : e);
            throw e;
        } finally {
            running.remove(key, mine);
        }
    }

    /**
     * Whether {@code failure} belongs to the encoding caller rather than to the item: its thread is interrupted, or
     * the failure comes from an interruption or a cancellation.
     */
    static boolean callersOwn(Throwable failure) {
        if (Thread.currentThread().isInterrupted()) {
            return true;
        }
        for (Throwable t = failure; t != null; t = t.getCause() == t ? null : t.getCause()) {
            if (t instanceof InterruptedException || t instanceof InterruptedIOException
                    || t instanceof ClosedByInterruptException || t instanceof CancellationException) {
                return true;
            }
        }
        return false;
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
            throw new Abandoned(e);
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
