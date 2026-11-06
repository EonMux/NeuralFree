package com.eonmux.neuralfree.server;

import jakarta.inject.Singleton;

import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * The gRPC requests in flight, by the client's {@code request_id}: the {@code Control.Abort} call cancels the named
 * ones, and a duplicate id is refused while its request runs so an abort can never hit the wrong generation.
 * Entries are removed when their request reaches its terminal state, whatever the outcome. Thread-safe.
 */
@Singleton
public final class GrpcInFlight {

    private final ConcurrentMap<String, Runnable> requests = new ConcurrentHashMap<>();

    /**
     * Registers {@code id} with {@code cancel}.
     *
     * @throws IllegalArgumentException when a request with that id is still running
     */
    void register(String id, Runnable cancel) {
        if (!id.isEmpty() && requests.putIfAbsent(id, cancel) != null) {
            throw new IllegalArgumentException("request_id '" + id + "' is already in flight");
        }
    }

    /** Removes {@code id}; an id that is not running is ignored. */
    void remove(String id) {
        if (!id.isEmpty()) {
            requests.remove(id);
        }
    }

    /**
     * Cancels the in-flight requests named in {@code ids}; ids that are not running are ignored, so a repeated
     * abort is harmless. Returns how many running requests it cancelled.
     */
    int cancel(Collection<String> ids) {
        int cancelled = 0;
        for (String id : ids) {
            Runnable cancel = requests.remove(id);
            if (cancel != null) {
                cancel.run();
                cancelled++;
            }
        }
        return cancelled;
    }
}
