package com.eonmux.neuralfree.gguf.engine.kvevents;

/**
 * Receives the batch engine's KV cache event batches (vLLM's {@code EventPublisher}), one per engine step that
 * changed the prefix cache, on the engine's thread. An implementation named by {@code neuralfree.kv.events} needs
 * a public no-argument constructor. A publisher that throws loses that batch only: the engine logs the failure and
 * keeps running. The engine closes its publisher when it closes.
 */
public interface KvEventPublisher extends AutoCloseable {

    /** Delivers one step's events. */
    void publish(KvEventBatch batch);

    /** Whether the engine should collect events at all; only the null publisher answers false. */
    default boolean enabled() {
        return true;
    }

    /** Releases the publisher's resources; the default holds none. */
    @Override
    default void close() {
    }
}
