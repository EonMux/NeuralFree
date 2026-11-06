package com.eonmux.neuralfree.gguf.engine.kvevents;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The KV cache events of one engine step, in emission order (vLLM's {@code KVEventBatch}): the wall-clock time the
 * batch was published in seconds, the events, and the data-parallel rank of the engine replica that emitted them
 * (null for an engine outside a replica set). Immutable.
 */
public final class KvEventBatch {

    private final double ts;
    private final List<KvCacheEvent> events;
    private final Integer dataParallelRank;

    /** @throws IllegalArgumentException when the batch holds no event or a negative rank */
    public KvEventBatch(double ts, List<KvCacheEvent> events, Integer dataParallelRank) {
        Objects.requireNonNull(events, "events");
        if (events.isEmpty()) {
            throw new IllegalArgumentException("an event batch holds at least one event");
        }
        if (dataParallelRank != null && dataParallelRank < 0) {
            throw new IllegalArgumentException("data-parallel rank must be >= 0, got " + dataParallelRank);
        }
        for (KvCacheEvent e : events) {
            Objects.requireNonNull(e, "event");
        }
        this.ts = ts;
        this.events = Collections.unmodifiableList(new ArrayList<>(events));
        this.dataParallelRank = dataParallelRank;
    }

    /** Publication time, seconds since the epoch. */
    public double ts() { return ts; }

    /** The step's events in emission order. */
    public List<KvCacheEvent> events() { return events; }

    /** The emitting replica's data-parallel rank, or null. */
    public Integer dataParallelRank() { return dataParallelRank; }

    /** This batch stamped with {@code rank}. */
    public KvEventBatch withRank(int rank) {
        return new KvEventBatch(ts, events, rank);
    }

    @Override
    public String toString() {
        return "KvEventBatch{ts=" + ts + (dataParallelRank == null ? "" : ", data_parallel_rank=" + dataParallelRank)
                + ", events=" + events + "}";
    }
}
