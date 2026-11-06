package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublisher;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * A {@link KvEventPublisher} the engine instantiates by class name ({@code neuralfree.kv.events}) and that records
 * every batch it receives in {@link #BATCHES}, shared by every instance, for the server tests to read.
 */
public final class RecordingKvEventPublisher implements KvEventPublisher {

    /** Every batch any instance received, in publication order. */
    static final List<KvEventBatch> BATCHES = new CopyOnWriteArrayList<>();

    @Override
    public void publish(KvEventBatch batch) {
        BATCHES.add(batch);
    }
}
