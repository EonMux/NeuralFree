package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The Responses API store (vLLM's {@code response_store} and {@code msg_store}): stored responses by id, each with
 * its input items, the conversation a later {@code previous_response_id} continues, and for a background response
 * its event log and engine batch. At most {@code capacity} entries stay, least recently used first out; a background
 * response still running is never evicted. Thread-safe.
 */
final class ResponseStore {

    /** One stored response. Thread-safe: the producer appends, readers wait on it. */
    static final class Entry {
        private final String id;
        private final List<Object> inputItems;
        private final List<Map<String, Object>> events = new ArrayList<>();
        private final GenerationBatch batch;
        private Map<String, Object> response;
        private List<Map<String, Object>> conversation;
        private boolean finished;

        Entry(String id, List<Object> inputItems, Map<String, Object> response, GenerationBatch batch) {
            this.id = id;
            this.inputItems = List.copyOf(inputItems);
            this.response = response;
            this.batch = batch;
        }

        String id() {
            return id;
        }

        /** The request's input items. */
        List<Object> inputItems() {
            return inputItems;
        }

        /** The latest state of the response. */
        synchronized Map<String, Object> response() {
            return response;
        }

        /** The conversation through this response's output, or null while it runs. */
        synchronized List<Map<String, Object>> conversation() {
            return conversation;
        }

        /** Whether the response is still being produced. */
        synchronized boolean running() {
            return !finished;
        }

        /** Whether a background generation produces it ({@link #cancel} applies). */
        boolean background() {
            return batch != null;
        }

        /** Records one event of a background response; a response-carrying event updates its state. */
        synchronized void event(String type, Map<String, Object> data) {
            events.add(data);
            if (data.get("response") instanceof Map<?, ?> r) {
                @SuppressWarnings("unchecked")
                Map<String, Object> latest = (Map<String, Object>) r;
                response = latest;
            }
            notifyAll();
        }

        /** Ends the response with its final state and the conversation through its output. */
        synchronized void finish(Map<String, Object> finalResponse, List<Map<String, Object>> through) {
            response = finalResponse;
            conversation = List.copyOf(through);
            finished = true;
            notifyAll();
        }

        /**
         * Cancels a running background response; its generation stops and the response ends {@code cancelled}.
         *
         * @return whether it was running
         */
        boolean cancel() {
            synchronized (this) {
                if (finished || batch == null) {
                    return false;
                }
            }
            batch.cancel();
            return true;
        }

        /**
         * The events whose sequence number is above {@code after}, waiting at most {@code millis} for one; empty
         * when none came, and empty at once when the response finished and every event was read.
         */
        synchronized List<Map<String, Object>> eventsAfter(int after, long millis) throws InterruptedException {
            long deadline = System.currentTimeMillis() + millis;
            long next = (long) after + 1;
            while (events.size() <= next && !finished) {
                long left = deadline - System.currentTimeMillis();
                if (left <= 0) {
                    return List.of();
                }
                wait(left);
            }
            if (next >= events.size()) {
                return List.of();
            }
            int from = (int) Math.max(0, next);
            return List.copyOf(events.subList(from, events.size()));
        }

        /** Waits at most {@code millis} for the response to finish; whether it did. */
        synchronized boolean awaitFinished(long millis) throws InterruptedException {
            long deadline = System.currentTimeMillis() + millis;
            while (!finished) {
                long left = deadline - System.currentTimeMillis();
                if (left <= 0) {
                    return false;
                }
                wait(left);
            }
            return true;
        }

        /** Whether every event up to {@code after} was read and no more will come. */
        synchronized boolean drained(int after) {
            return finished && (long) after + 1 >= events.size();
        }
    }

    private final int capacity;
    private final Map<String, Entry> entries = new LinkedHashMap<>(16, 0.75f, true);

    /** A store of at most {@code capacity} responses; 0 keeps none. */
    ResponseStore(int capacity) {
        if (capacity < 0) {
            throw new IllegalArgumentException("capacity must be 0 or more, got " + capacity);
        }
        this.capacity = capacity;
    }

    /** Whether the store keeps responses at all. */
    boolean enabled() {
        return capacity > 0;
    }

    /**
     * Adds {@code entry}, evicting the least recently used finished entries beyond the capacity; the new entry stays
     * even when running background responses hold the other places.
     */
    synchronized void put(Entry entry) {
        if (!enabled()) {
            throw new IllegalStateException("the response store is disabled");
        }
        entries.put(entry.id(), entry);
        Iterator<Entry> it = entries.values().iterator();
        while (entries.size() > capacity && it.hasNext()) {
            Entry e = it.next();
            if (e != entry && !e.running()) {
                it.remove();
            }
        }
    }

    /** The entry of {@code id}, or null. */
    synchronized Entry get(String id) {
        return entries.get(id);
    }

    /** Removes the entry of {@code id}; whether there was one. */
    synchronized boolean remove(String id) {
        return entries.remove(id) != null;
    }

    /** The stored ids, least recently used first (for tests). */
    synchronized List<String> ids() {
        return Collections.unmodifiableList(new ArrayList<>(entries.keySet()));
    }
}
