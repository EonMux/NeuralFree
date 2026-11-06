package com.eonmux.neuralfree.gguf.mm;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.ToLongFunction;

/**
 * A least-recently-used cache of media, bounded by a byte budget: vLLM's multimodal processor cache
 * ({@code --mm-processor-cache-gb}, {@code mm_processor_cache_type=lru}), one per layer of a {@link MediaContext}'s
 * media cache (the encoder outputs, the processed media). A key names a media item (a client's media {@code uuid}
 * or a content digest); a value's size is what the sizer reports. An entry larger than the whole budget is not
 * stored (vLLM serves such an item uncached), and a budget of 0 stores nothing. Thread-safe.
 *
 * @param <K> the key type
 * @param <V> the cached value type
 */
final class EncoderOutputCache<K, V> {

    private final long maxBytes;
    private final ToLongFunction<V> sizer;
    private final Map<K, Entry<V>> entries = new LinkedHashMap<>(16, 0.75f, true);
    private long bytes;
    private long hits;
    private long queries;

    /**
     * A cache of at most {@code maxBytes}.
     *
     * @throws IllegalArgumentException when the budget is negative
     */
    EncoderOutputCache(long maxBytes, ToLongFunction<V> sizer) {
        if (maxBytes < 0) {
            throw new IllegalArgumentException("the media cache budget must not be negative, got " + maxBytes);
        }
        this.maxBytes = maxBytes;
        this.sizer = Objects.requireNonNull(sizer, "sizer");
    }

    /** The value under {@code key}, now the most recently used, or null; counts one query (and a hit). */
    synchronized V get(K key) {
        queries++;
        Entry<V> e = entries.get(key);
        if (e == null) {
            return null;
        }
        hits++;
        return e.value;
    }

    /**
     * The value under {@code key}, now the most recently used, or null; counts a hit but no query. The re-check
     * inside a serialized encode uses it, so the counters keep one lookup per appended item.
     */
    synchronized V peek(K key) {
        Entry<V> e = entries.get(key);
        if (e == null) {
            return null;
        }
        hits++;
        return e.value;
    }

    /** Whether {@code key} is cached; neither counts a query nor refreshes the entry. */
    synchronized boolean contains(K key) {
        return entries.containsKey(key);
    }

    /**
     * Stores {@code value} under {@code key} as the most recently used entry, evicting the least recently used
     * ones until the cache fits its budget; a value larger than the budget is not stored.
     */
    synchronized void put(K key, V value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        long size = sizer.applyAsLong(value);
        if (size > maxBytes) {
            return;
        }
        Entry<V> old = entries.put(key, new Entry<>(value, size));
        if (old != null) {
            bytes -= old.size;
        }
        bytes += size;
        Iterator<Map.Entry<K, Entry<V>>> it = entries.entrySet().iterator();
        while (bytes > maxBytes && it.hasNext()) {
            Map.Entry<K, Entry<V>> eldest = it.next();
            if (eldest.getKey().equals(key)) {
                continue;
            }
            bytes -= eldest.getValue().size;
            it.remove();
        }
    }

    /** Drops every entry (vLLM's {@code reset_mm_cache}); the hit counters stay. */
    synchronized void clear() {
        entries.clear();
        bytes = 0;
    }

    /** The bytes the cached values hold. */
    synchronized long bytes() {
        return bytes;
    }

    /** The number of cached values. */
    synchronized int size() {
        return entries.size();
    }

    /** The lookups that found their key, since the cache was built. */
    synchronized long hits() {
        return hits;
    }

    /** The lookups made, since the cache was built. */
    synchronized long queries() {
        return queries;
    }

    /** The byte budget. */
    long maxBytes() {
        return maxBytes;
    }

    /** One cached value and its size. */
    private static final class Entry<V> {
        final V value;
        final long size;

        Entry(V value, long size) {
            this.value = value;
            this.size = size;
        }
    }
}
