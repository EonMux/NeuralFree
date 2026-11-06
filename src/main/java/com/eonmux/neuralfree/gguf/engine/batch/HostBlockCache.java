package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The host-RAM tier of the prefix cache (llama.cpp's {@code --cache-ram} prompt cache, vLLM's CPU KV offloading): the
 * blocks the device pool evicted for room, kept as host copies ({@link BlockImage}) under a byte budget, least
 * recently used first out. Each entry records its chain hash and its parent's (the block hasher's digests, a
 * chain's first block's parent the hasher's {@code NONE_HASH} seed), the chain's extra keys and every token from
 * the chain's root through its own rows, so a lookup verifies the whole prefix a request walked and a hash
 * collision can only miss, never serve foreign rows. {@link PrefixCache#promote} brings matching entries back
 * into fresh pages before a request adopts them. Not thread-safe: the engine thread owns it.
 */
final class HostBlockCache {

    /** One host copy of an evicted block. Immutable. */
    static final class Entry {
        final byte[] hash;
        final byte[] parentHash;
        /** The value-equality key {@code parentHash} sits under in the children map. */
        private final DigestKey parentKey;
        final PrefixKey key;
        /** The tokens from the chain's root through the block's own rows. */
        private final int[] chain;
        final BlockImage image;

        Entry(byte[] hash, byte[] parentHash, PrefixKey key, int[] chain, BlockImage image) {
            this.hash = hash.clone();
            this.parentHash = parentHash.clone();
            this.parentKey = new DigestKey(this.parentHash);
            this.key = Objects.requireNonNull(key, "key");
            this.chain = chain.clone();
            this.image = Objects.requireNonNull(image, "image");
            if (chain.length < image.rows()) {
                throw new IllegalArgumentException("a chain of " + chain.length + " tokens cannot end in a block of "
                        + image.rows());
            }
        }

        /** The row the block starts at in its chain. */
        int start() {
            return chain.length - image.rows();
        }

        /** The tokens from the chain's root through the block's own rows. */
        int[] chain() {
            return chain.clone();
        }

        /** Host bytes the entry holds. */
        long bytes() {
            return image.bytes() + 4L * chain.length;
        }

        /**
         * The rows of this block that continue {@code want} at {@code start()} when {@code want} holds this chain's
         * prefix there: 0 when the prefix differs, else the common length of the block and {@code want[start(),
         * limit)}.
         */
        int continues(int[] want, int limit) {
            int s = start();
            if (s > limit) {
                return 0;
            }
            for (int i = 0; i < s; i++) {
                if (chain[i] != want[i]) {
                    return 0;
                }
            }
            int n = Math.min(image.rows(), limit - s);
            for (int i = 0; i < n; i++) {
                if (chain[s + i] != want[s + i]) {
                    return i;
                }
            }
            return n;
        }

        /** Whether this entry holds the same chain of rows as {@code other}. */
        boolean sameChain(Entry other) {
            return Arrays.equals(hash, other.hash) && key.equals(other.key) && Arrays.equals(chain, other.chain);
        }
    }

    /** A digest as a map key: value equality over the bytes ({@code byte[]} itself compares by identity). */
    static final class DigestKey {
        private final byte[] bytes;

        DigestKey(byte[] bytes) {
            this.bytes = bytes;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof DigestKey && Arrays.equals(bytes, ((DigestKey) o).bytes);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(bytes);
        }
    }

    /** A lookup's answer: the entry and how many of its rows continue the request. */
    static final class Hit {
        final Entry entry;
        final int rows;

        Hit(Entry entry, int rows) {
            this.entry = entry;
            this.rows = rows;
        }
    }

    private final long budget;
    private final Map<DigestKey, List<Entry>> children = new HashMap<>();
    private final LinkedHashSet<Entry> lru = new LinkedHashSet<>();
    private long bytes;

    /**
     * A tier of at most {@code budgetBytes} host bytes; a negative budget leaves it unbounded.
     *
     * @throws IllegalArgumentException for a budget of 0 (no tier)
     */
    HostBlockCache(long budgetBytes) {
        if (budgetBytes == 0) {
            throw new IllegalArgumentException("a host-RAM prompt cache needs a non-zero budget (negative for none)");
        }
        this.budget = budgetBytes;
    }

    /** Entries held. */
    int size() {
        return lru.size();
    }

    /** Host bytes the entries hold. */
    long bytes() {
        return bytes;
    }

    /**
     * Keeps {@code entry}, replacing an entry of the same chain, then drops least recently used entries while the
     * tier holds more than its budget. An entry larger than the whole budget is not kept.
     *
     * @return the entries dropped for room (the replaced one excluded), oldest first; {@code entry} itself when it
     *         was not kept
     */
    List<Entry> put(Entry entry) {
        if (budget > 0 && entry.bytes() > budget) {
            return Collections.singletonList(entry);
        }
        List<Entry> siblings = children.computeIfAbsent(entry.parentKey, h -> new ArrayList<>(2));
        for (Entry e : new ArrayList<>(siblings)) {
            if (e.sameChain(entry)) {
                remove(e);
            }
        }
        children.computeIfAbsent(entry.parentKey, h -> new ArrayList<>(2)).add(entry);
        lru.add(entry);
        bytes += entry.bytes();
        List<Entry> dropped = new ArrayList<>();
        while (budget > 0 && bytes > budget) {
            Entry oldest = lru.iterator().next();
            remove(oldest);
            dropped.add(oldest);
        }
        return dropped;
    }

    /**
     * The whole block after the block hashed {@code parentHash} under {@code key} whose chain holds
     * {@code want[0, start + pageSize)}, starting at {@code start}; null when none does.
     */
    Entry whole(byte[] parentHash, PrefixKey key, int[] want, int start, int pageSize) {
        for (Entry e : children.getOrDefault(new DigestKey(parentHash), Collections.emptyList())) {
            if (e.image.full() && e.start() == start && e.key.equals(key)
                    && e.continues(want, start + pageSize) == pageSize) {
                return e;
            }
        }
        return null;
    }

    /**
     * The block after the block hashed {@code parentHash} under {@code key} sharing the longest row run with
     * {@code want[start, limit)} behind the same chain prefix, the latest kept winning ties; null when none shares
     * a row.
     */
    Hit best(byte[] parentHash, PrefixKey key, int[] want, int start, int limit) {
        Hit best = null;
        for (Entry e : children.getOrDefault(new DigestKey(parentHash), Collections.emptyList())) {
            if (e.start() != start || !e.key.equals(key)) {
                continue;
            }
            int n = e.continues(want, limit);
            if (n > 0 && (best == null || n >= best.rows)) {
                best = new Hit(e, n);
            }
        }
        return best;
    }

    /**
     * The entries under {@code key} whose whole chain is a prefix of {@code want}: the host blocks a history of those
     * tokens left.
     */
    List<Entry> within(int[] want, PrefixKey key) {
        List<Entry> out = new ArrayList<>();
        for (Entry e : lru) {
            if (e.key.equals(key) && e.chain.length <= want.length && e.continues(want, e.chain.length)
                    == e.image.rows()) {
                out.add(e);
            }
        }
        return out;
    }

    /** Marks {@code entry} the most recently used. */
    void touch(Entry entry) {
        if (lru.remove(entry)) {
            lru.add(entry);
        }
    }

    /** Drops {@code entry} (it moved back to the device, or the tier clears). */
    void remove(Entry entry) {
        if (!lru.remove(entry)) {
            return;
        }
        bytes -= entry.bytes();
        List<Entry> siblings = children.get(entry.parentKey);
        siblings.remove(entry);
        if (siblings.isEmpty()) {
            children.remove(entry.parentKey);
        }
    }

    /** Drops every entry, answering them oldest first. */
    List<Entry> clear() {
        List<Entry> all = new ArrayList<>(lru);
        lru.clear();
        children.clear();
        bytes = 0;
        return all;
    }

    @Override
    public String toString() {
        return "HostBlockCache[" + lru.size() + " blocks, " + bytes + " bytes of "
                + (budget < 0 ? "unbounded" : budget + "") + "]";
    }
}
