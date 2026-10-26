package com.eonmux.neuralfree.gguf.engine.draft;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A map from n-grams (sizes 1..{@value #MAX_KEY}) to the empirical distribution of the token that follows them,
 * llama.cpp {@code common_ngram_cache} (common/ngram-cache.h). A part of the cache (one n-gram) counts how often
 * each token followed that n-gram, which drafters compare against sample-size and percentage thresholds
 * ({@code common/ngram-cache.cpp try_draft}).
 *
 * <p>An n-gram key of size {@code s} stores its {@code s} token ids and pads the missing slots to
 * {@value #MAX_KEY} tokens with {@value #NULL} (llama.cpp's zero-padded {@code common_ngram} struct, where the
 * padding is {@code LLAMA_TOKEN_NULL} == -1)), so two n-grams of different sizes never share a key. Keys and counts are
 * produced only by {@link #update} and {@link #merge}. The token sequences given to {@code update} may only be
 * appended to; changes in the middle need a fresh cache, as llama.cpp documents.
 */
public final class NgramCache {

    /** Padding token id of an n-gram key ({@code LLAMA_TOKEN_NULL}). */
    public static final int NULL = -1;

    /** Largest n-gram size stored ({@code LLAMA_NGRAM_MAX}); llama.cpp's cache files always store 4-token keys. */
    public static final int MAX_KEY = 4;

    /**
     * One cache key: up to {@value #MAX_KEY} token ids, padded with {@value #NULL} to the full width so keys of
     * different sizes stay distinct. Immutable and identity-free (equals/hashCode over the four token ids).
     */
    public static final class Key {

        private static final Key EMPTY = new Key(NULL, NULL, NULL, NULL);

        /** The key's first token. */
        public final int t0;
        /** The key's second token ({@link #NULL} when the key is shorter). */
        public final int t1;
        /** The key's third token ({@link #NULL} when the key is shorter). */
        public final int t2;
        /** The key's fourth token ({@link #NULL} when the key is shorter). */
        public final int t3;

        Key(int t0, int t1, int t2, int t3) {
            this.t0 = t0;
            this.t1 = t1;
            this.t2 = t2;
            this.t3 = t3;
        }

        /** The key's first token (record-style accessor). */
        public int t0() { return t0; }

        /** The key's second token (record-style accessor). */
        public int t1() { return t1; }

        /** The key's third token (record-style accessor). */
        public int t2() { return t2; }

        /** The key's fourth token (record-style accessor). */
        public int t3() { return t3; }

        /**
         * @throws IllegalArgumentException for a size outside {@code [0, 4]}
         */
        public static Key of(int[] tokens, int offset, int size) {
            if (offset < 0 || size < 0 || size > MAX_KEY || offset + size > tokens.length) {
                throw new IllegalArgumentException("n-gram window [" + offset + ", " + (offset + size)
                        + ") outside a key token array of " + tokens.length + " tokens");
            }
            int[] padded = new int[MAX_KEY];
            Arrays.fill(padded, NULL);
            System.arraycopy(tokens, offset, padded, 0, size);
            if (size == MAX_KEY) {
                return new Key(tokens[offset], tokens[offset + 1], tokens[offset + 2], tokens[offset + 3]);
            }
            return new Key(padded[0], padded[1], padded[2], padded[3]);
        }

        /** The key of the padding-free and token-free cache. */
        static Key empty() {
            return EMPTY;
        }

        /** Number of non-padding tokens of this key. */
        public int size() {
            int size = 0;
            for (int t : new int[]{t0, t1, t2, t3}) {
                if (t == NULL) {
                    break;
                }
                size++;
            }
            return size;
        }

        @Override
        public String toString() {
            int size = size();
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < size; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(token(i));
            }
            return sb.append(']').toString();
        }

        /** {@code i}-th token of the key, {@code i} in {@code [0, size)}. */
        public int token(int i) {
            if (i < 0 || i >= MAX_KEY) {
                throw new IndexOutOfBoundsException("key token " + i + " outside [0, " + MAX_KEY + ")");
            }
            if (i == 0) {
                return t0;
            } else if (i == 1) {
                return t1;
            } else if (i == 2) {
                return t2;
            } else {
                return t3;
            }
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof Key)) {
                return false;
            }
            Key that = (Key) other;
            return t0 == that.t0 && t1 == that.t1 && t2 == that.t2 && t3 == that.t3;
        }

        @Override
        public int hashCode() {
            int result = t0;
            result = 31 * result + t1;
            result = 31 * result + t2;
            result = 31 * result + t3;
            return result;
        }
    }

    private final Map<Key, HashMap<Integer, Integer>> parts = new HashMap<>();

    /** An empty cache. */
    public NgramCache() {
    }

    /** Whether no n-gram was added yet ({@code merge} of empties or nothing updated). */
    public boolean isEmpty() {
        return parts.isEmpty();
    }

    /** Number of stored n-grams. */
    public int ngrams() {
        return parts.size();
    }

    /**
     * Adds the n-grams each of the tokens {@code [last - nNew, last)} of a sequence adds, for sizes
     * {@code [ngramMin, ngramMax]}, llama.cpp {@code common_ngram_cache_update}. {@code tokens} holds the whole
     * sequence ({@code tokens.length >= last}); {@code nNew} counts the new tokens ({@code 0} appends nothing).
     * Sequences may only be appended to between calls: a middle change produces wrong parts.
     *
     * @throws IllegalArgumentException for an out-of-range n-gram size pair or an impossible sequence window
     */
    public void update(int ngramMin, int ngramMax, int[] tokens, int last, int nNew) {
        if (ngramMin < 1 || ngramMax > MAX_KEY || ngramMin > ngramMax) {
            throw new IllegalArgumentException("n-gram sizes must be in [1, " + MAX_KEY + "] with min <= max, got ["
                    + ngramMin + ", " + ngramMax + "]");
        }
        if (last < 0 || last > tokens.length) {
            throw new IllegalArgumentException("sequence window [0, " + last + ") outside a cache update array of "
                    + tokens.length + " tokens");
        }
        if (nNew < 0 || last - nNew < 0) {
            throw new IllegalArgumentException(nNew + " new tokens cannot fit before " + last);
        }
        for (int size = ngramMin; size <= ngramMax; size++) {
            int iStart = Math.max(last - nNew, size);
            for (int i = Math.max(iStart, size); i < last; i++) {
                addOne(tokens, i - size, size, tokens[i]);
            }
        }
    }

    /** Adds one whole sequence as a fresh cache input ({@code nNew = length}). */
    public void addSequence(int ngramMin, int ngramMax, int[] tokens) {
        update(ngramMin, ngramMax, tokens, tokens.length, tokens.length);
    }

    /**
     * Adds {@code count} occurrences of {@code token} following {@code key} (the file loader's entry point).
     *
     * @throws IllegalArgumentException for a non-positive count
     */
    public void addCount(Key key, int token, int count) {
        if (count <= 0) {
            throw new IllegalArgumentException("the count of token " + token + " after n-gram " + key + " must be"
                    + " positive, got " + count);
        }
        parts.computeIfAbsent(key, k -> new HashMap<>()).merge(token, count, Integer::sum);
    }

    private void addOne(int[] tokens, int keyStart, int size, int token) {
        Key key = Key.of(tokens, keyStart, size);
        HashMap<Integer, Integer> part = parts.computeIfAbsent(key, k -> new HashMap<>());
        part.merge(token, 1, Integer::sum);
    }

    /**
     * Adds every part and count of {@code other}, llama.cpp {@code common_ngram_cache_merge}.
     *
     * @throws IllegalArgumentException when {@code other} is null
     */
    public void merge(NgramCache other) {
        Objects.requireNonNull(other, "merged cache");
        for (Map.Entry<Key, HashMap<Integer, Integer>> e : other.parts.entrySet()) {
            HashMap<Integer, Integer> part = parts.computeIfAbsent(e.getKey(), k -> new HashMap<>());
            for (Map.Entry<Integer, Integer> c : e.getValue().entrySet()) {
                part.merge(c.getKey(), c.getValue(), Integer::sum);
            }
        }
    }

    /** Whether {@code key} has a part in this cache. */
    public boolean hasPart(Key key) {
        return parts.containsKey(key);
    }

    /** How often {@code token} followed {@code key}, 0 when unknown. */
    public int count(Key key, int token) {
        HashMap<Integer, Integer> part = parts.get(key);
        return part == null ? 0 : part.getOrDefault(token, 0);
    }

    /** Sum of the counts of every token that followed {@code key} (0 when unknown). */
    public long partTotal(Key key) {
        HashMap<Integer, Integer> part = parts.get(key);
        if (part == null) {
            return 0L;
        }
        long total = 0;
        for (int c : part.values()) {
            total += c;
        }
        return total;
    }

    /**
     * All parts: an unmodifiable map from key to an unmodifiable map token id to count. A serializing caller
     * receives a stable iteration order.
     */
    public Map<Key, Map<Integer, Integer>> entries() {
        Map<Key, Map<Integer, Integer>> out = new LinkedHashMap<>();
        for (Map.Entry<Key, HashMap<Integer, Integer>> e : parts.entrySet()) {
            out.put(e.getKey(), Collections.unmodifiableMap(new LinkedHashMap<>(e.getValue())));
        }
        return out;
    }

    /**
     * The token ids {@code key}'s part counts; empty when the part is unknown.
     */
    public List<Integer> tokensOf(Key key) {
        HashMap<Integer, Integer> part = parts.get(key);
        return part == null ? List.of() : List.copyOf(part.keySet());
    }

    /** The keys of {@link #entries()} (unmodifiable). */
    public List<Key> keys() {
        return List.copyOf(parts.keySet());
    }

    @Override
    public String toString() {
        return "NgramCache{" + parts.size() + " n-grams}";
    }
}
