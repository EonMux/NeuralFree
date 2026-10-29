package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The longest-prefix cache over the paged memory (PARITY plan P3, SGLang's RadixAttention in the shape the
 * substrate gives): a retired sequence stays in the memory with its pages, the cache indexes those sequences by
 * their exact token arrays, and an admitted request whose prompt extends a cached prefix copies the shared pages
 * ({@code PagedSequences.copy} — whole pages share, one leading partial page privatizes) and prefills only its
 * tail. The cache holds no pages itself: every entry is a real sequence occupying one slot of the memory, so the
 * entry count is bounded by the slot count and eviction frees a slot and its pages at once.
 *
 * <p>Entries match exactly by tokens (a hash chain or tree adds nothing under that bound); ties resolve to the
 * most recently used entry. Not thread-safe: the engine thread owns it.
 */
public final class PrefixCache {

    /** One cached sequence: its slot id, its exact token array and its last-use stamp. */
    static final class Entry {
        final int seq;
        final int[] tokens;
        long lastUse;

        Entry(int seq, int[] tokens, long lastUse) {
            this.seq = seq;
            this.tokens = tokens;
            this.lastUse = lastUse;
        }
    }

    private final List<Entry> entries = new ArrayList<>();
    private long clock;

    /** The entry whose tokens form the longest prefix of {@code prompt} (over one token), or null. */
    public Entry longestPrefix(int[] prompt) {
        Objects.requireNonNull(prompt, "prompt");
        Entry best = null;
        int bestLen = 0;
        for (Entry entry : entries) {
            int len = commonLength(entry.tokens, prompt);
            if (len > bestLen || (len == bestLen && best != null && entry.lastUse > best.lastUse)) {
                best = entry;
                bestLen = len;
            }
        }
        return best;
    }

    /** The longest common prefix length of the two arrays. */
    static int commonLength(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                return i;
            }
        }
        return n;
    }

    /** Caches a retired sequence with its exact tokens; the caller owns the slot budget and eviction. */
    public void offer(int seq, int[] tokens) {
        discard(seq);
        entries.add(new Entry(seq, tokens.clone(), ++clock));
    }

    /** Drops the entry for {@code seq} (the caller then frees or reuses the sequence itself). */
    public boolean discard(int seq) {
        for (int i = 0; i < entries.size(); i++) {
            if (entries.get(i).seq == seq) {
                entries.remove(i);
                return true;
            }
        }
        return false;
    }

    /** Whether a sequence id is cached. */
    public boolean holds(int seq) {
        for (Entry entry : entries) {
            if (entry.seq == seq) {
                return true;
            }
        }
        return false;
    }

    /** Drops and returns the least recently used entry, or null when the cache is empty. */
    public Entry evictOldest() {
        int oldest = -1;
        for (int i = 0; i < entries.size(); i++) {
            if (oldest < 0 || entries.get(i).lastUse < entries.get(oldest).lastUse) {
                oldest = i;
            }
        }
        return oldest < 0 ? null : entries.remove(oldest);
    }

    /** Cached sequences. */
    public int size() {
        return entries.size();
    }

    /** The cached tokens of {@code seq}, or null. */
    public int[] tokensOf(int seq) {
        for (Entry entry : entries) {
            if (entry.seq == seq) {
                return entry.tokens;
            }
        }
        return null;
    }

    /** The entry's tokens, for assertions. */
    static int[] tokens(Entry entry) {
        return entry.tokens;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("PrefixCache[");
        for (Entry entry : entries) {
            sb.append("seq ").append(entry.seq).append(": ").append(Arrays.toString(
                    Arrays.copyOf(entry.tokens, Math.min(entry.tokens.length, 8))));
            if (entry.tokens.length > 8) {
                sb.append("..");
            }
            sb.append(", ");
        }
        return sb.append(']').toString();
    }
}
