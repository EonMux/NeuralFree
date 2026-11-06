package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The encoder output cache of a {@link BatchEngine} (vLLM V1's {@code EncoderCacheManager}): the embedding rows
 * of identified media chunks ({@link com.eonmux.neuralfree.gguf.engine.MediaChunk#id()}, the projector's content
 * digest, vLLM's {@code mm_hash}) stored once and shared by every submission that carries the same identity, with
 * the reference each live submission holds and an eviction of the least recently used unreferenced entries when a
 * new one needs the space. The budget is rows of encoder output ({@code neuralfree.encoder.cache.rows}, vLLM's
 * {@code encoder_cache_size}); an entry whose last reference retires stays cached until its space is claimed —
 * a prompt with the same medium then hits instead of holding a second copy. Rows a chunk larger than the whole
 * budget, and every row while the cache is off, pass through unshared: the chunk's own arrays stay its prompt's.
 * Confined to the engine's thread.
 */
final class EncoderCache {

    /** One cached encoder output: the canonical rows and the live submissions referencing them. */
    private static final class Entry {
        final float[][] rows;
        final int nRows;
        int refs;

        Entry(float[][] rows) {
            this.rows = rows;
            this.nRows = rows.length;
        }
    }

    /** Rows of encoder output the cache holds at most; at most 0 turns the cache off. */
    private final int budgetRows;
    /** Cached entries in access order (the eviction candidates are the least recently touched unreferenced). */
    private final LinkedHashMap<String, Entry> entries = new LinkedHashMap<>(16, 0.75f, true);
    /** References live submissions still hold to entries a {@link #clear} dropped, per identity. */
    private final Map<String, Integer> clearedRefs = new HashMap<>();
    private int usedRows;
    private long hits;
    private long evictions;

    /**
     * A cache of {@code budgetRows} rows; at most 0 runs disabled and every intern passes through.
     */
    EncoderCache(int budgetRows) {
        this.budgetRows = budgetRows;
    }

    /** Whether the cache stores anything ({@code neuralfree.encoder.cache.rows} above 0). */
    boolean enabled() {
        return budgetRows > 0;
    }

    /**
     * The canonical rows the cache holds for {@code id}, storing {@code rows} as them on a miss (evicting the
     * least recently used unreferenced entries for the space, never a referenced one): the returned arrays are
     * what every chunk of that identity holds from then on. The caller's rows come back unshared, and nothing
     * evicts, when the cache is off, {@code rows} alone exceed the whole budget, the referenced entries leave too
     * little room, or the entry cached under {@code id} has another shape than {@code rows}.
     */
    float[][] intern(String id, float[][] rows) {
        if (id == null) {
            return rows;
        }
        Entry hit = entries.get(id);
        if (hit != null) {
            if (!sameShape(hit.rows, rows)) {
                return rows;
            }
            hits++;
            return hit.rows;
        }
        if (!enabled() || rows.length > budgetRows) {
            return rows;
        }
        if (!makeRoom(rows.length)) {
            return rows;
        }
        entries.put(id, new Entry(rows));
        usedRows += rows.length;
        return rows;
    }

    /**
     * {@link #intern} for a submission whose references are {@code held}: when the returned rows are the cache's
     * entry for {@code id} and the submission does not reference it yet, the submission takes one reference
     * (vLLM's reference add) and {@code id} joins {@code held}. A referenced entry never evicts, so a prompt's
     * later chunks cannot evict its earlier ones; {@code held} is what the submission {@link #release}s when it
     * retires.
     */
    float[][] reference(String id, float[][] rows, Set<String> held) {
        float[][] canonical = intern(id, rows);
        Entry e = id == null ? null : entries.get(id);
        if (e != null && e.rows == canonical && held.add(id)) {
            e.refs++;
        }
        return canonical;
    }

    /**
     * Drops one submission's reference to each id it {@link #reference}d; an entry keeps its rows until eviction
     * claims them.
     */
    void release(Collection<String> ids) {
        for (String id : ids) {
            Integer orphans = clearedRefs.get(id);
            if (orphans != null) {
                if (orphans == 1) {
                    clearedRefs.remove(id);
                } else {
                    clearedRefs.put(id, orphans - 1);
                }
                continue;
            }
            Entry e = entries.get(id);
            if (e != null && e.refs > 0) {
                e.refs--;
            }
        }
    }

    /**
     * Drops every cached entry (vLLM's {@code reset_encoder_cache}): the next intern of any identity stores its
     * own rows again. Live submissions keep the rows their prompts already hold; the references they took count
     * against the cleared entries, so their later {@link #release} never frees a reference taken after the reset.
     * The hit and eviction counters stay.
     */
    void clear() {
        for (Map.Entry<String, Entry> it : entries.entrySet()) {
            int refs = it.getValue().refs;
            if (refs > 0) {
                clearedRefs.merge(it.getKey(), refs, Integer::sum);
            }
        }
        entries.clear();
        usedRows = 0;
    }

    /**
     * Evicts least recently used unreferenced entries until {@code rows} fit the free budget; false, evicting
     * nothing, when even every unreferenced entry's rows would not make the room.
     */
    private boolean makeRoom(int rows) {
        if (usedRows + rows <= budgetRows) {
            return true;
        }
        int reclaimable = 0;
        for (Entry e : entries.values()) {
            if (e.refs == 0) {
                reclaimable += e.nRows;
            }
        }
        if (usedRows - reclaimable + rows > budgetRows) {
            return false;
        }
        List<String> spent = null;
        for (Map.Entry<String, Entry> it : entries.entrySet()) {
            if (usedRows + rows <= budgetRows) {
                break;
            }
            Entry e = it.getValue();
            if (e.refs != 0) {
                continue;
            }
            usedRows -= e.nRows;
            evictions++;
            if (spent == null) {
                spent = new ArrayList<>();
            }
            spent.add(it.getKey());
        }
        if (spent != null) {
            for (String id : spent) {
                entries.remove(id);
            }
        }
        return true;
    }

    /** Whether two non-empty row sets hold the same row count and width, so one can stand in for the other. */
    private static boolean sameShape(float[][] a, float[][] b) {
        return a.length == b.length && a[0].length == b[0].length;
    }

    /** Rows of encoder output the cache holds at the moment. */
    int usedRows() {
        return usedRows;
    }

    /** Entries evicted since the engine opened (vLLM's freed {@code mm_hash} count). */
    long evictions() {
        return evictions;
    }

    /** Interns that found their identity's encoder output cached instead of storing their own rows. */
    long hits() {
        return hits;
    }

    @Override
    public String toString() {
        return "EncoderCache{budget=" + budgetRows + ", used=" + usedRows + ", entries=" + entries.size()
                + ", hits=" + hits + ", evictions=" + evictions + "}";
    }
}
