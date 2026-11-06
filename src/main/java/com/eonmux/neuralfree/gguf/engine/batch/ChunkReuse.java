package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * The KV chunk reuse of a {@link BatchEngine} admission (llama.cpp server's {@code n_cache_reuse},
 * tools/server/server-context.cpp's cached-prompt loop): past the common prefix, the prompt's runs of at least
 * {@code n_cache_reuse} tokens that the block tree still holds skip their prefill — the memory re-appends the
 * cached rows at the prompt's own positions ({@link ForwardBackend#reuseChunk}, the re-append llama.cpp's
 * {@code seq_add} performs over the match's tail), each chunk's keys re-rotating by the uniform shift between the
 * cached positions and the prompt ones, so the rows sit at the positions their tokens take in the prompt. The
 * re-appended rows keep the deeper-layer state their donor prefix produced — llama.cpp's opt-in trade of accuracy
 * for the prefill it saves, exact only where the donor prefix is the prompt's own.
 * The scan walks the cached chain the way llama.cpp walks its slot's cached tokens: anchored at the prompt row the
 * last reuse ended on, aligned runs of cached rows against that one prompt row crossing whole blocks, a run of at
 * least the reuse length becoming a chunk ({@code kv_shift} = prompt row minus cached row, never positive) and a
 * shorter run advancing the cached row alone. The chunks tile the prompt rows back to back from the match's
 * boundary (llama.cpp's {@code head_p} advances only with a chunk, and its {@code seq_rm} drops the stale cached
 * rows between two chunks — rows this memory never applied, there being nothing to remove), so the stored rows
 * are the prompt's rows {@code [0, promptEnd)} in order and the prompt view keeps composing. The cached side is
 * the block tree's most recently used chain from the boundary, continued into the child whose first row matches
 * the anchored prompt row when one does, else into the most recently used child whose rows the walk steps past one
 * by one (the cache cursor llama.cpp advances alone); every chunk's rows are token-exact against the prompt before
 * it applies, so any branch serves. Chunks never cover the prompt's last row (the row whose forward produces the
 * first output token). A memory short of the pages a re-append takes degrades the admission to prefix-only reuse:
 * the applied rows roll back and the prompt prefills past the common prefix.
 */
final class ChunkReuse {

    private final BatchState state;
    private final ForwardBackend backend;
    /** The id an embedding row records in a sequence's token record ({@code -1} for none). */
    private final int embeddingToken;

    ChunkReuse(BatchState state, ForwardBackend backend, int embeddingToken) {
        this.state = state;
        this.backend = backend;
        this.embeddingToken = embeddingToken;
    }

    /** Whether the backend can re-append cached rows: it edits paged sequences whose keys it can re-rotate. */
    boolean supported() {
        return backend.sequenceEditRefusal() == null;
    }

    /**
     * One chunk: the contiguous run of cached page rows it re-appends, the prompt row the run's first row takes,
     * the chain row the run's first cached row holds, and its row count. The run crosses whole blocks in chain
     * order (its page rows are contiguous, {@code srcRow0} opening the first page), and a block whose page rows
     * end short closes it (a partial block's rows have no continuation to chain on).
     */
    static final class Chunk {
        /** The run's blocks in chain order, one page per KV group each ({@code pages}). */
        final List<PrefixCache.Block> blocks = new ArrayList<>(4);
        /** The page row of the run's first row within the first block's page. */
        final int srcRow0;
        /** Prompt row the chunk's first row takes; the cached rows move here. */
        final int promptFrom;
        /** Chain row the chunk's first cached row holds; the shift is {@code promptFrom - cacheFrom}. */
        final int cacheFrom;
        final int rows;

        Chunk(int srcRow0, int promptFrom, int cacheFrom, int rows) {
            this.srcRow0 = srcRow0;
            this.promptFrom = promptFrom;
            this.cacheFrom = cacheFrom;
            this.rows = rows;
        }

        /** The uniform RoPE shift the chunk's re-appended keys re-rotate by ({@code kv_shift}). */
        int shift() {
            return promptFrom - cacheFrom;
        }
    }

    /** What a scan found: the chunks, the prompt rows they cover, and the page accounting their apply takes. */
    static final class Plan {
        final List<Chunk> chunks = new ArrayList<>(2);
        /** Prompt rows the chunks cover, past the reused prefix; the prefill starts here. */
        int promptEnd;
        /** Chunk rows a zero-shift apply shares as whole cached pages (they cost no page of their own). */
        int sharedRows;
        /** Every cached block the chunks read: eviction spares them while the plan stands. */
        final Set<PrefixCache.Block> blocks = Collections.newSetFromMap(new IdentityHashMap<>());

        boolean isEmpty() {
            return chunks.isEmpty();
        }

        /** The rows the chunks re-append. */
        int rows() {
            int rows = 0;
            for (Chunk chunk : chunks) {
                rows += chunk.rows;
            }
            return rows;
        }
    }

    /** A cached chain cursor: the block being walked and the page row inside it. */
    private static final class Cursor {
        PrefixCache.Block block;
        int row;

        Cursor(PrefixCache.Block block, int row) {
            this.block = block;
            this.row = row;
        }
    }

    /**
     * The chunks the block tree's cached chain yields past the match's boundary: llama.cpp's scan over the slot's
     * cached tokens, the cached side being the most recently used chain from the boundary. A run of fewer than
     * {@code n_cache_reuse} aligned rows never forms a chunk, and such a run advances the cached cursor alone.
     * Empty when the request turns chunk reuse off, the cache is off, or no run qualifies.
     */
    Plan scan(Submission head, PrefixCache.Match match) {
        Plan plan = new Plan();
        int nReuse = head.request.nCacheReuse();
        if (nReuse <= 0 || state.cache == null || state.checkpoints != null) {
            return plan;
        }
        CacheRows prompt = head.promptRows();
        int promptLast = prompt.length() - 1; // the last prompt row always prefills
        int nPast = match.tokens();
        int headP = nPast;
        Cursor cursor = boundaryCursor(head.prefixKey, match, prompt, headP);
        int headC = nPast;
        while (cursor.block != null && headP < promptLast) {
            if (cursor.row >= cursor.block.rows.length()) {
                cursor.block = descend(cursor.block, prompt, headP);
                cursor.row = 0;
                continue;
            }
            int run = runLength(cursor.block, cursor.row, prompt, headP, promptLast - headP);
            if (run >= nReuse) {
                plan.chunks.add(chunk(plan, cursor, prompt, headC, headP, run));
                headC += run;
                headP += run;
            } else {
                cursor.row += 1;
                headC += 1;
            }
        }
        if (!plan.chunks.isEmpty()) {
            plan.promptEnd = headP;
            countSharedPages(plan, state.cache.pageSize(), nPast);
        }
        return plan;
    }

    /**
     * Re-appends every chunk of {@code plan} at {@code seq}'s append cursor ({@link ForwardBackend#reuseChunk},
     * one call per chunk): its rows land at the prompt rows they replace, re-rotating by its shift, the rows
     * tiling the prompt from the match's boundary on. True when every chunk applied; false when the memory lacked
     * the pages the re-append takes (any applied rows rolled back, the sequence still serving prefix-only reuse).
     * Another memory failure rolls the applied rows back and rethrows: the caller frees the sequence and lets the
     * submission retry.
     */
    boolean apply(int seq, Submission head, Plan plan) {
        CacheRows prompt = head.promptRows();
        int from = head.reused;
        int applied = 0;
        try {
            for (Chunk chunk : plan.chunks) {
                int[][] srcPages = new int[chunk.blocks.size()][];
                for (int k = 0; k < chunk.blocks.size(); k++) {
                    srcPages[k] = chunk.blocks.get(k).pages;
                }
                backend.reuseChunk(seq, srcPages, chunk.srcRow0, from + applied, chunk.rows, chunk.shift(),
                        prompt.tokens(chunk.promptFrom, chunk.promptFrom + chunk.rows), embeddingToken);
                applied += chunk.rows;
                state.stats.chunkRowsReused += chunk.rows;
            }
            return true;
        } catch (RuntimeException e) {
            if (applied > 0) {
                state.sequences.remove(seq, from + applied, -1);
            }
            if (e instanceof MemoryFullException || e instanceof UnsupportedOperationException) {
                return false;
            }
            throw e;
        }
    }

    /**
     * Where the scan enters the cached chain: inside the match's boundary block at its shared rows when the match
     * ends there, else at page row 0 of the child the walk continues the match's last whole block (or the chain
     * roots) with. Null when nothing extends the match.
     */
    private Cursor boundaryCursor(PrefixKey key, PrefixCache.Match match, CacheRows prompt, int headP) {
        if (match.partial != null && match.partialRows > 0) {
            return new Cursor(match.partial, match.partialRows);
        }
        PrefixCache.Block parent = match.blocks.isEmpty() ? null : match.blocks.get(match.blocks.size() - 1);
        return new Cursor(descend(parent, key, prompt, headP), 0);
    }

    /** The cached block the walk continues {@code block} with once its rows walked out ({@link #descend}). */
    private PrefixCache.Block descend(PrefixCache.Block block, CacheRows prompt, int promptAt) {
        return descend(block, null, prompt, promptAt);
    }

    /** The cached child the run measure and the chunk walk continue a run's block with ({@link #descend}). */
    private PrefixCache.Block descendMatching(PrefixCache.Block block, CacheRows prompt, int promptAt) {
        return descend(block, null, prompt, promptAt);
    }

    /**
     * The cached child of {@code parent} ({@code parent}'s children, or the chain roots of {@code key} for null)
     * the walk continues into: the most recently used cached child whose first row matches the prompt row
     * {@code promptAt} — the run continues at once — or, when none matches, the most recently used cached child
     * whatever its first row, whose rows the walk steps past one by one hunting the next run (llama.cpp's scan
     * advances its cache cursor past mismatching rows the same way). Null when {@code parent} is uncached (a
     * taken-over boundary's own rows walk, its children do not serve) or no cached child exists.
     */
    private PrefixCache.Block descend(PrefixCache.Block parent, PrefixKey key, CacheRows prompt, int promptAt) {
        if (parent != null && !parent.cached) {
            return null;
        }
        PrefixCache.Block best = null;
        PrefixCache.Block matching = null;
        long bestUse = Long.MIN_VALUE;
        long matchingUse = Long.MIN_VALUE;
        for (PrefixCache.Block child : state.cache.chainChildren(parent, key)) {
            if (!child.cached || child.rows.length() < 1) {
                continue;
            }
            if (child.lastUse > bestUse) {
                best = child;
                bestUse = child.lastUse;
            }
            if (CacheRows.commonLength(child.rows, 0, prompt, promptAt, 1) == 1 && child.lastUse > matchingUse) {
                matching = child;
                matchingUse = child.lastUse;
            }
        }
        return matching != null ? matching : best;
    }

    /**
     * The aligned run the cached chain holds from cursor ({@code block}, {@code row}) against the prompt rows
     * from {@code headP}, at most {@code maxRun} rows: the rows match by token across whole blocks, the next
     * block of the chain ({@link #descend}) continuing the run, until a row mismatches, a block's end finds no
     * cached child, or the cap is reached.
     */
    private int runLength(PrefixCache.Block block, int row, CacheRows prompt, int headP, int maxRun) {
        int run = 0;
        PrefixCache.Block cur = block;
        int r = row;
        while (run < maxRun && cur != null) {
            if (r >= cur.rows.length()) {
                cur = descendMatching(cur, prompt, headP + run);
                r = 0;
                continue;
            }
            int cap = Math.min(cur.rows.length() - r, maxRun - run);
            int got = CacheRows.commonLength(cur.rows, r, prompt, headP + run, cap);
            run += got;
            r += got;
            if (got < cap) {
                break; // a row mismatches inside the block
            }
        }
        return run;
    }

    /**
     * Adds the chunk the aligned run of {@code run} rows from cached row {@code cacheFrom} (at cursor {@code at})
     * to prompt row {@code promptFrom} forms: its blocks, chained while the run's rows continue (a whole block's
     * page rows chain onto the matching child's page row 0; a block whose rows end short closes the chunk). The
     * cursor ends past the chunk's last row, on the block the run walked into and at its next row.
     */
    private Chunk chunk(Plan plan, Cursor at, CacheRows prompt, int cacheFrom, int promptFrom, int run) {
        Chunk chunk = new Chunk(at.row, promptFrom, cacheFrom, run);
        int remain = run;
        while (remain > 0) {
            int take = Math.min(at.block.rows.length() - at.row, remain);
            chunk.blocks.add(at.block);
            plan.blocks.add(at.block);
            at.row += take;
            remain -= take;
            if (remain <= 0 || at.row < at.block.rows.length()) {
                break; // the chunk's rows are covered, or the run ends inside the block
            }
            // the run's rows walked out of the block: the chain child the run measure continued them with takes
            // the walk over at page row 0; a block whose page rows end short has none, which the scan never
            // reaches mid-run
            PrefixCache.Block next = descendMatching(at.block, prompt, promptFrom + (run - remain));
            if (next == null) {
                throw new IllegalStateException("the chunk scan walked past the cached chain's end");
            }
            at.block = next;
            at.row = 0;
        }
        return chunk;
    }

    /**
     * The chunk rows a zero-shift apply shares as whole cached pages: per chunk whose rows start at page row 0
     * with the apply cursor page-aligned, the whole pages of its run. The apply cursor walks the re-appended rows
     * in chunk order; a shifted chunk re-rotates its rows, so none of its pages shares.
     */
    private static void countSharedPages(Plan plan, int pageSize, int nPast) {
        int cursor = nPast;
        for (Chunk chunk : plan.chunks) {
            if (chunk.shift() == 0 && chunk.srcRow0 == 0 && cursor % pageSize == 0) {
                plan.sharedRows += chunk.rows - chunk.rows % pageSize;
            }
            cursor += chunk.rows;
        }
    }
}
