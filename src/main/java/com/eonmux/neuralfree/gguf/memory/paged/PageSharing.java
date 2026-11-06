package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The page-sharing half of {@link PagedSequences}: the block holders ({@link #holdPages}, {@link #importPages},
 * {@link #releasePages} and the {@link PageWindow} of a context checkpoint), the adoption of held blocks by an empty
 * sequence ({@link #adoptPages}), the copy of one sequence's rows onto another ({@link #copy}), and the page-pool and
 * window queries a holder asks. It drives the owner's tables (pools, page lists, row mirrors and slot cursors)
 * directly and keeps the owner the single authority for the page-reference bookkeeping.
 *
 * <p>Not thread-safe: it runs on the owner's engine thread, as the owner does.
 */
final class PageSharing {

    private final PagedSequences owner;

    PageSharing(PagedSequences owner) {
        this.owner = Objects.requireNonNull(owner, "owner");
    }

    /** KV groups of the memory: a block's pages hold one page per group. */
    int groups() {
        return owner.groupsCount();
    }

    /** The fewest idle pages of any KV group: 0 when some group could not allocate a page now. */
    int minFreePages() {
        int min = Integer.MAX_VALUE;
        for (int g = 0; g < groups(); g++) {
            min = Math.min(min, pools()[g].free());
        }
        return min;
    }

    /** Idle pages of KV group {@code g}. */
    int freePages(int g) {
        return pools()[Objects.checkIndex(g, groups())].free();
    }

    /**
     * The pages a write of {@code rows} rows by {@code seq} takes in KV group {@code g}, the count
     * {@link PagedSequences#prepare} reserves: a private copy of every shared page of the span plus the new tail
     * pages.
     */
    int pagesToWrite(int seq, int g, int rows) {
        owner.checkSeq(seq);
        return needPages(Objects.checkIndex(g, groups()), new int[] {seq}, new int[] {rows});
    }

    /**
     * The pages of block {@code block} of {@code seq}, one per KV group, when the block's first {@code rows} rows
     * hold the sequence's own positions {@code [block * P, block * P + rows)} at page rows {@code [0, rows)} in
     * every full-attention group; null when the sequence stores them otherwise (it starts above position 0, its
     * cursor stops short of them, or a row holds another position). A sliding-window group ({@link #trims}) gives
     * -1: its rows leave the sequence as the window moves, so a block never holds them ({@link #holdWindow} holds a
     * window's pages instead).
     *
     * @throws IllegalArgumentException when {@code block} is negative or {@code rows} is outside {@code [1, P]}
     */
    int[] blockPages(int seq, int block, int rows) {
        return blockPages(seq, block, rows, null);
    }

    /**
     * {@link #blockPages(int, int, int)} for rows that store explicit positions: page row {@code r} must hold
     * {@code positions[r]} in every full-attention group (a media chunk's rows share their base position, so a
     * sequence's rows and positions part after it); null expects the positions {@code [block * P, block * P + rows)}.
     *
     * @throws IllegalArgumentException when {@code block} is negative, {@code rows} is outside {@code [1, P]} or
     *                                  {@code positions} holds another count than {@code rows}
     */
    int[] blockPages(int seq, int block, int rows, int[] positions) {
        owner.checkSeq(seq);
        int pageSize = owner.pageSize();
        if (block < 0 || rows < 1 || rows > pageSize) {
            throw new IllegalArgumentException("block " + block + " of " + rows + " rows outside a page of "
                    + pageSize);
        }
        if (positions != null && positions.length != rows) {
            throw new IllegalArgumentException(positions.length + " positions for a block of " + rows + " rows");
        }
        if (owner.posMinArr()[seq] != 0) {
            return null;
        }
        int first = block * pageSize;
        int[] pages = new int[groups()];
        for (int g = 0; g < groups(); g++) {
            pages[g] = -1;
            if (owner.swaGroups()[g]) {
                continue;
            }
            PageList list = lists()[seq][g];
            if (block >= list.size() || slots()[g][seq] < first + rows) {
                return null;
            }
            int base = list.page(block) * pageSize;
            for (int r = 0; r < rows; r++) {
                if (kvPos()[g][base + r] != (positions == null ? first + r : positions[r])) {
                    return null;
                }
            }
            pages[g] = list.page(block);
        }
        return pages;
    }

    /**
     * Takes a holder's reference on each page of {@code pages} (one per KV group, -1 for a sliding-window group):
     * the pages stay allocated, with their rows, after every sequence released them, until {@link #releasePages}
     * drops the reference.
     *
     * @throws IllegalStateException when a page is free (a holder references only pages a sequence holds)
     */
    void holdPages(int[] pages) {
        checkBlock(pages);
        for (int g = 0; g < pages.length; g++) {
            if (pages[g] >= 0) {
                pools()[g].ref(pages[g]);
            }
        }
    }

    /**
     * Takes one idle page per KV group for rows imported from the host (a restored slot or a host-RAM cached
     * block), the caller holding the reference as {@link #holdPages} would: page row {@code r} stores
     * {@code positions[r]} in every group, the rest of the page stays empty, and the device learns the new
     * {@code kv_pos} values through {@link PagedSequences#pendingReports()}. The caller writes the rows' K/V before
     * the next step reads them, and drops the reference with {@link #releasePages}.
     *
     * @return the pages, one per KV group
     * @throws IllegalArgumentException when {@code positions} holds no row, more rows than a page, a negative or a
     *                                  descending position
     * @throws IllegalStateException    when some KV group has no idle page ({@link #minFreePages()} is 0); nothing
     *                                  changes then
     */
    int[] importPages(int[] positions) {
        Objects.requireNonNull(positions, "positions");
        int pageSize = owner.pageSize();
        int rows = positions.length;
        if (rows < 1 || rows > pageSize) {
            throw new IllegalArgumentException("an imported page holds 1 to " + pageSize + " rows, got " + rows);
        }
        for (int r = 0; r < rows; r++) {
            if (positions[r] < 0 || (r > 0 && positions[r] < positions[r - 1])) {
                throw new IllegalArgumentException("imported row " + r + " stores position " + positions[r]
                        + (r > 0 ? " after " + positions[r - 1] : ""));
            }
        }
        if (minFreePages() == 0) {
            throw new IllegalStateException("no idle page in some KV group for an imported block: " + owner);
        }
        int[] pages = new int[groups()];
        int[] rowIds = new int[rows];
        for (int g = 0; g < groups(); g++) {
            pages[g] = pools()[g].alloc();
            int base = pages[g] * pageSize;
            for (int r = 0; r < rows; r++) {
                kvPos()[g][base + r] = positions[r];
                rowIds[r] = base + r;
            }
            usedRows()[g] += rows;
            pendingReports().add(RowReport.sets(g, rowIds, positions));
        }
        return pages;
    }

    /**
     * The positions page rows {@code [0, rows)} of {@code page} store in KV group {@code group}, -1 for an empty
     * row (a copy of the host mirror).
     *
     * @throws IndexOutOfBoundsException when the group or the page is outside the pools
     * @throws IllegalArgumentException  when {@code rows} is outside {@code [0, P]}
     */
    int[] pagePositions(int group, int page, int rows) {
        int pageSize = owner.pageSize();
        if (group < 0 || group >= groups()) {
            throw new IndexOutOfBoundsException("KV group " + group + " outside [0, " + groups() + ")");
        }
        if (page < 0 || page >= pools()[group].pages()) {
            throw new IndexOutOfBoundsException("page " + page + " outside the " + pools()[group].pages()
                    + " pages of group " + group);
        }
        if (rows < 0 || rows > pageSize) {
            throw new IllegalArgumentException(rows + " rows outside a page of " + pageSize);
        }
        int base = page * pageSize;
        return Arrays.copyOfRange(kvPos()[group], base, base + rows);
    }

    /**
     * Drops a holder's reference on each page of {@code pages}; a page no sequence holds then returns to its pool
     * with its rows empty (reported through {@link PagedSequences#pendingReports()}).
     */
    void releasePages(int[] pages) {
        checkBlock(pages);
        for (int g = 0; g < pages.length; g++) {
            if (pages[g] >= 0) {
                unref(true, g, pages[g]);
            }
        }
    }

    /**
     * Whether a page of {@code pages} has more than one reference: beside its holder, a sequence or a pending copy
     * still uses it.
     */
    boolean pagesShared(int[] pages) {
        checkBlock(pages);
        for (int g = 0; g < pages.length; g++) {
            if (pages[g] >= 0 && pools()[g].refCount(pages[g]) > 1) {
                return true;
            }
        }
        return false;
    }

    /** Pages of {@code window} that only its holder references: the pages releasing it returns to the pool. */
    int windowOnlyPages(PageWindow window) {
        int only = 0;
        for (int k = 0; k < window.size(); k++) {
            only += pools()[window.group()].refCount(window.page(k)) == 1 ? 1 : 0;
        }
        return only;
    }

    /** Whether KV group {@code g} trims a sliding window: its sequences drop the pages below their window. */
    boolean trims(int g) {
        return owner.swaGroups()[Objects.checkIndex(g, groups())];
    }

    /** The widest reader window of a trimming group {@code g} ({@link #trims}); 0 for a full-attention group. */
    int windowWidth(int g) {
        return owner.windowWidths()[Objects.checkIndex(g, groups())];
    }

    /**
     * Takes a holder's reference on every page sequence {@code seq} holds in the trimming group {@code g} when it
     * has stored {@code rows} rows: the window of a context checkpoint at {@code rows}, which a later sequence
     * adopts. The sequence's next write into its last page copies its own rows first (the page is shared now).
     *
     * @throws IllegalArgumentException when {@code g} does not trim, or {@code rows} is not the row count the
     *                                  sequence's pages of the group cover
     */
    PageWindow holdWindow(int seq, int g, int rows) {
        owner.checkSeq(seq);
        int pageSize = owner.pageSize();
        if (!trims(g)) {
            throw new IllegalArgumentException("KV group " + g + " keeps every row; a window holds a trimming group");
        }
        PageList list = lists()[seq][g];
        int trimmed = rows - slots()[g][seq];
        if (list.isEmpty() || trimmed < 0 || trimmed % pageSize != 0
                || (rows + pageSize - 1) / pageSize - trimmed / pageSize != list.size()) {
            throw new IllegalArgumentException("sequence " + seq + " holds " + list.size() + " pages for "
                    + slots()[g][seq] + " slots in group " + g + ", not the window of " + rows + " rows");
        }
        int[] pages = new int[list.size()];
        for (int k = 0; k < pages.length; k++) {
            pages[k] = list.page(k);
            pools()[g].ref(pages[k]);
        }
        return new PageWindow(g, trimmed / pageSize, pages, rows);
    }

    /** Drops a holder's reference on every page of {@code window}; a page no one else holds returns to its pool. */
    void releaseWindow(PageWindow window) {
        Objects.requireNonNull(window, "window");
        for (int k = 0; k < window.size(); k++) {
            unref(true, window.group(), window.page(k));
        }
    }

    /**
     * Gives the empty sequence {@code seq} the held pages of consecutive blocks as its positions from 0:
     * {@code blocks[k]} holds block {@code k}'s page per KV group, every block whole but the last, whose first
     * {@code lastRows} rows join. The sequence shares the pages (its first write into a shared page copies its own
     * rows first), its cursor continues after the adopted rows, and its token record takes {@code tokens}. Checks
     * everything before it changes anything.
     *
     * @throws IllegalStateException    when {@code seq} stores positions, or a page is free or holds other
     *                                  positions than its block's
     * @throws IllegalArgumentException when the block shapes or the token count disagree
     */
    void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens) {
        adoptPages(seq, blocks, lastRows, tokens, null);
    }

    /**
     * {@link #adoptPages(int, int[][], int, int[], int[], int)} for a model whose embedding rows record no token
     * ({@code -1}).
     */
    void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens, int[] positions) {
        adoptPages(seq, blocks, lastRows, tokens, positions, -1);
    }

    /**
     * {@link #adoptPages(int, int[][], int, int[])} for rows that store explicit positions: adopted row {@code r}
     * must hold {@code positions[r]} in every group (null expects position {@code r}), and the token record takes
     * one entry per row at its stored position, as {@link PagedSequences#appendRows} records a forward's rows: a row
     * whose token is {@code -1} (a media or prompt-embedding row) records {@code embeddingToken} as an embedding
     * row. Without positions the rows are the token run of {@link #adoptPages(int, int[][], int, int[])}. The
     * sequence's highest position becomes the last row's. A memory with trimming groups adopts through
     * {@link #adoptPages(int, int[][], int, int[], int[], PageWindow[], int)}.
     *
     * @param embeddingToken the id an embedding row records: {@code -1} for none, or its model's stand-in
     * @throws IllegalStateException    when {@code seq} stores positions, or a page is free or holds other
     *                                  positions than its rows'
     * @throws IllegalArgumentException when the block shapes, the token count or the position count disagree, or a
     *                                  position descends
     */
    void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens, int[] positions, int embeddingToken) {
        adoptPages(seq, blocks, lastRows, tokens, positions, new PageWindow[groups()], embeddingToken);
    }

    /**
     * {@link #adoptPages(int, int[][], int, int[], int[], int)} for a memory with trimming groups: the blocks give
     * the full-attention groups' pages (-1 for a trimming group), and {@code windows[g]} gives trimming group
     * {@code g}'s pages, a window held at exactly the adopted row count ({@link #holdWindow}); its rows of the last
     * {@link #windowWidth} adopted rows must be live. The sequence's list of such a group starts at the window's
     * first page, as if it had trimmed the pages before it.
     *
     * @throws IllegalStateException    when {@code seq} stores positions, or a page is free or holds other
     *                                  positions than its rows'
     * @throws IllegalArgumentException when the block shapes, the token or position count, or the windows disagree
     */
    void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens, int[] positions,
                    PageWindow[] windows, int embeddingToken) {
        owner.checkSeq(seq);
        int pageSize = owner.pageSize();
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(windows, "windows");
        for (int[] block : blocks) {
            checkBlock(block);
        }
        int n = blocks.length;
        int total = (n - 1) * pageSize + lastRows;
        if (n == 0 || lastRows < 1 || lastRows > pageSize || tokens.length != total
                || (positions != null && positions.length != total) || windows.length != groups()) {
            throw new IllegalArgumentException(n + " blocks with " + lastRows + " last rows do not hold "
                    + tokens.length + " tokens" + (positions == null ? "" : " at " + positions.length
                    + " positions") + " over " + windows.length + " group windows");
        }
        int[] stored = positions != null ? positions : identityPositions(total);
        for (int r = 1; r < total; r++) {
            if (stored[r] < stored[r - 1]) {
                throw new IllegalArgumentException("adopted row " + r + " stores position " + stored[r] + " below "
                        + stored[r - 1]);
            }
        }
        int lastPos = stored[total - 1];
        if (lastPos + 1 > owner.cap()) {
            throw new MemoryFullException(lastPos + 1, owner.cap(), total, owner.freeCells());
        }
        for (int g = 0; g < groups(); g++) {
            if (!lists()[seq][g].isEmpty()) {
                throw new IllegalStateException("sequence " + seq + " holds pages; adoption needs an empty one");
            }
            if (owner.swaGroups()[g]) {
                PageChecks.checkWindow(pools()[g], kvPos()[g], owner.windowWidths()[g], g, windows[g], blocks,
                        total, positions);
            } else {
                PageChecks.checkBlockPages(pools()[g], kvPos()[g], g, blocks, lastRows, positions);
            }
        }
        for (int g = 0; g < groups(); g++) {
            int[] pages = owner.swaGroups()[g] ? windows[g].pages() : groupPages(g, blocks);
            for (int page : pages) {
                pools()[g].ref(page);
                lists()[seq][g].append(page);
            }
            slots()[g][seq] = owner.swaGroups()[g] ? total - windows[g].firstPage() * pageSize : total;
        }
        owner.posMinArr()[seq] = stored[0];
        owner.posMaxArr()[seq] = lastPos;
        SequenceState state = owner.sequence(seq);
        state.clear();
        if (positions == null) {
            state.append(tokens, 0, total);
        } else {
            state.appendRows(tokens, stored, 0, total, embeddingToken);
        }
    }

    /** The positions {@code [0, n)}. */
    private static int[] identityPositions(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = i;
        }
        return out;
    }

    /** Group {@code g}'s page of every block. */
    private static int[] groupPages(int g, int[][] blocks) {
        int[] pages = new int[blocks.length];
        for (int k = 0; k < blocks.length; k++) {
            pages[k] = blocks[k][g];
        }
        return pages;
    }

    private void checkBlock(int[] pages) {
        Objects.requireNonNull(pages, "pages");
        if (pages.length != groups()) {
            throw new IllegalArgumentException("a block holds one page per KV group (" + groups() + "), got "
                    + pages.length);
        }
    }

    @Override
    public String toString() {
        return "PageSharing{" + groups() + " KV groups}";
    }

    /**
     * Copies positions {@code [p0, p1)} (or up to {@code posMax(src)} for {@code p1 = -1}) of sequence {@code src}
     * to the empty sequence {@code dst}: whole pages are shared, a leading partial page is copied into a private
     * page, and the destination's token record follows. See {@link PagedSequences#copy(int, int, int, int)}.
     */
    void copy(int src, int dst, int p0, int p1) {
        owner.checkSeq(src);
        owner.checkSeq(dst);
        if (src == dst) {
            throw new IllegalArgumentException("copy needs two different sequences, got " + src);
        }
        int lo = p0;
        int hi = normalizeRange(src, p0, p1);
        int srcMax = owner.posMaxArr()[src];
        if (lo < 0) {
            throw new IllegalArgumentException("copy starts before the source: position " + lo);
        }
        if (srcMax < 0 || hi <= lo || lo > srcMax) {
            throw new IllegalArgumentException("copy range [" + lo + ", " + hi + ") holds nothing of sequence "
                    + src + " (" + owner.posMinArr()[src] + ".." + srcMax + ")");
        }
        if (owner.posMaxArr()[dst] >= 0) {
            throw new IllegalStateException("copy shares pages with sequence " + dst + ", which stores positions "
                    + owner.posMinArr()[dst] + ".." + owner.posMaxArr()[dst] + "; the shared rows keep the source"
                    + " positions, so the destination must be empty");
        }
        // all-or-nothing: count the private leading pages first, no pool touched
        for (int g = 0; g < groups(); g++) {
            int need = 0;
            PageList from = lists()[src][g];
            for (int k = 0; k < from.size(); k++) {
                if (copyShareKind(g, from.page(k), lo, hi) == 2) {
                    need++;
                }
            }
            if (need > pools()[g].free()) {
                throw new MemoryFullException(hi - lo, owner.cap(), hi - lo, owner.freeCells());
            }
        }
        for (int g = 0; g < groups(); g++) {
            copyGroup(g, src, dst, lo, hi);
        }
        owner.sequence(dst).copyFrom(owner.sequence(src), lo, hi);
        owner.posMinArr()[dst] = lo;
        owner.posMaxArr()[dst] = hi - 1;
    }

    /** Part of {@code page} in the copy range: 0 none, 1 the whole in-range part (share it), 2 a leading partial. */
    private int copyShareKind(int g, int page, int lo, int hi) {
        int pageSize = owner.pageSize();
        int base = page * pageSize;
        boolean liveInRange = false;
        boolean liveBefore = false;
        for (int r = 0; r < pageSize; r++) {
            int pos = kvPos()[g][base + r];
            if (pos < 0) {
                continue;
            }
            if (pos >= lo && pos < hi) {
                liveInRange = true;
            } else if (pos < lo) {
                liveBefore = true;
            }
        }
        if (!liveInRange) {
            return 0;
        }
        return liveBefore ? 2 : 1;
    }

    private void copyGroup(int g, int src, int dst, int lo, int hi) {
        int pageSize = owner.pageSize();
        PageList from = lists()[src][g];
        PageList to = lists()[dst][g];
        int first = 0;
        while (first < from.size() && copyShareKind(g, from.page(first), lo, hi) == 0) {
            first++; // leading pages outside the range are neither shared nor copied
        }
        for (int k = first; k < from.size(); k++) {
            int page = from.page(k);
            int kind = copyShareKind(g, page, lo, hi);
            if (kind == 0) {
                break;
            }
            if (kind == 1) {
                pools()[g].ref(page);
                to.append(page);
                continue;
            }
            // rows before the range stay on the source alone: the destination gets a private page and copies only
            // its in-range rows into rows [firstRow, lastRowPlusOne)
            int pageBase = page * pageSize;
            int row0 = pageSize;
            int lastRowPlusOne = 0;
            for (int r = 0; r < pageSize; r++) {
                int pos = kvPos()[g][pageBase + r];
                if (pos >= lo && pos < hi) {
                    row0 = Math.min(row0, r);
                    lastRowPlusOne = Math.max(lastRowPlusOne, r + 1);
                }
            }
            int priv = pools()[g].alloc();
            // the pending copy holds one reference of its own on the source page, so no holder's release can
            // recycle the rows before the copy's execution; releaseCopies hands it back
            pools()[g].ref(page);
            pendingCopies().add(PageCopy.ofRange(g, page, priv, row0, lastRowPlusOne - row0));
            to.append(priv);
            int[] rows = new int[lastRowPlusOne - row0];
            int[] values = new int[rows.length];
            int n = 0;
            for (int r = row0; r < lastRowPlusOne; r++) {
                int pos = kvPos()[g][pageBase + r];
                kvPos()[g][priv * pageSize + r] = pos;
                if (pos >= 0) {
                    usedRows()[g]++;
                    rows[n] = priv * pageSize + r;
                    values[n] = pos;
                    n++;
                }
            }
            if (n > 0) {
                // the device applies the copy besides the step's writes: it needs the copied rows' positions
                pendingReports().add(RowReport.sets(g, Arrays.copyOf(rows, n), Arrays.copyOf(values, n)));
            }
        }
        // the copied rows end inside the last page (a page fully covered by the range ends whole): the next append
        // continues there and privatizes the last page first when it is shared
        slots()[g][dst] = slotsOfList(to, g, lo, hi);
    }

    /**
     * Slots the copied list covers: a page per shorter page plus the in-range rows of the last one, so the next
     * append continues on the last copied row. Appends that reach a shared partial tail page privatize it first.
     */
    private int slotsOfList(PageList list, int g, int lo, int hi) {
        if (list.isEmpty()) {
            return 0;
        }
        int pageSize = owner.pageSize();
        int last = list.size() - 1;
        int base = list.page(last) * pageSize;
        int count = 0;
        for (int r = 0; r < pageSize; r++) {
            int pos = kvPos()[g][base + r];
            if (pos >= lo && pos < hi) {
                count = r + 1;
            }
        }
        return last * pageSize + count;
    }

    /**
     * Re-appends the held rows of cached pages at the append cursor of sequence {@code seq} (llama.cpp server's KV
     * chunk reuse, the copy side of its {@code seq_rm} + {@code seq_add}): the run of {@code rows} rows starts at
     * source row {@code srcRow0} of run page 0 and lands at the sequence's own slots, at the positions
     * {@code [pos0, pos0 + rows)} its prompt rows take — copied rows re-rotate by the uniform {@code delta}
     * between a source row's stored position and its new one ({@link RowReport#shift}), and a {@code delta} 0 run
     * whose rows meet the cursor page-aligned shares whole cached pages instead of copying them. The token record
     * takes one entry per row ({@code tokens[r]}, an embedding row negative, recording {@code embeddingToken}).
     * All-or-nothing: the new tail pages and the private copies of every written shared page are counted per group
     * and refused as a whole ({@link MemoryFullException}) before any pool changes.
     *
     * @param srcPages the run's pages: {@code srcPages[runPage][group]}, covering source rows
     *                 {@code [srcRow0, srcRow0 + rows)} in page order
     * @throws IllegalArgumentException  for a run outside the pages, a token count or position mismatch, or source
     *                                   rows the cache does not hold
     * @throws IllegalStateException     when the sequence's groups disagree on the cursor or a group's page list
     *                                   does not cover its cursor, or the source cells disagree between groups
     * @throws MemoryFullException       when the run needs pages or rows the pools or the sequence lack
     * @throws UnsupportedOperationException for a trimming group (chunk reuse needs a full-attention memory)
     */
    void reusePages(int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                    int[] tokens, int embeddingToken) {
        owner.checkSeq(seq);
        if (rows <= 0) {
            throw new IllegalArgumentException("a reused chunk holds rows > 0, got " + rows);
        }
        Objects.requireNonNull(tokens, "tokens");
        int pageSize = owner.pageSize();
        if (tokens.length != rows) {
            throw new IllegalArgumentException("a reused chunk of " + rows + " rows takes " + rows + " tokens, got "
                    + tokens.length);
        }
        if (srcRow0 < 0 || srcRow0 >= pageSize) {
            throw new IllegalArgumentException("reused chunk starts at source row " + srcRow0 + " of its first"
                    + " page, outside the page's " + pageSize + " rows");
        }
        int runPages = (srcRow0 + rows + pageSize - 1) / pageSize;
        if (srcPages.length != runPages) {
            throw new IllegalArgumentException("a reused chunk of " + rows + " rows from source row " + srcRow0
                    + " spans " + runPages + " pages, got " + srcPages.length);
        }
        for (int[] block : srcPages) {
            checkBlock(block);
        }
        int cursor = slots()[0][seq];
        for (int g = 0; g < groups(); g++) {
            if (slots()[g][seq] != cursor) {
                throw new IllegalStateException("sequence " + seq + " holds " + cursor + " slots in group 0 but "
                        + slots()[g][seq] + " in group " + g + "; chunk reuse appends whole rows at one cursor");
            }
            if (owner.trims(g)) {
                throw new UnsupportedOperationException("chunk reuse appends whole cached rows, which the trimming"
                        + " group " + g + " does not hold: it needs a full-attention memory");
            }
            int listLen = lists()[seq][g].size();
            if (listLen != (cursor + pageSize - 1) / pageSize) {
                throw new IllegalStateException("page list of sequence " + seq + " group " + g + " covers "
                        + listLen + " pages for " + cursor + " slots");
            }
        }
        if (owner.posMaxArr()[seq] < 0) {
            throw new IllegalArgumentException("sequence " + seq + " stores nothing; chunk reuse re-appends held"
                    + " rows past an adopted prefix");
        }
        if (pos0 < owner.posMaxArr()[seq] + 1) {
            throw new IllegalArgumentException("sequence " + seq + " stores positions up to "
                    + owner.posMaxArr()[seq] + ", so its reused chunk starts at " + (owner.posMaxArr()[seq] + 1)
                    + " or later, got " + pos0);
        }
        if (pos0 + rows > owner.cap()) {
            throw new MemoryFullException(pos0 + rows, owner.cap(), rows, owner.freeCells());
        }
        if (owner.sequence(seq).nPast() + rows > owner.sequence(seq).capacity()) {
            throw new MemoryFullException(owner.sequence(seq).nPast() + rows, owner.sequence(seq).capacity(), rows,
                    owner.freeCells());
        }
        // the source rows' positions, checked live and equal across groups: the run's destination positions follow
        int[] srcPositions = new int[rows];
        for (int r = 0; r < rows; r++) {
            int srcAbs = srcRow0 + r;
            int page = srcPages[srcAbs / pageSize][0];
            int row = srcAbs % pageSize;
            int pos = kvPos()[0][page * pageSize + row];
            if (pos < 0) {
                throw new IllegalArgumentException("the reused chunk's source row " + r + " (cached page " + page
                        + ", row " + row + " of group 0) holds no position");
            }
            for (int g = 1; g < groups(); g++) {
                int other = kvPos()[g][srcPages[srcAbs / pageSize][g] * pageSize + row];
                if (other != pos) {
                    throw new IllegalStateException("the reused chunk's source row " + r + " holds position " + pos
                            + " in group 0 but " + other + " in group " + g);
                }
            }
            if (r > 0 && pos <= srcPositions[r - 1]) {
                throw new IllegalArgumentException("the reused chunk's source row " + r + " stores position " + pos
                        + " at or below row " + (r - 1) + "'s " + srcPositions[r - 1]);
            }
            if (pos + delta != pos0 + r) {
                throw new IllegalArgumentException("the reused chunk's source row " + r + " stores position " + pos
                        + ", which lands at " + (pos + delta) + " by the shift " + delta + " instead of the"
                        + " requested " + (pos0 + r));
            }
            srcPositions[r] = pos;
        }
        for (int[] block : srcPages) {
            for (int g = 0; g < groups(); g++) {
                if (pools()[g].refCount(block[g]) < 1) {
                    throw new IllegalArgumentException("the reused chunk's cached page " + block[g] + " of group "
                            + g + " holds no reference (it is free)");
                }
            }
        }
        // all-or-nothing: the exact page count of every group, no pool touched yet; a destination page a whole
        // cached page shares allocates none, so it is subtracted from the new tail pages
        for (int g = 0; g < groups(); g++) {
            int listLen = lists()[seq][g].size();
            int spanLast = (cursor + rows - 1) / pageSize;
            int need = 0;
            for (int k = cursor / pageSize; k <= spanLast && k < listLen; k++) {
                if (pools()[g].refCount(lists()[seq][g].page(k)) > 1) {
                    need++;
                }
            }
            int newPages = Math.max((cursor + rows + pageSize - 1) / pageSize - listLen, 0);
            need += newPages - sharedSegments(g, seq, cursor, rows, delta, srcRow0);
            if (need > pools()[g].free()) {
                throw new MemoryFullException(rows, owner.cap(), rows, owner.freeCells());
            }
        }
        for (int g = 0; g < groups(); g++) {
            reuseGroup(g, seq, srcPages, srcRow0, pos0, rows, delta, cursor);
        }
        owner.posMaxArr()[seq] = Math.max(owner.posMaxArr()[seq], pos0 + rows - 1);
        int[] stored = new int[rows];
        for (int r = 0; r < rows; r++) {
            stored[r] = pos0 + r;
        }
        owner.sequence(seq).appendRows(tokens, stored, 0, rows, embeddingToken);
    }

    /**
     * Destination pages of a run at {@code cursor} that whole cached pages share, so none allocates: the count
     * {@link #reusePages} subtracts from its new tail pages, mirroring the share rule of {@code reuseGroup} (a
     * {@code delta} 0 run whose rows cover one destination page whole, at the source rows' own page offsets, and
     * at the page list's end — which the walk's own appends move forward, so a later segment counts against the
     * pages the walk added before it). A whole cached page's source reference is already checked, so none is here.
     */
    private int sharedSegments(int g, int seq, int cursor, int rows, int delta, int srcRow0) {
        if (delta != 0) {
            return 0;
        }
        int pageSize = owner.pageSize();
        int firstPage = cursor / pageSize;
        int destPages = (cursor + rows + pageSize - 1) / pageSize - firstPage;
        int listLen = lists()[seq][g].size();
        int shares = 0;
        int grown = 0;
        for (int dk = 0; dk < destPages; dk++) {
            int listIndex = firstPage + dk;
            int slotBase = listIndex * pageSize;
            if (Math.max(cursor - slotBase, 0) == 0 && Math.min(pageSize, cursor + rows - slotBase) == pageSize
                    && listIndex == listLen + grown
                    && (srcRow0 + slotBase - cursor) % pageSize == 0) {
                shares++;
            }
            if (listIndex >= listLen) {
                grown++;
            }
        }
        return shares;
    }

    /** One group of {@link #reusePages}: appends the run's pages at the cursor, sharing or copying each. */
    private void reuseGroup(int g, int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                            int cursor) {
        int pageSize = owner.pageSize();
        PageList list = lists()[seq][g];
        int firstPage = cursor / pageSize;
        int destPages = (cursor + rows + pageSize - 1) / pageSize - firstPage;
        int[] reportRows = new int[rows];
        int[] reportValues = new int[rows];
        int filled = 0;
        for (int dk = 0; dk < destPages; dk++) {
            int listIndex = firstPage + dk;
            int slotBase = listIndex * pageSize;
            int t0 = Math.max(cursor - slotBase, 0);
            int t1 = Math.min(pageSize, cursor + rows - slotBase);
            int dst;
            boolean share = delta == 0 && t0 == 0 && t1 == pageSize && listIndex == list.size()
                    && (srcRow0 + slotBase - cursor) % pageSize == 0
                    && pools()[g].refCount(srcPages[(srcRow0 + slotBase - cursor) / pageSize][g]) >= 1;
            if (listIndex < list.size()) {
                dst = list.page(listIndex);
                if (pools()[g].refCount(dst) > 1) {
                    // the destination's rows on the shared page move to a fresh private copy first (the
                    // copy-on-write rule); the run's rows then land on the private page
                    int priv = pools()[g].alloc();
                    PageCopy cow = PageCopy.of(g, dst, priv, t0);
                    pendingCopies().add(cow);
                    int srcBase = dst * pageSize;
                    int dstBase = priv * pageSize;
                    System.arraycopy(kvPos()[g], srcBase, kvPos()[g], dstBase, t0);
                    int[] cowRows = new int[t0];
                    int[] cowValues = new int[t0];
                    int cowFill = 0;
                    for (int r = 0; r < t0; r++) {
                        int pos = kvPos()[g][dstBase + r];
                        if (pos >= 0) {
                            usedRows()[g]++;
                            cowRows[cowFill] = dstBase + r;
                            cowValues[cowFill] = pos;
                            cowFill++;
                        }
                    }
                    if (cowFill > 0) {
                        pendingReports().add(RowReport.sets(g, Arrays.copyOf(cowRows, cowFill),
                                Arrays.copyOf(cowValues, cowFill)));
                    }
                    list.set(listIndex, priv);
                    dst = priv;
                }
            } else if (share) {
                int src = srcPages[(srcRow0 + slotBase - cursor) / pageSize][g];
                pools()[g].ref(src);
                list.append(src);
                continue;
            } else {
                while (list.size() < listIndex) {
                    list.append(pools()[g].alloc());
                }
                dst = pools()[g].alloc();
                list.append(dst);
            }
            int s = t0;
            while (s < t1) {
                int runRow = slotBase + s - cursor;
                int srcAbs = srcRow0 + runRow;
                int ks = srcAbs / pageSize;
                int srcRow = srcAbs % pageSize;
                int count = Math.min(t1 - s, pageSize - srcRow);
                int src = srcPages[ks][g];
                // the pending copy holds one reference of its own on the source page (the cache's block keeps
                // its rows alive regardless), released when the copy's execution hands it back
                pools()[g].ref(src);
                pendingCopies().add(PageCopy.ofShifted(g, src, dst, srcRow, s, count));
                int srcBase = src * pageSize;
                int dstBase = dst * pageSize;
                for (int r = 0; r < count; r++) {
                    int pos = kvPos()[g][srcBase + srcRow + r];
                    if (kvPos()[g][dstBase + s + r] < 0) {
                        usedRows()[g]++;
                    }
                    kvPos()[g][dstBase + s + r] = pos + delta;
                    reportRows[filled] = dstBase + s + r;
                    reportValues[filled] = pos + delta;
                    filled++;
                }
                s += count;
            }
        }
        if (filled > 0) {
            sortByRow(reportRows, reportValues, filled);
            // the copied rows' positions: a shifted run adds delta on the device (its keys re-rotate by it), an
            // unshifted one carries the source positions unchanged
            pendingReports().add(delta == 0
                    ? RowReport.sets(g, Arrays.copyOf(reportRows, filled), Arrays.copyOf(reportValues, filled))
                    : RowReport.shift(g, Arrays.copyOf(reportRows, filled), Arrays.copyOf(reportValues, filled),
                            delta));
        }
        slots()[g][seq] = cursor + rows;
    }

    /** Sorts the first {@code n} row/value pairs by row, so a report carries one ascending run. */
    private static void sortByRow(int[] rows, int[] values, int n) {
        for (int i = 1; i < n; i++) {
            int r = rows[i];
            int v = values[i];
            int j = i - 1;
            while (j >= 0 && rows[j] > r) {
                rows[j + 1] = rows[j];
                values[j + 1] = values[j];
                j--;
            }
            rows[j + 1] = r;
            values[j + 1] = v;
        }
    }

    /** Existing shared pages of each write span plus new tail pages, in the commit allocation order, one group. */
    int needPages(int g, int[] segSeq, int[] segLen) {
        int pageSize = owner.pageSize();
        int need = 0;
        for (int s = 0; s < segSeq.length; s++) {
            int seq = segSeq[s];
            int slot0 = slots()[g][seq];
            int end = slot0 + segLen[s];
            PageList list = lists()[seq][g];
            int listLen = list.size();
            int spanLast = (end - 1) / pageSize;
            for (int k = slot0 / pageSize; k <= spanLast && k < listLen; k++) {
                if (pools()[g].refCount(list.page(k)) > 1) {
                    need++;
                }
            }
            int newPages = (end + pageSize - 1) / pageSize - listLen;
            if (newPages > 0) {
                need += newPages;
            }
        }
        return need;
    }

    /**
     * Drops one reference of {@code page} in {@code g}. A page freed to 0 goes back to the pool with its rows empty;
     * {@code rowReported} reports that emptiness to the device ({@link PagedSequences#pendingReports()}), skipping
     * rows an operation already emptied.
     */
    void unref(boolean rowReported, int g, int page) {
        int pageSize = owner.pageSize();
        boolean freed = pools()[g].refCount(page) == 1;
        pools()[g].unref(page);
        if (!freed) {
            return;
        }
        int base = page * pageSize;
        int[] rows = new int[pageSize];
        int n = 0;
        for (int r = 0; r < pageSize; r++) {
            if (kvPos()[g][base + r] >= 0) {
                kvPos()[g][base + r] = -1;
                usedRows()[g]--;
                rows[n++] = base + r;
            }
        }
        if (rowReported && n > 0) {
            pendingReports().add(RowReport.clears(g, Arrays.copyOf(rows, n)));
        }
    }

    /** The stored range of {@code seq} from its own rows (a shared page's rows above its cursor are another's). */
    int normalizeRange(int seq, int p0, int p1) {
        if (p0 < 0) {
            throw new IllegalArgumentException("range starts before the sequence: position " + p0);
        }
        if (p1 < -1) {
            throw new IllegalArgumentException("range end " + p1 + " is below -1");
        }
        if (p1 == -1) {
            return owner.posMaxArr()[seq] < 0 ? p0 : owner.posMaxArr()[seq] + 1;
        }
        return Math.max(p0, p1);
    }

    /** Rows of the pool that hold a position, summed over the groups: {@link PagedSequences#freeCells()}. */
    int freeCells() {
        int free = 0;
        for (int g = 0; g < groups(); g++) {
            free += owner.plan().group(g).rows() - usedRows()[g];
        }
        return free;
    }

    private PagePool[] pools() {
        return owner.pools();
    }

    private PageList[][] lists() {
        return owner.lists();
    }

    private int[][] kvPos() {
        return owner.kvPos();
    }

    private int[][] slots() {
        return owner.slots();
    }

    private int[] usedRows() {
        return owner.usedRows();
    }

    private List<PageCopy> pendingCopies() {
        return owner.pendingCopyList();
    }

    private List<RowReport> pendingReports() {
        return owner.pendingReportList();
    }
}
