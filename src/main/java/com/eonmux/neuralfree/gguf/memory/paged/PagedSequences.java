package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The bookkeeping half of the paged {@link SequenceMemory}: page pools per KV group
 * ({@link PagePool}), per-sequence page lists ({@link PageList}) and the copy-on-write discipline for every row some
 * other holder can still read: a page with {@code refCount > 1} is never written in place; every
 * mutation first replaces it in the mutating sequence's list by a private {@link PageCopy} - a write in prepare, the
 * removal of part of a page, an {@code add} re-rotation and the copy over a shared partial page; the SWA trim leaves
 * a shared page's rows alone). Holders outside the sequences keep pages alive: the prefix cache's blocks
 * ({@link #holdPages}) and the windows of context checkpoints ({@link #holdWindow}), which a later sequence adopts.
 *
 * <p><b>Ownership and threading.</b> Mutable, used by the engine thread that runs this memory's steps, one thread at
 * a time (as every {@code ForwardBackend}); not thread-safe. The device side uploads what prepare, the step
 * tables and the row reports address.
 *
 * <p><b>Model.</b> A KV group's rows live in a pool of pages of {@link #pageSize()} rows. Logical slot {@code j} of a
 * sequence maps to physical row {@code pages[j >> log2P] * P + (j & (P-1))} of its group's {@link PageList}. Slots
 * are append ordered: a sequence that was never shifted or trimmed has slot {@code j} = position {@code j}; shifting
 * and trimming remove whole pages from the list (re-basing the slot indices) and empty the partially covered rows on
 * a private page only, so at most two partially holed pages exist per sequence. {@code kv_pos[row]} stays
 * authoritative for the causal, SWA and visibility masks; the host mirrors it and reports every change a step does
 * not carry through {@link #pendingReports()}.
 *
 * <p><b>Reservation.</b> {@link #prepare} is pure: it validates the whole step and picks every page it needs with
 * {@link PagePool#peek} without mutating a pool, and throws {@link MemoryFullException} before any state changed
 * (all-or-nothing reservation). {@link #commit} applies the reservation exactly once per prepare; if the pools
 * changed in between it fails loudly instead of reusing wrong pages.
 */
public final class PagedSequences implements SequenceMemory {

    private final MemoryPlan plan;
    private final int parallel;
    private final int pageSize;
    private final int maxPages;
    private final int cap;
    private final PagePool[] pools;
    private final PageList[][] lists;   // [seq][group]
    private final int[][] kvPos;        // [group][row]: the host mirror, position per row or -1
    private final int[][] slots;        // [group][seq]: slots appended so far (next append slot)
    private final int[] usedRows;       // [group]: rows of the pool that hold a position
    private final int[] posMinArr;
    private final int[] posMaxArr;
    private final SequenceState[] states;
    /** The sliding-window trim: the per-group trim kinds and the page drops below a sequence's window. */
    private final PagedSwaTrim swaTrim;
    private final List<PageCopy> pendingCopies = new ArrayList<>();
    private final List<RowMove> pendingRowMoves = new ArrayList<>();
    private final List<RowReport> pendingReports = new ArrayList<>();
    /** The block-holder and copy operations over this memory's tables. */
    private final PageSharing sharing;
    private final PagedCompaction compaction;

    private int clears;

    /**
     * Builds the bookkeeping over a paged plan: one {@link PagePool} per KV group over its {@code pages} pages of
     * {@code pageSize} rows, {@code parallel} sequences, and a per-sequence context cap of {@code nCtx}
     * ({@code neuralfree.kv.unified}) or {@code nCtx / parallel}.
     *
     * @throws IllegalArgumentException when the plan holds no settings, is not a paged plan (a group without pages)
     *                                  or its groups differ in page size, or {@code parallel} is outside
     *                                  {@code [1, 256]}
     */
    public PagedSequences(MemoryPlan plan, int parallel) {
        this.plan = Objects.requireNonNull(plan, "plan");
        if (plan.config() == null) {
            throw new IllegalArgumentException("PagedSequences needs the plan's settings (a plan with config)");
        }
        if (parallel < 1 || parallel > MemoryConfig.MAX_PARALLEL) {
            throw new IllegalArgumentException("parallel must be in [1, " + MemoryConfig.MAX_PARALLEL + "], got "
                    + parallel);
        }
        this.parallel = parallel;
        MemoryPlan.KvGroup first = plan.group(0);
        if (!first.paged()) {
            throw new IllegalArgumentException("PagedSequences needs a paged plan; group 0 has pageSize 0");
        }
        this.pageSize = first.pageSize();
        for (MemoryPlan.KvGroup g : plan.groups()) {
            if (g.pageSize() != pageSize) {
                throw new IllegalArgumentException("paged groups share one page size, got " + g + " after "
                        + pageSize);
            }
        }
        boolean unified = plan.config().kvUnified();
        this.cap = unified ? plan.nCtx() : Math.max(1, plan.nCtx() / parallel);
        this.maxPages = (cap + pageSize - 1) / pageSize;
        int groups = plan.groups().size();
        this.pools = new PagePool[groups];
        this.lists = new PageList[parallel][groups];
        this.kvPos = new int[groups][];
        this.slots = new int[groups][parallel];
        this.usedRows = new int[groups];
        this.posMinArr = new int[parallel];
        this.posMaxArr = new int[parallel];
        this.states = new SequenceState[parallel];
        for (int g = 0; g < groups; g++) {
            MemoryPlan.KvGroup group = plan.group(g);
            pools[g] = new PagePool(pageSize, group.pages());
            kvPos[g] = new int[group.rows()];
            Arrays.fill(kvPos[g], -1);
            for (int s = 0; s < parallel; s++) {
                lists[s][g] = new PageList(pageSize);
            }
        }
        for (int s = 0; s < parallel; s++) {
            posMinArr[s] = -1;
            posMaxArr[s] = -1;
            // one record entry per stored row: media rows sharing a position can outnumber the positions
            states[s] = new SequenceState(cap, maxPages * pageSize);
        }
        this.swaTrim = new PagedSwaTrim(this, groups);
        this.sharing = new PageSharing(this);
        this.compaction = new PagedCompaction(this);
    }

    /** Rows of a page (the configured {@code neuralfree.kv.page.size}). */
    public int pageSize() {
        return pageSize;
    }

    /** Pages of the largest group's pool; the page ids of every group stay below it. */
    public int pageCount() {
        return plan.maxGroupPages();
    }

    /** Page table entries per sequence per group ({@code ceil(cap / pageSize)}). */
    public int maxPages() {
        return maxPages;
    }

    /** The planned memory. */
    public MemoryPlan plan() {
        return plan;
    }

    int groupsCount() {
        return plan.groups().size();
    }

    /**
     * Steps whose {@link PageCopy}s did not complete on the device: every pending copy holds the reference its
     * sequence had on the source page until the caller observes the executing submission's completion and hands the
     * copy back through {@link #releaseCopies}.
     */
    public List<PageCopy> pendingCopies() {
        return new ArrayList<>(pendingCopies);
    }

    /**
     * Releases the sources of completed page copies: drops the reference each given copy holds on its source page,
     * freeing a page back to the pool at reference count 0 (its rows go empty and are reported through
     * {@link #pendingReports()}).
     *
     * @throws IllegalArgumentException for a null copy
     * @throws IllegalStateException    when a copy is not pending on this memory
     */
    public void releaseCopies(PageCopy... copies) {
        for (PageCopy c : copies) {
            if (c == null) {
                throw new IllegalArgumentException("page copy is null");
            }
            int i = pendingCopies.indexOf(c);
            if (i < 0) {
                throw new IllegalStateException("page copy " + c + " is not pending on this memory");
            }
            pendingCopies.remove(i);
            sharing.unref(true, c.group(), c.srcPage());
        }
    }

    /**
     * The row moves a shifted sequence's compaction queued and the backend has not carried yet
     * ({@link RowMove}): the keys and values of the moved cells follow them. The consumer runs the moves on its
     * layers and empties the queue with {@link #clearRowMoves()}.
     */
    public List<RowMove> pendingRowMoves() {
        return new ArrayList<>(pendingRowMoves);
    }

    /** Consumes the row moves that {@link #pendingRowMoves()} returned. */
    public void clearRowMoves() {
        pendingRowMoves.clear();
    }

    /**
     * Row {@code kv_pos} changes applied on the host besides a step's own writes: the rows a removal or a trim
     * cleared ({@link RowReport#values()} all -1, absolute), a shift moved ({@link RowReport#delta()} carries the
     * RoPE shift, {@code LLM_KV_RopeShift_*}), a division re-positioned ({@link RowReport#rowDelta(int)} carries
     * each row's own shift) or a shared partial copy mirrored ({@link RowReport#sets absolute}).
     * The device applies them between steps; the consumer consumes the list with {@link #clearReports()}.
     */
    public List<RowReport> pendingReports() {
        return new ArrayList<>(pendingReports);
    }

    /** Consumes the reports that {@link #pendingReports()} returned. */
    public void clearReports() {
        pendingReports.clear();
    }

    /**
     * Extends the token record of {@code seq} with text rows a committed step wrote at the positions following the
     * record's last one (the engine's partner of {@link #commit}; the reserved step carries addressing only).
     *
     * @throws IllegalStateException     when the record would overflow the sequence's row capacity
     * @throws IndexOutOfBoundsException when {@code seq} is outside the sequences
     */
    public void appendTokens(int seq, int[] tokens) {
        checkSeq(seq);
        Objects.requireNonNull(tokens, "tokens");
        states[seq].append(tokens, 0, tokens.length);
    }

    /**
     * Extends the token record of {@code seq} with {@code n} rows of a committed step, one entry per row stored at
     * {@code positions[off + i]}: a token row ({@code tokens[off + i] >= 0}) records its id and an embedding row
     * (negative) {@code embeddingToken}, the id its model gives it, as llama.cpp's cells keep {@code ext.tok}; media
     * rows that share one M-RoPE position each keep their own entry.
     *
     * @param embeddingToken the id an embedding row records: {@code -1} for none, or its model's stand-in
     * @throws IllegalArgumentException  when a range is outside its array or a position descends
     * @throws IllegalStateException     when the record would overflow the sequence's row capacity
     * @throws IndexOutOfBoundsException when {@code seq} is outside the sequences
     */
    public void appendRows(int seq, int[] tokens, int[] positions, int off, int n, int embeddingToken) {
        checkSeq(seq);
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(positions, "positions");
        states[seq].appendRows(tokens, positions, off, n, embeddingToken);
    }

    // ------------------------------------------------------------------ prepare / commit

    /**
     * Plans the step over {@code layout} without touching any pool: page ids are picked with {@link PagePool#peek}
     * in the exact order {@link #commit} takes them, so the pooled pages are known before anything is reserved.
     * Every row stores the position {@code segPos0(s) + t} of its segment.
     *
     * @throws IllegalArgumentException when the layout is null or names a sequence twice
     * @throws IllegalStateException    when a step of a sequence does not append in stored order, or the page
     *                                  bookkeeping is out of its invariants
     * @throws MemoryFullException      when the step needs more pages or positions than the pools or the sequence
     *                                  cap hold
     */
    public PagedStep prepare(StepLayout layout) {
        return prepare(layout, null);
    }

    /**
     * {@link #prepare(StepLayout)} with explicit stored positions: {@code rowPos[i]} is the position the row's KV
     * cells take in every group, in step-row order. Rows of a media segment share the segment's base position, so
     * the values are non-decreasing per segment and a segment's first value appends strictly above the sequence's
     * highest stored position (an empty sequence starts at 0). Text steps pass positions equal to
     * {@code segPos0(s) + t} and behave like {@link #prepare(StepLayout)}.
     *
     * @param rowPos stored position of every row, or null for the layout's consecutive positions
     * @throws IllegalArgumentException when the layout is null, {@code rowPos} holds another value than the layout
     *                                  has rows, a value is negative or a segment's values descend
     * @throws IllegalStateException    when a step of a sequence does not append above its stored positions
     * @throws MemoryFullException      when the step needs more pages or positions than the pools or the sequence
     *                                  cap hold
     */
    public PagedStep prepare(StepLayout layout, int[] rowPos) {
        Objects.requireNonNull(layout, "layout");
        int n = layout.nTokens();
        int nSeg = layout.nSeg();
        if (rowPos != null && rowPos.length != n) {
            throw new IllegalArgumentException("row positions hold " + rowPos.length + " values, the layout runs "
                    + n + " rows");
        }
        int[] segSeq = new int[nSeg];
        int[] segStart = new int[nSeg];
        int[] segLen = new int[nSeg];
        int[] segPos0 = new int[nSeg];
        int[] segPosMax = new int[nSeg];
        for (int s = 0; s < nSeg; s++) {
            int seq = layout.seq(s);
            checkSeq(seq);
            for (int k = 0; k < s; k++) {
                if (layout.seq(k) == seq) {
                    throw new IllegalArgumentException("step layout repeats sequence " + seq);
                }
            }
            int len = layout.len(s);
            int p0 = layout.pos0(s);
            segSeq[s] = seq;
            segStart[s] = layout.tokStart(s);
            segLen[s] = len;
            segPos0[s] = p0;
            int pMax = p0 + len - 1;
            if (rowPos != null) {
                pMax = PageChecks.checkRowPos(segStart[s], len, rowPos, seq, posMaxArr[seq]);
            }
            if (posMaxArr[seq] < 0) {
                if (p0 != 0) {
                    throw new IllegalStateException("sequence " + seq + " stores nothing, so its step starts at"
                            + " position 0, got " + p0);
                }
            } else if (p0 < posMaxArr[seq]) {
                // a media segment's rows can share the stored base position, so the step appends at or above it;
                // below it would rewrite committed rows
                throw new IllegalStateException("sequence " + seq + " stores positions up to " + posMaxArr[seq]
                        + ", so its step starts at or above it, got " + p0);
            }
            if (pMax + 1 > cap) {
                throw new MemoryFullException(pMax + 1, cap, n, freeCells());
            }
            if (states[seq].nPast() + len > states[seq].capacity()) {
                // the token record keeps one entry per row; refuse before any page is reserved
                throw new MemoryFullException(states[seq].nPast() + len, states[seq].capacity(), n, freeCells());
            }
            segPosMax[s] = pMax;
        }
        // all-or-nothing: the exact page count of every group, no pool touched yet
        for (int g = 0; g < groupsCount(); g++) {
            if (sharing.needPages(g, segSeq, segLen) > pools[g].free()) {
                throw new MemoryFullException(cap, cap, n, freeCells());
            }
        }
        // build the step; same page order per (group, segment) that commit takes
        PagedStep.Builder b = PagedStep.builder(this, n).segments(segSeq, segStart, segLen, segPos0)
                .segPosMax(segPosMax);
        int[][] kvRows = new int[groupsCount()][n];
        int[][] kvValues = new int[groupsCount()][n];
        int[] kvFill = new int[groupsCount()];
        for (int g = 0; g < groupsCount(); g++) {
            int[] peek = pools[g].peek(sharing.needPages(g, segSeq, segLen));
            int next = 0;
            for (int s = 0; s < nSeg; s++) {
                int seq = segSeq[s];
                int len = segLen[s];
                int slot0 = slots[g][seq];
                int end = slot0 + len;
                PageList list = lists[seq][g];
                int listLen = list.size();
                if (listLen != (slot0 + pageSize - 1) / pageSize && !(listLen == 0 && slot0 == 0)) {
                    throw new IllegalStateException("page list of sequence " + seq + " group " + g + " covers "
                            + listLen + " pages for " + slot0 + " slots (pageSize " + pageSize + ", posMax "
                            + posMaxArr[seq] + ", nPast " + states[seq].nPast() + ")");
                }
                int[] table = new int[Math.max(listLen, (end + pageSize - 1) / pageSize)];
                Arrays.fill(table, -1);
                // the shared pages of the write span get private copies first
                int spanLast = (end - 1) / pageSize;
                for (int k = slot0 / pageSize; k <= spanLast && k < listLen; k++) {
                    int page = list.page(k);
                    if (pools[g].refCount(page) > 1) {
                        int dst = peek[next++];
                        b.copy(PageCopy.of(g, page, dst, ownRows(g, seq, k)));
                        table[k] = dst;
                    }
                }
                // then the new tail pages
                int newPages = (end + pageSize - 1) / pageSize - listLen;
                for (int k = 0; k < newPages; k++) {
                    table[listLen + k] = peek[next++];
                }
                for (int k = 0; k < listLen && table[k] < 0; k++) {
                    table[k] = list.page(k);
                }
                b.pageTableRow(g, seq, table);
                b.seqSlots(g, seq, end);
                for (int t = 0; t < len; t++) {
                    int slot = slot0 + t;
                    int page = table[slot / pageSize];
                    int cell = page * pageSize + (slot & (pageSize - 1));
                    b.cell(g, segStart[s] + t, cell);
                    kvRows[g][kvFill[g]] = cell;
                    kvValues[g][kvFill[g]] = rowPos != null ? rowPos[segStart[s] + t] : segPos0[s] + t;
                    kvFill[g]++;
                }
            }
            // one kv_pos write per group over all segments' rows, in ubatch row order; an explicit rowPos
            // carries the stored positions (a media segment's rows share the chunk's base position)
            b.kvPos(g, Arrays.copyOf(kvRows[g], kvFill[g]), Arrays.copyOf(kvValues[g], kvFill[g]));
        }
        return b.build();
    }


    /**
     * Applies a reserved step: takes its pages in the prepare order, swaps privatized pages into the lists, applies
     * the new {@code kv_pos} values of every group, then trims the SWA pages of the touched sequences. The commit
     * runs before the step's layers, so each sequence trims to the window of its segment's first row: every row of
     * the step still reads its whole window, as llama.cpp frees a SWA cell only once it is masked for the first token
     * of the ubatch ({@code seq_pos_max + 1} before the ubatch, llama-kv-cache.cpp:1104-1110).
     *
     * @throws IllegalStateException when a step of another memory was committed, or the pools changed between
     *                               prepare and commit
     */
    public void commit(PagedStep step) {
        Objects.requireNonNull(step, "step");
        if (step.pageSize() != pageSize || step.maxSequences() != parallel || step.groups() != groupsCount()
                || step.maxPages() != maxPages || step.nPages() != pageCount()) {
            throw new IllegalStateException("the step does not belong to this paged memory");
        }
        int nSeg = step.nSeg();
        for (int g = 0; g < groupsCount(); g++) {
            for (int s = 0; s < nSeg; s++) {
                int seq = step.segSeq(s);
                int len = step.segLen(s);
                int slot0 = slots[g][seq];
                int end = slot0 + len;
                PageList list = lists[seq][g];
                int listLen = list.size();
                int spanLast = (end - 1) / pageSize;
                for (int k = slot0 / pageSize; k <= spanLast && k < listLen; k++) {
                    if (pools[g].refCount(list.page(k)) > 1) {
                        int src = list.page(k);
                        int dst = pools[g].alloc();
                        if (dst != step.pageTableEntry(g, seq, k)) {
                            throw new IllegalStateException("the pages changed between prepare and commit");
                        }
                        mirroredPageRows(g, src, dst, ownRows(g, seq, k));
                        list.set(k, dst);
                    }
                }
                int newPages = (end + pageSize - 1) / pageSize - listLen;
                for (int k = 0; k < newPages; k++) {
                    int page = pools[g].alloc();
                    if (page != step.pageTableEntry(g, seq, listLen + k)) {
                        throw new IllegalStateException("the pages changed between prepare and commit");
                    }
                    list.append(page);
                }
                slots[g][seq] = end;
            }
            int[] rows = step.kvPosRows(g);
            int[] values = step.kvPosValues(g);
            for (int i = 0; i < rows.length; i++) {
                int r = rows[i];
                if (kvPos[g][r] == -1) {
                    usedRows[g]++;
                }
                kvPos[g][r] = values[i];
            }
        }
        // verify the applied tables before the step's page ids reach the device; a table's entries past its
        // length are -1 by construction, so only the stored ones are compared
        for (int g = 0; g < groupsCount(); g++) {
            for (int s = 0; s < nSeg; s++) {
                int seq = step.segSeq(s);
                PageList list = lists[seq][g];
                for (int k = 0; k < list.size(); k++) {
                    if (step.pageTableEntry(g, seq, k) != list.page(k)) {
                        throw new IllegalStateException("page table disagrees with the list of sequence "
                                + seq + " at page " + k + " of group " + g);
                    }
                }
                if (step.pageTableLength(g, seq) > list.size()) {
                    throw new IllegalStateException("page table entry beyond the list of sequence " + seq
                            + " is not empty at page " + list.size() + " of group " + g);
                }
            }
        }
        for (int s = 0; s < nSeg; s++) {
            int seq = step.segSeq(s);
            int end = step.segPosMax(s);
            posMaxArr[seq] = Math.max(posMaxArr[seq], end);
            if (posMinArr[seq] < 0) {
                posMinArr[seq] = step.segPos0(s);
            }
        }
        pendingCopies.addAll(step.pageCopies());
        // the step's queries still read their windows (the trunk runs after commit): its first one bounds the trim
        for (int s = 0; s < nSeg; s++) {
            swaTrim.trimSequence(step.segSeq(s), step.segPos0(s) + 1);
        }
    }

    /**
     * Rows of the page at list index {@code k} that sequence {@code seq} owns in group {@code g}: the slots below
     * its append cursor. Rows at or above the cursor of a shared page belong to the holders it shares the page with.
     */
    private int ownRows(int g, int seq, int k) {
        return Math.min(Math.max(slots[g][seq] - k * pageSize, 0), pageSize);
    }

    /**
     * Moves the sequence's reference of {@code src} onto the fresh private page {@code dst} (the source's reference
     * stays on the pending copy until its completion releases it), mirrors the host {@code kv_pos} of the copy's
     * first {@code rows} rows (the privatizing sequence's own; the rest of {@code dst} stays empty), counting the
     * mirrored live rows into the group's used rows and reporting them, so the device mirror holds the private
     * page's positions before the next step reads it.
     */
    private void mirroredPageRows(int g, int src, int dst, int rows) {
        System.arraycopy(kvPos[g], src * pageSize, kvPos[g], dst * pageSize, rows);
        int base = dst * pageSize;
        int[] rowIds = new int[rows];
        int[] values = new int[rows];
        int n = 0;
        for (int r = 0; r < rows; r++) {
            int pos = kvPos[g][base + r];
            if (pos >= 0) {
                usedRows[g]++;
                rowIds[n] = base + r;
                values[n] = pos;
                n++;
            }
        }
        if (n > 0) {
            pendingReports.add(RowReport.sets(g, Arrays.copyOf(rowIds, n), Arrays.copyOf(values, n)));
        }
    }

    // ------------------------------------------------------------------ copy-on-write mutation paths

    /**
     * Invariant 7's private helper: when the page of {@code (seq, group, listIndex)} has {@code refCount > 1}, it
     * allocates a page, emits a {@link PageCopy} of the rows the sequence owns on it ({@link #ownRows}), swaps the
     * page in the sequence's list and moves the sequence's reference on the source to the pending copy. The caller
     * mutates the private copy only. A page with {@code refCount == 1} is left alone and mutated in place.
     *
     * @throws MemoryFullException when the pools hold no free page for the private copy ({@code prepare} and
     *                             {@code copy} pre-count theirs, so only a caller that skipped that count lands
     *                             here)
     */
    void privatize(int g, int seq, int listIndex) {
        PageList list = lists[seq][g];
        int src = list.page(listIndex);
        if (pools[g].refCount(src) == 1) {
            return;
        }
        int dst = pools[g].alloc();
        if (dst < 0) {
            throw new MemoryFullException(pageSize, 0, pageSize, 0);
        }
        int own = Math.max(ownRows(g, seq, listIndex), 1);
        PageCopy copy = PageCopy.of(g, src, dst, own);
        mirroredPageRows(g, src, dst, own);
        pendingCopies.add(copy);
        list.set(listIndex, dst);
    }

    @Override
    public String removeRefusal(int seq, int p0, int p1) {
        checkSeq(seq);
        if (p0 < 0) {
            return "remove starts before the sequence: position " + p0;
        }
        if (p1 < -1) {
            return "remove range ends before -1: " + p1;
        }
        return null;
    }

    @Override
    public void remove(int seq, int p0, int p1) {
        String refusal = removeRefusal(seq, p0, p1);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        int lo = p0;
        int hi = sharing.normalizeRange(seq, p0, p1);
        int max = posMaxArr[seq];
        if (max < 0 || lo >= hi || lo > max) {
            return;
        }
        dropRowMoves(seq); // the removal frees the rows the queued moves would read
        hi = Math.min(hi, max + 1);
        boolean tail = hi == max + 1;
        // the kept rows' position range, gathered by the removal's own walk over every page
        int[] kept = {-1, -1};
        for (int g = 0; g < groupsCount(); g++) {
            removeGroup(g, seq, lo, hi, kept);
            if (tail) {
                // the cursor follows the live rows: a row move (a verify commit's compaction) unmaps its source
                // before the removal runs, so the range can hold no live row at all and the dead slots it leaves
                // must not hold the next append above them. The fresh slots then start at the first slot past the
                // live rows, where the stored positions equal the slots, as the paged attention's fresh-window
                // rule (a fresh row indexes its tree mask bits by kv_pos - (kv_end - n_fresh)) requires. An
                // interior removal keeps the cursor: live rows sit above its holes.
                slots[g][seq] = highestLiveSlot(g, seq) + 1;
            }
            if (lists[seq][g].isEmpty()) {
                // an emptied list holds no rows: the counter drops with it (the armed-but-unfed token's slot
                // never held a row, so the live-row sum can leave a residue an admission would trip over)
                slots[g][seq] = 0;
            }
        }
        states[seq].removePositions(lo, hi);
        posMinArr[seq] = kept[0];
        posMaxArr[seq] = kept[1];
    }

    /**
     * The paged rows' removal never rewrites the per-sequence recurrent state rows ({@code remove} keeps them, the
     * tree verify's replay re-derives them), so the keep-recurrent drop runs the plain suffix removal.
     */
    @Override
    public String removeKeepRecurrentRefusal(int seq, int p0) {
        return removeRefusal(seq, p0, -1);
    }

    @Override
    public void removeKeepRecurrent(int seq, int p0) {
        remove(seq, p0, -1);
    }

    /**
     * The highest slot of {@code seq} in KV group {@code g} whose cell holds a row (its kv_pos is set), or -1 when
     * every own slot is dead. The walk runs from the back over the sequence's own rows only: rows of a shared page
     * above the cursor belong to another holder.
     */
    private int highestLiveSlot(int g, int seq) {
        PageList list = lists[seq][g];
        int ownSlots = Math.min(slots[g][seq], list.size() * pageSize);
        for (int k = list.size() - 1; k >= 0; k--) {
            int base = list.page(k) * pageSize;
            int ownRows = Math.min(Math.max(ownSlots - k * pageSize, 0), pageSize);
            for (int r = ownRows - 1; r >= 0; r--) {
                if (kvPos[g][base + r] >= 0) {
                    return k * pageSize + r;
                }
            }
        }
        return -1;
    }

    /**
     * Removes one group's rows in {@code [lo, hi)}. The walk runs from the back so every whole page in the range
     * leaves through {@code removeBack}: a page that is wholly in the range always sits above the pages still to
     * visit, so it is the last entry when the walk reaches it and no interior tombstone can ever stay behind in
     * the list. A shared page counts only the rows of the removing sequence's own slots: rows above its cursor
     * belong to the holder it shares the page with and stay untouched. The positions of the live own rows the
     * removal keeps widen {@code kept} ({@code {min, max}}, -1 while empty): they are the sequence's stored range
     * afterwards, since a page that leaves keeps none and the cursor moves only over dead rows.
     */
    private void removeGroup(int g, int seq, int lo, int hi, int[] kept) {
        PageList list = lists[seq][g];
        int ownSlots = Math.min(slots[g][seq], list.size() * pageSize);
        for (int k = list.size() - 1; k >= 0; k--) {
            int page = list.page(k);
            int base = page * pageSize;
            int ownRows = Math.min(Math.max(ownSlots - k * pageSize, 0), pageSize);
            int live = 0;
            int liveIn = 0;
            for (int r = 0; r < ownRows; r++) {
                int pos = kvPos[g][base + r];
                if (pos < 0) {
                    continue;
                }
                live++;
                if (pos >= lo && pos < hi) {
                    liveIn++;
                } else {
                    kept[0] = kept[0] < 0 ? pos : Math.min(kept[0], pos);
                    kept[1] = Math.max(kept[1], pos);
                }
            }
            if (live == 0 || liveIn == live) {
                // the whole page leaves the sequence
                if (k == list.size() - 1) {
                    // every own cell of the back page ends dead, so the cursor pulls to the page's logical
                    // start; a page whose own cells lie past the cursor leaves it alone
                    slots[g][seq] = Math.min(slots[g][seq], k * pageSize);
                    list.removeBack(1);
                    sharing.unref(true, g, page);
                    continue;
                }
                if (k == 0) {
                    list.removeFront(1);
                    slots[g][seq] = Math.max(0, slots[g][seq] - pageSize);
                    sharing.unref(true, g, page);
                    continue;
                }
                // defensive middle case (a contiguous range cannot reach it): the rows empty and the page stays
                // HELD in its list slot, so the slot addressing after it stays put and the freed id can never be
                // reallocated under a live entry of this list
                if (pools[g].refCount(page) > 0) {
                    privatize(g, seq, k);
                    clearRowsIn(g, lists[seq][g].page(k) * pageSize, Integer.MIN_VALUE, Integer.MAX_VALUE);
                }
                continue;
            }
            if (liveIn > 0) {
                privatize(g, seq, k);
                clearRowsIn(g, lists[seq][g].page(k) * pageSize, lo, hi);
            }
        }
    }

    /** Sets the rows of {@code base} whose position is selected empty and reports any position that was dropped. */
    void clearRowsIn(int g, int base, int lo, int hi) {
        int[] rows = new int[pageSize];
        int n = 0;
        for (int r = 0; r < pageSize; r++) {
            int pos = kvPos[g][base + r];
            if (pos >= lo && pos < hi) {
                if (pos >= 0) {
                    usedRows[g]--;
                    rows[n++] = base + r;
                }
                kvPos[g][base + r] = -1;
            }
        }
        if (n > 0) {
            pendingReports.add(RowReport.clears(g, Arrays.copyOf(rows, n)));
        }
    }

    @Override
    public String moveRowsRefusal(int seq, int[] srcPos, int[] dstPos) {
        String structural = checkMoveRows(srcPos, dstPos);
        if (structural != null) {
            return structural;
        }
        checkSeq(seq);
        return null;
    }

    @Override
    public void moveRows(int seq, int[] srcPos, int[] dstPos) {
        String refusal = moveRowsRefusal(seq, srcPos, dstPos);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        for (int g = 0; g < groupsCount(); g++) {
            moveGroupRows(g, seq, srcPos, dstPos);
        }
    }

    /** The ordering contract of a compaction: equal lengths, sources above their destinations, both ascending. */
    private static String checkMoveRows(int[] srcPos, int[] dstPos) {
        if (srcPos == null || dstPos == null) {
            throw new IllegalArgumentException("a row compaction names sources and destinations");
        }
        if (srcPos.length != dstPos.length) {
            return "the compaction names " + srcPos.length + " sources and " + dstPos.length + " destinations";
        }
        for (int i = 0; i < srcPos.length; i++) {
            if (srcPos[i] <= dstPos[i]) {
                return "the compaction moves position " + srcPos[i] + " down onto " + dstPos[i] + ", which is not"
                        + " below its source";
            }
            if (i > 0 && (srcPos[i] <= srcPos[i - 1] || dstPos[i] <= dstPos[i - 1])) {
                return "the compaction's positions must ascend strictly, got " + srcPos[i] + "/" + dstPos[i]
                        + " after " + srcPos[i - 1] + "/" + dstPos[i - 1];
            }
        }
        return null;
    }

    /**
     * One group's gather: maps every position to its slot, privatizes the destination pages (a shared page is
     * never written in place), then moves the cells and mirrors them. The queue the moves join runs in order, so a
     * move that overwrites an earlier move's source lands after it.
     */
    private void moveGroupRows(int g, int seq, int[] srcPos, int[] dstPos) {
        PageList list = lists[seq][g];
        int cursor = slots[g][seq];
        int[] srcSlot = new int[srcPos.length];
        int[] dstSlot = new int[dstPos.length];
        for (int s = 0; s < cursor; s++) {
            int pos = kvPos[g][cellOf(list, s)];
            for (int i = 0; i < srcPos.length; i++) {
                if (pos == srcPos[i]) {
                    srcSlot[i] = s + 1;
                }
                if (pos == dstPos[i]) {
                    dstSlot[i] = s + 1;
                }
            }
        }
        for (int i = 0; i < srcPos.length; i++) {
            if (srcSlot[i] == 0) {
                throw new UnsupportedOperationException("the compaction moves position " + srcPos[i]
                        + ", which the sequence holds no row for");
            }
            if (dstSlot[i] == 0) {
                throw new UnsupportedOperationException("the compaction's destination position " + dstPos[i]
                        + " holds no row to overwrite");
            }
        }
        // the destination pages privatize before any cell moves: a page replaced itself in the list, so the cells
        // re-address afterwards
        int lastListIndex = -1;
        for (int i = 0; i < dstSlot.length; i++) {
            int listIndex = dstSlot[i] - 1;
            int k = listIndex / pageSize;
            if (k != lastListIndex) {
                privatize(g, seq, k);
                lastListIndex = k;
            }
        }
        list = lists[seq][g];
        // a cell can be one move's source and a later move's destination: the reports name every touched cell by
        // its final state
        int[] touched = new int[2 * srcPos.length];
        int moved = 0;
        for (int i = 0; i < srcPos.length; i++) {
            int src = cellOf(list, srcSlot[i] - 1);
            int dst = cellOf(list, dstSlot[i] - 1);
            if (src == dst) {
                continue;
            }
            if (kvPos[g][dst] < 0) {
                usedRows[g]++;
            }
            kvPos[g][src] = -1;
            kvPos[g][dst] = dstPos[i];
            pendingRowMoves.add(RowMove.of(seq, g, src, dst, 1));
            touched[moved++] = src;
            touched[moved++] = dst;
        }
        PagedCompaction.reportFinal(pendingReports, g, kvPos[g], touched, moved);
    }

    @Override
    public String copyRefusal() {
        return null;
    }

    /** Whether any page of {@code seq} has a reference count over 1 (another holder keeps it too). */
    @Override
    public boolean sharesPages(int seq) {
        checkSeq(seq);
        for (int g = 0; g < groupsCount(); g++) {
            PageList list = lists[seq][g];
            for (int k = 0; k < list.size(); k++) {
                if (pools[g].refCount(list.page(k)) > 1) {
                    return true;
                }
            }
        }
        return false;
    }


    @Override
    public void keep(int seq) {
        checkSeq(seq);
        for (int s = 0; s < parallel; s++) {
            if (s != seq) {
                dropSequence(s);
            }
        }
    }

    @Override
    public void clear() {
        for (int s = 0; s < parallel; s++) {
            dropSequence(s);
        }
        clears++;
    }

    /**
     * How many times {@link #clear()} emptied the memory. A clear comes with a reset of the memory's row mirrors,
     * so a holder of pages outside the sequences ({@link #holdPages}) compares the count to learn that the rows of
     * its pages no longer serve attention.
     */
    public int clears() {
        return clears;
    }

    // ------------------------------------------------------------------ block holders and copies (PageSharing)

    /** KV groups of the memory: a block's pages hold one page per group. */
    public int groups() {
        return sharing.groups();
    }

    /** The fewest idle pages of any KV group: 0 when some group could not allocate a page now. */
    public int minFreePages() {
        return sharing.minFreePages();
    }

    /** The pages of block {@code block} of {@code seq}, one per KV group; see {@link PageSharing#blockPages}. */
    public int[] blockPages(int seq, int block, int rows) {
        return sharing.blockPages(seq, block, rows);
    }

    /** {@link #blockPages(int, int, int)} for rows that store explicit positions. */
    public int[] blockPages(int seq, int block, int rows, int[] positions) {
        return sharing.blockPages(seq, block, rows, positions);
    }

    /** Takes a holder's reference on each page of {@code pages}; see {@link PageSharing#holdPages}. */
    public void holdPages(int[] pages) {
        sharing.holdPages(pages);
    }

    /** Takes one idle page per KV group for rows imported from the host. */
    public int[] importPages(int[] positions) {
        return sharing.importPages(positions);
    }

    /** The positions page rows {@code [0, rows)} of {@code page} store in KV group {@code group}. */
    public int[] pagePositions(int group, int page, int rows) {
        return sharing.pagePositions(group, page, rows);
    }

    /** Drops a holder's reference on each page of {@code pages}. */
    public void releasePages(int[] pages) {
        sharing.releasePages(pages);
    }

    /** Whether a page of {@code pages} has more than one reference. */
    public boolean pagesShared(int[] pages) {
        return sharing.pagesShared(pages);
    }

    /** Idle pages of KV group {@code g}. */
    public int freePages(int g) {
        return sharing.freePages(g);
    }

    /** The pages a write of {@code rows} rows by {@code seq} takes in KV group {@code g}. */
    public int pagesToWrite(int seq, int g, int rows) {
        return sharing.pagesToWrite(seq, g, rows);
    }

    /** Pages of {@code window} that only its holder references. */
    public int windowOnlyPages(PageWindow window) {
        return sharing.windowOnlyPages(window);
    }

    /** Whether KV group {@code g} trims a sliding window. */
    public boolean trims(int g) {
        return sharing.trims(g);
    }

    /** The widest reader window of a trimming group {@code g}; 0 for a full-attention group. */
    public int windowWidth(int g) {
        return sharing.windowWidth(g);
    }

    /** Takes a holder's reference on the window of sequence {@code seq} in group {@code g} at {@code rows} rows. */
    public PageWindow holdWindow(int seq, int g, int rows) {
        return sharing.holdWindow(seq, g, rows);
    }

    /** Drops a holder's reference on every page of {@code window}. */
    public void releaseWindow(PageWindow window) {
        sharing.releaseWindow(window);
    }

    /** Gives the empty sequence {@code seq} the held pages of consecutive blocks as its positions from 0. */
    public void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens) {
        sharing.adoptPages(seq, blocks, lastRows, tokens);
    }

    /** {@link #adoptPages(int, int[][], int, int[])} for a model whose embedding rows record no token. */
    public void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens, int[] positions) {
        sharing.adoptPages(seq, blocks, lastRows, tokens, positions);
    }

    /** {@link #adoptPages(int, int[][], int, int[])} for rows that store explicit positions. */
    public void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens, int[] positions,
                           int embeddingToken) {
        sharing.adoptPages(seq, blocks, lastRows, tokens, positions, embeddingToken);
    }

    /** {@link #adoptPages(int, int[][], int, int[], int[], int)} for a memory with trimming groups. */
    public void adoptPages(int seq, int[][] blocks, int lastRows, int[] tokens, int[] positions,
                           PageWindow[] windows, int embeddingToken) {
        sharing.adoptPages(seq, blocks, lastRows, tokens, positions, windows, embeddingToken);
    }

    @Override
    public void copy(int src, int dst, int p0, int p1) {
        sharing.copy(src, dst, p0, p1);
    }

    /**
     * Re-appends the held rows of cached pages at the append cursor of sequence {@code seq} (llama.cpp server's KV
     * chunk reuse): the run of {@code rows} rows starts at source row {@code srcRow0} of run page 0 of
     * {@code srcPages} (one page per KV group per run page, the cache blocks') and lands at the sequence's own
     * slots, at the positions {@code [pos0, pos0 + rows)} its prompt rows take — the sequence's next position.
     * Copied rows re-rotate by the uniform {@code delta} between a source row's stored position and its new one
     * ({@link RowReport#shift}, the re-rotation {@link #pendingReports()} names); a {@code delta} 0 run whose rows
     * meet the cursor page-aligned shares whole cached pages instead of copying them. The token record takes one
     * entry per row ({@code tokens[r]}, an embedding row negative, recording {@code embeddingToken}).
     * All-or-nothing: the new tail pages and the private copies of every written shared page are counted per group
     * and refused as a whole before any pool changes.
     *
     * @throws IllegalArgumentException      for a run outside the pages, a token count or position mismatch, or
     *                                       source rows the cache does not hold
     * @throws IllegalStateException         when the sequence's groups disagree on the cursor or a group's page list
     *                                       does not cover its cursor
     * @throws MemoryFullException           when the run needs pages or rows the pools or the sequence lack
     * @throws UnsupportedOperationException for a trimming group (chunk reuse needs a full-attention memory)
     */
    public void reusePages(int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                           int[] tokens, int embeddingToken) {
        sharing.reusePages(seq, srcPages, srcRow0, pos0, rows, delta, tokens, embeddingToken);
    }

    // ------------------------------------------------------------------ PageSharing accessors

    PageSharing sharing() {
        return sharing;
    }

    PagePool[] pools() {
        return pools;
    }

    PageList[][] lists() {
        return lists;
    }

    int[][] kvPos() {
        return kvPos;
    }

    int[][] slots() {
        return slots;
    }

    int[] usedRows() {
        return usedRows;
    }

    int[] posMinArr() {
        return posMinArr;
    }

    int[] posMaxArr() {
        return posMaxArr;
    }

    boolean[] swaGroups() {
        return swaTrim.swaGroups();
    }

    int[] windowWidths() {
        return swaTrim.windowWidths();
    }

    int cap() {
        return cap;
    }

    List<PageCopy> pendingCopyList() {
        return pendingCopies;
    }

    List<RowReport> pendingReportList() {
        return pendingReports;
    }

    List<RowMove> pendingRowMoveList() {
        return pendingRowMoves;
    }

    private void dropSequence(int seq) {
        dropRowMoves(seq);
        for (int g = 0; g < groupsCount(); g++) {
            PageList list = lists[seq][g];
            while (!list.isEmpty()) {
                sharing.unref(true, g, list.page(list.size() - 1));
                list.removeBack(1);
            }
            slots[g][seq] = 0;
            recountUsed(g);
        }
        states[seq].clear();
        posMinArr[seq] = -1;
        posMaxArr[seq] = -1;
    }

    /** Drops the queued row moves of {@code seq}: its rows leave, so their moves would read freed cells. */
    private void dropRowMoves(int seq) {
        pendingRowMoves.removeIf(m -> m.seq() == seq);
    }

    @Override
    public String shiftRefusal() {
        return null;
    }

    @Override
    public String divRefusal() {
        return null;
    }

    @Override
    public void div(int seq, int p0, int p1, int d) {
        checkSeq(seq);
        if (d < 2) {
            throw new IllegalArgumentException("position division needs a divisor of at least 2, got " + d);
        }
        int lo = p0;
        if (lo < 0) {
            throw new IllegalArgumentException("division starts before the sequence: position " + lo);
        }
        int hi = sharing.normalizeRange(seq, p0, p1);
        if (hi <= lo) {
            return;
        }
        for (int g = 0; g < groupsCount(); g++) {
            divGroup(g, seq, lo, hi, d);
        }
        states[seq].divPositions(lo, hi, d);
        recomputePos(seq);
    }

    /**
     * Divides one group's row positions in {@code [lo, hi)}: every page with a row in the range privatizes first
     * (a shared page is never written in place), then each of its rows takes {@code pos / d} and reports the
     * per-row RoPE shift {@code pos - pos / d} the backend re-rotates its key by. Rows keep their cells, so the
     * slot cursor and the page lists do not move.
     */
    private void divGroup(int g, int seq, int lo, int hi, int d) {
        PageList list = lists[seq][g];
        for (int k = 0; k < list.size(); k++) {
            int base = list.page(k) * pageSize;
            boolean mutates = false;
            for (int r = 0; r < pageSize && !mutates; r++) {
                int pos = kvPos[g][base + r];
                mutates = pos >= lo && pos < hi;
            }
            if (!mutates) {
                continue;
            }
            privatize(g, seq, k);
            int privBase = lists[seq][g].page(k) * pageSize;
            int[] rows = new int[pageSize];
            int[] values = new int[pageSize];
            int[] deltas = new int[pageSize];
            int n = 0;
            for (int r = 0; r < pageSize; r++) {
                int pos = kvPos[g][privBase + r];
                if (pos >= lo && pos < hi) {
                    int divided = pos / d;
                    kvPos[g][privBase + r] = divided;
                    rows[n] = privBase + r;
                    values[n] = divided;
                    deltas[n] = pos - divided;
                    n++;
                }
            }
            if (n > 0) {
                pendingReports.add(RowReport.reposition(g, Arrays.copyOf(rows, n), Arrays.copyOf(values, n),
                        Arrays.copyOf(deltas, n)));
            }
        }
    }

    @Override
    public void add(int seq, int p0, int p1, int delta) {
        checkSeq(seq);
        if (delta == 0) {
            return;
        }
        int lo = p0;
        int hi = sharing.normalizeRange(seq, p0, p1);
        if (lo < 0) {
            throw new IllegalArgumentException("shift starts before the sequence: position " + lo);
        }
        if (hi <= lo) {
            return;
        }
        for (int g = 0; g < groupsCount(); g++) {
            addGroup(g, seq, lo, hi, delta);
        }
        states[seq].shiftPositions(lo, hi, delta);
        recomputePos(seq);
        if (delta < 0) {
            compact(seq);
        }
    }

    /**
     * Slides the sequence's live rows down over the dead slots below them ({@link PagedCompaction}): a context
     * shift's interior removal strands the removed slots under the live tail, and an append cursor that keeps them
     * would grow every later append past the pages the plan sized. The rows' positions and the token record stay;
     * the cells move, their {@code kv_pos} mirrors and the device reports with them, and the backend carries the
     * keys and values through the queued {@link RowMove}s.
     */
    private void compact(int seq) {
        for (int g = 0; g < groupsCount(); g++) {
            compaction.compactGroup(g, seq);
        }
    }

    /** The cell of slot {@code s} of the list (page table entry {@code s / pageSize}, row {@code s % pageSize}). */
    int cellOf(PageList list, int s) {
        return list.page(s / pageSize) * pageSize + (s % pageSize);
    }

    private void addGroup(int g, int seq, int lo, int hi, int delta) {
        PageList list = lists[seq][g];
        for (int k = 0; k < list.size(); k++) {
            int page = list.page(k);
            int base = page * pageSize;
            boolean mutates = false;
            for (int r = 0; r < pageSize && !mutates; r++) {
                int pos = kvPos[g][base + r];
                mutates = pos >= lo && pos < hi;
            }
            if (!mutates) {
                continue;
            }
            privatize(g, seq, k);
            int privBase = lists[seq][g].page(k) * pageSize;
            int[] rows = new int[pageSize];
            int[] values = new int[pageSize];
            int n = 0;
            for (int r = 0; r < pageSize; r++) {
                int pos = kvPos[g][privBase + r];
                if (pos >= lo && pos < hi) {
                    pos += delta;
                    kvPos[g][privBase + r] = pos;
                    rows[n] = privBase + r;
                    values[n] = pos;
                    n++;
                }
            }
            if (n > 0) {
                pendingReports.add(RowReport.shift(g, Arrays.copyOf(rows, n), Arrays.copyOf(values, n), delta));
            }
        }
    }

    // ------------------------------------------------------------------ SWA trimming

    /**
     * Trims the SWA pages of {@code seq}: whole pages whose rows all lie below the window leave the sequence, and
     * the leading page's rows below the window go empty when the sequence alone holds the page. A page another
     * holder shares (a context checkpoint's window, a fork) is never written: a whole one leaves the
     * sequence by dropping its reference, a straddling one keeps its rows below the window, which the SWA mask
     * hides from every query of the sequence. So a trim never allocates a page. Committed is the sequence's
     * committed element count, so trimming follows {@code committed - W_g} with W_g the group's widest reader
     * window: the window of the sequence's next position.
     */
    public void trim(int seq) {
        checkSeq(seq);
        swaTrim.trim(seq);
    }

    /**
     * Ends a committed step once the trunk ran over it: its queries read no more, so each of its sequences trims
     * to the window of its next position ({@link #trim}) and the window rows below it leave before the next step
     * reserves pages.
     */
    public void finish(PagedStep step) {
        Objects.requireNonNull(step, "step");
        swaTrim.finish(step);
    }

    // ------------------------------------------------------------------ pool helpers


    private void recountUsed(int g) {
        usedRows[g] = 0;
        for (int p = 0; p < pools[g].pages(); p++) {
            if (pools[g].isFree(p)) {
                continue;
            }
            int base = p * pageSize;
            for (int r = 0; r < pageSize; r++) {
                if (kvPos[g][base + r] >= 0) {
                    usedRows[g]++;
                }
            }
        }
    }

    /** The stored range of {@code seq} from its own rows (a shared page's rows above its cursor are another's). */
    private void recomputePos(int seq) {
        int[] range = scannedRange(seq);
        posMinArr[seq] = range[0];
        posMaxArr[seq] = range[1];
    }

    /**
     * {@code {min, max}} of the positions {@code seq}'s own live rows hold over every group, by a walk over all of
     * its pages ({@code {-1, -1}} when it holds none).
     */
    int[] scannedRange(int seq) {
        int min = -1;
        int max = -1;
        for (int g = 0; g < groupsCount(); g++) {
            PageList list = lists[seq][g];
            for (int k = 0; k < list.size(); k++) {
                int base = list.page(k) * pageSize;
                int own = ownRows(g, seq, k);
                for (int r = 0; r < own; r++) {
                    int pos = kvPos[g][base + r];
                    if (pos < 0) {
                        continue;
                    }
                    min = min < 0 ? pos : Math.min(min, pos);
                    max = Math.max(max, pos);
                }
            }
        }
        return new int[] {min, max};
    }

    // ------------------------------------------------------------------ SequenceMemory

    @Override
    public int maxSequences() {
        return parallel;
    }

    @Override
    public SequenceState sequence(int seq) {
        checkSeq(seq);
        return states[seq];
    }

    @Override
    public int posMin(int seq) {
        checkSeq(seq);
        return posMinArr[seq];
    }

    @Override
    public int posMax(int seq) {
        checkSeq(seq);
        return posMaxArr[seq];
    }

    @Override
    public int freeCells() {
        return sharing.freeCells();
    }

    @Override
    public int totalCells() {
        int total = 0;
        for (int g = 0; g < groupsCount(); g++) {
            total += plan.group(g).rows();
        }
        return total;
    }

    void checkSeq(int seq) {
        if (seq < 0 || seq >= parallel) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + parallel + ")");
        }
    }

    @Override
    public String toString() {
        return "PagedSequences{parallel=" + parallel + ", pageSize=" + pageSize + ", groups=" + groupsCount()
                + ", " + freeCells() + "/" + totalCells() + " free rows}";
    }
}
