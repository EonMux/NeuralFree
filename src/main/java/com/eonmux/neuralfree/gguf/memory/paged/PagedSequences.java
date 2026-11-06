package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
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
 * The bookkeeping half of the paged {@link SequenceMemory} (PARITY MS-2): page pools per KV group
 * ({@link PagePool}), per-sequence page lists ({@link PageList}) and the copy-on-write discipline for every row some
 * other holder can still read (invariant 7: a page with {@code refCount > 1} is never written in place; every
 * mutation first replaces it in the mutating sequence's list by a private {@link PageCopy} - a write in prepare, the
 * removal of part of a page, an {@code add} re-rotation, the SWA trim of a page and the copy over a shared partial
 * page; the recurrent-slot and checkpoint restores of later work packages go through the same helpers).
 *
 * <p><b>Ownership and threading.</b> Mutable, used by the engine thread that runs this memory's steps, one thread at
 * a time (as every {@code ForwardBackend}); not thread-safe. The device side (MS-3) uploads what prepare, the step
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
 * (all-or-nothing reservation, T-4). {@link #commit} applies the reservation exactly once per prepare; if the pools
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
    private final boolean[] swaGroup;   // [group]: groups whose sequences trim (the widest reader window, M-2)
    private final int[] windowWidth;    // [group]: W_g of a trimming group
    private final List<PageCopy> pendingCopies = new ArrayList<>();
    private final List<RowReport> pendingReports = new ArrayList<>();

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
        this.swaGroup = new boolean[groups];
        this.windowWidth = new int[groups];
        for (int g = 0; g < groups; g++) {
            MemoryPlan.KvGroup group = plan.group(g);
            pools[g] = new PagePool(pageSize, group.pages());
            kvPos[g] = new int[group.rows()];
            Arrays.fill(kvPos[g], -1);
            for (int s = 0; s < parallel; s++) {
                lists[s][g] = new PageList(pageSize);
            }
            describeGroupWindow(g);
        }
        for (int s = 0; s < parallel; s++) {
            posMinArr[s] = -1;
            posMaxArr[s] = -1;
            states[s] = new SequenceState(cap);
        }
    }

    /**
     * The group's trim kind (M-2 readers): the widest window over every layer reading the group, with differing
     * window types or any window-less reader making it a full-attention group. Only windowed (same-type) groups trim.
     */
    private void describeGroupWindow(int g) {
        swaGroup[g] = false;
        windowWidth[g] = 0;
        SwaType type = null;
        boolean sameWindowType = true;
        int width = 0;
        for (LayerMemoryPlan l : plan.layers()) {
            if (l.kvGroup() != g || l.kv() == null) {
                continue;
            }
            KvWindow w = l.kv().window();
            if (w.type() == SwaType.NONE) {
                sameWindowType = false;
                break;
            }
            if (type == null) {
                type = w.type();
            } else if (type != w.type()) {
                sameWindowType = false;
                break;
            }
            width = Math.max(width, w.nSwa());
        }
        if (sameWindowType && type != null) {
            swaGroup[g] = true;
            windowWidth[g] = width;
        }
    }

    /** Rows of a page (the configured {@code neuralfree.kv.page.size}). */
    public int pageSize() {
        return pageSize;
    }

    /** Pages of each group's pool. */
    public int pageCount() {
        return plan.groups().isEmpty() ? 0 : plan.group(0).pages();
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
            unref(true, c.group(), c.srcPage());
        }
    }

    /**
     * Row {@code kv_pos} changes applied on the host besides a step's own writes: the rows a removal or a trim
     * cleared ({@link RowReport#values()} all -1, absolute), a shift moved ({@link RowReport#delta()} carries the
     * RoPE shift, {@code LLM_KV_RopeShift_*}) or a shared partial copy mirrored ({@link RowReport#sets absolute}).
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
     * Updates the token record of {@code seq} with the tokens whose rows a committed step wrote (the engine's
     * partner of {@link #commit}; the reserved step carries addressing only).
     *
     * @throws IllegalStateException     when the record would overflow the sequence's context cap
     * @throws IndexOutOfBoundsException when {@code seq} is outside the sequences
     */
    public void appendTokens(int seq, int[] tokens) {
        checkSeq(seq);
        Objects.requireNonNull(tokens, "tokens");
        states[seq].append(tokens, 0, tokens.length);
    }

    // ------------------------------------------------------------------ prepare / commit

    /**
     * Plans the step over {@code layout} without touching any pool: page ids are picked with {@link PagePool#peek}
     * in the exact order {@link #commit} takes them, so the pooled pages are known before anything is reserved.
     *
     * @throws IllegalArgumentException when the layout is null or names a sequence twice
     * @throws IllegalStateException    when a step of a sequence does not append in stored order, or the page
     *                                  bookkeeping is out of its invariants
     * @throws MemoryFullException      when the step needs more pages or positions than the pools or the sequence
     *                                  cap hold
     */
    public PagedStep prepare(StepLayout layout) {
        Objects.requireNonNull(layout, "layout");
        int n = layout.nTokens();
        int nSeg = layout.nSeg();
        int[] segSeq = new int[nSeg];
        int[] segStart = new int[nSeg];
        int[] segLen = new int[nSeg];
        int[] segPos0 = new int[nSeg];
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
            if (posMaxArr[seq] < 0) {
                if (p0 != 0) {
                    throw new IllegalStateException("sequence " + seq + " stores nothing, so its step starts at"
                            + " position 0, got " + p0);
                }
            } else if (p0 != posMaxArr[seq] + 1) {
                throw new IllegalStateException("sequence " + seq + " stores positions from " + posMinArr[seq]
                        + " to " + posMaxArr[seq] + ", so its step starts at " + (posMaxArr[seq] + 1)
                        + ", got " + p0);
            }
            if ((long) p0 + len > cap) {
                throw new MemoryFullException(p0 + len, cap, n, freeCells());
            }
        }
        // all-or-nothing: the exact page count of every group, no pool touched yet
        for (int g = 0; g < groupsCount(); g++) {
            if (needPages(g, segSeq, segLen) > pools[g].free()) {
                throw new MemoryFullException(cap, cap, n, freeCells());
            }
        }
        // build the step; same page order per (group, segment) that commit takes
        PagedStep.Builder b = PagedStep.builder(this, n).segments(segSeq, segStart, segLen, segPos0);
        int[][] cells = new int[groupsCount()][n];
        for (int g = 0; g < groupsCount(); g++) {
            int[] peek = pools[g].peek(needPages(g, segSeq, segLen));
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
                            + listLen + " pages for " + slot0 + " slots");
                }
                int[] table = new int[maxPages];
                Arrays.fill(table, -1);
                // the shared pages of the write span get private copies first
                int spanLast = (end - 1) / pageSize;
                for (int k = slot0 / pageSize; k <= spanLast && k < listLen; k++) {
                    int page = list.page(k);
                    if (pools[g].refCount(page) > 1) {
                        int dst = peek[next++];
                        b.copy(PageCopy.of(g, page, dst, pageSize));
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
                for (int k = 0; k < maxPages; k++) {
                    if (table[k] >= 0) {
                        b.pageTable(g, seq, k, table[k]);
                    }
                }
                b.seqSlots(g, seq, end);
                for (int t = 0; t < len; t++) {
                    int slot = slot0 + t;
                    int page = table[slot / pageSize];
                    int cell = page * pageSize + (slot & (pageSize - 1));
                    b.cell(g, segStart[s] + t, cell);
                    cells[g][segStart[s] + t] = cell;
                }
                int[] rows = new int[len];
                int[] values = new int[len];
                for (int t = 0; t < len; t++) {
                    rows[t] = cells[g][segStart[s] + t];
                    values[t] = segPos0[s] + t;
                }
                b.kvPos(g, rows, values);
            }
        }
        return b.build();
    }

    /** Existing shared pages of each write span plus new tail pages, in the commit allocation order, one group. */
    private int needPages(int g, int[] segSeq, int[] segLen) {
        int need = 0;
        for (int s = 0; s < segSeq.length; s++) {
            int seq = segSeq[s];
            int slot0 = slots[g][seq];
            int end = slot0 + segLen[s];
            PageList list = lists[seq][g];
            int listLen = list.size();
            int spanLast = (end - 1) / pageSize;
            for (int k = slot0 / pageSize; k <= spanLast && k < listLen; k++) {
                if (pools[g].refCount(list.page(k)) > 1) {
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
     * Applies a reserved step: takes its pages in the prepare order, swaps privatized pages into the lists, applies
     * the new {@code kv_pos} values of every group, then trims the SWA pages of the touched sequences.
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
                        mirroredPageRows(g, src, dst);
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
        // verify the applied tables before the step's page ids reach the device
        for (int g = 0; g < groupsCount(); g++) {
            for (int s = 0; s < nSeg; s++) {
                PageList list = lists[step.segSeq(s)][g];
                for (int k = 0; k < list.size(); k++) {
                    if (step.pageTableEntry(g, step.segSeq(s), k) != list.page(k)) {
                        throw new IllegalStateException("page table disagrees with the list of sequence "
                                + step.segSeq(s) + " at page " + k + " of group " + g);
                    }
                }
                for (int k = list.size(); k < maxPages; k++) {
                    if (step.pageTableEntry(g, step.segSeq(s), k) != -1) {
                        throw new IllegalStateException("page table entry beyond the list of sequence "
                                + step.segSeq(s) + " is not empty at page " + k + " of group " + g);
                    }
                }
            }
        }
        for (int s = 0; s < nSeg; s++) {
            int seq = step.segSeq(s);
            int end = step.segPos0(s) + step.segLen(s) - 1;
            posMaxArr[seq] = Math.max(posMaxArr[seq], end);
            if (posMinArr[seq] < 0) {
                posMinArr[seq] = step.segPos0(s);
            }
        }
        pendingCopies.addAll(step.pageCopies());
        for (int s = 0; s < nSeg; s++) {
            trimSequence(step.segSeq(s));
        }
    }

    /**
     * Moves the sequence's reference of {@code src} onto the fresh private page {@code dst} (the source's reference
     * stays on the pending copy until its completion releases it) and mirrors the host {@code kv_pos} of the copy.
     */
    private void mirroredPageRows(int g, int src, int dst) {
        System.arraycopy(kvPos[g], src * pageSize, kvPos[g], dst * pageSize, pageSize);
    }

    // ------------------------------------------------------------------ copy-on-write mutation paths

    /**
     * Invariant 7's private helper: when the page of {@code (seq, group, listIndex)} has {@code refCount > 1}, it
     * allocates a page, emits a {@link PageCopy} of the whole page, swaps the page in the sequence's list and moves
     * the sequence's reference on the source to the pending copy. The caller mutates the private copy only. A page
     * with {@code refCount == 1} is left alone and mutated in place.
     */
    private void privatize(int g, int seq, int listIndex) {
        PageList list = lists[seq][g];
        int src = list.page(listIndex);
        if (pools[g].refCount(src) == 1) {
            return;
        }
        int dst = pools[g].alloc();
        PageCopy copy = PageCopy.of(g, src, dst, pageSize);
        mirroredPageRows(g, src, dst);
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
        int hi = normalizeRange(seq, p0, p1);
        int max = posMaxArr[seq];
        if (max < 0 || lo >= hi || lo > max) {
            return;
        }
        hi = Math.min(hi, max + 1);
        boolean tail = hi == max + 1;
        for (int g = 0; g < groupsCount(); g++) {
            int firstCleared = removeGroup(g, seq, lo, hi);
            if (tail && firstCleared >= 0 && slots[g][seq] > firstCleared) {
                // a tail removal that partially clears a page strands its dead rows below the cursor: the next
                // append continues at the first dead slot, so the page addressing stays dense and attention reads
                // the fresh rows. An interior removal keeps the cursor: live rows sit above its holes.
                slots[g][seq] = firstCleared;
            }
            if (lists[seq][g].isEmpty()) {
                // an emptied list holds no rows: the counter drops with it (the armed-but-unfed token's slot
                // never held a row, so the live-row sum can leave a residue an admission would trip over)
                slots[g][seq] = 0;
            }
        }
        if (hi == max + 1 && posMinArr[seq] == 0) {
            states[seq].truncate(Math.min(states[seq].nPast(), hi - lo));
        } else {
            // an interior or non-aligned removal leaves no reusable token prefix
            states[seq].clear();
        }
        recomputePos(seq);
    }

    /**
     * Removes one group's rows in {@code [lo, hi)} and returns the slot of the first cleared row of a partially
     * cleared page, or -1 when every touched page left whole (the cursor then needs no pull-down). The walk runs
     * from the back so every whole page in the range leaves through {@code removeBack}: a page that is wholly in
     * the range always sits above the pages still to visit, so it is the last entry when the walk reaches it and
     * no interior tombstone can ever stay behind in the list.
     */
    private int removeGroup(int g, int seq, int lo, int hi) {
        PageList list = lists[seq][g];
        int firstCleared = -1;
        for (int k = list.size() - 1; k >= 0; k--) {
            int page = list.page(k);
            int base = page * pageSize;
            int live = 0;
            int liveIn = 0;
            for (int r = 0; r < pageSize; r++) {
                int pos = kvPos[g][base + r];
                if (pos < 0) {
                    continue;
                }
                live++;
                if (pos >= lo && pos < hi) {
                    liveIn++;
                }
            }
            if (live == 0 || liveIn == live) {
                // the whole page leaves the sequence
                if (k == list.size() - 1) {
                    slots[g][seq] = Math.max(0, slots[g][seq] - live);
                    list.removeBack(1);
                    unref(true, g, page);
                    continue;
                }
                if (k == 0) {
                    list.removeFront(1);
                    slots[g][seq] = Math.max(0, slots[g][seq] - pageSize);
                    unref(true, g, page);
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
                int base2 = lists[seq][g].page(k) * pageSize;
                clearRowsIn(g, base2, lo, hi);
                for (int r = 0; r < pageSize; r++) {
                    if (kvPos[g][base2 + r] == -1) {
                        // the cursor is a logical slot of the sequence (list index * P + row), not a pool cell
                        int slot = k * pageSize + r;
                        if (firstCleared < 0 || slot < firstCleared) {
                            firstCleared = slot;
                        }
                    }
                }
            }
        }
        return firstCleared;
    }

    /** Sets the rows of {@code base} whose position is selected empty and reports any position that was dropped. */
    private void clearRowsIn(int g, int base, int lo, int hi) {
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
    public String copyRefusal() {
        return null;
    }

    @Override
    public void copy(int src, int dst, int p0, int p1) {
        checkSeq(src);
        checkSeq(dst);
        if (src == dst) {
            throw new IllegalArgumentException("copy needs two different sequences, got " + src);
        }
        int lo = p0;
        int hi = normalizeRange(src, p0, p1);
        int srcMax = posMaxArr[src];
        if (lo < 0) {
            throw new IllegalArgumentException("copy starts before the source: position " + lo);
        }
        if (srcMax < 0 || hi <= lo || lo > srcMax) {
            throw new IllegalArgumentException("copy range [" + lo + ", " + hi + ") holds nothing of sequence "
                    + src + " (" + posMinArr[src] + ".." + srcMax + ")");
        }
        if (posMaxArr[dst] >= 0) {
            throw new IllegalStateException("copy shares pages with sequence " + dst + ", which stores positions "
                    + posMinArr[dst] + ".." + posMaxArr[dst] + "; the shared rows keep the source positions, so the"
                    + " destination must be empty");
        }
        // all-or-nothing: count the private leading pages first, no pool touched
        for (int g = 0; g < groupsCount(); g++) {
            int need = 0;
            PageList from = lists[src][g];
            for (int k = 0; k < from.size(); k++) {
                if (copyShareKind(g, from.page(k), lo, hi) == 2) {
                    need++;
                }
            }
            if (need > pools[g].free()) {
                throw new MemoryFullException(hi - lo, cap, hi - lo, freeCells());
            }
        }
        for (int g = 0; g < groupsCount(); g++) {
            copyGroup(g, src, dst, lo, hi);
        }
        posMinArr[dst] = lo;
        posMaxArr[dst] = hi - 1;
    }

    /** Part of {@code page} in the copy range: 0 none, 1 the whole in-range part (share it), 2 a leading partial. */
    private int copyShareKind(int g, int page, int lo, int hi) {
        int base = page * pageSize;
        boolean liveInRange = false;
        boolean liveBefore = false;
        for (int r = 0; r < pageSize; r++) {
            int pos = kvPos[g][base + r];
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
        PageList from = lists[src][g];
        PageList to = lists[dst][g];
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
                pools[g].ref(page);
                to.append(page);
                continue;
            }
            // rows before the range stay on the source alone: the destination gets a private page and copies only
            // its in-range rows into rows [firstRow, lastRowPlusOne)
            int pageBase = page * pageSize;
            int row0 = pageSize;
            int lastRowPlusOne = 0;
            for (int r = 0; r < pageSize; r++) {
                int pos = kvPos[g][pageBase + r];
                if (pos >= lo && pos < hi) {
                    row0 = Math.min(row0, r);
                    lastRowPlusOne = Math.max(lastRowPlusOne, r + 1);
                }
            }
            int priv = pools[g].alloc();
            pendingCopies.add(PageCopy.ofRange(g, page, priv, row0, lastRowPlusOne - row0));
            to.append(priv);
            int[] rows = new int[lastRowPlusOne - row0];
            int[] values = new int[rows.length];
            int n = 0;
            for (int r = row0; r < lastRowPlusOne; r++) {
                int pos = kvPos[g][pageBase + r];
                kvPos[g][priv * pageSize + r] = pos;
                if (pos >= 0) {
                    usedRows[g]++;
                    rows[n] = priv * pageSize + r;
                    values[n] = pos;
                    n++;
                }
            }
            if (n > 0) {
                // the device applies the copy besides the step's writes: it needs the copied rows' positions
                pendingReports.add(RowReport.sets(g, Arrays.copyOf(rows, n), Arrays.copyOf(values, n)));
            }
        }
        // the copied rows end inside the last page (a page fully covered by the range ends whole): the next append
        // continues there and privatizes the last page first when it is shared
        slots[g][dst] = slotsOfList(to, g, lo, hi);
    }

    /**
     * Slots the copied list covers: a page per shorter page plus the in-range rows of the last one, so the next
     * append continues on the last copied row. Appends that reach a shared partial tail page privatize it first.
     */
    private int slotsOfList(PageList list, int g, int lo, int hi) {
        if (list.isEmpty()) {
            return 0;
        }
        int last = list.size() - 1;
        int base = list.page(last) * pageSize;
        int count = 0;
        for (int r = 0; r < pageSize; r++) {
            int pos = kvPos[g][base + r];
            if (pos >= lo && pos < hi) {
                count = r + 1;
            }
        }
        return last * pageSize + count;
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
    }

    private void dropSequence(int seq) {
        for (int g = 0; g < groupsCount(); g++) {
            PageList list = lists[seq][g];
            while (!list.isEmpty()) {
                unref(true, g, list.page(list.size() - 1));
                list.removeBack(1);
            }
            slots[g][seq] = 0;
            recountUsed(g);
        }
        states[seq].clear();
        posMinArr[seq] = -1;
        posMaxArr[seq] = -1;
    }

    @Override
    public String shiftRefusal() {
        return null;
    }

    @Override
    public void add(int seq, int p0, int p1, int delta) {
        checkSeq(seq);
        if (delta == 0) {
            return;
        }
        int lo = p0;
        int hi = normalizeRange(seq, p0, p1);
        if (lo < 0) {
            throw new IllegalArgumentException("shift starts before the sequence: position " + lo);
        }
        if (hi <= lo) {
            return;
        }
        for (int g = 0; g < groupsCount(); g++) {
            addGroup(g, seq, lo, hi, delta);
        }
        recomputePos(seq);
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
     * the leading page's rows below the window go empty on a private page only (invariant 7). Committed is the
     * sequence's committed element count, so trimming follows {@code committed - W_g} with W_g the group's widest
     * reader window (M-2). The prefix cache's completion order (O-3) is a later work package's concern.
     */
    public void trim(int seq) {
        checkSeq(seq);
        trimSequence(seq);
    }

    private void trimSequence(int seq) {
        if (posMaxArr[seq] < 0) {
            return;
        }
        int committed = posMaxArr[seq] + 1;
        for (int g = 0; g < groupsCount(); g++) {
            if (swaGroup[g]) {
                trimGroup(g, seq, committed - windowWidth[g]);
            }
        }
    }

    private void trimGroup(int g, int seq, int threshold) {
        if (threshold <= 0) {
            return;
        }
        PageList list = lists[seq][g];
        while (!list.isEmpty()) {
            int page = list.page(0);
            int base = page * pageSize;
            int lastLive = -1;
            boolean hasBelow = false;
            for (int r = pageSize - 1; r >= 0; r--) {
                int pos = kvPos[g][base + r];
                if (pos >= 0) {
                    if (lastLive < 0) {
                        lastLive = r;
                    }
                    if (pos < threshold) {
                        hasBelow = true;
                    }
                }
            }
            if (lastLive >= 0 && kvPos[g][base + lastLive] >= threshold) {
                // straddling page: the rows below the window go empty, the page stays
                if (hasBelow) {
                    privatize(g, seq, 0);
                    clearRowsIn(g, lists[seq][g].page(0) * pageSize, 0, threshold);
                }
                return;
            }
            // the whole page lies below the window (or has no live row): drop it
            if (pools[g].refCount(page) > 1) {
                // one PageCopy: the trim is written on the private page, whose rows go empty, then the page leaves
                privatize(g, seq, 0);
                int priv = list.page(0);
                clearRowsIn(g, priv * pageSize, Integer.MIN_VALUE, Integer.MAX_VALUE);
                unref(false, g, priv);
                list.removeFront(1);
            } else {
                clearRowsIn(g, base, Integer.MIN_VALUE, Integer.MAX_VALUE);
                unref(false, g, page);
                list.removeFront(1);
            }
            slots[g][seq] = Math.max(0, slots[g][seq] - pageSize);
        }
    }

    // ------------------------------------------------------------------ pool helpers

    /**
     * Drops one reference of {@code page} in {@code g}. A page freed to 0 goes back to the pool with its rows empty;
     * {@code rowReported} reports that emptiness to the device ({@link #pendingReports()}), skipping rows an
     * operation already emptied.
     */
    private void unref(boolean rowReported, int g, int page) {
        boolean freed = pools[g].refCount(page) == 1;
        pools[g].unref(page);
        if (!freed) {
            return;
        }
        int base = page * pageSize;
        int[] rows = new int[pageSize];
        int n = 0;
        for (int r = 0; r < pageSize; r++) {
            if (kvPos[g][base + r] >= 0) {
                kvPos[g][base + r] = -1;
                usedRows[g]--;
                rows[n++] = base + r;
            }
        }
        if (rowReported && n > 0) {
            pendingReports.add(RowReport.clears(g, Arrays.copyOf(rows, n)));
        }
    }

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

    private void recomputePos(int seq) {
        int min = -1;
        int max = -1;
        for (int g = 0; g < groupsCount(); g++) {
            PageList list = lists[seq][g];
            for (int k = 0; k < list.size(); k++) {
                int base = list.page(k) * pageSize;
                for (int r = 0; r < pageSize; r++) {
                    int pos = kvPos[g][base + r];
                    if (pos < 0) {
                        continue;
                    }
                    min = min < 0 ? pos : Math.min(min, pos);
                    max = Math.max(max, pos);
                }
            }
        }
        posMinArr[seq] = min;
        posMaxArr[seq] = max;
    }

    private int normalizeRange(int seq, int p0, int p1) {
        if (p0 < 0) {
            throw new IllegalArgumentException("range starts before the sequence: position " + p0);
        }
        if (p1 < -1) {
            throw new IllegalArgumentException("range end " + p1 + " is below -1");
        }
        if (p1 == -1) {
            return posMaxArr[seq] < 0 ? p0 : posMaxArr[seq] + 1;
        }
        return Math.max(p0, p1);
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
        int free = 0;
        for (int g = 0; g < groupsCount(); g++) {
            free += plan.group(g).rows() - usedRows[g];
        }
        return free;
    }

    @Override
    public int totalCells() {
        int total = 0;
        for (int g = 0; g < groupsCount(); g++) {
            total += plan.group(g).rows();
        }
        return total;
    }

    private void checkSeq(int seq) {
        if (seq < 0 || seq >= parallel) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + parallel + ")");
        }
    }

    @Override
    public String toString() {
        return "PagedSequences{parallel=" + parallel + ", pageSize=" + pageSize + ", groups=" + groupsCount()
                + ", " + freeCells() + "/" + totalCells() + " free rows}";
    }

    /**
     * One host-side {@code kv_pos} change the device must apply besides a step's own row writes: the rows of one
     * KV group together with their target values. Absolute reports ({@code values[i]}) set
     * {@code kv_pos[rows[i]] = values[i]}; a relative ({@code delta != 0}) report instead adds {@code delta}
     * ({@code LLM_KV_RopeShift_*}), with {@code values[i]} carrying the host's expected post-shift position.
     * Immutable; rows are ascending.
     */
    public static final class RowReport {

        private final int group;
        private final int[] rows;
        private final int[] values;
        private final int delta;

        private RowReport(int group, int[] rows, int[] values, int delta) {
            this.group = group;
            this.rows = rows;
            this.values = values;
            this.delta = delta;
        }

        /** Clearing rows: {@code kv_pos[rows[i]] = -1}. */
        public static RowReport clears(int group, int[] rowsAscending) {
            if (rowsAscending.length == 0) {
                throw new IllegalArgumentException("clears carry at least one row");
            }
            int[] values = new int[rowsAscending.length];
            Arrays.fill(values, -1);
            return new RowReport(group, rowsAscending.clone(), values, 0);
        }

        /** Shifting rows: {@code kv_pos[rows[i]] += delta}, the kernel's RoPE-shift inputs. */
        public static RowReport shift(int group, int[] rowsAscending, int[] newPositions, int delta) {
            if (rowsAscending.length == 0 || rowsAscending.length != newPositions.length) {
                throw new IllegalArgumentException("shift rows and new positions share one non-empty length");
            }
            return new RowReport(group, rowsAscending.clone(), newPositions.clone(), delta);
        }

        /** Setting rows: {@code kv_pos[rows[i]] = values[i]} (the host's mirror of a partial page copy). */
        public static RowReport sets(int group, int[] rowsAscending, int[] values) {
            if (rowsAscending.length == 0 || rowsAscending.length != values.length) {
                throw new IllegalArgumentException("sets carry rows and values of one non-empty length");
            }
            return new RowReport(group, rowsAscending.clone(), values.clone(), 0);
        }

        /** KV group of the rows. */
        public int group() { return group; }

        /** Rows, ascending. */
        public int[] rows() { return rows.clone(); }

        /**
         * Target values: -1 for a cleared row, the new position for a shifted or set one. {@code delta} 0 applies
         * these as absolute values.
         */
        public int[] values() { return values.clone(); }

        /**
         * The RoPE shift amount ({@code delta != 0} means relative: the device adds it instead of setting);
         * 0 for an absolute report.
         */
        public int delta() { return delta; }

        /** True when the device sets {@code values[]}; false when it adds {@code delta}. */
        public boolean absolute() {
            return delta == 0;
        }

        @Override
        public String toString() {
            return "RowReport{group=" + group + ", rows=" + rows.length
                    + (absolute() ? ", absolute" : ", delta=" + delta) + "}";
        }
    }
}
