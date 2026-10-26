package com.eonmux.neuralfree.gguf.memory.paged;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * One reserved forward step over a {@code PagedSequences} memory, the device addressing op of the step (built by
 * {@code PagedSequences.prepare}, applied by {@code PagedSequences.commit}): for every token the physical cell of
 * each KV group, per group the page table of every sequence, and the row writes the step executes.
 *
 * <p>Immutable, and validated: every page id is in {@code [0, nPages)} and the arrays share the declared sizes. The
 * device side runs every {@link PageCopy} of {@link #pageCopies()} before the step's kernels read the tables; the
 * host applies {@code commit} only after the step completed.
 */
public final class PagedStep {

    private final int nTokens;
    private final int maxSequences;
    private final int groups;
    private final int pageSize;
    private final int nPages;
    private final int maxPages;
    private final int[] segSeq;       // [nSeg] sequence of each segment
    private final int[] segStart;     // [nSeg] first ubatch row of each segment
    private final int[] segLen;       // [nSeg] rows of each segment
    private final int[] segPos0;      // [nSeg] first position of each segment
    private final int[][] cellIdx;    // [groups][nTokens] physical row per token
    private final int[][] pageTable;  // [groups][maxSequences * maxPages], -1 = this list element does not exist
    private final int[][] seqSlots;   // [groups][maxSequences] slots the sequence holds after commit
    private final int[][] kvPosRows;  // [groups][rows written], rows ascending
    private final int[][] kvPosValues;// [groups][counts match kvPosRows], the positions they take
    private final List<PageCopy> pageCopies;

    private PagedStep(Step s) {
        this.nTokens = s.nTokens;
        this.maxSequences = s.maxSequences;
        this.groups = s.groups;
        this.pageSize = s.pageSize;
        this.nPages = s.nPages;
        this.maxPages = s.maxPages;
        this.segSeq = s.segSeq;
        this.segStart = s.segStart;
        this.segLen = s.segLen;
        this.segPos0 = s.segPos0;
        this.cellIdx = s.cellIdx;
        this.pageTable = s.pageTable;
        this.seqSlots = s.seqSlots;
        this.kvPosRows = s.kvPosRows;
        this.kvPosValues = s.kvPosValues;
        this.pageCopies = s.pageCopies;
    }

    static final class Step {
        transient int nTokens;
        transient int maxSequences;
        transient int groups;
        transient int pageSize;
        transient int nPages;
        transient int maxPages;
        int[] segSeq;
        int[] segStart;
        int[] segLen;
        int[] segPos0;
        int[][] cellIdx;
        int[][] pageTable;
        int[][] seqSlots;
        int[][] kvPosRows;
        int[][] kvPosValues;
        List<PageCopy> pageCopies;
    }

    /** Rows of the step (ubatch-local token count). */
    public int nTokens() { return nTokens; }

    /** Sequences of the memory. */
    public int maxSequences() { return maxSequences; }

    /** KV groups. */
    public int groups() { return groups; }

    /** Rows per page. */
    public int pageSize() { return pageSize; }

    /** Pages of each group's pool (the validation bound {@code [0, nPages)} of every table entry). */
    public int nPages() { return nPages; }

    /** Page table entries per sequence per group ({@code ceil(cap / pageSize)}). */
    public int maxPages() { return maxPages; }

    /** Segments of the step (one per sequence, as the step layout carried). */
    public int nSeg() {
        return segSeq.length;
    }

    /** Sequence of segment {@code seg}.
     *
     * @throws IndexOutOfBoundsException when {@code seg} is outside the segments
     */
    public int segSeq(int seg) {
        return segSeq[seg];
    }

    /** First ubatch row of segment {@code seg}. */
    public int segStart(int seg) {
        return segStart[seg];
    }

    /** Rows of segment {@code seg}. */
    public int segLen(int seg) {
        return segLen[seg];
    }

    /** First position of segment {@code seg}. */
    public int segPos0(int seg) {
        return segPos0[seg];
    }

    /**
     * Physical cell of each token in group {@code g}: the row its K/V is stored in.
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] cellIdx(int g) {
        checkGroup(g);
        return cellIdx[g].clone();
    }

    /**
     * Page table of group {@code g}, laid out {@code pageTable[seq * maxPages + k]}: page id after commit, or -1
     * when sequence {@code seq}'s list has no page at index {@code k}.
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] pageTable(int g) {
        checkGroup(g);
        return pageTable[g].clone();
    }

    /**
     * Slots each sequence holds in group {@code g} after the commit ({@code seqSlots[seq]}).
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] seqSlots(int g) {
        checkGroup(g);
        return seqSlots[g].clone();
    }

    /**
     * Rows of group {@code g} whose {@code kv_pos} the step writes, ascending (the rows the step fills).
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] kvPosRows(int g) {
        checkGroup(g);
        return kvPosRows[g].clone();
    }

    /**
     * The new {@code kv_pos} values of {@link #kvPosRows(int)} ({@code kvPosValues[i]} pairs with
     * {@code kvPosRows[i]}), the positions of the written rows.
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] kvPosValues(int g) {
        checkGroup(g);
        return kvPosValues[g].clone();
    }

    /** The page copies the device executes before the step (one per privatization, PARITY invariant 7). */
    public List<PageCopy> pageCopies() {
        return pageCopies;
    }

    /**
     * Page table entry of sequence {@code seq} group {@code g} list index {@code k} (the entry of
     * {@code pageTable(g)[seq * maxPages + k]}).
     *
     * @throws IndexOutOfBoundsException outside the group or the table
     */
    int pageTableEntry(int g, int seq, int k) {
        if (g < 0 || g >= groups || seq < 0 || seq >= maxSequences || k < 0 || k >= maxPages) {
            throw new IndexOutOfBoundsException("page table entry g=" + g + " seq=" + seq + " k=" + k
                    + " outside groups=" + groups + ", seqs=" + maxSequences + ", maxPages=" + maxPages);
        }
        return pageTable[g][seq * maxPages + k];
    }

    @Override
    public String toString() {
        return "PagedStep{nTokens=" + nTokens + ", groups=" + groups + ", pageSize=" + pageSize + ", maxPages="
                + maxPages + ", copies=" + pageCopies.size() + "}";
    }

    static Builder builder(PagedSequences memory, int nTokens) {
        return new Builder(memory, nTokens);
    }

    /** Validates and freezes one step; mutable only while {@link Builder#build()} runs. */
    static final class Builder {

        private final PagedSequences memory;
        private final Step s;
        private final int n;

        private Builder(PagedSequences memory, int nTokens) {
            if (nTokens < 0) {
                throw new IllegalArgumentException("step token count " + nTokens + " is negative");
            }
            this.memory = memory;
            this.n = nTokens;
            this.s = new Step();
            s.nTokens = nTokens;
            s.maxSequences = memory.maxSequences();
            s.groups = memory.groupsCount();
            s.pageSize = memory.pageSize();
            s.nPages = memory.pageCount();
            s.maxPages = memory.maxPages();
            s.segSeq = null;
            s.cellIdx = new int[s.groups][n];
            s.pageTable = new int[s.groups][s.maxSequences * s.maxPages];
            for (int g = 0; g < s.groups; g++) {
                Arrays.fill(s.pageTable[g], -1);
            }
            s.seqSlots = new int[s.groups][s.maxSequences];
            s.kvPosRows = new int[s.groups][0];
            s.kvPosValues = new int[s.groups][0];
            s.pageCopies = new java.util.ArrayList<>();
        }

        /**
         * Segment metadata of the step (one segment per sequence), taken from the step layout.
         *
         * @throws IllegalArgumentException when the arrays differ in length or hold negative values
         */
        Builder segments(int[] seq, int[] start, int[] len, int[] pos0) {
            if (seq.length != start.length || seq.length != len.length || seq.length != pos0.length) {
                throw new IllegalArgumentException("segment arrays share one length");
            }
            s.segSeq = seq.clone();
            s.segStart = start.clone();
            s.segLen = len.clone();
            s.segPos0 = pos0.clone();
            return this;
        }

        /**
         * Cell of ubatch row {@code row} in group {@code g}.
         *
         * @throws IllegalStateException when the cell is out of range
         */
        Builder cell(int g, int row, int cell) {
            if (cell < 0 || cell >= s.nPages * s.pageSize) {
                throw new IllegalStateException("cell " + cell + " outside the " + s.nPages + "x" + s.pageSize
                        + " rows of group " + g);
            }
            s.cellIdx[g][row] = cell;
            return this;
        }

        Builder pageTable(int g, int seq, int pageIdx, int page) {
            if (pageIdx < 0 || pageIdx >= s.maxPages) {
                throw new IllegalStateException("page index " + pageIdx + " outside maxPages " + s.maxPages);
            }
            if (page >= s.nPages) {
                throw new IllegalStateException("page id " + page + " outside the pool of " + s.nPages + " pages");
            }
            s.pageTable[g][seq * s.maxPages + pageIdx] = page;
            return this;
        }

        Builder seqSlots(int g, int seq, int slots) {
            s.seqSlots[g][seq] = slots;
            return this;
        }

        Builder kvPos(int g, int[] rows, int[] values) {
            if (rows.length != values.length) {
                throw new IllegalStateException("kv_pos rows/values length mismatch");
            }
            s.kvPosRows[g] = rows;
            s.kvPosValues[g] = values;
            return this;
        }

        Builder copy(PageCopy c) {
            s.pageCopies.add(Objects.requireNonNull(c, "copy"));
            return this;
        }

        PagedStep build() {
            if (s.segSeq == null) {
                throw new IllegalStateException("a step carries its segment metadata");
            }
            for (int g = 0; g < s.groups; g++) {
                for (int row = 0; row < s.kvPosRows[g].length; row++) {
                    int r = s.kvPosRows[g][row];
                    if (r < 0 || r >= s.nPages * s.pageSize) {
                        throw new IllegalStateException("kv_pos row " + r + " outside the rows of group " + g);
                    }
                }
            }
            s.pageCopies = java.util.Collections.unmodifiableList(new java.util.ArrayList<>(s.pageCopies));
            return new PagedStep(s);
        }
    }

    private void checkGroup(int g) {
        if (g < 0 || g >= groups) {
            throw new IndexOutOfBoundsException("KV group " + g + " outside " + groups);
        }
    }
}
