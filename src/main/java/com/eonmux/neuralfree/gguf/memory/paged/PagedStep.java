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
    private final int[] segPosMax;    // [nSeg] highest stored position of each segment
    private final int[][] cellIdx;    // [groups][nTokens] physical row per token
    private final int[][][] tables;   // [groups][maxSequences] each sequence's page list after commit, null = none
    private final int[][] tableLen;   // [groups][maxSequences] entries of each table; the entries past it are -1
    private final int[][] seqSlots;   // [groups][maxSequences] slots the sequence holds after commit
    private final int[][] kvPosRows;  // [groups][rows written], ascending within each segment, in segment order
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
        this.segPosMax = s.segPosMax;
        this.cellIdx = s.cellIdx;
        this.tables = s.tables;
        this.tableLen = s.tableLen;
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
        int[] segPosMax;
        int[][] cellIdx;
        int[][][] tables;
        int[][] tableLen;
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
     * Highest stored position of segment {@code seg}: the rows of a media segment can share their base position,
     * so the maximum is not always {@code segPos0(seg) + segLen(seg) - 1}.
     */
    public int segPosMax(int seg) {
        return segPosMax[seg];
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
     * Page table of group {@code g} as one new array laid out {@code pageTable[seq * maxPages + k]}: page id after
     * commit, or -1 when sequence {@code seq}'s list has no page at index {@code k}. The step stores each sequence's
     * table compacted; the per-step readers take {@link #pageTableRow} and {@link #pageTableEntry} instead.
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] pageTable(int g) {
        checkGroup(g);
        int[] out = new int[maxSequences * maxPages];
        Arrays.fill(out, -1);
        for (int seq = 0; seq < maxSequences; seq++) {
            if (tableLen[g][seq] > 0) {
                System.arraycopy(tables[g][seq], 0, out, seq * maxPages, tableLen[g][seq]);
            }
        }
        return out;
    }

    /**
     * Entries of sequence {@code seq}'s page table in group {@code g}: every entry from this index on is -1, so a
     * reader of the slots {@code [0, seqSlot(g, seq))} needs only the entries below it.
     *
     * @throws IndexOutOfBoundsException when {@code g} or {@code seq} is outside the step
     */
    public int pageTableLength(int g, int seq) {
        checkGroup(g);
        checkSequence(seq);
        return tableLen[g][seq];
    }

    /**
     * Copies sequence {@code seq}'s page table in group {@code g} into {@code dst[0, pageTableLength(g, seq))} and
     * returns that length; the entries past it are -1 and {@code dst} keeps whatever it held there.
     *
     * @throws IndexOutOfBoundsException when {@code g} or {@code seq} is outside the step
     * @throws IllegalArgumentException  when {@code dst} is shorter than the table
     */
    public int pageTableRow(int g, int seq, int[] dst) {
        int len = pageTableLength(g, seq);
        if (Objects.requireNonNull(dst, "dst").length < len) {
            throw new IllegalArgumentException("sequence " + seq + " holds " + len + " page table entries in group "
                    + g + ", the destination " + dst.length);
        }
        if (len > 0) {
            System.arraycopy(tables[g][seq], 0, dst, 0, len);
        }
        return len;
    }

    /**
     * Slots each sequence holds in group {@code g} after the commit ({@code seqSlots[seq]}), as a new array; the
     * per-layer readers take {@link #seqSlot} instead.
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] seqSlots(int g) {
        checkGroup(g);
        return seqSlots[g].clone();
    }

    /**
     * Slots sequence {@code seq} holds in group {@code g} after the commit (0 for a sequence outside the step).
     *
     * @throws IndexOutOfBoundsException when {@code g} or {@code seq} is outside the step
     */
    public int seqSlot(int g, int seq) {
        checkGroup(g);
        checkSequence(seq);
        return seqSlots[g][seq];
    }

    /**
     * Rows of group {@code g} whose {@code kv_pos} the step writes (the rows the step fills, one segment's rows
     * ascending, the segments in order).
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] kvPosRows(int g) {
        checkGroup(g);
        return kvPosRows[g].clone();
    }

    /**
     * The positions the step writes: {@code kvPosValues[i]} pairs with {@code kvPosRows[i]}.
     *
     * @throws IndexOutOfBoundsException when {@code g} is outside {@link #groups()}
     */
    public int[] kvPosValues(int g) {
        checkGroup(g);
        return kvPosValues[g].clone();
    }

    /**
     * Writes the physical cells of sequence {@code seq}'s slots in group {@code g} into {@code dst[0, slots)} in slot
     * order: slot {@code j} lives in cell {@code pageTable[seq * maxPages + j / pageSize] * pageSize + j % pageSize},
     * the addressing of the paged attention kernels. A sparse selection over one sequence's slots maps its selected
     * slots to the cells the caches are read at this way.
     *
     * @return the slots the sequence holds after the commit ({@code seqSlots(g)[seq]})
     * @throws IndexOutOfBoundsException when {@code g} or {@code seq} is outside the step
     * @throws IllegalArgumentException  when {@code dst} is shorter than the sequence's slots
     */
    public int seqCells(int g, int seq, int[] dst) {
        checkGroup(g);
        if (seq < 0 || seq >= maxSequences) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + maxSequences + ")");
        }
        int slots = seqSlots[g][seq];
        if (Objects.requireNonNull(dst, "dst").length < slots) {
            throw new IllegalArgumentException("sequence " + seq + " holds " + slots + " slots in group " + g
                    + ", the cell array " + dst.length);
        }
        int[] table = tables[g][seq];
        for (int j = 0; j < slots; j++) {
            dst[j] = table[j / pageSize] * pageSize + j % pageSize;
        }
        return slots;
    }

    /** The page copies the device executes before the step (one per privatization). */
    public List<PageCopy> pageCopies() {
        return pageCopies;
    }

    /**
     * Page table entry of sequence {@code seq} group {@code g} list index {@code k} (the entry of
     * {@code pageTable(g)[seq * maxPages + k]}): -1 from {@link #pageTableLength} on.
     *
     * @throws IndexOutOfBoundsException outside the group or the table
     */
    public int pageTableEntry(int g, int seq, int k) {
        if (g < 0 || g >= groups || seq < 0 || seq >= maxSequences || k < 0 || k >= maxPages) {
            throw new IndexOutOfBoundsException("page table entry g=" + g + " seq=" + seq + " k=" + k
                    + " outside groups=" + groups + ", seqs=" + maxSequences + ", maxPages=" + maxPages);
        }
        return k < tableLen[g][seq] ? tables[g][seq][k] : -1;
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
            s.tables = new int[s.groups][s.maxSequences][];
            s.tableLen = new int[s.groups][s.maxSequences];
            s.seqSlots = new int[s.groups][s.maxSequences];
            s.kvPosRows = new int[s.groups][0];
            s.kvPosValues = new int[s.groups][0];
            s.pageCopies = new java.util.ArrayList<>();
        }

        /**
         * Segment metadata of the step (one segment per sequence), taken from the step layout; each segment's
         * highest stored position is {@code pos0 + len - 1}.
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
            s.segPosMax = new int[seq.length];
            for (int k = 0; k < seq.length; k++) {
                s.segPosMax[k] = pos0[k] + len[k] - 1;
            }
            return this;
        }

        /** Overrides {@link #segments}' per-segment highest stored position with explicit values. */
        Builder segPosMax(int[] posMax) {
            if (posMax.length != s.segPosMax.length) {
                throw new IllegalArgumentException("segment position maxima cover " + posMax.length
                        + " segments, the step holds " + s.segPosMax.length);
            }
            s.segPosMax = posMax.clone();
            return this;
        }

        /**
         * Cell of ubatch row {@code row} in group {@code g}. A cell of {@code -1} marks a row this memory does not
         * store (a context-parallel page shard's foreign-page row); every store kernel skips negative cells.
         *
         * @throws IllegalStateException when the cell is out of range
         */
        Builder cell(int g, int row, int cell) {
            if (cell < -1 || cell >= s.nPages * s.pageSize) {
                throw new IllegalStateException("cell " + cell + " outside the " + s.nPages + "x" + s.pageSize
                        + " rows of group " + g);
            }
            s.cellIdx[g][row] = cell;
            return this;
        }

        /** Sets entry {@code pageIdx} of sequence {@code seq}'s table in group {@code g}; unset entries stay -1. */
        Builder pageTable(int g, int seq, int pageIdx, int page) {
            if (pageIdx < 0 || pageIdx >= s.maxPages) {
                throw new IllegalStateException("page index " + pageIdx + " outside maxPages " + s.maxPages);
            }
            checkPage(page);
            int[] row = s.tables[g][seq];
            if (row == null || row.length <= pageIdx) {
                int grown = Math.min(s.maxPages, Math.max(pageIdx + 1, row == null ? 4 : row.length * 2));
                int from = row == null ? 0 : row.length;
                row = row == null ? new int[grown] : Arrays.copyOf(row, grown);
                Arrays.fill(row, from, grown, -1);
                s.tables[g][seq] = row;
            }
            row[pageIdx] = page;
            if (page >= 0) {
                s.tableLen[g][seq] = Math.max(s.tableLen[g][seq], pageIdx + 1);
            }
            return this;
        }

        /**
         * Sequence {@code seq}'s whole table in group {@code g}: entry {@code k} of {@code row} is list index
         * {@code k}, -1 where the list holds no page. The step adopts {@code row}, which its caller built for it
         * and no longer touches.
         *
         * @throws IllegalStateException when the row exceeds {@code maxPages} or names a page outside the pool
         */
        Builder pageTableRow(int g, int seq, int[] row) {
            if (row.length > s.maxPages) {
                throw new IllegalStateException("page table of " + row.length + " entries exceeds maxPages "
                        + s.maxPages);
            }
            int len = 0;
            for (int k = 0; k < row.length; k++) {
                checkPage(row[k]);
                if (row[k] >= 0) {
                    len = k + 1;
                }
            }
            s.tables[g][seq] = row;
            s.tableLen[g][seq] = len;
            return this;
        }

        private void checkPage(int page) {
            if (page < -1 || page >= s.nPages) {
                throw new IllegalStateException("page id " + page + " outside the pool of " + s.nPages + " pages");
            }
        }

        Builder seqSlots(int g, int seq, int slots) {
            s.seqSlots[g][seq] = slots;
            return this;
        }

        /**
         * Appends the rows of one segment whose {@code kv_pos} the step writes (the rows and their positions, a
         * segment's rows ascending); the step's arrays carry every segment's rows in segment order.
         *
         * @throws IllegalStateException when the arrays differ in length
         */
        Builder kvPos(int g, int[] rows, int[] values) {
            if (rows.length != values.length) {
                throw new IllegalStateException("kv_pos rows/values length mismatch");
            }
            s.kvPosRows[g] = concat(s.kvPosRows[g], rows);
            s.kvPosValues[g] = concat(s.kvPosValues[g], values);
            return this;
        }

        private static int[] concat(int[] head, int[] tail) {
            if (head.length == 0) {
                return tail.clone();
            }
            int[] out = Arrays.copyOf(head, head.length + tail.length);
            System.arraycopy(tail, 0, out, head.length, tail.length);
            return out;
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

    private void checkSequence(int seq) {
        if (seq < 0 || seq >= maxSequences) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside [0, " + maxSequences + ")");
        }
    }
}
