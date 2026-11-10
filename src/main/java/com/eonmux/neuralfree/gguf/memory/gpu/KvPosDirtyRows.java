package com.eonmux.neuralfree.gguf.memory.gpu;

import java.util.Arrays;

/**
 * The rows of one KV group whose device {@code kv_pos} may differ from the host mirror: every mirror change marks
 * its row, every device write of mirror rows clears them. A paged step uploads only the marked rows, grouped into
 * runs ({@link #runs}): two marked rows at most {@link #MERGE_GAP} clean rows apart share one write, so a decode
 * step over {@code n} sequences writes at most {@code n} short runs instead of the whole pool; past
 * {@link #MAX_RUNS} runs, or once the runs cover a quarter of the rows, one write spans the first to the last
 * marked row. Not thread-safe.
 */
final class KvPosDirtyRows {

    /** Clean rows between two marked ones that still share a write (a write's fixed cost buys ~256 bytes). */
    static final int MERGE_GAP = 64;
    /** Runs above which one spanning write replaces them. */
    static final int MAX_RUNS = 32;

    private final int rows;
    private final long[] words;
    private int count;

    KvPosDirtyRows(int rows) {
        if (rows < 0) {
            throw new IllegalArgumentException("a KV group of " + rows + " rows");
        }
        this.rows = rows;
        this.words = new long[(rows + 63) >>> 6];
    }

    /** Marks {@code row} as changed in the mirror since the device last received it. */
    void mark(int row) {
        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException("row " + row + " outside the KV group of " + rows + " rows");
        }
        long bit = 1L << row;
        int w = row >>> 6;
        if ((words[w] & bit) == 0) {
            words[w] |= bit;
            count++;
        }
    }

    /** Marks every row (the device copy diverged from the mirror as a whole). */
    void markAll() {
        if (rows == 0) {
            return;
        }
        Arrays.fill(words, -1L);
        int tail = rows & 63;
        if (tail != 0) {
            words[words.length - 1] = (1L << tail) - 1;
        }
        count = rows;
    }

    /** Clears the marks of rows {@code [from, to)}, which a write just carried from the mirror. */
    void clear(int from, int to) {
        if (from < 0 || to > rows || to < from) {
            throw new IndexOutOfBoundsException("rows [" + from + ", " + to + ") outside the KV group of " + rows
                    + " rows");
        }
        if (count == 0 || from == to) {
            return;
        }
        if (from == 0 && to == rows) {
            clearAll();
            return;
        }
        for (int r = from; r < to; ) {
            int w = r >>> 6;
            int lo = r & 63;
            int hi = Math.min(64, lo + (to - r));
            long mask = hi == 64 ? -1L << lo : ((1L << hi) - 1) & (-1L << lo);
            count -= Long.bitCount(words[w] & mask);
            words[w] &= ~mask;
            r += hi - lo;
        }
    }

    /** Clears every mark. */
    void clearAll() {
        Arrays.fill(words, 0L);
        count = 0;
    }

    /** Marked rows. */
    int count() {
        return count;
    }

    /** Whether {@code row} is marked. */
    boolean marked(int row) {
        return (words[row >>> 6] & (1L << row)) != 0;
    }

    /**
     * The write ranges that carry every marked row, as {@code [from, to)} pairs in {@code out} (ascending, disjoint);
     * returns the number of ranges. {@code out} holds at least {@code 2 * MAX_RUNS} ints.
     */
    int runs(int[] out) {
        if (out.length < 2 * MAX_RUNS) {
            throw new IllegalArgumentException("a run buffer of " + out.length + " ints holds fewer than "
                    + MAX_RUNS + " runs");
        }
        if (count == 0) {
            return 0;
        }
        int n = 0;
        int covered = 0;
        int first = -1;
        int last = -1;
        int runFrom = -1;
        int runTo = -1;
        boolean spill = false;
        for (int r = nextMarked(0); r >= 0; r = nextMarked(runTo)) {
            int end = nextClear(r);
            if (first < 0) {
                first = r;
            }
            last = end;
            if (runFrom >= 0 && r - runTo <= MERGE_GAP) {
                runTo = end;
                continue;
            }
            if (runFrom >= 0) {
                if (n == MAX_RUNS) {
                    spill = true;
                } else {
                    out[2 * n] = runFrom;
                    out[2 * n + 1] = runTo;
                    n++;
                    covered += runTo - runFrom;
                }
            }
            runFrom = r;
            runTo = end;
        }
        if (n == MAX_RUNS) {
            spill = true;
        } else {
            out[2 * n] = runFrom;
            out[2 * n + 1] = runTo;
            n++;
            covered += runTo - runFrom;
        }
        if (spill || (long) covered * 4 > rows) {
            out[0] = first;
            out[1] = last;
            return 1;
        }
        return n;
    }

    /** The first marked row at or after {@code from}, or -1. */
    private int nextMarked(int from) {
        if (from >= rows) {
            return -1;
        }
        int w = from >>> 6;
        long bits = words[w] & (-1L << (from & 63));
        while (bits == 0) {
            if (++w == words.length) {
                return -1;
            }
            bits = words[w];
        }
        return (w << 6) + Long.numberOfTrailingZeros(bits);
    }

    /** The first unmarked row at or after {@code from} ({@code rows} when the marks run to the end). */
    private int nextClear(int from) {
        int w = from >>> 6;
        long bits = ~words[w] & (-1L << (from & 63));
        while (bits == 0) {
            if (++w == words.length) {
                return rows;
            }
            bits = ~words[w];
        }
        return Math.min(rows, (w << 6) + Long.numberOfTrailingZeros(bits));
    }
}
