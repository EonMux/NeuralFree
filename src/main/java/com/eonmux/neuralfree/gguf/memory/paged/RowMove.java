package com.eonmux.neuralfree.gguf.memory.paged;

/**
 * One host-side row move the backend must run besides a step's writes: the rows {@code [srcRow, srcRow + rows)}
 * of one KV group (cell addresses) carry their keys and values to the rows {@code [dstRow, dstRow + rows)}. The
 * compaction of a shifted sequence and the tree verify's commit emit these: the cells the rows live in move, so
 * the data moves with them. Immutable.
 *
 * <p>Ordering: the moves of one queue run in the order they were queued. A move never reads a cell an earlier move
 * of the queue wrote, but its destination may be an earlier move's source (a compaction that slides rows by less
 * than a run's length), so a backend must not reorder them. The source and destination ranges of one move never
 * overlap, as a device buffer copy requires.
 */
public final class RowMove {

    private final int seq;
    private final int group;
    private final int srcRow;
    private final int dstRow;
    private final int rows;

    private RowMove(int seq, int group, int srcRow, int dstRow, int rows) {
        this.seq = seq;
        this.group = group;
        this.srcRow = srcRow;
        this.dstRow = dstRow;
        this.rows = rows;
    }

    /**
     * A move of {@code rows} rows from {@code srcRow} to the empty rows at {@code dstRow} of one KV group, for the
     * sequence the rows belong to (the caller drops the moves of a sequence whose rows leave).
     *
     * @throws IllegalArgumentException for a non-positive row count, overlapping source and destination ranges,
     *                                  or a negative row
     */
    public static RowMove of(int seq, int group, int srcRow, int dstRow, int rows) {
        if (rows <= 0) {
            throw new IllegalArgumentException("row move needs rows > 0, got " + rows);
        }
        if (srcRow < 0 || dstRow < 0) {
            throw new IllegalArgumentException("row move needs non-negative rows, got src " + srcRow + ", dst "
                    + dstRow);
        }
        if (Math.abs((long) srcRow - dstRow) < rows) {
            throw new IllegalArgumentException("row move of " + rows + " rows from " + srcRow + " to " + dstRow
                    + " overlaps itself");
        }
        return new RowMove(seq, group, srcRow, dstRow, rows);
    }

    /** The sequence whose rows move. */
    public int seq() { return seq; }

    /** KV group of the rows. */
    public int group() { return group; }

    /** First source row (a cell address). */
    public int srcRow() { return srcRow; }

    /** First destination row (a cell address). */
    public int dstRow() { return dstRow; }

    /** Rows the move carries. */
    public int rows() { return rows; }

    @Override
    public String toString() {
        return "RowMove{seq=" + seq + ", group=" + group + ", " + srcRow + " -> " + dstRow + " x " + rows + "}";
    }
}
