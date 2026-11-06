package com.eonmux.neuralfree.gguf.memory.paged;

/**
 * One host-side row move the backend must run besides a step's writes: the rows {@code [srcRow, srcRow + rows)}
 * of one KV group (cell addresses) carry their keys and values to the rows at {@code dstRow}, which hold no live
 * position. The compaction of a shifted sequence emits these — the cells the rows live in move, so the data moves
 * with them. Immutable; the compaction's moves never read a cell another move wrote (every destination cell held
 * no live row, so it is no move's source), and any order is safe.
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
     * @throws IllegalArgumentException for a non-positive row count, a destination equal to the source, or a
     *                                  negative row
     */
    public static RowMove of(int seq, int group, int srcRow, int dstRow, int rows) {
        if (rows <= 0) {
            throw new IllegalArgumentException("row move needs rows > 0, got " + rows);
        }
        if (srcRow < 0 || dstRow < 0) {
            throw new IllegalArgumentException("row move needs non-negative rows, got src " + srcRow + ", dst "
                    + dstRow);
        }
        if (dstRow == srcRow) {
            throw new IllegalArgumentException("row move needs two different rows, got " + srcRow);
        }
        return new RowMove(seq, group, srcRow, dstRow, rows);
    }

    /** The sequence whose rows move. */
    public int seq() { return seq; }

    /** KV group of the rows. */
    public int group() { return group; }

    /** First source row (a cell address). */
    public int srcRow() { return srcRow; }

    /** First destination row (a cell address below the source). */
    public int dstRow() { return dstRow; }

    /** Rows the move carries. */
    public int rows() { return rows; }

    @Override
    public String toString() {
        return "RowMove{seq=" + seq + ", group=" + group + ", " + srcRow + " -> " + dstRow + " x " + rows + "}";
    }
}
