package com.eonmux.neuralfree.gguf.memory.gpu;

import java.util.List;

/**
 * Immutable plan of one group's context shift: positions after the shift, the per-row RoPE deltas (indexed by
 * the row after the shift) and the row moves, as runs {@code (src, dst, n)} in ascending source order.
 *
 * <p>Executing the moves on a cache: when every move goes down ({@code dst < src}, always the case for a full
 * cache, where {@code dst = src - nDiscard}) the runs are copied in order, each in chunks of at most
 * {@code src - dst} rows, so no chunk overlaps itself or a row still to be read. Otherwise (a ring rotation)
 * the moved rows are first gathered into a staging area and then scattered ({@link #needsStaging()}).
 */
public final class KvShift {

    private final int[] positionsAfter;
    private final int[] delta;
    private final int[][] runs;
    private final int movedRows;
    private final boolean needsStaging;

    KvShift(int[] positionsAfter, int[] delta, List<int[]> runs) {
        this.positionsAfter = positionsAfter;
        this.delta = delta;
        this.runs = runs.toArray(new int[0][]);
        int moved = 0;
        boolean staging = false;
        for (int[] run : this.runs) {
            moved += run[2];
            staging |= run[1] > run[0];
        }
        this.movedRows = moved;
        this.needsStaging = staging;
    }

    public int rows() {
        return positionsAfter.length;
    }

    /** Position of a row after the shift, -1 when empty. */
    public int positionAfter(int row) {
        return positionsAfter[row];
    }

    int[] positionsAfter() {
        return positionsAfter.clone();
    }

    /** RoPE delta of a row after the shift (0 for rows not shifted). */
    public int delta(int row) {
        return delta[row];
    }

    /** Number of move runs. */
    public int runCount() {
        return runs.length;
    }

    public int runSource(int i) {
        return runs[i][0];
    }

    public int runTarget(int i) {
        return runs[i][1];
    }

    public int runRows(int i) {
        return runs[i][2];
    }

    /** Total rows moved. */
    public int movedRows() {
        return movedRows;
    }

    /** True when some row moves up, so the moves need a staging area. */
    public boolean needsStaging() {
        return needsStaging;
    }
}
