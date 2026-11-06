package com.eonmux.neuralfree.gguf.memory.paged;

import java.util.Arrays;

/**
 * One host-side {@code kv_pos} change the device must apply besides a step's own row writes: the rows of one
 * KV group together with their target values. Absolute reports ({@code values[i]}) set
 * {@code kv_pos[rows[i]] = values[i]}; a relative ({@code delta != 0} or an explicit {@code deltas[]}) report
 * instead adds the delta ({@code LLM_KV_RopeShift_*}), with {@code values[i]} carrying the host's expected
 * post-shift position. Immutable; rows are one ascending run.
 */
public final class RowReport {

    private final int group;
    private final int[] rows;
    private final int[] values;
    private final int delta;
    /** Per-row deltas of a position division, or null for a uniform (or absolute) report. */
    private final int[] deltas;

    private RowReport(int group, int[] rows, int[] values, int delta) {
        this(group, rows, values, delta, null);
    }

    private RowReport(int group, int[] rows, int[] values, int delta, int[] deltas) {
        this.group = group;
        this.rows = rows;
        this.values = values;
        this.delta = delta;
        this.deltas = deltas;
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

    /**
     * Re-positioning rows: {@code kv_pos[rows[i]] = newPositions[i]}, each row re-rotated by its own shift
     * {@code deltas[i]} (a position division, where the shift of a row depends on its old position).
     */
    public static RowReport reposition(int group, int[] rowsAscending, int[] newPositions, int[] deltas) {
        if (rowsAscending.length == 0 || rowsAscending.length != newPositions.length
                || rowsAscending.length != deltas.length) {
            throw new IllegalArgumentException("reposition rows, positions and deltas share one non-empty length");
        }
        return new RowReport(group, rowsAscending.clone(), newPositions.clone(), 0, deltas.clone());
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
     * The RoPE shift amount of a uniform report ({@code delta != 0} means relative: the device adds it instead of
     * setting); 0 for an absolute report or a per-row one ({@link #rowDelta(int)}).
     */
    public int delta() { return delta; }

    /**
     * The RoPE shift of the report's row {@code i}: the uniform {@link #delta()}, or the row's own delta of a
     * {@link #reposition} report.
     */
    public int rowDelta(int i) {
        return deltas != null ? deltas[i] : delta;
    }

    /** True when the device sets {@code values[]}; false when it adds each row's {@link #rowDelta(int)}. */
    public boolean absolute() {
        return delta == 0 && deltas == null;
    }

    @Override
    public String toString() {
        return "RowReport{group=" + group + ", rows=" + rows.length
                + (absolute() ? ", absolute" : deltas != null ? ", per-row deltas" : ", delta=" + delta) + "}";
    }
}
