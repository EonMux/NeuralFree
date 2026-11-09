package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.gguf.memory.MemoryPlan;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Host record of one KV group's {@code kv_pos}: the position stored in each row, -1 when empty. Every stored
 * position {@code p} sits in row {@code group().rowOf(p)}; the operations below keep that invariant. Shared by
 * the device step inputs and the heap memory. Not thread-safe.
 */
public final class GroupPositions {

    private final MemoryPlan.KvGroup group;
    private int[] pos;

    public GroupPositions(MemoryPlan.KvGroup group) {
        this.group = Objects.requireNonNull(group, "group");
        this.pos = new int[group.rows()];
        Arrays.fill(pos, -1);
    }

    public MemoryPlan.KvGroup group() {
        return group;
    }

    public int rows() {
        return pos.length;
    }

    /** Position stored in {@code row}, -1 when empty. */
    public int position(int row) {
        if (row < 0 || row >= pos.length) {
            throw new IndexOutOfBoundsException("row " + row + " outside the KV group of " + pos.length + " rows");
        }
        return pos[row];
    }

    /** Copy of every row's position. */
    public int[] snapshot() {
        return pos.clone();
    }

    /**
     * Validates a step of {@code n} positions from {@code pos0}: every position must have a row, and a ring may
     * not hold fewer rows than the step (its tokens would overwrite each other).
     *
     * @throws IllegalArgumentException otherwise
     */
    public void checkStep(int pos0, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("step of " + n + " positions");
        }
        if (group.ring()) {
            if (n > pos.length) {
                throw new IllegalArgumentException("step of " + n + " tokens exceeds the ring of " + pos.length
                        + " rows of " + group);
            }
            group.rowOf(pos0);
        } else if (group.sharded()) {
            /* The step may lie wholly or partly outside this rank's block; only the overlap is stored. */
            if (pos0 < 0) {
                throw new IllegalArgumentException("step positions are negative: " + pos0);
            }
        } else {
            group.rowOf(pos0);
            group.rowOf(pos0 + n - 1);
        }
    }

    /**
     * Stores positions {@code [pos0, pos0 + n)} in their rows; on a context shard only the positions its block
     * holds are stored.
     *
     * @return true when a ring row holding a different stored position was overwritten
     */
    public boolean store(int pos0, int n) {
        checkStep(pos0, n);
        boolean overwrote = false;
        for (int t = 0; t < n; t++) {
            int p = pos0 + t;
            if (!group.holds(p)) {
                continue;
            }
            int r = group.rowOf(p);
            if (pos[r] >= 0 && pos[r] != p) {
                overwrote = true;
            }
            pos[r] = p;
        }
        return overwrote;
    }

    /** Empties the rows of positions {@code [pos0, pos0 + n)} that still hold those positions. */
    public void drop(int pos0, int n) {
        for (int t = 0; t < n; t++) {
            int p = pos0 + t;
            if (!group.holds(p)) {
                continue;
            }
            int r = group.rowOf(p);
            if (pos[r] == p) {
                pos[r] = -1;
            }
        }
    }

    /** Empties every row. */
    public void clear() {
        Arrays.fill(pos, -1);
    }

    /**
     * Sets the position stored in {@code row} directly (the paged step's partner of {@link #store}:
     * {@code PagedStep} addresses rows by page table, not by consecutive position, so {@code upload} cannot
     * derive the rows from the positions alone). Host mirror only; the caller uploads the range to the device
     * like every other mirror change.
     */
    public void set(int row, int position) {
        pos[row] = position;
    }

    /**
     * First row of the smallest contiguous row range covering positions {@code [pos0, pos0 + n)}: the run of
     * rows when it does not wrap, the whole group when it does.
     */
    public int spanFrom(int pos0, int n) {
        if (group.sharded()) {
            int lo = Math.max(pos0, group.rowBase());
            int hi = Math.min(pos0 + n, group.rowBase() + group.rows());
            return hi <= lo ? 0 : lo - group.rowBase();
        }
        int r0 = group.rowOf(pos0);
        return r0 + n <= pos.length ? r0 : 0;
    }

    /** End (exclusive) of the row range of {@link #spanFrom(int, int)}. */
    public int spanTo(int pos0, int n) {
        if (group.sharded()) {
            int lo = Math.max(pos0, group.rowBase());
            int hi = Math.min(pos0 + n, group.rowBase() + group.rows());
            return hi <= lo ? 0 : hi - group.rowBase();
        }
        int r0 = group.rowOf(pos0);
        return r0 + n <= pos.length ? r0 + n : pos.length;
    }

    /**
     * Plans the context shift of llama.cpp {@code seq_rm(nKeep, nKeep + nDiscard)} followed by
     * {@code seq_add(nKeep + nDiscard, inf, -nDiscard)} (src/llama-kv-cache.cpp:404, 592): positions below
     * {@code nKeep} stay, positions in {@code [nKeep, nKeep + nDiscard)} are removed, and every later position
     * {@code p} becomes {@code p - nDiscard}. Because rows are addressed by {@code rowOf}, a shifted position
     * moves to {@code rowOf(p - nDiscard)}; the returned plan lists those row moves and the per-row deltas.
     * Nothing changes until {@link #apply(KvShift)}.
     *
     * @throws IllegalArgumentException for {@code nKeep < 0} or {@code nDiscard <= 0}
     * @throws IllegalStateException    when two positions would land in one row (a broken invariant)
     */
    public KvShift planShift(int nKeep, int nDiscard) {
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep
                    + "/" + nDiscard);
        }
        int rows = pos.length;
        int[] next = new int[rows];
        int[] delta = new int[rows];
        Arrays.fill(next, -1);
        List<int[]> moves = new ArrayList<>();
        long discardEnd = (long) nKeep + nDiscard;
        for (int r = 0; r < rows; r++) {
            int p = pos[r];
            if (p < 0 || (p >= nKeep && p < discardEnd)) {
                continue;
            }
            int q = p < nKeep ? p : p - nDiscard;
            int d;
            if (p < nKeep) {
                d = r;
            } else {
                if (!group.holds(q)) {
                    // a context shard cannot carry a position across its block boundary
                    continue;
                }
                d = group.rowOf(q);
            }
            if (next[d] >= 0) {
                throw new IllegalStateException("context shift maps positions " + next[d] + " and " + q
                        + " to row " + d + " of " + group);
            }
            next[d] = q;
            if (p >= nKeep) {
                delta[d] = -nDiscard;
                if (d != r) {
                    appendMove(moves, r, d);
                }
            }
        }
        return new KvShift(next, delta, moves);
    }

    private static void appendMove(List<int[]> runs, int src, int dst) {
        if (!runs.isEmpty()) {
            int[] last = runs.get(runs.size() - 1);
            if (last[0] + last[2] == src && last[1] + last[2] == dst) {
                last[2]++;
                return;
            }
        }
        runs.add(new int[]{src, dst, 1});
    }

    /**
     * Whether the first query after a context shift, at position {@code nPastAfter}, still finds every kept
     * position (below {@code nKeep}) its window covers. Only a ring can fail: rows of kept positions may have been
     * overwritten, and the shift cannot bring them back.
     */
    public boolean keepsWindowAfterShift(int nKeep, int nPastAfter) {
        if (!group.ring() || nKeep <= 0) {
            return true;
        }
        for (int p = group.window().firstVisible(nPastAfter); p < nKeep; p++) {
            if (pos[group.rowOf(p)] != p) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether the first query at {@code newPast} still finds every position of its window
     * ({@code [firstVisible(newPast), newPast)}) in its row once positions {@code [nPast, nPast + nWrite)} are
     * stored and every position from {@code newPast} on is dropped. With {@code nWrite == 0} this is the rule of
     * a plain rollback to {@code newPast}. Read from the mirror, so rows overwritten by later positions and rows
     * emptied by earlier drops both count as lost. Only a ring can fail: a full cache keeps every position below
     * {@code nPast} in its own row.
     *
     * @param nPast   positions stored now
     * @param nWrite  positions about to be stored from {@code nPast}
     * @param newPast positions left after the drop, at most {@code nPast + nWrite}
     * @throws IllegalArgumentException for negative arguments or {@code newPast > nPast + nWrite}
     */
    public boolean keepsWindowAfter(int nPast, int nWrite, int newPast) {
        if (nPast < 0 || nWrite < 0 || newPast < 0 || (long) newPast > (long) nPast + nWrite) {
            throw new IllegalArgumentException("window check needs 0 <= newPast <= nPast + nWrite, got nPast="
                    + nPast + " nWrite=" + nWrite + " newPast=" + newPast);
        }
        if (!group.ring() || newPast == 0) {
            return true;
        }
        int rows = pos.length;
        if (nWrite > rows) {
            return false;
        }
        long writeEnd = (long) nPast + nWrite;
        for (int p = group.window().firstVisible(newPast); p < newPast; p++) {
            if (p >= nPast) {
                return true;
            }
            if (pos[group.rowOf(p)] != p) {
                return false;
            }
            long firstOverwrite = p + (long) rows * ((nPast - p + rows - 1) / rows);
            if (firstOverwrite < writeEnd) {
                return false;
            }
        }
        return true;
    }

    /** Installs the positions of a shift planned on this group. */
    public void apply(KvShift shift) {
        if (shift.rows() != pos.length) {
            throw new IllegalArgumentException("shift of " + shift.rows() + " rows applied to a group of "
                    + pos.length);
        }
        pos = shift.positionsAfter();
    }
}
