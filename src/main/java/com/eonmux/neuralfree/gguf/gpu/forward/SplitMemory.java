package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The {@link ModelMemory} of a layer split over the memories of its sides: every device's memory of its own span
 * and, when the first layers run on the host, their heap memory. The sides hold identical sequence records, which
 * the split forward backend appends in lockstep, so every operation runs on all of them and succeeds only when each
 * side allows it: {@link #reset()} empties all, {@link #rollback(int)} drops the tail of all, and
 * {@link #snapshot()} / {@link #restore()} save and restore the recurrent state of every side that holds one, while
 * a side without recurrent state rolls its sequence back to the saved length on restore. The plan it reports is the
 * plan of the whole model. The sides belong to the stages that allocated them; {@link #close()} releases nothing.
 * Not thread-safe.
 */
public final class SplitMemory implements ModelMemory {

    private final MemoryPlan plan;
    private final List<ModelMemory> sides;
    private final ModelMemory primary;
    private final List<ModelMemory> snapshotted = new ArrayList<>();
    private int snapshotPast = -1;

    /**
     * @param plan    the memory plan of the whole model
     * @param sides   the memory of every side, each over its own layers of {@code plan}
     * @param primary the side whose sequence record the facade reports (the main device's)
     * @throws IllegalArgumentException when {@code sides} is empty or does not contain {@code primary}
     */
    public SplitMemory(MemoryPlan plan, List<? extends ModelMemory> sides, ModelMemory primary) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sides = Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(sides, "sides")));
        this.primary = Objects.requireNonNull(primary, "primary");
        if (this.sides.isEmpty() || !this.sides.contains(primary)) {
            throw new IllegalArgumentException("a split memory needs its sides and a primary among them, got "
                    + this.sides.size() + " side(s)");
        }
    }

    /** The memories of the sides, in pipeline order (unmodifiable). */
    public List<ModelMemory> sides() {
        return sides;
    }

    /** Records the step of {@code n} tokens from {@code off} of {@code tokens} on every side. */
    void append(int[] tokens, int off, int n) {
        for (ModelMemory m : sides) {
            m.sequence().append(tokens, off, n);
        }
    }

    @Override
    public MemoryPlan plan() {
        return plan;
    }

    @Override
    public SequenceState sequence() {
        return primary.sequence();
    }

    @Override
    public void reset() {
        RuntimeException first = null;
        for (ModelMemory m : sides) {
            try {
                m.reset();
            } catch (RuntimeException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        snapshotted.clear();
        snapshotPast = -1;
        if (first != null) {
            throw first;
        }
    }

    @Override
    public boolean canRollback(int nTokens) {
        for (ModelMemory m : sides) {
            if (!m.canRollback(nTokens)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Drops the last {@code nTokens} positions of every side.
     *
     * @throws IllegalStateException when a side cannot drop them; no side changes then
     */
    @Override
    public void rollback(int nTokens) {
        if (!canRollback(nTokens)) {
            throw new IllegalStateException("the split memory cannot drop the last " + nTokens + " of its "
                    + sequence().nPast() + " positions on every device");
        }
        for (ModelMemory m : sides) {
            m.rollback(nTokens);
        }
    }

    @Override
    public boolean ringsKeepWindow(int nWrite, int nDrop) {
        for (ModelMemory m : sides) {
            if (!m.ringsKeepWindow(nWrite, nDrop)) {
                return false;
            }
        }
        return true;
    }

    /** True when every side with recurrent state can take a snapshot. */
    @Override
    public boolean canSnapshot() {
        if (!plan.snapshotsReserved()) {
            return false;
        }
        for (ModelMemory m : sides) {
            if (m.hasRecurrentState() && !m.canSnapshot()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Saves the recurrent state of every side that holds one, and the sequence length the other sides return to.
     *
     * @throws UnsupportedOperationException when {@link #canSnapshot()} is false
     */
    @Override
    public void snapshot() {
        if (!canSnapshot()) {
            throw new UnsupportedOperationException("the split memory reserved no recurrent-state snapshots on"
                    + " every device");
        }
        snapshotted.clear();
        for (ModelMemory m : sides) {
            if (m.hasRecurrentState()) {
                m.snapshot();
                snapshotted.add(m);
            }
        }
        snapshotPast = sequence().nPast();
    }

    /**
     * Restores the last snapshot: the recurrent sides restore their own, the others drop the positions stored since.
     *
     * @throws IllegalStateException when no snapshot exists or a side cannot drop the positions
     */
    @Override
    public void restore() {
        if (snapshotPast < 0) {
            throw new IllegalStateException("the split memory holds no snapshot to restore");
        }
        for (ModelMemory m : sides) {
            if (!snapshotted.contains(m)) {
                int drop = m.sequence().nPast() - snapshotPast;
                if (drop < 0 || !m.canRollback(drop)) {
                    throw new IllegalStateException("a device of the split cannot return from "
                            + m.sequence().nPast() + " to the " + snapshotPast + " positions of the snapshot");
                }
            }
        }
        for (ModelMemory m : sides) {
            if (snapshotted.contains(m)) {
                m.restore();
            } else {
                m.rollback(m.sequence().nPast() - snapshotPast);
            }
        }
    }

    /** Releases nothing: every side belongs to the stage that allocated it. */
    @Override
    public void close() {
    }

    @Override
    public String toString() {
        return "SplitMemory{" + sides.size() + " sides, " + plan + '}';
    }
}
