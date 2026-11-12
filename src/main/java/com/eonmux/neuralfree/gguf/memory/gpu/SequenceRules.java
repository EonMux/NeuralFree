package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.gguf.memory.CompressorRollback;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

/**
 * Sequence rules shared by the device and heap memories, decided from the plan, the sequence record and the host
 * mirrors of the KV groups: when the last positions can be dropped, when a snapshot can be taken or restored, and
 * when the context can shift.
 * Each check returns null when the operation is allowed, or the reason it is not.
 */
public final class SequenceRules {

    private SequenceRules() {
    }

    /** True when any layer has DSV4 compressed caches, whose compressor state accumulates across positions. */
    public static boolean hasCompressed(MemoryPlan plan) {
        return plan.hasCompressed();
    }

    /**
     * Why the last {@code nTokens > 0} positions cannot be dropped: too few stored, recurrent state (which rolls
     * back only through snapshot / restore), DSV4 compressor state without a usable rollback snapshot
     * ({@link CompressorRollback#blocker}), or {@link #ringBlocker}.
     *
     * @param groups     the host mirrors of every KV group of the memory
     * @param compressor the memory's compressor rollback record (consulted only when the plan has compressed
     *                   caches)
     */
    public static String rollbackBlocker(MemoryPlan plan, GroupPositions[] groups, SequenceState sequence,
                                         CompressorRollback compressor, int nTokens) {
        if (nTokens > sequence.nPast()) {
            return "only " + sequence.nPast() + " positions are stored";
        }
        if (plan.hasRecurrentState()) {
            return "the model keeps recurrent state; use snapshot() and restore()";
        }
        if (plan.hasCompressed()) {
            String why = compressor.blocker(nTokens);
            if (why != null) {
                return why;
            }
        }
        return ringBlocker(groups, sequence, 0, nTokens);
    }

    /**
     * Why storing {@code nWrite} more positions and then dropping the last {@code nDrop} would leave a
     * sliding-window ring without keys the next query attends ({@link GroupPositions#keepsWindowAfter}); null
     * when every ring still holds the window of the query at {@code nPast + nWrite - nDrop}. This is the
     * condition llama.cpp checks through {@code seq_pos_min} before reusing a cached prefix of an SWA model
     * (tools/server/server-context.cpp:4061-4070): a ring row overwritten by a later position, or emptied by an
     * earlier drop, cannot be brought back.
     *
     * @throws IllegalArgumentException for negative counts or {@code nDrop > nPast + nWrite}
     */
    public static String ringBlocker(GroupPositions[] groups, SequenceState sequence, int nWrite, int nDrop) {
        int nPast = sequence.nPast();
        if (nWrite < 0 || nDrop < 0 || (long) nDrop > (long) nPast + nWrite) {
            throw new IllegalArgumentException("cannot store " + nWrite + " and drop " + nDrop + " of " + nPast
                    + " stored positions");
        }
        int newPast = (int) ((long) nPast + nWrite - nDrop);
        for (GroupPositions g : groups) {
            if (!g.keepsWindowAfter(nPast, nWrite, newPast)) {
                return "the sliding-window ring of " + g.group() + " no longer holds every key the query at "
                        + "position " + newPast + " attends";
            }
        }
        return null;
    }

    /**
     * Why a snapshot cannot be taken: no recurrent state, DSV4 compressor state the snapshot would miss, or a
     * plan that reserved no shadow buffers ({@link MemoryPlan#snapshotsReserved()}).
     */
    public static String snapshotBlocker(MemoryPlan plan) {
        if (!plan.hasRecurrentState()) {
            return "memory has no recurrent state to snapshot";
        }
        if (plan.hasCompressed()) {
            return "memory with DSV4 compressed caches cannot be snapshot";
        }
        if (!plan.snapshotsReserved()) {
            return "the memory was planned without recurrent-state snapshot buffers; set "
                    + MemoryConfig.PROP_STATE_SNAPSHOTS + "=true";
        }
        return null;
    }

    /** Why the context cannot shift ({@link MemoryPlan#shiftBlocker()}); null when it can. */
    public static String shiftBlocker(MemoryPlan plan) {
        return plan.shiftBlocker();
    }

    /**
     * Why a context shift would leave a sliding-window ring short of keys: the first query after the shift
     * ({@code nPast - nDiscard}) must still find every kept position its window covers. llama.cpp refuses any
     * shift of a cache with rings (src/llama-kv-cache-iswa.cpp:251-255); the shifts allowed here give the result
     * of the same shift on full-size caches.
     */
    public static String ringShiftBlocker(GroupPositions group, SequenceState sequence, int nKeep, int nDiscard) {
        int nPastAfter = sequence.nPast() - nDiscard;
        if (group.keepsWindowAfterShift(nKeep, nPastAfter)) {
            return null;
        }
        return "the sliding-window ring of " + group.group() + " overwrote kept positions below " + nKeep
                + " that the query at position " + nPastAfter + " still attends";
    }

    /** Validates the range of a context shift against the stored positions. */
    public static void checkShift(SequenceState sequence, int nKeep, int nDiscard) {
        if (nKeep < 0 || nDiscard <= 0 || (long) nKeep + nDiscard > sequence.nPast()) {
            throw new IllegalArgumentException("context shift of keep=" + nKeep + " discard=" + nDiscard
                    + " is invalid for " + sequence.nPast() + " stored positions");
        }
    }

    /**
     * Removes positions {@code [nKeep, nKeep + nDiscard)} from the sequence record, keeping its wrap flag. The
     * record itself rewrites the tokens in place and caps its contiguous reuse frontier at the kept count
     * ({@link SequenceState#shift}), so a following generation can never reuse the rows
     * that were computed with the discarded context.
     */
    public static void shiftSequence(SequenceState sequence, int nKeep, int nDiscard) {
        sequence.shift(nKeep, nDiscard);
    }
}
