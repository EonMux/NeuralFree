package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Runtime memory of one model (KV caches, recurrent states, side caches) for a single sequence, allocated per
 * {@link #plan()}. Implemented by {@code DeviceModelMemory} (GPU) and {@code HeapModelMemory} (CPU).
 *
 * <p>Contract:
 * <ul>
 *   <li>{@link #reset()} empties every cache ({@code kv_pos = -1}) and zeroes every state; called per
 *       {@code generate()} unless the prefix is reused.</li>
 *   <li>{@link #canRollback(int)}: true for full caches; for sliding-window rings when every key the query at the
 *       new end attends is still in its row ({@link #ringsKeepWindow(int, int)} with nothing written): rows
 *       overwritten by later positions or emptied by earlier rollbacks cannot be brought back. Memory with recurrent
 *       state rolls back only through {@link #snapshot()} / {@link #restore()}. Memory with DSV4 compressed caches
 *       rolls back at most {@link MemoryPlan#rollbackSnapshots()} positions, once per step and within the last
 *       step's positions ({@link CompressorRollback}).</li>
 *   <li>{@link #rollback(int)} drops the last {@code nTokens} positions (rows marked empty; DSV4 compressor state
 *       restored from its snapshot plane) and fails when {@link #canRollback(int)} is false.</li>
 *   <li>{@link #snapshot()} copies every recurrent state into the shadow buffers the plan reserved
 *       ({@link MemoryPlan#snapshotsReserved()}); {@link #restore()} copies it back.</li>
 * </ul>
 */
public interface ModelMemory extends AutoCloseable {

    /** The plan this memory was allocated from. */
    MemoryPlan plan();

    /** Host record of the stored sequence. */
    SequenceState sequence();

    /**
     * The sequence-addressing API over this memory (llama.cpp {@code llama_memory_seq_*}). Default: the adapter
     * over this memory's single contiguous sequence; memories that hold their own pages replace it.
     */
    default SequenceMemory sequences() {
        return new SingleSequenceMemory(this);
    }

    /**
     * The paged sequence bookkeeping of this memory, or null for the contiguous form. Default: null, today's
     * contiguous single-sequence memory; a paged plan ({@code neuralfree.parallel > 1} or
     * {@code neuralfree.kv.paged}) replaces it ({@link #sequences}).
     */
    default PagedSequences pagedSequences() {
        return null;
    }

    /**
     * Why a paged step over {@code step} would refuse on this memory, or null when
     * {@link #beginPagedStep} can run it. Default: refused, the contiguous memory has no page tables.
     */
    default String pagedStepRefusal(PagedStep step) {
        return getClass().getSimpleName() + " holds no paged sequences";
    }

    /**
     * Marks the rows a paged step writes before the step's K/V are written: on the host mirror of every
     * {@code kv_pos}, the step's rows ({@code PagedStep.kvPosRows / kvPosValues}) and the mirror-only
     * {@code PagedSequences.pendingReports()} changes (removals, shifts, partial page copies); on a device memory
     * also the page tables, per-step {@code cell_idx} and the per-sequence addressing the paged kernels read. The
     * step stays current until {@link #endPagedStep}.
     *
     * @param positions    position of each ubatch row, in ubatch order (a device memory uploads them)
     * @param ropeSections section-major M-RoPE positions ({@code [section][row]}), or null to derive them from the
     *                     step's segments
     * @throws UnsupportedOperationException carrying {@link #pagedStepRefusal} when this memory is not paged
     * @throws IllegalArgumentException      when the step belongs to another memory
     */
    default void beginPagedStep(PagedStep step, int[] positions, int[][] ropeSections) {
        throw new UnsupportedOperationException(pagedStepRefusal(step));
    }

    /** Clears the paged step begun by the last {@link #beginPagedStep}. No-op without one. */
    default void endPagedStep() {
    }

    /**
     * The paged step the current forward pass runs, set by {@link #beginPagedStep} and cleared by
     * {@link #endPagedStep}, or null between steps. A mixer on a paged memory reads it to store its K/V rows at the
     * step's {@code cellIdx} rows and to attend through the step's page tables; null means the contiguous step,
     * whose store and attend rules the mixer runs unchanged. Default: always null, the contiguous form.
     */
    default PagedStep currentPagedStep() {
        return null;
    }

    /**
     * Runs one page copy of the copy-on-write protocol on this memory's storage: every cache layer of the copy's
     * KV group moves rows {@code [srcPage * pageSize + srcRow0, .. + rows)} to the same rows of the destination
     * page.
     *
     * @throws UnsupportedOperationException for a memory without paged sequences
     * @throws IllegalArgumentException      when no cache layer belongs to the copy's group
     */
    default void copyPage(PageCopy copy) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " holds no paged sequences");
    }

    /**
     * Why {@link #movePositionRows} would refuse, or null when it can run. Default: refused, a memory whose rows
     * address no re-addressable positions (a memory with its own paged sequences moves rows through
     * {@code PagedSequences} instead).
     */
    default String movePositionRowsRefusal(int[] srcPos, int[] dstPos) {
        return getClass().getSimpleName() + " moves no rows of its contiguous sequence";
    }

    /**
     * Moves the contiguous sequence's row of position {@code srcPos[i]} onto the row of position
     * {@code dstPos[i]}, the row taking the destination's position (the {@code SequenceMemory#moveRows} primitive
     * over a single contiguous sequence): every cache layer of every KV group carries the row's bytes, the
     * {@code GroupPositions} mirrors follow, and the token record is the caller's concern. Default: refused.
     *
     * @throws UnsupportedOperationException carrying {@link #movePositionRowsRefusal(int[], int[])}
     */
    default void movePositionRows(int[] srcPos, int[] dstPos) {
        throw new UnsupportedOperationException(movePositionRowsRefusal(srcPos, dstPos));
    }

    /**
     * Runs what a sequence-level compaction queued on this memory: the pending copy-on-write page copies, then the
     * pending row moves ({@code PagedSequences#pendingRowMoves()}), so the queued bytes land before the caller
     * goes on. A no-op when the memory keeps nothing queued (a memory whose row moves are immediate, or a device
     * memory whose backend carries them). Default: nothing to carry.
     */
    default void carryRowMoves() {
    }

    /** Empties all caches and zeroes all states. */
    void reset();

    /** Whether the last {@code nTokens} positions can be dropped. */
    boolean canRollback(int nTokens);

    /**
     * Drops the last {@code nTokens} positions.
     *
     * @throws IllegalStateException when {@link #canRollback(int)} is false for {@code nTokens}
     */
    void rollback(int nTokens);

    /**
     * Drops the last {@code nTokens} positions without touching the recurrent state, whose rows the positions after
     * the drop re-evaluate: the Step5 region summaries and decode ring ({@link MemoryPlan#hasOnlyPositionalState()}).
     * A summary whose region the dropped rows had closed is rewritten when the region closes again, and ring rows
     * are rewritten by the positions they hold. Used where a memory must rewind rows it will re-write (a draft
     * context's own caches), which {@link #rollback(int)} refuses for recurrent state.
     *
     * @throws UnsupportedOperationException when the plan keeps no positional state, or keeps other recurrent state
     *                                       or DSV4 compressor state
     * @throws IllegalStateException         when fewer than {@code nTokens} positions are stored
     */
    default void dropRecomputed(int nTokens) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " cannot drop positions without"
                + " restoring its recurrent state");
    }

    /**
     * Drops the last {@code nTokens} positions' cache rows and truncates the sequence without touching the
     * recurrent layer state, whose values for the accepted tokens the caller re-derives right after: the recurrent
     * tree verify's commit drops the round's dead rows, then replays the accepted path through the recurrent
     * layers against the committed state, leaving the state paired with the truncated sequence. DSV4 compressor
     * state and positional state the dropped rows feed are refused: the replay re-derives only the recurrent
     * layers' state. The drop keeps every sliding-window ring holding the keys the query at the new end attends
     * ({@link #ringsKeepWindow}, like {@link #rollback(int)}), which the proposal length already asked for.
     *
     * @throws UnsupportedOperationException when the memory keeps DSV4 compressor or positional state
     * @throws IllegalStateException         when fewer than {@code nTokens} positions are stored, or the drop would
     *                                       empty a ring of keys the next query attends
     */
    default void dropKeepRecurrent(int nTokens) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " cannot drop cache rows while keeping"
                + " its recurrent state");
    }

    /**
     * Whether, after {@code nWrite} more positions are stored from {@code nPast}, dropping the last {@code nDrop}
     * positions (by {@link #rollback(int)} or {@link #restore()}) would leave every sliding-window ring holding each
     * key the query at the new end attends. Speculative decoding asks this before verifying a draft: writing
     * {@code k + 1} tokens and rejecting up to {@code k} of them is safe when {@code ringsKeepWindow(k + 1, k)}.
     * Only the KV rows decide it; recurrent and DSV4 compressor state are not considered. Memory without rings
     * always answers true; this default, which cannot see the stored rows, answers false for a drop on rings.
     *
     * @throws IllegalArgumentException for negative counts or {@code nDrop > nPast + nWrite}
     */
    default boolean ringsKeepWindow(int nWrite, int nDrop) {
        if (nWrite < 0 || nDrop < 0 || (long) nDrop > (long) sequence().nPast() + nWrite) {
            throw new IllegalArgumentException("cannot store " + nWrite + " and drop " + nDrop + " of "
                    + sequence().nPast() + " stored positions");
        }
        return nDrop == 0 || !plan().hasRing();
    }

    /** True when the model keeps recurrent state. */
    default boolean hasRecurrentState() {
        return plan().hasRecurrentState();
    }

    /** Whether {@link #snapshot()} can be taken: the plan reserved shadow buffers for every recurrent state. */
    default boolean canSnapshot() {
        return plan().snapshotsReserved();
    }

    /**
     * Saves every recurrent state (and the sequence length) for a later {@link #restore()}.
     *
     * @throws UnsupportedOperationException when {@link #canSnapshot()} is false: no recurrent state, DSV4
     *                                       compressor state, or no shadow buffers reserved
     */
    void snapshot();

    /**
     * Restores the state saved by the last {@link #snapshot()}.
     *
     * @throws IllegalStateException when no snapshot exists
     */
    void restore();

    /**
     * Why this memory cannot copy, restore and clear the recurrent state of one paged sequence
     * ({@link #saveSequenceState}, {@link #restoreSequenceState}, {@link #clearSequenceState}), or null when it can.
     * A memory without recurrent state always can: there is nothing to copy. Default: refused for recurrent state.
     */
    default String sequenceStateRefusal() {
        return hasRecurrentState() ? getClass().getSimpleName() + " holds no per-sequence recurrent state rows"
                : null;
    }

    /**
     * Copies the recurrent state row of paged sequence {@code seq} of every state tensor
     * ({@link RecurrentStateCopy#EMPTY} without recurrent state).
     *
     * @throws UnsupportedOperationException carrying {@link #sequenceStateRefusal()} when it is not null
     */
    default RecurrentStateCopy saveSequenceState(int seq) {
        String refusal = sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        return RecurrentStateCopy.EMPTY;
    }

    /**
     * Writes {@code copy} (taken by {@link #saveSequenceState} on this memory) into the recurrent state row of
     * paged sequence {@code seq}; the other sequences' rows stay.
     *
     * @throws UnsupportedOperationException carrying {@link #sequenceStateRefusal()} when it is not null
     * @throws IllegalArgumentException      when the copy's rows do not match this memory's state tensors
     */
    default void restoreSequenceState(int seq, RecurrentStateCopy copy) {
        String refusal = sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (copy.rows() != 0) {
            throw new IllegalArgumentException("a memory without recurrent state restores no state rows, got "
                    + copy);
        }
    }

    /**
     * Zeroes the recurrent state row of paged sequence {@code seq}, so the next sequence in the slot starts from a
     * fresh state. A no-op without recurrent state.
     *
     * @throws UnsupportedOperationException carrying {@link #sequenceStateRefusal()} when it is not null
     */
    default void clearSequenceState(int seq) {
        String refusal = sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
    }

    /**
     * The host transfer of this memory's stored state: raw K/V rows, recurrent state tensors and position mirrors
     * ({@link StateTransfer}), the primitive of session files, slot files and the host-RAM prompt cache. Default:
     * refused, a memory without host transfer.
     *
     * @throws UnsupportedOperationException when this memory holds no host transfer
     */
    default StateTransfer stateTransfer() {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " holds no host state transfer");
    }

    /**
     * Writes the session file of the held contiguous sequence to {@code out} ({@code llama_state_save_file},
     * {@link SessionFile}): its tokens, position mirrors, K/V rows and recurrent states, whichever backend holds
     * them.
     *
     * @throws UnsupportedOperationException when the memory is outside the session-file scope
     * @throws IOException                   when the write fails
     */
    default void saveSession(Path out) throws IOException {
        SessionFile.save(this, out);
    }

    /**
     * Reads the session file at {@code in} into this memory ({@code llama_state_load_file}, {@link SessionFile}),
     * resetting it first; the file may come from either backend. The next decode continues the saved sequence.
     *
     * @throws UnsupportedOperationException when the memory is outside the session-file scope
     * @throws IOException                   for an unreadable file, a foreign format or a plan mismatch
     */
    default void loadSession(Path in) throws IOException {
        SessionFile.load(this, in);
    }

    /**
     * Why this memory cannot copy and restore the DSV4 compressed streams of one paged sequence
     * ({@link #saveCompressedState}, {@link #restoreCompressedState}), or null when it can. A memory without
     * compressed caches always can: there is nothing to copy. Default: refused.
     */
    default String compressedStateRefusal() {
        return plan().hasCompressed() ? getClass().getSimpleName() + " cannot copy a sequence's DSV4 compressed"
                + " streams" : null;
    }

    /**
     * Copies the DSV4 compressed streams of paged sequence {@code seq}: every compressed cache's compressed rows,
     * both compressor tensors with all their planes, and the sequence's compressor rollback record
     * ({@link CompressedStateCopy#EMPTY} without compressed caches).
     *
     * @throws UnsupportedOperationException carrying {@link #compressedStateRefusal()} when it is not null
     */
    default CompressedStateCopy saveCompressedState(int seq) {
        String refusal = compressedStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        return CompressedStateCopy.EMPTY;
    }

    /**
     * Writes {@code copy} (taken by {@link #saveCompressedState} on this memory) into the DSV4 compressed streams
     * of paged sequence {@code seq}: the compressed rows, every compressor plane, and the rollback record; the
     * other sequences' streams stay.
     *
     * @throws UnsupportedOperationException carrying {@link #compressedStateRefusal()} when it is not null
     * @throws IllegalArgumentException      when the copy's streams do not match this memory's compressed caches
     */
    default void restoreCompressedState(int seq, CompressedStateCopy copy) {
        String refusal = compressedStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        if (copy.streams() != 0) {
            throw new IllegalArgumentException("a memory without compressed caches restores no compressed streams,"
                    + " got " + copy);
        }
    }

    /** Releases all buffers. */
    @Override
    void close();
}
