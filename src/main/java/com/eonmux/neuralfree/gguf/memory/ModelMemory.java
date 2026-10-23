package com.eonmux.neuralfree.gguf.memory;

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

    /** Releases all buffers. */
    @Override
    void close();
}
