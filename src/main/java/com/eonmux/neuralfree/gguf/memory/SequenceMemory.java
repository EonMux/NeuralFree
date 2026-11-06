package com.eonmux.neuralfree.gguf.memory;

/**
 * The sequence-addressing API of a memory: the {@code llama_memory_seq_*} analog over one or more sequences of KV
 * cells. The generation engine drives it per request ({@code remove} for rollback, {@code copy} for shared
 * prompts, {@code keep} and so on), while the storage of pages and cells stays inside the memory implementation.
 *
 * <p>Position ranges are stored positions, {@code p0} inclusive and {@code p1} exclusive; {@code p1 = -1} means up
 * to and including {@link #posMax(int)}. Every mutating method refuses (an {@code UnsupportedOperationException}
 * carrying the reason) when the memory implementation cannot serve it for its storage form; the
 * {@code *Refusal} methods report the reason in advance. Implemented by
 * {@link SingleSequenceMemory}, the adapter over today's contiguous single-sequence {@link ModelMemory}.</p>
 *
 * <p>Used by the generation thread of the model this memory belongs to; not thread-safe.</p>
 */
public interface SequenceMemory {

    /** Sequences the memory can hold simultaneously (llama.cpp {@code n_seq_max}). */
    int maxSequences();

    /**
     * The host token record of sequence {@code seq} (positions sit at {@code [posMin, posMax]}). On paged memory the
     * record follows the rows: {@link #remove}, {@link #copy} and {@link #add} drop, copy and move its entries as
     * they do the rows' cells.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside {@link #maxSequences()}
     */
    SequenceState sequence(int seq);

    /**
     * The lowest stored position of sequence {@code seq}, or -1 when it holds nothing. Contiguous memory stores
     * positions from 0; paged memory reports the first row the sequence's page list covers.
     */
    int posMin(int seq);

    /** The highest stored position of sequence {@code seq}, or -1 when it holds nothing. */
    int posMax(int seq);

    /**
     * Why {@code remove(seq, p0, p1)} would refuse, or null when it can run.
     *
     * @param seq the sequence
     * @param p0  first position (inclusive)
     * @param p1  first position after the range, or -1 for up to and including {@code posMax(seq)}
     */
    String removeRefusal(int seq, int p0, int p1);

    /**
     * Removes positions {@code [p0, p1)} (or {@code [p0, posMax(seq)]} for {@code p1 = -1}) of sequence
     * {@code seq}: the cells of whole pages leave the sequence, partial pages get their rows marked empty.
     *
     * @throws UnsupportedOperationException when {@link #removeRefusal(int, int, int)} answers a reason
     */
    void remove(int seq, int p0, int p1);

    /**
     * Why {@code removeKeepRecurrent(seq, p0)} would refuse, or null when it can run.
     *
     * @param seq the sequence
     * @param p0  first position (inclusive); the range runs to {@code posMax(seq)}
     */
    String removeKeepRecurrentRefusal(int seq, int p0);

    /**
     * Removes the positions {@code [p0, posMax(seq)]} of sequence {@code seq} without touching the recurrent layer
     * state, whose accepted-token values the caller re-derives right after: the recurrent tree verify's commit
     * drops the round's dead rows and then replays the accepted path against the committed state. A plain
     * {@link #remove} of these positions would refuse, or rewind state, because the memory could no longer pair
     * its state with its sequence; this method's caller takes over that pairing. DSV4 compressor state and
     * positional state the dropped rows feed are refused: the replay cannot re-derive them.
     *
     * @throws UnsupportedOperationException when {@link #removeKeepRecurrentRefusal(int, int)} answers a reason
     */
    void removeKeepRecurrent(int seq, int p0);

    /** Why {@code copy} would refuse on this memory, or null when it can copy between sequences. */
    String copyRefusal();

    /**
     * Whether any page of {@code seq} is also held by another sequence, so a removal that cuts inside a shared
     * page ({@code remove(seq, p0, p1)}) would copy that page on write. The paged memory answers from the page
     * reference counts; the contiguous default answers false (no page is shared).
     */
    default boolean sharesPages(int seq) {
        return false;
    }

    /**
     * Copies positions {@code [p0, p1)} (or up to {@code posMax(src)} for {@code p1 = -1}) of sequence {@code src}
     * to the end of sequence {@code dst}: page sharing, not a byte copy.
     *
     * @throws UnsupportedOperationException when {@link #copyRefusal()} answers a reason
     */
    void copy(int src, int dst, int p0, int p1);

    /**
     * Keeps only sequence {@code seq}: every other sequence of the memory becomes empty.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside {@link #maxSequences()}
     */
    void keep(int seq);

    /** Empties every sequence of the memory (all {@code kv_pos} go to -1, states are zeroed). */
    void clear();

    /** Why {@code add} (the position shift with RoPE re-rotation) would refuse, or null when it can shift. */
    String shiftRefusal();

    /**
     * Moves every stored position of sequence {@code seq} in {@code [p0, p1)} (or those at or above {@code p0} for
     * {@code p1 = -1}) by {@code delta}, re-rotating the stored keys: context shift, self-extend and chunk reuse.
     *
     * @throws UnsupportedOperationException when {@link #shiftRefusal()} answers a reason
     */
    void add(int seq, int p0, int p1, int delta);

    /** Why {@code div} (the position division with RoPE re-rotation) would refuse, or null when it can divide. */
    String divRefusal();

    /**
     * Divides every stored position of sequence {@code seq} in {@code [p0, p1)} (or those at or above {@code p0}
     * for {@code p1 = -1}) by {@code d >= 2}, re-rotating the stored keys by each row's own shift
     * ({@code old - old / d}): llama.cpp's {@code llama_memory_seq_div}, the position compression of self-extend
     * (group attention). Several rows can come to share one position.
     *
     * @throws IllegalArgumentException      when {@code p0} is negative or {@code d} is below 2
     * @throws UnsupportedOperationException when {@link #divRefusal()} answers a reason
     */
    void div(int seq, int p0, int p1, int d);

    /**
     * Why {@code moveRows(seq, srcPos, dstPos)} would refuse, or null when it can run.
     *
     * @param seq    the sequence
     * @param srcPos the stored position of each row's source cell, ascending strictly
     * @param dstPos the position each row takes over, ascending strictly, every one below its source
     */
    String moveRowsRefusal(int seq, int[] srcPos, int[] dstPos);

    /**
     * Moves the row stored at {@code srcPos[i]} onto the cell of position {@code dstPos[i]}: the tree verify's
     * commit compaction, which gathers an accepted path's KV rows from their verify slots onto the consecutive
     * positions they occupy once committed (SpecPilot's KV slot permutation). The row's bytes move; its position
     * becomes the destination's, so a row roped at its depth position lands exactly where a linear decode would
     * have computed it and needs no re-rotation. The moves run in list order, each reading only cells no earlier
     * move wrote: the sources ascend above their destinations, so an ascending list never reads a moved cell. The
     * host token record is the caller's concern (it truncates the unaccepted tail and retokenizes the kept entries
     * afterwards); the queue the moves join ({@code pendingRowMoves} on a paged memory) is byte-carried by
     * {@code ForwardBackend#carryRowMoves}.
     *
     * @throws UnsupportedOperationException carrying {@link #moveRowsRefusal(int[], int[])} when the storage form
     *                                        cannot move rows, or when a position is not stored
     * @throws IllegalArgumentException            when the arrays are null, differ in length or break the ordering
     *                                             contract
     */
    void moveRows(int seq, int[] srcPos, int[] dstPos);

    /**
     * Cells of the memory that hold no position, summed over its KV groups. For paged memories these are the free
     * rows of the page pools; for contiguous (single-sequence) memory the rows after the stored range.
     */
    int freeCells();

    /** Cells of the memory, summed over its KV groups ({@code P} per page, one page pool per KV group). */
    int totalCells();
}
