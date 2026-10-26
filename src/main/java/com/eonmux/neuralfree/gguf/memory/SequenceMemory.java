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

    /** The host token record of sequence {@code seq} (positions sit at {@code [posMin, posMax]}).
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

    /** Why {@code copy} would refuse on this memory, or null when it can copy between sequences. */
    String copyRefusal();

    /**
     * Copies positions {@code [p0, p1)} (or up to {@code posMax(src)} for {@code p1 = -1}) of sequence {@code src}
     * to the end of sequence {@code dst}: page sharing, not a byte copy (invariant 5).
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

    /**
     * Cells of the memory that hold no position, summed over its KV groups. For paged memories these are the free
     * rows of the page pools; for contiguous (single-sequence) memory the rows after the stored range.
     */
    int freeCells();

    /** Cells of the memory, summed over its KV groups ({@code P} per page, one page pool per KV group). */
    int totalCells();
}
