package com.eonmux.neuralfree.demo.tools.eval;

/**
 * The questions of one task-accuracy suite, in a stable order. The runner shuffles the indices per
 * pass, so a dataset needs no randomness of its own.
 */
public interface EvalDataset {

    /** The suite's name ({@code gsm8k}, {@code aime}, {@code aime2025} or {@code aime2026}). */
    String type();

    /** The number of distinct questions. */
    int size();

    /**
     * The question at {@code index}.
     *
     * @throws IndexOutOfBoundsException for an index outside {@code [0, size())}
     */
    EvalQuestion question(int index);
}
