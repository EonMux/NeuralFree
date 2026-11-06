package com.eonmux.neuralfree.gguf.engine.draft;

/**
 * The committed token record of one sequence that a {@link HistoryDrafter} drafts over: the single-sequence
 * target's stored tokens, or one batch sequence's stored text tokens in the paged memory. Read on the drafting
 * thread at every draft.
 */
@FunctionalInterface
public interface TokenHistory {

    /** A copy of the sequence's committed tokens, oldest first. */
    int[] tokens();
}
