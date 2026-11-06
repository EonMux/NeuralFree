package com.eonmux.neuralfree.gguf.engine.beam;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The outcome of one beam search: the best beams in ranking order (the length-penalized score, best first), the
 * list as long as the search still had sequences to rank — at most the beam width. Immutable.
 */
public final class BeamSearchOutput {

    private final int promptTokens;
    private final List<BeamSearchSequence> sequences;
    private final int steps;

    BeamSearchOutput(int promptTokens, List<BeamSearchSequence> sequences, int steps) {
        this.promptTokens = promptTokens;
        this.sequences = Collections.unmodifiableList(new ArrayList<>(sequences));
        this.steps = steps;
    }

    /** The prompt's token count the beams grew from. */
    public int promptTokens() {
        return promptTokens;
    }

    /** The best beams, best first ({@link BeamSearchScoring}). */
    public List<BeamSearchSequence> sequences() {
        return sequences;
    }

    /** How many scored steps the search ran before it stopped. */
    public int steps() {
        return steps;
    }

    @Override
    public String toString() {
        return "BeamSearchOutput{" + sequences.size() + " beams over " + promptTokens + " prompt tokens, "
                + steps + " steps}";
    }
}
