package com.eonmux.neuralfree.gguf.engine;

/**
 * A speculative-decoding drafter (NextN/MTP head, eagle3, gemma4-assistant, dflash) bound to one target model.
 *
 * <p>Protocol per generation: {@link #reset()}, then repeatedly {@link #draft} after the target sampled a token,
 * the target verifies the proposal with {@link ForwardBackend#forwardRows}, and {@link #accept} reports how many
 * proposed tokens were kept. The drafter tracks its own position from the target's sequence.
 */
public interface DraftModel extends AutoCloseable {

    /**
     * Binds this drafter to its target: shared memory, hidden taps and vocabulary mapping. Called once, before
     * the first {@link #draft}.
     *
     * @throws IllegalArgumentException when the target is incompatible (architecture, vocabulary, width), with
     *                                  the reason
     */
    void bindTarget(EngineModel target);

    /**
     * Proposes the tokens following {@code lastToken}, the token the target just sampled.
     *
     * @param n the maximum number of tokens to propose
     * @return between 0 and {@code n} proposed token ids, in target vocabulary
     */
    int[] draft(int lastToken, int n);

    /**
     * Reports verification of the last {@link #draft}: its first {@code nAccepted} tokens were kept and the rest
     * discarded, so the drafter rolls back any state it advanced for them.
     */
    void accept(int nAccepted);

    /** Starts a new sequence; drops every per-sequence state. */
    default void reset() {
    }

    /** Releases the drafter's resources. */
    @Override
    default void close() {
    }
}
