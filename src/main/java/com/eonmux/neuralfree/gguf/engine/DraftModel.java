package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;

/**
 * A speculative-decoding drafter (NextN/MTP head, eagle, eagle3, gemma4-assistant, dflash) bound to one target
 * model.
 *
 * <p>Protocol per generation: {@link #beginGeneration(int)} (which drops or keeps the per-sequence state as the
 * target's reused prefix requires), then repeatedly {@link #draft} after the target sampled a token, the target
 * verifies the proposal with {@link ForwardBackend#forwardRows}, and {@link #accept} reports how many proposed
 * tokens were kept. The drafter tracks its own position from the target's sequence.
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
     * {@link #draft(int, int)} for a request that samples at {@code temperature}: a drafter that supports
     * speculative watermarking draws its proposals under the draft role at that temperature, so a confirmed
     * proposal carries the draft key; a greedy request ({@code temperature <= 0}) keeps the ordinary pick. The
     * default is the temperature-independent proposal.
     */
    default int[] draft(int lastToken, int n, float temperature) {
        return draft(lastToken, n);
    }

    /**
     * Whether this drafter can draw its proposals under a draft watermark key ({@link WatermarkConfig}'s draft
     * role): a model drafter that samples its logits does, a lookup or suffix drafter that proposes stored tokens
     * does not. A watermarked speculative request is refused by name on a drafter that cannot.
     */
    default boolean supportsWatermarking() {
        return false;
    }

    /**
     * Prepares draft-side watermarking for a generation: this drafter draws its proposals under
     * {@code config}'s draft role ({@link WatermarkConfig#draftWatermarker()}), keyed on the target's committed
     * tokens and the round's earlier proposals, with {@code seed} driving the ordinary draws of repeated contexts.
     * A null configuration, or one whose draft role has no key, returns the drafter to its ordinary pick.
     *
     * @param promptLength the prompt tokens of the generation, which the {@code single_turn} repeated-context scope
     *                     leaves out of its search
     */
    default void watermark(WatermarkConfig config, int promptLength, long seed) {
    }

    /**
     * Reports verification of the last {@link #draft}: its first {@code nAccepted} tokens were kept and the rest
     * discarded, so the drafter rolls back any state it advanced for them.
     */
    void accept(int nAccepted);

    /**
     * Starts a generation whose target memory already holds the prompt's first {@code reused} tokens (prefix reuse):
     * a drafter with caches of its own keeps the rows of those positions aligned with the target's instead of
     * dropping them, and the target's next forward call starts at {@code reused - replay}, the returned number of
     * stored positions to re-evaluate.
     *
     * <p>A drafter whose state the target's next calls rebuild (a stateless or token-stream drafter, a separate
     * draft model that re-prefills by prefix comparison) drops every per-sequence state and replays nothing. A
     * drafter whose caches cannot keep the prefix asks for the whole prefix back ({@code reused}); the target then
     * evaluates it again so the drafter's rows are rebuilt from its hidden states.
     *
     * <p>A context shift calls this for a drafter that reads the target's hidden rows
     * ({@link #readsTargetHidden()}), after the target shifted its cache: {@code reused} is then the number of
     * positions the shift kept, which hold the same tokens the drafter cached. A drafter that keeps its rows asks
     * for the boundary position back; a drafter that cannot keep them asks for all of them, so the target replays
     * the whole post-shift context through it.
     *
     * <p>The decoder calls this at a generation start in place of {@link #reset()}, and again after a context shift
     * for a drafter that reads the target's hidden rows; it calls {@link #reset()} itself on failures.
     *
     * @param reused the prompt positions the target holds, over the prompt's first {@code reused} tokens; 0 when
     *               the target starts empty
     * @return the number of stored positions to re-evaluate before the tail, in {@code [0, reused]}
     * @throws IllegalArgumentException when {@code reused} is negative
     */
    default int beginGeneration(int reused) {
        if (reused < 0) {
            throw new IllegalArgumentException("reused prefix length " + reused + " is negative");
        }
        reset();
        return 0;
    }

    /** Starts a new sequence; drops every per-sequence state. */
    default void reset() {
    }

    /**
     * Suspends or resumes this drafter's feature injection from the target's hidden rows for one forward call: a
     * single-pass tree verification runs target rows of several proposal branches in one batch, whose rows are not
     * the committed sequence's rows, so a drafter that injects everything a call computes would store rows of
     * branches the round may drop. The decoder suspends the injection around that one batch; the committed rows
     * reach the drafter through the calls before and after it, which replay them on the target's committed
     * sequence. The default ignores the request (a drafter without target-hidden injection).
     */
    default void featureTapSuspended(boolean suspended) {
    }

    /**
     * Whether this drafter rebuilds its state from the target's hidden rows, so a context shift re-evaluates the
     * post-shift window through the target for it ({@link #beginGeneration(int)} decides how much). False for a
     * drafter that reads only the target's tokens or runs a model of its own.
     */
    default boolean readsTargetHidden() {
        return false;
    }

    /** Releases the drafter's resources. */
    @Override
    default void close() {
    }
}
