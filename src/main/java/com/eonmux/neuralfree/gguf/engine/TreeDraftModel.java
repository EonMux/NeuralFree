package com.eonmux.neuralfree.gguf.engine;

import java.util.SplittableRandom;

/**
 * A {@link DraftModel} that proposes a {@link DraftTree} over a paged draft memory with several sequences: the
 * multi-branch speculation of llama.cpp's {@code examples/speculative/speculative.cpp}. An implementing drafter
 * binds the target and evaluates one branch per draft sequence, splitting a branch into a new one when a secondary
 * candidate's probability exceeds the configured split probability.
 *
 * <p>The decoder asks {@link #draftTree} in place of {@link #draft} when the target memory holds more than one
 * sequence; {@link #acceptTree} reports how many drafted tokens the round confirmed so a drafter with caches can
 * realign them.
 */
public interface TreeDraftModel extends DraftModel {

    /**
     * Proposes a tree after {@code lastToken}, the token the target just sampled.
     *
     * @param lastToken   the target's pending token
     * @param maxTokens   the most draft depths to propose (the linear chain's node count)
     * @param branches    the draft sequences the proposal may use, at least one
     * @param temperature the request's sampling temperature, 0 or less drafting greedily
     * @param rng         the request's draft stream, used when {@code temperature} is positive
     * @return the proposal, possibly empty
     */
    DraftTree draftTree(int lastToken, int maxTokens, int branches, float temperature, SplittableRandom rng);

    /**
     * {@link #draftTree(int, int, int, float, SplittableRandom)} with an upper bound on the nodes one single-pass
     * verification lays the proposal out over (the pending token's own row excluded). A drafter whose verify runs
     * one forward per depth ignores it; a single-pass drafter keeps its branching proposal's node count within the
     * bound instead, so the round never needs more rows than one ubatch holds. The default delegates without the
     * bound.
     */
    default DraftTree draftTree(int lastToken, int maxTokens, int branches, int maxRows, float temperature,
                                SplittableRandom rng) {
        return draftTree(lastToken, maxTokens, branches, temperature, rng);
    }

    /**
     * Whether the decoder verifies this drafter's branching proposals in one target forward (a layout over the
     * target's page-shared sequences), instead of one forward per depth. Single-pass drafters keep the proposal
     * within {@code maxRows} rows through {@link #draftTree(int, int, int, int, float, SplittableRandom)}.
     */
    default boolean singlePassVerify() {
        return false;
    }

    /**
     * Reinjects the accepted rows of a single-pass verify from the forward that computed them, so the commit needs
     * no re-decode: {@code verifyRow[i]} names the batch row of the accepted path's depth {@code i} (0 for the
     * pending token's row, a node's row index + 1 for the depths above) and {@code pos0} is the position the path's
     * first row commits at. A drafter that reads the target's hidden rows ({@code DraftModel#readsTargetHidden()})
     * captures the verify forward's rows while its tap is suspended ({@code DraftModel#featureTapSuspended}) and
     * gathers exactly the accepted ones here, after the KV compaction has landed them at their final positions.
     * The default ignores the call (a drafter without target-hidden injection).
     *
     * @throws IllegalStateException when no verify forward's rows were captured or a row is outside them
     */
    default void injectVerifyRows(int[] verifyRow, int pos0) {
    }

    /** The draft sequences the bound target's paged memory holds, or 0 when the drafter proposes linear chains. */
    int branches();

    /** Reports verification of the last {@link #draftTree}: its first {@code nAccepted} depth levels were kept. */
    void acceptTree(int nAccepted);
}
