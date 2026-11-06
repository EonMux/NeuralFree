package com.eonmux.neuralfree.gguf.engine.draft;

/**
 * A {@link com.eonmux.neuralfree.gguf.engine.DraftModel} that knows the distribution each of its proposals was
 * drawn from (vLLM's {@code draft_sample_method} probabilistic): rejection sampling then uses the full {@code q}
 * for the ratio test and the residual, instead of the one-hot drafters without probabilities fall back to.
 */
public interface DraftProbs {

    /**
     * The drafter's proposal distribution for the token it proposed at {@code position} of the most recent
     * {@link com.eonmux.neuralfree.gguf.engine.DraftModel#draft} call, over the target's vocabulary: the entry at
     * the proposed token's id is the probability the drafter assigned it. Called between that draft call and the
     * verification of its proposal.
     *
     * @return a probability row over the target's vocabulary (no NaN, no negatives) whose drafted token's entry is
     *         positive, or null when the token was proposed deterministically (a greedy draw), which the verifier
     *         treats as one-hot. A watermarked draft draws every proposal and reports each one.
     * @throws IllegalStateException when the drafter has no proposal at {@code position}
     */
    float[] draftProbs(int position);
}
