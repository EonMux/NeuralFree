package com.eonmux.neuralfree.gguf.engine;

import java.util.Objects;

/**
 * One token's score in a logprobs capture: the token id and its value in the capture's {@link LogprobsMode} —
 * {@code ln(p)} of the raw or the processed distribution, or the raw or processed logit. A generated step's
 * capture ({@link LogprobCapture}) leads with the sampled token, a prompt capture with the prompt token.
 * {@link #rank()} carries the token's vocabulary rank (1 = highest score) when the producer knows it — the
 * leading token and the best alternatives carry it — and -1 when it does not (a requested token id).
 */
public final class TokenScore {

    private final int tokenId;
    private final float logprob;
    private final int rank;

    public TokenScore(int tokenId, float logprob) {
        this(tokenId, logprob, -1);
    }

    /** @param rank the token's 1-based vocabulary rank, or -1 when unknown */
    public TokenScore(int tokenId, float logprob, int rank) {
        this.tokenId = tokenId;
        this.logprob = logprob;
        this.rank = rank;
    }

    /** The token id. */
    public int tokenId() {
        return tokenId;
    }

    /**
     * The token's score: {@code ln(p)} of its probability under a logprobs mode ({@code -Infinity} for a zero
     * probability), its logit under a logits mode ({@code -Infinity} for a removed token).
     */
    public float logprob() {
        return logprob;
    }

    /** The token's 1-based vocabulary rank, or -1 when the producer did not compute it. */
    public int rank() {
        return rank;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof TokenScore)) {
            return false;
        }
        TokenScore t = (TokenScore) o;
        return tokenId == t.tokenId && Float.compare(logprob, t.logprob) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(tokenId, logprob);
    }

    @Override
    public String toString() {
        return tokenId + ":" + logprob;
    }
}
