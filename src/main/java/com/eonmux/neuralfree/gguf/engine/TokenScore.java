package com.eonmux.neuralfree.gguf.engine;

import java.util.Objects;

/**
 * One token's log probability of a sampled step's post-chain distribution (llama.cpp's
 * {@code n_probs} array after the chain): the token id and {@code ln(p)} of its probability among
 * the chain's surviving candidates. A capture's first entry is the sampled token.
 */
public final class TokenScore {

    private final int tokenId;
    private final float logprob;

    public TokenScore(int tokenId, float logprob) {
        this.tokenId = tokenId;
        this.logprob = logprob;
    }

    /** The token id. */
    public int tokenId() {
        return tokenId;
    }

    /** {@code ln(p)} of the token's probability ({@code -Infinity} for a zero probability). */
    public float logprob() {
        return logprob;
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
