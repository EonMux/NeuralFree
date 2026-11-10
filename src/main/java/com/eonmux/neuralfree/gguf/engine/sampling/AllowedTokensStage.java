package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

import java.util.Arrays;
import java.util.Objects;

/**
 * The allowed-token-ids stage of a request (vLLM {@code allowed_token_ids}): every candidate id outside
 * {@code allowed} takes a {@code -inf} logit. A masking stage ({@link #masks()}): the chain applies it from the
 * constraint head, before the configured chain, so a banned id never reaches a truncating stage and a truncation
 * cannot cut the allowed set out. Immutable apart from the shared candidate set.
 */
public final class AllowedTokensStage extends SamplerStage implements TokenGate {

    private final int[] allowed;

    /** Copies and sorts {@code allowedTokenIds}; ids must be non-negative and the list non-empty. */
    public AllowedTokensStage(int[] allowedTokenIds) {
        Objects.requireNonNull(allowedTokenIds, "allowedTokenIds");
        if (allowedTokenIds.length == 0) {
            throw new IllegalArgumentException("allowed token ids must not be empty");
        }
        for (int id : allowedTokenIds) {
            if (id < 0) {
                throw new IllegalArgumentException("allowed token id " + id + " is negative");
            }
        }
        int[] sorted = allowedTokenIds.clone();
        Arrays.sort(sorted);
        int distinct = 0;
        for (int i = 0; i < sorted.length; i++) {
            if (distinct == 0 || sorted[distinct - 1] != sorted[i]) {
                sorted[distinct++] = sorted[i];
            }
        }
        this.allowed = distinct == sorted.length ? sorted : Arrays.copyOf(sorted, distinct);
    }

    /** The allowed ids (a copy). */
    public int[] allowed() {
        return allowed.clone();
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        int next = 0;
        for (int i = 0; i < candidates.size(); i++) {
            int id = candidates.id(i);
            while (next < allowed.length && allowed[next] < id) {
                next++;
            }
            if (next >= allowed.length || allowed[next] != id) {
                candidates.setLogit(i, Float.NEGATIVE_INFINITY);
            }
        }
    }

    /** Whether {@code token} is one of the allowed ids. */
    @Override
    public boolean keeps(int token) {
        return Arrays.binarySearch(allowed, token) >= 0;
    }

    @Override
    public boolean masks() {
        return true;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof AllowedTokensStage && Arrays.equals(allowed, ((AllowedTokensStage) o).allowed);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(allowed);
    }

    @Override
    public String toString() {
        return "AllowedTokensStage{allowed=" + allowed.length + "}";
    }
}
