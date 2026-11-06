package com.eonmux.neuralfree.gguf.ops.moe;

import java.util.Objects;

/**
 * Avalon's hosted tail-skip rule (Strata's {@code STRATA_ROUTE_TAIL_SKIP},
 * ggml/src/ggml-cuda/moe-cache.cu:6805-6860): a routed expert that the GPU expert cache does not hold and that
 * every token of the step routes only at rank {@code rank} or lower (its 0-based position in the top-k, by weight)
 * is not computed on the host. Its expert output rows stay zero and the other experts keep the weights the router
 * gave them, so there is no renormalisation. Cached experts are always computed, and the rule never drops a pair
 * some token routes at a better rank. NeuralFree keeps the rule opt-in
 * ({@code neuralfree.moe.cache.tail.skip}); Avalon turns it on with rank 7.
 */
public final class MoeTailSkip {

    /** Rank that turns the rule off. */
    public static final int OFF = 0;
    /** Highest accepted rank (Avalon accepts 1 to 9). */
    public static final int MAX_RANK = 9;

    private MoeTailSkip() {
    }

    /**
     * Marks the experts of a step that the rule drops.
     *
     * @param ids     routed ids of the step, token-major ({@code t * k + j})
     * @param nIds    routed pairs of the step
     * @param k       experts selected per token
     * @param rank    lowest 0-based top-k position the rule drops; {@link #OFF} disables it
     * @param cached  per expert, whether the cache holds it at the start of the step
     * @param nExpert experts of the layer
     * @return a mask over {@code [0, nExpert)}: true for an uncached expert every one of its pairs routes at
     *         position {@code rank} or lower; all false when the rule is off or the step routes nothing
     * @throws IllegalArgumentException for a pair count or top-k that does not divide it, a rank above
     *                                  {@link #MAX_RANK}, or an id outside the experts
     */
    public static boolean[] skippedExperts(int[] ids, int nIds, int k, int rank, boolean[] cached, int nExpert) {
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(cached, "cached");
        if (nIds < 0 || nIds > ids.length) {
            throw new IllegalArgumentException("pair count " + nIds + " outside [0, " + ids.length + "]");
        }
        if (k < 1 || nIds % k != 0) {
            throw new IllegalArgumentException(nIds + " routed ids are not a whole number of " + k
                    + "-expert tokens");
        }
        if (nExpert < 1 || cached.length < nExpert) {
            throw new IllegalArgumentException("expert mask holds " + cached.length + " experts, the layer has "
                    + nExpert);
        }
        if (rank < OFF || rank > MAX_RANK) {
            throw new IllegalArgumentException("tail-skip rank must be " + OFF + " (off) or 1-" + MAX_RANK
                    + ", got " + rank);
        }
        boolean[] skip = new boolean[nExpert];
        if (rank == OFF || nIds == 0) {
            return skip;
        }
        boolean[] seen = new boolean[nExpert];
        for (int p = 0; p < nIds; p++) {
            int e = checkId(ids[p], nExpert);
            if (seen[e] || cached[e]) {
                continue;
            }
            seen[e] = true;
            skip[e] = tailOnly(ids, nIds, k, e, rank);
        }
        return skip;
    }

    /** Whether every pair of expert {@code e} sits at position {@code rank} or lower. */
    private static boolean tailOnly(int[] ids, int nIds, int k, int e, int rank) {
        for (int q = 0; q < nIds; q++) {
            if (ids[q] == e && q % k < rank) {
                return false;
            }
        }
        return true;
    }

    private static int checkId(int e, int nExpert) {
        if (e < 0 || e >= nExpert) {
            throw new IllegalArgumentException("routed expert " + e + " outside [0, " + nExpert + ")");
        }
        return e;
    }
}
