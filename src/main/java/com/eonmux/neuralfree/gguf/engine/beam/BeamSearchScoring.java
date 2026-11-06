package com.eonmux.neuralfree.gguf.engine.beam;

/**
 * The beam-search score (vLLM's {@code get_beam_search_score}, Transformers' length-penalized rank): a beam's
 * cumulative log probability divided by its generated-token count to the {@code lengthPenalty} power, the count
 * including a terminating EOS. A {@code lengthPenalty} above 0 favors longer beams, below 0 shorter ones, and 0
 * ranks by the raw cumulative log probability.
 */
public final class BeamSearchScoring {

    private BeamSearchScoring() {
    }

    /**
     * The score of a beam with {@code cumulativeLogprob} over {@code generatedTokens} scored tokens; a beam that
     * has scored none yet scores {@code cumulativeLogprob} (vLLM divides by {@code max(n, 1)} so an aborted beam
     * with no token keeps a finite key).
     */
    public static double score(int generatedTokens, double cumulativeLogprob, float lengthPenalty) {
        double n = Math.pow(Math.max(generatedTokens, 1), lengthPenalty);
        return cumulativeLogprob / n;
    }

    /** Compares two scores for ranking: the larger first. */
    public static int compare(double a, double b) {
        return Double.compare(b, a);
    }
}
