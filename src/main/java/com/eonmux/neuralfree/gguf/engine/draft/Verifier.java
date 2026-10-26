package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.StopReason;

/**
 * One verify round's decision maker: a {@link VerifyMode} other than {@link VerifyMode#MATCH} decides the emissions
 * of a proposal through this interface instead of the sampler's own draw ({@code SpeculativeDecoder}). The
 * implementation drives the round over {@link Round}: it asks for the target's processed distributions, chooses per
 * draft position whether that position is confirmed, and emits replacements and bonuses through the round, which
 * routes every emission through the same session as a sampled token.
 *
 * <p>Implementations hold no per-round state; one instance is shared by a decoder's rounds and generations. Not
 * thread-safe by way of the round they run on the backend's thread.
 */
public interface Verifier {

    /** Decides and emits one verify round. */
    void verify(Round round);

    /**
     * The driver's view of one verify round. An implementation must ask for the target rows in ascending order and
     * stop as soon as an emission reports a {@link StopReason}; a confirm's {@code position} must be the round's
     * current accepted count.
     */
    interface Round {

        /** Number {@code k} of proposed tokens; verify rows {@code 0 .. k} exist. */
        int draftCount();

        /** The proposed token at {@code position} ({@code 0 .. draftCount() - 1}). */
        int draftToken(int position);

        /**
         * The target's processed distribution of verify row {@code row} (which validates draft token {@code row}):
         * each row at most once, and only in ascending order (row {@code n} after a row below {@code n} was last
         * asked). Row {@code i} answers with the distribution the target chain gives the token after the first
         * {@code i} batched tokens.
         */
        Distribution target(int row);

        /** The drafter's proposal distribution at draft {@code position} (one-hot for drafters without probabilities). */
        Distribution draft(int position);

        /** One uniform draw in {@code [0, 1)} from the round's RNG. */
        double uniform();

        /** Emits draft token {@code position} as confirmed; returns the stop reason, or null to continue. */
        StopReason confirm(int position);

        /** Emits any other token (a rejection's replacement or a bonus); returns the stop reason, or null to continue. */
        StopReason emit(int token);

        /** Number of draft tokens confirmed so far. */
        int accepted();

        /** Whether an emission reported a stop reason. */
        boolean stopped();
    }

    /**
     * A processed distribution of one row: its surviving candidates with token ids, the probabilities the chain's
     * transformations left, their logits and the chain's own draw. Immutable. A greedy chain's probabilities are
     * one-hot on the argmax; the logits then still carry every candidate's post-chain logit for FLy's entropy gate.
     */
    final class Distribution {

        private final int[] tokens;
        private final float[] probs;
        private final float[] logits;
        private final int selected;

        /**
         * @param tokens   the candidate token ids (copied)
         * @param probs    each candidate's processed probability (copied; non-negative, summing to more than 0)
         * @param logits   each candidate's processed logit (may be {@code -inf}; never NaN)
         * @param selected index of the chain's own draw, or -1 when the distribution has no draw
         * @throws IllegalArgumentException when the arrays mismatch, hold negatives/NaN, sum to 0 or {@code selected} is out of range
         */
        public Distribution(int[] tokens, float[] probs, float[] logits, int selected) {
            if (tokens == null || probs == null || logits == null) {
                throw new IllegalArgumentException("a distribution needs token, probability and logit arrays");
            }
            int n = tokens.length;
            if (probs.length != n || logits.length != n) {
                throw new IllegalArgumentException("the distribution arrays disagree: " + tokens.length + " tokens, "
                        + probs.length + " probabilities, " + logits.length + " logits");
            }
            if (n == 0) {
                throw new IllegalArgumentException("a distribution needs at least one candidate");
            }
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                float p = probs[i];
                if (Float.isNaN(p) || p < 0f) {
                    throw new IllegalArgumentException("candidate " + i + " has probability " + p);
                }
                if (Float.isNaN(logits[i])) {
                    throw new IllegalArgumentException("candidate " + i + " has a NaN logit");
                }
                sum += p;
            }
            if (sum <= 0.0) {
                throw new IllegalArgumentException("the distribution probabilities sum to " + sum);
            }
            if (selected < -1 || selected >= n) {
                throw new IllegalArgumentException("selected candidate " + selected + " outside [0, " + n + ")");
            }
            this.tokens = tokens.clone();
            this.probs = probs.clone();
            this.logits = logits.clone();
            this.selected = selected;
        }

        /** The chain's own draw: offer of a single candidate {@code token} with probability 1. */
        public static Distribution oneHot(int token) {
            return new Distribution(new int[]{token}, new float[]{1f}, new float[]{0f}, 0);
        }

        /** Number of surviving candidates. */
        public int size() {
            return tokens.length;
        }

        /** Candidate {@code i}'s token id. */
        public int token(int i) {
            return tokens[i];
        }

        /** Candidate {@code i}'s processed probability. */
        public float prob(int i) {
            return probs[i];
        }

        /** Candidate {@code i}'s processed logit. */
        public float logit(int i) {
            return logits[i];
        }

        /** Index of the candidate with token id {@code token}, or -1 outside the support. */
        public int indexOf(int token) {
            for (int i = 0; i < tokens.length; i++) {
                if (tokens[i] == token) {
                    return i;
                }
            }
            return -1;
        }

        /** Probability of token id {@code token}; 0 outside the support. */
        public float probOf(int token) {
            int i = indexOf(token);
            return i < 0 ? 0f : probs[i];
        }

        /** Processed logit of token id {@code token}; {@code -inf} outside the support. */
        public float logitOf(int token) {
            int i = indexOf(token);
            return i < 0 ? Float.NEGATIVE_INFINITY : logits[i];
        }

        /** Index of the chain's own draw, or -1. */
        public int selected() {
            return selected;
        }

        /** The chain's own draw's token id. */
        public int selectedToken() {
            if (selected < 0) {
                throw new IllegalStateException("the distribution has no chain draw");
            }
            return tokens[selected];
        }

        @Override
        public String toString() {
            StringBuilder s = new StringBuilder("Distribution{");
            for (int i = 0; i < tokens.length; i++) {
                if (i > 0) {
                    s.append(", ");
                }
                s.append(tokens[i]).append('=').append(probs[i]).append(" <- ").append(logits[i]);
            }
            return s.append(", selected=").append(selected).append('}').toString();
        }
    }

    /**
     * Draws one rejection's replacement token from the residual {@code norm(max(0, a·p − b·q))} (the true residual
     * has {@code a = b = 1}; block verification's correction scales the pair), with one uniform draw from
     * {@code round}. Candidates are visited in the target row's candidate order; probability 0 terms never win. A
     * residual with no positive term anywhere ({@code p = q} and {@code a = b}) falls back to the chain's own draw.
     */
    static int residual(Round round, Distribution p, Distribution q, double a, double b) {
        int n = p.size();
        double[] weights = new double[n];
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            double w = a * p.prob(i) - b * q.probOf(p.token(i));
            if (w > 0.0) {
                weights[i] = w;
                total += w;
            }
        }
        if (total <= 0.0) {
            return p.selectedToken();
        }
        double target = total * round.uniform();
        double run = 0.0;
        int lastPositive = -1;
        for (int i = 0; i < n; i++) {
            if (weights[i] > 0.0) {
                lastPositive = i;
                run += weights[i];
                if (run >= target) {
                    return p.token(i);
                }
            }
        }
        return p.token(lastPositive);
    }
}
