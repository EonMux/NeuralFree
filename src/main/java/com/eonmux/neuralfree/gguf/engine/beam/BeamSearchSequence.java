package com.eonmux.neuralfree.gguf.engine.beam;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One beam-search sequence: the generated tokens after the prompt, their raw log probabilities summed
 * ({@code cum_logprob}, a terminating EOS included), the per-step captures the search scored, and how it ended.
 * The prompt the beams grow from stays with the search; {@link #tokens()} carries the generated ids alone, a
 * final EOS dropped from the output text unless the request keeps stop strings in it (vLLM's
 * {@code include_stop_str_in_output}). Immutable; the search derives each candidate from its parent.
 */
public final class BeamSearchSequence {

    /** The {@link #stopTokenId()} value of a beam that did not end on a stop token id. */
    public static final int NO_STOP_TOKEN = GenerationResult.NO_STOP_TOKEN;

    private final int[] tokens;
    private final double cumulativeLogprob;
    private final List<TokenScore[]> steps;
    private final StopReason stopReason;
    private final int stopTokenId;
    private final String text;

    private BeamSearchSequence(int[] tokens, double cumulativeLogprob, List<TokenScore[]> steps,
                               StopReason stopReason, int stopTokenId, String text) {
        this.tokens = tokens;
        this.cumulativeLogprob = cumulativeLogprob;
        this.steps = Collections.unmodifiableList(new ArrayList<>(steps));
        this.stopReason = stopReason;
        this.stopTokenId = stopTokenId;
        this.text = text;
    }

    /** The search's first beam: the prompt generated nothing yet. */
    static BeamSearchSequence initial() {
        return new BeamSearchSequence(new int[0], 0.0, new ArrayList<>(), null, NO_STOP_TOKEN, null);
    }

    /**
     * The candidate extending {@code parent} with one scored token: its tokens and captures grow by
     * {@code tokenId} and {@code step}, its cumulative log probability by {@code logprob}; a beam that ended on
     * an end-of-generation token carries the reason and the token's id, an output-less end (the search ran out
     * of context or budget) the given reason.
     */
    static BeamSearchSequence extend(BeamSearchSequence parent, int tokenId, float logprob, TokenScore[] step) {
        int[] tokens = extend(parent.tokens, tokenId);
        List<TokenScore[]> steps = new ArrayList<>(parent.steps.size() + 1);
        steps.addAll(parent.steps);
        steps.add(step);
        return new BeamSearchSequence(tokens, parent.cumulativeLogprob + logprob, steps, null, NO_STOP_TOKEN, null);
    }

    /** The finished form of a beam: its end reason and stop token recorded, its text decoded. */
    static BeamSearchSequence finish(BeamSearchSequence beam, StopReason reason, int stopTokenId, String text) {
        return new BeamSearchSequence(beam.tokens, beam.cumulativeLogprob, beam.steps, reason, stopTokenId, text);
    }

    /** Copy of the generated token ids, a final EOS included when the beam ended on one. */
    public int[] tokens() {
        return tokens.clone();
    }

    /**
     * The token ids the output reports (vLLM's {@code token_ids}): {@link #tokens()}, with a terminating EOS
     * dropped unless {@code includeEndToken} (vLLM's {@code include_stop_str_in_output}).
     */
    public int[] outputTokens(boolean includeEndToken) {
        if (includeEndToken || stopReason != StopReason.EOS || tokens.length == 0) {
            return tokens.clone();
        }
        return java.util.Arrays.copyOf(tokens, tokens.length - 1);
    }

    /** The number of scored steps, a terminating EOS counted (vLLM's {@code len(logprobs)}). */
    public int stepCount() {
        return steps.size();
    }

    /** The sum of the scored tokens' log probabilities, a terminating EOS included (vLLM's {@code cum_logprob}). */
    public double cumulativeLogprob() {
        return cumulativeLogprob;
    }

    /** The length-penalized beam score ({@link BeamSearchScoring#score}) the beams rank by. */
    public double score(float lengthPenalty) {
        return BeamSearchScoring.score(steps.size(), cumulativeLogprob, lengthPenalty);
    }

    /**
     * The per-step captures the search scored, one row per generated token: the row the step's distribution was
     * read from, the beam's own token first then the alternatives it contended with (raw log probabilities).
     */
    public List<TokenScore[]> logprobs() {
        return steps;
    }

    /** Why the beam ended: EOS, the token budget, a full context, or null while it was still growing. */
    public StopReason stopReason() {
        return stopReason;
    }

    /** The request's stop token id the beam ended on, or {@link #NO_STOP_TOKEN}. */
    public int stopTokenId() {
        return stopTokenId;
    }

    /** The decoded output text (the prompt and a dropped EOS excluded), or null before the search finishes. */
    public String text() {
        return text;
    }

    @Override
    public String toString() {
        return "BeamSearchSequence{" + steps.size() + " steps, cum=" + cumulativeLogprob
                + (stopReason == null ? "" : ", " + stopReason) + '}';
    }

    /** {@code xs} with {@code v} appended. */
    private static int[] extend(int[] xs, int v) {
        int[] out = java.util.Arrays.copyOf(xs, xs.length + 1);
        out[xs.length] = v;
        return out;
    }

    /** Builds a sequence outside the search (tests and callers re-deriving one). */
    public static final class Builder {
        private final int[] tokens;
        private final double cumulativeLogprob;
        private final List<TokenScore[]> steps = new ArrayList<>();
        private StopReason stopReason;
        private int stopTokenId = NO_STOP_TOKEN;
        private String text;

        /** A sequence over already generated {@code tokens} with their summed {@code cumulativeLogprob}. */
        public Builder(int[] tokens, double cumulativeLogprob) {
            this.tokens = Objects.requireNonNull(tokens, "tokens").clone();
            this.cumulativeLogprob = cumulativeLogprob;
        }

        /** Appends one step's capture (its first entry the token the beam took). */
        public Builder step(TokenScore[] capture) {
            steps.add(Objects.requireNonNull(capture, "capture"));
            return this;
        }

        /** Records how the sequence ended. */
        public Builder end(StopReason reason, int stopTokenId) {
            this.stopReason = Objects.requireNonNull(reason, "reason");
            this.stopTokenId = stopTokenId;
            return this;
        }

        /** Records the decoded output text. */
        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public BeamSearchSequence build() {
            return new BeamSearchSequence(tokens, cumulativeLogprob, steps, stopReason, stopTokenId, text);
        }
    }
}
