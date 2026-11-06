package com.eonmux.neuralfree.gguf.engine.beam;

import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.BooleanSupplier;

/**
 * Beam search over a scored stepper (vLLM's {@code beam_search}, the Transformers scheme): each step scores
 * every live beam's next-token distribution, expands it into its best candidates, the children that ended on an
 * end-of-generation token join the completed set, and the surviving candidates prune to the {@code beamWidth}
 * best by the length-penalized score. The search ends when the token budget ran out, no candidate survived a
 * step (every beam ended), the context could not fit the next token, or the cancellation hook fired; the
 * completed set and the still-growing beams then rank once more and the best {@code beamWidth} come out.
 *
 * <p>Candidates score {@code ln p / n^lengthPenalty} over the raw log probabilities their stepper captured
 * (vLLM's {@code get_beam_search_score}); ties rank in beam and capture order, as vLLM's stable sort does. The
 * stepper receives every live beam of a step at once, so the runner behind it decides how they share the model's
 * KV cache.
 */
public final class BeamSearch {

    /** One scored step: runs the live beam sequences and returns their capture rows (best tokens first). */
    public interface Stepper {

        /**
         * Scores the next token after every sequence in {@code sequences} (each a full beam sequence, prompt
         * included) and returns one capture row per sequence, in order: the sampled token first, then its best
         * alternatives by raw log probability — at most the search's requested width, distinct ids, best first.
         *
         * @throws RuntimeException when the step failed; the search aborts with it
         */
        List<TokenScore[]> step(List<int[]> sequences);
    }

    /** Renders a beam's output tokens to text (the runner's decoded form). */
    public interface TextRenderer {
        String render(int[] tokens);
    }

    /** The search's fixed settings; build with {@link #settings}. */
    public static final class Settings {

        private final int beamWidth;
        private final int maxSteps;
        private final float lengthPenalty;
        private final boolean ignoreEos;
        private final Set<Integer> endOfGenerationIds;
        private final int[] stopTokenIds;
        private final int[] allowedTokenIds;
        private final int maxSequenceLength;
        private final BooleanSupplier cancelled;
        private final TextRenderer renderer;
        private final boolean includeEndToken;

        Settings(Builder b) {
            this.beamWidth = b.beamWidth;
            this.maxSteps = b.maxSteps;
            this.lengthPenalty = b.lengthPenalty;
            this.ignoreEos = b.ignoreEos;
            this.endOfGenerationIds = b.endOfGenerationIds;
            this.stopTokenIds = b.stopTokenIds;
            this.allowedTokenIds = b.allowedTokenIds;
            this.maxSequenceLength = b.maxSequenceLength;
            this.cancelled = b.cancelled;
            this.renderer = b.renderer;
            this.includeEndToken = b.includeEndToken;
        }

        /** The number of beams the search keeps per step (vLLM's {@code beam_width}). */
        public int beamWidth() {
            return beamWidth;
        }
    }

    /** A new settings builder: one beam, unlimited steps, the plain penalty, no enders or filters. */
    public static Builder settings() {
        return new Builder();
    }

    /** Builds {@link Settings}. */
    public static final class Builder {
        private int beamWidth = 1;
        private int maxSteps = Integer.MAX_VALUE;
        private float lengthPenalty = 1.0f;
        private boolean ignoreEos;
        private Set<Integer> endOfGenerationIds = Set.of();
        private int[] stopTokenIds = new int[0];
        private int[] allowedTokenIds = new int[0];
        private int maxSequenceLength = Integer.MAX_VALUE;
        private BooleanSupplier cancelled = () -> false;
        private TextRenderer renderer = null;
        private boolean includeEndToken;

        /** The number of beams kept per step; at least 1. */
        public Builder beamWidth(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("beam_width must be at least 1, got " + v);
            }
            this.beamWidth = v;
            return this;
        }

        /** The scored-step budget (vLLM's {@code max_tokens}); a beam still growing at the end ends
         *  with {@link StopReason#MAX_TOKENS}. */
        public Builder maxSteps(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("the step budget must be non-negative, got " + v);
            }
            this.maxSteps = v;
            return this;
        }

        /** The length penalty of the ranking score (vLLM's {@code length_penalty}). */
        public Builder lengthPenalty(float v) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                throw new IllegalArgumentException("length_penalty must be finite, got " + v);
            }
            this.lengthPenalty = v;
            return this;
        }

        /**
         * Whether end-of-generation tokens sample through as ordinary candidates instead of ending a beam
         * (vLLM's {@code ignore_eos}); true keeps every such token in the search.
         */
        public Builder ignoreEos(boolean v) {
            this.ignoreEos = v;
            return this;
        }

        /** The vocabulary's end-of-generation ids (vLLM's eos set); a candidate among them completes a beam. */
        public Builder endOfGenerationIds(Set<Integer> ids) {
            this.endOfGenerationIds = Set.copyOf(Objects.requireNonNull(ids, "endOfGenerationIds"));
            return this;
        }

        /** The request's stop token ids; a candidate among them completes a beam and names its
         *  {@link BeamSearchSequence#stopTokenId()}. */
        public Builder stopTokenIds(int[] ids) {
            this.stopTokenIds = Objects.requireNonNull(ids, "stopTokenIds").clone();
            return this;
        }

        /** The only ids the search may extend a beam with (vLLM's {@code allowed_token_ids}); empty allows all. */
        public Builder allowedTokenIds(int[] ids) {
            this.allowedTokenIds = Objects.requireNonNull(ids, "allowedTokenIds").clone();
            return this;
        }

        /** The positions one full sequence may hold (the context); a next token past it ends the growing
         *  beams with {@link StopReason#CONTEXT_FULL}. */
        public Builder maxSequenceLength(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("max_sequence_length must be positive, got " + v);
            }
            this.maxSequenceLength = v;
            return this;
        }

        /** Polled before every step; true ends the growing beams with {@link StopReason#CANCELLED}. */
        public Builder cancelled(BooleanSupplier v) {
            this.cancelled = Objects.requireNonNull(v, "cancelled");
            return this;
        }

        /** Renders a beam's output tokens to text; without one {@link BeamSearchSequence#text()} stays null. */
        public Builder renderer(TextRenderer v) {
            this.renderer = v;
            return this;
        }

        /** Whether a beam's reported output keeps its terminating end token (vLLM's
         *  {@code include_stop_str_in_output}); false drops it from the text. */
        public Builder includeEndToken(boolean v) {
            this.includeEndToken = v;
            return this;
        }

        public Settings build() {
            return new Settings(this);
        }
    }

    private BeamSearch() {
    }

    /**
     * Runs the search over {@code prompt} through {@code stepper}.
     *
     * @throws IllegalArgumentException for an empty prompt or a negative token id
     * @throws RuntimeException         when a step failed (the stepper's error, with the failed beams' context)
     */
    public static BeamSearchOutput run(Settings s, int[] prompt, Stepper stepper) {
        Objects.requireNonNull(s, "settings");
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(stepper, "stepper");
        requirePrompt(prompt);
        List<BeamSearchSequence> beams = new ArrayList<>();
        beams.add(BeamSearchSequence.initial());
        List<BeamSearchSequence> completed = new ArrayList<>();
        StopReason liveEnd = null;
        int step = 0;
        while (step < s.maxSteps && !beams.isEmpty()) {
            if (s.cancelled.getAsBoolean()) {
                liveEnd = StopReason.CANCELLED;
                break;
            }
            if (prompt.length + beams.get(0).stepCount() + 1 > s.maxSequenceLength) {
                liveEnd = StopReason.CONTEXT_FULL;
                break;
            }
            beams = step(s, prompt, beams, completed, stepper);
            step++;
        }
        for (BeamSearchSequence beam : beams) {
            completed.add(beam.stopReason() != null ? beam
                    : BeamSearchSequence.finish(beam, liveEnd != null ? liveEnd : StopReason.MAX_TOKENS,
                            BeamSearchSequence.NO_STOP_TOKEN, null));
        }
        completed.sort((a, b) -> Double.compare(b.score(s.lengthPenalty), a.score(s.lengthPenalty)));
        List<BeamSearchSequence> best = new ArrayList<>();
        int kept = Math.min(s.beamWidth, completed.size());
        for (int i = 0; i < kept; i++) {
            best.add(withText(s, completed.get(i)));
        }
        return new BeamSearchOutput(prompt.length, best, step);
    }

    /**
     * One scored step: every live beam expands into its capture row's candidates — end-of-generation children
     * to {@code completed}, the rest pruned to the beam width. Returns the surviving beams.
     */
    private static List<BeamSearchSequence> step(Settings s, int[] prompt, List<BeamSearchSequence> beams,
                                                 List<BeamSearchSequence> completed, Stepper stepper) {
        List<int[]> sequences = new ArrayList<>();
        for (BeamSearchSequence beam : beams) {
            sequences.add(fullSequence(prompt, beam));
        }
        List<TokenScore[]> rows = stepper.step(sequences);
        if (rows == null || rows.size() != sequences.size()) {
            throw new IllegalStateException("the beam search's step " + beams.get(0).stepCount() + " returned "
                    + (rows == null ? "no" : String.valueOf(rows.size())) + " rows for " + sequences.size()
                    + " beams");
        }
        List<BeamSearchSequence> candidates = new ArrayList<>();
        for (int b = 0; b < beams.size(); b++) {
            BeamSearchSequence beam = beams.get(b);
            TokenScore[] row = rows.get(b);
            if (row == null || row.length == 0) {
                throw new IllegalStateException("the beam search's step " + beam.stepCount() + " returned "
                        + (row == null ? "no" : "an empty") + " capture row for beam " + b);
            }
            for (TokenScore t : row) {
                if (!admits(s, t)) {
                    continue;
                }
                TokenScore[] captured = leading(row, t);
                if (endsGeneration(s, t.tokenId())) {
                    BeamSearchSequence child = BeamSearchSequence.extend(beam, t.tokenId(), t.logprob(), captured);
                    boolean stopToken = isRequestStop(s, t.tokenId());
                    completed.add(BeamSearchSequence.finish(child, StopReason.EOS,
                            stopToken ? t.tokenId() : BeamSearchSequence.NO_STOP_TOKEN, null));
                } else {
                    candidates.add(BeamSearchSequence.extend(beam, t.tokenId(), t.logprob(), captured));
                }
            }
        }
        candidates.sort((a, b) -> Double.compare(b.score(s.lengthPenalty), a.score(s.lengthPenalty)));
        if (candidates.size() > s.beamWidth) {
            candidates = new ArrayList<>(candidates.subList(0, s.beamWidth));
        }
        return candidates;
    }

    /**
     * The step's capture row with {@code chosen} first, the beam's own token leading its alternatives exactly as
     * the capture of a sampled step leads with the sampled token.
     */
    private static TokenScore[] leading(TokenScore[] row, TokenScore chosen) {
        TokenScore[] out = new TokenScore[row.length];
        out[0] = chosen;
        int kept = 1;
        for (TokenScore t : row) {
            if (t.tokenId() != chosen.tokenId()) {
                out[kept++] = t;
            }
        }
        return kept == out.length ? out : java.util.Arrays.copyOf(out, kept);
    }

    /** The prompt followed by the beam's generated tokens. */
    private static int[] fullSequence(int[] prompt, BeamSearchSequence beam) {
        int[] generated = beam.tokens();
        int[] out = new int[prompt.length + generated.length];
        System.arraycopy(prompt, 0, out, 0, prompt.length);
        System.arraycopy(generated, 0, out, prompt.length, generated.length);
        return out;
    }

    /** Whether the capture may seed a candidate: its id allowed and its score finite. */
    private static boolean admits(Settings s, TokenScore t) {
        if (!Float.isFinite(t.logprob())) {
            return false;
        }
        if (s.allowedTokenIds.length == 0) {
            return true;
        }
        for (int id : s.allowedTokenIds) {
            if (id == t.tokenId()) {
                return true;
            }
        }
        return false;
    }

    /** Whether {@code token} ends a beam: a request stop token id always does; a vocabulary end-of-generation
     *  id does unless the search samples it through ({@code ignore_eos}) — {@code StopCriteria}'s rule. */
    private static boolean endsGeneration(Settings s, int token) {
        if (isRequestStop(s, token)) {
            return true;
        }
        return !s.ignoreEos && s.endOfGenerationIds.contains(token);
    }

    /** Whether {@code token} is one of the request's stop token ids. */
    private static boolean isRequestStop(Settings s, int token) {
        for (int id : s.stopTokenIds) {
            if (id == token) {
                return true;
            }
        }
        return false;
    }

    /** The finished beam with its decoded output text, the terminating end token dropped unless the settings
     *  keep it, when a renderer is set. */
    private static BeamSearchSequence withText(Settings s, BeamSearchSequence beam) {
        if (s.renderer == null) {
            return beam;
        }
        return BeamSearchSequence.finish(beam, beam.stopReason(), beam.stopTokenId(),
                s.renderer.render(beam.outputTokens(s.includeEndToken)));
    }

    private static void requirePrompt(int[] prompt) {
        if (prompt.length == 0) {
            throw new IllegalArgumentException("prompt must not be empty");
        }
        for (int i = 0; i < prompt.length; i++) {
            if (prompt[i] < 0) {
                throw new IllegalArgumentException("prompt token " + i + " has negative id " + prompt[i]);
            }
        }
    }
}
