package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Outcome of one generation: the generated tokens (prompt excluded), their decoded text (any matched stop string
 * excluded), why generation stopped, and timings. Immutable.
 */
public final class GenerationResult {

    private final int[] tokens;
    private final String text;
    private final StopReason stopReason;
    private final String stopString;
    /** A stop token id the generation ended on, or {@link #NO_STOP_TOKEN}. */
    private final int stopTokenId;
    private final Timings timings;
    private final List<TokenScore[]> logprobs;
    private final List<TokenScore[]> promptLogprobs;
    private final float[][] promptTokenIdLogprobs;
    private final int reasoningTokens;

    /** {@link #stopTokenId()} value meaning the generation did not end on a request's stop token id. */
    public static final int NO_STOP_TOKEN = -1;

    /**
     * @param tokens     generated token ids, prompt excluded (copied)
     * @param text       decoded text of {@code tokens}, cut before the matched stop string
     * @param stopReason why generation ended
     * @param stopString the matched stop string; non-null exactly when {@code stopReason} is
     *                   {@link StopReason#STOP_STRING}
     * @param timings    measured times
     * @throws IllegalArgumentException when {@code stopString} disagrees with {@code stopReason}
     */
    public GenerationResult(int[] tokens, String text, StopReason stopReason, String stopString, Timings timings) {
        this(tokens, text, stopReason, stopString, NO_STOP_TOKEN, timings, List.of(), List.of(), null, 0);
    }

    private GenerationResult(int[] tokens, String text, StopReason stopReason, String stopString, int stopTokenId,
                             Timings timings, List<TokenScore[]> logprobs, List<TokenScore[]> promptLogprobs,
                             float[][] promptTokenIdLogprobs, int reasoningTokens) {
        Objects.requireNonNull(tokens, "tokens");
        this.text = Objects.requireNonNull(text, "text");
        this.stopReason = Objects.requireNonNull(stopReason, "stopReason");
        this.timings = Objects.requireNonNull(timings, "timings");
        this.logprobs = Objects.requireNonNull(logprobs, "logprobs");
        this.promptLogprobs = Objects.requireNonNull(promptLogprobs, "promptLogprobs");
        this.promptTokenIdLogprobs = promptTokenIdLogprobs;
        if ((stopReason == StopReason.STOP_STRING) != (stopString != null)) {
            throw new IllegalArgumentException("stop string " + (stopString == null ? "missing" : "'" + stopString + "' given")
                    + " for stop reason " + stopReason);
        }
        if (stopTokenId != NO_STOP_TOKEN && stopTokenId < 0) {
            throw new IllegalArgumentException("stop token id must be " + NO_STOP_TOKEN + " (none) or a token id,"
                    + " got " + stopTokenId);
        }
        if (reasoningTokens < 0 || reasoningTokens > tokens.length) {
            throw new IllegalArgumentException("reasoning tokens " + reasoningTokens + " outside [0, "
                    + tokens.length + "]");
        }
        this.tokens = tokens.clone();
        this.stopString = stopString;
        this.stopTokenId = stopTokenId;
        this.reasoningTokens = reasoningTokens;
    }

    /** Copy of the generated token ids, prompt excluded. */
    public int[] tokens() {
        return tokens.clone();
    }

    /** Number of generated tokens. */
    public int tokenCount() {
        return tokens.length;
    }

    /** Decoded text of the generated tokens, cut before the matched stop string. */
    public String text() {
        return text;
    }

    public StopReason stopReason() {
        return stopReason;
    }

    /** The matched stop string, or null unless {@link #stopReason()} is {@link StopReason#STOP_STRING}. */
    public String stopString() {
        return stopString;
    }

    /**
     * The request's stop token id the generation ended on (vLLM's {@code stop_reason} for a stop-token end),
     * or {@link #NO_STOP_TOKEN}: the vocabulary's own end-of-generation tokens, a stop string and every other
     * reason leave it unset.
     */
    public int stopTokenId() {
        return stopTokenId;
    }

    /** A copy naming {@code stopTokenId} as the stop token id the generation ended on. */
    public GenerationResult withStopTokenId(int stopTokenId) {
        return new GenerationResult(tokens, text, stopReason, stopString, stopTokenId, timings, logprobs,
                promptLogprobs, promptTokenIdLogprobs, reasoningTokens);
    }

    public Timings timings() {
        return timings;
    }

    /** {@code prompt} followed by the generated tokens. */
    public int[] withPrompt(int[] prompt) {
        int[] all = new int[prompt.length + tokens.length];
        System.arraycopy(prompt, 0, all, 0, prompt.length);
        System.arraycopy(tokens, 0, all, prompt.length, tokens.length);
        return all;
    }

    /**
     * The per-token log probabilities when the request asked for them ({@link
     * GenerationRequest.Builder#logprobs(int)} or {@link GenerationRequest.Builder#logprobTokenIds}), one entry
     * per generated token: the sampled token first, then its best alternatives or the requested token ids, scored
     * in the request's {@link LogprobsMode}. Empty when the request asked for none.
     */
    public List<TokenScore[]> logprobs() {
        return logprobs;
    }

    /** A copy carrying the per-token log probabilities ({@link SequenceRun}'s capture). */
    public GenerationResult withLogprobs(List<TokenScore[]> logprobs) {
        return new GenerationResult(tokens, text, stopReason, stopString, stopTokenId, timings,
                List.copyOf(logprobs), promptLogprobs, promptTokenIdLogprobs, reasoningTokens);
    }

    /**
     * The per-prompt-token log probabilities when the request asked for them ({@link
     * GenerationRequest.Builder#promptLogprobs}), one entry per prompt token: the prompt token plus its top
     * alternatives of the distribution that predicted it; empty when the request asked for none.
     */
    public List<TokenScore[]> promptLogprobs() {
        return promptLogprobs;
    }

    /** A copy carrying the per-prompt-token log probabilities (the batch engine's prompt-row readback). */
    public GenerationResult withPromptLogprobs(List<TokenScore[]> promptLogprobs) {
        return new GenerationResult(tokens, text, stopReason, stopString, stopTokenId, timings, logprobs,
                Collections.unmodifiableList(new ArrayList<>(promptLogprobs)), promptTokenIdLogprobs,
                reasoningTokens);
    }

    /**
     * The scores of the request's prompt token-id rows ({@link GenerationRequest#promptLogprobTokenIds()}, vLLM's
     * {@code prompt_token_id_logprobs}): {@code [rows][ids]}, row {@code i} scoring its ids as predictions of
     * prompt token {@code promptLogprobStart + i + 1}, a padding entry at {@code -inf}. A deep copy; null when
     * the request named no rows.
     */
    public float[][] promptTokenIdLogprobs() {
        return copy(promptTokenIdLogprobs);
    }

    /** A copy carrying the prompt token-id scores (copied; the batch engine's prompt-row readback). */
    public GenerationResult withPromptTokenIdLogprobs(float[][] scores) {
        return new GenerationResult(tokens, text, stopReason, stopString, stopTokenId, timings, logprobs,
                promptLogprobs, copy(Objects.requireNonNull(scores, "scores")), reasoningTokens);
    }

    private static float[][] copy(float[][] rows) {
        if (rows == null) {
            return null;
        }
        float[][] out = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            out[i] = rows[i].clone();
        }
        return out;
    }

    /**
     * The generated tokens inside a reasoning block, its open and close tags excluded (vLLM's
     * {@code count_reasoning_tokens}); 0 for a model without a reasoning parser.
     */
    public int reasoningTokens() {
        return reasoningTokens;
    }

    /** A copy counting {@code reasoningTokens} generated tokens as reasoning. */
    public GenerationResult withReasoningTokens(int reasoningTokens) {
        return new GenerationResult(tokens, text, stopReason, stopString, stopTokenId, timings, logprobs,
                promptLogprobs, promptTokenIdLogprobs, reasoningTokens);
    }

    @Override
    public String toString() {
        return "GenerationResult{tokens=" + tokens.length + ", stop=" + stopReason
                + (stopString != null ? " '" + stopString + "'" : "") + ", " + timings + '}';
    }

    /**
     * Prompt-evaluation and generation times (llama.cpp prompt eval / eval timings). Implementations that cannot
     * separate the two phases report a single total ({@link #split()} false). Immutable.
     */
    public static final class Timings {

        private final int promptTokens;
        private final long promptNanos;
        private final int generatedTokens;
        private final long generationNanos;
        private final boolean split;

        private Timings(int promptTokens, long promptNanos, int generatedTokens, long generationNanos, boolean split) {
            if (promptTokens < 0 || generatedTokens < 0 || promptNanos < 0 || generationNanos < 0) {
                throw new IllegalArgumentException("timings must be non-negative: prompt " + promptTokens + " tokens / "
                        + promptNanos + " ns, generation " + generatedTokens + " tokens / " + generationNanos + " ns");
            }
            this.promptTokens = promptTokens;
            this.promptNanos = promptNanos;
            this.generatedTokens = generatedTokens;
            this.generationNanos = generationNanos;
            this.split = split;
        }

        /**
         * @param promptTokens    prompt tokens evaluated (after prefix reuse)
         * @param promptNanos     time spent evaluating them
         * @param generatedTokens tokens sampled, including any discarded after a stop string
         * @param generationNanos time spent generating them
         */
        public static Timings of(int promptTokens, long promptNanos, int generatedTokens, long generationNanos) {
            return new Timings(promptTokens, promptNanos, generatedTokens, generationNanos, true);
        }

        /** Timings with only the total time known. */
        public static Timings total(int promptTokens, int generatedTokens, long totalNanos) {
            return new Timings(promptTokens, 0L, generatedTokens, totalNanos, false);
        }

        public int promptTokens() { return promptTokens; }
        /** Prompt time; 0 when {@link #split()} is false. */
        public long promptNanos() { return promptNanos; }
        public int generatedTokens() { return generatedTokens; }
        /** Generation time; the whole time when {@link #split()} is false. */
        public long generationNanos() { return generationNanos; }
        /** Whether prompt and generation times are measured separately. */
        public boolean split() { return split; }

        public long totalNanos() {
            return promptNanos + generationNanos;
        }

        /** Prompt tokens per second, or NaN when not measured separately or no time elapsed. */
        public double promptTokensPerSecond() {
            return split ? rate(promptTokens, promptNanos) : Double.NaN;
        }

        /** Generated tokens per second (over the total time when not split), or NaN when no time elapsed. */
        public double generationTokensPerSecond() {
            return rate(generatedTokens, generationNanos);
        }

        private static double rate(int tokens, long nanos) {
            return nanos > 0 ? tokens * 1e9 / nanos : Double.NaN;
        }

        @Override
        public String toString() {
            if (!split) {
                return String.format(Locale.ROOT, "Timings{%d prompt + %d generated tokens in %.1f ms}",
                        promptTokens, generatedTokens, totalNanos() / 1e6);
            }
            return String.format(Locale.ROOT, "Timings{prompt %d tokens %.1f ms (%.2f t/s), generation %d tokens %.1f ms (%.2f t/s)}",
                    promptTokens, promptNanos / 1e6, promptTokensPerSecond(),
                    generatedTokens, generationNanos / 1e6, generationTokensPerSecond());
        }
    }
}
