package com.eonmux.neuralfree.gguf.engine;

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
    private final Timings timings;

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
        Objects.requireNonNull(tokens, "tokens");
        this.text = Objects.requireNonNull(text, "text");
        this.stopReason = Objects.requireNonNull(stopReason, "stopReason");
        this.timings = Objects.requireNonNull(timings, "timings");
        if ((stopReason == StopReason.STOP_STRING) != (stopString != null)) {
            throw new IllegalArgumentException("stop string " + (stopString == null ? "missing" : "'" + stopString + "' given")
                    + " for stop reason " + stopReason);
        }
        this.tokens = tokens.clone();
        this.stopString = stopString;
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
