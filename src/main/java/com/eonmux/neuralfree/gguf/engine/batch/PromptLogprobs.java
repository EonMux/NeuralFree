package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogprobCapture;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The prompt-row scores of one prefill: vLLM's {@code prompt_logprobs} and {@code prompt_logprob_token_ids}. Each
 * prompt row's raw logits are scored — their log-softmax, or the logits themselves under a logits
 * {@code logprobs_mode} — with no sampling chain. Row {@code r} predicts prompt token {@code r + 1}; rows before
 * {@code prompt_logprob_start} are not scored, so the first prompt token and every position up to and including
 * the start have a null entry, as vLLM reports them. A {@code prompt_logprobs} entry is vLLM's dict: the prompt
 * token first with its exact vocabulary rank, then the best tokens of the requested count that are not the
 * prompt token. {@link #capture} reads a whole prompt at once (the batch engine's full-prefill path and the
 * single-sequence {@code GenerationLoop}'s row readback); {@link Collector} fills the same capture piece by piece
 * for the chunked prefill ({@code StepExecutor#run}).
 */
public final class PromptLogprobs {

    private PromptLogprobs() {
    }

    /**
     * Captures over {@code prompt} (fully prefilled, no prefix reuse) from {@code rows}, one raw logits row per
     * prompt token: row {@code r} predicts {@code prompt[r + 1]}. {@code count} is the request's entry count
     * ({@link GenerationRequest#promptLogprobs()}) and the scores are log probabilities.
     */
    public static List<TokenScore[]> capture(float[][] rows, int[] prompt, int count) {
        return fill(new Collector(prompt, count, 0, null, false), rows).result();
    }

    /**
     * The prompt-row scores {@code request} asks for ({@link GenerationRequest#readsPromptRows()}) over
     * {@code prompt} (fully prefilled, no prefix reuse) from {@code rows}, one raw logits row per prompt token;
     * {@link Collector#attach} carries them onto the generation's result.
     */
    public static Collector capture(float[][] rows, int[] prompt, GenerationRequest request) {
        return fill(Collector.of(prompt, request), rows);
    }

    private static Collector fill(Collector c, float[][] rows) {
        for (int r = 0; r < rows.length; r++) {
            c.captureRow(r, rows[r]);
        }
        return c;
    }

    /** The prompt-row scores of one prefill, filled one logits row at a time. */
    public static final class Collector {
        private final int[] prompt;
        private final int count;
        private final int start;
        private final int[][] tokenIds;
        private final boolean logits;
        private final List<TokenScore[]> entries;
        private final float[][] tokenIdScores;
        private final boolean[] tokenIdRowsDone;

        /**
         * @param prompt   the prompt's token ids, one per prompt row
         * @param count    the {@code prompt_logprobs} entry count, 0 for none
         * @param start    the first scored row ({@code prompt_logprob_start})
         * @param tokenIds the {@code prompt_logprob_token_ids} rows (row {@code i} scores prompt row
         *                 {@code start + i}), or null
         * @param logits   whether scores are raw logits rather than log probabilities
         */
        Collector(int[] prompt, int count, int start, int[][] tokenIds, boolean logits) {
            this.prompt = prompt;
            this.count = count;
            this.start = start;
            this.tokenIds = tokenIds;
            this.logits = logits;
            this.entries = new ArrayList<>(prompt.length);
            for (int i = 0; i < prompt.length; i++) {
                entries.add(null);
            }
            this.tokenIdScores = tokenIds == null ? null : new float[tokenIds.length][];
            this.tokenIdRowsDone = tokenIds == null ? null : new boolean[tokenIds.length];
        }

        /** The collector {@code request} asks for over {@code prompt}. */
        static Collector of(int[] prompt, GenerationRequest request) {
            return new Collector(prompt, request.promptLogprobs(), request.promptLogprobStart(),
                    request.promptLogprobTokenIds(), request.logprobsMode().logits());
        }

        /**
         * Captures the logits row that processes prompt row {@code r}: it predicts prompt token {@code r + 1}.
         * A row before the start, or at or past the prompt's last row (it predicts a generated token; a recompute
         * replays the fed decodes behind the prompt), adds nothing.
         */
        void captureRow(int r, float[] row) {
            if (r < start || r + 1 >= prompt.length) {
                return;
            }
            float[] values = LogprobCapture.scores(row, logits);
            if (count != 0) {
                entries.set(r + 1, entry(values, prompt[r + 1], count));
            }
            int i = r - start;
            if (tokenIds != null && i < tokenIds.length) {
                float[] scored = new float[tokenIds[i].length];
                for (int j = 0; j < scored.length; j++) {
                    int id = tokenIds[i][j];
                    scored[j] = id >= 0 && id < values.length ? values[id] : Float.NEGATIVE_INFINITY;
                }
                tokenIdScores[i] = scored;
                tokenIdRowsDone[i] = true;
            }
        }

        /** Whether every scored row has its entry (the prefill covered the whole prompt). */
        boolean complete() {
            if (count != 0) {
                for (int i = start + 1; i < entries.size(); i++) {
                    if (entries.get(i) == null) {
                        return false;
                    }
                }
            }
            if (tokenIdRowsDone != null) {
                for (boolean done : tokenIdRowsDone) {
                    if (!done) {
                        return false;
                    }
                }
            }
            return true;
        }

        /** Whether the request asked for {@code prompt_logprobs} entries. */
        boolean hasEntries() {
            return count != 0;
        }

        /** One entry per prompt token; null where no scored row predicts it. */
        List<TokenScore[]> result() {
            return entries;
        }

        /** The {@code prompt_logprob_token_ids} scores, {@code [rows][ids]}, or null when none were asked. */
        float[][] tokenIdScores() {
            return tokenIdScores;
        }

        /**
         * {@code result} carrying this capture: its {@code prompt_logprobs} entries and its
         * {@code prompt_logprob_token_ids} scores, each when asked. Only a capture that covers every scored row
         * ({@link #complete()}) attaches; a partial one, left by a run that stopped before its prefill ended,
         * leaves {@code result} as it is.
         */
        public GenerationResult attach(GenerationResult result) {
            if (!complete()) {
                return result;
            }
            GenerationResult out = hasEntries() ? result.withPromptLogprobs(entries) : result;
            return tokenIdScores != null ? out.withPromptTokenIdLogprobs(tokenIdScores) : out;
        }
    }

    /**
     * One {@code prompt_logprobs} entry over a scored row: {@code actual} leads with its exact rank, then the best
     * {@code count - 1} tokens of the row that are not {@code actual} — fewer when {@code actual} is among them, as
     * vLLM's dict merges it ({@link GenerationRequest#PROMPT_LOGPROBS_ALL}: every other token, best first).
     */
    private static TokenScore[] entry(float[] values, int actual, int count) {
        if (count == GenerationRequest.PROMPT_LOGPROBS_ALL) {
            return everyToken(values, actual);
        }
        int top = Math.max(count, 1) - 1;
        int others = LogprobCapture.rank(values, actual) <= top ? top - 1 : top;
        return LogprobCapture.withBest(values, actual, others);
    }

    /** Every vocabulary token: {@code actual} first, then the rest best first, each with its rank. */
    private static TokenScore[] everyToken(float[] values, int actual) {
        long[] keys = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            keys[i] = Sampler.Candidates.descendingKey(values[i], i);
        }
        Arrays.sort(keys);
        boolean inVocab = actual >= 0 && actual < values.length;
        TokenScore[] out = new TokenScore[inVocab ? values.length : values.length + 1];
        float lead = inVocab ? values[actual] : Float.NEGATIVE_INFINITY;
        out[0] = new TokenScore(actual, lead, LogprobCapture.rank(values, actual));
        int k = 1;
        int greater = 0;
        for (int j = 0; j < keys.length; j++) {
            int id = (int) keys[j];
            if (j > 0 && values[(int) keys[j - 1]] > values[id]) {
                greater = j;
            }
            if (id != actual) {
                out[k++] = new TokenScore(id, values[id], greater + 1);
            }
        }
        return out;
    }
}
