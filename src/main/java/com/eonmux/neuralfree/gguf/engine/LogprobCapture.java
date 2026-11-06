package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.Verifier;

import java.util.Arrays;

/**
 * The per-token logprobs capture of one generation: for every sampled token, the token first with its score and
 * vocabulary rank, then either the {@code count - 1} best other tokens (best first, ties to the lower id) or —
 * when the request names {@link GenerationRequest#logprobTokenIds()} — exactly those tokens in the request's
 * order (vLLM's {@code logprob_token_ids}, which take precedence over the count). Scores follow the request's
 * {@link LogprobsMode}: the raw modes read the model's logits row, the processed modes the sampler chain's final
 * candidates, where every removed token scores {@code -inf}.
 *
 * <p>Holds one scratch row, so one capture belongs to one sequence and is not thread-safe; the static helpers
 * ({@link #scores}, {@link #rank}, {@link #withBest}) are pure.
 */
public final class LogprobCapture {

    private static final TokenScore[] NONE = new TokenScore[0];
    /** The largest alternative count picked by insertion; larger counts sort the row once. */
    private static final int INSERTION_LIMIT = 64;

    private final LogprobsMode mode;
    private final int count;
    private final int[] tokenIds;
    private float[] dense = new float[0];

    private LogprobCapture(LogprobsMode mode, int count, int[] tokenIds) {
        this.mode = mode;
        this.count = count;
        this.tokenIds = tokenIds;
    }

    /** The capture {@code request} asks for: its mode, its {@code logprobs} count and its explicit token ids. */
    public static LogprobCapture of(GenerationRequest request) {
        return new LogprobCapture(request.logprobsMode(), Math.max(0, request.logprobs()),
                request.logprobTokenIds());
    }

    /** Whether the request captures anything (a positive count or explicit token ids). */
    public boolean active() {
        return count > 0 || tokenIds.length > 0;
    }

    /** The distribution the capture scores. */
    public LogprobsMode mode() {
        return mode;
    }

    /**
     * Scores one sampled step: {@code raw} is the logits row the chain ran over, {@code processed} the chain's
     * final candidates for that row (read under the processed modes only).
     *
     * @return the capture, the sampled token first; empty when the capture is not {@link #active()}
     */
    public TokenScore[] capture(float[] raw, Sampler.Candidates processed, int sampled) {
        if (!active()) {
            return NONE;
        }
        if (!mode.processed()) {
            return score(raw, sampled);
        }
        float[] row = denseRow(raw.length);
        for (int i = 0; i < processed.size(); i++) {
            put(row, processed.id(i), processed.logit(i));
        }
        return score(row, sampled);
    }

    /**
     * Scores one verified emission: {@code raw} is the verify row, {@code processed} the target's processed
     * distribution of that row — the one the verifier decided over (read under the processed modes only).
     */
    public TokenScore[] capture(float[] raw, Verifier.Distribution processed, int sampled) {
        if (!active()) {
            return NONE;
        }
        if (!mode.processed()) {
            return score(raw, sampled);
        }
        float[] row = denseRow(raw.length);
        for (int i = 0; i < processed.size(); i++) {
            put(row, processed.token(i), processed.logit(i));
        }
        return score(row, sampled);
    }

    private float[] denseRow(int vocab) {
        if (dense.length != vocab) {
            dense = new float[vocab];
        }
        Arrays.fill(dense, Float.NEGATIVE_INFINITY);
        return dense;
    }

    private static void put(float[] row, int id, float logit) {
        if (id >= 0 && id < row.length) {
            row[id] = logit;
        }
    }

    private TokenScore[] score(float[] row, int sampled) {
        float[] values = scores(row, mode.logits());
        if (tokenIds.length > 0) {
            TokenScore[] out = new TokenScore[tokenIds.length + 1];
            out[0] = new TokenScore(sampled, valueOf(values, sampled), rank(values, sampled));
            for (int i = 0; i < tokenIds.length; i++) {
                out[i + 1] = new TokenScore(tokenIds[i], valueOf(values, tokenIds[i]));
            }
            return out;
        }
        return withBest(values, sampled, count - 1);
    }

    /**
     * The scores of one row: its logits ({@code logits} true) or their log-softmax, with every NaN or
     * {@code -inf} logit at {@code -inf} (it takes no part in the softmax). Returns a new array.
     */
    public static float[] scores(float[] row, boolean logits) {
        float[] out = new float[row.length];
        if (logits) {
            for (int i = 0; i < row.length; i++) {
                out[i] = takesPart(row[i]) ? row[i] : Float.NEGATIVE_INFINITY;
            }
            return out;
        }
        double max = Double.NEGATIVE_INFINITY;
        for (float v : row) {
            if (takesPart(v) && v > max) {
                max = v;
            }
        }
        double sum = 0;
        for (float v : row) {
            if (takesPart(v)) {
                sum += Math.exp(v - max);
            }
        }
        double logSum = max + Math.log(sum);
        for (int i = 0; i < row.length; i++) {
            out[i] = takesPart(row[i]) ? (float) (row[i] - logSum) : Float.NEGATIVE_INFINITY;
        }
        return out;
    }

    /** The 1-based rank of {@code id} in {@code values}: one plus the tokens scoring strictly higher. */
    public static int rank(float[] values, int id) {
        float v = valueOf(values, id);
        int greater = 0;
        for (float x : values) {
            if (x > v) {
                greater++;
            }
        }
        return greater + 1;
    }

    /**
     * {@code lead} with its score and rank, then the {@code others} best other tokens of finite score, best first
     * (ties to the lower id), each with its rank. Fewer follow when fewer tokens have a finite score.
     */
    public static TokenScore[] withBest(float[] values, int lead, int others) {
        int[] best = best(values, lead, Math.max(0, others));
        float leadValue = valueOf(values, lead);
        TokenScore[] out = new TokenScore[best.length + 1];
        out[0] = new TokenScore(lead, leadValue, rank(values, lead));
        int greaterInList = 0;
        for (int k = 0; k < best.length; k++) {
            float v = values[best[k]];
            if (k > 0 && values[best[k - 1]] > v) {
                greaterInList = k;
            }
            int greater = greaterInList + (leadValue > v ? 1 : 0);
            out[k + 1] = new TokenScore(best[k], v, greater + 1);
        }
        return out;
    }

    /** The ids of the {@code n} best finite-scored tokens other than {@code skip}, best first. */
    private static int[] best(float[] values, int skip, int n) {
        if (n > INSERTION_LIMIT) {
            return sortedBest(values, skip, n);
        }
        int[] ids = new int[Math.min(n, values.length)];
        int size = 0;
        for (int i = 0; i < values.length && ids.length > 0; i++) {
            float v = values[i];
            if (i == skip || v == Float.NEGATIVE_INFINITY) {
                continue;
            }
            if (size == ids.length && v <= values[ids[size - 1]]) {
                continue;
            }
            int at = size == ids.length ? size - 1 : size++;
            while (at > 0 && values[ids[at - 1]] < v) {
                ids[at] = ids[at - 1];
                at--;
            }
            ids[at] = i;
        }
        return size == ids.length ? ids : Arrays.copyOf(ids, size);
    }

    /** {@link #best} by one sort of the finite scores, for counts past {@link #INSERTION_LIMIT}. */
    private static int[] sortedBest(float[] values, int skip, int n) {
        long[] keys = new long[values.length];
        int size = 0;
        for (int i = 0; i < values.length; i++) {
            if (i != skip && values[i] != Float.NEGATIVE_INFINITY) {
                keys[size++] = Sampler.Candidates.descendingKey(values[i], i);
            }
        }
        Arrays.sort(keys, 0, size);
        int[] ids = new int[Math.min(n, size)];
        for (int k = 0; k < ids.length; k++) {
            ids[k] = (int) keys[k];
        }
        return ids;
    }

    private static float valueOf(float[] values, int id) {
        return id >= 0 && id < values.length ? values[id] : Float.NEGATIVE_INFINITY;
    }

    private static boolean takesPart(float logit) {
        return logit != Float.NEGATIVE_INFINITY && !Float.isNaN(logit);
    }

    @Override
    public String toString() {
        return "LogprobCapture{" + mode.label() + (tokenIds.length > 0 ? ", ids=" + Arrays.toString(tokenIds)
                : ", count=" + count) + '}';
    }
}
