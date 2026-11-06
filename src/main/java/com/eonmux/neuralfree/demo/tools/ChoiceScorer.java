package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.ForwardBackend;

/**
 * Evaluates the candidate sequences of one multiple-choice task over the backend: each candidate is prefilled
 * from position 0 (llama.cpp shares the candidates' common prefix over several sequences of one batch; a
 * single-sequence backend scores them one by one, and causal attention returns the same logits), and the
 * requested rows' log-softmax values at their next tokens are summed (perplexity.cpp
 * {@code compute_logprobs}).
 */
final class ChoiceScorer {

    private final ForwardBackend backend;

    /** The scorer over one open model's backend; the backend serves one sequence at a time. */
    ChoiceScorer(ForwardBackend backend) {
        this.backend = backend;
    }

    /**
     * The sum of the log-softmax of every row in {@code [fromRow, lastRowIncl]} evaluated at the sequence's
     * next token (row {@code r} predicts {@code tokens[r + 1]}), the quantity the task scorers average. The
     * backend's memory is reset first, so every candidate scores from position 0.
     *
     * @throws IllegalArgumentException when a requested row predicts no token of the sequence
     */
    double sumLogSoftmax(int[] tokens, int fromRow, int lastRowIncl) {
        if (fromRow < 0 || lastRowIncl < fromRow || lastRowIncl > tokens.length - 2) {
            throw new IllegalArgumentException("rows [" + fromRow + ", " + lastRowIncl + "] score no next"
                    + " token of a " + tokens.length + " token sequence");
        }
        int n = lastRowIncl - fromRow + 1;
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            rows[i] = fromRow + i;
        }
        backend.reset();
        float[][] logits = backend.forwardRows(tokens, 0, rows);
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += PerplexityMath.logSoftmax(logits[i], tokens[fromRow + i + 1]);
        }
        return sum;
    }
}
