package com.eonmux.neuralfree.gguf.engine;

import java.util.Arrays;
import java.util.Objects;

/**
 * Scores produced by a classifier, reranker or decision head for one input: one score per output with its label.
 *
 * <p>Labels follow llama.cpp's embedding example: a missing or empty {@code classifier.output_labels} entry is
 * shown as the output index. {@link #normalized()} is true when the head already produced probabilities (the
 * qwen3 / qwen3vl reranker applies a softmax, llama.cpp src/llama-graph.cpp build_pooling RANK); otherwise the scores
 * are raw head outputs. Immutable.
 */
public final class ClassifierOutput {

    private final float[] scores;
    private final String[] labels;
    private final boolean normalized;

    private ClassifierOutput(float[] scores, String[] labels, boolean normalized) {
        this.scores = scores;
        this.labels = labels;
        this.normalized = normalized;
    }

    /**
     * @param scores     one score per output (copied)
     * @param labels     labels per output, or null / shorter for index labels; null or empty entries become the
     *                   index (copied)
     * @param normalized whether {@code scores} are probabilities
     * @throws IllegalArgumentException when {@code scores} is empty or {@code labels} has more entries than scores
     */
    public static ClassifierOutput of(float[] scores, String[] labels, boolean normalized) {
        Objects.requireNonNull(scores, "scores");
        if (scores.length == 0) {
            throw new IllegalArgumentException("a classifier output needs at least one score");
        }
        int nLabels = labels == null ? 0 : labels.length;
        if (nLabels > scores.length) {
            throw new IllegalArgumentException(nLabels + " labels for " + scores.length + " scores");
        }
        String[] resolved = new String[scores.length];
        for (int i = 0; i < resolved.length; i++) {
            String l = i < nLabels ? labels[i] : null;
            resolved[i] = l == null || l.isEmpty() ? Integer.toString(i) : l;
        }
        return new ClassifierOutput(scores.clone(), resolved, normalized);
    }

    /** Number of outputs. */
    public int size() {
        return scores.length;
    }

    /** Score of output {@code i}. */
    public float score(int i) {
        return scores[i];
    }

    /** Label of output {@code i}. */
    public String label(int i) {
        return labels[i];
    }

    /** Copy of all scores. */
    public float[] scores() {
        return scores.clone();
    }

    /** Copy of all labels. */
    public String[] labels() {
        return labels.clone();
    }

    /** Whether the scores are probabilities. */
    public boolean normalized() {
        return normalized;
    }

    /** Index of the highest score (first on ties). */
    public int best() {
        return ForwardBackend.argmax(scores);
    }

    /** Label of the highest score. */
    public String bestLabel() {
        return labels[best()];
    }

    /** The single score of a one-output reranker. */
    public float rankScore() {
        if (scores.length != 1) {
            throw new IllegalStateException("rank score requested from a head with " + scores.length + " outputs");
        }
        return scores[0];
    }

    /** Probabilities: the scores when already normalized, otherwise their softmax. */
    public float[] probabilities() {
        if (normalized) {
            return scores.clone();
        }
        float max = Float.NEGATIVE_INFINITY;
        for (float s : scores) {
            max = Math.max(max, s);
        }
        float[] p = new float[scores.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            double e = Math.exp(scores[i] - max);
            p[i] = (float) e;
            sum += e;
        }
        for (int i = 0; i < p.length; i++) {
            p[i] = (float) (p[i] / sum);
        }
        return p;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("ClassifierOutput{");
        for (int i = 0; i < scores.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(labels[i]).append('=').append(scores[i]);
        }
        return sb.append(normalized ? ", normalized}" : "}").toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ClassifierOutput)) {
            return false;
        }
        ClassifierOutput other = (ClassifierOutput) o;
        return normalized == other.normalized && Arrays.equals(scores, other.scores)
                && Arrays.equals(labels, other.labels);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * Arrays.hashCode(scores) + Arrays.hashCode(labels)) + (normalized ? 1 : 0);
    }
}
