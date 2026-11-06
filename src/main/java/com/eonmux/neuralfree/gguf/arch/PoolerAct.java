package com.eonmux.neuralfree.gguf.arch;

/**
 * The pooler-level activation a classifier head applies to its scores after the chain
 * (vLLM's pooler activation): the configuration's {@code problem_type} and sentence-transformers
 * activation function resolve to one of these names, written to {@code classifier.pooler_activation}.
 * The names are the GGUF metadata values.
 */
public enum PoolerAct {

    /** The scores are reported unchanged (a regression head). */
    IDENTITY("identity"),
    /** {@code 1 / (1 + exp(-x))} per score (a single-label one-output or a multi-label head). */
    SIGMOID("sigmoid"),
    /** {@code exp(x) / sum exp(x)} over the scores (a single-label multi-output head). */
    SOFTMAX("softmax"),
    /** {@code tanh(x)} per score (a sentence-transformers cross-encoder activation). */
    TANH("tanh");

    /** The GGUF metadata value of {@code classifier.pooler_activation}. */
    private final String ggufName;

    PoolerAct(String ggufName) {
        this.ggufName = ggufName;
    }

    /** The GGUF metadata value. */
    public String ggufName() {
        return ggufName;
    }

    /**
     * The activation of a GGUF {@code classifier.pooler_activation} value.
     *
     * @throws IllegalArgumentException for an unknown value
     */
    public static PoolerAct fromGgufName(String name) {
        for (PoolerAct act : values()) {
            if (act.ggufName.equals(name)) {
                return act;
            }
        }
        throw new IllegalArgumentException("unknown pooler activation '" + name + "': expected identity, sigmoid,"
                + " softmax or tanh");
    }
}
