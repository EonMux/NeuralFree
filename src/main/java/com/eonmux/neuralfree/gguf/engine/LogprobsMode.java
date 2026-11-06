package com.eonmux.neuralfree.gguf.engine;

import java.util.Locale;

/**
 * Which distribution a logprobs capture scores (vLLM's {@code --logprobs-mode}). The raw modes read the model's
 * logits row as it came out of the forward pass, before any sampling stage (logit bias, allowed and banned ids,
 * penalties, grammar, temperature, truncation); the processed modes read the sampler chain's final candidates —
 * a greedy chain's penalized logits, a sampling chain's logits after temperature and truncation, with removed
 * tokens at {@code -inf}. The {@code *_LOGPROBS} modes report {@code ln(p)} of the log-softmax, the
 * {@code *_LOGITS} modes the logit values themselves. Prompt captures have no sampling chain: both logprobs modes
 * report the raw log-softmax there and both logits modes the raw logits, as vLLM does.
 */
public enum LogprobsMode {

    /** The log-softmax of the raw logits (vLLM's default {@code raw_logprobs}, llama.cpp's default probabilities). */
    RAW_LOGPROBS("raw_logprobs", false, false),
    /** The raw logits ({@code raw_logits}). */
    RAW_LOGITS("raw_logits", false, true),
    /** The log-softmax of the chain's final candidates ({@code processed_logprobs}, llama.cpp {@code post_sampling_probs}). */
    PROCESSED_LOGPROBS("processed_logprobs", true, false),
    /** The chain's final candidate logits ({@code processed_logits}). */
    PROCESSED_LOGITS("processed_logits", true, true);

    private final String label;
    private final boolean processed;
    private final boolean logits;

    LogprobsMode(String label, boolean processed, boolean logits) {
        this.label = label;
        this.processed = processed;
        this.logits = logits;
    }

    /** vLLM's name of the mode ({@code raw_logprobs}, ...). */
    public String label() {
        return label;
    }

    /** Whether the capture reads the sampler chain's final candidates rather than the raw row. */
    public boolean processed() {
        return processed;
    }

    /** Whether the capture reports logit values rather than log probabilities. */
    public boolean logits() {
        return logits;
    }

    /**
     * The mode named {@code name} (vLLM's spelling, case-insensitive).
     *
     * @throws IllegalArgumentException naming {@code logprobs_mode} for any other value
     */
    public static LogprobsMode parse(String name) {
        if (name != null) {
            String key = name.trim().toLowerCase(Locale.ROOT);
            for (LogprobsMode mode : values()) {
                if (mode.label.equals(key)) {
                    return mode;
                }
            }
        }
        throw new IllegalArgumentException("logprobs_mode must be one of raw_logprobs, raw_logits,"
                + " processed_logprobs or processed_logits, got " + name);
    }
}
