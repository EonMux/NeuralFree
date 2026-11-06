package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Locale;

/**
 * The watermarking algorithms: a single-key Gumbel-max watermark ({@code gumbel}) and two independent Gumbel-max
 * keys routed per token ({@code dual_key_gumbel}). The labels are the ones vLLM's {@code WatermarkConfig.algorithm}
 * carries, and the configuration spells them exactly.
 */
public enum WatermarkAlgorithm {

    /** One key: the sampled token is the argmax of {@code logits + Gumbel(PRF(key, context, token))}. */
    GUMBEL("gumbel"),
    /**
     * Two keys: the draft key A and the target key B, chosen per token by a draw over {@code [1 - alpha, alpha]}
     * (key B with probability {@code alpha}). Detection fuses both keys' scores with the same weights.
     */
    DUAL_KEY_GUMBEL("dual_key_gumbel");

    private final String label;

    WatermarkAlgorithm(String label) {
        this.label = label;
    }

    /** The configuration label. */
    public String label() {
        return label;
    }

    /**
     * The algorithm of a configuration label, case-insensitively and with {@code -} accepted for {@code _}.
     *
     * @throws IllegalArgumentException when the label names no algorithm
     */
    public static WatermarkAlgorithm ofLabel(String label) {
        String key = label == null ? "" : label.trim().toLowerCase(Locale.ROOT).replace('-', '_');
        for (WatermarkAlgorithm algorithm : values()) {
            if (algorithm.label.equals(key)) {
                return algorithm;
            }
        }
        throw new IllegalArgumentException("unknown watermarking algorithm '" + label
                + "'; the algorithms are gumbel and dual_key_gumbel");
    }
}
