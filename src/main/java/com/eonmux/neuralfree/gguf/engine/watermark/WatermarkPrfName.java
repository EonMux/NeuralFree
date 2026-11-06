package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Locale;

/**
 * The pseudorandom functions a watermark can be keyed with. {@link #PHILOX} is Philox4x32-10 (Random123,
 * <a href="https://doi.org/10.1145/2063384.2063405">Salmon et al.</a>), the one vLLM configures as {@code philox}.
 */
public enum WatermarkPrfName {

    /** Philox4x32-10 over a 64-bit key ({@link PhiloxPrf}). */
    PHILOX("philox");

    private final String label;

    WatermarkPrfName(String label) {
        this.label = label;
    }

    /** The configuration label. */
    public String label() {
        return label;
    }

    /**
     * The PRF of a configuration label, case-insensitively.
     *
     * @throws IllegalArgumentException when the label names no PRF
     */
    public static WatermarkPrfName ofLabel(String label) {
        String key = label == null ? "" : label.trim().toLowerCase(Locale.ROOT);
        for (WatermarkPrfName prf : values()) {
            if (prf.label.equals(key)) {
                return prf;
            }
        }
        throw new IllegalArgumentException("unknown watermark PRF '" + label + "'; the PRFs are philox");
    }
}
