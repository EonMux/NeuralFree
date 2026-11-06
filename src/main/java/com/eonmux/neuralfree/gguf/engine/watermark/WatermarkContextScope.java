package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Locale;

/**
 * Which history the generation search for a repeated context looks over before it samples: a repeated context would
 * reuse one keyed random vector, so it is sampled without the watermark ({@code WatermarkConfig.deduplicate_contexts}).
 */
public enum WatermarkContextScope {

    /** No search: every token is watermarked. */
    NONE("none"),
    /** The request's generated tokens (the default). */
    SINGLE_TURN("single_turn"),
    /** The generated tokens and the prompt; the first {@code context_width} generated tokens stay unwatermarked. */
    ALL("all");

    private final String label;

    WatermarkContextScope(String label) {
        this.label = label;
    }

    /** The configuration label. */
    public String label() {
        return label;
    }

    /** The default scope of generation ({@link #SINGLE_TURN}) and of detection (every context scored once). */
    public static final WatermarkContextScope DEFAULT = SINGLE_TURN;

    /**
     * The scope of a configuration label, case-insensitively and with {@code -} accepted for {@code _}.
     *
     * @throws IllegalArgumentException when the label names no scope
     */
    public static WatermarkContextScope ofLabel(String label) {
        String key = label == null ? "" : label.trim().toLowerCase(Locale.ROOT).replace('-', '_');
        for (WatermarkContextScope scope : values()) {
            if (scope.label.equals(key)) {
                return scope;
            }
        }
        throw new IllegalArgumentException("unknown deduplicate_contexts '" + label
                + "'; the scopes are none, single_turn and all");
    }
}
