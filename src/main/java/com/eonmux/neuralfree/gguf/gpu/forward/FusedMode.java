package com.eonmux.neuralfree.gguf.gpu.forward;

import java.util.Locale;

/**
 * Selection mode of the fused decode-path kernels: a layer dispatch may replace two back-to-back kernels
 * (norm then rotation, norm then gate, hyper-connection expand then norm) with one fused kernel, and independent
 * small kernels with one persistent multi-task launch. The fused forms are arithmetically identical to the pairs
 * they replace.
 *
 * <ul>
 *   <li>{@link #AUTO}: a fused kernel is used when the backend provides it; otherwise the unfused pair runs (the
 *       default).</li>
 *   <li>{@link #ON}: the fused kernels must be available; a backend that lacks one fails the bind naming it
 *       instead of silently running the unfused path.</li>
 *   <li>{@link #OFF}: only the unfused kernels run, for an exact comparison against the fused path.</li>
 * </ul>
 *
 * The mode is read from the system property {@value #PROPERTY}; an unknown value is refused by name.
 */
public enum FusedMode {

    AUTO, ON, OFF;

    /** System property holding the mode: {@code auto} (default), {@code on} or {@code off}. */
    public static final String PROPERTY = "neuralfree.gpu.fused";

    /**
     * The configured mode: {@value #PROPERTY}, or {@link #AUTO} when unset or blank.
     *
     * @throws IllegalArgumentException naming the property, the value and the allowed values when the property
     *                                  holds anything else
     */
    public static FusedMode configured() {
        String value = System.getProperty(PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return AUTO;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        for (FusedMode mode : values()) {
            if (mode.name().toLowerCase(Locale.ROOT).equals(normalized)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("-D" + PROPERTY + " must be one of auto, on, off, got '" + value + "'");
    }

    /** Whether the mode permits fused kernels at all. */
    public boolean permitsFusion() {
        return this != OFF;
    }

    /** Whether a missing fused kernel is an error ({@link #ON}) rather than a fallback ({@link #AUTO}). */
    public boolean requiresFusion() {
        return this == ON;
    }
}
