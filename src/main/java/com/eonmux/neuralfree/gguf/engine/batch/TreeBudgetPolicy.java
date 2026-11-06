package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.Locale;

/**
 * How a verify step's tree-row total divides over its proposals
 * ({@code neuralfree.draft.tree.budget}, SpecPilot's per-request node budget policy, arXiv 2611.02501): every
 * request keeps at least its root, every cut a per-request node prefix.
 */
public enum TreeBudgetPolicy {

    /** Every drafted node verifies; the strongest chains first when the step's row caps bind. */
    FULL,
    /** The {@code total - bs} strongest nodes over all requests, keyed by path score (the reference's {@code topt}). */
    TOPT,
    /** A request keeps nodes while {@code gamma * score >= log lambda} (the reference's {@code thresh}). */
    THRESH,
    /** Every request keeps the same node count (the reference's {@code fixed}). */
    FIXED;

    /**
     * Parses a policy value, case-insensitive.
     *
     * @throws IllegalArgumentException naming {@code key} for any other value
     */
    public static TreeBudgetPolicy parse(String key, String v) {
        String p = v.trim().toLowerCase(Locale.ROOT);
        switch (p) {
            case "full":
                return FULL;
            case "topt":
                return TOPT;
            case "thresh":
                return THRESH;
            case "fixed":
                return FIXED;
            default:
                throw new IllegalArgumentException(key + "=" + v + ": expected full, topt, thresh or fixed");
            }
    }
}
