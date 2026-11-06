package com.eonmux.neuralfree.gguf;

import java.util.Locale;

/**
 * Process-wide batch-invariance mode, the parity of vLLM's {@code VLLM_BATCH_INVARIANT}: while it is on, every
 * kernel choice that would otherwise depend on how many rows a forward step carries is pinned to the choice the
 * single-row path makes, so a request's logits do not depend on the other requests batched beside it.
 *
 * <p>The mode is read from the system property {@value #PROPERTY} ({@code true} or {@code false}, also
 * {@code 1} or {@code 0} as vLLM's environment variable takes) and, when the property is unset, from the
 * environment variable {@value #ENV}. A server sets it with {@code --batch-invariant}. The value is resolved once
 * and remembered; tests override it with {@link #setEnabled(boolean)} and release the override with
 * {@link #clearOverride()}. An unknown value is refused by name.</p>
 *
 * <p>The kernels the mode pins, all to the single-row reduction order:</p>
 * <ul>
 *   <li>projections: a weight type with a fast GEMM lane (the int8 MMQ GEMM, its cooperative-matrix form or the
 *       tensor-core MMA GEMM) takes it for one row and for a full batch alike, and its split-K twins are never
 *       taken; a type without one keeps the float matvec family ({@code {T}_MatVec} /
 *       {@code {T}_BatchMatMul_2D}) at every row count, and the exact tiled GEMM and the single-token Q8_1
 *       integer-dot lane are never taken;</li>
 *   <li>projection accumulate and fused gate/up: the unfused products that every row count takes;</li>
 *   <li>flash attention: one per-row decode launch per query row (its split over the live key count is the one
 *       row's own), both on the dense and on the paged cache, instead of a multi-row prefill kernel or one paged
 *       launch split for the batch's largest sequence;</li>
 *   <li>MoE routed experts: the sorted per-pair {@code {T}_BatchMatMulId} path for every row count instead of the
 *       one-token matvec path and the grouped expert GEMM;</li>
 *   <li>row norms and reductions: their work-groups stay one per row, as they already are;</li>
 *   <li>CPU attention: the key-split count of a call, taken from one query row's work units instead of the whole
 *       batch's.</li>
 * </ul>
 *
 * <p>The mode costs throughput: attention runs one decode launch per query row instead of a tiled or batched
 * prefill kernel, split-K is off, and a weight type without a fast GEMM lane projects through the matvec family at
 * every row count. The documentation records the measured cost. Immutable configuration read through static
 * methods.</p>
 */
public final class BatchInvariance {

    /** System property holding the mode: {@code true} (on) or {@code false} (off, the default). */
    public static final String PROPERTY = "neuralfree.batch.invariant";

    /** Environment variable holding the mode when the property is unset, as vLLM's {@code VLLM_BATCH_INVARIANT}. */
    public static final String ENV = "NEURALFREE_BATCH_INVARIANT";

    /** A test's explicit setting, or null when the mode is resolved from the property and the environment. */
    private static volatile Boolean override;

    /** The resolved property/environment value, or null before the first resolution. */
    private static volatile Boolean resolved;

    private BatchInvariance() {
    }

    /** Whether the process runs batch-invariant kernels. */
    public static boolean enabled() {
        Boolean forced = override;
        if (forced != null) {
            return forced;
        }
        Boolean value = resolved;
        if (value == null) {
            value = resolve();
            resolved = value;
        }
        return value;
    }

    /**
     * Forces the mode for this process, for tests; {@code null} clears the override.
     *
     * @see #clearOverride()
     */
    public static void setEnabled(Boolean on) {
        override = on;
    }

    /** Releases a {@link #setEnabled(Boolean)} override and forgets the resolved value, so the property and the
     *  environment are read again. */
    public static void clearOverride() {
        override = null;
        resolved = null;
    }

    /**
     * Parses a mode value: {@code true}, {@code 1}, {@code false} or {@code 0}, ignoring case and surrounding
     * blanks.
     *
     * @param value  the text to parse
     * @param source the property or variable the value came from, named in the refusal
     * @throws IllegalArgumentException when the value is not one of the four accepted forms
     */
    public static boolean parse(String value, String source) {
        String v = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        if (v.equals("true") || v.equals("1")) {
            return true;
        }
        if (v.equals("false") || v.equals("0")) {
            return false;
        }
        throw new IllegalArgumentException(source + "=" + value + " is not a boolean; use true or false"
                + " (vLLM's environment variable also takes 1 and 0)");
    }

    private static boolean resolve() {
        String property = System.getProperty(PROPERTY);
        if (property != null && !property.trim().isEmpty()) {
            return parse(property, "-D" + PROPERTY);
        }
        String environment = System.getenv(ENV);
        if (environment != null && !environment.trim().isEmpty()) {
            return parse(environment, ENV);
        }
        return false;
    }
}
