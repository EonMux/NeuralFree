package com.eonmux.neuralfree.gguf.grammar;

/**
 * Bounds for parsing and validating GBNF grammars, an untrusted input class for every engine
 * (PARITY §3.6 and §5.7 SO-1). A grammar that exceeds a bound fails to parse with an explicit
 * {@link GrammarException} naming the limit; nothing is silently truncated.
 *
 * <p>{@link #MAX_REPETITION_EXPANSION} takes over llama.cpp's {@code MAX_REPETITION_THRESHOLD}
 * ({@code src/llama-grammar.cpp:13}): it bounds the repetition count {@code {m,n}} per expansion
 * step and the product of nested repetitions built into one sequence ({@code llama-grammar.cpp:470-532}).
 * {@link #MAX_ELEMENT_MODEL} is also checked over the final rule set, matching the sum of all rule
 * element counts at construction time.</p>
 */
public final class GrammarLimits {

    /** Longest grammar source in characters; a larger source is rejected before parsing. */
    public static final int MAX_SOURCE_CHARS = 1 << 20;

    /** Largest number of distinct rule names one grammar may declare or reference. */
    public static final int MAX_RULES = 10_000;

    /** Largest number of elements the parsed rule set holds, counted while parsing. */
    public static final int MAX_ELEMENTS = 1 << 20;

    /** Largest per-axis repetition; counts above it are rejected or fall back to unbounded. */
    public static final int MAX_REPETITION_EXPANSION = 10_000;

    private GrammarLimits() {}
}
