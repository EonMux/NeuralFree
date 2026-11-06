package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.util.Objects;
import java.util.Properties;

/**
 * The settings of lookahead (Jacobi) decoding, the multi-branch decoding scheme of Fu et al. (BREAKING THE
 * SEQUENTIAL DEPENDENCY OF LLM DECODING, arXiv 2402.02057) that llama.cpp runs in its {@code lookahead} example: a
 * window of {@code W} Jacobi guesses refreshed every step, a fixed n-gram size {@code N}, and at most {@code G}
 * observed n-grams verified per step. Immutable; read with {@link #fromProperties(Properties)} or built with
 * {@link #builder()}.
 *
 * <p>Lookahead decoding evaluates its Jacobi iterations and its verification n-grams as parallel sequences of the
 * target's paged memory: {@code W + G + 1} sequences, exactly the {@code n_parallel} llama.cpp's example sets
 * ({@code W + G + 1} coupled sequences, unified KV cache). {@link LookaheadDecoder} therefore runs on a paged memory
 * whose {@code neuralfree.parallel} holds that many sequences and whose {@code neuralfree.kv.unified} lets each of
 * them span the context.
 *
 * <p>Properties (absent keys keep the defaults):
 * <ul>
 *   <li>{@code neuralfree.lookahead}: {@code true} turns lookahead decoding on with the defaults, {@code false}
 *       (default) leaves it off. Lookahead decoding replaces speculative decoding; a configuration that sets both is
 *       refused.</li>
 *   <li>{@code neuralfree.lookahead.window}: the Jacobi window {@code W}, how many guesses the lookahead maintains
 *       (default 15, llama.cpp's example {@code W}).</li>
 *   <li>{@code neuralfree.lookahead.ngram.size}: the n-gram size {@code N} (default 5, the example's {@code N}, at
 *       least 3); it bounds the tokens one verified n-gram accepts to {@code N - 1}.</li>
 *   <li>{@code neuralfree.lookahead.max.ngrams}: the verification n-grams {@code G} kept per first token and verified
 *       per step (default 15, the example's {@code G}).</li>
 * </ul>
 */
public final class LookaheadOptions {

    public static final String PROP_ENABLED = "neuralfree.lookahead";
    public static final String PROP_WINDOW = "neuralfree.lookahead.window";
    public static final String PROP_NGRAM_SIZE = "neuralfree.lookahead.ngram.size";
    public static final String PROP_MAX_NGRAMS = "neuralfree.lookahead.max.ngrams";

    /** Default Jacobi window (llama.cpp's lookahead example {@code W}). */
    public static final int DEFAULT_WINDOW = 15;
    /** Default n-gram size (llama.cpp's lookahead example {@code N}). */
    public static final int DEFAULT_NGRAM_SIZE = 5;
    /** Default verification n-grams per first token (llama.cpp's lookahead example {@code G}). */
    public static final int DEFAULT_MAX_NGRAMS = 15;

    /** Bounds the window, the n-gram size and the n-gram count to one sequence each. */
    public static final int MAX_WINDOW = 64;
    public static final int MAX_NGRAM_SIZE = 64;
    public static final int MAX_NGRAMS = 256;

    private static final LookaheadOptions DISABLED = new LookaheadOptions(false, DEFAULT_WINDOW, DEFAULT_NGRAM_SIZE,
            DEFAULT_MAX_NGRAMS);

    private final boolean enabled;
    private final int window;
    private final int ngramSize;
    private final int maxNgrams;

    private LookaheadOptions(boolean enabled, int window, int ngramSize, int maxNgrams) {
        if (window < 1 || window > MAX_WINDOW) {
            throw new IllegalArgumentException(PROP_WINDOW + " must be in [1, " + MAX_WINDOW + "], got " + window);
        }
        if (ngramSize < 3 || ngramSize > MAX_NGRAM_SIZE) {
            throw new IllegalArgumentException(PROP_NGRAM_SIZE + " must be in [3, " + MAX_NGRAM_SIZE + "], got "
                    + ngramSize);
        }
        if (maxNgrams < 1 || maxNgrams > MAX_NGRAMS) {
            throw new IllegalArgumentException(PROP_MAX_NGRAMS + " must be in [1, " + MAX_NGRAMS + "], got "
                    + maxNgrams);
        }
        if (enabled && window + maxNgrams + 1 > MemoryConfig.MAX_PARALLEL) {
            throw new IllegalArgumentException(PROP_WINDOW + "=" + window + " and " + PROP_MAX_NGRAMS + "="
                    + maxNgrams + " need " + (window + maxNgrams + 1) + " sequences, above the "
                    + MemoryConfig.MAX_PARALLEL + " a memory holds");
        }
        this.enabled = enabled;
        this.window = window;
        this.ngramSize = ngramSize;
        this.maxNgrams = maxNgrams;
    }

    /** Lookahead decoding off (the default). */
    public static LookaheadOptions disabled() {
        return DISABLED;
    }

    /** Lookahead decoding on with the default window, n-gram size and n-gram count. */
    public static LookaheadOptions defaults() {
        return new LookaheadOptions(true, DEFAULT_WINDOW, DEFAULT_NGRAM_SIZE, DEFAULT_MAX_NGRAMS);
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * The options of a property set; absent keys keep the defaults, so {@code neuralfree.lookahead} alone turns
     * decoding on with the llama.cpp example's sizes.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static LookaheadOptions fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        return builder()
                .enabled(EngineProperties.booleanProperty(p, PROP_ENABLED, false))
                .window(EngineProperties.intProperty(p, PROP_WINDOW, DEFAULT_WINDOW))
                .ngramSize(EngineProperties.intProperty(p, PROP_NGRAM_SIZE, DEFAULT_NGRAM_SIZE))
                .maxNgrams(EngineProperties.intProperty(p, PROP_MAX_NGRAMS, DEFAULT_MAX_NGRAMS))
                .build();
    }

    /** Whether lookahead decoding runs. */
    public boolean enabled() {
        return enabled;
    }

    /** The Jacobi window {@code W}. */
    public int window() {
        return window;
    }

    /** The n-gram size {@code N}: a verified n-gram accepts up to {@code N - 1} tokens. */
    public int ngramSize() {
        return ngramSize;
    }

    /** The verification n-grams {@code G} kept per first token. */
    public int maxNgrams() {
        return maxNgrams;
    }

    /** The sequences one lookahead step runs: the input row plus {@code W} Jacobi levels and {@code G} n-grams. */
    public int sequenceCount() {
        return window + maxNgrams + 1;
    }

    /**
     * The stored positions one lookahead step writes at most: one row per sequence for the input token, the
     * {@code G} verification n-grams of {@code N - 1} tokens, the {@code W - 1} first-level rows a sequence shares
     * (each written into every sequence that reads it) and the {@code N - 2} later levels.
     */
    public int stepRows() {
        return sequenceCount() + maxNgrams * (ngramSize - 1) + window * (window - 1) / 2 + window * (ngramSize - 2);
    }

    /**
     * The positions one step spans at most, from the input row through the deepest Jacobi or n-gram row: the
     * deepest row of a step sits at {@code n_past + window + ngramSize - 3}, or at {@code n_past + ngramSize - 1}
     * when a window of one writes no Jacobi levels.
     */
    public int positionSpan() {
        return window >= 2 ? window + ngramSize - 2 : ngramSize;
    }

    /** These options restated as a {@link Builder}. */
    public Builder toBuilder() {
        return builder().enabled(enabled).window(window).ngramSize(ngramSize).maxNgrams(maxNgrams);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof LookaheadOptions)) {
            return false;
        }
        LookaheadOptions other = (LookaheadOptions) o;
        return enabled == other.enabled && window == other.window && ngramSize == other.ngramSize
                && maxNgrams == other.maxNgrams;
    }

    @Override
    public int hashCode() {
        return Objects.hash(enabled, window, ngramSize, maxNgrams);
    }

    @Override
    public String toString() {
        return enabled ? "Lookahead{window=" + window + ", ngram=" + ngramSize + ", maxNgrams=" + maxNgrams
                + ", sequences=" + sequenceCount() + '}' : "Lookahead{off}";
    }

    /** Configures {@link LookaheadOptions}. */
    public static final class Builder {
        private boolean enabled;
        private int window = DEFAULT_WINDOW;
        private int ngramSize = DEFAULT_NGRAM_SIZE;
        private int maxNgrams = DEFAULT_MAX_NGRAMS;

        private Builder() {
        }

        /** Whether lookahead decoding runs. */
        public Builder enabled(boolean v) {
            this.enabled = v;
            return this;
        }

        /** The Jacobi window {@code W}. */
        public Builder window(int v) {
            this.window = v;
            return this;
        }

        /** The n-gram size {@code N}. */
        public Builder ngramSize(int v) {
            this.ngramSize = v;
            return this;
        }

        /** The verification n-grams {@code G} kept per first token. */
        public Builder maxNgrams(int v) {
            this.maxNgrams = v;
            return this;
        }

        /**
         * Builds the options.
         *
         * @throws IllegalArgumentException for a window, n-gram size or n-gram count out of range, or a sequence
         *                                  count above the memory's
         */
        public LookaheadOptions build() {
            return new LookaheadOptions(enabled, window, ngramSize, maxNgrams);
        }
    }
}
