package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * A watermark detector: it scores the tokens of a text under the keyed PRF and turns the score into a p-value under
 * the null hypothesis that the text was not watermarked. This is vLLM's {@code WatermarkDetector}: every token is
 * scored as {@code -log(1 - PRF(context, token))}, which is 1 in expectation for unwatermarked text and larger when
 * the token was drawn to favour the key's high draws. A text is reported watermarked when the p-value reaches the
 * threshold.
 *
 * <p>Each token's context is the {@code context_width} tokens before it, padded with {@code -1} before the start. With
 * {@code deduplicateContexts} on, a context that already occurred is scored once: a repeated context reuses one keyed
 * random vector, so its later tokens are not independent evidence. Immutable and thread-safe; subclasses define the
 * per-token score and the p-value.
 */
public abstract class WatermarkDetector {

    /** vLLM's default p-value threshold. */
    public static final float DEFAULT_P_VALUE_THRESHOLD = 0.01f;

    private final int contextWidth;
    private final float pValueThreshold;
    private final boolean deduplicateContexts;

    /**
     * @param contextWidth        how many prior tokens a context holds, at least 1
     * @param pValueThreshold     the p-value at or below which a text is watermarked, in {@code (0, 1)}
     * @param deduplicateContexts whether a repeated context is scored once
     * @throws IllegalArgumentException for a width below 1 or a threshold outside {@code (0, 1)}
     */
    protected WatermarkDetector(int contextWidth, float pValueThreshold, boolean deduplicateContexts) {
        if (contextWidth < 1) {
            throw new IllegalArgumentException("watermark context width must be 1 or more, got " + contextWidth);
        }
        if (!(pValueThreshold > 0f && pValueThreshold < 1f)) {
            throw new IllegalArgumentException("p-value threshold must be in (0, 1), got " + pValueThreshold);
        }
        this.contextWidth = contextWidth;
        this.pValueThreshold = pValueThreshold;
        this.deduplicateContexts = deduplicateContexts;
    }

    /**
     * The detector of a configuration: the single-key detector for {@link WatermarkAlgorithm#GUMBEL}, the fused
     * dual-key detector for {@link WatermarkAlgorithm#DUAL_KEY_GUMBEL}.
     *
     * @param deduplicateContexts whether a repeated context is scored once (vLLM's detection default is true)
     */
    public static WatermarkDetector create(WatermarkConfig config, float pValueThreshold,
                                           boolean deduplicateContexts) {
        if (config == null) {
            throw new NullPointerException("config");
        }
        if (config.algorithm() == WatermarkAlgorithm.DUAL_KEY_GUMBEL) {
            return new DualKeyGumbelWatermarkDetector(config.key(), config.contextWidth(), pValueThreshold,
                    config.prf(), deduplicateContexts, config.alpha());
        }
        if (config.algorithm() == WatermarkAlgorithm.GUMBEL) {
            return new GumbelWatermarkDetector(config.key(), config.contextWidth(), pValueThreshold, config.prf(),
                    deduplicateContexts);
        }
        throw new IllegalArgumentException("unknown watermarking algorithm " + config.algorithm());
    }

    /** {@link #create(WatermarkConfig, float, boolean)} at vLLM's defaults (threshold 0.01, contexts deduplicated). */
    public static WatermarkDetector create(WatermarkConfig config) {
        return create(config, DEFAULT_P_VALUE_THRESHOLD, true);
    }

    /** The number of prior tokens a context holds. */
    public int contextWidth() {
        return contextWidth;
    }

    /** The p-value at or below which a text is watermarked. */
    public float pValueThreshold() {
        return pValueThreshold;
    }

    /** Whether a repeated context is scored once. */
    public boolean deduplicateContexts() {
        return deduplicateContexts;
    }

    /**
     * Scores {@code tokenIds} and returns the detection.
     *
     * @throws IllegalArgumentException for a negative token id
     */
    public final WatermarkDetection detect(int[] tokenIds) {
        if (tokenIds == null) {
            throw new NullPointerException("tokenIds");
        }
        int width = contextWidth;
        int[] prefix = new int[width];
        Arrays.fill(prefix, -1);
        int[] contexts = new int[tokenIds.length * width];
        int[] targets = new int[tokenIds.length];
        Set<ContextTokens> seen = deduplicateContexts ? new HashSet<>() : null;
        int scored = 0;
        for (int tokenId : tokenIds) {
            if (tokenId < 0) {
                throw new IllegalArgumentException("token id " + tokenId + " is negative");
            }
            ContextTokens context = new ContextTokens(prefix);
            if (seen == null || seen.add(context)) {
                System.arraycopy(prefix, 0, contexts, scored * width, width);
                targets[scored] = tokenId;
                scored++;
            }
            if (width > 1) {
                System.arraycopy(prefix, 1, prefix, 0, width - 1);
            }
            prefix[width - 1] = tokenId;
        }
        if (scored == 0) {
            return new WatermarkDetection(0.0, 1.0, 0, false);
        }
        double score = score(contexts, targets, scored);
        double pValue = pValue(score, scored);
        return new WatermarkDetection(score, pValue, scored, pValue <= pValueThreshold);
    }

    /**
     * The summed score of the scored tokens.
     *
     * @param contexts {@code count * contextWidth()} token ids, oldest first per token
     * @param targets  the scored token ids
     * @param count    how many tokens are scored
     */
    protected abstract double score(int[] contexts, int[] targets, int count);

    /** The probability of a score this large or larger under unwatermarked text. */
    protected abstract double pValue(double score, int numScoredTokens);

    /** Reads one token's context from the flat context array. */
    static int[] contextAt(int[] contexts, int width, int index) {
        return Arrays.copyOfRange(contexts, index * width, index * width + width);
    }

    /** A token context usable as a set key. */
    private static final class ContextTokens {
        private final int[] tokens;
        private final int hash;

        ContextTokens(int[] tokens) {
            this.tokens = tokens.clone();
            this.hash = Arrays.hashCode(this.tokens);
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof ContextTokens && Arrays.equals(tokens, ((ContextTokens) o).tokens);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }
}
