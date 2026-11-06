package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Objects;

/**
 * The dual-key Gumbel watermark detector of vLLM's {@code DualKeyGumbelWatermarkDetector}: each token is scored under
 * both role keys and fused as {@code (1 - alpha) * scoreA + alpha * scoreB}. The fused score is a weighted sum of
 * Gamma variables, so the p-value divides the score and the token count by the fused variance
 * {@code (1 - alpha)^2 + alpha^2}; when the weights are equal, or one key takes everything, the null distribution is a
 * Gamma with an integer shape and the closed integer survival is used, otherwise its real-shaped survival.
 */
public final class DualKeyGumbelWatermarkDetector extends WatermarkDetector {

    private final WatermarkPrf keyAPrf;
    private final WatermarkPrf keyBPrf;
    private final double alpha;

    /**
     * @param key             the 64-bit secret key; the role keys are derived from it
     * @param contextWidth    how many prior tokens a context holds
     * @param pValueThreshold the p-value at or below which a text is watermarked
     * @param prf             the pseudorandom function both roles use
     * @param deduplicate     whether a repeated context is scored once
     * @param alpha           the weight of key B, in {@code [0, 1]}
     * @throws IllegalArgumentException when {@code alpha} is outside {@code [0, 1]}
     */
    public DualKeyGumbelWatermarkDetector(long key, int contextWidth, float pValueThreshold, WatermarkPrfName prf,
                                          boolean deduplicate, double alpha) {
        super(contextWidth, pValueThreshold, deduplicate);
        if (!(alpha >= 0.0 && alpha <= 1.0)) {
            throw new IllegalArgumentException("watermark alpha must be in [0, 1], got " + alpha);
        }
        Objects.requireNonNull(prf, "prf");
        if (prf != WatermarkPrfName.PHILOX) {
            throw new IllegalArgumentException("unsupported watermark PRF " + prf.label());
        }
        this.keyAPrf = new PhiloxPrf(WatermarkConfig.keyA(key));
        this.keyBPrf = new PhiloxPrf(WatermarkConfig.keyB(key));
        this.alpha = alpha;
    }

    /** The weight of key B. */
    public double alpha() {
        return alpha;
    }

    /** The key-A (draft role) PRF. */
    public WatermarkPrf keyAPrf() {
        return keyAPrf;
    }

    /** The key-B (target role) PRF. */
    public WatermarkPrf keyBPrf() {
        return keyBPrf;
    }

    @Override
    protected double score(int[] contexts, int[] targets, int count) {
        int width = contextWidth();
        double weightA = 1.0 - alpha;
        double weightB = alpha;
        double total = 0.0;
        for (int i = 0; i < count; i++) {
            int[] context = contextAt(contexts, width, i);
            double scoreA = -Math.log1p(-keyAPrf.uniform(context, targets[i]));
            double scoreB = -Math.log1p(-keyBPrf.uniform(context, targets[i]));
            total += weightA * scoreA + weightB * scoreB;
        }
        return total;
    }

    @Override
    protected double pValue(double score, int numScoredTokens) {
        double variance = (1.0 - alpha) * (1.0 - alpha) + alpha * alpha;
        double scaledScore = score / variance;
        if (alpha == 0.0 || alpha == 0.5 || alpha == 1.0) {
            return WatermarkGamma.integerSurvival(scaledScore, (int) Math.round(numScoredTokens / variance));
        }
        return WatermarkGamma.upperRegularized(numScoredTokens / variance, scaledScore);
    }
}
