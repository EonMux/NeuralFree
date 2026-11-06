package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Objects;

/**
 * The single-key Gumbel watermark detector of vLLM's {@code GumbelWatermarkDetector}: a token's score is
 * {@code -log(1 - PRF(context, token))}, and the sum's p-value is the upper tail of a Gamma distribution with a shape
 * equal to the scored token count, which is the null distribution of the sum when the draws are uniform.
 */
public final class GumbelWatermarkDetector extends WatermarkDetector {

    private final WatermarkPrf prf;

    /**
     * @param key             the 64-bit secret key (a {@code long} bit pattern)
     * @param contextWidth    how many prior tokens a context holds
     * @param pValueThreshold the p-value at or below which a text is watermarked
     * @param prf             the pseudorandom function
     * @param deduplicate     whether a repeated context is scored once
     */
    public GumbelWatermarkDetector(long key, int contextWidth, float pValueThreshold, WatermarkPrfName prf,
                                   boolean deduplicate) {
        super(contextWidth, pValueThreshold, deduplicate);
        Objects.requireNonNull(prf, "prf");
        if (prf != WatermarkPrfName.PHILOX) {
            throw new IllegalArgumentException("unsupported watermark PRF " + prf.label());
        }
        this.prf = new PhiloxPrf(key);
    }

    /** The pseudorandom function the detector scores with. */
    public WatermarkPrf prf() {
        return prf;
    }

    @Override
    protected double score(int[] contexts, int[] targets, int count) {
        int width = contextWidth();
        double total = 0.0;
        for (int i = 0; i < count; i++) {
            double uniform = prf.uniform(contextAt(contexts, width, i), targets[i]);
            total += -Math.log1p(-uniform);
        }
        return total;
    }

    @Override
    protected double pValue(double score, int numScoredTokens) {
        return WatermarkGamma.integerSurvival(score, numScoredTokens);
    }
}
