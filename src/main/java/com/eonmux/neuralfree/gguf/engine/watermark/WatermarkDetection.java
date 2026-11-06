package com.eonmux.neuralfree.gguf.engine.watermark;

/**
 * The result of {@link WatermarkDetector#detect}: the summed token score, the p-value of the score under the null
 * hypothesis that the text was not watermarked, the number of scored tokens, and whether the p-value reached the
 * detector's threshold. Immutable.
 */
public final class WatermarkDetection {

    private final double score;
    private final double pValue;
    private final int numScoredTokens;
    private final boolean isWatermarked;

    /** A detection result. */
    public WatermarkDetection(double score, double pValue, int numScoredTokens, boolean isWatermarked) {
        if (numScoredTokens < 0) {
            throw new IllegalArgumentException("numScoredTokens must be 0 or more, got " + numScoredTokens);
        }
        this.score = score;
        this.pValue = pValue;
        this.numScoredTokens = numScoredTokens;
        this.isWatermarked = isWatermarked;
    }

    /** The sum of the per-token scores; larger means more evidence of the watermark. */
    public double score() {
        return score;
    }

    /** The probability of a score this large or larger under unwatermarked text. */
    public double pValue() {
        return pValue;
    }

    /** How many tokens the detector scored (a repeated context is scored once). */
    public int numScoredTokens() {
        return numScoredTokens;
    }

    /** Whether {@link #pValue()} is at most the detector's threshold. */
    public boolean isWatermarked() {
        return isWatermarked;
    }

    @Override
    public String toString() {
        return "WatermarkDetection{score=" + score + ", pValue=" + pValue + ", numScoredTokens="
                + numScoredTokens + ", isWatermarked=" + isWatermarked + '}';
    }
}
