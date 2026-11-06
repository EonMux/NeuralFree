package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Objects;
import java.util.function.DoubleSupplier;

/**
 * The dual-key Gumbel-max watermarker of vLLM's {@code DualKeyGumbelWatermarker}: two independent single-key
 * watermarkers (key A, the draft role, and key B, the target role) each select a candidate, and one routing draw
 * chooses between them — key B with probability {@code alpha}, key A with probability {@code 1 - alpha}. The routing
 * draw is taken on every call, including calls whose result the caller discards because the context repeats, so a
 * sequence's routing stream does not depend on the repeated-context search. Immutable.
 *
 * <p>{@code alpha == 0} and {@code alpha == 1} reduce to one key with no routing draw, as vLLM does. Detection fuses
 * the two keys' scores with the same weights ({@code DualKeyGumbelWatermarkDetector}).
 */
public final class DualKeyGumbelWatermarker extends Watermarker {

    private final GumbelWatermarker draftWatermarker;
    private final GumbelWatermarker targetWatermarker;
    private final double alpha;
    private final double keyAProbability;

    /**
     * @param key          the 64-bit secret key; the role keys are derived from it ({@link WatermarkConfig#keyA})
     * @param contextWidth how many prior tokens the PRF keys on, at least 1
     * @param alpha        the probability of routing a token to key B, in {@code [0, 1]}
     * @param prf          the pseudorandom function both roles use
     * @throws IllegalArgumentException when the context width is below 1 or {@code alpha} is outside {@code [0, 1]}
     */
    public DualKeyGumbelWatermarker(long key, int contextWidth, double alpha, WatermarkPrfName prf) {
        if (!(alpha >= 0.0 && alpha <= 1.0)) {
            throw new IllegalArgumentException("watermark alpha must be in [0, 1], got " + alpha);
        }
        Objects.requireNonNull(prf, "prf");
        this.alpha = alpha;
        this.draftWatermarker = new GumbelWatermarker(WatermarkConfig.keyA(key), contextWidth, prf);
        this.targetWatermarker = new GumbelWatermarker(WatermarkConfig.keyB(key), contextWidth, prf);
        this.keyAProbability = keyAProbability(alpha);
    }

    /** The routing probability of key A as vLLM's softmax over {@code log([1 - alpha, alpha])} resolves it. */
    private static double keyAProbability(double alpha) {
        if (alpha <= 0.0) {
            return 1.0;
        }
        if (alpha >= 1.0) {
            return 0.0;
        }
        double logA = Math.log(1.0 - alpha);
        double logB = Math.log(alpha);
        double max = Math.max(logA, logB);
        double weightA = Math.exp(logA - max);
        double weightB = Math.exp(logB - max);
        return weightA / (weightA + weightB);
    }

    /** The probability of routing a token to key B. */
    public double alpha() {
        return alpha;
    }

    /** The key-A (draft role) watermarker. */
    public GumbelWatermarker draftWatermarker() {
        return draftWatermarker;
    }

    /** The key-B (target role) watermarker. */
    public GumbelWatermarker targetWatermarker() {
        return targetWatermarker;
    }

    @Override
    public int contextWidth() {
        return draftWatermarker.contextWidth();
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalArgumentException when {@code alpha} is strictly between 0 and 1 and {@code routing} is null
     */
    @Override
    public int sample(int[] tokenIds, float[] logits, int count, int[] context, DoubleSupplier routing) {
        if (alpha <= 0.0) {
            return draftWatermarker.sample(tokenIds, logits, count, context, routing);
        }
        if (alpha >= 1.0) {
            return targetWatermarker.sample(tokenIds, logits, count, context, routing);
        }
        Objects.requireNonNull(routing, "routing");
        int keyAToken = draftWatermarker.sample(tokenIds, logits, count, context, routing);
        int keyBToken = targetWatermarker.sample(tokenIds, logits, count, context, routing);
        return routing.getAsDouble() < keyAProbability ? keyAToken : keyBToken;
    }

    @Override
    public String toString() {
        return "DualKeyGumbelWatermarker{alpha=" + alpha + ", contextWidth=" + contextWidth() + '}';
    }
}
