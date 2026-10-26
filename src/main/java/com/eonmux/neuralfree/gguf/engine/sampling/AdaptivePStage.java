package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's adaptive-p sampler (llama_sampler_adaptive_p_apply, llama.cpp src/llama-sampler.cpp:3756): a terminal
 * selector that reshapes the candidate distribution around the exponential moving average of the selected tokens'
 * original probabilities - a quadratic-around-target logit transform ({@code 5 - 10 * d^2/(1 + d)}, {@code d} the
 * normalized distance to the adapted target; the algebraic constants DISTRIBUTION_WIDTH 0.3, PEAK_LOGIT_VALUE 5,
 * SHARPNESS 10), then a softmax and an llama_sample_dist draw with its own std::mt19937. It replaces the chain's dist
 * stage; llama.cpp appends it only when the order asks for it and never builds a dist stage behind it
 * (common/sampling.cpp:387-407). A negative {@code target} draws the untouched distribution instead, as llama.cpp's
 * no-op branch does.
 *
 * <p>{@code decay} clamps to [0, 0.99] at construction as llama.cpp's init does; {@code accept} folds the selected
 * token's original probability into the EMA exactly once per apply (the pending-token latch); {@link #reset()} restores
 * the fresh EMA and reseeds the draw, as llama_sampler_adaptive_p_reset does.
 */
public final class AdaptivePStage extends SamplerStage {

    private static final float DISTRIBUTION_WIDTH = 0.3f;
    private static final float PEAK_LOGIT_VALUE = 5.0f;
    private static final float SHARPNESS = 10.0f;
    private static final float INV_WIDTH = 1.0f / DISTRIBUTION_WIDTH;

    private final float target;
    private final float decay;
    private final long seed;
    private final Mt19937 rng;
    private float weightedSum;
    private float totalWeight;
    private float[] originalProbs = new float[0];
    private int pendingTokenId = -1;
    private int pendingTokenIdx = -1;

    public AdaptivePStage(float target, float decay, long seed) {
        this.target = target;
        this.decay = Math.max(0.0f, Math.min(0.99f, decay));
        this.seed = seed;
        this.rng = new Mt19937(seed);
        this.weightedSum = target / (1.0f - this.decay);
        this.totalWeight = 1.0f / (1.0f - this.decay);
    }

    @Override
    public StageKind kind() {
        return StageKind.ADAPTIVE_P;
    }

    @Override
    public void apply(Candidates candidates) {
        candidates.softmaxUnsorted();

        // store the original probabilities
        if (originalProbs.length < candidates.size()) {
            originalProbs = new float[candidates.size()];
        }
        for (int i = 0; i < candidates.size(); i++) {
            originalProbs[i] = candidates.prob(i);
        }

        if (target < 0.0f) {
            // at negative target values, adaptive-p is a no-op over the existing distribution; llama.cpp sets only
            // the selection here, so the EMA never folds these draws
            int idx = rng.distDraw(used(candidates, candidates.size()));
            candidates.select(idx);
            return;
        }

        float adaptedTarget = totalWeight == 0.0f
                ? Math.max(0.0f, Math.min(1.0f, target))
                : Math.max(0.0f, Math.min(1.0f, 2.0f * target - (weightedSum / totalWeight)));

        // adaptive probability transform: quadratic near the target, linear in the tails
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.logit(i) == Float.NEGATIVE_INFINITY) {
                continue;
            }
            float dist = Math.abs((candidates.prob(i) - adaptedTarget) * INV_WIDTH);
            candidates.setLogit(i, PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0f + dist));
        }

        candidates.softmaxUnsorted();
        int idx = rng.distDraw(used(candidates, candidates.size()));
        candidates.select(idx);
        pendingTokenId = candidates.id(idx);
        pendingTokenIdx = idx;
    }

    /** The current probabilities (the llama_sample_dist iterator range over cur_p). */
    private static float[] used(Candidates candidates, int n) {
        float[] probs = new float[n];
        for (int i = 0; i < n; i++) {
            probs[i] = candidates.prob(i);
        }
        return probs;
    }

    @Override
    public void accept(int token) {
        if (pendingTokenId == token) {
            weightedSum = originalProbs[pendingTokenIdx] + decay * weightedSum;
            totalWeight = 1.0f + decay * totalWeight;
        }
        pendingTokenId = -1;
        pendingTokenIdx = -1;
    }

    @Override
    public void reset() {
        weightedSum = target / (1.0f - decay);
        totalWeight = 1.0f / (1.0f - decay);
        pendingTokenId = -1;
        pendingTokenIdx = -1;
        rng.seed(seed);
    }
}
