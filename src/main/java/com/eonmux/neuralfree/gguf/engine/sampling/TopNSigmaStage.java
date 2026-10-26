package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's top-n-sigma sampler (llama_sampler_top_n_sigma_apply, llama.cpp src/llama-sampler.cpp:3243): every
 * candidate whose logit is below {@code max - n * std} (mean and population standard deviation over the non
 * {@code -inf} logits) gets {@code -inf}; the surviving logits stay untouched, and the {@code sorted} flag is not
 * rewritten. {@code n <= 0} and single-candidate sets apply nothing, exactly as llama.cpp's empty sampler does.
 */
public final class TopNSigmaStage extends SamplerStage {

    private final float n;

    public TopNSigmaStage(float n) {
        this.n = n;
    }

    @Override
    public StageKind kind() {
        return StageKind.TOP_N_SIGMA;
    }

    @Override
    public boolean modifies() {
        return n > 0.0f;
    }

    @Override
    public void apply(Candidates candidates) {
        if (n <= 0.0f || candidates.size() <= 1) {
            return;
        }
        // find max logit and calculate mean (max over the non -inf logits, llama.cpp starts from candidate 0)
        float max = candidates.logit(0);
        float logitsSum = 0f;
        int validCount = 0;
        for (int i = 0; i < candidates.size(); i++) {
            float logit = candidates.logit(i);
            if (logit != Float.NEGATIVE_INFINITY) {
                if (logit > max) {
                    max = logit;
                }
                logitsSum += logit;
                validCount++;
            }
        }
        float mean = validCount > 0 ? logitsSum / validCount : 0f;

        // calculate standard deviation (over the non -inf logits)
        float acc = 0f;
        for (int i = 0; i < candidates.size(); i++) {
            float logit = candidates.logit(i);
            if (logit != Float.NEGATIVE_INFINITY) {
                float d = logit - mean;
                acc += d * d;
            }
        }
        float std = validCount > 0 ? (float) Math.sqrt(acc / validCount) : 0f;

        // apply mask
        float cutoff = max - (n * std);
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.logit(i) < cutoff) {
                candidates.setLogit(i, Float.NEGATIVE_INFINITY);
            }
        }
    }
}
