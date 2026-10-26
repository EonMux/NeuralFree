package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's nucleus sampler (llama_sampler_top_p_apply): the smallest prefix of the descending logit order whose
 * cumulative probability reaches {@code topP}, probabilities coming from the last softmax over the current logits. A
 * {@code topP >= 1} applies nothing. {@code minKeep} stops the truncation only once that many candidates are kept
 * (llama.cpp's {@code i + 1 >= min_keep} check). The survivors stay in descending order.
 */
public final class TopPStage extends SamplerStage {

    private final float topP;
    private final int minKeep;

    public TopPStage(float topP, int minKeep) {
        this.topP = topP;
        this.minKeep = minKeep;
    }

    @Override
    public StageKind kind() {
        return StageKind.TOP_P;
    }

    @Override
    public void apply(Candidates candidates) {
        if (topP >= 1f) {
            return;
        }
        candidates.softmax();
        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        float cum = 0f;
        int last = candidates.size();
        for (int i = 0; i < candidates.size(); i++) {
            cum += candidates.prob(i);
            if (cum >= topP && i + 1 >= minKeep) {
                last = i + 1;
                break;
            }
        }
        candidates.truncate(last);
    }
}
