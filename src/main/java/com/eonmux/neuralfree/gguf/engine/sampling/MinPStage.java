package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's min-p sampler (llama_sampler_min_p_apply): candidates whose logit is within {@code log(minP)} of the
 * maximum logit. A {@code minP <= 0} applies nothing. Over an unsorted set it moves the kept candidates to the front
 * in their current (vocabulary) order; {@code minKeep} candidates or more make that pass the result (llama.cpp's
 * {@code filtered_tokens}), otherwise the set is left untouched and llama.cpp's sorted fallback runs over all of it:
 * a full descending sort and a prefix scan that
 * only stops once {@code minKeep} candidates are past ({@code i >= min_keep}). The sorted set keeps its order.
 */
public final class MinPStage extends SamplerStage {

    private final float minP;
    private final int minKeep;

    public MinPStage(float minP, int minKeep) {
        this.minP = minP;
        this.minKeep = minKeep;
    }

    @Override
    public StageKind kind() {
        return StageKind.MIN_P;
    }

    @Override
    public void apply(Candidates candidates) {
        if (minP <= 0f || candidates.size() == 0) {
            return;
        }
        if (!candidates.sorted()) {
            float max = candidates.maxLogit();
            float minLogit = max + (float) Math.log(minP);
            int passing = 0;
            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.logit(i) >= minLogit) {
                    passing++;
                }
            }
            if (passing > 0 && passing >= minKeep) {
                int kept = 0;
                for (int i = 0; i < candidates.size(); i++) {
                    if (candidates.logit(i) >= minLogit) {
                        candidates.move(i, kept++);
                    }
                }
                candidates.truncate(kept);
                return;
            }
        }
        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        float minLogit = candidates.logit(0) + (float) Math.log(minP);
        int i = 1;
        while (i < candidates.size() && !(candidates.logit(i) < minLogit && i >= minKeep)) {
            i++;
        }
        candidates.truncate(i);
    }
}
