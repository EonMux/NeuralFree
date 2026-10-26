package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's top-k sampler (llama_sampler_top_k_apply and llama_sampler_top_k_impl): the {@code topK} highest logits
 * in descending order; the order of the candidates below the cut is what {@link Candidates#partialSortDescending}
 * leaves (the full sort when {@code topK} is close to the set size, llama.cpp's
 * {@code llama_token_data_array_partial_sort_inplace}). A {@code topK <= 0} applies nothing. {@code minKeep} keeps at
 * least that many candidates, as the chain's {@code min_keep} would.
 */
public final class TopKStage extends SamplerStage {

    private final int topK;
    private final int minKeep;

    public TopKStage(int topK, int minKeep) {
        this.topK = topK;
        this.minKeep = minKeep;
    }

    @Override
    public StageKind kind() {
        return StageKind.TOP_K;
    }

    @Override
    public void apply(Candidates candidates) {
        if (topK <= 0) {
            return;
        }
        int k = Math.min(topK, candidates.size());
        if (k < minKeep) {
            k = Math.min(minKeep, candidates.size());
        }
        if (!candidates.sorted()) {
            candidates.partialSortDescending(k);
        }
        candidates.truncate(k);
    }
}
