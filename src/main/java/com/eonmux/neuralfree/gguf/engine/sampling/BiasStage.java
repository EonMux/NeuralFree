package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

/**
 * The logit bias stage every chain starts with (llama.cpp llama_sampler_logit_bias_apply over the banned tokens:
 * {@code tokenizer.ggml.suppress_tokens}, end-of-generation tokens under ignore-eos): a banned token id whose
 * candidate index is still its token id, before any stage reordered the set, gets a {@code -inf} logit. Not part of
 * the configurable order. Immutable apart from the shared candidate set.
 */
public final class BiasStage extends SamplerStage {

    private final int[] banned;

    /** Copies {@code bannedTokens}; values must be non-negative. */
    public BiasStage(int[] bannedTokens) {
        this.banned = bannedTokens.clone();
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        for (int id : banned) {
            if (id < candidates.size()) {
                candidates.setLogit(id, Float.NEGATIVE_INFINITY);
            }
        }
    }
}
