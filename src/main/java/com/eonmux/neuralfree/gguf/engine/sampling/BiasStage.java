package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

import java.util.Arrays;

/**
 * The logit bias stage every chain starts with (llama.cpp {@code llama_sampler_init_logit_bias}): before any
 * stage reordered the set — a candidate's index is still its token id — banned ids take a {@code -inf} logit and
 * biased ids take their bias added to the logit, a bias of {@code -100} or below acting as the ban (OpenAI's
 * {@code logit_bias} range and llama.cpp's threshold). The banned ids are {@code tokenizer.ggml.suppress_tokens},
 * the end-of-generation tokens under ignore-eos, and the request's {@code -100} entries. Not part of the
 * configurable order. Immutable apart from the shared candidate set.
 */
public final class BiasStage extends SamplerStage {

    private final int[] banned;
    private final int[] biasedIds;
    private final float[] biases;

    /** Copies {@code bannedTokens}; values must be non-negative. */
    public BiasStage(int[] bannedTokens) {
        this(bannedTokens, new int[0], new float[0]);
    }

    /**
     * A stage over banned ids and additive biases.
     *
     * @throws IllegalArgumentException for a negative id or a non-finite bias
     */
    public BiasStage(int[] bannedTokens, int[] biasedIds, float[] biases) {
        for (int id : bannedTokens) {
            if (id < 0) {
                throw new IllegalArgumentException("banned token id " + id + " is negative");
            }
        }
        if (biasedIds.length != biases.length) {
            throw new IllegalArgumentException(biasedIds.length + " biased ids carry " + biases.length + " biases");
        }
        for (int i = 0; i < biasedIds.length; i++) {
            if (biasedIds[i] < 0) {
                throw new IllegalArgumentException("biased token id " + biasedIds[i] + " is negative");
            }
            if (!Float.isFinite(biases[i])) {
                throw new IllegalArgumentException("the bias of token " + biasedIds[i] + " is not finite");
            }
        }
        this.banned = bannedTokens.clone();
        this.biasedIds = biasedIds.clone();
        this.biases = biases.clone();
    }

    /** The banned ids (a copy). */
    public int[] banned() {
        return banned.clone();
    }

    /** The additively biased ids (a copy). */
    public int[] biasedIds() {
        return biasedIds.clone();
    }

    /** The additive bias of each biased id (a copy). */
    public float[] biases() {
        return biases.clone();
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        for (int id : banned) {
            if (id < candidates.size()) {
                candidates.setLogit(id, Float.NEGATIVE_INFINITY);
            }
        }
        for (int i = 0; i < biasedIds.length; i++) {
            int id = biasedIds[i];
            if (id < candidates.size()) {
                if (biases[i] <= -100f) {
                    candidates.setLogit(id, Float.NEGATIVE_INFINITY);
                } else {
                    candidates.setLogit(id, candidates.logit(id) + biases[i]);
                }
            }
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof BiasStage)) {
            return false;
        }
        BiasStage other = (BiasStage) o;
        return Arrays.equals(banned, other.banned) && Arrays.equals(biasedIds, other.biasedIds)
                && Arrays.equals(biases, other.biases);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(banned);
        result = 31 * result + Arrays.hashCode(biasedIds);
        return 31 * result + Arrays.hashCode(biases);
    }

    @Override
    public String toString() {
        return "BiasStage{banned=" + banned.length + ", biased=" + biasedIds.length + "}";
    }
}
