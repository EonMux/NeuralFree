package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

import java.util.Objects;

/**
 * The min-tokens stage (vLLM's {@code min_tokens}; the server bans end-of-generation tokens until the request's
 * minimum is generated): while fewer than {@code minTokens} tokens were accepted, every end-of-generation id of
 * the vocabulary takes a {@code -inf} logit, so the model cannot stop before its budget. A Java-API custom stage
 * ({@code Sampler.create(config, bannedTokens, extra, vocab)}): it always applies, and its {@link #modifies()}
 * keeps an argmax chain off the on-device argmax path while it still bans.
 */
public final class MinTokensStage extends SamplerStage {

    private final int minTokens;
    private final SamplerVocab vocab;
    private int accepted;

    /** A stage that bans the vocabulary's end-of-generation ids for the first {@code minTokens} accepted tokens. */
    public MinTokensStage(int minTokens, SamplerVocab vocab) {
        if (minTokens <= 0) {
            throw new IllegalArgumentException("min tokens needs a positive budget, got " + minTokens);
        }
        this.minTokens = minTokens;
        this.vocab = Objects.requireNonNull(vocab, "vocab");
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        if (accepted >= minTokens) {
            return;
        }
        for (int i = 0; i < candidates.size(); i++) {
            if (vocab.isEog(candidates.id(i))) {
                candidates.setLogit(i, Float.NEGATIVE_INFINITY);
            }
        }
    }

    @Override
    public void accept(int token) {
        accepted++;
    }

    @Override
    public boolean modifies() {
        return accepted < minTokens;
    }

    @Override
    public void reset() {
        accepted = 0;
    }
}
