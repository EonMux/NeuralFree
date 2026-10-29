package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;

import java.util.ArrayList;
import java.util.List;

/**
 * The request-driven parity stages (PARITY P5): {@link MinTokensStage} for {@code min_tokens} and
 * {@link BadWordsStage} for {@code bad_words}, built from one {@link GenerationRequest} for the sampler's
 * Java-API extra stages ({@code Sampler.create(config, bannedTokens, extra, vocab)}). Empty when the request
 * sets neither.
 */
public final class RequestStages {

    private RequestStages() {
    }

    /** The request's stages; empty when it sets neither {@code minTokens} nor {@code badWords}. */
    public static List<SamplerStage> of(GenerationRequest request, SamplerVocab vocab) {
        List<SamplerStage> stages = new ArrayList<>(2);
        if (request.minTokens() > 0) {
            stages.add(new MinTokensStage(request.minTokens(), vocab));
        }
        BadWordsStage bad = BadWordsStage.of(request.badWords(), vocab);
        if (bad != null) {
            stages.add(bad);
        }
        return stages;
    }
}
