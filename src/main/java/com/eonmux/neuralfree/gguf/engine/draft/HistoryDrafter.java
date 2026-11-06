package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.engine.DraftModel;

/**
 * A {@link DraftModel} that drafts from a token record alone, with no hidden states, draft model or target
 * forward ({@link NgramDrafter}, {@link SuffixDrafter}). Beside {@link DraftModel#bindTarget}, which reads the
 * target memory's single sequence, it binds to any {@link TokenHistory}: the batch engine binds one drafter per
 * running sequence to that sequence's paged token record, so every sequence keeps its own drafter state (llama.cpp
 * server's per-slot speculative state, vLLM's per-request n-gram and suffix proposers).
 */
public interface HistoryDrafter extends DraftModel {

    /**
     * Binds this drafter to {@code history} instead of a target model; called once, in place of
     * {@link #bindTarget}.
     *
     * @throws IllegalStateException when the drafter is closed or already bound
     */
    void bindHistory(TokenHistory history);
}
