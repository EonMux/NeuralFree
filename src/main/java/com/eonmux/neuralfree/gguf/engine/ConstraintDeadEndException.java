package com.eonmux.neuralfree.gguf.engine;

/**
 * A constraint whose admitted set is empty: after the constraint stages masked the candidate set, no token keeps
 * a finite logit, so the step has nothing to sample (a grammar dead end — the text generated so far left no
 * partial parse that any vocabulary token continues — or the request's masks banning every id, its {@code
 * allowed_token_ids} behind its bad words for example; llama.cpp samples behind its grammar mask, vLLM applies
 * the structured-output bitmask and its {@code allowed_token_ids} and bad words before any sampling stage and
 * has no candidate to draw either). The request's own inputs produced it, so the server answers it with a 400
 * naming the constraint rather than the 500 a plain {@link IllegalStateException} maps to.
 */
public final class ConstraintDeadEndException extends IllegalStateException {

    private static final long serialVersionUID = 1L;

    /** The refusal naming the constraint state that admitted no candidate. */
    public ConstraintDeadEndException(String message) {
        super(message);
    }
}
