package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

/**
 * One mutable stage of a {@link Sampler} chain, applying llama.cpp's sampler semantics to the candidate set of one
 * chain run ({@link Sampler.Candidates}). {@code Sampler} owns the stages it builds: it applies them in the configured
 * chain order, tells them which tokens were accepted, and resets them. A stage keeps mutable per-sampler state (the
 * penalty history, scratch buffers), so it is not thread-safe, and it must not be shared between samplers.
 *
 * <p>The fixed stages around the configured chain have {@link #kind()} {@code null}: the bias stage always runs before
 * the chain and the dist stage after it, and they are not part of the order. Java-API custom stages (an argument of
 * {@code Sampler.create(config, banned, extra)}) also report {@code null} and always apply.
 */
public abstract class SamplerStage {

    /**
     * The llama.cpp kind of a built-in (configurable) stage, or {@code null} for a fixed stage (bias) or a Java-API
     * custom stage.
     */
    public StageKind kind() {
        return null;
    }

    /** Applies the stage to {@code candidates}. */
    public abstract void apply(Sampler.Candidates candidates);

    /** Records an accepted token with the stage's history; the token id must be non-negative. */
    public void accept(int token) {
    }

    /** Returns the stage to its state after construction (before any accepted token). */
    public void reset() {
    }
}
