package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

/**
 * One mutable stage of a {@link Sampler} chain, applying llama.cpp's sampler semantics to the candidate set of one
 * chain run ({@link Sampler.Candidates}). {@code Sampler} owns the stages it builds: it applies them in the configured
 * chain order, tells them which tokens were accepted, and resets them. A stage keeps mutable per-sampler state (the
 * penalty history, scratch buffers), so it is not thread-safe, and it must not be shared between samplers. A stage
 * can save and restore its token history ({@link #saveState}) and, when it constrains the output, tell whether a
 * token may come next ({@link #admits}).
 *
 * <p>The fixed stages around the configured chain have {@link #kind()} {@code null}: the bias stage always runs before
 * the chain and the dist stage after it, and they are not part of the order. Java-API custom stages (an argument of
 * {@code Sampler.create(config, banned, extra)} or of its logits-processor form) also report {@code null} and always
 * apply; a logits processor runs at the head of the configured chain.
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

    /**
     * Whether the stage edits the candidate set under its configuration: false only when it provably applies nothing
     * (a llama.cpp disable gate), defaulting to true. {@code Sampler} uses it to tell an argmax-only chain apart.
     */
    public boolean modifies() {
        return true;
    }

    /** Records an accepted token with the stage's history; the token id must be non-negative. */
    public void accept(int token) {
    }

    /** Returns the stage to its state after construction (before any accepted token). */
    public void reset() {
    }

    /**
     * A copy of the stage's state, everything {@link #accept} has recorded and whatever its applies carry to the
     * next row (an armed flag, mirostat's surprise target), which {@link #restoreState} puts back; null (the default)
     * for a stage that keeps nothing. Speculative verification saves the chain before it feeds draft tokens ahead of
     * their confirmation and restores it when a rejected draft token has to leave the history (vLLM's grammar
     * matcher rollback). A stage that keeps any such state overrides both methods. Random streams and scratch
     * buffers are not part of the state, so a restore never replays a draw.
     */
    public Object saveState() {
        return null;
    }

    /**
     * Puts back a state {@link #saveState} returned on this stage; the stage then behaves as it did when the state
     * was saved. The state stays valid, so it can be restored again.
     */
    public void restoreState(Object state) {
    }

    /**
     * Whether the stage's current state lets {@code token} be accepted next. A constraint (a grammar) answers false
     * for a token its {@link #accept} would refuse; every other stage admits every token (the default).
     */
    public boolean admits(int token) {
        return true;
    }

    /**
     * Whether {@link #admits} can refuse a token: the stage constrains the accepted sequence (default false).
     * A chain applies its constraining stages as its constraint head, before every configured stage, so the
     * truncating stages operate on the admitted set ({@code Sampler.apply}).
     */
    public boolean constrains() {
        return false;
    }

    /**
     * Whether the stage hard-masks the current step's candidates (logits to {@code -inf}) and so belongs in the
     * chain's constraint head too, before every truncating stage (vLLM applies {@code allowed_token_ids},
     * bad words and {@code min_tokens} to the logits before any sampling stage; default false). The head reads
     * it alongside {@link #constrains} and applies both kinds in chain order. A masking stage does not answer
     * {@link #admits}, so it stays out of the sequence checks ({@code Sampler#admittedPrefix}) that run over the
     * constraining stages; the reasoning budget's forcing mask is not one — it keeps the final say, applied
     * behind the truncating stages.
     */
    public boolean masks() {
        return false;
    }
}
