package com.eonmux.neuralfree.gguf.engine.sampling;

/**
 * A constraint-head stage ({@link SamplerStage#constrains}, {@link SamplerStage#masks}) whose mask can be asked about
 * one token: {@link #keeps} answers what its {@link SamplerStage#apply} would do to that token's logit in the current
 * state. A greedy chain whose constraint head consists of such stages selects its token sample-first: it takes the
 * argmax of the unmasked row and asks the head whether it keeps that token, computing the whole-row mask only when a
 * stage refuses it ({@code Sampler#sample}). The masks set logits to {@code -inf} by token id alone and the stages
 * that run after the head are per-token edits, so a kept argmax is the masked argmax.
 */
public interface TokenGate {

    /**
     * Whether {@link SamplerStage#apply} over a vocabulary-ordered row would leave {@code token}'s finite logit
     * finite — the stages before it in the constraint head keeping it too. False when the stage would mask it, and
     * also when the answer depends on the rest of the candidate set (the caller then applies the whole mask). The
     * call makes the state changes an apply makes besides the logits (a grammar arms), so a kept token can be
     * accepted as if the stage had applied.
     */
    boolean keeps(int token);
}
