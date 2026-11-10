package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;

import java.util.Objects;

/**
 * The min-tokens stage (vLLM's {@code min_tokens}; the server bans end-of-generation tokens until the request's
 * minimum is generated): while fewer than {@code minTokens} generated tokens were accepted, every
 * end-of-generation id of the vocabulary takes a {@code -inf} logit, so the model cannot stop before its budget.
 * Only generated tokens spend the budget: {@code accept} counts from the stage's first {@link #apply} on, so the
 * prompt tokens the sampler's history takes (the penalty stages' history) never spend it. The ban holds only
 * while a different candidate stays selectable: when the stages before it (a completed grammar, the request's
 * other masks) left nothing but end-of-generation candidates, banning those too would empty the set, so the ban
 * yields and the generation can stop (vLLM's {@code MinTokensLogitsProcessor} restores the stop-token logits in
 * that case); the request's stage order therefore builds it behind the other masks. A masking stage ({@link
 * #masks()}): the chain applies it from the constraint head, before the configured chain. A Java-API custom
 * stage ({@code Sampler.create(config, bannedTokens, extra, vocab)}): it always applies, and its {@link
 * #modifies()} keeps an argmax chain off the on-device argmax path while it still bans.
 */
public final class MinTokensStage extends SamplerStage implements TokenGate {

    private final int minTokens;
    private final SamplerVocab vocab;
    private int accepted;
    private boolean armed;

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
        armed = true;
        if (accepted >= minTokens) {
            return;
        }
        for (int i = 0; i < candidates.size(); i++) {
            if (!vocab.isEog(candidates.id(i)) && Float.isFinite(candidates.logit(i))) {
                ban(candidates);
                return;
            }
        }
    }

    /**
     * Whether {@link #apply} keeps {@code token}: always once the budget is spent; a token that is not an
     * end-of-generation id while it lasts, since that finite token keeps the ban from yielding. An end-of-generation
     * token is refused while the budget lasts, as the ban yields only when nothing else is left, which the whole
     * row decides. Arms the stage, as an apply does.
     */
    @Override
    public boolean keeps(int token) {
        armed = true;
        return accepted >= minTokens || !vocab.isEog(token);
    }

    /** Sets every end-of-generation logit to {@code -inf}. */
    private void ban(Sampler.Candidates candidates) {
        for (int i = 0; i < candidates.size(); i++) {
            if (vocab.isEog(candidates.id(i))) {
                candidates.setLogit(i, Float.NEGATIVE_INFINITY);
            }
        }
    }

    @Override
    public void accept(int token) {
        if (!armed) {
            return; // prompt history: only generated tokens spend the budget (see class doc)
        }
        accepted++;
    }

    @Override
    public boolean modifies() {
        return accepted < minTokens;
    }

    @Override
    public boolean masks() {
        return true;
    }

    @Override
    public void reset() {
        accepted = 0;
        armed = false;
    }

    /** The generated-token count and the armed flag. */
    @Override
    public Object saveState() {
        return new int[]{accepted, armed ? 1 : 0};
    }

    @Override
    public void restoreState(Object state) {
        if (!(state instanceof int[]) || ((int[]) state).length != 2) {
            throw new IllegalArgumentException("not a min-tokens stage state: " + state);
        }
        int[] s = (int[]) state;
        accepted = s[0];
        armed = s[1] != 0;
    }
}
