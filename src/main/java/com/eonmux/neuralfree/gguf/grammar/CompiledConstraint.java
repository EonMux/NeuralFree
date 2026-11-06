package com.eonmux.neuralfree.gguf.grammar;

import java.util.Objects;

/**
 * A {@link ConstraintSpec} compiled for a sampler: its validated {@link Grammar} and the {@link GrammarTriggers}
 * that gate it, or null triggers when the grammar constrains the output from the first token
 * ({@link GbnfBuilder#compile}). Immutable.
 */
public final class CompiledConstraint {

    private final Grammar grammar;
    private final GrammarTriggers triggers;

    /**
     * @param grammar  the validated grammar
     * @param triggers the gating triggers, or null for an always-on grammar
     */
    public CompiledConstraint(Grammar grammar, GrammarTriggers triggers) {
        this.grammar = Objects.requireNonNull(grammar, "grammar");
        this.triggers = triggers;
    }

    public Grammar grammar() {
        return grammar;
    }

    /** The gating triggers, or null for an always-on grammar. */
    public GrammarTriggers triggers() {
        return triggers;
    }

    /** Whether the grammar waits for a trigger before it constrains anything. */
    public boolean gated() {
        return triggers != null;
    }
}
