package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A {@link ConstraintSpec} compiled for a sampler: its validated {@link Grammar} and the {@link GrammarTriggers}
 * that gate it, or null triggers when the grammar constrains the output from the first token
 * ({@link GbnfBuilder#compile}). A structural tag whose dispatching format names {@code excludes} or
 * {@code exclude_tokens} also carries the strings and token ids its free text must never write; the stage that
 * runs the triggers masks them while the output is free. Immutable.
 */
public final class CompiledConstraint {

    private final Grammar grammar;
    private final GrammarTriggers triggers;
    private final List<String> freeExcludes;
    private final int[] freeExcludeTokens;

    /**
     * @param grammar  the validated grammar
     * @param triggers the gating triggers, or null for an always-on grammar
     */
    public CompiledConstraint(Grammar grammar, GrammarTriggers triggers) {
        this(grammar, triggers, null, null);
    }

    /**
     * @param grammar           the validated grammar
     * @param triggers          the gating triggers, or null for an always-on grammar
     * @param freeExcludes      strings the free text of a gated grammar must not write, or null
     * @param freeExcludeTokens token ids the free text of a gated grammar must not sample, or null
     */
    public CompiledConstraint(Grammar grammar, GrammarTriggers triggers, List<String> freeExcludes,
            int[] freeExcludeTokens) {
        this.grammar = Objects.requireNonNull(grammar, "grammar");
        this.triggers = triggers;
        this.freeExcludes = freeExcludes == null ? Collections.emptyList()
                : Collections.unmodifiableList(new ArrayList<>(freeExcludes));
        this.freeExcludeTokens = freeExcludeTokens == null ? new int[0] : freeExcludeTokens.clone();
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

    /** The strings the free text must not write; empty when none. */
    public List<String> freeExcludes() {
        return freeExcludes;
    }

    /** The token ids the free text must not sample, as a copy; empty when none. */
    public int[] freeExcludeTokens() {
        return freeExcludeTokens.clone();
    }
}
