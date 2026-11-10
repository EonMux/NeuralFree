package com.eonmux.neuralfree.gguf.grammar;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Objects;

/**
 * {@link GbnfBuilder#compile(ConstraintSpec, GrammarVocab)} over one vocabulary, remembering the most recently used
 * compiled constraints by spec. Agents and tool-calling clients send the same schema, tool set or grammar request
 * after request; a remembered spec skips the schema translation and the grammar parse. The compiled constraint is
 * immutable, so every request of an engine shares one; equal specs ({@link ConstraintSpec#equals}) compile to equal
 * grammars over the same vocabulary. A spec that fails to compile is not remembered: it throws again on every
 * request. Thread-safe.
 *
 * <p>What the compiler keeps is bounded twice: by entry count, and by the heap the kept specs and grammars hold,
 * estimated from the grammar's element count ({@link Grammar#elementCount}) and the spec's text
 * ({@link #estimatedBytes}). Least recently used entries go first, and a constraint whose estimate exceeds a quarter
 * of the budget is compiled for its request and never kept, so a few very large schemas or tool sets cannot pin the
 * budget or push out the constraints that recur.
 */
public final class ConstraintCompiler {

    /** How many compiled constraints an engine keeps by default. */
    public static final int DEFAULT_CAPACITY = 32;
    /** The heap, in estimated bytes, an engine's kept constraints may hold by default. */
    public static final long DEFAULT_BUDGET_BYTES = 16L << 20;
    /** Estimated heap of one grammar element, with its share of the rule lists and symbol tables. */
    static final long BYTES_PER_ELEMENT = 64;

    private static final class Entry {
        final CompiledConstraint compiled;
        final long bytes;

        Entry(CompiledConstraint compiled, long bytes) {
            this.compiled = compiled;
            this.bytes = bytes;
        }
    }

    private final GrammarVocab vocab;
    private final int capacity;
    private final long budgetBytes;
    private final LinkedHashMap<ConstraintSpec, Entry> recent = new LinkedHashMap<>(16, 0.75f, true);
    private long heldBytes;

    /**
     * @param vocab       the vocabulary {@code <token>} references resolve through, or null to reject them
     * @param capacity    how many compiled constraints to keep, at least 1
     * @param budgetBytes the estimated heap the kept constraints may hold, at least 1
     */
    public ConstraintCompiler(GrammarVocab vocab, int capacity, long budgetBytes) {
        if (capacity < 1) {
            throw new IllegalArgumentException("capacity must be at least 1, got " + capacity);
        }
        if (budgetBytes < 1) {
            throw new IllegalArgumentException("budget must be at least 1 byte, got " + budgetBytes);
        }
        this.vocab = vocab;
        this.capacity = capacity;
        this.budgetBytes = budgetBytes;
    }

    /** A compiler keeping {@code capacity} constraints within {@link #DEFAULT_BUDGET_BYTES}. */
    public ConstraintCompiler(GrammarVocab vocab, int capacity) {
        this(vocab, capacity, DEFAULT_BUDGET_BYTES);
    }

    /** A compiler keeping {@link #DEFAULT_CAPACITY} constraints within {@link #DEFAULT_BUDGET_BYTES}. */
    public ConstraintCompiler(GrammarVocab vocab) {
        this(vocab, DEFAULT_CAPACITY, DEFAULT_BUDGET_BYTES);
    }

    /**
     * The compiled {@code spec}: the remembered one when an equal spec compiled before and is still kept, else a
     * fresh compilation, which is then kept unless it is too large to.
     *
     * @throws IllegalArgumentException when the constraint is malformed or untranslatable
     */
    public CompiledConstraint compile(ConstraintSpec spec) {
        Objects.requireNonNull(spec, "spec");
        synchronized (recent) {
            Entry kept = recent.get(spec);
            if (kept != null) {
                return kept.compiled;
            }
        }
        CompiledConstraint compiled = GbnfBuilder.compile(spec, vocab);
        long bytes = estimatedBytes(spec, compiled);
        if (bytes > budgetBytes / 4) {
            return compiled;
        }
        synchronized (recent) {
            Entry raced = recent.get(spec);
            if (raced != null) {
                return raced.compiled;
            }
            recent.put(spec, new Entry(compiled, bytes));
            heldBytes += bytes;
            evict();
            return compiled;
        }
    }

    /** Drops the least recently used entries until the count and the estimated heap are within bounds. */
    private void evict() {
        Iterator<Entry> eldest = recent.values().iterator();
        while ((recent.size() > capacity || heldBytes > budgetBytes) && eldest.hasNext()) {
            heldBytes -= eldest.next().bytes;
            eldest.remove();
        }
    }

    /**
     * The heap {@code spec} and its compilation are estimated to hold while kept: {@link #BYTES_PER_ELEMENT} per
     * grammar element, and two bytes per character of the spec's text, choices, whitespace pattern and triggers.
     */
    static long estimatedBytes(ConstraintSpec spec, CompiledConstraint compiled) {
        long chars = length(spec.text()) + length(spec.whitespacePattern());
        for (String choice : spec.choices()) {
            chars += length(choice);
        }
        if (spec.triggers() != null) {
            for (GrammarTriggers.Trigger trigger : spec.triggers().triggers()) {
                chars += length(trigger.source());
            }
        }
        for (String exclude : compiled.freeExcludes()) {
            chars += length(exclude);
        }
        return BYTES_PER_ELEMENT * compiled.grammar().elementCount() + 2 * chars;
    }

    private static int length(String s) {
        return s == null ? 0 : s.length();
    }

    /** How many compiled constraints are kept now. */
    public int size() {
        synchronized (recent) {
            return recent.size();
        }
    }

    /** The estimated heap the kept constraints hold now ({@link #estimatedBytes}). */
    public long heldBytes() {
        synchronized (recent) {
            return heldBytes;
        }
    }
}
