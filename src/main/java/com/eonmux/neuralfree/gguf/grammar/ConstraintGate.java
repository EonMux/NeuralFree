package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Where a constraint starts when the model writes free text first: the output runs unconstrained until one of the
 * {@link #triggers()} fires, and from there it must match the {@link #prefix()} rule, optional whitespace and then the
 * constraint's own grammar — vLLM's structured output that waits for the end of reasoning
 * ({@code StructuredOutputManager} with a reasoning parser) and llama.cpp's response-format grammar behind the
 * thinking section or the harmony final header. The grammar reads the output from where the trigger fires
 * ({@link GrammarTriggers.Trigger#find}), so the prefix must match the triggering text. Immutable.
 */
public final class ConstraintGate {

    private final GrammarTriggers triggers;
    private final String prefix;

    /**
     * @param triggers the lazy triggers that open the gate
     * @param prefix   the GBNF right-hand side the output matches from the trigger on
     */
    public ConstraintGate(GrammarTriggers triggers, String prefix) {
        Objects.requireNonNull(triggers, "triggers");
        if (triggers.mode() != GrammarTriggers.Mode.LAZY) {
            throw new IllegalArgumentException("a gate opens once: its triggers must be lazy");
        }
        if (prefix == null || prefix.trim().isEmpty()) {
            throw new IllegalArgumentException("a gate needs a prefix rule");
        }
        this.triggers = triggers;
        this.prefix = prefix;
    }

    /** A gate that opens at the first of {@code words}, its prefix exactly that word. */
    public static ConstraintGate afterWords(List<String> words) {
        List<String> literals = new ArrayList<>();
        for (String w : words) {
            literals.add(JsonSchemaToGbnf.formatLiteral(w));
        }
        return new ConstraintGate(GrammarTriggers.lazyWords(words), String.join(" | ", literals));
    }

    /** The lazy triggers that open the gate. */
    public GrammarTriggers triggers() {
        return triggers;
    }

    /** The GBNF right-hand side matched from the trigger on. */
    public String prefix() {
        return prefix;
    }

    /**
     * {@code grammar} behind this gate: {@code root ::= prefix [ \t\n\r]* grammar-root}, the grammar's rules renamed
     * under {@code gated-} ({@link GbnfPrinter}).
     */
    Grammar wrap(Grammar grammar) {
        String inner = GbnfPrinter.ruleName(grammar, grammar.rootRuleIndex(), "gated-");
        String text = "root ::= gate-prefix gate-ws " + inner + "\n"
                + "gate-prefix ::= " + prefix + "\n"
                + "gate-ws ::= [ \\t\\n\\r]*\n"
                + GbnfPrinter.print(grammar, "gated-");
        return GbnfParser.parse(text);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ConstraintGate)) {
            return false;
        }
        ConstraintGate other = (ConstraintGate) o;
        return triggers.equals(other.triggers) && prefix.equals(other.prefix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(triggers, prefix);
    }

    @Override
    public String toString() {
        return "ConstraintGate[" + triggers + "]";
    }
}
