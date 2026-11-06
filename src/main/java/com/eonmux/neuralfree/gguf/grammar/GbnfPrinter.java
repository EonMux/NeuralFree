package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.List;

/**
 * Writes a compiled {@link Grammar} back as GBNF text that {@link GbnfParser} parses to the same rules (llama.cpp's
 * {@code llama_grammar_print} in valid syntax). Every rule is renamed {@code prefix + name + "-" + id}: the id keeps
 * names unique, the prefix lets the text join another grammar without clashes, and the parser's synthesized
 * {@code name_id} rules become valid names. Characters print as bracket sets and tokens by id ({@code <[42]>}), so the
 * text needs no vocabulary.
 */
public final class GbnfPrinter {

    private GbnfPrinter() {
    }

    /** The grammar as GBNF rules, each named {@link #ruleName}; no rule is called {@code root} unless prefixed so. */
    public static String print(Grammar grammar, String prefix) {
        StringBuilder out = new StringBuilder();
        for (int id = 0; id < grammar.ruleCount(); id++) {
            out.append(ruleName(grammar, id, prefix)).append(" ::=");
            appendRule(out, grammar, grammar.rule(id), prefix);
            out.append('\n');
        }
        return out.toString();
    }

    /** The printed name of rule {@code id}. */
    public static String ruleName(Grammar grammar, int id, String prefix) {
        return prefix + grammar.symbolName(id).replaceAll("[^a-zA-Z0-9-]", "-") + "-" + id;
    }

    /**
     * One rule's alternatives. An empty alternative prints as {@code ""}: GBNF lets a newline follow {@code |}, so a
     * bare trailing {@code |} would join the next rule to this one.
     */
    private static void appendRule(StringBuilder out, Grammar grammar, List<GrammarElement> rule, String prefix) {
        int i = 0;
        boolean empty = true;
        while (i < rule.size()) {
            GrammarElement e = rule.get(i);
            if (Grammar.isEndOfSequence(e)) {
                if (empty) {
                    out.append(" \"\"");
                }
                if (e.type() == GrammarElement.Type.END) {
                    return;
                }
                out.append(" |");
                empty = true;
                i++;
                continue;
            }
            out.append(' ');
            i = appendElement(out, grammar, rule, i, prefix);
            empty = false;
        }
    }

    /** The element at {@code i} (a whole character set for CHAR/CHAR_NOT); returns the index after it. */
    private static int appendElement(StringBuilder out, Grammar grammar, List<GrammarElement> rule, int i,
                                     String prefix) {
        GrammarElement e = rule.get(i);
        switch (e.type()) {
            case RULE_REF:
                out.append(ruleName(grammar, e.value(), prefix));
                return i + 1;
            case CHAR_ANY:
                out.append('.');
                return i + 1;
            case TOKEN:
                out.append("<[").append(Integer.toUnsignedString(e.value())).append("]>");
                return i + 1;
            case TOKEN_NOT:
                out.append("!<[").append(Integer.toUnsignedString(e.value())).append("]>");
                return i + 1;
            case CHAR:
            case CHAR_NOT:
                return appendSet(out, rule, i);
            default:
                throw new IllegalStateException("grammar element " + e.type() + " outside a character set");
        }
    }

    /** One character set starting at {@code start}: the CHAR/CHAR_NOT and its CHAR_ALT and range elements. */
    private static int appendSet(StringBuilder out, List<GrammarElement> rule, int start) {
        GrammarElement head = rule.get(start);
        out.append(head.type() == GrammarElement.Type.CHAR_NOT ? "[^" : "[");
        appendChar(out, head.value());
        int i = start + 1;
        while (i < rule.size()) {
            GrammarElement e = rule.get(i);
            if (e.type() == GrammarElement.Type.CHAR_RNG_UPPER) {
                out.append('-');
                appendChar(out, e.value());
            } else if (e.type() == GrammarElement.Type.CHAR_ALT) {
                appendChar(out, e.value());
            } else {
                break;
            }
            i++;
        }
        out.append(']');
        return i;
    }

    private static void appendChar(StringBuilder out, int c) {
        List<Integer> one = new ArrayList<>(1);
        one.add(c);
        String set = GbnfExclusion.charClass(one, false);
        out.append(set, 1, set.length() - 1);
    }
}
