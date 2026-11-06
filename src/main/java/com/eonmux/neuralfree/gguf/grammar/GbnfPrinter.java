package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.List;

/**
 * Writes a compiled {@link Grammar} back as GBNF text that {@link GbnfParser} parses to the same rules (llama.cpp's
 * {@code llama_grammar_print} in valid syntax). Every rule is renamed {@code prefix + name + "-" + id}: the id keeps
 * names unique, the prefix lets the text join another grammar without clashes, and the parser's synthesized
 * {@code name_id} rules become valid names. Characters print as bracket sets and tokens by id ({@code <[42]>}, any
 * token as {@code !<>}), so the text needs no vocabulary. The budget elements a single-alternative rule opens with
 * print as xgrammar's rule attributes ({@code name[max_tokens=n, max_chars=m] ::= …}), the one place
 * {@link GbnfParser} reads them.
 */
public final class GbnfPrinter {

    private GbnfPrinter() {
    }

    /** The grammar as GBNF rules, each named {@link #ruleName}; no rule is called {@code root} unless prefixed so. */
    public static String print(Grammar grammar, String prefix) {
        StringBuilder out = new StringBuilder();
        for (int id = 0; id < grammar.ruleCount(); id++) {
            List<GrammarElement> rule = grammar.rule(id);
            int head = budgetHead(rule);
            out.append(ruleName(grammar, id, prefix));
            appendBudgets(out, rule, head);
            out.append(" ::=");
            appendRule(out, grammar, rule.subList(head, rule.size()), prefix);
            out.append('\n');
        }
        return out.toString();
    }

    /** The number of budget elements {@code rule} opens with; a budget anywhere else has no GBNF form. */
    private static int budgetHead(List<GrammarElement> rule) {
        int head = 0;
        while (GrammarElement.isBudget(rule.get(head).type())) {
            head++;
        }
        for (int i = head; i < rule.size(); i++) {
            GrammarElement.Type type = rule.get(i).type();
            if (GrammarElement.isBudget(type) || (head > 0 && type == GrammarElement.Type.ALT)) {
                throw new IllegalStateException("a budget element prints only at the head of a single-alternative"
                        + " rule");
            }
        }
        return head;
    }

    /** The rule attributes {@code [max_tokens=n, max_chars=m]} of the first {@code head} elements. */
    private static void appendBudgets(StringBuilder out, List<GrammarElement> rule, int head) {
        if (head == 0) {
            return;
        }
        out.append('[');
        for (int i = 0; i < head; i++) {
            GrammarElement e = rule.get(i);
            out.append(i > 0 ? ", " : "").append(e.type() == GrammarElement.Type.TOKEN_BUDGET
                    ? "max_tokens=" : "max_chars=").append(e.value());
        }
        out.append(']');
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
            case TOKEN_NOT:
                return appendTokenSet(out, rule, i);
            case TOKEN_ANY:
                out.append("!<>");
                return i + 1;
            case CHAR:
            case CHAR_NOT:
                return appendSet(out, rule, i);
            default:
                throw new IllegalStateException("grammar element " + e.type() + " outside a character set");
        }
    }

    /** One token element starting at {@code start}: {@code <[a]>}, {@code !<[a]>}, or a set with its TOKEN_ALTs. */
    private static int appendTokenSet(StringBuilder out, List<GrammarElement> rule, int start) {
        GrammarElement head = rule.get(start);
        out.append(head.type() == GrammarElement.Type.TOKEN_NOT ? "!<[" : "<[")
                .append(Integer.toUnsignedString(head.value())).append(']');
        int i = start + 1;
        while (i < rule.size() && rule.get(i).type() == GrammarElement.Type.TOKEN_ALT) {
            out.append(",[").append(Integer.toUnsignedString(rule.get(i).value())).append(']');
            i++;
        }
        out.append('>');
        return i;
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
