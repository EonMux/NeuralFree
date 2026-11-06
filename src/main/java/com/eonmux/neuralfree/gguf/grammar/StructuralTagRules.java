package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The rule accumulator behind {@link StructuralTag}: one rule list with its symbol table, into which the legacy
 * payload's structures and the modern {@code format} object both compile. Sub-grammars ({@link #merge}) rename
 * under a prefix and re-index their rule references; literals become code-point runs; every rule body ends with
 * {@link GrammarElement.Type#END}. {@link #finish} adds the root rule and returns the validated {@link Grammar}
 * the sampling runtime consumes. A self-recursive rule (a repetition) reserves its slot first and fills the body once
 * its own index is known. Package-private; confined to the thread building the grammar.
 */
final class StructuralTagRules {

    private final List<List<GrammarElement>> rules = new ArrayList<>();
    private final Map<String, Integer> symbolIds = new TreeMap<>();
    private final List<String> symbolNames = new ArrayList<>();
    private int elements;

    /** A rollback mark over the rules added so far. */
    int[] mark() {
        return new int[]{rules.size(), symbolNames.size()};
    }

    /** Discards every rule added after {@code mark}, the speculative-compile rollback of format inference. */
    void undo(int[] mark) {
        while (rules.size() > mark[0]) {
            elements -= rules.remove(rules.size() - 1).size();
        }
        while (symbolNames.size() > mark[1]) {
            symbolIds.remove(symbolNames.remove(symbolNames.size() - 1));
        }
    }

    /**
     * Appends {@code schema}'s rules under the name {@code prefix}: every rule renames to {@code prefix + name}
     * and its rule references re-index by the appended offset. Returns the schema's root rule's new index.
     */
    int merge(Grammar schema, String prefix) {
        int offset = rules.size();
        for (int r = 0; r < schema.ruleCount(); r++) {
            List<GrammarElement> renamed = new ArrayList<>(schema.rule(r).size());
            for (GrammarElement element : schema.rule(r)) {
                renamed.add(element.type() == GrammarElement.Type.RULE_REF
                        ? new GrammarElement(GrammarElement.Type.RULE_REF, element.value() + offset)
                        : element);
            }
            store(renamed);
            define(prefix + schema.symbolName(r), offset + r);
        }
        return offset + schema.rootRuleIndex();
    }

    /** A rule matching the literal's code points; an empty literal matches the empty string. */
    int literal(String name, String text) {
        List<GrammarElement> body = new ArrayList<>(text.codePointCount(0, text.length()));
        text.codePoints().forEach(cp -> body.add(new GrammarElement(GrammarElement.Type.CHAR, cp)));
        return rule(name, body);
    }

    /** A rule matching exactly the token id ({@code <[id]>}). */
    int token(String name, int id) {
        return rule(name, Collections.singletonList(new GrammarElement(GrammarElement.Type.TOKEN, id)));
    }

    /**
     * A rule matching one token outside {@code ids}: the token set {@code !<[a],[b],...>}, or any token
     * ({@code !<>}) when {@code ids} is empty.
     */
    int anyTokenBut(String name, int[] ids) {
        if (ids.length == 0) {
            return rule(name, Collections.singletonList(new GrammarElement(GrammarElement.Type.TOKEN_ANY, 0)));
        }
        List<GrammarElement> body = new ArrayList<>(ids.length);
        for (int i = 0; i < ids.length; i++) {
            body.add(new GrammarElement(i == 0 ? GrammarElement.Type.TOKEN_NOT : GrammarElement.Type.TOKEN_ALT,
                    ids[i]));
        }
        return rule(name, body);
    }

    /**
     * A rule matching {@code inner} within budgets, xgrammar's {@code name[max_tokens=n, max_chars=m]}: at most
     * {@code maxTokens} tokens and {@code maxChars} code points; a negative budget is no bound. Returns
     * {@code inner} itself when neither bound is set.
     */
    int budgeted(String name, int maxTokens, int maxChars, int inner) {
        if (maxTokens < 0 && maxChars < 0) {
            return inner;
        }
        List<GrammarElement> body = new ArrayList<>(3);
        if (maxTokens >= 0) {
            body.add(new GrammarElement(GrammarElement.Type.TOKEN_BUDGET, maxTokens));
        }
        if (maxChars >= 0) {
            body.add(new GrammarElement(GrammarElement.Type.CHAR_BUDGET, maxChars));
        }
        body.add(new GrammarElement(GrammarElement.Type.RULE_REF, inner));
        return rule(name, body);
    }

    /** The literal's code points as CHAR elements, the body pieces of a literal inside a larger rule. */
    static List<GrammarElement> codePoints(String text) {
        List<GrammarElement> body = new ArrayList<>(text.codePointCount(0, text.length()));
        text.codePoints().forEach(cp -> body.add(new GrammarElement(GrammarElement.Type.CHAR, cp)));
        return body;
    }

    /** The rule refs in order, the body pieces of a sequence inside a larger rule. */
    static List<GrammarElement> refSequence(int... refs) {
        List<GrammarElement> body = new ArrayList<>(refs.length);
        for (int ref : refs) {
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, ref));
        }
        return body;
    }

    /** A rule whose body is the refs in order (an empty list matches the empty string). */
    int rule(String name, List<GrammarElement> body) {
        List<GrammarElement> full = new ArrayList<>(body.size() + 1);
        full.addAll(body);
        add(full);
        define(name, rules.size() - 1);
        return rules.size() - 1;
    }

    /** Reserves a rule slot under {@code name}, its body still unset; {@link #fill} completes it. */
    int reserve(String name) {
        add(new ArrayList<>());
        define(name, rules.size() - 1);
        return rules.size() - 1;
    }

    /** Sets the body of the rule reserved under {@code index}; a self-reference to {@code index} is a repetition. */
    void fill(int index, List<GrammarElement> body) {
        List<GrammarElement> full = new ArrayList<>(body.size() + 1);
        full.addAll(body);
        full.add(new GrammarElement(GrammarElement.Type.END, 0));
        elements += body.size(); // the reserved placeholder already counted its END
        if (elements > GrammarLimits.MAX_ELEMENTS) {
            throw new IllegalArgumentException("structural tag builds more than " + GrammarLimits.MAX_ELEMENTS
                    + " grammar elements");
        }
        rules.set(index, full);
    }

    /** A rule holding one alternate per ref: {@code a | b | ...}. */
    int alternation(String name, List<Integer> refs) {
        List<GrammarElement> body = new ArrayList<>(refs.size() * 2);
        for (int i = 0; i < refs.size(); i++) {
            if (i > 0) {
                body.add(new GrammarElement(GrammarElement.Type.ALT, 0));
            }
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, refs.get(i)));
        }
        return rule(name, body);
    }

    /** The finished grammar: the root rule over {@code rootBody} and every rule added so far. */
    Grammar finish(List<GrammarElement> rootBody) {
        List<GrammarElement> root = new ArrayList<>(rootBody.size() + 1);
        root.addAll(rootBody);
        add(root);
        define(Grammar.ROOT_RULE, rules.size() - 1);
        return new Grammar(rules, symbolIds, symbolNames, rules.size() - 1,
                new int[rules.size()], new int[rules.size()]);
    }

    private void add(List<GrammarElement> body) {
        body.add(new GrammarElement(GrammarElement.Type.END, 0));
        store(body);
    }

    private void store(List<GrammarElement> body) {
        elements += body.size();
        if (rules.size() >= GrammarLimits.MAX_RULES) {
            throw new IllegalArgumentException("structural tag builds more than " + GrammarLimits.MAX_RULES
                    + " rules");
        }
        if (elements > GrammarLimits.MAX_ELEMENTS) {
            throw new IllegalArgumentException("structural tag builds more than " + GrammarLimits.MAX_ELEMENTS
                    + " grammar elements");
        }
        rules.add(body);
    }

    private void define(String name, int index) {
        if (symbolIds.containsKey(name)) {
            throw new IllegalArgumentException("structural tag builds rule '" + name + "' twice");
        }
        symbolIds.put(name, index);
        symbolNames.add(name);
    }
}
