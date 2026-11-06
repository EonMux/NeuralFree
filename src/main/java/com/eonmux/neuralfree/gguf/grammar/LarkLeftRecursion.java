package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The left-recursion rewrite of a Lark grammar's rules. Lark parses with Earley or LALR, so its grammars are
 * routinely left-recursive ({@code sum: sum "+" product | product}); the GBNF runtime is a pushdown matcher that
 * refuses left recursion ({@link Grammar}). The rewrite keeps the language and removes the recursion: the rules
 * whose leftmost references form a cycle are rewritten in source order by Paull's algorithm — a leading reference
 * to an earlier rule of the cycle is replaced by that rule's alternatives, then each rule's direct recursion
 * {@code A: A a1 | ... | b1 | ...} becomes {@code A: b1 T | ...} with the tail rule {@code T: (a1 | ...)*}. A
 * leading group is opened into its alternatives first; an alternative that is just {@code A} is dropped. Rules
 * outside a cycle are returned untouched. A rule whose every alternative starts with itself (directly, or through
 * the other rules of its cycle) is accepted and encodes as a rule that matches nothing, as xgrammar does.
 * Recursion behind a nullable leading item is not rewritten; the {@link Grammar} check refuses it.
 */
final class LarkLeftRecursion {

    /** The most alternatives a rewritten rule may hold. */
    static final int MAX_ALTERNATIVES = 1024;

    /** The name prefix of the tail rules; no Lark name starts with a digit. */
    static final String TAIL_PREFIX = "0lr_";

    private final Map<String, LarkExpr> rules;
    private final Map<String, LarkParser.Definition> definitions;

    private LarkLeftRecursion(Map<String, LarkExpr> rules, Map<String, LarkParser.Definition> definitions) {
        this.rules = rules;
        this.definitions = definitions;
    }

    /**
     * The rules with their left recursion rewritten: the same names in the same order, each tail rule right after
     * its rule.
     *
     * @param rules       the rule bodies by Lark name, in source order
     * @param definitions the definitions the rules come from, for the positions of refusals
     * @throws GrammarException when a recursive rule has no alternative free of the recursion, or the rewrite
     *                          exceeds {@link #MAX_ALTERNATIVES}
     */
    static Map<String, LarkExpr> rewrite(Map<String, LarkExpr> rules,
            Map<String, LarkParser.Definition> definitions) {
        return new LarkLeftRecursion(rules, definitions).rewrite();
    }

    private Map<String, LarkExpr> rewrite() {
        Map<String, List<List<LarkExpr>>> rewritten = new HashMap<>();
        Map<String, LarkExpr> tails = new HashMap<>();
        for (List<String> cycle : cycles()) {
            for (int i = 0; i < cycle.size(); i++) {
                String name = cycle.get(i);
                List<List<LarkExpr>> alternatives = substitute(name, cycle.subList(0, i), rewritten);
                rewritten.put(name, removeDirect(name, alternatives, tails));
            }
        }
        Map<String, LarkExpr> out = new LinkedHashMap<>();
        for (Map.Entry<String, LarkExpr> rule : rules.entrySet()) {
            String name = rule.getKey();
            List<List<LarkExpr>> alternatives = rewritten.get(name);
            out.put(name, alternatives == null ? rule.getValue() : body(alternatives, rule.getValue()));
            if (tails.containsKey(name)) {
                out.put(TAIL_PREFIX + name, tails.get(name));
            }
        }
        return out;
    }

    /** The alternatives of {@code name} with every leading reference to an {@code earlier} rule substituted. */
    private List<List<LarkExpr>> substitute(String name, List<String> earlier,
            Map<String, List<List<LarkExpr>>> rewritten) {
        List<List<LarkExpr>> alternatives = alternatives(rules.get(name), name);
        boolean changed = true;
        while (changed) {
            changed = false;
            List<List<LarkExpr>> next = new ArrayList<>();
            for (List<LarkExpr> alternative : alternatives) {
                String lead = leadingRule(alternative);
                if (lead != null && earlier.contains(lead)) {
                    changed = true;
                    for (List<LarkExpr> replacement : rewritten.get(lead)) {
                        List<LarkExpr> joined = new ArrayList<>(replacement);
                        joined.addAll(alternative.subList(1, alternative.size()));
                        open(joined, next, name);
                    }
                } else {
                    next.add(alternative);
                }
            }
            alternatives = next;
        }
        return alternatives;
    }

    /** {@code A: A a1 | b1} as {@code A: b1 T} and {@code T: (a1)*}; a rule without direct recursion is kept. */
    private List<List<LarkExpr>> removeDirect(String name, List<List<LarkExpr>> alternatives,
            Map<String, LarkExpr> tails) {
        List<LarkExpr> recursive = new ArrayList<>();
        List<List<LarkExpr>> base = new ArrayList<>();
        LarkParser.Definition at = definitions.get(name);
        for (List<LarkExpr> alternative : alternatives) {
            if (!name.equals(leadingRule(alternative))) {
                base.add(alternative);
            } else if (alternative.size() > 1) {
                recursive.add(LarkExpr.seq(alternative.subList(1, alternative.size()), at.line(), at.column()));
            }
        }
        if (recursive.isEmpty()) {
            return base;
        }
        if (base.isEmpty()) {
            // xgrammar accepts a rule whose every alternative starts with itself: it derives nothing. The
            // rewrite encodes that as a NUL literal before the rule itself — no token carries a NUL byte, so
            // no derivation of the rule can ever complete.
            List<LarkExpr> dead = new ArrayList<>(2);
            dead.add(LarkExpr.literal(String.valueOf('\0'), false, at.line(), at.column()));
            dead.add(LarkExpr.ref(name, at.line(), at.column()));
            List<List<LarkExpr>> out = new ArrayList<>(1);
            out.add(dead);
            return out;
        }
        LarkExpr tailBody = recursive.size() == 1 ? recursive.get(0)
                : LarkExpr.alt(recursive, at.line(), at.column());
        tails.put(name, LarkExpr.repeat(tailBody, 0, LarkExpr.UNBOUNDED, at.line(), at.column()));
        List<List<LarkExpr>> out = new ArrayList<>(base.size());
        for (List<LarkExpr> alternative : base) {
            List<LarkExpr> withTail = new ArrayList<>(alternative);
            withTail.add(LarkExpr.ref(TAIL_PREFIX + name, at.line(), at.column()));
            out.add(withTail);
        }
        return out;
    }

    // ------------------------------------------------------------------ the left-corner graph

    /** The left-recursive cycles: strongly connected components of the leftmost-reference graph, in source order. */
    private List<List<String>> cycles() {
        Map<String, Set<String>> edges = new LinkedHashMap<>();
        for (Map.Entry<String, LarkExpr> rule : rules.entrySet()) {
            Set<String> leads = new LinkedHashSet<>();
            leadingRules(rule.getValue(), leads);
            edges.put(rule.getKey(), leads);
        }
        List<List<String>> out = new ArrayList<>();
        for (Set<String> component : new Tarjan(edges).components()) {
            String only = component.iterator().next();
            if (component.size() > 1 || edges.get(only).contains(only)) {
                List<String> ordered = new ArrayList<>();
                for (String name : rules.keySet()) {
                    if (component.contains(name)) {
                        ordered.add(name);
                    }
                }
                out.add(ordered);
            }
        }
        return out;
    }

    /** The rules {@code e} may start with: the leftmost reference of each of its alternatives. */
    private void leadingRules(LarkExpr e, Set<String> out) {
        switch (e.kind()) {
            case REF:
                if (rules.containsKey(e.text())) {
                    out.add(e.text());
                }
                return;
            case SEQ:
                if (!e.children().isEmpty()) {
                    leadingRules(e.children().get(0), out);
                }
                return;
            case ALT:
                for (LarkExpr child : e.children()) {
                    leadingRules(child, out);
                }
                return;
            default:
                return;
        }
    }

    /**
     * Tarjan's strongly connected components over the rule graph, iterative so a long chain of rules cannot
     * exhaust the thread stack.
     */
    private static final class Tarjan {
        private final Map<String, Set<String>> edges;
        private final Map<String, Integer> index = new HashMap<>();
        private final Map<String, Integer> low = new HashMap<>();
        private final List<String> stack = new ArrayList<>();
        private final Set<String> onStack = new LinkedHashSet<>();
        private final List<Set<String>> components = new ArrayList<>();

        Tarjan(Map<String, Set<String>> edges) {
            this.edges = edges;
        }

        List<Set<String>> components() {
            for (String node : edges.keySet()) {
                if (!index.containsKey(node)) {
                    visit(node);
                }
            }
            return components;
        }

        /** The depth-first visit from {@code root}: an explicit stack of nodes and their edge iterators. */
        private void visit(String root) {
            List<String> path = new ArrayList<>();
            List<Iterator<String>> pending = new ArrayList<>();
            enter(root, path, pending);
            while (!path.isEmpty()) {
                String node = path.get(path.size() - 1);
                Iterator<String> next = pending.get(pending.size() - 1);
                if (next.hasNext()) {
                    String target = next.next();
                    if (!index.containsKey(target)) {
                        enter(target, path, pending);
                    } else if (onStack.contains(target)) {
                        low.put(node, Math.min(low.get(node), index.get(target)));
                    }
                    continue;
                }
                path.remove(path.size() - 1);
                pending.remove(pending.size() - 1);
                if (!path.isEmpty()) {
                    String parent = path.get(path.size() - 1);
                    low.put(parent, Math.min(low.get(parent), low.get(node)));
                }
                if (low.get(node).equals(index.get(node))) {
                    emit(node);
                }
            }
        }

        private void enter(String node, List<String> path, List<Iterator<String>> pending) {
            index.put(node, index.size());
            low.put(node, index.get(node));
            stack.add(node);
            onStack.add(node);
            path.add(node);
            pending.add(edges.get(node).iterator());
        }

        private void emit(String root) {
            Set<String> component = new LinkedHashSet<>();
            String member;
            do {
                member = stack.remove(stack.size() - 1);
                onStack.remove(member);
                component.add(member);
            } while (!member.equals(root));
            components.add(component);
        }
    }

    // ------------------------------------------------------------------ alternatives

    /** The rule body's alternatives as item sequences, every leading group opened. */
    private List<List<LarkExpr>> alternatives(LarkExpr body, String owner) {
        List<List<LarkExpr>> out = new ArrayList<>();
        for (LarkExpr alternative : body.kind() == LarkExpr.Kind.ALT ? body.children()
                : Collections.singletonList(body)) {
            open(items(alternative), out, owner);
        }
        return out;
    }

    /** Adds {@code sequence} to {@code out}, a leading group or sequence opened into its alternatives. */
    private void open(List<LarkExpr> sequence, List<List<LarkExpr>> out, String owner) {
        if (!sequence.isEmpty() && (sequence.get(0).kind() == LarkExpr.Kind.ALT
                || sequence.get(0).kind() == LarkExpr.Kind.SEQ)) {
            LarkExpr head = sequence.get(0);
            List<LarkExpr> rest = sequence.subList(1, sequence.size());
            List<LarkExpr> choices = head.kind() == LarkExpr.Kind.ALT ? head.children()
                    : Collections.singletonList(head);
            for (LarkExpr choice : choices) {
                List<LarkExpr> joined = new ArrayList<>(items(choice));
                joined.addAll(rest);
                open(joined, out, owner);
            }
            return;
        }
        if (out.size() >= MAX_ALTERNATIVES) {
            LarkParser.Definition at = definitions.get(owner);
            throw new GrammarException("invalid Lark grammar: rewriting the left recursion of rule '" + owner
                    + "' needs more than " + MAX_ALTERNATIVES + " alternatives", at.line(), at.column());
        }
        out.add(sequence);
    }

    private static List<LarkExpr> items(LarkExpr e) {
        return e.kind() == LarkExpr.Kind.SEQ ? e.children() : Collections.singletonList(e);
    }

    /** The rule an alternative starts with, or null when it starts with anything else or is empty. */
    private String leadingRule(List<LarkExpr> alternative) {
        if (alternative.isEmpty() || alternative.get(0).kind() != LarkExpr.Kind.REF) {
            return null;
        }
        String name = alternative.get(0).text();
        return rules.containsKey(name) ? name : null;
    }

    /** The alternatives as one body expression, positioned at {@code original}. */
    private static LarkExpr body(List<List<LarkExpr>> alternatives, LarkExpr original) {
        List<LarkExpr> sequences = new ArrayList<>(alternatives.size());
        for (List<LarkExpr> alternative : alternatives) {
            sequences.add(LarkExpr.seq(alternative, original.line(), original.column()));
        }
        return sequences.size() == 1 ? sequences.get(0)
                : LarkExpr.alt(sequences, original.line(), original.column());
    }
}
