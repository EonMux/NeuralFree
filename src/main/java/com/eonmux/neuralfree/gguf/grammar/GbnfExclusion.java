package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.BiFunction;

/**
 * GBNF rules matching any text that contains none of a set of strings: the complement of the strings'
 * Aho-Corasick automaton, one rule per non-matching state (llama.cpp {@code gbnf_excluding_grammar},
 * {@code common/peg-parser.cpp}). Every state accepts, and a code point that would complete a string has no
 * alternative, so a forbidden string can never be matched — the {@code until(delimiter)} body of a chat format
 * grammar.
 */
public final class GbnfExclusion {

    private GbnfExclusion() {
    }

    /**
     * Adds the exclusion rules of {@code strings} through {@code addRule} and returns the start rule's name. State 0
     * is named {@code prefix}, state {@code s} {@code prefix-ss} (two digits at least).
     *
     * @param addRule adds {@code name ::= definition} and returns the stored name, which must be the name asked for
     * @param prefix  the rule name prefix, not yet used by the grammar
     * @param strings the excluded strings, none of them empty
     * @throws IllegalArgumentException when {@code strings} is empty or holds an empty string
     * @throws IllegalStateException    when {@code addRule} stores a rule under another name (the prefix collides)
     */
    public static String excluding(BiFunction<String, String, String> addRule, String prefix, List<String> strings) {
        Automaton ac = new Automaton(strings);
        for (int q = 0; q < ac.size(); q++) {
            if (ac.terminal(q)) {
                continue;
            }
            claim(addRule, stateName(prefix, q), definition(ac, q, prefix));
        }
        return prefix;
    }

    private static String definition(Automaton ac, int q, String prefix) {
        Map<Integer, List<Integer>> buckets = new TreeMap<>();
        List<Integer> specific = new ArrayList<>();
        for (int c : ac.alphabet()) {
            int d = ac.next(q, c);
            if (ac.terminal(d)) {
                specific.add(c);
            } else if (d != 0) {
                buckets.computeIfAbsent(d, k -> new ArrayList<>()).add(c);
                specific.add(c);
            }
        }
        StringBuilder rhs = new StringBuilder("|");
        for (Map.Entry<Integer, List<Integer>> e : buckets.entrySet()) {
            rhs.append(' ').append(charClass(e.getValue(), false)).append(' ')
                    .append(stateName(prefix, e.getKey())).append(" |");
        }
        rhs.append(' ').append(charClass(specific, true)).append(' ').append(stateName(prefix, 0));
        return rhs.toString();
    }

    private static void claim(BiFunction<String, String, String> addRule, String name, String definition) {
        String stored = addRule.apply(name, definition);
        if (!name.equals(stored)) {
            throw new IllegalStateException("exclusion rule '" + name + "' was stored as '" + stored
                    + "': the prefix collides with an existing rule");
        }
    }

    private static String stateName(String prefix, int state) {
        if (state == 0) {
            return prefix;
        }
        return prefix + "-" + (state < 10 ? "0" + state : String.valueOf(state));
    }

    /** A GBNF character class of {@code chars}, negated on request. */
    static String charClass(List<Integer> chars, boolean negate) {
        StringBuilder out = new StringBuilder(negate ? "[^" : "[");
        for (int c : chars) {
            appendClassChar(out, c);
        }
        return out.append(']').toString();
    }

    private static void appendClassChar(StringBuilder out, int c) {
        if (c == '\\' || c == ']' || c == '[' || c == '-' || c == '"') {
            out.append('\\').appendCodePoint(c);
        } else if (c == '^') {
            out.append("\\x5E"); // a leading ^ negates the class, and GBNF has no \^ escape
        } else if (c == '\n') {
            out.append("\\n");
        } else if (c == '\r') {
            out.append("\\r");
        } else if (c == '\t') {
            out.append("\\t");
        } else if (c >= 0x20 && c < 0x7f) {
            out.append((char) c);
        } else if (c <= 0xff) {
            out.append(String.format("\\x%02X", c));
        } else if (c <= 0xffff) {
            out.append(String.format("\\u%04X", c));
        } else {
            out.append(String.format("\\U%08X", c));
        }
    }

    /** The Aho-Corasick automaton over the strings' code points: trie edges completed by failure links. */
    private static final class Automaton {

        private final List<Map<Integer, Integer>> edges = new ArrayList<>();
        private final List<Integer> fail = new ArrayList<>();
        private final List<Boolean> terminal = new ArrayList<>();
        private final TreeSet<Integer> alphabet = new TreeSet<>();
        private final Map<Long, Integer> transitions = new HashMap<>();

        Automaton(List<String> strings) {
            if (strings == null || strings.isEmpty()) {
                throw new IllegalArgumentException("an exclusion needs at least one string");
            }
            addState();
            for (String s : strings) {
                if (s == null || s.isEmpty()) {
                    throw new IllegalArgumentException("an excluded string must not be empty");
                }
                insert(s);
            }
            link();
        }

        int size() {
            return edges.size();
        }

        boolean terminal(int q) {
            return terminal.get(q);
        }

        TreeSet<Integer> alphabet() {
            return alphabet;
        }

        /** The DFA transition: the trie edge, else the failure state's transition, else the root. */
        int next(int q, int c) {
            long key = ((long) q << 32) | (c & 0xffffffffL);
            Integer cached = transitions.get(key);
            if (cached != null) {
                return cached;
            }
            int state = q;
            Integer d = edges.get(state).get(c);
            while (d == null && state != 0) {
                state = fail.get(state);
                d = edges.get(state).get(c);
            }
            int result = d == null ? 0 : d;
            transitions.put(key, result);
            return result;
        }

        private int addState() {
            edges.add(new HashMap<>());
            fail.add(0);
            terminal.add(Boolean.FALSE);
            return edges.size() - 1;
        }

        private void insert(String s) {
            int q = 0;
            for (int i = 0; i < s.length(); ) {
                int c = s.codePointAt(i);
                i += Character.charCount(c);
                alphabet.add(c);
                Integer d = edges.get(q).get(c);
                if (d == null) {
                    d = addState();
                    edges.get(q).put(c, d);
                }
                q = d;
            }
            terminal.set(q, Boolean.TRUE);
        }

        /** Breadth-first failure links; a state whose failure state matches matches too. */
        private void link() {
            List<Integer> queue = new ArrayList<>(edges.get(0).values());
            for (int head = 0; head < queue.size(); head++) {
                int q = queue.get(head);
                for (Map.Entry<Integer, Integer> e : edges.get(q).entrySet()) {
                    int child = e.getValue();
                    int f = fail.get(q);
                    Integer d = edges.get(f).get(e.getKey());
                    while (d == null && f != 0) {
                        f = fail.get(f);
                        d = edges.get(f).get(e.getKey());
                    }
                    int target = d == null || d == child ? 0 : d;
                    fail.set(child, target);
                    if (terminal.get(target)) {
                        terminal.set(child, Boolean.TRUE);
                    }
                    queue.add(child);
                }
            }
        }
    }
}
