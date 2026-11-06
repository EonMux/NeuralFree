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
 * grammar. {@link #excludingJsonStringBody} builds the same automaton into the JSON string body a free JSON
 * string writes, and {@link #dispatching} into structural-tag rules with the dispatch that follows the free
 * text, at the earliest occurrence of its opening string.
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

    /**
     * Adds the rules of a JSON string body — the text between the quotes, unescaped characters and escapes as the
     * {@code char*} of a JSON string — whose raw text contains none of {@code strings}, and returns the start rule's
     * name. The rules are the product of the strings' Aho-Corasick automaton with the escape states of a JSON string
     * (plain, after a backslash, inside the four hex digits of a {@code \\u} escape): an escape spells its raw
     * characters too, so {@code \\"x} holds the text {@code "x}. Every plain state accepts.
     *
     * @param addRule adds {@code name ::= definition} and returns the stored name, which must be the name asked for
     * @param prefix  the rule name prefix, not yet used by the grammar
     * @param strings the excluded strings, none of them empty
     * @throws IllegalArgumentException when {@code strings} is empty or holds an empty string
     * @throws IllegalStateException    when {@code addRule} stores a rule under another name (the prefix collides)
     */
    public static String excludingJsonStringBody(BiFunction<String, String, String> addRule, String prefix,
                                                 List<String> strings) {
        Automaton ac = new Automaton(strings);
        List<int[]> queue = new ArrayList<>();
        Map<Long, String> names = new HashMap<>();
        String start = jsonStateName(prefix, 0, PLAIN, names, queue);
        for (int head = 0; head < queue.size(); head++) {
            int q = queue.get(head)[0];
            int mode = queue.get(head)[1];
            claim(addRule, jsonStateName(prefix, q, mode, names, queue), jsonDefinition(ac, q, mode, prefix, names,
                    queue));
        }
        return start;
    }

    /** JSON string body states: plain text, after a backslash, then the four hex digits of a unicode escape. */
    private static final int PLAIN = 0;
    private static final int ESCAPE = 1;
    private static final int HEX_FIRST = 2;
    private static final int HEX_LAST = 5;
    private static final String SIMPLE_ESCAPES = "\"\\/bfnrt";
    private static final String HEX_DIGITS = "0123456789ABCDEFabcdef";

    private static String jsonStateName(String prefix, int q, int mode, Map<Long, String> names, List<int[]> queue) {
        long key = ((long) q << 3) | mode;
        String name = names.get(key);
        if (name == null) {
            name = q == 0 && mode == PLAIN ? prefix : prefix + "-" + q + "-" + mode;
            names.put(key, name);
            queue.add(new int[]{q, mode});
        }
        return name;
    }

    private static String jsonDefinition(Automaton ac, int q, int mode, String prefix, Map<Long, String> names,
                                         List<int[]> queue) {
        Map<String, List<Integer>> buckets = new TreeMap<>();
        StringBuilder rhs = new StringBuilder();
        if (mode == PLAIN) {
            List<Integer> specific = new ArrayList<>();
            for (int c : ac.alphabet()) {
                if (c == '"' || c == '\\' || c < 0x20 || c == 0x7F) {
                    continue;
                }
                specific.add(c);
                jsonBucket(ac, q, c, PLAIN, prefix, names, queue, buckets);
            }
            int afterBackslash = ac.next(q, '\\');
            rhs.append('|');
            if (!ac.terminal(afterBackslash)) {
                rhs.append(" \"\\\\\" ").append(jsonStateName(prefix, afterBackslash, ESCAPE, names, queue))
                        .append(" |");
            }
            StringBuilder other = new StringBuilder("[^\\\"\\\\\\x7F\\x00-\\x1F");
            for (int c : specific) {
                appendClassChar(other, c);
            }
            rhs.append(' ').append(other).append("] ").append(jsonStateName(prefix, 0, PLAIN, names, queue));
        } else {
            String allowed = mode == ESCAPE ? SIMPLE_ESCAPES + "u" : HEX_DIGITS;
            for (int k = 0; k < allowed.length(); k++) {
                int c = allowed.charAt(k);
                int nextMode = mode == ESCAPE ? (c == 'u' ? HEX_FIRST : PLAIN) : mode == HEX_LAST ? PLAIN : mode + 1;
                jsonBucket(ac, q, c, nextMode, prefix, names, queue, buckets);
            }
        }
        for (Map.Entry<String, List<Integer>> e : buckets.entrySet()) {
            if (rhs.length() > 0) {
                rhs.append(" | ");
            }
            rhs.append(charClass(e.getValue(), false)).append(' ').append(e.getKey());
        }
        if (rhs.length() == 0) {
            throw new IllegalArgumentException("the excluded strings leave a JSON escape no way to continue");
        }
        return rhs.toString();
    }

    /** Files {@code c} under the JSON string state it leads to, unless it completes an excluded string. */
    private static void jsonBucket(Automaton ac, int q, int c, int nextMode, String prefix, Map<Long, String> names,
                                   List<int[]> queue, Map<String, List<Integer>> buckets) {
        int d = ac.next(q, c);
        if (ac.terminal(d)) {
            return;
        }
        buckets.computeIfAbsent(jsonStateName(prefix, d, nextMode, names, queue), k -> new ArrayList<>()).add(c);
    }

    /**
     * Adds to {@code rules} any text holding none of {@code strings}, followed by one of {@code continuations}, and
     * returns the start state's rule: continuation {@code k}, whose text opens with {@code openings.get(k)}, follows
     * a state only when writing its opening from there completes no string before the opening's last code point.
     * The text therefore dispatches at the earliest occurrence of an opening, as xgrammar's {@code TagDispatch} does
     * when an opening overlaps itself ({@code aba} in {@code ababa}) or another one. State {@code s} is named
     * {@code prefix-ss}, state 0 {@code prefix}.
     *
     * @param strings       the banned strings, every opening among them, none of them empty
     * @param openings      the opening string of each continuation
     * @param continuations the continuation rules, one per opening
     */
    static int dispatching(StructuralTagRules rules, String prefix, List<String> strings, List<String> openings,
            List<Integer> continuations) {
        Automaton ac = new Automaton(strings);
        int[] index = new int[ac.size()];
        for (int q = 0; q < ac.size(); q++) {
            index[q] = ac.terminal(q) ? -1 : rules.reserve(stateName(prefix, q));
        }
        for (int q = 0; q < ac.size(); q++) {
            if (ac.terminal(q)) {
                continue;
            }
            Edges edges = new Edges(ac, q);
            List<GrammarElement> body = new ArrayList<>();
            for (Map.Entry<Integer, List<Integer>> e : edges.buckets.entrySet()) {
                body.addAll(classElements(e.getValue(), false));
                body.add(new GrammarElement(GrammarElement.Type.RULE_REF, index[e.getKey()]));
                body.add(new GrammarElement(GrammarElement.Type.ALT, 0));
            }
            body.addAll(classElements(edges.specific, true));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, index[0]));
            for (int k = 0; k < openings.size(); k++) {
                if (opensFirst(ac, q, openings.get(k))) {
                    body.add(new GrammarElement(GrammarElement.Type.ALT, 0));
                    body.add(new GrammarElement(GrammarElement.Type.RULE_REF, continuations.get(k)));
                }
            }
            rules.fill(index[q], body);
        }
        return index[0];
    }

    /** Whether writing {@code opening} from state {@code q} completes no string before its last code point. */
    private static boolean opensFirst(Automaton ac, int q, String opening) {
        int state = q;
        int last = opening.offsetByCodePoints(opening.length(), -1);
        for (int i = 0; i < last; ) {
            int c = opening.codePointAt(i);
            i += Character.charCount(c);
            state = ac.next(state, c);
            if (ac.terminal(state)) {
                return false;
            }
        }
        return true;
    }

    /** A character set as grammar elements: {@code CHAR} (or {@code CHAR_NOT}) then one {@code CHAR_ALT} each. */
    private static List<GrammarElement> classElements(List<Integer> chars, boolean negate) {
        List<GrammarElement> out = new ArrayList<>(chars.size());
        for (int i = 0; i < chars.size(); i++) {
            GrammarElement.Type type = i > 0 ? GrammarElement.Type.CHAR_ALT
                    : negate ? GrammarElement.Type.CHAR_NOT : GrammarElement.Type.CHAR;
            out.add(new GrammarElement(type, chars.get(i)));
        }
        return out;
    }

    private static String definition(Automaton ac, int q, String prefix) {
        Edges edges = new Edges(ac, q);
        StringBuilder rhs = new StringBuilder("|");
        for (Map.Entry<Integer, List<Integer>> e : edges.buckets.entrySet()) {
            rhs.append(' ').append(charClass(e.getValue(), false)).append(' ')
                    .append(stateName(prefix, e.getKey())).append(" |");
        }
        rhs.append(' ').append(charClass(edges.specific, true)).append(' ').append(stateName(prefix, 0));
        return rhs.toString();
    }

    /**
     * The transitions out of one non-matching state: the code points leading to each other non-matching state but
     * the root, and every code point with a transition of its own (the rest return to the root).
     */
    private static final class Edges {
        final Map<Integer, List<Integer>> buckets = new TreeMap<>();
        final List<Integer> specific = new ArrayList<>();

        Edges(Automaton ac, int q) {
            for (int c : ac.alphabet()) {
                int d = ac.next(q, c);
                if (ac.terminal(d)) {
                    specific.add(c);
                } else if (d != 0) {
                    buckets.computeIfAbsent(d, k -> new ArrayList<>()).add(c);
                    specific.add(c);
                }
            }
        }
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
    static final class Automaton {

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
