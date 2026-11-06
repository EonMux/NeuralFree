package com.eonmux.neuralfree.gguf.chat.response;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * The literal strings a delimiter regex spells (vLLM {@code response_template_grammar._literals}): alternations,
 * non-capturing groups, escaped characters ({@code \n}, {@code \r}, {@code \t} and escaped punctuation) and an
 * optional {@code ?} after an atom. Two forms read only on request: a {@code ^}, which matches only at the message
 * start ({@code messageStart}: the literals that start there, without it; a literal with {@code ^} anywhere else is
 * dropped), and a final lookahead {@code (?=...)} ({@code lookahead}: its text follows {@link #AHEAD} in the literal).
 * Any other construct, a {@code ^} or lookahead not asked for, or more than {@value #MAX_LITERALS} strings, means the
 * pattern matches more than literals.
 */
final class LiteralPatterns {

    /** vLLM's bound on the strings expanded from one delimiter pattern. */
    static final int MAX_LITERALS = 16;
    /** Stands for {@code ^} while expanding. */
    static final String START = "\0";
    /** Precedes the text of a final lookahead in an expanded literal. */
    static final String AHEAD = "\1";
    private static final String META = ".^$*+?{}[]\\|()";

    private LiteralPatterns() {
    }

    /** The literals {@code pattern} spells without {@code ^} or a lookahead, or null when it spells anything else. */
    static List<String> expand(String pattern) {
        return expand(pattern, false, false);
    }

    /**
     * The literals {@code pattern} spells, or null when it spells anything else.
     *
     * @param messageStart whether a {@code ^} may start a literal, which then matches only at the message start
     * @param lookahead    whether a literal may end in a lookahead, kept after {@link #AHEAD}
     */
    static List<String> expand(String pattern, boolean messageStart, boolean lookahead) {
        int[] at = {0};
        List<String> literals = alternatives(pattern, at);
        if (literals == null || at[0] != pattern.length()) return null;
        for (String l : literals) {
            if (count(l, AHEAD) > (lookahead ? 1 : 0)) return null;
        }
        if (!messageStart) {
            for (String l : literals) {
                if (l.contains(START)) return null;
            }
            return Collections.unmodifiableList(literals);
        }
        List<String> out = new ArrayList<>();
        for (String l : literals) {
            if (l.equals(START) || l.indexOf(START, 1) >= 0) continue;
            out.add(l.startsWith(START) ? l.substring(1) : l);
        }
        return Collections.unmodifiableList(out);
    }

    private static int count(String s, String part) {
        int n = 0;
        for (int i = s.indexOf(part); i >= 0; i = s.indexOf(part, i + 1)) n++;
        return n;
    }

    private static List<String> alternatives(String p, int[] at) {
        LinkedHashSet<String> literals = new LinkedHashSet<>();
        while (true) {
            List<String> branch = concatenation(p, at);
            if (branch == null) return null;
            literals.addAll(branch);
            if (!p.startsWith("|", at[0])) break;
            at[0]++;
        }
        return literals.size() <= MAX_LITERALS ? new ArrayList<>(literals) : null;
    }

    private static List<String> concatenation(String p, int[] at) {
        List<String> literals = new ArrayList<>();
        literals.add("");
        while (at[0] < p.length() && p.charAt(at[0]) != '|' && p.charAt(at[0]) != ')') {
            List<String> atom = atom(p, at);
            if (atom == null) return null;
            if (p.startsWith("?", at[0])) {
                atom.add(0, "");
                at[0]++;
            }
            for (String prefix : literals) {
                // a lookahead reads as a literal only at the end of the pattern
                if (prefix.contains(AHEAD)) return null;
            }
            List<String> next = new ArrayList<>();
            for (String prefix : literals) {
                for (String suffix : atom) next.add(prefix + suffix);
            }
            if (next.size() > MAX_LITERALS) return null;
            literals = next;
        }
        return literals;
    }

    private static List<String> atom(String p, int[] at) {
        int i = at[0];
        char c = p.charAt(i);
        if (p.startsWith("(?:", i) || p.startsWith("(?=", i)) {
            boolean ahead = p.charAt(i + 2) == '=';
            at[0] = i + 3;
            List<String> inner = alternatives(p, at);
            if (inner == null || !p.startsWith(")", at[0])) return null;
            at[0]++;
            if (!ahead) return inner;
            List<String> marked = new ArrayList<>();
            for (String l : inner) marked.add(AHEAD + l);
            return marked;
        }
        List<String> atom = new ArrayList<>();
        if (c == '\\' && i + 1 < p.length()) {
            char e = p.charAt(i + 1);
            String text = e == 'n' ? "\n" : e == 'r' ? "\r" : e == 't' ? "\t"
                    : Character.isLetterOrDigit(e) ? null : String.valueOf(e);
            if (text == null) return null;
            atom.add(text);
            at[0] = i + 2;
            return atom;
        }
        if (c == '^') {
            atom.add(START);
            at[0] = i + 1;
            return atom;
        }
        if (META.indexOf(c) >= 0) return null;
        atom.add(String.valueOf(c));
        at[0] = i + 1;
        return atom;
    }
}
