package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The case-insensitive form of a compiled grammar, the {@code i} flag of a Lark regular expression: every
 * character set widens to the other case of the ASCII letters it holds ({@link RegexCharSet#asciiCaseClosure}), so
 * {@code [a-c]} accepts {@code A-C} too; as in xgrammar, no non-ASCII code point gains a variant. A negated set
 * excludes both cases of what it lists ({@code [^a]} refuses {@code A}). Rule references, any-character elements
 * and tokens are kept.
 */
final class GrammarCaseFold {

    private GrammarCaseFold() {
    }

    /** {@code grammar} with every character set case folded; the rules, names and root are the same. */
    static Grammar fold(Grammar grammar) {
        List<List<GrammarElement>> rules = new ArrayList<>(grammar.ruleCount());
        Map<String, Integer> ids = new TreeMap<>();
        List<String> names = new ArrayList<>(grammar.ruleCount());
        int[] lines = new int[grammar.ruleCount()];
        int[] columns = new int[grammar.ruleCount()];
        for (int r = 0; r < grammar.ruleCount(); r++) {
            rules.add(foldRule(grammar.rule(r)));
            names.add(grammar.symbolName(r));
            ids.put(grammar.symbolName(r), r);
            lines[r] = grammar.ruleLine(r);
            columns[r] = grammar.ruleColumn(r);
        }
        return new Grammar(rules, ids, names, grammar.rootRuleIndex(), lines, columns);
    }

    private static List<GrammarElement> foldRule(List<GrammarElement> rule) {
        List<GrammarElement> out = new ArrayList<>(rule.size());
        int i = 0;
        while (i < rule.size()) {
            GrammarElement head = rule.get(i);
            if (head.type() != GrammarElement.Type.CHAR && head.type() != GrammarElement.Type.CHAR_NOT) {
                out.add(head);
                i++;
                continue;
            }
            List<Integer> pairs = new ArrayList<>();
            i = readSet(rule, i, pairs);
            int[] bounds = new int[pairs.size()];
            for (int k = 0; k < bounds.length; k++) {
                bounds[k] = pairs.get(k);
            }
            boolean negated = head.type() == GrammarElement.Type.CHAR_NOT;
            writeSet(RegexCharSet.ofRanges(bounds).asciiCaseClosure(), negated, out);
        }
        return out;
    }

    /** Reads the set starting at {@code start} into range pairs; returns the index after it. */
    private static int readSet(List<GrammarElement> rule, int start, List<Integer> pairs) {
        int i = start;
        while (i < rule.size()) {
            GrammarElement e = rule.get(i);
            boolean member = i == start || e.type() == GrammarElement.Type.CHAR_ALT;
            if (!member) {
                break;
            }
            int lo = e.value();
            int hi = lo;
            i++;
            if (i < rule.size() && rule.get(i).type() == GrammarElement.Type.CHAR_RNG_UPPER) {
                hi = rule.get(i).value();
                i++;
            }
            pairs.add(lo);
            pairs.add(hi);
        }
        return i;
    }

    private static void writeSet(RegexCharSet set, boolean negated, List<GrammarElement> out) {
        int[] bounds = set.bounds();
        for (int k = 0; k < bounds.length; k += 2) {
            GrammarElement.Type type = k > 0 ? GrammarElement.Type.CHAR_ALT
                    : negated ? GrammarElement.Type.CHAR_NOT : GrammarElement.Type.CHAR;
            out.add(new GrammarElement(type, bounds[k]));
            if (bounds[k + 1] != bounds[k]) {
                out.add(new GrammarElement(GrammarElement.Type.CHAR_RNG_UPPER, bounds[k + 1]));
            }
        }
    }
}
