package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The parsed and validated GBNF rule set, the Java shape of llama.cpp's grammar model
 * ({@code llama_grammar_rules}, {@code src/llama-grammar.h:64-69}). The parser
 * ({@link GbnfParser}) produces it; the sampling runtime of SO-2 consumes it.
 *
 * <p>{@link GbnfParser#parse} performs the checks {@code llama_grammar_init_impl} runs on a
 * parsed grammar ({@code src/llama-grammar.cpp:1215-1267}): the rule set must not be empty, must
 * declare the start rule {@link #ROOT_RULE}, must have every {@code RULE_REF} point at a defined
 * rule ({@code llama-grammar.cpp:700-718}) and must be free of left recursion
 * ({@code llama-grammar.cpp:963-1014}); each violation throws {@link GrammarException} at the
 * source line that caused it, which construction tracks per rule.</p>
 *
 * <p>Rule {@code i} of {@link #rule(int)} is the definition of the symbol named
 * {@link #symbolName(int)}; it always ends with {@code GrammarElement.Type.END}. Immutable and
 * thread-safe.</p>
 */
public final class Grammar {

    /** The start symbol every grammar must define ({@code LLAMA_VOCAB_GRAMMAR_ROOT}, "root"). */
    public static final String ROOT_RULE = "root";

    private final List<List<GrammarElement>> rules;
    private final Map<String, Integer> symbolIds;
    private final List<String> symbolNames;
    private final int rootRuleIndex;
    private final int[] ruleLines;
    private final int[] ruleColumns;

    Grammar(List<List<GrammarElement>> rules,
            Map<String, Integer> symbolIds,
            List<String> symbolNames,
            int rootRuleIndex,
            int[] ruleLines,
            int[] ruleColumns) {
        List<List<GrammarElement>> copy = new ArrayList<>(rules.size());
        for (List<GrammarElement> rule : rules) {
            copy.add(Collections.unmodifiableList(new ArrayList<>(rule)));
        }
        this.rules = Collections.unmodifiableList(copy);
        this.symbolIds = Collections.unmodifiableMap(new TreeMap<>(symbolIds));
        this.symbolNames = Collections.unmodifiableList(new ArrayList<>(symbolNames));
        this.rootRuleIndex = rootRuleIndex;
        this.ruleLines = ruleLines.clone();
        this.ruleColumns = ruleColumns.clone();
        detectLeftRecursion();
    }

    /** Number of rules; every id in {@code [0, ruleCount())} names a defined rule. */
    public int ruleCount() {
        return rules.size();
    }

    /** Definition of rule {@code index}; unmodifiable view, ends with {@code Type.END}. */
    public List<GrammarElement> rule(int index) {
        return rules.get(index);
    }

    /** Rule id of {@code name}, or {@code -1} when the grammar has no such symbol. */
    public int symbolId(String name) {
        Integer id = symbolIds.get(name);
        return id == null ? -1 : id;
    }

    /** Name of rule {@code id}, the inverse of {@link #symbolId}. */
    public String symbolName(int id) {
        return symbolNames.get(id);
    }

    /** Rule id of {@link #ROOT_RULE}, which construction guarantees to be defined. */
    public int rootRuleIndex() {
        return rootRuleIndex;
    }

    /** 1-based definition line of rule {@code id}, as parsed from the grammar source. */
    int ruleLine(int id) {
        return ruleLines[id];
    }

    /** 1-based definition column of rule {@code id}, as parsed from the grammar source. */
    int ruleColumn(int id) {
        return ruleColumns[id];
    }

    /** A rule reference ends one alternate and may start another ({@code llama-grammar.cpp:754-761}). */
    static boolean isEndOfSequence(GrammarElement element) {
        return element.type() == GrammarElement.Type.END
                || element.type() == GrammarElement.Type.ALT;
    }

    /**
     * The left-recursion detection of {@code llama_grammar_detect_left_recursion}
     * ({@code llama-grammar.cpp:963-1014}): depth-first re-entry marks recursion; a rule that can
     * produce the empty string lets the scan continue into the next non-terminal of the sequence.
     */
    private void detectLeftRecursion() {
        boolean[] visited = new boolean[rules.size()];
        boolean[] inProgress = new boolean[rules.size()];
        boolean[] mayBeEmpty = new boolean[rules.size()];
        for (int i = 0; i < rules.size(); i++) {
            if (visited[i]) {
                continue;
            }
            if (detectLeftRecursion(i, visited, inProgress, mayBeEmpty)) {
                throw new GrammarException(
                        "unsupported grammar, left recursion detected for nonterminal '" + symbolName(i) + "'",
                        ruleLines[i], ruleColumns[i]);
            }
        }
    }

    private boolean detectLeftRecursion(int ruleIndex, boolean[] visited, boolean[] inProgress,
            boolean[] mayBeEmpty) {
        if (inProgress[ruleIndex]) {
            return true;
        }
        inProgress[ruleIndex] = true;
        List<GrammarElement> rule = rules.get(ruleIndex);

        // First check if the rule might produce the empty string.
        boolean atRuleStart = true;
        for (GrammarElement element : rule) {
            if (isEndOfSequence(element)) {
                if (atRuleStart) {
                    mayBeEmpty[ruleIndex] = true;
                    break;
                }
                atRuleStart = true;
            } else {
                atRuleStart = false;
            }
        }

        // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous
        // nonterminal may be empty).
        boolean recurseIntoNonterminal = true;
        for (GrammarElement element : rule) {
            if (element.type() == GrammarElement.Type.RULE_REF && recurseIntoNonterminal) {
                if (detectLeftRecursion(element.value(), visited, inProgress, mayBeEmpty)) {
                    return true;
                }
                if (!mayBeEmpty[element.value()]) {
                    recurseIntoNonterminal = false;
                }
            } else if (isEndOfSequence(element)) {
                recurseIntoNonterminal = true;
            } else {
                recurseIntoNonterminal = false;
            }
        }
        inProgress[ruleIndex] = false;
        visited[ruleIndex] = true;
        return false;
    }
}
