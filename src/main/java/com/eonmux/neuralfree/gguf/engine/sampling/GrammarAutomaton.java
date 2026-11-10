package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarElement;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * The pushdown machinery of llama.cpp's grammar runtime ({@code llama-grammar.cpp:755-940}) over one {@link Grammar}
 * flattened into primitive arrays: the element types and values of every rule laid end to end, and each rule's start.
 * A partial parse is a {@link Stack} of flattened positions (top last); a budget region also keeps its frame on the
 * stack, the remaining count right below a negative marker ({@link #TOKEN_FRAME}, {@link #CHAR_FRAME}), as
 * {@link GrammarStage} documents. Immutable and thread-safe: every operation returns new stacks.
 */
final class GrammarAutomaton {

    static final int END = 0;
    static final int ALT = 1;
    static final int RULE_REF = 2;
    static final int CHAR = 3;
    static final int CHAR_NOT = 4;
    static final int CHAR_RNG_UPPER = 5;
    static final int CHAR_ALT = 6;
    static final int CHAR_ANY = 7;
    static final int TOKEN = 8;
    static final int TOKEN_NOT = 9;
    static final int TOKEN_ALT = 10;
    static final int TOKEN_ANY = 11;
    static final int TOKEN_BUDGET = 12;
    static final int CHAR_BUDGET = 13;

    /** The stack marker of a token budget frame; the remaining token count lies right below it. */
    static final int TOKEN_FRAME = -1;

    /** The stack marker of a character budget frame; the remaining code point count lies right below it. */
    static final int CHAR_FRAME = -2;

    /**
     * One partial parse: flattened positions and budget frames, top last. Immutable, with its hash cached, so sets of
     * stacks deduplicate in constant time per stack.
     */
    static final class Stack {
        final int[] entries;
        private final int hash;

        Stack(int[] entries) {
            this.entries = entries;
            this.hash = Arrays.hashCode(entries);
        }

        int size() {
            return entries.length;
        }

        boolean isEmpty() {
            return entries.length == 0;
        }

        int top() {
            return entries[entries.length - 1];
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Stack && ((Stack) o).hash == hash && Arrays.equals(((Stack) o).entries, entries);
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public String toString() {
            return Arrays.toString(entries);
        }
    }

    static final Stack EMPTY = new Stack(new int[0]);

    private final int[] type;
    private final int[] value;
    private final int[] ruleStart;
    private final int rootStart;

    GrammarAutomaton(Grammar grammar) {
        int total = 0;
        for (int r = 0; r < grammar.ruleCount(); r++) {
            total += grammar.rule(r).size();
        }
        this.type = new int[total];
        this.value = new int[total];
        this.ruleStart = new int[grammar.ruleCount()];
        int at = 0;
        for (int r = 0; r < grammar.ruleCount(); r++) {
            ruleStart[r] = at;
            for (GrammarElement element : grammar.rule(r)) {
                type[at] = code(element.type());
                value[at] = element.value();
                at++;
            }
        }
        this.rootStart = ruleStart[grammar.rootRuleIndex()];
    }

    private static int code(GrammarElement.Type t) {
        switch (t) {
            case END: return END;
            case ALT: return ALT;
            case RULE_REF: return RULE_REF;
            case CHAR: return CHAR;
            case CHAR_NOT: return CHAR_NOT;
            case CHAR_RNG_UPPER: return CHAR_RNG_UPPER;
            case CHAR_ALT: return CHAR_ALT;
            case CHAR_ANY: return CHAR_ANY;
            case TOKEN: return TOKEN;
            case TOKEN_NOT: return TOKEN_NOT;
            case TOKEN_ALT: return TOKEN_ALT;
            case TOKEN_ANY: return TOKEN_ANY;
            case TOKEN_BUDGET: return TOKEN_BUDGET;
            case CHAR_BUDGET: return CHAR_BUDGET;
            default: throw new IllegalArgumentException("unknown grammar element type " + t);
        }
    }

    /** Whether the element at {@code pos} matches one whole token (TOKEN, TOKEN_NOT, TOKEN_ANY). */
    boolean isToken(int pos) {
        int t = type[pos];
        return t == TOKEN || t == TOKEN_NOT || t == TOKEN_ANY;
    }

    private boolean isEndOfSequence(int pos) {
        int t = type[pos];
        return t == END || t == ALT;
    }

    /**
     * {@code llama_grammar_init_impl}'s start state: one partial parse per alternative of the root rule, an empty
     * alternative being the empty (completed) stack.
     */
    Stack[] initialStacks() {
        Set<Stack> initial = new LinkedHashSet<>();
        int pos = rootStart;
        while (true) {
            advanceStack(isEndOfSequence(pos) ? EMPTY : new Stack(new int[]{pos}), initial);
            while (!isEndOfSequence(pos)) {
                pos++;
            }
            if (type[pos] != ALT) {
                return initial.toArray(new Stack[0]);
            }
            pos++;
        }
    }

    /**
     * {@code llama_grammar_advance_stack}: expands rule references into every terminal-reaching stack, added to
     * {@code out} in discovery order unless already present. A budget element pushes its frame under the rest of its
     * sequence; a frame on top (its region complete) pops.
     */
    void advanceStack(Stack stack, Set<Stack> out) {
        Deque<Stack> todo = new ArrayDeque<>();
        todo.push(stack);
        Set<Stack> seen = new HashSet<>();
        while (!todo.isEmpty()) {
            Stack curr = todo.pop();
            if (!seen.add(curr)) {
                continue;
            }
            if (curr.isEmpty()) {
                out.add(curr);
                continue;
            }
            int[] e = curr.entries;
            int pos = e[e.length - 1];
            if (pos < 0) {
                todo.push(new Stack(Arrays.copyOf(e, e.length - 2)));
                continue;
            }
            int t = type[pos];
            if (t == TOKEN_BUDGET || t == CHAR_BUDGET) {
                boolean rest = !isEndOfSequence(pos + 1);
                int[] framed = Arrays.copyOf(e, e.length + (rest ? 2 : 1));
                framed[e.length - 1] = value[pos];
                framed[e.length] = t == TOKEN_BUDGET ? TOKEN_FRAME : CHAR_FRAME;
                if (rest) {
                    framed[e.length + 1] = pos + 1;
                }
                todo.push(new Stack(framed));
            } else if (t == RULE_REF) {
                int sub = ruleStart[value[pos]];
                boolean rest = !isEndOfSequence(pos + 1);
                while (true) {
                    boolean body = !isEndOfSequence(sub);
                    int[] next = Arrays.copyOf(e, e.length - 1 + (rest ? 1 : 0) + (body ? 1 : 0));
                    int at = e.length - 1;
                    if (rest) {
                        next[at++] = pos + 1;
                    }
                    if (body) {
                        next[at] = sub;
                    }
                    todo.push(new Stack(next));
                    while (!isEndOfSequence(sub)) {
                        sub++;
                    }
                    if (type[sub] == ALT) {
                        sub++;
                    } else {
                        break;
                    }
                }
            } else if (t == CHAR || t == CHAR_NOT || t == CHAR_ANY || isToken(pos)) {
                out.add(curr);
            } else {
                throw new IllegalStateException("grammar stack left on " + GrammarElement.Type.values()[t]
                        + " (a malformed rule ends mid-sequence)");
            }
        }
    }

    /**
     * The stacks after the char cluster on top of {@code stack} consumed one code point and moved to {@code next}
     * ({@link #matchChar}'s position after the cluster): the open character frames spend one, {@code next} replaces
     * the top unless the sequence ends there, and the result is advanced to its terminals.
     */
    Stack[] afterChar(Stack stack, int next) {
        int[] e = consumeChar(stack.entries);
        boolean rest = !isEndOfSequence(next);
        int[] moved = Arrays.copyOf(e, e.length - 1 + (rest ? 1 : 0));
        if (rest) {
            moved[e.length - 1] = next;
        }
        Set<Stack> out = new LinkedHashSet<>();
        advanceStack(new Stack(moved), out);
        return out.toArray(new Stack[0]);
    }

    /** {@code llama_grammar_accept_chr}: consumes one code point at {@code stack}'s top into {@code out}. */
    void acceptChr(Stack stack, int chr, Set<Stack> out) {
        if (stack.isEmpty()) {
            return;
        }
        int pos = stack.top();
        if (isToken(pos) || !charsLeft(stack)) {
            return;
        }
        int match = matchChar(pos, chr);
        if (match >= 0) {
            int[] e = consumeChar(stack.entries);
            boolean rest = !isEndOfSequence(match);
            int[] moved = Arrays.copyOf(e, e.length - 1 + (rest ? 1 : 0));
            if (rest) {
                moved[e.length - 1] = match;
            }
            advanceStack(new Stack(moved), out);
        }
    }

    /**
     * The token step of {@code llama_grammar_accept_impl}: {@code stack}, whose top is a token element at
     * {@code pos} matching the token, with that element popped and the sequence after its id set pushed, advanced
     * into {@code out}.
     */
    void acceptToken(Stack stack, int pos, Set<Stack> out) {
        int after = afterTokenSet(pos);
        int[] e = stack.entries;
        boolean rest = !isEndOfSequence(after);
        int[] moved = Arrays.copyOf(e, e.length - 1 + (rest ? 1 : 0));
        if (rest) {
            moved[e.length - 1] = after;
        }
        advanceStack(new Stack(moved), out);
    }

    /**
     * {@code llama_grammar_match_char}: the code point against the char cluster at {@code pos}. Returns the position
     * after the cluster when the code point matches, else that position's complement ({@code ~next}, negative).
     */
    int matchChar(int pos, int chr) {
        boolean found = false;
        boolean positive = type[pos] == CHAR || type[pos] == CHAR_ANY;
        do {
            if (type[pos + 1] == CHAR_RNG_UPPER) {
                found = found || (value[pos] <= chr && chr <= value[pos + 1]);
                pos += 2;
            } else if (type[pos] == CHAR_ANY) {
                found = true;
                pos += 1;
            } else {
                found = found || value[pos] == chr;
                pos += 1;
            }
        } while (type[pos] == CHAR_ALT);
        return found == positive ? pos : ~pos;
    }

    /** {@code llama_grammar_match_partial_char}: whether the partial UTF-8 tail could satisfy {@code pos}. */
    boolean matchPartialChar(int pos, int partialValue, int nRemain) {
        boolean positive = type[pos] == CHAR || type[pos] == CHAR_ANY;
        if (nRemain < 0 || (nRemain == 1 && partialValue < 2)) {
            return false;
        }
        int low = partialValue << (nRemain * 6);
        int high = low | ((1 << (nRemain * 6)) - 1);
        if (low == 0) {
            if (nRemain == 2) {
                low = 1 << 11;
            } else if (nRemain == 3) {
                low = 1 << 16;
            }
        }
        do {
            if (type[pos + 1] == CHAR_RNG_UPPER) {
                if (value[pos] <= high && low <= value[pos + 1]) {
                    return positive;
                }
                pos += 2;
            } else if (type[pos] == CHAR_ANY) {
                return true;
            } else {
                if (low <= value[pos] && value[pos] <= high) {
                    return positive;
                }
                pos += 1;
            }
        } while (type[pos] == CHAR_ALT);
        return !positive;
    }

    /**
     * {@code llama_grammar_match_token}: the token id against a TOKEN / TOKEN_NOT element and the TOKEN_ALT ids
     * after it: a TOKEN set matches any of its ids, a TOKEN_NOT set any token outside them, TOKEN_ANY any token.
     */
    boolean matchToken(int pos, int token) {
        int t = type[pos];
        if (t == TOKEN_ANY) {
            return true;
        }
        if (t != TOKEN && t != TOKEN_NOT) {
            throw new IllegalStateException("token match outside a token element: "
                    + GrammarElement.Type.values()[t]);
        }
        boolean found = value[pos] == token;
        for (int i = pos + 1; !found && type[i] == TOKEN_ALT; i++) {
            found = value[i] == token;
        }
        return found == (t == TOKEN);
    }

    /** The position after the token element at {@code pos} and its TOKEN_ALT ids. */
    private int afterTokenSet(int pos) {
        int i = pos + 1;
        while (type[i] == TOKEN_ALT) {
            i++;
        }
        return i;
    }

    // ------------------------------------------------------------------ budget frames

    /**
     * {@code stack} as a new token starts on it: every open token frame spends one, or null when one is already
     * spent (the parse cannot take the token inside its region). The same stack when no token frame is open.
     */
    static Stack tokenStart(Stack stack) {
        int[] spent = spend(stack.entries, TOKEN_FRAME);
        return spent == null ? null : spent == stack.entries ? stack : new Stack(spent);
    }

    /** Whether {@code stack} may consume one more code point: none of its character frames is spent. */
    static boolean charsLeft(Stack stack) {
        int[] e = stack.entries;
        for (int i = e.length - 1; i > 0; i--) {
            if (e[i] == CHAR_FRAME) {
                if (e[i - 1] == 0) {
                    return false;
                }
                i--;
            } else if (e[i] == TOKEN_FRAME) {
                i--;
            }
        }
        return true;
    }

    /** {@code entries} after one code point: every open character frame spends one ({@link #charsLeft} holds). */
    private static int[] consumeChar(int[] entries) {
        int[] spent = spend(entries, CHAR_FRAME);
        if (spent == null) {
            throw new IllegalStateException("a code point consumed past a spent character budget");
        }
        return spent;
    }

    /**
     * {@code entries} with every {@code marker} frame's count spent by one, or null when one of them is already 0.
     * Frames are read from the top: a marker's count lies right below it, so a count is never read as a marker.
     * The same array when no such frame is open.
     */
    private static int[] spend(int[] entries, int marker) {
        int[] out = null;
        for (int i = entries.length - 1; i > 0; i--) {
            int entry = entries[i];
            if (entry >= 0) {
                continue;
            }
            if (entry == marker) {
                int left = entries[i - 1];
                if (left == 0) {
                    return null;
                }
                if (out == null) {
                    out = entries.clone();
                }
                out[i - 1] = left - 1;
            }
            i--;
        }
        return out == null ? entries : out;
    }
}
