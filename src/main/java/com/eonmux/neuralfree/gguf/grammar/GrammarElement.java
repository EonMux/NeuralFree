package com.eonmux.neuralfree.gguf.grammar;

import java.util.Objects;

/**
 * One element of a grammar rule, the port of {@code llama_grammar_element}
 * ({@code src/llama-grammar.h:47-50}). The element type mirrors {@code llama_gretype}
 * ({@code llama-grammar.h:13-45}), plus the token-set alternate {@link Type#TOKEN_ALT}, the any-token
 * {@link Type#TOKEN_ANY} and the region budgets {@link Type#TOKEN_BUDGET} and {@link Type#CHAR_BUDGET}:
 *
 * <ul>
 *   <li>{@link Type#END} ends a rule definition (also terminates one alternate of a rule);</li>
 *   <li>{@link Type#ALT} starts an alternate definition for the rule;</li>
 *   <li>{@link Type#RULE_REF} references a rule by id;</li>
 *   <li>{@link Type#CHAR}, {@link Type#CHAR_NOT}, {@link Type#CHAR_ALT} and
 *       {@link Type#CHAR_RNG_UPPER} form character sets ({@code [a]}, {@code [^a-z]}, {@code [ab]},
 *       {@code [a-b]});</li>
 *   <li>{@link Type#CHAR_ANY} matches any character ({@code .});</li>
 *   <li>{@link Type#TOKEN} and {@link Type#TOKEN_NOT} constrain token ids
 *       ({@code <[id]>} and {@code !<[id]>}); trailing {@link Type#TOKEN_ALT} elements widen either into a
 *       token set ({@code <[a],[b]>} matches a or b, {@code !<[a],[b]>} any token but a and b), the token
 *       counterpart of {@code CHAR_ALT} that xgrammar's {@code ExcludeToken} needs;</li>
 *   <li>{@link Type#TOKEN_ANY} matches any one token ({@code !<>}, xgrammar's {@code ExcludeToken()} with an
 *       empty set);</li>
 *   <li>{@link Type#TOKEN_BUDGET} and {@link Type#CHAR_BUDGET} open a counted region over the rest of their
 *       sequence: it consumes at most {@code value} tokens or code points (xgrammar's rule attributes
 *       {@code name[max_tokens=n]} and {@code name[max_chars=n]}, the syntax {@link GbnfParser} reads). A token
 *       that opens the region does not count against its token budget; every code point inside it counts
 *       against its character budget.</li>
 * </ul>
 *
 * <p>{@code value} carries the Unicode code point for the {@code CHAR_*} family, the rule id for
 * {@link Type#RULE_REF}, the token id for {@code TOKEN}/{@code TOKEN_NOT}/{@code TOKEN_ALT}, the budget for
 * {@code TOKEN_BUDGET}/{@code CHAR_BUDGET} and {@code 0} for the remaining types. Immutable.</p>
 */
public final class GrammarElement {

    /** Element type of a grammar rule item; the mirror of {@code llama_gretype}. */
    public enum Type {
        /** End of the rule definition. */
        END,
        /** Start of an alternate rule definition. */
        ALT,
        /** Non-terminal element: reference to a rule by id. */
        RULE_REF,
        /** Terminal element: one character (code point). */
        CHAR,
        /** Inverse character set ({@code [^a]}, {@code [^a-b]}, {@code [^abc]}). */
        CHAR_NOT,
        /** Inclusive upper range bound of the preceding character ({@code [a-z]}). */
        CHAR_RNG_UPPER,
        /** Alternate character of the set ({@code [ab]}, {@code [a-zA]}). */
        CHAR_ALT,
        /** Any character ({@code .}). */
        CHAR_ANY,
        /** Terminal element: token id ({@code <[id]>}). */
        TOKEN,
        /** Inverse token ({@code !<[id]>}). */
        TOKEN_NOT,
        /** Alternate token id of the preceding TOKEN/TOKEN_NOT set ({@code <[a],[b]>}, {@code !<[a],[b]>}). */
        TOKEN_ALT,
        /** Any one token ({@code !<>}). */
        TOKEN_ANY,
        /** The rest of the sequence consumes at most {@code value} tokens ({@code name[max_tokens=n]}). */
        TOKEN_BUDGET,
        /** The rest of the sequence consumes at most {@code value} code points ({@code name[max_chars=n]}). */
        CHAR_BUDGET,
    }

    /** Whether {@code type} matches one whole token: {@link Type#TOKEN}, {@link Type#TOKEN_NOT} or {@link Type#TOKEN_ANY}. */
    public static boolean isToken(Type type) {
        return type == Type.TOKEN || type == Type.TOKEN_NOT || type == Type.TOKEN_ANY;
    }

    /** Whether {@code type} opens a counted region: {@link Type#TOKEN_BUDGET} or {@link Type#CHAR_BUDGET}. */
    public static boolean isBudget(Type type) {
        return type == Type.TOKEN_BUDGET || type == Type.CHAR_BUDGET;
    }

    private final Type type;
    private final int value;

    /** An element of {@code type} carrying {@code value}; see the class contract for its meaning. */
    public GrammarElement(Type type, int value) {
        this.type = Objects.requireNonNull(type, "type");
        this.value = value;
    }

    public Type type() {
        return type;
    }

    public int value() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GrammarElement)) return false;
        GrammarElement other = (GrammarElement) o;
        return type == other.type && value == other.value;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, value);
    }

    /**
     * Short form for debugging, shaped like llama.cpp's {@code print_rule_binary}
     * ({@code llama-grammar.cpp:256-298}): the type name and its bracketed value.
     */
    @Override
    public String toString() {
        return type + "(" + value + ")";
    }
}
