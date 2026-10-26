package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The ECMAScript regex subset to GBNF translator, the Java port of
 * {@code common_chat_schema_converter::_pattern_to_rule} and its helpers
 * ({@code common/json-schema-to-grammar.cpp:284-340,390-631}). A pattern is translated into
 * GBNF rule text in a {@link GbnfBuilder}; whole strings and lazy trigger patterns are wrapped
 * {@code "\"" ( … ) "\""} exactly like llama.cpp writes them, so a match is one JSON string.
 *
 * <p>The supported subset, llama.cpp's: the pattern must be anchored {@code ^…$}; literal
 * runs and GBNF escape forms; {@code .} (non-{@code dotAll}: any code point except CR and LF);
 * character classes copied verbatim, whose escapes must already be GBNF escapes
 * ({@code \\xNN}, uNNNN, {@code \\UNNNNNNNN}, {@code \\t \\r \\n \\ \\\" \\[ \\] \\-});
 * non-capturing groups {@code (?: … )}; alternation {@code |}; and the repetition operators
 * {@code * + ? {m} {m,} {m,n}}. Anything else is refused with a message naming the construct
 * ({@link UnsupportedRegexException}); a structurally broken pattern (a stray operator,
 * unbalanced or unterminated escapes, anchors inside the pattern) is an
 * {@link InvalidRegexException}. Like the rest of the grammar engine, this does not consult
 * {@code java.util.regex} (PARITY row G2 note): the translation is a single direct scan.
 *
 * <p>{@code JsonSchemaToGbnf} catches {@link UnsupportedRegexException} and degrades the
 * schema property to the generic string rule as llama.cpp does (a warning, not an error);
 * an invalid pattern is a conversion error. A standalone regex constraint
 * ({@code ConstraintSpec.regex}, G4) must not degrade, so it is refused instead. SO-2 lazy
 * triggers and SA-3 {@code stop_regex} consume {@link #compile}: the returned {@link Grammar}
 * is the compiled automaton, advanced like any other grammar's rule set.</p>
 *
 * <p>One instance per pattern; the scan state ({@code pos}, {@code parenDepth},
 * {@code subRuleIds}) is thread-confined to the thread translating the pattern.</p>
 */
public final class RegexToGbnf {

    /** Deepest group nesting accepted, llama.cpp {@code MAX_PATTERN_DEPTH} ({@code :282}). */
    public static final int MAX_PATTERN_DEPTH = 100;

    /**
     * The pattern is a valid regex with no GBNF equivalent; the message names the construct.
     * Unchecked so it can cross the translator's recursion freely, as llama.cpp does.
     */
    public static final class UnsupportedRegexException extends RuntimeException {
        UnsupportedRegexException(String construct) {
            super(construct);
        }
    }

    /**
     * The pattern is not a valid regex; the message names the problem, matching llama.cpp's
     * {@code invalid_pattern} (an error, never a graceful fallback).
     */
    public static final class InvalidRegexException extends RuntimeException {
        InvalidRegexException(String problem) {
            super(problem);
        }
    }

    /** Whether {@code pattern} is anchored {@code ^…$}, the precondition of every translation. */
    public static boolean isAnchored(String pattern) {
        return pattern.length() >= 2 && pattern.charAt(0) == '^'
                && pattern.charAt(pattern.length() - 1) == '$';
    }

    /**
     * Translates {@code pattern} and adds its rule under {@code name} in {@code builder} (the
     * port of llama.cpp {@code _pattern_to_rule},
     * {@code common/json-schema-to-grammar.cpp:406-631}).
     *
     * @param pattern the ECMAScript regex, anchored {@code ^…$}
     * @param name    the requested rule name (sanitized by the builder)
     * @param builder receives the rule and helper rules such as the {@code .} rule
     * @param dotAll  when {@code true} a {@code .} accepts any code point, otherwise any code
     *                point except CR and LF (llama.cpp {@code get_dot}, {@code :426-434})
     * @return the stored rule name, llama.cpp's return of {@code _add_rule}
     * @throws UnsupportedRegexException naming the unsupported construct
     * @throws InvalidRegexException     naming the structural problem
     */
    public static String patternRule(String pattern, String name, GbnfBuilder builder,
            boolean dotAll) {
        if (!isAnchored(pattern)) {
            throw new UnsupportedRegexException("not anchored with '^' and '$'");
        }
        return new RegexToGbnf(pattern.substring(1, pattern.length() - 1), name, builder, dotAll)
                .translate();
    }

    /**
     * Translates {@code pattern} into its own builder and parses the result with
     * {@link GbnfParser}: the compiled automaton of the regex, which SO-2's lazy triggers and
     * SA-3's {@code stop_regex} drive the ordinary grammar stacks over.
     *
     * @return the grammar whose wrapped-string rule is stored under {@code name}
     * @throws UnsupportedRegexException naming the unsupported construct (unanchored, or outside
     *                                   llama.cpp's subset)
     * @throws InvalidRegexException     naming the structural problem
     */
    public static Grammar compile(String pattern, String name, boolean dotAll) {
        GbnfBuilder builder = new GbnfBuilder();
        patternRule(pattern, name, builder, dotAll);
        return builder.grammar();
    }

    private final String sub;
    private final String name;
    private final GbnfBuilder builder;
    private final boolean dotAll;
    /** Helper rule text to stored name, llama.cpp {@code sub_rule_ids} ({@code :411}). */
    private final Map<String, String> subRuleIds = new HashMap<>();
    private int pos;
    private int parenDepth;

    private RegexToGbnf(String sub, String name, GbnfBuilder builder, boolean dotAll) {
        this.sub = sub;
        this.name = name;
        this.builder = builder;
        this.dotAll = dotAll;
    }

    private String translate() {
        String rule = transform();
        if (parenDepth != 0) {
            throw new InvalidRegexException("unbalanced parentheses");
        }
        return builder.addRule(name, "\"\\\"\" (" + rule + ") \"\\\"");
    }

    /**
     * One alternation-or-sequence group of the scan (llama.cpp's {@code transform} lambda,
     * {@code :423-623}). Consumes the closing parenthesis of its group before returning
     * {@code join_seq}'s joined text; a group never closed before the pattern end leaves the
     * depth counter up, which {@link #translate} turns into an invalid-pattern error.
     */
    private String transform() {
        List<Item> seq = new ArrayList<>();
        while (pos < sub.length()) {
            char c = sub.charAt(pos);
            if (c == '.') {
                seq.add(new Item(dotRuleText(), false));
                pos++;
            } else if (c == '(') {
                pos++;
                if (pos < sub.length() && sub.charAt(pos) == '?') {
                    if (pos + 1 < sub.length() && sub.charAt(pos + 1) == ':') {
                        pos += 2; // non-capturing group, treated as a regular group
                    } else {
                        // lookarounds, named groups, inline flags
                        throw new UnsupportedRegexException("unsupported group syntax");
                    }
                }
                parenDepth++;
                if (parenDepth > MAX_PATTERN_DEPTH) {
                    throw new UnsupportedRegexException("pattern nesting too deep");
                }
                seq.add(new Item("(" + transform() + ")", false));
            } else if (c == ')') {
                pos++;
                if (parenDepth == 0) {
                    throw new InvalidRegexException("unbalanced parentheses");
                }
                parenDepth--;
                return joinSeq(seq);
            } else if (c == '^' || c == '$') {
                throw new UnsupportedRegexException("anchor inside the pattern");
            } else if (c == '[') {
                seq.add(new Item(charClassText(), false));
            } else if (c == '|') {
                seq.add(new Item("|", false));
                pos++;
            } else if (c == '*' || c == '+' || c == '?') {
                pos++;
                applyStar(seq, c);
            } else if (c == '{') {
                curly(seq);
            } else {
                literalRun(seq);
            }
        }
        return joinSeq(seq);
    }

    private String dotRuleText() {
        String rule = dotAll ? "[\\U00000000-\\U0010FFFF]" : "[^\\x0A\\x0D]";
        return builder.addRule("dot", rule);
    }

    /**
     * A {@code [ … ]} class copied verbatim; escapes inside a class must already be GBNF
     * escapes (llama.cpp {@code :498-519}).
     */
    private String charClassText() {
        StringBuilder out = new StringBuilder("[");
        pos++;
        while (pos < sub.length() && sub.charAt(pos) != ']') {
            if (sub.charAt(pos) == '\\') {
                int escapeLength = gbnfEscapeLength(sub, pos);
                if (escapeLength == 0) {
                    throw new UnsupportedRegexException("unsupported escape in character class: "
                            + sub.substring(pos, Math.min(pos + 2, sub.length())));
                }
                out.append(sub, pos, pos + escapeLength);
                pos += escapeLength;
            } else {
                out.append(sub.charAt(pos));
                pos++;
            }
        }
        if (pos >= sub.length()) {
            throw new InvalidRegexException("unterminated character class");
        }
        out.append(']');
        pos++;
        return out.toString();
    }

    /** {@code * + ?} on the last item, llama.cpp {@code :523-528}. */
    private static void applyStar(List<Item> seq, char operator) {
        if (seq.isEmpty()) {
            throw new InvalidRegexException("nothing to repeat");
        }
        int last = seq.size() - 1;
        Item item = seq.get(last);
        seq.set(last, new Item(toRule(item) + operator, false));
    }

    /**
     * {@code {m}, {m,}, {m,n}} bounds on the last item (llama.cpp {@code :529-581}); a
     * non-literal item is first hoisted into a helper rule named {@code <name>-<n>}.
     */
    private void curly(List<Item> seq) {
        StringBuilder brackets = new StringBuilder("{");
        pos++;
        while (pos < sub.length() && sub.charAt(pos) != '}') {
            brackets.append(sub.charAt(pos));
            pos++;
        }
        if (pos >= sub.length()) {
            throw new UnsupportedRegexException("unterminated curly brackets");
        }
        brackets.append('}');
        pos++;

        List<String> parts = splitComma(brackets.substring(1, brackets.length() - 1));
        if (parts.size() != 1 && parts.size() != 2) {
            throw new UnsupportedRegexException("wrong number of values in curly brackets");
        }
        int minTimes = 0;
        int maxTimes = Integer.MAX_VALUE;
        if (parts.size() == 1) {
            minTimes = maxTimes = stoiLike(parts.get(0));
        } else {
            if (!parts.get(0).isEmpty()) {
                minTimes = stoiLike(parts.get(0));
            }
            if (!parts.get(1).isEmpty()) {
                maxTimes = stoiLike(parts.get(1));
            }
        }
        if (seq.isEmpty()) {
            throw new InvalidRegexException("nothing to repeat");
        }
        int last = seq.size() - 1;
        Item item = seq.get(last);
        String subId = item.li ? null : subRuleIds.get(item.text);
        if (!item.li && subId == null) {
            // llama.cpp inserts an empty id first, so the first helper lands on <name>-1
            subId = builder.addRule(name + "-" + (subRuleIds.size() + 1), item.text);
            subRuleIds.put(item.text, subId);
        }
        seq.set(last, new Item(GbnfBuilder.repetition(
                item.li ? "\"" + item.text + "\"" : subId, minTimes, maxTimes, ""), false));
    }

    /**
     * A run of literal characters with their escapes (llama.cpp {@code :582-620}): '"' becomes
     * {@code \"}, the GBNF escape forms pass through verbatim; the run stops before every
     * operator/meta character unless it would be empty there.
     */
    private void literalRun(List<Item> seq) {
        StringBuilder out = new StringBuilder();
        while (pos < sub.length()) {
            char c = sub.charAt(pos);
            if (c == '\\') {
                if (pos == sub.length() - 1) {
                    throw new InvalidRegexException("trailing backslash");
                }
                char next = sub.charAt(pos + 1);
                if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.indexOf(next) >= 0) {
                    pos++;
                    out.append(sub.charAt(pos));
                    pos++;
                } else {
                    int escapeLength = gbnfEscapeLength(sub, pos);
                    if (escapeLength == 0) {
                        throw new UnsupportedRegexException("unsupported escape: "
                                + sub.substring(pos, Math.min(pos + 2, sub.length())));
                    }
                    out.append(sub, pos, pos + escapeLength);
                    pos += escapeLength;
                }
            } else if (c == '"') {
                out.append("\\\"");
                pos++;
            } else if (!isNonLiteral(c) && (pos == sub.length() - 1 || out.length() == 0
                    || sub.charAt(pos + 1) == '.' || !isNonLiteral(sub.charAt(pos + 1)))) {
                out.append(c);
                pos++;
            } else {
                break;
            }
        }
        if (out.length() == 0) {
            // nothing was consumed, e.g. a stray ']' or '}'
            throw new UnsupportedRegexException("unsupported character: " + sub.charAt(pos));
        }
        seq.add(new Item(out.toString(), true));
    }

    /**
     * Joins the sequence, merging consecutive literal items (llama.cpp {@code join_seq},
     * {@code :437-467}).
     */
    private static String joinSeq(List<Item> seq) {
        List<Item> ret = new ArrayList<>();
        StringBuilder literal = new StringBuilder();
        for (Item item : seq) {
            if (item.li) {
                literal.append(item.text);
            } else {
                flushLiteral(ret, literal);
                ret.add(item);
            }
        }
        flushLiteral(ret, literal);
        StringBuilder out = new StringBuilder();
        for (Item item : ret) {
            if (out.length() > 0) {
                out.append(' ');
            }
            out.append(toRule(item));
        }
        return out.toString();
    }

    private static void flushLiteral(List<Item> ret, StringBuilder literal) {
        if (literal.length() > 0) {
            ret.add(new Item(literal.toString(), true));
            literal.setLength(0);
        }
    }

    private static String toRule(Item item) {
        return item.li ? "\"" + item.text + "\"" : item.text;
    }

    // ------------------------------------------------------------------ scan infrastructure

    /** One sequence item: {@code li} marks a literal run. */
    private static final class Item {
        final String text;
        final boolean li;

        Item(String text, boolean li) {
            this.text = text;
            this.li = li;
        }
    }

    /** llama.cpp's comma {@code string_split} ({@code common/common.cpp:543-557}). */
    private static List<String> splitComma(String body) {
        List<String> parts = new ArrayList<>();
        int start = 0;
        int comma = body.indexOf(',', start);
        while (comma >= 0) {
            parts.add(body.substring(start, comma));
            start = comma + 1;
            comma = body.indexOf(',', start);
        }
        parts.add(body.substring(start));
        return parts;
    }

    // ------------------------------------------------------------------ character sets

    /** llama.cpp {@code NON_LITERAL_SET} ({@code :284}). */
    private static final String NON_LITERAL_SET = "|.()[]{}*+?^$";

    /** llama.cpp {@code ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS} ({@code :285}). */
    private static final String ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = "^$[]()|{}*+?";

    private static boolean isNonLiteral(char c) {
        return NON_LITERAL_SET.indexOf(c) >= 0;
    }

    /** {@code std::stoi} semantics trimmed to what a curly bracket body may hold. */
    private static int stoiLike(String part) {
        int i = 0;
        while (i < part.length() && (part.charAt(i) == ' ' || part.charAt(i) == '\t')) {
            i++;
        }
        boolean negative = false;
        if (i < part.length() && (part.charAt(i) == '+' || part.charAt(i) == '-')) {
            negative = part.charAt(i) == '-';
            i++;
        }
        int end = i;
        while (end < part.length() && part.charAt(end) >= '0' && part.charAt(end) <= '9') {
            end++;
        }
        if (end == i) {
            throw new UnsupportedRegexException("invalid number in curly brackets");
        }
        try {
            int value = Integer.parseInt(part.substring(i, end));
            return negative ? -value : value;
        } catch (NumberFormatException e) {
            throw new UnsupportedRegexException("invalid number in curly brackets");
        }
    }

    /**
     * Length of the GBNF escape starting at {@code pos}, the port of llama.cpp
     * {@code gbnf_escape_length} ({@code :315-340}), 0 when {@code sub} does not start one at
     * {@code pos} (keep in sync with {@link GbnfParser}'s character escapes).
     */
    static int gbnfEscapeLength(String sub, int pos) {
        if (pos + 1 >= sub.length() || sub.charAt(pos) != '\\') {
            return 0;
        }
        int hexDigits;
        switch (sub.charAt(pos + 1)) {
            case 'x': hexDigits = 2; break;
            case 'u': hexDigits = 4; break;
            case 'U': hexDigits = 8; break;
            case 't': case 'r': case 'n': case '\\': case '"': case '[': case ']': case '-':
                return 2;
            default:
                return 0;
        }
        if (pos + 2 + hexDigits > sub.length()) {
            return 0;
        }
        for (int i = pos + 2; i < pos + 2 + hexDigits; i++) {
            char h = sub.charAt(i);
            if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'))) {
                return 0;
            }
        }
        return 2 + hexDigits;
    }
}
