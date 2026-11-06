package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * One immutable node of a parsed Lark expression ({@link LarkParser}): a sequence, an alternation, a repetition, or
 * one of the atoms a Lark grammar writes — a string literal (optionally case-insensitive), a character range
 * {@code "a".."z"}, a regular expression {@code /.../flags}, a rule or terminal reference, an inline {@code %json}
 * schema, a token set {@code <[...]>}, and the {@code %ignore} slot {@link LarkGrammar} inserts after tokens. Every
 * node keeps the 1-based source line and column it was read from, for diagnostics.
 */
final class LarkExpr {

    /** Which construct the node is. */
    enum Kind {
        /** The children in order; no children matches the empty string. */
        SEQ,
        /** One of the children. */
        ALT,
        /** The one child repeated {@code min..max} times ({@link #UNBOUNDED} for no upper bound). */
        REPEAT,
        /** The literal {@link #text()}, ASCII and Unicode case folded when {@link #caseInsensitive()}. */
        LITERAL,
        /** One code point in {@code low()..high()}. */
        RANGE,
        /** The regular expression {@link #text()} with its {@link #flags()}. */
        REGEX,
        /** The rule or terminal named {@link #text()}. */
        REF,
        /** The inline JSON schema {@link #text()}. */
        JSON,
        /** One token of {@link #tokens()}, or one token outside them when {@link #negated()}. */
        TOKENS,
        /** The ignored-text slot of {@code %ignore}. */
        IGNORE
    }

    /** The upper bound of an unbounded repetition. */
    static final int UNBOUNDED = -1;

    private final Kind kind;
    private final List<LarkExpr> children;
    private final String text;
    private final String flags;
    private final int low;
    private final int high;
    private final int[] tokens;
    private final boolean negated;
    private final int line;
    private final int column;

    private LarkExpr(Kind kind, List<LarkExpr> children, String text, String flags, int low, int high, int[] tokens,
            boolean negated, int line, int column) {
        this.kind = kind;
        this.children = children == null ? Collections.emptyList()
                : Collections.unmodifiableList(new ArrayList<>(children));
        this.text = text;
        this.flags = flags == null ? "" : flags;
        this.low = low;
        this.high = high;
        this.tokens = tokens == null ? new int[0] : tokens.clone();
        this.negated = negated;
        this.line = line;
        this.column = column;
    }

    static LarkExpr seq(List<LarkExpr> items, int line, int column) {
        return new LarkExpr(Kind.SEQ, items, null, null, 0, 0, null, false, line, column);
    }

    static LarkExpr alt(List<LarkExpr> alternatives, int line, int column) {
        return new LarkExpr(Kind.ALT, alternatives, null, null, 0, 0, null, false, line, column);
    }

    /** {@code item} repeated {@code min..max} times; {@code max} is {@link #UNBOUNDED} for no upper bound. */
    static LarkExpr repeat(LarkExpr item, int min, int max, int line, int column) {
        return new LarkExpr(Kind.REPEAT, Collections.singletonList(item), null, null, min, max, null, false,
                line, column);
    }

    static LarkExpr literal(String text, boolean caseInsensitive, int line, int column) {
        return new LarkExpr(Kind.LITERAL, null, text, caseInsensitive ? "i" : "", 0, 0, null, false, line, column);
    }

    static LarkExpr range(int low, int high, int line, int column) {
        return new LarkExpr(Kind.RANGE, null, null, null, low, high, null, false, line, column);
    }

    static LarkExpr regex(String pattern, String flags, int line, int column) {
        return new LarkExpr(Kind.REGEX, null, pattern, flags, 0, 0, null, false, line, column);
    }

    static LarkExpr ref(String name, int line, int column) {
        return new LarkExpr(Kind.REF, null, name, null, 0, 0, null, false, line, column);
    }

    static LarkExpr json(String schema, int line, int column) {
        return new LarkExpr(Kind.JSON, null, schema, null, 0, 0, null, false, line, column);
    }

    static LarkExpr tokens(int[] ids, boolean negated, int line, int column) {
        return new LarkExpr(Kind.TOKENS, null, null, null, 0, 0, ids, negated, line, column);
    }

    static LarkExpr ignore() {
        return new LarkExpr(Kind.IGNORE, null, null, null, 0, 0, null, false, 0, 0);
    }

    /** The node with {@code children} in place of its own, everything else kept (SEQ, ALT, REPEAT). */
    LarkExpr withChildren(List<LarkExpr> replacement) {
        return new LarkExpr(kind, replacement, text, flags, low, high, tokens, negated, line, column);
    }

    Kind kind() {
        return kind;
    }

    /** The children of a SEQ, ALT or REPEAT; empty for the atoms. */
    List<LarkExpr> children() {
        return children;
    }

    /** A literal's text, a regex's pattern, a reference's name or a JSON schema's text; null otherwise. */
    String text() {
        return text;
    }

    /** A regex's flag letters, {@code "i"} for a case-insensitive literal, else empty. */
    String flags() {
        return flags;
    }

    boolean caseInsensitive() {
        return kind == Kind.LITERAL && flags.equals("i");
    }

    /** A repetition's minimum or a range's lowest code point. */
    int low() {
        return low;
    }

    /** A repetition's maximum ({@link #UNBOUNDED} for none) or a range's highest code point. */
    int high() {
        return high;
    }

    /** A token set's ids; a copy. */
    int[] tokens() {
        return tokens.clone();
    }

    boolean negated() {
        return negated;
    }

    int line() {
        return line;
    }

    int column() {
        return column;
    }

    /** Whether the node is one token of the grammar: everything but a sequence, alternation, repetition or rule. */
    boolean isTokenAtom() {
        return kind == Kind.LITERAL || kind == Kind.RANGE || kind == Kind.REGEX || kind == Kind.JSON
                || kind == Kind.TOKENS;
    }
}
