package com.eonmux.neuralfree.gguf.grammar;

/**
 * A GBNF parse or validation failure with the position in the grammar source that caused it.
 *
 * <p>Syntax errors, undefined rules, left recursion, a missing {@code root} rule and the
 * {@link GrammarLimits} bounds all surface as this exception. {@link #getLine()} and
 * {@link #getColumn()} are 1-based and point at the offending character or rule definition; the
 * column counts code points. It extends {@code IllegalArgumentException} because a malformed
 * grammar is malformed engine input.</p>
 */
public final class GrammarException extends IllegalArgumentException {

    private final int line;
    private final int column;

    /**
     * A failure at {@code line}/{@code column} (1-based); the carried message is the caller's
     * message with the position appended.
     */
    public GrammarException(String message, int line, int column) {
        super(message + " (line " + line + ", column " + column + ")");
        this.line = line;
        this.column = column;
    }

    /** 1-based source line of the failure. */
    public int getLine() {
        return line;
    }

    /** 1-based position in the failure's line, counted in code points. */
    public int getColumn() {
        return column;
    }
}
