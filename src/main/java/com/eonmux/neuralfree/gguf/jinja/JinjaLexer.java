package com.eonmux.neuralfree.gguf.jinja;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Tokenizer for Jinja templates.
 *
 * <p>Produces one flat token stream: {@link Kind#TEXT} for template data and, for every
 * {@code {{ }}} or {@code {% %}} tag, an open token, the expression tokens and a close token.
 * Comments disappear and {@code {% raw %}} blocks become text. Whitespace control follows the
 * HF/llama.cpp chat-template environment (llama.cpp common/jinja/lexer.cpp:132-225): {@code trim_blocks}
 * and {@code lstrip_blocks} are on, {@code -} strips all adjacent whitespace, {@code +} on a block
 * opener disables {@code lstrip_blocks} and on a block closer disables {@code trim_blocks}, line
 * endings are normalized to {@code \n} and a single trailing newline is dropped.
 */
public final class JinjaLexer {

    /** Token categories. */
    public enum Kind { TEXT, EXPR_OPEN, EXPR_CLOSE, STMT_OPEN, STMT_CLOSE, NAME, STRING, INT, FLOAT, OP, EOF }

    /** Immutable token with its 1-based source line. */
    public static final class Token {
        final Kind kind;
        final String text;
        final int line;

        Token(Kind kind, String text, int line) {
            this.kind = kind;
            this.text = text;
            this.line = line;
        }

        boolean is(Kind k, String t) { return kind == k && text.equals(t); }

        boolean isName(String t) { return is(Kind.NAME, t); }

        boolean isOp(String t) { return is(Kind.OP, t); }

        @Override public String toString() {
            switch (kind) {
                case EOF: return "end of template";
                case TEXT: return "template text";
                case STRING: return "string literal";
                default: return "'" + text + "'";
            }
        }
    }

    private static final String[] OPERATORS = {
        "**", "//", "==", "!=", "<=", ">=",
        "+", "-", "*", "/", "%", "~", "<", ">", "=", ".", ",", "|", ":", "(", ")", "[", "]", "{", "}"
    };

    private final String src;
    private final List<Token> tokens = new ArrayList<>();
    private int pos;
    private int line = 1;
    private boolean prevStripAfter;
    private boolean prevTrimNewline;

    private JinjaLexer(String source) {
        String s = source.replace("\r\n", "\n").replace('\r', '\n');
        this.src = s.endsWith("\n") ? s.substring(0, s.length() - 1) : s;
    }

    /** Tokenizes a template. */
    public static List<Token> tokenize(String source) {
        JinjaLexer lx = new JinjaLexer(source);
        lx.run();
        return Collections.unmodifiableList(lx.tokens);
    }

    private void run() {
        while (pos < src.length()) {
            int open = findTagOpen(pos);
            int textEnd = open < 0 ? src.length() : open;
            int textLine = line;
            String raw = src.substring(pos, textEnd);
            advanceLines(raw);
            char kind = open < 0 ? 0 : src.charAt(open + 1);
            char modifier = open >= 0 && open + 2 < src.length() ? src.charAt(open + 2) : 0;
            emitText(raw, textLine, pos == 0, kind, modifier);
            if (open < 0) break;
            pos = open + 2 + (modifier == '-' || modifier == '+' ? 1 : 0);
            if (kind == '#') lexComment();
            else if (kind == '%' && isRawStart()) lexRaw();
            else lexTag(kind == '{');
        }
        tokens.add(new Token(Kind.EOF, "", line));
    }

    private int findTagOpen(int from) {
        for (int i = from; i + 1 < src.length(); i++) {
            if (src.charAt(i) == '{') {
                char c = src.charAt(i + 1);
                if (c == '{' || c == '%' || c == '#') return i;
            }
        }
        return -1;
    }

    /**
     * Emits template text after applying the closing control of the previous tag and the opening
     * control of the next one ({@code kind} 0 when the text runs to the end of the template).
     */
    private void emitText(String raw, int textLine, boolean atTemplateStart, char kind, char modifier) {
        String text = raw;
        if (kind != 0) {
            if (modifier == '-') {
                text = rstrip(text);
            } else if ((kind == '%' || kind == '#') && modifier != '+') {
                text = lstripBlock(text, atTemplateStart);
            }
        }
        if (prevStripAfter) text = lstrip(text);
        else if (prevTrimNewline && text.startsWith("\n")) text = text.substring(1);
        prevStripAfter = false;
        prevTrimNewline = false;
        if (!text.isEmpty()) tokens.add(new Token(Kind.TEXT, text, textLine));
    }

    /** Removes the spaces and tabs between the last line start and a block tag. */
    private static String lstripBlock(String text, boolean atTemplateStart) {
        int i = text.length();
        while (i > 0 && (text.charAt(i - 1) == ' ' || text.charAt(i - 1) == '\t')) i--;
        boolean lineStart = i == 0 ? atTemplateStart : text.charAt(i - 1) == '\n';
        return lineStart ? text.substring(0, i) : text;
    }

    private void lexComment() {
        int end = src.indexOf("#}", pos);
        if (end < 0) throw error("unterminated comment");
        String body = src.substring(pos, end);
        advanceLines(body);
        setCloseControl(end > pos && src.charAt(end - 1) == '-', end > pos && src.charAt(end - 1) == '+', true);
        pos = end + 2;
    }

    private boolean isRawStart() {
        int i = skipSpaces(pos);
        if (!src.startsWith("raw", i)) return false;
        i = skipSpaces(i + 3);
        if (i < src.length() && (src.charAt(i) == '-' || src.charAt(i) == '+')) i++;
        return src.startsWith("%}", i);
    }

    private void lexRaw() {
        int close = src.indexOf("%}", pos);
        boolean stripBody = src.charAt(close - 1) == '-';
        int bodyStart = close + 2;
        int search = bodyStart;
        while (true) {
            int tag = src.indexOf("{%", search);
            if (tag < 0) throw error("unterminated raw block");
            int i = tag + 2;
            boolean stripEnd = i < src.length() && src.charAt(i) == '-';
            if (stripEnd || i < src.length() && src.charAt(i) == '+') i++;
            i = skipSpaces(i);
            if (src.startsWith("endraw", i)) {
                int j = skipSpaces(i + 6);
                boolean dashClose = j < src.length() && src.charAt(j) == '-';
                boolean plusClose = j < src.length() && src.charAt(j) == '+';
                if (dashClose || plusClose) j++;
                if (!src.startsWith("%}", j)) throw error("malformed endraw tag");
                String body = src.substring(bodyStart, tag);
                if (stripBody) body = lstrip(body);
                if (stripEnd) body = rstrip(body);
                advanceLines(src.substring(pos, j + 2));
                if (!body.isEmpty()) tokens.add(new Token(Kind.TEXT, body, line));
                setCloseControl(dashClose, plusClose, true);
                pos = j + 2;
                return;
            }
            search = tag + 2;
        }
    }

    private void lexTag(boolean expression) {
        tokens.add(new Token(expression ? Kind.EXPR_OPEN : Kind.STMT_OPEN, expression ? "{{" : "{%", line));
        String close = expression ? "}}" : "%}";
        int braceDepth = 0;
        while (true) {
            skipWhitespace();
            if (pos >= src.length()) throw error("unterminated " + (expression ? "expression" : "statement") + " tag");
            char c = src.charAt(pos);
            if ((c == '-' || c == '+') && src.startsWith(close, pos + 1) && (braceDepth == 0 || !expression)) {
                closeTag(expression, c == '-', c == '+', pos + 1 + close.length());
                return;
            }
            if (src.startsWith(close, pos) && (braceDepth == 0 || !expression)) {
                closeTag(expression, false, false, pos + close.length());
                return;
            }
            if (c == '\'' || c == '"') {
                lexString(c);
            } else if (Character.isDigit(c)) {
                lexNumber();
            } else if (Character.isLetter(c) || c == '_') {
                int start = pos;
                while (pos < src.length() && (Character.isLetterOrDigit(src.charAt(pos)) || src.charAt(pos) == '_')) pos++;
                tokens.add(new Token(Kind.NAME, src.substring(start, pos), line));
            } else {
                String op = matchOperator();
                if (op == null) throw error("unexpected character '" + c + "'");
                if ("{".equals(op)) braceDepth++;
                else if ("}".equals(op)) braceDepth--;
                tokens.add(new Token(Kind.OP, op, line));
                pos += op.length();
            }
        }
    }

    private void closeTag(boolean expression, boolean dash, boolean plus, int next) {
        tokens.add(new Token(expression ? Kind.EXPR_CLOSE : Kind.STMT_CLOSE, expression ? "}}" : "%}", line));
        setCloseControl(dash, plus, !expression);
        pos = next;
    }

    private void setCloseControl(boolean dash, boolean plus, boolean block) {
        prevStripAfter = dash;
        prevTrimNewline = !dash && block && !plus;
    }

    private String matchOperator() {
        for (String op : OPERATORS) if (src.startsWith(op, pos)) return op;
        return null;
    }

    private void lexNumber() {
        int start = pos;
        boolean isFloat = false;
        consumeDigits();
        if (pos + 1 < src.length() && src.charAt(pos) == '.' && Character.isDigit(src.charAt(pos + 1))) {
            isFloat = true;
            pos++;
            consumeDigits();
        }
        if (pos < src.length() && (src.charAt(pos) == 'e' || src.charAt(pos) == 'E')) {
            int save = pos;
            pos++;
            if (pos < src.length() && (src.charAt(pos) == '+' || src.charAt(pos) == '-')) pos++;
            if (pos < src.length() && Character.isDigit(src.charAt(pos))) {
                isFloat = true;
                consumeDigits();
            } else {
                pos = save;
            }
        }
        String text = src.substring(start, pos).replace("_", "");
        tokens.add(new Token(isFloat ? Kind.FLOAT : Kind.INT, text, line));
    }

    private void consumeDigits() {
        while (pos < src.length() && (Character.isDigit(src.charAt(pos))
                || src.charAt(pos) == '_' && pos + 1 < src.length() && Character.isDigit(src.charAt(pos + 1)))) {
            pos++;
        }
    }

    /** Python string literal with escape sequences; literals may span lines. */
    private void lexString(char quote) {
        int startLine = line;
        StringBuilder sb = new StringBuilder();
        pos++;
        while (true) {
            if (pos >= src.length()) throw error("unterminated string literal starting on line " + startLine);
            char c = src.charAt(pos++);
            if (c == quote) break;
            if (c == '\n') line++;
            if (c != '\\') {
                sb.append(c);
                continue;
            }
            if (pos >= src.length()) throw error("unterminated string literal starting on line " + startLine);
            char e = src.charAt(pos++);
            switch (e) {
                case 'n': sb.append('\n'); break;
                case 't': sb.append('\t'); break;
                case 'r': sb.append('\r'); break;
                case 'b': sb.append('\b'); break;
                case 'f': sb.append('\f'); break;
                case 'v': sb.append('\u000b'); break;
                case 'a': sb.append('\u0007'); break;
                case '0': sb.append('\0'); break;
                case '\\': case '\'': case '"': sb.append(e); break;
                case '\n': line++; break;
                case 'x': sb.appendCodePoint(hexEscape(2)); break;
                case 'u': sb.appendCodePoint(hexEscape(4)); break;
                case 'U': sb.appendCodePoint(hexEscape(8)); break;
                default: sb.append('\\').append(e); break;
            }
        }
        tokens.add(new Token(Kind.STRING, sb.toString(), startLine));
    }

    private int hexEscape(int digits) {
        if (pos + digits > src.length()) throw error("truncated \\x/\\u escape in string literal");
        try {
            long cp = Long.parseLong(src.substring(pos, pos + digits), 16);
            if (cp > Character.MAX_CODE_POINT) throw error("escape \\U" + src.substring(pos, pos + digits) + " is not a valid code point");
            pos += digits;
            return (int) cp;
        } catch (NumberFormatException e) {
            throw error("invalid hexadecimal escape in string literal");
        }
    }

    private void skipWhitespace() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            if (src.charAt(pos) == '\n') line++;
            pos++;
        }
    }

    private int skipSpaces(int i) {
        while (i < src.length() && Character.isWhitespace(src.charAt(i))) i++;
        return i;
    }

    private void advanceLines(String s) {
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) == '\n') line++;
    }

    private static String lstrip(String s) {
        int i = 0;
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        return s.substring(i);
    }

    private static String rstrip(String s) {
        int i = s.length();
        while (i > 0 && Character.isWhitespace(s.charAt(i - 1))) i--;
        return s.substring(0, i);
    }

    private JinjaRuntime.TemplateException error(String message) {
        return new JinjaRuntime.TemplateException(line, "syntax error: " + message);
    }
}
