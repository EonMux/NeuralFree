package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * The hand-written recursive-descent parser of the Lark grammar dialect vLLM's xgrammar backend compiles
 * ({@code Grammar.from_lark}, the llguidance flavour of Lark): rule definitions {@code name: ...} (with the tree
 * modifiers {@code ?} and {@code !}, which do not change the language), terminal definitions {@code NAME: ...},
 * alternatives {@code |} (a new line may open with one), aliases {@code -> name}, groups {@code ( )}, optional
 * groups {@code [ ]}, the operators {@code ? * +}, {@code ~ n} / {@code ~ n..m} and llguidance's brace forms
 * {@code {n}}, {@code {n,}}, {@code {,m}} and {@code {n,m}}, double-quoted string literals
 * with JSON escapes and the {@code i} flag, character ranges {@code "a".."z"}, regular expressions {@code /.../}
 * with the {@code i}, {@code s} and {@code u} flags, the inline schema {@code %json {...}}, token sets
 * {@code <[1]>}, {@code <[1-5,7]>}, {@code <[^1,2]>} and named special tokens {@code <|im_end|>} (resolved through
 * the model vocabulary), the rule attributes {@code name[max_tokens=n, max_chars=m, stop="s"]} (xgrammar's budget
 * of the rule's text, and the strings its text ends at), and the directives {@code %import common.NAME [-> ALIAS]},
 * {@code %import common (A, B)}, {@code %ignore} and {@code %llguidance <JSON options>}. Comments start at
 * {@code //} or {@code #}.
 * </p>
 *
 * <p>Refusals name the construct, as xgrammar's do: rule or terminal priorities, templates, other rule attributes,
 * and attributes on a terminal, other directives, named grammar references {@code @name},
 * single-quoted strings, terminal intersection {@code &}, a GBNF {@code ::=} rule, unknown names, a terminal that
 * references a rule or itself, {@code %json} or a token inside a terminal, a non-ASCII case-insensitive literal,
 * and a missing {@code start} rule. Each is
 * a {@link GrammarException} at the 1-based line and column of the Lark source. One instance per source; confined
 * to the parsing thread.</p>
 */
final class LarkParser {

    /** The largest token set a {@code <[...]>} reference may name. */
    static final int MAX_TOKEN_SET = 1 << 16;

    /** The deepest nesting of groups {@code ( )} and {@code [ ]}. */
    static final int MAX_NESTING = 100;

    /** One rule or terminal definition. */
    static final class Definition {
        private final String name;
        private final boolean terminal;
        private final LarkExpr body;
        /** The rule's token budget ({@code max_tokens}), or -1. */
        private final int maxTokens;
        /** The rule's code-point budget ({@code max_chars}), or -1. */
        private final int maxChars;
        /** The rule's stop strings ({@code stop}), empty when it declares none. */
        private final List<String> stops;
        private final int line;
        private final int column;

        Definition(String name, boolean terminal, LarkExpr body, int[] budgets, List<String> stops, int line,
                int column) {
            this.name = name;
            this.terminal = terminal;
            this.body = body;
            this.maxTokens = budgets[0];
            this.maxChars = budgets[1];
            this.stops = Collections.unmodifiableList(new ArrayList<>(stops));
            this.line = line;
            this.column = column;
        }

        String name() {
            return name;
        }

        /** Whether the definition is a terminal (its name's first letter is upper case). */
        boolean terminal() {
            return terminal;
        }

        LarkExpr body() {
            return body;
        }

        /** The rule's token budget, or -1. */
        int maxTokens() {
            return maxTokens;
        }

        /** The rule's code-point budget, or -1. */
        int maxChars() {
            return maxChars;
        }

        /** The rule's stop strings, empty when it declares none. */
        List<String> stops() {
            return stops;
        }

        int line() {
            return line;
        }

        int column() {
            return column;
        }
    }

    /** The parsed grammar: its definitions in source order and its {@code %ignore} expressions. */
    static final class Result {
        private final Map<String, Definition> definitions;
        private final List<LarkExpr> ignores;

        Result(Map<String, Definition> definitions, List<LarkExpr> ignores) {
            this.definitions = Collections.unmodifiableMap(new LinkedHashMap<>(definitions));
            this.ignores = Collections.unmodifiableList(new ArrayList<>(ignores));
        }

        Map<String, Definition> definitions() {
            return definitions;
        }

        List<LarkExpr> ignores() {
            return ignores;
        }
    }

    private final String src;
    private final int[] lineStarts;
    private final GrammarVocab vocab;
    private final Map<String, Definition> definitions = new LinkedHashMap<>();
    private final List<LarkExpr> ignores = new ArrayList<>();
    private int pos;
    private int depth;

    private LarkParser(String src, GrammarVocab vocab) {
        this.src = src;
        this.lineStarts = lineStarts(src);
        this.vocab = vocab;
    }

    /**
     * Parses and validates a Lark grammar.
     *
     * @param vocab the model vocabulary named special tokens resolve through, or null to refuse them
     * @throws GrammarException at the Lark source position of a syntax error or a refused construct
     */
    static Result parse(String source, GrammarVocab vocab) {
        if (source.length() > GrammarLimits.MAX_SOURCE_CHARS) {
            throw new GrammarException("invalid Lark grammar: the source is longer than "
                    + GrammarLimits.MAX_SOURCE_CHARS + " characters", 1, 1);
        }
        LarkParser parser = new LarkParser(source, vocab);
        parser.statements();
        parser.validate();
        return new Result(parser.definitions, parser.ignores);
    }

    /** Whether a Lark name is a terminal's: its first letter is upper case. */
    static boolean isTerminalName(String name) {
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isLetter(c)) {
                return Character.isUpperCase(c);
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ statements

    private void statements() {
        while (true) {
            skipBlank();
            if (pos >= src.length()) {
                return;
            }
            char c = src.charAt(pos);
            if (c == '%') {
                directive();
            } else if (c == '?' || c == '!' || isNameStart(c)) {
                definition();
            } else {
                throw error("expected rule or terminal name", pos);
            }
            skipInline();
            if (pos < src.length() && src.charAt(pos) != '\n') {
                throw error("unexpected '" + src.charAt(pos) + "' after the statement", pos);
            }
        }
    }

    private void definition() {
        int start = pos;
        boolean modifier = src.charAt(pos) == '?' || src.charAt(pos) == '!';
        if (modifier) {
            pos++;
        }
        String name = name("expected rule or terminal name");
        boolean terminal = isTerminalName(name);
        if (modifier && terminal) {
            throw error("the ? and ! modifiers apply to rules, not to terminal '" + name + "'", start);
        }
        char next = peek(pos);
        if (next == '.') {
            throw error("rule and terminal priorities are not supported", pos);
        }
        if (next == '{') {
            throw error("Lark templates are not supported", pos);
        }
        int[] budgets = NO_BUDGETS;
        List<String> stops = NO_STOPS;
        if (next == '[') {
            if (terminal) {
                throw error("attributes are only supported on rules", pos);
            }
            budgets = new int[]{-1, -1};
            stops = attributes(budgets);
        }
        skipInline();
        if (src.startsWith("::=", pos)) {
            throw error("'::=' is GBNF rule syntax; a Lark rule is written '" + name + ": ...'", pos);
        }
        if (peek(pos) != ':') {
            throw error("expected ':' after '" + name + "'", pos);
        }
        pos++;
        LarkExpr body = expansions(false, terminal);
        define(new Definition(name, terminal, body, budgets, stops, lineOf(start), columnOf(start)), start);
    }

    /** The budgets of a rule without attributes. */
    private static final int[] NO_BUDGETS = {-1, -1};

    /** The stop strings of a rule without attributes. */
    private static final List<String> NO_STOPS = Collections.emptyList();

    /**
     * The rule attributes {@code [max_tokens=n, max_chars=m, stop="s"]} at {@code pos}: the budget of the rule's
     * text (xgrammar's budgets) and the strings its text ends at; any other attribute is refused by name.
     */
    private List<String> attributes(int[] budgets) {
        List<String> stops = new ArrayList<>();
        pos++;
        while (true) {
            skipInline();
            int keyAt = pos;
            String key = name("expected a rule attribute");
            int slot = "max_tokens".equals(key) ? 0 : "max_chars".equals(key) ? 1 : -1;
            if (slot < 0 && !key.equals("stop")) {
                throw error("rule attribute '" + key + "' is not supported; the engine reads max_tokens,"
                        + " max_chars and stop", keyAt);
            }
            if (slot >= 0 && budgets[slot] >= 0) {
                throw error(key + " attribute is specified more than once", keyAt);
            }
            if (slot < 0 && !stops.isEmpty()) {
                throw error("stop attribute is specified more than once", keyAt);
            }
            skipInline();
            if (peek(pos) != '=') {
                throw error("expected '=' after rule attribute '" + key + "'", pos);
            }
            pos++;
            skipInline();
            if (slot >= 0) {
                budgets[slot] = attributeValue();
            } else {
                stops.add(stopValue(keyAt));
            }
            skipInline();
            char c = peek(pos);
            pos++;
            if (c == ']') {
                return stops;
            }
            if (c != ',') {
                throw error("expected ',' or ']' in rule attributes", pos - 1);
            }
        }
    }

    /** The value of a {@code stop} attribute: a non-empty string literal without the {@code i} flag. */
    private String stopValue(int at) {
        if (peek(pos) != '"') {
            throw error("expected a string literal after stop=", pos);
        }
        int start = pos;
        String text = string();
        if (peek(pos) == 'i') {
            throw error("case-insensitive flags are not supported on stop", start);
        }
        if (text.isEmpty()) {
            throw error("stop must not be empty", start);
        }
        return text;
    }

    /** The value of a rule attribute: a non-negative integer at most {@link Integer#MAX_VALUE}. */
    private int attributeValue() {
        int start = pos;
        while (pos < src.length() && isDigit(src.charAt(pos))) {
            pos++;
        }
        if (pos == start) {
            throw error("expected a rule attribute value", start);
        }
        String digits = src.substring(start, pos);
        if (digits.length() > 10 || Long.parseLong(digits) > Integer.MAX_VALUE) {
            throw error("rule attribute value is larger than " + Integer.MAX_VALUE, start);
        }
        return Integer.parseInt(digits);
    }

    private void define(Definition definition, int at) {
        if (definitions.containsKey(definition.name())) {
            throw error("duplicate rule or terminal '" + definition.name() + "'", at);
        }
        if (definitions.size() >= GrammarLimits.MAX_RULES) {
            throw error("the grammar defines more than " + GrammarLimits.MAX_RULES + " rules and terminals", at);
        }
        definitions.put(definition.name(), definition);
    }

    private void directive() {
        int start = pos;
        pos++;
        int wordStart = pos;
        while (pos < src.length() && Character.isLetter(src.charAt(pos))) {
            pos++;
        }
        String word = src.substring(wordStart, pos);
        if (word.equals("import")) {
            importDirective(start);
        } else if (word.equals("ignore")) {
            ignores.add(expansions(false, true));
        } else if (word.equals("llguidance")) {
            llguidanceDirective(start);
        } else {
            throw error("directive %" + word + " is not supported", start);
        }
    }

    /**
     * {@code %llguidance <JSON options>}: the header llama.cpp's llguidance backend and llguidance's
     * {@code gbnf_to_lark} put in front of a Lark grammar. The options ({@code no_forcing} and
     * {@code allow_invalid_utf8}) tune llguidance's runtime, not the language, so they are validated and the rest of
     * the grammar compiles unchanged. llguidance reads the value as a JSON object, or as the same struct in sequence
     * form (at most two booleans); the value may sit on a later line.
     */
    private void llguidanceDirective(int start) {
        skipBlank();
        String text = jsonValue(start);
        Object options;
        try {
            options = Json.parse(text, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw error("failed to parse %llguidance declaration: " + e.getMessage(), start);
        }
        checkLlguidanceOptions(options, start);
    }

    /** The options of a {@code %llguidance} declaration: an object with boolean fields, or up to two booleans. */
    private void checkLlguidanceOptions(Object options, int start) {
        if (options instanceof Map) {
            Map<?, ?> fields = (Map<?, ?>) options;
            for (String field : new String[]{"no_forcing", "allow_invalid_utf8"}) {
                Object value = fields.get(field);
                if (value != null && !(value instanceof Boolean)) {
                    throw error("failed to parse %llguidance declaration: '" + field + "' must be a boolean", start);
                }
            }
            return;
        }
        if (options instanceof List) {
            List<?> values = (List<?>) options;
            if (values.size() > 2) {
                throw error("failed to parse %llguidance declaration: an options array holds at most 2 booleans",
                        start);
            }
            for (Object value : values) {
                if (!(value instanceof Boolean)) {
                    throw error("failed to parse %llguidance declaration: an options array holds booleans", start);
                }
            }
            return;
        }
        throw error("failed to parse %llguidance declaration: expected a JSON object", start);
    }

    /** The text of the JSON value at {@code pos}: a container, a string, a number or a keyword. */
    private String jsonValue(int at) {
        int start = pos;
        char first = peek(pos);
        if (first == '{' || first == '[') {
            int depth = 0;
            boolean inString = false;
            while (pos < src.length()) {
                char c = src.charAt(pos++);
                if (inString) {
                    if (c == '\\') {
                        pos++;
                    } else if (c == '"') {
                        inString = false;
                    }
                } else if (c == '"') {
                    inString = true;
                } else if (c == '{' || c == '[' || c == '}' || c == ']') {
                    if (c == '{' || c == '[') {
                        depth++;
                    } else if (--depth == 0) {
                        return src.substring(start, pos);
                    }
                }
            }
            throw error("unterminated %llguidance declaration", at);
        }
        if (first == '"') {
            pos++;
            while (pos < src.length()) {
                char c = src.charAt(pos++);
                if (c == '\\') {
                    pos++;
                } else if (c == '"') {
                    return src.substring(start, pos);
                } else if (c == '\n') {
                    break;
                }
            }
            throw error("unterminated %llguidance declaration", at);
        }
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == '\n' || c == ' ' || c == '\t' || c == '\r') {
                break;
            }
            pos++;
        }
        if (pos == start) {
            throw error("expected a JSON value after %llguidance", at);
        }
        return src.substring(start, pos);
    }

    /** {@code %import common.NAME [-> ALIAS]} or {@code %import common (A, B, ...)}. */
    private void importDirective(int start) {
        skipInline();
        int pathStart = pos;
        StringBuilder path = new StringBuilder(name("expected a module name after %import"));
        while (peek(pos) == '.') {
            pos++;
            path.append('.').append(name("expected a name after '.'"));
        }
        String dotted = path.toString();
        skipInline();
        if (dotted.indexOf('.') < 0 && peek(pos) == '(') {
            pos++;
            for (String imported : importList()) {
                importCommon(dotted, imported, imported, pathStart, start);
            }
            return;
        }
        int dot = dotted.lastIndexOf('.');
        if (dot < 0) {
            throw error("expected %import module.NAME or %import module (NAMES)", pathStart);
        }
        String alias = dotted.substring(dot + 1);
        if (src.startsWith("->", pos)) {
            pos += 2;
            skipInline();
            alias = name("expected an alias name after '->'");
        }
        importCommon(dotted.substring(0, dot), dotted.substring(dot + 1), alias, pathStart, start);
    }

    /** The names of {@code (A, B, ...)} after its opening parenthesis, through the closing one. */
    private List<String> importList() {
        List<String> names = new ArrayList<>();
        while (true) {
            skipBlank();
            names.add(name("expected a name in the %import list"));
            skipBlank();
            char c = peek(pos);
            pos++;
            if (c == ')') {
                return names;
            }
            if (c != ',') {
                throw error("expected ',' or ')' in the %import list", pos - 1);
            }
        }
    }

    private void importCommon(String module, String name, String alias, int pathAt, int statementAt) {
        String pattern = module.equals("common") ? LarkCommon.pattern(name) : null;
        if (pattern == null) {
            throw error("unknown common import '" + module + "." + name + "'", pathAt);
        }
        if (!isTerminalName(alias)) {
            throw error("the imported terminal '" + name + "' needs a terminal name, not '" + alias + "'", pathAt);
        }
        int line = lineOf(statementAt);
        int column = columnOf(statementAt);
        define(new Definition(alias, true, LarkExpr.regex(pattern, "", line, column), NO_BUDGETS, NO_STOPS, line,
                column), statementAt);
    }

    // ------------------------------------------------------------------ expressions

    /** Alternatives separated by {@code |}; at the top level a following line may continue them with one. */
    private LarkExpr expansions(boolean nested, boolean inTerminal) {
        int start = pos;
        List<LarkExpr> alternatives = new ArrayList<>();
        alternatives.add(alias(nested, inTerminal));
        while (true) {
            skipSpace(nested);
            if (peek(pos) != '|' && (nested || !continuesOnNextLine())) {
                break;
            }
            pos++;
            alternatives.add(alias(nested, inTerminal));
        }
        return alternatives.size() == 1 ? alternatives.get(0)
                : LarkExpr.alt(alternatives, lineOf(start), columnOf(start));
    }

    /** Whether the next non-blank line opens with {@code |}; if so, {@code pos} moves to it. */
    private boolean continuesOnNextLine() {
        if (peek(pos) != '\n') {
            return false;
        }
        int saved = pos;
        skipBlank();
        if (peek(pos) == '|') {
            return true;
        }
        pos = saved;
        return false;
    }

    private LarkExpr alias(boolean nested, boolean inTerminal) {
        LarkExpr expansion = expansion(nested, inTerminal);
        skipSpace(nested);
        if (src.startsWith("->", pos)) {
            if (nested || inTerminal) {
                throw error("an alias (->) may only follow a top-level alternative of a rule", pos);
            }
            pos += 2;
            skipInline();
            name("expected an alias name after '->'");
        }
        return expansion;
    }

    private LarkExpr expansion(boolean nested, boolean inTerminal) {
        int start = pos;
        List<LarkExpr> items = new ArrayList<>();
        while (true) {
            skipSpace(nested);
            char c = peek(pos);
            if (pos >= src.length() || c == '|' || c == ')' || c == ']' || c == '\n' || src.startsWith("->", pos)) {
                break;
            }
            items.add(postfix(atom(inTerminal), nested));
        }
        return items.size() == 1 ? items.get(0) : LarkExpr.seq(items, lineOf(start), columnOf(start));
    }

    /**
     * The atom with its one repetition operator, if any: {@code ? * +}, {@code ~ n[..m]}, or llguidance's brace form
     * {@code {n}}, {@code {n,}}, {@code {,m}} or {@code {n,m}}.
     */
    private LarkExpr postfix(LarkExpr atom, boolean nested) {
        skipSpace(nested);
        int at = pos;
        char c = peek(pos);
        int min;
        int max;
        if (c == '?' || c == '*' || c == '+') {
            pos++;
            min = c == '+' ? 1 : 0;
            max = c == '?' ? 1 : LarkExpr.UNBOUNDED;
        } else if (c == '~') {
            pos++;
            skipSpace(nested);
            min = count();
            skipSpace(nested);
            max = min;
            if (src.startsWith("..", pos)) {
                pos += 2;
                skipSpace(nested);
                max = count();
                if (max < min) {
                    throw error("repetition end must be greater than or equal to start", at);
                }
            }
        } else if (c == '{') {
            pos++;
            skipSpace(nested);
            min = peek(pos) == ',' ? 0 : count();
            skipSpace(nested);
            max = min;
            if (peek(pos) == ',') {
                pos++;
                skipSpace(nested);
                max = peek(pos) == '}' ? LarkExpr.UNBOUNDED : count();
                if (max != LarkExpr.UNBOUNDED && max < min) {
                    throw error("repetition end must be greater than or equal to start", at);
                }
            }
            skipSpace(nested);
            if (peek(pos) != '}') {
                throw error("expected '}' to close the repetition", pos);
            }
            pos++;
        } else {
            return atom;
        }
        skipSpace(nested);
        char after = peek(pos);
        if (after == '?' || after == '*' || after == '+' || after == '~' || after == '{') {
            throw error("a repetition operator cannot follow another; group the expression first", pos);
        }
        return LarkExpr.repeat(atom, min, max, atom.line(), atom.column());
    }

    /** A repetition count of {@code ~} or a brace bound, at most {@link Integer#MAX_VALUE}. */
    private int count() {
        int start = pos;
        while (pos < src.length() && isDigit(src.charAt(pos))) {
            pos++;
        }
        if (pos == start) {
            throw error("expected integer", start);
        }
        String digits = src.substring(start, pos);
        if (digits.length() > 10 || Long.parseLong(digits) > Integer.MAX_VALUE) {
            throw error("invalid non-negative repetition count", start);
        }
        return Integer.parseInt(digits);
    }

    private LarkExpr atom(boolean inTerminal) {
        int start = pos;
        char c = peek(pos);
        switch (c) {
            case '(':
                return group(')', inTerminal);
            case '[': {
                LarkExpr inner = group(']', inTerminal);
                return LarkExpr.repeat(inner, 0, 1, lineOf(start), columnOf(start));
            }
            case '"':
                return literal();
            case '/':
                return regex();
            case '<':
                if (inTerminal) {
                    throw error("special tokens cannot be used in terminals", start);
                }
                return token();
            case '%':
                return percentAtom(inTerminal);
            case '@':
                pos++;
                throw error("named grammar references (@" + nameOrEmpty()
                        + ") are not supported: the request names no grammars", start);
            case '\'':
                throw error("unexpected character '''; Lark string literals use double quotes", start);
            case '&':
                throw error("terminal intersection '&' is not supported", start);
            default:
                if (isNameStart(c)) {
                    String name = name("expected a name");
                    if (peek(pos) == '{') {
                        throw error("Lark templates are not supported", pos);
                    }
                    return LarkExpr.ref(name, lineOf(start), columnOf(start));
                }
                throw error("expected grammar expression", start);
        }
    }

    /** The alternatives inside the bracket at {@code pos} through {@code closing}, nested {@link #MAX_NESTING} deep. */
    private LarkExpr group(char closing, boolean inTerminal) {
        int openedAt = pos;
        if (++depth > MAX_NESTING) {
            throw error("groups nest deeper than " + MAX_NESTING, openedAt);
        }
        pos++;
        LarkExpr inner = expansions(true, inTerminal);
        skipBlank();
        if (peek(pos) != closing) {
            throw error("expected '" + closing + "' to close the '" + src.charAt(openedAt) + "'", pos);
        }
        pos++;
        depth--;
        return inner;
    }

    /** A string literal, its {@code i} flag, or a range {@code "a".."z"} of two one-character literals. */
    private LarkExpr literal() {
        int start = pos;
        String text = string();
        boolean caseInsensitive = peek(pos) == 'i';
        if (caseInsensitive) {
            pos++;
            if (text.chars().anyMatch(ch -> ch >= 0x80)) {
                throw error("case-insensitive string literals support ASCII characters only", start);
            }
        }
        int saved = pos;
        skipInline();
        if (!src.startsWith("..", pos)) {
            pos = saved;
            return LarkExpr.literal(text, caseInsensitive, lineOf(start), columnOf(start));
        }
        pos += 2;
        skipInline();
        if (peek(pos) != '"') {
            throw error("expected a string literal after '..'", pos);
        }
        String high = string();
        if (caseInsensitive || peek(pos) == 'i') {
            throw error("character range endpoints take no flags", start);
        }
        if (text.codePointCount(0, text.length()) != 1 || high.codePointCount(0, high.length()) != 1) {
            throw error("character range endpoints must be one character", start);
        }
        int lo = text.codePointAt(0);
        int hi = high.codePointAt(0);
        if (hi < lo) {
            throw error("character range \"" + text + "\"..\"" + high + "\" is reversed", start);
        }
        return LarkExpr.range(lo, hi, lineOf(start), columnOf(start));
    }

    /** A double-quoted string with JSON escapes, decoded. */
    private String string() {
        int start = pos;
        pos++;
        while (true) {
            char c = peek(pos);
            if (pos >= src.length() || c == '\n') {
                throw error("unterminated string literal", start);
            }
            if (c == '"') {
                break;
            }
            pos += c == '\\' ? 2 : 1;
        }
        pos++;
        try {
            return (String) Json.parse(src.substring(start, pos), Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw error("invalid string literal: " + e.getMessage(), start);
        }
    }

    /** A regular expression {@code /pattern/flags}; {@code \/} stands for a slash. */
    private LarkExpr regex() {
        int start = pos;
        pos++;
        StringBuilder pattern = new StringBuilder();
        while (true) {
            char c = peek(pos);
            if (pos >= src.length() || c == '\n') {
                throw error("unterminated regular expression", start);
            }
            if (c == '/') {
                break;
            }
            if (c == '\\' && peek(pos + 1) == '/') {
                pattern.append('/');
            } else if (c == '\\') {
                pattern.append(c).append(peek(pos + 1));
            } else {
                pattern.append(c);
            }
            pos += c == '\\' ? 2 : 1;
        }
        pos++;
        StringBuilder flags = new StringBuilder();
        while (pos < src.length() && Character.isLetter(src.charAt(pos))) {
            char flag = src.charAt(pos);
            if (flag != 'i' && flag != 's' && flag != 'u') {
                throw error("regular-expression flag '" + flag + "' is not supported", start);
            }
            flags.append(flag);
            pos++;
        }
        return LarkExpr.regex(pattern.toString(), flags.toString(), lineOf(start), columnOf(start));
    }

    /**
     * The {@code %} form at {@code pos}: the inline schema {@code %json}, or a named refusal of llguidance's other
     * forms ({@code %lark} nested grammars and {@code %regex} substring constraints).
     */
    private LarkExpr percentAtom(boolean inTerminal) {
        int start = pos;
        if (src.startsWith("%lark", pos) && !isNameChar(peek(pos + 5))) {
            throw error("nested %lark grammars are not supported", start);
        }
        if (src.startsWith("%regex", pos) && !isNameChar(peek(pos + 6))) {
            throw error("%regex substring constraints are not supported", start);
        }
        return json(inTerminal);
    }

    /** {@code %json} followed by a JSON schema object or boolean. */
    private LarkExpr json(boolean inTerminal) {
        int start = pos;
        if (!src.startsWith("%json", pos) || isNameChar(peek(pos + 5))) {
            throw error("only %json may appear inside an expression", start);
        }
        if (inTerminal) {
            throw error("%json cannot be used in terminals", start);
        }
        pos += 5;
        skipInline();
        String schema;
        if (peek(pos) == '{') {
            schema = jsonObject();
        } else if (src.startsWith("true", pos) || src.startsWith("false", pos)) {
            schema = src.startsWith("true", pos) ? "true" : "false";
            pos += schema.length();
        } else {
            throw error("%json takes a JSON schema object or boolean", pos);
        }
        return LarkExpr.json(schema, lineOf(start), columnOf(start));
    }

    /** The balanced JSON object text at {@code pos}, strings and their escapes honoured. */
    private String jsonObject() {
        int start = pos;
        int depth = 0;
        boolean inString = false;
        while (pos < src.length()) {
            char c = src.charAt(pos++);
            if (inString) {
                if (c == '\\') {
                    pos++;
                } else if (c == '"') {
                    inString = false;
                }
            } else if (c == '"') {
                inString = true;
            } else if (c == '{' || c == '[') {
                depth++;
            } else if ((c == '}' || c == ']') && --depth == 0) {
                return src.substring(start, pos);
            }
        }
        throw error("unterminated %json schema", start);
    }

    /** A token set {@code <[...]>} or a named special token {@code <...>}. */
    private LarkExpr token() {
        int start = pos;
        int close = src.indexOf('>', pos);
        int eol = src.indexOf('\n', pos);
        if (close < 0 || (eol >= 0 && eol < close)) {
            throw error("invalid special token", start);
        }
        String inner = src.substring(pos + 1, close);
        pos = close + 1;
        if (inner.startsWith("[")) {
            if (!inner.endsWith("]")) {
                throw error("invalid special token", start);
            }
            boolean negated = inner.startsWith("[^");
            int[] ids = tokenIds(inner.substring(negated ? 2 : 1, inner.length() - 1), start);
            return LarkExpr.tokens(ids, negated, lineOf(start), columnOf(start));
        }
        if (inner.isEmpty() || !inner.trim().equals(inner) || inner.indexOf(' ') >= 0) {
            throw error("invalid special token", start);
        }
        return LarkExpr.tokens(new int[]{namedToken("<" + inner + ">", start)}, false, lineOf(start),
                columnOf(start));
    }

    /** The ids of a numeric token set body: comma-separated ids and inclusive ranges {@code a-b}. */
    private int[] tokenIds(String body, int at) {
        Set<Integer> ids = new TreeSet<>();
        if (body.trim().isEmpty()) {
            throw error("empty numeric special-token range", at);
        }
        for (String part : body.split(",", -1)) {
            String item = part.trim();
            int dash = item.indexOf('-');
            long lo = tokenId(dash < 0 ? item : item.substring(0, dash), item, at);
            long hi = dash < 0 ? lo : tokenId(item.substring(dash + 1), item, at);
            if (hi < lo) {
                throw error("invalid numeric special-token range '" + item + "'", at);
            }
            if (hi - lo + ids.size() >= MAX_TOKEN_SET) {
                throw error("a token set names more than " + MAX_TOKEN_SET + " tokens", at);
            }
            for (long id = lo; id <= hi; id++) {
                ids.add((int) id);
            }
        }
        int[] out = new int[ids.size()];
        int i = 0;
        for (int id : ids) {
            out[i++] = id;
        }
        return out;
    }

    private long tokenId(String digits, String item, int at) {
        String d = digits.trim();
        if (d.isEmpty() || d.length() > 10 || !d.chars().allMatch(LarkParser::isDigit)
                || Long.parseLong(d) > Integer.MAX_VALUE) {
            throw error("invalid numeric special-token range '" + item + "'", at);
        }
        return Long.parseLong(d);
    }

    private int namedToken(String text, int at) {
        if (vocab == null) {
            throw error("named special token " + text + " requires a model vocabulary", at);
        }
        int[] ids = vocab.tokenize(text);
        if (ids == null || ids.length != 1) {
            throw error("special token " + text + " is not one token of the model vocabulary", at);
        }
        return ids[0];
    }

    // ------------------------------------------------------------------ validation

    /** The whole-grammar checks: a start rule, known names, and terminals free of rules and of cycles. */
    private void validate() {
        Definition start = definitions.get("start");
        if (start == null || start.terminal()) {
            throw new GrammarException("invalid Lark grammar: no start rule found", 1, 1);
        }
        for (Definition d : definitions.values()) {
            checkRefs(d.body(), d.terminal() ? "terminal '" + d.name() + "'" : null);
        }
        for (LarkExpr ignore : ignores) {
            checkRefs(ignore, "%ignore");
        }
        checkTerminalsAcyclic();
    }

    /** Every reference names a definition; inside {@code terminalOwner} (when non-null) a terminal one. */
    private void checkRefs(LarkExpr e, String terminalOwner) {
        if (e.kind() == LarkExpr.Kind.REF) {
            Definition target = definitions.get(e.text());
            if (target == null) {
                throw new GrammarException("invalid Lark grammar: unknown name '" + e.text() + "'", e.line(),
                        e.column());
            }
            if (terminalOwner != null && !target.terminal()) {
                throw new GrammarException("invalid Lark grammar: " + terminalOwner + " cannot reference rule '"
                        + e.text() + "'", e.line(), e.column());
            }
        }
        for (LarkExpr child : e.children()) {
            checkRefs(child, terminalOwner);
        }
    }

    /**
     * No terminal reaches itself: terminals referencing only finished terminals finish, repeatedly; any terminal
     * left over reaches a cycle, and following its references among the left-over ones finds a terminal on it.
     */
    private void checkTerminalsAcyclic() {
        Map<String, List<String>> refs = new LinkedHashMap<>();
        Map<String, List<String>> users = new HashMap<>();
        Map<String, Integer> open = new HashMap<>();
        List<String> ready = new ArrayList<>();
        for (Definition d : definitions.values()) {
            if (d.terminal()) {
                List<String> out = new ArrayList<>();
                collectRefs(d.body(), out);
                Set<String> distinct = new LinkedHashSet<>(out);
                refs.put(d.name(), new ArrayList<>(distinct));
                open.put(d.name(), distinct.size());
                for (String target : distinct) {
                    users.computeIfAbsent(target, k -> new ArrayList<>()).add(d.name());
                }
                if (distinct.isEmpty()) {
                    ready.add(d.name());
                }
            }
        }
        Set<String> finished = new HashSet<>();
        while (!ready.isEmpty()) {
            String done = ready.remove(ready.size() - 1);
            finished.add(done);
            for (String user : users.getOrDefault(done, Collections.emptyList())) {
                if (open.merge(user, -1, Integer::sum) == 0) {
                    ready.add(user);
                }
            }
        }
        for (String name : refs.keySet()) {
            if (!finished.contains(name)) {
                Definition onCycle = definitions.get(cycleMember(name, refs, finished));
                throw new GrammarException("invalid Lark grammar: circular reference in terminal '"
                        + onCycle.name() + "'", onCycle.line(), onCycle.column());
            }
        }
    }

    /** A terminal on the cycle {@code start} reaches, following unfinished references until one repeats. */
    private static String cycleMember(String start, Map<String, List<String>> refs, Set<String> finished) {
        Set<String> seen = new HashSet<>();
        String at = start;
        while (seen.add(at)) {
            for (String next : refs.get(at)) {
                if (!finished.contains(next)) {
                    at = next;
                    break;
                }
            }
        }
        return at;
    }

    static void collectRefs(LarkExpr e, List<String> out) {
        if (e.kind() == LarkExpr.Kind.REF) {
            out.add(e.text());
        }
        for (LarkExpr child : e.children()) {
            collectRefs(child, out);
        }
    }

    // ------------------------------------------------------------------ lexing

    private String name(String expected) {
        int start = pos;
        if (!isNameStart(peek(pos))) {
            throw error(expected, start);
        }
        while (pos < src.length() && isNameChar(src.charAt(pos))) {
            pos++;
        }
        return src.substring(start, pos);
    }

    private String nameOrEmpty() {
        int start = pos;
        while (pos < src.length() && isNameChar(src.charAt(pos))) {
            pos++;
        }
        return src.substring(start, pos);
    }

    /** Skips blanks and comments within the line ({@code //} and {@code #} run to its end). */
    private void skipInline() {
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == ' ' || c == '\t' || c == '\r') {
                pos++;
            } else if (c == '#' || (c == '/' && peek(pos + 1) == '/')) {
                while (pos < src.length() && src.charAt(pos) != '\n') {
                    pos++;
                }
            } else {
                return;
            }
        }
    }

    /** Skips blanks, comments and line breaks. */
    private void skipBlank() {
        while (true) {
            skipInline();
            if (peek(pos) != '\n' || pos >= src.length()) {
                return;
            }
            pos++;
        }
    }

    private void skipSpace(boolean nested) {
        if (nested) {
            skipBlank();
        } else {
            skipInline();
        }
    }

    private char peek(int at) {
        return at < src.length() ? src.charAt(at) : '\0';
    }

    private static boolean isNameStart(char c) {
        return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    private static boolean isNameChar(char c) {
        return isNameStart(c) || isDigit(c);
    }

    private static boolean isDigit(int c) {
        return c >= '0' && c <= '9';
    }

    private GrammarException error(String message, int at) {
        return new GrammarException("invalid Lark grammar: " + message, lineOf(at), columnOf(at));
    }

    /** The 1-based line of {@code offset}: a binary search of the line starts. */
    private int lineOf(int offset) {
        int found = Arrays.binarySearch(lineStarts, Math.min(offset, src.length()));
        return found >= 0 ? found + 1 : -found - 1;
    }

    /** The 1-based column of {@code offset} in its line, counted in code points. */
    private int columnOf(int offset) {
        int end = Math.min(offset, src.length());
        return src.codePointCount(lineStarts[lineOf(end) - 1], end) + 1;
    }

    private static int[] lineStarts(String text) {
        int lines = 1;
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) == '\n') {
                lines++;
            }
        }
        int[] starts = new int[lines];
        int line = 1;
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) == '\n') {
                starts[line++] = i + 1;
            }
        }
        return starts;
    }
}
