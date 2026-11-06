package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Hand-written recursive-descent parser for GBNF, the Java port of llama.cpp's
 * {@code llama_grammar_parser} ({@code src/llama-grammar.cpp:14-752}); no ANTLR or other
 * dependency is used. {@link #parse} parses a whole grammar into a validated {@link Grammar}.
 *
 * <p>Syntax and port rules, mirroring llama.cpp:</p>
 *
 * <ul>
 *   <li>rules {@code name ::= alternates}, one per line, separated by a newline; comments start
 *       at {@code #} and run to the end of the line ({@code parse_space}, {@code :125-138});</li>
 *   <li>literal strings {@code "chars"}, character sets {@code [^a-zAB]} with ranges and
 *       alternates, any-character {@code .}, rule references by name, grouping {@code (...)},
 *       and the operators {@code *}, {@code +}, {@code ?}, {@code {m}}, {@code {m,}} and
 *       {@code {m,n}} — the repetition expansion rewrites each operator into synthesized
 *       {@code <name>_<id>} rules exactly as llama.cpp does ({@code :467-532});</li>
 *   <li>escapes: hex forms of 2, 4 and 8 digits (leading {@code x}, {@code u} or {@code U}, as
 *       llama.cpp writes them), control escapes for tab, CR and LF, and the specials backslash,
 *       double quote, brackets and dash ({@code parse_char}, {@code :162-184}); the source is a
 *       Java string, so a written UTF-8 character yields its code point directly, like llama.cpp's
 *       byte-wise {@code decode_utf8} over UTF-8 text;</li>
 *   <li>token references {@code <[id]>} and inverse {@code !<[id]>} ({@code :578-589}), and the
 *       name form {@code <token>} / {@code !<token>}, which resolves through a {@link GrammarVocab}
 *       to the one token its text tokenizes to; without a vocabulary the parser rejects the name
 *       form, as llama.cpp does with a null vocab ({@code :214-232}).</li>
 * </ul>
 *
 * <p>After parsing, the undefined-rule validation of {@code llama_grammar_parser::parse}
 * ({@code :700-718}) and the start-rule check of {@code llama_grammar_init_impl}
 * ({@code :1233-1237}) run; {@link Grammar} construction performs the left-recursion check
 * ({@code :963-1014}). Untrusted input (§3.6) is bounded by {@link GrammarLimits}; violations
 * throw {@link GrammarException} with the 1-based line and column of the offending grammar
 * text.</p>
 */
public final class GbnfParser {

    /** Sentinel for an unbounded repetition upper bound ({@code {m,}}, {@code *}, {@code +}). */
    private static final long NO_MAX_TIMES = Long.MAX_VALUE;

    private final String src;
    private final List<List<GrammarElement>> rules = new ArrayList<>();
    private final TreeMap<String, Integer> symbolIds = new TreeMap<>();
    /** Names in symbol-id order; the inverse of {@code symbolIds} for diagnostics. */
    private final List<String> symbolNames = new ArrayList<>();
    /** Rule id to the 1-based line/column of its source definition, for error positions. */
    private final Map<Integer, int[]> ruleLocations = new HashMap<>();
    /** First reference site of each symbol created by a reference, for undefined-rule errors. */
    private final Map<Integer, int[]> refLocations = new HashMap<>();
    private final GrammarVocab vocab;
    private long elementCount;

    private GbnfParser(String src, GrammarVocab vocab) {
        this.src = src;
        this.vocab = vocab;
    }

    /**
     * Parses a complete GBNF grammar into its validated rule set.
     *
     * @param source the grammar text; must not be null; at most
     *               {@link GrammarLimits#MAX_SOURCE_CHARS} characters
     * @throws IllegalArgumentException if {@code source} is null
     * @throws GrammarException on a syntax error, an undefined rule, a missing {@code root} rule,
     *                          left recursion or any {@link GrammarLimits} violation, at the
     *                          source position of the cause
     */
    public static Grammar parse(String source) {
        return parse(source, null);
    }

    /**
     * Parses a complete GBNF grammar whose {@code <token>} references resolve through {@code vocab}.
     *
     * @param source the grammar text, as for {@link #parse(String)}
     * @param vocab  the model vocabulary, or null to reject the {@code <token>} name form
     * @throws GrammarException as for {@link #parse(String)}, and for a {@code <token>} reference
     *                          that does not tokenize to exactly one token
     */
    public static Grammar parse(String source, GrammarVocab vocab) {
        if (source == null) {
            throw new IllegalArgumentException("grammar source must not be null");
        }
        if (source.length() > GrammarLimits.MAX_SOURCE_CHARS) {
            GbnfParser bounds = new GbnfParser(source, null);
            throw bounds.error("grammar source is " + source.length() + " characters, more than "
                    + GrammarLimits.MAX_SOURCE_CHARS, GrammarLimits.MAX_SOURCE_CHARS);
        }
        return new GbnfParser(source, vocab).parseAll();
    }

    // ------------------------------------------------------------------ top level

    /** {@code llama_grammar_parser::parse} followed by the init checks; the parse entry point. */
    private Grammar parseAll() {
        int pos = parseSpace(0, true);
        while (pos < src.length()) {
            pos = parseRule(pos);
        }
        validateUndefinedRules(pos);
        if (rules.isEmpty()) {
            // llama_grammar_init_impl refuses a grammar without rules before looking for its root
            throw error("failed to parse grammar: no rules defined", pos);
        }

        Integer rootId = symbolIds.get(Grammar.ROOT_RULE);
        if (rootId == null) {
            throw error("grammar does not contain a 'root' symbol", pos);
        }
        int[] lines = new int[rules.size()];
        int[] columns = new int[rules.size()];
        for (int i = 0; i < rules.size(); i++) {
            int[] at = ruleLocations.get(i);
            lines[i] = at == null ? 1 : at[0];
            columns[i] = at == null ? 1 : at[1];
        }
        return new Grammar(rules, symbolIds, symbolNames, rootId, lines, columns);
    }

    /**
     * Every {@code RULE_REF} must point at a filled rule slot; a reference into a never-defined
     * slot is an undefined rule ({@code llama-grammar.cpp:700-718}).
     */
    private void validateUndefinedRules(int endPos) {
        int nRules = rules.size();
        for (int i = 0; i < nRules; i++) {
            List<GrammarElement> rule = rules.get(i);
            if (rule == null) {
                throw error("Undefined rule identifier '" + symbolName(i) + "'", at(refLocations, i, endPos));
            }
            for (GrammarElement element : rule) {
                if (element.type() != GrammarElement.Type.RULE_REF) {
                    continue;
                }
                if (element.value() >= nRules || rules.get(element.value()) == null) {
                    throw error("Undefined rule identifier '" + symbolName(element.value())
                            + "', referenced by rule '" + symbolName(i) + "'",
                            at(refLocations, element.value(), endPos));
                }
            }
        }
    }

    // ------------------------------------------------------------------ rule parsing

    /** {@code llama_grammar_parser::parse_rule} ({@code llama-grammar.cpp:670-692}). */
    private int parseRule(int pos) {
        int nameStart = pos;
        int nameEnd = parseName(pos);
        int afterName = parseSpace(nameEnd, false);
        String name = src.substring(nameStart, nameEnd);
        int ruleId = acquireSymbolId(name, nameStart, false);
        if (!(peek(afterName) == ':' && peek(afterName + 1) == ':' && peek(afterName + 2) == '=')) {
            throw error("expecting '::='", afterName);
        }
        ruleLocations.put(ruleId, location(nameStart));
        pos = parseSpace(afterName + 3, true);
        pos = parseAlternates(pos, name, ruleId, false);
        if (peek(pos) == '\r') {
            pos += peek(pos + 1) == '\n' ? 2 : 1;
        } else if (peek(pos) == '\n') {
            pos++;
        } else if (pos < src.length()) {
            throw error("expecting newline or end", pos);
        }
        return parseSpace(pos, true);
    }

    /** {@code llama_grammar_parser::parse_alternates} ({@code llama-grammar.cpp:439-454}). */
    private int parseAlternates(int pos, String ruleName, int ruleId, boolean isNested) {
        List<GrammarElement> rule = new ArrayList<>();
        pos = parseSequence(pos, ruleName, rule, isNested, new Sequence());
        while (peek(pos) == '|') {
            addElement(rule, alt(), pos);
            pos = parseSpace(pos + 1, true);
            pos = parseSequence(pos, ruleName, rule, isNested, new Sequence());
        }
        addElement(rule, end(), pos);
        addRule(ruleId, rule, pos);
        return pos;
    }

    /**
     * Per-sequence capture state of {@code parse_sequence}: where the current symbol (the item a
     * repetition operator applies to) starts in the rule and how many rules the repetitions that
     * built it generated.
     */
    private static final class Sequence {
        int lastSymStart;
        long nPrevRules = 1;
    }

    /** {@code llama_grammar_parser::parse_sequence} ({@code llama-grammar.cpp:456-668}). */
    private int parseSequence(int pos, String ruleName, List<GrammarElement> rule, boolean isNested,
            Sequence seq) {
        seq.lastSymStart = rule.size();
        while (pos < src.length()) {
            char c = peek(pos);
            if (c == '"') {
                pos = parseLiteral(pos, rule, isNested, seq);
            } else if (c == '[') {
                pos = parseCharRange(pos, rule, isNested, seq);
            } else if (c == '<' || c == '!') {
                pos = parseTokenElement(pos, rule, isNested, seq);
            } else if (isWordChar(c)) {
                int nameEnd = parseName(pos);
                int refId = acquireSymbolId(src.substring(pos, nameEnd), pos, true);
                pos = parseSpace(nameEnd, isNested);
                seq.lastSymStart = rule.size();
                seq.nPrevRules = 1;
                addElement(rule, new GrammarElement(GrammarElement.Type.RULE_REF, refId), pos);
            } else if (c == '(') {
                pos = parseGroup(pos, rule, ruleName, isNested, seq);
            } else if (c == '.') {
                seq.lastSymStart = rule.size();
                seq.nPrevRules = 1;
                addElement(rule, new GrammarElement(GrammarElement.Type.CHAR_ANY, 0), pos);
                pos = parseSpace(pos + 1, isNested);
            } else if (c == '*') {
                pos = parseSpace(pos + 1, isNested);
                handleRepetitions(seq, rule, ruleName, 0, NO_MAX_TIMES, pos);
            } else if (c == '+') {
                pos = parseSpace(pos + 1, isNested);
                handleRepetitions(seq, rule, ruleName, 1, NO_MAX_TIMES, pos);
            } else if (c == '?') {
                pos = parseSpace(pos + 1, isNested);
                handleRepetitions(seq, rule, ruleName, 0, 1, pos);
            } else if (c == '{') {
                pos = parseRepetitionBounds(pos, rule, ruleName, isNested, seq);
            } else {
                break;
            }
        }
        return pos;
    }

    /** Literal string {@code "..."} ({@code llama-grammar.cpp:535-547}). */
    private int parseLiteral(int pos, List<GrammarElement> rule, boolean isNested, Sequence seq) {
        pos++;
        seq.lastSymStart = rule.size();
        seq.nPrevRules = 1;
        while (peek(pos) != '"') {
            if (pos >= src.length()) {
                throw error("unexpected end of input: unterminated string literal", pos);
            }
            Span span = parseChar(pos);
            addElement(rule, charOf((int) span.value), pos);
            pos = span.nextPos;
        }
        return parseSpace(pos + 1, isNested);
    }

    /** Character set {@code "[^chars]"} ({@code llama-grammar.cpp:548-577}). */
    private int parseCharRange(int pos, List<GrammarElement> rule, boolean isNested, Sequence seq) {
        pos++;
        GrammarElement.Type startType = GrammarElement.Type.CHAR;
        if (peek(pos) == '^') {
            pos++;
            startType = GrammarElement.Type.CHAR_NOT;
        }
        seq.lastSymStart = rule.size();
        seq.nPrevRules = 1;
        while (peek(pos) != ']') {
            if (pos >= src.length()) {
                throw error("unexpected end of input: unterminated character set", pos);
            }
            Span span = parseChar(pos);
            addElement(rule, new GrammarElement(
                    seq.lastSymStart < rule.size() ? GrammarElement.Type.CHAR_ALT : startType,
                    (int) span.value), pos);
            pos = span.nextPos;
            if (peek(pos) == '-' && peek(pos + 1) != ']') {
                if (pos + 1 >= src.length()) {
                    throw error("unexpected end of input: unterminated character range", pos);
                }
                Span endSpan = parseChar(pos + 1);
                pos = endSpan.nextPos;
                addElement(rule, new GrammarElement(GrammarElement.Type.CHAR_RNG_UPPER, (int) endSpan.value), pos);
            }
        }
        return parseSpace(pos + 1, isNested);
    }

    /** Token element {@code <[id]>} or {@code !<[id]>} ({@code llama-grammar.cpp:578-589}). */
    private int parseTokenElement(int pos, List<GrammarElement> rule, boolean isNested, Sequence seq) {
        GrammarElement.Type type = GrammarElement.Type.TOKEN;
        if (peek(pos) == '!') {
            type = GrammarElement.Type.TOKEN_NOT;
            pos++;
        }
        Span span = parseToken(pos);
        seq.lastSymStart = rule.size();
        seq.nPrevRules = 1;
        addElement(rule, new GrammarElement(type, (int) span.value), pos);
        return parseSpace(span.nextPos, isNested);
    }

    /** Grouping {@code (...)}; nested alternates live in a synthesized rule ({@code :597-610}). */
    private int parseGroup(int pos, List<GrammarElement> rule, String ruleName, boolean isNested,
            Sequence seq) {
        pos = parseSpace(pos + 1, true);
        int nRulesBefore = symbolIds.size();
        int subRuleId = generateSymbolId(ruleName, pos);
        pos = parseAlternates(pos, ruleName, subRuleId, true);
        seq.nPrevRules = Math.max(1, symbolIds.size() - nRulesBefore);
        seq.lastSymStart = rule.size();
        addElement(rule, new GrammarElement(GrammarElement.Type.RULE_REF, subRuleId), pos);
        if (peek(pos) != ')') {
            throw error("expecting ')'", pos);
        }
        return parseSpace(pos + 1, isNested);
    }

    /**
     * Repetition bounds {@code {m}, {m,}, {m,n}}; {@code {m}} above the expansion limit is
     * rejected, a larger {@code n} falls back to unbounded ({@code llama-grammar.cpp:625-662}).
     */
    private int parseRepetitionBounds(int pos, List<GrammarElement> rule, String ruleName,
            boolean isNested, Sequence seq) {
        pos = parseSpace(pos + 1, isNested);
        if (!isDigit(peek(pos))) {
            throw error("expecting an int", pos);
        }
        Span min = parseInt(pos);
        pos = parseSpace(min.nextPos, isNested);

        long maxTimes;
        if (peek(pos) == '}') {
            maxTimes = min.value;
            pos = parseSpace(pos + 1, isNested);
        } else if (peek(pos) == ',') {
            pos = parseSpace(pos + 1, isNested);
            maxTimes = NO_MAX_TIMES;
            if (isDigit(peek(pos))) {
                Span maxInt = parseInt(pos);
                maxTimes = maxInt.value;
                pos = parseSpace(maxInt.nextPos, isNested);
            }
            if (peek(pos) != '}') {
                throw error("expecting '}'", pos);
            }
            pos = parseSpace(pos + 1, isNested);
        } else {
            throw error("expecting ','", pos);
        }
        if (min.value > GrammarLimits.MAX_REPETITION_EXPANSION) {
            throw error("number of repetitions exceeds sane defaults, please reduce the number "
                    + "of repetitions", pos);
        }
        if (maxTimes != NO_MAX_TIMES && maxTimes > GrammarLimits.MAX_REPETITION_EXPANSION) {
            maxTimes = NO_MAX_TIMES;
        }
        handleRepetitions(seq, rule, ruleName, min.value, maxTimes, pos);
        return pos;
    }

    // ------------------------------------------------------------------ repetition expansion

    /**
     * The {@code handle_repetitions} rewrite ({@code llama-grammar.cpp:467-532}):
     * {@code S{m,n}} becomes {@code m} copies of {@code S} plus {@code n-m} synthesized
     * {@code <name>_<id>} rules of the form {@code S' ::= S S' | }, and the unbounded forms chain
     * one recursive rule ({@code S' ::= S S' | }).
     */
    private void handleRepetitions(Sequence seq, List<GrammarElement> rule, String ruleName,
            long min, long max, int pos) {
        boolean noMax = max == NO_MAX_TIMES;
        if (seq.lastSymStart == rule.size()) {
            throw error("expecting preceding item to */+/?/{", pos);
        }
        long totalRules = 1;
        if (!noMax && max > 0) {
            totalRules = max;
        } else if (min > 0) {
            totalRules = min;
        }

        if (seq.nPrevRules * totalRules > GrammarLimits.MAX_REPETITION_EXPANSION) {
            throw error("number of rules that are going to be repeated multiplied by the new "
                            + "repetition exceeds sane defaults, please reduce the number of "
                            + "repetitions or rule complexity", pos);
        }
        List<GrammarElement> prevRule = new ArrayList<>(rule.subList(seq.lastSymStart, rule.size()));
        elementCount += prevRule.size();
        checkElementLimit(pos);

        if (min == 0) {
            removeTail(rule, seq.lastSymStart);
        } else {
            // Repeat the previous elements (min - 1) additional times.
            for (long i = 1; i < min; i++) {
                rule.addAll(prevRule);
                elementCount += prevRule.size();
                checkElementLimit(pos);
            }
        }

        long lastRecRuleId = 0;
        long nOpt = noMax ? 1 : max - min;
        List<GrammarElement> recRule = new ArrayList<>(prevRule);
        for (long i = 0; i < nOpt; i++) {
            removeTail(recRule, prevRule.size());
            long recRuleId = generateSymbolId(ruleName, pos);
            if (i > 0 || noMax) {
                addElement(recRule, new GrammarElement(GrammarElement.Type.RULE_REF,
                        noMax ? (int) recRuleId : (int) lastRecRuleId), pos);
            }
            addElement(recRule, alt(), pos);
            addElement(recRule, end(), pos);
            addRule((int) recRuleId, recRule, pos);
            lastRecRuleId = recRuleId;
        }
        if (nOpt > 0) {
            addElement(rule, new GrammarElement(GrammarElement.Type.RULE_REF, (int) lastRecRuleId), pos);
        }
        seq.nPrevRules *= totalRules;
    }

    // ------------------------------------------------------------------ llama.cpp helpers

    /** {@code parse_space}: blanks and tabs, comments; newlines only when {@code newlineOk}. */
    private int parseSpace(int pos, boolean newlineOk) {
        while (pos < src.length()) {
            char c = peek(pos);
            if (c == '#') {
                while (pos < src.length() && peek(pos) != '\r' && peek(pos) != '\n') {
                    pos++;
                }
            } else if (c == ' ' || c == '\t' || (newlineOk && (c == '\r' || c == '\n'))) {
                pos++;
            } else {
                break;
            }
        }
        return pos;
    }

    /** Rule names are letters, digits and '-'; an empty name is an error ({@code :140-149}). */
    private int parseName(int pos) {
        int end = pos;
        while (end < src.length() && isWordChar(src.charAt(end))) {
            end++;
        }
        if (end == pos) {
            throw error("expecting name", pos);
        }
        return end;
    }

    /**
     * Digits only; the value saturates well above every bound llama.cpp compares it against
     * ({@code token ids} against {@code uint32}, repetition counts against the limit).
     */
    private Span parseInt(int pos) {
        int end = pos;
        long value = 0;
        while (end < src.length() && isDigit(src.charAt(end))) {
            value = value * 10 + (src.charAt(end) - '0');
            if (value > 1L << 40) {
                value = 1L << 40;
            }
            end++;
        }
        if (end == pos) {
            throw error("expecting integer", pos);
        }
        return new Span(value, end);
    }

    /** {@code parse_hex} ({@code llama-grammar.cpp:102-123}): exactly {@code size} hex digits. */
    private Span parseHex(int pos, int size) {
        int end = Math.min(pos + size, src.length());
        int value = 0;
        int scan = pos;
        for (; scan < end; scan++) {
            char c = src.charAt(scan);
            if ('a' <= c && c <= 'f') {
                value = (value << 4) + (c - 'a' + 10);
            } else if ('A' <= c && c <= 'F') {
                value = (value << 4) + (c - 'A' + 10);
            } else if (isDigit(c)) {
                value = (value << 4) + (c - '0');
            } else {
                break;
            }
        }
        if (scan != end) {
            throw error("expecting " + size + " hex chars", pos);
        }
        return new Span(value, end);
    }

    /**
     * {@code parse_char} ({@code llama-grammar.cpp:162-184}); the sequence is a Java string, so
     * the written character is its code point directly (a surrogate pair decodes as the code
     * point it encodes, like llama.cpp's byte decode of UTF-8 text).
     */
    private Span parseChar(int pos) {
        if (pos >= src.length()) {
            throw error("unexpected end of input", pos);
        }
        char c = src.charAt(pos);
        if (c == '\\') {
            if (pos + 1 >= src.length()) {
                throw error("unexpected end of input", pos);
            }
            char e = src.charAt(pos + 1);
            switch (e) {
                case 'x': return parseHex(pos + 2, 2);
                case 'u': return parseHex(pos + 2, 4);
                case 'U': return parseHex(pos + 2, 8);
                case 't': return new Span('\t', pos + 2);
                case 'r': return new Span('\r', pos + 2);
                case 'n': return new Span('\n', pos + 2);
                case '\\':
                case '"':
                case '[':
                case ']':
                case '-':
                    return new Span(e, pos + 2);
                default:
                    throw error("unknown escape", pos);
            }
        }
        int codePoint = src.codePointAt(pos);
        return new Span(codePoint, pos + Character.charCount(codePoint));
    }

    /**
     * {@code parse_token} ({@code llama-grammar.cpp:186-234}): {@code <[id]>} by id, or
     * {@code <token>} through the vocabulary, whose tokenization of the whole reference (angle
     * brackets included) must be exactly one token.
     */
    private Span parseToken(int pos) {
        int start = pos;
        if (peek(pos) != '<') {
            throw error("expecting '<'", pos);
        }
        pos++;
        if (peek(pos) == '[') {
            pos++;
            Span id = parseInt(pos);
            if (id.value > 0xFFFFFFFFL) {
                throw error("parsed token id is too big", pos);
            }
            pos = id.nextPos;
            if (peek(pos) != ']') {
                throw error("expecting ']'", pos);
            }
            pos++;
            if (peek(pos) != '>') {
                throw error("expecting '>'", pos);
            }
            pos++;
            return new Span(id.value, pos);
        }
        if (vocab == null) {
            throw error("no vocab to parse token", start);
        }
        while (pos < src.length() && peek(pos) != '>') {
            pos++;
        }
        if (pos >= src.length()) {
            throw error("expecting '>'", pos);
        }
        pos++;
        String text = src.substring(start, pos);
        int[] ids = vocab.tokenize(text);
        if (ids == null || ids.length != 1) {
            throw error("invalid token '" + text + "'", start);
        }
        return new Span(ids[0], pos);
    }

    // ------------------------------------------------------------------ symbols and rules

    private String symbolName(int id) {
        return symbolNames.get(id);
    }

    /** {@code get_symbol_id}; {@code isRef} marks a reference site for undefined-rule errors. */
    private int acquireSymbolId(String name, int pos, boolean isRef) {
        Integer existing = symbolIds.get(name);
        if (existing != null) {
            if (isRef) {
                refLocations.putIfAbsent(existing, location(pos));
            }
            return existing;
        }
        int nextId = symbolIds.size();
        if (nextId >= GrammarLimits.MAX_RULES) {
            throw error("grammar defines " + nextId + " rule names, more than "
                    + GrammarLimits.MAX_RULES, pos);
        }
        symbolIds.put(name, nextId);
        symbolNames.add(name);
        if (isRef) {
            refLocations.put(nextId, location(pos));
        }
        return nextId;
    }

    /** {@code generate_symbol_id}; synthesized names look like {@code <rule>_<id>}. */
    private int generateSymbolId(String baseName, int pos) {
        int nextId = symbolIds.size();
        if (nextId >= GrammarLimits.MAX_RULES) {
            throw error("grammar defines " + nextId + " rule names, more than "
                    + GrammarLimits.MAX_RULES, pos);
        }
        String name = baseName + '_' + nextId;
        symbolIds.put(name, nextId);
        symbolNames.add(name);
        return nextId;
    }

    /**
     * {@code add_rule} ({@code llama-grammar.cpp:432-437}): expands the rule list to
     * {@code ruleId} and stores the definition; padded slots of not-yet-defined rules stay null
     * until their own {@code add_rule} fills them.
     */
    private void addRule(int ruleId, List<GrammarElement> rule, int pos) {
        while (rules.size() <= ruleId) {
            rules.add(null);
        }
        rules.set(ruleId, new ArrayList<>(rule));
        ruleLocations.putIfAbsent(ruleId, location(pos));
    }

    private void addElement(List<GrammarElement> rule, GrammarElement element, int pos) {
        rule.add(element);
        elementCount++;
        checkElementLimit(pos);
    }

    private void checkElementLimit(int pos) {
        if (elementCount > GrammarLimits.MAX_ELEMENTS) {
            throw error("grammar has more than " + GrammarLimits.MAX_ELEMENTS + " elements", pos);
        }
    }

    private static void removeTail(List<GrammarElement> rule, int from) {
        if (from < rule.size()) {
            rule.subList(from, rule.size()).clear();
        }
    }

    private GrammarElement end() {
        return new GrammarElement(GrammarElement.Type.END, 0);
    }

    private GrammarElement alt() {
        return new GrammarElement(GrammarElement.Type.ALT, 0);
    }

    private GrammarElement charOf(int codePoint) {
        return new GrammarElement(GrammarElement.Type.CHAR, codePoint);
    }

    // ------------------------------------------------------------------ characters and errors

    private char peek(int pos) {
        return pos < src.length() ? src.charAt(pos) : '\0';
    }

    private static boolean isDigit(char c) {
        return '0' <= c && c <= '9';
    }

    /** {@code is_word_char} ({@code llama-grammar.cpp:98-100}): letters, digits and '-'. */
    private static boolean isWordChar(char c) {
        return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || isDigit(c);
    }

    /** Location of {@code id}, or the parse position {@code fallback} when none is known. */
    private int[] at(Map<Integer, int[]> locations, int id, int fallback) {
        int[] found = locations.get(id);
        if (found != null) {
            return found;
        }
        int[] defined = ruleLocations.get(id);
        return defined == null ? location(fallback) : defined;
    }

    /** 1-based line and column of {@code offset}; the column counts code points. */
    private int[] location(int offset) {
        int line = 1;
        int lineStart = 0;
        int bound = Math.min(offset, src.length());
        for (int i = 0; i < bound; i++) {
            if (src.charAt(i) == '\n') {
                line++;
                lineStart = i + 1;
            }
        }
        return new int[] {line, bound - lineStart + 1};
    }

    private GrammarException error(String message, int pos) {
        int[] at = location(pos);
        return new GrammarException(message, at[0], at[1]);
    }

    private GrammarException error(String message, int[] at) {
        return new GrammarException(message, at[0], at[1]);
    }

    // ------------------------------------------------------------------ span results

    /** Value and follow-up position of a parsed character, integer or token reference. */
    private static final class Span {
        final long value;
        final int nextPos;

        Span(long value, int nextPos) {
            this.value = value;
            this.nextPos = nextPos;
        }
    }
}
