package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * The regex to GBNF translator: a pattern becomes GBNF rule text in a {@link GbnfBuilder}. One
 * scan reads ECMAScript syntax (plus the Python {@code re} forms vLLM clients send) for six
 * uses, which differ in what the pattern matches and how a matched character is written.
 *
 * <p>A standalone output regex ({@link #compile}, the {@code guided_regex} /
 * {@code structured_outputs.regex} constraint) matches the whole raw output, as vLLM's
 * xgrammar backend compiles it ({@code compile_regex}), and reads ECMAScript syntax: literals
 * and every escape {@link RegexElement} decodes; the shorthands {@code \d \D \w \W \s \S}
 * inside and outside classes; character classes with ranges, negation and shorthands, built
 * as exact code-point sets ({@link RegexCharSet}); {@code .}; capturing, non-capturing and
 * named groups ({@code ( … ) (?: … ) (?<name> … ) (?P<name> … )}); alternation; the greedy and
 * lazy quantifiers {@code * + ? {m} {m,} {m,n} {,n}} (laziness does not change a full-match
 * language); a {@code '{'} that starts no quantifier and a lone {@code ']'} or {@code '}'} as
 * literals; and {@code ^} / {@code $} at the start / end of a top-level alternative (implied
 * elsewhere, so the pattern need not be anchored). An output regex never degrades: what has no
 * grammar equivalent (lookarounds, backreferences, word boundaries, inline flags, possessive
 * quantifiers, Unicode property escapes, a mid-pattern anchor) is refused by name. The regex of a Lark
 * terminal ({@link #larkRule}) reads and matches the same way, its {@code .} Python's (any code point but
 * LF, any code point under the {@code s} flag), its {@code i} flag matching both cases of every ASCII
 * letter.</p>
 *
 * <p>A JSON-schema value {@code pattern} ({@link #valueRule}) matches the whole decoded string,
 * the way xgrammar's JSON-schema converter (vLLM's default backend) compiles it: the anchors are
 * optional as for an output regex, so {@code [0-9]+} accepts {@code "123"} but not
 * {@code "a123"}. Its {@code .} is ECMAScript's (no line terminator), {@code []} and
 * {@code [^]} are ECMAScript's empty and full classes, and every matched character is written
 * JSON-encoded ({@link JsonStringCharacters}), so the value is always a valid JSON string; the
 * match is wrapped in the string's quotes. The value pattern of an XML parameter
 * ({@link #rawValueRule}) reads and matches the same way, its characters written raw and unquoted
 * between the parameter tags, as xgrammar's XML styles write them.</p>
 *
 * <p>A JSON object key pattern ({@link #keyRule}, {@code patternProperties} and
 * {@code propertyNames}) reads the same dialect with JSON schema's semantics: the pattern
 * is searched in the decoded key, so a top-level alternative without its {@code ^} or
 * {@code $} matches any run of key characters on that side, and every matched key character is
 * written JSON-encoded as for a value pattern. {@link #searches} runs the same scan to write a
 * key pattern as a {@code java.util.regex} and searches a known key with it.</p>
 *
 * <p>A structurally broken pattern (a stray operator, unbalanced parentheses, an unterminated
 * class or escape) is an {@link InvalidRegexException} in every use. Like the rest of the
 * grammar engine, the translation does not consult {@code java.util.regex}:
 * it is a single direct scan; only {@link #searches}, which matches a pattern against a
 * known text, evaluates one with it. One instance per pattern; the scan state ({@code pos},
 * {@code parenDepth}, {@code subRuleIds}, {@code groupNames}) is thread-confined to the thread
 * translating the pattern.</p>
 */
public final class RegexToGbnf {

    /**
     * Deepest group nesting accepted in every use, llama.cpp's {@code COMMON_RECURSIVE_LIMIT}
     * ({@code common/common.h}), the bound its schema-pattern converter applies.
     */
    public static final int MAX_PATTERN_DEPTH = GrammarLimits.MAX_NESTING_DEPTH;

    /** The {@code .} of the {@code dotAll} mode: any code point. */
    private static final String DOT_ALL = "[\\U00000000-\\U0010FFFF]";

    /** llama.cpp's {@code .} ({@code get_dot}): any code point but CR and LF. */
    private static final String DOT_NO_CR_LF = "[^\\x0A\\x0D]";

    /** Python {@code re}'s {@code .}: any code point but LF. */
    private static final String PYTHON_DOT = "[^\\x0A]";

    /**
     * The pattern is a valid regex with no GBNF equivalent; the message names the construct.
     * Unchecked so it can cross the translator's recursion freely, as llama.cpp does.
     */
    public static final class UnsupportedRegexException extends IllegalArgumentException {
        UnsupportedRegexException(String construct) {
            super(construct);
        }
    }

    /**
     * The pattern is not a valid regex; the message names the problem, matching llama.cpp's
     * {@code invalid_pattern} (an error, never a graceful fallback).
     */
    public static final class InvalidRegexException extends IllegalArgumentException {
        InvalidRegexException(String problem) {
            super(problem);
        }
    }

    /** What a translation matches and the text it writes. */
    private enum Mode {
        /** A standalone output regex: matched against the whole raw output. */
        OUTPUT_REGEX,
        /** A Lark terminal regex: matched whole, Python's {@code .} (any code point but LF). */
        LARK_TERMINAL,
        /** A JSON-schema value pattern: matched against the whole decoded string, written JSON-encoded. */
        JSON_VALUE,
        /** A JSON-schema value pattern of an XML parameter: matched against the whole raw value, written raw. */
        RAW_VALUE,
        /** A JSON object key pattern: searched in the decoded key, written JSON-encoded. */
        JSON_KEY,
        /** A JSON object key pattern written as the {@code java.util.regex} {@link #searches} runs. */
        KEY_SEARCH;

        /** Whether the mode reads a JSON key pattern (top-level anchors kept for the search). */
        boolean key() {
            return this == JSON_KEY || this == KEY_SEARCH;
        }

        /** Whether the pattern is a JSON schema's (ECMAScript only: {@code []} is the empty class). */
        boolean schema() {
            return this != OUTPUT_REGEX && this != LARK_TERMINAL;
        }

        /** Whether matched characters are written JSON-encoded inside a JSON string. */
        boolean jsonEncoded() {
            return this == JSON_VALUE || this == JSON_KEY;
        }
    }

    /**
     * Translates the JSON-schema value {@code pattern} and adds its rule under {@code name} in
     * {@code builder}: the JSON strings whose whole decoded text {@code pattern} matches, xgrammar's
     * full-match semantics (the anchors are optional), every matched character written
     * JSON-encoded ({@link JsonStringCharacters}).
     *
     * @param pattern the ECMAScript regex
     * @param name    the requested rule name (sanitized by the builder)
     * @param builder receives the rule and helper rules such as the {@code .} rule
     * @return the stored rule name
     * @throws UnsupportedRegexException naming the construct with no grammar equivalent
     * @throws InvalidRegexException     naming the structural problem
     */
    static String valueRule(String pattern, String name, GbnfBuilder builder) {
        return new RegexToGbnf(pattern, name, builder, false, Mode.JSON_VALUE).translate();
    }

    /**
     * Translates the JSON-schema value {@code pattern} of an XML parameter ({@link XmlParameterRules}) and adds its
     * rule under {@code name}: the raw texts {@code pattern} matches whole, read as {@link #valueRule} reads it, every
     * matched character written as is between the parameter tags.
     *
     * @return the stored rule name
     * @throws UnsupportedRegexException naming the construct with no grammar equivalent
     * @throws InvalidRegexException     naming the structural problem
     */
    static String rawValueRule(String pattern, String name, GbnfBuilder builder) {
        return new RegexToGbnf(pattern, name, builder, false, Mode.RAW_VALUE).translate();
    }

    /**
     * Translates the regex of a Lark terminal ({@code /pattern/}, {@link LarkGrammar}) and adds its rule under
     * {@code name}: the texts the whole regex matches, read as an output regex is read, with Python's {@code .} (any
     * code point but LF), as xgrammar's Lark front end compiles it.
     *
     * @param dotAll          the {@code s} flag: a {@code .} accepts any code point
     * @param caseInsensitive the {@code i} flag: a letter or class holding an ASCII letter matches both its cases
     * @return the stored rule name
     * @throws UnsupportedRegexException naming the construct with no grammar equivalent
     * @throws InvalidRegexException     naming the structural problem
     */
    static String larkRule(String pattern, String name, GbnfBuilder builder, boolean dotAll,
            boolean caseInsensitive) {
        return new RegexToGbnf(pattern, name, builder, dotAll ? DOT_ALL : PYTHON_DOT, Mode.LARK_TERMINAL,
                caseInsensitive).translate();
    }

    /**
     * Translates the output regex {@code pattern} into its own builder and parses the result
     * with {@link GbnfParser}: the grammar accepting exactly the outputs the whole regex
     * matches.
     *
     * @param pattern the output regex in ECMAScript syntax; {@code ^} and {@code $} are optional
     * @param name    the rule name the raw-text rule is stored under
     * @param dotAll  when {@code true} a {@code .} accepts any code point, otherwise any code
     *                point except CR and LF
     * @return the grammar whose raw-text rule is stored under {@code name}
     * @throws UnsupportedRegexException naming the construct with no grammar equivalent
     * @throws InvalidRegexException     naming the structural problem
     */
    public static Grammar compile(String pattern, String name, boolean dotAll) {
        if (pattern == null) {
            throw new InvalidRegexException("the pattern is missing");
        }
        GbnfBuilder builder = new GbnfBuilder();
        new RegexToGbnf(pattern, name, builder, dotAll, Mode.OUTPUT_REGEX).translate();
        return builder.grammar();
    }

    /**
     * {@link #compile} with Python {@code re}'s {@code .}, the dialect of a Lark regular expression
     * ({@link LarkGrammar}): without {@code dotAll} a {@code .} accepts any code point but LF.
     *
     * @throws UnsupportedRegexException naming the construct with no grammar equivalent
     * @throws InvalidRegexException     naming the structural problem
     */
    static Grammar compilePython(String pattern, String name, boolean dotAll) {
        GbnfBuilder builder = new GbnfBuilder();
        new RegexToGbnf(pattern, name, builder, dotAll ? DOT_ALL : PYTHON_DOT, Mode.OUTPUT_REGEX).translate();
        return builder.grammar();
    }

    /**
     * Translates the JSON object key pattern {@code pattern} and adds its rule under
     * {@code name}: the JSON strings whose decoded text {@code pattern} matches somewhere, the
     * unanchored search of JSON schema's {@code patternProperties} and {@code propertyNames}.
     * A top-level alternative without its {@code ^} (or {@code $}) matches any run of key
     * characters before (or after) it; the grammar writes every matched character JSON-encoded
     * ({@link JsonStringCharacters}).
     *
     * @return the stored rule name
     * @throws UnsupportedRegexException naming the construct with no grammar equivalent, among
     *                                   them a class holding no character a key is written with
     * @throws InvalidRegexException     naming the structural problem
     */
    static String keyRule(String pattern, String name, GbnfBuilder builder) {
        return new RegexToGbnf(pattern, name, builder, false, Mode.JSON_KEY).translate();
    }

    /**
     * Whether the JSON key pattern {@code pattern} matches somewhere in {@code text}, the
     * ECMAScript search {@link #keyRule} translates: the same scan writes the pattern as a
     * {@code java.util.regex} (classes as exact code-point sets, {@code .} without line
     * terminators, {@code $} at the very end only), which then searches {@code text}.
     *
     * @throws UnsupportedRegexException naming the construct {@link #keyRule} refuses too
     * @throws InvalidRegexException     naming the structural problem
     */
    static boolean searches(String pattern, String text) {
        String regex = new RegexToGbnf(pattern, "search", new GbnfBuilder(), false,
                Mode.KEY_SEARCH).scan();
        return Pattern.compile(regex).matcher(text).find();
    }

    /**
     * Whether a match of the output regex {@code pattern} ({@link #compile}, {@code .} matching any code point) may
     * hold one of the code points {@code chars} (vLLM {@code response_template_grammar._may_write}): whether a
     * character set its compiled grammar reaches from the root admits one. True for a pattern that does not compile.
     */
    public static boolean mayWrite(String pattern, Set<Integer> chars) {
        Grammar grammar;
        try {
            grammar = compile(pattern, Grammar.ROOT_RULE, true);
        } catch (RuntimeException e) {
            return true;
        }
        List<Integer> pending = new ArrayList<>();
        Set<Integer> seen = new HashSet<>();
        pending.add(grammar.rootRuleIndex());
        while (!pending.isEmpty()) {
            int r = pending.remove(pending.size() - 1);
            if (seen.add(r) && ruleMayWrite(grammar.rule(r), chars, pending)) {
                return true;
            }
        }
        return false;
    }

    /** Whether a character set of {@code rule} admits one of {@code chars}; queues the rules it references. */
    private static boolean ruleMayWrite(List<GrammarElement> rule, Set<Integer> chars, List<Integer> pending) {
        for (int i = 0; i < rule.size(); i++) {
            GrammarElement.Type t = rule.get(i).type();
            if (t == GrammarElement.Type.RULE_REF) {
                pending.add(rule.get(i).value());
            } else if (t == GrammarElement.Type.CHAR_ANY) {
                if (!chars.isEmpty()) {
                    return true;
                }
            } else if (t == GrammarElement.Type.CHAR || t == GrammarElement.Type.CHAR_NOT) {
                int end = i + 1;
                while (end < rule.size() && (rule.get(end).type() == GrammarElement.Type.CHAR_ALT
                        || rule.get(end).type() == GrammarElement.Type.CHAR_RNG_UPPER)) {
                    end++;
                }
                for (int c : chars) {
                    if (setAdmits(rule, i, end, c)) {
                        return true;
                    }
                }
                i = end - 1;
            }
        }
        return false;
    }

    /** Whether the character set {@code rule[from, to)} admits {@code c}. */
    private static boolean setAdmits(List<GrammarElement> rule, int from, int to, int c) {
        boolean in = false;
        for (int k = from; k < to; k++) {
            int lo = rule.get(k).value();
            int hi = k + 1 < to && rule.get(k + 1).type() == GrammarElement.Type.CHAR_RNG_UPPER
                    ? rule.get(++k).value() : lo;
            in |= c >= lo && c <= hi;
        }
        return rule.get(from).type() == GrammarElement.Type.CHAR_NOT != in;
    }

    private final String sub;
    private final String name;
    private final GbnfBuilder builder;
    /** The GBNF character set a {@code .} matches. */
    private final String dotSet;
    private final Mode mode;
    /** Whether ASCII letters match both cases (a Lark regex's {@code i} flag). */
    private final boolean caseInsensitive;
    /** Helper rule text to stored name, llama.cpp {@code sub_rule_ids}. */
    private final Map<String, String> subRuleIds = new HashMap<>();
    /** The named groups seen so far; a repeated name is an invalid pattern. */
    private final Set<String> groupNames = new HashSet<>();
    private int pos;
    private int parenDepth;

    private RegexToGbnf(String sub, String name, GbnfBuilder builder, boolean dotAll, Mode mode) {
        this(sub, name, builder, mode == Mode.LARK_TERMINAL ? (dotAll ? DOT_ALL : PYTHON_DOT)
                : dotAll ? DOT_ALL : DOT_NO_CR_LF, mode, false);
    }

    private RegexToGbnf(String sub, String name, GbnfBuilder builder, String dotSet, Mode mode) {
        this(sub, name, builder, dotSet, mode, false);
    }

    private RegexToGbnf(String sub, String name, GbnfBuilder builder, String dotSet, Mode mode,
            boolean caseInsensitive) {
        this.sub = sub;
        this.name = name;
        this.builder = builder;
        this.dotSet = dotSet;
        this.mode = mode;
        this.caseInsensitive = caseInsensitive;
    }

    /** Scans the pattern and stores its rule under {@link #name}; returns the stored name. */
    private String translate() {
        String rule = scan();
        switch (mode) {
            case JSON_VALUE:
                return builder.addRule(name, "\"\\\"\" (" + rule + ") \"\\\"\"");
            case JSON_KEY:
                return builder.addRule(name, "\"\\\"\" " + rule + " \"\\\"\"");
            default:
                return builder.addRule(name, "(" + rule + ")");
        }
    }

    /** The text of the whole pattern; every group must be closed. */
    private String scan() {
        String text = transform();
        if (parenDepth != 0) {
            throw new InvalidRegexException("unbalanced parentheses");
        }
        return text;
    }

    /**
     * One alternation-or-sequence group of the scan (llama.cpp's {@code transform} lambda).
     * Consumes the closing parenthesis of its group before returning {@code join_seq}'s
     * joined text; a group never closed before the pattern end leaves the depth counter up,
     * which {@link #scan} turns into an invalid-pattern error. At the pattern end a JSON key
     * pattern joins its top-level alternatives with their search sides ({@link #keyAlternatives}).
     */
    private String transform() {
        List<Item> seq = new ArrayList<>();
        while (pos < sub.length()) {
            if (sub.charAt(pos) == ')') {
                pos++;
                if (parenDepth == 0) {
                    throw new InvalidRegexException("unbalanced parentheses");
                }
                parenDepth--;
                return joinSeq(seq);
            }
            scanItem(seq);
        }
        return mode == Mode.JSON_KEY && parenDepth == 0 ? keyAlternatives(seq) : joinSeq(seq);
    }

    /** Scans the item at {@code pos} into {@code seq}, dispatching on its first character. */
    private void scanItem(List<Item> seq) {
        char c = sub.charAt(pos);
        switch (c) {
            case '.':
                seq.add(Item.atom(dotText()));
                pos++;
                break;
            case '(':
                String body = group();
                seq.add(Item.atom(mode == Mode.KEY_SEARCH ? "(?:" + body + ")" : "(" + body + ")"));
                break;
            case '^':
            case '$':
                anchor(seq, c);
                break;
            case '[':
                seq.add(Item.atom(classSet()));
                break;
            case '|':
                seq.add(Item.ALTERNATION);
                pos++;
                break;
            case '*':
            case '+':
            case '?':
                pos++;
                quantify(seq, c);
                break;
            case '{':
                scanBrace(seq);
                break;
            default:
                fullAtom(seq);
        }
    }

    /** The text of a {@code .}: the dot rule of an output regex, or ECMAScript's for a JSON schema pattern. */
    private String dotText() {
        switch (mode) {
            case JSON_VALUE:
            case JSON_KEY:
                return builder.addRule("json-string-dot", JsonStringCharacters.dot());
            case RAW_VALUE:
                return builder.addRule("raw-string-dot", RegexCharSet.NON_LINE_TERMINATOR.complement().toGbnf(true));
            case LARK_TERMINAL:
                return dotRuleText();
            case KEY_SEARCH:
                return RegexCharSet.NON_LINE_TERMINATOR.toJavaRegex();
            default:
                return dotRuleText();
        }
    }

    private String dotRuleText() {
        return builder.addRule("dot", dotSet);
    }

    // ------------------------------------------------------------------ groups and anchors

    /** A group from its {@code '('}: reads the group prefix, then scans the body. */
    private String group() {
        pos++;
        if (pos < sub.length() && sub.charAt(pos) == '?') {
            fullGroupPrefix();
        }
        parenDepth++;
        if (parenDepth > MAX_PATTERN_DEPTH) {
            throw new UnsupportedRegexException("pattern nesting too deep");
        }
        return transform();
    }

    /**
     * The {@code (?…} prefix of a group at {@code pos}: a non-capturing or named
     * group is consumed (a group only groups here), every other form refuses by name.
     */
    private void fullGroupPrefix() {
        char kind = charAt(pos + 1);
        char next = charAt(pos + 2);
        if (kind == ':') {
            pos += 2;
        } else if (kind == '<' && next != '=' && next != '!') {
            pos = groupNameEnd(pos + 2);
        } else if (kind == 'P' && next == '<') {
            pos = groupNameEnd(pos + 3);
        } else if (kind == '=' || kind == '!') {
            throw new UnsupportedRegexException("lookahead assertion: (?" + kind);
        } else if (kind == '<') {
            throw new UnsupportedRegexException("lookbehind assertion: (?<" + next);
        } else if (kind == 'P' && next == '=') {
            throw new UnsupportedRegexException("named backreference: (?P=");
        } else if (kind == '>') {
            throw new UnsupportedRegexException("atomic group: (?>");
        } else if (kind == '#') {
            throw new UnsupportedRegexException("comment group: (?#");
        } else if (Character.isLetter(kind) || kind == '-') {
            throw new UnsupportedRegexException("inline flag group: (?" + kind);
        } else if (kind == 0) {
            throw new InvalidRegexException("incomplete group syntax: (?");
        } else {
            throw new UnsupportedRegexException("unsupported group syntax: (?" + kind);
        }
    }

    /** Reads a group name ending in {@code '>'} from {@code from}; returns the index past it. */
    private int groupNameEnd(int from) {
        int close = sub.indexOf('>', from);
        if (close < 0) {
            throw new InvalidRegexException("unterminated group name");
        }
        String groupName = sub.substring(from, close);
        boolean valid = !groupName.isEmpty() && !Character.isDigit(groupName.charAt(0));
        for (int i = 0; i < groupName.length() && valid; i++) {
            char c = groupName.charAt(i);
            valid = Character.isLetterOrDigit(c) || c == '_' || c == '$';
        }
        if (!valid) {
            throw new InvalidRegexException("invalid group name: " + groupName);
        }
        if (!groupNames.add(groupName)) {
            throw new InvalidRegexException("duplicate group name: " + groupName);
        }
        return close + 1;
    }

    /**
     * An anchor: {@code ^} at the start and {@code $} at the end of a top-level alternative,
     * implied by a full match and kept as an anchor item by a key pattern's search; refused
     * anywhere else.
     */
    private void anchor(List<Item> seq, char c) {
        Item last = seq.isEmpty() ? null : seq.get(seq.size() - 1);
        boolean implied = parenDepth == 0 && (c == '^'
                ? last == null || last.alternation || last.anchor == '^'
                : pos + 1 == sub.length() || sub.charAt(pos + 1) == '|');
        if (!implied) {
            throw new UnsupportedRegexException("anchor inside the pattern");
        }
        pos++;
        if (mode.key()) {
            seq.add(Item.anchor(c, mode == Mode.KEY_SEARCH ? (c == '^' ? "^" : "\\z") : ""));
        }
    }

    /**
     * The JSON key body of the top-level {@code seq}: each top-level alternative searched, a
     * side without its anchor matching any run of key characters, the alternatives joined.
     */
    private String keyAlternatives(List<Item> seq) {
        List<String> alternatives = new ArrayList<>();
        int from = 0;
        for (int i = 0; i <= seq.size(); i++) {
            if (i == seq.size() || seq.get(i).alternation) {
                alternatives.add(keyAlternative(seq.subList(from, i)));
                from = i + 1;
            }
        }
        return alternatives.size() == 1 ? alternatives.get(0)
                : "(" + String.join(" | ", alternatives) + ")";
    }

    /** One searched top-level alternative of a JSON key pattern; {@code ""} for the empty key. */
    private String keyAlternative(List<Item> items) {
        boolean start = false;
        boolean end = false;
        List<Item> body = new ArrayList<>();
        for (Item item : items) {
            start |= item.anchor == '^';
            end |= item.anchor == '$';
            if (item.anchor == 0) {
                body.add(item);
            }
        }
        String core = joinSeq(body);
        List<String> parts = new ArrayList<>();
        if (!start) {
            parts.add(keyRun());
        }
        if (!core.isEmpty()) {
            parts.add("(" + core + ")");
        }
        if (!end && (start || !core.isEmpty())) {
            parts.add(keyRun());
        }
        return parts.isEmpty() ? "\"\"" : String.join(" ", parts);
    }

    /** Any run of key characters, the unanchored side of a key search. */
    private String keyRun() {
        return builder.addRule("json-string-char", JsonStringCharacters.any()) + "*";
    }

    // ------------------------------------------------------------------ quantifiers

    /** {@code * + ?} on the last item, then its lazy/possessive suffix. */
    private void quantify(List<Item> seq, char operator) {
        Item item = repeatable(seq);
        seq.set(seq.size() - 1, Item.quantified(toRule(item) + operator));
        quantifierSuffix();
    }

    /** The last item of {@code seq} when a quantifier may follow it. */
    private static Item repeatable(List<Item> seq) {
        Item item = seq.isEmpty() ? null : seq.get(seq.size() - 1);
        if (item == null || item.alternation || item.quantified || item.anchor != 0) {
            throw new InvalidRegexException("nothing to repeat");
        }
        return item;
    }

    /**
     * After a quantifier: a lazy {@code ?} is consumed (a full match or a search
     * accepts the same strings either way), a possessive {@code +} refuses.
     */
    private void quantifierSuffix() {
        char next = charAt(pos);
        if (next == '?') {
            pos++;
        } else if (next == '+') {
            throw new UnsupportedRegexException("possessive quantifier");
        }
    }

    /**
     * A {@code '{'}: the bounds of a quantifier, or a literal when it starts none (as
     * ECMAScript Annex B and Python read it).
     */
    private void scanBrace(List<Item> seq) {
        int[] bounds = braceBounds(pos);
        if (bounds == null) {
            fullAtom(seq);
            return;
        }
        Item item = repeatable(seq);
        pos = bounds[2];
        seq.set(seq.size() - 1, Item.quantified(repeated(item, bounds[0], bounds[1])));
        quantifierSuffix();
    }

    /**
     * The repetition text of {@code item}: in GBNF a non-literal item is first hoisted into a
     * helper rule named {@code <name>-<n>} (llama.cpp {@code :588-605}); a key search writes
     * the Java bounds {@code {m,n}} on the atom.
     */
    private String repeated(Item item, int minTimes, int maxTimes) {
        if (mode == Mode.KEY_SEARCH) {
            return item.text + "{" + minTimes + ","
                    + (maxTimes == Integer.MAX_VALUE ? "" : String.valueOf(maxTimes)) + "}";
        }
        String subId = item.li ? null : subRuleIds.get(item.text);
        if (!item.li && subId == null) {
            // llama.cpp inserts an empty id first, so the first helper lands on <name>-1
            subId = builder.addRule(name + "-" + (subRuleIds.size() + 1), item.text);
            subRuleIds.put(item.text, subId);
        }
        return GbnfBuilder.repetition(item.li ? "\"" + item.text + "\"" : subId,
                minTimes, maxTimes, "");
    }

    /**
     * The bounds of the quantifier {@code {m} {m,} {m,n} {,n}} at {@code from}:
     * {@code {min, max, end}} with {@code max} Integer.MAX_VALUE when unbounded and
     * {@code end} the index past {@code '}'}, or null when no quantifier starts there.
     */
    private int[] braceBounds(int from) {
        int close = sub.indexOf('}', from + 1);
        if (charAt(from) != '{' || close < 0) {
            return null;
        }
        String body = sub.substring(from + 1, close);
        int comma = body.indexOf(',');
        String low = comma < 0 ? body : body.substring(0, comma);
        String high = comma < 0 ? body : body.substring(comma + 1);
        boolean lowOk = isDigits(low) || (comma >= 0 && low.isEmpty());
        boolean highOk = isDigits(high) || (comma >= 0 && high.isEmpty());
        if (!lowOk || !highOk || (low.isEmpty() && high.isEmpty())) {
            return null;
        }
        int minTimes = low.isEmpty() ? 0 : bound(low);
        int maxTimes = high.isEmpty() ? Integer.MAX_VALUE : bound(high);
        if (minTimes > maxTimes) {
            throw new InvalidRegexException("numbers out of order in curly brackets: {" + body + "}");
        }
        return new int[]{minTimes, maxTimes, close + 1};
    }

    private static boolean isDigits(String text) {
        if (text.isEmpty()) {
            return false;
        }
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) < '0' || text.charAt(i) > '9') {
                return false;
            }
        }
        return true;
    }

    private static int bound(String digits) {
        try {
            return Integer.parseInt(digits);
        } catch (NumberFormatException e) {
            throw new UnsupportedRegexException("repetition bound too large: " + digits);
        }
    }

    // ------------------------------------------------------------------ atoms

    /**
     * The item at {@code pos} that is no structural character: a class shorthand
     * as its character set ({@code \D} as {@code [^0-9]}), else a literal run.
     */
    private void fullAtom(List<Item> seq) {
        RegexElement first = RegexElement.at(sub, pos, false);
        if (first.isShorthand()) {
            char letter = first.shorthand;
            int from = pos;
            pos += first.length;
            seq.add(Item.atom(setText(folded(RegexCharSet.shorthand(Character.toLowerCase(letter))),
                    Character.isUpperCase(letter), from)));
            return;
        }
        if (isFolded(first.codePoint)) {
            int from = pos;
            pos += first.length;
            seq.add(Item.atom(setText(folded(RegexCharSet.single(first.codePoint)), false, from)));
            return;
        }
        fullLiteral(seq);
    }

    /**
     * A run of literal code points, plain or escaped, written as one literal. The run stops
     * before a metacharacter or shorthand, and leaves a code point a quantifier follows to an
     * item of its own; its first element is always taken ({@link #fullAtom} has ruled out a
     * shorthand there).
     */
    private void fullLiteral(List<Item> seq) {
        StringBuilder out = new StringBuilder();
        while (pos < sub.length()) {
            if (out.length() > 0 && isFullMeta(pos)) {
                break;
            }
            RegexElement element = RegexElement.at(sub, pos, false);
            int next = pos + element.length;
            if (out.length() > 0 && (element.isShorthand() || isQuantifier(next) || isFolded(element.codePoint))) {
                break;
            }
            out.append(literalText(element.codePoint));
            pos = next;
        }
        seq.add(Item.literal(out.toString()));
    }

    /** One literal code point: in a GBNF literal, JSON-encoded for a JSON schema pattern, or as Java regex. */
    private String literalText(int codePoint) {
        switch (mode) {
            case JSON_VALUE:
            case JSON_KEY:
                return JsonStringCharacters.literal(codePoint);
            case KEY_SEARCH:
                return RegexCharSet.javaChar(codePoint);
            default:
                return RegexCharSet.literalChar(codePoint);
        }
    }

    private boolean isFullMeta(int at) {
        char c = sub.charAt(at);
        return "|.()[^$".indexOf(c) >= 0 || isQuantifier(at);
    }

    private boolean isQuantifier(int at) {
        char c = charAt(at);
        return c == '*' || c == '+' || c == '?' || (c == '{' && braceBounds(at) != null);
    }

    /**
     * A character class from its {@code '['}: members, ranges and shorthands united into one
     * {@link RegexCharSet}, negated by a leading {@code ^}. A JSON schema pattern reads
     * ECMAScript's {@code []} (no character) and {@code [^]} (any character); an output regex
     * refuses a leading {@code ']'}, which Python reads as a literal.
     */
    private String classSet() {
        int open = pos;
        pos++;
        boolean negated = charAt(pos) == '^';
        if (negated) {
            pos++;
        }
        if (charAt(pos) == ']' && !mode.schema()) {
            throw new UnsupportedRegexException("']' first in a character class (an empty class in"
                    + " ECMAScript, a literal in Python); write it as \\]");
        }
        RegexCharSet set = RegexCharSet.EMPTY;
        while (pos < sub.length() && sub.charAt(pos) != ']') {
            set = set.union(classMember());
        }
        if (pos >= sub.length()) {
            throw new InvalidRegexException("unterminated character class");
        }
        pos++;
        return setText(folded(set), negated, open);
    }

    /** {@code set} with both cases of its ASCII letters under the {@code i} flag, else {@code set}. */
    private RegexCharSet folded(RegexCharSet set) {
        return caseInsensitive ? set.asciiCaseClosure() : set;
    }

    /** Whether the literal {@code codePoint} is an ASCII letter the {@code i} flag writes as a class of its cases. */
    private boolean isFolded(int codePoint) {
        return caseInsensitive && codePoint < 128 && Character.isLetter(codePoint);
    }

    /**
     * The text of the code points {@code set} (its complement when {@code negated}) read from
     * {@code from} to {@code pos}: a GBNF character set, JSON-encoded string characters, or a
     * Java class.
     *
     * @throws UnsupportedRegexException when a JSON schema pattern's class matches no character
     */
    private String setText(RegexCharSet set, boolean negated, int from) {
        switch (mode) {
            case JSON_VALUE:
            case JSON_KEY:
                String text = JsonStringCharacters.set(negated ? set.complement() : set);
                if (text == null) {
                    throw new UnsupportedRegexException("character class " + sub.substring(from, pos)
                            + " matches no character");
                }
                return text;
            case RAW_VALUE: {
                RegexCharSet members = negated ? set.complement() : set;
                if (members.isEmpty()) {
                    throw new UnsupportedRegexException("character class " + sub.substring(from, pos)
                            + " matches no character");
                }
                return set.isEmpty() ? members.toGbnf(false) : set.toGbnf(negated);
            }
            case KEY_SEARCH:
                return (negated ? set.complement() : set).toJavaRegex();
            default:
                return set.toGbnf(negated);
        }
    }

    /** One class member at {@code pos}: a code point, a shorthand or a range {@code a-z}. */
    private RegexCharSet classMember() {
        RegexElement low = RegexElement.at(sub, pos, true);
        pos += low.length;
        if (charAt(pos) != '-' || pos + 1 >= sub.length() || sub.charAt(pos + 1) == ']') {
            return low.toSet();
        }
        pos++;
        RegexElement high = RegexElement.at(sub, pos, true);
        pos += high.length;
        if (low.isShorthand() || high.isShorthand()) {
            throw new InvalidRegexException("character range with a class shorthand");
        }
        if (high.codePoint < low.codePoint) {
            throw new InvalidRegexException("character range out of order");
        }
        return RegexCharSet.range(low.codePoint, high.codePoint);
    }

    // ------------------------------------------------------------------ sequence joining

    /**
     * Joins the sequence, merging consecutive literal items (llama.cpp {@code join_seq},
     * {@code :460-491}); a key search concatenates its Java regex items.
     */
    private String joinSeq(List<Item> seq) {
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
        String separator = mode == Mode.KEY_SEARCH ? "" : " ";
        StringBuilder out = new StringBuilder();
        for (Item item : ret) {
            if (out.length() > 0) {
                out.append(separator);
            }
            out.append(toRule(item));
        }
        return out.toString();
    }

    private static void flushLiteral(List<Item> ret, StringBuilder literal) {
        if (literal.length() > 0) {
            ret.add(Item.literal(literal.toString()));
            literal.setLength(0);
        }
    }

    /** The item's text: a literal run quoted as a GBNF literal, a key search's regex as is. */
    private String toRule(Item item) {
        return item.li && mode != Mode.KEY_SEARCH ? "\"" + item.text + "\"" : item.text;
    }

    // ------------------------------------------------------------------ scan infrastructure

    /** The character at {@code at}, or 0 past the pattern end. */
    private char charAt(int at) {
        return at < sub.length() ? sub.charAt(at) : 0;
    }

    /**
     * One sequence item: {@code li} marks a literal run (GBNF literal body text),
     * {@code quantified} a item that already carries its quantifier,
     * {@code alternation} the {@code |} separator and {@code anchor} ({@code ^} or {@code $},
     * 0 otherwise) a key pattern's top-level anchor.
     */
    private static final class Item {
        static final Item ALTERNATION = new Item("|", false, false, true, (char) 0);

        final String text;
        final boolean li;
        final boolean quantified;
        final boolean alternation;
        final char anchor;

        private Item(String text, boolean li, boolean quantified, boolean alternation, char anchor) {
            this.text = text;
            this.li = li;
            this.quantified = quantified;
            this.alternation = alternation;
            this.anchor = anchor;
        }

        static Item literal(String text) {
            return new Item(text, true, false, false, (char) 0);
        }

        static Item atom(String text) {
            return new Item(text, false, false, false, (char) 0);
        }

        static Item quantified(String text) {
            return new Item(text, false, true, false, (char) 0);
        }

        static Item anchor(char anchor, String text) {
            return new Item(text, false, false, false, anchor);
        }
    }
}
