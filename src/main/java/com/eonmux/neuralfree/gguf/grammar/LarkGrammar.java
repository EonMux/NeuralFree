package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Lark grammars as an output constraint: vLLM's {@code structured_outputs.grammar} (and {@code guided_grammar}) in
 * Lark syntax, which its xgrammar backend compiles with {@code compile_lark}. The source parses with
 * {@link LarkParser} and compiles to one GBNF grammar the engine's own runtime enforces: rules and terminals
 * become GBNF rules ({@code start} is the root), literals and ranges become literals and character sets, a regular
 * expression compiles through {@link RegexToGbnf} in Python's dialect ({@code s} lets {@code .} match a line
 * break, {@code i} case folds it through {@link GrammarCaseFold}), {@code %json} through {@link JsonSchemaToGbnf}
 * in the constraint's whitespace mode, and token sets become GBNF token references. {@code %ignore} works as
 * xgrammar's does: inside rules, the ignored text may follow every token (a literal, range, regex, terminal
 * reference, {@code %json} or token set). Left-recursive rules are rewritten ({@link LarkLeftRecursion}); a rule
 * whose every alternative starts with itself is accepted and matches nothing, as xgrammar's do. A rule's
 * attributes {@code name[max_tokens=n, max_chars=m]} become GBNF rule attributes, whose regions the sampling
 * stage spends ({@code engine/sampling/GrammarStage}), and {@code name[stop="s"]} compiles as the product of the
 * body with no stop string inside ({@link GrammarStop}) followed by the stop: the rule's text ends at the stop.
 *
 * <p>{@link #isLikelyLark} is vLLM's {@code grammar_is_likely_lark}: a grammar text with no {@code ::=} rule line
 * is Lark, so one {@code grammar} field takes both syntaxes. A text starting with llama.cpp's
 * {@code %llguidance} header is Lark too ({@link #isLark}); the {@link LarkParser} accepts and validates that header
 * and compiles the rest of the grammar unchanged.</p>
 */
public final class LarkGrammar {

    /** The rule of the {@code %ignore} text; no Lark name starts with a digit. */
    private static final String IGNORE_RULE = "0ignore";

    /** The header llama.cpp routes to its llguidance Lark backend. */
    private static final String LLGUIDANCE_PREFIX = "%llguidance";

    private LarkGrammar() {
    }

    /**
     * Whether {@code grammar} looks like Lark rather than GBNF, vLLM's {@code grammar_is_likely_lark}: no line, its
     * {@code #} or {@code //} comment removed, holds {@code ::=}. Null and empty text is not Lark.
     */
    public static boolean isLikelyLark(String grammar) {
        if (grammar == null || grammar.isEmpty()) {
            return false;
        }
        for (String line : grammar.split("\n", -1)) {
            int hash = line.indexOf('#');
            int slashes = line.indexOf("//");
            int cut = hash < 0 ? slashes : slashes < 0 ? hash : Math.min(hash, slashes);
            if ((cut < 0 ? line : line.substring(0, cut)).contains("::=")) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether {@code grammar} is Lark: llama.cpp's {@code %llguidance} header at the start of the text (which selects
     * its llguidance Lark backend), or vLLM's {@link #isLikelyLark} (no {@code ::=} rule line). Null and empty text
     * is not Lark.
     */
    public static boolean isLark(String grammar) {
        return (grammar != null && grammar.startsWith(LLGUIDANCE_PREFIX)) || isLikelyLark(grammar);
    }

    /**
     * The validated grammar of a Lark source.
     *
     * @param whitespace the whitespace mode of {@code %json} schemas; null is flexible
     * @param vocab      the vocabulary named special tokens ({@code <|im_end|>}) resolve through, or null to
     *                   refuse them
     * @throws GrammarException         at the Lark source position of a syntax error or a refused construct
     * @throws IllegalArgumentException when the compiled grammar breaks a {@link GrammarLimits} bound or keeps a
     *                                  left recursion the rewrite does not remove
     */
    public static Grammar compile(String source, JsonSchemaToGbnf.WhitespaceMode whitespace, GrammarVocab vocab) {
        String gbnf = gbnf(source, whitespace, vocab);
        try {
            return GbnfParser.parse(gbnf);
        } catch (GrammarException e) {
            throw new IllegalArgumentException("Lark grammar is not supported by the grammar engine: "
                    + e.getMessage().replaceFirst(" \\(line \\d+, column \\d+\\)$", ""), e);
        }
    }

    /** The GBNF text of a Lark source, every token reference already resolved to its id. */
    static String gbnf(String source, JsonSchemaToGbnf.WhitespaceMode whitespace, GrammarVocab vocab) {
        Objects.requireNonNull(source, "source");
        LarkParser.Result parsed = LarkParser.parse(source, vocab);
        Set<String> terminals = new HashSet<>();
        Map<String, int[]> budgets = new HashMap<>();
        for (LarkParser.Definition d : parsed.definitions().values()) {
            if (d.terminal()) {
                terminals.add(d.name());
            } else if (d.maxTokens() >= 0 || d.maxChars() >= 0) {
                budgets.put(d.name(), new int[]{d.maxTokens(), d.maxChars()});
            }
        }
        boolean ignoring = !parsed.ignores().isEmpty();
        Map<String, LarkExpr> rules = new LinkedHashMap<>();
        for (LarkParser.Definition d : parsed.definitions().values()) {
            if (!d.terminal()) {
                rules.put(d.name(), ignoring ? ignoringAlone(d.body(), terminals) : d.body());
            }
        }
        Emitter emitter = new Emitter(whitespace == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : whitespace);
        for (Map.Entry<String, LarkExpr> rule : LarkLeftRecursion.rewrite(rules, parsed.definitions()).entrySet()) {
            int[] budget = budgets.get(rule.getKey());
            LarkParser.Definition d = parsed.definitions().get(rule.getKey());
            if (d != null && !d.stops().isEmpty()) {
                emitter.stopRule(rule.getKey(), d.stops(), stopBodyGrammar(rule.getValue(), d.stops(), parsed,
                        terminals, ignoring, whitespace, vocab, rule.getKey()), budget == null ? -1 : budget[0],
                        budget == null ? -1 : budget[1]);
                continue;
            }
            emitter.rule(rule.getKey(), rule.getValue(), budget == null ? -1 : budget[0],
                    budget == null ? -1 : budget[1]);
        }
        for (LarkParser.Definition d : parsed.definitions().values()) {
            if (d.terminal()) {
                emitter.rule(d.name(), d.body(), -1, -1);
            }
        }
        if (ignoring) {
            LarkExpr ignored = parsed.ignores().size() == 1 ? parsed.ignores().get(0)
                    : LarkExpr.alt(parsed.ignores(), 0, 0);
            emitter.ignoreRule(LarkExpr.repeat(ignored, 0, LarkExpr.UNBOUNDED, 0, 0));
        }
        return emitter.text();
    }

    /** The GBNF rule name of a Lark name: {@code start} is the root, {@code root} moves aside, {@code _} is a dash. */
    static String ruleName(String larkName) {
        if (larkName.equals("start")) {
            return Grammar.ROOT_RULE;
        }
        if (larkName.equals(Grammar.ROOT_RULE)) {
            return "0root";
        }
        return larkName.replace('_', '-');
    }

    // ------------------------------------------------------------------ %ignore

    /** {@code e} as a whole rule body or group member, the ignore slot after it when it is a token. */
    private static LarkExpr ignoringAlone(LarkExpr e, Set<String> terminals) {
        if (isToken(e, terminals)) {
            List<LarkExpr> pair = new ArrayList<>(2);
            pair.add(e);
            pair.add(LarkExpr.ignore());
            return LarkExpr.seq(pair, e.line(), e.column());
        }
        return ignoring(e, terminals);
    }

    /** {@code e} with the ignore slot after every token inside it. */
    private static LarkExpr ignoring(LarkExpr e, Set<String> terminals) {
        switch (e.kind()) {
            case SEQ: {
                List<LarkExpr> items = new ArrayList<>();
                for (LarkExpr item : e.children()) {
                    if (isToken(item, terminals)) {
                        items.add(item);
                        items.add(LarkExpr.ignore());
                    } else {
                        items.add(ignoring(item, terminals));
                    }
                }
                return e.withChildren(items);
            }
            case ALT:
            case REPEAT: {
                List<LarkExpr> children = new ArrayList<>();
                for (LarkExpr child : e.children()) {
                    children.add(ignoringAlone(child, terminals));
                }
                return e.withChildren(children);
            }
            default:
                return e;
        }
    }

    private static boolean isToken(LarkExpr e, Set<String> terminals) {
        return e.isTokenAtom() || (e.kind() == LarkExpr.Kind.REF && terminals.contains(e.text()));
    }

    // ------------------------------------------------------------------ stop attributes

    /**
     * The grammar of a stop rule's body with no stop string inside any derivation ({@link GrammarStop}): the body
     * compiles on its own — the terminals it names, the {@code %ignore} rule, and the body at the root — and the
     * product runs over the compiled rules, so regexes and terminal references carry the stop-free state too.
     */
    private static Grammar stopBodyGrammar(LarkExpr body, List<String> stops, LarkParser.Result parsed,
            Set<String> terminals, boolean ignoring, JsonSchemaToGbnf.WhitespaceMode whitespace,
            GrammarVocab vocab, String ruleName) {
        checkStopBody(body, terminals);
        Emitter bodyEmitter = new Emitter(whitespace == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE
                : whitespace);
        for (LarkParser.Definition d : parsed.definitions().values()) {
            if (d.terminal()) {
                bodyEmitter.rule(d.name(), d.body(), -1, -1);
            }
        }
        if (ignoring) {
            LarkExpr ignored = parsed.ignores().size() == 1 ? parsed.ignores().get(0)
                    : LarkExpr.alt(parsed.ignores(), 0, 0);
            bodyEmitter.ignoreRule(LarkExpr.repeat(ignored, 0, LarkExpr.UNBOUNDED, 0, 0));
        }
        bodyEmitter.rule("start", body, -1, -1);
        Grammar compiled = GbnfParser.parse(bodyEmitter.text(), vocab);
        return GrammarStop.apply(compiled, stops, ruleName(ruleName));
    }

    /**
     * What a stop body may hold, as xgrammar's terminal-like stop bodies do: literals, ranges, regular
     * expressions, terminal references, and the combinators over them; no rule references, no {@code %json}, no
     * tokens, and no case-insensitive regular expression.
     */
    private static void checkStopBody(LarkExpr e, Set<String> terminals) {
        switch (e.kind()) {
            case REF:
                if (!terminals.contains(e.text())) {
                    throw atExpr(e, "a stop body cannot reference rule '" + e.text() + "'");
                }
                return;
            case JSON:
                throw atExpr(e, "%json cannot be used in terminals");
            case TOKENS:
                throw atExpr(e, "special tokens cannot be used in terminals");
            case REGEX:
                if (e.flags().indexOf('i') >= 0) {
                    throw atExpr(e, "regular-expression flag 'i' is not supported with suffix or"
                            + " stop attributes");
                }
                return;
            default:
                for (LarkExpr child : e.children()) {
                    checkStopBody(child, terminals);
                }
        }
    }

    /** A refusal at the 1-based position of an expression, the shape {@link LarkParser} refuses with. */
    private static GrammarException atExpr(LarkExpr e, String message) {
        return new GrammarException("invalid Lark grammar: " + message, e.line(), e.column());
    }

    // ------------------------------------------------------------------ GBNF text

    /**
     * Writes the GBNF text: one {@code name ::= body} line per rule, then the regular expressions and JSON schemas
     * compiled on the way, each embedded once under its own prefix ({@link GbnfPrinter}).
     */
    private static final class Emitter {
        private final JsonSchemaToGbnf.WhitespaceMode whitespace;
        private final StringBuilder rules = new StringBuilder();
        private final StringBuilder embedded = new StringBuilder();
        private final Map<String, String> embeddedRoots = new HashMap<>();

        Emitter(JsonSchemaToGbnf.WhitespaceMode whitespace) {
            this.whitespace = whitespace;
        }

        void rule(String larkName, LarkExpr body, int maxTokens, int maxChars) {
            rules.append(ruleName(larkName)).append(attributeText(maxTokens, maxChars)).append(" ::= ")
                    .append(render(body)).append('\n');
        }

        /**
         * A rule with a {@code stop} attribute: the product of its body with "no stop inside"
         * ({@code product}, embedded once), then the stop itself, so the rule's text ends at the stop.
         */
        void stopRule(String larkName, List<String> stops, Grammar product, int maxTokens, int maxChars) {
            String root = embed("stop" + embeddedRoots.size(), product, "0stop");
            rules.append(ruleName(larkName)).append(attributeText(maxTokens, maxChars)).append(" ::= ")
                    .append(root).append(' ').append(literal(stops.get(0), false)).append('\n');
        }

        /** The rule's budget attributes, xgrammar's {@code [max_tokens=n, max_chars=m]}. */
        private static String attributeText(int maxTokens, int maxChars) {
            if (maxTokens < 0 && maxChars < 0) {
                return "";
            }
            StringBuilder out = new StringBuilder("[");
            if (maxTokens >= 0) {
                out.append("max_tokens=").append(maxTokens);
            }
            if (maxChars >= 0) {
                if (out.length() > 1) {
                    out.append(", ");
                }
                out.append("max_chars=").append(maxChars);
            }
            return out.append(']').toString();
        }

        void ignoreRule(LarkExpr body) {
            rules.append(IGNORE_RULE).append(" ::= ").append(render(body)).append('\n');
        }

        String text() {
            return rules.toString() + embedded;
        }

        private String render(LarkExpr e) {
            switch (e.kind()) {
                case SEQ:
                    return sequence(e.children());
                case ALT: {
                    List<String> alternatives = new ArrayList<>(e.children().size());
                    for (LarkExpr child : e.children()) {
                        alternatives.add(render(child));
                    }
                    return "(" + String.join(" | ", alternatives) + ")";
                }
                case REPEAT:
                    return repetition(e);
                case LITERAL:
                    return literal(e.text(), e.caseInsensitive());
                case RANGE:
                    return RegexCharSet.range(e.low(), e.high()).toGbnf(false);
                case REGEX:
                    return regex(e);
                case REF:
                    return ruleName(e.text());
                case JSON:
                    return json(e);
                case TOKENS:
                    return tokens(e.tokens(), e.negated());
                case IGNORE:
                    return IGNORE_RULE;
                default:
                    throw new IllegalStateException("Lark expression " + e.kind() + " has no GBNF form");
            }
        }

        private String sequence(List<LarkExpr> items) {
            if (items.isEmpty()) {
                return "\"\"";
            }
            List<String> parts = new ArrayList<>(items.size());
            for (LarkExpr item : items) {
                parts.add(render(item));
            }
            return String.join(" ", parts);
        }

        private String repetition(LarkExpr e) {
            int min = e.low();
            int max = e.high();
            if (max == 0) {
                return "\"\"";
            }
            String inner = "(" + render(e.children().get(0)) + ")";
            if (max == LarkExpr.UNBOUNDED) {
                return inner + (min == 0 ? "*" : min == 1 ? "+" : "{" + min + ",}");
            }
            if (min == 0 && max == 1) {
                return inner + "?";
            }
            if (min == max) {
                if (min <= GrammarLimits.MAX_REPETITION_EXPANSION) {
                    return inner + "{" + min + "}";
                }
                // an exact count above the GBNF expansion limit: the copies are written out, as xgrammar
                // keeps the count exact
                return expanded(e.children().get(0), min);
            }
            if (min > GrammarLimits.MAX_REPETITION_EXPANSION) {
                return expanded(e.children().get(0), min) + " (" + inner + "){" + (max - min) + "}";
            }
            return inner + "{" + min + "," + max + "}";
        }

        /** {@code child} written {@code count} times; a group per copy would exceed the runtime's rule count. */
        private String expanded(LarkExpr child, int count) {
            if (child.kind() == LarkExpr.Kind.ALT) {
                String inner = "(" + render(child) + ")";
                StringBuilder out = new StringBuilder(inner.length() * count);
                for (int i = 0; i < count; i++) {
                    if (i > 0) {
                        out.append(' ');
                    }
                    out.append(inner);
                }
                return out.toString();
            }
            String one = child.kind() == LarkExpr.Kind.SEQ ? sequence(child.children()) : render(child);
            StringBuilder out = new StringBuilder(one.length() * count);
            for (int i = 0; i < count; i++) {
                if (i > 0) {
                    out.append(' ');
                }
                out.append(one);
            }
            return out.toString();
        }

        /** A literal; case-insensitive (ASCII only, as the parser checks), each letter is the set of both cases. */
        private static String literal(String text, boolean caseInsensitive) {
            if (text.isEmpty()) {
                return "\"\"";
            }
            List<String> parts = new ArrayList<>();
            StringBuilder run = new StringBuilder();
            text.codePoints().forEach(cp -> {
                boolean letter = (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
                if (!caseInsensitive || !letter) {
                    run.append(RegexCharSet.literalChar(cp));
                    return;
                }
                if (run.length() > 0) {
                    parts.add("\"" + run + "\"");
                    run.setLength(0);
                }
                parts.add(RegexCharSet.single(cp).asciiCaseClosure().toGbnf(false));
            });
            if (run.length() > 0) {
                parts.add("\"" + run + "\"");
            }
            return String.join(" ", parts);
        }

        private static String tokens(int[] ids, boolean negated) {
            StringBuilder out = new StringBuilder(negated ? "!<" : "<");
            for (int i = 0; i < ids.length; i++) {
                out.append(i == 0 ? "[" : ",[").append(Integer.toUnsignedString(ids[i])).append(']');
            }
            return out.append('>').toString();
        }

        /** The root rule of the regex's compiled grammar, embedded once per pattern and flags. */
        private String regex(LarkExpr e) {
            String key = "re/" + e.flags() + "/" + e.text();
            String known = embeddedRoots.get(key);
            if (known != null) {
                return known;
            }
            String shown = "/" + e.text() + "/" + e.flags();
            Grammar compiled;
            try {
                compiled = RegexToGbnf.compilePython(e.text(), Grammar.ROOT_RULE, e.flags().indexOf('s') >= 0);
            } catch (RegexToGbnf.UnsupportedRegexException ex) {
                throw atExpr(e, "regular expression " + shown + " uses an unsupported construct: " + ex.getMessage());
            } catch (RegexToGbnf.InvalidRegexException ex) {
                throw atExpr(e, "regular expression " + shown + " is not valid: " + ex.getMessage());
            } catch (IllegalArgumentException ex) {
                throw atExpr(e, "regular expression " + shown + " does not compile: " + ex.getMessage());
            }
            if (e.flags().indexOf('i') >= 0) {
                compiled = GrammarCaseFold.fold(compiled);
            }
            return embed(key, compiled, "0re");
        }

        /** The root rule of the inline schema's compiled grammar, embedded once per schema text. */
        private String json(LarkExpr e) {
            String key = "json/" + e.text();
            String known = embeddedRoots.get(key);
            if (known != null) {
                return known;
            }
            Grammar compiled;
            try {
                compiled = JsonSchemaToGbnf.convert(e.text(), whitespace);
            } catch (IllegalArgumentException ex) {
                throw atExpr(e, "failed to compile inline JSON schema: " + ex.getMessage());
            }
            return embed(key, compiled, "0json");
        }

        private String embed(String key, Grammar compiled, String kind) {
            String prefix = kind + embeddedRoots.size() + "-";
            embedded.append(GbnfPrinter.print(compiled, prefix));
            String root = GbnfPrinter.ruleName(compiled, compiled.rootRuleIndex(), prefix);
            embeddedRoots.put(key, root);
            return root;
        }
    }
}
