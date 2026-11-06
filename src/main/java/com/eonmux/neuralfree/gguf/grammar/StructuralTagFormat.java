package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The modern xgrammar structural-tag {@code format} object: {@code {"type": "structural_tag", "format": {...}}}
 * compiles to one grammar over the same format tree xgrammar accepts — the value formats {@code const_string},
 * {@code json_schema}, {@code any_text}, {@code grammar} (EBNF, or Lark with {@code "syntax": "lark"}),
 * {@code regex} and {@code token}, the
 * combinators {@code sequence}, {@code or}, {@code tag} (with its multi-end and token ends), {@code optional},
 * {@code plus}, {@code star}, {@code repeat} and {@code tags_with_separator}, the token sets
 * {@code exclude_token} and {@code any_tokens}, and the dispatching formats {@code triggered_tags},
 * {@code dispatch}, {@code token_triggered_tags} and {@code token_dispatch}. A {@code format} node without a
 * {@code type} field infers its type from its members, in xgrammar's order. A {@code json_schema} with
 * {@code excludes} writes none of them in the raw text of a free JSON string (value or key), and a {@code regex}
 * matches whole with {@code .} matching line breaks too, as in xgrammar.
 *
 * <p>A {@code json_schema} compiles in its {@code style} — standard JSON, or the top-level object as the tagged
 * parameters of {@code qwen_xml}, {@code minimax_xml}, {@code deepseek_xml}, {@code deepseek_v4_1_xml} or
 * {@code glm_xml} ({@link XmlParameterRules}; the deprecated {@code qwen_xml_parameter} format is the
 * {@code qwen_xml} style) — with {@code any_order} and {@code max_whitespace_cnt} ({@link JsonSchemaOptions}), its
 * properties in the order the payload declares them. An {@code any_text} takes {@code max_tokens} and
 * {@code max_chars}, an {@code any_tokens} {@code max_tokens}: the region is a counted rule
 * ({@code rule[max_tokens=n, max_chars=m]}, {@link GbnfParser}) whose budget the sampling stage spends, and
 * {@code exclude_token} and {@code any_tokens} take an empty exclusion set (any token).
 * A {@code grammar} is GBNF, or Lark under {@code "syntax": "lark"} ({@link LarkGrammar}).
 *
 * <p>A dispatching format lets free text run until a trigger (a string, or a token for the token variants), then
 * holds the output to the alternative the trigger opens. At the top level it compiles through the trigger stage
 * the legacy payload uses ({@link GrammarTriggers}): the grammar is the alternation of the alternatives,
 * {@link GrammarTriggers.Mode#DISPATCH} resumes free text after each one, {@link GrammarTriggers.Mode#LAZY} ends
 * the format after the first ({@code stop_after_first}, {@code loop=false}), and the {@code excludes} and
 * {@code exclude_tokens} travel as {@link CompiledConstraint#freeExcludes}. Nested inside another format (the
 * {@code <think>} prefix of a tool-calling sequence), or when {@code at_least_one} makes an alternative open the
 * output, it compiles to plain grammar rules: free text that never writes a trigger or an exclusion, then an
 * alternative at the earliest occurrence of its trigger ({@link GbnfExclusion#dispatching}), looping as
 * configured. A {@code triggered_tags} or {@code any_text} inside a tag also never writes
 * the tag's end strings, and a {@code token_triggered_tags} or {@code any_tokens} never its end token, as in
 * xgrammar.
 *
 * <p>Refusals name their field and mirror xgrammar's checks: a missing member, a wrong-typed member, an unknown
 * or unsupported type (the {@code minimax_m3_xml}, {@code cohere_xml} and {@code kimi_k3_xml} styles, a grammar
 * {@code syntax} other than {@code ebnf} or {@code lark}, {@code json_schema} {@code excludes} with an XML
 * parameter style), and a tag matching no trigger or several — all as {@link IllegalArgumentException}.
 */
final class StructuralTagFormat {

    /** The compiled payload: the grammar, its triggers, and its free-text exclusions. */
    static final class Compiled {
        private final Grammar grammar;
        private final GrammarTriggers triggers;
        private final List<String> excludes;
        private final int[] excludeTokens;

        Compiled(Grammar grammar, GrammarTriggers triggers, List<String> excludes, int[] excludeTokens) {
            this.grammar = grammar;
            this.triggers = triggers;
            this.excludes = excludes == null ? Collections.emptyList() : excludes;
            this.excludeTokens = excludeTokens == null ? new int[0] : excludeTokens;
        }

        Grammar grammar() {
            return grammar;
        }

        GrammarTriggers triggers() {
            return triggers;
        }

        List<String> excludes() {
            return excludes;
        }

        int[] excludeTokens() {
            return excludeTokens;
        }
    }

    /** The type inference order of xgrammar's parser (its {@code ParseFormat} fallbacks). */
    private static final String[] INFER_ORDER = {"tag", "const_string", "json_schema", "any_text", "sequence",
            "or", "triggered_tags", "tags_with_separator", "optional", "plus", "star", "repeat", "dispatch",
            "token_dispatch"};

    /** Deepest nesting of format nodes, xgrammar's {@code kMaxFormatDepth}. */
    private static final int MAX_DEPTH = 100;

    /**
     * How a schema object is written back to the text {@link JsonSchemaToGbnf} reads: compact, in insertion order,
     * so the properties keep the order the request declares them in (the order of the object's members).
     */
    static final Json.Style SCHEMA_TEXT = new Json.Style(false, null, ",", ":", false);

    private final StructuralTagRules rules;
    private final GrammarVocab vocab;
    private final JsonSchemaToGbnf.WhitespaceMode whitespace;
    private int nodes;
    private int depth;

    private StructuralTagFormat(JsonSchemaToGbnf.WhitespaceMode whitespace, GrammarVocab vocab) {
        this.rules = new StructuralTagRules();
        this.whitespace = whitespace;
        this.vocab = vocab;
    }

    /** Compiles the payload's {@code format} object: grammar, triggers and free-text exclusions in one pass. */
    static Compiled compile(JsonView payload, JsonSchemaToGbnf.WhitespaceMode whitespace, GrammarVocab vocab) {
        Object value = payload.raw("format");
        if (!(value instanceof Map)) {
            throw new IllegalArgumentException("structural tag format must be an object with a type");
        }
        StructuralTagFormat format = new StructuralTagFormat(whitespace, vocab);
        JsonView view = view(value, "format");
        String type = typeOf(view);
        if (type != null && dispatching(type)) {
            return format.topLevelDispatch(view, type);
        }
        if (type != null) {
            int root = format.format(value, "format", Collections.emptyList(), Collections.emptyList());
            return format.finish(root, null, null, null);
        }
        return format.infer(value, "format");
    }

    // ------------------------------------------------------------------ dispatching formats

    /**
     * A parsed dispatching format: one rule per dispatch alternative (each opening with its trigger), the string
     * trigger each string alternative opens with, the string or token triggers, the free-text exclusions, whether
     * free text resumes after a dispatch ({@code loop}) and whether an alternative must come first
     * ({@code at_least_one}).
     */
    private static final class Dispatch {
        final List<Integer> groups;
        final List<String> openings;
        final List<String> words;
        final List<Integer> tokens;
        final List<String> excludes;
        final List<Integer> excludeTokens;
        final boolean loop;
        final boolean atLeastOne;

        Dispatch(List<Integer> groups, List<String> openings, List<String> words, List<Integer> tokens,
                List<String> excludes, List<Integer> excludeTokens, boolean loop, boolean atLeastOne) {
            this.groups = groups;
            this.openings = openings;
            this.words = words;
            this.tokens = tokens;
            this.excludes = excludes;
            this.excludeTokens = excludeTokens;
            this.loop = loop;
            this.atLeastOne = atLeastOne;
        }
    }

    /**
     * Compiles a top-level dispatching format. The free text runs under the trigger stage: the grammar is the
     * alternation of the dispatch alternatives, the triggers gate it ({@link GrammarTriggers.Mode#DISPATCH} when
     * free text resumes, {@link GrammarTriggers.Mode#LAZY} when the first dispatch ends the format) and the
     * exclusions travel as {@link CompiledConstraint#freeExcludes}. An {@code at_least_one} format must open with
     * an alternative, which the trigger stage cannot force, so it compiles to a plain grammar instead.
     */
    private Compiled topLevelDispatch(JsonView v, String type) {
        Dispatch d = parseDispatch(v, type, "format", Collections.emptyList(), Collections.emptyList());
        if (d.atLeastOne) {
            return finish(dispatchGrammar(d), null, null, null);
        }
        GrammarTriggers.Builder b = GrammarTriggers.builder(d.loop
                ? GrammarTriggers.Mode.DISPATCH : GrammarTriggers.Mode.LAZY);
        d.words.forEach(b::word);
        d.tokens.forEach(b::token);
        int root = rules.alternation(nextName("dispatch-root"), d.groups);
        return finish(root, b.build(), d.excludes, d.excludeTokens.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * The dispatching format as plain grammar rules, xgrammar's {@code TagDispatch} for a format nested in another
     * one: the free text {@code F} never writes a trigger or an exclusion, and {@code FG} is free text followed by an
     * alternative at the earliest occurrence of its trigger ({@link GbnfExclusion#dispatching}; for tokens, free
     * tokens then an alternative); {@code loop} resumes the free text after it. With {@code D ::= FG D | F}
     * (looping) or {@code D ::= FG | F} (once), the format is {@code D}, or {@code G D} / {@code G} when one
     * alternative {@code G} must open it.
     */
    private int dispatchGrammar(Dispatch d) {
        int alternatives = rules.alternation(nextName("dispatch-group"), d.groups);
        if (d.atLeastOne && !d.loop) {
            return alternatives;
        }
        int free = d.words.isEmpty() ? freeTokens(d) : freeText(d);
        int dispatch = rules.reserve(nextName("dispatch"));
        List<GrammarElement> body = new ArrayList<>(6);
        if (d.words.isEmpty()) {
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, free));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, alternatives));
        } else {
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, GbnfExclusion.dispatching(rules,
                    nextName("dispatch-free"), banned(d), d.openings, d.groups)));
        }
        if (d.loop) {
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, dispatch));
        }
        body.add(new GrammarElement(GrammarElement.Type.ALT, 0));
        body.add(new GrammarElement(GrammarElement.Type.RULE_REF, free));
        rules.fill(dispatch, body);
        return d.atLeastOne
                ? rules.rule(nextName("dispatch-first"), StructuralTagRules.refSequence(alternatives, dispatch))
                : dispatch;
    }

    /** The free text of a string dispatch: any text holding none of the triggers and exclusions. */
    private int freeText(Dispatch d) {
        return excludingText(new LinkedHashSet<>(banned(d)));
    }

    /** The strings the free text of a string dispatch never writes: its triggers and exclusions. */
    private static List<String> banned(Dispatch d) {
        LinkedHashSet<String> banned = new LinkedHashSet<>(d.words);
        banned.addAll(d.excludes);
        return new ArrayList<>(banned);
    }

    /** The free text of a token dispatch: any run of tokens outside the trigger and excluded tokens. */
    private int freeTokens(Dispatch d) {
        Set<Integer> banned = new LinkedHashSet<>(d.tokens);
        banned.addAll(d.excludeTokens);
        int one = rules.anyTokenBut(nextName("free-token"), banned.stream().mapToInt(Integer::intValue).toArray());
        int self = rules.reserve(nextName("free-tokens"));
        rules.fill(self, Arrays.asList(new GrammarElement(GrammarElement.Type.RULE_REF, one),
                new GrammarElement(GrammarElement.Type.RULE_REF, self), new GrammarElement(GrammarElement.Type.ALT, 0)));
        return self;
    }

    /**
     * Parses a dispatching format of {@code type}. The end strings ({@code ends}) or end token ({@code endTokens})
     * of the tag around a {@code triggered_tags} or {@code token_triggered_tags} join its exclusions, as xgrammar
     * adds them; {@code dispatch} and {@code token_dispatch} keep their own exclusions only, as xgrammar does.
     */
    private Dispatch parseDispatch(JsonView v, String type, String path, List<String> ends,
            List<Integer> endTokens) {
        switch (type) {
            case "triggered_tags":
                return triggeredTags(v, path, ends);
            case "dispatch":
                return dispatch(v, path);
            case "token_triggered_tags":
                return tokenTriggeredTags(v, path, endTokens);
            case "token_dispatch":
                return tokenDispatch(v, path);
            default:
                throw new IllegalArgumentException("structural tag format type not recognized: " + type);
        }
    }

    /** The free text waits for a trigger word, then the matched tag's begin + content + end runs to its end. */
    private Dispatch triggeredTags(JsonView v, String path, List<String> ends) {
        List<String> triggers = nonEmptyStrings(v, "triggers", "triggered_tags format");
        if (triggers.isEmpty()) {
            throw new IllegalArgumentException(
                    "structural tag format triggered_tags needs at least one trigger");
        }
        List<Object> tags = rawList(v, "tags");
        if (tags.isEmpty()) {
            throw new IllegalArgumentException("structural tag format triggered_tags needs at least one tag");
        }
        boolean atLeastOne = boolField(v, "at_least_one");
        boolean stopAfterFirst = boolField(v, "stop_after_first");
        LinkedHashSet<String> excludes = new LinkedHashSet<>(nonEmptyStrings(v, "excludes", "triggered_tags format"));
        excludes.addAll(ends);

        List<Integer> tagRules = new ArrayList<>(tags.size());
        List<String> openings = new ArrayList<>(tags.size());
        for (int i = 0; i < tags.size(); i++) {
            JsonView tag = tagView(tags.get(i), path + ".tags[" + i + "]");
            String begin = stringBegin(tag, i, "triggered_tags");
            openings.add(matchExactlyOneTrigger(begin, triggers, i));
            tagRules.add(tagRule(tag, path + ".tags[" + i + "]"));
        }
        return new Dispatch(tagRules, openings, triggers, Collections.emptyList(), new ArrayList<>(excludes),
                Collections.emptyList(), !stopAfterFirst, atLeastOne);
    }

    /** The free text waits for a rule's pattern, then the rule's format runs; {@code loop=false} runs once. */
    private Dispatch dispatch(JsonView v, String path) {
        List<Object> ruleList = rawList(v, "rules");
        if (ruleList.isEmpty()) {
            throw new IllegalArgumentException("structural tag format dispatch needs at least one rule");
        }
        boolean loop = boolField(v, "loop", true);
        List<String> excludes = nonEmptyStrings(v, "excludes", "dispatch format");

        List<Integer> groups = new ArrayList<>(ruleList.size());
        List<String> patterns = new ArrayList<>(ruleList.size());
        for (int i = 0; i < ruleList.size(); i++) {
            List<?> pair = dispatchRule(ruleList.get(i), "dispatch", i, "pattern");
            Object pattern = pair.get(0);
            if (!(pattern instanceof String) || ((String) pattern).isEmpty()) {
                throw new IllegalArgumentException("structural tag format dispatch rule " + i
                        + " needs a non-empty pattern string");
            }
            int content = format(pair.get(1), path + ".rules[" + i + "][1]",
                    Collections.emptyList(), Collections.emptyList());
            List<GrammarElement> body = new ArrayList<>(StructuralTagRules.codePoints((String) pattern));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, content));
            groups.add(rules.rule(nextName("dispatch-rule"), body));
            patterns.add((String) pattern);
        }
        return new Dispatch(groups, patterns, patterns, Collections.emptyList(), excludes, Collections.emptyList(),
                loop, false);
    }

    /** The free text waits for a trigger token, then the tag whose begin token it is runs to its end. */
    private Dispatch tokenTriggeredTags(JsonView v, String path, List<Integer> endTokens) {
        List<Integer> triggerTokens = tokenList(v, "trigger_tokens", "token_triggered_tags format");
        if (triggerTokens.isEmpty()) {
            throw new IllegalArgumentException(
                    "structural tag format token_triggered_tags needs at least one trigger token");
        }
        List<Object> tags = rawList(v, "tags");
        if (tags.isEmpty()) {
            throw new IllegalArgumentException(
                    "structural tag format token_triggered_tags needs at least one tag");
        }
        boolean atLeastOne = boolField(v, "at_least_one");
        boolean stopAfterFirst = boolField(v, "stop_after_first");
        Set<Integer> excludeTokens = new LinkedHashSet<>(
                tokenList(v, "exclude_tokens", "token_triggered_tags format"));
        excludeTokens.addAll(endTokens);

        List<Integer> tagRules = new ArrayList<>(tags.size());
        for (int i = 0; i < tags.size(); i++) {
            JsonView tag = tagView(tags.get(i), path + ".tags[" + i + "]");
            int begin = tokenBegin(tag, i);
            matchExactlyOneTokenTrigger(begin, triggerTokens, i);
            tagRules.add(tagRule(tag, path + ".tags[" + i + "]"));
        }
        return new Dispatch(tagRules, Collections.emptyList(), Collections.emptyList(), triggerTokens,
                Collections.emptyList(), new ArrayList<>(excludeTokens), !stopAfterFirst, atLeastOne);
    }

    /** The free text waits for a rule's trigger token, then the rule's format runs; {@code loop=false} runs once. */
    private Dispatch tokenDispatch(JsonView v, String path) {
        List<Object> ruleList = rawList(v, "rules");
        if (ruleList.isEmpty()) {
            throw new IllegalArgumentException("structural tag format token_dispatch needs at least one rule");
        }
        boolean loop = boolField(v, "loop", true);
        List<Integer> excludeTokens = tokenList(v, "exclude_tokens", "token_dispatch format");

        List<Integer> groups = new ArrayList<>(ruleList.size());
        List<Integer> tokens = new ArrayList<>(ruleList.size());
        for (int i = 0; i < ruleList.size(); i++) {
            List<?> pair = dispatchRule(ruleList.get(i), "token_dispatch", i, "token");
            int token = tokenId(pair.get(0), "structural tag format token_dispatch rule " + i);
            int content = format(pair.get(1), path + ".rules[" + i + "][1]",
                    Collections.emptyList(), Collections.emptyList());
            List<GrammarElement> body = new ArrayList<>(2);
            body.add(new GrammarElement(GrammarElement.Type.TOKEN, token));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, content));
            groups.add(rules.rule(nextName("token-dispatch-rule"), body));
            tokens.add(token);
        }
        return new Dispatch(groups, Collections.emptyList(), Collections.emptyList(), tokens, Collections.emptyList(),
                excludeTokens, loop, false);
    }

    /** One {@code [trigger, format]} pair of a dispatch's {@code rules}. */
    private static List<?> dispatchRule(Object pair, String format, int index, String trigger) {
        if (!(pair instanceof List) || ((List<?>) pair).size() != 2) {
            throw new IllegalArgumentException("structural tag format " + format + " rule " + index
                    + " must be a two-element array of " + trigger + " and format");
        }
        List<?> elements = (List<?>) pair;
        if (elements.get(1) == null) {
            throw new IllegalArgumentException("structural tag format " + format + " rule " + index
                    + " needs a format");
        }
        return elements;
    }

    // ------------------------------------------------------------------ format nodes

    /**
     * One format node nested in another: a rule index. {@code ends} and {@code endTokens} are the end strings or
     * end token of the tag around it, which its free text must never write. A node without a {@code type} infers
     * its type in xgrammar's order.
     */
    private int format(Object value, String path, List<String> ends, List<Integer> endTokens) {
        if (++depth > MAX_DEPTH) {
            throw new IllegalArgumentException("structural tag format is nested deeper than " + MAX_DEPTH
                    + " levels");
        }
        try {
            if (!(value instanceof Map)) {
                throw new IllegalArgumentException("structural tag " + path + " must be an object with a type");
            }
            JsonView v = view(value, path);
            String type = typeOf(v);
            if (type != null) {
                return visit(type, v, path, ends, endTokens);
            }
            for (String candidate : INFER_ORDER) {
                int[] mark = rules.mark();
                try {
                    return visit(candidate, v, path, ends, endTokens);
                } catch (RuntimeException e) {
                    rules.undo(mark);
                }
            }
            throw new IllegalArgumentException("structural tag " + path + " matches no format type; add a type"
                    + " field naming one of the structural tag formats");
        } finally {
            depth--;
        }
    }

    /** One format node of {@code type}; a dispatching type compiles to grammar rules ({@link #dispatchGrammar}). */
    private int visit(String type, JsonView v, String path, List<String> ends, List<Integer> endTokens) {
        if (dispatching(type)) {
            return dispatchGrammar(parseDispatch(v, type, path, ends, endTokens));
        }
        switch (type) {
            case "const_string":
                return constString(v);
            case "json_schema":
                return jsonSchema(v);
            case "any_text":
                return anyText(v, ends);
            case "grammar":
                return grammar(v);
            case "regex":
                return regex(v);
            case "sequence":
                return elements(v, path, ends, endTokens, true);
            case "or":
                return elements(v, path, ends, endTokens, false);
            case "tag":
                return tagRule(v, path);
            case "tags_with_separator":
                return tagsWithSeparator(v, path);
            case "optional":
                return optional(v, path, ends, endTokens);
            case "plus":
                return repeated(v, path, ends, endTokens, 1, -1);
            case "star":
                return repeated(v, path, ends, endTokens, 0, -1);
            case "repeat":
                return repeat(v, path, ends, endTokens);
            case "token":
                return token(v);
            case "exclude_token":
                return excludeToken(v, endTokens);
            case "any_tokens":
                return anyTokens(v, endTokens);
            case "qwen_xml_parameter":
                return schemaRule(v, "qwen_xml_parameter format", JsonSchemaOptions.json(whitespace)
                        .withStyle(XmlParameterStyle.QWEN));
            default:
                throw new IllegalArgumentException("structural tag format type not recognized: " + type);
        }
    }

    /** One inferred top-level format: each type in xgrammar's order, the first that compiles. */
    private Compiled infer(Object value, String path) {
        JsonView v = view(value, path);
        for (String candidate : INFER_ORDER) {
            int[] mark = rules.mark();
            try {
                return dispatching(candidate) ? topLevelDispatch(v, candidate)
                        : finish(visit(candidate, v, path, Collections.emptyList(), Collections.emptyList()),
                                null, null, null);
            } catch (RuntimeException e) {
                rules.undo(mark);
            }
        }
        throw new IllegalArgumentException("structural tag format matches no supported type; add a type field"
                + " naming one of the structural tag formats");
    }

    /** The constant string. */
    private int constString(JsonView v) {
        Object value = require(v, "value", "const_string format");
        if (!(value instanceof String)) {
            throw new IllegalArgumentException("structural tag const_string format needs a value string");
        }
        return rules.literal(nextName("const-string"), (String) value);
    }

    /**
     * The JSON schema in its {@code style} (standard JSON, or the top-level object as {@link XmlParameterStyle XML
     * parameters}), with xgrammar's {@code any_order} and {@code max_whitespace_cnt}; the strings of an
     * {@code excludes} member hold no free JSON string, in the standard {@code json} style.
     */
    private int jsonSchema(JsonView v) {
        JsonSchemaOptions options = JsonSchemaOptions.json(whitespace).withStyle(schemaStyle(v.raw("style")));
        Object anyOrder = v.raw("any_order");
        if (anyOrder != null && !(anyOrder instanceof Boolean)) {
            throw new IllegalArgumentException("structural tag json_schema any_order must be a boolean, got "
                    + anyOrder);
        }
        options = options.withAnyOrder(Boolean.TRUE.equals(anyOrder));
        Object maxWhitespace = v.raw("max_whitespace_cnt");
        if (maxWhitespace != null) {
            if (!isIntegral(maxWhitespace) || ((Number) maxWhitespace).longValue() < 1
                    || ((Number) maxWhitespace).longValue() > GrammarLimits.MAX_REPETITION_EXPANSION) {
                throw new IllegalArgumentException("structural tag json_schema max_whitespace_cnt must be an"
                        + " integer from 1 to " + GrammarLimits.MAX_REPETITION_EXPANSION + ", got " + maxWhitespace);
            }
            options = options.withMaxWhitespace((int) ((Number) maxWhitespace).longValue());
        }
        List<String> excludes = nonEmptyStrings(v, "excludes", "json_schema format");
        if (excludes.isEmpty()) {
            return schemaRule(v, "json_schema format", options);
        }
        if (options.style() != null) {
            throw new IllegalArgumentException("structural tag json_schema excludes compiles in the standard"
                    + " \"json\" style only; drop the style or put the exclusion on an any_text format");
        }
        return schemaRule(v, "json_schema format", options, excludes);
    }

    /** The XML parameter style named {@code raw}; null for the standard {@code "json"} style or no style. */
    private static XmlParameterStyle schemaStyle(Object raw) {
        if (raw == null || "json".equals(raw)) {
            return null;
        }
        XmlParameterStyle style = raw instanceof String ? XmlParameterStyle.named((String) raw) : null;
        if (style != null) {
            return style;
        }
        StringBuilder supported = new StringBuilder("json");
        for (XmlParameterStyle candidate : XmlParameterStyle.values()) {
            supported.append(", ").append(candidate.keyword);
        }
        throw new IllegalArgumentException("structural tag json_schema style '" + raw + "' is "
                + (XmlParameterStyle.UNSUPPORTED.contains(raw) ? "not supported" : "not recognized")
                + "; the engine compiles the styles " + supported);
    }

    /** The {@code json_schema} field of {@code v} converted under {@code options}. */
    private int schemaRule(JsonView v, String what, JsonSchemaOptions options) {
        return schemaRule(v, what, options, Collections.<String>emptyList());
    }

    /**
     * The {@code json_schema} field of {@code v} converted under {@code options}; the strings of {@code excludes}
     * hold no free JSON string, value or key ({@code json} style only).
     */
    private int schemaRule(JsonView v, String what, JsonSchemaOptions options, List<String> excludes) {
        Object schema = require(v, "json_schema", what);
        String schemaJson;
        if (schema instanceof Map) {
            StringBuilder out = new StringBuilder();
            try {
                Json.write(schema, SCHEMA_TEXT, out);
            } catch (java.io.IOException e) {
                throw new IllegalArgumentException("structural tag " + what + ": its schema does not write back"
                        + " to JSON", e);
            }
            schemaJson = out.toString();
        } else if (Boolean.TRUE.equals(schema)) {
            schemaJson = options.style() != null ? "{}" : "{\"type\": [\"object\", \"array\", \"string\","
                    + " \"number\", \"integer\", \"boolean\", \"null\"]}";
        } else {
            throw new IllegalArgumentException("structural tag " + what + " needs a json_schema object"
                    + " or true" + (Boolean.FALSE.equals(schema) ? "; false admits no output and is refused" : ""));
        }
        if (excludes.isEmpty()) {
            return rules.merge(JsonSchemaToGbnf.convert(schemaJson, options), nextName("schema-"));
        }
        return rules.merge(excludingSchema(schemaJson, options, excludes), nextName("schema-"));
    }

    /**
     * The schema's grammar with none of {@code excludes} in the raw text of a free JSON string, value or key
     * ({@link JsonSchemaToGbnf.Composer#addSchema(String, String, List)}).
     *
     * @throws IllegalArgumentException naming the string the schema bounds, so it cannot hold the exclusion
     */
    private Grammar excludingSchema(String schemaJson, JsonSchemaOptions options, List<String> excludes) {
        JsonSchemaToGbnf.Composer composer = JsonSchemaToGbnf.composer(options);
        String gbnf;
        try {
            composer.addRule(Grammar.ROOT_RULE, composer.addSchema("schema", schemaJson, excludes));
            gbnf = composer.gbnf();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("structural tag json_schema: " + e.getMessage(), e);
        }
        return GbnfParser.parse(gbnf, vocab);
    }

    /** Any run of text; the excludes, plus the ends of the tag around it, can never be written. */
    private int anyText(JsonView v, List<String> ends) {
        if (!v.has("type") && !v.has("excludes")) {
            // xgrammar infers no type from an empty object: a fieldless any_text must name its type
            throw new IllegalArgumentException("structural tag any_text should not have any fields other than"
                    + " type and excludes");
        }
        int maxTokens = budget(v, "any_text", "max_tokens");
        int maxChars = budget(v, "any_text", "max_chars");
        LinkedHashSet<String> excludes = new LinkedHashSet<>();
        Object raw = v.raw("excludes");
        if (raw != null) {
            if (!(raw instanceof List)) {
                throw new IllegalArgumentException("structural tag any_text excludes must be an array of strings");
            }
            List<?> list = (List<?>) raw;
            for (int i = 0; i < list.size(); i++) {
                Object s = list.get(i);
                if (!(s instanceof String) || ((String) s).isEmpty()) {
                    throw new IllegalArgumentException("structural tag any_text excludes " + i
                            + " is not a non-empty string");
                }
                excludes.add((String) s);
            }
        }
        excludes.addAll(ends);
        return rules.budgeted(nextName("any-text-budget"), maxTokens, maxChars, excludingText(excludes));
    }

    /**
     * Any run of text holding none of {@code banned} (non-empty strings): the exclusion automaton of
     * {@link GbnfExclusion}, or any text at all when nothing is banned.
     */
    private int excludingText(Set<String> banned) {
        if (banned.isEmpty()) {
            int self = rules.reserve(nextName("any-text"));
            rules.fill(self, Arrays.asList(new GrammarElement(GrammarElement.Type.CHAR_ANY, 0),
                    new GrammarElement(GrammarElement.Type.RULE_REF, self),
                    new GrammarElement(GrammarElement.Type.ALT, 0)));
            return self;
        }
        GbnfBuilder builder = new GbnfBuilder("");
        String start = GbnfExclusion.excluding(builder::addRule, "excluding", new ArrayList<>(banned));
        builder.addRule(Grammar.ROOT_RULE, start);
        return rules.merge(builder.grammar(), nextName("excluding-"));
    }

    /** An EBNF grammar compiled with this engine's GBNF parser, or a Lark one ({@code "syntax": "lark"}). */
    private int grammar(JsonView v) {
        Object grammar = require(v, "grammar", "grammar format");
        if (!(grammar instanceof String) || ((String) grammar).isEmpty()) {
            throw new IllegalArgumentException("structural tag grammar format needs a non-empty grammar string");
        }
        Object syntax = v.has("syntax") ? v.raw("syntax") : "ebnf";
        if ("lark".equals(syntax)) {
            return rules.merge(LarkGrammar.compile((String) grammar, whitespace, vocab), nextName("lark-"));
        }
        if (!"ebnf".equals(syntax)) {
            throw new IllegalArgumentException("structural tag grammar syntax '" + syntax
                    + "' is not supported; the engine parses \"ebnf\" and \"lark\"");
        }
        return rules.merge(GbnfParser.parse((String) grammar, vocab), nextName("ebnf-"));
    }

    /**
     * A regex compiled with this engine's ECMAScript subset, matched whole; {@code .} matches any code point, line
     * breaks included, as xgrammar's regex converter writes it.
     */
    private int regex(JsonView v) {
        Object pattern = require(v, "pattern", "regex format");
        if (!(pattern instanceof String) || ((String) pattern).isEmpty()) {
            throw new IllegalArgumentException("structural tag regex format needs a non-empty pattern string");
        }
        return rules.merge(RegexToGbnf.compile((String) pattern, Grammar.ROOT_RULE, true), nextName("regex-"));
    }

    /** The elements in order ({@code sequence}) or as alternatives ({@code or}). */
    private int elements(JsonView v, String path, List<String> ends, List<Integer> endTokens, boolean inOrder) {
        Object raw = require(v, "elements", inOrder ? "sequence format" : "or format");
        if (!(raw instanceof List)) {
            throw new IllegalArgumentException("structural tag " + (inOrder ? "sequence" : "or")
                    + " format needs an elements array");
        }
        List<?> list = (List<?>) raw;
        if (list.isEmpty()) {
            throw new IllegalArgumentException("structural tag " + (inOrder ? "sequence" : "or")
                    + " format needs at least one element");
        }
        List<Integer> refs = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            refs.add(format(list.get(i), path + ".elements[" + i + "]", ends, endTokens));
        }
        return inOrder ? rules.rule(nextName("sequence"), StructuralTagRules.refSequence(
                refs.stream().mapToInt(Integer::intValue).toArray()))
                : rules.alternation(nextName("or"), refs);
    }

    /** The tags run back to back, separated by the separator, with no free text. */
    private int tagsWithSeparator(JsonView v, String path) {
        List<Object> tags = rawList(v, "tags");
        if (tags.isEmpty()) {
            throw new IllegalArgumentException(
                    "structural tag format tags_with_separator needs at least one tag");
        }
        Object separator = require(v, "separator", "tags_with_separator format");
        if (!(separator instanceof String)) {
            throw new IllegalArgumentException(
                    "structural tag tags_with_separator format needs a separator string");
        }
        boolean atLeastOne = boolField(v, "at_least_one");
        boolean stopAfterFirst = boolField(v, "stop_after_first");
        List<Integer> tagRules = new ArrayList<>(tags.size());
        for (int i = 0; i < tags.size(); i++) {
            tagRules.add(tagRule(tagView(tags.get(i), path + ".tags[" + i + "]"), path + ".tags[" + i + "]"));
        }
        int allTags = rules.alternation(nextName("tags"), tagRules);
        if (stopAfterFirst) {
            List<Integer> root = new ArrayList<>(2);
            root.add(allTags);
            if (!atLeastOne) {
                root.add(rules.rule(nextName("empty"), new ArrayList<>()));
            }
            return rules.alternation(nextName("tags-with-separator"), root);
        }
        int sub = rules.reserve(nextName("tags-sep"));
        // sub ::= sep tags sub | "" ; the wrapper keeps a nullable tags rule from reading as left recursion
        int wrap = rules.rule(nextName("tags-wrap"), Collections.singletonList(
                new GrammarElement(GrammarElement.Type.RULE_REF, allTags)));
        List<GrammarElement> subBody = new ArrayList<>(4);
        subBody.addAll(StructuralTagRules.codePoints((String) separator));
        subBody.add(new GrammarElement(GrammarElement.Type.RULE_REF, wrap));
        subBody.add(new GrammarElement(GrammarElement.Type.RULE_REF, sub));
        subBody.add(new GrammarElement(GrammarElement.Type.ALT, 0));
        rules.fill(sub, subBody);
        List<GrammarElement> root = new ArrayList<>(4);
        root.add(new GrammarElement(GrammarElement.Type.RULE_REF, allTags));
        root.add(new GrammarElement(GrammarElement.Type.RULE_REF, sub));
        if (!atLeastOne) {
            root.add(new GrammarElement(GrammarElement.Type.ALT, 0));
            root.add(new GrammarElement(GrammarElement.Type.RULE_REF,
                    rules.rule(nextName("empty"), new ArrayList<>())));
        }
        return rules.rule(nextName("tags-with-separator"), root);
    }

    private int optional(JsonView v, String path, List<String> ends, List<Integer> endTokens) {
        int content = contentRule(v, path, ends, endTokens, "optional format");
        List<GrammarElement> body = new ArrayList<>(2);
        body.add(new GrammarElement(GrammarElement.Type.RULE_REF, content));
        body.add(new GrammarElement(GrammarElement.Type.ALT, 0));
        return rules.rule(nextName("optional"), body);
    }

    private int repeat(JsonView v, String path, List<String> ends, List<Integer> endTokens) {
        Object rawMin = require(v, "min", "repeat format");
        Object rawMax = require(v, "max", "repeat format");
        int min = repeatBound(rawMin, "structural tag repeat min");
        int max = repeatBound(rawMax, "structural tag repeat max");
        if (min < 0) {
            throw new IllegalArgumentException("structural tag repeat min must be >= 0, got " + min);
        }
        if (max >= 0 && min > max) {
            throw new IllegalArgumentException("structural tag repeat min " + min + " is above max " + max);
        }
        if (max < -1) {
            throw new IllegalArgumentException("structural tag repeat max must be -1 (unbounded) or >= 0");
        }
        boolean unbounded = max < 0 || max - min > GrammarLimits.MAX_REPETITION_EXPANSION;
        return repeated(v, path, ends, endTokens, min, unbounded ? -1 : max);
    }

    /** {@code optional}, {@code plus}, {@code star} and {@code repeat} share this repetition shape. */
    private int repeated(JsonView v, String path, List<String> ends, List<Integer> endTokens, int min, int max) {
        int content = contentRule(v, path, ends, endTokens,
                min == 0 && max < 0 ? "star format" : min == 1 && max < 0 ? "plus format"
                        : max == 1 && min <= 1 ? "optional format" : "repeat format");
        List<GrammarElement> body = new ArrayList<>(Math.max(min, 1) + 2);
        for (int i = 0; i < min; i++) {
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, content));
        }
        if (max < 0) {
            int loop = rules.reserve(nextName("repeat-loop"));
            // the wrapper keeps a directly nullable content from reading as left recursion
            int wrap = rules.rule(nextName("repeat-wrap"), Collections.singletonList(
                    new GrammarElement(GrammarElement.Type.RULE_REF, content)));
            List<GrammarElement> loopBody = new ArrayList<>(3);
            loopBody.add(new GrammarElement(GrammarElement.Type.RULE_REF, wrap));
            loopBody.add(new GrammarElement(GrammarElement.Type.RULE_REF, loop));
            loopBody.add(new GrammarElement(GrammarElement.Type.ALT, 0));
            rules.fill(loop, loopBody);
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, loop));
        } else {
            int extra = max - min;
            int chain = rules.rule(nextName("repeat-opt"), Collections.emptyList());
            List<GrammarElement> last = new ArrayList<>(2);
            last.add(new GrammarElement(GrammarElement.Type.RULE_REF, content));
            last.add(new GrammarElement(GrammarElement.Type.ALT, 0));
            rules.fill(chain, last);
            for (int i = 1; i < extra; i++) {
                int prev = chain;
                chain = rules.rule(nextName("repeat-opt"), Arrays.asList(
                        new GrammarElement(GrammarElement.Type.RULE_REF, content),
                        new GrammarElement(GrammarElement.Type.RULE_REF, prev),
                        new GrammarElement(GrammarElement.Type.ALT, 0)));
            }
            if (extra > 0) {
                body.add(new GrammarElement(GrammarElement.Type.RULE_REF, chain));
            }
        }
        return rules.rule(nextName("repeat"), body);
    }

    /** Exactly one token. */
    private int token(JsonView v) {
        return rules.token(nextName("token"),
                tokenId(require(v, "token", "token format"), "structural tag format token"));
    }
    /** Exactly one token outside the excluded set, plus the end token of the tag around it; any token if none. */
    private int excludeToken(JsonView v, List<Integer> endTokens) {
        return rules.anyTokenBut(nextName("exclude-token"), mergedTokens(v, endTokens));
    }

    /**
     * Zero or more tokens outside the excluded set, plus the end token of the tag around it (any tokens if none),
     * at most {@code max_tokens} of them.
     */
    private int anyTokens(JsonView v, List<Integer> endTokens) {
        int maxTokens = budget(v, "any_tokens", "max_tokens");
        int inner = rules.anyTokenBut(nextName("any-token"), mergedTokens(v, endTokens));
        int self = rules.reserve(nextName("any-tokens"));
        List<GrammarElement> body = new ArrayList<>(3);
        body.add(new GrammarElement(GrammarElement.Type.RULE_REF, inner));
        body.add(new GrammarElement(GrammarElement.Type.RULE_REF, self));
        body.add(new GrammarElement(GrammarElement.Type.ALT, 0));
        rules.fill(self, body);
        return rules.budgeted(nextName("any-tokens-budget"), maxTokens, -1, self);
    }

    // ------------------------------------------------------------------ tag pieces

    /** One tag node of a {@code tags} array: begin + content + end. */
    private int tagRule(JsonView tag, String path) {
        Object content = require(tag, "content", "tag format");
        Object end = require(tag, "end", "tag format");
        List<String> endStrings = new ArrayList<>();
        Integer endToken = null;
        if (end instanceof String) {
            endStrings.add((String) end);
        } else if (end instanceof List) {
            List<?> list = (List<?>) end;
            if (list.isEmpty()) {
                throw new IllegalArgumentException("structural tag " + path + " end array cannot be empty");
            }
            for (int i = 0; i < list.size(); i++) {
                if (!(list.get(i) instanceof String)) {
                    throw new IllegalArgumentException("structural tag " + path + " end " + i + " is not a string");
                }
                endStrings.add((String) list.get(i));
            }
        } else if (end instanceof Map) {
            endToken = tokenMember(end, "structural tag " + path + " end");
        } else {
            throw new IllegalArgumentException("structural tag " + path + " end must be a string, an array of"
                    + " strings, or a token object");
        }
        if (endToken == null && unlimited(content) && !excluded(content)
                && endStrings.stream().allMatch(String::isEmpty)) {
            throw new IllegalArgumentException("structural tag " + path + ": its content is unlimited, so at"
                    + " least one end string must be non-empty");
        }
        int contentRule = format(content, path + ".content", endToken == null ? nonEmpty(endStrings)
                : Collections.emptyList(), endToken == null ? Collections.emptyList()
                : Collections.singletonList(endToken));
        int endRule = endToken != null ? rules.token(nextName("tag-end"), endToken) : endRule(endStrings);
        int begin = beginRule(tag, path);
        return rules.rule(nextName("tag"), StructuralTagRules.refSequence(begin, contentRule, endRule));
    }

    /** The begin rule of a tag: a literal, or one token. */
    private int beginRule(JsonView v, String path) {
        Object begin = require(v, "begin", "tag format");
        if (begin instanceof String) {
            return rules.literal(nextName("tag-begin"), (String) begin);
        }
        if (begin instanceof Map) {
            return rules.token(nextName("tag-begin"),
                    tokenMember(begin, "structural tag " + path + " begin"));
        }
        throw new IllegalArgumentException("structural tag " + path + " begin must be a string or a token object");
    }

    /** The token id of a token object ({@code {"type": "token", "token": id-or-string}}). */
    private int tokenMember(Object object, String at) {
        Object token = ((Map<?, ?>) object).get("token");
        if (token == null) {
            throw new IllegalArgumentException(at + " needs a token field");
        }
        return tokenId(token, at);
    }

    /** The end rule: one literal, several literals, or one token; an empty end matches the empty string. */
    private int endRule(List<String> endStrings) {
        if (endStrings.size() == 1) {
            return rules.literal(nextName("tag-end"), endStrings.get(0));
        }
        List<Integer> literals = new ArrayList<>(endStrings.size());
        for (String end : endStrings) {
            literals.add(rules.literal(nextName("tag-end"), end));
        }
        return rules.alternation(nextName("tag-ends"), literals);
    }

    /** The one trigger {@code begin} starts with. */
    private String matchExactlyOneTrigger(String begin, List<String> triggers, int index) {
        int matches = 0;
        String matched = null;
        for (String trigger : triggers) {
            if (begin.startsWith(trigger)) {
                matches++;
                matched = trigger;
            }
        }
        if (matches == 0) {
            throw new IllegalArgumentException("structural tag format tag " + index + " begins with '" + begin
                    + "', which no trigger " + triggers + " opens");
        }
        if (matches > 1) {
            throw new IllegalArgumentException("structural tag format tag " + index + " begins with '" + begin
                    + "', which several triggers open; one tag must match exactly one trigger");
        }
        return matched;
    }

    private void matchExactlyOneTokenTrigger(int begin, List<Integer> triggerTokens, int index) {
        int matches = 0;
        for (int trigger : triggerTokens) {
            if (trigger == begin) {
                matches++;
            }
        }
        if (matches == 0) {
            throw new IllegalArgumentException("structural tag format tag " + index + " begins with token "
                    + begin + ", which no trigger token opens");
        }
        if (matches > 1) {
            throw new IllegalArgumentException("structural tag format tag " + index + " begins with token "
                    + begin + ", which several trigger tokens open; one tag must match exactly one trigger");
        }
    }

    /** The tag's begin: a string for the string dispatch, or a token object for the token dispatch. */
    private String stringBegin(JsonView tag, int index, String format) {
        Object begin = require(tag, "begin", "tag format");
        if (!(begin instanceof String)) {
            throw new IllegalArgumentException("structural tag format " + format + " tag " + index
                    + " needs a string begin" + (begin instanceof Map ? "; token begins dispatch through"
                    + " token_triggered_tags" : ""));
        }
        return (String) begin;
    }

    /** The tag's begin: a token object for the token dispatch, or a string for the string dispatch. */
    private int tokenBegin(JsonView tag, int index) {
        Object begin = require(tag, "begin", "tag format");
        if (begin instanceof Map) {
            return tokenMember(begin, "structural tag format token_triggered_tags tag " + index + " begin");
        }
        throw new IllegalArgumentException("structural tag format token_triggered_tags tag " + index
                + " needs a token begin" + (begin instanceof String ? "; string begins dispatch through"
                + " triggered_tags" : ""));
    }

    // ------------------------------------------------------------------ value helpers

    /** The repeated format's content compiled to its rule index, shared by optional/plus/star/repeat. */
    private int contentRule(JsonView v, String path, List<String> ends, List<Integer> endTokens, String what) {
        Object content = require(v, "content", what);
        return format(content, path + ".content", ends, endTokens);
    }

    private int[] mergedTokens(JsonView v, List<Integer> endTokens) {
        Set<Integer> ids = new LinkedHashSet<>(tokenList(v, "exclude_tokens", "token format"));
        ids.addAll(endTokens);
        int[] out = new int[ids.size()];
        int i = 0;
        for (int id : ids) {
            out[i++] = id;
        }
        return out;
    }

    /** The token ids of a member of ints or token strings; a string resolves through the vocabulary. */
    private List<Integer> tokenList(JsonView v, String key, String what) {
        Object raw = v.raw(key);
        List<Integer> out = new ArrayList<>();
        if (raw == null) {
            return out;
        }
        if (!(raw instanceof List)) {
            throw new IllegalArgumentException("structural tag " + what + " " + key + " must be an array of"
                    + " token ids or token strings");
        }
        List<?> list = (List<?>) raw;
        for (int i = 0; i < list.size(); i++) {
            out.add(tokenId(list.get(i), "structural tag " + what + " " + key + "[" + i + "]"));
        }
        return out;
    }

    /** One token id: a non-negative integer, or a token string resolving to exactly one vocabulary token. */
    private int tokenId(Object value, String at) {
        if (isIntegral(value)) {
            long id = ((Number) value).longValue();
            if (id < 0 || id > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(at + " must be a non-negative token id, got " + value);
            }
            return (int) id;
        }
        if (value instanceof String) {
            String token = (String) value;
            if (token.isEmpty()) {
                throw new IllegalArgumentException(at + " must not be an empty token string");
            }
            if (vocab == null) {
                throw new IllegalArgumentException(at + " names the token '" + token + "', which needs the model"
                        + " vocabulary, and this compile received none");
            }
            int[] ids = vocab.tokenize(token);
            if (ids.length != 1) {
                throw new IllegalArgumentException(at + " token '" + token + "' does not tokenize to exactly one"
                        + " token");
            }
            return ids[0];
        }
        throw new IllegalArgumentException(at + " must be a token id or a token string");
    }

    private static boolean isIntegral(Object value) {
        if (value instanceof Long || value instanceof Integer || value instanceof BigInteger) {
            return true;
        }
        return value instanceof Double && ((Double) value) == Math.rint((Double) value);
    }

    private int repeatBound(Object value, String at) {
        if (!isIntegral(value)) {
            throw new IllegalArgumentException(at + " must be an integer, got " + value);
        }
        long bound = ((Number) value).longValue();
        if (bound < -1 || bound > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(at + " must be -1 (unbounded) or a repetition count that fits"
                    + " an int, got " + bound);
        }
        return (int) bound;
    }

    /** The region budget {@code key} ({@code max_tokens}, {@code max_chars}): 0 to 2^31-1, or -1 when unset. */
    private int budget(JsonView v, String what, String key) {
        Object raw = v.raw(key);
        if (raw == null) {
            return -1;
        }
        if (!isIntegral(raw) || ((Number) raw).longValue() < 0 || ((Number) raw).longValue() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("structural tag " + what + " " + key + " must be an integer from 0 to "
                    + Integer.MAX_VALUE + ", got " + raw);
        }
        return (int) ((Number) raw).longValue();
    }

    private List<String> nonEmptyStrings(JsonView v, String key, String what) {
        Object raw = v.raw(key);
        if (raw == null) {
            return Collections.emptyList();
        }
        if (!(raw instanceof List)) {
            throw new IllegalArgumentException("structural tag " + what + " " + key + " must be an array of"
                    + " non-empty strings");
        }
        List<?> list = (List<?>) raw;
        List<String> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            Object s = list.get(i);
            if (!(s instanceof String) || ((String) s).isEmpty()) {
                throw new IllegalArgumentException("structural tag " + what + " " + key + "[" + i + "] is not a"
                        + " non-empty string");
            }
            out.add((String) s);
        }
        return out;
    }

    private boolean boolField(JsonView v, String key) {
        return boolField(v, key, false);
    }

    private boolean boolField(JsonView v, String key, boolean dflt) {
        Object raw = v.raw(key);
        if (raw == null) {
            return dflt;
        }
        if (!(raw instanceof Boolean)) {
            throw new IllegalArgumentException("structural tag format " + key + " must be a boolean");
        }
        return (Boolean) raw;
    }

    private List<Object> rawList(JsonView v, String key) {
        Object raw = v.raw(key);
        if (!(raw instanceof List)) {
            throw new IllegalArgumentException("structural tag format " + key + " must be an array");
        }
        return new ArrayList<>((List<?>) raw);
    }

    private Object require(JsonView v, String key, String what) {
        Object raw = v.raw(key);
        if (raw == null && !v.has(key)) {
            throw new IllegalArgumentException("structural tag " + what + " needs a " + key + " field");
        }
        return raw;
    }

    private static JsonView view(Object value, String path) {
        Map<?, ?> map = (Map<?, ?>) value;
        Map<String, Object> typed = new java.util.LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            typed.put(String.valueOf(e.getKey()), e.getValue());
        }
        return JsonView.of(typed, "structural_tag", path);
    }

    private static JsonView tagView(Object value, String path) {
        if (!(value instanceof Map)) {
            throw new IllegalArgumentException("structural tag " + path
                    + " must be a tag object with begin, content and end");
        }
        return view(value, path);
    }

    private static String typeOf(JsonView v) {
        Object type = v.raw("type");
        if (type == null) {
            return null;
        }
        if (!(type instanceof String)) {
            throw new IllegalArgumentException("structural tag format type must be a string");
        }
        return (String) type;
    }

    private static boolean dispatching(String type) {
        return "triggered_tags".equals(type) || "dispatch".equals(type)
                || "token_triggered_tags".equals(type) || "token_dispatch".equals(type);
    }

    /** xgrammar's unlimited-content analysis over the raw format node. */
    private static boolean unlimited(Object value) {
        if (!(value instanceof Map)) {
            return false;
        }
        Map<?, ?> map = (Map<?, ?>) value;
        String type = map.get("type") instanceof String ? (String) map.get("type") : null;
        switch (type == null ? "" : type) {
            case "any_text": case "triggered_tags": case "token_triggered_tags": case "dispatch":
            case "token_dispatch": case "any_tokens": case "tags_with_separator": case "star": case "plus":
                return true;
            case "sequence": case "or": {
                Object elements = map.get("elements");
                if (!(elements instanceof List)) {
                    return false;
                }
                for (Object element : (List<?>) elements) {
                    if (unlimited(element) && !excluded(element)) {
                        return true;
                    }
                }
                return false;
            }
            case "optional":
                return unlimited(map.get("content"));
            case "repeat": {
                Object max = map.get("max");
                if (isIntegral(max) && ((Number) max).longValue() == 0) {
                    return false;
                }
                if (isIntegral(max) && ((Number) max).longValue() < 0) {
                    return true;
                }
                return unlimited(map.get("content"));
            }
            default:
                return false;
        }
    }

    /** xgrammar's excluded-content analysis over the raw format node. */
    private static boolean excluded(Object value) {
        if (!(value instanceof Map)) {
            return false;
        }
        Map<?, ?> map = (Map<?, ?>) value;
        String type = map.get("type") instanceof String ? (String) map.get("type") : null;
        String field = "token_triggered_tags".equals(type) || "token_dispatch".equals(type)
                || "any_tokens".equals(type) ? "exclude_tokens" : "excludes";
        return map.get(field) instanceof List && !((List<?>) map.get(field)).isEmpty();
    }

    private static List<String> nonEmpty(List<String> strings) {
        List<String> out = new ArrayList<>(strings.size());
        for (String s : strings) {
            if (!s.isEmpty()) {
                out.add(s);
            }
        }
        return out;
    }

    private Compiled finish(int rootRule, GrammarTriggers triggers, List<String> excludes, int[] excludeTokens) {
        return new Compiled(rules.finish(Collections.singletonList(
                new GrammarElement(GrammarElement.Type.RULE_REF, rootRule))), triggers, excludes, excludeTokens);
    }

    /** The next unique rule-name prefix, one per compiled format node. */
    private String nextName(String hint) {
        return "st-f" + (nodes++) + "-" + hint;
    }
}
