package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The immutable constraint of one generation request, the sum type the request wiring passes
 * around: {@code Gbnf | Lark | JsonSchema | JsonObject | Regex |
 * Choice | StructuralTag}. Each instance names its {@link Kind} and carries only that kind's
 * data; the conversion into a {@link Grammar} lives with {@link GbnfBuilder}, which dispatches
 * on the kind ({@link JsonSchemaToGbnf} for the schema kinds, {@link RegexToGbnf} for
 * {@link Kind#REGEX}, an alternation of literals for {@link Kind#CHOICE}, {@link LarkGrammar}
 * for Lark and {@link GbnfParser} for raw GBNF). Invalid or untranslatable requests surface as
 * {@link GrammarException} there, never here: construction only checks shape.
 *
 * <p>The {@link Kind#STRUCTURAL_TAG} variant carries the raw structural-tag text in either form vLLM accepts:
 * the legacy {@code structures}/{@code triggers} payload ({@link StructuralTag}) or xgrammar's
 * {@code {"type": "structural_tag", "format": ...}} tree of formats ({@link StructuralTagFormat}).</p>
 *
 * <p>The {@link Kind#JSON_SCHEMA} and {@link Kind#JSON_OBJECT} variants carry the structured-output JSON options of
 * vLLM's {@code StructuredOutputsParams}: the whitespace mode ({@link #whitespace}), an explicit
 * {@code whitespace_pattern} regex ({@link #whitespacePattern}) and {@code disable_additional_properties}
 * ({@link #disableAdditionalProperties}), applied together by {@link #withJsonOptions}.</p>
 */
public final class ConstraintSpec {

    /** Which constraint kind this instance is. */
    public enum Kind {
        /**
         * A grammar text: GBNF, or Lark when it starts with llama.cpp's {@code %llguidance} header or has no line
         * defining a {@code ::=} rule ({@link LarkGrammar#isLark}, vLLM's detection plus llama.cpp's header),
         * translated by {@link LarkGrammar}.
         */
        GBNF,
        /** A Lark grammar text, the dialect vLLM's xgrammar backend compiles. */
        LARK,
        /** A JSON schema text. */
        JSON_SCHEMA,
        /** The generic JSON-value grammar, no schema constraints. */
        JSON_OBJECT,
        /** A regex the whole output must match (ECMAScript syntax, anchors optional). */
        REGEX,
        /** One of the given literal strings. */
        CHOICE,
        /** A structural tag: legacy structures, or xgrammar's format tree. */
        STRUCTURAL_TAG
    }

    private final Kind kind;
    private final String text;
    private final List<String> choices;
    private final boolean dotAll;
    private final JsonSchemaToGbnf.WhitespaceMode whitespace;
    private final String whitespacePattern;
    private final boolean disableAdditionalProperties;
    private final GrammarTriggers triggers;
    private final ConstraintGate gate;

    private ConstraintSpec(Kind kind, String text, List<String> choices, boolean dotAll,
            JsonSchemaToGbnf.WhitespaceMode whitespace) {
        this(kind, text, choices, dotAll, whitespace, null, false, null, null);
    }

    private ConstraintSpec(Kind kind, String text, List<String> choices, boolean dotAll,
            JsonSchemaToGbnf.WhitespaceMode whitespace, GrammarTriggers triggers, ConstraintGate gate) {
        this(kind, text, choices, dotAll, whitespace, null, false, triggers, gate);
    }

    private ConstraintSpec(Kind kind, String text, List<String> choices, boolean dotAll,
            JsonSchemaToGbnf.WhitespaceMode whitespace, String whitespacePattern,
            boolean disableAdditionalProperties, GrammarTriggers triggers, ConstraintGate gate) {
        this.kind = kind;
        this.text = text;
        this.choices = choices == null ? null : Collections.unmodifiableList(new ArrayList<>(choices));
        this.dotAll = dotAll;
        this.whitespace = whitespace;
        this.whitespacePattern = whitespacePattern;
        this.disableAdditionalProperties = disableAdditionalProperties;
        this.triggers = triggers;
        this.gate = gate;
    }

    /**
     * This constraint gated by {@code triggers}: the output runs free until a trigger fires, and the grammar
     * constrains it from the trigger on ({@link GrammarTriggers}); null returns the always-on form.
     */
    public ConstraintSpec withTriggers(GrammarTriggers triggers) {
        return new ConstraintSpec(kind, text, choices, dotAll, whitespace, whitespacePattern,
                disableAdditionalProperties, triggers, gate);
    }

    /**
     * This constraint behind {@code gate}: free output until a gate word, then the gate's prefix and this grammar
     * ({@link ConstraintGate}); null removes the gate. A constraint that carries its own triggers (or a structural
     * tag whose payload names triggers) is lazy already and compiles without the gate.
     */
    public ConstraintSpec withGate(ConstraintGate gate) {
        return new ConstraintSpec(kind, text, choices, dotAll, whitespace, whitespacePattern,
                disableAdditionalProperties, triggers, gate);
    }

    /**
     * This JSON constraint with vLLM's {@code structured_outputs} JSON options: {@code disableAnyWhitespace} writes
     * the compact form (its {@link JsonSchemaToGbnf.WhitespaceMode#COMPACT}), {@code whitespacePattern} (when not
     * null) makes that regex the whitespace between JSON tokens, and {@code disableAdditionalProperties} closes
     * every object that declares properties.
     *
     * @throws IllegalArgumentException when an option is requested for another constraint kind, where neither the
     *                                  whitespace options nor the closing of objects has meaning (a regex, choice,
     *                                  grammar or structural tag), or when the pattern is empty
     */
    public ConstraintSpec withJsonOptions(boolean disableAnyWhitespace, String whitespacePattern,
            boolean disableAdditionalProperties) {
        boolean requested = disableAnyWhitespace || whitespacePattern != null || disableAdditionalProperties;
        if (requested && kind != Kind.JSON_SCHEMA && kind != Kind.JSON_OBJECT) {
            throw new IllegalArgumentException("the JSON output options disable_any_whitespace, whitespace_pattern"
                    + " and disable_additional_properties apply to json and json_object constraints only, not to "
                    + kind);
        }
        if (whitespacePattern != null && whitespacePattern.isEmpty()) {
            throw new IllegalArgumentException("whitespace_pattern must not be empty");
        }
        JsonSchemaToGbnf.WhitespaceMode effective = disableAnyWhitespace
                ? JsonSchemaToGbnf.WhitespaceMode.COMPACT : whitespace;
        return new ConstraintSpec(kind, text, choices, dotAll, effective, whitespacePattern,
                disableAdditionalProperties, triggers, gate);
    }

    /** The gate the constraint waits behind, or null. */
    public ConstraintGate gate() {
        return gate;
    }

    /**
     * The triggers gating the grammar, or null when it constrains the output from the first token. A
     * {@link Kind#STRUCTURAL_TAG} payload's own {@code triggers} are not reported here: they compile with the
     * payload ({@link GbnfBuilder#compile}).
     */
    public GrammarTriggers triggers() {
        return triggers;
    }

    /** A {@link Kind#GBNF} constraint: GBNF text under {@link GbnfParser} bounds, or a Lark grammar. */
    public static ConstraintSpec gbnf(String grammarSource) {
        Objects.requireNonNull(grammarSource, "grammarSource");
        return new ConstraintSpec(Kind.GBNF, grammarSource, null, false,
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    /**
     * A {@link Kind#LARK} constraint: the Lark grammar text {@link LarkGrammar} compiles, its {@code %json}
     * schemas in the flexible whitespace mode.
     */
    public static ConstraintSpec lark(String grammarSource) {
        Objects.requireNonNull(grammarSource, "grammarSource");
        return new ConstraintSpec(Kind.LARK, grammarSource, null, false,
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    /**
     * The constraint of a grammar text in either syntax, as vLLM reads {@code structured_outputs.grammar}:
     * {@link Kind#LARK} when the text looks like Lark ({@link LarkGrammar#isLark}: llama.cpp's {@code %llguidance}
     * header, or no {@code ::=} rule line), else {@link Kind#GBNF}.
     */
    public static ConstraintSpec grammar(String grammarSource) {
        Objects.requireNonNull(grammarSource, "grammarSource");
        return LarkGrammar.isLark(grammarSource) ? lark(grammarSource) : gbnf(grammarSource);
    }

    /**
     * A {@link Kind#JSON_SCHEMA} constraint in {@code mode}; {@code mode} null means
     * llama.cpp's default, {@link JsonSchemaToGbnf.WhitespaceMode#FLEXIBLE}.
     */
    public static ConstraintSpec jsonSchema(String schemaJson,
            JsonSchemaToGbnf.WhitespaceMode mode) {
        Objects.requireNonNull(schemaJson, "schemaJson");
        return new ConstraintSpec(Kind.JSON_SCHEMA, schemaJson, null, false,
                mode == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : mode);
    }

    /**
     * A {@link Kind#JSON_SCHEMA} constraint under the structured-output JSON options of
     * {@link #withJsonOptions}: {@code mode} (null is flexible), the {@code whitespace_pattern} regex (null for the
     * mode's rule) and {@code disableAdditionalProperties}.
     */
    public static ConstraintSpec jsonSchema(String schemaJson, JsonSchemaToGbnf.WhitespaceMode mode,
            String whitespacePattern, boolean disableAdditionalProperties) {
        Objects.requireNonNull(schemaJson, "schemaJson");
        return new ConstraintSpec(Kind.JSON_SCHEMA, schemaJson, null, false,
                mode == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : mode, whitespacePattern,
                disableAdditionalProperties, null, null);
    }

    /** A {@link Kind#JSON_OBJECT} constraint in {@code mode}; null {@code mode} is flexible. */
    public static ConstraintSpec jsonObject(JsonSchemaToGbnf.WhitespaceMode mode) {
        return new ConstraintSpec(Kind.JSON_OBJECT, "{}", null, false,
                mode == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : mode);
    }

    /** A {@link Kind#JSON_OBJECT} constraint under the structured-output JSON options of {@link #withJsonOptions}. */
    public static ConstraintSpec jsonObject(JsonSchemaToGbnf.WhitespaceMode mode, String whitespacePattern,
            boolean disableAdditionalProperties) {
        return new ConstraintSpec(Kind.JSON_OBJECT, "{}", null, false,
                mode == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : mode, whitespacePattern,
                disableAdditionalProperties, null, null);
    }

    /**
     * A {@link Kind#REGEX} constraint: the whole output must match {@code pattern}, an
     * ECMAScript regex as {@link RegexToGbnf#compile} reads it; {@code dotAll} sets llama.cpp's
     * {@code common_grammar_options.dotall} ({@code .} matches newlines too).
     */
    public static ConstraintSpec regex(String pattern, boolean dotAll) {
        Objects.requireNonNull(pattern, "pattern");
        return new ConstraintSpec(Kind.REGEX, pattern, null, dotAll,
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    /** A {@link Kind#CHOICE} constraint: exactly one of {@code choices}. */
    public static ConstraintSpec choice(List<String> choices) {
        if (choices == null || choices.isEmpty()) {
            throw new IllegalArgumentException("choice constraint needs at least one choice");
        }
        for (String choice : choices) {
            if (choice == null) {
                throw new IllegalArgumentException("choices must not contain null");
            }
        }
        return new ConstraintSpec(Kind.CHOICE, null, choices, false,
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    /**
     * A {@link Kind#STRUCTURAL_TAG} constraint carrying the raw tag text; {@link StructuralTag} parses it into
     * the begin/end segments and per-segment schemas.
     */
    public static ConstraintSpec structuralTag(String tagText) {
        Objects.requireNonNull(tagText, "tagText");
        return new ConstraintSpec(Kind.STRUCTURAL_TAG, tagText, null, false,
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    /** The kind of this constraint. */
    public Kind kind() {
        return kind;
    }

    /** The source text: GBNF, Lark, schema JSON, regex pattern or structural tag, per {@code kind}. */
    public String text() {
        return text;
    }

    /** The {@link Kind#CHOICE} literals; empty for the other kinds. */
    public List<String> choices() {
        return choices == null ? List.of() : choices;
    }

    /** Whether {@link Kind#REGEX} runs dotAll. */
    public boolean dotAll() {
        return dotAll;
    }

    /** The whitespace mode of the JSON kinds. */
    public JsonSchemaToGbnf.WhitespaceMode whitespace() {
        return whitespace;
    }

    /** The {@code whitespace_pattern} regex of the JSON kinds, or null when the mode's rule applies. */
    public String whitespacePattern() {
        return whitespacePattern;
    }

    /** Whether the JSON kinds close every object that declares properties ({@code disable_additional_properties}). */
    public boolean disableAdditionalProperties() {
        return disableAdditionalProperties;
    }

    /** The conversion options of a JSON kind, carrying the whitespace pattern and the closing of objects. */
    JsonSchemaOptions jsonOptions() {
        return JsonSchemaOptions.json(whitespace).withWhitespacePattern(whitespacePattern)
                .withAdditionalPropertiesDisabled(disableAdditionalProperties);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ConstraintSpec)) {
            return false;
        }
        ConstraintSpec other = (ConstraintSpec) o;
        return kind == other.kind
                && dotAll == other.dotAll
                && disableAdditionalProperties == other.disableAdditionalProperties
                && whitespace == other.whitespace
                && Objects.equals(whitespacePattern, other.whitespacePattern)
                && Objects.equals(text, other.text)
                && Objects.equals(choices, other.choices)
                && Objects.equals(triggers, other.triggers)
                && Objects.equals(gate, other.gate);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, text, choices, dotAll, whitespace, whitespacePattern,
                disableAdditionalProperties, triggers, gate);
    }

    @Override
    public String toString() {
        return "ConstraintSpec[" + kind
                + (text == null ? "" : ", text=" + (kind == Kind.JSON_SCHEMA
                        || kind == Kind.STRUCTURAL_TAG ? "<" + text.length() + " chars>" : text))
                + (choices == null ? "" : ", choices=" + choices)
                + (kind == Kind.REGEX ? ", dotAll=" + dotAll : "")
                + (whitespacePattern == null ? "" : ", whitespacePattern=" + whitespacePattern)
                + (disableAdditionalProperties ? ", disableAdditionalProperties" : "")
                + (triggers == null ? "" : ", triggers=" + triggers)
                + (gate == null ? "" : ", gate=" + gate)
                + "]";
    }
}
