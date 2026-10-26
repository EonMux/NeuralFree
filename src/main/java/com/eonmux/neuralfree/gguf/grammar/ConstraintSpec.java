package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The immutable constraint of one generation request, the sum type row G3/G4/G8 and the PARITY
 * grammar epic request wiring pass around: {@code Gbnf | JsonSchema | JsonObject | Regex |
 * Choice | StructuralTag}. Each instance names its {@link Kind} and carries only that kind's
 * data; the conversion into a {@link Grammar} lives with {@link GbnfBuilder}, which dispatches
 * on the kind ({@link JsonSchemaToGbnf} for the schema kinds, {@link RegexToGbnf} for
 * {@link Kind#REGEX}, an alternation of literals for {@link Kind#CHOICE} and
 * {@link GbnfParser} for raw GBNF). Invalid or untranslatable requests surface as
 * {@link GrammarException} there, never here: construction only checks shape.
 *
 * <p>The {@link Kind#STRUCTURAL_TAG} variant carries the raw structural-tag text; its parsing
 * into begin/trigger/end segments and grammars is the SO-4 WP ({@code StructuralTag}), which
 * converts the segment fragments with this package's converters. SO-3 delivers the variant,
 * SO-4 the parser.</p>
 */
public final class ConstraintSpec {

    /** Which constraint kind this instance is. */
    public enum Kind {
        /** A raw GBNF grammar text. */
        GBNF,
        /** A JSON schema text. */
        JSON_SCHEMA,
        /** The generic JSON-value grammar, no schema constraints. */
        JSON_OBJECT,
        /** An anchored regex over the output. */
        REGEX,
        /** One of the given literal strings. */
        CHOICE,
        /** A structural tag (begin/schema/end), parsed by SO-4. */
        STRUCTURAL_TAG
    }

    private final Kind kind;
    private final String text;
    private final List<String> choices;
    private final boolean dotAll;
    private final JsonSchemaToGbnf.WhitespaceMode whitespace;

    private ConstraintSpec(Kind kind, String text, List<String> choices, boolean dotAll,
            JsonSchemaToGbnf.WhitespaceMode whitespace) {
        this.kind = kind;
        this.text = text;
        this.choices = choices == null ? null : Collections.unmodifiableList(new ArrayList<>(choices));
        this.dotAll = dotAll;
        this.whitespace = whitespace;
    }

    /** A {@link Kind#GBNF} constraint; the text under {@link GbnfParser} bounds. */
    public static ConstraintSpec gbnf(String grammarSource) {
        Objects.requireNonNull(grammarSource, "grammarSource");
        return new ConstraintSpec(Kind.GBNF, grammarSource, null, false,
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    /**
     * A {@link Kind#JSON_SCHEMA} constraint in {@code mode} (row G8); {@code mode} null means
     * llama.cpp's default, {@link JsonSchemaToGbnf.WhitespaceMode#FLEXIBLE}.
     */
    public static ConstraintSpec jsonSchema(String schemaJson,
            JsonSchemaToGbnf.WhitespaceMode mode) {
        Objects.requireNonNull(schemaJson, "schemaJson");
        return new ConstraintSpec(Kind.JSON_SCHEMA, schemaJson, null, false,
                mode == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : mode);
    }

    /** A {@link Kind#JSON_OBJECT} constraint in {@code mode}; null {@code mode} is flexible. */
    public static ConstraintSpec jsonObject(JsonSchemaToGbnf.WhitespaceMode mode) {
        return new ConstraintSpec(Kind.JSON_OBJECT, "{}", null, false,
                mode == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : mode);
    }

    /**
     * A {@link Kind#REGEX} constraint in llama.cpp's ECMAScript subset; {@code dotAll} sets
     * llama.cpp's {@code common_grammar_options.dotall} ({@code .} matches newlines too).
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
     * A {@link Kind#STRUCTURAL_TAG} constraint carrying the raw tag text; SO-4 parses it into
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

    /** The source text: GBNF, schema JSON, regex pattern or structural tag, per {@code kind}. */
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

    /** The whitespace mode of the JSON kinds (row G8). */
    public JsonSchemaToGbnf.WhitespaceMode whitespace() {
        return whitespace;
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
                && whitespace == other.whitespace
                && Objects.equals(text, other.text)
                && Objects.equals(choices, other.choices);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, text, choices, dotAll, whitespace);
    }

    @Override
    public String toString() {
        return "ConstraintSpec[" + kind
                + (text == null ? "" : ", text=" + (kind == Kind.JSON_SCHEMA
                        || kind == Kind.STRUCTURAL_TAG ? "<" + text.length() + " chars>" : text))
                + (choices == null ? "" : ", choices=" + choices)
                + (kind == Kind.REGEX ? ", dotAll=" + dotAll : "")
                + "]";
    }
}
