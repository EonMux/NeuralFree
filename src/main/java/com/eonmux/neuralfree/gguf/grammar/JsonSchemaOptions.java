package com.eonmux.neuralfree.gguf.grammar;

/**
 * How {@link JsonSchemaToGbnf} writes a schema's values, the knobs of vLLM's structured output requests: the
 * whitespace mode of the JSON tokens, an explicit {@code whitespace_pattern} regex where the mode is too coarse,
 * {@code disable_additional_properties} (every object that declares properties is closed), the output {@code style}
 * (standard JSON, or the top-level object as one of the {@link XmlParameterStyle XML parameter styles}),
 * {@code any_order} (every object's members in any order, keys repeatable, the count bounded by the required
 * members and {@code minProperties} / {@code maxProperties}) and {@code max_whitespace_cnt} (every whitespace run
 * at most that many of space, tab, LF and CR). Immutable.
 */
final class JsonSchemaOptions {

    /** xgrammar's whitespace between XML parameters and around their values: any run of space, LF, CR and tab. */
    static final String XML_SPACE_RULE = "[ \\t\\n\\r]*";

    private final JsonSchemaToGbnf.WhitespaceMode mode;
    private final XmlParameterStyle style;
    private final boolean anyOrder;
    private final int maxWhitespace;
    private final String whitespacePattern;
    private final boolean disableAdditionalProperties;

    private JsonSchemaOptions(JsonSchemaToGbnf.WhitespaceMode mode, XmlParameterStyle style, boolean anyOrder,
            int maxWhitespace, String whitespacePattern, boolean disableAdditionalProperties) {
        this.mode = mode;
        this.style = style;
        this.anyOrder = anyOrder;
        this.maxWhitespace = maxWhitespace;
        this.whitespacePattern = whitespacePattern;
        this.disableAdditionalProperties = disableAdditionalProperties;
    }

    /** Standard JSON in {@code mode}, members in schema order, whitespace runs as the mode allows. */
    static JsonSchemaOptions json(JsonSchemaToGbnf.WhitespaceMode mode) {
        return new JsonSchemaOptions(mode == null ? JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE : mode, null, false, -1,
                null, false);
    }

    /** These options with the top-level object written in {@code xmlStyle}; null is standard JSON. */
    JsonSchemaOptions withStyle(XmlParameterStyle xmlStyle) {
        return new JsonSchemaOptions(mode, xmlStyle, anyOrder, maxWhitespace, whitespacePattern,
                disableAdditionalProperties);
    }

    /** These options with every object's members in any order when {@code value}. */
    JsonSchemaOptions withAnyOrder(boolean value) {
        return new JsonSchemaOptions(mode, style, value, maxWhitespace, whitespacePattern,
                disableAdditionalProperties);
    }

    /**
     * These options with every whitespace run limited to {@code count} characters of space, tab, LF and CR.
     *
     * @throws IllegalArgumentException when {@code count} is not 1 to {@link GrammarLimits#MAX_REPETITION_EXPANSION}
     */
    JsonSchemaOptions withMaxWhitespace(int count) {
        if (count < 1 || count > GrammarLimits.MAX_REPETITION_EXPANSION) {
            throw new IllegalArgumentException("max_whitespace_cnt must be an integer from 1 to "
                    + GrammarLimits.MAX_REPETITION_EXPANSION + ", got " + count);
        }
        return new JsonSchemaOptions(mode, style, anyOrder, count, whitespacePattern, disableAdditionalProperties);
    }

    /**
     * These options with the {@code space} rule matching the ECMAScript regex {@code pattern}, vLLM's
     * {@code whitespace_pattern}; null keeps the mode's rule, and a non-null pattern wins over the mode.
     *
     * @throws IllegalArgumentException when {@code pattern} is empty
     */
    JsonSchemaOptions withWhitespacePattern(String pattern) {
        if (pattern != null && pattern.isEmpty()) {
            throw new IllegalArgumentException("whitespace_pattern must not be empty");
        }
        return new JsonSchemaOptions(mode, style, anyOrder, maxWhitespace, pattern, disableAdditionalProperties);
    }

    /** These options closing every object that declares properties when {@code value} (guidance's option). */
    JsonSchemaOptions withAdditionalPropertiesDisabled(boolean value) {
        return new JsonSchemaOptions(mode, style, anyOrder, maxWhitespace, whitespacePattern, value);
    }

    /** The XML parameter style of the top-level object, or null for standard JSON. */
    XmlParameterStyle style() {
        return style;
    }

    /** Whether every object's members come in any order. */
    boolean anyOrder() {
        return anyOrder;
    }

    /** The explicit whitespace regex, or null when the {@link #mode} rule applies. */
    String whitespacePattern() {
        return whitespacePattern;
    }

    /** Whether every object that declares properties is closed ({@code additionalProperties: false}). */
    boolean disableAdditionalProperties() {
        return disableAdditionalProperties;
    }

    /** The {@code space} rule text between JSON tokens. */
    String spaceRule() {
        return maxWhitespace < 0 ? mode.spaceRule() : boundedWhitespace();
    }

    /** The whitespace rule text between and around XML parameters. */
    String xmlSpaceRule() {
        return maxWhitespace < 0 ? XML_SPACE_RULE : boundedWhitespace();
    }

    private String boundedWhitespace() {
        return "[ \\t\\n\\r]{0," + maxWhitespace + "}";
    }
}
