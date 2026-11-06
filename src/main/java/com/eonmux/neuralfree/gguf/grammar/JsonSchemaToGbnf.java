package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.Kind;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.PatternProperty;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.Property;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.Schema;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.SchemaBuilder;
import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * JSON schema to GBNF translator, the Java port of llama.cpp
 * {@code common/json-schema-to-grammar.cpp} ({@code :993} tag) and the schema normalization of
 * {@code common/json-schema.cpp}, extended to the keyword coverage of vLLM's xgrammar backend. The schema JSON
 * parses with the {@code Json} API under {@link Json.Limits#UNTRUSTED}, normalizes into a schema node tree
 * ({@link JsonSchemaTree}: properties with their required flags, array item and tuple shapes, numeric bounds,
 * formats), and the
 * converter visits it in llama.cpp's order, adding one GBNF rule per schema node into a
 * {@link GbnfBuilder}:
 *
 * <ul>
 *   <li>{@code $ref} into the same document ({@link SchemaDocument}): JSON pointers, {@code $id} resources and
 *       {@code $anchor} names resolved against the base URI in scope; schemas are never fetched, so a reference
 *       leaving the document is refused; cycles handled and the visit deduplicated per target;</li>
 *   <li>{@code anyOf}/{@code oneOf} as a union, {@code allOf} as required/optional property
 *       merging plus the enum intersection of llama.cpp {@code _visit_all_of};</li>
 *   <li>{@code enum} and {@code const} as the dumped constants in GBNF literal form;</li>
 *   <li>{@code pattern} through {@link RegexToGbnf#valueRule}: the whole decoded string matches the regex
 *       (xgrammar's full match; the anchors are optional), its characters written JSON-encoded; an
 *       unsupported or invalid pattern fails the conversion;</li>
 *   <li>{@code minLength}/{@code maxLength} and {@code minItems}/{@code maxItems} as built
 *       repetitions ({@link GbnfBuilder#repetition});</li>
 *   <li>{@code minimum}/{@code maximum}/{@code exclusiveMinimum}/{@code exclusiveMaximum} (number or draft 4
 *       flag form) as the integer ranges ({@code _build_min_max_int} digit ranges, any integral magnitude) and
 *       the decimal ranges of {@link JsonNumberRanges};</li>
 *   <li>{@code additionalProperties} (a bare object's implicit true, {@code false} closed), incl.
 *       the property-name exclusion trie of llama.cpp {@code _not_strings};</li>
 *   <li>{@code patternProperties}: keys matching a pattern (ECMAScript search semantics) take its schema; the
 *       key grammar writes the matched characters JSON-encoded ({@link RegexToGbnf#keyRule});</li>
 *   <li>{@code propertyNames}: the schema every key of an object without declared properties satisfies;</li>
 *   <li>{@code minProperties}/{@code maxProperties} as counted member sequences;</li>
 *   <li>{@code required} ordering, which follows the original property order;</li>
 *   <li>the string formats of {@link JsonStringFormats};</li>
 *   <li>boolean schemas: {@code true} admits any value; {@code false} drops the property, pattern or
 *       alternative it stands for, closes {@code items} after the prefix and empties the keys of
 *       {@code propertyNames}, and is refused where a value is required.</li>
 * </ul>
 *
 * <p>{@code json_object}, the generic JSON constraint, is the conversion of the empty
 * schema, the primitive {@code value} rules of llama.cpp's {@code PRIMITIVE_RULES}.</p>
 *
 * <p>The structural-tag {@code json_schema} format converts under {@link JsonSchemaOptions}: xgrammar's
 * {@code any_order} writes every object as a count-bounded run of any of its members, {@code max_whitespace_cnt}
 * bounds the {@code space} rule, and an XML {@code style} writes the top-level object as tagged parameters
 * ({@link XmlParameterRules}).</p>
 *
 * <p>Whitespace is one {@code space} rule: {@link WhitespaceMode#FLEXIBLE} uses
 * llama.cpp's {@code SPACE_RULE}, {@link WhitespaceMode#COMPACT} the empty literal {@code ""}, so
 * nothing but the token separators stands between JSON tokens. An explicit {@code whitespace_pattern}
 * ({@link JsonSchemaOptions#withWhitespacePattern}) translates that regex into the {@code space} rule instead, and
 * {@code disable_additional_properties} ({@link JsonSchemaOptions#withAdditionalPropertiesDisabled}) closes every
 * object that declares properties. Refusals stay refusals: an
 * unknown or non-string {@code type}, an unresolvable {@code $ref}, a negative or oversized
 * length or count, bounds no value satisfies, an invalid pattern, a key constraint a grammar cannot keep
 * (overlapping {@code patternProperties} with different schemas, {@code patternProperties} beside
 * {@code additionalProperties}, {@code propertyNames} beside declared or pattern properties) — named in the
 * failure, never silently relaxed; the conversion surfaces them with llama.cpp's
 * {@code "JSON schema conversion failed:"} summary.</p>
 */
public final class JsonSchemaToGbnf {

    // ------------------------------------------------------------------ whitespace modes

    /** The compact mode's {@code space} rule: the empty literal, so the rule matches nothing. */
    private static final String COMPACT_SPACE_RULE = "\"\"";

    /** The two whitespace modes. */
    public enum WhitespaceMode {
        /** Nothing between JSON tokens but their own separators. */
        COMPACT,
        /** llama.cpp's flexible rule: a space, or one or two newlines followed by blanks/tabs. */
        FLEXIBLE;

        String spaceRule() {
            return this == COMPACT ? COMPACT_SPACE_RULE : GbnfBuilder.SPACE_RULE;
        }
    }

    // ------------------------------------------------------------------ entry points

    /**
     * The GBNF text of the conversion of {@code schemaJson}, the port of llama.cpp
     * {@code json_schema_to_grammar} ({@code common/json-schema-to-grammar.cpp:993}).
     *
     * @param schemaJson the JSON schema text
     * @param mode       the whitespace mode
     * @return the grammar text, one {@code name ::= rule} line per registry entry
     * @throws IllegalArgumentException with the {@code "JSON schema conversion failed:"}
     *                                  summary of every refusal, llama.cpp's
     *                                  {@code std::invalid_argument} mirror
     * @throws com.eonmux.neuralfree.utils.json.JsonException when the schema is not valid JSON
     *                                                        or trips {@code UNTRUSTED}
     */
    public static String gbnf(String schemaJson, WhitespaceMode mode) {
        return gbnf(schemaJson, JsonSchemaOptions.json(mode));
    }

    /**
     * {@link #gbnf(String, WhitespaceMode)} under {@code options}: in an {@link XmlParameterStyle XML style} the
     * top-level object becomes its tagged parameters ({@link XmlParameterRules}), nested values staying JSON.
     */
    static String gbnf(String schemaJson, JsonSchemaOptions options) {
        try {
            Object document = Json.parse(schemaJson, Json.Limits.UNTRUSTED);
            Converter converter = new Converter(options);
            Schema schema = converter.schema(document);
            if (options.style() == null) {
                converter.visit(schema, "");
            } else {
                new XmlParameterRules(converter, options).root(schema);
            }
            converter.checkErrors();
            return converter.rules.format();
        } catch (SchemaErrorException e) {
            throw new IllegalArgumentException("JSON schema conversion failed:\n" + e.getMessage());
        }
    }

    /**
     * The converted {@code schemaJson} parsed with {@link GbnfParser}: the validated
     * {@link Grammar} of a {@code json_schema} constraint (no left recursion, defined rules,
     * a {@code root} start symbol).
     */
    public static Grammar convert(String schemaJson, WhitespaceMode mode) {
        return GbnfParser.parse(gbnf(schemaJson, mode));
    }

    /** {@link #convert(String, WhitespaceMode)} under {@code options}. */
    static Grammar convert(String schemaJson, JsonSchemaOptions options) {
        return GbnfParser.parse(gbnf(schemaJson, options));
    }

    /** A {@link Composer} in {@code mode}. */
    public static Composer composer(WhitespaceMode mode) {
        return new Composer(mode == null ? WhitespaceMode.FLEXIBLE : mode);
    }

    /** A {@link Composer} under {@code options} (its {@code style} is written through {@code addXmlParameters}). */
    static Composer composer(JsonSchemaOptions options) {
        return new Composer(options == null ? JsonSchemaOptions.json(null) : options);
    }

    /**
     * One grammar composed of hand-written rules and converted schemas, llama.cpp's {@code common_grammar_builder}
     * ({@code build_grammar}'s {@code add_rule} / {@code add_schema}), as the tool-call grammars use it: schemas
     * convert under their own rule names into one shared registry, so rules reference them by the returned names.
     * The composition needs a {@code root} rule. Not thread-safe.
     */
    public static final class Composer {
        private final Converter converter;

        private Composer(WhitespaceMode mode) {
            this(JsonSchemaOptions.json(mode));
        }

        private Composer(JsonSchemaOptions options) {
            this.converter = new Converter(options);
        }

        /**
         * Adds a rule ({@code name ::= definition}); returns the stored name, deduplicated as
         * {@link GbnfBuilder#addRule} does.
         */
        public String addRule(String name, String definition) {
            return converter.rules.addRule(name, definition);
        }

        /**
         * Converts {@code schemaJson} under {@code name}; returns the rule name that matches it.
         *
         * @throws IllegalArgumentException when the schema cannot be converted
         */
        public String addSchema(String name, String schemaJson) {
            return addSchema(name, schemaJson, Collections.<String>emptyList());
        }

        /**
         * Converts {@code schemaJson} under {@code name} with no JSON string in it — a value or a key whose characters
         * the schema leaves free — holding any of {@code excludes} in its raw text (xgrammar's {@code json_schema}
         * format {@code excludes}, the closers a structural tag's parser would end the value at); returns the rule
         * name that matches it. A string the schema bounds by length or pattern, a property name set that excludes
         * declared names, or a constant that holds an excluded string cannot hold the exclusion and is a conversion
         * error naming it.
         *
         * @throws IllegalArgumentException when the schema cannot be converted
         */
        public String addSchema(String name, String schemaJson, List<String> excludes) {
            if (excludes == null || excludes.contains(null) || excludes.contains("")) {
                throw new IllegalArgumentException("string excludes must be non-empty strings");
            }
            converter.excluding(excludes);
            try {
                return addConverted(name, schemaJson);
            } finally {
                converter.excluding(Collections.<String>emptyList());
            }
        }

        private String addConverted(String name, String schemaJson) {
            if (name == null || name.isEmpty() || Grammar.ROOT_RULE.equals(name)) {
                throw new IllegalArgumentException("a composed schema needs its own rule name, got '" + name + "'");
            }
            try {
                Object document = Json.parse(schemaJson, Json.Limits.UNTRUSTED);
                Schema schema = converter.schema(document);
                converter.beginDocument();
                return converter.visit(schema, name);
            } catch (SchemaErrorException e) {
                throw new IllegalArgumentException("JSON schema conversion failed:\n" + e.getMessage());
            }
        }

        /**
         * Converts the JSON-schema value {@code pattern} under {@code name} as the raw, unquoted text it matches whole
         * ({@link RegexToGbnf#rawValueRule}), the value of a format that writes a string unescaped between its own
         * delimiters; returns the stored rule name.
         *
         * @throws IllegalArgumentException when the pattern has no grammar equivalent ({@link RegexToGbnf.UnsupportedRegexException})
         *                                  or is structurally invalid
         */
        public String addRawPattern(String name, String pattern) {
            if (name == null || name.isEmpty() || Grammar.ROOT_RULE.equals(name)) {
                throw new IllegalArgumentException("a composed pattern needs its own rule name, got '" + name + "'");
            }
            Map<String, String> snapshot = converter.rules.snapshot();
            try {
                return RegexToGbnf.rawValueRule(pattern, name, converter.rules);
            } catch (RuntimeException e) {
                converter.rules.restore(snapshot);
                throw e;
            }
        }

        /**
         * Converts {@code schemaJson} under {@code name} as the tagged parameters of the xgrammar XML style
         * {@code style} ({@code qwen_xml}, {@code deepseek_xml}, {@code deepseek_v4_1_xml}, ...,
         * {@link XmlParameterRules}), xgrammar's {@code json_schema} format with that {@code style}; returns the rule
         * name. One composition writes one XML style.
         *
         * @throws IllegalArgumentException when the style is unknown or the schema cannot be converted
         */
        public String addXmlParameters(String name, String schemaJson, String style) {
            if (name == null || name.isEmpty() || Grammar.ROOT_RULE.equals(name)) {
                throw new IllegalArgumentException("composed XML parameters need their own rule name, got '" + name
                        + "'");
            }
            XmlParameterStyle xml = XmlParameterStyle.named(style);
            if (xml == null) {
                throw new IllegalArgumentException("XML parameter style '" + style + "' is not recognized");
            }
            try {
                Object document = Json.parse(schemaJson, Json.Limits.UNTRUSTED);
                Schema schema = converter.schema(document);
                converter.beginDocument();
                return new XmlParameterRules(converter, converter.options.withStyle(xml)).rule(schema, name);
            } catch (SchemaErrorException e) {
                throw new IllegalArgumentException("JSON schema conversion failed:\n" + e.getMessage());
            }
        }

        /** The rule registry the composition writes into, for callers storing rules with attributes. */
        GbnfBuilder builder() {
            return converter.rules;
        }

        /**
         * The composed GBNF text.
         *
         * @throws IllegalArgumentException when a schema recorded a conversion error, or no root rule is defined
         */
        public String gbnf() {
            converter.checkErrors();
            if (!converter.rules.hasRule(Grammar.ROOT_RULE)) {
                throw new IllegalArgumentException("a composed grammar needs a root rule");
            }
            return converter.rules.format();
        }
    }

    private JsonSchemaToGbnf() {
    }

    // ------------------------------------------------------------------ primitive tables

    /** One builtin rule, llama.cpp {@code BuiltinRule}: text plus dependency names. */
    static final class BuiltinRule {
        final String content;
        final String[] deps;

        BuiltinRule(String content, String[] deps) {
            this.content = content;
            this.deps = deps;
        }
    }

    /** llama.cpp {@code PRIMITIVE_RULES} ({@code common/json-schema-to-grammar.cpp:236-249}). */
    private static final Map<String, BuiltinRule> PRIMITIVE_RULES = new LinkedHashMap<>();

    /** llama.cpp {@code STRING_FORMAT_RULES} ({@code :251-258}) and the xgrammar formats. */
    private static final Map<String, BuiltinRule> STRING_FORMAT_RULES = JsonStringFormats.RULES;

    static {
        PRIMITIVE_RULES.put("boolean", new BuiltinRule("(\"true\" | \"false\")", new String[0]));
        PRIMITIVE_RULES.put("decimal-part", new BuiltinRule("[0-9]{1,16}", new String[0]));
        PRIMITIVE_RULES.put("integral-part", new BuiltinRule("[0] | [1-9] [0-9]{0,15}",
                new String[0]));
        PRIMITIVE_RULES.put("number", new BuiltinRule("(\"-\"? integral-part) (\".\" decimal-part)?"
                + " ([eE] [-+]? integral-part)?", new String[] {"integral-part", "decimal-part"}));
        PRIMITIVE_RULES.put("integer", new BuiltinRule("(\"-\"? integral-part)",
                new String[] {"integral-part"}));
        PRIMITIVE_RULES.put("value", new BuiltinRule("object | array | string | number | boolean"
                + " | null", new String[] {"object", "array", "string", "number", "boolean",
                "null"}));
        PRIMITIVE_RULES.put("object", new BuiltinRule("\"{\" space ( string \":\" space value"
                + " (\",\" space string \":\" space value)* )? space \"}\"",
                new String[] {"string", "value"}));
        PRIMITIVE_RULES.put("array", new BuiltinRule("\"[\" space ( value (\",\" space value)* )?"
                + " space \"]\"", new String[] {"value"}));
        PRIMITIVE_RULES.put("uuid", new BuiltinRule("\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4}"
                + " \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"",
                new String[0]));
        PRIMITIVE_RULES.put("char", new BuiltinRule("[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt]"
                + " | \"u\" [0-9a-fA-F]{4})", new String[0]));
        PRIMITIVE_RULES.put("string", new BuiltinRule("\"\\\"\" char* \"\\\"\"",
                new String[] {"char"}));
        PRIMITIVE_RULES.put("null", new BuiltinRule("\"null\"", new String[0]));
    }

    /** Every name a schema property may not claim, llama.cpp {@code is_reserved_name}. */
    private static boolean isReservedName(String name) {
        return RESERVED_NAMES.contains(name);
    }

    private static final Set<String> RESERVED_NAMES = new HashSet<>();

    /** The primitives that hold free JSON strings, directly or through their members. */
    private static final Set<String> EXCLUDABLE_TYPES = new HashSet<>(
            Arrays.asList("string", "value", "object", "array"));
    /** A reference to one of {@link #EXCLUDABLE_TYPES} in a primitive's rule text. */
    private static final Pattern EXCLUDABLE = Pattern.compile(
            "(?<![\\w-])(string|value|object|array)(?![\\w-])");

    static {
        RESERVED_NAMES.add("root");
        RESERVED_NAMES.addAll(PRIMITIVE_RULES.keySet());
        RESERVED_NAMES.addAll(STRING_FORMAT_RULES.keySet());
    }


    /** A schema normalization failure, the shape of llama.cpp {@code fail(path, msg)}. */
    static final class SchemaErrorException extends RuntimeException {
        SchemaErrorException(String path, String msg) {
            super(path.isEmpty() ? msg : "JSON schema error at " + path + ": " + msg);
        }
    }

    // ------------------------------------------------------------------ conversion

    /** The port of {@code common_chat_schema_converter} ({@code :342-991}). */
    static final class Converter {
        final JsonSchemaOptions options;
        final GbnfBuilder rules;
        /** Rule names of the {@code $ref} targets resolved in the current document, by canonical key. */
        final Map<String, String> resolvedRefs = new HashMap<>();
        /** Rule names reserved for the targets whose resolution is under way (cycles return these). */
        final Map<String, String> pendingRefs = new HashMap<>();
        final List<String> errors = new ArrayList<>();

        /** The strings no free JSON string may hold, empty when none; {@link #excluding} sets them per schema. */
        private List<String> excludes = new ArrayList<>();
        /** The rule-name suffix of the primitives of the current {@link #excludes}. */
        private String excludeSuffix = "";
        private int exclusionCount;

        Converter(JsonSchemaOptions options) {
            this.options = options;
            this.rules = new GbnfBuilder(options.spaceRule());
            if (options.whitespacePattern() != null) {
                configureWhitespacePattern(options.whitespacePattern());
            }
        }

        /**
         * Makes the {@code space} rule the grammar of the explicit whitespace regex, so every JSON token is
         * separated by exactly what the pattern admits.
         */
        private void configureWhitespacePattern(String pattern) {
            try {
                rules.setSpaceRule(RegexToGbnf.rawValueRule(pattern, "space-pattern", rules));
            } catch (RegexToGbnf.UnsupportedRegexException e) {
                throw new IllegalArgumentException("whitespace_pattern " + pattern
                        + " uses an unsupported construct: " + e.getMessage(), e);
            } catch (RegexToGbnf.InvalidRegexException e) {
                throw new IllegalArgumentException("whitespace_pattern " + pattern
                        + " is not a valid regex: " + e.getMessage(), e);
            }
        }

        /** The schema tree of a parsed document under the conversion options. */
        Schema schema(Object document) {
            return new SchemaBuilder(options.disableAdditionalProperties()
                    ? JsonSchemaAdditionalProperties.closed(document) : document).build();
        }

        /** Starts a new schema document: its {@code $ref} keys name its own subschemas. */
        void beginDocument() {
            resolvedRefs.clear();
            pendingRefs.clear();
        }

        /** Converts the next schemas with {@code strings} excluded from their free JSON strings. */
        void excluding(List<String> strings) {
            excludes = new ArrayList<>(strings);
            excludeSuffix = strings.isEmpty() ? "" : "x" + (exclusionCount++);
        }

        private boolean excluding() {
            return !excludes.isEmpty();
        }

        /**
         * The primitive {@code type} ({@code string}, {@code value}, {@code object}, {@code array}) whose free strings
         * hold no excluded string: the string body is {@link GbnfExclusion#excludingJsonStringBody}, and the composite
         * primitives reference the excluding variants. Added once per exclusion.
         */
        private String excludingPrimitive(String type) {
            String name = type + "-" + excludeSuffix;
            if (rules.hasRule(name)) {
                return name;
            }
            if (type.equals("string")) {
                String body = GbnfExclusion.excludingJsonStringBody(rules::addRule, name + "-body", excludes);
                return rules.addRule(name, "\"\\\"\" " + body + " \"\\\"\"");
            }
            BuiltinRule rule = PRIMITIVE_RULES.get(type);
            String stored = rules.addRule(name, EXCLUDABLE.matcher(rule.content).replaceAll("$1-" + excludeSuffix));
            for (String dep : rule.deps) {
                if (EXCLUDABLE_TYPES.contains(dep)) {
                    excludingPrimitive(dep);
                } else if (!rules.hasRule(dep)) {
                    addPrimitive(dep, PRIMITIVE_RULES.get(dep));
                }
            }
            return stored;
        }

        /** The literal of a constant, which must not write an excluded string. */
        private String constantRule(Object value, String name) {
            if (excluding()) {
                String written = dump(value);
                for (String x : excludes) {
                    if (written.contains(x)) {
                        return exclusionError("the constant " + written + " at '" + name + "'");
                    }
                }
            }
            return generateConstantRule(value);
        }

        /** Records that the current exclusion cannot hold in {@code what}; returns an empty rule text. */
        private String exclusionError(String what) {
            errors.add(what + " cannot hold the string excludes " + excludes);
            return "";
        }

        /**
         * Whether a {@code format} string could spell an excluded string. The date, time and uuid formats write only
         * ASCII letters, digits and {@code :+.-}, so an exclude holding any other character is safe beside them; the
         * other formats (email, hostname, the URI forms, ...) may spell anything, so they never are.
         */
        private boolean formatMaySpellExclude(JsonStringFormats.Format format) {
            if (format != JsonStringFormats.Format.UUID && format != JsonStringFormats.Format.DATE
                    && format != JsonStringFormats.Format.TIME && format != JsonStringFormats.Format.DATE_TIME) {
                return true;
            }
            for (String x : excludes) {
                if (x.chars().allMatch(c -> Character.isLetterOrDigit(c) && c < 0x80 || ":+.-".indexOf(c) >= 0)) {
                    return true;
                }
            }
            return false;
        }

        /** {@code visit} ({@code :877-973}). */
        String visit(Schema schema, String name) {
            String ruleName = isReservedName(GbnfBuilder.sanitizeRuleName(name)) ? name + "-"
                    : name.isEmpty() ? "root" : name;
            String subName = name + (name.isEmpty() ? "" : "-");

            switch (schema.kind) {
                case REF:
                    return rules.addRule(ruleName, resolveRef(schema));
                case ANY_OF:
                    return rules.addRule(ruleName, generateUnionRule(name, schema.children));
                case ALL_OF:
                    return visitAllOf(schema, name, ruleName);
                case CONST:
                    return rules.addRule(ruleName, constantRule(schema.constValue, name));
                case ENUM: {
                    List<String> constants = new ArrayList<>();
                    for (Object value : schema.values) {
                        constants.add(constantRule(value, name));
                    }
                    return rules.addRule(ruleName, "(" + String.join(" | ", constants) + ")");
                }
                case OBJECT:
                    return visitObject(schema, name, ruleName);
                case TUPLE: {
                    StringBuilder out = new StringBuilder("\"[\" space ");
                    for (int i = 0; i < schema.children.size(); i++) {
                        if (i > 0) {
                            out.append(" \",\" space ");
                        }
                        out.append(visit(schema.children.get(i), subName + "tuple-" + i));
                    }
                    out.append(" space \"]\"");
                    return rules.addRule(ruleName, out.toString());
                }
                case ARRAY: {
                    if (schema.items.kind == Kind.ANY && schema.minItems == 0
                            && schema.maxItems < 0) {
                        return primitiveVisit(ruleName, "array");
                    }
                    String itemRuleName = visit(schema.items, subName + "item");
                    int maxItems = schema.maxItems < 0 ? Integer.MAX_VALUE : schema.maxItems;
                    return rules.addRule(ruleName, "\"[\" space " + GbnfBuilder.repetition(
                            itemRuleName, schema.minItems, maxItems, "\",\" space")
                            + " space \"]\"");
                }
                case STRING:
                    return visitString(schema, ruleName, name);
                case INTEGER: {
                    String range = JsonNumberRanges.integerRange(
                            schema.lower == null ? null : schema.lower.lowestInteger(),
                            schema.upper == null ? null : schema.upper.highestInteger());
                    return range == null ? primitiveVisit(ruleName, "integer")
                            : rules.addRule(ruleName, "(" + range + ")");
                }
                case NUMBER:
                    return visitNumber(schema, ruleName);
                case BOOLEAN:
                    return primitiveVisit(ruleName, "boolean");
                case NULL_KIND:
                    return primitiveVisit(ruleName, "null");
                case ANY:
                    return rules.addRule(ruleName, addPrimitive("value",
                            PRIMITIVE_RULES.get("value")));
                default:
                    throw new IllegalStateException("unexpected schema kind " + schema.kind);
            }
        }

        /** {@code _visit_primitive} ({@code :828-830}): primitives take their type's name. */
        String primitiveVisit(String ruleName, String type) {
            if (excluding() && EXCLUDABLE_TYPES.contains(type)) {
                String excluded = excludingPrimitive(type);
                return ruleName.equals("root") ? rules.addRule("root", excluded) : excluded;
            }
            return addPrimitive(ruleName.equals("root") ? "root" : type, PRIMITIVE_RULES.get(type));
        }

        private String visitObject(Schema schema, String name, String ruleName) {
            if (schema.properties.isEmpty() && schema.additionalProperties != null
                    && schema.additionalProperties.kind == Kind.ANY && schema.propertyNames == null
                    && schema.forbiddenProperties.isEmpty() && !schema.countsProperties()) {
                return rules.addRule(ruleName, addPrimitive("object", PRIMITIVE_RULES.get("object")));
            }
            Set<String> required = new LinkedHashSet<>();
            for (Property property : schema.properties) {
                if (property.required) {
                    required.add(property.name);
                }
            }
            return rules.addRule(ruleName, buildObjectRule(schema.properties, required, name,
                    schema.additionalProperties, schema));
        }

        private String visitString(Schema schema, String ruleName, String name) {
            if (excluding() && (!schema.pattern.isEmpty() || schema.minLength > 0 || schema.maxLength >= 0)) {
                return exclusionError("a string bounded by a pattern or length at '" + name + "'");
            }
            if (excluding() && schema.format != JsonStringFormats.Format.NONE && formatMaySpellExclude(schema.format)) {
                return exclusionError("a format string at '" + name + "'");
            }
            if (!schema.pattern.isEmpty()) {
                return visitPattern(schema.pattern, ruleName);
            }
            if (schema.format == JsonStringFormats.Format.UUID) {
                return primitiveVisit(ruleName, "uuid");
            }
            if (schema.format != JsonStringFormats.Format.NONE) {
                String prim = schema.format.rule;
                return rules.addRule(ruleName, addPrimitive(prim, STRING_FORMAT_RULES.get(prim)));
            }
            if (schema.minLength > 0 || schema.maxLength >= 0) {
                String charRule = addPrimitive("char", PRIMITIVE_RULES.get("char"));
                int maxLength = schema.maxLength < 0 ? Integer.MAX_VALUE : schema.maxLength;
                return rules.addRule(ruleName, "\"\\\"\" " + GbnfBuilder.repetition(
                        charRule, schema.minLength, maxLength, "") + " \"\\\"\"");
            }
            return primitiveVisit(ruleName, "string");
        }

        /** A bounded number as {@link JsonNumberRanges#decimalRange}; an unbounded one is the primitive. */
        private String visitNumber(Schema schema, String ruleName) {
            if (schema.lower == null && schema.upper == null) {
                return primitiveVisit(ruleName, "number");
            }
            String range = JsonNumberRanges.decimalRange(
                    schema.lower == null ? null : schema.lower.value, schema.lower != null && schema.lower.exclusive,
                    schema.upper == null ? null : schema.upper.value, schema.upper != null && schema.upper.exclusive);
            if (range == null) {
                errors.add("no number of at most " + JsonNumberRanges.MAX_FRACTION_DIGITS
                        + " fraction digits lies within the bounds of " + ruleName);
                return "";
            }
            return rules.addRule(ruleName, "(" + range + ")");
        }

        /** {@code _generate_union_rule} ({@code :371-378}). */
        private String generateUnionRule(String name, List<Schema> alternatives) {
            List<String> parts = new ArrayList<>();
            for (int i = 0; i < alternatives.size(); i++) {
                parts.add(visit(alternatives.get(i),
                        name + (name.isEmpty() ? "alternative-" : "-") + i));
            }
            return String.join(" | ", parts);
        }

        /**
         * A value {@code pattern}: the strings whose whole decoded text it matches ({@link RegexToGbnf#valueRule},
         * xgrammar's full match). A pattern the translator cannot express is a conversion error, as xgrammar
         * refuses it, never a wider string rule.
         */
        private String visitPattern(String pattern, String ruleName) {
            Map<String, String> snapshot = rules.snapshot();
            try {
                return RegexToGbnf.valueRule(pattern, ruleName, rules);
            } catch (RegexToGbnf.UnsupportedRegexException e) {
                rules.restore(snapshot);
                errors.add("pattern " + pattern + " is not supported: " + e.getMessage());
                return "";
            } catch (RegexToGbnf.InvalidRegexException e) {
                rules.restore(snapshot);
                errors.add("Invalid pattern " + pattern + ": " + e.getMessage());
                return "";
            }
        }

        /**
         * The raw value {@code pattern} of an XML parameter ({@link RegexToGbnf#rawValueRule}), refused as
         * {@link #visitPattern} refuses a JSON value pattern.
         */
        String rawPattern(String pattern, String ruleName) {
            Map<String, String> snapshot = rules.snapshot();
            try {
                return RegexToGbnf.rawValueRule(pattern, ruleName, rules);
            } catch (RegexToGbnf.UnsupportedRegexException e) {
                rules.restore(snapshot);
                errors.add("pattern " + pattern + " is not supported: " + e.getMessage());
                return "";
            } catch (RegexToGbnf.InvalidRegexException e) {
                rules.restore(snapshot);
                errors.add("Invalid pattern " + pattern + ": " + e.getMessage());
                return "";
            }
        }

        /** The builtin rule of the raw text of {@code format}, unquoted ({@link JsonStringFormats.Format#raw}). */
        String rawFormat(JsonStringFormats.Format format) {
            return addPrimitive(format.raw, STRING_FORMAT_RULES.get(format.raw));
        }

        /** The llama.cpp primitive rule {@code name} ({@code value}, {@code object}, {@code number}, ...). */
        String primitive(String name) {
            return addPrimitive(name, PRIMITIVE_RULES.get(name));
        }

        /**
         * {@code _resolve_ref} ({@code :682-697}): one rule per target of the document, named after its pointer
         * ({@code ref-definitions-foo}); a name another target already holds gets a numeric suffix, and a
         * reference back into a target under resolution takes the name reserved for it.
         */
        private String resolveRef(Schema ref) {
            String resolved = resolvedRefs.get(ref.ref);
            if (resolved != null) {
                return resolved;
            }
            String pending = pendingRefs.get(ref.ref);
            if (pending != null) {
                return pending;
            }
            String base = "ref" + ref.ref.substring(ref.ref.indexOf('#') + 1).replaceAll("[^a-zA-Z0-9-]+", "-");
            String refName = base;
            for (int suffix = 0; rules.hasRule(refName) || pendingRefs.containsValue(refName); suffix++) {
                refName = base + suffix;
            }
            pendingRefs.put(ref.ref, refName);
            String stored;
            try {
                stored = visit(ref.target, refName);
            } finally {
                pendingRefs.remove(ref.ref);
            }
            resolvedRefs.put(ref.ref, stored);
            return stored;
        }

        /** {@code _build_object_rule} ({@code :699-794}) with the pattern, key and count keywords of {@code shape}. */
        private String buildObjectRule(List<Property> properties, Set<String> required,
                String name, Schema additionalProperties, Schema shape) {
            List<String> requiredProps = new ArrayList<>();
            List<String> optionalProps = new ArrayList<>();
            Map<String, String> propKvRuleNames = new LinkedHashMap<>();
            List<String> propNames = new ArrayList<>();
            for (Property property : properties) {
                String propRuleName = visit(property.schema,
                        name + (name.isEmpty() ? "" : "-") + property.name);
                propKvRuleNames.put(property.name, rules.addRule(
                        name + (name.isEmpty() ? "" : "-") + property.name + "-kv",
                        formatLiteral(dump(property.name)) + " space \":\" space "
                                + propRuleName));
                if (required.contains(property.name)) {
                    requiredProps.add(property.name);
                } else {
                    optionalProps.add(property.name);
                }
                propNames.add(property.name);
            }
            if (shape != null) {
                propNames.addAll(shape.forbiddenProperties);
            }
            String repeatedKv = shape != null && !shape.patternProperties.isEmpty()
                    ? patternKv(shape.patternProperties, name)
                    : additionalProperties != null
                    ? additionalKv(additionalProperties, propNames, name, shape == null ? null : shape.propertyNames)
                    : null;

            if (options.anyOrder()) {
                List<String> members = new ArrayList<>(propKvRuleNames.values());
                if (repeatedKv != null) {
                    members.add(repeatedKv);
                }
                int low = Math.max(shape == null ? 0 : shape.minProperties, requiredProps.size());
                int high = shape == null || shape.maxProperties < 0 ? CountedMembers.UNBOUNDED : shape.maxProperties;
                return anyOrderObject(members, low, high, ObjectSyntax.JSON, name);
            }
            if (shape != null && shape.countsProperties()) {
                List<String> requiredKvs = new ArrayList<>();
                for (String prop : requiredProps) {
                    requiredKvs.add(propKvRuleNames.get(prop));
                }
                List<String> optionalKvs = new ArrayList<>();
                for (String prop : optionalProps) {
                    optionalKvs.add(propKvRuleNames.get(prop));
                }
                return countedObjectRule(requiredKvs, optionalKvs, repeatedKv, shape, name);
            }
            if (repeatedKv != null) {
                propKvRuleNames.put("*", repeatedKv);
                optionalProps.add("*");
            }

            if (requiredProps.isEmpty() && optionalProps.isEmpty()) {
                return "\"{\" space \"}\"";
            }

            StringBuilder out = new StringBuilder("\"{\" space ");
            for (int i = 0; i < requiredProps.size(); i++) {
                if (i > 0) {
                    out.append(" \",\" space ");
                }
                out.append(propKvRuleNames.get(requiredProps.get(i)));
            }

            if (!optionalProps.isEmpty()) {
                out.append(" (");
                if (!requiredProps.isEmpty()) {
                    out.append(" \",\" space ( ");
                }
                for (int i = 0; i < optionalProps.size(); i++) {
                    if (i > 0) {
                        out.append(" | ");
                    }
                    out.append(getRecursiveRefs(propKvRuleNames, name, optionalProps, i, false));
                }
                if (!requiredProps.isEmpty()) {
                    out.append(" )");
                }
                out.append(" )?");
            }
            out.append(" space \"}\"");
            return out.toString();
        }

        /**
         * The member rule of the keys outside the declared properties (llama.cpp {@code :743-753}): the
         * {@code propertyNames} key rule when given, else any string but a declared name. Null when no key
         * qualifies.
         */
        private String additionalKv(Schema additionalProperties, List<String> propNames, String name,
                Schema propertyNames) {
            String subName = name + (name.isEmpty() ? "" : "-") + "additional";
            String keyRule;
            if (excluding() && (propertyNames != null || !propNames.isEmpty())) {
                exclusionError("additional property names beside declared ones or under propertyNames at '"
                        + name + "'");
                return null;
            }
            if (propertyNames != null) {
                keyRule = propertyNameRule(propertyNames, subName + "-k", Collections.newSetFromMap(
                        new IdentityHashMap<>()));
                if (keyRule == null) {
                    return null;
                }
            } else {
                keyRule = propNames.isEmpty()
                        ? addPrimitive("string", PRIMITIVE_RULES.get("string"))
                        : rules.addRule(subName + "-k", notStrings(propNames, this));
            }
            String valueRule = additionalProperties.kind != Kind.ANY
                    ? visit(additionalProperties, subName + "-value")
                    : addPrimitive("value", PRIMITIVE_RULES.get("value"));
            return rules.addRule(subName + "-kv", keyRule + " \":\" space " + valueRule);
        }

        /** The member rule of the {@code patternProperties} keys, all patterns sharing one value schema. */
        private String patternKv(List<PatternProperty> patterns, String name) {
            if (excluding()) {
                exclusionError("patternProperties keys at '" + name + "'");
                return null;
            }
            String subName = name + (name.isEmpty() ? "" : "-") + "pattern";
            List<String> keyRules = new ArrayList<>();
            for (int i = 0; i < patterns.size(); i++) {
                String keyRule = patternKeyRule(patterns.get(i).pattern,
                        subName + "-k" + (patterns.size() > 1 ? "-" + i : ""));
                if (keyRule == null) {
                    return null;
                }
                keyRules.add(keyRule);
            }
            String keyRule = keyRules.size() == 1 ? keyRules.get(0)
                    : rules.addRule(subName + "-k", String.join(" | ", keyRules));
            Schema value = patterns.get(0).schema;
            String valueRule = value.kind != Kind.ANY ? visit(value, subName + "-value")
                    : addPrimitive("value", PRIMITIVE_RULES.get("value"));
            return rules.addRule(subName + "-kv", keyRule + " \":\" space " + valueRule);
        }

        /**
         * The JSON string rule of the keys {@code pattern} matches somewhere, JSON schema's unanchored regex
         * semantics: an unanchored side matches any JSON-safe characters. A pattern the translator cannot express
         * is a conversion error here, never a degradation: a wider key would escape the pattern's schema.
         */
        private String patternKeyRule(String pattern, String ruleName) {
            Map<String, String> snapshot = rules.snapshot();
            try {
                return RegexToGbnf.keyRule(pattern, ruleName, rules);
            } catch (RegexToGbnf.UnsupportedRegexException | RegexToGbnf.InvalidRegexException e) {
                rules.restore(snapshot);
                errors.add("key pattern " + pattern + " is not supported: " + e.getMessage());
                return null;
            }
        }

        /**
         * The key rule of a {@code propertyNames} schema: the strings it admits, null when it admits none (an
         * object, number or other non-string schema).
         */
        private String propertyNameRule(Schema keys, String ruleName, Set<Schema> visiting) {
            if (!visiting.add(keys)) {
                errors.add("propertyNames refers back to itself");
                return null;
            }
            try {
                switch (keys.kind) {
                    case ANY:
                        return addPrimitive("string", PRIMITIVE_RULES.get("string"));
                    case STRING:
                        return keys.pattern.isEmpty() ? visitString(keys, ruleName, ruleName)
                                : patternKeyRule(keys.pattern, ruleName);
                    case CONST:
                        return keys.constValue instanceof String
                                ? rules.addRule(ruleName, generateConstantRule(keys.constValue)) : null;
                    case ENUM:
                        return stringConstantsRule(keys.values, ruleName);
                    case REF:
                        return keys.target == null ? null : propertyNameRule(keys.target, ruleName, visiting);
                    case ANY_OF: {
                        List<String> alternatives = new ArrayList<>();
                        for (int i = 0; i < keys.children.size(); i++) {
                            String alternative = propertyNameRule(keys.children.get(i), ruleName + "-" + i,
                                    visiting);
                            if (alternative != null) {
                                alternatives.add(alternative);
                            }
                        }
                        return alternatives.isEmpty() ? null
                                : rules.addRule(ruleName, String.join(" | ", alternatives));
                    }
                    case ALL_OF:
                        errors.add("propertyNames with allOf is not supported");
                        return null;
                    default:
                        return null;
                }
            } finally {
                visiting.remove(keys);
            }
        }

        private String stringConstantsRule(List<Object> values, String ruleName) {
            List<String> constants = new ArrayList<>();
            for (Object value : values) {
                if (value instanceof String) {
                    constants.add(generateConstantRule(value));
                }
            }
            return constants.isEmpty() ? null : rules.addRule(ruleName, "(" + String.join(" | ", constants) + ")");
        }

        /**
         * The object rule under {@code minProperties}/{@code maxProperties}: the required members, then the
         * {@link CountedMembers} selection of the optional ones and the repeated (additional or pattern) member.
         */
        private String countedObjectRule(List<String> requiredKvs, List<String> optionalKvs, String repeatedKv,
                Schema shape, String name) {
            String object = name.isEmpty() ? "root" : name;
            CountedMembers members = new CountedMembers(rules, optionalKvs, new boolean[optionalKvs.size()],
                    repeatedKv, "\",\" space", name + (name.isEmpty() ? "" : "-") + "count-");
            int required = requiredKvs.size();
            int high = shape.maxProperties < 0 ? CountedMembers.UNBOUNDED : shape.maxProperties - required;
            if (high < 0) {
                errors.add("maxProperties " + shape.maxProperties + " is less than the " + required
                        + " required properties of " + object);
                return "\"{\" space \"}\"";
            }
            int low = Math.max(0, shape.minProperties - required);
            if (low > members.capacity()) {
                errors.add("minProperties " + shape.minProperties + " exceeds the " + (required + members.capacity())
                        + " properties " + object + " can hold");
                return "\"{\" space \"}\"";
            }
            List<String> parts = new ArrayList<>();
            parts.add("\"{\" space");
            if (required == 0) {
                parts.add(members.first(low, high));
            } else {
                parts.add(String.join(" \",\" space ", requiredKvs));
                parts.add(members.rest(low, high));
            }
            parts.add("space \"}\"");
            parts.removeIf(String::isEmpty);
            return String.join(" ", parts);
        }

        /**
         * xgrammar's {@code any_order} object: {@code low..high} members joined by the syntax's separator, each any
         * of {@code members}, so a key may repeat and a required one be missing; only the count is bounded.
         */
        String anyOrderObject(List<String> members, int low, int high, ObjectSyntax syntax, String name) {
            String object = name.isEmpty() ? "root" : name;
            if (low > high) {
                errors.add("maxProperties " + high + " is less than the " + low + " properties " + object
                        + " needs");
                return syntax.empty;
            }
            if (members.isEmpty() || high == 0) {
                if (low > 0) {
                    errors.add(object + " can hold no property but needs " + low);
                }
                return syntax.empty;
            }
            String item = members.size() == 1 ? members.get(0)
                    : rules.addRule(name + (name.isEmpty() ? "" : "-") + "item", String.join(" | ", members));
            String some = syntax.open + " " + GbnfBuilder.repetition(item, Math.max(1, low), high, syntax.separator)
                    + " " + syntax.close;
            return low == 0 ? some + " | " + syntax.empty : some;
        }

        /** {@code get_recursive_refs} ({@code :757-777}), index-carrying instead of splicing. */
        private String getRecursiveRefs(Map<String, String> propKvRuleNames, String name,
                List<String> keys, int from, boolean firstIsOptional) {
            if (from >= keys.size()) {
                return "";
            }
            String key = keys.get(from);
            String kvRuleName = propKvRuleNames.get(key);
            String commaRef = "( \",\" space " + kvRuleName + " )";
            String result;
            if (firstIsOptional) {
                result = commaRef + ("*".equals(key) ? "*" : "?");
            } else {
                result = "*".equals(key) ? kvRuleName + " " + commaRef + "*" : kvRuleName;
            }
            if (from + 1 < keys.size()) {
                String restName = rules.addRule(
                        name + (name.isEmpty() ? "" : "-") + key + "-rest",
                        getRecursiveRefs(propKvRuleNames, name, keys, from + 1, true));
                result += " " + restName;
            }
            return result;
        }

        /** {@code _add_primitive} ({@code :796-813}). */
        private String addPrimitive(String name, BuiltinRule rule) {
            if (excluding() && EXCLUDABLE_TYPES.contains(name)) {
                return excludingPrimitive(name);
            }
            String stored = rules.addRule(name, rule.content);
            for (String dep : rule.deps) {
                BuiltinRule depRule = PRIMITIVE_RULES.containsKey(dep) ? PRIMITIVE_RULES.get(dep)
                        : STRING_FORMAT_RULES.get(dep);
                if (depRule == null) {
                    errors.add("Rule " + dep + " not known");
                    continue;
                }
                if (!rules.hasRule(dep)) {
                    addPrimitive(dep, depRule);
                }
            }
            return stored;
        }

        /** {@code _visit_all_of} ({@code :832-875}). */
        private String visitAllOf(Schema schema, String name, String ruleName) {
            AllOfParts parts = allOfParts(schema);
            if (!parts.intersection.isEmpty()) {
                return rules.addRule(ruleName, "(" + String.join(" | ", parts.intersection) + ")");
            }
            return rules.addRule(ruleName, buildObjectRule(parts.properties, parts.required, name, null, null));
        }

        /** The merge of {@code _visit_all_of}: the components' properties and required names, the enum intersection. */
        AllOfParts allOfParts(Schema schema) {
            Set<String> required = new HashSet<>();
            List<Property> properties = new ArrayList<>();
            Map<String, Integer> enumConstants = new TreeMap<>();

            for (Schema child : schema.children) {
                if (child.kind == Kind.ANY_OF) {
                    for (Schema alt : child.children) {
                        addComponent(alt, false, properties, required, enumConstants);
                    }
                } else {
                    addComponent(child, true, properties, required, enumConstants);
                }
            }

            List<String> intersection = new ArrayList<>();
            for (Map.Entry<String, Integer> entry : enumConstants.entrySet()) {
                if (entry.getValue() == schema.children.size()) {
                    intersection.add(entry.getKey());
                }
            }
            return new AllOfParts(properties, required, intersection);
        }

        /** {@code add_component} ({@code :836-853}). */
        private void addComponent(Schema component, boolean isRequired, List<Property> properties,
                Set<String> required, Map<String, Integer> enumConstants) {
            if (component.kind == Kind.REF) {
                if (component.target != null) {
                    addComponent(component.target, isRequired, properties, required,
                            enumConstants);
                }
            } else if (component.kind == Kind.OBJECT) {
                for (Property property : component.properties) {
                    properties.add(property);
                    if (isRequired) {
                        required.add(property.name);
                    }
                }
            } else if (component.kind == Kind.ENUM) {
                for (Object value : component.values) {
                    enumConstants.merge(generateConstantRule(value), 1, Integer::sum);
                }
            }
        }

        /** {@code check_errors} ({@code :975-982}). */
        void checkErrors() {
            if (!errors.isEmpty()) {
                throw new IllegalArgumentException("JSON schema conversion failed:\n"
                        + String.join("\n", errors));
            }
        }
    }

    /**
     * The merge of an {@code allOf}: the components' properties, the names a required component declares, and
     * the constant rules every component's enum holds (empty when they share none).
     */
    static final class AllOfParts {
        final List<Property> properties;
        final Set<String> required;
        final List<String> intersection;

        AllOfParts(List<Property> properties, Set<String> required, List<String> intersection) {
            this.properties = Collections.unmodifiableList(new ArrayList<>(properties));
            this.required = Collections.unmodifiableSet(new HashSet<>(required));
            this.intersection = Collections.unmodifiableList(new ArrayList<>(intersection));
        }
    }

    /** The delimiters of an object's members: JSON's braces and comma, or an XML style's whitespace. */
    static final class ObjectSyntax {
        static final ObjectSyntax JSON = new ObjectSyntax("\"{\" space", "\",\" space", "space \"}\"",
                "\"{\" space \"}\"");

        /** The text before the first member. */
        final String open;
        /** The text between two members. */
        final String separator;
        /** The text after the last member. */
        final String close;
        /** The whole object without members. */
        final String empty;

        ObjectSyntax(String open, String separator, String close, String empty) {
            this.open = open;
            this.separator = separator;
            this.close = close;
            this.empty = empty;
        }
    }

    // ------------------------------------------------------------------ name exclusion trie

    /**
     * The rule that matches a JSON string that is none of the provided names, the port of
     * llama.cpp's {@code _not_strings} over {@code common_trie} ({@code :633-680}): {@code
     * not_strings({"and", "also"})} is {@code ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) |
     * [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]}.
     */
    private static String notStrings(List<String> names, Converter converter) {
        Trie trie = new Trie(names);
        String charRule = converter.addPrimitive("char", PRIMITIVE_RULES.get("char"));
        StringBuilder out = new StringBuilder("[\"] ( ");
        trieVisit(trie, 0, charRule, out);
        out.append(" )");
        if (trie.nodes.get(0).pattern < 0) {
            out.append('?');
        }
        out.append(" [\"]");
        return out.toString();
    }

    private static void trieVisit(Trie trie, int index, String charRule, StringBuilder out) {
        Trie.Node node = trie.nodes.get(index);
        StringBuilder rejects = new StringBuilder();
        boolean first = true;
        for (Map.Entry<Integer, Integer> child : node.children.entrySet()) {
            String c = new String(Character.toChars(child.getKey()));
            rejects.append(c);
            if (!first) {
                out.append(" | ");
            }
            first = false;
            out.append('[').append(c).append(']');
            Trie.Node childNode = trie.nodes.get(child.getValue());
            if (!childNode.children.isEmpty()) {
                out.append(" (");
                trieVisit(trie, child.getValue(), charRule, out);
                out.append(')');
            } else {
                out.append(' ').append(charRule).append('+');
            }
        }
        if (!node.children.isEmpty()) {
            out.append(" | [^\"").append(rejects).append("] ").append(charRule).append('*');
        }
    }

    /** A byte-view trie, the port of {@code common_trie} ({@code common/trie.h}) for the map
     * iteration and pattern indices {@code _not_strings} consumes. */
    private static final class Trie {
        static final class Node {
            /** Child code points in ascending order, each entry its node id. */
            final TreeMap<Integer, Integer> children = new TreeMap<>();
            /** Index of a string ending at this node, -1 when none ends here. */
            int pattern = -1;
        }

        final List<Node> nodes = new ArrayList<>();
        int patternCount;

        Trie(List<String> names) {
            nodes.add(new Node());
            for (String name : names) {
                insert(name);
            }
        }

        /** llama.cpp {@code common_trie::insert}: duplicates keep the first index. */
        void insert(String name) {
            Node current = nodes.get(0);
            int position = 0;
            while (position < name.length()) {
                int codePoint = name.codePointAt(position);
                position += Character.charCount(codePoint);
                if (current.children.get(codePoint) == null) {
                    Node next = new Node();
                    nodes.add(next);
                    current.children.put(codePoint, nodes.size() - 1);
                }
                current = nodes.get(current.children.get(codePoint));
            }
            if (current.pattern < 0) {
                current.pattern = patternCount++;
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    /**
     * llama.cpp {@code format_literal} ({@code :305-311}): GBNF escapes of CR, LF, '"' and '\'.
     */
    static String formatLiteral(String literal) {
        StringBuilder out = new StringBuilder("\"");
        for (int i = 0; i < literal.length(); i++) {
            switch (literal.charAt(i)) {
                case '\r':
                    out.append("\\r");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '"':
                    out.append("\\\"");
                    break;
                case '\\':
                    out.append("\\\\");
                    break;
                default:
                    out.append(literal.charAt(i));
                    break;
            }
        }
        out.append('"');
        return out.toString();
    }

    /**
     * The {@code .dump()} of a schema value, llama.cpp's compact sorted-key form; the canonical
     * style of {@code Json} agrees with it key for key and escape for escape.
     */
    static String dump(Object value) {
        return Json.toString(value, Json.Style.CANONICAL);
    }

    private static String generateConstantRule(Object value) {
        return formatLiteral(dump(value));
    }
}
