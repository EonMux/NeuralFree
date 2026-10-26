package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * JSON schema to GBNF translator, the Java port of llama.cpp
 * {@code common/json-schema-to-grammar.cpp} ({@code :993} tag) and the schema normalization of
 * {@code common/json-schema.cpp}. The schema JSON parses with the {@code Json} API under
 * {@link Json.Limits#UNTRUSTED} (§3.6), normalizes into a schema node tree (properties with
 * their required flags, array item and tuple shapes, integer bounds, formats), and the
 * converter visits it in llama.cpp's order, adding one GBNF rule per schema node into a
 * {@link GbnfBuilder}:
 *
 * <ul>
 *   <li>{@code $ref} with local {@code #/} pointers only (remote refs refused), cycles handled
 *       and the visit deduplicated through the rule registry;</li>
 *   <li>{@code anyOf}/{@code oneOf} as a union, {@code allOf} as required/optional property
 *       merging plus the enum intersection of llama.cpp {@code _visit_all_of};</li>
 *   <li>{@code enum} and {@code const} as the dumped constants in GBNF literal form;</li>
 *   <li>{@code pattern} through {@link RegexToGbnf}: an unsupported pattern degrades the
 *       property to the generic string rule (a warning, llama.cpp {@code _visit_pattern}), an
 *       invalid one fails the conversion;</li>
 *   <li>{@code minLength}/{@code maxLength} and {@code minItems}/{@code maxItems} as built
 *       repetitions ({@link GbnfBuilder#repetition});</li>
 *   <li>{@code minimum}/{@code maximum} as the {@code _build_min_max_int} digit ranges;</li>
 *   <li>{@code additionalProperties} (a bare object's implicit true, {@code false} closed), incl.
 *       the property-name exclusion trie of llama.cpp {@code _not_strings};</li>
 *   <li>{@code required} ordering, which follows the original property order;</li>
 *   <li>the {@code date}, {@code time}, {@code date-time} and {@code uuid} formats.</li>
 * </ul>
 *
 * <p>{@code json_object} — row G3's generic JSON constraint — is the conversion of the empty
 * schema, the primitive {@code value} rules of llama.cpp's {@code PRIMITIVE_RULES}.</p>
 *
 * <p>Whitespace (row G8) is one {@code space} rule: {@link WhitespaceMode#FLEXIBLE} uses
 * llama.cpp's {@code SPACE_RULE}, {@link WhitespaceMode#COMPACT} the empty alternate, so
 * nothing but the token separators stands between JSON tokens. Refusals stay refusals: an
 * unknown or non-string {@code type}, an unresolvable {@code $ref}, a negative or oversized
 * length or count, an invalid pattern — named in the failure, never silently relaxed; the
 * conversion surfaces them with llama.cpp's {@code "JSON schema conversion failed:"} summary.</p>
 */
public final class JsonSchemaToGbnf {

    // ------------------------------------------------------------------ whitespace (row G8)

    /** The two whitespace modes of row G8. */
    public enum WhitespaceMode {
        /** Nothing between JSON tokens but their own separators. */
        COMPACT,
        /** llama.cpp's flexible rule: a space, or one or two newlines followed by blanks/tabs. */
        FLEXIBLE;

        String spaceRule() {
            return this == COMPACT ? "" : GbnfBuilder.SPACE_RULE;
        }
    }

    // ------------------------------------------------------------------ entry points

    /**
     * The GBNF text of the conversion of {@code schemaJson}, the port of llama.cpp
     * {@code json_schema_to_grammar} ({@code common/json-schema-to-grammar.cpp:993}).
     *
     * @param schemaJson the JSON schema text
     * @param mode       the whitespace mode of row G8
     * @return the grammar text, one {@code name ::= rule} line per registry entry
     * @throws IllegalArgumentException with the {@code "JSON schema conversion failed:"}
     *                                  summary of every refusal, llama.cpp's
     *                                  {@code std::invalid_argument} mirror
     * @throws com.eonmux.neuralfree.utils.json.JsonException when the schema is not valid JSON
     *                                                        or trips {@code UNTRUSTED}
     */
    public static String gbnf(String schemaJson, WhitespaceMode mode) {
        try {
            Object document = Json.parse(schemaJson, Json.Limits.UNTRUSTED);
            SchemaBuilder builder = new SchemaBuilder(document);
            Converter converter = new Converter(mode);
            converter.visit(builder.build(), "");
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

    private JsonSchemaToGbnf() {
    }

    // ------------------------------------------------------------------ primitive tables

    /** One builtin rule, llama.cpp {@code BuiltinRule}: text plus dependency names. */
    private static final class BuiltinRule {
        final String content;
        final String[] deps;

        BuiltinRule(String content, String[] deps) {
            this.content = content;
            this.deps = deps;
        }
    }

    /** llama.cpp {@code PRIMITIVE_RULES} ({@code common/json-schema-to-grammar.cpp:236-249}). */
    private static final Map<String, BuiltinRule> PRIMITIVE_RULES = new LinkedHashMap<>();

    /** llama.cpp {@code STRING_FORMAT_RULES} ({@code :251-258}). */
    private static final Map<String, BuiltinRule> STRING_FORMAT_RULES = new LinkedHashMap<>();

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
                + " \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"",
                new String[0]));
        PRIMITIVE_RULES.put("char", new BuiltinRule("[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt]"
                + " | \"u\" [0-9a-fA-F]{4})", new String[0]));
        PRIMITIVE_RULES.put("string", new BuiltinRule("\"\\\"\" char* \"\\\"\"",
                new String[] {"char"}));
        PRIMITIVE_RULES.put("null", new BuiltinRule("\"null\"", new String[0]));

        STRING_FORMAT_RULES.put("date", new BuiltinRule("[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] )"
                + " \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", new String[0]));
        STRING_FORMAT_RULES.put("time", new BuiltinRule("([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9]"
                + " \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" )"
                + " ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", new String[0]));
        STRING_FORMAT_RULES.put("date-time", new BuiltinRule("date \"T\" time",
                new String[] {"date", "time"}));
        STRING_FORMAT_RULES.put("date-string", new BuiltinRule("\"\\\"\" date \"\\\"\"",
                new String[] {"date"}));
        STRING_FORMAT_RULES.put("time-string", new BuiltinRule("\"\\\"\" time \"\\\"\"",
                new String[] {"time"}));
        STRING_FORMAT_RULES.put("date-time-string", new BuiltinRule(
                "\"\\\"\" date-time \"\\\"\"", new String[] {"date-time"}));
    }

    /** Every name a schema property may not claim, llama.cpp {@code is_reserved_name}. */
    private static boolean isReservedName(String name) {
        return RESERVED_NAMES.contains(name);
    }

    private static final Set<String> RESERVED_NAMES = new HashSet<>();

    static {
        RESERVED_NAMES.add("root");
        RESERVED_NAMES.addAll(PRIMITIVE_RULES.keySet());
        RESERVED_NAMES.addAll(STRING_FORMAT_RULES.keySet());
    }

    // ------------------------------------------------------------------ schema model

    /** Kind of a normalized schema node; the mirror of llama.cpp's {@code node_kind}. */
    private enum Kind {
        REF, ANY_OF, ALL_OF, CONST, ENUM, OBJECT, TUPLE, ARRAY, STRING, INTEGER, NUMBER,
        BOOLEAN, NULL_KIND, ANY
    }

    private enum StringFormat {
        NONE, UUID, DATE, TIME, DATE_TIME
    }

    /** One normalized schema node; the fields are valid per {@link Kind}. */
    private static final class Schema {
        Kind kind;
        String ref;
        Schema target;
        List<Schema> children;
        List<Object> values;
        Object constValue;
        List<Property> properties;
        Schema additionalProperties;
        Schema items;
        int minItems;
        int maxItems = -1;
        String pattern = "";
        StringFormat format = StringFormat.NONE;
        int minLength;
        int maxLength = -1;
        long minimum = Long.MIN_VALUE;
        long maximum = Long.MAX_VALUE;
        boolean hasMin;
        boolean hasMax;
    }

    /** One object property with its required flag. */
    private static final class Property {
        final String name;
        final Schema schema;
        final boolean required;

        Property(String name, Schema schema, boolean required) {
            this.name = name;
            this.schema = schema;
            this.required = required;
        }
    }

    /** A schema normalization failure, the shape of llama.cpp {@code fail(path, msg)}. */
    static final class SchemaErrorException extends RuntimeException {
        SchemaErrorException(String path, String msg) {
            super(path.isEmpty() ? msg : "JSON schema error at " + path + ": " + msg);
        }
    }

    // ------------------------------------------------------------------ schema building

    /**
     * The port of {@code common_chat_schema_builder} ({@code common/json-schema.cpp:12-338}).
     */
    private static final class SchemaBuilder {
        private final Object root;
        private final Map<String, Schema> refs = new LinkedHashMap<>();
        private final List<Schema> pendingRefs = new ArrayList<>();

        SchemaBuilder(Object root) {
            this.root = root;
        }

        Schema build() {
            Schema schema = buildNode(root, "#");
            for (Schema ref : pendingRefs) {
                ref.target = refs.get(ref.ref);
            }
            return schema;
        }

        /** {@code get_count} ({@code common/json-schema.cpp:26-35}). */
        private static int getCount(Object schema, String key, String path, int def) {
            if (!(schema instanceof Map) || !((Map<?, ?>) schema).containsKey(key)) {
                return def;
            }
            Object value = ((Map<?, ?>) schema).get(key);
            if (!(value instanceof Long) && !(value instanceof Integer)) {
                fail(path, key + " must be a non-negative integer");
            }
            long count = ((Number) value).longValue();
            if (count < 0) {
                fail(path, key + " must be a non-negative integer");
            }
            if (count > Integer.MAX_VALUE) {
                fail(path, key + " " + count + " does not fit an int");
            }
            return (int) count;
        }

        /**
         * {@code get_bound} ({@code :38-48}); a fractional bound rounds inwards, towards the
         * integers the bound still admits.
         */
        private static long getBound(Object schema, String key, String path, boolean roundUp) {
            if (!(schema instanceof Map) || !((Map<?, ?>) schema).containsKey(key)) {
                fail(path, key + " must be a number");
            }
            Object value = ((Map<?, ?>) schema).get(key);
            if (value instanceof Long || value instanceof Integer) {
                return ((Number) value).longValue();
            }
            if (!(value instanceof Double) && !(value instanceof BigInteger)) {
                fail(path, key + " must be a number");
            }
            // beyond the int64 range the value is used as a float, as nlohmann parses it
            double d = ((Number) value).doubleValue();
            return (long) (roundUp ? Math.ceil(d) : Math.floor(d));
        }

        /** {@code get_format} ({@code :50-72}). */
        private static StringFormat getFormat(Object schema, String path) {
            if (!(schema instanceof Map) || !((Map<?, ?>) schema).containsKey("format")) {
                return StringFormat.NONE;
            }
            Object value = ((Map<?, ?>) schema).get("format");
            if (!(value instanceof String)) {
                fail(path, "format must be a string");
            }
            String format = (String) value;
            if ("date".equals(format)) {
                return StringFormat.DATE;
            }
            if ("time".equals(format)) {
                return StringFormat.TIME;
            }
            if ("date-time".equals(format)) {
                return StringFormat.DATE_TIME;
            }
            if ("uuid".equals(format) || (format.length() == 5 && format.startsWith("uuid")
                    && format.charAt(4) >= '1' && format.charAt(4) <= '5')) {
                return StringFormat.UUID;
            }
            return StringFormat.NONE;
        }

        /** {@code resolve_ref} ({@code :74-97}): pointer navigation into the document. */
        private Object resolveRef(String ref, String path) {
            Object target = root;
            List<String> tokens = splitSlash(ref.substring(1));
            for (int i = 1; i < tokens.size(); i++) {
                String token = tokens.get(i);
                if (target instanceof Map && ((Map<?, ?>) target).containsKey(token)) {
                    target = ((Map<?, ?>) target).get(token);
                } else if (target instanceof List) {
                    long index;
                    try {
                        index = Long.parseLong(token);
                    } catch (NumberFormatException e) {
                        index = ((List<?>) target).size();
                    }
                    if (index < 0 || index >= ((List<?>) target).size()) {
                        fail(path, "cannot resolve $ref " + ref + ", " + token + " is out of range");
                    }
                    target = ((List<?>) target).get((int) index);
                } else {
                    fail(path, "cannot resolve $ref " + ref + ", " + token + " not found");
                }
            }
            return target;
        }

        /** llama.cpp {@code string_split(s, "/")}: every side kept, empty ones included. */
        private static List<String> splitSlash(String text) {
            List<String> parts = new ArrayList<>();
            int start = 0;
            int slash = text.indexOf('/');
            while (slash >= 0) {
                parts.add(text.substring(start, slash));
                start = slash + 1;
                slash = text.indexOf('/', start);
            }
            parts.add(text.substring(start));
            return parts;
        }

        /** {@code build_ref} ({@code :99-115}) with the local-only and cycle rules. */
        private Schema buildRef(Object value, String path) {
            if (!(value instanceof String)) {
                fail(path, "$ref must be a string");
            }
            String ref = (String) value;
            if (!ref.startsWith("#/")) {
                fail(path, "unsupported $ref " + ref
                        + ", only references into the same document are supported");
            }
            Schema node = new Schema();
            node.kind = Kind.REF;
            node.ref = ref;
            if (!refs.containsKey(ref)) {
                // reserve the key first, so a cycle back to this $ref stops here
                refs.put(ref, null);
                refs.put(ref, buildNode(resolveRef(ref, path), ref));
            }
            pendingRefs.add(node);
            return node;
        }

        /** {@code build_alternatives} ({@code :117-131}) for anyOf/oneOf/allOf. */
        private Schema buildAlternatives(Object value, String path, Kind kind) {
            if (!(value instanceof List)) {
                fail(path, "must be an array of schemas");
            }
            List<?> alternatives = (List<?>) value;
            if (alternatives.isEmpty()) {
                fail(path, "must not be empty");
            }
            Schema node = new Schema();
            node.kind = kind;
            node.children = new ArrayList<>();
            for (int i = 0; i < alternatives.size(); i++) {
                node.children.add(buildNode(alternatives.get(i), path + "/" + i));
            }
            return node;
        }

        /** {@code build_object} ({@code :133-172}). */
        private Schema buildObject(Object schema, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            Schema node = new Schema();
            node.kind = Kind.OBJECT;
            node.properties = new ArrayList<>();

            Set<String> required = new HashSet<>();
            if (map.containsKey("required") && map.get("required") instanceof List) {
                for (Object candidate : (List<?>) map.get("required")) {
                    if (candidate instanceof String) {
                        required.add((String) candidate);
                    }
                }
            }

            if (map.containsKey("properties")) {
                Object properties = map.get("properties");
                if (!(properties instanceof Map)) {
                    fail(path, "properties must be an object");
                }
                for (Map.Entry<?, ?> entry : ((Map<?, ?>) properties).entrySet()) {
                    node.properties.add(new Property((String) entry.getKey(),
                            buildNode(entry.getValue(), path + "/properties/" + entry.getKey()),
                            required.contains(entry.getKey())));
                }
            }

            if (map.containsKey("additionalProperties")) {
                Object additional = map.get("additionalProperties");
                if (Boolean.TRUE.equals(additional)) {
                    node.additionalProperties = any();
                } else if (additional instanceof Map) {
                    node.additionalProperties =
                            buildNode(additional, path + "/additionalProperties");
                } else if (Boolean.FALSE.equals(additional)) {
                    // false closes the object
                } else {
                    fail(path, "additionalProperties must be a boolean or a schema");
                }
            } else if (!map.containsKey("properties")) {
                // {"type": "object"} on its own accepts any object
                node.additionalProperties = any();
            }
            return node;
        }

        /** {@code build_array} ({@code :174-195}); "items" wins over "prefixItems". */
        private Schema buildArray(Object schema, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            if (map.containsKey("items") || map.containsKey("prefixItems")) {
                String key = map.containsKey("items") ? "items" : "prefixItems";
                Object items = map.get(key);
                if (items instanceof List) {
                    Schema tuple = new Schema();
                    tuple.kind = Kind.TUPLE;
                    tuple.children = new ArrayList<>();
                    for (int i = 0; i < ((List<?>) items).size(); i++) {
                        tuple.children.add(buildNode(((List<?>) items).get(i),
                                path + "/" + key + "/" + i));
                    }
                    return tuple;
                }
                Schema node = new Schema();
                node.kind = Kind.ARRAY;
                node.items = buildNode(items, path + "/" + key);
                node.minItems = getCount(schema, "minItems", path, 0);
                node.maxItems = getCount(schema, "maxItems", path, -1);
                return node;
            }
            Schema node = new Schema();
            node.kind = Kind.ARRAY;
            node.items = any();
            node.minItems = getCount(schema, "minItems", path, 0);
            node.maxItems = getCount(schema, "maxItems", path, -1);
            return node;
        }

        /** {@code build_string} ({@code :197-210}). */
        private Schema buildString(Object schema, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            Schema node = new Schema();
            node.kind = Kind.STRING;
            if (map.containsKey("pattern")) {
                Object pattern = map.get("pattern");
                if (!(pattern instanceof String)) {
                    fail(path, "pattern must be a string");
                }
                node.pattern = (String) pattern;
            }
            node.format = getFormat(schema, path);
            node.minLength = getCount(schema, "minLength", path, 0);
            node.maxLength = getCount(schema, "maxLength", path, -1);
            return node;
        }

        /** {@code build_integer} ({@code :212-225}) with the exclusive variants. */
        private Schema buildInteger(Object schema, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            Schema node = new Schema();
            node.kind = Kind.INTEGER;
            if (map.containsKey("minimum")) {
                node.minimum = getBound(schema, "minimum", path, true);
                node.hasMin = true;
            } else if (map.containsKey("exclusiveMinimum")) {
                node.minimum = getBound(schema, "exclusiveMinimum", path, false) + 1;
                node.hasMin = true;
            }
            if (map.containsKey("maximum")) {
                node.maximum = getBound(schema, "maximum", path, false);
                node.hasMax = true;
            } else if (map.containsKey("exclusiveMaximum")) {
                node.maximum = getBound(schema, "exclusiveMaximum", path, true) - 1;
                node.hasMax = true;
            }
            return node;
        }

        /** {@code build_node} ({@code :227-323}). */
        private Schema buildNode(Object schema, String path) {
            if (!(schema instanceof Map)) {
                fail(path, "schema must be an object");
            }
            Map<?, ?> map = (Map<?, ?>) schema;

            if (map.containsKey("$ref")) {
                return buildRef(map.get("$ref"), path);
            }
            if (map.containsKey("oneOf") || map.containsKey("anyOf")) {
                String key = map.containsKey("oneOf") ? "oneOf" : "anyOf";
                return buildAlternatives(map.get(key), path + "/" + key, Kind.ANY_OF);
            }

            Object type = map.get("type");
            if (type instanceof List) {
                // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
                List<?> alternatives = (List<?>) type;
                if (alternatives.isEmpty()) {
                    fail(path, "type must not be empty");
                }
                Schema node = new Schema();
                node.kind = Kind.ANY_OF;
                node.children = new ArrayList<>();
                for (int i = 0; i < alternatives.size(); i++) {
                    Map<String, Object> alt = new LinkedHashMap<>();
                    for (Map.Entry<?, ?> entry : map.entrySet()) {
                        alt.put((String) entry.getKey(), entry.getValue());
                    }
                    alt.put("type", alternatives.get(i));
                    node.children.add(buildNode(alt, path + "/type/" + i));
                }
                return node;
            }

            if (map.containsKey("const")) {
                Schema node = new Schema();
                node.kind = Kind.CONST;
                node.constValue = map.get("const");
                return node;
            }
            if (map.containsKey("enum")) {
                Object values = map.get("enum");
                if (!(values instanceof List) || ((List<?>) values).isEmpty()) {
                    fail(path, "enum must be a non-empty array");
                }
                Schema node = new Schema();
                node.kind = Kind.ENUM;
                node.values = new ArrayList<>((List<?>) values);
                return node;
            }
            if (type != null && !(type instanceof String)) {
                fail(path, "type must be a string or an array of strings");
            }

            String typeName = type instanceof String ? (String) type : "";
            boolean hasProperties = map.containsKey("properties")
                    || (map.containsKey("additionalProperties")
                    && !Boolean.TRUE.equals(map.get("additionalProperties")));

            if (typeName.isEmpty()) {
                // without a type the structural keywords decide, in the converter's order
                if (hasProperties) {
                    return buildObject(schema, path);
                }
                if (map.containsKey("allOf")) {
                    return buildAlternatives(map.get("allOf"), path + "/allOf", Kind.ALL_OF);
                }
                if (map.containsKey("items") || map.containsKey("prefixItems")) {
                    return buildArray(schema, path);
                }
                if (map.containsKey("pattern") || map.containsKey("minLength")
                        || map.containsKey("maxLength")
                        || getFormat(schema, path) != StringFormat.NONE) {
                    return buildString(schema, path);
                }
                return any();
            }
            if (typeName.equals("object")) {
                if (!hasProperties && map.containsKey("allOf")) {
                    return buildAlternatives(map.get("allOf"), path + "/allOf", Kind.ALL_OF);
                }
                return buildObject(schema, path);
            }
            if (typeName.equals("string")) {
                if (map.containsKey("allOf")) {
                    return buildAlternatives(map.get("allOf"), path + "/allOf", Kind.ALL_OF);
                }
                return buildString(schema, path);
            }
            if (typeName.equals("array")) {
                return buildArray(schema, path);
            }
            if (typeName.equals("integer")) {
                return buildInteger(schema, path);
            }
            if (typeName.equals("number")) {
                return simple(Kind.NUMBER);
            }
            if (typeName.equals("boolean")) {
                return simple(Kind.BOOLEAN);
            }
            if (typeName.equals("null")) {
                return simple(Kind.NULL_KIND);
            }
            fail(path, "unrecognized type " + typeName);
            return null;
        }

        private Schema any() {
            return simple(Kind.ANY);
        }

        private Schema simple(Kind kind) {
            Schema node = new Schema();
            node.kind = kind;
            return node;
        }

        private static void fail(String path, String msg) {
            throw new SchemaErrorException(path, msg);
        }
    }

    // ------------------------------------------------------------------ conversion

    /** The port of {@code common_chat_schema_converter} ({@code :342-991}). */
    private static final class Converter {
        final GbnfBuilder rules;
        final Set<String> refsBeingResolved = new HashSet<>();
        final List<String> errors = new ArrayList<>();
        final List<String> warnings = new ArrayList<>();

        Converter(WhitespaceMode mode) {
            rules = new GbnfBuilder(mode.spaceRule());
        }

        /** {@code visit} ({@code :877-973}). */
        String visit(Schema schema, String name) {
            String ruleName = isReservedName(name) ? name + "-" : name.isEmpty() ? "root" : name;
            String subName = name + (name.isEmpty() ? "" : "-");

            switch (schema.kind) {
                case REF:
                    return rules.addRule(ruleName, resolveRef(schema));
                case ANY_OF:
                    return rules.addRule(ruleName, generateUnionRule(name, schema.children));
                case ALL_OF:
                    return visitAllOf(schema, name, ruleName);
                case CONST:
                    return rules.addRule(ruleName, generateConstantRule(schema.constValue));
                case ENUM: {
                    List<String> constants = new ArrayList<>();
                    for (Object value : schema.values) {
                        constants.add(generateConstantRule(value));
                    }
                    return rules.addRule(ruleName, "(" + String.join(" | ", constants) + ")");
                }
                case OBJECT: {
                    if (schema.properties.isEmpty() && schema.additionalProperties != null
                            && schema.additionalProperties.kind == Kind.ANY) {
                        return rules.addRule(ruleName, addPrimitive("object",
                                PRIMITIVE_RULES.get("object")));
                    }
                    List<Property> properties = schema.properties;
                    Set<String> required = new LinkedHashSet<>();
                    for (Property property : properties) {
                        if (property.required) {
                            required.add(property.name);
                        }
                    }
                    return rules.addRule(ruleName, buildObjectRule(properties, required, name,
                            schema.additionalProperties));
                }
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
                case STRING: {
                    if (!schema.pattern.isEmpty()) {
                        return visitPattern(schema.pattern, ruleName);
                    }
                    if (schema.format == StringFormat.UUID) {
                        return primitiveVisit(ruleName, "uuid");
                    }
                    if (schema.format != StringFormat.NONE) {
                        String prim = (schema.format == StringFormat.DATE ? "date"
                                : schema.format == StringFormat.TIME ? "time" : "date-time")
                                + "-string";
                        return rules.addRule(ruleName, addPrimitive(prim,
                                STRING_FORMAT_RULES.get(prim)));
                    }
                    if (schema.minLength > 0 || schema.maxLength >= 0) {
                        String charRule = addPrimitive("char", PRIMITIVE_RULES.get("char"));
                        int maxLength = schema.maxLength < 0 ? Integer.MAX_VALUE
                                : schema.maxLength;
                        return rules.addRule(ruleName, "\"\\\"\" " + GbnfBuilder.repetition(
                                charRule, schema.minLength, maxLength, "") + " \"\\\"\"");
                    }
                    return primitiveVisit(ruleName, "string");
                }
                case INTEGER: {
                    if (!schema.hasMin && !schema.hasMax) {
                        return primitiveVisit(ruleName, "integer");
                    }
                    StringBuilder out = new StringBuilder("(");
                    buildMinMaxInt(schema.hasMin ? schema.minimum : Long.MIN_VALUE,
                            schema.hasMax ? schema.maximum : Long.MAX_VALUE, out, 16, true);
                    out.append(")");
                    return rules.addRule(ruleName, out.toString());
                }
                case NUMBER:
                    return primitiveVisit(ruleName, "number");
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
            return addPrimitive(ruleName.equals("root") ? "root" : type, PRIMITIVE_RULES.get(type));
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

        /** {@code _visit_pattern} ({@code :390-404}): unsupported patterns degrade. */
        private String visitPattern(String pattern, String ruleName) {
            Map<String, String> snapshot = rules.snapshot();
            try {
                return RegexToGbnf.patternRule(pattern, ruleName, rules, false);
            } catch (RegexToGbnf.UnsupportedRegexException e) {
                rules.restore(snapshot);
                warnings.add("pattern " + pattern + " is not supported (" + e.getMessage()
                        + "), accepting any string");
                return rules.addRule(ruleName, addPrimitive("string",
                        PRIMITIVE_RULES.get("string")));
            } catch (RegexToGbnf.InvalidRegexException e) {
                rules.restore(snapshot);
                errors.add("Invalid pattern " + pattern + ": " + e.getMessage());
                return "";
            }
        }

        /** {@code _resolve_ref} ({@code :682-697}). */
        private String resolveRef(Schema ref) {
            String refName = "ref"
                    + ref.ref.substring(ref.ref.indexOf('#') + 1)
                    .replaceAll("[^a-zA-Z0-9-]+", "-");
            if (!rules.hasRule(refName) && !refsBeingResolved.contains(ref.ref)) {
                refsBeingResolved.add(ref.ref);
                refName = visit(ref.target, refName);
                refsBeingResolved.remove(ref.ref);
            }
            return refName;
        }

        /** {@code _build_object_rule} ({@code :699-794}). */
        private String buildObjectRule(List<Property> properties, Set<String> required,
                String name, Schema additionalProperties) {
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
            if (additionalProperties != null) {
                String subName = name + (name.isEmpty() ? "" : "-") + "additional";
                String valueRule = additionalProperties.kind != Kind.ANY
                        ? visit(additionalProperties, subName + "-value")
                        : addPrimitive("value", PRIMITIVE_RULES.get("value"));
                String keyRule = propNames.isEmpty()
                        ? addPrimitive("string", PRIMITIVE_RULES.get("string"))
                        : rules.addRule(subName + "-k", notStrings(propNames, this));
                String kvRule = rules.addRule(subName + "-kv",
                        keyRule + " \":\" space " + valueRule);
                propKvRuleNames.put("*", kvRule);
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

            if (!enumConstants.isEmpty()) {
                List<String> intersection = new ArrayList<>();
                for (Map.Entry<String, Integer> entry : enumConstants.entrySet()) {
                    if (entry.getValue() == schema.children.size()) {
                        intersection.add(entry.getKey());
                    }
                }
                if (!intersection.isEmpty()) {
                    return rules.addRule(ruleName,
                            "(" + String.join(" | ", intersection) + ")");
                }
            }
            return rules.addRule(ruleName, buildObjectRule(properties, required, name, null));
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

        /** {@code check_errors} ({@code :975-982}); warnings are informational in llama.cpp. */
        void checkErrors() {
            if (!errors.isEmpty()) {
                throw new IllegalArgumentException("JSON schema conversion failed:\n"
                        + String.join("\n", errors));
            }
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
    private static String dump(Object value) {
        return Json.toString(value, Json.Style.CANONICAL);
    }

    private static String generateConstantRule(Object value) {
        return formatLiteral(dump(value));
    }

    // ------------------------------------------------------------------ integer ranges

    /**
     * {@code _build_min_max_int} ({@code :45-227}) ported verbatim; the exact output of every
     * bounded-integer case in llama.cpp's tests is the golden reference.
     */
    private static void buildMinMaxInt(long minValue, long maxValue, StringBuilder out,
            int decimalsLeft, boolean topLevel) {
        boolean hasMin = minValue != Long.MIN_VALUE;
        boolean hasMax = maxValue != Long.MAX_VALUE;

        if (hasMin && hasMax) {
            if (minValue < 0 && maxValue < 0) {
                out.append("\"-\" (");
                buildMinMaxInt(-maxValue, -minValue, out, decimalsLeft, true);
                out.append(')');
                return;
            }
            if (minValue < 0) {
                out.append("\"-\" (");
                buildMinMaxInt(0, -minValue, out, decimalsLeft, true);
                out.append(") | ");
                minValue = 0;
            }

            String minDigits = Long.toString(minValue);
            String maxDigits = Long.toString(maxValue);
            for (int digits = minDigits.length(); digits < maxDigits.length(); digits++) {
                uniformRange(minDigits, "9".repeat(digits), out);
                minDigits = "1" + "0".repeat(digits);
                out.append(" | ");
            }
            uniformRange(minDigits, maxDigits, out);
            return;
        }

        int lessDecimals = Math.max(decimalsLeft - 1, 1);

        if (hasMin) {
            if (minValue < 0) {
                out.append("\"-\" (");
                buildMinMaxInt(Long.MIN_VALUE, -minValue, out, decimalsLeft, false);
                out.append(") | [0] | [1-9] ");
                moreDigits(0, decimalsLeft - 1, out);
            } else if (minValue == 0) {
                if (topLevel) {
                    out.append("[0] | [1-9] ");
                    moreDigits(0, lessDecimals, out);
                } else {
                    moreDigits(1, decimalsLeft, out);
                }
            } else if (minValue <= 9) {
                char c = (char) ('0' + minValue);
                char rangeStart = topLevel ? '1' : '0';
                if (c > rangeStart) {
                    digitRange(rangeStart, (char) (c - 1), out);
                    out.append(' ');
                    moreDigits(1, lessDecimals, out);
                    out.append(" | ");
                }
                digitRange(c, '9', out);
                out.append(' ');
                moreDigits(0, lessDecimals, out);
            } else {
                String digits = Long.toString(minValue);
                int length = digits.length();
                char c = digits.charAt(0);

                if (c > '1') {
                    digitRange(topLevel ? '1' : '0', (char) (c - 1), out);
                    out.append(' ');
                    moreDigits(length, lessDecimals, out);
                    out.append(" | ");
                }
                digitRange(c, c, out);
                out.append(" (");
                buildMinMaxInt(Long.parseLong(digits.substring(1)), Long.MAX_VALUE, out,
                        lessDecimals, false);
                out.append(')');
                if (c < '9') {
                    out.append(" | ");
                    digitRange((char) (c + 1), '9', out);
                    out.append(' ');
                    moreDigits(length - 1, lessDecimals, out);
                }
            }
            return;
        }

        if (hasMax) {
            if (maxValue >= 0) {
                if (topLevel) {
                    out.append("\"-\" [1-9] ");
                    moreDigits(0, lessDecimals, out);
                    out.append(" | ");
                }
                buildMinMaxInt(0, maxValue, out, decimalsLeft, true);
            } else {
                out.append("\"-\" (");
                buildMinMaxInt(-maxValue, Long.MAX_VALUE, out, decimalsLeft, false);
                out.append(')');
            }
            return;
        }
        throw new SchemaErrorException("", "At least one of min_value or max_value must be set");
    }

    private static void digitRange(char from, char to, StringBuilder out) {
        out.append('[');
        if (from == to) {
            out.append(from);
        } else {
            out.append(from).append('-').append(to);
        }
        out.append(']');
    }

    private static void moreDigits(int minDigits, int maxDigits, StringBuilder out) {
        out.append("[0-9]");
        if (minDigits == maxDigits && minDigits == 1) {
            return;
        }
        out.append('{').append(minDigits);
        if (maxDigits != minDigits) {
            out.append(',');
            if (maxDigits != Integer.MAX_VALUE) {
                out.append(maxDigits);
            }
        }
        out.append('}');
    }

    private static void uniformRange(String from, String to, StringBuilder out) {
        int i = 0;
        while (i < from.length() && i < to.length() && from.charAt(i) == to.charAt(i)) {
            i++;
        }
        if (i > 0) {
            out.append('"').append(from, 0, i).append('"');
        }
        if (i < from.length() && i < to.length()) {
            if (i > 0) {
                out.append(' ');
            }
            int subLength = from.length() - i - 1;
            if (subLength > 0) {
                String fromSub = from.substring(i + 1);
                String toSub = to.substring(i + 1);
                String subZeros = "0".repeat(subLength);
                String subNines = "9".repeat(subLength);

                boolean toReached = false;
                out.append('(');
                if (fromSub.equals(subZeros)) {
                    digitRange(from.charAt(i), (char) (to.charAt(i) - 1), out);
                    out.append(' ');
                    moreDigits(subLength, subLength, out);
                } else {
                    out.append('[').append(from.charAt(i)).append("] (");
                    uniformRange(fromSub, subNines, out);
                    out.append(')');
                    if (from.charAt(i) < to.charAt(i) - 1) {
                        out.append(" | ");
                        if (toSub.equals(subNines)) {
                            digitRange((char) (from.charAt(i) + 1), to.charAt(i), out);
                            toReached = true;
                        } else {
                            digitRange((char) (from.charAt(i) + 1),
                                    (char) (to.charAt(i) - 1), out);
                        }
                        out.append(' ');
                        moreDigits(subLength, subLength, out);
                    }
                }
                if (!toReached) {
                    out.append(" | ");
                    digitRange(to.charAt(i), to.charAt(i), out);
                    out.append(' ');
                    uniformRange(subZeros, toSub, out);
                }
                out.append(')');
            } else {
                out.append('[').append(from.charAt(i)).append('-').append(to.charAt(i))
                        .append(']');
            }
        }
    }
}
