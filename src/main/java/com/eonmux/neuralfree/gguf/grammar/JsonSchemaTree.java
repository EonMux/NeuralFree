package com.eonmux.neuralfree.gguf.grammar;

import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The normalized schema tree {@link JsonSchemaToGbnf} converts: one {@link Schema} node per subschema with its
 * {@link Kind}, the object properties with their required flags, the pattern properties, the array item and tuple
 * shapes, the string, numeric and count bounds and the {@code $ref} targets. {@link SchemaBuilder} is the port of
 * llama.cpp's {@code common_chat_schema_builder} ({@code common/json-schema.cpp:12-338}) that builds it from the
 * parsed schema document, refusing what it cannot normalize with a {@link JsonSchemaToGbnf.SchemaErrorException}.
 */
final class JsonSchemaTree {

    private JsonSchemaTree() {
    }

    /** Kind of a normalized schema node; the mirror of llama.cpp's {@code node_kind}. */
    enum Kind {
        REF, ANY_OF, ALL_OF, CONST, ENUM, OBJECT, TUPLE, ARRAY, STRING, INTEGER, NUMBER,
        BOOLEAN, NULL_KIND, ANY
    }

    /** One normalized schema node; the fields are valid per {@link Kind}. */
    static final class Schema {
        Kind kind;
        String ref;
        Schema target;
        List<Schema> children;
        List<Object> values;
        Object constValue;
        List<Property> properties;
        /** The property names whose schema is {@code false}: keys the object never holds. */
        List<String> forbiddenProperties = Collections.emptyList();
        Schema additionalProperties;
        List<PatternProperty> patternProperties = Collections.emptyList();
        Schema propertyNames;
        int minProperties;
        int maxProperties = -1;
        Schema items;
        int minItems;
        int maxItems = -1;
        String pattern = "";
        JsonStringFormats.Format format = JsonStringFormats.Format.NONE;
        int minLength;
        int maxLength = -1;
        SchemaBound lower;
        SchemaBound upper;

        boolean countsProperties() {
            return minProperties > 0 || maxProperties >= 0;
        }
    }

    /** One object property with its required flag. */
    static final class Property {
        final String name;
        final Schema schema;
        final boolean required;

        Property(String name, Schema schema, boolean required) {
            this.name = name;
            this.schema = schema;
            this.required = required;
        }
    }

    /** One {@code patternProperties} entry: the key pattern and the schema of its values. */
    static final class PatternProperty {
        final String pattern;
        final Schema schema;

        PatternProperty(String pattern, Schema schema) {
            this.pattern = pattern;
            this.schema = schema;
        }
    }

    /**
     * The port of {@code common_chat_schema_builder} ({@code common/json-schema.cpp:12-338}).
     */
    static final class SchemaBuilder {
        private final Object root;
        private final SchemaDocument document;
        private final Map<String, Schema> refs = new LinkedHashMap<>();
        private final List<Schema> pendingRefs = new ArrayList<>();
        private URI base = SchemaDocument.DEFAULT_BASE;

        SchemaBuilder(Object root) {
            this.root = root;
            this.document = new SchemaDocument(root);
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

        /** {@code get_format} ({@code :50-72}) over the formats of {@link JsonStringFormats}. */
        private static JsonStringFormats.Format getFormat(Object schema, String path) {
            if (!(schema instanceof Map) || !((Map<?, ?>) schema).containsKey("format")) {
                return JsonStringFormats.Format.NONE;
            }
            Object value = ((Map<?, ?>) schema).get("format");
            if (!(value instanceof String)) {
                fail(path, "format must be a string");
            }
            return JsonStringFormats.of((String) value);
        }

        /** {@code build_ref} ({@code :99-115}) over the document's base URIs and anchors. */
        private Schema buildRef(Object value, String path) {
            if (!(value instanceof String)) {
                fail(path, "$ref must be a string");
            }
            SchemaDocument.Target target = document.resolve((String) value, base, path);
            Schema node = new Schema();
            node.kind = Kind.REF;
            node.ref = target.key;
            if (!refs.containsKey(target.key)) {
                // reserve the key first, so a cycle back to this target stops here
                refs.put(target.key, null);
                refs.put(target.key, buildNode(target.node, target.key));
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
                if (kind != Kind.ANY_OF || !Boolean.FALSE.equals(alternatives.get(i))) {
                    node.children.add(buildNode(alternatives.get(i), path + "/" + i));
                }
            }
            if (node.children.isEmpty()) {
                fail(path, "every alternative is the schema false, which admits no value");
            }
            return node;
        }

        /** {@code build_object} ({@code :133-172}) with the key and count keywords. */
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
                node.forbiddenProperties = new ArrayList<>();
                for (Map.Entry<?, ?> entry : ((Map<?, ?>) properties).entrySet()) {
                    String key = (String) entry.getKey();
                    if (Boolean.FALSE.equals(entry.getValue())) {
                        if (required.contains(key)) {
                            fail(path, "required property " + key + " has the schema false, which admits no value");
                        }
                        node.forbiddenProperties.add(key);
                        continue;
                    }
                    node.properties.add(new Property(key, buildNode(entry.getValue(), path + "/properties/" + key),
                            required.contains(key)));
                }
            }
            buildPatternProperties(map, node, path);
            buildPropertyNames(map, node, path);

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
                if (node.additionalProperties != null && map.containsKey("patternProperties")) {
                    fail(path, "additionalProperties cannot be combined with patternProperties: the keys outside"
                            + " the patterns cannot be kept from matching them");
                }
            } else if (!map.containsKey("properties") && !map.containsKey("patternProperties")) {
                // {"type": "object"} on its own accepts any object
                node.additionalProperties = any();
            }
            if (Boolean.FALSE.equals(map.get("propertyNames"))) {
                // no key is a valid name: the object is empty
                node.additionalProperties = null;
            }

            node.minProperties = getCount(schema, "minProperties", path, 0);
            node.maxProperties = getCount(schema, "maxProperties", path, -1);
            if (node.maxProperties >= 0 && node.minProperties > node.maxProperties) {
                fail(path, "minProperties " + node.minProperties + " exceeds maxProperties "
                        + node.maxProperties);
            }
            return node;
        }

        /**
         * {@code patternProperties}: every pattern must share one schema (a key matching two patterns needs both
         * schemas), and no declared property may match a pattern (its value would need both schemas).
         */
        private void buildPatternProperties(Map<?, ?> map, Schema node, String path) {
            if (!map.containsKey("patternProperties")) {
                return;
            }
            Object patterns = map.get("patternProperties");
            if (!(patterns instanceof Map)) {
                fail(path, "patternProperties must be an object");
            }
            node.patternProperties = new ArrayList<>();
            List<String> declared = new ArrayList<>(node.forbiddenProperties);
            for (Property property : node.properties) {
                declared.add(property.name);
            }
            String shared = null;
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) patterns).entrySet()) {
                String pattern = (String) entry.getKey();
                for (String name : declared) {
                    if (searches(pattern, name, path)) {
                        fail(path, "property " + name + " also matches patternProperties pattern " + pattern
                                + ": its value would need both schemas");
                    }
                }
                if (Boolean.FALSE.equals(entry.getValue())) {
                    continue; // the keys it matches are never written
                }
                String dumped = JsonSchemaToGbnf.dump(entry.getValue());
                if (shared != null && !shared.equals(dumped)) {
                    fail(path, "patternProperties patterns with different schemas are not supported: a key"
                            + " matching several patterns would need all of their schemas");
                }
                shared = dumped;
                node.patternProperties.add(new PatternProperty(pattern,
                        buildNode(entry.getValue(), path + "/patternProperties/" + pattern)));
            }
        }

        /** Whether {@code pattern} matches somewhere in {@code name}, JSON schema's unanchored regex semantics. */
        private static boolean searches(String pattern, String name, String path) {
            try {
                return RegexToGbnf.searches(pattern, name);
            } catch (RegexToGbnf.InvalidRegexException e) {
                fail(path, "patternProperties pattern " + pattern + " is not a valid regular expression: "
                        + e.getMessage());
                return false;
            } catch (RegexToGbnf.UnsupportedRegexException e) {
                fail(path, "key pattern " + pattern + " is not supported: " + e.getMessage());
                return false;
            }
        }

        /** {@code propertyNames}: the key schema of an object whose keys are all additional ones. */
        private void buildPropertyNames(Map<?, ?> map, Schema node, String path) {
            Object names = map.get("propertyNames");
            if (names == null || Boolean.TRUE.equals(names)) {
                return;
            }
            if (map.containsKey("properties")) {
                fail(path, "propertyNames cannot be combined with properties");
            }
            if (map.containsKey("patternProperties")) {
                fail(path, "propertyNames cannot be combined with patternProperties");
            }
            if (!Boolean.FALSE.equals(names)) {
                node.propertyNames = buildNode(names, path + "/propertyNames");
            }
        }

        /** {@code build_array} ({@code :174-195}); "items" wins over "prefixItems". */
        private Schema buildArray(Object schema, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            if (map.containsKey("items") || map.containsKey("prefixItems")) {
                String key = map.containsKey("items") ? "items" : "prefixItems";
                Object items = map.get(key);
                if (Boolean.FALSE.equals(items)) {
                    // no item beyond the prefix: the prefix tuple, or the empty array
                    if (map.get("prefixItems") instanceof List) {
                        key = "prefixItems";
                        items = map.get(key);
                    } else {
                        if (getCount(schema, "minItems", path, 0) > 0) {
                            fail(path, "minItems needs items, but items is the schema false");
                        }
                        Schema empty = simple(Kind.ARRAY);
                        empty.items = any();
                        empty.maxItems = 0;
                        return empty;
                    }
                }
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

        /** {@code build_integer} ({@code :212-225}) over {@link SchemaBound}, any integral bound. */
        private Schema buildInteger(Object schema, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            Schema node = simple(Kind.INTEGER);
            node.lower = SchemaBound.lower(map, path);
            node.upper = SchemaBound.upper(map, path);
            if (node.lower != null && node.upper != null
                    && node.lower.lowestInteger().compareTo(node.upper.highestInteger()) > 0) {
                fail(path, "no integer lies within the minimum and maximum bounds");
            }
            return node;
        }

        /** A {@code number} node with its decimal bounds. */
        private Schema buildNumber(Object schema, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            Schema node = simple(Kind.NUMBER);
            node.lower = SchemaBound.lower(map, path);
            node.upper = SchemaBound.upper(map, path);
            if (!SchemaBound.admitsValue(node.lower, node.upper)) {
                fail(path, "no number lies within the minimum and maximum bounds");
            }
            return node;
        }

        /** {@code build_node} ({@code :227-323}) within the base URI the node declares. */
        private Schema buildNode(Object schema, String path) {
            if (Boolean.TRUE.equals(schema)) {
                return any();
            }
            if (Boolean.FALSE.equals(schema)) {
                fail(path, "the schema false admits no value");
            }
            URI saved = base;
            URI own = document.baseOf(schema);
            if (own != null) {
                base = own;
            }
            try {
                return buildNodeInScope(schema, path);
            } finally {
                base = saved;
            }
        }

        private Schema buildNodeInScope(Object schema, String path) {
            if (!(schema instanceof Map)) {
                fail(path, "schema must be an object or a boolean");
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
                return buildTypeUnion(map, (List<?>) type, path);
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
            return buildTyped(schema, type instanceof String ? (String) type : "", path);
        }

        /** {@code {"type": ["a", "b"], ...}} is {@code {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}}. */
        private Schema buildTypeUnion(Map<?, ?> map, List<?> alternatives, String path) {
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

        private Schema buildTyped(Object schema, String typeName, String path) {
            Map<?, ?> map = (Map<?, ?>) schema;
            boolean hasProperties = map.containsKey("properties")
                    || map.containsKey("patternProperties") || map.containsKey("propertyNames")
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
                        || getFormat(schema, path) != JsonStringFormats.Format.NONE) {
                    return buildString(schema, path);
                }
                if (map.containsKey("minProperties") || map.containsKey("maxProperties")) {
                    return buildObject(schema, path);
                }
                return any();
            }
            switch (typeName) {
                case "object":
                    if (!hasProperties && map.containsKey("allOf")) {
                        return buildAlternatives(map.get("allOf"), path + "/allOf", Kind.ALL_OF);
                    }
                    return buildObject(schema, path);
                case "string":
                    if (map.containsKey("allOf")) {
                        return buildAlternatives(map.get("allOf"), path + "/allOf", Kind.ALL_OF);
                    }
                    return buildString(schema, path);
                case "array":
                    return buildArray(schema, path);
                case "integer":
                    return buildInteger(schema, path);
                case "number":
                    return buildNumber(schema, path);
                case "boolean":
                    return simple(Kind.BOOLEAN);
                case "null":
                    return simple(Kind.NULL_KIND);
                default:
                    fail(path, "unrecognized type " + typeName);
                    return null;
            }
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
            throw new JsonSchemaToGbnf.SchemaErrorException(path, msg);
        }
    }
}
