package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * vLLM's {@code disable_additional_properties} schema preprocessing (guidance's
 * {@code process_for_additional_properties}): every schema object that declares {@code properties} or
 * {@code patternProperties} and no {@code additionalProperties} gains {@code "additionalProperties": false}.
 *
 * <p>Only schema positions are visited — the member maps {@code properties}, {@code patternProperties},
 * {@code $defs}, {@code definitions}, {@code dependentSchemas} and {@code dependencies}, and the sub-schema
 * keywords {@code additionalProperties}, {@code unevaluatedProperties}, {@code propertyNames}, {@code contains},
 * {@code additionalItems}, {@code unevaluatedItems}, {@code not}, {@code if}/{@code then}/{@code else},
 * {@code contentSchema}, {@code items}, {@code prefixItems}, {@code allOf}, {@code anyOf} and {@code oneOf} — so a
 * literal value under {@code const}, {@code enum}, {@code default} or {@code examples} that happens to hold a
 * {@code properties} member is never rewritten. An explicit {@code additionalProperties} is kept as written. The
 * walk builds new maps and lists for the schema positions it visits and never modifies its input; a value outside a
 * schema position (a {@code const}, {@code enum}, {@code default} or {@code examples} literal) is kept as it is.
 */
final class JsonSchemaAdditionalProperties {

    /** Keywords whose value maps names to subschemas. */
    private static final Set<String> MEMBER_MAPS = new HashSet<>(Arrays.asList(
            "properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"));

    /** Keywords whose value is a subschema, or a list of subschemas. */
    private static final Set<String> SUBSCHEMAS = new HashSet<>(Arrays.asList(
            "additionalProperties", "unevaluatedProperties", "propertyNames", "contains", "additionalItems",
            "unevaluatedItems", "not", "if", "then", "else", "contentSchema", "items", "prefixItems", "allOf",
            "anyOf", "oneOf"));

    private JsonSchemaAdditionalProperties() {
    }

    /**
     * The schema document {@code document} with every object that declares properties closed: its schema members
     * are rebuilt recursively, its other values stay as they are.
     */
    static Object closed(Object document) {
        if (document instanceof Map) {
            return closedObject((Map<?, ?>) document);
        }
        if (document instanceof List) {
            return closedList((List<?>) document);
        }
        return document;
    }

    private static Map<String, Object> closedObject(Map<?, ?> schema) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : schema.entrySet()) {
            String key = (String) entry.getKey();
            out.put(key, closedValue(key, entry.getValue()));
        }
        if (!out.containsKey("additionalProperties")
                && (out.get("properties") instanceof Map || out.get("patternProperties") instanceof Map)) {
            out.put("additionalProperties", Boolean.FALSE);
        }
        return out;
    }

    private static List<Object> closedList(List<?> schemas) {
        List<Object> out = new ArrayList<>(schemas.size());
        for (Object schema : schemas) {
            out.add(schema instanceof Map ? closedObject((Map<?, ?>) schema) : schema);
        }
        return out;
    }

    /** The value of {@code keyword}: a schema position is visited, any other value is kept. */
    private static Object closedValue(String keyword, Object value) {
        if (MEMBER_MAPS.contains(keyword) && value instanceof Map) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                Object subschema = entry.getValue();
                out.put((String) entry.getKey(),
                        subschema instanceof Map ? closedObject((Map<?, ?>) subschema) : subschema);
            }
            return out;
        }
        if (SUBSCHEMAS.contains(keyword) && (value instanceof Map || value instanceof List)) {
            return closed(value);
        }
        return value;
    }
}
