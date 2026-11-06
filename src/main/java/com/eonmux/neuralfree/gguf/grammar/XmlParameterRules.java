package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.Kind;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.Property;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaTree.Schema;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * The top-level object of a schema as the tagged parameters of an {@link XmlParameterStyle}, the XML styles of
 * xgrammar's structural-tag {@code json_schema} format (its {@code JSONSchemaConverter} XML mode). Whitespace
 * ({@code xml-ws}, {@link JsonSchemaOptions#xmlSpaceRule}) stands before every parameter and after the last; the
 * properties come in schema order, the required ones always, the {@code additionalProperties} after them under
 * {@code [a-zA-Z_][a-zA-Z0-9_]*} names, and {@code minProperties} / {@code maxProperties} bound the count
 * ({@link CountedMembers}); under {@code any_order} each parameter is any of them ({@code anyOrderObject}).
 *
 * <p>A parameter's value is written by its schema: an unconstrained string is any text up to the closing tag
 * ({@code xml-string}, {@link GbnfExclusion}), with no whitespace around it; a string with a {@code pattern},
 * {@code format} or length bounds, a string constant and a string enum member are their raw, unquoted text; every
 * other value (numbers, booleans, null, arrays, objects, the non-string constants) is its JSON text, nested objects
 * staying JSON; a union is the union of its alternatives' values; an unconstrained value is a string, a JSON array
 * or a JSON object. Outside the unconstrained string, whitespace may surround the value. In a typed style
 * ({@code deepseek_xml}) the opening tag declares {@code string="true"} before a string value, written with no
 * whitespace around it, and {@code string="false"} before any other value.</p>
 *
 * <p>A non-object schema at the top level is written as one parameter value is: an unconstrained string as
 * {@code xml-string}, any other value as JSON. {@code patternProperties} and {@code propertyNames} of the top-level
 * object are refused by name: the parameter names are raw text that no JSON key pattern describes.</p>
 */
final class XmlParameterRules {

    private static final String XML_NAME = "[a-zA-Z_] [a-zA-Z0-9_]*";
    private static final String ANY_CHARACTER = "[\\U00000000-\\U0010FFFF]";

    private final JsonSchemaToGbnf.Converter converter;
    private final GbnfBuilder rules;
    private final XmlParameterStyle style;
    private final boolean anyOrder;
    private final String ws;
    private final String text;
    private final String name;
    private final JsonSchemaToGbnf.ObjectSyntax syntax;

    /**
     * Seeds the XML rules ({@code xml-ws}, {@code xml-string}, {@code xml-name}) into {@code converter}'s registry
     * before any schema rule takes a name.
     */
    XmlParameterRules(JsonSchemaToGbnf.Converter converter, JsonSchemaOptions options) {
        this.converter = converter;
        this.rules = converter.rules;
        this.style = options.style();
        this.anyOrder = options.anyOrder();
        this.ws = rules.addRule("xml-ws", options.xmlSpaceRule());
        this.text = GbnfExclusion.excluding(rules::addRule, "xml-string", Collections.singletonList(style.valueClose));
        this.name = rules.addRule("xml-name", XML_NAME);
        this.syntax = new JsonSchemaToGbnf.ObjectSyntax(ws, ws, ws, ws);
    }

    /** Adds the {@code root} rule of {@code schema}. */
    void root(Schema schema) {
        String body = topLevel(schema, "", identitySet());
        if (body == null) {
            converter.visit(schema, "");
        } else {
            rules.addRule(Grammar.ROOT_RULE, body);
        }
    }

    /** Adds the rule of {@code schema} under {@code name}, its helper rules prefixed by it; returns the stored name. */
    String rule(Schema schema, String name) {
        String body = topLevel(schema, name, identitySet());
        return body == null ? converter.visit(schema, name) : rules.addRule(name, body);
    }

    /** The rule text of a top-level {@code schema}, or null when it is a JSON value. */
    private String topLevel(Schema schema, String base, Set<Schema> visiting) {
        switch (schema.kind) {
            case OBJECT:
                return object(schema, base);
            case ANY:
                return anyObject();
            case STRING:
                return string(schema, base.isEmpty() ? "root-value" : base);
            case REF:
                if (schema.target == null || !visiting.add(schema.target)) {
                    return null;
                }
                try {
                    return topLevel(schema.target, base, visiting);
                } finally {
                    visiting.remove(schema.target);
                }
            case ANY_OF:
                return alternatives(schema.children, base, visiting);
            case ALL_OF: {
                JsonSchemaToGbnf.AllOfParts parts = converter.allOfParts(schema);
                return parts.intersection.isEmpty() ? object(merged(parts), base) : null;
            }
            default:
                return null;
        }
    }

    private String alternatives(List<Schema> children, String base, Set<Schema> visiting) {
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < children.size(); i++) {
            String alternativeName = base + (base.isEmpty() ? "alternative-" : "-") + i;
            String body = topLevel(children.get(i), alternativeName, visiting);
            parts.add(body == null ? converter.visit(children.get(i), alternativeName)
                    : rules.addRule(alternativeName, body));
        }
        return String.join(" | ", parts);
    }

    /** The object an {@code allOf} merges, its properties required as a required component declares them. */
    private static Schema merged(JsonSchemaToGbnf.AllOfParts parts) {
        Schema node = new Schema();
        node.kind = Kind.OBJECT;
        node.properties = new ArrayList<>();
        for (Property property : parts.properties) {
            node.properties.add(new Property(property.name, property.schema, parts.required.contains(property.name)));
        }
        return node;
    }

    // ------------------------------------------------------------------ objects

    /** Any parameters with any values: {@code json_schema} {@code true}, {@code {}} or a bare object. */
    private String anyObject() {
        String member = rules.addRule("xml-any-param", parameter(literal(style.keyOpen) + " " + name, null,
                "xml-any"));
        return rules.addRule("xml-object", ws + " (" + member + " " + ws + ")*");
    }

    private String object(Schema schema, String base) {
        String object = base.isEmpty() ? "root" : base;
        if (!schema.patternProperties.isEmpty() || schema.propertyNames != null) {
            converter.errors.add((schema.propertyNames != null ? "propertyNames" : "patternProperties") + " of "
                    + object + " are not supported in the " + style.keyword + " style");
            return ws;
        }
        if (schema.properties.isEmpty() && schema.additionalProperties != null
                && schema.additionalProperties.kind == Kind.ANY && !schema.countsProperties()) {
            return anyObject();
        }
        String prefix = base.isEmpty() ? "" : base + "-";
        List<String> members = new ArrayList<>();
        boolean[] required = new boolean[schema.properties.size()];
        for (int i = 0; i < schema.properties.size(); i++) {
            Property property = schema.properties.get(i);
            required[i] = property.required;
            members.add(rules.addRule(prefix + property.name + "-param", parameter(
                    literal(style.keyOpen + property.name), property.schema, prefix + property.name)));
        }
        String additional = schema.additionalProperties == null ? null
                : rules.addRule(prefix + "additional-param", parameter(literal(style.keyOpen) + " " + name,
                        schema.additionalProperties.kind == Kind.ANY ? null : schema.additionalProperties,
                        prefix + "additional-value"));
        int high = schema.maxProperties < 0 ? CountedMembers.UNBOUNDED : schema.maxProperties;
        if (anyOrder) {
            List<String> items = new ArrayList<>(members);
            if (additional != null) {
                items.add(additional);
            }
            int requiredCount = 0;
            for (boolean flag : required) {
                requiredCount += flag ? 1 : 0;
            }
            return converter.anyOrderObject(items, Math.max(schema.minProperties, requiredCount), high, syntax, base);
        }
        return counted(new CountedMembers(rules, members, required, additional, ws, prefix + "count-"),
                schema.minProperties, high, object);
    }

    private String counted(CountedMembers members, int low, int high, String object) {
        if (high < members.requiredCount()) {
            converter.errors.add("maxProperties " + high + " is less than the " + members.requiredCount()
                    + " required properties of " + object);
            return ws;
        }
        if (low > members.capacity()) {
            converter.errors.add("minProperties " + low + " exceeds the " + members.capacity() + " properties "
                    + object + " can hold");
            return ws;
        }
        String selection = members.first(low, high);
        return selection.isEmpty() ? ws : ws + " " + selection + " " + ws;
    }

    // ------------------------------------------------------------------ parameters and values

    /**
     * One parameter: {@code key} (the GBNF of the opening text through the name), then the value of
     * {@code value} (null: any value) up to the closing tag.
     */
    private String parameter(String key, Schema value, String valueName) {
        List<String> strings = new ArrayList<>();
        List<String> others = new ArrayList<>();
        values(value, valueName, strings, others, identitySet());
        if (strings.isEmpty() && others.isEmpty()) {
            converter.errors.add("no value of " + valueName + " can be written");
            return key;
        }
        String close = literal(style.valueClose);
        if (style.typed) {
            List<String> kinds = new ArrayList<>();
            if (!strings.isEmpty()) {
                kinds.add(style.keyEnd(ws, true) + " " + group(strings) + " " + close);
            }
            if (!others.isEmpty()) {
                kinds.add(style.keyEnd(ws, false) + " " + ws + " " + group(others) + " " + ws + " " + close);
            }
            return key + " (" + String.join(" | ", kinds) + ")";
        }
        List<String> all = new ArrayList<>(strings);
        all.addAll(others);
        String body = isPlainString(value) ? text : ws + " " + group(all) + " " + ws;
        return key + " " + style.keyEnd(ws, false) + " " + body + " " + close;
    }

    /** Sorts the value rules of {@code schema} (null: any value) into the string values and all others. */
    private void values(Schema schema, String valueName, List<String> strings, List<String> others,
            Set<Schema> visiting) {
        if (schema == null || schema.kind == Kind.ANY) {
            strings.add(text);
            if (style.typed) {
                others.add(converter.primitive("number"));
                others.add(converter.primitive("boolean"));
                others.add(converter.primitive("null"));
            }
            others.add(converter.primitive("array"));
            others.add(converter.primitive("object"));
            return;
        }
        switch (schema.kind) {
            case STRING:
                strings.add(string(schema, valueName));
                return;
            case CONST:
                constant(schema.constValue, strings, others);
                return;
            case ENUM:
                for (Object constant : schema.values) {
                    constant(constant, strings, others);
                }
                return;
            case ANY_OF:
                for (int i = 0; i < schema.children.size(); i++) {
                    values(schema.children.get(i), valueName + "-" + i, strings, others, visiting);
                }
                return;
            case REF:
                if (schema.target != null && splits(schema.target.kind) && visiting.add(schema.target)) {
                    values(schema.target, valueName, strings, others, visiting);
                    visiting.remove(schema.target);
                    return;
                }
                others.add(converter.visit(schema, valueName));
                return;
            default:
                others.add(converter.visit(schema, valueName));
        }
    }

    /** Whether a schema of {@code kind} may hold string values, so its alternatives are sorted one by one. */
    private static boolean splits(Kind kind) {
        return kind == Kind.STRING || kind == Kind.CONST || kind == Kind.ENUM || kind == Kind.ANY_OF
                || kind == Kind.REF || kind == Kind.ANY;
    }

    private static void constant(Object value, List<String> strings, List<String> others) {
        if (value instanceof String) {
            strings.add(literal((String) value));
        } else {
            others.add(literal(JsonSchemaToGbnf.dump(value)));
        }
    }

    /** The raw text of a string schema: its pattern, format or length bounds, else any text up to the tag. */
    private String string(Schema schema, String valueName) {
        if (!schema.pattern.isEmpty()) {
            return converter.rawPattern(schema.pattern, valueName);
        }
        if (schema.format != JsonStringFormats.Format.NONE) {
            return converter.rawFormat(schema.format);
        }
        if (schema.minLength > 0 || schema.maxLength >= 0) {
            String character = rules.addRule("xml-char", ANY_CHARACTER);
            int maxLength = schema.maxLength < 0 ? Integer.MAX_VALUE : schema.maxLength;
            return rules.addRule(valueName, GbnfBuilder.repetition(character, schema.minLength, maxLength, ""));
        }
        return text;
    }

    /** Whether {@code schema} is a string without pattern, format or length bounds. */
    private static boolean isPlainString(Schema schema) {
        return schema != null && schema.kind == Kind.STRING && schema.pattern.isEmpty()
                && schema.format == JsonStringFormats.Format.NONE && schema.minLength == 0 && schema.maxLength < 0;
    }

    private static String group(List<String> alternatives) {
        return alternatives.size() == 1 ? alternatives.get(0) : "(" + String.join(" | ", alternatives) + ")";
    }

    private static String literal(String raw) {
        return JsonSchemaToGbnf.formatLiteral(raw);
    }

    private static Set<Schema> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }
}
