package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The structural-tag parser (PARITY SO-4): the vLLM legacy {@code structural_tag} payload —
 * {@code {"structures": [{"begin", "schema", "end"}...], "triggers": [...]}} — compiles to one grammar whose
 * root is the alternation of the structures, each {@code begin} literal, then the {@code schema}'s grammar
 * ({@link JsonSchemaToGbnf} in the spec's whitespace mode), then the {@code end} literal. The structures'
 * grammars merge at the rule level: every rule renames under a per-structure prefix and its rule references
 * re-index, so several structures compose without collisions.
 *
 * <p>A non-empty {@code triggers} array gates the grammar as xgrammar's tag dispatch does ({@link #triggers}): the
 * output is free text until a trigger appears, the structure whose {@code begin} the trigger opens is then
 * enforced to its {@code end}, and free text resumes. Every structure's {@code begin} must start with one of the
 * triggers, which xgrammar requires too. Without triggers the grammar requires a structure outright.
 *
 * <p>Refusals name their field: a payload without a {@code structures} array (the modern xgrammar
 * {@code format} object is pending), a trigger that is not a non-empty string, a structure whose {@code begin} no
 * trigger opens, or a structure missing one of its three members, all as {@link IllegalArgumentException}.
 */
public final class StructuralTag {

    private StructuralTag() {
    }

    /** The validated grammar of the legacy payload text ({@link ConstraintSpec#structuralTag}). */
    public static Grammar parse(String payload, JsonSchemaToGbnf.WhitespaceMode whitespace) {
        JsonView root = root(payload);
        triggerWords(root);
        List<?> structures = asList(root, "structures");
        if (structures.isEmpty()) {
            throw new IllegalArgumentException("structural tag carries no structures");
        }

        List<List<GrammarElement>> rules = new ArrayList<>();
        Map<String, Integer> symbolIds = new TreeMap<>();
        List<String> symbolNames = new ArrayList<>();
        List<Integer> structureRules = new ArrayList<>();
        for (int i = 0; i < structures.size(); i++) {
            JsonView structure = asView(structures.get(i), "structures[" + i + "]");
            Grammar schema = JsonSchemaToGbnf.convert(schemaJson(structure, i), whitespace);
            int schemaRoot = merge(rules, symbolIds, symbolNames, schema, "st" + i + "-");
            // begin literal rule, then the schema root, then the end literal rule
            int begin = addLiteralRule(rules, symbolIds, symbolNames,
                    "st" + i + "-begin", requireText(structure, "begin", i));
            int end = addLiteralRule(rules, symbolIds, symbolNames,
                    "st" + i + "-end", requireText(structure, "end", i));
            int structureRule = rules.size();
            List<GrammarElement> body = new ArrayList<>();
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, begin));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, schemaRoot));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, end));
            body.add(new GrammarElement(GrammarElement.Type.END, 0));
            rules.add(body);
            define(symbolIds, symbolNames, "st" + i + "-structure", structureRule);
            structureRules.add(structureRule);
        }
        int rootIndex = rules.size();
        List<GrammarElement> rootRule = new ArrayList<>();
        for (int i = 0; i < structureRules.size(); i++) {
            if (i > 0) {
                rootRule.add(new GrammarElement(GrammarElement.Type.ALT, 0));
            }
            rootRule.add(new GrammarElement(GrammarElement.Type.RULE_REF, structureRules.get(i)));
        }
        rootRule.add(new GrammarElement(GrammarElement.Type.END, 0));
        rules.add(rootRule);
        define(symbolIds, symbolNames, Grammar.ROOT_RULE, rootIndex);
        return new Grammar(rules, symbolIds, symbolNames, rootIndex, new int[rules.size()],
                new int[rules.size()]);
    }

    /**
     * The tag-dispatch triggers of the payload, or null when it names none: {@link GrammarTriggers.Mode#DISPATCH}
     * words, after which the structure grammar runs and free text resumes once it completes.
     *
     * @throws IllegalArgumentException when the payload is malformed, a trigger is not a non-empty string, or a
     *                                  structure's {@code begin} starts with no trigger
     */
    public static GrammarTriggers triggers(String payload) {
        List<String> words = triggerWords(root(payload));
        if (words.isEmpty()) {
            return null;
        }
        GrammarTriggers.Builder b = GrammarTriggers.builder(GrammarTriggers.Mode.DISPATCH);
        words.forEach(b::word);
        return b.build();
    }

    private static JsonView root(String payload) {
        JsonView root;
        try {
            root = JsonView.of(Json.parse(payload, Json.Limits.UNTRUSTED), "structural_tag");
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("structural tag payload is not JSON: " + e.getMessage(), e);
        }
        if (!root.has("structures")) {
            throw new IllegalArgumentException("structural tag needs a structures array (the modern xgrammar"
                    + " format object is not compiled yet)");
        }
        return root;
    }

    /** The payload's trigger words, validated against the structures' begin texts. */
    private static List<String> triggerWords(JsonView root) {
        List<String> words = new ArrayList<>();
        Object raw = root.raw("triggers");
        if (raw == null) {
            return words;
        }
        if (!(raw instanceof List)) {
            throw new IllegalArgumentException("structural tag triggers must be an array of strings");
        }
        List<?> list = (List<?>) raw;
        for (int i = 0; i < list.size(); i++) {
            Object w = list.get(i);
            if (!(w instanceof String) || ((String) w).isEmpty()) {
                throw new IllegalArgumentException("structural tag trigger " + i + " is not a non-empty string");
            }
            words.add((String) w);
        }
        if (words.isEmpty()) {
            return words;
        }
        List<?> structures = asList(root, "structures");
        for (int i = 0; i < structures.size(); i++) {
            String begin = requireText(asView(structures.get(i), "structures[" + i + "]"), "begin", i);
            if (words.stream().noneMatch(begin::startsWith)) {
                throw new IllegalArgumentException("structural tag structure " + i + " begins with '" + begin
                        + "', which no trigger " + words + " opens");
            }
        }
        return words;
    }

    /**
     * Appends {@code schema}'s rules to {@code rules} under the name {@code prefix}: every rule renames to
     * {@code prefix + name} and its rule references re-index by the appended offset. Returns the schema's root
     * rule's new index.
     */
    private static int merge(List<List<GrammarElement>> rules, Map<String, Integer> symbolIds,
                             List<String> symbolNames, Grammar schema, String prefix) {
        int offset = rules.size();
        for (int r = 0; r < schema.ruleCount(); r++) {
            List<GrammarElement> renamed = new ArrayList<>(schema.rule(r).size());
            for (GrammarElement element : schema.rule(r)) {
                renamed.add(element.type() == GrammarElement.Type.RULE_REF
                        ? new GrammarElement(GrammarElement.Type.RULE_REF, element.value() + offset)
                        : element);
            }
            rules.add(renamed);
            define(symbolIds, symbolNames, prefix + schema.symbolName(r), offset + r);
        }
        return offset + schema.rootRuleIndex();
    }

    /** A rule matching the literal's code points, for a structure's begin and end text. */
    private static int addLiteralRule(List<List<GrammarElement>> rules, Map<String, Integer> symbolIds,
                                      List<String> symbolNames, String name, String literal) {
        int index = rules.size();
        List<GrammarElement> body = new ArrayList<>();
        literal.codePoints().forEach(cp -> body.add(new GrammarElement(GrammarElement.Type.CHAR, cp)));
        body.add(new GrammarElement(GrammarElement.Type.END, 0));
        rules.add(body);
        define(symbolIds, symbolNames, name, index);
        return index;
    }

    private static void define(Map<String, Integer> symbolIds, List<String> symbolNames, String name,
                               int index) {
        if (symbolIds.containsKey(name)) {
            throw new IllegalArgumentException("structural tag builds rule '" + name + "' twice");
        }
        symbolIds.put(name, index);
        symbolNames.add(name);
    }

    private static String requireText(JsonView structure, String key, int index) {
        if (!structure.has(key)) {
            throw new IllegalArgumentException("structural tag structure " + index + " needs a " + key + " string");
        }
        return structure.str(key);
    }

    /** The structure's schema as canonical JSON text; an object is required. */
    private static String schemaJson(JsonView structure, int index) {
        if (!structure.has("schema")) {
            throw new IllegalArgumentException("structural tag structure " + index + " needs a schema object");
        }
        Object schema = structure.raw("schema");
        if (!(schema instanceof Map)) {
            throw new IllegalArgumentException("structural tag structure " + index + " needs a schema object");
        }
        StringBuilder out = new StringBuilder();
        try {
            Json.write(schema, Json.Style.CANONICAL, out);
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException("structural tag structure " + index
                    + ": its schema does not write back to JSON", e);
        }
        return out.toString();
    }

    private static List<?> asList(JsonView root, String key) {
        Object raw = root.raw(key);
        if (!(raw instanceof List)) {
            throw new IllegalArgumentException("structural tag " + key + " must be an array");
        }
        return (List<?>) raw;
    }

    private static JsonView asView(Object element, String where) {
        if (!(element instanceof Map)) {
            throw new IllegalArgumentException(where + " must be an object with begin, schema and end");
        }
        Map<?, ?> map = (Map<?, ?>) element;
        Map<String, Object> typed = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            typed.put(String.valueOf(e.getKey()), e.getValue());
        }
        return JsonView.of(typed, where);
    }
}
