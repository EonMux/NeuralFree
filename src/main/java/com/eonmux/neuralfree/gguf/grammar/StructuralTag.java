package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;

/**
 * The structural-tag parser: the vLLM {@code structural_tag} payload compiles to one grammar, in
 * either of the two shapes xgrammar accepts. The legacy payload {@code {"structures": [{"begin", "schema",
 * "end"}...], "triggers": [...]}} compiles to the alternation of the structures, each {@code begin} literal, then
 * the {@code schema}'s grammar ({@link JsonSchemaToGbnf} in the spec's whitespace mode), then the {@code end}
 * literal, rules renaming under a per-structure prefix so several structures compose. The modern payload
 * {@code {"type": "structural_tag", "format": {...}}} compiles through {@link StructuralTagFormat} — the xgrammar
 * format object with {@code const_string}, {@code json_schema}, {@code grammar}, {@code regex}, {@code sequence},
 * {@code or}, {@code tag}, {@code triggered_tags}, {@code tags_with_separator}, {@code any_text} and the rest.
 *
 * <p>A non-empty {@code triggers} array of the legacy shape gates the grammar as xgrammar's tag dispatch does
 * ({@link GrammarTriggers.Mode#DISPATCH}): the output is free text until a trigger appears, the structure whose
 * {@code begin} the trigger opens is then enforced to its {@code end}, and free text resumes. Every structure's
 * {@code begin} must start with one of the triggers, which xgrammar requires too. Without triggers the grammar
 * requires a structure outright; a dispatching modern format ({@code triggered_tags}, {@code dispatch} and the
 * token variants) carries its triggers inside the {@code format} tree.
 *
 * <p>Refusals name their field: a payload with neither a {@code structures} array nor a {@code format} object, a
 * trigger that is not a non-empty string, a structure whose {@code begin} no trigger opens, a structure missing
 * one of its three members, or a malformed or unsupported {@code format} node — all as
 * {@link IllegalArgumentException}.
 */
public final class StructuralTag {

    private StructuralTag() {
    }

    /** The validated grammar of the payload text ({@link ConstraintSpec#structuralTag}); no vocabulary. */
    public static Grammar parse(String payload, JsonSchemaToGbnf.WhitespaceMode whitespace) {
        return parse(payload, whitespace, null);
    }

    /**
     * The validated grammar of the payload text, the token references of a {@code format} tree resolving through
     * {@code vocab} ({@code null} refuses token-string forms).
     *
     * @throws IllegalArgumentException when the payload is malformed or names an unsupported construct
     */
    public static Grammar parse(String payload, JsonSchemaToGbnf.WhitespaceMode whitespace,
            GrammarVocab vocab) {
        JsonView root = root(payload);
        if (root.has("structures")) {
            return legacy(root, whitespace);
        }
        return StructuralTagFormat.compile(root, whitespace, vocab).grammar();
    }

    /**
     * The payload compiled in one pass for a sampler: its grammar, its triggers (the legacy {@code triggers}
     * words, or the dispatching {@code format}'s own), and the strings and token ids its free text must never
     * write ({@link CompiledConstraint#freeExcludes}).
     *
     * @throws IllegalArgumentException when the payload is malformed or names an unsupported construct
     */
    public static CompiledConstraint compileConstraint(String payload,
            JsonSchemaToGbnf.WhitespaceMode whitespace, GrammarVocab vocab) {
        JsonView root = root(payload);
        if (root.has("structures")) {
            return new CompiledConstraint(legacy(root, whitespace), legacyTriggers(root));
        }
        StructuralTagFormat.Compiled compiled = StructuralTagFormat.compile(root, whitespace, vocab);
        return new CompiledConstraint(compiled.grammar(), compiled.triggers(), compiled.excludes(),
                compiled.excludeTokens());
    }

    /**
     * The triggers gating the payload: the legacy {@code triggers} words ({@link GrammarTriggers.Mode#DISPATCH}),
     * or the top-level dispatching {@code format}'s own; null when the payload names none and its grammar
     * constrains the output from the first token. Token-string forms are refused (no vocabulary).
     *
     * @throws IllegalArgumentException when the payload is malformed, a trigger is not a non-empty string, or a
     *                                  structure's {@code begin} starts with no trigger
     */
    public static GrammarTriggers triggers(String payload) {
        JsonView root = root(payload);
        if (root.has("structures")) {
            return legacyTriggers(root);
        }
        return StructuralTagFormat.compile(root, JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE, null).triggers();
    }

    /** The legacy grammar: the alternation of the structures, each begin + schema + end. */
    private static Grammar legacy(JsonView root, JsonSchemaToGbnf.WhitespaceMode whitespace) {
        triggerWords(root);
        List<?> structures = asList(root, "structures");
        if (structures.isEmpty()) {
            throw new IllegalArgumentException("structural tag carries no structures");
        }

        StructuralTagRules rules = new StructuralTagRules();
        List<Integer> structureRules = new ArrayList<>();
        for (int i = 0; i < structures.size(); i++) {
            JsonView structure = asView(structures.get(i), "structures[" + i + "]");
            Grammar schema = JsonSchemaToGbnf.convert(schemaJson(structure, i), whitespace);
            int schemaRoot = rules.merge(schema, "st" + i + "-");
            // begin literal rule, then the schema root, then the end literal rule
            int begin = rules.literal("st" + i + "-begin", text(structure, "begin", i));
            int end = rules.literal("st" + i + "-end", text(structure, "end", i));
            List<GrammarElement> body = new ArrayList<>();
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, begin));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, schemaRoot));
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, end));
            structureRules.add(rules.rule("st" + i + "-structure", body));
        }
        return rules.finish(refs(structureRules));
    }

    /** The legacy payload's trigger words, DISPATCH-gated; null when it names none. */
    private static GrammarTriggers legacyTriggers(JsonView root) {
        List<String> words = triggerWords(root);
        if (words.isEmpty()) {
            return null;
        }
        GrammarTriggers.Builder b = GrammarTriggers.builder(GrammarTriggers.Mode.DISPATCH);
        words.forEach(b::word);
        return b.build();
    }

    private static List<GrammarElement> refs(List<Integer> ruleIds) {
        List<GrammarElement> body = new ArrayList<>(ruleIds.size());
        for (int i = 0; i < ruleIds.size(); i++) {
            if (i > 0) {
                body.add(new GrammarElement(GrammarElement.Type.ALT, 0));
            }
            body.add(new GrammarElement(GrammarElement.Type.RULE_REF, ruleIds.get(i)));
        }
        return body;
    }

    private static JsonView root(String payload) {
        JsonView root;
        try {
            root = JsonView.of(Json.parse(payload, Json.Limits.UNTRUSTED), "structural_tag");
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("structural tag payload is not JSON: " + e.getMessage(), e);
        }
        Object type = root.raw("type");
        if (root.has("type") && !"structural_tag".equals(type)) {
            throw new IllegalArgumentException("structural tag type must be \"structural_tag\"");
        }
        if (!root.has("structures") && !root.has("format")) {
            throw new IllegalArgumentException("structural tag needs a structures array or a format object");
        }
        return root;
    }

    /** The legacy payload's trigger words, validated against the structures' begin texts. */
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
            String begin = text(asView(structures.get(i), "structures[" + i + "]"), "begin", i);
            if (words.stream().noneMatch(begin::startsWith)) {
                throw new IllegalArgumentException("structural tag structure " + i + " begins with '" + begin
                        + "', which no trigger " + words + " opens");
            }
        }
        return words;
    }

    private static String text(JsonView structure, String key, int index) {
        if (!structure.has(key) || !(structure.raw(key) instanceof String)) {
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
        if (!(schema instanceof java.util.Map)) {
            throw new IllegalArgumentException("structural tag structure " + index + " needs a schema object");
        }
        StringBuilder out = new StringBuilder();
        try {
            Json.write(schema, StructuralTagFormat.SCHEMA_TEXT, out);
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
        if (!(element instanceof java.util.Map)) {
            throw new IllegalArgumentException(where + " must be an object with begin, schema and end");
        }
        java.util.Map<?, ?> map = (java.util.Map<?, ?>) element;
        java.util.Map<String, Object> typed = new java.util.LinkedHashMap<>();
        for (java.util.Map.Entry<?, ?> e : map.entrySet()) {
            typed.put(String.valueOf(e.getKey()), e.getValue());
        }
        return JsonView.of(typed, where);
    }
}
