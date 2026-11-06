package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.tools.ToolSchemas;
import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The one PEG definition of a template's tool calls, built from {@link AutoParserAnalysis}: {@link #call} reads a
 * call out of text and {@link #grammar} emits the GBNF that admits the declared tools in the same syntax. Every
 * marker the analysis derived becomes a literal; the declared tools' parameter schemas become the constrained
 * argument rules, so a grammar-constrained generation reads back through the same parser.
 *
 * <p>The three tool formats differ only in their call body: a JSON object whose fields name the function and hold
 * its arguments ({@link AutoParserAnalysis.ToolFormat#JSON_NATIVE}), a function tag followed by a JSON argument
 * object ({@link AutoParserAnalysis.ToolFormat#TAG_WITH_JSON}), or a function tag followed by tagged parameter
 * names and values ({@link AutoParserAnalysis.ToolFormat#TAG_WITH_TAGGED}).
 */
final class AutoParserPeg {

    /** Any JSON object, the free arguments of a tool whose schema is not enforced. */
    private static final String ANY_OBJECT = "{\"type\": \"object\", \"additionalProperties\": true}";

    private AutoParserPeg() {
    }

    /**
     * A parser for one call of the declared tools.
     *
     * @param tools            the analysis
     * @param declared         the request's tools
     * @param withCallOpener   whether the expression starts with the per-call opener (false when the block reader
     *                         already consumed it)
     */
    static Peg call(AutoParserAnalysis.Tools tools, List<ChatTool> declared, boolean withCallOpener) {
        switch (tools.mode()) {
            case JSON_NATIVE: {
                List<Peg> parts = new ArrayList<>();
                if (withCallOpener) {
                    addLiteral(parts, tools.perCallStart());
                }
                parts.add(Peg.tag("json", Peg.schema("call-json", ANY_OBJECT)));
                addLiteral(parts, tools.perCallEnd());
                return parts.size() == 1 ? parts.get(0) : Peg.sequence(parts.toArray(new Peg[0]));
            }
            case TAG_WITH_JSON:
                return taggedJsonCall(tools, declared, withCallOpener);
            case TAG_WITH_TAGGED:
                return taggedArgCall(tools, declared, withCallOpener);
            default:
                throw new IllegalArgumentException("the template has no tool-call syntax to parse");
        }
    }

    private static Peg taggedJsonCall(AutoParserAnalysis.Tools tools, List<ChatTool> declared,
                                      boolean withCallOpener) {
        List<Peg> alternatives = new ArrayList<>();
        for (int i = 0; i < declared.size(); i++) {
            ChatTool tool = declared.get(i);
            List<Peg> parts = new ArrayList<>();
            if (withCallOpener) {
                addLiteral(parts, tools.perCallStart());
            }
            addCallIdIfBeforeName(parts, tools);
            addLiteral(parts, tools.functionNamePrefix());
            parts.add(Peg.tag("name", Peg.literal(tool.name())));
            addSuffixAndCallId(parts, tools);
            addLiteral(parts, tools.functionArgsSeparator());
            addLiteral(parts, tools.argsStart());
            parts.add(Peg.tag("args", Peg.schema("tool-" + i + "-args", argumentsSchema(tool))));
            addLiteral(parts, tools.argsEnd());
            addCallIdIfAfterArgs(parts, tools);
            addLiteral(parts, tools.functionClose());
            addIntertagSpace(parts, tools);
            addLiteral(parts, tools.perCallEnd());
            alternatives.add(parts.size() == 1 ? parts.get(0) : Peg.sequence(parts.toArray(new Peg[0])));
        }
        if (alternatives.isEmpty()) {
            return Peg.epsilon();
        }
        return Peg.choice(alternatives.toArray(new Peg[0]));
    }

    private static Peg taggedArgCall(AutoParserAnalysis.Tools tools, List<ChatTool> declared,
                                     boolean withCallOpener) {
        List<Peg> alternatives = new ArrayList<>();
        for (int i = 0; i < declared.size(); i++) {
            ChatTool tool = declared.get(i);
            List<Peg> parts = new ArrayList<>();
            if (withCallOpener) {
                addLiteral(parts, tools.perCallStart());
            }
            addCallIdIfBeforeName(parts, tools);
            addLiteral(parts, tools.functionNamePrefix());
            parts.add(Peg.tag("name", Peg.literal(tool.name())));
            addSuffixAndCallId(parts, tools);
            addLiteral(parts, tools.functionArgsSeparator());
            parts.add(taggedArguments(tools, tool, i));
            addCallIdIfAfterArgs(parts, tools);
            addLiteral(parts, tools.functionClose());
            addIntertagSpace(parts, tools);
            addLiteral(parts, tools.perCallEnd());
            alternatives.add(parts.size() == 1 ? parts.get(0) : Peg.sequence(parts.toArray(new Peg[0])));
        }
        if (alternatives.isEmpty()) {
            return Peg.epsilon();
        }
        return Peg.choice(alternatives.toArray(new Peg[0]));
    }

    /** The optional call id the format writes before the function name ({@code PRE_FUNC_NAME}). */
    private static void addCallIdIfBeforeName(List<Peg> parts, AutoParserAnalysis.Tools tools) {
        if (tools.callIdPosition() == AutoParserAnalysis.CallIdPosition.PRE_FUNC_NAME && carriesCallId(tools)) {
            addCallId(parts, tools);
        }
    }

    /** The optional call id the format writes after the arguments ({@code POST_ARGS}). */
    private static void addCallIdIfAfterArgs(List<Peg> parts, AutoParserAnalysis.Tools tools) {
        if (tools.callIdPosition() == AutoParserAnalysis.CallIdPosition.POST_ARGS && carriesCallId(tools)) {
            addCallId(parts, tools);
        }
    }

    /**
     * Whitespace a tolerant format may write between the last argument tag and the call end (Laguna, Bailing V3);
     * the template renders the two tight but the model may not.
     */
    private static void addIntertagSpace(List<Peg> parts, AutoParserAnalysis.Tools tools) {
        if (tools.tolerateIntertagWhitespace() && tools.functionClose().isEmpty()
                && !tools.perCallEnd().isEmpty()) {
            parts.add(Peg.space());
        }
    }

    /**
     * The text after the function name and, when the format carries one there, the call id. The analysis's
     * function-name suffix can absorb the id's closer (it extends to the argument object), so a suffix that ends
     * with the id's closer is split: the rest follows the name, the closer stays with the id.
     */
    private static void addSuffixAndCallId(List<Peg> parts, AutoParserAnalysis.Tools tools) {
        String suffix = tools.functionNameSuffix();
        String closer = tools.callIdSuffix();
        if (carriesCallId(tools) && !closer.isEmpty() && suffix.endsWith(closer)) {
            addLiteral(parts, suffix.substring(0, suffix.length() - closer.length()));
        } else {
            addLiteral(parts, suffix);
        }
        if (tools.callIdPosition() == AutoParserAnalysis.CallIdPosition.BETWEEN_FUNC_AND_ARGS) {
            addCallId(parts, tools);
        }
    }

    /** An optional call id between the function name and the arguments, when the analysis found one there. */
    private static void addCallId(List<Peg> parts, AutoParserAnalysis.Tools tools) {
        if (!carriesCallId(tools)) {
            return;
        }
        List<Peg> id = new ArrayList<>();
        addLiteral(id, tools.callIdPrefix());
        if (!tools.callIdSuffix().isEmpty()) {
            id.add(Peg.tag("id", Peg.until(tools.callIdSuffix())));
            id.add(Peg.literal(tools.callIdSuffix()));
        } else if (!tools.argsStart().isEmpty()) {
            id.add(Peg.tag("id", Peg.until(tools.argsStart())));
        } else if (!tools.functionArgsSeparator().isEmpty()) {
            id.add(Peg.tag("id", Peg.until(tools.functionArgsSeparator())));
        } else {
            id.add(Peg.tag("id", Peg.jsonValue()));
        }
        parts.add(Peg.opt(Peg.sequence(id.toArray(new Peg[0]))));
    }

    private static boolean carriesCallId(AutoParserAnalysis.Tools tools) {
        return tools.callIdPosition() != AutoParserAnalysis.CallIdPosition.NONE
                && !tools.callIdPrefix().isEmpty();
    }

    /** The declared parameters as tagged names and values: required ones in declaration order, then any optional. */
    private static Peg taggedArguments(AutoParserAnalysis.Tools tools, ChatTool tool, int toolIndex) {
        List<Peg> required = new ArrayList<>();
        List<Peg> optional = new ArrayList<>();
        Map<String, Object> params = tool.parameters();
        for (Map.Entry<String, Object> property : properties(params).entrySet()) {
            Peg parameter = taggedParameter(tools, property.getKey(), property.getValue(), toolIndex);
            (required(params, property.getKey()) ? required : optional).add(parameter);
        }
        List<Peg> parts = new ArrayList<>();
        for (int i = 0; i < required.size(); i++) {
            if (i > 0) {
                addLiteral(parts, tools.argSeparator());
            }
            parts.add(required.get(i));
        }
        if (!optional.isEmpty()) {
            List<Peg> any = new ArrayList<>();
            for (Peg p : optional) {
                any.add(p);
            }
            List<Peg> repeated = new ArrayList<>();
            if (!tools.argSeparator().isEmpty()) {
                repeated.add(Peg.literal(tools.argSeparator()));
            }
            repeated.add(Peg.choice(any.toArray(new Peg[0])));
            if (required.isEmpty()) {
                parts.add(optional.get(0));
            }
            parts.add(Peg.star(Peg.sequence(repeated.toArray(new Peg[0]))));
        }
        return parts.isEmpty() ? Peg.epsilon() : Peg.sequence(parts.toArray(new Peg[0]));
    }

    private static Peg taggedParameter(AutoParserAnalysis.Tools tools, String key, Object schema, int toolIndex) {
        String name = "tool-" + toolIndex + "-arg-" + key;
        List<Peg> parts = new ArrayList<>();
        addLiteral(parts, tools.argNamePrefix());
        parts.add(Peg.tag("argname", Peg.literal(key)));
        addLiteral(parts, tools.argNameSuffix());
        addLiteral(parts, tools.argValuePrefix());
        Peg value;
        if (!tools.argValueSuffix().isEmpty() && (schema == null || ToolSchemas.textValuedSchema(schema))) {
            value = Peg.tag("argvalue", Peg.until(tools.argValueSuffix()));
        } else if (!tools.argValueSuffix().isEmpty()) {
            value = Peg.tag("argvalue", Peg.schema(name, dump(schema)));
        } else {
            value = Peg.tag("argvalue", Peg.jsonValue());
        }
        parts.add(value);
        addLiteral(parts, tools.argValueSuffix());
        return Peg.tag("arg", Peg.sequence(parts.toArray(new Peg[0])));
    }

    /**
     * The grammar of a structured-output request on this template: the JSON schema's value, bare or in a
     * {@code ```json} fence, as llama.cpp's autoparser constrains a {@code response_format} (the reasoning the
     * prompt leaves open is the constraint gate's, not a rule here).
     */
    static String responseFormatGrammar(String schemaJson) {
        Peg.GbnfEmitter emitter = new Peg.GbnfEmitter();
        Peg value = Peg.tag("response-format", Peg.schema("response-format-schema", schemaJson));
        Peg fenced = Peg.sequence(Peg.literal("```json"), Peg.space(), value, Peg.space(), Peg.literal("```"));
        Peg bare = Peg.sequence(Peg.space(), value, Peg.space());
        return emitter.complete(Peg.choice(fenced, bare).gbnf(emitter));
    }

    /**
     * The GBNF of the declared tools in the template's format: the envelope with the analysis's markers and, for
     * each tool, its parameter schema, so a generation under the grammar reads back through {@link #call}.
     *
     * @param tools    the analysis
     * @param declared the request's tools
     * @param parallel whether more than one call is admitted
     */
    static String grammar(AutoParserAnalysis.Tools tools, List<ChatTool> declared, boolean parallel) {
        if (declared.isEmpty()) {
            throw new IllegalArgumentException("a tool grammar needs at least one tool");
        }
        Peg.GbnfEmitter emitter = new Peg.GbnfEmitter();
        boolean arrayWrapped = tools.mode() == AutoParserAnalysis.ToolFormat.JSON_NATIVE
                && tools.toolsArrayWrapped();
        Peg call = tools.mode() == AutoParserAnalysis.ToolFormat.JSON_NATIVE
                ? jsonNativeCall(tools, declared, emitter, arrayWrapped)
                : call(tools, declared, true);
        Peg body = parallel
                ? Peg.sequence(call, Peg.star(Peg.sequence(Peg.literal(separator(tools, arrayWrapped)),
                        call)))
                : call;
        List<Peg> section = new ArrayList<>();
        if (!tools.sectionStart().isEmpty()) {
            section.add(Peg.literal(tools.sectionStart()));
            section.add(Peg.space());
        }
        if (arrayWrapped) {
            section.add(Peg.literal("["));
            section.add(Peg.space());
        }
        section.add(body);
        if (arrayWrapped) {
            section.add(Peg.space());
            section.add(Peg.literal("]"));
        }
        if (!tools.sectionEnd().isEmpty()) {
            section.add(Peg.space());
            section.add(Peg.literal(tools.sectionEnd()));
        }
        Peg root = section.size() == 1 ? section.get(0)
                : Peg.sequence(section.toArray(new Peg[0]));
        return emitter.complete(root.gbnf(emitter));
    }

    private static String separator(AutoParserAnalysis.Tools tools, boolean arrayWrapped) {
        if (arrayWrapped) {
            return ",";
        }
        return tools.perCallStart().isEmpty() && tools.sectionStart().isEmpty() ? ", " : "";
    }

    /**
     * The JSON-native call grammar: one object per declared tool, its name field the tool's name and its arguments
     * field the tool's parameter schema, ordered as the analysis found the fields.
     */
    private static Peg jsonNativeCall(AutoParserAnalysis.Tools tools, List<ChatTool> declared,
                                      Peg.GbnfEmitter emitter, boolean arrayWrapped) {
        List<Peg> alternatives = new ArrayList<>();
        for (int i = 0; i < declared.size(); i++) {
            ChatTool tool = declared.get(i);
            String argsRule = emitter.schema("tool-" + i + "-args", argumentsSchema(tool));
            Peg object;
            if (tools.funNameIsKey()) {
                object = braces(Peg.sequence(jsonString(tool.name()), Peg.space(), Peg.literal(":"), Peg.space(),
                        Peg.raw(argsRule)));
            } else {
                String namePath = tools.effectiveNameField().isEmpty() ? "name" : tools.effectiveNameField();
                String argsPath = tools.effectiveArgsField().isEmpty() ? "arguments" : tools.effectiveArgsField();
                List<JsonField> fields = new ArrayList<>();
                fields.add(new JsonField(namePath, Peg.literal("\"" + MarkerDiff.literal(tool.name()) + "\"")));
                fields.add(new JsonField(argsPath, Peg.raw(argsRule)));
                Map<String, JsonField> byPath = new LinkedHashMap<>();
                for (JsonField field : fields) {
                    byPath.put(field.path, field);
                }
                List<JsonField> ordered = new ArrayList<>();
                for (String path : tools.parameterOrder()) {
                    JsonField field = byPath.remove(path);
                    if (field != null) {
                        ordered.add(field);
                    }
                }
                ordered.addAll(byPath.values());
                object = objectOf(ordered, openaiTypeField(tools));
            }
            alternatives.add(arrayWrapped ? object : Peg.seq(
                    tools.perCallStart().isEmpty() ? Peg.eps() : Peg.literal(tools.perCallStart()),
                    object,
                    tools.perCallEnd().isEmpty() ? Peg.eps() : Peg.literal(tools.perCallEnd())));
        }
        return Peg.choice(alternatives.toArray(new Peg[0]));
    }

    /**
     * The optional leading {@code "type": "function",} field of the OpenAI function wrapper: the templates matched
     * by the {@code openai_wrapper_trigger} patch have their model emit it before the analysed fields.
     */
    private static Peg openaiTypeField(AutoParserAnalysis.Tools tools) {
        if (!tools.openaiWrapperTrigger()) {
            return Peg.eps();
        }
        return Peg.opt(Peg.seq(jsonString("type"), Peg.space(), Peg.literal(":"), Peg.space(),
                jsonString("function"), Peg.space(), Peg.literal(","), Peg.space()));
    }

    /** One field of a JSON-native call: its dotted path and the expression that matches its value. */
    private static final class JsonField {
        final String path;
        final Peg value;

        JsonField(String path, Peg value) {
            this.path = path;
            this.value = value;
        }
    }

    /** A JSON object over {@code fields}, nesting dotted paths into inner objects, after the {@code prefix}. */
    private static Peg objectOf(List<JsonField> fields, Peg prefix) {
        Map<String, List<JsonField>> groups = new LinkedHashMap<>();
        for (JsonField field : fields) {
            int dot = field.path.indexOf('.');
            String head = dot < 0 ? field.path : field.path.substring(0, dot);
            String rest = dot < 0 ? "" : field.path.substring(dot + 1);
            List<JsonField> group = groups.get(head);
            if (group == null) {
                group = new ArrayList<>();
                groups.put(head, group);
            }
            group.add(new JsonField(rest, field.value));
        }
        List<Peg> members = new ArrayList<>();
        for (Map.Entry<String, List<JsonField>> group : groups.entrySet()) {
            List<JsonField> sub = group.getValue();
            Peg value = sub.size() == 1 && sub.get(0).path.isEmpty() ? sub.get(0).value
                    : objectOf(sub, Peg.eps());
            members.add(Peg.sequence(jsonString(group.getKey()), Peg.space(), Peg.literal(":"), Peg.space(), value));
        }
        return braces(Peg.seq(prefix, joinComma(members)));
    }

    private static Peg braces(Peg body) {
        return Peg.sequence(Peg.literal("{"), Peg.space(), body, Peg.space(), Peg.literal("}"));
    }

    private static Peg joinComma(List<Peg> parts) {
        List<Peg> all = new ArrayList<>();
        for (int i = 0; i < parts.size(); i++) {
            if (i > 0) {
                all.add(Peg.literal(","));
                all.add(Peg.space());
            }
            all.add(parts.get(i));
        }
        return Peg.sequence(all.toArray(new Peg[0]));
    }

    private static Peg jsonString(String text) {
        return Peg.literal("\"" + MarkerDiff.literal(text) + "\"");
    }

    private static String argumentsSchema(ChatTool tool) {
        Map<String, Object> parameters = tool.parameters();
        return parameters == null || parameters.isEmpty() ? ANY_OBJECT : dump(parameters);
    }

    private static String dump(Object value) {
        return Json.toString(value, Json.Style.API);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> properties(Map<String, Object> parameters) {
        Object props = parameters == null ? null : parameters.get("properties");
        return props instanceof Map ? (Map<String, Object>) props : new LinkedHashMap<String, Object>();
    }

    @SuppressWarnings("unchecked")
    private static boolean required(Map<String, Object> parameters, String key) {
        Object required = parameters == null ? null : parameters.get("required");
        if (!(required instanceof List)) {
            return false;
        }
        for (Object name : (List<Object>) required) {
            if (key.equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static void addLiteral(List<Peg> parts, String text) {
        if (text != null && !text.isEmpty()) {
            parts.add(Peg.literal(text));
        }
    }
}
