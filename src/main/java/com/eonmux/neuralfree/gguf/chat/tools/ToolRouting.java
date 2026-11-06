package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.HarmonyDialect;
import com.eonmux.neuralfree.gguf.chat.OutputGates;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintGate;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

/**
 * How one chat request's tools reach the output: the parser that reads calls out of the content, whether harmony
 * tool recipients start calls, and the grammar a forcing {@code tool_choice} adds (vLLM's
 * {@code serving_chat} tool handling, llama.cpp's {@code tool_choice} grammars).
 *
 * <ul>
 *   <li>No tools, or {@code tool_choice: none}: no parsing; the template still renders the tools.</li>
 *   <li>Harmony-family models: calls are the messages addressed {@code to=functions.NAME} (gpt-oss), or the
 *       ATEM block of a message addressed to a tool (Muse); {@code required} and a named function add the
 *       dialect's forcing grammar ({@link HarmonyToolGrammar}).</li>
 *   <li>{@code auto}: the configured parser ({@code neuralfree.tool.parser}); {@code auto} detects the format
 *       from the chat template ({@link ToolParserRegistry#detect}), {@code none} leaves the syntax in the
 *       content.</li>
 *   <li>{@code required}: a grammar holds the output to a JSON array of {@code {"name", "parameters"}} objects
 *       over the request's tools (one item without {@code parallel_tool_calls}), read back as calls.</li>
 *   <li>A named function: a grammar holds the output to that function's parameter object.</li>
 * </ul>
 *
 * When the rendered prompt leaves reasoning open (a thinking template's forced {@code <think>}), the forcing
 * grammar waits for the reasoning end tag (a lazy trigger) and constrains from the tag on, so the model still
 * reasons first; a harmony grammar lets the model write its reasoning messages before the call. Immutable.
 */
public final class ToolRouting {

    private static final ToolRouting NONE = new ToolRouting(null, false, null, null, Integer.MAX_VALUE);

    private final ToolParser parser;
    private final boolean channelCalls;
    private final ToolParser recipientParser;
    private final ConstraintSpec constraint;
    private final int maxCalls;

    private ToolRouting(ToolParser parser, boolean channelCalls, ToolParser recipientParser,
                        ConstraintSpec constraint, int maxCalls) {
        this.parser = parser;
        this.channelCalls = channelCalls;
        this.recipientParser = recipientParser;
        this.constraint = constraint;
        this.maxCalls = maxCalls;
    }

    /** No tool handling: content passes through. */
    public static ToolRouting none() {
        return NONE;
    }

    /**
     * The routing of {@code chat}.
     *
     * @param chat              the request (tools, tool choice)
     * @param configuredParser  the {@code neuralfree.tool.parser} name: {@code auto}, {@code none} or a format
     * @param chatTemplate      the model's chat template, for {@code auto} detection; may be null
     * @param reasoning         the model's reasoning filter
     * @param renderedPrompt    the rendered prompt, to tell whether it leaves reasoning open; may be null
     * @param parallelToolCalls whether several calls are allowed ({@code parallel_tool_calls}, default true)
     * @throws IllegalArgumentException for a named function the tools do not declare, or an unknown parser name
     */
    public static ToolRouting plan(ChatRequest chat, String configuredParser, String chatTemplate,
                                   ReasoningFilter reasoning, String renderedPrompt, boolean parallelToolCalls) {
        List<ChatTool> tools = chat.tools();
        ChatRequest.ToolChoice choice = chat.toolChoice() == null ? ChatRequest.ToolChoice.auto() : chat.toolChoice();
        if (tools.isEmpty() || choice.kind() == ChatRequest.ToolChoice.Kind.NONE) {
            return NONE;
        }
        int maxCalls = parallelToolCalls ? Integer.MAX_VALUE : 1;
        if (reasoning.mode() == ReasoningFilter.Mode.HARMONY_CHANNELS) {
            return harmony(chat, choice, chatTemplate, reasoning, renderedPrompt, parallelToolCalls);
        }
        switch (choice.kind()) {
            case REQUIRED:
                return new ToolRouting(new JsonToolParser(Collections.<String>emptyList(), null,
                        JsonToolParser.Body.ARRAY, true), false, null,
                        forcing(requiredSchema(tools, parallelToolCalls), reasoning, renderedPrompt), maxCalls);
            case FUNCTION:
                ChatTool tool = find(tools, choice.functionName());
                return new ToolRouting(new NamedFunctionToolParser(tool.name()), false, null,
                        forcing(write(parameters(tool)), reasoning, renderedPrompt), 1);
            default:
                return new ToolRouting(parser(configuredParser, chatTemplate, tools), false, null, null, maxCalls);
        }
    }

    /**
     * The routing of a harmony-family model: recipients start calls, a non-{@code functions.} recipient's body is
     * read as a Muse ATEM block, and a forcing choice adds the dialect's grammar.
     */
    private static ToolRouting harmony(ChatRequest chat, ChatRequest.ToolChoice choice, String chatTemplate,
                                       ReasoningFilter reasoning, String renderedPrompt, boolean parallel) {
        List<ChatTool> tools = chat.tools();
        ToolParser bodies = ToolParserRegistry.atem(tools);
        int maxCalls = parallel ? Integer.MAX_VALUE : 1;
        HarmonyDialect dialect = HarmonyDialect.of(chatTemplate);
        boolean inReasoning = renderedPrompt != null
                && reasoning.newState((channel, tokenId, text) -> { }, renderedPrompt).inReasoning();
        switch (choice.kind()) {
            case REQUIRED:
                return new ToolRouting(null, true, bodies,
                        HarmonyToolGrammar.forcing(dialect, tools, parallel, inReasoning), maxCalls);
            case FUNCTION:
                ChatTool tool = find(tools, choice.functionName());
                return new ToolRouting(null, true, bodies, HarmonyToolGrammar.forcing(dialect,
                        Collections.singletonList(tool), false, inReasoning), 1);
            default:
                return new ToolRouting(null, true, bodies, null, maxCalls);
        }
    }

    /** Whether the output carries tool calls to read. */
    public boolean active() {
        return parser != null || channelCalls;
    }

    /** The format parser of the content, or null. */
    public ToolParser parser() {
        return parser;
    }

    /** The grammar a forcing tool choice adds, or null. */
    public ConstraintSpec constraint() {
        return constraint;
    }

    /** A fresh output for one choice, with random call ids. */
    public ToolOutput newOutput() {
        return newOutput(ToolCallStream::randomId);
    }

    /** A fresh output for one choice whose generated call ids come from {@code ids}. */
    public ToolOutput newOutput(Supplier<String> ids) {
        return new ToolOutput(parser, channelCalls, recipientParser, maxCalls, ids);
    }

    private static ToolParser parser(String configured, String template, List<ChatTool> tools) {
        String name = configured == null ? "auto" : configured.trim().toLowerCase(Locale.ROOT);
        if (name.equals("none")) {
            return null;
        }
        ToolParser.Factory f = name.isEmpty() || name.equals("auto") ? ToolParserRegistry.detect(template)
                : ToolParserRegistry.forName(name);
        if (f == null && !(name.isEmpty() || name.equals("auto"))) {
            throw new IllegalArgumentException("tool parser '" + configured + "' is not registered; known: "
                    + ToolParserRegistry.names());
        }
        return f == null ? null : f.create(tools);
    }

    private static ChatTool find(List<ChatTool> tools, String name) {
        for (ChatTool t : tools) {
            if (t.name().equals(name)) {
                return t;
            }
        }
        List<String> names = new ArrayList<>();
        for (ChatTool t : tools) {
            names.add(t.name());
        }
        throw new IllegalArgumentException("tool_choice names function '" + name + "', which the tools do not"
                + " declare (declared: " + names + ")");
    }

    /**
     * The forcing grammar of {@code schemaJson}: the schema itself, behind the reasoning end tag when the prompt leaves
     * reasoning open ({@link OutputGates}), so the model still reasons first.
     */
    static ConstraintSpec forcing(String schemaJson, ReasoningFilter reasoning, String renderedPrompt) {
        ConstraintSpec spec = ConstraintSpec.jsonSchema(schemaJson, null);
        return OutputGates.opensReasoning(reasoning, renderedPrompt)
                ? spec.withGate(ConstraintGate.afterWords(reasoning.endTags())) : spec;
    }

    /** vLLM's required-choice schema: an array of {@code {"name": one of the tools, "parameters": its schema}}. */
    static String requiredSchema(List<ChatTool> tools, boolean parallel) {
        List<Object> branches = new ArrayList<>();
        for (ChatTool t : tools) {
            Map<String, Object> name = new LinkedHashMap<>();
            name.put("type", "string");
            name.put("enum", Collections.singletonList(t.name()));
            Map<String, Object> props = new LinkedHashMap<>();
            props.put("name", name);
            props.put("parameters", parameters(t));
            Map<String, Object> branch = new LinkedHashMap<>();
            branch.put("type", "object");
            branch.put("properties", props);
            branch.put("required", Arrays.asList("name", "parameters"));
            branch.put("additionalProperties", Boolean.FALSE);
            branches.add(branch);
        }
        Map<String, Object> items = new LinkedHashMap<>();
        items.put("anyOf", branches);
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "array");
        schema.put("items", items);
        schema.put("minItems", 1L);
        if (!parallel) {
            schema.put("maxItems", 1L);
        }
        return write(schema);
    }

    /** The tool's parameter schema, an empty object schema when it declares none. */
    static Map<String, Object> parameters(ChatTool tool) {
        Map<String, Object> p = tool.parameters();
        if (p == null || p.isEmpty()) {
            Map<String, Object> empty = new LinkedHashMap<>();
            empty.put("type", "object");
            empty.put("properties", new LinkedHashMap<String, Object>());
            return empty;
        }
        return p;
    }

    /** {@code value} as canonical JSON text. */
    static String write(Object value) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(value, Json.Style.CANONICAL, out);
        } catch (IOException e) {
            throw new IllegalStateException("a tool schema does not write back to JSON", e);
        }
        return out.toString();
    }

    /** {@code text} as a GBNF string literal. */
    static String literal(String text) {
        StringBuilder out = new StringBuilder("\"");
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                default: out.append(c); break;
            }
        }
        return out.append('"').toString();
    }
}
