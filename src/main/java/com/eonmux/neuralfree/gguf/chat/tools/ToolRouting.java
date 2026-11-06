package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.HarmonyDialect;
import com.eonmux.neuralfree.gguf.chat.OutputGates;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
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
import java.util.Objects;
import java.util.function.Supplier;

/**
 * How one chat request's tools reach the output: the parser that reads calls out of the content, whether harmony
 * tool recipients start calls, and the grammar the tool choice adds (vLLM's {@code serving_chat} tool handling,
 * llama.cpp's {@code tool_choice} grammars).
 *
 * <ul>
 *   <li>No tools, or {@code tool_choice: none}: no parsing; the template still renders the tools.</li>
 *   <li>Harmony-family models: calls are the messages addressed {@code to=functions.NAME} (gpt-oss), or the
 *       ATEM block of a message addressed to a tool (Muse); {@code required} and a named function add the
 *       dialect's forcing grammar ({@link HarmonyToolGrammar}).</li>
 *   <li>Every other model with a known format: the format's own call syntax carries the calls
 *       ({@link NativeToolGrammar}) — {@code required} and a named function force it from the first token,
 *       {@code auto} applies it as a lazy grammar behind the format's call openers when the strictness asks
 *       for one, and the output is read back with the format's parser. A format without a native grammar
 *       keeps the generic behavior: {@code auto} leaves the output to the parser alone, {@code required}
 *       forces a JSON array of {@code {"name", "parameters"}} objects, a named function its parameter
 *       object.</li>
 *   <li>{@code auto}: the configured parser ({@code neuralfree.tool.parser}); {@code auto} detects the format
 *       from the chat template ({@link ToolParserRegistry#detect}), {@code none} leaves the syntax in the
 *       content.</li>
 *   <li>The strict level ({@link ToolStrict}, {@code neuralfree.tool.strict}) decides whether the arguments
 *       are held to a tool's parameter schema: a forcing choice holds each tool's unless the tool declares
 *       {@code strict: false}; under {@code auto} a tool's schema holds only when the tool declares
 *       {@code strict: true}, unless the level is {@code parameter} (vLLM's {@code resolve_tool_strictness}).
 *       Under {@code tool_choice: auto} with level {@code auto} and no strict tool, no grammar applies.</li>
 * </ul>
 *
 * The response-template parsers {@code hf} and {@code beam} read the calls with the model's {@code response_template}
 * and add no routing grammar: {@code hf} refuses the tool guarantees by name ({@link TemplateParsing#checkTools}), and
 * {@code beam} holds the whole output to its template's grammar ({@code BeamGrammar}), which the request's planner
 * derives beside the routing.
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
     * The routing of {@code chat} at strict level {@code auto}.
     *
     * @see #plan(ChatRequest, String, String, ReasoningFilter, String, ParallelToolCalls, ToolStrict)
     */
    public static ToolRouting plan(ChatRequest chat, String configuredParser, String chatTemplate,
                                   ReasoningFilter reasoning, String renderedPrompt,
                                   ParallelToolCalls parallelToolCalls) {
        return plan(chat, configuredParser, chatTemplate, reasoning, renderedPrompt, parallelToolCalls,
                ToolStrict.AUTO);
    }

    /**
     * The routing of {@code chat}.
     *
     * @param chat              the request (tools, tool choice)
     * @param configuredParser  the {@code neuralfree.tool.parser} name: {@code auto}, {@code none} or a format
     * @param chatTemplate      the model's chat template, for {@code auto} detection; may be null
     * @param reasoning         the model's reasoning filter
     * @param renderedPrompt    the rendered prompt, to tell whether it leaves reasoning open; may be null
     * @param parallelToolCalls the request's {@code parallel_tool_calls}: an explicit false is refused as a
     *                          guarantee ({@link TemplateParsing#checkTools}), the served value limits the calls
     *                          to one ({@link ParallelToolCalls#allowed})
     * @param strictLevel       the {@code neuralfree.tool.strict} floor
     * @throws IllegalArgumentException for a named function the tools do not declare, or an unknown parser name
     */
    public static ToolRouting plan(ChatRequest chat, String configuredParser, String chatTemplate,
                                   ReasoningFilter reasoning, String renderedPrompt,
                                   ParallelToolCalls parallelToolCalls, ToolStrict strictLevel) {
        return plan(chat, configuredParser, chatTemplate, null, reasoning, renderedPrompt, parallelToolCalls,
                strictLevel);
    }

    /**
     * The routing of {@code chat} on a loaded model: {@link #plan(ChatRequest, String, String, ReasoningFilter,
     * String, ParallelToolCalls, ToolStrict)} over the tokenizer's chat template, with the selected format resolved
     * for the model ({@link ToolParser.Factory#forModel}), so a named {@code hy_v3} / {@code hy_v4} parser takes the
     * vocabulary's structural-token suffix as {@code auto} takes the template's.
     *
     * @throws IllegalArgumentException for a named function the tools do not declare, an unknown parser name, or a
     *                                  format the model cannot serve
     */
    public static ToolRouting plan(ChatRequest chat, String configuredParser, GGUFTokenizer tokenizer,
                                   ReasoningFilter reasoning, String renderedPrompt,
                                   ParallelToolCalls parallelToolCalls, ToolStrict strictLevel) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        return plan(chat, configuredParser, tokenizer.chatTemplate(), tokenizer, reasoning, renderedPrompt,
                parallelToolCalls, strictLevel);
    }

    private static ToolRouting plan(ChatRequest chat, String configuredParser, String chatTemplate,
                                    GGUFTokenizer tokenizer, ReasoningFilter reasoning, String renderedPrompt,
                                    ParallelToolCalls parallelToolCalls, ToolStrict strictLevel) {
        List<ChatTool> tools = chat.tools();
        ChatRequest.ToolChoice choice = chat.toolChoice() == null ? ChatRequest.ToolChoice.auto() : chat.toolChoice();
        boolean none = choice.kind() == ChatRequest.ToolChoice.Kind.NONE;
        if (tools.isEmpty() || (none && !TemplateParsing.names(configuredParser))) {
            return NONE;
        }
        int maxCalls = parallelToolCalls.allowed() ? Integer.MAX_VALUE : 1;
        if (!none && reasoning.mode() == ReasoningFilter.Mode.HARMONY_CHANNELS) {
            return harmony(chat, choice, chatTemplate, reasoning, renderedPrompt, parallelToolCalls);
        }
        ToolParser.Factory format = formatFactory(configuredParser, chatTemplate);
        if (format != null && tokenizer != null) {
            format = format.forModel(tokenizer);
        }
        if (format instanceof ToolParserRegistry.TemplateFormat) {
            checkTemplateRequest(((ToolParserRegistry.TemplateFormat) format).parsing(), chat, reasoning,
                    parallelToolCalls);
            return none ? NONE : new ToolRouting(format.create(tools), false, null, null, maxCalls);
        }
        if (none) {
            return NONE;
        }
        switch (choice.kind()) {
            case REQUIRED: {
                List<ChatTool> pinned = pinned(tools, choice, strictLevel);
                ConstraintSpec tagged = pinned == null || format == null ? null
                        : NativeToolGrammar.forcing(format, pinned, parallelToolCalls.allowed(),
                        !parallelToolCalls.allowed(), chat.templateVariables());
                if (tagged != null) {
                    return new ToolRouting(format.create(tools), false, null,
                            forcing(tagged, reasoning, renderedPrompt), maxCalls);
                }
                return new ToolRouting(new JsonToolParser(Collections.<String>emptyList(), null,
                        JsonToolParser.Body.ARRAY, true), false, null,
                        forcing(requiredSchema(tools, parallelToolCalls.allowed()), reasoning, renderedPrompt),
                        maxCalls);
            }
            case FUNCTION: {
                ChatTool tool = find(tools, choice.functionName());
                List<ChatTool> pinned = pinned(Collections.singletonList(tool), choice, strictLevel);
                ConstraintSpec tagged = pinned == null || format == null ? null
                        : NativeToolGrammar.forcing(format, pinned, false, true, chat.templateVariables());
                if (tagged != null) {
                    return new ToolRouting(format.create(tools), false, null,
                            forcing(tagged, reasoning, renderedPrompt), 1);
                }
                return new ToolRouting(new NamedFunctionToolParser(tool.name()), false, null,
                        forcing(write(parameters(tool)), reasoning, renderedPrompt), 1);
            }
            default: {
                ToolParser parser = format == null ? null : format.create(tools);
                List<ChatTool> pinned = pinned(tools, choice, strictLevel);
                if (pinned != null && parser != null && !parser.triggers().isEmpty()) {
                    ConstraintSpec lazy = NativeToolGrammar.lazy(format, pinned, parser.triggers(),
                            chat.templateVariables());
                    if (lazy != null) {
                        return new ToolRouting(parser, false, null, lazy, maxCalls);
                    }
                }
                return new ToolRouting(parser, false, null, null, maxCalls);
            }
        }
    }

    /**
     * The guarantees a response-template tool parser refuses for {@code chat} ({@link TemplateParsing#checkTools}), and
     * a reasoning parser of the other template family, which vLLM cannot combine with it either.
     *
     * @throws IllegalArgumentException naming the guarantee and its request parameter, or the two parsers
     */
    private static void checkTemplateRequest(TemplateParsing parsing, ChatRequest chat, ReasoningFilter reasoning,
                                              ParallelToolCalls parallelToolCalls) {
        TemplateParsing reasoningTemplate = reasoning.responseTemplate();
        if (reasoningTemplate != null && !reasoningTemplate.name().equals(parsing.name())) {
            throw new IllegalArgumentException("--reasoning-parser " + reasoningTemplate.name()
                    + " and --tool-call-parser " + parsing.name() + " cannot be combined");
        }
        parsing.checkTools(chat, parallelToolCalls);
    }

    /**
     * The routing of a harmony-family model: recipients start calls, a non-{@code functions.} recipient's body is
     * read as a Muse ATEM block, and a forcing choice adds the dialect's grammar.
     */
    private static ToolRouting harmony(ChatRequest chat, ChatRequest.ToolChoice choice, String chatTemplate,
                                       ReasoningFilter reasoning, String renderedPrompt,
                                       ParallelToolCalls parallelToolCalls) {
        List<ChatTool> tools = chat.tools();
        ToolParser bodies = ToolParserRegistry.atem(tools);
        int maxCalls = parallelToolCalls.allowed() ? Integer.MAX_VALUE : 1;
        HarmonyDialect dialect = HarmonyDialect.of(chatTemplate);
        boolean inReasoning = renderedPrompt != null
                && reasoning.newState((channel, tokenId, text) -> { }, renderedPrompt).inReasoning();
        switch (choice.kind()) {
            case REQUIRED:
                return new ToolRouting(null, true, bodies,
                        HarmonyToolGrammar.forcing(dialect, tools, parallelToolCalls.allowed(), inReasoning),
                        maxCalls);
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

    /**
     * Checks at model load that a named tool parser can serve the model (vLLM
     * {@code validate_tokenizer_response_template}): the format resolved for it ({@link ToolParser.Factory#forModel}),
     * so {@code hf} or {@code beam} over a model without a usable {@code response_template} with a {@code tool_calls}
     * field fails before the first request. {@code auto} and {@code none} pass: they follow the chat template.
     *
     * @throws IllegalArgumentException for an unregistered name or a format the model cannot serve, naming what it
     *                                  lacks
     */
    public static void checkFormat(String configuredParser, GGUFTokenizer tokenizer) {
        String name = configuredParser == null ? "" : configuredParser.trim().toLowerCase(Locale.ROOT);
        if (name.isEmpty() || name.equals("auto") || name.equals("none")) {
            return;
        }
        formatFactory(configuredParser, null).forModel(Objects.requireNonNull(tokenizer, "tokenizer"));
    }

    /**
     * The format the configured name or the template selects ({@link ToolParserRegistry}), or null for
     * {@code none} or a template that writes no known syntax.
     *
     * @throws IllegalArgumentException for a configured name that is not registered
     */
    private static ToolParser.Factory formatFactory(String configured, String template) {
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
        return f;
    }

    /**
     * The tools with each strict flag pinned per the level (vLLM's {@code resolve_tool_strictness}), or null
     * when no native grammar applies: {@code tool_choice: auto} at level {@code auto} with no tool declaring
     * {@code strict: true}. A forcing choice constrains every call and keeps each tool's schema enforced
     * unless the tool declares {@code strict: false}; under {@code auto} only a tool that declares
     * {@code strict: true} is schema-pinned, unless the level is {@link ToolStrict#PARAMETER}.
     */
    static List<ChatTool> pinned(List<ChatTool> tools, ChatRequest.ToolChoice choice, ToolStrict level) {
        boolean forcing = choice.kind() != ChatRequest.ToolChoice.Kind.AUTO;
        boolean anyStrict = false;
        for (ChatTool t : tools) {
            anyStrict |= Boolean.TRUE.equals(t.strict());
        }
        if (!forcing && level == ToolStrict.AUTO && !anyStrict) {
            return null;
        }
        List<ChatTool> out = new ArrayList<>(tools.size());
        for (ChatTool t : tools) {
            boolean strict;
            if (level.atLeast(ToolStrict.PARAMETER)) {
                strict = true;
            } else if (forcing) {
                strict = t.strict() == null || t.strict();
            } else {
                strict = Boolean.TRUE.equals(t.strict());
            }
            out.add(t.toBuilder().strict(strict).build());
        }
        return out;
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
        return forcing(ConstraintSpec.jsonSchema(schemaJson, null), reasoning, renderedPrompt);
    }

    /** {@code spec} behind the reasoning end tag when the prompt leaves reasoning open ({@link OutputGates}). */
    static ConstraintSpec forcing(ConstraintSpec spec, ReasoningFilter reasoning, String renderedPrompt) {
        if (reasoning.mode() != ReasoningFilter.Mode.THINK_TAGS) {
            return spec;
        }
        ConstraintGate gate = OutputGates.forChat(reasoning, renderedPrompt, null);
        return gate == null ? spec : spec.withGate(gate);
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
