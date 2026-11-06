package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.GbnfPrinter;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;
import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The grammar the {@code beam} parser holds one request's output to (vLLM {@code BeamParser._constrain_request} and
 * {@code _template_tag}): a structural tag ({@link ConstraintSpec#structuralTag}) in xgrammar's format derived from the
 * response template ({@link TemplateGrammar}), which replaces the request's own output constraint.
 *
 * <ul>
 *   <li>Tools a template tool parser reads: calls in the template's format to the declared functions, the named one
 *       for a named {@code tool_choice}, with no answer beside them for a required or named choice, one call when
 *       the served {@code parallel_tool_calls} is false, and each function's arguments held to its schema where the
 *       template can spell it (strict tools: a tool's {@code strict: true}, or every tool at the {@code parameter}
 *       strict level); otherwise any arguments. When the model may answer, calls may also name another function with
 *       any arguments.</li>
 *   <li>{@code tool_choice: none} beside tools of a template with a tool-call field: the answer alone.</li>
 *   <li>An output constraint ({@code response_format}, a guided grammar) answered inside a content field the template
 *       opens explicitly: the answer field's opener, then the constrained answer. An output constraint the template
 *       answers without an opener, a structural tag, or one with no xgrammar format, stays the request's own and
 *       holds the output alone.</li>
 *   <li>Otherwise, when {@link Request#alwaysConstrain} (vLLM {@code VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT}): the
 *       template's framing alone.</li>
 * </ul>
 *
 * When the prompt ends at the assistant message's start anchor, the tag covers the output from its first token,
 * opening with non-empty reasoning unless the request turns thinking off ({@code enable_thinking}); elsewhere it
 * follows the reasoning and waits for its end like any output constraint. Inside a content region the prompt left
 * open only the rest of a free answer is held; inside an open thinking region a tag needs the reasoning gate. When
 * no tag is derived the request's own output constraint, if any, still holds; a tool guarantee the template cannot
 * express is refused by name: a required or named {@code tool_choice}, strict tools, an explicitly false
 * {@code parallel_tool_calls} of a repeating tool field, or {@code tool_choice: none}. Immutable.
 */
public final class BeamGrammar {

    /** vLLM's schema of a {@code json_object} answer. */
    private static final String OBJECT_SCHEMA = "{\"type\": \"object\"}";
    /** The whitespace mode a structural tag compiles its {@code json_schema} formats in. */
    private static final JsonSchemaToGbnf.WhitespaceMode TAG_WHITESPACE =
            ConstraintSpec.structuralTag("{}").whitespace();

    private final ConstraintSpec constraint;
    private final boolean fromMessageStart;

    private BeamGrammar(ConstraintSpec constraint, boolean fromMessageStart) {
        this.constraint = constraint;
        this.fromMessageStart = fromMessageStart;
    }

    /** The structural tag the output follows. */
    public ConstraintSpec constraint() {
        return constraint;
    }

    /**
     * Whether the tag covers the output from its first token, reasoning included (vLLM {@code _grammar_from_parser}):
     * no reasoning gate applies.
     */
    public boolean fromMessageStart() {
        return fromMessageStart;
    }

    /** The request a {@link BeamGrammar} is derived for. Build with {@link #of}; not thread-safe. */
    public static final class Request {
        private final ChatRequest chat;
        private String prompt = "";
        private ParallelToolCalls parallelToolCalls = ParallelToolCalls.PARALLEL;
        private boolean functionStrict;
        private boolean parameterStrict;
        private ConstraintSpec output;
        private boolean reasoningGated;
        private boolean toolParsing;
        private boolean alwaysConstrain = true;

        private Request(ChatRequest chat) {
            this.chat = Objects.requireNonNull(chat, "chat");
        }

        /** A request for {@code chat}: tools parsed by no template parser, no output constraint, no reasoning gate. */
        public static Request of(ChatRequest chat) {
            return new Request(chat);
        }

        /** The rendered prompt the output continues. */
        public Request prompt(String v) { this.prompt = v == null ? "" : v; return this; }

        /** {@code parallel_tool_calls}; default parallel. */
        public Request parallelToolCalls(ParallelToolCalls v) {
            this.parallelToolCalls = v == null ? ParallelToolCalls.PARALLEL : v;
            return this;
        }

        /**
         * Whether the server's strict level constrains the calls of every request with tools ({@code function} or
         * {@code parameter}), not only those with a forcing choice or a strict tool.
         */
        public Request functionStrict(boolean v) { this.functionStrict = v; return this; }

        /** Whether the server's strict level holds every tool's arguments to its schema ({@code parameter}). */
        public Request parameterStrict(boolean v) { this.parameterStrict = v; return this; }

        /** The request's own output constraint ({@code response_format}, a guided grammar), or null. */
        public Request output(ConstraintSpec v) { this.output = v; return this; }

        /**
         * Whether a reasoning parser reads the reasoning and it reaches the response ({@code include_reasoning}), so a
         * grammar can wait for the reasoning's end.
         */
        public Request reasoningGated(boolean v) { this.reasoningGated = v; return this; }

        /** Whether the configured tool parser is a response-template parser, which reads the template's calls. */
        public Request toolParsing(boolean v) { this.toolParsing = v; return this; }

        /** Whether the output follows the template also without a tool or output constraint; default true. */
        public Request alwaysConstrain(boolean v) { this.alwaysConstrain = v; return this; }
    }

    /**
     * The grammar of {@code request} under the {@code beam} parser {@code parsing}, or null to leave the output to
     * the request's own constraint, if any.
     *
     * @throws IllegalArgumentException for a guarantee the template cannot express, a tool or parameter name its
     *                                  format cannot encode, or a forcing {@code tool_choice} beside an output
     *                                  constraint, naming the request parameter
     */
    public static BeamGrammar derive(TemplateParsing parsing, Request request) {
        if (!parsing.beam()) throw new IllegalArgumentException("'" + parsing.name() + "' derives no grammar");
        return new Derivation(parsing.template(), request).derive();
    }

    /** One derivation: the request's tool and output state over the template. */
    private static final class Derivation {
        private final ResponseTemplate template;
        private final Request r;
        private final ChatRequest.ToolChoice.Kind choice;
        private final boolean named;
        private final boolean toolsEnabled;
        private final boolean forced;
        private final boolean toolsForbidden;
        private final boolean constrainedTools;
        private final boolean strictAsked;

        Derivation(ResponseTemplate template, Request r) {
            this.template = template;
            this.r = r;
            List<ChatTool> tools = r.chat.tools();
            this.choice = r.chat.toolChoice() == null ? ChatRequest.ToolChoice.Kind.AUTO : r.chat.toolChoice().kind();
            this.named = choice == ChatRequest.ToolChoice.Kind.FUNCTION;
            this.toolsEnabled = r.toolParsing && !tools.isEmpty() && choice != ChatRequest.ToolChoice.Kind.NONE;
            this.forced = toolsEnabled && (named || choice == ChatRequest.ToolChoice.Kind.REQUIRED);
            // the prompt may still list the tools, so only the grammar keeps the model from calling them
            this.toolsForbidden = !tools.isEmpty() && choice == ChatRequest.ToolChoice.Kind.NONE
                    && template.field(ResponseTemplate.TOOL_CALLS) != null;
            boolean anyStrict = false;
            for (ChatTool t : tools) anyStrict |= Boolean.TRUE.equals(t.strict());
            this.strictAsked = toolsEnabled && (r.parameterStrict || anyStrict);
            // vLLM resolve_tool_strictness: no tool constraint for an automatic choice at the auto level without a
            // strict tool
            this.constrainedTools = toolsEnabled && (forced || r.functionStrict || strictAsked);
        }

        BeamGrammar derive() {
            if (forced && r.output != null) {
                throw refusal("tool_choice " + r.chat.toolChoice() + " constrains the output; it cannot be combined"
                        + " with another output constraint", "response_format");
            }
            ResponseTemplate.Field answer = template.field(ResponseTemplate.CONTENT);
            Map<String, Object> answerFormat = null;
            if (r.output != null) {
                answerFormat = answer != null && answer.opens() ? answerFormat(r.output) : null;
                if (answerFormat == null) {
                    // the request's own constraint holds the output, which leaves the calls without a grammar
                    if (strictAsked) throw unguaranteed("strict tools beside this structured output", "tools");
                    return null;
                }
            }
            boolean alwaysConstrain = r.alwaysConstrain && !forced;
            if (!constrainedTools && r.output == null && !toolsForbidden && !alwaysConstrain) return null;
            String open = openField(template, r.prompt);
            boolean messageStart = startsMessage(template, r.prompt);
            Map<String, Object> tag = null;
            if (ResponseTemplate.CONTENT.equals(open)) {
                // inside an open answer only the always-on grammar applies, finishing a free answer
                if (alwaysConstrain && r.output == null) tag = TemplateGrammar.answerRestTag(answer);
            } else if (!ResponseTemplate.THINKING.equals(open) || r.reasoningGated) {
                tag = toolCallTag(answer, answerFormat, messageStart);
            }
            if (tag == null) {
                refuseLostGuarantees();
                return null;
            }
            return new BeamGrammar(ConstraintSpec.structuralTag(Json.toString(tag, Json.Style.CANONICAL)), messageStart);
        }

        private Map<String, Object> toolCallTag(ResponseTemplate.Field answer, Map<String, Object> answerFormat,
                                                boolean messageStart) {
            List<ChatTool> functions = new ArrayList<>();
            if (toolsEnabled) {
                for (ChatTool t : r.chat.tools()) {
                    if (named && !t.name().equals(r.chat.toolChoice().functionName())) continue;
                    // vLLM resolve_tool_strictness: a tool's own strict flag, or every tool at the parameter level
                    functions.add(t.toBuilder().strict(r.parameterStrict || Boolean.TRUE.equals(t.strict())).build());
                }
            }
            TemplateGrammar.CallOptions o = new TemplateGrammar.CallOptions();
            o.parallelToolCalls = !named && r.parallelToolCalls.allowed();
            o.thinking = template.field(ResponseTemplate.THINKING);
            o.answer = forced ? null : answer;
            o.answerFormat = answerFormat;
            o.messageStart = messageStart ? TemplateGrammar.messageStartAnchors(template) : null;
            o.requiresReasoning = requiresReasoning(r.chat);
            // beam drifts from schemas the grammar does not enforce, and also writes arguments sorted by name
            o.sortedArguments = true;
            o.strictArguments = true;
            try {
                return TemplateGrammar.toolCallTag(template.field(ResponseTemplate.TOOL_CALLS), functions, o);
            } catch (IllegalArgumentException e) {
                throw refusal(e.getMessage(), "tools");
            }
        }

        /**
         * Refuses the tool guarantees a missing grammar loses, naming the parameter that asked for each; an output
         * constraint stays the request's own. The {@code parallel_tool_calls} refusal answers an explicit false
         * only: the template's one-call default limits the calls the parsers keep, it asks for nothing.
         */
        private void refuseLostGuarantees() {
            if (forced) throw unguaranteed((named ? "named" : "required") + " tool choice", "tool_choice");
            if (toolsForbidden) throw unguaranteed("tool_choice none", "tool_choice");
            if (strictAsked) throw unguaranteed("strict tools", "tools");
            ResponseTemplate.Field calls = template.field(ResponseTemplate.TOOL_CALLS);
            if (toolsEnabled && r.parallelToolCalls.singleAsked() && calls != null && calls.repeats()) {
                throw unguaranteed("parallel_tool_calls=False", "parallel_tool_calls");
            }
        }
    }

    /**
     * The xgrammar format of an output constraint inside the answer (vLLM {@code structured_outputs_to_format}): JSON
     * as a {@code json_schema}, a choice as its strings, a dot-all regex as a {@code regex}, and any other grammar as
     * its GBNF; null for a structural tag or a constraint with triggers or a gate. A JSON object is the schema
     * {@code {"type": "object"}}, as vLLM's; JSON in a whitespace mode other than the tag's travels as GBNF.
     */
    static Map<String, Object> answerFormat(ConstraintSpec output) {
        if (output.triggers() != null || output.gate() != null) return null;
        switch (output.kind()) {
            case CHOICE:
                return TemplateGrammar.constStrings(output.choices());
            case JSON_SCHEMA:
            case JSON_OBJECT: {
                String schema = output.kind() == ConstraintSpec.Kind.JSON_OBJECT ? OBJECT_SCHEMA : output.text();
                // the tag compiles its json_schema formats in its own whitespace mode; another policy travels as GBNF
                if (output.whitespace() != TAG_WHITESPACE || output.whitespacePattern() != null
                        || output.disableAdditionalProperties()) {
                    return grammarFormat(ConstraintSpec.jsonSchema(schema, output.whitespace(),
                            output.whitespacePattern(), output.disableAdditionalProperties()));
                }
                Map<String, Object> format = new LinkedHashMap<>();
                format.put("type", "json_schema");
                format.put("json_schema", Json.parse(schema, Json.Limits.UNTRUSTED));
                return format;
            }
            case REGEX:
                return output.dotAll() ? TemplateGrammar.regex(output.text()) : grammarFormat(output);
            case GBNF:
                return grammar(output.text());
            default:
                return null;
        }
    }

    /** The constraint's compiled grammar as a {@code grammar} format. */
    private static Map<String, Object> grammarFormat(ConstraintSpec output) {
        Grammar g = GbnfBuilder.grammarOf(output);
        String prefix = "answer-";
        return grammar(Grammar.ROOT_RULE + " ::= " + GbnfPrinter.ruleName(g, g.rootRuleIndex(), prefix) + "\n"
                + GbnfPrinter.print(g, prefix));
    }

    private static Map<String, Object> grammar(String gbnf) {
        Map<String, Object> f = new LinkedHashMap<>();
        f.put("type", "grammar");
        f.put("grammar", gbnf);
        return f;
    }

    /** {@code enable_thinking} of the request's template variables, true when unset (vLLM {@code bool(...)}). */
    static boolean requiresReasoning(ChatRequest chat) {
        Object v = chat.templateVariables().get("enable_thinking");
        return v == null || ContentParser.truthy(v);
    }

    /** Whether output after {@code prompt} starts the assistant message (vLLM {@code _starts_message}). */
    static boolean startsMessage(ResponseTemplate template, String prompt) {
        List<TemplateRegex.Match> anchors = template.startAnchorRegex().findAll(prompt);
        return !anchors.isEmpty() && anchors.get(anchors.size() - 1).end == prompt.length();
    }

    /** The field {@code prompt} leaves open with its explicit opener, or null (vLLM {@code _open_field}). */
    static String openField(ResponseTemplate template, String prompt) {
        String[] field = {null};
        new ResponseParser(template, prompt, ToolArguments.NONE, new ResponseParser.Listener() {
            @Override public void opened(String name, int start, int end, Map<String, String> captures) {
                if (end > start) field[0] = name;
            }

            @Override public void closed(String name, int start, int end, Object value) {
                field[0] = null;
            }

            @Override public void malformed(String name, int start, int end, String error) {
                field[0] = null;
            }
        });
        return field[0];
    }

    private static IllegalArgumentException unguaranteed(String guarantee, String parameter) {
        return refusal("response_template parsing (beam) cannot guarantee " + guarantee + " for this request", parameter);
    }

    private static IllegalArgumentException refusal(String message, String parameter) {
        return new IllegalArgumentException(message + "; request parameter '" + parameter + "'");
    }
}
