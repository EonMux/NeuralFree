package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.GbnfExclusion;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.RegexToGbnf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.chat.tools.NativeToolGrammar.WS;
import static com.eonmux.neuralfree.gguf.chat.tools.NativeToolGrammar.strict;

/**
 * The native tool-call grammars of the remaining registered formats, one builder per format: the envelope each
 * format's own parser reads — MiniMax-M3's namespace-tagged elements, K2 Horizon's arg tags, Granite 20B's
 * {@code <function_call>} objects, GigaChat 3's JSON header, MiniCPM5's {@code <param>} tags, Step 3's StepTML
 * invokes, Apertus' single-key array, FunctionGemma's escape-delimited values, Gemma 4's dict syntax, Inkling's
 * content blocks, Cohere Command's action array and Functionary's {@code >>>} recipients — with the arguments held
 * to the tool's parameter schema when the tool is strict and left free otherwise. The builders share
 * {@link NativeToolGrammar.Writer} and {@link NativeToolGrammar.Format} with the core formats.
 */
final class NativeToolGrammarFormats {

    /** The opening and closing tags of a Cohere Command action block. */
    private static final String COHERE_ACTION_OPEN = "<|START_ACTION|>";
    private static final String COHERE_ACTION_CLOSE = "<|END_ACTION|>";

    private NativeToolGrammarFormats() {
    }

    /**
     * The builder of {@code name}, or null when no builder here serves it; {@code k2CallFormat} is the K2 Horizon
     * template's selected call layout ({@code xml}, {@code xml_typed} or {@code json}).
     */
    static NativeToolGrammar.Format format(String name, String k2CallFormat) {
        if (name == null) {
            return null;
        }
        switch (name.toLowerCase(Locale.ROOT)) {
            case "minimax_m3":
                return minimaxM3();
            case "k2_horizon":
                return k2Horizon(k2CallFormat);
            case "granite-20b-fc":
                return granite20bFc();
            case "gigachat3":
                return gigachat3();
            case "minicpm5":
                return minicpm5();
            case "step3":
                return step3();
            case "apertus":
                return apertus();
            case "functiongemma":
                return functiongemma();
            case "gemma4":
                return gemma4();
            case "inkling":
                return inkling();
            case "cohere_command3":
                return cohere();
            case "cohere_command4":
                return cohere();
            case "functionary_v3_2":
                return functionary();
            default:
                return null;
        }
    }

    private static String literal(String text) {
        return HarmonyToolGrammar.literal(text);
    }

    /** One or more {@code unit}s, whitespace between them (the parsers skip it). */
    private static String repeated(String unit) {
        return unit + " ( " + WS + " " + unit + " )*";
    }

    // ------------------------------------------------------------------ MiniMax M3

    /** MiniMax-M3's namespace token, which every tool tag carries. */
    private static final String MM_NS = "]<]minimax[>[";
    private static final String MM_CALL_OPEN = MM_NS + "<tool_call>";
    private static final String MM_CALL_CLOSE = MM_NS + "</tool_call>";
    private static final String MM_INVOKE_OPEN = MM_NS + "<invoke name=\"";
    private static final String MM_INVOKE_CLOSE = MM_NS + "</invoke>";

    /**
     * MiniMax-M3: one call block of {@code <invoke name="NAME">} elements, each argument an element whose tag is the
     * parameter name — a string value raw text, a nested object or array the template's recursive element form
     * ({@code <item>} elements), any other value the JSON text its schema admits — inside the namespace
     * {@code ]<]minimax[>[}. A bare invoke is the parser's other opener under {@code auto}.
     */
    private static NativeToolGrammar.Format minimaxM3() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                return invoke(w, index, tool);
            }

            private String invoke(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                String params = w.taggedParams(index, tool," ", (writer, i, p, key) ->
                        element(writer, writer.base + "-tool-" + i + "-param-" + p, key, tool,
                                NativeToolGrammar.property(tool, key)));
                return w.g.addRule(w.base + "-tool-" + index,
                        literal(MM_INVOKE_OPEN + tool.name() + "\">") + " "
                                + (params.isEmpty() ? "" : params + " " + WS)
                                + literal(MM_INVOKE_CLOSE));
            }

            /** One argument element: {@code <tag>} then its value in the template's recursive form. */
            private String element(NativeToolGrammar.Writer w, String rule, String tag, ChatTool tool,
                                   Object schema) {
                Object property = schema instanceof Map ? schema : null;
                return w.g.addRule(rule, literal(MM_NS + "<" + tag + ">") + " "
                        + value(w, rule, tool, property) + " " + literal(MM_NS + "</" + tag + ">"));
            }

            /**
             * The value of an element: raw text for a string, the template's child elements for a nested object
             * ({@code <key>}) or array ({@code <item>}), JSON text for any other type, and any JSON value for an
             * undeclared one. A tool that does not hold its arguments to the schema keeps every value free text.
             */
            private String value(NativeToolGrammar.Writer w, String rule, ChatTool tool, Object schema) {
                if (!strict(tool) || (schema != null && ToolSchemas.textValuedSchema(schema))) {
                    return free(w, rule + "-text");
                }
                if (ToolSchemas.objectSchema(schema)) {
                    return members(w, rule, tool, schema);
                }
                if (ToolSchemas.arraySchema(schema)) {
                    String item = element(w, rule + "-item", "item", tool, ToolSchemas.itemsSchema(schema));
                    return "( " + WS + " " + item + " )*";
                }
                return w.g.addSchema(rule + "-json",
                        schema == null ? NativeToolGrammar.ANY_VALUE : ToolRouting.write(schema));
            }

            /** The members of a nested object: the required properties in declared order, then any optional one. */
            private String members(NativeToolGrammar.Writer w, String rule, ChatTool tool, Object schema) {
                Map<?, ?> props = (Map<?, ?>) ((Map<?, ?>) schema).get("properties");
                List<?> required = ToolSchemas.requiredProperties(schema);
                List<String> requiredMembers = new ArrayList<>();
                List<String> optionalMembers = new ArrayList<>();
                int p = 0;
                for (Map.Entry<?, ?> e : props.entrySet()) {
                    String key = String.valueOf(e.getKey());
                    String member = element(w, rule + "-m" + p, key, tool, e.getValue());
                    (required.contains(key) ? requiredMembers : optionalMembers).add(member);
                    p++;
                }
                StringBuilder out = new StringBuilder(String.join(" ", requiredMembers));
                if (!optionalMembers.isEmpty()) {
                    String alts = optionalMembers.size() == 1 ? optionalMembers.get(0)
                            : "( " + String.join(" | ", optionalMembers) + " )";
                    out.append(" ( ").append(alts).append(" )*");
                }
                return out.toString();
            }

            /** Raw text that never writes the namespace token, so it cannot open a nested element. */
            private String free(NativeToolGrammar.Writer w, String rule) {
                return GbnfExclusion.excluding(w.g::addRule, rule, Collections.singletonList(MM_NS));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String section = section(w, unit, atMostOne);
                String repeated = "( " + section + " | " + unit + " )";
                return repeated + " ( " + WS + " " + repeated + " )*";
            }

            @Override
            public String forcingRoot(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools,
                                      boolean parallel, boolean atMostOne) {
                return section(w, unit, atMostOne);
            }

            private String section(NativeToolGrammar.Writer w, String unit, boolean atMostOne) {
                String calls = atMostOne ? unit : repeated(unit);
                return w.g.addRule(w.base + "-section", literal(MM_CALL_OPEN) + " " + WS + " " + calls
                        + " " + WS + " " + literal(MM_CALL_CLOSE));
            }
        };
    }

    // ------------------------------------------------------------------ K2 Horizon

    private static final String K2_SECTION_OPEN = "<ifm|tool_calls>";
    private static final String K2_SECTION_CLOSE = "</ifm|tool_calls>";
    private static final String K2_CALL_OPEN = "<ifm|tool_call>";
    private static final String K2_CALL_CLOSE = "</ifm|tool_call>";
    private static final String K2_KEY_OPEN = "<ifm|arg_key>";
    private static final String K2_KEY_CLOSE = "</ifm|arg_key>";
    private static final String K2_TYPE_OPEN = "<ifm|arg_type>";
    private static final String K2_TYPE_CLOSE = "</ifm|arg_type>";
    private static final String K2_VALUE_OPEN = "<ifm|arg_value>";
    private static final String K2_VALUE_CLOSE = "</ifm|arg_value>";

    /**
     * K2 Horizon: a section of {@code <ifm|tool_call>NAME} calls, one {@code <ifm|arg_key>K</ifm|arg_key>
     * <ifm|arg_value>V</ifm|arg_value>} pair per argument; a value its schema types as a string stays raw text,
     * any other is its JSON. The template's {@code xml_typed} layout adds the optional {@code <ifm|arg_type>} tag
     * before each value, and its {@code json} layout writes the call body as
     * {@code {"name": ..., "arguments": ...}} inside the same call tags. A bare call is the parser's other opener
     * under {@code auto}.
     */
    private static NativeToolGrammar.Format k2Horizon(String callFormat) {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                if ("json".equals(callFormat)) {
                    return jsonCall(w, index, tool);
                }
                List<String> excludes = Arrays.asList(K2_VALUE_CLOSE, K2_KEY_OPEN, K2_CALL_CLOSE);
                boolean typed = "xml_typed".equals(callFormat);
                String params = w.taggedParams(index, tool,"", (writer, i, p, key) ->
                        writer.g.addRule(writer.base + "-tool-" + i + "-param-" + p,
                                literal(K2_KEY_OPEN + key + K2_KEY_CLOSE) + " "
                                        + (typed ? argumentType(writer, i, p) : "")
                                        + literal(K2_VALUE_OPEN) + " "
                                        + writer.rawValue(i, p, tool, key, excludes)
                                        + " " + literal(K2_VALUE_CLOSE)));
                return w.g.addRule(w.base + "-tool-" + index,
                        literal(K2_CALL_OPEN + tool.name()) + " " + WS + " "
                                + (params.isEmpty() ? "" : params + " " + WS + " ")
                                + literal(K2_CALL_CLOSE));
            }

            /** The optional {@code arg_type} tag of the {@code xml_typed} layout. */
            private String argumentType(NativeToolGrammar.Writer w, int index, int p) {
                String body = GbnfExclusion.excluding(w.g::addRule,
                        w.base + "-tool-" + index + "-type-" + p, Collections.singletonList(K2_TYPE_CLOSE));
                return "( " + literal(K2_TYPE_OPEN) + " " + body + " " + literal(K2_TYPE_CLOSE) + " "
                        + WS + " )? ";
            }

            /** The JSON call body the template's {@code json} layout writes. */
            private String jsonCall(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                String args = w.args(index, tool);
                return w.g.addRule(w.base + "-tool-" + index,
                        literal(K2_CALL_OPEN) + " " + WS + " "
                                + literal("{\"name\": \"" + tool.name() + "\", \"arguments\": ") + " " + args
                                + " " + literal("}") + " " + WS + " " + literal(K2_CALL_CLOSE));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String section = section(w, unit, atMostOne);
                String repeated = "( " + section + " | " + unit + " )";
                return repeated + " ( " + WS + " " + repeated + " )*";
            }

            @Override
            public String forcingRoot(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools,
                                      boolean parallel, boolean atMostOne) {
                return section(w, unit, atMostOne);
            }

            private String section(NativeToolGrammar.Writer w, String unit, boolean atMostOne) {
                String calls = atMostOne ? unit : repeated(unit);
                return w.g.addRule(w.base + "-section", literal(K2_SECTION_OPEN) + " " + WS + " " + calls
                        + " " + WS + " " + literal(K2_SECTION_CLOSE));
            }
        };
    }

    // ------------------------------------------------------------------ Granite 20B FC

    /**
     * Granite 20B FC: {@code <function_call>{"name": "NAME", "arguments": ARGS}} blocks, one call each, the
     * opener repeated for further calls (the parser reads an object run after the opener).
     */
    private static NativeToolGrammar.Format granite20bFc() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                return w.g.addRule(w.base + "-tool-" + index,
                        literal("<function_call>") + " " + WS + " "
                                + literal("{\"name\": \"" + tool.name() + "\", \"arguments\": ") + " "
                                + w.args(index, tool) + " " + literal("}"));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return atMostOne ? unit : repeated(unit);
            }
        };
    }

    // ------------------------------------------------------------------ GigaChat 3

    private static final String GIGA_ROLE_HEADER = "function call<|role_sep|>\n";
    private static final String GIGA_FLAT_HEADER = "<|function_call|>";
    private static final String GIGA_MESSAGE_SEP = "<|message_sep|>\n\n";

    /**
     * GigaChat 3: one JSON call object after the {@code function call<|role_sep|>} header or the flat
     * {@code <|function_call|>} token, optionally preceded by the message separator the template writes.
     */
    private static NativeToolGrammar.Format gigachat3() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                String open = "( " + literal(GIGA_ROLE_HEADER) + " | " + literal(GIGA_FLAT_HEADER) + " )";
                String body = literal("{\"name\": \"" + tool.name() + "\", \"arguments\": ") + " "
                        + w.args(index, tool) + " " + literal("}");
                return w.g.addRule(w.base + "-tool-" + index,
                        "( " + literal(GIGA_MESSAGE_SEP) + " )? " + open + " " + WS + " " + body);
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return atMostOne ? unit : repeated(unit);
            }
        };
    }

    // ------------------------------------------------------------------ MiniCPM5

    /**
     * MiniCPM5: bare {@code <function name="NAME">} blocks, one call each, with one
     * {@code <param name="KEY">VALUE</param>} per argument — a string value raw text, any other its JSON.
     */
    private static NativeToolGrammar.Format minicpm5() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                List<String> excludes = Arrays.asList("</param>", "<param name=", "</function>");
                String params = w.taggedParams(index, tool,"", (writer, i, p, key) ->
                        writer.g.addRule(writer.base + "-tool-" + i + "-param-" + p,
                                literal("<param name=\"" + key + "\">") + " "
                                        + writer.rawValue(i, p, tool, key,excludes)
                                        + " " + literal("</param>")));
                return w.g.addRule(w.base + "-tool-" + index,
                        literal("<function name=\"" + tool.name() + "\">") + " " + WS + " "
                                + (params.isEmpty() ? "" : params + " " + WS + " ")
                                + literal("</function>"));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return atMostOne ? unit : repeated(unit);
            }
        };
    }

    // ------------------------------------------------------------------ Step 3

    private static final String STEP_SECTION_OPEN = "<｜tool_calls_begin｜>";
    private static final String STEP_SECTION_CLOSE = "<｜tool_calls_end｜>";
    private static final String STEP_CALL_OPEN = "<｜tool_call_begin｜>";
    private static final String STEP_CALL_CLOSE = "<｜tool_call_end｜>";
    private static final String STEP_CALL_PREFIX = "function<｜tool_sep｜>";
    private static final String STEP_SEP = "<｜tool_sep｜>";
    private static final String STEP_INVOKE_OPEN = "<steptml:invoke name=\"";
    private static final String STEP_INVOKE_CLOSE = "</steptml:invoke>";
    private static final String STEP_PARAM_OPEN = "<steptml:parameter name=\"";
    private static final String STEP_PARAM_CLOSE = "</steptml:parameter>";

    /**
     * Step 3: a section of {@code <｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="NAME">} calls,
     * one {@code <steptml:parameter name="KEY">VALUE</steptml:parameter>} per argument, the template's
     * {@code <｜tool_sep｜>} between adjacent calls.
     */
    private static NativeToolGrammar.Format step3() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                List<String> excludes = Arrays.asList(STEP_PARAM_CLOSE, STEP_PARAM_OPEN, STEP_INVOKE_CLOSE);
                String params = w.taggedParams(index, tool,"", (writer, i, p, key) ->
                        writer.g.addRule(writer.base + "-tool-" + i + "-param-" + p,
                                literal(STEP_PARAM_OPEN + key + "\">") + " "
                                        + writer.rawValue(i, p, tool, key,excludes)
                                        + " " + literal(STEP_PARAM_CLOSE)));
                return w.g.addRule(w.base + "-tool-" + index,
                        literal(STEP_CALL_OPEN + STEP_CALL_PREFIX + STEP_INVOKE_OPEN + tool.name() + "\">") + " "
                                + WS + " " + (params.isEmpty() ? "" : params + " " + WS + " ")
                                + literal(STEP_INVOKE_CLOSE + STEP_CALL_CLOSE));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return section(w, unit, false);
            }

            @Override
            public String forcingRoot(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools,
                                      boolean parallel, boolean atMostOne) {
                return section(w, unit, atMostOne);
            }

            private String section(NativeToolGrammar.Writer w, String unit, boolean atMostOne) {
                String calls = atMostOne ? unit : unit + " ( " + WS + " ( " + literal(STEP_SEP) + " )? "
                        + WS + " " + unit + " )*";
                return w.g.addRule(w.base + "-section", literal(STEP_SECTION_OPEN) + " " + WS + " " + calls
                        + " " + WS + " " + literal(STEP_SECTION_CLOSE));
            }
        };
    }

    // ------------------------------------------------------------------ Apertus

    private static final String APERTUS_PREFIX = "<|tools_prefix|>";
    private static final String APERTUS_SUFFIX = "<|tools_suffix|>";

    /**
     * Apertus: an array of single-key objects — the function name mapped to its arguments object — between the
     * tools prefix and the optional suffix; parallel calls are array elements. The suffix is optional, as vLLM's
     * parser reads a block that ends after the array.
     */
    private static NativeToolGrammar.Format apertus() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                return w.g.addRule(w.base + "-tool-" + index,
                        literal("{\"" + tool.name() + "\": ") + " " + w.args(index, tool) + " " + literal("}"));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String list = atMostOne ? unit
                        : unit + " ( " + WS + " " + literal(",") + " " + WS + " " + unit + " )*";
                return w.g.addRule(w.base + "-array",
                        literal(APERTUS_PREFIX) + " " + WS + " " + literal("[") + " " + WS + " " + list
                                + " " + WS + " " + literal("]") + " ( " + WS + " "
                                + literal(APERTUS_SUFFIX) + " )?");
            }
        };
    }

    // ------------------------------------------------------------------ FunctionGemma

    private static final String FG_OPEN = "<start_function_call>";
    private static final String FG_CLOSE = "<end_function_call>";
    private static final String FG_ESCAPE = "<escape>";

    /**
     * FunctionGemma: {@code <start_function_call>call:NAME{KEY:<escape>VALUE<escape>}} calls, the keys the tool's
     * properties, a value its schema types as a string raw text, any other its JSON, and the arguments comma
     * separated.
     */
    private static NativeToolGrammar.Format functiongemma() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                String params = w.taggedParams(index, tool,",", (writer, i, p, key) ->
                        writer.g.addRule(writer.base + "-tool-" + i + "-param-" + p,
                                literal(key + ":" + FG_ESCAPE) + " "
                                        + writer.rawValue(i, p, tool, key, Collections.singletonList(FG_ESCAPE))
                                        + " " + literal(FG_ESCAPE)));
                return w.g.addRule(w.base + "-tool-" + index,
                        literal(FG_OPEN + "call:" + tool.name() + "{") + " "
                                + (params.isEmpty() ? "" : params + " ")
                                + literal("}" + FG_CLOSE));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return atMostOne ? unit : repeated(unit);
            }
        };
    }

    // ------------------------------------------------------------------ Gemma 4

    private static final String GEMMA_OPEN = "<|tool_call>";
    private static final String GEMMA_CLOSE = "<tool_call|>";
    private static final String GEMMA_STRING = "<|\"|>";

    /**
     * Gemma 4: {@code <|tool_call>call:NAME{KEY:VALUE,...}<tool_call|>} calls (llama.cpp's {@code gemma4}
     * grammar). A value is a string between {@code <|"|>} delimiters, a nested dict or array written in the same
     * syntax, or a bare JSON number, boolean or null; a strict tool's declared keys are pinned and its values
     * written by their schemas — a string's {@code enum}, {@code pattern} and length bounds inside the delimiters —
     * a loose tool's arguments are a free dict.
     */
    private static NativeToolGrammar.Format gemma4() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                String base = w.base + "-tool-" + index;
                if (!strict(tool)) {
                    return w.g.addRule(base, literal(GEMMA_OPEN + "call:" + tool.name()) + " "
                            + gemmaFreeDict(w, base) + " " + literal(GEMMA_CLOSE));
                }
                String members = gemmaMembers(w, base, ToolRouting.parameters(tool));
                String body = members.isEmpty()
                        ? literal(GEMMA_OPEN + "call:" + tool.name() + "{}" + GEMMA_CLOSE)
                        : literal(GEMMA_OPEN + "call:" + tool.name() + "{") + " " + members + " " + WS + " "
                        + literal("}" + GEMMA_CLOSE);
                return w.g.addRule(base, body);
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return atMostOne ? unit : repeated(unit);
            }
        };
    }

    /** The members of one Gemma dict: the required properties in declared order, then any optional one. */
    private static String gemmaMembers(NativeToolGrammar.Writer w, String base, Map<?, ?> parameters) {
        Object propsRaw = parameters.get("properties");
        if (!(propsRaw instanceof Map) || ((Map<?, ?>) propsRaw).isEmpty()) {
            return "";
        }
        List<?> requiredNames = parameters.get("required") instanceof List
                ? (List<?>) parameters.get("required") : Collections.emptyList();
        List<String> requiredMembers = new ArrayList<>();
        List<String> optionalMembers = new ArrayList<>();
        int p = 0;
        for (Map.Entry<?, ?> e : ((Map<?, ?>) propsRaw).entrySet()) {
            String key = String.valueOf(e.getKey());
            String member = w.g.addRule(base + "-m" + p,
                    literal(key + ":") + " " + gemmaValue(w, base + "-m" + p, e.getValue()));
            (requiredNames.contains(key) ? requiredMembers : optionalMembers).add(member);
            p++;
        }
        String comma = " " + literal(",") + " " + WS + " ";
        StringBuilder out = new StringBuilder(String.join(comma, requiredMembers));
        if (!optionalMembers.isEmpty()) {
            String alts = optionalMembers.size() == 1 ? optionalMembers.get(0)
                    : "( " + String.join(" | ", optionalMembers) + " )";
            if (out.length() == 0) {
                out.append(alts).append(" ( ").append(comma).append(alts).append(" )*");
            } else {
                out.append(" ( ").append(comma).append(alts).append(" )*");
            }
        }
        return out.toString();
    }

    /** One Gemma value of the schema {@code schema}. */
    private static String gemmaValue(NativeToolGrammar.Writer w, String base, Object schema) {
        if (!(schema instanceof Map)) {
            return gemmaAny(w, base);
        }
        Map<?, ?> m = (Map<?, ?>) schema;
        Object type = m.get("type");
        if (isStringSchema(m)) {
            return gemmaString(w, base, m);
        }
        if ("object".equals(type)) {
            Object props = m.get("properties");
            if (props instanceof Map && !((Map<?, ?>) props).isEmpty()) {
                return literal("{") + " " + gemmaMembers(w, base, m) + " " + WS + " " + literal("}");
            }
            return gemmaAny(w, base);
        }
        if ("array".equals(type)) {
            String item = gemmaValue(w, base + "-item", m.get("items"));
            return w.g.addRule(base + "-array", literal("[") + " " + WS + " ( " + item + " ( "
                    + literal(",") + " " + WS + " " + item + " )* )? " + WS + " " + literal("]"));
        }
        if ("integer".equals(type) || "number".equals(type) || "boolean".equals(type)
                || "null".equals(type)) {
            return w.g.addSchema(base + "-json", ToolRouting.write(schema));
        }
        return gemmaAny(w, base);
    }

    /** Whether the schema is a string: it declares {@code string}, or an {@code enum}/{@code const} of strings. */
    private static boolean isStringSchema(Map<?, ?> schema) {
        Object type = schema.get("type");
        if ("string".equals(type)) {
            return true;
        }
        if (type != null && !"string".equals(type)) {
            return false;
        }
        List<?> values = stringConstants(schema);
        return values != null;
    }

    /** The string values of an {@code enum} or {@code const}; null when the schema has neither or a value is no string. */
    private static List<?> stringConstants(Map<?, ?> schema) {
        Object raw;
        if (schema.containsKey("const")) {
            raw = Collections.singletonList(schema.get("const"));
        } else if (schema.get("enum") instanceof List) {
            raw = schema.get("enum");
        } else {
            return null;
        }
        List<?> values = (List<?>) raw;
        if (values.isEmpty()) {
            return null;
        }
        for (Object value : values) {
            if (!(value instanceof String)) {
                return null;
            }
        }
        return values;
    }

    /**
     * A Gemma string value: raw text between two {@code <|"|>} delimiters, held to the schema's {@code enum},
     * {@code pattern} and length bounds where that constrained text cannot hold the delimiter itself, else free
     * text. A constrained form is written only when it cannot spell {@code <|"|>}: an enum of delimiter-holding
     * values, or a pattern that may write the delimiter's first character, falls back to the free text.
     */
    private static String gemmaString(NativeToolGrammar.Writer w, String base, Map<?, ?> schema) {
        String content = gemmaStringContent(w, base, schema);
        if (content == null) {
            content = GbnfExclusion.excluding(w.g::addRule, base + "-string-free",
                    Collections.singletonList(GEMMA_STRING));
        }
        return w.g.addRule(base + "-string", literal(GEMMA_STRING) + " " + content + " " + literal(GEMMA_STRING));
    }

    /** A Gemma string value that holds no delimiter: any raw text between two {@code <|"|>} delimiters. */
    private static String gemmaFreeString(NativeToolGrammar.Writer w, String base) {
        String free = GbnfExclusion.excluding(w.g::addRule, base + "-string-free",
                Collections.singletonList(GEMMA_STRING));
        return w.g.addRule(base + "-string", literal(GEMMA_STRING) + " " + free + " " + literal(GEMMA_STRING));
    }

    /**
     * The constrained content of a Gemma string, or null when the schema constrains nothing the delimiters can
     * hold: its enum or const, else a pattern and no delimiter, else a length bound (the text then holds no
     * {@code '<'}, so no delimiter can start).
     */
    private static String gemmaStringContent(NativeToolGrammar.Writer w, String base, Map<?, ?> schema) {
        List<?> values = stringConstants(schema);
        if (values != null) {
            for (Object value : values) {
                if (((String) value).contains(GEMMA_STRING)) {
                    return null;
                }
            }
            List<String> literals = new ArrayList<>(values.size());
            for (Object value : values) {
                literals.add(literal((String) value));
            }
            return w.g.addRule(base + "-enum", literals.size() == 1 ? literals.get(0)
                    : "( " + String.join(" | ", literals) + " )");
        }
        Object pattern = schema.get("pattern");
        if (pattern instanceof String && !((String) pattern).isEmpty()) {
            String regex = (String) pattern;
            if (!RegexToGbnf.mayWrite(regex, Collections.singleton((int) '<'))) {
                try {
                    return w.g.addRawPattern(base + "-pattern", regex);
                } catch (RegexToGbnf.UnsupportedRegexException e) {
                    // no grammar equivalent: the string stays free rather than breaking the format
                } catch (RegexToGbnf.InvalidRegexException e) {
                    throw invalidPattern(regex, e);
                }
            } else {
                checkPattern(regex);
            }
        }
        int min = lengthBound(schema.get("minLength"), "minLength", 0);
        int max = lengthBound(schema.get("maxLength"), "maxLength", -1);
        if (min > 0 || max >= 0) {
            return GbnfBuilder.repetition("[^<]", min, max < 0 ? Integer.MAX_VALUE : max, "");
        }
        return null;
    }

    /**
     * Checks that a pattern the grammar cannot hold in a string value is still a valid regex, so a broken schema is
     * refused rather than silently left free; a valid pattern with no grammar equivalent stays free.
     */
    private static void checkPattern(String pattern) {
        try {
            RegexToGbnf.compile(pattern, Grammar.ROOT_RULE, true);
        } catch (RegexToGbnf.InvalidRegexException e) {
            throw invalidPattern(pattern, e);
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            // a construct the translator cannot express: the string stays free
        }
    }

    private static IllegalArgumentException invalidPattern(String pattern, RuntimeException cause) {
        return new IllegalArgumentException("a tool parameter's pattern '" + pattern + "' is invalid: "
                + cause.getMessage(), cause);
    }

    /**
     * A string length bound: {@code absent} when the schema does not declare it, the value when it is a
     * non-negative integer.
     *
     * @throws IllegalArgumentException naming the keyword when the value is not one
     */
    private static int lengthBound(Object value, String keyword, int absent) {
        if (value == null) {
            return absent;
        }
        if (value instanceof Number) {
            double d = ((Number) value).doubleValue();
            int n = ((Number) value).intValue();
            if (n >= 0 && d == n) {
                return n;
            }
        }
        throw new IllegalArgumentException("a tool parameter's " + keyword + " must be a non-negative integer, got "
                + value);
    }

    /** Any Gemma value: a string, a free dict, a free array, or a bare JSON scalar. */
    private static String gemmaAny(NativeToolGrammar.Writer w, String base) {
        gemmaGenericDict(w, base);
        return gemmaValueUnion(w, base);
    }

    /** A free Gemma dict: any keys, each an arbitrary Gemma value. */
    private static String gemmaFreeDict(NativeToolGrammar.Writer w, String base) {
        String dict = gemmaGenericDict(w, base);
        gemmaValueUnion(w, base);
        return dict;
    }

    /** The members of a free Gemma dict, valid while {@link #gemmaValueUnion} defines its values. */
    private static String gemmaGenericDict(NativeToolGrammar.Writer w, String base) {
        String keyRest = GbnfExclusion.excluding(w.g::addRule, base + "-key-rest",
                Arrays.asList(":", "}", ","));
        String key = w.g.addRule(base + "-key", "[^:},] " + keyRest);
        String kv = w.g.addRule(base + "-kv", key + " " + literal(":") + " " + base + "-value");
        return w.g.addRule(base + "-dict", literal("{") + " " + WS + " ( " + kv + " ( "
                + literal(",") + " " + WS + " " + kv + " )* )? " + WS + " " + literal("}"));
    }

    /** The union of the Gemma value alternatives, a free dict and array included. */
    private static String gemmaValueUnion(NativeToolGrammar.Writer w, String base) {
        String string = gemmaFreeString(w, base);
        String number = w.g.addSchema(base + "-number", "{\"type\": \"number\"}");
        String bool = w.g.addSchema(base + "-boolean", "{\"type\": \"boolean\"}");
        String nul = w.g.addSchema(base + "-null", "{\"type\": \"null\"}");
        String array = w.g.addRule(base + "-array", literal("[") + " " + WS + " ( " + base + "-value ( "
                + literal(",") + " " + WS + " " + base + "-value )* )? " + WS + " " + literal("]"));
        return w.g.addRule(base + "-value", "( " + string + " | " + base + "-dict | " + array + " | "
                + number + " | " + bool + " | " + nul + " )");
    }

    // ------------------------------------------------------------------ Inkling

    private static final String INK_HEADER = "<|message_model|>";
    private static final String INK_TEXT = "<|content_text|>";
    private static final String INK_THINKING = "<|content_thinking|>";
    private static final String INK_TOOL_JSON = "<|content_invoke_tool_json|>";
    private static final String INK_TOOL_TEXT = "<|content_invoke_tool_text|>";
    private static final String INK_TOOL_ERROR = "<|content_tool_error|>";
    private static final String INK_END = "<|end_message|>";
    private static final String INK_SAMPLING = "<|content_model_end_sampling|>";

    /**
     * Inkling: self-describing message blocks, a tool block's body one JSON object of {@code name} and
     * {@code args}. A forcing choice writes only tool blocks; from a lazy trigger the closed text, thinking and
     * tool blocks the parser reads may follow one another.
     */
    private static NativeToolGrammar.Format inkling() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                return w.g.addRule(w.base + "-tool-" + index,
                        literal(INK_TOOL_JSON) + " " + WS + " "
                                + literal("{\"name\": \"" + tool.name() + "\", \"args\": ") + " "
                                + w.args(index, tool) + " " + literal("}") + " " + WS + " " + close(w));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String text = GbnfExclusion.excluding(w.g::addRule, w.base + "-body", markers());
                String header = header(w);
                String textBlock = w.g.addRule(w.base + "-text-block", header + " ( " + literal(INK_TEXT)
                        + " | " + literal(INK_TOOL_TEXT) + " | " + literal(INK_TOOL_ERROR) + " ) "
                        + WS + " " + text + " " + WS + " " + close(w));
                String thinkingBlock = w.g.addRule(w.base + "-thinking-block",
                        header + literal(INK_THINKING) + " " + WS + " " + text + " " + WS + " " + close(w));
                String toolBlock = w.g.addRule(w.base + "-tool-block", header + " " + unit);
                String block = "( " + toolBlock + " | " + textBlock + " | " + thinkingBlock + " )";
                return block + " ( " + WS + " " + block + " )*";
            }

            @Override
            public String forcingRoot(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools,
                                      boolean parallel, boolean atMostOne) {
                String headed = header(w) + " " + unit;
                return atMostOne ? headed : repeated(headed);
            }

            private String header(NativeToolGrammar.Writer w) {
                return "( " + literal(INK_HEADER) + " " + WS + " )?";
            }

            private String close(NativeToolGrammar.Writer w) {
                return "( " + literal(INK_END) + " | " + literal(INK_SAMPLING) + " )";
            }

            private List<String> markers() {
                return Arrays.asList(INK_HEADER, INK_TEXT, INK_THINKING, INK_TOOL_JSON, INK_TOOL_TEXT,
                        INK_TOOL_ERROR, INK_END, INK_SAMPLING);
            }
        };
    }

    // ------------------------------------------------------------------ Cohere Command 3/4

    /**
     * Cohere Command 3/4: one action block holding a JSON array of
     * {@code {"tool_call_id": "N", "tool_name": "NAME", "parameters": ARGS}} objects; the two formats differ only
     * in the content blocks beside the calls, so their call grammars are the same.
     */
    private static NativeToolGrammar.Format cohere() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                return w.g.addRule(w.base + "-tool-" + index,
                        literal("{\"tool_call_id\": \"") + " [0-9]+ "
                                + literal("\", \"tool_name\": \"" + tool.name() + "\", \"parameters\": ") + " "
                                + w.args(index, tool) + " " + literal("}"));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String list = atMostOne ? unit
                        : unit + " ( " + WS + " " + literal(",") + " " + WS + " " + unit + " )*";
                return w.g.addRule(w.base + "-action",
                        literal(COHERE_ACTION_OPEN) + " " + WS + " " + literal("[") + " " + WS + " "
                                + list + " " + WS + " " + literal("]") + " " + WS
                                + " " + literal(COHERE_ACTION_CLOSE));
            }
        };
    }

    // ------------------------------------------------------------------ Functionary v3.2

    /**
     * Functionary v3.2: {@code >>>NAME\nARGS} recipients, one JSON arguments object each. A forcing choice writes
     * the calls; from a lazy trigger on {@code >>>} a {@code >>>all} reply may precede them (the reply text runs to
     * the next {@code >>>}), as the parser reads it.
     */
    private static NativeToolGrammar.Format functionary() {
        return new NativeToolGrammar.Format() {
            @Override
            public String unit(NativeToolGrammar.Writer w, int index, ChatTool tool) {
                return w.g.addRule(w.base + "-tool-" + index,
                        literal(">>>" + tool.name() + "\n") + " " + WS + " " + w.args(index, tool));
            }

            @Override
            public String root(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String calls = repeated(unit);
                String content = GbnfExclusion.excluding(w.g::addRule, w.base + "-content",
                        Collections.singletonList(">>>"));
                String all = literal(">>>all\n") + " " + WS + " " + content + " ( " + calls + " )?";
                return w.g.addRule(w.base + "-root", "( " + all + " | " + calls + " )");
            }

            @Override
            public String forcingRoot(NativeToolGrammar.Writer w, String unit, List<ChatTool> tools,
                                      boolean parallel, boolean atMostOne) {
                return atMostOne ? unit : repeated(unit);
            }
        };
    }
}
