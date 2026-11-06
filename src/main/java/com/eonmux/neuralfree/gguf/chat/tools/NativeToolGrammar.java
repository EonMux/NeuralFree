package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserFormat;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfExclusion;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The tool-call grammars written in a registered format's own syntax, one builder per format (vLLM's
 * {@code tool_parsers/structural_tag_registry.py} per-model builders, llama.cpp's per-parser tool-call
 * grammars): a {@link #forcing} call is the format's own envelope — Hermes' {@code <tool_call>{json}
 * </tool_call>}, Qwen3-Coder's {@code <function=...>} parameters, DeepSeek's sectioned calls, the pythonic
 * {@code [f(k=v)]} list — with the arguments held to the tool's parameter schema when the tool is strict and
 * left free (any JSON object, or the format's raw value text) when it is not. {@link #lazy} is the same
 * envelope as a lazy grammar behind the format's call openers, for a {@code tool_choice: auto} request whose
 * strictness asks for one (llama.cpp's {@code grammar_lazy} with word triggers on the call opener).
 *
 * <p>The rules reuse this package's parser shapes ({@link JsonToolParser}, {@link XmlToolParser},
 * {@link NameArgsToolParser}, {@link MistralToolParser}, {@link PythonicToolParser}), so a constrained
 * output reads back through the format's own parser. A format without a row keeps the routing's generic
 * behavior. Tagged parameters follow the declared schema order: the required properties first, then any
 * optional one, any number of times — the parsers read the values whichever order they arrive in. The DSML
 * (DeepSeek V3.2, V4, V4.1) and MiMo rows are xgrammar's builtin structural tags for those models, their
 * parameters the {@code json_schema} XML styles ({@code deepseek_xml}, {@code deepseek_v4_1_xml},
 * {@code qwen_xml}) of {@link JsonSchemaToGbnf.Composer#addXmlParameters}; the Kimi K3, PLaMo 3, HY v4 and
 * GLM 4.7 rows are vLLM's own builtin structural-tag builders
 * ({@code tool_parsers/structural_tag_registry.py}) for those models, written in their formats' syntax. The
 * remaining registered formats — MiniMax-M3, K2 Horizon, Granite 20B FC, GigaChat 3, MiniCPM5, Step 3, Apertus,
 * FunctionGemma, Gemma 4, Inkling, Cohere Command 3/4 and Functionary v3.2 — have their builders in
 * {@link NativeToolGrammarFormats}.
 */
final class NativeToolGrammar {

    /** Any JSON object, the free arguments of a non-strict call in a JSON format. */
    static final String ANY_OBJECT = "{\"type\": \"object\", \"additionalProperties\": true}";

    /** Any JSON value, the free argument of a non-strict pythonic keyword (a JSON literal is a Python one). */
    static final String ANY_VALUE = "{}";

    /** The whitespace the tool parsers skip between tokens of a call. */
    static final String WS = "[ \\t\\n\\r]*";

    private NativeToolGrammar() {
    }

    /**
     * The forcing grammar of {@code tools} in the format's own syntax: one call of the single tool a named
     * {@code tool_choice} selects ({@code atMostOne} over one tool), or calls of the declared tools — as many
     * as the model writes when {@code parallel}, one when not. Null when the format has no native grammar.
     */
    static ConstraintSpec forcing(ToolParser.Factory format, List<ChatTool> tools, boolean parallel,
            boolean atMostOne) {
        return forcing(format, tools, parallel, atMostOne, Collections.<String, Object>emptyMap());
    }

    /**
     * The forcing grammar of {@code tools} in the format's own syntax, with the request's template variables
     * ({@code chat_template_kwargs}) that a format's grammar depends on — K2 Horizon's {@code tool_call_format}.
     */
    static ConstraintSpec forcing(ToolParser.Factory format, List<ChatTool> tools, boolean parallel,
            boolean atMostOne, Map<String, Object> templateVariables) {
        if (format instanceof AutoParserFormat) {
            return ((AutoParserFormat) format).forcing(tools, parallel);
        }
        Format f = format(format.name(), hunyuanSuffix(format), templateVariables);
        if (f == null || tools.isEmpty()) {
            return null;
        }
        Writer w = new Writer(format.name());
        String call = w.choice(f, tools);
        return ConstraintSpec.gbnf(w.withRoot(f.forcingRoot(w, call, tools, parallel, atMostOne)));
    }

    /**
     * The lazy grammar of {@code tools} in the format's own syntax behind {@code triggers}, the format's call
     * openers: the output runs free until an opener appears and is a call from there on. Null when the format
     * has no native grammar or no opener to trigger on.
     */
    static ConstraintSpec lazy(ToolParser.Factory format, List<ChatTool> tools, List<String> triggers) {
        return lazy(format, tools, triggers, Collections.<String, Object>emptyMap());
    }

    /** {@link #lazy(ToolParser.Factory, List, List)} with the request's template variables. */
    static ConstraintSpec lazy(ToolParser.Factory format, List<ChatTool> tools, List<String> triggers,
            Map<String, Object> templateVariables) {
        if (format instanceof AutoParserFormat) {
            return ((AutoParserFormat) format).lazy(tools, triggers);
        }
        Format f = format(format.name(), hunyuanSuffix(format), templateVariables);
        if (f == null || tools.isEmpty() || triggers == null || triggers.isEmpty()) {
            return null;
        }
        Writer w = new Writer(format.name());
        String call = w.choice(f, tools);
        return ConstraintSpec.gbnf(w.withRoot(f.root(w, call, tools, true, false)))
                .withTriggers(GrammarTriggers.lazyWords(triggers));
    }

    /**
     * The Hunyuan structural-token suffix a resolved {@code hy_v3} / {@code hy_v4} factory bakes into its
     * markers: the text between {@code <tool_calls} and the closing angle of the parser's opener, empty for the
     * other formats.
     */
    private static String hunyuanSuffix(ToolParser.Factory format) {
        if (!format.name().startsWith("hy_")) {
            return "";
        }
        for (String trigger : format.create(Collections.<ChatTool>emptyList()).triggers()) {
            if (trigger.startsWith("<tool_calls") && trigger.endsWith(">")) {
                return trigger.substring("<tool_calls".length(), trigger.length() - 1);
            }
        }
        return "";
    }

    // ------------------------------------------------------------------ formats

    /**
     * The builder of a format name under the request's template variables: the format's own syntax, or the layout
     * the K2 Horizon template selects through {@code tool_call_format} ({@code xml}, {@code xml_typed} or
     * {@code json}).
     *
     * @throws IllegalArgumentException when K2 Horizon's {@code tool_call_format} is not one of those values
     */
    private static Format format(String name, String hySuffix, Map<String, Object> templateVariables) {
        if (name == null) {
            return null;
        }
        switch (name.toLowerCase(Locale.ROOT)) {
            case "hermes": case "ernie45":
                return jsonObjectFormat("<tool_call>", "</tool_call>", "arguments", true);
            case "longcat":
                return jsonObjectFormat("<longcat_tool_call>", "</longcat_tool_call>", "arguments", true);
            case "internlm":
                return jsonObjectFormat("<|action_start|><|plugin|>", "<|action_end|>", "arguments", false);
            case "llama3_json":
                return llama3Json();
            case "jamba":
                return jsonArrayFormat("<tool_calls>", "</tool_calls>");
            case "granite":
                return jsonArrayFormat("<|tool_call|>", null);
            case "phi4_mini_json":
                return jsonArrayFormat("functools", null);
            case "xlam":
                return jsonArrayFormat(null, null);
            case "mistral":
                return jsonArrayFormat(MistralToolParser.TOOL_CALLS, null);
            case "qwen3_coder":
                return xmlFunctionFormat("<tool_call>", "</tool_call>", true, true);
            case "mimo":
                return mimoFormat();
            case "seed_oss":
                return xmlFunctionFormat("<seed:tool_call>", "</seed:tool_call>", false, false);
            case "minimax_m2":
                return xmlInvokeFormat("<minimax:tool_call>", "</minimax:tool_call>");
            case "dots":
                return xmlInvokeFormat("<dots_function_call>", "</dots_function_call>");
            case "glm45":
                return argKeyFormat();
            case "glm_4_7":
                return glm47Format();
            case "deepseek_v3":
                return nameArgsFormat("function<｜tool▁sep｜>", "\n```json\n", "\n```");
            case "deepseek_v31":
                return nameArgsFormat(null, "<｜tool▁sep｜>", null);
            case "kimi_k2":
                return kimiFormat();
            case "deepseek_v32":
                return dsmlFormat(Collections.singletonList("<｜DSML｜function_calls>"), "</｜DSML｜function_calls>",
                        "invoke", "deepseek_xml");
            case "deepseek_v4":
                return dsmlFormat(Arrays.asList("<｜DSML｜tool_calls>", "<｜DSML｜toolcalls>", "<｜DSML｜tool>"),
                        "</｜DSML｜tool_calls>", "invoke", "deepseek_xml");
            case "deepseek_v41":
                return dsmlFormat(Collections.singletonList("<｜DSML｜ calls>"), "</｜DSML｜ calls>", " invoke",
                        "deepseek_v4_1_xml");
            case "kimi_k3":
                return kimiK3Format();
            case "plamo3":
                return plamo3Format();
            case "hy_v3":
                return hyFormat(hySuffix, "<tool_sep" + hySuffix + ">");
            case "hy_v4":
                return hyFormat(hySuffix, null);
            case "pythonic":
                return pythonicFormat(null, null);
            case "llama4_pythonic":
                return pythonicFormat("<|python_start|>", "<|python_end|>");
            case "lfm2":
                return pythonicFormat("<|tool_call_start|>", "<|tool_call_end|>");
            case "olmo3":
                return olmo3Format();
            default:
                return NativeToolGrammarFormats.format(name, k2CallFormat(name, templateVariables));
        }
    }

    /**
     * The K2 Horizon call layout the template selects through its {@code tool_call_format} variable
     * (chat_template_kwargs): {@code xml} when unset, otherwise {@code xml}, {@code xml_typed} or {@code json}.
     *
     * @throws IllegalArgumentException naming the variable when its value is another one, or is no string
     */
    static String k2CallFormat(String name, Map<String, Object> templateVariables) {
        if (name == null || !"k2_horizon".equals(name.toLowerCase(Locale.ROOT))) {
            return "xml";
        }
        Object value = templateVariables == null ? null : templateVariables.get("tool_call_format");
        if (value == null || "auto".equals(value)) {
            return "xml";
        }
        if (value instanceof String) {
            String format = (String) value;
            if ("xml".equals(format) || "xml_typed".equals(format) || "json".equals(format)) {
                return format;
            }
        }
        throw new IllegalArgumentException("chat_template_kwargs.tool_call_format must be one of xml, xml_typed,"
                + " json; got " + value);
    }

    private static String prefix(String format) {
        // rule names must be safe before the composer sanitizes them: the exclusion rules claim theirs verbatim
        return "native-" + format.toLowerCase(Locale.ROOT).replaceAll("[^a-zA-Z0-9-]+", "-");
    }

    /** One format: the rule of one call unit and the root over the units. */
    interface Format {

        /** The rule name of one call unit of tool {@code index}, added through {@code w}. */
        String unit(Writer w, int index, ChatTool tool);

        /**
         * The root right-hand side over the unit rule: {@code atMostOne} holds the output to one call, else
         * {@code parallel} lets calls repeat — the whole output for a forcing choice, everything from a
         * trigger on for a lazy one.
         */
        String root(Writer w, String unit, List<ChatTool> tools, boolean parallel, boolean atMostOne);

        /** The root of a forcing choice, the whole output; {@link #root} unless the format writes it otherwise. */
        default String forcingRoot(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                                   boolean atMostOne) {
            return root(w, unit, tools, parallel, atMostOne);
        }
    }

    /** The rule and schema helpers shared by the formats; rule names carry the format prefix. */
    static final class Writer {

        final JsonSchemaToGbnf.Composer g = JsonSchemaToGbnf.composer(null);
        final String base;

        Writer(String format) {
            this.base = prefix(format);
        }

        /** The composed grammar with {@code root} set. */
        String withRoot(String root) {
            g.addRule("root", root);
            return g.gbnf();
        }

        /** The alternation rule over one unit per tool: the calls the choice admits. */
        String choice(Format f, List<ChatTool> tools) {
            List<String> units = new ArrayList<>();
            for (int i = 0; i < tools.size(); i++) {
                units.add(f.unit(this, i, tools.get(i)));
            }
            return units.size() == 1 ? units.get(0)
                    : g.addRule(base + "-choice", "( " + String.join(" | ", units) + " )");
        }

        /** The arguments rule of tool {@code index}: its schema when strict, else any JSON object. */
        String args(int index, ChatTool tool) {
            String name = base + "-tool-" + index + "-args";
            return strict(tool) ? g.addSchema(name, ToolRouting.write(ToolRouting.parameters(tool)))
                    : g.addSchema(name, ANY_OBJECT);
        }

        /** The value rule of parameter {@code key}: its schema when strict, else any JSON value. */
        String jsonValue(int index, int p, ChatTool tool, String key) {
            String name = base + "-tool-" + index + "-arg-" + p;
            Object prop = property(tool, key);
            return strict(tool) && prop instanceof Map ? g.addSchema(name, ToolRouting.write(prop))
                    : g.addSchema(name, ANY_VALUE);
        }

        /**
         * The raw text value of parameter {@code key}: its JSON schema when strict and the schema admits no
         * string (the parser reads such a value as JSON), else text that never writes {@code excludes} —
         * the parameter's closer and the tags that could start the next parameter or end the call.
         */
        String rawValue(int index, int p, ChatTool tool, String key, List<String> excludes) {
            String name = base + "-tool-" + index + "-arg-" + p;
            Object prop = property(tool, key);
            if (strict(tool) && prop instanceof Map && !ToolSchemas.textValuedSchema(prop)) {
                return g.addSchema(name, ToolRouting.write(prop));
            }
            return GbnfExclusion.excluding(g::addRule, name, excludes);
        }

        /** The tagged parameters block: the required properties in declared order, then any optional one. */
        String taggedParams(int index, ChatTool tool, String sep, Parameter parameter) {
            Map<String, Object> props = properties(tool);
            if (props.isEmpty()) {
                return "";
            }
            List<String> required = new ArrayList<>();
            List<String> optional = new ArrayList<>();
            int p = 0;
            for (Map.Entry<String, Object> e : props.entrySet()) {
                String rule = parameter.of(this, index, p, e.getKey());
                (required(e, tool) ? required : optional).add(rule);
                p++;
            }
            String gap = sep.isEmpty() ? "" : " " + sep + " ";
            StringBuilder out = new StringBuilder(String.join(gap, required));
            if (optional.isEmpty()) {
                return out.toString();
            }
            String alts = String.join(" | ", optional);
            String group = gap.isEmpty() ? "( " + alts + " )*"
                    : "( " + sep + " ( " + alts + " ) )*";
            if (required.isEmpty()) {
                // the first optional carries no leading separator, or "f(, k=v)" and its like would be forced
                if (out.length() > 0) {
                    out.append(" ");
                }
                return out.append(optional.get(0)).append(" ").append(group).toString();
            }
            if (out.length() > 0) {
                out.append(" ");
            }
            return out.append(group).toString();
        }

        /** One tagged parameter as the format writes it. */
        interface Parameter {

            /** The rule name of parameter {@code p} of tool {@code index}. */
            String of(Writer w, int index, int p, String key);
        }
    }

    /** Whether tool {@code tool} holds its arguments to its schema. */
    static boolean strict(ChatTool tool) {
        return Boolean.TRUE.equals(tool.strict());
    }

    /** The tool's properties map, empty when it declares none. */
    static Map<String, Object> properties(ChatTool tool) {
        Object props = ToolRouting.parameters(tool).get("properties");
        Map<String, Object> out = new LinkedHashMap<>();
        if (props instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) props).entrySet()) {
                out.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        return out;
    }

    /** The schema of one property, or null. */
    static Object property(ChatTool tool, String key) {
        return properties(tool).get(key);
    }

    /** Whether the property is required by the tool's schema. */
    static boolean required(Map.Entry<String, Object> property, ChatTool tool) {
        Object required = ToolRouting.parameters(tool).get("required");
        return required instanceof List && ((List<?>) required).contains(property.getKey());
    }

    // ------------------------------------------------------------------ JSON formats

    /**
     * The formats whose call is one JSON object of {@code name} and the arguments key after an opener and
     * before a closer (Hermes, Longcat, InternLM); {@code spaced} adds the newline-leading variant the
     * templates write (vLLM's two hermes tags).
     */
    private static Format jsonObjectFormat(String open, String close, String argsKey, boolean spaced) {
        return new Format() {
            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String lead = spaced ? "( " + HarmonyToolGrammar.literal("\n{\"name\": \"") + " | "
                        + HarmonyToolGrammar.literal("{\"name\": \"") + " )"
                        : HarmonyToolGrammar.literal("{\"name\": \"");
                String tail = spaced ? "( " + HarmonyToolGrammar.literal("}\n" + close) + " | "
                        + HarmonyToolGrammar.literal("}" + close) + " )"
                        : HarmonyToolGrammar.literal("}" + close);
                String args = w.args(index, tool);
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal(open) + " " + lead + " "
                                + HarmonyToolGrammar.literal(tool.name()) + " "
                                + HarmonyToolGrammar.literal("\", \"" + argsKey + "\": ") + " " + args + " "
                                + tail);
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return atMostOne ? unit : "( " + unit + " )+";
            }
        };
    }

    /**
     * Llama 3: {@code <|python_tag|>} then call objects with the {@code parameters} key, {@code ;} between
     * the calls of one block (the parser's object-run separators).
     */
    private static Format llama3Json() {
        return new Format() {
            private final String tag = "<|python_tag|>";

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String args = w.args(index, tool);
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal("{\"name\": \"") + " "
                                + HarmonyToolGrammar.literal(tool.name()) + " "
                                + HarmonyToolGrammar.literal("\", \"parameters\": ") + " " + args + " "
                                + HarmonyToolGrammar.literal("}"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String repeat = atMostOne ? ""
                        : " ( " + WS + " \";\" " + WS + " " + unit + " )*";
                String block = w.g.addRule(w.base + "-block",
                        HarmonyToolGrammar.literal(tag) + " " + WS + " " + unit + repeat);
                return atMostOne ? block : "( " + block + " )+";
            }
        };
    }

    /**
     * The formats whose calls form a JSON array after an opener (Jamba, Granite, Phi-4-mini, xLAM, Mistral);
     * a null {@code open} means a bare array.
     */
    private static Format jsonArrayFormat(String open, String close) {
        return new Format() {
            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String args = w.args(index, tool);
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal("{\"name\": \"") + " "
                                + HarmonyToolGrammar.literal(tool.name()) + " "
                                + HarmonyToolGrammar.literal("\", \"arguments\": ") + " " + args + " "
                                + HarmonyToolGrammar.literal("}"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String lead = open == null ? "" : HarmonyToolGrammar.literal(open) + " " + WS + " ";
                String tail = close == null ? "" : " " + WS + " " + HarmonyToolGrammar.literal(close);
                String list = atMostOne ? unit
                        : unit + " ( " + WS + " \",\" " + WS + " " + unit + " )*";
                return w.g.addRule(w.base + "-array", lead + HarmonyToolGrammar.literal("[") + " " + WS + " "
                        + list + " " + WS + " " + HarmonyToolGrammar.literal("]") + tail);
            }
        };
    }

    // ------------------------------------------------------------------ tagged-text formats

    /**
     * Qwen3-Coder and Seed-OSS: {@code <function=NAME>} with one {@code <parameter=KEY>} per declared
     * property; Qwen3-Coder trims one newline off each value and may open a call without the block opener,
     * so the lazy grammar continues from either opener.
     */
    private static Format xmlFunctionFormat(String wrapOpen, String wrapClose, boolean trim, boolean bare) {
        return new Format() {
            private final String sep = trim ? "\n" : "";

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String fn = fn(w, index, tool);
                return w.g.addRule(w.base + "-tool-" + index, HarmonyToolGrammar.literal(wrapOpen + sep) + " "
                        + fn + " " + HarmonyToolGrammar.literal(sep + wrapClose));
            }

            /** The bare {@code <function=NAME>} call, without the block wrapper. */
            private String fn(Writer w, int index, ChatTool tool) {
                List<String> excludes = Arrays.asList("</parameter>", "<parameter=", "</function>");
                String params = w.taggedParams(index, tool, sep, (writer, i, p, key) -> writer.g.addRule(
                        writer.base + "-tool-" + i + "-param-" + p,
                        HarmonyToolGrammar.literal("<parameter=" + key + ">" + sep) + " "
                                + writer.rawValue(i, p, tool, key, excludes)
                                + " " + HarmonyToolGrammar.literal(sep + "</parameter>")));
                return w.g.addRule(w.base + "-tool-" + index + "-fn",
                        HarmonyToolGrammar.literal("<function=" + tool.name() + ">" + sep) + " "
                                + (params.isEmpty() ? "" : params + " " + HarmonyToolGrammar.literal(sep) + " ")
                                + HarmonyToolGrammar.literal("</function>"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                if (atMostOne) {
                    return unit;
                }
                String root = "( " + unit + " )+";
                if (!bare) {
                    return root;
                }
                List<String> fns = new ArrayList<>();
                for (int i = 0; i < tools.size(); i++) {
                    fns.add(fn(w, i, tools.get(i)));
                }
                return w.g.addRule(w.base + "-root", "( " + unit + " | " + String.join(" | ", fns) + " )+");
            }
        };
    }

    /** MiniMax-M2 and Dots: one block of {@code <invoke name="NAME">} calls, one {@code <parameter name="KEY">} each. */
    private static Format xmlInvokeFormat(String blockOpen, String blockClose) {
        return new Format() {
            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                List<String> excludes = Arrays.asList("</parameter>", "<parameter name=", "</invoke>");
                String params = w.taggedParams(index, tool, "\n", (writer, i, p, key) -> writer.g.addRule(
                        writer.base + "-tool-" + i + "-param-" + p,
                        HarmonyToolGrammar.literal("<parameter name=\"" + key + "\">") + " "
                                + writer.rawValue(i, p, tool, key, excludes)
                                + " " + HarmonyToolGrammar.literal("</parameter>")));
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal("<invoke name=\"" + tool.name() + "\">\n") + " "
                                + params + (params.isEmpty() ? "" : " " + HarmonyToolGrammar.literal("\n") + " ")
                                + HarmonyToolGrammar.literal("</invoke>"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String block = w.g.addRule(w.base + "-block",
                        HarmonyToolGrammar.literal(blockOpen + "\n") + " " + "( " + unit + " "
                                + HarmonyToolGrammar.literal("\n") + " )* " + unit + " "
                                + HarmonyToolGrammar.literal("\n" + blockClose));
                return atMostOne ? block : "( " + block + " )+";
            }
        };
    }

    /**
     * MiMo, xgrammar's builtin {@code mimo} tag: {@code <tool_call><function=NAME>} with the tool's schema as
     * {@code qwen_xml} parameters ({@link JsonSchemaToGbnf.Composer#addXmlParameters}; any parameters when the tool
     * is not strict) and {@code </function></tool_call>}, no newlines around the wrapper tags. The parser also opens
     * a bare {@code <function=NAME>}, so the lazy grammar continues from either.
     */
    private static Format mimoFormat() {
        return new Format() {
            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                return w.g.addRule(w.base + "-tool-" + index, HarmonyToolGrammar.literal("<tool_call>") + " "
                        + fn(w, index, tool) + " " + HarmonyToolGrammar.literal("</tool_call>"));
            }

            private String fn(Writer w, int index, ChatTool tool) {
                String schema = strict(tool) ? ToolRouting.write(ToolRouting.parameters(tool)) : ANY_VALUE;
                String params = w.g.addXmlParameters(w.base + "-tool-" + index + "-params", schema, "qwen_xml");
                return w.g.addRule(w.base + "-tool-" + index + "-fn",
                        HarmonyToolGrammar.literal("<function=" + tool.name() + ">") + " " + params + " "
                                + HarmonyToolGrammar.literal("</function>"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                List<String> fns = new ArrayList<>();
                for (int i = 0; i < tools.size(); i++) {
                    fns.add(fn(w, i, tools.get(i)));
                }
                return "( " + unit + " | " + String.join(" | ", fns) + " )+";
            }

            @Override
            public String forcingRoot(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                                      boolean atMostOne) {
                return atMostOne ? unit : "( " + unit + " )+";
            }
        };
    }

    /**
     * PLaMo 3, vLLM's builtin {@code plamo3} tag: one request per call — the request opener, the tool name
     * between its markers, the arguments as JSON under the constrain chain — inside a requests block that ends
     * with the block closer alone or followed by the end-of-turn tag. A forcing choice holds the output to one
     * block ({@code at_least_one} over the calls); from a lazy trigger the output is blocks or bare requests,
     * the parser's other trigger.
     */
    private static Format plamo3Format() {
        return new Format() {
            private static final String REQUESTS_OPEN = "<|plamo:begin_tool_requests:plamo|>";
            private static final String REQUESTS_CLOSE = "<|plamo:end_tool_requests:plamo|>";
            private static final String REQUEST_OPEN = "<|plamo:begin_tool_request:plamo|>";
            private static final String NAME_OPEN = "<|plamo:begin_tool_name:plamo|>";
            private static final String NAME_CLOSE = "<|plamo:end_tool_name:plamo|>";
            private static final String ARGS_OPEN =
                    "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>";
            private static final String REQUEST_CLOSE =
                    "<|plamo:end_tool_arguments:plamo|><|plamo:end_tool_request:plamo|>";
            private static final String EOT = "<|plamo:tag|>";

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String args = w.args(index, tool);
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal(REQUEST_OPEN + NAME_OPEN + tool.name() + NAME_CLOSE
                                + ARGS_OPEN) + " " + args + " " + HarmonyToolGrammar.literal(REQUEST_CLOSE));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel, boolean atMostOne) {
                String bare = w.g.addRule(w.base + "-bare", "( " + unit + " "
                        + HarmonyToolGrammar.literal(EOT) + " | " + unit + " )");
                return "( " + block(w, unit, parallel, atMostOne) + " | " + bare + " )+";
            }

            @Override
            public String forcingRoot(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                                      boolean atMostOne) {
                return block(w, unit, parallel, atMostOne);
            }

            /** The requests block: the opener, the calls, and the closer with its end-of-turn variant. */
            private String block(Writer w, String unit, boolean parallel, boolean atMostOne) {
                String calls = atMostOne ? unit : "( " + unit + " )+";
                return w.g.addRule(w.base + "-block", HarmonyToolGrammar.literal(REQUESTS_OPEN) + " " + calls
                        + " ( " + HarmonyToolGrammar.literal(REQUESTS_CLOSE) + " | "
                        + HarmonyToolGrammar.literal(REQUESTS_CLOSE + EOT) + " )");
            }
        };
    }

    /**
     * HY v3 and HY v4, vLLM's builtin {@code hy_v4} tag and the v3 layout it extends: {@code <tool_call{S}>NAME}
     * with one {@code <arg_key{S}>K</arg_key{S}><arg_value{S}>V</arg_value{S}> pair per argument, the required keys
     * in declared order and the optional ones after them in any order; values stay free text, as vLLM's builder
     * writes them. HY v3 writes {@code nameSep} (the {@code <tool_sep{S}>} token) between the name and the first
     * pair. Adjacent calls carry no separator. The suffix {@code S} is the vocabulary's structural-token suffix,
     * and a bare {@code <tool_call{S}>} opens a call too under {@code auto}.
     */
    private static Format hyFormat(String suffix, String nameSep) {
        return new Format() {
            private final String callsOpen = "<tool_calls" + suffix + ">";
            private final String callsClose = "</tool_calls" + suffix + ">";
            private final String callOpen = "<tool_call" + suffix + ">";
            private final String callClose = "</tool_call" + suffix + ">";
            private final String keyOpen = "<arg_key" + suffix + ">";
            private final String keyClose = "</arg_key" + suffix + ">";
            private final String valueOpen = "<arg_value" + suffix + ">";
            private final String valueClose = "</arg_value" + suffix + ">";
            private final String sep = nameSep == null ? "" : HarmonyToolGrammar.literal(nameSep);

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                Map<String, Object> props = properties(tool);
                List<String> requiredKeys = requiredKeys(tool, props);
                StringBuilder params = new StringBuilder();
                int p = 0;
                for (String key : requiredKeys) {
                    if (params.length() > 0) {
                        params.append(' ');
                    }
                    params.append(pair(w, index, p++, tool, key));
                }
                List<String> optional = new ArrayList<>();
                for (String key : props.keySet()) {
                    if (!requiredKeys.contains(key)) {
                        optional.add(pair(w, index, p++, tool, key));
                    }
                }
                if (!optional.isEmpty()) {
                    String alts = String.join(" | ", optional);
                    String group = "( " + (optional.size() == 1 ? alts : "( " + alts + " )") + " )*";
                    params.append(params.length() > 0 ? " " : "").append(group);
                }
                String head = HarmonyToolGrammar.literal(callOpen + tool.name()) + " "
                        + (nameSep == null ? "" : sep + " ");
                if (params.length() == 0) {
                    // vLLM leaves the content of a call with no arguments free: any text but the closer
                    String free = GbnfExclusion.excluding(w.g::addRule, w.base + "-tool-" + index + "-free",
                            Collections.singletonList(callClose));
                    return w.g.addRule(w.base + "-tool-" + index, head + free + " "
                            + HarmonyToolGrammar.literal(callClose));
                }
                return w.g.addRule(w.base + "-tool-" + index, head + params + " "
                        + HarmonyToolGrammar.literal(callClose));
            }

            /** One {@code arg_key}/{@code arg_value} pair; the value stays free text the parser can cut out of. */
            private String pair(Writer w, int index, int p, ChatTool tool, String key) {
                String free = GbnfExclusion.excluding(w.g::addRule, w.base + "-tool-" + index + "-arg-" + p,
                        Arrays.asList(keyOpen, valueClose, callClose));
                return w.g.addRule(w.base + "-tool-" + index + "-pair-" + p,
                        HarmonyToolGrammar.literal(keyOpen + key + keyClose + valueOpen) + " " + free + " "
                                + HarmonyToolGrammar.literal(valueClose));
            }

            /** The required keys: the declared ones that are required, then any required name without a property. */
            private List<String> requiredKeys(ChatTool tool, Map<String, Object> props) {
                List<String> out = new ArrayList<>();
                Object required = ToolRouting.parameters(tool).get("required");
                List<?> names = required instanceof List ? (List<?>) required : Collections.emptyList();
                for (String key : props.keySet()) {
                    if (names.contains(key)) {
                        out.add(key);
                    }
                }
                for (Object name : names) {
                    String key = String.valueOf(name);
                    if (!props.containsKey(key) && !out.contains(key)) {
                        out.add(key);
                    }
                }
                return out;
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel, boolean atMostOne) {
                String section = w.g.addRule(w.base + "-section", HarmonyToolGrammar.literal(callsOpen) + " ( "
                        + unit + " )+ " + HarmonyToolGrammar.literal(callsClose));
                return "( " + section + " | " + unit + " )+";
            }

            @Override
            public String forcingRoot(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                                      boolean atMostOne) {
                String calls = atMostOne ? unit : "( " + unit + " )+";
                return w.g.addRule(w.base + "-section", HarmonyToolGrammar.literal(callsOpen) + " " + calls + " "
                        + HarmonyToolGrammar.literal(callsClose));
            }
        };
    }

    /**
     * Kimi K3, vLLM's builtin {@code kimi_k3} tag: XTML channels. A forcing choice holds the whole output to
     * the response channel (its opener optional, its text holding none of the reserved markers), the tools
     * channel of calls — each {@code <|open|>call tool="N" index="<digits>"} with one {@code argument} block
     * per strict argument, string values raw (an enum pinned, a bounded string by length, the rest free but the
     * markers) and any other type JSON per its property schema — and the optional message closer. Under
     * {@code auto} the grammar runs from the tools opener on, the trigger the parser reports. A loose tool's
     * arguments stay permissive: any attributes, any value.
     */
    private static Format kimiK3Format() {
        return new Format() {
            private static final String OPEN = "<|open|>";
            private static final String CLOSE = "<|close|>";
            private static final String SEP = "<|sep|>";
            private static final String RESPONSE_OPEN = OPEN + "response" + SEP;
            private static final String RESPONSE_CLOSE = CLOSE + "response" + SEP;
            private static final String TOOLS_OPEN = OPEN + "tools" + SEP;
            private static final String TOOLS_CLOSE = CLOSE + "tools" + SEP;
            private static final String CALL_CLOSE = CLOSE + "call" + SEP;
            private static final String ARG_CLOSE = CLOSE + "argument" + SEP;
            private static final String MESSAGE_CLOSE = CLOSE + "message" + SEP;
            private static final String END_OF_MSG = "<|end_of_msg|>";

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                return callTag(w, index, tool, 0);
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel, boolean atMostOne) {
                String channel = w.g.addRule(w.base + "-channel", HarmonyToolGrammar.literal(TOOLS_OPEN) + " ( "
                        + unit + " )+ " + HarmonyToolGrammar.literal(TOOLS_CLOSE) + " " + trailer(w));
                return channel;
            }

            @Override
            public String forcingRoot(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                                      boolean atMostOne) {
                String calls = atMostOne || !parallel ? unit : "( " + unit + " )+";
                String channel = w.g.addRule(w.base + "-channel", HarmonyToolGrammar.literal(TOOLS_OPEN) + " "
                        + calls + " " + HarmonyToolGrammar.literal(TOOLS_CLOSE));
                return w.g.addRule(w.base + "-root", responsePrefix(w) + " " + channel + " " + trailer(w));
            }

            /** The response channel: an optional opener, text with no reserved marker, the closer. */
            private String responsePrefix(Writer w) {
                String text = GbnfExclusion.excluding(w.g::addRule, w.base + "-response",
                        Arrays.asList(OPEN, CLOSE, END_OF_MSG));
                return w.g.addRule(w.base + "-response-prefix", "( "
                        + HarmonyToolGrammar.literal(RESPONSE_OPEN) + " | \"\" ) " + text + " "
                        + HarmonyToolGrammar.literal(RESPONSE_CLOSE));
            }

            /** The model's natural turn end, which the tag permits so it never masks the terminator. */
            private String trailer(Writer w) {
                return "( " + HarmonyToolGrammar.literal(MESSAGE_CLOSE) + " | \"\" )";
            }

            /** One {@code call} tag: the tool name and index in the opener, then the argument blocks. */
            private String callTag(Writer w, int index, ChatTool tool, int argBase) {
                String arguments = argumentsBlock(w, index, tool, argBase);
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal(OPEN + "call tool=\"" + escapeAttr(tool.name())
                                + "\" index=\"") + " [0-9]+ "
                                + HarmonyToolGrammar.literal("\"" + SEP) + " " + arguments + " "
                                + HarmonyToolGrammar.literal(CALL_CLOSE));
            }

            /** The arguments of one call: one argument tag per declared property, or permissive blocks. */
            private String argumentsBlock(Writer w, int index, ChatTool tool, int argBase) {
                Map<String, Object> parameters = ToolRouting.parameters(tool);
                Map<String, Object> props = properties(tool);
                if (!strict(tool) || props.isEmpty()) {
                    return w.g.addRule(w.base + "-tool-" + index + "-args",
                            "( " + permissiveTag(w, index, argBase) + " )*");
                }
                Map<String, Object> rootDefs = rootDefs(parameters);
                List<String> tags = new ArrayList<>();
                int p = 0;
                for (Map.Entry<String, Object> e : props.entrySet()) {
                    if (!(e.getValue() instanceof Map)) {
                        tags.add(permissiveTag(w, index, argBase + p));
                    } else {
                        String tag = argumentTag(w, index, argBase + p, e.getKey(),
                                (Map<?, ?>) e.getValue(), rootDefs);
                        tags.add(tag == null ? permissiveTag(w, index, argBase + p) : tag);
                    }
                    p++;
                }
                String inner = tags.size() == 1 ? tags.get(0) : "( " + String.join(" | ", tags) + " )";
                boolean anyRequired = required(parameters);
                return w.g.addRule(w.base + "-tool-" + index + "-args",
                        "( " + inner + " " + (anyRequired ? ")+" : ")*"));
            }

            /**
             * One {@code argument} tag for a property: the value raw for {@code string} (an enum pinned as
             * constant strings, a length-bounded string by its bound, the rest free but the close marker) and
             * JSON per the property schema otherwise; null when the type is unknown or a union.
             */
            private String argumentTag(Writer w, int index, int p, String key, Map<?, ?> prop,
                    Map<String, Object> rootDefs) {
                Object type = prop.get("type");
                String xtml = type instanceof String ? xtmlType((String) type) : null;
                if (xtml == null) {
                    return null;
                }
                String begin = OPEN + "argument key=\"" + escapeAttr(key) + "\" type=\"" + xtml + "\"" + SEP;
                String content = "string".equals(xtml)
                        ? stringContent(w, index, p, prop)
                        : w.g.addSchema(w.base + "-tool-" + index + "-arg-" + p + "-value",
                                ToolRouting.write(embedded(prop, rootDefs)));
                return w.g.addRule(w.base + "-tool-" + index + "-arg-" + p,
                        HarmonyToolGrammar.literal(begin) + " " + content + " "
                                + HarmonyToolGrammar.literal(ARG_CLOSE));
            }

            /** The XTML type of a JSON schema type, or null; the integer collapses onto number. */
            private String xtmlType(String jsonType) {
                switch (jsonType) {
                    case "string": return "string";
                    case "integer": case "number": return "number";
                    case "boolean": return "boolean";
                    case "null": return "null";
                    case "object": return "object";
                    case "array": return "array";
                    default: return null;
                }
            }

            /** The content of a string argument: its enum, its length bound, or free text but the close marker. */
            private String stringContent(Writer w, int index, int p, Map<?, ?> prop) {
                List<String> values = stringEnum(prop);
                if (!values.isEmpty()) {
                    List<String> literals = new ArrayList<>(values.size());
                    for (String value : values) {
                        literals.add(HarmonyToolGrammar.literal(value));
                    }
                    return w.g.addRule(w.base + "-tool-" + index + "-arg-" + p + "-enum",
                            literals.size() == 1 ? literals.get(0)
                                    : "( " + String.join(" | ", literals) + " )");
                }
                int[] bound = stringBound(prop);
                if (bound != null) {
                    return w.g.addRule(w.base + "-tool-" + index + "-arg-" + p + "-bounded",
                            "( [^<] | \"<\" [^|] ){" + (bound[0] == bound[1] ? String.valueOf(bound[0])
                                    : bound[0] + "," + bound[1]) + "}");
                }
                return GbnfExclusion.excluding(w.g::addRule, w.base + "-tool-" + index + "-arg-" + p + "-free",
                        Collections.singletonList(CLOSE));
            }

            /** The enum or const of a string property, when they are plain strings without the marker head. */
            private List<String> stringEnum(Map<?, ?> prop) {
                List<?> raw = prop.containsKey("enum") && prop.get("enum") instanceof List
                        ? (List<?>) prop.get("enum")
                        : prop.get("const") instanceof String
                        ? Collections.singletonList(prop.get("const")) : null;
                if (raw == null || raw.isEmpty() || raw.size() > 256) {
                    return Collections.emptyList();
                }
                for (Object value : raw) {
                    if (!(value instanceof String) || ((String) value).contains("<|")) {
                        return Collections.emptyList();
                    }
                }
                List<String> out = new ArrayList<>(raw.size());
                for (Object value : raw) {
                    out.add((String) value);
                }
                return out;
            }

            /** The {@code [minLength, maxLength]} of a string property, or null when unbounded or over 4096. */
            private int[] stringBound(Map<?, ?> prop) {
                Object maxRaw = prop.get("maxLength");
                if (!(maxRaw instanceof Number)) {
                    return null;
                }
                int max = ((Number) maxRaw).intValue();
                if (max < 0 || max > 4096 || (double) max != ((Number) maxRaw).doubleValue()) {
                    return null;
                }
                Object minRaw = prop.get("minLength");
                int min = minRaw instanceof Number ? ((Number) minRaw).intValue() : 0;
                if (min < 0 || min > max || (minRaw instanceof Number
                        && (double) min != ((Number) minRaw).doubleValue())) {
                    min = 0;
                }
                return new int[]{min, max};
            }

            /** A key-agnostic {@code argument} tag: any attributes, then any value but the close marker. */
            private String permissiveTag(Writer w, int index, int p) {
                String free = GbnfExclusion.excluding(w.g::addRule, w.base + "-tool-" + index + "-arg-" + p
                        + "-free", Collections.singletonList(CLOSE));
                return w.g.addRule(w.base + "-tool-" + index + "-arg-" + p, "[^<]* "
                        + HarmonyToolGrammar.literal(SEP) + " " + free + " "
                        + HarmonyToolGrammar.literal(ARG_CLOSE));
            }

            /** The root {@code $defs} and {@code definitions} of the parameters, for embedded property schemas. */
            private Map<String, Object> rootDefs(Map<String, Object> parameters) {
                Map<String, Object> out = new LinkedHashMap<>();
                for (String key : Arrays.asList("$defs", "definitions")) {
                    if (parameters.get(key) instanceof Map) {
                        out.put(key, parameters.get(key));
                    }
                }

                return out;
            }

            /** A property schema with the root definition tables re-attached, so its references resolve. */
            private Map<String, Object> embedded(Map<?, ?> prop, Map<String, Object> rootDefs) {
                Map<String, Object> out = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : prop.entrySet()) {
                    out.put(String.valueOf(e.getKey()), e.getValue());
                }
                for (Map.Entry<String, Object> e : rootDefs.entrySet()) {
                    out.putIfAbsent(e.getKey(), e.getValue());
                }
                return out;
            }

            /** Whether the parameters name at least one required property. */
            private boolean required(Map<String, Object> parameters) {
                Object required = parameters.get("required");
                return required instanceof List && !((List<?>) required).isEmpty();
            }
        };

    }

    /** An XTML attribute value: {@code &} escapes first, then quotes, as the format writes them. */
    private static String escapeAttr(String value) {
        return value.replace("&", "&amp;").replace("\"", "&quot;");
    }

    /**
     * GLM 4.7, vLLM's builtin {@code glm_4_7} tag: {@code <tool_call>NAME} with one
     * {@code <arg_key>K</arg_key><arg_value>V</arg_value>} pair per argument. A strict tool's pairs are its
     * schema in the {@code glm_xml} style, as the xgrammar builtin writes them; a loose tool's keys and scalar
     * value types stay pinned while its other values stay free text, and a loose tool without plain properties
     * keeps free {@code glm_xml} parameters.
     */
    private static Format glm47Format() {
        return new Format() {
            private static final String CALL_OPEN = "<tool_call>";
            private static final String CALL_CLOSE = "</tool_call>";
            /** The strings a free value never writes, vLLM's {@code _GLM_4_7_VALUE_EXCLUDES}. */
            private final List<String> valueExcludes = Arrays.asList("<think>", "</think>", "<tool_call>",
                    "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>");
            /** The JSON types pinned as bare JSON; other values stay raw text. */
            private final List<String> jsonScalars = Arrays.asList("integer", "number", "boolean", "null");

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                Map<String, Object> props = properties(tool);
                if (strict(tool) || props.isEmpty() || !plainParameters(tool)) {
                    String schema = strict(tool) ? ToolRouting.write(ToolRouting.parameters(tool)) : "{}";
                    String params = w.g.addXmlParameters(w.base + "-tool-" + index + "-params", schema, "glm_xml");
                    return w.g.addRule(w.base + "-tool-" + index,
                            HarmonyToolGrammar.literal(CALL_OPEN + tool.name()) + " " + params + " "
                                    + HarmonyToolGrammar.literal(CALL_CLOSE));
                }
                StringBuilder pairs = new StringBuilder();
                int p = 0;
                for (Map.Entry<String, Object> e : props.entrySet()) {
                    if (!(e.getValue() instanceof Map)) {
                        continue;
                    }
                    if (pairs.length() > 0) {
                        pairs.append(" | ");
                    }
                    pairs.append(pair(w, index, p, e.getKey(), (Map<?, ?>) e.getValue()));
                    p++;
                }                if (pairs.length() == 0) {
                    return w.g.addRule(w.base + "-tool-" + index, HarmonyToolGrammar.literal(CALL_OPEN
                            + tool.name()) + " " + HarmonyToolGrammar.literal(CALL_CLOSE));
                }
                String inner = p == 1 ? pairs.toString() : "( " + pairs + " )";
                return w.g.addRule(w.base + "-tool-" + index, HarmonyToolGrammar.literal(CALL_OPEN
                        + tool.name()) + " ( " + inner + " )* " + HarmonyToolGrammar.literal(CALL_CLOSE));
            }

            /** One pair: the key pinned, its value an enum, a bare JSON scalar, or free text but the markers. */
            private String pair(Writer w, int index, int p, String key, Map<?, ?> prop) {
                String value = valueFormat(w, index, p, prop);
                return w.g.addRule(w.base + "-tool-" + index + "-pair-" + p,
                        HarmonyToolGrammar.literal("<arg_key>" + key + "</arg_key><arg_value>") + " " + value
                                + " " + HarmonyToolGrammar.literal("</arg_value>"));
            }

            /** The value of one loose argument: its enum, its scalar JSON type, or free text but the markers. */
            private String valueFormat(Writer w, int index, int p, Map<?, ?> prop) {
                if (prop.containsKey("enum")) {
                    List<String> literals = new ArrayList<>();
                    for (Object value : (List<?>) prop.get("enum")) {
                        literals.add(HarmonyToolGrammar.literal(
                                value instanceof String ? (String) value : ToolRouting.write(value)));
                    }
                    return w.g.addRule(w.base + "-tool-" + index + "-value-" + p, literals.size() == 1
                            ? literals.get(0) : "( " + String.join(" | ", literals) + " )");
                }
                Object type = prop.get("type");
                List<String> types = new ArrayList<>();
                if (type instanceof List) {
                    for (Object t : (List<?>) type) {
                        types.add(String.valueOf(t));
                    }
                } else if (type != null) {
                    types.add(String.valueOf(type));
                }
                boolean scalar = !types.isEmpty();
                for (String t : types) {
                    scalar &= jsonScalars.contains(t);
                }
                if (scalar) {
                    List<String> branches = new ArrayList<>(types.size());
                    for (String t : types) {
                        Map<String, Object> schema = new LinkedHashMap<>();
                        schema.put("type", t);
                        branches.add(w.g.addSchema(w.base + "-tool-" + index + "-value-" + p,
                                ToolRouting.write(schema)));
                    }
                    return branches.size() == 1 ? branches.get(0)
                            : w.g.addRule(w.base + "-tool-" + index + "-value-" + p + "-alt",
                                    "( " + String.join(" | ", branches) + " )");
                }
                return GbnfExclusion.excluding(w.g::addRule, w.base + "-tool-" + index + "-value-" + p,
                        valueExcludes);
            }

            /** Whether the parameters are a plain object schema: declared properties and no composition keys. */
            private boolean plainParameters(ChatTool tool) {
                Map<String, Object> parameters = ToolRouting.parameters(tool);
                if (Boolean.TRUE.equals(parameters.get("additionalProperties"))) {
                    return false;
                }
                for (String key : Arrays.asList("$ref", "allOf", "anyOf", "oneOf", "patternProperties")) {
                    if (parameters.containsKey(key)) {
                        return false;
                    }
                }
                return true;
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel, boolean atMostOne) {
                return "( " + unit + " )+";
            }

            @Override
            public String forcingRoot(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                                      boolean atMostOne) {
                return atMostOne ? unit : "( " + unit + " )+";
            }
        };
    }

    /** GLM 4.5-4.7: {@code <tool_call>NAME} with one {@code <arg_key>KEY</arg_key><arg_value>V</arg_value>} pair. */
    private static Format argKeyFormat() {
        return new Format() {
            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                List<String> excludes = Arrays.asList("<arg_key>", "</arg_value>", "</tool_call>");
                String params = w.taggedParams(index, tool, "", (writer, i, p, key) -> writer.g.addRule(
                        writer.base + "-tool-" + i + "-param-" + p,
                        HarmonyToolGrammar.literal("<arg_key>" + key + "</arg_key>\n<arg_value>") + " "
                                + writer.rawValue(i, p, tool, key, excludes)
                                + " " + HarmonyToolGrammar.literal("</arg_value>\n")));
                String head = HarmonyToolGrammar.literal("<tool_call>" + tool.name());
                return w.g.addRule(w.base + "-tool-" + index, params.isEmpty()
                        ? head + " " + HarmonyToolGrammar.literal("</tool_call>")
                        : head + " " + HarmonyToolGrammar.literal("\n") + " " + params + " "
                        + HarmonyToolGrammar.literal("</tool_call>"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                return atMostOne ? unit : "( " + unit + " )+";
            }
        };
    }

    /**
     * DeepSeek V3.2 / V4 / V4.1 DSML, xgrammar's builtin {@code deepseek_v3_2} / {@code deepseek_v4} /
     * {@code deepseek_v4_1} tags: an {@code <｜DSML｜invoke name="f">} per call, its parameters the tool's schema in
     * the {@code style} XML parameters ({@link JsonSchemaToGbnf.Composer#addXmlParameters}; any parameters when the
     * tool is not strict), inside a calls section. A forcing choice is xgrammar's: a blank line, then one section
     * of one call, or of one or more under parallel calls. From a lazy trigger on, the output is sections opened
     * by any of the parser's {@code openers}, or bare invokes, the parser's other trigger.
     */
    private static Format dsmlFormat(List<String> openers, String close, String invoke, String style) {
        return new Format() {
            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String schema = strict(tool) ? ToolRouting.write(ToolRouting.parameters(tool)) : ANY_VALUE;
                String params = w.g.addXmlParameters(w.base + "-tool-" + index + "-params", schema, style);
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal("<｜DSML｜" + invoke + " name=\"" + tool.name() + "\">\n") + " "
                                + params + " " + HarmonyToolGrammar.literal("</｜DSML｜" + invoke + ">\n"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                List<String> opens = new ArrayList<>();
                for (String opener : openers) {
                    opens.add(HarmonyToolGrammar.literal(opener + "\n"));
                }
                String section = w.g.addRule(w.base + "-section", "( " + String.join(" | ", opens) + " ) ( "
                        + unit + " )+ " + HarmonyToolGrammar.literal(close));
                return "( " + section + " | " + unit + " )+";
            }

            @Override
            public String forcingRoot(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                                      boolean atMostOne) {
                String calls = atMostOne || !parallel ? unit : "( " + unit + " )+";
                return HarmonyToolGrammar.literal("\n\n" + openers.get(0) + "\n") + " " + calls + " "
                        + HarmonyToolGrammar.literal(close);
            }
        };
    }

    // ------------------------------------------------------------------ delimited name-and-JSON formats

    /** DeepSeek V3/V3.1: a section of calls, the name then the JSON arguments under {@code pre}/{@code post}. */
    private static Format nameArgsFormat(String typePrefix, String pre, String post) {
        return new Format() {
            private static final String SECTION = "<｜tool▁calls▁begin｜>";
            private static final String SECTION_END = "<｜tool▁calls▁end｜>";
            private static final String CALL = "<｜tool▁call▁begin｜>";
            private static final String CALL_END = "<｜tool▁call▁end｜>";

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String args = w.args(index, tool);
                String type = typePrefix == null ? "" : HarmonyToolGrammar.literal(typePrefix) + " ";
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal(CALL) + " " + type
                                + HarmonyToolGrammar.literal(tool.name()) + " "
                                + HarmonyToolGrammar.literal(pre) + " " + args + " "
                                + (post == null ? "" : HarmonyToolGrammar.literal(post) + " ")
                                + HarmonyToolGrammar.literal(CALL_END));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                if (atMostOne) {
                    return w.g.addRule(w.base + "-section", HarmonyToolGrammar.literal(SECTION) + " " + WS + " "
                            + unit + " " + WS + " " + HarmonyToolGrammar.literal(SECTION_END));
                }
                String section = w.g.addRule(w.base + "-section", HarmonyToolGrammar.literal(SECTION) + " "
                        + WS + " " + "( " + unit + " " + WS + " )+ " + HarmonyToolGrammar.literal(SECTION_END));
                return w.g.addRule(w.base + "-root", "( " + section + " | " + unit + " )+");
            }
        };
    }

    /** Kimi K2: a section of calls addressed {@code functions.NAME:INDEX}, the JSON arguments after the separator. */
    private static Format kimiFormat() {
        return new Format() {
            private static final String SECTION = "<|tool_calls_section_begin|>";
            private static final String SECTION_END = "<|tool_calls_section_end|>";
            private static final String CALL = "<|tool_call_begin|>";
            private static final String CALL_END = "<|tool_call_end|>";
            private static final String ARGS = "<|tool_call_argument_begin|>";

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String args = w.args(index, tool);
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal(CALL + "functions." + tool.name() + ":0" + ARGS) + " "
                                + args + " " + HarmonyToolGrammar.literal(CALL_END));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                if (atMostOne) {
                    return w.g.addRule(w.base + "-section", HarmonyToolGrammar.literal(SECTION) + " " + WS + " "
                            + unit + " " + WS + " " + HarmonyToolGrammar.literal(SECTION_END));
                }
                String section = w.g.addRule(w.base + "-section", HarmonyToolGrammar.literal(SECTION) + " "
                        + WS + " " + "( " + unit + " " + WS + " )+ " + HarmonyToolGrammar.literal(SECTION_END));
                return w.g.addRule(w.base + "-root", "( " + section + " | " + unit + " )+");
            }
        };
    }

    // ------------------------------------------------------------------ pythonic formats

    /**
     * The pythonic list {@code [f(k=v)]}, wrapped in the format's tags (Llama 4, LFM2) or bare (pythonic);
     * the values are JSON literals, which the parser reads as Python ones.
     */
    private static Format pythonicFormat(String open, String close) {
        return new Format() {
            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String params = w.taggedParams(index, tool, ",", (writer, i, p, key) -> writer.g.addRule(
                        writer.base + "-tool-" + i + "-param-" + p,
                        HarmonyToolGrammar.literal(key + "=") + " "
                                + writer.jsonValue(i, p, tool, key)));
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal(tool.name() + "(") + " " + params + " "
                                + HarmonyToolGrammar.literal(")"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String list = atMostOne ? unit
                        : unit + " ( " + WS + " \",\" " + WS + " " + unit + " )*";
                String lead = open == null ? "" : HarmonyToolGrammar.literal(open) + " " + WS + " ";
                String tail = open == null ? "" : " " + WS + " " + HarmonyToolGrammar.literal(close);
                return w.g.addRule(w.base + "-list", lead + HarmonyToolGrammar.literal("[") + " " + WS + " "
                        + list + " " + WS + " " + HarmonyToolGrammar.literal("]") + tail);
            }
        };
    }

    /** OLMo 3: one pythonic call per line inside {@code <function_calls>}. */
    private static Format olmo3Format() {
        return new Format() {
            private static final String OPEN = "<function_calls>";
            private static final String CLOSE = "</function_calls>";

            @Override
            public String unit(Writer w, int index, ChatTool tool) {
                String params = w.taggedParams(index, tool, ",", (writer, i, p, key) -> writer.g.addRule(
                        writer.base + "-tool-" + i + "-param-" + p,
                        HarmonyToolGrammar.literal(key + "=") + " "
                                + writer.jsonValue(i, p, tool, key)));
                return w.g.addRule(w.base + "-tool-" + index,
                        HarmonyToolGrammar.literal(tool.name() + "(") + " " + params + " "
                                + HarmonyToolGrammar.literal(")"));
            }

            @Override
            public String root(Writer w, String unit, List<ChatTool> tools, boolean parallel,
                               boolean atMostOne) {
                String calls = atMostOne ? unit
                        : unit + " ( " + HarmonyToolGrammar.literal("\n") + " " + unit + " )*";
                String block = w.g.addRule(w.base + "-block", HarmonyToolGrammar.literal(OPEN) + " " + WS + " "
                        + calls + " " + WS + " " + HarmonyToolGrammar.literal(CLOSE));
                return atMostOne ? block : "( " + block + " )+";
            }
        };
    }
}
