package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserFormat;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.plugin.Plugins;
import com.eonmux.neuralfree.gguf.plugin.ToolParserProvider;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

/**
 * The tool-call parser registry, serving the parser names of vLLM's {@code --tool-call-parser} and the
 * {@code neuralfree.tool.parser} property: the built-in formats below, plus the factories plugins contribute through
 * {@link ToolParserProvider ToolParserProvider&lt;ToolParser.Factory&gt;}, merged in provider-class order. A plugin
 * name colliding with a built-in name or with another plugin's name fails the table build with both sources named:
 * providers never replace built-ins.
 *
 * <p>{@link #detect} picks a format from the chat template, as llama.cpp's template dispatch does
 * (common/chat.cpp {@code common_chat_template_detect_params_init}): each format is recognized by the tool-call
 * syntax its template writes.
 *
 * <table>
 *   <caption>Built-in formats</caption>
 *   <tr><th>name (aliases)</th><th>syntax</th></tr>
 *   <tr><td>hermes (qwen2_5, qwen3_json, granite4)</td><td>{@code <tool_call>{json}</tool_call>}</td></tr>
 *   <tr><td>ernie45</td><td>{@code <tool_call>{json}</tool_call>}, the {@code <response>} tags left out of the content</td></tr>
 *   <tr><td>longcat</td><td>{@code <longcat_tool_call>{json}</longcat_tool_call>}</td></tr>
 *   <tr><td>qwen3_coder (qwen3_xml, qwen3, qwen3_5, nemotron_v3, step3p5, step5)</td><td>{@code <tool_call><function=f><parameter=k>v</parameter></function></tool_call>}</td></tr>
 *   <tr><td>mimo</td><td>the qwen3_coder syntax, every value kept as its text</td></tr>
 *   <tr><td>seed_oss</td><td>{@code <seed:tool_call><function=f><parameter=k>v</parameter></function></seed:tool_call>}</td></tr>
 *   <tr><td>minimax_m2</td><td>{@code <minimax:tool_call><invoke name="f"><parameter name="k">v</parameter></invoke></minimax:tool_call>}</td></tr>
 *   <tr><td>glm45 (glm47, glm4_moe, ling3, poolside_v1)</td><td>{@code <tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>}</td></tr>
 *   <tr><td>glm_4_7</td><td>the glm45 syntax (vLLM's {@code glm_4_7} structural-tag model)</td></tr>
 *   <tr><td>llama3_json (llama4_json)</td><td>{@code <|python_tag|>{"name", "parameters"}} or a bare object</td></tr>
 *   <tr><td>mistral (ministral3)</td><td>{@code [TOOL_CALLS][{json}]} or {@code [TOOL_CALLS]f[ARGS]{json}}</td></tr>
 *   <tr><td>deepseek_v3 (deepseek_r1)</td><td>{@code <｜tool▁call▁begin｜>function<｜tool▁sep｜>f\n```json ...```<｜tool▁call▁end｜>}</td></tr>
 *   <tr><td>deepseek_v31</td><td>{@code <｜tool▁call▁begin｜>f<｜tool▁sep｜>{json}<｜tool▁call▁end｜>}</td></tr>
 *   <tr><td>deepseek_v32</td><td>{@code <｜DSML｜function_calls><｜DSML｜invoke name="f">...</｜DSML｜function_calls>}</td></tr>
 *   <tr><td>deepseek_v4</td><td>DSML inside {@code <｜DSML｜tool_calls>} (also written {@code toolcalls} or {@code tool})</td></tr>
 *   <tr><td>deepseek_v41</td><td>DSML with the spaced {@code <｜DSML｜ calls>} markers</td></tr>
 *   <tr><td>kimi_k2</td><td>{@code <|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{json}<|tool_call_end|>}</td></tr>
 *   <tr><td>kimi_k3</td><td>XTML channels: {@code <|open|>tools<|sep|><|open|>call tool="f"<|sep|>...}</td></tr>
 *   <tr><td>k2_horizon</td><td>{@code <ifm|tool_call>f<ifm|arg_key>k</ifm|arg_key><ifm|arg_value>v</ifm|arg_value></ifm|tool_call>}</td></tr>
 *   <tr><td>internlm (internlm2)</td><td>{@code <|action_start|><|plugin|>{json}<|action_end|>}</td></tr>
 *   <tr><td>jamba (apriel, hunyuan_a13b)</td><td>{@code <tool_calls>[{json}]</tool_calls>}</td></tr>
 *   <tr><td>granite</td><td>{@code <|tool_call|>[{json}]}</td></tr>
 *   <tr><td>granite-20b-fc</td><td>{@code <function_call>{"name", "arguments"}}</td></tr>
 *   <tr><td>hy_v3</td><td>{@code <tool_calls{S}>...<tool_call{S}>f<tool_sep{S}><arg_key{S}>k</arg_key{S}>...}, the Hunyuan token suffix {@code S}: the template's under {@code auto}, the vocabulary's for the name on a loaded model ({@link ToolParser.Factory#forModel})</td></tr>
 *   <tr><td>hy_v4</td><td>the hy_v3 layout without {@code <tool_sep{S}>}</td></tr>
 *   <tr><td>phi4_mini_json (firefunction)</td><td>{@code functools[{json}]}</td></tr>
 *   <tr><td>xlam</td><td>a bare {@code [{json}]}, also in a {@code ```json} fence</td></tr>
 *   <tr><td>pythonic</td><td>a bare {@code [f(k=v)]}</td></tr>
 *   <tr><td>llama4_pythonic</td><td>{@code <|python_start|>[f(k=v)]<|python_end|>} or a bare list</td></tr>
 *   <tr><td>lfm2</td><td>{@code <|tool_call_start|>[f(k=v)]<|tool_call_end|>}</td></tr>
 *   <tr><td>olmo3</td><td>{@code <function_calls>f(k=v)\n...</function_calls>}</td></tr>
 *   <tr><td>dots</td><td>{@code <dots_function_call><invoke name="f"><parameter name="k">v</parameter>...</dots_function_call>}</td></tr>
 *   <tr><td>gigachat3</td><td>{@code function call<|role_sep|>\n{json}} or {@code <|function_call|>{json}}</td></tr>
 *   <tr><td>minimax_m3</td><td>{@code ]<]minimax[>[<tool_call>...<invoke name="f">...]<]minimax[>[</tool_call>}, element tags for the arguments</td></tr>
 *   <tr><td>minicpm5</td><td>{@code <function name="f"><param name="k">v</param>...</function>}</td></tr>
 *   <tr><td>step3</td><td>{@code <｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="f">...}</td></tr>
 *   <tr><td>plamo3</td><td>{@code <|plamo:begin_tool_request:plamo|><|plamo:begin_tool_name:plamo|>f...}</td></tr>
 *   <tr><td>apertus</td><td>{@code <|tools_prefix|>[{"f": {k: v}}]<|tools_suffix|>}</td></tr>
 *   <tr><td>functiongemma</td><td>{@code <start_function_call>call:f{k:<escape>v<escape>}<end_function_call>}</td></tr>
 *   <tr><td>gemma4</td><td>{@code <|tool_call>call:f{k:<|"|>v<|"|>}<tool_call|>}</td></tr>
 *   <tr><td>inkling</td><td>{@code <|content_invoke_tool_json|>{"name", "args"}<|end_message|>} blocks</td></tr>
 *   <tr><td>cohere_command3</td><td>{@code <|START_ACTION|>[{"tool_name", "parameters"}]<|END_ACTION|>}, replies in {@code <|START_RESPONSE|>}</td></tr>
 *   <tr><td>cohere_command4</td><td>the cohere_command3 layout, replies also in {@code <|START_TEXT|>}</td></tr>
 *   <tr><td>functionary_v3_2</td><td>{@code >>>f\n{json}} calls after the prompt's {@code >>>}, {@code >>>all\n} carries the reply</td></tr>
 *   <tr><td>hf, beam</td><td>the {@code tool_calls} regions of the loaded model's {@code response_template} (vLLM's response-template parsers, {@link ResponseTemplateToolParser}); refused without a model or a template with a {@code tool_calls} field</td></tr>
 *   <tr><td>autoparser</td><td>whatever syntax the loaded model's template writes, derived by differential analysis ({@link AutoParserFormat}); also {@code auto} detection's last resort and refused without a model or a template with tool-call syntax</td></tr>
 * </table>
 */
public final class ToolParserRegistry {

    /** The opener of a Muse tool message's ATEM body. */
    static final String ATEM_CALLS = "<atem:function_calls>";

    private static volatile Map<String, ToolParser.Factory> table;

    private ToolParserRegistry() {
    }

    /** The parser of a Muse tool message body: the ATEM invoke block ({@link XmlToolParser.Style#ATEM}). */
    static ToolParser atem(List<ChatTool> tools) {
        return new XmlToolParser(XmlToolParser.Style.ATEM, ATEM_CALLS, "</atem:function_calls>", false, false,
                ToolSchemas.of(tools));
    }

    /** The factory a name or alias selects (case-insensitive, {@code -} and {@code _} alike), or null. */
    public static ToolParser.Factory forName(String name) {
        if (name == null) {
            return null;
        }
        return table().get(key(name));
    }

    /** The primary names of every registered format, built-ins first. */
    public static List<String> names() {
        List<String> out = new ArrayList<>();
        for (ToolParser.Factory f : table().values()) {
            if (!out.contains(f.name())) {
                out.add(f.name());
            }
        }
        return out;
    }

    /**
     * The format whose tool-call syntax {@code template} writes, or null when the template cannot be compiled or
     * writes no syntax a format can read. More specific syntaxes are tested first (a Qwen3-Coder template also
     * contains Hermes' {@code <tool_call>}). A template none of the hand-written formations recognizes is offered to
     * the differential autoparser ({@link AutoParserFormat#forTemplate}), which derives the syntax by rendering
     * probe conversations; its analysis is cached per template source.
     */
    public static ToolParser.Factory detect(String template) {
        if (template == null) {
            return null;
        }
        String t = template;
        if (t.contains("<|tool_calls_section_begin|>") && t.contains("<|tool_call_begin|>")) {
            return forName("kimi_k2");
        }
        if (t.contains("tool▁call▁begin")) {
            return forName(t.contains("function<｜tool▁sep｜>") || t.contains("```json")
                    ? "deepseek_v3" : "deepseek_v31");
        }
        if (t.contains("<｜tool_calls_begin｜>") && t.contains("<｜tool_call_begin｜>")) {
            return forName("step3");
        }
        if (t.contains("<｜DSML｜function_calls>") || t.contains("<｜DSML｜invoke")) {
            return forName("deepseek_v32");
        }
        if (t.contains("<｜DSML｜ calls>") || t.contains("<｜DSML｜ invoke")) {
            return forName("deepseek_v41");
        }
        if (t.contains("<｜DSML｜tool_calls>") || t.contains("<｜DSML｜toolcalls>") || t.contains("<｜DSML｜tool>")) {
            return forName("deepseek_v4");
        }
        if (t.contains("<|plamo:begin_tool_request")) {
            return forName("plamo3");
        }
        if (t.contains("<|open|>tools<|sep|>")) {
            return forName("kimi_k3");
        }
        if (t.contains("<ifm|tool_call")) {
            return forName("k2_horizon");
        }
        if (t.contains("<tool_calls") && t.contains("<arg_key")) {
            return t.contains("<tool_sep") ? hyV3(hySuffix(t)) : hyV4(hySuffix(t));
        }
        if (t.contains("<|content_invoke_tool_json|>") || t.contains("<|content_text|>")) {
            return forName("inkling");
        }
        if (t.contains("<|START_ACTION|>")) {
            return forName(t.contains("<|START_TEXT|>") ? "cohere_command4" : "cohere_command3");
        }
        if (t.contains("<|tools_prefix|>")) {
            return forName("apertus");
        }
        if (t.contains("<start_function_call>")) {
            return forName("functiongemma");
        }
        if (t.contains("<|tool_call>") && t.contains("<tool_call|>")) {
            return forName("gemma4");
        }
        if (t.contains("<param name=") || t.contains("<function name=")) {
            return forName("minicpm5");
        }
        if (t.contains("<dots_function_call>")) {
            return forName("dots");
        }
        if (t.contains("<|function_call|>") || t.contains("function call<|role_sep|>")) {
            return forName("gigachat3");
        }
        if (t.contains("<function_call>")) {
            return forName("granite-20b-fc");
        }
        if (t.contains("[TOOL_CALLS]")) {
            return forName("mistral");
        }
        if (t.contains("<minimax:tool_call>") && t.contains("<invoke name=")) {
            return forName("minimax_m2");
        }
        if (t.contains("]<]minimax[>[<tool_call>")) {
            return forName("minimax_m3");
        }
        if (t.contains("<seed:tool_call>")) {
            return forName("seed_oss");
        }
        if (t.contains("<tool_call>") && t.contains("<arg_key>")) {
            return forName("glm45");
        }
        if (t.contains("<tool_call>") && t.contains("<function=") && t.contains("<parameter=")) {
            return forName("qwen3_coder");
        }
        if (t.contains("<|tool_call_start|>")) {
            return forName("lfm2");
        }
        if (t.contains("<|python_start|>")) {
            return forName("llama4_pythonic");
        }
        if (t.contains("<function_calls>")) {
            return forName("olmo3");
        }
        if (t.contains("<tool_calls>")) {
            return forName("jamba");
        }
        if (t.contains("<|tool_call|>")) {
            return forName("granite");
        }
        if (t.contains("functools[")) {
            return forName("phi4_mini_json");
        }
        if (t.contains("<|action_start|>") && t.contains("<|plugin|>")) {
            return forName("internlm");
        }
        if (t.contains(">>>") && t.contains("<|start_header_id|>")) {
            return forName("functionary_v3_2");
        }
        if (t.contains("<longcat_tool_call>")) {
            return forName("longcat");
        }
        if (t.contains("<tool_call>") && t.contains("<response>")) {
            return forName("ernie45");
        }
        if (t.contains("<tool_call>")) {
            return forName("hermes");
        }
        if (t.contains("<|python_tag|>") || (t.contains("ipython") && t.contains("\"parameters\""))) {
            return forName("llama3_json");
        }
        return AutoParserFormat.forTemplate(template);
    }

    private static String key(String name) {
        return name.trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }

    private static Map<String, ToolParser.Factory> table() {
        Map<String, ToolParser.Factory> t = table;
        if (t == null) {
            synchronized (ToolParserRegistry.class) {
                t = table;
                if (t == null) {
                    t = build(builtIns(), loadPlugins());
                    table = t;
                }
            }
        }
        return t;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static List<ToolParserProvider<ToolParser.Factory>> loadPlugins() {
        List<ToolParserProvider<ToolParser.Factory>> out = new ArrayList<>();
        for (ToolParserProvider p : Plugins.load(ToolParserProvider.class)) {
            out.add((ToolParserProvider<ToolParser.Factory>) p);
        }
        return out;
    }

    /**
     * The name table of {@code builtIns} plus the providers' factories, in provider order.
     *
     * @throws IllegalStateException when a name or alias is registered twice, naming both sources
     */
    static Map<String, ToolParser.Factory> build(List<ToolParser.Factory> builtIns,
                                                 List<? extends ToolParserProvider<ToolParser.Factory>> providers) {
        Map<String, ToolParser.Factory> out = new LinkedHashMap<>();
        Map<String, String> source = new LinkedHashMap<>();
        for (ToolParser.Factory f : builtIns) {
            register(out, source, f, "built-in");
        }
        for (ToolParserProvider<ToolParser.Factory> p : providers) {
            List<ToolParser.Factory> rows = p.toolParserFactories();
            if (rows == null) {
                throw new IllegalStateException("tool parser provider " + p.getClass().getName()
                        + " returned null factories");
            }
            for (Object row : rows) {
                if (!(row instanceof ToolParser.Factory)) {
                    throw new IllegalStateException("tool parser provider " + p.getClass().getName()
                            + " contributed " + (row == null ? "null" : row.getClass().getName())
                            + ", not a ToolParser.Factory");
                }
                register(out, source, (ToolParser.Factory) row, p.getClass().getName());
            }
        }
        return Collections.unmodifiableMap(out);
    }

    private static void register(Map<String, ToolParser.Factory> out, Map<String, String> source,
                                 ToolParser.Factory f, String from) {
        List<String> names = new ArrayList<>();
        names.add(f.name());
        names.addAll(f.aliases());
        for (String n : names) {
            String k = key(n);
            if (out.containsKey(k)) {
                throw new IllegalStateException("tool parser name '" + n + "' of " + from
                        + " is already registered by " + source.get(k));
            }
            out.put(k, f);
            source.put(k, from);
        }
    }

    /** A built-in row. */
    private static final class Format implements ToolParser.Factory {
        private final String name;
        private final List<String> aliases;
        private final Function<List<ChatTool>, ToolParser> create;
        /** The factory for a loaded model, or null when the model does not change the format. */
        private final Function<GGUFTokenizer, ToolParser.Factory> forModel;

        Format(String name, List<String> aliases, Function<List<ChatTool>, ToolParser> create,
               Function<GGUFTokenizer, ToolParser.Factory> forModel) {
            this.name = name;
            this.aliases = aliases;
            this.create = create;
            this.forModel = forModel;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public List<String> aliases() {
            return aliases;
        }

        @Override
        public ToolParser create(List<ChatTool> tools) {
            return create.apply(tools == null ? Collections.<ChatTool>emptyList() : tools);
        }

        @Override
        public ToolParser.Factory forModel(GGUFTokenizer tokenizer) {
            return forModel == null ? this : forModel.apply(tokenizer);
        }

        @Override
        public String toString() {
            return "ToolParser.Factory(" + name + ")";
        }
    }

    private static Format format(String name, List<String> aliases, Function<List<ChatTool>, ToolParser> create) {
        return new Format(name, Collections.unmodifiableList(new ArrayList<>(aliases)), create, null);
    }

    /** A row whose factory for a loaded model is {@code forModel}'s; the row itself serves without a model. */
    private static Format modelFormat(Format row, Function<GGUFTokenizer, ToolParser.Factory> forModel) {
        return new Format(row.name, row.aliases, row.create, forModel);
    }

    private static List<ToolParser.Factory> builtIns() {
        String dsCalls = "<｜tool▁calls▁begin｜>";
        String dsCallsEnd = "<｜tool▁calls▁end｜>";
        String dsCall = "<｜tool▁call▁begin｜>";
        String dsCallEnd = "<｜tool▁call▁end｜>";
        String dsSep = "<｜tool▁sep｜>";
        List<String> none = Collections.emptyList();
        return Arrays.<ToolParser.Factory>asList(
                format("hermes", Arrays.asList("qwen2_5", "qwen3_json", "granite4"),
                        tools -> new JsonToolParser(Arrays.asList("<tool_call>"), "</tool_call>",
                                JsonToolParser.Body.ANY, false)),
                format("ernie45", none,
                        tools -> new ResponseTagToolParser(new JsonToolParser(Arrays.asList("<tool_call>"),
                                "</tool_call>", JsonToolParser.Body.ANY, false))),
                format("longcat", none,
                        tools -> new JsonToolParser(Arrays.asList("<longcat_tool_call>"), "</longcat_tool_call>",
                                JsonToolParser.Body.ANY, false)),
                format("qwen3_coder", Arrays.asList("qwen3_xml", "qwen3", "qwen3_5", "nemotron_v3", "step3p5",
                                "step5"),
                        tools -> new XmlToolParser(XmlToolParser.Style.FUNCTION, "<tool_call>", "</tool_call>",
                                true, true, ToolSchemas.of(tools))),
                format("mimo", none,
                        tools -> new XmlToolParser(XmlToolParser.Style.FUNCTION, "<tool_call>", "</tool_call>",
                                true, true, ToolSchemas.none())),
                format("seed_oss", none,
                        tools -> new XmlToolParser(XmlToolParser.Style.FUNCTION, "<seed:tool_call>",
                                "</seed:tool_call>", false, false, ToolSchemas.of(tools))),
                format("minimax_m2", Arrays.asList("minimax"),
                        tools -> new XmlToolParser(XmlToolParser.Style.INVOKE, "<minimax:tool_call>",
                                "</minimax:tool_call>", false, false, ToolSchemas.of(tools))),
                format("glm45", Arrays.asList("glm47", "glm4_moe", "ling3", "poolside_v1"),
                        tools -> new XmlToolParser(XmlToolParser.Style.ARG_KEY, "<tool_call>", "</tool_call>",
                                false, false, ToolSchemas.of(tools))),
                format("glm_4_7", none,
                        tools -> new XmlToolParser(XmlToolParser.Style.ARG_KEY, "<tool_call>", "</tool_call>",
                                false, false, ToolSchemas.of(tools))),
                format("llama3_json", Arrays.asList("llama4_json"),
                        tools -> new JsonToolParser(Arrays.asList("<|python_tag|>"), null,
                                JsonToolParser.Body.OBJECTS, true)),
                format("mistral", Arrays.asList("ministral3"), tools -> new MistralToolParser()),
                format("deepseek_v3", Arrays.asList("deepseek_r1"),
                        tools -> new NameArgsToolParser(dsCalls, dsCallsEnd, dsCall, "function" + dsSep, null,
                                "\n", null, dsCallEnd, NameArgsToolParser::plain)),
                format("deepseek_v31", none,
                        tools -> new NameArgsToolParser(dsCalls, dsCallsEnd, dsCall, null, null, dsSep, null,
                                dsCallEnd, NameArgsToolParser::plain)),
                format("kimi_k2", none,
                        tools -> new NameArgsToolParser("<|tool_calls_section_begin|>",
                                "<|tool_calls_section_end|>", "<|tool_call_begin|>", null, null,
                                "<|tool_call_argument_begin|>", null, "<|tool_call_end|>", NameArgsToolParser::kimi)),
                format("internlm", Arrays.asList("internlm2"),
                        tools -> new JsonToolParser(Arrays.asList("<|action_start|><|plugin|>"), "<|action_end|>",
                                JsonToolParser.Body.OBJECT, false)),
                format("jamba", Arrays.asList("apriel", "hunyuan_a13b"),
                        tools -> new JsonToolParser(Arrays.asList("<tool_calls>"), "</tool_calls>",
                                JsonToolParser.Body.ANY, false)),
                format("granite", Arrays.asList("granite_3"),
                        tools -> new JsonToolParser(Arrays.asList("<|tool_call|>"), null,
                                JsonToolParser.Body.ARRAY, false)),
                format("phi4_mini_json", Arrays.asList("firefunction", "phi4mini"),
                        tools -> new JsonToolParser(Arrays.asList("functools"), null,
                                JsonToolParser.Body.ARRAY, false)),
                format("xlam", none,
                        tools -> new JsonToolParser(Collections.<String>emptyList(), null,
                                JsonToolParser.Body.ARRAY, true)),
                format("pythonic", none, tools -> new PythonicToolParser(null, null, true, true)),
                format("llama4_pythonic", none,
                        tools -> new PythonicToolParser("<|python_start|>", "<|python_end|>", true, true)),
                format("lfm2", Arrays.asList("lfm2_5"),
                        tools -> new PythonicToolParser("<|tool_call_start|>", "<|tool_call_end|>", true, false)),
                format("olmo3", none,
                        tools -> new PythonicToolParser("<function_calls>", "</function_calls>", false, false)),
                format("dots", none,
                        tools -> new XmlToolParser(XmlToolParser.Style.INVOKE, "<dots_function_call>",
                                "</dots_function_call>", false, false, ToolSchemas.of(tools))),
                format("granite-20b-fc", none,
                        tools -> new JsonToolParser(Arrays.asList("<function_call>"), null,
                                JsonToolParser.Body.OBJECTS, false)),
                format("gigachat3", none,
                        tools -> new ContentEndToolParser(new JsonToolParser(
                                        Arrays.asList("<|function_call|>", "function call<|role_sep|>\n"), null,
                                        JsonToolParser.Body.OBJECT, false),
                                "<|message_sep|>")),
                format("minicpm5", none,
                        tools -> new XmlToolParser(XmlToolParser.Style.PARAM_ATTR, "", "", false, false,
                                ToolSchemas.of(tools))),
                format("apertus", none, tools -> new ApertusToolParser()),
                format("deepseek_v32", none, tools -> new DsmlToolParser(
                        Collections.singletonList("<｜DSML｜function_calls>"), "</｜DSML｜function_calls>",
                        "<｜DSML｜invoke name=\"", "\">", "</｜DSML｜invoke>",
                        "<｜DSML｜parameter", "</｜DSML｜parameter>")),
                format("deepseek_v4", none, tools -> new DsmlToolParser(
                        Arrays.asList("<｜DSML｜tool_calls>", "<｜DSML｜toolcalls>", "<｜DSML｜tool>"),
                        "</｜DSML｜tool_calls>", "<｜DSML｜invoke name=\"", "\">", "</｜DSML｜invoke>",
                        "<｜DSML｜parameter", "</｜DSML｜parameter>")),
                format("deepseek_v41", none, tools -> new DsmlToolParser(
                        Collections.singletonList("<｜DSML｜ calls>"), "</｜DSML｜ calls>",
                        "<｜DSML｜ invoke name=\"", "\">", "</｜DSML｜ invoke>",
                        "<｜DSML｜ parameter", "</｜DSML｜ parameter>")),
                format("kimi_k3", none, tools -> new XtmlToolParser()),
                format("k2_horizon", none,
                        tools -> new ArgKeyToolParser("<ifm|tool_calls>", "</ifm|tool_calls>",
                                "<ifm|tool_call>", "</ifm|tool_call>", null,
                                "<ifm|arg_key>", "</ifm|arg_key>", "<ifm|arg_type>", "</ifm|arg_type>",
                                "<ifm|arg_value>", "</ifm|arg_value>", ToolSchemas.of(tools))),
                format("minimax_m3", none, tools -> new MinimaxM3ToolParser(ToolSchemas.of(tools))),
                format("step3", none, tools -> new SteptmlToolParser(ToolSchemas.of(tools))),
                format("plamo3", none,
                        tools -> new NameArgsToolParser("<|plamo:begin_tool_requests:plamo|>",
                                "<|plamo:end_tool_requests:plamo|>", "<|plamo:begin_tool_request:plamo|>", null,
                                "<|plamo:begin_tool_name:plamo|>", "<|plamo:end_tool_name:plamo|>",
                                "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>",
                                "<|plamo:end_tool_arguments:plamo|><|plamo:end_tool_request:plamo|>",
                                NameArgsToolParser::plain)),
                modelFormat(hyV3(""), tok -> hyV3(ReasoningParsers.hunyuanSuffix(tok))),
                modelFormat(hyV4(""), tok -> hyV4(ReasoningParsers.hunyuanSuffix(tok))),
                format("functiongemma", none, tools -> new FunctionGemmaToolParser(ToolSchemas.of(tools))),
                format("gemma4", none, tools -> new Gemma4ToolParser()),
                format("inkling", none, tools -> new InklingToolParser()),
                format("cohere_command3", none, tools -> CohereCommandToolParser.command3()),
                format("cohere_command4", none, tools -> CohereCommandToolParser.command4()),
                format("functionary_v3_2", none, tools -> new FunctionaryToolParser()),
                modelFormat(format("autoparser", none, tools -> {
                    throw new IllegalArgumentException("tool parser 'autoparser' analyses the loaded model's chat"
                            + " template; it needs the model's tokenizer");
                }), AutoParserFormat::forModelAnalysis),
                responseTemplateFormat("hf"),
                responseTemplateFormat("beam"));
    }

    /**
     * vLLM's response-template tool parser {@code name} ({@code hf} or {@code beam}): the loaded model's
     * {@code response_template} reads the calls ({@link ResponseTemplateToolParser}); without a model there is no
     * template, so the row refuses by name.
     */
    private static Format responseTemplateFormat(String name) {
        Format row = format(name, Collections.<String>emptyList(), tools -> {
            throw new IllegalArgumentException("tool parser '" + name + "' reads the loaded model's response_template;"
                    + " it needs the model's tokenizer");
        });
        return modelFormat(row, tok -> {
            TemplateParsing parsing = TemplateParsing.forModel(name, tok, "--tool-call-parser " + name,
                    ResponseTemplate.TOOL_CALLS);
            return new TemplateFormat(format(name, Collections.<String>emptyList(),
                    tools -> new ResponseTemplateToolParser(parsing, tools)), parsing);
        });
    }

    /** A response-template row resolved for a model: the format and the template parsing its requests are held to. */
    static final class TemplateFormat implements ToolParser.Factory {
        private final Format format;
        private final TemplateParsing parsing;

        TemplateFormat(Format format, TemplateParsing parsing) {
            this.format = format;
            this.parsing = parsing;
        }

        TemplateParsing parsing() {
            return parsing;
        }

        @Override
        public String name() {
            return format.name();
        }

        @Override
        public List<String> aliases() {
            return format.aliases();
        }

        @Override
        public ToolParser create(List<ChatTool> tools) {
            return format.create(tools);
        }

        @Override
        public String toString() {
            return format.toString();
        }
    }

    /** HY v3: the Hunyuan structural-token suffix {@code suffix} baked into every marker. */
    private static Format hyV3(String suffix) {
        return format("hy_v3", Collections.emptyList(),
                tools -> new ArgKeyToolParser("<tool_calls" + suffix + ">", "</tool_calls" + suffix + ">",
                        "<tool_call" + suffix + ">", "</tool_call" + suffix + ">", "<tool_sep" + suffix + ">",
                        "<arg_key" + suffix + ">", "</arg_key" + suffix + ">", null, null,
                        "<arg_value" + suffix + ">", "</arg_value" + suffix + ">", ToolSchemas.of(tools)));
    }

    /** HY v4: the Hunyuan structural-token suffix {@code suffix}, the name running to the first key tag. */
    private static Format hyV4(String suffix) {
        return format("hy_v4", Collections.emptyList(),
                tools -> new ArgKeyToolParser("<tool_calls" + suffix + ">", "</tool_calls" + suffix + ">",
                        "<tool_call" + suffix + ">", "</tool_call" + suffix + ">", null,
                        "<arg_key" + suffix + ">", "</arg_key" + suffix + ">", null, null,
                        "<arg_value" + suffix + ">", "</arg_value" + suffix + ">", ToolSchemas.of(tools)));
    }

    /**
     * The Hunyuan structural-token suffix of a template: its {@code HYTK} assignment when it spells the markers
     * {@code '<tool_calls{}>'.format(HYTK)} (tencent-Hy3.jinja), else the text it writes after {@code <tool_calls},
     * else the empty one.
     */
    private static String hySuffix(String template) {
        String assigned = ReasoningFilter.hunyuanTemplateSuffix(template);
        if (assigned != null) {
            return assigned;
        }
        int at = template.indexOf("<tool_calls");
        if (at < 0) {
            return "";
        }
        int gt = template.indexOf('>', at);
        String suffix = gt > at ? template.substring(at + "<tool_calls".length(), gt) : "";
        return suffix.isEmpty() || suffix.startsWith(":") ? suffix : "";
    }
}
