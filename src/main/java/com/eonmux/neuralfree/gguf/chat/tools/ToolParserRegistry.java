package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
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
 *   <tr><td>hermes (qwen2_5, qwen3_json)</td><td>{@code <tool_call>{json}</tool_call>}</td></tr>
 *   <tr><td>ernie45</td><td>{@code <tool_call>{json}</tool_call>}, the {@code <response>} tags left out of the content</td></tr>
 *   <tr><td>longcat</td><td>{@code <longcat_tool_call>{json}</longcat_tool_call>}</td></tr>
 *   <tr><td>qwen3_coder (qwen3_xml, qwen3, qwen3_5, nemotron_v3)</td><td>{@code <tool_call><function=f><parameter=k>v</parameter></function></tool_call>}</td></tr>
 *   <tr><td>seed_oss</td><td>{@code <seed:tool_call><function=f><parameter=k>v</parameter></function></seed:tool_call>}</td></tr>
 *   <tr><td>minimax_m2</td><td>{@code <minimax:tool_call><invoke name="f"><parameter name="k">v</parameter></invoke></minimax:tool_call>}</td></tr>
 *   <tr><td>glm45 (glm47, glm4_moe)</td><td>{@code <tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>}</td></tr>
 *   <tr><td>llama3_json (llama4_json)</td><td>{@code <|python_tag|>{"name", "parameters"}} or a bare object</td></tr>
 *   <tr><td>mistral</td><td>{@code [TOOL_CALLS][{json}]} or {@code [TOOL_CALLS]f[ARGS]{json}}</td></tr>
 *   <tr><td>deepseek_v3 (deepseek_r1)</td><td>{@code <｜tool▁call▁begin｜>function<｜tool▁sep｜>f\n```json ...```<｜tool▁call▁end｜>}</td></tr>
 *   <tr><td>deepseek_v31</td><td>{@code <｜tool▁call▁begin｜>f<｜tool▁sep｜>{json}<｜tool▁call▁end｜>}</td></tr>
 *   <tr><td>kimi_k2</td><td>{@code <|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{json}<|tool_call_end|>}</td></tr>
 *   <tr><td>internlm (internlm2)</td><td>{@code <|action_start|><|plugin|>{json}<|action_end|>}</td></tr>
 *   <tr><td>jamba (apriel, hunyuan_a13b)</td><td>{@code <tool_calls>[{json}]</tool_calls>}</td></tr>
 *   <tr><td>granite</td><td>{@code <|tool_call|>[{json}]}</td></tr>
 *   <tr><td>phi4_mini_json (firefunction)</td><td>{@code functools[{json}]}</td></tr>
 *   <tr><td>xlam</td><td>a bare {@code [{json}]}, also in a {@code ```json} fence</td></tr>
 *   <tr><td>pythonic</td><td>a bare {@code [f(k=v)]}</td></tr>
 *   <tr><td>llama4_pythonic</td><td>{@code <|python_start|>[f(k=v)]<|python_end|>} or a bare list</td></tr>
 *   <tr><td>lfm2</td><td>{@code <|tool_call_start|>[f(k=v)]<|tool_call_end|>}</td></tr>
 *   <tr><td>olmo3</td><td>{@code <function_calls>f(k=v)\n...</function_calls>}</td></tr>
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
     * The format whose tool-call syntax {@code template} writes, or null when the template writes none this
     * registry knows. More specific syntaxes are tested first (a Qwen3-Coder template also contains Hermes'
     * {@code <tool_call>}).
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
        if (t.contains("[TOOL_CALLS]")) {
            return forName("mistral");
        }
        if (t.contains("<minimax:tool_call>") && t.contains("<invoke name=")) {
            return forName("minimax_m2");
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
        return null;
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

        Format(String name, List<String> aliases, Function<List<ChatTool>, ToolParser> create) {
            this.name = name;
            this.aliases = aliases;
            this.create = create;
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
        public String toString() {
            return "ToolParser.Factory(" + name + ")";
        }
    }

    private static Format format(String name, List<String> aliases, Function<List<ChatTool>, ToolParser> create) {
        return new Format(name, Collections.unmodifiableList(new ArrayList<>(aliases)), create);
    }

    private static List<ToolParser.Factory> builtIns() {
        String dsCalls = "<｜tool▁calls▁begin｜>";
        String dsCallsEnd = "<｜tool▁calls▁end｜>";
        String dsCall = "<｜tool▁call▁begin｜>";
        String dsCallEnd = "<｜tool▁call▁end｜>";
        String dsSep = "<｜tool▁sep｜>";
        List<String> none = Collections.emptyList();
        return Arrays.<ToolParser.Factory>asList(
                format("hermes", Arrays.asList("qwen2_5", "qwen3_json"),
                        tools -> new JsonToolParser(Arrays.asList("<tool_call>"), "</tool_call>",
                                JsonToolParser.Body.ANY, false)),
                format("ernie45", none,
                        tools -> new ResponseTagToolParser(new JsonToolParser(Arrays.asList("<tool_call>"),
                                "</tool_call>", JsonToolParser.Body.ANY, false))),
                format("longcat", none,
                        tools -> new JsonToolParser(Arrays.asList("<longcat_tool_call>"), "</longcat_tool_call>",
                                JsonToolParser.Body.ANY, false)),
                format("qwen3_coder", Arrays.asList("qwen3_xml", "qwen3", "qwen3_5", "nemotron_v3"),
                        tools -> new XmlToolParser(XmlToolParser.Style.FUNCTION, "<tool_call>", "</tool_call>",
                                true, true, ToolSchemas.of(tools))),
                format("seed_oss", none,
                        tools -> new XmlToolParser(XmlToolParser.Style.FUNCTION, "<seed:tool_call>",
                                "</seed:tool_call>", false, false, ToolSchemas.of(tools))),
                format("minimax_m2", Arrays.asList("minimax"),
                        tools -> new XmlToolParser(XmlToolParser.Style.INVOKE, "<minimax:tool_call>",
                                "</minimax:tool_call>", false, false, ToolSchemas.of(tools))),
                format("glm45", Arrays.asList("glm47", "glm4_moe"),
                        tools -> new XmlToolParser(XmlToolParser.Style.ARG_KEY, "<tool_call>", "</tool_call>",
                                false, false, ToolSchemas.of(tools))),
                format("llama3_json", Arrays.asList("llama4_json"),
                        tools -> new JsonToolParser(Arrays.asList("<|python_tag|>"), null,
                                JsonToolParser.Body.OBJECTS, true)),
                format("mistral", none, tools -> new MistralToolParser()),
                format("deepseek_v3", Arrays.asList("deepseek_r1"),
                        tools -> new NameArgsToolParser(dsCalls, dsCallsEnd, dsCall, "function" + dsSep, "\n",
                                dsCallEnd, NameArgsToolParser::plain)),
                format("deepseek_v31", none,
                        tools -> new NameArgsToolParser(dsCalls, dsCallsEnd, dsCall, null, dsSep, dsCallEnd,
                                NameArgsToolParser::plain)),
                format("kimi_k2", none,
                        tools -> new NameArgsToolParser("<|tool_calls_section_begin|>",
                                "<|tool_calls_section_end|>", "<|tool_call_begin|>", null,
                                "<|tool_call_argument_begin|>", "<|tool_call_end|>", NameArgsToolParser::kimi)),
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
                        tools -> new PythonicToolParser("<function_calls>", "</function_calls>", false, false)));
    }
}
