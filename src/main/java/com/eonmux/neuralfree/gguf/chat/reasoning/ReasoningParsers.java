package com.eonmux.neuralfree.gguf.chat.reasoning;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserFormat;
import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.plugin.Plugins;
import com.eonmux.neuralfree.gguf.plugin.ReasoningParserProvider;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * The reasoning parser registry, serving the names of vLLM's {@code --reasoning-parser} and the
 * {@code neuralfree.reasoning.parser} property: each name selects a {@link ReasoningFilter} configuration — the
 * delimiters its model family writes around reasoning — and {@code auto} (or no name) keeps the filter the chat
 * template implies ({@link ReasoningFilter#forChatTemplate}). Plugins contribute further rows through
 * {@link ReasoningParserProvider ReasoningParserProvider&lt;ReasoningParsers.Factory&gt;}, merged in
 * provider-class order; a name colliding with a built-in or with another plugin's name fails the table build with
 * both sources named.
 *
 * <p>Built-in rows, after vLLM's reasoning parsers and llama.cpp's per-format delimiters: {@code deepseek_r1},
 * {@code deepseek_v3}, {@code glm45}, {@code kimi_k2}, {@code minimax_m2}, {@code ernie45}, {@code hunyuan_a13b},
 * {@code olmo3}, {@code step3}, {@code mimo}, {@code poolside_v1} and {@code holo2}
 * ({@code <think>}/{@code </think>}); {@code qwen3} and {@code ling3} (also ended by a {@code <tool_call>});
 * {@code deepseek_v4} (also ended by a DSML call block); {@code minimax_m3} ({@code <mm:think>});
 * {@code seed_oss} ({@code <seed:think>}); {@code mistral} ({@code [THINK]}); {@code gemma4}
 * ({@code <|channel>thought}/{@code <channel|>}); {@code granite} ({@code Here is my thought process:} /
 * {@code Here is my response:}); {@code cohere_command3} and {@code cohere_command4}
 * ({@code <|START_THINKING|>}); {@code kimi_k3} ({@code <|open|>think<|sep|>}); {@code k2_horizon}
 * ({@code <ifm|think>} and its fast variants); {@code openai_gptoss} and {@code muse} (the harmony channels);
 * {@code inkling} (the Inkling block layout, {@link ReasoningFilter#inkling}); {@code minimax_m2_append_think}
 * (everything is content, after the {@code <think>} the prompt opened); {@code none} (alias {@code identity}:
 * everything is content).
 *
 * <p>Rows that read the request's chat template variables ({@link ReasoningFilter#perRequest}), as vLLM's parsers
 * read {@code chat_template_kwargs}:
 * <ul>
 *   <li>{@code hy_v3} and {@code hy_v4}: {@code <think{S}>}/{@code </think{S}>} with the checkpoint's
 *       structural-token suffix {@code S} (the vocabulary's suffixed {@code <think:...>} tokens), reasoning from
 *       the output's start; {@code reasoning_effort: no_think} makes everything content, and an unset effort
 *       means {@code no_think} for {@code hy_v3} and {@code high} for {@code hy_v4}.</li>
 *   <li>{@code nemotron_v3} and {@code granite_thinking_parser}: the Qwen3 tags, reasoning from the output's
 *       start, a stray {@code </think>} dropped; with {@code enable_thinking: false} no reasoning is parsed at all —
 *       the whole output is content, a {@code <think>} in it included — and with {@code force_nonempty_content: true}
 *       an answer without content gets its reasoning as the content (an assembled message as its content only,
 *       {@link ReasoningFilter.Sink#contentFallback}).
 *       {@code granite_thinking_parser} also drops the newlines the content starts with, and swaps only an answer
 *       with no content at all.</li>
 *   <li>{@code minicpmv}: the Qwen3 tags, reasoning from the output's start, {@code <|im_end|>} and the reserved
 *       markers 12-15 removed, and the tokenizer's {@code \n} escapes recovered as newlines; with
 *       {@code enable_thinking: false} a {@code <think>} block is discarded instead.</li>
 *   <li>{@code plamo3}: {@code <|plamo:begin_think:plamo|>}/{@code <|plamo:end_think:plamo|>} (the end of turn and
 *       the tool-request section also end it), reasoning from the output's start and reopened by a later start
 *       tag, a marker cut off by the end of generation dropped; {@code enable_thinking: false} makes everything
 *       content.</li>
 *   <li>{@code hf} and {@code beam} (vLLM's response-template parsers, {@link TemplateParsing}): the model's
 *       {@code response_template} metadata decides the regions ({@link ReasoningFilter#responseTemplate}); a model
 *       without one, or whose template has no thinking field, is refused by name. {@code enable_thinking: false}
 *       tells an output constraint the reasoning is over unless the prompt left it open.</li>
 * </ul>
 * A per-request variable of the wrong type (a non-boolean {@code enable_thinking}, a non-string
 * {@code reasoning_effort}) is refused by name ({@link IllegalArgumentException}).
 */
public final class ReasoningParsers {

    /** The {@code auto} name: the chat template decides. */
    public static final String AUTO = "auto";

    private static final String ENABLE_THINKING = "enable_thinking";
    private static final String REASONING_EFFORT = "reasoning_effort";
    private static final String FORCE_NONEMPTY_CONTENT = "force_nonempty_content";
    /** The text every PLaMo 3 control marker starts with. */
    private static final String PLAMO_MARKER_STEM = "<|plamo:";
    /** The HY {@code reasoning_effort} that turns thinking off. */
    private static final String NO_THINK = "no_think";
    /** The Hunyuan structural markers whose token texts carry the checkpoint suffix; longer stems first. */
    private static final List<String> HUNYUAN_STEMS = Collections.unmodifiableList(Arrays.asList(
            "think", "tool_calls", "tool_call", "tool_sep", "arg_key", "arg_value"));

    /**
     * One registry row: a name, its aliases and the filter it selects. The factory type plugins contribute through
     * {@code ReasoningParserProvider<ReasoningParsers.Factory>}.
     */
    public interface Factory {

        String name();

        /** Further names selecting the same filter; empty when none. */
        List<String> aliases();

        /** The filter (tag tokens are resolved later, per model, with {@link ReasoningFilter#withTagTokens}). */
        ReasoningFilter create();

        /**
         * The filter for a loaded model, whose vocabulary may decide its tags (the Hunyuan structural-token suffix);
         * {@link #create()} by default.
         */
        default ReasoningFilter create(GGUFTokenizer tokenizer) {
            return create();
        }
    }

    private static volatile Map<String, Factory> table;

    private ReasoningParsers() {
    }

    /** The row a name or alias selects (case-insensitive, {@code -} and {@code _} alike), or null. */
    public static Factory forName(String name) {
        return name == null ? null : table().get(key(name));
    }

    /** Whether {@code name} is {@code auto}, null or blank: the template decides. */
    public static boolean isAuto(String name) {
        return name == null || name.trim().isEmpty() || key(name).equals(AUTO);
    }

    /** The primary names of every row, built-ins first. */
    public static List<String> names() {
        List<String> out = new ArrayList<>();
        for (Factory f : table().values()) {
            if (!out.contains(f.name())) {
                out.add(f.name());
            }
        }
        return out;
    }

    /**
     * The filter of a loaded model: the named row ({@code name} not {@link #isAuto auto}), else the chat
     * template's, with the template's tag tokens matched by id.
     *
     * @throws IllegalArgumentException when {@code name} selects no row
     */
    public static ReasoningFilter forModel(String name, GGUFTokenizer tokenizer) {
        ReasoningFilter base;
        if (isAuto(name)) {
            base = ReasoningFilter.forChatTemplate(tokenizer.chatTemplate());
            if (base.mode() == ReasoningFilter.Mode.NONE) {
                ReasoningFilter derived = AutoParserFormat.reasoningFilter(tokenizer);
                if (derived != null) {
                    base = derived;
                }
            }
        } else {
            Factory f = forName(name);
            if (f == null) {
                throw new IllegalArgumentException("reasoning parser '" + name + "' is not registered; known: "
                        + names() + " or " + AUTO);
            }
            base = f.create(tokenizer);
        }
        return base.withTagTokens(tokenizer::getTokenId);
    }

    /**
     * The filter of a loaded model served with {@code toolParser}: {@link #forModel(String, GGUFTokenizer)}, except that
     * an {@link #isAuto auto} reasoning parser beside a response-template tool parser ({@code hf} or {@code beam})
     * parses no reasoning, as vLLM runs no reasoning parser there: the template routes the output, its thinking
     * region's text streaming as content without its delimiters ({@link TemplateParsing#withoutReasoning}). The model's
     * template is parsed here, at load.
     *
     * @throws IllegalArgumentException when {@code name} selects no row, or the template tool parser cannot serve the
     *                                  model
     */
    public static ReasoningFilter forModel(String name, String toolParser, GGUFTokenizer tokenizer) {
        if (!isAuto(name) || !TemplateParsing.names(toolParser)) {
            return forModel(name, tokenizer);
        }
        TemplateParsing parsing = TemplateParsing.forModel(toolParser, tokenizer, "--tool-call-parser " + toolParser,
                ResponseTemplate.TOOL_CALLS);
        return ReasoningFilter.responseTemplate(parsing.withoutReasoning()).withTagTokens(tokenizer::getTokenId);
    }

    /**
     * The Hunyuan structural-token suffix of a vocabulary, as vLLM's HY parsers detect it: the suffix of the
     * lowest-id token that opens or closes a {@code think}, {@code tool_calls}, {@code tool_call}, {@code tool_sep},
     * {@code arg_key} or {@code arg_value} marker ({@code <think:6124c78e>} gives {@code :6124c78e}, a plain
     * {@code <think>} the empty suffix); empty when the vocabulary has none.
     */
    public static String hunyuanSuffix(GGUFTokenizer tokenizer) {
        for (int id = 0; id < tokenizer.vocabSize(); id++) {
            String suffix = hunyuanMarkerSuffix(tokenizer.getToken(id));
            if (suffix != null) {
                return suffix;
            }
        }
        return "";
    }

    /** The suffix of one Hunyuan marker token text, or null when the text is no such marker. */
    static String hunyuanMarkerSuffix(String token) {
        if (token == null || token.length() < 3 || token.charAt(0) != '<' || token.charAt(token.length() - 1) != '>') {
            return null;
        }
        String body = token.substring(1, token.length() - 1);
        if (body.startsWith("/")) {
            body = body.substring(1);
        }
        for (String stem : HUNYUAN_STEMS) {
            if (body.startsWith(stem)) {
                String suffix = body.substring(stem.length());
                if (suffix.isEmpty() || (suffix.startsWith(":") && suffix.length() > 1 && !hasSpaceOrAngle(suffix))) {
                    return suffix;
                }
            }
        }
        return null;
    }

    private static boolean hasSpaceOrAngle(String s) {
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isWhitespace(c) || c == '<' || c == '>') {
                return true;
            }
        }
        return false;
    }

    private static String key(String name) {
        return name.trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }

    private static Map<String, Factory> table() {
        Map<String, Factory> t = table;
        if (t == null) {
            synchronized (ReasoningParsers.class) {
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
    private static List<ReasoningParserProvider<Factory>> loadPlugins() {
        List<ReasoningParserProvider<Factory>> out = new ArrayList<>();
        for (ReasoningParserProvider p : Plugins.load(ReasoningParserProvider.class)) {
            out.add((ReasoningParserProvider<Factory>) p);
        }
        return out;
    }

    /**
     * The name table of {@code builtIns} plus the providers' rows.
     *
     * @throws IllegalStateException when a name or alias is registered twice, naming both sources
     */
    static Map<String, Factory> build(List<Factory> builtIns,
                                      List<? extends ReasoningParserProvider<Factory>> providers) {
        Map<String, Factory> out = new LinkedHashMap<>();
        Map<String, String> source = new LinkedHashMap<>();
        for (Factory f : builtIns) {
            register(out, source, f, "built-in");
        }
        for (ReasoningParserProvider<Factory> p : providers) {
            List<Factory> rows = p.reasoningParserFactories();
            if (rows == null) {
                throw new IllegalStateException("reasoning parser provider " + p.getClass().getName()
                        + " returned null factories");
            }
            for (Object row : rows) {
                if (!(row instanceof Factory)) {
                    throw new IllegalStateException("reasoning parser provider " + p.getClass().getName()
                            + " contributed " + (row == null ? "null" : row.getClass().getName())
                            + ", not a ReasoningParsers.Factory");
                }
                register(out, source, (Factory) row, p.getClass().getName());
            }
        }
        return Collections.unmodifiableMap(out);
    }

    private static void register(Map<String, Factory> out, Map<String, String> source, Factory f, String from) {
        List<String> names = new ArrayList<>();
        names.add(f.name());
        names.addAll(f.aliases());
        for (String n : names) {
            String k = key(n);
            if (k.equals(AUTO) || out.containsKey(k)) {
                throw new IllegalStateException("reasoning parser name '" + n + "' of " + from
                        + " is already registered by " + (k.equals(AUTO) ? "the registry (auto)" : source.get(k)));
            }
            out.put(k, f);
            source.put(k, from);
        }
    }

    private static final class Row implements Factory {
        private final String name;
        private final List<String> aliases;
        private final Supplier<ReasoningFilter> create;
        private final Function<GGUFTokenizer, ReasoningFilter> forModel;

        Row(String name, List<String> aliases, Supplier<ReasoningFilter> create) {
            this(name, aliases, create, null);
        }

        /** A row whose filter for a loaded model is {@code forModel}'s; {@code create} serves without a model. */
        Row(String name, List<String> aliases, Supplier<ReasoningFilter> create,
            Function<GGUFTokenizer, ReasoningFilter> forModel) {
            this.name = name;
            this.aliases = Collections.unmodifiableList(new ArrayList<>(aliases));
            this.create = create;
            this.forModel = forModel;
        }

        @Override public String name() { return name; }
        @Override public List<String> aliases() { return aliases; }
        @Override public ReasoningFilter create() { return create.get(); }

        @Override public ReasoningFilter create(GGUFTokenizer tokenizer) {
            return forModel == null ? create.get() : forModel.apply(tokenizer);
        }

        @Override public String toString() { return "ReasoningParsers.Factory(" + name + ")"; }
    }

    private static ReasoningFilter think(String start, String end, String... interrupts) {
        return ReasoningFilter.thinkTags(Collections.singletonList(start), Collections.singletonList(end),
                Arrays.asList(interrupts));
    }

    /** The Qwen3 tags (vLLM {@code qwen3_config}): a {@code <tool_call>} also ends the reasoning. */
    private static ReasoningFilter qwen3() {
        return ReasoningFilter.thinkTags(Collections.singletonList("<think>"), Arrays.asList("\n</think>", "</think>"),
                Collections.singletonList("<tool_call>"));
    }

    /**
     * HY v3 / v4 (vLLM {@code HYV3ReasoningParser}, {@code HYV4ReasoningParser}): the suffixed think tags with the
     * reasoning open from the output's start (the prompt injects the start tag), or everything content when the
     * request's {@code reasoning_effort}, else {@code defaultEffort}, is {@code no_think}.
     */
    private static ReasoningFilter hunyuan(String suffix, String defaultEffort) {
        ReasoningFilter thinking = think("<think" + suffix + ">", "</think" + suffix + ">").opensImplicitly();
        return ReasoningFilter.perRequest(vars -> NO_THINK.equals(effort(vars, defaultEffort))
                ? ReasoningFilter.none() : thinking);
    }

    /**
     * Nemotron V3 and Granite 4.2 thinking (vLLM {@code nemotron_v3_config}, {@code GraniteThinkingParser}): the
     * Qwen3 tags, a stray end tag in the content dropped and the content's leading whitespace with it; reasoning
     * open from the output's start, or none at all when {@code enable_thinking} is false, and the reasoning-to-content
     * fallback when it is false or {@code force_nonempty_content} is true.
     */
    private static ReasoningFilter nemotronV3() {
        return ReasoningFilter.perRequest(vars -> {
            ReasoningFilter f = nemotronTags(vars);
            return swapsToContent(vars) ? f.withContentFallback() : f;
        });
    }

    /**
     * Granite 4.2 thinking (vLLM {@code GraniteThinkingParser}): {@link #nemotronV3}, with the newlines the content
     * starts with dropped wherever it starts, and the reasoning-to-content fallback only for an output with no content
     * at all.
     */
    private static ReasoningFilter graniteThinking() {
        return ReasoningFilter.perRequest(vars -> {
            ReasoningFilter f = nemotronTags(vars).strippingLeadingNewlines();
            return swapsToContent(vars) ? f.withContentFallbackWhenAbsent() : f;
        });
    }

    /**
     * The Nemotron V3 tags for a request: Qwen3's with a stray end tag dropped, open from the start with thinking;
     * without it the whole output is content, a start tag included (vLLM's engine starts in its content state, which
     * has no transition into reasoning).
     */
    private static ReasoningFilter nemotronTags(Map<String, Object> vars) {
        ReasoningFilter f = qwen3().dropping(Collections.singletonList("</think>"));
        return flag(vars, ENABLE_THINKING, true) ? f.opensImplicitly() : f.contentOnly();
    }

    /** vLLM's {@code _should_force_content}: thinking off, or {@code force_nonempty_content} on. */
    private static boolean swapsToContent(Map<String, Object> vars) {
        boolean thinking = flag(vars, ENABLE_THINKING, true);
        return !thinking || flag(vars, FORCE_NONEMPTY_CONTENT, false);
    }

    /**
     * MiniCPM-V (vLLM {@code MiniCPMVParser}): the Qwen3 grammar with {@code <|im_end|>} and the reserved markers
     * 12-15 removed, a stray end tag dropped, and the tokenizer's escaped newlines recovered outside code, LaTeX and
     * tool calls. With thinking (the default) reasoning is open from the output's start; with
     * {@code enable_thinking} false the answer is content and a {@code <think>...</think>} block in it is discarded.
     */
    private static ReasoningFilter minicpmv() {
        return ReasoningFilter.perRequest(vars -> {
            List<String> dropped = new ArrayList<>(Arrays.asList("</think>", "<|im_end|>"));
            for (int i = 12; i <= 15; i++) {
                dropped.add("<reserved_" + i + ">");
                dropped.add("<|reserved_" + i + "|>");
            }
            List<String> verbatim = Arrays.asList("<tool_call>", "</tool_call>");
            if (flag(vars, ENABLE_THINKING, true)) {
                return qwen3().dropping(dropped).opensImplicitly().recoveringEscapedNewlines(verbatim);
            }
            return ReasoningFilter.thinkTags(Collections.singletonList("<think>"),
                    Collections.singletonList("</think>"), Collections.<String>emptyList())
                    .dropping(dropped).reopening().discardingReasoning().recoveringEscapedNewlines(verbatim);
        });
    }

    /**
     * PLaMo 3 (vLLM {@code plamo3_config}): reasoning open from the output's start, ended by its end tag or the end
     * of turn (both dropped wherever they appear) or the tool-request section, and reopened by a later start tag; a
     * marker the end of generation cut off ({@code <|plamo:end_}) is dropped; everything content when
     * {@code enable_thinking} is false.
     */
    private static ReasoningFilter plamo3() {
        ReasoningFilter thinking = ReasoningFilter.thinkTags(Collections.singletonList(ReasoningFilter.PLAMO_BEGIN_THINK),
                Arrays.asList(ReasoningFilter.PLAMO_END_THINK, ReasoningFilter.PLAMO_EOT),
                Collections.singletonList(ReasoningFilter.PLAMO_TOOL_REQUESTS))
                .dropping(Arrays.asList(ReasoningFilter.PLAMO_END_THINK, ReasoningFilter.PLAMO_EOT))
                .opensImplicitly().reopening().droppingTruncatedTags(PLAMO_MARKER_STEM);
        return ReasoningFilter.perRequest(vars -> flag(vars, ENABLE_THINKING, true) ? thinking : ReasoningFilter.none());
    }

    /**
     * vLLM's {@code hf} / {@code beam} reasoning parser over the model's response template, per request for
     * {@code enable_thinking}.
     *
     * @throws IllegalArgumentException when the model has no servable response template with a thinking field
     */
    private static ReasoningFilter responseTemplate(String name, GGUFTokenizer tokenizer) {
        TemplateParsing parsing = TemplateParsing.forModel(name, tokenizer, "--reasoning-parser " + name,
                ResponseTemplate.THINKING);
        return ReasoningFilter.perRequest(vars ->
                ReasoningFilter.responseTemplate(parsing.withThinking(flag(vars, ENABLE_THINKING, true))));
    }

    /** A row that reads the loaded model's metadata has no filter without one. */
    private static ReasoningFilter needsModel(String name) {
        throw new IllegalStateException("reasoning parser '" + name + "' reads the loaded model's response_template;"
                + " create it with the model's tokenizer");
    }

    /**
     * A boolean template variable, {@code byDefault} when unset.
     *
     * @throws IllegalArgumentException when it is set to a non-boolean
     */
    private static boolean flag(Map<String, Object> vars, String name, boolean byDefault) {
        Object v = vars.get(name);
        if (v == null) {
            return byDefault;
        }
        if (!(v instanceof Boolean)) {
            throw new IllegalArgumentException(name + " must be a boolean for this reasoning parser, got "
                    + v.getClass().getSimpleName() + " " + v);
        }
        return (Boolean) v;
    }

    /**
     * The {@code reasoning_effort} template variable, {@code byDefault} when unset or empty.
     *
     * @throws IllegalArgumentException when it is set to a non-string
     */
    private static String effort(Map<String, Object> vars, String byDefault) {
        Object v = vars.get(REASONING_EFFORT);
        if (v == null) {
            return byDefault;
        }
        if (!(v instanceof String)) {
            throw new IllegalArgumentException(REASONING_EFFORT + " must be a string for this reasoning parser, got "
                    + v.getClass().getSimpleName() + " " + v);
        }
        return ((String) v).isEmpty() ? byDefault : (String) v;
    }

    private static List<Factory> builtIns() {
        List<String> none = Collections.emptyList();
        return Arrays.<Factory>asList(
                new Row("deepseek_r1", Arrays.asList("step5"), ReasoningFilter::thinkTags),
                new Row("deepseek_v3", Arrays.asList("deepseek_v31"), ReasoningFilter::thinkTags),
                new Row("deepseek_v4", Arrays.asList("deepseek_v41", "deepseek_v32"),
                        () -> ReasoningFilter.thinkTags(Collections.singletonList("<think>"),
                                Collections.singletonList("</think>"),
                                Arrays.asList(ReasoningFilter.DSML_FUNCTION_CALLS, ReasoningFilter.DSML_TOOL_CALLS))),
                new Row("qwen3", Arrays.asList("qwen3_5", "qwen3_coder"),
                        () -> ReasoningFilter.thinkTags(Collections.singletonList("<think>"),
                                Arrays.asList("\n</think>", "</think>"), Collections.singletonList("<tool_call>"))),
                new Row("glm45", Arrays.asList("glm47", "glm4_moe"), ReasoningFilter::thinkTags),
                new Row("kimi_k2", none, ReasoningFilter::thinkTags),
                new Row("kimi_k3", none, () -> think("<|open|>think<|sep|>", "<|close|>think<|sep|>")),
                new Row("k2_horizon", none, () -> ReasoningFilter.thinkTags(
                        Arrays.asList("<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"),
                        Arrays.asList("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"),
                        Collections.<String>emptyList())),
                new Row("minimax_m2", none, ReasoningFilter::thinkTags),
                new Row("minimax_m2_append_think", none,
                        () -> ReasoningFilter.thinkTags().asRaw().withContentPrefix("<think>")),
                new Row("minimax_m3", none, () -> think("<mm:think>", "</mm:think>", "]<]minimax[>[<tool_call>")),
                new Row("ernie45", none, ReasoningFilter::thinkTags),
                new Row("hunyuan_a13b", none, ReasoningFilter::thinkTags),
                new Row("hy_v3", none, () -> hunyuan("", NO_THINK), tok -> hunyuan(hunyuanSuffix(tok), NO_THINK)),
                new Row("hy_v4", none, () -> hunyuan("", "high"), tok -> hunyuan(hunyuanSuffix(tok), "high")),
                new Row("olmo3", none, ReasoningFilter::thinkTags),
                new Row("step3", Arrays.asList("step3p5"), ReasoningFilter::thinkTags),
                new Row("nemotron_v3", none, ReasoningParsers::nemotronV3),
                new Row("granite_thinking_parser", none, ReasoningParsers::graniteThinking),
                new Row("minicpmv", none, ReasoningParsers::minicpmv),
                new Row("plamo3", none, ReasoningParsers::plamo3),
                new Row("inkling", none, ReasoningFilter::inkling),
                new Row("mimo", none, ReasoningFilter::thinkTags),
                new Row("poolside_v1", none, ReasoningFilter::thinkTags),
                new Row("holo2", none, ReasoningFilter::thinkTags),
                new Row("ling3", none, () -> think("<think>", "</think>", "<tool_call>")),
                new Row("seed_oss", none, () -> think("<seed:think>", "</seed:think>")),
                new Row("mistral", none, () -> think("[THINK]", "[/THINK]")),
                new Row("gemma4", none, () -> think("<|channel>thought", "<channel|>")),
                new Row("granite", none, () -> ReasoningFilter.thinkTags(
                        Arrays.asList("Here is my thought process:", "Here's my thought process:"),
                        Arrays.asList("Here is my response:", "Here's my response:"), Collections.<String>emptyList())),
                new Row("cohere_command3", Arrays.asList("cohere_command4"),
                        () -> think("<|START_THINKING|>", "<|END_THINKING|>")),
                new Row("openai_gptoss", Arrays.asList("gpt_oss", "harmony"), ReasoningFilter::harmony),
                new Row("muse", Arrays.asList("muse_glimmer", "musespark1x"), ReasoningFilter::harmony),
                new Row("hf", none, () -> needsModel("hf"), tok -> responseTemplate("hf", tok)),
                new Row("beam", none, () -> needsModel("beam"), tok -> responseTemplate("beam", tok)),
                new Row("none", Arrays.asList("identity"), ReasoningFilter::none));
    }
}
