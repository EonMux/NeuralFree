package com.eonmux.neuralfree.gguf.chat.reasoning;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
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
 * {@code olmo3}, {@code step3}, {@code nemotron_v3}, {@code mimo}, {@code poolside_v1} and {@code holo2}
 * ({@code <think>}/{@code </think>}); {@code qwen3} and {@code ling3} (also ended by a {@code <tool_call>});
 * {@code deepseek_v4} (also ended by a DSML call block); {@code minimax_m3} ({@code <mm:think>});
 * {@code seed_oss} ({@code <seed:think>}); {@code mistral} ({@code [THINK]}); {@code gemma4}
 * ({@code <|channel>thought}/{@code <channel|>}); {@code granite} ({@code Here is my thought process:} /
 * {@code Here is my response:}); {@code cohere_command3} and {@code cohere_command4}
 * ({@code <|START_THINKING|>}); {@code kimi_k3} ({@code <|open|>think<|sep|>}); {@code k2_horizon}
 * ({@code <ifm|think>} and its fast variants); {@code openai_gptoss} and {@code muse} (the harmony channels);
 * {@code none} (alias {@code identity}: everything is content).
 */
public final class ReasoningParsers {

    /** The {@code auto} name: the chat template decides. */
    public static final String AUTO = "auto";

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
        } else {
            Factory f = forName(name);
            if (f == null) {
                throw new IllegalArgumentException("reasoning parser '" + name + "' is not registered; known: "
                        + names() + " or " + AUTO);
            }
            base = f.create();
        }
        return base.withTagTokens(tokenizer::getTokenId);
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

        Row(String name, List<String> aliases, Supplier<ReasoningFilter> create) {
            this.name = name;
            this.aliases = Collections.unmodifiableList(new ArrayList<>(aliases));
            this.create = create;
        }

        @Override public String name() { return name; }
        @Override public List<String> aliases() { return aliases; }
        @Override public ReasoningFilter create() { return create.get(); }
        @Override public String toString() { return "ReasoningParsers.Factory(" + name + ")"; }
    }

    private static ReasoningFilter think(String start, String end, String... interrupts) {
        return ReasoningFilter.thinkTags(Collections.singletonList(start), Collections.singletonList(end),
                Arrays.asList(interrupts));
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
                new Row("minimax_m2", Arrays.asList("minimax_m2_append_think"), ReasoningFilter::thinkTags),
                new Row("minimax_m3", none, () -> think("<mm:think>", "</mm:think>", "]<]minimax[>[<tool_call>")),
                new Row("ernie45", none, ReasoningFilter::thinkTags),
                new Row("hunyuan_a13b", Arrays.asList("hy_v3", "hy_v4"), ReasoningFilter::thinkTags),
                new Row("olmo3", none, ReasoningFilter::thinkTags),
                new Row("step3", Arrays.asList("step3p5"), ReasoningFilter::thinkTags),
                new Row("nemotron_v3", none, ReasoningFilter::thinkTags),
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
                new Row("none", Arrays.asList("identity"), ReasoningFilter::none));
    }
}
