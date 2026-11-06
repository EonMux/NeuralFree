package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;

/**
 * vLLM's response-template parsers {@code hf} ({@code ResponseTemplateParser}) and {@code beam} ({@code BeamParser}):
 * the model's {@code response_template} ({@link GGUFTokenizer#responseTemplate}) parses the output — its thinking
 * region is the reasoning, its content region the answer, its tool-call regions the calls.
 *
 * <p>A tool guarantee needs a structural grammar of the template's format. {@code hf} has none and refuses the
 * guarantees by name ({@link #checkTools}): strict tools, a required or named {@code tool_choice}, and an
 * explicitly false {@code parallel_tool_calls}. {@code beam} derives the grammar from the template as an
 * xgrammar structural tag ({@link BeamGrammar}), which enforces them; everything else parses as {@code hf} does.
 *
 * <p>A model's template is parsed once: the first parser built for a tokenizer parses and validates its
 * {@code response_template} metadata, and every later one, per model or per request, reuses that result. Immutable.
 */
public final class TemplateParsing {

    /** The parser names. */
    public static final List<String> NAMES = Collections.unmodifiableList(Arrays.asList("hf", "beam"));

    /** Each tokenizer's parsed serving template, or the refusal of its metadata; held while the tokenizer lives. */
    private static final Map<GGUFTokenizer, Loaded> LOADED = Collections.synchronizedMap(
            new WeakHashMap<GGUFTokenizer, Loaded>());

    private final ResponseTemplate template;
    private final String name;
    private final boolean thinkingDisabled;
    private final boolean parsesReasoning;

    private TemplateParsing(ResponseTemplate template, String name, boolean thinkingDisabled,
                            boolean parsesReasoning) {
        this.template = template;
        this.name = name;
        this.thinkingDisabled = thinkingDisabled;
        this.parsesReasoning = parsesReasoning;
    }

    /** Whether {@code parser} names a response-template parser ({@code hf} or {@code beam}, any case). */
    public static boolean names(String parser) {
        return parser != null && NAMES.contains(parser.trim().toLowerCase(Locale.ROOT));
    }

    /**
     * The parser {@code name} over a model's response template (vLLM {@code validate_tokenizer_response_template}).
     *
     * @param name      {@code hf} or {@code beam}
     * @param tokenizer the model's tokenizer
     * @param flag      the selecting option, for errors ({@code --reasoning-parser hf})
     * @param field     the field the selection needs: {@link ResponseTemplate#THINKING} for a reasoning parser,
     *                  {@link ResponseTemplate#TOOL_CALLS} for a tool parser
     * @throws IllegalArgumentException when the model has no response template, the template is invalid or cannot be
     *                                  served, or it lacks {@code field}
     */
    public static TemplateParsing forModel(String name, GGUFTokenizer tokenizer, String flag, String field) {
        String key = Objects.requireNonNull(name, "name").trim().toLowerCase(Locale.ROOT);
        if (!NAMES.contains(key)) throw new IllegalArgumentException("'" + name + "' is not hf or beam");
        Loaded loaded = LOADED.computeIfAbsent(Objects.requireNonNull(tokenizer, "tokenizer"), TemplateParsing::load);
        if (loaded.serving == null) {
            String reason = loaded.error != null ? loaded.error : flag + " requires `response_template` metadata in the"
                    + " tokenizer configuration (GGUF " + GgufKeys.TOKENIZER_RESPONSE_TEMPLATE + ")";
            throw new IllegalArgumentException(reason);
        }
        ResponseTemplate serving = loaded.serving;
        if (serving.field(field) == null) {
            throw new IllegalArgumentException(flag + " requires a response_template with a " + field + " field");
        }
        return new TemplateParsing(serving, key, false, true);
    }

    /** Parses a tokenizer's {@code response_template} metadata for serving, or records why it cannot be served. */
    private static Loaded load(GGUFTokenizer tokenizer) {
        try {
            String json = tokenizer.responseTemplate();
            return json == null ? new Loaded(null, null) : new Loaded(ResponseTemplate.parse(json).forServing(), null);
        } catch (IllegalStateException | IllegalArgumentException e) {
            return new Loaded(null, "Invalid response_template metadata: " + e.getMessage());
        }
    }

    /** A tokenizer's serving template, or null with the refusal of its metadata (null error: no metadata). */
    private static final class Loaded {
        final ResponseTemplate serving;
        final String error;

        Loaded(ResponseTemplate serving, String error) {
            this.serving = serving;
            this.error = error;
        }
    }

    /**
     * The parser {@code name} over {@code template}, held to serving ({@link ResponseTemplate#forServing}).
     *
     * @throws IllegalArgumentException for a name other than hf or beam, or a template serving cannot route
     */
    public static TemplateParsing of(String name, ResponseTemplate template) {
        String key = Objects.requireNonNull(name, "name").trim().toLowerCase(Locale.ROOT);
        if (!NAMES.contains(key)) throw new IllegalArgumentException("'" + name + "' is not hf or beam");
        return new TemplateParsing(template.forServing(), key, false, true);
    }

    /** This parser for a request with thinking on or off ({@code enable_thinking}). */
    public TemplateParsing withThinking(boolean enabled) {
        return enabled == !thinkingDisabled ? this : new TemplateParsing(template, name, !enabled, parsesReasoning);
    }

    /**
     * This parser with no reasoning parser beside it (vLLM's {@code parse_reasoning} false: a tool parser {@code hf} or
     * {@code beam} with no reasoning parser): the thinking region's text streams as content, without its delimiters,
     * and no output constraint waits for its end.
     */
    public TemplateParsing withoutReasoning() {
        return parsesReasoning ? new TemplateParsing(template, name, thinkingDisabled, false) : this;
    }

    /** Whether the thinking region is parsed as the reasoning ({@link #withoutReasoning}). */
    public boolean parsesReasoning() {
        return parsesReasoning;
    }

    /** The serving template. */
    public ResponseTemplate template() {
        return template;
    }

    /** {@code hf} or {@code beam}. */
    public String name() {
        return name;
    }

    public boolean beam() {
        return name.equals("beam");
    }

    /** Whether the request turned thinking off. */
    public boolean thinkingDisabled() {
        return thinkingDisabled;
    }

    /** Whether a generation after {@code prompt} has its reasoning behind it ({@link ResponseRouting#reasoningEnded}). */
    public boolean reasoningEnded(String prompt) {
        return ResponseRouting.reasoningEnded(template, prompt, thinkingDisabled);
    }

    /**
     * The tool guarantees of a request {@code hf}, parsing its tools, cannot enforce (vLLM
     * {@code ResponseTemplateParser._constrain_request}); {@code beam} enforces them with its grammar
     * ({@link BeamGrammar}) and refuses none here. A {@code parallel_tool_calls} the client did not set is no
     * guarantee: only its explicit false is refused (vLLM {@code request.parallel_tool_calls is False}), the
     * template's default leaves the calls to the parser.
     *
     * @throws IllegalArgumentException naming the guarantee and the request parameter that asked for it
     */
    public void checkTools(ChatRequest chat, ParallelToolCalls parallelToolCalls) {
        List<ChatTool> tools = chat.tools();
        ChatRequest.ToolChoice.Kind choice = chat.toolChoice() == null ? ChatRequest.ToolChoice.Kind.AUTO
                : chat.toolChoice().kind();
        if (beam() || tools.isEmpty() || choice == ChatRequest.ToolChoice.Kind.NONE) return;
        for (ChatTool t : tools) {
            if (Boolean.TRUE.equals(t.strict())) throw refusal("strict tools", "tools");
        }
        if (choice == ChatRequest.ToolChoice.Kind.REQUIRED) throw refusal("required tool choice", "tool_choice");
        if (choice == ChatRequest.ToolChoice.Kind.FUNCTION) throw refusal("named tool choice", "tool_choice");
        if (parallelToolCalls.singleAsked()) {
            throw refusal("parallel_tool_calls=False", "parallel_tool_calls");
        }
    }

    private IllegalArgumentException refusal(String guarantee, String parameter) {
        return new IllegalArgumentException("response_template parsing (" + name + ") cannot guarantee " + guarantee
                + " without a format-compatible structural grammar; request parameter '" + parameter + "'");
    }

    @Override public String toString() {
        return name + " " + template + (thinkingDisabled ? " thinking-off" : "")
                + (parsesReasoning ? "" : " no-reasoning");
    }
}
