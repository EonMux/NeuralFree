package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.chat.response.ResponseRouting;
import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Splits streamed model output into content, reasoning and tool-call text.
 *
 * <p>The filter itself is an immutable configuration; {@link #newState} creates the per-generation
 * streaming {@link State}, which is a {@link TokenCallback} that routes each piece of text to a
 * {@link Sink} together with its {@link Channel}. Three modes exist:
 * <ul>
 *   <li>{@link Mode#THINK_TAGS}: an optional leading {@code start ... end} reasoning block
 *       (llama.cpp common/chat-auto-parser-generator.cpp analyze_reasoning::build_parser:
 *       {@code optional(optspace(start) + reasoning(until(end)) + optspace(end))}); interrupt tags
 *       such as a tool-call opener also end reasoning but stay in the content.</li>
 *   <li>{@link Mode#HARMONY_CHANNELS}: header/body messages ({@code <|start|>header<|message|>body}
 *       closed by a terminator) in two dialects. OpenAI harmony: the {@code analysis} channel is
 *       reasoning, {@code final} and recipient-less {@code commentary} are content, and a {@code to=}
 *       recipient other than {@code assistant} marks a tool call (llama.cpp common/parsers/gpt-oss.cpp:68-127).
 *       Muse recipients: {@code to=self} is reasoning, {@code to=user}, {@code to=commentary} and no
 *       recipient are content, any other recipient is a tool call, and {@code <|eom|>}/{@code <|eot|>}
 *       end a message (llama.cpp common/parsers/muse-spark.cpp:61-76, muse-glimmer.cpp:59-67). The raw variant
 *       ({@link #asRaw}) still routes tool calls but keeps reasoning inline in the content.</li>
 *   <li>{@link Mode#NONE}: everything is content.</li>
 * </ul>
 * Tags are matched in the decoded text; {@link #withTagTokens} additionally maps special-token ids
 * to their tag text so the filter works even when the detokenizer hides special tokens.
 *
 * <p>Tag filters take further behaviors of vLLM's per-family parsers: reasoning that opens at the
 * output's first token ({@link #opensImplicitly}), start tags that open further blocks after the
 * content ({@link #reopening}), markers removed wherever they appear ({@link #dropping}), discarded
 * reasoning ({@link #discardingReasoning}), reasoning that becomes the content of an answer without
 * one ({@link #withContentFallback}), recovery of newlines a tokenizer decodes as {@code \n}
 * ({@link #recoveringEscapedNewlines}), the Inkling block layout ({@link #inkling}) and a checkpoint's response
 * template ({@link #responseTemplate(TemplateParsing)}). A raw filter
 * can prefix its content ({@link #withContentPrefix}). A filter built by {@link #perRequest} picks
 * its configuration per request from the chat template variables ({@link #forRequest}).
 */
public final class ReasoningFilter {

    /** Parsing mode. */
    public enum Mode { THINK_TAGS, HARMONY_CHANNELS, NONE }

    /** Destination of a piece of output text. */
    public enum Channel { CONTENT, REASONING, TOOL_CALL }

    /** Receives routed text. {@code tokenId} is the token that released the text, or -1 when {@link State#finish()} did. */
    @FunctionalInterface
    public interface Sink {
        void accept(Channel channel, int tokenId, String text);

        /**
         * Routed text with the step's logprobs capture: the accepting token's {@code LogprobCapture} (the sampled
         * token first, then its best alternatives or the requested ids, {@code SequenceRun.Step.logprobs()}) when
         * the request captures them, empty otherwise. The default drops the capture and routes the text only; the
         * logprobs-capturing runs ({@code request.wantsLogprobs()}) take this overload, and a token whose text is
         * fully held back arrives as an empty {@code text} on {@link Channel#CONTENT} with its capture.
         */
        default void accept(Channel channel, int tokenId, String text, com.eonmux.neuralfree.gguf.engine.TokenScore[] logprobs) {
            accept(channel, tokenId, text);
        }

        /**
         * A harmony or Muse message header named the tool {@code recipient} (gpt-oss {@code to=functions.f}); the
         * message body that follows arrives on {@link Channel#TOOL_CALL} until the next header. Called in order with
         * the routed text, before the body's first piece. The default ignores it.
         */
        default void toolCallStarted(String recipient) {
        }

        /**
         * One more generated token joined the output, before any of its text routes: called once per accepted
         * token (an end-of-generation token is not one), so a sink counts the tokens behind each piece of text.
         * The default ignores it.
         */
        default void stepped(int tokenId) {
        }

        /**
         * The output ended without content and its reasoning follows again on {@link Channel#CONTENT}
         * ({@link #withContentFallback}): called once, at {@link State#finish}, right before that content. The stream
         * keeps both copies; an assembled message carries the reasoning as its content only, as vLLM's non-streamed
         * swap does. The default ignores it.
         */
        default void contentFallback() {
        }
    }

    /** Harmony message opener. */
    public static final String H_START = "<|start|>";
    /** Harmony channel marker ({@code <|channel|>analysis}). */
    public static final String H_CHANNEL = "<|channel|>";
    /** Harmony header/body separator. */
    public static final String H_MESSAGE = "<|message|>";
    /** Harmony message terminator: more messages follow. */
    public static final String H_END = "<|end|>";
    /** Harmony turn terminator: the final message. */
    public static final String H_RETURN = "<|return|>";
    /** Harmony tool-call terminator: the assistant awaits the tool result. */
    public static final String H_CALL = "<|call|>";
    static final String H_CONSTRAIN = "<|constrain|>";
    /** Muse message terminators: more messages follow ({@code <|eom|>}) or the turn ends ({@code <|eot|>}). */
    public static final String M_EOM = "<|eom|>";
    public static final String M_EOT = "<|eot|>";
    private static final List<String> HARMONY_TAGS = Collections.unmodifiableList(Arrays.asList(
            H_START, H_CHANNEL, H_MESSAGE, H_END, H_RETURN, H_CALL, H_CONSTRAIN, M_EOM, M_EOT));

    /** Apriel 1.6 reasoning: the generation prompt opens it, the final-response marker closes it. */
    private static final String APRIEL_REASONING_START = "Here are my reasoning steps:";
    private static final String APRIEL_FINAL_RESPONSE = "[BEGIN FINAL RESPONSE]";

    /** DeepSeek V3.2 / V4 DSML tool-call block openers; the DSML delimiters are fullwidth vertical bars (U+FF5C). */
    /** DeepSeek V3.2 / V4 DSML function-call block opener, which also ends reasoning. */
    public static final String DSML_FUNCTION_CALLS = "<\uFF5CDSML\uFF5Cfunction_calls>";
    /** DeepSeek V3.2 / V4 DSML tool-call block opener, which also ends reasoning. */
    public static final String DSML_TOOL_CALLS = "<\uFF5CDSML\uFF5Ctool_calls>";

    /** Inkling model-message header opener (vLLM vllm/parser/inkling.py). */
    public static final String INKLING_MESSAGE = "<|message_model|>";
    /** Inkling text block marker. */
    public static final String INKLING_TEXT = "<|content_text|>";
    /** Inkling thinking block marker: its body is reasoning. */
    public static final String INKLING_THINKING = "<|content_thinking|>";
    /** Inkling JSON tool-call block marker. */
    public static final String INKLING_TOOL_JSON = "<|content_invoke_tool_json|>";
    /** Inkling raw tool-text block marker: its body is visible text. */
    public static final String INKLING_TOOL_TEXT = "<|content_invoke_tool_text|>";
    /** Inkling tool-error block marker: its body is visible text. */
    public static final String INKLING_TOOL_ERROR = "<|content_tool_error|>";
    /** Inkling block terminator. */
    public static final String INKLING_END = "<|end_message|>";
    /** Inkling standalone end of sampling, which also ends a block. */
    public static final String INKLING_END_SAMPLING = "<|content_model_end_sampling|>";

    /** PLaMo 3 reasoning opener (vLLM vllm/parser/plamo3.py). */
    public static final String PLAMO_BEGIN_THINK = "<|plamo:begin_think:plamo|>";
    /** PLaMo 3 reasoning closer. */
    public static final String PLAMO_END_THINK = "<|plamo:end_think:plamo|>";
    /** PLaMo 3 end of turn, which also ends reasoning. */
    public static final String PLAMO_EOT = "<|plamo:tag|>";
    /** PLaMo 3 tool-request section opener, which ends reasoning and stays in the content. */
    public static final String PLAMO_TOOL_REQUESTS = "<|plamo:begin_tool_requests:plamo|>";

    /** The Hunyuan template's suffix assignment, {@code {%- set HYTK = ':opensource' %}}. */
    private static final Pattern HYTK_ASSIGNMENT = Pattern.compile("set\\s+HYTK\\s*=\\s*['\"]([^'\"\\s>]*)['\"]");

    /** Single special-token candidates inside a tag: {@code <...>} and {@code [...]} pieces. */
    private static final Pattern TOKEN_PIECE = Pattern.compile("<[^<>]*>|\\[[^\\[\\]]*\\]");

    private static final ReasoningFilter NONE = new ReasoningFilter(Mode.NONE, Collections.<String>emptyList(),
            Collections.<String>emptyList(), Collections.<String>emptyList(), Collections.<Integer, String>emptyMap(),
            false, Options.DEFAULT);

    private final Mode mode;
    private final List<String> startTags;
    private final List<String> endTags;
    private final List<String> interruptTags;
    private final Map<Integer, String> tagTokens;
    private final boolean reasoningInline;
    private final Options options;

    private ReasoningFilter(Mode mode, List<String> startTags, List<String> endTags, List<String> interruptTags,
                            Map<Integer, String> tagTokens, boolean reasoningInline, Options options) {
        this.mode = mode;
        this.startTags = Collections.unmodifiableList(new ArrayList<>(startTags));
        this.endTags = Collections.unmodifiableList(new ArrayList<>(endTags));
        this.interruptTags = Collections.unmodifiableList(new ArrayList<>(interruptTags));
        this.tagTokens = Collections.unmodifiableMap(new HashMap<>(tagTokens));
        this.reasoningInline = reasoningInline;
        this.options = options;
    }

    /** A copy of this filter with {@code o} as its options. */
    private ReasoningFilter with(Options o) {
        return new ReasoningFilter(mode, startTags, endTags, interruptTags, tagTokens, reasoningInline, o);
    }

    /**
     * The behaviors of a filter beyond its tags and mode. Immutable once built: each {@code with...} method returns a
     * modified copy. {@code verbatimSpans} is null without newline recovery, else the open/close pairs whose text it
     * keeps as written; {@code truncatedTagStem} is null unless a tag cut off by the end of generation is dropped.
     */
    private static final class Options implements Cloneable {
        static final Options DEFAULT = new Options();

        private boolean implicitOpen;
        private boolean contentOnly;
        private boolean reopens;
        private boolean dropsReasoning;
        private boolean contentFallback;
        private boolean fallbackOnlyWithoutContent;
        private boolean stripsLeadingNewlines;
        private List<String> droppedTags = Collections.<String>emptyList();
        private String contentPrefix = "";
        private List<String> verbatimSpans;
        private String truncatedTagStem;
        private boolean inklingBlocks;
        private TemplateParsing responseTemplate;
        private Variants variants;

        private Options copy() {
            try {
                return (Options) clone();
            } catch (CloneNotSupportedException e) {
                throw new AssertionError(e);
            }
        }

        Options implicitOpen() {
            Options o = copy();
            o.implicitOpen = true;
            return o;
        }

        Options contentOnly() {
            Options o = copy();
            o.contentOnly = true;
            return o;
        }

        Options reopens() {
            Options o = copy();
            o.reopens = true;
            return o;
        }

        Options dropsReasoning() {
            Options o = copy();
            o.dropsReasoning = true;
            return o;
        }

        Options contentFallback(boolean onlyWithoutContent) {
            Options o = copy();
            o.contentFallback = true;
            o.fallbackOnlyWithoutContent = onlyWithoutContent;
            return o;
        }

        Options stripsLeadingNewlines() {
            Options o = copy();
            o.stripsLeadingNewlines = true;
            return o;
        }

        Options droppedTags(List<String> tags) {
            List<String> all = new ArrayList<>(droppedTags);
            for (String t : tags) {
                if (!all.contains(t)) all.add(t);
            }
            Options o = copy();
            o.droppedTags = Collections.unmodifiableList(all);
            return o;
        }

        Options contentPrefix(String prefix) {
            Options o = copy();
            o.contentPrefix = prefix;
            return o;
        }

        Options verbatimSpans(List<String> spans) {
            Options o = copy();
            o.verbatimSpans = spans == null ? null : Collections.unmodifiableList(new ArrayList<>(spans));
            return o;
        }

        Options truncatedTagStem(String stem) {
            Options o = copy();
            o.truncatedTagStem = stem;
            return o;
        }

        Options inklingBlocks() {
            Options o = copy();
            o.inklingBlocks = true;
            return o;
        }

        Options responseTemplate(TemplateParsing template) {
            Options o = copy();
            o.responseTemplate = template;
            return o;
        }

        Options variants(Variants v) {
            Options o = copy();
            o.variants = v;
            return o;
        }

        @Override public String toString() {
            StringBuilder b = new StringBuilder();
            if (implicitOpen) b.append(" implicit-open");
            if (contentOnly) b.append(" content-only");
            if (reopens) b.append(" reopening");
            if (dropsReasoning) b.append(" reasoning-discarded");
            if (contentFallback) b.append(fallbackOnlyWithoutContent ? " content-fallback(absent)" : " content-fallback");
            if (stripsLeadingNewlines) b.append(" content-newlines-stripped");
            if (!droppedTags.isEmpty()) b.append(" dropping ").append(droppedTags);
            if (!contentPrefix.isEmpty()) b.append(" content-prefix ").append(contentPrefix);
            if (verbatimSpans != null) b.append(" escaped-newlines");
            if (truncatedTagStem != null) b.append(" truncated-tags ").append(truncatedTagStem);
            if (inklingBlocks) b.append(" inkling-blocks");
            if (responseTemplate != null) b.append(" response-template");
            if (variants != null) b.append(" per-request");
            return b.toString();
        }
    }

    /**
     * The per-request selection of a {@link #perRequest} filter: the selector, then the derivations
     * ({@link #withTagTokens}, {@link #asRaw}, ...) applied to the default since, replayed on each selected filter.
     */
    private static final class Variants {
        final Function<Map<String, Object>, ReasoningFilter> select;
        final List<UnaryOperator<ReasoningFilter>> derivations;

        Variants(Function<Map<String, Object>, ReasoningFilter> select, List<UnaryOperator<ReasoningFilter>> derivations) {
            this.select = select;
            this.derivations = Collections.unmodifiableList(new ArrayList<>(derivations));
        }

        Variants then(UnaryOperator<ReasoningFilter> derivation) {
            List<UnaryOperator<ReasoningFilter>> d = new ArrayList<>(derivations);
            d.add(derivation);
            return new Variants(select, d);
        }

        ReasoningFilter resolve(Map<String, Object> variables) {
            ReasoningFilter f = Objects.requireNonNull(select.apply(variables), "the per-request selector returned null");
            if (f.options.variants != null) {
                f = f.with(f.options.variants(null));
            }
            for (UnaryOperator<ReasoningFilter> d : derivations) {
                f = d.apply(f);
            }
            return f;
        }
    }

    /** {@code derived}, carrying this filter's per-request selection with {@code derivation} appended. */
    private ReasoningFilter carry(ReasoningFilter derived, UnaryOperator<ReasoningFilter> derivation) {
        Variants v = options.variants;
        if (v == null) return derived;
        return new ReasoningFilter(derived.mode, derived.startTags, derived.endTags, derived.interruptTags,
                derived.tagTokens, derived.reasoningInline, derived.options.variants(v.then(derivation)));
    }

    // ================================================================= factories

    /** Passes all text through as content. */
    public static ReasoningFilter none() { return NONE; }

    /**
     * The raw-output variant of this filter (llama.cpp's {@code reasoning_format none}). A tag filter
     * becomes {@link Mode#NONE}: every piece routes to content, but the special tokens of this
     * filter's tags keep rendering as their tag text, so the raw generated text keeps its tags. A
     * harmony filter keeps parsing messages, so tool calls still route to {@link Channel#TOOL_CALL},
     * but a reasoning message routes to content: a gpt-oss one whole, from its first
     * {@code <|channel|>} through its terminator ({@code <|channel|>analysis<|message|>...<|end|>}),
     * a Muse {@code to=self} one as its body (llama.cpp common/parsers/gpt-oss.cpp and
     * muse-spark.cpp with {@code extract_reasoning} false).
     */
    public ReasoningFilter asRaw() {
        ReasoningFilter raw = mode == Mode.HARMONY_CHANNELS
                ? new ReasoningFilter(mode, startTags, endTags, interruptTags, tagTokens, true, options)
                : new ReasoningFilter(Mode.NONE, startTags, endTags, interruptTags, tagTokens, false, options);
        return carry(raw, ReasoningFilter::asRaw);
    }

    /** {@code <think>...</think>} reasoning. */
    public static ReasoningFilter thinkTags() {
        return thinkTags(Collections.singletonList("<think>"), Collections.singletonList("</think>"),
                Collections.<String>emptyList());
    }

    /**
     * Tag-delimited reasoning.
     *
     * @param starts     tags that open reasoning (at least one)
     * @param ends       tags that close reasoning and are dropped (at least one)
     * @param interrupts tags that close reasoning and remain part of the content (may be empty)
     * @throws IllegalArgumentException for missing, null or empty tags
     */
    public static ReasoningFilter thinkTags(List<String> starts, List<String> ends, List<String> interrupts) {
        requireTags("start", starts, false);
        requireTags("end", ends, false);
        requireTags("interrupt", interrupts, true);
        return new ReasoningFilter(Mode.THINK_TAGS, starts, ends, interrupts, Collections.<Integer, String>emptyMap(),
                false, Options.DEFAULT);
    }

    /**
     * Reasoning that opens at the output's first token and closes at {@code end}, the delimiter style of a template
     * whose reasoning has no opening marker (llama.cpp {@code analyze_reasoning} with an empty start tag).
     *
     * @param end        the tag that closes reasoning and is dropped
     * @param interrupts tags that close reasoning and remain part of the content (may be empty)
     * @throws IllegalArgumentException for a missing, null or empty tag
     */
    public static ReasoningFilter implicitReasoning(String end, List<String> interrupts) {
        requireTags("end", Collections.singletonList(end), false);
        requireTags("interrupt", interrupts, true);
        ReasoningFilter filter = new ReasoningFilter(Mode.THINK_TAGS, Collections.<String>emptyList(),
                Collections.singletonList(end), interrupts, Collections.<Integer, String>emptyMap(), false,
                Options.DEFAULT);
        return filter.with(filter.options.implicitOpen().droppedTags(Collections.<String>emptyList()));
    }

    /** Header/body messages: OpenAI harmony channels (gpt-oss and the harmony dialects) and Muse recipients. */
    public static ReasoningFilter harmony() {
        return new ReasoningFilter(Mode.HARMONY_CHANNELS, Collections.<String>emptyList(),
                Collections.<String>emptyList(),
                Collections.<String>emptyList(), Collections.<Integer, String>emptyMap(), false, Options.DEFAULT);
    }

    /**
     * The Inkling block layout (vLLM vllm/parser/inkling.py): {@code <|message_model|>} opens a header — the
     * optional function name — that a content-kind marker ends, both dropped. A {@code <|content_thinking|>} block's
     * body is reasoning; a text, raw tool-text or tool-error block's body is content without its markers; a
     * {@code <|content_invoke_tool_json|>} block passes whole (marker, body and end marker) to the content, where the
     * {@code inkling} tool parser reads the call, and it also ends an unclosed thinking block. Blocks end with
     * {@code <|end_message|>} or {@code <|content_model_end_sampling|>}, repeat in any order, and text between them is
     * content. Generation starts in a header (the prompt prefills {@code <|message_model|>}); a prompt whose last
     * Inkling marker opens a thinking or text block starts inside it.
     */
    public static ReasoningFilter inkling() {
        return new ReasoningFilter(Mode.THINK_TAGS, Collections.singletonList(INKLING_THINKING),
                Arrays.asList(INKLING_END, INKLING_END_SAMPLING), Collections.singletonList(INKLING_TOOL_JSON),
                Collections.<Integer, String>emptyMap(), false,
                Options.DEFAULT.droppedTags(Arrays.asList(INKLING_MESSAGE, INKLING_TEXT, INKLING_TOOL_TEXT,
                        INKLING_TOOL_ERROR)).inklingBlocks());
    }

    /**
     * The regions of a checkpoint's {@code response_template} (vLLM's {@code hf} and {@code beam} parsers,
     * {@link ResponseRouting}): the thinking region is reasoning, the content region content, and a tool-call region
     * passes whole to the content, where the template's tool parser reads it. The start and end tags are the thinking
     * region's literal opener and closer, empty for a regex or implicit one.
     */
    public static ReasoningFilter responseTemplate(TemplateParsing parsing) {
        Objects.requireNonNull(parsing, "parsing");
        ResponseTemplate.Field thinking = parsing.template().field(ResponseTemplate.THINKING);
        List<String> starts = thinking == null || thinking.openLiterals() == null ? Collections.<String>emptyList()
                : thinking.openLiterals();
        List<String> ends = thinking == null || thinking.closeLiterals() == null ? Collections.<String>emptyList()
                : thinking.closeLiterals();
        return new ReasoningFilter(Mode.THINK_TAGS, starts, ends, Collections.<String>emptyList(),
                Collections.<Integer, String>emptyMap(), false, Options.DEFAULT.responseTemplate(parsing));
    }

    /** The response-template parsing of a {@link #responseTemplate(TemplateParsing)} filter, else null. */
    public TemplateParsing responseTemplate() { return options.responseTemplate; }

    /**
     * Whether this filter routes any output to the reasoning: false for {@link Mode#NONE}, a {@link #contentOnly}
     * filter, and a response-template filter without reasoning parsing ({@link TemplateParsing#withoutReasoning}).
     */
    public boolean parsesReasoning() {
        if (mode == Mode.NONE || options.contentOnly) return false;
        return options.responseTemplate == null || options.responseTemplate.parsesReasoning();
    }

    /**
     * A copy whose reasoning opens at the output's first token (vLLM parsers whose engine starts in its reasoning
     * state: the start tag is the prompt's, not the output's), unless the prompt's generation prefix closes a
     * reasoning block itself, as a template's empty {@code <think></think>} with thinking off does. A start tag the
     * output still writes inside the reasoning is dropped.
     *
     * @throws IllegalStateException when this is not a tag filter
     */
    public ReasoningFilter opensImplicitly() {
        requireTagMode("opensImplicitly");
        return with(options.implicitOpen().droppedTags(startTags));
    }

    /**
     * A copy that parses no reasoning: the output is content from its first token and a start tag in it stays text,
     * while the dropped tags are still removed (vLLM tag parsers with {@code enable_thinking} false, whose engine
     * starts in its content state with no transition into reasoning, as Nemotron V3's does).
     *
     * @throws IllegalStateException when this is not a tag filter
     */
    public ReasoningFilter contentOnly() {
        requireTagMode("contentOnly");
        return with(options.contentOnly());
    }

    /**
     * A copy in which a start tag after the reasoning opens another reasoning block, as vLLM's
     * {@code (CONTENT, THINK_START) -> REASONING} transitions do; without it the text after the first block is
     * content whatever it holds.
     *
     * @throws IllegalStateException when this is not a tag filter
     */
    public ReasoningFilter reopening() {
        requireTagMode("reopening");
        return with(options.reopens());
    }

    /**
     * A copy that removes {@code tags} from the output wherever they appear outside the tags this filter acts on
     * (vLLM pass-through terminals such as MiniCPM-V's {@code <|im_end|>}, and stray end tags in the content).
     * Their special tokens render as their text ({@link #withTagTokens}), so they are removed when hidden too.
     *
     * @throws IllegalArgumentException for a null, empty or null-holding list
     * @throws IllegalStateException    when this is not a tag filter
     */
    public ReasoningFilter dropping(List<String> tags) {
        requireTagMode("dropping");
        requireTags("dropped", tags, false);
        return with(options.droppedTags(tags));
    }

    /**
     * A copy whose reasoning is discarded: neither reasoning nor content (MiniCPM-V with thinking off, where a stray
     * {@code <think>...</think>} block is dropped, not printed).
     *
     * @throws IllegalStateException when this is not a tag filter
     */
    public ReasoningFilter discardingReasoning() {
        requireTagMode("discardingReasoning");
        return with(options.dropsReasoning());
    }

    /**
     * A copy that, when the output ends without non-whitespace content, releases its reasoning again as the content
     * (vLLM's Nemotron V3 reasoning-to-content swap, streamed as {@code get_streaming_fallback_content} does: the
     * reasoning keeps its channel and the whole of it follows on the content at {@link State#finish}).
     *
     * @throws IllegalStateException when this is not a tag filter
     */
    public ReasoningFilter withContentFallback() {
        requireTagMode("withContentFallback");
        return with(options.contentFallback(false));
    }

    /**
     * {@link #withContentFallback} that releases the reasoning again only when the output has no content at all, not
     * even whitespace (vLLM's Granite thinking parser, whose swap needs the content truly absent).
     *
     * @throws IllegalStateException when this is not a tag filter
     */
    public ReasoningFilter withContentFallbackWhenAbsent() {
        requireTagMode("withContentFallbackWhenAbsent");
        return with(options.contentFallback(true));
    }

    /**
     * A copy that drops the newlines the content starts with, wherever the content starts: after an end tag or from
     * the output's first token when no reasoning precedes it (vLLM's Granite thinking parser, whose template writes
     * {@code \n</think>\n}).
     *
     * @throws IllegalStateException when this is not a tag filter
     */
    public ReasoningFilter strippingLeadingNewlines() {
        requireTagMode("strippingLeadingNewlines");
        return with(options.stripsLeadingNewlines());
    }

    /**
     * A copy that drops a tag the end of generation cut off: text held back at {@link State#finish} as a proper
     * prefix of one of the filter's tags, starting with {@code stem}, is removed instead of released (vLLM's PLaMo 3
     * parser, which strips a truncated {@code <|plamo:end_} from the reasoning).
     *
     * @throws IllegalArgumentException for a null or empty stem
     * @throws IllegalStateException    when this is not a tag filter
     */
    public ReasoningFilter droppingTruncatedTags(String stem) {
        if (stem == null || stem.isEmpty()) throw new IllegalArgumentException("truncated tag stem must be non-empty");
        requireTagMode("droppingTruncatedTags");
        return with(options.truncatedTagStem(stem));
    }

    /**
     * A copy that writes {@code prefix} before the content: on the first token's text, or alone at {@link State#finish}
     * when no token came (vLLM's {@code minimax_m2_append_think}, which prepends the {@code <think>} the prompt
     * opened).
     *
     * @throws IllegalArgumentException for a null or empty prefix
     * @throws IllegalStateException    when this is not a {@link Mode#NONE} filter
     */
    public ReasoningFilter withContentPrefix(String prefix) {
        if (prefix == null || prefix.isEmpty()) throw new IllegalArgumentException("content prefix must be non-empty");
        if (mode != Mode.NONE) {
            throw new IllegalStateException("withContentPrefix applies to a raw (Mode.NONE) filter, not " + mode);
        }
        return with(options.contentPrefix(prefix));
    }

    /**
     * A copy that recovers newlines from the escape text a tokenizer decodes them as ({@code \n}, {@code \r},
     * {@code \r\n}), in the reasoning and in the content (vLLM's MiniCPM-V {@code EscapedNewlineNormalizer}). Code
     * blocks, inline code, LaTeX spans and doubled escapes keep their text, and so does every span from an
     * {@code open} to its {@code close} in {@code verbatimSpans} (pairs; a tool call's arguments are parsed as
     * written).
     *
     * @throws IllegalArgumentException for an odd-length list or an empty marker
     */
    public ReasoningFilter recoveringEscapedNewlines(List<String> verbatimSpans) {
        requireTags("verbatim span", verbatimSpans, true);
        if (verbatimSpans.size() % 2 != 0) {
            throw new IllegalArgumentException("verbatim spans are open/close pairs; got " + verbatimSpans.size()
                    + " markers");
        }
        return with(options.verbatimSpans(verbatimSpans));
    }

    /**
     * A filter that picks its configuration per request: {@code select} maps a request's chat template variables
     * ({@code enable_thinking}, {@code reasoning_effort} and the {@code chat_template_kwargs}) to the filter that
     * request parses with, as vLLM's parsers read their {@code chat_template_kwargs}. The result behaves as the
     * selection of a request that sets none; {@link #forRequest} selects for one request.
     *
     * @throws NullPointerException when {@code select} is null or returns null for no variables
     */
    public static ReasoningFilter perRequest(Function<Map<String, Object>, ReasoningFilter> select) {
        Objects.requireNonNull(select, "select");
        Variants v = new Variants(select, Collections.<UnaryOperator<ReasoningFilter>>emptyList());
        ReasoningFilter defaults = v.resolve(Collections.<String, Object>emptyMap());
        return defaults.with(defaults.options.variants(v));
    }

    /**
     * The filter one request parses with: this filter's per-request selection ({@link #perRequest}) for the
     * request's chat template variables, with every derivation of this filter applied; this filter itself when it
     * selects nothing per request.
     *
     * @throws IllegalArgumentException when a variable the selection reads has the wrong type
     */
    public ReasoningFilter forRequest(Map<String, ?> variables) {
        Variants v = options.variants;
        if (v == null) return this;
        Map<String, Object> vars = variables == null ? Collections.<String, Object>emptyMap()
                : Collections.unmodifiableMap(new HashMap<String, Object>(variables));
        return v.resolve(vars);
    }

    /** Whether this filter picks its configuration per request ({@link #perRequest}). */
    public boolean selectsPerRequest() { return options.variants != null; }

    private void requireTagMode(String what) {
        if (mode != Mode.THINK_TAGS || options.inklingBlocks || options.responseTemplate != null) {
            throw new IllegalStateException(what + " applies to a tag filter (Mode.THINK_TAGS), not " + this);
        }
    }

    private static void requireTags(String what, List<String> tags, boolean mayBeEmpty) {
        if (tags == null) throw new IllegalArgumentException(what + " tags must not be null");
        if (!mayBeEmpty && tags.isEmpty()) throw new IllegalArgumentException("at least one " + what + " tag is required");
        for (String t : tags) {
            if (t == null || t.isEmpty()) throw new IllegalArgumentException(what + " tags must be non-empty strings");
        }
    }

    /**
     * Picks the filter for a chat template, mirroring llama.cpp's template dispatch
     * (llama.cpp common/chat.cpp common_chat_template_detect_params_init) and the reasoning tags of each
     * dedicated parser. Templates handled by the autoparser use the reasoning delimiters it derives for
     * them: Apriel 1.6's forced-open reasoning ended by its final-response marker, otherwise the think
     * tags the template contains. A null template, or one without reasoning delimiters, yields {@link #none()}.
     */
    public static ReasoningFilter forChatTemplate(String template) {
        if (template == null) return NONE;
        String s = template;
        if (s.contains("[SYSTEM_PROMPT]") && s.contains("[TOOL_CALLS]") && s.contains("[ARGS]") && !s.contains("[CALL_ID]")) {
            return think("[THINK]", "[/THINK]");                              // parsers/ministral3.cpp:52-53
        }
        if (s.contains("chat_format=llm-jp-harmony-v1") || s.contains(H_CHANNEL)) {
            return harmony();                                                 // parsers/gpt-oss.cpp:43-44
        }
        if (s.contains("to=commentary") && s.contains(M_EOM)) {
            return harmony();                                                 // chat.cpp:1121-1124, parsers/muse-spark.cpp:61-76
        }
        if (s.contains("<atem:function_calls>") && s.contains(M_EOM)) {
            return harmony();                                                 // chat.cpp:1127-1130, parsers/muse-glimmer.cpp:59-67
        }
        if (s.contains("<|tool_calls_section_begin|>") && s.contains("<|tool_call_begin|>")) {
            return think("<think>", "</think>");                              // parsers/kimi-k2.cpp:33-34
        }
        if (s.contains("<|open|>") && s.contains("<|close|>") && s.contains("<|end_of_msg|>")) {
            return think("<|open|>think<|sep|>", "<|close|>think<|sep|>");    // parsers/kimi-k3.cpp:19-20
        }
        if (s.contains("<|ifm|im_start|>") && s.contains("<ifm|tool_calls>")) {
            return thinkTags(Arrays.asList("<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"),
                    Arrays.asList("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"),
                    Collections.<String>emptyList());                          // parsers/k2-horizon.cpp:30-36
        }
        if (s.contains("<role>ASSISTANT</role>") && s.contains("<arg_key>")) {
            return thinkTags(Collections.singletonList("<think>"), Collections.singletonList("</think>"),
                    Collections.singletonList("<tool_call>"));                 // parsers/ling3.cpp:36-39
        }
        if (s.contains("<|START_TEXT|>") && s.contains("<|START_ACTION|>")) {
            return think("<|START_THINKING|>", "<|END_THINKING|>");           // parsers/cohere2moe.cpp:28-29
        }
        if (s.contains("]<]minimax[>[") && s.contains("<tool_call>") && s.contains("<invoke name=")) {
            return thinkTags(Collections.singletonList("<mm:think>"), Collections.singletonList("</mm:think>"),
                    Collections.singletonList("]<]minimax[>[<tool_call>"));     // parsers/minimax-m3.cpp:11-19
        }
        if (s.contains("dsml_token") && s.contains("DSML") && (s.contains("function_calls") || s.contains("tool_calls"))) {
            return thinkTags(Collections.singletonList("<think>"), Collections.singletonList("</think>"),
                    Arrays.asList(DSML_FUNCTION_CALLS, DSML_TOOL_CALLS));         // parsers/deepseek.cpp:93-120
        }
        if (s.contains("'<|tool_call>call:'")) {
            return think("<|channel>thought", "<channel|>");                  // parsers/gemma4.cpp:173-174
        }
        if (s.contains("<tool_call>") && s.contains("<function=") && s.contains("<parameter=")
                && !s.contains("'<tool_call><function=' ~ tool_call.name ~ '>'") && s.contains("<think>")) {
            return thinkTags(Collections.singletonList("<think>"), Arrays.asList("\n</think>", "</think>"),
                    Collections.singletonList("<tool_call>"));                 // chat.cpp:1245-1252, parsers/qwen3-coder.cpp:13-28
        }
        if (s.contains(APRIEL_REASONING_START) && s.contains(APRIEL_FINAL_RESPONSE)) {
            // chat-diff-analyzer.cpp:160-166; the generation prompt opens reasoning and "\n[BEGIN FINAL RESPONSE]\n"
            // closes it (models/templates/Apriel-1.6-15b-Thinker-fixed.jinja:21, :106, :167-168)
            return thinkTags(Collections.singletonList(APRIEL_REASONING_START),
                    Arrays.asList("\n" + APRIEL_FINAL_RESPONSE, APRIEL_FINAL_RESPONSE), Collections.<String>emptyList());
        }
        if (s.contains(INKLING_THINKING) || (s.contains(INKLING_MESSAGE) && s.contains(INKLING_TEXT))) {
            return inkling();                                                 // vllm/parser/inkling.py
        }
        if (s.contains(PLAMO_BEGIN_THINK)) {
            return thinkTags(Collections.singletonList(PLAMO_BEGIN_THINK),
                    Arrays.asList(PLAMO_END_THINK, PLAMO_EOT), Collections.singletonList(PLAMO_TOOL_REQUESTS));
        }
        String hunyuan = hunyuanTemplateSuffix(s);
        if (hunyuan != null) {
            return think("<think" + hunyuan + ">", "</think" + hunyuan + ">");  // tencent-Hy3.jinja:2-9
        }
        if (s.contains("[THINK]")) return think("[THINK]", "[/THINK]");
        if (s.contains("<seed:think>")) return think("<seed:think>", "</seed:think>");
        if (s.contains("<|START_THINKING|>")) return think("<|START_THINKING|>", "<|END_THINKING|>");
        if (s.contains("<think>") || s.contains("</think>")) return thinkTags();
        return NONE;
    }

    private static ReasoningFilter think(String start, String end) {
        return thinkTags(Collections.singletonList(start), Collections.singletonList(end), Collections.<String>emptyList());
    }

    /**
     * The structural-token suffix of a Hunyuan (HY v3 / v4) template, which spells its tags
     * {@code '<think{}>'.format(HYTK)} after {@code {%- set HYTK = ':opensource' %}}; null for any other template.
     */
    public static String hunyuanTemplateSuffix(String template) {
        if (template == null || !template.contains("'<think{}>'.format(HYTK)")) return null;
        Matcher m = HYTK_ASSIGNMENT.matcher(template);
        if (!m.find()) return null;
        String suffix = m.group(1);
        return suffix.isEmpty() || suffix.startsWith(":") ? suffix : null;
    }

    /**
     * Returns a copy that also recognizes the special tokens of its tags by id: every tag, and every
     * {@code <...>} or {@code [...]} piece of a tag, for which {@code lookup} returns an id &gt;= 0
     * is matched by id, its text replaced by the tag text. Typical lookup: {@code tokenizer::getTokenId}.
     */
    public ReasoningFilter withTagTokens(ToIntFunction<String> lookup) {
        Objects.requireNonNull(lookup, "lookup");
        List<String> tags = new ArrayList<>(mode == Mode.HARMONY_CHANNELS ? HARMONY_TAGS : startTags);
        tags.addAll(endTags);
        tags.addAll(interruptTags);
        tags.addAll(options.droppedTags);
        if (options.responseTemplate != null) tags.addAll(options.responseTemplate.template().literals());
        if (mode == Mode.NONE && tags.isEmpty() && options.variants == null) return this;
        Set<String> candidates = new LinkedHashSet<>();
        for (String tag : tags) {
            candidates.add(tag);
            Matcher m = TOKEN_PIECE.matcher(tag);
            while (m.find()) candidates.add(m.group());
        }
        Map<Integer, String> ids = new HashMap<>(tagTokens);
        for (String c : candidates) {
            int id = lookup.applyAsInt(c);
            if (id >= 0) ids.put(id, c);
        }
        return carry(new ReasoningFilter(mode, startTags, endTags, interruptTags, ids, reasoningInline, options),
                f -> f.withTagTokens(lookup));
    }

    /** A copy that matches its tags in the decoded text only, without the token-id mapping of {@link #withTagTokens}. */
    public ReasoningFilter withoutTagTokens() {
        if (tagTokens.isEmpty() && options.variants == null) return this;
        return carry(new ReasoningFilter(mode, startTags, endTags, interruptTags,
                Collections.<Integer, String>emptyMap(), reasoningInline, options), ReasoningFilter::withoutTagTokens);
    }

    public Mode mode() { return mode; }

    public List<String> startTags() { return startTags; }

    public List<String> endTags() { return endTags; }

    public List<String> interruptTags() { return interruptTags; }

    /** Markers removed from the output wherever they appear ({@link #dropping}); empty when none. */
    public List<String> droppedTags() { return options.droppedTags; }

    /** Token id to tag text mapping set by {@link #withTagTokens}. */
    public Map<Integer, String> tagTokens() { return tagTokens; }

    /** Whether this is the raw harmony variant ({@link #asRaw}): reasoning messages route to content. */
    public boolean reasoningInline() { return reasoningInline; }

    /** A sink forwarding only content to {@code callback}; text released by {@code finish()} carries id -1. */
    public static Sink contentTo(TokenCallback callback) {
        Objects.requireNonNull(callback, "callback");
        return (channel, tokenId, text) -> {
            if (channel == Channel.CONTENT) callback.onToken(tokenId, text);
        };
    }

    /** Starts filtering a generation whose prompt state is unknown (no generation-prompt prefix). */
    public State newState(Sink sink) { return newState(sink, null); }

    /**
     * Starts filtering a generation that continues {@code renderedPrompt}. A prompt whose generation
     * prefix opens a reasoning block (it ends, ignoring trailing whitespace, with a start tag, as the
     * {@code <think>\n} prefill of DeepSeek-R1 and Qwen3.5 templates does) starts in reasoning; a
     * harmony prompt is replayed from its last {@code <|start|>} so a prefilled header or channel
     * carries over.
     */
    public State newState(Sink sink, String renderedPrompt) {
        Objects.requireNonNull(sink, "sink");
        EscapedNewlines.RecoveringSink recovering = options.verbatimSpans == null ? null
                : new EscapedNewlines.RecoveringSink(sink, options.verbatimSpans);
        Sink out = recovering == null ? sink : recovering;
        ReasoningMachine m;
        switch (mode) {
            case THINK_TAGS:
                m = options.responseTemplate != null ? new TemplateMachine(out, options.responseTemplate, renderedPrompt)
                        : options.inklingBlocks ? new InklingMachine(out, renderedPrompt)
                        : options.contentOnly ? new ThinkMachine(this, out, ThinkMachine.Phase.CONTENT)
                        : new ThinkMachine(this, out, (options.implicitOpen ? !closesReasoning(renderedPrompt)
                                : openReasoning(renderedPrompt)) ? ThinkMachine.Phase.REASONING
                                : ThinkMachine.Phase.PRELUDE);
                break;
            case HARMONY_CHANNELS: m = new HarmonyMachine(out, renderedPrompt, reasoningInline); break;
            default: m = new PassMachine(out, options.contentPrefix); break;
        }
        return new State(tagTokens, m, recovering);
    }

    /**
     * Whether a generation after {@code renderedPrompt} starts inside reasoning, as {@link #newState} would start it:
     * a prompt ending with a start tag, a filter whose reasoning opens at the output's first token
     * ({@link #opensImplicitly}) under a prompt that does not close a block, an Inkling prompt inside a thinking
     * block, or a harmony prompt whose last header opens an analysis message.
     *
     * @param renderedPrompt the prompt as text, or null when unknown
     */
    public boolean startsInReasoning(String renderedPrompt) {
        return newState((channel, tokenId, text) -> { }, renderedPrompt).inReasoning();
    }

    /**
     * Whether the prompt's generation prefix leaves reasoning open. Only the prompt tail counts: a
     * start tag quoted earlier in the conversation (a user asking about {@code <think>}, a system
     * prompt describing the format) must not route the whole answer to reasoning. Trailing whitespace is
     * ignored on both sides, so a tag that ends in a newline ({@code <|channel>thought\n}) matches too.
     */
    private boolean openReasoning(String prompt) {
        if (prompt == null) return false;
        String tail = stripTrailingWhitespace(prompt);
        for (String t : startTags) {
            String tag = stripTrailingWhitespace(t);
            if (!tag.isEmpty() && tail.endsWith(tag)) return true;
        }
        return false;
    }

    /**
     * Whether the prompt's generation prefix closes a reasoning block: it ends, ignoring whitespace, with an end
     * tag, as the empty {@code <think></think>} a template writes with thinking off does.
     */
    private boolean closesReasoning(String prompt) {
        if (prompt == null) return false;
        String tail = prompt.trim();
        for (String t : endTags) {
            String tag = t.trim();
            if (!tag.isEmpty() && tail.endsWith(tag)) return true;
        }
        return false;
    }

    private static String stripTrailingWhitespace(String s) {
        int end = s.length();
        while (end > 0 && Character.isWhitespace(s.charAt(end - 1))) end--;
        return s.substring(0, end);
    }

    @Override public String toString() {
        return "ReasoningFilter{" + mode + (mode == Mode.THINK_TAGS ? " " + startTags + " " + endTags + " " + interruptTags : "")
                + (reasoningInline ? " reasoning-inline" : "") + options + "}";
    }

    // ================================================================= streaming state

    /** Per-generation streaming state. Not thread-safe; feed tokens in order, then call {@link #finish()}. */
    public static final class State implements TokenCallback {
        private final Map<Integer, String> tagTokens;
        private final ReasoningMachine machine;
        /** The newline-recovering sink between the machine and the caller's sink, or null. */
        private final EscapedNewlines.RecoveringSink recovering;
        private boolean finished;

        State(Map<Integer, String> tagTokens, ReasoningMachine machine, EscapedNewlines.RecoveringSink recovering) {
            this.tagTokens = tagTokens;
            this.machine = machine;
            this.recovering = recovering;
        }

        /** Feeds one generated token. */
        @Override public void onToken(int tokenId, String text) {
            if (finished) throw new IllegalStateException("ReasoningFilter state already finished");
            String tag = tagTokens.get(tokenId);
            String t = tag != null ? tag : text;
            if (t != null && !t.isEmpty()) machine.feed(tokenId, t);
        }

        /** Releases held-back text at the end of generation. Further tokens are rejected. */
        public void finish() {
            if (finished) return;
            finished = true;
            machine.finish();
            if (recovering != null) recovering.finish();
        }

        /**
         * Whether the generation is inside reasoning: its text streams to {@link Channel#REASONING}, or to
         * content when the raw harmony variant keeps the reasoning inline.
         */
        public boolean inReasoning() { return machine.inReasoning(); }

        /** Whether reasoning text is held back as the possible start of a closing tag. */
        public boolean holding() { return machine.holding(); }

        /** Recipient of the current harmony tool call, or null. */
        public String recipient() { return machine.recipient(); }
    }

    private static final class PassMachine extends ReasoningMachine {
        /** Written before the first content; emptied once written. */
        private String prefix;

        PassMachine(Sink sink, String prefix) {
            super(sink);
            this.prefix = prefix;
        }

        @Override void feed(int tokenId, String text) {
            String t = prefix.isEmpty() ? text : prefix + text;
            prefix = "";
            emit(Channel.CONTENT, tokenId, t, 0, t.length());
        }

        @Override void finish() {
            if (!prefix.isEmpty()) emit(Channel.CONTENT, -1, prefix, 0, prefix.length());
            prefix = "";
        }

        @Override boolean inReasoning() { return false; }
    }

    /**
     * THINK_TAGS: PRELUDE (optional start tag) -> REASONING -> AFTER_END (drop whitespace) -> CONTENT, or back to
     * PRELUDE when the filter reopens. Dropped tags are removed in every phase, after the tags the phase acts on.
     */
    private static final class ThinkMachine extends ReasoningMachine {
        enum Phase { PRELUDE, REASONING, AFTER_END, CONTENT }

        private final List<String> starts;
        private final List<String> closers;
        private final int endCount;
        private final List<String> dropped;
        private final boolean reopens;
        private final boolean dropsReasoning;
        /** Every reasoning piece so far when the filter has a content fallback, else null. */
        private final StringBuilder fallback;
        private final boolean fallbackOnlyWithoutContent;
        private final boolean stripsLeadingNewlines;
        /** The prefix of a tag the end of generation may cut off, or null. */
        private final String truncatedTagStem;
        private final List<String> allTags;
        private final StringBuilder pending = new StringBuilder();
        /** Content with a non-whitespace character was routed. */
        private boolean contentSeen;
        /** Any content character was routed or dropped (whitespace after an end tag, stripped newlines). */
        private boolean contentPresent;
        /** The content's leading newlines are behind (or not stripped at all). */
        private boolean contentStarted;
        private Phase phase;

        ThinkMachine(ReasoningFilter f, Sink sink, Phase start) {
            super(sink);
            this.starts = f.startTags;
            List<String> c = new ArrayList<>(f.endTags);
            c.addAll(f.interruptTags);
            this.closers = Collections.unmodifiableList(c);
            this.endCount = f.endTags.size();
            this.dropped = f.options.droppedTags;
            this.reopens = f.options.reopens;
            this.dropsReasoning = f.options.dropsReasoning;
            this.fallback = f.options.contentFallback ? new StringBuilder() : null;
            this.fallbackOnlyWithoutContent = f.options.fallbackOnlyWithoutContent;
            this.stripsLeadingNewlines = f.options.stripsLeadingNewlines;
            this.contentStarted = !stripsLeadingNewlines;
            this.truncatedTagStem = f.options.truncatedTagStem;
            List<String> all = new ArrayList<>(starts);
            all.addAll(closers);
            all.addAll(dropped);
            this.allTags = Collections.unmodifiableList(all);
            this.phase = start;
        }

        @Override void feed(int tokenId, String text) {
            if (phase == Phase.CONTENT && dropped.isEmpty()) {
                route(Channel.CONTENT, tokenId, text, text.length());
                return;
            }
            pending.append(text);
            process(tokenId);
        }

        private void process(int tokenId) {
            while (true) {
                switch (phase) {
                    case PRELUDE: {
                        // the opener is recognized wherever it arrives (vLLM's thinking parser): the prelude
                        // text before it streams as content, a partial opener holds its suffix back
                        int hit = routeUntil(tokenId, Channel.CONTENT, starts);
                        if (hit < 0) return;
                        pending.delete(0, starts.get(hit).length());
                        phase = Phase.REASONING;
                        break;
                    }
                    case REASONING: {
                        int hit = routeUntil(tokenId, Channel.REASONING, closers);
                        if (hit < 0) return;
                        if (hit < endCount) {
                            pending.delete(0, closers.get(hit).length());
                            phase = Phase.AFTER_END;
                        } else {
                            phase = reopens ? Phase.PRELUDE : Phase.CONTENT;
                        }
                        break;
                    }
                    case AFTER_END: {
                        int blank = leadingWhitespace(pending);
                        contentPresent |= blank > 0;
                        pending.delete(0, blank);
                        if (pending.length() == 0) return;
                        phase = reopens ? Phase.PRELUDE : Phase.CONTENT;
                        break;
                    }
                    default: {
                        routeUntil(tokenId, Channel.CONTENT, Collections.<String>emptyList());
                        return;
                    }
                }
            }
        }

        /**
         * Routes pending text to {@code channel} up to the earliest of {@code tags}, removing dropped tags before it,
         * and returns that tag's index with the pending text starting at the tag; or routes all but a possible
         * partial tag and returns -1.
         */
        private int routeUntil(int tokenId, Channel channel, List<String> tags) {
            while (true) {
                int[] hit = earliest(pending, tags, 0);
                int[] drop = dropped.isEmpty() ? null : earliest(pending, dropped, 0);
                if (drop != null && (hit == null || drop[0] < hit[0])) {
                    route(channel, tokenId, pending, drop[0]);
                    pending.delete(0, drop[0] + dropped.get(drop[1]).length());
                    continue;
                }
                if (hit == null) {
                    int upTo = pending.length() - Math.max(partialSuffix(pending, tags), partialSuffix(pending, dropped));
                    route(channel, tokenId, pending, upTo);
                    pending.delete(0, upTo);
                    return -1;
                }
                route(channel, tokenId, pending, hit[0]);
                pending.delete(0, hit[0]);
                return hit[1];
            }
        }

        /** Routes {@code text[0, to)}: reasoning unless discarded (kept for the fallback), content marked seen. */
        private void route(Channel channel, int tokenId, CharSequence text, int to) {
            if (to <= 0) return;
            if (channel == Channel.REASONING) {
                if (fallback != null) fallback.append(text, 0, to);
                if (!dropsReasoning) emit(Channel.REASONING, tokenId, text, 0, to);
                return;
            }
            contentPresent = true;
            int from = 0;
            if (!contentStarted) {
                while (from < to && text.charAt(from) == '\n') from++;
                contentStarted = from < to;
            }
            if (!contentSeen) contentSeen = leadingWhitespace(text.subSequence(from, to)) < to - from;
            emit(Channel.CONTENT, tokenId, text, from, to);
        }

        @Override void finish() {
            if (truncatedTag()) pending.setLength(0);
            if (phase == Phase.PRELUDE || phase == Phase.CONTENT) route(Channel.CONTENT, -1, pending, pending.length());
            else if (phase == Phase.REASONING) route(Channel.REASONING, -1, pending, pending.length());
            pending.setLength(0);
            boolean noContent = fallbackOnlyWithoutContent ? !contentPresent : !contentSeen;
            if (fallback != null && noContent && fallback.length() > 0) {
                announceContentFallback();
                emit(Channel.CONTENT, -1, fallback, 0, fallback.length());
            }
        }

        /** Whether the held text is a tag the end of generation cut off: it starts with the stem and begins a tag. */
        private boolean truncatedTag() {
            if (truncatedTagStem == null || pending.length() == 0 || pending.indexOf(truncatedTagStem) != 0) {
                return false;
            }
            String held = pending.toString();
            for (String tag : allTags) {
                if (tag.length() > held.length() && tag.startsWith(held)) return true;
            }
            return false;
        }

        @Override boolean inReasoning() { return phase == Phase.REASONING; }

        @Override boolean holding() { return phase == Phase.REASONING && pending.length() > 0; }
    }

    /**
     * The Inkling block layout ({@link #inkling}): HEADER (after {@code <|message_model|>}, up to a content-kind
     * marker) -> REASONING (a thinking body), CONTENT (a text body or text between blocks) or TOOL (a JSON tool
     * block, passed whole to the content) -> CONTENT at the block's end marker. vLLM's inkling grammar: the header
     * is metadata, a marker of either kind opens its block from the content as from a header, a repeated thinking
     * marker inside reasoning is absorbed, and an end marker in the content is dropped.
     */
    private static final class InklingMachine extends ReasoningMachine {
        private enum Phase { HEADER, CONTENT, REASONING, TOOL }

        /** A header longer than this without a content-kind marker means the output is not Inkling blocks. */
        private static final int HEADER_LIMIT = 256;
        private static final List<String> ENDS = Collections.unmodifiableList(
                Arrays.asList(INKLING_END, INKLING_END_SAMPLING));
        private static final List<String> MARKERS = Collections.unmodifiableList(Arrays.asList(INKLING_MESSAGE,
                INKLING_TEXT, INKLING_THINKING, INKLING_TOOL_JSON, INKLING_TOOL_TEXT, INKLING_TOOL_ERROR, INKLING_END,
                INKLING_END_SAMPLING));
        private static final List<String> REASONING_MARKERS = Collections.unmodifiableList(Arrays.asList(
                INKLING_END, INKLING_END_SAMPLING, INKLING_TOOL_JSON, INKLING_THINKING, INKLING_MESSAGE));

        private final StringBuilder pending = new StringBuilder();
        private Phase phase;

        InklingMachine(Sink sink, String prompt) {
            super(sink);
            this.phase = seed(prompt);
        }

        /**
         * The phase a prompt leaves generation in (vLLM {@code InklingParser.adjust_initial_state_from_prompt}): its
         * last Inkling marker decides — a thinking marker reasoning, a text marker content, a header opener (or any
         * other marker, or none) the header the prompt prefills.
         */
        private static Phase seed(String prompt) {
            if (prompt == null) return Phase.HEADER;
            int best = -1;
            String last = null;
            for (String m : MARKERS) {
                int at = prompt.lastIndexOf(m);
                if (at > best) {
                    best = at;
                    last = m;
                }
            }
            if (INKLING_THINKING.equals(last)) return Phase.REASONING;
            if (INKLING_TEXT.equals(last)) return Phase.CONTENT;
            return Phase.HEADER;
        }

        @Override void feed(int tokenId, String text) {
            pending.append(text);
            while (step(tokenId)) {
                // each step consumes one marker
            }
        }

        /** Routes pending text up to the next marker of the phase and takes it; false once only a partial is left. */
        private boolean step(int tokenId) {
            List<String> markers = phase == Phase.REASONING ? REASONING_MARKERS : phase == Phase.TOOL ? ENDS : MARKERS;
            int[] hit = earliest(pending, markers, 0);
            if (hit == null) {
                if (phase == Phase.HEADER) {
                    if (pending.length() > HEADER_LIMIT) phase = Phase.CONTENT;
                    else return false;
                }
                int upTo = pending.length() - partialSuffix(pending, markers);
                emit(channel(), tokenId, pending, 0, upTo);
                pending.delete(0, upTo);
                return false;
            }
            String marker = markers.get(hit[1]);
            boolean endsHeader = phase == Phase.HEADER && (INKLING_END.equals(marker)
                    || INKLING_END_SAMPLING.equals(marker));
            if (phase != Phase.HEADER || endsHeader) emit(endsHeader ? Channel.CONTENT : channel(), tokenId, pending, 0, hit[0]);
            if (phase == Phase.TOOL) emit(Channel.CONTENT, tokenId, marker, 0, marker.length());
            pending.delete(0, hit[0] + marker.length());
            phase = next(marker);
            if (phase == Phase.TOOL && INKLING_TOOL_JSON.equals(marker)) {
                emit(Channel.CONTENT, tokenId, marker, 0, marker.length());
            }
            return true;
        }

        private Channel channel() {
            return phase == Phase.REASONING ? Channel.REASONING : Channel.CONTENT;
        }

        /** The phase after {@code marker} in the current phase. */
        private Phase next(String marker) {
            if (phase == Phase.TOOL) return Phase.CONTENT;
            if (INKLING_MESSAGE.equals(marker)) return Phase.HEADER;
            if (INKLING_THINKING.equals(marker)) return Phase.REASONING;
            if (INKLING_TOOL_JSON.equals(marker)) return Phase.TOOL;
            return Phase.CONTENT;
        }

        @Override void finish() {
            emit(channel(), -1, pending, 0, pending.length());
            pending.setLength(0);
        }

        @Override boolean inReasoning() { return phase == Phase.REASONING; }

        @Override boolean holding() { return phase == Phase.REASONING && pending.length() > 0; }
    }

    /** A response template's regions ({@link #responseTemplate(TemplateParsing)}), routed by {@link ResponseRouting}. */
    private static final class TemplateMachine extends ReasoningMachine {
        private final ResponseRouting routing;
        private int tokenId = -1;

        TemplateMachine(Sink sink, TemplateParsing parsing, String prompt) {
            super(sink);
            this.routing = new ResponseRouting(parsing.template(), prompt,
                    (channel, text) -> emit(channel, tokenId, text, 0, text.length()), parsing.parsesReasoning());
        }

        @Override void feed(int id, String text) {
            tokenId = id;
            routing.feed(text);
        }

        @Override void finish() {
            tokenId = -1;
            routing.finish();
        }

        @Override boolean inReasoning() { return routing.inThinking(); }

        @Override boolean holding() { return routing.inThinking() && routing.holding(); }
    }
}
