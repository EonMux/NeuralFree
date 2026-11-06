package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParser;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserAnalysis;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Renders chat prompts with a model's resolved chat template ({@link ChatTemplateSource}), as llama.cpp's
 * Jinja path does (common/chat.cpp common_chat_templates_apply_jinja).
 *
 * <p>At construction the template is parsed once and its capabilities are probed the way llama.cpp's
 * {@code caps_get} probes them ({@link TemplateCaps}): whether it renders a system message, describes tools,
 * renders tool calls (several per message, with object arguments), keeps earlier reasoning, reads the
 * reasoning effort, and reads message content as a string or as typed parts. The messages reach the template in
 * the forms it reads ({@link ChatMessages#prepare}): typed parts joined to a string or a string wrapped as a
 * text part, tool-call arguments as an object or as JSON text, and for a template without the system role the
 * system prompt merged into the next message. A built-in template ({@link ChatTemplateSource#isBuiltIn})
 * reproduces llama.cpp's formatter for its style, which handles system messages itself, so it counts as
 * rendering the system role and its conversations keep their system messages. A forced content format (vLLM's
 * {@code --chat-template-content-format}, {@link #CONTENT_FORMAT_PROPERTY}) replaces the detected content
 * capabilities.</p>
 *
 * <p>A template that fails to parse or render leaves the prompt to a fallback, logged as a warning: the
 * gpt-oss harmony format for gpt-oss models, otherwise the built-in ChatML template.</p>
 *
 * <p>Instances are immutable and thread-safe.</p>
 */
public final class ChatFormatter {

    private static final Logger LOGGER = LogManager.getLogger(ChatFormatter.class);

    /**
     * The system property that sets the default of the {@code preserve_reasoning} template variable
     * ({@code true} or {@code false}), as llama.cpp's {@code --reasoning-preserve} and
     * {@code --no-reasoning-preserve} do; unset leaves llama.cpp's default
     * ({@link #DEFAULT_PRESERVE_REASONING}).
     */
    public static final String PRESERVE_REASONING_PROPERTY = "neuralfree.chat.reasoning.preserve";

    /**
     * The {@code preserve_reasoning} a request without that template kwarg renders with when neither the
     * {@link #PRESERVE_REASONING_PROPERTY} nor a formatter default says otherwise: llama.cpp's, which enables
     * the reasoning-history aliases at start-up when no flag names them (common/arg.cpp:948-950).
     */
    public static final boolean DEFAULT_PRESERVE_REASONING = true;

    /**
     * The system property that forces the content format ({@code auto}, {@code string} or {@code openai}), as
     * vLLM's {@code --chat-template-content-format} does ({@link TemplateCaps.ContentFormat}); unset is
     * {@code auto}, the detected one.
     */
    public static final String CONTENT_FORMAT_PROPERTY = "neuralfree.chat.template.content.format";

    private static final Compiled CHATML =
            new Compiled(ChatTemplateSource.CHATML, MiniJinja.compile(ChatTemplateSource.CHATML));
    /** The chat_template overrides one formatter keeps compiled (vLLM compiles per template object). */
    private static final int OVERRIDE_CACHE_MAX = 8;
    /** llama.cpp's {@code datetime} and {@code date_string} formats, {@code %b %d %Y} and {@code %d %b %Y} in the C locale. */
    private static final DateTimeFormatter DATETIME_FORMAT = DateTimeFormatter.ofPattern("MMM dd yyyy", Locale.ENGLISH);
    private static final DateTimeFormatter DATE_STRING_FORMAT = DateTimeFormatter.ofPattern("dd MMM yyyy", Locale.ENGLISH);

    private final String source;
    /** The model template with its capabilities; null when the source does not parse. */
    private final Compiled template;
    private final String bosText;
    private final String eosText;
    private final boolean addBos;
    private final boolean addEos;
    private final boolean harmonyFallback;
    private final Boolean preserveReasoning;
    private final TemplateCaps.ContentFormat contentFormat;
    /** Compiled chat_template overrides by source, least recently used out. */
    private final Map<String, Compiled> overrides = Collections.synchronizedMap(
            new LinkedHashMap<String, Compiled>(16, 0.75f, true) {
                @Override protected boolean removeEldestEntry(Map.Entry<String, Compiled> eldest) {
                    return size() > OVERRIDE_CACHE_MAX;
                }
            });

    /**
     * A formatter that leaves {@code preserve_reasoning} to the request and the content format to detection
     * ({@link #ChatFormatter(String, String, String, boolean, boolean, String, Boolean, TemplateCaps.ContentFormat)}
     * with a null default and {@code auto}).
     */
    public ChatFormatter(String source, String bosText, String eosText, boolean addBos, boolean addEos,
                         String architecture) {
        this(source, bosText, eosText, addBos, addEos, architecture, null, TemplateCaps.ContentFormat.AUTO);
    }

    /**
     * @param source            the resolved template source ({@link ChatTemplateSource#resolve}); not null
     * @param bosText           the BOS token text for {@code bos_token}, empty when the vocabulary has none
     * @param eosText           the EOS token text for {@code eos_token}, empty when the vocabulary has none
     * @param addBos            whether tokenization adds BOS ({@code llama_vocab_get_add_bos})
     * @param addEos            whether tokenization adds EOS ({@code llama_vocab_get_add_eos})
     * @param architecture      the GGUF architecture name, selecting the fallback format; may be null
     * @param preserveReasoning the {@code preserve_reasoning} a request without that template kwarg renders
     *                          with ({@link #PRESERVE_REASONING_PROPERTY}); null leaves llama.cpp's default
     *                          ({@link #DEFAULT_PRESERVE_REASONING})
     * @param contentFormat     the content format every template renders with ({@link #CONTENT_FORMAT_PROPERTY});
     *                          {@code auto} uses the detected one
     */
    public ChatFormatter(String source, String bosText, String eosText, boolean addBos, boolean addEos,
                         String architecture, Boolean preserveReasoning, TemplateCaps.ContentFormat contentFormat) {
        if (source == null) throw new IllegalArgumentException("chat template source is null");
        if (contentFormat == null) throw new IllegalArgumentException("chat template content format is null");
        this.source = source;
        this.bosText = bosText == null ? "" : bosText;
        this.eosText = eosText == null ? "" : eosText;
        this.addBos = addBos;
        this.addEos = addEos;
        this.harmonyFallback = "gpt-oss".equals(architecture) || "gpt_oss".equals(architecture);
        this.preserveReasoning = preserveReasoning;
        this.contentFormat = contentFormat;
        this.template = compile(source);
        if (template != null) {
            reportContentFormat(template.caps);
            reportPreserveReasoning(template.caps);
        }
    }

    /**
     * The {@link #CONTENT_FORMAT_PROPERTY} format: {@code auto} when unset.
     *
     * @throws IllegalArgumentException when the property names no format
     */
    public static TemplateCaps.ContentFormat contentFormatFromProperties() {
        String value = System.getProperty(CONTENT_FORMAT_PROPERTY);
        if (value == null) return TemplateCaps.ContentFormat.AUTO;
        try {
            return TemplateCaps.ContentFormat.parse(value.trim().toLowerCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(CONTENT_FORMAT_PROPERTY + ": " + e.getMessage(), e);
        }
    }

    /** vLLM's load-time note on the content format (renderers/hf.py resolve_chat_template_content_format). */
    private void reportContentFormat(TemplateCaps detected) {
        TemplateCaps.ContentFormat format = detected.detectedContentFormat();
        if (contentFormat == TemplateCaps.ContentFormat.AUTO) {
            LOGGER.info("Detected the chat template content format to be '{}'; --chat-template-content-format"
                    + " overrides it", format.value());
        } else if (contentFormat != format) {
            LOGGER.warn("--chat-template-content-format {} differs from the detected format '{}'",
                    contentFormat.value(), format.value());
        }
    }

    /**
     * The {@link #PRESERVE_REASONING_PROPERTY} default: {@code true}, {@code false}, or null when unset, which
     * leaves llama.cpp's {@link #DEFAULT_PRESERVE_REASONING}.
     *
     * @throws IllegalArgumentException when the property is set to anything else
     */
    public static Boolean preserveReasoningFromProperties() {
        String value = System.getProperty(PRESERVE_REASONING_PROPERTY);
        if (value == null) return null;
        String v = value.trim();
        if ("true".equalsIgnoreCase(v)) return Boolean.TRUE;
        if ("false".equalsIgnoreCase(v)) return Boolean.FALSE;
        throw new IllegalArgumentException(PRESERVE_REASONING_PROPERTY + " must be true or false, got '" + value + "'");
    }

    private static Compiled compile(String source) {
        if (ChatTemplateSource.CHATML.equals(source)) return CHATML;
        try {
            return new Compiled(source, MiniJinja.compile(source));
        } catch (RuntimeException e) {
            LOGGER.warn("The chat template ({} chars) does not parse, prompts use the fallback format: {}",
                    source.length(), e.getMessage());
            return null;
        }
    }

    /**
     * llama.cpp's load-time notes on the preserve-reasoning capability (server-context.cpp:1746-1762): a template
     * that can keep earlier reasoning renders with it enabled by default ({@link #DEFAULT_PRESERVE_REASONING}),
     * which may use more tokens; a template that cannot, with an explicit {@code --reasoning-preserve}, gains
     * nothing.
     */
    private void reportPreserveReasoning(TemplateCaps caps) {
        boolean enabled = preserveReasoning == null ? DEFAULT_PRESERVE_REASONING : preserveReasoning;
        if (caps.supportsPreserveReasoning() && preserveReasoning == null) {
            LOGGER.warn("The chat template supports preserving reasoning; it is enabled by default (may use more"
                    + " tokens, disable with --no-reasoning-preserve)");
        } else if (caps.supportsPreserveReasoning() && !enabled) {
            LOGGER.info("The chat template supports preserving reasoning; enable it with --reasoning-preserve"
                    + " or the preserve_reasoning template kwarg");
        } else if (!caps.supportsPreserveReasoning() && preserveReasoning != null && enabled) {
            LOGGER.warn("The chat template does not support preserving reasoning; --reasoning-preserve has no"
                    + " effect");
        }
    }

    /** The template source prompts are rendered with. */
    public String source() { return source; }

    /**
     * The capabilities the model template renders with ({@link TemplateCaps}), its content capabilities those
     * of a forced content format; those of the built-in ChatML template, which then renders, when the model
     * template does not parse.
     */
    public TemplateCaps caps() { return effective(template != null ? template : CHATML); }

    /**
     * The capabilities the {@code request} renders with: its {@code chat_template} override's when it carries
     * one, else the model template's ({@link #caps()}) — the request's normalization
     * ({@link ChatMessages#prepare}) and its capability-driven defaults answer to these.
     *
     * @throws IllegalArgumentException when the override does not parse
     */
    public TemplateCaps caps(ChatRequest request) {
        if (request == null || request.chatTemplate() == null) return caps();
        return effective(compiledOverride(request.chatTemplate()));
    }

    /** The capabilities {@code compiled} renders with: its detected ones under the forced content format. */
    private TemplateCaps effective(Compiled compiled) {
        return compiled.caps.withContentFormat(contentFormat);
    }

    /**
     * Whether the template renders a leading system message; when not, it is merged into the next message.
     * True for every built-in template, which places system messages as llama.cpp's formatter does.
     */
    public boolean supportsSystemRole() { return caps().supportsSystemRole(); }

    /**
     * Renders a conversation followed by the assistant generation prompt.
     *
     * @param messages       maps with a string {@code role} and a {@code content}; not modified
     * @param enableThinking the template's {@code enable_thinking} variable
     * @throws IllegalArgumentException for malformed messages
     */
    public String format(List<? extends Map<String, ?>> messages, boolean enableThinking) {
        List<Map<String, Object>> copy = ChatMessages.copyOf(messages);
        LocalDate today = LocalDate.now();
        Map<String, Object> noKwargs = Collections.emptyMap();
        if (template != null) {
            try {
                List<Map<String, Object>> input = ChatMessages.trimForTemplate(copy, template.source);
                return render(template.template, ChatMessages.prepare(input, template.source, effective(template)),
                        true, enableThinking, Collections.<Map<String, Object>>emptyList(), null,
                        Collections.<Map<String, Object>>emptyList(), noKwargs, today, null);
            } catch (RuntimeException e) {
                LOGGER.warn("Chat template rendering failed ({}-char template, {} messages), using the fallback"
                        + " format: {}", source.length(), copy.size(), e.getMessage());
            }
        }
        if (harmonyFallback) return harmony(copy, today);
        return render(CHATML.template, ChatMessages.prepare(copy, CHATML.source, effective(CHATML)),
                true, enableThinking, Collections.<Map<String, Object>>emptyList(), null,
                Collections.<Map<String, Object>>emptyList(), noKwargs, today, null);
    }

    /**
     * Renders a chat request ({@link ChatRequest}): its messages after llama.cpp's template-input
     * conventions ({@link ChatMessages#forTemplate}, then {@link ChatMessages#prepare} in the content
     * and argument forms the template reads), the offered tools bound as the template's {@code tools}
     * list (undefined when none, as llama.cpp binds an empty tool list, common/chat.cpp:894), the
     * request's {@code documents} as the {@code documents} variable (bound only when some are given,
     * as vLLM binds them into the template context), the {@code chat_template_kwargs} (bound last, so
     * a template variable can be carried explicitly), {@code enable_thinking} and {@code reasoning_effort}
     * (both bound only when set, the effort also as {@code reasoning_strength}), and for a boolean
     * {@code preserve_reasoning} kwarg, or this formatter's default of it, llama.cpp's reasoning-history
     * aliases ({@link TemplateCaps#PRESERVE_REASONING}).
     *
     * <p>A request carrying a {@code chat_template} override renders with that template in place
     * of the model's, resolved as the model's is ({@link ChatTemplateSource#resolve}: a built-in name
     * selects that template, and llama.cpp's source patches apply) and its capabilities probed the
     * same way; the override must parse and
     * render — a failure refuses the request by name rather than falling back (vLLM raises). The
     * BOS/EOS stripping stays this formatter's, as the override renders for the same vocabulary.</p>
     *
     * <p>With {@code continue_final_message} the render happens without the generation prompt and
     * the template's end-of-turn after the last assistant message is stripped, as vLLM does.</p>
     *
     * <p>A request whose messages carry an image or an audio content part is refused: the part's
     * content lands on {@link ChatMessages#MEDIA_MARKER}, and the encoding that consumes the
     * marker is the multimodal prompt path's ({@link #format(ChatRequest, boolean)}).</p>
     *
     * @throws IllegalArgumentException for malformed messages, tool calls or template kwargs, and
     *                                  for a chat_template override that does not parse or render
     * @throws UnsupportedOperationException for a request with image or audio content parts
     */
    public String format(ChatRequest request) {
        return format(request, false, null);
    }

    /**
     * The speech prompt of one text (common_chat_get_tts_prompt, common/chat.cpp:859-868): this formatter's
     * template — the model's speech chat template — renders the single-user-message conversation with the
     * {@code language} and {@code voice} variables and only the token variables beside them
     * (common_chat_template_render, common/chat.cpp:808-815). The result passes
     * {@link #stripAddedSpecials}, so tokenizing it yields exactly one BOS. This formatter must be the speech
     * one ({@code GGUFTokenizer} builds it from the model's {@code tts} template); the plain
     * {@link #format(ChatRequest)} renders with the full chat context instead.
     */
    public String formatTtsPrompt(String text, String language, String voice) {
        Map<String, Object> user = new LinkedHashMap<>();
        user.put("role", "user");
        user.put("content", text);
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("messages", List.of(user));
        context.put("language", language);
        context.put("voice", voice);
        context.put("bos_token", bosText);
        context.put("eos_token", eosText);
        return stripAddedSpecials(template.template.render(context), bosText, eosText, addBos, addEos);
    }

    /**
     * {@link #format(ChatRequest)}, with {@code mediaMarkers} set for a caller that runs the
     * multimodal prompt path: every image and audio content part then renders as one
     * {@link ChatMessages#MEDIA_MARKER} in the prompt, in message and part order, for the caller
     * to replace with the encoded media (llama.cpp's {@code mtmd_tokenize} input).
     *
     * @throws IllegalArgumentException      as {@link #format(ChatRequest)}
     * @throws UnsupportedOperationException for media content parts when {@code mediaMarkers} is false
     */
    public String format(ChatRequest request, boolean mediaMarkers) {
        return format(request, mediaMarkers, null);
    }

    /**
     * {@link #format(ChatRequest, boolean)} over the autoparser's analysis: with {@code continuation} set and
     * {@code continue_final_message}, a continued final message is written from its reasoning and content around
     * the derived reasoning markers, as llama.cpp's autoparser does, instead of rendering the message through the
     * template. A null analysis keeps the template-rendering path.
     *
     * @throws IllegalArgumentException      as {@link #format(ChatRequest)}
     * @throws UnsupportedOperationException for media content parts when {@code mediaMarkers} is false
     */
    public String format(ChatRequest request, boolean mediaMarkers, AutoParserAnalysis continuation) {
        return format(request, mediaMarkers, continuation, null);
    }

    /**
     * The speech prompt of a conversation: {@link #format(ChatRequest)} over this formatter's template — the
     * model's conversation template — with the {@code language} and {@code voice} variables bound beside the
     * usual context, so a template that conditions speech on them (the LFM3 speech template, or a conversation
     * template that reads them) renders them. A template that names neither ignores them, as llama.cpp's
     * conversation render does.
     *
     * @throws IllegalArgumentException as {@link #format(ChatRequest)}
     */
    public String formatSpeech(ChatRequest request, String language, String voice) {
        Map<String, Object> variables = new LinkedHashMap<>();
        variables.put("language", language == null ? "" : language);
        variables.put("voice", voice == null ? "" : voice);
        return format(request, false, null, variables);
    }

    private String format(ChatRequest request, boolean mediaMarkers, AutoParserAnalysis continuation,
                          Map<String, Object> variables) {
        if (request == null) throw new IllegalArgumentException("chat request is null");
        List<Map<String, Object>> messages = ChatMessages.forTemplate(ChatMessages.copyOf(request.messages()),
                profileSource(request.chatTemplate()));
        if (!mediaMarkers) refuseMedia(messages);
        if (request.chatTemplate() != null) {
            return formatWithOverride(request, messages, continuation, variables);
        }
        LocalDate today = LocalDate.now();
        if (template != null) {
            try {
                return renderRequest(template, request, messages, today, continuation, variables);
            } catch (RuntimeException e) {
                LOGGER.warn("Chat template rendering failed ({}-char template, {} messages), using the fallback"
                        + " format: {}", source.length(), messages.size(), e.getMessage());
            }
        }
        if (harmonyFallback) return harmony(messages, today);
        return renderRequest(CHATML, request, messages, today, continuation, variables);
    }

    /**
     * The gpt-oss harmony fallback, which writes each content as text: typed text parts reach it joined, as for
     * the string-only ChatML template.
     */
    private static String harmony(List<Map<String, Object>> messages, LocalDate today) {
        return HarmonyFallback.format(ChatMessages.normalize(messages, CHATML.caps), today);
    }

    /**
     * The resolved source whose template input conventions shape {@code request}'s messages
     * ({@link ChatMessages#forTemplate(List, String)}): a {@code chat_template} override's, else the model
     * template's, else the built-in ChatML one.
     *
     * @throws IllegalArgumentException when the override does not parse
     */
    private String profileSource(String override) {
        if (override != null) return compiledOverride(override).source;
        return template != null ? template.source : ChatTemplateSource.CHATML;
    }

    /**
     * Renders with the request's template override: compiled and probed once per distinct source,
     * then rendered like the model's template. A parse or render failure is an
     * {@link IllegalArgumentException}, since the caller named this template explicitly.
     */
    private String formatWithOverride(ChatRequest request, List<Map<String, Object>> messages,
                                      AutoParserAnalysis continuation, Map<String, Object> variables) {
        Compiled override = compiledOverride(request.chatTemplate());
        AutoParserAnalysis analysis = continuation != null ? continuation
                : continuationOf(override.source, request);
        try {
            return renderRequest(override, request, messages, LocalDate.now(), analysis, variables);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("the chat_template override does not render this conversation: "
                    + e.getMessage());
        }
    }

    /**
     * The analysis a continued final message is written around for a request's own template override, which the
     * model template's analysis does not describe; null unless the final message is continued. A template that
     * does not analyse keeps the template-rendering path.
     */
    private static AutoParserAnalysis continuationOf(String source, ChatRequest request) {
        return request.continueFinalMessage() ? AutoParser.analyze(source) : null;
    }

    /**
     * One request rendered with {@code compiled}: the messages prepared for its source and
     * capabilities, the generation prompt unless the final message is continued (vLLM's contract),
     * the end-of-turn stripped after a continued message. With {@code continuation} set, a continued final
     * message is written from the autoparser's derived reasoning markers instead ({@link #renderContinued});
     * {@code variables} are the render's extra template variables (a speech render's language and voice).
     */
    private String renderRequest(Compiled compiled, ChatRequest request, List<Map<String, Object>> messages,
                                 LocalDate today, AutoParserAnalysis continuation, Map<String, Object> variables) {
        compiled.reportTools(request);
        if (request.continueFinalMessage() && continuation != null) {
            String continued = renderContinued(compiled, request, messages, continuation, today);
            if (continued != null) {
                return continued;
            }
        }
        String rendered = render(compiled.template,
                ChatMessages.prepare(messages, compiled.source, effective(compiled)),
                request.continueFinalMessage() ? false : request.addGenerationPrompt(),
                request.enableThinking(), toolTemplates(request),
                request.reasoningEffort(), request.documents(), request.templateKwargs(), today, variables);
        return request.continueFinalMessage() ? stripEndOfTurn(rendered) : rendered;
    }

    /**
     * The prompt that continues the final assistant message around the autoparser's derived reasoning markers
     * (llama.cpp's {@code autoparser::peg_generator::generate_parser} continuation): the conversation before the
     * message, then the generation prompt truncated at the reasoning start, the reasoning content, and — unless
     * only the reasoning is continued — the reasoning end and the content. Null when the message's content is not
     * plain text, so the caller renders it through the template instead.
     */
    private String renderContinued(Compiled compiled, ChatRequest request, List<Map<String, Object>> messages,
                                   AutoParserAnalysis analysis, LocalDate today) {
        if (messages.isEmpty() || !"assistant".equals(messages.get(messages.size() - 1).get(ChatMessages.ROLE))) {
            return null;
        }
        Map<String, Object> last = messages.get(messages.size() - 1);
        String content = continuedContent(last.get(ChatMessages.CONTENT));
        if (content == null) {
            return null;
        }
        Object rawReasoning = last.get(ChatMessages.REASONING_CONTENT);
        String reasoning = rawReasoning instanceof String ? (String) rawReasoning : "";
        if (reasoning.isEmpty() && content.isEmpty()) {
            // llama.cpp leaves an empty continued message to the template: there is nothing to write around
            return null;
        }
        // an assistant message with reasoning only continues its reasoning, any other its content
        boolean continueContent = !(reasoning != null && !reasoning.isEmpty() && content.isEmpty());
        List<Map<String, Object>> prior = messages.subList(0, messages.size() - 1);
        List<Map<String, Object>> prepared = ChatMessages.prepare(new ArrayList<>(prior), compiled.source,
                effective(compiled));
        List<Map<String, Object>> tools = toolTemplates(request);
        String prefix = render(compiled.template, prepared, false, request.enableThinking(), tools,
                request.reasoningEffort(), request.documents(), request.templateKwargs(), today, null);
        String withGeneration = render(compiled.template, prepared, true, request.enableThinking(), tools,
                request.reasoningEffort(), request.documents(), request.templateKwargs(), today, null);
        String generation = withGeneration.substring(commonPrefix(prefix, withGeneration));
        String start = analysis.reasoning().start();
        StringBuilder out = new StringBuilder(prefix);
        if (!start.isEmpty()) {
            int at = generation.indexOf(start);
            if (at >= 0) {
                generation = generation.substring(0, at);
            }
            generation += start + reasoning;
            if (continueContent) {
                generation += analysis.reasoning().end();
            }
        }
        if (continueContent) {
            generation += content;
        }
        return out.append(generation).toString();
    }

    /** The length of the two texts' common prefix. */
    private static int commonPrefix(String a, String b) {
        int limit = Math.min(a.length(), b.length());
        int at = 0;
        while (at < limit && a.charAt(at) == b.charAt(at)) {
            at++;
        }
        return at;
    }

    /** A message's content as plain text ({@code content_parts}' texts joined), or null for other parts. */
    private static String continuedContent(Object content) {
        String text = plainContent(content);
        if (text == null || text.contains(ChatMessages.MEDIA_MARKER)
                || text.contains(ChatMessages.PROMPT_EMBEDS_MARKER)) {
            return null;
        }
        return text;
    }

    /** A message's content as plain text ({@code content_parts}' texts joined), or null for other parts. */
    private static String plainContent(Object content) {
        if (content == null) {
            return "";
        }
        if (content instanceof String) {
            return (String) content;
        }
        if (!(content instanceof List)) {
            return null;
        }
        StringBuilder out = new StringBuilder();
        for (Object part : (List<?>) content) {
            if (!(part instanceof Map) || !"text".equals(((Map<?, ?>) part).get("type"))
                    || !(((Map<?, ?>) part).get("text") instanceof String)) {
                return null;
            }
            out.append((String) ((Map<?, ?>) part).get("text"));
        }
        return out.toString();
    }

    /**
     * The compiled and probed override of {@code template}, resolved as a model template is.
     *
     * @throws IllegalArgumentException when the resolved source does not parse
     */
    private Compiled compiledOverride(String template) {
        Compiled cached = overrides.get(template);
        if (cached != null) return cached;
        String resolved = ChatTemplateSource.resolve(template);
        MiniJinja.Template compiled;
        try {
            compiled = MiniJinja.compile(resolved);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("the chat_template override does not parse: " + e.getMessage());
        }
        Compiled built = new Compiled(resolved, compiled);
        overrides.put(template, built);
        return built;
    }

    /**
     * One compiled template: its source, the template, and its probed capabilities, the system role counted
     * as rendered for a built-in template ({@link ChatTemplateSource#isBuiltIn}).
     */
    private static final class Compiled {
        final String source;
        final MiniJinja.Template template;
        final TemplateCaps caps;
        private final AtomicBoolean toolsReported = new AtomicBoolean();

        Compiled(String source, MiniJinja.Template template) {
            this.source = source;
            this.template = template;
            TemplateCaps probed = TemplateCaps.probe(template);
            this.caps = ChatTemplateSource.isBuiltIn(source) ? probed.withSystemRole() : probed;
        }

        /**
         * Warns, once per template, when a request offers tools the template does not describe: llama.cpp's
         * warning for a template that renders tool calls but not the tools (common/chat.cpp:1551-1555), and the
         * same for one that renders neither, whose prompt then carries no trace of the tools.
         */
        void reportTools(ChatRequest request) {
            if (request.tools().isEmpty() || caps.supportsTools() || !toolsReported.compareAndSet(false, true)) {
                return;
            }
            if (caps.supportsToolCalls()) {
                LOGGER.warn("The chat template supports tool calls but does not natively describe tools; the"
                        + " prompt may produce bad results, consider overriding the template");
            } else {
                LOGGER.warn("The chat template renders neither tools nor tool calls; the offered tools do not"
                        + " reach the prompt, consider overriding the template");
            }
        }
    }

    /** The tools bound as the template's {@code tools} list; empty when no tools are offered. */
    private static List<Map<String, Object>> toolTemplates(ChatRequest request) {
        if (request.tools().isEmpty()) return Collections.emptyList();
        List<Map<String, Object>> out = new ArrayList<>(request.tools().size());
        for (ChatTool tool : request.tools()) {
            out.add(tool.templateObject());
        }
        return out;
    }

    /**
     * Raises a multimodal refusal for a request whose messages carry image or audio content
     * parts, before any template runs: the parts produce {@link ChatMessages#MEDIA_MARKER}, which
     * only the multimodal prompt path consumes.
     */
    private static void refuseMedia(List<Map<String, Object>> messages) {
        for (Map<String, Object> m : messages) {
            if (carriesMediaMarker(m.get(ChatMessages.CONTENT))) {
                throw new UnsupportedOperationException("multimodal chat input is refused:"
                        + " message content parts of kind image or audio need a multimodal projector"
                        + " (--mmproj) to encode them, and this prompt is rendered as text only");
            }
        }
    }

    /** Whether a string content, or a text part of a typed content, carries {@link ChatMessages#MEDIA_MARKER}. */
    private static boolean carriesMediaMarker(Object content) {
        if (content instanceof String) return ((String) content).contains(ChatMessages.MEDIA_MARKER);
        if (!(content instanceof List)) return false;
        for (Object part : (List<?>) content) {
            Object text = part instanceof Map ? ((Map<?, ?>) part).get("text") : null;
            if (text instanceof String && ((String) text).contains(ChatMessages.MEDIA_MARKER)) return true;
        }
        return false;
    }

    /**
     * Renders with the llama.cpp context variables (common/chat.cpp:877-911): messages, add_generation_prompt,
     * bos_token, eos_token, tools (bound only when some are offered, as llama.cpp binds an empty tool list,
     * common/chat.cpp:894), the documents (bound only when some are given, as vLLM binds them into the template
     * context), enable_thinking and reasoning_effort (both only when the caller set them, the effort also as
     * {@code reasoning_strength}), the reasoning-history aliases of a boolean {@code preserve_reasoning}
     * ({@link #bindPreserveReasoning}), the chat_template_kwargs last (server-chat.cpp:593), and the local-date
     * {@code datetime} and {@code date_string} of common_chat_extra_context (:1065-1073), which templates such as
     * Llama 3.1 write into their system header, and the caller's extra variables (a speech render's
     * {@code language} and {@code voice}) before the kwargs, which can still override them. The result passes
     * {@link #stripAddedSpecials}, so tokenizing it yields exactly one BOS.
     */
    private String render(MiniJinja.Template t, List<Map<String, Object>> messages, boolean addGenerationPrompt,
                          Boolean enableThinking, List<Map<String, Object>> tools, String reasoningEffort,
                          List<Map<String, Object>> documents, Map<String, Object> templateKwargs,
                          LocalDate today, Map<String, Object> variables) {
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("messages", messages);
        context.put("add_generation_prompt", addGenerationPrompt);
        context.put("bos_token", bosText);
        context.put("eos_token", eosText);
        if (!tools.isEmpty()) context.put("tools", tools);
        if (!documents.isEmpty()) context.put("documents", documents);
        if (reasoningEffort != null) TemplateCaps.applyReasoningEffort(context, reasoningEffort);
        if (enableThinking != null) context.put("enable_thinking", enableThinking);
        context.put("datetime", DATETIME_FORMAT.format(today));
        context.put("date_string", DATE_STRING_FORMAT.format(today));
        bindPreserveReasoning(context, templateKwargs);
        if (variables != null) context.putAll(variables);
        context.putAll(templateKwargs);
        return stripAddedSpecials(t.render(context), bosText, eosText, addBos, addEos);
    }

    /**
     * llama.cpp's {@code preserve_reasoning} handling (common/chat.cpp:1077-1080): the kwarg, or this formatter's
     * default when the request does not carry it, is bound with its aliases ({@link TemplateCaps}), which the
     * kwargs bound after it can still override one by one. Without any of them, llama.cpp's default
     * ({@link #DEFAULT_PRESERVE_REASONING}) holds.
     *
     * @throws IllegalArgumentException when the {@code preserve_reasoning} kwarg is not a boolean
     */
    private void bindPreserveReasoning(Map<String, Object> context, Map<String, Object> templateKwargs) {
        Object requested = templateKwargs.get(TemplateCaps.PRESERVE_REASONING);
        if (requested != null && !(requested instanceof Boolean)) {
            throw new IllegalArgumentException("chat_template_kwargs." + TemplateCaps.PRESERVE_REASONING
                    + " must be a boolean, got " + requested.getClass().getSimpleName());
        }
        Boolean preserve = requested != null ? (Boolean) requested
                : preserveReasoning != null ? preserveReasoning : DEFAULT_PRESERVE_REASONING;
        context.put(TemplateCaps.PRESERVE_REASONING, preserve);
        TemplateCaps.applyPreserveReasoning(context, preserve);
    }

    /**
     * vLLM's {@code continue_final_message} post-processing: the template's end-of-turn after the
     * last assistant message is removed, so the model continues that message's turn. Trailing
     * whitespace is dropped and, when the vocabulary's EOS text ends the prompt, it is removed
     * once; the EOS a template emits after a final assistant message is the model EOS in
     * llama.cpp templates, exactly as vLLM strips it (ChatCompletionRequest.continue_final_message).
     */
    public static String stripEndOfTurn(String rendered, String eosText) {
        String out = rendered;
        while (!out.isEmpty() && isTrailingSpace(out.charAt(out.length() - 1))) {
            out = out.substring(0, out.length() - 1);
        }
        if (!eosText.isEmpty() && out.endsWith(eosText)) {
            out = out.substring(0, out.length() - eosText.length());
            while (!out.isEmpty() && isTrailingSpace(out.charAt(out.length() - 1))) {
                out = out.substring(0, out.length() - 1);
            }
        }
        return out;
    }

    /** {@code continue_final_message} strip with this formatter's EOS text ({@link #stripEndOfTurn}). */
    private String stripEndOfTurn(String rendered) {
        return stripEndOfTurn(rendered, eosText);
    }

    private static boolean isTrailingSpace(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\u000B' || c == '\f';
    }

    /**
     * Removes the BOS text a template wrote at the start of the prompt when the tokenizer adds BOS itself, and
     * the EOS text it wrote at the end when the tokenizer adds EOS, as llama.cpp does after rendering
     * (common/chat.cpp:938-949, with {@code add_bos}/{@code add_eos} from {@code llama_vocab_get_add_bos/eos},
     * :804-805). Without this the encoded prompt would start with two BOS tokens (src/llama-vocab.cpp:3767-3768).
     */
    public static String stripAddedSpecials(String rendered, String bosText, String eosText, boolean addBos,
                                            boolean addEos) {
        String out = rendered;
        if (addBos && !bosText.isEmpty() && out.startsWith(bosText)) {
            out = out.substring(bosText.length());
        }
        if (addEos && !eosText.isEmpty() && out.endsWith(eosText)) {
            out = out.substring(0, out.length() - eosText.length());
        }
        return out;
    }
}
