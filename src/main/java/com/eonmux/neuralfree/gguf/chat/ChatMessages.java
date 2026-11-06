package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Chat message lists as llama.cpp hands them to a template (common/chat.cpp
 * common_chat_templates_apply_jinja, :1299-1360, and the {@code workaround} functions, :987-1061).
 * Every method returns new lists and maps; caller data is never modified.
 */
public final class ChatMessages {

    private static final Logger LOGGER = LogManager.getLogger(ChatMessages.class);

    static final String ROLE = "role";
    /** The message content key, public for the serving layers that read rendered messages back. */
    public static final String CONTENT = "content";
    /** The assistant message's reasoning text key, public for the continuation prompt's builder. */
    public static final String REASONING_CONTENT = "reasoning_content";
    /** vLLM's current assistant reasoning key; {@link #REASONING_CONTENT} is its deprecated alias. */
    public static final String REASONING = "reasoning";
    private static final String TYPE = "type";
    private static final String TEXT = "text";
    private static final String SYSTEM = "system";
    private static final String DEVELOPER = "developer";

    /** Marker of the gpt-oss harmony templates, which keep the developer role (common/chat.cpp:1347). */
    private static final String HARMONY_CHANNEL = "<|channel|>";
    /** Marker of the StepFun templates, whose message text is trimmed before rendering (common/chat.cpp:1476-1482). */
    private static final String STEPFUN_TOOLS_PROMPT = "You have access to the following functions in JSONSchema format";
    /**
     * Marker of the DeepSeek V4.1 templates, whose DSML tag names carry a leading space (common/chat.cpp:1483);
     * their message rendering keeps the message's own keys and carries media markers as separate text parts
     * ({@code render_message_to_json_deepseek_v4_1}, common/chat.cpp:568-601).
     */
    static final String DEEPSEEK_V41_DSML_CALLS = "\uFF5CDSML\uFF5C calls>";
    /** The part type that names a request tool a template renders the definition of (common/chat.cpp:385-387). */
    public static final String PART_TOOL_REFERENCE = "tool_reference";
    /** The tool a {@code tool_reference} content part names ({@code part.at("name")}). */
    private static final String PART_NAME_FIELD = "name";

    /** OpenAI content-part types the template input handles; every other part type is an error. */
    private static final String PART_TEXT = "text";
    private static final String PART_TEXT_FIELD = "text";
    /** The OpenAI image part, {@code {"type": "image_url", "image_url": {"url": ...}}}. */
    public static final String PART_IMAGE = "image_url";
    /** The OpenAI audio part, {@code {"type": "input_audio", "input_audio": {"data": ..., "format": ...}}}. */
    public static final String PART_AUDIO = "input_audio";
    /** vLLM's audio URL part, {@code {"type": "audio_url", "audio_url": {"url": ...}}}. */
    public static final String PART_AUDIO_URL = "audio_url";
    /** vLLM's video URL part, {@code {"type": "video_url", "video_url": {"url": ...}}}. */
    public static final String PART_VIDEO_URL = "video_url";
    /** llama.cpp's video part, {@code {"type": "input_video", "input_video": {"data": ...}}}. */
    public static final String PART_INPUT_VIDEO = "input_video";
    /** vLLM's precomputed-embedding content parts, whose rows a multimodal prompt splices into the tokens. */
    public static final String PART_IMAGE_EMBEDS = "image_embeds";
    public static final String PART_AUDIO_EMBEDS = "audio_embeds";
    private static final String TOOL_CALLS = "tool_calls";
    private static final String TOOL_CALL_ID = "tool_call_id";
    private static final String TOOL_ID = "id";
    private static final String TOOL_NAME = "name";
    private static final String TOOL_FUNCTION = "function";
    private static final String TOOL_ARGUMENTS = "arguments";
    private static final String TOOL_TYPE = "type";
    private static final String TOOL_TYPE_FUNCTION = "function";
    /** Tool-call arguments as JSON text: compact, insertion order, non-ASCII kept (nlohmann {@code dump()}). */
    private static final Json.Style ARGUMENTS_JSON = new Json.Style(false, null, ",", ":", false);

    private ChatMessages() {}

    /**
     * The media marker a flattened image or an audio part leaves in the message content, at the
     * place llama.cpp's mtmd request markers take (tools/mtmd/mtmd-helper.h), where a multimodal
     * prompt injects the encoded media. A render without media support refuses such a request
     * ({@code ChatFormatter#format(ChatRequest)}); a media-aware render keeps the markers for the
     * serving layer to replace ({@code ChatFormatter#format(ChatRequest, boolean)}).
     */
    public static final String MEDIA_MARKER = "<|__media__|>";

    /**
     * The marker an {@code image_embeds} or {@code audio_embeds} content part leaves in the message
     * content, vLLM's {@code <##PROMPT_EMBEDS##>} placeholder: the serving layer splits the rendered
     * prompt on it and splices the parts' precomputed embedding rows at those positions
     * (vllm/renderers/chat_utils/content_parser.py). Unlike {@link #MEDIA_MARKER}, it survives the text
     * chat path: the rows are fed without a media encoder.
     */
    public static final String PROMPT_EMBEDS_MARKER = "<##PROMPT_EMBEDS##>";

    /**
     * Validated copy of {@code messages}: every message is a new map; a missing or null {@code content}
     * becomes the empty string, as llama.cpp's string-typed message content is (common/chat.h common_chat_msg).
     * An assistant message's reasoning is carried under both {@code reasoning} and the deprecated
     * {@code reasoning_content} ({@link #normalizeReasoning}).
     *
     * @throws IllegalArgumentException for a null list, a null message, a message without a string role, or a
     *         non-string {@code reasoning}
     */
    public static List<Map<String, Object>> copyOf(List<? extends Map<String, ?>> messages) {
        if (messages == null) throw new IllegalArgumentException("chat messages are null");
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        for (int i = 0; i < messages.size(); i++) {
            Map<String, ?> m = messages.get(i);
            if (m == null) throw new IllegalArgumentException("chat message " + i + " is null");
            if (!(m.get(ROLE) instanceof String)) {
                throw new IllegalArgumentException("chat message " + i + " has no string 'role'");
            }
            Map<String, Object> copy = new LinkedHashMap<>(m);
            if (copy.get(CONTENT) == null) copy.put(CONTENT, "");
            normalizeReasoning(copy, i);
            out.add(copy);
        }
        return out;
    }

    /**
     * vLLM's message reasoning keys on an assistant message: {@code reasoning} is current and
     * {@code reasoning_content} its deprecated alias (protocol.py {@code _normalize_messages_before} and
     * {@code parse_chat_message_content}). Both carry the same text so a template written for either name renders
     * it, and {@code reasoning} wins where a caller sent both. A non-string {@code reasoning} (vLLM's
     * prompt-embedding form) is refused by name.
     */
    private static void normalizeReasoning(Map<String, Object> message, int index) {
        if (!"assistant".equals(message.get(ROLE))) {
            return;
        }
        Object reasoning = message.get(REASONING);
        if (reasoning != null && !(reasoning instanceof String)) {
            throw new IllegalArgumentException("chat message " + index
                    + ": 'reasoning' must be a string; this build takes reasoning text only");
        }
        Object text = reasoning != null ? reasoning : message.get(REASONING_CONTENT);
        if (!(text instanceof String)) {
            return;
        }
        message.put(REASONING_CONTENT, text);
        message.put(REASONING, text);
    }

    /**
     * The messages a template renders, after llama.cpp's input handling in its order: the content and argument
     * forms of the template's capabilities ({@link #normalize}), developer-to-system mapping for templates without
     * harmony channels, and the merge of a leading system message for templates that do not support the system
     * role. The StepFun trimming comes before, on the messages as sent ({@link #trimForTemplate}).
     *
     * @param messages       messages from {@link #trimForTemplate}, or {@link #forTemplate}
     * @param templateSource the resolved template source
     * @param caps           the template's capabilities ({@link TemplateCaps#probe})
     * @throws IllegalArgumentException for a content part or tool-call arguments the template input cannot carry
     */
    public static List<Map<String, Object>> prepare(List<Map<String, Object>> messages, String templateSource,
                                                    TemplateCaps caps) {
        List<Map<String, Object>> out = normalize(messages, caps);
        if (!templateSource.contains(HARMONY_CHANNEL)) out = mapDeveloperToSystem(out);
        if (!caps.supportsSystemRole()) out = mergeLeadingSystem(out);
        return out;
    }

    /** Whether {@code templateSource} is a StepFun template, whose message text is trimmed before rendering. */
    static boolean isStepFunTemplate(String templateSource) {
        return templateSource.contains(STEPFUN_TOOLS_PROMPT);
    }

    /**
     * The messages as sent, trimmed for a StepFun template ({@link #trimAllContent}) and unchanged for any other:
     * llama.cpp trims the messages before their parts join (common/chat.cpp:1476-1482), so a text part's own
     * whitespace goes even where the parts later join to one string.
     */
    static List<Map<String, Object>> trimForTemplate(List<Map<String, Object>> messages, String templateSource) {
        return isStepFunTemplate(templateSource) ? trimAllContent(messages) : messages;
    }

    /** Whether {@code templateSource} is a DeepSeek V4.1 template ({@code render_message_to_json_deepseek_v4_1}). */
    static boolean isDeepSeekV41Template(String templateSource) {
        return templateSource.contains(DEEPSEEK_V41_DSML_CALLS);
    }

    /**
     * Trims whitespace around string {@code content}, {@code reasoning_content} and {@code reasoning}, and around
     * the {@code text} of typed {@code text} content parts (common/chat.cpp {@code trim_all_content}).
     */
    static List<Map<String, Object>> trimAllContent(List<Map<String, Object>> messages) {
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        for (Map<String, Object> m : messages) {
            Map<String, Object> copy = new LinkedHashMap<>(m);
            copy.put(CONTENT, trimContent(copy.get(CONTENT)));
            trimReasoning(copy, REASONING_CONTENT);
            trimReasoning(copy, REASONING);
            out.add(copy);
        }
        return out;
    }

    /** Trims the reasoning text under {@code key}, when that key holds a string. */
    private static void trimReasoning(Map<String, Object> message, String key) {
        Object reasoning = message.get(key);
        if (reasoning instanceof String) message.put(key, trimWhitespace((String) reasoning));
    }

    /** A string content trimmed, or a typed part list with every {@code text} part's text trimmed, as new objects. */
    private static Object trimContent(Object content) {
        if (content instanceof String) return trimWhitespace((String) content);
        if (!(content instanceof List)) return content;
        List<?> parts = (List<?>) content;
        List<Object> out = new ArrayList<>(parts.size());
        for (Object part : parts) {
            if (part instanceof Map && TEXT.equals(((Map<?, ?>) part).get(TYPE))
                    && ((Map<?, ?>) part).get(TEXT) instanceof String) {
                Map<Object, Object> copy = new LinkedHashMap<>((Map<?, ?>) part);
                copy.put(TEXT, trimWhitespace((String) copy.get(TEXT)));
                out.add(copy);
            } else {
                out.add(part);
            }
        }
        return out;
    }

    /** C {@code isspace} trimming, as llama.cpp's {@code trim_whitespace} (common/chat-auto-parser-helpers.cpp:12-28). */
    static String trimWhitespace(String s) {
        int start = 0;
        int end = s.length();
        while (start < end && isCSpace(s.charAt(start))) start++;
        while (end > start && isCSpace(s.charAt(end - 1))) end--;
        return s.substring(start, end);
    }

    private static boolean isCSpace(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\u000B' || c == '\f' || c == '\r';
    }

    /** Renames the {@code developer} role to {@code system} (common/chat.cpp:989-997). */
    static List<Map<String, Object>> mapDeveloperToSystem(List<Map<String, Object>> messages) {
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        for (Map<String, Object> m : messages) {
            if (DEVELOPER.equals(m.get(ROLE))) {
                Map<String, Object> copy = new LinkedHashMap<>(m);
                copy.put(ROLE, SYSTEM);
                out.add(copy);
            } else {
                out.add(m);
            }
        }
        return out;
    }

    /**
     * Merges a leading system message into the next message as {@code system + "\n" + content}, or drops
     * it when it is the only message (common/chat.cpp:1000-1015, system_message_not_supported). A typed
     * next content gets the system text and the newline as a leading text part, and a typed system content
     * contributes its text parts joined as {@link #normalize} joins them.
     */
    static List<Map<String, Object>> mergeLeadingSystem(List<Map<String, Object>> messages) {
        if (messages.isEmpty() || !SYSTEM.equals(messages.get(0).get(ROLE))) return messages;
        if (messages.size() == 1) {
            LOGGER.warn("Removing the system prompt: the chat template does not support the system role");
            return Collections.emptyList();
        }
        LOGGER.debug("Merging the system prompt into the next message: the chat template does not support the system role");
        List<Map<String, Object>> out = new ArrayList<>(messages.size() - 1);
        Map<String, Object> next = new LinkedHashMap<>(messages.get(1));
        String system = mergeText(0, messages.get(0)) + "\n";
        Object content = next.get(CONTENT);
        if (content instanceof List) {
            List<Object> parts = new ArrayList<>(((List<?>) content).size() + 1);
            parts.add(textPart(system));
            parts.addAll((List<?>) content);
            next.put(CONTENT, parts);
        } else {
            next.put(CONTENT, system + mergeText(1, next));
        }
        out.add(next);
        out.addAll(messages.subList(2, messages.size()));
        return out;
    }

    /** The text of a message merged with the system prompt: its string content, or its text parts joined. */
    private static String mergeText(int index, Map<String, Object> m) {
        Object c = m.get(CONTENT);
        if (c instanceof String) return (String) c;
        if (c instanceof List) return joinTextParts(index, (List<?>) c);
        throw new IllegalArgumentException("cannot merge the system prompt: message content of role '"
                + m.get(ROLE) + "' is neither a string nor a list of content parts");
    }

    // -----------------------------------------------------------------------------------------
    // llama.cpp template-input conventions (common/chat.cpp apply_jinja, :1299 onward)
    // -----------------------------------------------------------------------------------------

    /**
     * The messages in llama.cpp's template-input form ({@code render_message_to_json}), before the template's
     * capabilities choose the content and argument forms ({@link #normalize}).
     *
     * @see #forTemplate(List, String)
     */
    static List<Map<String, Object>> forTemplate(List<Map<String, Object>> messages) {
        return forTemplate(messages, null);
    }

    /**
     * The messages in llama.cpp's template-input form, before the template's capabilities choose the content
     * and argument forms ({@link #normalize}): a multi-part {@code content} of text parts only stays a list of
     * {@code {"type": "text", "text": ...}} parts; one that carries media becomes a string, its text parts
     * joined as {@link #normalize} joins them, with {@link #MEDIA_MARKER} in place of every image and audio
     * part and {@link #PROMPT_EMBEDS_MARKER} in place of every embedding part, no newline around a marker
     * (common/chat.cpp {@code to_json_oaicompat}). Assistant {@code tool_calls} take the shape
     * {@link #normalizeToolCalls} gives them, their {@code arguments} validated as a JSON object and kept in the
     * form the caller gave (an object, or its JSON text). A {@code tool_reference} part is carried as
     * {@code {"type": "tool_reference", "name": ...}}, and skipped with a warning where the content joins to a
     * string, as llama.cpp skips it there.
     *
     * <p>{@code templateSource} shapes the conversion where llama.cpp keys it off the template: a StepFun
     * template ({@link #isStepFunTemplate}) gets the messages trimmed before their parts join
     * ({@link #trimForTemplate}), and a DeepSeek V4.1 template ({@link #isDeepSeekV41Template}) keeps media
     * markers as separate text parts and the tool calls' extra keys ({@code render_message_to_json_deepseek_v4_1}).
     * Every template gets the message keys beyond the OpenAI ones as sent, as vLLM passes them and as llama.cpp
     * restores them for DeepSeek V4.1.</p>
     *
     * <p>Applied before {@link #prepare}. Regenerates every message map; input data is never modified.</p>
     *
     * @param templateSource the resolved source of the template the messages render with, or null for
     *                       llama.cpp's default rendering
     * @throws IllegalArgumentException for an empty or non-object tool call, a missing tool-call
     *                                  function name, arguments that are not a JSON object, or a
     *                                  part whose {@code type} is not text, tool reference, image,
     *                                  audio, video or one of the embedding kinds
     */
    static List<Map<String, Object>> forTemplate(List<Map<String, Object>> messages, String templateSource) {
        boolean deepseek = templateSource != null && isDeepSeekV41Template(templateSource);
        List<Map<String, Object>> in = templateSource != null ? trimForTemplate(messages, templateSource) : messages;
        List<Map<String, Object>> out = new ArrayList<>(in.size());
        for (int i = 0; i < in.size(); i++) {
            Map<String, Object> copy = new LinkedHashMap<>(in.get(i));
            Object content = copy.get(CONTENT);
            if (content instanceof List) {
                List<?> parts = (List<?>) content;
                copy.put(CONTENT, deepseek && hasMedia(parts) ? deepSeekMediaParts(i, parts) : contentParts(i, parts));
            }
            Object calls = copy.get(TOOL_CALLS);
            if (calls instanceof List) {
                copy.put(TOOL_CALLS, normalizeToolCalls(i, (List<?>) calls, deepseek));
            }
            out.add(copy);
        }
        return out;
    }

    /**
     * Multi-part content as the template input carries it: the text parts as fresh {@code {type, text}} parts
     * when every part is text (an empty list is the empty string), otherwise one string with the media and
     * embedding markers in part order. A {@code tool_reference} part carries
     * the tool's {@code name}; it is not media, and a string join skips it with a warning, as llama.cpp skips
     * it there ({@code to_json_oaicompat}, {@code concat_content_parts}). The media URLs are carried
     * by the request's media list, not the template context.
     */
    private static Object contentParts(int messageIndex, List<?> parts) {
        if (parts.isEmpty()) return "";
        List<Object> texts = new ArrayList<>(parts.size());
        PartJoiner joined = new PartJoiner();
        boolean media = false;
        boolean toolReference = false;
        for (int p = 0; p < parts.size(); p++) {
            Object part = parts.get(p);
            if (!(part instanceof Map)) {
                throw partError(messageIndex, p, "is not an object");
            }
            Object type = ((Map<?, ?>) part).get(TYPE);
            if (PART_TEXT.equals(type)) {
                Object t = ((Map<?, ?>) part).get(PART_TEXT_FIELD);
                if (!(t instanceof String)) {
                    throw partError(messageIndex, p, "text part has no string 'text'");
                }
                texts.add(textPart((String) t));
                joined.text((String) t);
            } else if (PART_TOOL_REFERENCE.equals(type)) {
                texts.add(toolReferencePart(messageIndex, p, (Map<?, ?>) part));
                toolReference = true;
            } else {
                joined.marker(markerOf(messageIndex, p, type));
                media = true;
            }
        }
        if (media && toolReference) {
            LOGGER.warn("Ignoring content part type: {}", PART_TOOL_REFERENCE);
        }
        return media ? joined.toString() : texts;
    }

    /**
     * The content of a message that carries media, for a DeepSeek V4.1 template
     * ({@code render_message_to_json_deepseek_v4_1}): every part becomes a text part, media and embedding parts
     * their marker, so the template joins the blocks itself. As llama.cpp retypes the parts, a
     * {@code tool_reference} part becomes an empty text part.
     */
    private static List<Object> deepSeekMediaParts(int messageIndex, List<?> parts) {
        List<Object> out = new ArrayList<>(parts.size());
        for (int p = 0; p < parts.size(); p++) {
            if (!(parts.get(p) instanceof Map)) {
                throw partError(messageIndex, p, "is not an object");
            }
            Map<?, ?> part = (Map<?, ?>) parts.get(p);
            Object type = part.get(TYPE);
            if (PART_TEXT.equals(type)) {
                Object t = part.get(PART_TEXT_FIELD);
                if (!(t instanceof String)) {
                    throw partError(messageIndex, p, "text part has no string 'text'");
                }
                out.add(textPart((String) t));
            } else if (PART_TOOL_REFERENCE.equals(type)) {
                toolReferencePart(messageIndex, p, part); // its name is checked as anywhere else
                out.add(textPart(""));
            } else {
                out.add(textPart(markerOf(messageIndex, p, type)));
            }
        }
        return out;
    }

    /**
     * Whether {@code parts} carry a part other than text and tool references, an image, audio or embedding
     * part that joins a message's content to one string ({@code common_chat_msg::contains_media}).
     */
    private static boolean hasMedia(List<?> parts) {
        for (Object part : parts) {
            if (!(part instanceof Map)) continue;
            Object type = ((Map<?, ?>) part).get(TYPE);
            if (!PART_TEXT.equals(type) && !PART_TOOL_REFERENCE.equals(type)) return true;
        }
        return false;
    }

    /**
     * The marker a media or embedding part leaves in the content: {@link #MEDIA_MARKER} for an image, audio or
     * video part, {@link #PROMPT_EMBEDS_MARKER} for an embedding part; any other type is refused by name.
     */
    private static String markerOf(int messageIndex, int partIndex, Object type) {
        if (PART_IMAGE.equals(type) || PART_AUDIO.equals(type) || PART_AUDIO_URL.equals(type)
                || PART_VIDEO_URL.equals(type) || PART_INPUT_VIDEO.equals(type)) {
            return MEDIA_MARKER;
        }
        if (PART_IMAGE_EMBEDS.equals(type) || PART_AUDIO_EMBEDS.equals(type)) {
            return PROMPT_EMBEDS_MARKER;
        }
        throw partError(messageIndex, partIndex, "has unsupported type '" + type + "'");
    }

    /**
     * A {@code tool_reference} content part the template renders ({@code {"type": "tool_reference", "name": ...}},
     * llama.cpp's {@code common_chat_msg_content_part} with a name).
     */
    private static Map<String, Object> toolReferencePart(int messageIndex, int partIndex, Map<?, ?> part) {
        Object name = part.get(PART_NAME_FIELD);
        if (!(name instanceof String)) {
            throw partError(messageIndex, partIndex, "tool_reference part has no string 'name'");
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(TYPE, PART_TOOL_REFERENCE);
        out.put(PART_NAME_FIELD, name);
        return out;
    }

    private static Map<String, Object> textPart(String text) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put(TYPE, PART_TEXT);
        part.put(PART_TEXT_FIELD, text);
        return part;
    }

    private static IllegalArgumentException partError(int messageIndex, int partIndex, String what) {
        return new IllegalArgumentException("message " + messageIndex + " content part " + partIndex
                + " " + what);
    }

    /**
     * llama.cpp's joining of content parts into one string ({@code concat_content_parts}): a newline between
     * text parts, none before or after a marker.
     */
    private static final class PartJoiner {
        private final StringBuilder text = new StringBuilder();
        private boolean lastWasMarker;

        void text(String t) {
            if (!lastWasMarker && text.length() > 0) text.append('\n');
            text.append(t);
            lastWasMarker = false;
        }

        void marker(String marker) {
            text.append(marker);
            lastWasMarker = true;
        }

        @Override public String toString() { return text.toString(); }
    }

    /**
     * The text parts of a typed content joined to one string; a {@code tool_reference} part is skipped with a
     * warning, as llama.cpp's join skips the types it cannot print ({@code concat_content_parts}); any other
     * non-text part is refused by name.
     */
    private static String joinTextParts(int messageIndex, List<?> parts) {
        PartJoiner joined = new PartJoiner();
        for (int p = 0; p < parts.size(); p++) {
            Object part = parts.get(p);
            if (part instanceof Map && PART_TOOL_REFERENCE.equals(((Map<?, ?>) part).get(TYPE))) {
                LOGGER.warn("Ignoring content part type: {}", PART_TOOL_REFERENCE);
                continue;
            }
            if (!(part instanceof Map) || !PART_TEXT.equals(((Map<?, ?>) part).get(TYPE))) {
                throw partError(messageIndex, p, "is not a text part, and the chat template reads message"
                        + " content as a string");
            }
            Object t = ((Map<?, ?>) part).get(PART_TEXT_FIELD);
            if (!(t instanceof String)) {
                throw partError(messageIndex, p, "text part has no string 'text'");
            }
            joined.text((String) t);
        }
        return joined.toString();
    }

    // -----------------------------------------------------------------------------------------
    // Capability normalisation (common/chat.cpp messages_inp_normalizer, func_args_not_string)
    // -----------------------------------------------------------------------------------------

    /**
     * The messages in the content and argument forms the template reads, as llama.cpp's
     * {@code messages_inp_normalizer} and {@code func_args_not_string} give them: for a template that reads
     * only string content, a typed content becomes its text parts joined with newlines; for one that reads
     * only typed content, a string content becomes one text part; a template that reads both (or, unexpectedly,
     * neither) gets the content as it is. Tool-call {@code arguments} become an object for a template that
     * reads them as one and JSON text otherwise, text the caller gave staying as given.
     *
     * @throws IllegalArgumentException for a non-text part in content joined to a string, and for tool-call
     *                                  arguments that are not JSON when the template reads an object
     */
    public static List<Map<String, Object>> normalize(List<Map<String, Object>> messages, TemplateCaps caps) {
        boolean onlyString = caps.supportsStringContent() && !caps.supportsTypedContent();
        boolean onlyTyped = !caps.supportsStringContent() && caps.supportsTypedContent();
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        for (int i = 0; i < messages.size(); i++) {
            Map<String, Object> copy = new LinkedHashMap<>(messages.get(i));
            Object content = copy.get(CONTENT);
            if (onlyTyped && content instanceof String) {
                List<Object> parts = new ArrayList<>(1);
                parts.add(textPart((String) content));
                copy.put(CONTENT, parts);
            } else if (onlyString && content instanceof List) {
                copy.put(CONTENT, joinTextParts(i, (List<?>) content));
            }
            Object calls = copy.get(TOOL_CALLS);
            if (calls instanceof List) {
                copy.put(TOOL_CALLS, argumentForms(i, (List<?>) calls, caps.supportsObjectArguments()));
            }
            out.add(copy);
        }
        return out;
    }

    /** Tool calls with their flat and nested {@code arguments} as an object or as JSON text. */
    private static List<Object> argumentForms(int messageIndex, List<?> calls, boolean object) {
        List<Object> out = new ArrayList<>(calls.size());
        for (int c = 0; c < calls.size(); c++) {
            Object call = calls.get(c);
            if (!(call instanceof Map)) {
                out.add(call);
                continue;
            }
            Map<Object, Object> copy = new LinkedHashMap<>((Map<?, ?>) call);
            if (copy.containsKey(TOOL_ARGUMENTS)) {
                copy.put(TOOL_ARGUMENTS, argumentForm(messageIndex, c, copy.get(TOOL_ARGUMENTS), object));
            }
            if (copy.get(TOOL_FUNCTION) instanceof Map) {
                Map<Object, Object> function = new LinkedHashMap<>((Map<?, ?>) copy.get(TOOL_FUNCTION));
                if (function.containsKey(TOOL_ARGUMENTS)) {
                    function.put(TOOL_ARGUMENTS, argumentForm(messageIndex, c, function.get(TOOL_ARGUMENTS), object));
                }
                copy.put(TOOL_FUNCTION, function);
            }
            out.add(copy);
        }
        return out;
    }

    private static Object argumentForm(int messageIndex, int callIndex, Object arguments, boolean object) {
        if (object && arguments instanceof String) {
            return parseArguments(messageIndex, callIndex, (String) arguments);
        }
        if (!object && (arguments instanceof Map || arguments instanceof List)) {
            return Json.toString(arguments, ARGUMENTS_JSON);
        }
        return arguments;
    }

    private static Object parseArguments(int messageIndex, int callIndex, String arguments) {
        try {
            return Json.parse(arguments, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException("tool call " + callIndex + " of message " + messageIndex
                    + " has arguments that are not valid JSON: " + e.getMessage(), e);
        }
    }

    /**
     * Assistant {@code tool_calls} the templates render ({@code message.tool_calls}): each entry
     * normalised, as llama.cpp's parsed tool calls are, to
     * {@code {name, arguments [, id] [, function: {name, arguments}]}} — the flat keys for
     * Mistral-style templates, the nested {@code function} for the OpenAI-style ones, both bound,
     * since every Jinja tool template reads one of the two shapes. The arguments must be a JSON
     * object; they keep the caller's form, an object or its JSON text, until {@link #normalize}.
     *
     * @param keepExtras whether keys beside the tool-call shape stay ({@code render_message_to_json_deepseek_v4_1}
     *                   restores them: every key beside {@code type}, {@code function} and {@code id}, and every
     *                   {@code function} key beside {@code name} and {@code arguments})
     */
    private static List<Map<String, Object>> normalizeToolCalls(int messageIndex, List<?> calls,
                                                                boolean keepExtras) {
        List<Map<String, Object>> out = new ArrayList<>(calls.size());
        for (int c = 0; c < calls.size(); c++) {
            Object call = calls.get(c);
            if (!(call instanceof Map)) {
                throw new IllegalArgumentException("tool call " + c + " of message " + messageIndex
                        + " is not an object");
            }
            Map<?, ?> in = (Map<?, ?>) call;
            Map<?, ?> function = in.get(TOOL_FUNCTION) instanceof Map ? (Map<?, ?>) in.get(TOOL_FUNCTION) : null;
            String name = stringField(function, TOOL_NAME, stringField(in, TOOL_NAME, null));
            if (name == null) {
                throw new IllegalArgumentException("tool call " + c + " of message " + messageIndex
                        + " has no function name");
            }
            Object arguments = function != null && function.get(TOOL_ARGUMENTS) != null
                    ? function.get(TOOL_ARGUMENTS) : in.get(TOOL_ARGUMENTS);
            Object parsed = arguments instanceof String
                    ? parseArguments(messageIndex, c, (String) arguments) : arguments;
            if (!(parsed instanceof Map)) {
                throw new IllegalArgumentException("tool call " + c + " of message " + messageIndex
                        + " has arguments that are not a JSON object");
            }
            out.add(toolCall(in, function, name, arguments, keepExtras));
        }
        return out;
    }

    /** One tool call in the render shape: optional id, type, flat name and arguments, nested function. */
    private static Map<String, Object> toolCall(Map<?, ?> in, Map<?, ?> function, String name, Object arguments,
                                                boolean keepExtras) {
        Map<String, Object> functionCopy = new LinkedHashMap<>();
        functionCopy.put(TOOL_NAME, name);
        functionCopy.put(TOOL_ARGUMENTS, arguments);
        Map<String, Object> outCall = new LinkedHashMap<>();
        String id = stringField(in, TOOL_ID, null);
        if (id != null) outCall.put(TOOL_ID, id);
        Object type = in.get(TOOL_TYPE);
        if (type instanceof String) outCall.put(TOOL_TYPE, type);
        else outCall.put(TOOL_TYPE, TOOL_TYPE_FUNCTION);
        outCall.put(TOOL_NAME, name);
        outCall.put(TOOL_ARGUMENTS, arguments);
        outCall.put(TOOL_FUNCTION, functionCopy);
        if (keepExtras) {
            for (Map.Entry<?, ?> e : in.entrySet()) {
                if (e.getKey() instanceof String && isToolCallExtraKey((String) e.getKey())) {
                    outCall.put((String) e.getKey(), e.getValue());
                }
            }
            if (function != null) {
                for (Map.Entry<?, ?> e : function.entrySet()) {
                    if (e.getKey() instanceof String && isToolCallFunctionExtraKey((String) e.getKey())) {
                        functionCopy.put((String) e.getKey(), e.getValue());
                    }
                }
            }
        }
        return outCall;
    }

    /** A tool-call key the render shape does not carry, which a DeepSeek V4.1 render keeps on the call. */
    private static boolean isToolCallExtraKey(String key) {
        return !TOOL_TYPE.equals(key) && !TOOL_FUNCTION.equals(key) && !TOOL_ID.equals(key);
    }

    /** A tool-call function key the render shape does not carry, kept beside {@code name} and {@code arguments}. */
    private static boolean isToolCallFunctionExtraKey(String key) {
        return !TOOL_NAME.equals(key) && !TOOL_ARGUMENTS.equals(key);
    }

    private static String stringField(Map<?, ?> m, String key, String fallback) {
        Object v = m == null ? null : m.get(key);
        return v instanceof String ? (String) v : fallback;
    }

    // -----------------------------------------------------------------------------------------
    // Legacy render shape
    // -----------------------------------------------------------------------------------------

    /**
     * A legacy-shape render — a message list and the always-set {@code enable_thinking} variable,
     * the shape llama.cpp and the OpenAI chat completions API use — as the equivalent
     * {@link ChatRequest}.
     */
    public static ChatRequest request(List<? extends Map<String, ?>> messages, boolean enableThinking) {
        return ChatRequest.builder().messages(messages).enableThinking(enableThinking).build();
    }

    // -----------------------------------------------------------------------------------------
    // JSON value trees
    // -----------------------------------------------------------------------------------------

    /**
     * Deep copy of a parsed JSON value (string, number, boolean, null, object with string keys,
     * array), or an error when the tree is not one. {@code subject} names the offending request
     * member in the message.
     */
    static Object copyValue(Object value, String subject) {
        if (value == null || value instanceof String || value instanceof Boolean
                || value instanceof Integer || value instanceof Long || value instanceof Short
                || value instanceof Byte || value instanceof Double || value instanceof Float) {
            return value;
        }
        if (value instanceof Map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) value).entrySet()) {
                if (!(e.getKey() instanceof String)) {
                    throw jsonError(subject, "a non-string " + e.getKey().getClass().getSimpleName() + " key");
                }
                copy.put((String) e.getKey(), copyValue(e.getValue(), subject + " key '" + e.getKey() + "'"));
            }
            return copy;
        }
        if (value instanceof List) {
            List<Object> copy = new ArrayList<>(((List<?>) value).size());
            for (Object item : (List<?>) value) {
                copy.add(copyValue(item, subject));
            }
            return copy;
        }
        if (value instanceof Character) {
            return value.toString();
        }
        throw jsonError(subject, "a " + value.getClass().getSimpleName() + " value, which is not JSON");
    }

    private static IllegalArgumentException jsonError(String subject, String what) {
        return new IllegalArgumentException(subject + " has " + what);
    }
}
