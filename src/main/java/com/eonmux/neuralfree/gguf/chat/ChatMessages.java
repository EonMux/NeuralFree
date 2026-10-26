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
    static final String CONTENT = "content";
    private static final String REASONING_CONTENT = "reasoning_content";
    private static final String TYPE = "type";
    private static final String TEXT = "text";
    private static final String SYSTEM = "system";
    private static final String DEVELOPER = "developer";

    /** Marker of the gpt-oss harmony templates, which keep the developer role (common/chat.cpp:1347). */
    private static final String HARMONY_CHANNEL = "<|channel|>";
    /** Marker of the StepFun templates, whose message text is trimmed before rendering (common/chat.cpp:1308-1315). */
    private static final String STEPFUN_TOOLS_PROMPT = "You have access to the following functions in JSONSchema format";

    /** OpenAI content-part types MediaSchema handles; every other part type is an error. */
    private static final String PART_TEXT = "text";
    private static final String PART_TEXT_FIELD = "text";
    private static final String PART_IMAGE = "image_url";
    private static final String PART_AUDIO = "input_audio";
    private static final String TOOL_CALLS = "tool_calls";
    private static final String TOOL_CALL_ID = "tool_call_id";
    private static final String TOOL_ID = "id";
    private static final String TOOL_NAME = "name";
    private static final String TOOL_FUNCTION = "function";
    private static final String TOOL_ARGUMENTS = "arguments";
    private static final String TOOL_TYPE = "type";
    private static final String TOOL_TYPE_FUNCTION = "function";

    private ChatMessages() {}

    /**
     * The media marker a flattened image or an audio part leaves in the message content: MM-5's
     * {@code MultimodalPrompt} splits the rendered prompt at this marker and injects the encoded
     * media there, as llama.cpp's mtmd request markers do (tools/mtmd/mtmd-helper.h). Until that
     * prompt path exists, a request whose messages contain a media part is refused
     * ({@code ChatFormatter#format(ChatRequest)}, §3.4).
     */
    static final String MEDIA_MARKER = "<|__media__|>";

    /**
     * Validated copy of {@code messages}: every message is a new map; a missing or null {@code content}
     * becomes the empty string, as llama.cpp's string-typed message content is (common/chat.h common_chat_msg).
     *
     * @throws IllegalArgumentException for a null list, a null message, or a message without a string role
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
            out.add(copy);
        }
        return out;
    }

    /**
     * The messages a template renders, after llama.cpp's workarounds in their order: StepFun content
     * trimming, developer-to-system mapping for templates without harmony channels, and the merge of a
     * leading system message for templates that do not support the system role.
     *
     * @param messages           messages from {@link #copyOf}
     * @param templateSource     the resolved template source
     * @param supportsSystemRole whether the template renders a system message (see {@link ChatFormatter})
     */
    public static List<Map<String, Object>> prepare(List<Map<String, Object>> messages, String templateSource,
                                                    boolean supportsSystemRole) {
        List<Map<String, Object>> out = messages;
        if (templateSource.contains(STEPFUN_TOOLS_PROMPT)) out = trimAllContent(out);
        if (!templateSource.contains(HARMONY_CHANNEL)) out = mapDeveloperToSystem(out);
        if (!supportsSystemRole) out = mergeLeadingSystem(out);
        return out;
    }

    /**
     * Trims whitespace around string {@code content} and {@code reasoning_content}, and around the {@code text}
     * of typed {@code text} content parts (common/chat.cpp:1051-1061).
     */
    static List<Map<String, Object>> trimAllContent(List<Map<String, Object>> messages) {
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        for (Map<String, Object> m : messages) {
            Map<String, Object> copy = new LinkedHashMap<>(m);
            copy.put(CONTENT, trimContent(copy.get(CONTENT)));
            Object reasoning = copy.get(REASONING_CONTENT);
            if (reasoning instanceof String) copy.put(REASONING_CONTENT, trimWhitespace((String) reasoning));
            out.add(copy);
        }
        return out;
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
     * it when it is the only message (common/chat.cpp:1000-1015, system_message_not_supported).
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
        next.put(CONTENT, text(messages.get(0)) + "\n" + text(next));
        out.add(next);
        out.addAll(messages.subList(2, messages.size()));
        return out;
    }

    /** The string content of a message; llama.cpp merges string contents only (common/chat.cpp:1007-1008). */
    private static String text(Map<String, Object> m) {
        Object c = m.get(CONTENT);
        if (!(c instanceof String)) {
            throw new IllegalArgumentException("cannot merge the system prompt: message content of role '"
                    + m.get(ROLE) + "' is not a string");
        }
        return (String) c;
    }

    // -----------------------------------------------------------------------------------------
    // llama.cpp template-input conventions (common/chat.cpp apply_jinja, :1299 onward)
    // -----------------------------------------------------------------------------------------

    /**
     * The messages as the template context receives them, after llama.cpp's input conventions:
     * multi-part {@code content} concatenated to text (text parts in order), image and audio
     * parts replaced by {@link #MEDIA_MARKER}, and assistant {@code tool_calls} arguments as
     * JSON objects (a string form is parsed, matching llama.cpp's convention of keeping the
     * arguments the model wrote but rendering them as objects).
     *
     * <p>Applied before {@link #prepare}, so its trimming and system-role workarounds see the
     * flattened contents. Regenerates every message map; input data is never modified.</p>
     *
     * @throws IllegalArgumentException for an empty or non-object tool call, a missing tool-call
     *                                  function name, arguments that parse to a non-object, or a
     *                                  part whose {@code type} is not text, image or audio
     */
    static List<Map<String, Object>> forTemplate(List<Map<String, Object>> messages) {
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        for (int i = 0; i < messages.size(); i++) {
            Map<String, Object> m = messages.get(i);
            Map<String, Object> copy = new LinkedHashMap<>(m);
            Object content = copy.get(CONTENT);
            if (content instanceof List) {
                copy.put(CONTENT, flattenParts(i, (List<?>) content));
            }
            Object calls = copy.get(TOOL_CALLS);
            if (calls instanceof List) {
                copy.put(TOOL_CALLS, normalizeToolCalls(i, (List<?>) calls));
            }
            out.add(copy);
        }
        return out;
    }

    /**
     * Multi-part content concatenated to text: every {@code text} part's quoted {@code text} in
     * order, and {@link #MEDIA_MARKER} in place of every image and audio part
     * (common/chat.cpp's content-part handling). The image and audio URLs are carried by the
     * request's media list, not the template context (MM-5).
     */
    private static String flattenParts(int messageIndex, List<?> parts) {
        StringBuilder sb = new StringBuilder();
        boolean any = false;
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
                sb.append(t);
            } else if (PART_IMAGE.equals(type) || PART_AUDIO.equals(type)) {
                sb.append(MEDIA_MARKER);
            } else {
                throw partError(messageIndex, p, "has unsupported type '" + type + "'");
            }
            any = true;
        }
        return any ? sb.toString() : "";
    }

    private static IllegalArgumentException partError(int messageIndex, int partIndex, String what) {
        return new IllegalArgumentException("message " + messageIndex + " content part " + partIndex
                + " " + what);
    }

    /**
     * Assistant {@code tool_calls} the templates render ({@code message.tool_calls}): each entry
     * normalised, as llama.cpp's parsed tool calls are, to
     * {@code {name, arguments(object) [, id] [, function: {name, arguments}]}} — the flat keys for
     * Mistral-style templates, the nested {@code function} for the OpenAI-style ones, both bound,
     * since every Jinja tool template reads one of the two shapes.
     */
    private static List<Map<String, Object>> normalizeToolCalls(int messageIndex, List<?> calls) {
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
            if (arguments instanceof String) {
                try {
                    arguments = Json.parse(arguments.toString(), Json.Limits.UNTRUSTED);
                } catch (JsonException e) {
                    throw new IllegalArgumentException("tool call " + c + " of message " + messageIndex
                            + " has arguments that are not valid JSON: " + e.getMessage(), e);
                }
            }
            if (!(arguments instanceof Map)) {
                throw new IllegalArgumentException("tool call " + c + " of message " + messageIndex
                        + " has arguments that are not a JSON object");
            }
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
            out.add(outCall);
        }
        return out;
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
