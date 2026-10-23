package com.eonmux.neuralfree.gguf.chat;

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

    private ChatMessages() {}

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
}
