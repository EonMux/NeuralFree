package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One Responses item as chat messages (llama.cpp's {@code server_chat_convert_responses_to_chatcmpl} item cases):
 * an input message ({@code user}, {@code system}, {@code developer}; string or {@code input_text} /
 * {@code input_image} / {@code input_audio} parts), an output message ({@code assistant}; {@code output_text} or
 * {@code refusal} parts), a {@code function_call}, a {@code function_call_output} and a {@code reasoning} item.
 * Assistant-side items extend the previous message when it is the assistant's, so one turn's reasoning, text and
 * calls render as one message. Stateless.
 */
final class ResponsesItems {

    private ResponsesItems() {
    }

    /** Appends the messages of {@code item} to {@code out}, merging into a trailing assistant message. */
    static void append(Map<String, Object> item, String where, List<Map<String, Object>> out) {
        String type = item.get("type") instanceof String t ? t : null;
        Object role = item.get("role");
        if (role instanceof String r && (type == null || "message".equals(type))) {
            if ("assistant".equals(r)) {
                assistantText(parts(item.get("content"), where, true), lastAssistant(out));
            } else if ("user".equals(r) || "system".equals(r) || "developer".equals(r)) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("role", r);
                m.put("content", parts(item.get("content"), where, false));
                out.add(m);
            } else {
                throw ServerJson.badRequest(where + ".role must be user, system, developer or assistant");
            }
            return;
        }
        if (type == null) {
            throw ServerJson.badRequest("Cannot determine the type of " + where);
        }
        switch (type) {
            case "function_call" -> functionCall(item, where, out);
            case "function_call_output" -> out.add(functionOutput(item, where));
            case "reasoning" -> reasoning(item, out);
            case "item_reference" -> throw ServerJson.badRequest(where
                    + ": item_reference is not supported; send the item itself");
            default -> throw ServerJson.badRequest(where + ".type '" + type + "' is not one of message,"
                    + " function_call, function_call_output, reasoning");
        }
    }

    /** The trailing assistant message, appended when the last message is another role's. */
    private static Map<String, Object> lastAssistant(List<Map<String, Object>> out) {
        if (!out.isEmpty() && "assistant".equals(out.get(out.size() - 1).get("role"))) {
            return out.get(out.size() - 1);
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", "assistant");
        m.put("content", new ArrayList<>());
        out.add(m);
        return m;
    }

    @SuppressWarnings("unchecked")
    private static void assistantText(List<Object> parts, Map<String, Object> message) {
        Object content = message.get("content");
        List<Object> merged = content instanceof List<?> list ? new ArrayList<>((List<Object>) list)
                : new ArrayList<>();
        if (content instanceof String s && !s.isEmpty()) {
            merged.add(textPart(s));
        }
        merged.addAll(parts);
        message.put("content", merged);
    }

    /**
     * Chat content parts of a Responses {@code content}: a string is one text part; {@code input_text} and
     * {@code output_text} are text, {@code input_image} an image URL, {@code input_audio} the chat audio part, and an
     * assistant's {@code refusal} its text. A media part's {@code uuid} (vLLM's media cache key) carries over, and
     * with it the media value may be absent.
     */
    private static List<Object> parts(Object content, String where, boolean assistant) {
        List<Object> out = new ArrayList<>();
        if (content instanceof String s) {
            out.add(textPart(s));
            return out;
        }
        if (!(content instanceof List<?> list)) {
            throw ServerJson.badRequest(where + ".content must be a string or an array of content parts");
        }
        for (int i = 0; i < list.size(); i++) {
            String at = where + ".content[" + i + "]";
            if (!(list.get(i) instanceof Map<?, ?> part)) {
                throw ServerJson.badRequest(at + " must be an object");
            }
            out.add(part(part, at, assistant));
        }
        return out;
    }

    private static Object part(Map<?, ?> part, String at, boolean assistant) {
        Object type = part.get("type");
        if ("input_text".equals(type) || "output_text".equals(type)) {
            if (!(part.get("text") instanceof String text)) {
                throw ServerJson.badRequest(at + " requires 'text'");
            }
            return textPart(text);
        }
        if (assistant && "refusal".equals(type)) {
            if (!(part.get("refusal") instanceof String refusal)) {
                throw ServerJson.badRequest(at + " requires 'refusal'");
            }
            return textPart(refusal);
        }
        if (!assistant && "input_image".equals(type)) {
            if (part.get("file_id") != null && part.get("image_url") == null) {
                throw ServerJson.badRequest(at + ".file_id is refused: this server has no Files API; pass the image"
                        + " as image_url (an http(s) or data URL)");
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("type", "image_url");
            if (part.get("image_url") instanceof String url) {
                out.put("image_url", Map.of("url", url));
            } else if (part.get("image_url") != null || part.get("uuid") == null) {
                throw ServerJson.badRequest(at + " requires 'image_url'");
            }
            return withUuid(out, part);
        }
        if (!assistant && "input_audio".equals(type)) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("type", "input_audio");
            if (part.get("input_audio") instanceof Map<?, ?> audio && audio.get("data") instanceof String data) {
                Map<String, Object> inner = new LinkedHashMap<>();
                inner.put("data", data);
                if (audio.get("format") instanceof String format) {
                    inner.put("format", format);
                }
                out.put("input_audio", inner);
            } else if (part.get("input_audio") != null || part.get("uuid") == null) {
                throw ServerJson.badRequest(at + " requires 'input_audio' with base64 'data'");
            }
            return withUuid(out, part);
        }
        if ("input_file".equals(type)) {
            throw ServerJson.badRequest(at + ": 'input_file' is not supported: this server has no Files API");
        }
        throw ServerJson.badRequest(at + ".type must be one of "
                + (assistant ? "output_text or refusal" : "input_text, input_image or input_audio"));
    }

    /** {@code out} with the {@code uuid} of the source {@code part}, when it has one (checked where it is used). */
    private static Map<String, Object> withUuid(Map<String, Object> out, Map<?, ?> part) {
        if (part.get("uuid") != null) {
            out.put("uuid", part.get("uuid"));
        }
        return out;
    }

    private static Map<String, Object> textPart(String text) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("type", "text");
        out.put("text", text);
        return out;
    }

    @SuppressWarnings("unchecked")
    private static void functionCall(Map<String, Object> item, String where, List<Map<String, Object>> out) {
        if (!(item.get("call_id") instanceof String callId) || !(item.get("name") instanceof String name)
                || !(item.get("arguments") instanceof String arguments)) {
            throw ServerJson.badRequest(where + ": a function_call needs string call_id, name and arguments");
        }
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", name);
        fn.put("arguments", arguments);
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("id", callId);
        call.put("type", "function");
        call.put("function", fn);
        Map<String, Object> message = lastAssistant(out);
        List<Object> calls = message.get("tool_calls") instanceof List<?> list
                ? new ArrayList<>((List<Object>) list) : new ArrayList<>();
        calls.add(call);
        message.put("tool_calls", calls);
    }

    private static Map<String, Object> functionOutput(Map<String, Object> item, String where) {
        if (!(item.get("call_id") instanceof String callId)) {
            throw ServerJson.badRequest(where + ": a function_call_output needs a string call_id");
        }
        Object output = item.get("output");
        Object content;
        if (output instanceof String s) {
            content = s;
        } else if (output instanceof List<?>) {
            content = parts(output, where + ".output", false);
        } else {
            throw ServerJson.badRequest(where + ".output must be a string or an array of content parts");
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", "tool");
        m.put("tool_call_id", callId);
        m.put("content", content);
        return m;
    }

    /**
     * A reasoning item's text ({@code reasoning_text} content, else its summary) as the assistant's
     * {@code reasoning_content}; an item carrying neither (only encrypted content) adds nothing.
     */
    private static void reasoning(Map<String, Object> item, List<Map<String, Object>> out) {
        String text = joinedText(item.get("content"));
        if (text.isEmpty()) {
            text = joinedText(item.get("summary"));
        }
        if (text.isEmpty()) {
            return;
        }
        Map<String, Object> message = lastAssistant(out);
        Object before = message.get("reasoning_content");
        message.put("reasoning_content", before instanceof String s && !s.isEmpty() ? s + "\n" + text : text);
    }

    private static String joinedText(Object parts) {
        if (!(parts instanceof List<?> list)) {
            return "";
        }
        StringBuilder b = new StringBuilder();
        for (Object o : list) {
            if (o instanceof Map<?, ?> part && part.get("text") instanceof String text) {
                b.append(b.length() == 0 ? "" : "\n").append(text);
            }
        }
        return b.toString();
    }
}
