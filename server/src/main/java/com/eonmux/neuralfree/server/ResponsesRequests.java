package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * The OpenAI Responses request as a chat-completions body (llama.cpp's
 * {@code server_chat_convert_responses_to_chatcmpl}, vLLM's {@code construct_input_messages}): {@code instructions}
 * become the system message, the stored conversation of {@code previous_response_id} follows (its system message
 * left out, as instructions do not carry over), then the {@code input} items. Consecutive assistant items (reasoning,
 * output message, function call) merge into one assistant message. {@code tools} of type {@code function} become chat
 * tools (built-in tool types have no chat equivalent and are skipped), {@code tool_choice}, {@code max_output_tokens},
 * {@code reasoning.effort}, {@code text.format} and {@code top_logprobs} map onto their chat fields, and
 * {@code extra_params} merge in last (llama.cpp's escape hatch for sampler fields). Every other field passes through.
 * Stateless.
 */
final class ResponsesRequests {

    private static final Logger LOG = Logger.getLogger(ResponsesRequests.class.getName());

    /** Responses fields that convert, or that only the Responses layer reads. */
    private static final Set<String> RESPONSES_ONLY = Set.of("input", "instructions", "tools", "tool_choice",
            "max_output_tokens", "reasoning", "text", "extra_params", "previous_response_id", "store", "background",
            "metadata", "include", "truncation", "top_logprobs", "stream", "stream_options", "user", "service_tier",
            "prompt_cache_key", "safety_identifier", "conversation", "max_tool_calls", "prompt");

    private ResponsesRequests() {
    }

    /**
     * The chat-completions body of {@code body}.
     *
     * @param history the stored conversation {@code previous_response_id} names, or an empty list
     * @throws jakarta.ws.rs.WebApplicationException 400 on a malformed or unsupported request
     */
    static Map<String, Object> chatBody(JsonView body, List<Map<String, Object>> history) {
        if (!body.has("input")) {
            throw ServerJson.badRequest("'input' is required");
        }
        Map<String, Object> chat = new LinkedHashMap<>();
        for (String key : body.keys()) {
            if (!RESPONSES_ONLY.contains(key)) {
                chat.put(key, body.raw(key));
            }
        }
        chat.put("messages", messages(body, history));
        if (body.raw("tools") != null) {
            List<Object> tools = tools(body.raw("tools"));
            if (!tools.isEmpty()) {
                chat.put("tools", tools);
            }
        }
        if (body.raw("tool_choice") != null) {
            chat.put("tool_choice", toolChoice(body.raw("tool_choice")));
        }
        if (body.raw("max_output_tokens") != null) {
            chat.put("max_tokens", body.raw("max_output_tokens"));
        }
        JsonView reasoning = body.optObj("reasoning");
        if (reasoning.has("effort") && reasoning.raw("effort") != null) {
            chat.put("reasoning_effort", reasoning.raw("effort"));
        }
        Map<String, Object> format = responseFormat(body.optObj("text"));
        if (format != null) {
            chat.put("response_format", format);
        }
        logprobs(body, chat);
        if (body.raw("extra_params") instanceof Map<?, ?> extra) {
            for (Map.Entry<?, ?> e : extra.entrySet()) {
                chat.put(String.valueOf(e.getKey()), e.getValue());
            }
        } else if (body.raw("extra_params") != null) {
            throw ServerJson.badRequest("'extra_params' must be an object");
        }
        return chat;
    }

    /** The input items of {@code body} as Responses items: a string input is one user message. */
    static List<Object> inputItems(JsonView body) {
        Object input = body.raw("input");
        if (input instanceof String text) {
            Map<String, Object> part = new LinkedHashMap<>();
            part.put("type", "input_text");
            part.put("text", text);
            Map<String, Object> message = new LinkedHashMap<>();
            message.put("type", "message");
            message.put("role", "user");
            message.put("content", List.of(part));
            return List.of(message);
        }
        if (input instanceof List<?> list) {
            // nulls stay, for the per-item check to refuse by position
            return Collections.unmodifiableList(new ArrayList<>(list));
        }
        throw ServerJson.badRequest("'input' must be a string or an array of items");
    }

    /**
     * The chat messages of Responses {@code items} (input items, or a response's output items), merging consecutive
     * assistant items into one message.
     */
    static List<Map<String, Object>> messagesOf(List<?> items) {
        List<Map<String, Object>> out = new ArrayList<>();
        appendItems(items, out);
        return out;
    }

    /** Appends the messages of {@code items} to {@code out}; the first assistant item may extend its last message. */
    private static void appendItems(List<?> items, List<Map<String, Object>> out) {
        for (int i = 0; i < items.size(); i++) {
            if (!(items.get(i) instanceof Map<?, ?> raw)) {
                throw ServerJson.badRequest("input[" + i + "] must be an object");
            }
            ResponsesItems.append(OpenAiRequests.stringKeys(raw), "input[" + i + "]", out);
        }
    }

    private static List<Map<String, Object>> messages(JsonView body, List<Map<String, Object>> history) {
        List<Map<String, Object>> out = new ArrayList<>();
        Object instructions = body.raw("instructions");
        if (instructions != null) {
            if (!(instructions instanceof String text)) {
                throw ServerJson.badRequest("'instructions' must be a string");
            }
            out.add(message("system", text));
        }
        for (Map<String, Object> m : history) {
            if (!"system".equals(m.get("role"))) {
                // a copy: a following assistant item merges into it, and the stored conversation stays as it was
                out.add(new LinkedHashMap<>(m));
            }
        }
        appendItems(inputItems(body), out);
        if (out.isEmpty()) {
            throw ServerJson.badRequest("'input' holds no message");
        }
        return out;
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    /** Responses function tools ({@code {"type": "function", "name", "parameters", ...}}) as chat tools. */
    private static List<Object> tools(Object raw) {
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("'tools' must be an array of objects");
        }
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof Map<?, ?> tool)) {
                throw ServerJson.badRequest("tools[" + i + "] must be an object");
            }
            Object type = tool.get("type");
            if (!"function".equals(type)) {
                LOG.warning("Responses tool type '" + type + "' has no chat equivalent; skipped");
                continue;
            }
            Map<String, Object> fn = OpenAiRequests.stringKeys(tool);
            fn.remove("type");
            Map<String, Object> chatTool = new LinkedHashMap<>();
            chatTool.put("type", "function");
            chatTool.put("function", fn);
            out.add(chatTool);
        }
        return out;
    }

    /** {@code none} / {@code auto} / {@code required}, or {@code {"type": "function", "name": ...}}. */
    private static Object toolChoice(Object raw) {
        if (raw instanceof String) {
            return raw;
        }
        if (raw instanceof Map<?, ?> m && "function".equals(m.get("type")) && m.get("name") instanceof String name) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("type", "function");
            out.put("function", Map.of("name", name));
            return out;
        }
        throw ServerJson.badRequest("tool_choice must be none, auto, required or"
                + " {\"type\": \"function\", \"name\": ...}");
    }

    /**
     * {@code text.format}: {@code text}, {@code json_object}, {@code json_schema} or {@code structural_tag} as chat's
     * response_format; a structural tag passes through whole (its {@code format} tree or the legacy
     * {@code structures} and {@code triggers}), as chat's {@code response_format} of that type takes it.
     */
    private static Map<String, Object> responseFormat(JsonView text) {
        if (!text.has("format")) {
            return null;
        }
        JsonView format = text.obj("format");
        String type = format.optStr("type", "text");
        switch (type) {
            case "text":
                return null;
            case "json_object":
                return Map.of("type", "json_object");
            case "json_schema": {
                if (!format.has("schema")) {
                    throw ServerJson.badRequest("text.format of type json_schema requires 'schema'");
                }
                Map<String, Object> spec = new LinkedHashMap<>();
                spec.put("name", format.optStr("name", "response"));
                spec.put("schema", format.raw("schema"));
                spec.put("strict", format.optBool("strict", false));
                Map<String, Object> out = new LinkedHashMap<>();
                out.put("type", "json_schema");
                out.put("json_schema", spec);
                return out;
            }
            case "structural_tag": {
                if (!format.has("format") && !format.has("structures")) {
                    throw ServerJson.badRequest("text.format of type structural_tag requires 'format'"
                            + " (or the legacy 'structures' and 'triggers')");
                }
                return OpenAiRequests.stringKeys((Map<?, ?>) text.raw("format"));
            }
            default:
                throw ServerJson.badRequest("text.format type '" + type
                        + "' is not one of text, json_object, json_schema, structural_tag");
        }
    }

    /** {@code top_logprobs}, or {@code include} naming {@code message.output_text.logprobs}, ask for logprobs. */
    private static void logprobs(JsonView body, Map<String, Object> chat) {
        boolean included = body.raw("include") instanceof List<?> include
                && include.contains("message.output_text.logprobs");
        Object top = body.raw("top_logprobs");
        if (top != null) {
            chat.put("logprobs", true);
            chat.put("top_logprobs", top);
        } else if (included) {
            chat.put("logprobs", true);
        }
    }

    /** Whether {@code body} asked for the output text's logprobs. */
    static boolean wantsLogprobs(JsonView body) {
        return body.raw("top_logprobs") != null || body.raw("include") instanceof List<?> include
                && include.contains("message.output_text.logprobs");
    }
}
