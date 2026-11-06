package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Maps Anthropic Messages bodies onto the engine's chat request (llama.cpp's {@code convert_anthropic_to_oai}):
 *
 * <ul>
 *   <li>{@code system}: a string or text blocks, as the first system message.</li>
 *   <li>User content: text blocks, and {@code tool_result} blocks, which become {@code tool} messages (their
 *       content a string or text blocks) ahead of the user's text.</li>
 *   <li>Assistant content: text, {@code thinking} (the message's {@code reasoning_content}) and
 *       {@code tool_use} blocks (its {@code tool_calls}).</li>
 *   <li>{@code tools} ({@code name}, {@code description}, {@code input_schema}) and {@code tool_choice}
 *       ({@code auto}, {@code any}, {@code tool} with its name, {@code none}; {@code disable_parallel_tool_use}).
 *       Anthropic's server tools (a {@code type} such as {@code web_search_20250305}) are refused by name.</li>
 *   <li>{@code thinking} ({@code {"type": "enabled"}} or {@code "disabled"}) sets the template's thinking
 *       switch.</li>
 * </ul>
 *
 * Image and document blocks are refused by name (the projector path is not wired here).
 */
final class AnthropicRequests {

    private AnthropicRequests() {
    }

    /** The chat request of the body. */
    static ChatRequest chat(JsonView body) {
        if (!(body.raw("messages") instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("messages is required and must be a non-empty array");
        }
        ChatRequest.Builder chat = ChatRequest.builder();
        String system = systemText(body.raw("system"));
        if (!system.isEmpty()) {
            chat.addMessage(Map.of("role", "system", "content", system));
        }
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> message)) {
                throw ServerJson.badRequest("every message must be an object with role and content");
            }
            String role = String.valueOf(message.get("role"));
            switch (role) {
                case "user" -> addUser(chat, message.get("content"));
                case "assistant" -> chat.addMessage(assistant(message.get("content")));
                default -> throw ServerJson.badRequest("message role must be user or assistant, got '" + role + "'");
            }
        }
        thinking(chat, body);
        List<ChatTool> tools = tools(body);
        chat.tools(tools);
        ChatRequest.ToolChoice choice = toolChoice(body);
        if (choice != null) {
            chat.toolChoice(choice);
        }
        try {
            return chat.build();
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** Whether several tool calls are allowed ({@code tool_choice.disable_parallel_tool_use} unset). */
    static boolean parallelToolCalls(JsonView body) {
        return !(body.raw("tool_choice") instanceof Map<?, ?> m && Boolean.TRUE.equals(m.get("disable_parallel_tool_use")));
    }

    /** The sampling settings; {@code max_tokens} is required, as Anthropic documents it. */
    static GenerationRequest request(JsonView body) {
        int maxTokens = body.optI32("max_tokens", -1);
        if (maxTokens < 1) {
            throw ServerJson.badRequest("max_tokens is required and must be a positive integer (Anthropic's contract)");
        }
        SamplerConfig.Builder sampler = GenerationRequest.defaults().sampler().toBuilder();
        if (body.has("temperature")) {
            sampler.temperature((float) body.f64("temperature"));
        }
        if (body.has("top_p")) {
            sampler.topP((float) body.f64("top_p"));
        }
        if (body.has("top_k")) {
            sampler.topK(body.optI32("top_k", 0));
        }
        List<String> stops = new ArrayList<>();
        if (body.raw("stop_sequences") instanceof List<?> list) {
            for (Object o : list) {
                if (!(o instanceof String s)) {
                    throw ServerJson.badRequest("stop_sequences must be an array of strings");
                }
                stops.add(s);
            }
        }
        return GenerationRequest.builder().maxNewTokens(maxTokens).sampler(sampler.build()).stopStrings(stops)
                .build();
    }

    private static String systemText(Object system) {
        if (system == null) {
            return "";
        }
        if (system instanceof String s) {
            return s;
        }
        if (system instanceof List<?> blocks) {
            StringBuilder text = new StringBuilder();
            for (Object b : blocks) {
                Map<?, ?> block = block(b, "system");
                if (!"text".equals(block.get("type"))) {
                    throw ServerJson.badRequest("system blocks must be text blocks");
                }
                text.append(stringOf(block.get("text")));
            }
            return text.toString();
        }
        throw ServerJson.badRequest("system must be a string or an array of text blocks");
    }

    private static void addUser(ChatRequest.Builder chat, Object content) {
        if (content instanceof String s) {
            chat.addMessage(Map.of("role", "user", "content", s));
            return;
        }
        if (!(content instanceof List<?> blocks)) {
            throw ServerJson.badRequest("message content must be a string or an array of content blocks");
        }
        StringBuilder text = new StringBuilder();
        boolean hasText = false;
        for (Object b : blocks) {
            Map<?, ?> block = block(b, "a user message");
            String type = String.valueOf(block.get("type"));
            switch (type) {
                case "text" -> {
                    text.append(stringOf(block.get("text")));
                    hasText = true;
                }
                case "tool_result" -> chat.addMessage(toolResult(block));
                default -> throw refusedBlock(type);
            }
        }
        if (hasText) {
            chat.addMessage(Map.of("role", "user", "content", text.toString()));
        }
    }

    private static Map<String, Object> toolResult(Map<?, ?> block) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "tool");
        message.put("tool_call_id", stringOf(block.get("tool_use_id")));
        Object content = block.get("content");
        String text;
        if (content == null) {
            text = "";
        } else if (content instanceof String s) {
            text = s;
        } else if (content instanceof List<?> parts) {
            StringBuilder sb = new StringBuilder();
            for (Object p : parts) {
                Map<?, ?> part = block(p, "a tool_result");
                if (!"text".equals(part.get("type"))) {
                    throw refusedBlock(String.valueOf(part.get("type")));
                }
                sb.append(stringOf(part.get("text")));
            }
            text = sb.toString();
        } else {
            throw ServerJson.badRequest("tool_result content must be a string or an array of text blocks");
        }
        message.put("content", text);
        return message;
    }

    private static Map<String, Object> assistant(Object content) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        if (content instanceof String s) {
            message.put("content", s);
            return message;
        }
        if (!(content instanceof List<?> blocks)) {
            throw ServerJson.badRequest("message content must be a string or an array of content blocks");
        }
        StringBuilder text = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        List<Object> calls = new ArrayList<>();
        for (Object b : blocks) {
            Map<?, ?> block = block(b, "an assistant message");
            String type = String.valueOf(block.get("type"));
            switch (type) {
                case "text" -> text.append(stringOf(block.get("text")));
                case "thinking" -> reasoning.append(stringOf(block.get("thinking")));
                case "redacted_thinking" -> { }
                case "tool_use" -> calls.add(toolUse(block));
                default -> throw refusedBlock(type);
            }
        }
        message.put("content", text.toString());
        if (reasoning.length() > 0) {
            message.put("reasoning_content", reasoning.toString());
        }
        if (!calls.isEmpty()) {
            message.put("tool_calls", calls);
        }
        return message;
    }

    private static Map<String, Object> toolUse(Map<?, ?> block) {
        if (!(block.get("name") instanceof String name) || name.isEmpty()) {
            throw ServerJson.badRequest("a tool_use block needs a name");
        }
        Object input = block.get("input");
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", name);
        fn.put("arguments", input instanceof Map<?, ?> m ? OpenAiRequests.stringKeys(m) : new LinkedHashMap<>());
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("id", stringOf(block.get("id")));
        call.put("type", "function");
        call.put("function", fn);
        return call;
    }

    private static void thinking(ChatRequest.Builder chat, JsonView body) {
        if (body.raw("thinking") instanceof Map<?, ?> t) {
            Object type = t.get("type");
            if ("enabled".equals(type) || "adaptive".equals(type)) {
                chat.enableThinking(true);
            } else if ("disabled".equals(type)) {
                chat.enableThinking(false);
            }
        }
    }

    private static List<ChatTool> tools(JsonView body) {
        List<ChatTool> out = new ArrayList<>();
        Object raw = body.raw("tools");
        if (raw == null) {
            return out;
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("tools must be an array");
        }
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof Map<?, ?> tool)) {
                throw ServerJson.badRequest("tools[" + i + "] must be an object");
            }
            Object type = tool.get("type");
            if (type != null && !"custom".equals(type)) {
                throw ServerJson.badRequest("tools[" + i + "] is the Anthropic server tool '" + type
                        + "', which this server does not run; declare client tools with name and input_schema");
            }
            out.add(OpenAiRequests.tool(OpenAiRequests.stringKeys(tool), "tools[" + i + "]", "input_schema"));
        }
        return out;
    }

    private static ChatRequest.ToolChoice toolChoice(JsonView body) {
        Object raw = body.raw("tool_choice");
        if (raw == null) {
            return null;
        }
        if (!(raw instanceof Map<?, ?> m)) {
            throw ServerJson.badRequest("tool_choice must be an object with a type");
        }
        String type = String.valueOf(m.get("type"));
        return switch (type) {
            case "auto" -> ChatRequest.ToolChoice.auto();
            case "any" -> ChatRequest.ToolChoice.required();
            case "none" -> ChatRequest.ToolChoice.none();
            case "tool" -> {
                if (!(m.get("name") instanceof String name) || name.isEmpty()) {
                    throw ServerJson.badRequest("tool_choice type tool needs a name");
                }
                yield ChatRequest.ToolChoice.function(name);
            }
            default -> throw ServerJson.badRequest("tool_choice type '" + type + "' is not one of auto, any, tool,"
                    + " none");
        };
    }

    private static Map<?, ?> block(Object b, String where) {
        if (!(b instanceof Map<?, ?> block)) {
            throw ServerJson.badRequest("every content block of " + where + " must be an object");
        }
        return block;
    }

    private static String stringOf(Object o) {
        return o instanceof String s ? s : "";
    }

    private static RuntimeException refusedBlock(String type) {
        return ServerJson.badRequest("content block type '" + type + "' is refused: only text, thinking,"
                + " tool_use and tool_result blocks are served (image and document blocks need the projector"
                + " path)");
    }
}
