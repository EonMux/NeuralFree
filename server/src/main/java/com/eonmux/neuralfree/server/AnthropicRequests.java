package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorArgs;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
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
 *   <li>User content: text and {@code image} blocks, and {@code tool_result} blocks, which become {@code tool}
 *       messages (their content a string, or text and image blocks) ahead of the user's text. An image block's
 *       {@code base64} source becomes a {@code data:} URL of its {@code media_type}, a {@code url} source its URL:
 *       an OpenAI {@code image_url} part either way, as llama.cpp's {@code convert_anthropic_to_oai} maps
 *       them.</li>
 *   <li>Assistant content: text, {@code thinking} (the message's {@code reasoning_content}) and
 *       {@code tool_use} blocks (its {@code tool_calls}).</li>
 *   <li>{@code tools} ({@code name}, {@code description}, {@code input_schema}) and {@code tool_choice}
 *       ({@code auto}, {@code any}, {@code tool} with its name, {@code none}; {@code disable_parallel_tool_use}).
 *       Anthropic's server tools (a {@code type} such as {@code web_search_20250305}) are refused by name.</li>
 *   <li>{@code thinking} ({@code {"type": "enabled"}} or {@code "disabled"}) sets the template's thinking
 *       switch; {@code budget_tokens} of an enabled thinking config becomes the engine's thinking-token budget
 *       (vLLM's {@code thinking_token_budget}).</li>
 *   <li>{@code cache_salt} (vLLM's extension) partitions the prefix cache as on the OpenAI endpoints, and
 *       {@code kv_transfer_params} (vLLM's extension) runs a disaggregated prefill/decode transfer as on the
 *       OpenAI endpoints; the message body answers the transfer's {@code kv_transfer_params}.</li>
 *   <li>{@code cache_control} on a system, content or tool block ({@code {"type": "ephemeral"}} with an optional
 *       {@code ttl} of {@code 5m} or {@code 1h}) is accepted and needs no action: the prefix cache keeps every
 *       computed prefix on its own, so the marked prefix is reused like any other. Another shape is refused by
 *       name.</li>
 * </ul>
 *
 * Document blocks, and image sources other than {@code base64} and {@code url}, are refused by name.
 */
final class AnthropicRequests {

    private AnthropicRequests() {
    }

    /** The chat request of the body over the server's default template variables
     *  ({@link ServerOptions#chatTemplateKwargs}, {@link ChatRequest#withDefaultTemplateKwargs}). */
    static ChatRequest chat(JsonView body, ServerOptions options) {
        return chat(body).withDefaultTemplateKwargs(options.chatTemplateKwargs());
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

    /**
     * The request's {@code parallel_tool_calls} from {@code tool_choice.disable_parallel_tool_use} (vLLM's
     * Anthropic mapping serves {@code not disable_parallel_tool_use}): false with the switch on, true with it
     * off, and null without it — the chat template's default then serves
     * ({@link EngineService#defaultParallelToolCalls}), and no guarantee is asked for.
     */
    static Boolean parallelToolCalls(JsonView body) {
        Object raw = body.raw("tool_choice") instanceof Map<?, ?> choice
                ? choice.get("disable_parallel_tool_use") : null;
        if (raw == null) {
            return null;
        }
        if (!(raw instanceof Boolean on)) {
            throw ServerJson.badRequest("tool_choice.disable_parallel_tool_use must be a boolean");
        }
        return !on;
    }

    /** The sampling settings and cache salt; {@code max_tokens} is required, as Anthropic documents it. */
    static GenerationRequest request(JsonView body) {
        return request(body, LogitsProcessors.empty());
    }

    /**
     * {@link #request(JsonView)} with the engine's custom logits processors validating the body's
     * {@code vllm_xargs} arguments (vLLM's Anthropic extension).
     */
    static GenerationRequest request(JsonView body, LogitsProcessors processors) {
        return request(body, processors, null);
    }

    /**
     * {@link #request(JsonView, LogitsProcessors)} with the engine's text watermark ({@code engineWatermark}, null
     * when the server serves none): every generated token of the message carries it, as on the OpenAI routes. The
     * body's {@code kv_transfer_params} (vLLM's Anthropic extension) parses here too, so a disaggregated
     * prefill/decode transfer runs on this route as on the OpenAI ones.
     */
    static GenerationRequest request(JsonView body, LogitsProcessors processors, WatermarkConfig engineWatermark) {
        int maxTokens = body.optI32("max_tokens", -1);
        if (maxTokens < 1) {
            throw ServerJson.badRequest("max_tokens is required and must be a positive integer (Anthropic's contract)");
        }
        SamplerConfig.Builder sampler = GenerationRequest.defaults().sampler().toBuilder();
        if (engineWatermark != null) {
            sampler.watermark(engineWatermark);
        }
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
        GenerationRequest.Builder builder = GenerationRequest.builder().maxNewTokens(maxTokens)
                .sampler(sampler.build()).stopStrings(stops).cacheSalt(OpenAiRequests.cacheSalt(body))
                .kvTransfer(OpenAiRequests.kvTransfer(body));
        int budget = ReasoningBudgets.anthropicBudget(body);
        if (budget != GenerationRequest.REASONING_BUDGET_DISABLED) {
            builder.reasoningBudget(budget);
        }
        LogitsProcessorArgs args = OpenAiRequests.logitsProcessorArgs(body);
        try {
            processors.validate(args);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        if (!args.isEmpty()) {
            builder.logitsProcessorArgs(args);
        }
        return builder.build();
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
        List<Object> parts = new ArrayList<>();
        for (Object b : blocks) {
            Map<?, ?> block = block(b, "a user message");
            String type = String.valueOf(block.get("type"));
            switch (type) {
                case "text" -> parts.add(textPart(stringOf(block.get("text"))));
                case "image" -> parts.add(imagePart(block));
                case "tool_result" -> chat.addMessage(toolResult(block));
                default -> throw refusedBlock(type);
            }
        }
        if (!parts.isEmpty()) {
            chat.addMessage(Map.of("role", "user", "content", content(parts)));
        }
    }

    /**
     * Message content of converted parts: the text joined into one string when every part is text (the template
     * sees exactly the text it saw before images were accepted), else the part list.
     */
    private static Object content(List<Object> parts) {
        StringBuilder text = new StringBuilder();
        for (Object p : parts) {
            Map<?, ?> part = (Map<?, ?>) p;
            if (!"text".equals(part.get("type"))) {
                return parts;
            }
            text.append(part.get("text"));
        }
        return text.toString();
    }

    private static Map<String, Object> textPart(String text) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "text");
        part.put("text", text);
        return part;
    }

    /** An image block as an OpenAI {@code image_url} part (llama.cpp server-chat.cpp, Anthropic conversion). */
    private static Map<String, Object> imagePart(Map<?, ?> block) {
        if (!(block.get("source") instanceof Map<?, ?> source)) {
            throw ServerJson.badRequest("an image block needs a 'source' object");
        }
        String kind = String.valueOf(source.get("type"));
        String url = switch (kind) {
            case "base64" -> {
                if (!(source.get("data") instanceof String data) || data.isEmpty()) {
                    throw ServerJson.badRequest("an image block's base64 source needs 'data'");
                }
                String mediaType = source.get("media_type") instanceof String m && !m.isEmpty() ? m : "image/jpeg";
                yield "data:" + mediaType + ";base64," + data;
            }
            case "url" -> {
                if (!(source.get("url") instanceof String u) || u.isEmpty()) {
                    throw ServerJson.badRequest("an image block's url source needs 'url'");
                }
                yield u;
            }
            default -> throw ServerJson.badRequest("image source type '" + kind + "' is refused: only base64 and"
                    + " url sources are served (file sources need the Files API, which this server does not run)");
        };
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "image_url");
        part.put("image_url", Map.of("url", url));
        return part;
    }

    private static Map<String, Object> toolResult(Map<?, ?> block) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "tool");
        message.put("tool_call_id", stringOf(block.get("tool_use_id")));
        Object content = block.get("content");
        Object converted;
        if (content == null) {
            converted = "";
        } else if (content instanceof String s) {
            converted = s;
        } else if (content instanceof List<?> blocks) {
            List<Object> parts = new ArrayList<>();
            for (Object p : blocks) {
                Map<?, ?> part = block(p, "a tool_result");
                String type = String.valueOf(part.get("type"));
                switch (type) {
                    case "text" -> parts.add(textPart(stringOf(part.get("text"))));
                    case "image" -> parts.add(imagePart(part));
                    default -> throw refusedBlock(type);
                }
            }
            converted = content(parts);
        } else {
            throw ServerJson.badRequest("tool_result content must be a string or an array of text and image blocks");
        }
        message.put("content", converted);
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
            cacheControl(tool, "tools[" + i + "]");
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
        cacheControl(block, "a content block of " + where);
        return block;
    }

    /**
     * Checks a block's {@code cache_control} breakpoint: absent, or {@code {"type": "ephemeral"}} with an optional
     * {@code ttl} of {@code 5m} or {@code 1h}, Anthropic's shape. The breakpoint needs no action here, as the
     * prefix cache stores every computed prefix itself.
     */
    private static void cacheControl(Map<?, ?> block, String where) {
        Object raw = block.get("cache_control");
        if (raw == null) {
            return;
        }
        if (!(raw instanceof Map<?, ?> control) || !"ephemeral".equals(control.get("type"))) {
            throw ServerJson.badRequest("cache_control of " + where + " must be {\"type\": \"ephemeral\"},"
                    + " got " + raw);
        }
        Object ttl = control.get("ttl");
        if (ttl != null && !"5m".equals(ttl) && !"1h".equals(ttl)) {
            throw ServerJson.badRequest("cache_control.ttl of " + where + " must be \"5m\" or \"1h\", got " + ttl);
        }
    }

    private static String stringOf(Object o) {
        return o instanceof String s ? s : "";
    }

    private static RuntimeException refusedBlock(String type) {
        return ServerJson.badRequest("content block type '" + type + "' is refused: only text, image,"
                + " thinking, tool_use and tool_result blocks are served (document blocks have no local"
                + " equivalent)");
    }
}
