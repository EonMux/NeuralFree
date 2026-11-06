package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One validated Cohere Chat v2 request ({@code POST /cohere/v2/chat}) and its translation into the OpenAI chat body
 * {@link ChatPlan} consumes, as vLLM's {@code CohereServingChatV2._convert_v2_to_chat_completion} does:
 *
 * <ul>
 *   <li>{@code system} content becomes a system message; the OpenAI {@code developer} role is aliased to
 *       {@code system}. User content carries text and {@code image_url} parts, assistant content text and
 *       {@code thinking} blocks (plus {@code tool_plan}) and its {@code tool_calls}, and tool content text and
 *       document blocks (a document becomes a JSON text part, as vLLM flattens it).</li>
 *   <li>{@code tools} and {@code tool_choice} ({@code REQUIRED}, {@code NONE}) map onto OpenAI's tool form;
 *       {@code strict_tools} sets every function's {@code strict} flag.</li>
 *   <li>{@code documents} become vLLM's document objects ({@code {"id", "data"}}) bound as the template's
 *       {@code documents} variable; {@code response_format} maps onto OpenAI's text, JSON object and JSON
 *       schema forms.</li>
 *   <li>The sampling fields map by meaning: {@code k} is {@code top_k}, {@code p} is {@code top_p}, and the other
 *       names match. Cohere's {@code thinking} switch becomes the template's {@code enable_thinking} and its
 *       {@code token_budget} the engine's {@code thinking_token_budget}.</li>
 * </ul>
 *
 * <p>Fields this server cannot honour refuse by name rather than being ignored: {@code safety_mode} and
 * {@code citation_options} (no Cohere safety or citation renderer). A message's {@code citations} are refused too,
 * for the same reason. Cohere's {@code watermarking} switch passes through as the request's per-request
 * watermarking opt-in, and {@code kv_transfer_params} passes through unchanged for disaggregated prefill/decode.</p>
 */
final class CohereChatRequest {

    /** The top-level fields of Cohere's v2 chat request. */
    private static final Set<String> FIELDS = Set.of("model", "messages", "stream", "tools", "strict_tools",
            "tool_choice", "documents", "citation_options", "response_format", "safety_mode", "max_tokens",
            "stop_sequences", "temperature", "seed", "frequency_penalty", "presence_penalty", "k", "p", "logprobs",
            "thinking", "priority", "watermarking", "kv_transfer_params", "chat_template_kwargs");

    /** The response format name vLLM gives a Cohere JSON schema. */
    private static final String JSON_SCHEMA_NAME = "cohere_v2_json_schema";

    private final Map<String, Object> openAiBody;
    private final boolean streamed;
    private final boolean logprobs;

    private CohereChatRequest(Map<String, Object> openAiBody, boolean streamed, boolean logprobs) {
        this.openAiBody = openAiBody;
        this.streamed = streamed;
        this.logprobs = logprobs;
    }

    /**
     * The request of {@code body}.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming the field of a malformed or unsupported request
     */
    static CohereChatRequest of(JsonView body) {
        for (String key : body.keys()) {
            if (!FIELDS.contains(key)) {
                throw ServerJson.badRequest("unknown field '" + key + "' in a Cohere chat request");
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model", model(body.raw("model")));
        out.put("messages", messages(body.raw("messages")));
        boolean streamed = bool(body.raw("stream"), "stream", false);
        out.put("stream", streamed);
        List<Object> tools = tools(body.raw("tools"), bool(body.raw("strict_tools"), "strict_tools", false));
        if (!tools.isEmpty()) {
            out.put("tools", tools);
        }
        String toolChoice = toolChoice(body.raw("tool_choice"), tools);
        if (toolChoice != null) {
            out.put("tool_choice", toolChoice);
        }
        List<Object> documents = documents(body.raw("documents"));
        if (!documents.isEmpty()) {
            out.put("documents", documents);
        }
        responseFormat(body.raw("response_format"), out);
        thinking(body.raw("thinking"), out);
        refuseUnsupported(body);
        sampling(body, out);
        if (body.raw("kv_transfer_params") != null) {
            out.put("kv_transfer_params", body.raw("kv_transfer_params"));
        }
        boolean logprobs = bool(body.raw("logprobs"), "logprobs", false);
        if (logprobs) {
            out.put("logprobs", true);
        }
        return new CohereChatRequest(out, streamed, logprobs);
    }

    /** The OpenAI chat body of this request. */
    Map<String, Object> openAiBody() {
        return openAiBody;
    }

    /** Whether the request asks for the Cohere SSE stream. */
    boolean streamed() {
        return streamed;
    }

    /** Whether the request asks for per-token log probabilities. */
    boolean logprobs() {
        return logprobs;
    }

    private static String model(Object raw) {
        if (!(raw instanceof String model) || model.isBlank()) {
            throw ServerJson.badRequest("model is required and must be a non-empty string");
        }
        return model;
    }

    /** The messages of the request, each translated to its OpenAI form. */
    private static List<Object> messages(Object raw) {
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("messages is required and must be a non-empty array");
        }
        List<Object> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            String where = "messages[" + i + "]";
            if (!(list.get(i) instanceof Map<?, ?> message)) {
                throw ServerJson.badRequest(where + " must be an object with a role");
            }
            Map<String, Object> fields = OpenAiRequests.stringKeys(message);
            Object rawRole = fields.get("role");
            if (!(rawRole instanceof String role)) {
                throw ServerJson.badRequest(where + ".role must be a string");
            }
            switch (role) {
                case "system", "developer" -> out.add(systemMessage(fields, where));
                case "user" -> out.add(userMessage(fields, where));
                case "assistant" -> out.add(assistantMessage(fields, where));
                case "tool" -> out.add(toolMessage(fields, where));
                default -> throw ServerJson.badRequest(where + ".role must be user, assistant, system, tool or"
                        + " developer, got '" + role + "'");
            }
        }
        return out;
    }

    /** A system message: its content is a string or text blocks, joined as vLLM joins them. */
    private static Map<String, Object> systemMessage(Map<String, Object> fields, String where) {
        only(fields, where, "role", "content");
        return message("system", text(fields.get("content"), where + ".content"), null, null);
    }

    /** A user message: a string, or text and {@code image_url} content parts. */
    private static Map<String, Object> userMessage(Map<String, Object> fields, String where) {
        only(fields, where, "role", "content");
        Object content = fields.get("content");
        if (content instanceof String text) {
            return message("user", text, null, null);
        }
        List<?> parts = requireList(content, where + ".content");
        List<Object> out = new ArrayList<>(parts.size());
        for (int j = 0; j < parts.size(); j++) {
            String at = where + ".content[" + j + "]";
            Map<String, Object> part = requireMap(parts.get(j), at);
            Object type = part.get("type");
            if ("text".equals(type)) {
                Map<String, Object> text = new LinkedHashMap<>();
                text.put("type", "text");
                text.put("text", requireString(part.get("text"), at + ".text"));
                out.add(text);
            } else if ("image_url".equals(type)) {
                Map<String, Object> image = requireMap(part.get("image_url"), at + ".image_url");
                Map<String, Object> url = new LinkedHashMap<>();
                url.put("url", requireString(image.get("url"), at + ".image_url.url"));
                if (image.get("detail") != null) {
                    url.put("detail", requireString(image.get("detail"), at + ".image_url.detail"));
                }
                Map<String, Object> imageUrl = new LinkedHashMap<>();
                imageUrl.put("type", "image_url");
                imageUrl.put("image_url", url);
                out.add(imageUrl);
            } else {
                throw ServerJson.badRequest(at + ".type must be text or image_url, got " + type);
            }
        }
        if (out.size() == 1 && "text".equals(((Map<?, ?>) out.get(0)).get("type"))) {
            return message("user", ((Map<?, ?>) out.get(0)).get("text"), null, null);
        }
        Map<String, Object> outMessage = new LinkedHashMap<>();
        outMessage.put("role", "user");
        outMessage.put("content", out);
        return outMessage;
    }

    /**
     * An assistant message: text blocks become {@code content}, {@code thinking} blocks and {@code tool_plan} the
     * OpenAI {@code reasoning}, and {@code tool_calls} keep their ids, names and argument text.
     */
    private static Map<String, Object> assistantMessage(Map<String, Object> fields, String where) {
        only(fields, where, "role", "content", "tool_calls", "tool_plan", "citations");
        refuseCitations(fields, where);
        StringBuilder text = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        Object content = fields.get("content");
        if (content instanceof String string) {
            text.append(string);
        } else if (content != null) {
            List<?> parts = requireList(content, where + ".content");
            for (int j = 0; j < parts.size(); j++) {
                String at = where + ".content[" + j + "]";
                Map<String, Object> part = requireMap(parts.get(j), at);
                Object type = part.get("type");
                if ("text".equals(type)) {
                    text.append(requireString(part.get("text"), at + ".text"));
                } else if ("thinking".equals(type)) {
                    reasoning.append(requireString(part.get("thinking"), at + ".thinking"));
                } else {
                    throw ServerJson.badRequest(at + ".type must be text or thinking, got " + type);
                }
            }
        }
        if (fields.get("tool_plan") != null) {
            reasoning.append(requireString(fields.get("tool_plan"), where + ".tool_plan"));
        }
        List<Object> calls = assistantToolCalls(fields.get("tool_calls"), where + ".tool_calls");
        return message("assistant", text.length() == 0 ? null : text.toString(),
                reasoning.length() == 0 ? null : reasoning.toString(), calls);
    }

    /** A tool message: its content's documents flatten to JSON text parts, as vLLM flattens them. */
    private static Map<String, Object> toolMessage(Map<String, Object> fields, String where) {
        only(fields, where, "role", "tool_call_id", "content");
        String id = requireString(fields.get("tool_call_id"), where + ".tool_call_id");
        Object content = fields.get("content");
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("role", "tool");
        out.put("tool_call_id", id);
        if (content instanceof String text) {
            out.put("content", text);
            return out;
        }
        List<?> parts = requireList(content, where + ".content");
        List<Object> outParts = new ArrayList<>(parts.size());
        for (int j = 0; j < parts.size(); j++) {
            String at = where + ".content[" + j + "]";
            Map<String, Object> part = requireMap(parts.get(j), at);
            Object type = part.get("type");
            Map<String, Object> text = new LinkedHashMap<>();
            text.put("type", "text");
            if ("text".equals(type)) {
                text.put("text", requireString(part.get("text"), at + ".text"));
            } else if ("document".equals(type)) {
                text.put("text", ServerJson.write(requireMap(part.get("document"), at + ".document")));
            } else {
                throw ServerJson.badRequest(at + ".type must be text or document, got " + type);
            }
            outParts.add(text);
        }
        out.put("content", outParts);
        return out;
    }

    /** The assistant {@code tool_calls} in OpenAI's form; absent sends none. */
    private static List<Object> assistantToolCalls(Object raw, String where) {
        if (raw == null) {
            return null;
        }
        List<?> list = requireList(raw, where);
        List<Object> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            String at = where + "[" + i + "]";
            Map<String, Object> call = requireMap(list.get(i), at);
            Map<String, Object> function = requireMap(call.get("function"), at + ".function");
            Map<String, Object> fn = new LinkedHashMap<>();
            fn.put("name", requireString(function.get("name"), at + ".function.name"));
            Object arguments = function.get("arguments");
            fn.put("arguments", arguments == null ? "{}" : requireString(arguments, at + ".function.arguments"));
            Map<String, Object> outCall = new LinkedHashMap<>();
            outCall.put("id", requireString(call.get("id"), at + ".id"));
            outCall.put("type", "function");
            outCall.put("function", fn);
            out.add(outCall);
        }
        return out;
    }

    /** An OpenAI message map with only the parts it carries. */
    private static Map<String, Object> message(String role, Object content, String reasoning, List<Object> calls) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("role", role);
        if (content != null) {
            out.put("content", content);
        }
        if (reasoning != null) {
            out.put("reasoning", reasoning);
        }
        if (calls != null) {
            out.put("tool_calls", calls);
        }
        return out;
    }

    /** The {@code tools} in OpenAI's form; {@code strictTools} marks every function {@code strict}. */
    private static List<Object> tools(Object raw, boolean strictTools) {
        if (raw == null) {
            return List.of();
        }
        List<?> list = requireList(raw, "tools");
        List<Object> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            String at = "tools[" + i + "]";
            Map<String, Object> tool = requireMap(list.get(i), at);
            if (!"function".equals(tool.get("type"))) {
                throw ServerJson.badRequest(at + ".type must be function, got " + tool.get("type"));
            }
            Map<String, Object> source = requireMap(tool.get("function"), at + ".function");
            Map<String, Object> fn = new LinkedHashMap<>();
            fn.put("name", requireString(source.get("name"), at + ".function.name"));
            if (source.get("description") != null) {
                fn.put("description", requireString(source.get("description"), at + ".function.description"));
            }
            Object parameters = source.get("parameters");
            fn.put("parameters", parameters == null ? Map.of() : requireMap(parameters, at + ".function.parameters"));
            if (strictTools) {
                fn.put("strict", true);
            }
            Map<String, Object> outTool = new LinkedHashMap<>();
            outTool.put("type", "function");
            outTool.put("function", fn);
            out.add(outTool);
        }
        return out;
    }

    /**
     * The OpenAI {@code tool_choice}: {@code REQUIRED} needs a tool and becomes {@code required}, {@code NONE}
     * becomes {@code none}, and tools without a choice serve the free {@code auto} default.
     */
    private static String toolChoice(Object raw, List<Object> tools) {
        String choice;
        if (raw == null) {
            choice = tools.isEmpty() ? null : "auto";
        } else if ("REQUIRED".equals(raw)) {
            if (tools.isEmpty()) {
                throw ServerJson.badRequest("tool_choice=\"REQUIRED\" requires at least one tool in tools.");
            }
            choice = "required";
        } else if ("NONE".equals(raw)) {
            choice = "none";
        } else {
            throw ServerJson.badRequest("tool_choice must be REQUIRED or NONE, got " + ServerJson.write(raw));
        }
        return choice;
    }

    /** Cohere's documents as vLLM's template documents: a string is text under a positional id. */
    private static List<Object> documents(Object raw) {
        if (raw == null) {
            return List.of();
        }
        List<?> list = requireList(raw, "documents");
        List<Object> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            String at = "documents[" + i + "]";
            Map<String, Object> document = new LinkedHashMap<>();
            if (list.get(i) instanceof String text) {
                document.put("id", "doc_" + i);
                document.put("data", Map.of("text", text));
            } else {
                Map<String, Object> source = requireMap(list.get(i), at);
                Object id = source.get("id");
                document.put("id", id == null ? "doc_" + i : requireString(id, at + ".id"));
                Object data = source.get("data");
                if (data instanceof String text) {
                    document.put("data", Map.of("text", text));
                } else {
                    document.put("data", requireMap(data, at + ".data"));
                }
            }
            out.add(document);
        }
        return out;
    }

    /** Cohere's {@code response_format} in OpenAI's shape; {@code text} is OpenAI's own default and adds none. */
    private static void responseFormat(Object raw, Map<String, Object> out) {
        if (raw == null) {
            return;
        }
        Map<String, Object> format = requireMap(raw, "response_format");
        Object type = format.get("type");
        if ("text".equals(type)) {
            return;
        }
        Map<String, Object> openAi = new LinkedHashMap<>();
        if ("json_object".equals(type)) {
            openAi.put("type", "json_object");
        } else if ("json_schema".equals(type)) {
            Map<String, Object> spec = new LinkedHashMap<>();
            spec.put("name", JSON_SCHEMA_NAME);
            spec.put("schema", requireMap(format.get("json_schema"), "response_format.json_schema"));
            openAi.put("type", "json_schema");
            openAi.put("json_schema", spec);
        } else {
            throw ServerJson.badRequest("response_format.type must be text, json_object or json_schema, got "
                    + type);
        }
        out.put("response_format", openAi);
    }

    /** Cohere's thinking switch and token budget as the template's {@code enable_thinking} and the engine budget. */
    private static void thinking(Object raw, Map<String, Object> out) {
        if (raw == null) {
            return;
        }
        Map<String, Object> thinking = requireMap(raw, "thinking");
        Object type = thinking.get("type");
        if (!"enabled".equals(type) && !"disabled".equals(type)) {
            throw ServerJson.badRequest("thinking.type must be enabled or disabled, got " + type);
        }
        for (String key : thinking.keySet()) {
            if (!key.equals("type") && !key.equals("token_budget")) {
                throw ServerJson.badRequest("unknown field 'thinking." + key + "' in a Cohere chat request");
            }
        }
        boolean enabled = "enabled".equals(type);
        out.put("enable_thinking", enabled);
        if (enabled && thinking.get("token_budget") != null) {
            long budget = integer(thinking.get("token_budget"), "thinking.token_budget");
            if (budget < -1) {
                throw ServerJson.badRequest("thinking.token_budget must be a non-negative integer or -1, got "
                        + budget);
            }
            out.put("thinking_token_budget", budget);
        }
    }

    /** The sampling fields, by meaning ({@code k} is {@code top_k}, {@code p} is {@code top_p}). */
    private static void sampling(JsonView body, Map<String, Object> out) {
        copy(body.raw("max_tokens"), "max_tokens", out);
        copy(body.raw("stop_sequences"), "stop", out);
        copy(body.raw("temperature"), "temperature", out);
        copy(body.raw("seed"), "seed", out);
        copy(body.raw("frequency_penalty"), "frequency_penalty", out);
        copy(body.raw("presence_penalty"), "presence_penalty", out);
        copy(body.raw("k"), "top_k", out);
        copy(body.raw("p"), "top_p", out);
        copy(body.raw("priority"), "priority", out);
        copy(body.raw("watermarking"), "watermarking", out);
        if (body.raw("chat_template_kwargs") != null) {
            out.put("chat_template_kwargs", requireMap(body.raw("chat_template_kwargs"), "chat_template_kwargs"));
        }
        // validate the shapes ChatPlan's sampling readers expect, naming the Cohere field
        if (integer(body.raw("max_tokens"), "max_tokens") < 0) {
            throw ServerJson.badRequest("max_tokens must be non-negative, got " + body.raw("max_tokens"));
        }
        integer(body.raw("seed"), "seed");
        integer(body.raw("k"), "k");
        integer(body.raw("priority"), "priority");
        real(body.raw("temperature"), "temperature");
        real(body.raw("p"), "p");
        real(body.raw("frequency_penalty"), "frequency_penalty");
        real(body.raw("presence_penalty"), "presence_penalty");
        strings(body.raw("stop_sequences"), "stop_sequences");
        if (body.raw("watermarking") != null && !(body.raw("watermarking") instanceof Boolean)) {
            throw ServerJson.badRequest("watermarking must be a boolean");
        }
    }

    /** Copies a Cohere field under its OpenAI name when present. */
    private static void copy(Object raw, String to, Map<String, Object> out) {
        if (raw != null) {
            out.put(to, raw);
        }
    }

    /** Refuses the fields this server cannot honour, naming each. */
    private static void refuseUnsupported(JsonView body) {
        if (body.raw("safety_mode") != null) {
            throw ServerJson.badRequest("safety_mode is not supported: this server has no Cohere safety renderer");
        }
        if (body.raw("citation_options") != null) {
            throw ServerJson.badRequest("citation_options is not supported: this server renders no citations");
        }
        OpenAiRequests.requireKvTransferObject(body);
    }

    private static void refuseCitations(Map<String, Object> fields, String where) {
        Object citations = fields.get("citations");
        if (citations != null && !(citations instanceof List<?> list && list.isEmpty())) {
            throw ServerJson.badRequest(where + ".citations is not supported: this server renders no citations");
        }
    }

    /** Refuses any message field but the listed ones, naming it. */
    private static void only(Map<String, Object> fields, String where, String... allowed) {
        Set<String> known = Set.of(allowed);
        for (String key : fields.keySet()) {
            if (!known.contains(key)) {
                throw ServerJson.badRequest(where + " has unsupported field '" + key + "'");
            }
        }
    }

    /** A message content that is a string or a joined list of text parts ({@code system} and text fallbacks). */
    private static String text(Object raw, String where) {
        if (raw instanceof String string) {
            return string;
        }
        List<?> parts = requireList(raw, where);
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < parts.size(); i++) {
            String at = where + "[" + i + "]";
            Map<String, Object> part = requireMap(parts.get(i), at);
            if (!"text".equals(part.get("type"))) {
                throw ServerJson.badRequest(at + ".type must be text, got " + part.get("type"));
            }
            out.append(requireString(part.get("text"), at + ".text"));
        }
        return out.toString();
    }

    private static boolean bool(Object raw, String field, boolean dflt) {
        if (raw == null) {
            return dflt;
        }
        if (!(raw instanceof Boolean flag)) {
            throw ServerJson.badRequest(field + " must be a boolean");
        }
        return flag;
    }

    private static long integer(Object raw, String field) {
        if (raw == null) {
            return 0;
        }
        if (!(raw instanceof Number number)) {
            throw ServerJson.badRequest(field + " must be an integer");
        }
        double value = number.doubleValue();
        if (value != Math.rint(value) || Double.isInfinite(value)) {
            throw ServerJson.badRequest(field + " must be an integer, got " + raw);
        }
        if (value < Long.MIN_VALUE || value > Long.MAX_VALUE) {
            throw ServerJson.badRequest(field + " is out of range: " + raw);
        }
        return (long) value;
    }

    private static double real(Object raw, String field) {
        if (raw == null) {
            return 0;
        }
        if (!(raw instanceof Number number) || !Double.isFinite(number.doubleValue())) {
            throw ServerJson.badRequest(field + " must be a number");
        }
        return number.doubleValue();
    }

    private static List<String> strings(Object raw, String field) {
        if (raw == null) {
            return List.of();
        }
        List<?> list = requireList(raw, field);
        List<String> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            out.add(requireString(list.get(i), field + "[" + i + "]"));
        }
        return out;
    }

    private static Map<String, Object> requireMap(Object raw, String where) {
        if (!(raw instanceof Map<?, ?> map)) {
            throw ServerJson.badRequest(where + " must be a JSON object");
        }
        return OpenAiRequests.stringKeys(map);
    }

    private static List<?> requireList(Object raw, String where) {
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest(where + " must be an array");
        }
        return list;
    }

    private static String requireString(Object raw, String where) {
        if (!(raw instanceof String text)) {
            throw ServerJson.badRequest(where + " must be a string");
        }
        return text;
    }
}
