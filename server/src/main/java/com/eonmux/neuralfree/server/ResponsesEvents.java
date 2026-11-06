package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * One Responses API response as it is produced: the chat deltas of its {@link ChatChoice} become output items and
 * the OpenAI streaming events (llama.cpp's {@code to_json_oaicompat_resp*}, vLLM's {@code responses_stream_generator}):
 * {@code response.created}, {@code response.in_progress}, then per item {@code output_item.added}, its deltas
 * ({@code reasoning_text.delta}, {@code content_part.added} and {@code output_text.delta},
 * {@code function_call_arguments.delta}), their {@code .done} events and {@code output_item.done}, and a terminal
 * {@code response.completed}, {@code response.incomplete} (the token budget ran out, or the response was cancelled)
 * or {@code response.failed}. Every event carries a {@code sequence_number}. The terminal event's response is the
 * whole response, so the non-streamed answer is that object. Not thread-safe: one response, one producer.
 */
final class ResponsesEvents {

    /** Where the events go: the SSE stream, a stored response's log, or nowhere. */
    @FunctionalInterface
    interface Sink {
        void event(String type, Map<String, Object> data) throws IOException;
    }

    private static final String HEX = "0123456789abcdef";

    private final Map<String, Object> head;
    private final Sink sink;
    private final OpenAiBodies bodies;
    private final boolean logprobs;
    private final List<Map<String, Object>> output = new ArrayList<>();
    private int sequence;
    private Item open;
    /**
     * Captures of tokens whose text was held before any item opened; the next text carries them. Captures while a
     * reasoning or function-call item is open belong to that item's tokens and are not output-text logprobs.
     */
    private final List<Object> heldLogprobs = new ArrayList<>();

    /** The item being produced. */
    private static final class Item {
        final String kind;
        final String id;
        final int index;
        final StringBuilder text = new StringBuilder();
        final List<Object> logprobs = new ArrayList<>();
        final String callId;
        final String name;
        final int callIndex;

        Item(String kind, String id, int index, String callId, String name, int callIndex) {
            this.kind = kind;
            this.id = id;
            this.index = index;
            this.callId = callId;
            this.name = name;
            this.callIndex = callIndex;
        }
    }

    /**
     * @param head     the response's fixed fields ({@link #head})
     * @param sink     where the events go
     * @param bodies   the logprobs rendering
     * @param logprobs whether the output text carries its logprobs
     */
    ResponsesEvents(Map<String, Object> head, Sink sink, OpenAiBodies bodies, boolean logprobs) {
        // the head keeps JSON nulls (no instructions, no temperature), which Map.copyOf refuses
        this.head = Collections.unmodifiableMap(new LinkedHashMap<>(head));
        this.sink = sink;
        this.bodies = bodies;
        this.logprobs = logprobs;
    }

    /** A response id, item id or similar: {@code prefix} and 32 hex digits. */
    static String newId(String prefix) {
        ThreadLocalRandom r = ThreadLocalRandom.current();
        StringBuilder b = new StringBuilder(prefix);
        for (int i = 0; i < 32; i++) {
            b.append(HEX.charAt(r.nextInt(16)));
        }
        return b.toString();
    }

    /**
     * The fields a response repeats from its request (the OpenAI {@code Response} object), keyed in its order.
     */
    static Map<String, Object> head(JsonView body, String id, String model,
                                    long created, boolean stored) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", id);
        out.put("object", "response");
        out.put("created_at", created);
        out.put("background", body.optBool("background", false));
        out.put("instructions", body.raw("instructions"));
        out.put("max_output_tokens", body.raw("max_output_tokens"));
        out.put("metadata", body.raw("metadata") == null ? Map.of() : body.raw("metadata"));
        out.put("model", model);
        out.put("parallel_tool_calls", body.optBool("parallel_tool_calls", true));
        out.put("previous_response_id", body.raw("previous_response_id"));
        Map<String, Object> reasoning = new LinkedHashMap<>();
        reasoning.put("effort", body.optObj("reasoning").raw("effort"));
        reasoning.put("summary", body.optObj("reasoning").raw("summary"));
        out.put("reasoning", reasoning);
        out.put("store", stored);
        out.put("temperature", body.raw("temperature"));
        out.put("text", body.raw("text") == null ? Map.of("format", Map.of("type", "text")) : body.raw("text"));
        out.put("tool_choice", body.raw("tool_choice") == null ? "auto" : body.raw("tool_choice"));
        out.put("tools", body.raw("tools") == null ? List.of() : body.raw("tools"));
        out.put("top_logprobs", body.raw("top_logprobs"));
        out.put("top_p", body.raw("top_p"));
        out.put("truncation", body.raw("truncation") == null ? "disabled" : body.raw("truncation"));
        out.put("user", body.raw("user"));
        return out;
    }

    /** The response as it stands at {@code status}, with no output yet ({@code queued}, {@code in_progress}). */
    /** The response {@code failed} with {@code message} and no output, for a run that could not report itself. */
    static Map<String, Object> failed(Map<String, Object> head, String message) {
        Map<String, Object> out = pending(head, "failed");
        out.put("error", error(message));
        return out;
    }

    private static Map<String, Object> error(String message) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("code", "server_error");
        error.put("message", message);
        return error;
    }

    static Map<String, Object> pending(Map<String, Object> head, String status) {
        Map<String, Object> out = new LinkedHashMap<>(head);
        out.put("status", status);
        out.put("error", null);
        out.put("incomplete_details", null);
        out.put("output", List.of());
        out.put("usage", null);
        return out;
    }

    /** Opens the response: {@code response.created} and {@code response.in_progress}. */
    void start() throws IOException {
        emit("response.created", Map.of("response", pending(head, "in_progress")));
        emit("response.in_progress", Map.of("response", pending(head, "in_progress")));
    }

    /**
     * Turns the chat deltas into item events. The tool parser reports a feed's content before its argument text,
     * so argument text for the open call goes first: text after a call must not close the call it completes.
     */
    void accept(List<ChatChoice.Delta> deltas) throws IOException {
        List<ChatChoice.Delta> rest = new ArrayList<>();
        for (ChatChoice.Delta d : deltas) {
            if (!continuesOpenCall(d)) {
                rest.add(d);
            } else {
                accept(d);
            }
        }
        for (ChatChoice.Delta d : rest) {
            accept(d);
        }
    }

    /** Whether {@code d} only adds argument text to the open function call. */
    private boolean continuesOpenCall(ChatChoice.Delta d) {
        if (open == null || !"function_call".equals(open.kind) || d.delta().size() != 1
                || !(d.delta().get("tool_calls") instanceof List<?> calls)) {
            return false;
        }
        for (Object c : calls) {
            Map<?, ?> call = (Map<?, ?>) c;
            if (call.get("id") != null || !(call.get("index") instanceof Integer i) || i != open.callIndex) {
                return false;
            }
        }
        return true;
    }

    private void accept(ChatChoice.Delta d) throws IOException {
        Map<String, Object> delta = d.delta();
        if (delta.get("reasoning_content") instanceof String reasoning) {
            ensure("reasoning", null, null, -1);
            open.text.append(reasoning);
            emit("response.reasoning_text.delta", itemDelta(reasoning));
        }
        if (delta.get("content") instanceof String content) {
            ensure("message", null, null, -1);
            open.text.append(content);
            Map<String, Object> data = itemDelta(content);
            List<Object> entries = new ArrayList<>(heldLogprobs);
            heldLogprobs.clear();
            entries.addAll(logprobEntries(d.logprobs()));
            open.logprobs.addAll(entries);
            data.put("logprobs", entries);
            emit("response.output_text.delta", data);
        } else if (delta.isEmpty()) {
            List<Object> entries = logprobEntries(d.logprobs());
            if (open == null) {
                heldLogprobs.addAll(entries);
            } else if ("message".equals(open.kind)) {
                open.logprobs.addAll(entries);
            }
        }
        if (delta.get("tool_calls") instanceof List<?> calls) {
            for (Object c : calls) {
                toolCallDelta((Map<?, ?>) c);
            }
        }
    }

    private void toolCallDelta(Map<?, ?> call) throws IOException {
        Map<?, ?> fn = (Map<?, ?>) call.get("function");
        if (call.get("id") instanceof String callId) {
            close();
            heldLogprobs.clear();
            int callIndex = call.get("index") instanceof Integer i ? i : -1;
            ensure("function_call", callId, String.valueOf(fn.get("name")), callIndex);
        }
        if (fn.get("arguments") instanceof String args && !args.isEmpty() && open != null
                && "function_call".equals(open.kind)) {
            open.text.append(args);
            Map<String, Object> data = new LinkedHashMap<>();
            data.put("item_id", open.id);
            data.put("output_index", open.index);
            data.put("delta", args);
            emit("response.function_call_arguments.delta", data);
        }
    }

    /**
     * Closes the response with its result: {@code completed}, {@code incomplete} when the budget or the context ran
     * out ({@code max_output_tokens}), or {@code cancelled}. The final output follows {@code parse}, the tool
     * parser's reading of the whole text, as the chat route's message does ({@link ResponsesOutput}).
     *
     * @param promptTokens the prompt's tokens
     * @param cachedTokens the prompt tokens reused from the cache
     * @return the final response
     */
    Map<String, Object> finish(GenerationResult result, int promptTokens, int cachedTokens, ToolParse parse)
            throws IOException {
        close();
        Map<String, Object> response = new LinkedHashMap<>(head);
        String status;
        Map<String, Object> incomplete = null;
        switch (result.stopReason()) {
            case MAX_TOKENS, CONTEXT_FULL -> {
                status = "incomplete";
                incomplete = Map.of("reason", "max_output_tokens");
            }
            case CANCELLED -> status = "cancelled";
            default -> status = "completed";
        }
        response.put("status", status);
        response.put("completed_at", "completed".equals(status) ? System.currentTimeMillis() / 1000 : null);
        response.put("error", null);
        response.put("incomplete_details", incomplete);
        response.put("output", ResponsesOutput.reconciled(output, parse));
        response.put("usage", usage(result, promptTokens, cachedTokens));
        emit("completed".equals(status) ? "response.completed" : "response.incomplete", Map.of("response", response));
        return response;
    }

    /** Closes the response as {@code failed} with {@code message}; returns the failed response. */
    Map<String, Object> fail(String message) throws IOException {
        close();
        Map<String, Object> response = new LinkedHashMap<>(head);
        response.put("status", "failed");
        response.put("error", error(message));
        response.put("incomplete_details", null);
        response.put("output", List.copyOf(output));
        response.put("usage", null);
        emit("response.failed", Map.of("response", response));
        return response;
    }

    private static Map<String, Object> usage(GenerationResult result, int promptTokens, int cachedTokens) {
        int generated = result.tokenCount();
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", promptTokens);
        usage.put("input_tokens_details", Map.of("cached_tokens", cachedTokens));
        usage.put("output_tokens", generated);
        usage.put("output_tokens_details", Map.of("reasoning_tokens", result.reasoningTokens()));
        usage.put("total_tokens", promptTokens + generated);
        return usage;
    }

    /** Opens an item of {@code kind} unless one is open; another kind's open item closes first. */
    private void ensure(String kind, String callId, String name, int callIndex) throws IOException {
        if (open != null && open.kind.equals(kind) && callId == null) {
            return;
        }
        close();
        String prefix = switch (kind) {
            case "reasoning" -> "rs_";
            case "message" -> "msg_";
            default -> "fc_";
        };
        open = new Item(kind, newId(prefix), output.size(), callId, name, callIndex);
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("output_index", open.index);
        data.put("item", item(open, "in_progress"));
        emit("response.output_item.added", data);
        if ("message".equals(kind)) {
            Map<String, Object> part = new LinkedHashMap<>();
            part.put("item_id", open.id);
            part.put("output_index", open.index);
            part.put("content_index", 0);
            part.put("part", textPart("", List.of()));
            emit("response.content_part.added", part);
        }
    }

    /** Closes the open item: its {@code .done} events and {@code output_item.done}. */
    private void close() throws IOException {
        if (open == null) {
            return;
        }
        Item item = open;
        open = null;
        String text = item.text.toString();
        switch (item.kind) {
            case "reasoning" -> {
                Map<String, Object> data = itemDone(item);
                data.put("text", text);
                emit("response.reasoning_text.done", data);
            }
            case "message" -> {
                Map<String, Object> data = itemDone(item);
                data.put("text", text);
                data.put("logprobs", List.copyOf(item.logprobs));
                emit("response.output_text.done", data);
                Map<String, Object> part = itemDone(item);
                part.put("part", textPart(text, item.logprobs));
                emit("response.content_part.done", part);
            }
            default -> {
                Map<String, Object> data = new LinkedHashMap<>();
                data.put("item_id", item.id);
                data.put("output_index", item.index);
                data.put("name", item.name);
                data.put("arguments", text);
                emit("response.function_call_arguments.done", data);
            }
        }
        Map<String, Object> done = item(item, "completed");
        output.add(done);
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("output_index", item.index);
        data.put("item", done);
        emit("response.output_item.done", data);
    }

    /** The item object at {@code status}: in progress it carries no content yet. */
    private Map<String, Object> item(Item item, String status) {
        boolean done = "completed".equals(status);
        String text = done ? item.text.toString() : "";
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", item.id);
        switch (item.kind) {
            case "reasoning" -> {
                out.put("type", "reasoning");
                out.put("summary", List.of());
                out.put("content", done ? List.of(Map.of("type", "reasoning_text", "text", text)) : List.of());
                out.put("encrypted_content", null);
            }
            case "message" -> {
                out.put("type", "message");
                out.put("role", "assistant");
                out.put("content", done ? List.of(textPart(text, item.logprobs)) : List.of());
            }
            default -> {
                out.put("type", "function_call");
                out.put("call_id", item.callId);
                out.put("name", item.name);
                out.put("arguments", text);
            }
        }
        out.put("status", status);
        return out;
    }

    private Map<String, Object> textPart(String text, List<Object> entries) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "output_text");
        part.put("text", text);
        part.put("annotations", List.of());
        part.put("logprobs", List.copyOf(entries));
        return part;
    }

    private Map<String, Object> itemDelta(String delta) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("item_id", open.id);
        data.put("output_index", open.index);
        data.put("content_index", 0);
        data.put("delta", delta);
        return data;
    }

    private static Map<String, Object> itemDone(Item item) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("item_id", item.id);
        data.put("output_index", item.index);
        data.put("content_index", 0);
        return data;
    }

    @SuppressWarnings("unchecked")
    private List<Object> logprobEntries(TokenScore[] step) {
        if (!logprobs || step == null) {
            return List.of();
        }
        return (List<Object>) bodies.chatLogprobs(step).get("content");
    }

    private void emit(String type, Map<String, Object> fields) throws IOException {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("type", type);
        data.put("sequence_number", sequence++);
        data.putAll(fields);
        sink.event(type, data);
    }
}
