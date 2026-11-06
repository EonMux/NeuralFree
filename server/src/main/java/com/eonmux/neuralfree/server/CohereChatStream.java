package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The Server-Sent Event stream of one Cohere Chat v2 answer, as vLLM's {@code _chat_completion_stream_to_v2}
 * writes it: {@code message-start}, then content blocks ({@code content-start}, {@code content-delta},
 * {@code content-end}) and tool-call blocks ({@code tool-call-start}, {@code tool-call-delta},
 * {@code tool-call-end}) as they open and close, then {@code message-end} carrying the finish reason and the usage,
 * then the {@code data: [DONE]} terminator. A block is closed when a block of another kind opens or the message
 * ends. Reasoning text opens a {@code thinking} block, content a {@code text} block. Frames carry no {@code event:}
 * line: each JSON object's {@code type} is its discriminator. Not thread-safe: one answer, one thread.
 */
final class CohereChatStream {

    private final OutputStream out;
    private final String id;
    private final Set<Integer> startedCalls = new HashSet<>();
    private int nextContentIndex;
    private String openKind;
    private int openContentIndex = -1;
    private Integer openToolIndex;
    private boolean started;

    CohereChatStream(OutputStream out, String id) {
        this.out = out;
        this.id = id;
    }

    /** Emits {@code message-start} once, before the first delta of the answer. */
    void start() throws IOException {
        if (started) {
            return;
        }
        started = true;
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "message-start");
        event.put("id", id);
        event.put("delta", Map.of("message", message));
        write(event);
    }

    /** Emits the events of one OpenAI-shaped chat delta: reasoning, content and tool-call pieces. */
    void delta(Map<String, Object> delta) throws IOException {
        start();
        String reasoning = string(delta.get("reasoning"));
        if (reasoning.isEmpty()) {
            reasoning = string(delta.get("reasoning_content"));
        }
        if (!reasoning.isEmpty()) {
            thinking(reasoning);
        }
        String content = string(delta.get("content"));
        if (!content.isEmpty()) {
            text(content);
        }
        if (delta.get("tool_calls") instanceof List<?> calls) {
            for (Object raw : calls) {
                if (raw instanceof Map<?, ?> call) {
                    toolCall(OpenAiRequests.stringKeys(call));
                }
            }
        }
    }

    /** Closes the open block, then emits {@code message-end} with the finish reason and usage. */
    void finish(String finishReason, Map<String, Object> usage) throws IOException {
        finish(finishReason, usage, null);
    }

    /** {@link #finish(String, Map)} carrying the request's {@code kv_transfer_params} when it ran a transfer. */
    void finish(String finishReason, Map<String, Object> usage, Map<String, Object> transfer) throws IOException {
        start();
        close();
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("finish_reason", finishReason);
        delta.put("usage", usage);
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "message-end");
        event.put("id", id);
        event.put("delta", delta);
        if (transfer != null) {
            event.put("kv_transfer_params", transfer);
        }
        write(event);
        writeDone();
    }

    /** Closes the open block and reports the failure as an {@code ERROR} message end. */
    void error(String message) throws IOException {
        start();
        close();
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("error", message);
        delta.put("finish_reason", "ERROR");
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "message-end");
        event.put("id", id);
        event.put("delta", delta);
        write(event);
        writeDone();
    }

    private void text(String piece) throws IOException {
        if (!"text".equals(openKind)) {
            close();
            openKind = "text";
            openContentIndex = nextContentIndex++;
            contentStart("text", "text");
        }
        contentDelta("text", piece);
    }

    private void thinking(String piece) throws IOException {
        if (!"thinking".equals(openKind)) {
            close();
            openKind = "thinking";
            openContentIndex = nextContentIndex++;
            contentStart("thinking", "thinking");
        }
        contentDelta("thinking", piece);
    }

    private void toolCall(Map<String, Object> call) throws IOException {
        Object rawIndex = call.get("index");
        int index = rawIndex instanceof Number number ? number.intValue() : startedCalls.size();
        Map<String, Object> function = call.get("function") instanceof Map<?, ?> fn
                ? OpenAiRequests.stringKeys(fn) : Map.of();
        String arguments = string(function.get("arguments"));
        if (!startedCalls.contains(index)) {
            close();
            startedCalls.add(index);
            openToolIndex = index;
            Map<String, Object> fn = new LinkedHashMap<>();
            fn.put("name", string(function.get("name")));
            fn.put("arguments", "");
            Map<String, Object> tool = new LinkedHashMap<>();
            tool.put("id", string(call.get("id")));
            tool.put("type", "function");
            tool.put("function", fn);
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("type", "tool-call-start");
            event.put("index", index);
            event.put("delta", Map.of("message", Map.of("tool_calls", tool)));
            write(event);
            return;
        }
        if (!arguments.isEmpty()) {
            Map<String, Object> fn = Map.of("arguments", arguments);
            Map<String, Object> tool = Map.of("function", fn);
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("type", "tool-call-delta");
            event.put("index", index);
            event.put("delta", Map.of("message", Map.of("tool_calls", tool)));
            write(event);
        }
    }

    private void contentStart(String kind, String field) throws IOException {
        Map<String, Object> content = new LinkedHashMap<>();
        content.put("type", kind);
        content.put(field, "");
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "content-start");
        event.put("index", openContentIndex);
        event.put("delta", Map.of("message", Map.of("content", content)));
        write(event);
    }

    private void contentDelta(String field, String piece) throws IOException {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "content-delta");
        event.put("index", openContentIndex);
        event.put("delta", Map.of("message", Map.of("content", Map.of(field, piece))));
        write(event);
    }

    /** Emits the end event of the open content or tool-call block, if any. */
    private void close() throws IOException {
        if (openKind != null) {
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("type", "content-end");
            event.put("index", openContentIndex);
            write(event);
            openKind = null;
            openContentIndex = -1;
        }
        if (openToolIndex != null) {
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("type", "tool-call-end");
            event.put("index", openToolIndex);
            write(event);
            openToolIndex = null;
        }
    }

    private void write(Map<String, Object> event) throws IOException {
        ServerJson.data(out, ServerJson.write(event));
    }

    private void writeDone() throws IOException {
        ServerJson.data(out, "[DONE]");
    }

    private static String string(Object raw) {
        return raw instanceof String text ? text : "";
    }
}
