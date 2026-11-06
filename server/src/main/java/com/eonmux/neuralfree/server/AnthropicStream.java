package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolCallStream;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The content blocks of one streamed Anthropic message: a block opens when its first delta arrives and closes when
 * a block of another kind opens or the message ends — {@code thinking} blocks ({@code thinking_delta}),
 * {@code text} blocks ({@code text_delta}) and one {@code tool_use} block per call ({@code input_json_delta}).
 * Not thread-safe.
 */
final class AnthropicStream implements ToolCallStream.Listener {

    private final OutputStream out;
    private int index = -1;
    private String open;
    private int openCall = -1;
    private boolean calls;

    AnthropicStream(OutputStream out) {
        this.out = out;
    }

    /** More reasoning text. */
    void thinking(String delta) {
        if (delta.isEmpty()) {
            return;
        }
        ensure("thinking");
        delta("thinking_delta", "thinking", delta);
    }

    @Override
    public void content(String delta) {
        if (delta.isEmpty()) {
            return;
        }
        ensure("text");
        delta("text_delta", "text", delta);
    }

    @Override
    public void callStarted(int callIndex, String id, String name) {
        calls = true;
        close();
        Map<String, Object> block = new LinkedHashMap<>();
        block.put("type", "tool_use");
        block.put("id", id);
        block.put("name", name);
        block.put("input", Map.of());
        start(block, "tool_use");
        openCall = callIndex;
    }

    /**
     * Argument text of the open call. A closed Anthropic block cannot take more deltas, so text for any other call
     * is not streamed; the tool parsers never produce it (a call's arguments end before the next call starts).
     */
    @Override
    public void arguments(int callIndex, String delta) {
        if (!"tool_use".equals(open) || callIndex != openCall) {
            return;
        }
        delta("input_json_delta", "partial_json", delta);
    }

    /** Closes the open block; a message that produced none still carries one empty text block. */
    void end() {
        if (index < 0) {
            ensure("text");
        }
        close();
    }

    /** Whether a tool_use block was streamed. */
    boolean calls() {
        return calls;
    }

    private void ensure(String kind) {
        if (kind.equals(open)) {
            return;
        }
        close();
        Map<String, Object> block = new LinkedHashMap<>();
        block.put("type", kind);
        block.put(kind, "");
        if (kind.equals("thinking")) {
            block.put("signature", "");
        }
        start(block, kind);
    }

    private void start(Map<String, Object> block, String kind) {
        index++;
        open = kind;
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "content_block_start");
        event.put("index", index);
        event.put("content_block", block);
        write("content_block_start", event);
    }

    private void close() {
        if (open == null) {
            return;
        }
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "content_block_stop");
        event.put("index", index);
        write("content_block_stop", event);
        open = null;
        openCall = -1;
    }

    private void delta(String type, String field, String text) {
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("type", type);
        delta.put(field, text);
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "content_block_delta");
        event.put("index", index);
        event.put("delta", delta);
        write("content_block_delta", event);
    }

    private void write(String type, Map<String, Object> event) {
        try {
            ServerJson.event(out, type, event);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
