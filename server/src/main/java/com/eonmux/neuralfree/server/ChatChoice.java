package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolCallStream;
import com.eonmux.neuralfree.gguf.chat.tools.ToolOutput;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One chat choice on its way to the client: reasoning text, content run through the request's {@link ToolOutput}
 * and the tool calls it reads. Each event yields the OpenAI {@code delta} objects it adds ({@link #drain}); the
 * logprobs capture of a piece rides its first delta, or an empty delta when the tool parser held the text back.
 * {@link #finish} closes the output and returns the final message parts. Not thread-safe: one choice, one
 * thread.
 */
final class ChatChoice implements ToolCallStream.Listener {

    /** One streamed delta and the logprobs capture it carries, or null. */
    record Delta(Map<String, Object> delta, TokenScore[] logprobs) { }

    private final ToolOutput tools;
    private final StringBuilder reasoning = new StringBuilder();
    private final List<TokenScore[]> logprobs = new ArrayList<>();
    private final List<Delta> pending = new ArrayList<>();
    private ToolParse result;

    ChatChoice(ToolOutput tools) {
        this.tools = tools;
    }

    /** Routes one event of the choice; its deltas wait in {@link #drain}. */
    void accept(GenerationBatch.Event event) {
        if (event instanceof GenerationBatch.ToolRecipient r) {
            tools.toolCallStarted(r.recipient(), this);
            return;
        }
        if (!(event instanceof GenerationBatch.Text t)) {
            return;
        }
        int before = pending.size();
        switch (t.channel()) {
            case REASONING -> {
                reasoning.append(t.text());
                if (!t.text().isEmpty()) {
                    pending.add(new Delta(Map.of("reasoning_content", t.text()), null));
                }
            }
            case TOOL_CALL -> tools.toolCallText(t.text(), this);
            default -> tools.content(t.text(), this);
        }
        if (t.logprobs() != null) {
            logprobs.add(t.logprobs());
            if (pending.size() > before) {
                Delta first = pending.get(before);
                pending.set(before, new Delta(first.delta(), t.logprobs()));
            } else {
                pending.add(new Delta(Map.of(), t.logprobs()));
            }
        }
    }

    /** Ends the output; the final parse's remaining deltas wait in {@link #drain}. */
    ToolParse finish() {
        if (result == null) {
            result = tools.finish(this);
        }
        return result;
    }

    /** The deltas produced since the last drain. */
    List<Delta> drain() {
        List<Delta> out = new ArrayList<>(pending);
        pending.clear();
        return out;
    }

    /** The reasoning text. */
    String reasoning() {
        return reasoning.toString();
    }

    /** Every logprobs capture, in token order. */
    List<TokenScore[]> logprobs() {
        return logprobs;
    }

    @Override
    public void content(String delta) {
        pending.add(new Delta(Map.of("content", delta), null));
    }

    @Override
    public void callStarted(int index, String id, String name) {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", name);
        fn.put("arguments", "");
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("index", index);
        call.put("id", id);
        call.put("type", "function");
        call.put("function", fn);
        pending.add(new Delta(Map.of("tool_calls", List.of(call)), null));
    }

    @Override
    public void arguments(int index, String delta) {
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("index", index);
        call.put("function", Map.of("arguments", delta));
        pending.add(new Delta(Map.of("tool_calls", List.of(call)), null));
    }

    /** The OpenAI {@code tool_calls} array of finished calls. */
    static List<Object> toolCallsBody(List<ToolCall> calls) {
        List<Object> out = new ArrayList<>();
        for (ToolCall c : calls) {
            Map<String, Object> fn = new LinkedHashMap<>();
            fn.put("name", c.name());
            fn.put("arguments", c.arguments());
            Map<String, Object> call = new LinkedHashMap<>();
            call.put("id", c.id());
            call.put("type", "function");
            call.put("function", fn);
            out.add(call);
        }
        return out;
    }
}
