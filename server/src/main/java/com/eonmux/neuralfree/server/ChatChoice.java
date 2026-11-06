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
 * One chat choice on its way to the client: reasoning text (dropped when the request set
 * {@code include_reasoning} false), content run through the request's {@link ToolOutput}
 * and the tool calls it reads. Each event yields the OpenAI {@code delta} objects it adds ({@link #drain}); the
 * logprobs capture of a piece rides its first delta, or an empty delta when the tool parser held the text back.
 * {@link #finish} closes the output and returns the final message parts. Reasoning carries both vLLM's
 * {@code reasoning} and llama.cpp's {@code reasoning_content}. A content fallback (the reasoning released
 * again as the content of an answer without one) streams both copies, while the assembled message keeps the
 * reasoning as its content only ({@link #messageReasoning}), as vLLM's non-streamed swap does. Not thread-safe: one choice, one
 * thread.
 */
final class ChatChoice implements ToolCallStream.Listener {

    /** One streamed delta, the logprobs capture it carries (or null) and its token id (or null). */
    record Delta(Map<String, Object> delta, TokenScore[] logprobs, Integer tokenId) {

        Delta(Map<String, Object> delta, TokenScore[] logprobs) {
            this(delta, logprobs, null);
        }
    }

    private final ToolOutput tools;
    private final boolean includeReasoning;
    private final StringBuilder reasoning = new StringBuilder();
    private final List<TokenScore[]> logprobs = new ArrayList<>();
    private final List<Delta> pending = new ArrayList<>();
    private ToolParse result;
    private boolean contentFallback;

    ChatChoice(ToolOutput tools, boolean includeReasoning) {
        this.tools = tools;
        this.includeReasoning = includeReasoning;
    }

    /** Routes one event of the choice; its deltas wait in {@link #drain}. */
    void accept(GenerationBatch.Event event) {
        if (event instanceof GenerationBatch.ToolRecipient r) {
            tools.toolCallStarted(r.recipient(), this);
            return;
        }
        if (event instanceof GenerationBatch.ContentFallback) {
            contentFallback = true;
            return;
        }
        if (!(event instanceof GenerationBatch.Text t)) {
            return;
        }
        int before = pending.size();
        switch (t.channel()) {
            case REASONING -> {
                if (includeReasoning) {
                    reasoning.append(t.text());
                    if (!t.text().isEmpty()) {
                        pending.add(new Delta(reasoningDelta(t.text()), null, t.tokenId()));
                    }
                }
                // include_reasoning false: the reasoning never reaches the response (vLLM drops it);
                // the token's logprobs capture below still rides its empty delta
            }
            case TOOL_CALL -> tools.toolCallText(t.text(), this);
            default -> {
                int at = pending.size();
                tools.content(t.text(), this);
                if (pending.size() > at) {
                    Delta first = pending.get(at);
                    pending.set(at, new Delta(first.delta(), first.logprobs(), t.tokenId()));
                }
            }
        }
        if (t.logprobs() != null) {
            logprobs.add(t.logprobs());
            if (pending.size() > before) {
                Delta first = pending.get(before);
                pending.set(before, new Delta(first.delta(), t.logprobs(), first.tokenId()));
            } else {
                pending.add(new Delta(Map.of(), t.logprobs(), t.tokenId()));
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

    /** The reasoning text, as streamed. */
    String reasoning() {
        return reasoning.toString();
    }

    /** The delta carrying {@code text} as vLLM's {@code reasoning} and llama.cpp's {@code reasoning_content}. */
    static Map<String, Object> reasoningDelta(String text) {
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("reasoning_content", text);
        delta.put("reasoning", text);
        return delta;
    }

    /** The reasoning of the assembled message: empty when it became the content (a content fallback). */
    String messageReasoning() {
        return contentFallback ? "" : reasoning.toString();
    }

    /** Whether the reasoning was released again as the content. */
    boolean contentFallback() {
        return contentFallback;
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
