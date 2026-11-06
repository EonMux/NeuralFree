package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The final output items of a response: the streamed items set right by the tool parser's reading of the whole
 * text ({@link ToolParse}), which can differ from what streamed (a call the final text turned into content, a
 * held-back tail). Reasoning items stay as streamed; the content becomes one message item at the first message's
 * place with every streamed message's logprobs in order (none when the content is empty); the calls fill the streamed function-call items in order, keeping their
 * item ids, and any call the stream did not show is appended. Stateless; the streamed items are not modified.
 */
final class ResponsesOutput {

    private ResponsesOutput() {
    }

    /** {@code streamed} reconciled with {@code parse}. */
    static List<Map<String, Object>> reconciled(List<Map<String, Object>> streamed, ToolParse parse) {
        List<ToolCall> calls = parse.calls();
        String content = parse.content();
        List<Map<String, Object>> out = new ArrayList<>();
        List<Object> logprobs = messageLogprobs(streamed);
        boolean message = false;
        int call = 0;
        int firstCall = -1;
        for (Map<String, Object> item : streamed) {
            Object type = item.get("type");
            if ("message".equals(type)) {
                if (!message && !content.isEmpty()) {
                    out.add(messageItem(String.valueOf(item.get("id")), content, logprobs));
                }
                message = true;
            } else if ("function_call".equals(type)) {
                if (call < calls.size()) {
                    firstCall = firstCall < 0 ? out.size() : firstCall;
                    out.add(withCall(item, calls.get(call++)));
                }
            } else {
                out.add(item);
            }
        }
        if (!message && !content.isEmpty()) {
            Map<String, Object> added = messageItem(ResponsesEvents.newId("msg_"), content, logprobs);
            out.add(firstCall < 0 ? out.size() : firstCall, added);
        }
        for (; call < calls.size(); call++) {
            Map<String, Object> fresh = new LinkedHashMap<>();
            fresh.put("id", ResponsesEvents.newId("fc_"));
            out.add(withCall(fresh, calls.get(call)));
        }
        return out;
    }

    /** The output-text logprobs of every streamed message item, in order. */
    private static List<Object> messageLogprobs(List<Map<String, Object>> streamed) {
        List<Object> out = new ArrayList<>();
        for (Map<String, Object> item : streamed) {
            if ("message".equals(item.get("type")) && item.get("content") instanceof List<?> parts) {
                for (Object p : parts) {
                    if (p instanceof Map<?, ?> part && part.get("logprobs") instanceof List<?> entries) {
                        out.addAll(entries);
                    }
                }
            }
        }
        return out;
    }

    private static Map<String, Object> messageItem(String id, String text, List<Object> logprobs) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "output_text");
        part.put("text", text);
        part.put("annotations", List.of());
        part.put("logprobs", Collections.unmodifiableList(logprobs));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", id);
        out.put("type", "message");
        out.put("role", "assistant");
        out.put("content", List.of(part));
        out.put("status", "completed");
        return out;
    }

    private static Map<String, Object> withCall(Map<String, Object> item, ToolCall call) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", item.get("id"));
        out.put("type", "function_call");
        Object callId = call.id() != null ? call.id() : item.get("call_id");
        out.put("call_id", callId != null ? callId : ResponsesEvents.newId("call_"));
        out.put("name", call.name());
        out.put("arguments", call.arguments());
        out.put("status", "completed");
        return out;
    }
}
