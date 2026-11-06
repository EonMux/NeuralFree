package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Mistral's two tool-call dialects after {@code [TOOL_CALLS]}: a JSON array of call objects (tokenizer versions
 * before v11, Mistral-Nemo) and {@code NAME[ARGS]{...}}, optionally with {@code [CALL_ID]ID} before the arguments
 * (v11 and later: Mistral Small 3.2, Devstral, Ministral 3). Each {@code [TOOL_CALLS]} of the newer dialect opens
 * one call.
 */
final class MistralToolParser extends BlockToolParser {

    static final String TOOL_CALLS = "[TOOL_CALLS]";
    private static final String ARGS = "[ARGS]";
    private static final String CALL_ID = "[CALL_ID]";

    MistralToolParser() {
        super(Collections.singletonList(TOOL_CALLS));
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        int i = JsonScan.skipWs(text, at + opener.length());
        if (i >= text.length()) {
            return partial ? Block.incomplete(new ArrayList<>(), at) : Block.malformed(text.length());
        }
        if (text.charAt(i) == '[') {
            JsonCalls.Run run = JsonCalls.array(text, i);
            if (run.invalid) {
                return Block.malformed(i + 1);
            }
            if (run.end < 0) {
                if (partial) {
                    return Block.incomplete(run.calls, at);
                }
                return Block.malformed(text.length());
            }
            return Block.complete(run.calls, run.end);
        }
        int args = text.indexOf(ARGS, i);
        int callId = text.indexOf(CALL_ID, i);
        int nameEnd = callId >= 0 && (args < 0 || callId < args) ? callId : args;
        if (nameEnd < 0) {
            boolean couldBeName = text.substring(i).chars().noneMatch(c -> c == '{' || Character.isWhitespace(c));
            return partial && couldBeName ? Block.incomplete(new ArrayList<>(), at) : Block.malformed(i + 1);
        }
        String name = text.substring(i, nameEnd).trim();
        if (name.isEmpty()) {
            return Block.malformed(nameEnd);
        }
        String id = null;
        if (nameEnd == callId) {
            int idStart = callId + CALL_ID.length();
            if (args < 0) {
                return partial ? Block.incomplete(new ArrayList<>(), at) : Block.malformed(text.length());
            }
            id = text.substring(idStart, args).trim();
        }
        int a = JsonScan.skipWs(text, args + ARGS.length());
        List<ToolCall> calls = new ArrayList<>();
        if (a >= text.length()) {
            if (partial) {
                calls.add(new ToolCall(name, "", id, false));
                return Block.incomplete(calls, at);
            }
            calls.add(new ToolCall(name, "{}", id, true));
            return Block.complete(calls, text.length());
        }
        if (text.charAt(a) != '{') {
            return Block.malformed(a);
        }
        int e = JsonScan.valueEnd(text, a);
        if (e == JsonScan.INVALID) {
            return Block.malformed(a);
        }
        if (e < 0) {
            if (!partial) {
                return Block.malformed(text.length());
            }
            calls.add(new ToolCall(name, rtrim(text.substring(a)), id, false));
            return Block.incomplete(calls, at);
        }
        String argText = text.substring(a, e);
        if (JsonScan.parseOrNull(argText) == null) {
            return Block.malformed(e);
        }
        calls.add(new ToolCall(name, argText, id, true));
        return Block.complete(calls, e);
    }
}
