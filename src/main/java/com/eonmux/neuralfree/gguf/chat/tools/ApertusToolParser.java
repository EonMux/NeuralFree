package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The Apertus format: a JSON array of single-key objects — the function name mapped to its arguments object —
 * between two special tokens:
 *
 * <pre>{@code
 * <|tools_prefix|>[{"get_weather": {"city": "Paris"}}]<|tools_suffix|>
 * }</pre>
 *
 * <p>The suffix a final text never writes is not required (vLLM's pattern ends the array or the text, whichever
 * comes first). While an object streams, its arguments grow as the value text does, so every partial argument text
 * is a prefix of the final one.
 */
final class ApertusToolParser extends BlockToolParser {

    private static final String PREFIX = "<|tools_prefix|>";
    private static final String SUFFIX = "<|tools_suffix|>";

    ApertusToolParser() {
        super(Arrays.asList(PREFIX));
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        int i = JsonScan.skipWs(text, at + opener.length());
        if (i >= text.length() || text.charAt(i) != '[') {
            return partial ? Block.incomplete(calls, at) : Block.malformed(Math.max(i, at + opener.length()));
        }
        i = JsonScan.skipWs(text, i + 1);
        while (true) {
            if (i >= text.length()) {
                return partial ? Block.incomplete(calls, at) : Block.complete(calls, text.length());
            }
            if (text.charAt(i) == ']') {
                return closed(text, calls, i + 1, at, partial);
            }
            int end = object(text, i, calls);
            if (end == JsonScan.INVALID) {
                return Block.malformed(Math.max(i + 1, at + opener.length()));
            }
            if (end < 0) {
                return partial ? Block.incomplete(calls, at) : Block.complete(calls, text.length());
            }
            int j = JsonScan.skipWs(text, end);
            if (j < text.length() && text.charAt(j) == ',') {
                i = JsonScan.skipWs(text, j + 1);
                continue;
            }
            if (j < text.length() && text.charAt(j) == ']') {
                return closed(text, calls, j + 1, at, partial);
            }
            if (j >= text.length()) {
                return partial ? Block.incomplete(calls, at) : Block.complete(calls, text.length());
            }
            return Block.malformed(Math.max(j + 1, at + opener.length()));
        }
    }

    /** The block's end after the array: past the suffix when it is there, cut off while it may still arrive. */
    private static Block closed(String text, List<ToolCall> calls, int afterArray, int at, boolean partial) {
        if (text.startsWith(SUFFIX, afterArray)) {
            return Block.complete(calls, afterArray + SUFFIX.length());
        }
        if (partial && (afterArray >= text.length() || prefixOf(text, afterArray, SUFFIX))) {
            return Block.incomplete(calls, at);
        }
        return Block.complete(calls, afterArray);
    }

    /**
     * Reads the single-key object at the brace {@code at} into {@code calls} — the key names the function, the
     * value is its arguments object — and returns where the object ends: -1 when the text ends inside it,
     * {@link JsonScan#INVALID} when it is none.
     */
    private static int object(String text, int at, List<ToolCall> calls) {
        List<JsonScan.Member> members = JsonScan.members(text, at);
        if (members == null) {
            return JsonScan.INVALID;
        }
        if (members.size() > 1 || (members.size() == 1 && members.get(0).key.isEmpty())) {
            return JsonScan.INVALID;
        }
        if (members.isEmpty()) {
            int end = JsonScan.valueEnd(text, at);
            return end == JsonScan.INVALID ? JsonScan.INVALID : end;
        }
        JsonScan.Member m = members.get(0);
        if (!m.complete()) {
            String source = BlockToolParser.rtrim(text.substring(m.valueStart));
            calls.add(new ToolCall(m.key, source.startsWith("{") ? source : "{", null, false));
            return -1;
        }
        String value = text.substring(m.valueStart, m.valueEnd);
        calls.add(new ToolCall(m.key, value.startsWith("{") ? value : "{}", null, true));
        return JsonScan.valueEnd(text, at);
    }
}
