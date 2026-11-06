package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The Cohere Command 3/4 format: tool calls as a JSON array inside an action block, the reply inside response or
 * text blocks (vLLM's {@code cohere_melody} filter, llama.cpp's {@code cohere2moe.cpp}):
 *
 * <pre>{@code
 * <|START_RESPONSE|>visible text<|END_RESPONSE|>
 * <|START_ACTION|>[{"tool_call_id": "0", "tool_name": "f", "parameters": {"a": 1}}]<|END_ACTION|>
 * }</pre>
 *
 * <p>Command 3 wraps replies in {@code <|START_RESPONSE|>}; Command 4 may use {@code <|START_TEXT|>} as well.
 * Unwrapped text outside the blocks is content. Each array object's {@code tool_name}, {@code parameters} and
 * {@code tool_call_id} are the call's name, arguments and id. While a call streams, its arguments grow as the
 * object does, and a trailing proper prefix of any marker is held back from the content.
 */
final class CohereCommandToolParser implements ToolParser {

    private static final String ACTION_OPEN = "<|START_ACTION|>";
    private static final String ACTION_CLOSE = "<|END_ACTION|>";

    private final List<String[]> contentWrappers;
    private final List<String> all;

    /**
     * @param contentWrappers the reply blocks' open and close markers, in model order
     */
    CohereCommandToolParser(List<String[]> contentWrappers) {
        this.contentWrappers = List.copyOf(contentWrappers);
        List<String> all = new ArrayList<>();
        all.add(ACTION_OPEN);
        all.add(ACTION_CLOSE);
        for (String[] wrapper : contentWrappers) {
            all.add(wrapper[0]);
            all.add(wrapper[1]);
        }
        this.all = all;
    }

    /** The wrappers of Command 3: the response block. */
    static CohereCommandToolParser command3() {
        return new CohereCommandToolParser(Collections.singletonList(
                new String[]{"<|START_RESPONSE|>", "<|END_RESPONSE|>"}));
    }

    /** The wrappers of Command 4: the response block and the text block. */
    static CohereCommandToolParser command4() {
        return new CohereCommandToolParser(Arrays.asList(
                new String[]{"<|START_RESPONSE|>", "<|END_RESPONSE|>"},
                new String[]{"<|START_TEXT|>", "<|END_TEXT|>"}));
    }

    @Override
    public List<String> triggers() {
        return Arrays.asList(ACTION_OPEN);
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        StringBuilder content = new StringBuilder();
        List<ToolCall> calls = new ArrayList<>();
        int pos = 0;
        while (pos < text.length()) {
            int at = -1;
            String[] block = null;
            for (String[] wrapper : contentWrappers) {
                int i = text.indexOf(wrapper[0], pos);
                if (i >= 0 && (at < 0 || i < at)) {
                    at = i;
                    block = wrapper;
                }
            }
            int action = text.indexOf(ACTION_OPEN, pos);
            if (action >= 0 && (at < 0 || action < at)) {
                at = action;
                block = null;
            }
            if (at < 0) {
                content.append(partial ? heldTail(text.substring(pos)) : text.substring(pos));
                break;
            }
            content.append(text, pos, at);
            String close = block == null ? ACTION_CLOSE : block[1];
            int bodyStart = at + (block == null ? ACTION_OPEN.length() : block[0].length());
            int bodyEnd = text.indexOf(close, bodyStart);
            boolean closed = bodyEnd >= 0;
            int end = closed ? bodyEnd : text.length();
            if (block == null) {
                calls.addAll(actionCalls(text, bodyStart, end));
            } else {
                content.append(closed ? text.substring(bodyStart, end) : heldTail(text.substring(bodyStart)));
            }
            pos = closed ? end + close.length() : text.length();
        }
        return new ToolParse(content.toString(), calls);
    }

    /** The calls of one action block's body: the JSON array of {@code tool_name}/{@code parameters} objects. */
    private static List<ToolCall> actionCalls(String text, int from, int end) {
        int bracket = JsonScan.skipWs(text, from);
        if (bracket >= end || text.charAt(bracket) != '[') {
            return new ArrayList<>();
        }
        JsonCalls.Run run = JsonCalls.array(text.substring(0, end), bracket);
        return run.invalid ? new ArrayList<>() : run.calls;
    }

    /** The tail of a text piece no marker can still start in: a trailing proper marker prefix is held. */
    private String heldTail(String seg) {
        int longest = 0;
        for (String marker : all) {
            longest = Math.max(longest, marker.length());
        }
        for (int k = Math.min(longest - 1, seg.length()); k > 0; k--) {
            for (String marker : all) {
                if (seg.regionMatches(seg.length() - k, marker, 0, k)) {
                    return seg.substring(0, seg.length() - k);
                }
            }
        }
        return seg;
    }
}
