package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The Inkling format: self-describing blocks, each a model-role header, a content-kind marker, a body and an end
 * marker — {@code <|message_model|><|content_text|>text<|end_message|>} — a block also ending with the standalone
 * sampling end {@code <|content_model_end_sampling|>}:
 *
 * <pre>{@code
 * <|message_model|><|content_text|>visible text<|end_message|>
 * <|message_model|><|content_invoke_tool_json|>{"name":"f","args":{"a":1}}<|end_message|>
 * }</pre>
 *
 * <p>The content is the bodies of the text, tool-text and tool-error blocks plus any raw text between blocks; a
 * thinking block's body is reasoning, not content. A tool block's body is one JSON object whose {@code name} and
 * {@code args} are the call. While a call streams, its arguments grow as the object does, and a trailing proper
 * prefix of any marker is held back from the content.
 */
final class InklingToolParser implements ToolParser {

    private static final String HEADER = "<|message_model|>";
    private static final String TEXT = "<|content_text|>";
    private static final String THINKING = "<|content_thinking|>";
    private static final String TOOL_JSON = "<|content_invoke_tool_json|>";
    private static final String TOOL_TEXT = "<|content_invoke_tool_text|>";
    private static final String TOOL_ERROR = "<|content_tool_error|>";
    private static final String END = "<|end_message|>";
    private static final String SAMPLING = "<|content_model_end_sampling|>";

    private static final List<String> OPENS = Arrays.asList(HEADER, TEXT, THINKING, TOOL_JSON, TOOL_TEXT, TOOL_ERROR);
    private static final List<String> ALL = Arrays.asList(HEADER, TEXT, THINKING, TOOL_JSON, TOOL_TEXT, TOOL_ERROR,
            END, SAMPLING);

    @Override
    public List<String> triggers() {
        return Arrays.asList(TOOL_JSON, TEXT);
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        StringBuilder content = new StringBuilder();
        List<ToolCall> calls = new ArrayList<>();
        int pos = 0;
        while (pos < text.length()) {
            int at = -1;
            String kind = null;
            for (String open : OPENS) {
                int i = text.indexOf(open, pos);
                if (i >= 0 && (at < 0 || i < at)) {
                    at = i;
                    kind = open;
                }
            }
            if (at < 0) {
                content.append(partial ? heldTail(text.substring(pos)) : text.substring(pos));
                break;
            }
            content.append(text, pos, at);
            pos = at + kind.length();
            if (kind.equals(HEADER)) {
                continue;
            }
            int bodyEnd = minPositive(text.indexOf(END, pos), text.indexOf(SAMPLING, pos));
            boolean closed = bodyEnd >= 0;
            int end = closed ? bodyEnd : text.length();
            if (kind.equals(TOOL_JSON)) {
                ToolCall c = jsonCall(text, pos, end, closed);
                if (c != null) {
                    calls.add(c);
                }
            } else if (!kind.equals(THINKING)) {
                content.append(closed ? text.substring(pos, end) : heldTail(text.substring(pos)));
            }
            pos = closed ? (text.startsWith(SAMPLING, bodyEnd) ? bodyEnd + SAMPLING.length()
                    : bodyEnd + END.length()) : text.length();
        }
        return new ToolParse(content.toString(), calls);
    }

    /** The call of one tool-JSON block's body, or null when it names no function (yet). */
    private static ToolCall jsonCall(String text, int from, int end, boolean closed) {
        int brace = JsonScan.skipWs(text, from);
        if (brace >= end || text.charAt(brace) != '{') {
            return null;
        }
        JsonCalls.ObjectCall o = JsonCalls.object(text.substring(0, end), brace);
        if (o.invalid || o.call == null) {
            return null;
        }
        boolean complete = o.end >= 0 && closed;
        return complete ? new ToolCall(o.call.name(), o.call.arguments(), o.call.id(), true)
                : new ToolCall(o.call.name(), o.call.arguments(), o.call.id(), false);
    }

    /** The tail of a text piece no marker can still start in: a trailing proper marker prefix is held. */
    private static String heldTail(String seg) {
        int longest = 0;
        for (String marker : ALL) {
            longest = Math.max(longest, marker.length());
        }
        for (int k = Math.min(longest - 1, seg.length()); k > 0; k--) {
            for (String marker : ALL) {
                if (seg.regionMatches(seg.length() - k, marker, 0, k)) {
                    return seg.substring(0, seg.length() - k);
                }
            }
        }
        return seg;
    }

    private static int minPositive(int a, int b) {
        if (a < 0) {
            return b;
        }
        return b < 0 ? a : Math.min(a, b);
    }
}
