package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The Functionary v3.2 format (llama.cpp's {@code functionary-v3-2.cpp}): the output opens with a recipient after
 * the generation prompt's {@code >>>} — {@code >>>all\n} carries the reply text, any other recipient starts a call
 * whose arguments are one JSON object:
 *
 * <pre>{@code
 * >>>all\nLet me check.
 * >>>get_weather
 * {"city": "Paris"}
 * }</pre>
 *
 * <p>The reply text runs to the next {@code >>>}. While a call streams, its arguments grow as the object does; the
 * reply text holds back a trailing proper prefix of {@code >>>}, and the whole text is held while it could still
 * become the first recipient.
 */
final class FunctionaryToolParser implements ToolParser {

    private static final String RECIPIENT_START = ">>>";

    @Override
    public List<String> triggers() {
        return Arrays.asList(RECIPIENT_START);
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        if (!text.startsWith(RECIPIENT_START)) {
            if (partial && RECIPIENT_START.startsWith(text)) {
                return ToolParse.empty();
            }
            return new ToolParse(partial ? heldTail(text) : text, new ArrayList<>());
        }
        int nl = text.indexOf('\n', RECIPIENT_START.length());
        if (nl < 0) {
            return ToolParse.empty();
        }
        String recipient = text.substring(RECIPIENT_START.length(), nl).trim();
        if (recipient.equals("all")) {
            return allBody(text, nl + 1, partial);
        }
        return new ToolParse("", calls(text, 0));
    }

    /** The reply of a {@code >>>all} body, to the next {@code >>>}; the calls follow it. */
    private static ToolParse allBody(String text, int from, boolean partial) {
        int end = text.indexOf(RECIPIENT_START, from);
        boolean callFollows = end >= 0;
        int contentEnd = callFollows ? end : text.length();
        if (partial) {
            contentEnd = heldEnd(text, contentEnd);
        }
        String content = contentEnd > from ? text.substring(from, contentEnd) : "";
        if (partial || callFollows) {
            content = BlockToolParser.rtrim(content);
        }
        int callsFrom = callFollows ? end : text.length();
        return new ToolParse(content, calls(text, callsFrom));
    }

    /** The calls from {@code from} on: each {@code >>>name\n{json}} whose object is the arguments. */
    private static List<ToolCall> calls(String text, int from) {
        List<ToolCall> out = new ArrayList<>();
        int i = JsonScan.skipWs(text, from);
        while (i < text.length()) {
            i = JsonScan.skipWs(text, i);
            if (!text.startsWith(RECIPIENT_START, i)) {
                break;
            }
            int nl = text.indexOf('\n', i + RECIPIENT_START.length());
            if (nl < 0) {
                break;
            }
            String name = text.substring(i + RECIPIENT_START.length(), nl).trim();
            if (name.isEmpty() || name.equals("all")) {
                break;
            }
            int brace = JsonScan.skipWs(text, nl + 1);
            if (brace >= text.length() || text.charAt(brace) != '{') {
                break;
            }
            int end = JsonScan.valueEnd(text, brace);
            if (end == JsonScan.INVALID) {
                break;
            }
            if (end < 0) {
                out.add(new ToolCall(name, BlockToolParser.rtrim(text.substring(brace)), null, false));
                break;
            }
            out.add(new ToolCall(name, text.substring(brace, end), null, true));
            i = end;
        }
        return out;
    }

    /** The end of a reply text no {@code >>>} can still start in: a trailing proper prefix is held. */
    private static int heldEnd(String text, int end) {
        for (int k = Math.min(RECIPIENT_START.length() - 1, end); k > 0; k--) {
            if (text.regionMatches(end - k, RECIPIENT_START, 0, k)) {
                return end - k;
            }
        }
        return end;
    }

    private static String heldTail(String seg) {
        return seg.substring(0, heldEnd(seg, seg.length()));
    }
}
