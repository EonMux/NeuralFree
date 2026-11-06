package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The Kimi K3 XTML format: the reply and the tool calls live in channels built from the special tokens
 * {@code <|open|>}, {@code <|close|>} and {@code <|sep|>}:
 *
 * <pre>{@code
 * <|open|>response<|sep|>visible text<|close|>response<|sep|>
 * <|open|>tools<|sep|>
 *   <|open|>call tool="f" index="1"<|sep|>
 *     <|open|>argument key="code" type="string"<|sep|>print(1)<|close|>argument<|sep|>
 *   <|close|>call<|sep|>
 * <|close|>tools<|sep|>
 * }</pre>
 *
 * <p>The content is the response channel's body; when the output opens with no response opener, the text before
 * the tools channel with any response or message closers removed. An {@code argument}'s value is the raw text when
 * its {@code type} reads {@code string} (the default), else JSON-decoded with the raw text as fallback. Attribute
 * values carry {@code &quot;}/{@code &amp;} escapes, which decode here. While a call streams, a string argument's
 * value grows inside the arguments JSON, a value of any other type is held back until its block closes, and every
 * streamed content text is a prefix of the final one.
 */
final class XtmlToolParser implements ToolParser {

    private static final String OPEN = "<|open|>";
    private static final String CLOSE = "<|close|>";
    private static final String SEP = "<|sep|>";

    private static final String RESPONSE_OPEN = OPEN + "response" + SEP;
    private static final String RESPONSE_CLOSE = CLOSE + "response" + SEP;
    private static final String MESSAGE_CLOSE = CLOSE + "message" + SEP;
    private static final String TOOLS_OPEN = OPEN + "tools" + SEP;
    private static final String TOOLS_CLOSE = CLOSE + "tools" + SEP;
    private static final String CALL_OPEN = OPEN + "call ";
    private static final String CALL_CLOSE = CLOSE + "call" + SEP;
    private static final String ARG_OPEN = OPEN + "argument ";
    private static final String ARG_CLOSE = CLOSE + "argument" + SEP;

    private static final List<String> MARKERS = Collections.unmodifiableList(Arrays.asList(
            RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, TOOLS_OPEN, TOOLS_CLOSE,
            CALL_CLOSE, ARG_OPEN, ARG_CLOSE, OPEN, CLOSE, SEP));

    @Override
    public List<String> triggers() {
        return Collections.singletonList(TOOLS_OPEN);
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        int responseOpen = text.indexOf(RESPONSE_OPEN);
        int bodyStart = responseOpen >= 0 ? responseOpen + RESPONSE_OPEN.length() : 0;
        int toolsOpen = text.indexOf(TOOLS_OPEN, bodyStart);
        int responseClose = text.indexOf(RESPONSE_CLOSE, bodyStart);
        int contentEnd = minPositive(toolsOpen, responseClose);
        return new ToolParse(content(text, bodyStart, contentEnd, responseOpen < 0, partial),
                calls(text, toolsOpen));
    }

    /**
     * The visible text: up to {@code contentEnd} — the tools channel or the response closer — with response and
     * message closers removed when the response channel never opened. A trailing proper prefix of a marker is held
     * back while the text may still grow, so every streamed content text is a prefix of the next.
     */
    private static String content(String text, int bodyStart, int contentEnd, boolean stripClosers,
                                  boolean partial) {
        int end = contentEnd < 0 ? text.length() : contentEnd;
        if (partial) {
            end = safeEnd(text, end);
        }
        if (end <= bodyStart) {
            return "";
        }
        String body = text.substring(bodyStart, end);
        return stripClosers ? body.replace(RESPONSE_CLOSE, "").replace(MESSAGE_CLOSE, "") : body;
    }

    /** The end of {@code text} no marker can still start in: a trailing proper marker prefix is held. */
    private static int safeEnd(String text, int end) {
        int longest = 0;
        for (String marker : MARKERS) {
            longest = Math.max(longest, marker.length());
        }
        for (int k = Math.min(longest - 1, end); k > 0; k--) {
            for (String marker : MARKERS) {
                if (end - k >= 0 && text.regionMatches(end - k, marker, 0, k)) {
                    return end - k;
                }
            }
        }
        return end;
    }

    private static int minPositive(int a, int b) {
        if (a < 0) {
            return b;
        }
        return b < 0 ? a : Math.min(a, b);
    }

    /** The calls of the tools channel opening at {@code toolsOpen} (-1 when none): each block with its closer. */
    private static List<ToolCall> calls(String text, int toolsOpen) {
        List<ToolCall> out = new ArrayList<>();
        if (toolsOpen < 0) {
            return out;
        }
        int toolsClose = text.indexOf(TOOLS_CLOSE, toolsOpen);
        int section = toolsClose < 0 ? text.length() : toolsClose;
        int i = toolsOpen + TOOLS_OPEN.length();
        while (i < section) {
            int open = text.indexOf(CALL_OPEN, i);
            if (open < 0 || open >= section) {
                break;
            }
            int attrsEnd = text.indexOf(SEP, open + CALL_OPEN.length());
            if (attrsEnd < 0 || attrsEnd >= section) {
                break;
            }
            int close = text.indexOf(CALL_CLOSE, attrsEnd);
            if (close < 0 || close >= section) {
                break;
            }
            Map<String, String> attrs = attrs(text.substring(open + CALL_OPEN.length(), attrsEnd));
            ToolCall c = call(attrs.get("tool"), text.substring(attrsEnd + SEP.length(), close));
            if (c != null) {
                out.add(c);
            }
            i = close + CALL_CLOSE.length();
        }
        return out;
    }

    /** The call of one {@code call} block: its arguments from the argument blocks, or null without a tool name. */
    private static ToolCall call(String tool, String body) {
        if (tool == null || tool.isEmpty()) {
            return null;
        }
        boolean complete = true;
        Map<String, Object> args = new LinkedHashMap<>();
        int i = 0;
        while (i < body.length()) {
            int open = body.indexOf(ARG_OPEN, i);
            if (open < 0) {
                break;
            }
            int attrsEnd = body.indexOf(SEP, open + ARG_OPEN.length());
            if (attrsEnd < 0) {
                complete = false;
                break;
            }
            int valueStart = attrsEnd + SEP.length();
            int close = body.indexOf(ARG_CLOSE, valueStart);
            Map<String, String> attrs = attrs(body.substring(open + ARG_OPEN.length(), attrsEnd));
            String key = attrs.get("key");
            String type = attrs.get("type");
            if (key == null || key.isEmpty()) {
                if (close < 0) {
                    complete = false;
                    break;
                }
                i = close + ARG_CLOSE.length();
                continue;
            }
            if (close < 0) {
                complete = false;
                if (type == null || "string".equals(type)) {
                    args.put(key, body.substring(valueStart));
                }
                break;
            }
            args.put(key, typed(body.substring(valueStart, close), type));
            i = close + ARG_CLOSE.length();
        }
        String json = ToolSchemas.arguments(args);
        return new ToolCall(tool, complete ? json : json.substring(0, json.length() - 1), null, complete);
    }

    /** The JSON value of an argument: raw text for {@code string} (the default), else its JSON or the text. */
    private static Object typed(String raw, String type) {
        if (type == null || "string".equals(type)) {
            return raw;
        }
        Object parsed = JsonScan.parseOrNull(raw);
        return parsed == null ? raw : parsed;
    }

    /** The attributes of one block header: {@code key="value"} pairs with their {@code &} escapes decoded. */
    private static Map<String, String> attrs(String header) {
        Map<String, String> out = new LinkedHashMap<>();
        int i = 0;
        while (i < header.length()) {
            int eq = header.indexOf('=', i);
            if (eq < 0) {
                break;
            }
            String key = header.substring(i, eq).trim();
            int q = header.indexOf('"', eq);
            int end = q < 0 ? -1 : header.indexOf('"', q + 1);
            if (q < 0 || end < 0) {
                break;
            }
            out.put(key, unescape(header.substring(q + 1, end)));
            i = end + 1;
        }
        return out;
    }

    private static String unescape(String value) {
        return value.replace("&quot;", "\"").replace("&amp;", "&");
    }
}
