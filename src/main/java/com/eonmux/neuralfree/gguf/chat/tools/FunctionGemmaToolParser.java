package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The FunctionGemma format: a call is the function name and {@code <escape>}-delimited argument values between two
 * special tokens:
 *
 * <pre>{@code
 * <start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>
 * }</pre>
 *
 * <p>Several calls follow one another. A value is read as JSON, keeping the text when it is no JSON value; while a
 * call streams, each value is reported once its second {@code <escape>} closes, so every partial argument text is
 * a prefix of the final one. A call the output ends inside of stands with the arguments read so far (vLLM's second
 * pattern accepts it).
 */
final class FunctionGemmaToolParser extends BlockToolParser {

    private static final String OPEN = "<start_function_call>";
    private static final String CLOSE = "<end_function_call>";
    private static final String ESCAPE = "<escape>";

    private final ToolSchemas schemas;

    /** @param schemas the request's parameter types */
    FunctionGemmaToolParser(ToolSchemas schemas) {
        super(Arrays.asList(OPEN));
        this.schemas = schemas;
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        int i = at + OPEN.length();
        if (!text.startsWith("call:", i)) {
            return partial && prefixOf(text, i, "call:") ? Block.incomplete(new ArrayList<>(), at)
                    : Block.malformed(i + 1);
        }
        i += "call:".length();
        int brace = text.indexOf('{', i);
        int close = text.indexOf(CLOSE, i);
        if (brace < 0) {
            if (partial && close < 0) {
                return Block.incomplete(new ArrayList<>(), at);
            }
            return Block.malformed(Math.max(i + 1, at + OPEN.length()));
        }
        String name = text.substring(i, brace).trim();
        if (name.isEmpty()) {
            return Block.malformed(i + 1);
        }
        Params p = readParams(text, brace + 1, name, partial);
        if (p.end < 0) {
            if (partial) {
                List<ToolCall> calls = new ArrayList<>();
                calls.add(new ToolCall(name, p.args(false), null, false));
                return Block.incomplete(calls, at);
            }
            // the output ends inside the call: it stands with the arguments read so far
            List<ToolCall> calls = new ArrayList<>();
            calls.add(new ToolCall(name, p.args(true), null, true));
            int k = text.indexOf(CLOSE, brace);
            return Block.complete(calls, k < 0 ? text.length() : k + CLOSE.length());
        }
        List<ToolCall> calls = new ArrayList<>();
        calls.add(new ToolCall(name, p.args(true), null, true));
        int k = JsonScan.skipWs(text, p.end);
        if (text.startsWith(CLOSE, k)) {
            return Block.complete(calls, k + CLOSE.length());
        }
        if (partial && (k >= text.length() || prefixOf(text, k, CLOSE))) {
            return Block.incomplete(calls, at);
        }
        return Block.complete(calls, k >= text.length() ? text.length() : k);
    }

    /** The parameters read before the call's closing brace: their values in order and where it ends (-1 cut off). */
    private static final class Params {
        final Map<String, Object> values = new LinkedHashMap<>();
        int end = -1;

        String args(boolean complete) {
            String obj = ToolSchemas.arguments(values);
            return complete ? obj : obj.substring(0, obj.length() - 1);
        }
    }

    private Params readParams(String text, int from, String function, boolean partial) {
        Params p = new Params();
        int i = from;
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (j < text.length() && text.charAt(j) == '}') {
                p.end = j + 1;
                return p;
            }
            if (j >= text.length()) {
                return p;
            }
            int colon = text.indexOf(':', j);
            int brace = text.indexOf('}', j);
            if (colon < 0 || (brace >= 0 && brace < colon)) {
                return p;
            }
            String key = text.substring(j, colon).trim();
            if (key.isEmpty()) {
                return p;
            }
            int vs = JsonScan.skipWs(text, colon + 1);
            if (!text.startsWith(ESCAPE, vs)) {
                if (partial && prefixOf(text, vs, ESCAPE)) {
                    return p;
                }
                return p;
            }
            int ve = text.indexOf(ESCAPE, vs + ESCAPE.length());
            if (ve < 0) {
                if (!partial) {
                    p.values.put(key, schemas.value(function, key, text.substring(vs + ESCAPE.length())));
                    p.end = text.length();
                }
                return p;
            }
            p.values.put(key, schemas.value(function, key, text.substring(vs + ESCAPE.length(), ve)));
            i = ve + ESCAPE.length();
            int k = JsonScan.skipWs(text, i);
            if (k < text.length() && text.charAt(k) == ',') {
                i = k + 1;
            }
        }
    }
}
