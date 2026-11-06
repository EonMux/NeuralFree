package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The Step 3 format: StepTML invokes inside a section of calls, each call carrying the {@code function} type before
 * its invoke:
 *
 * <pre>{@code
 * <｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="f">
 * <steptml:parameter name="city">Paris</steptml:parameter>
 * </steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>
 * }</pre>
 *
 * <p>A parameter's value is its text stripped of surrounding whitespace, typed by the parameter schema. While a
 * call streams, a parameter is reported once its tag closes, so every partial argument text is a prefix of the
 * final one.
 */
final class SteptmlToolParser extends BlockToolParser {

    private static final String SECTION_OPEN = "<｜tool_calls_begin｜>";
    private static final String SECTION_CLOSE = "<｜tool_calls_end｜>";
    private static final String CALL_OPEN = "<｜tool_call_begin｜>";
    private static final String CALL_PREFIX = "function<｜tool_sep｜>";
    private static final String CALL_CLOSE = "<｜tool_call_end｜>";
    private static final String INVOKE_OPEN = "<steptml:invoke name=\"";
    private static final String INVOKE_NAME_END = "\">";
    private static final String INVOKE_CLOSE = "</steptml:invoke>";
    private static final String PARAM_OPEN = "<steptml:parameter name=\"";
    private static final String PARAM_NAME_END = "\">";
    private static final String PARAM_CLOSE = "</steptml:parameter>";

    private final ToolSchemas schemas;

    /** @param schemas the request's parameter types */
    SteptmlToolParser(ToolSchemas schemas) {
        super(Arrays.asList(SECTION_OPEN));
        this.schemas = schemas;
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        int i = at + SECTION_OPEN.length();
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(SECTION_CLOSE, j)) {
                return Block.complete(calls, j + SECTION_CLOSE.length());
            }
            if (text.startsWith(CALL_OPEN, j)) {
                Call c = readCall(text, j, partial);
                if (c.malformed) {
                    return calls.isEmpty() ? Block.malformed(j + CALL_OPEN.length()) : Block.complete(calls, j);
                }
                if (c.call != null) {
                    calls.add(c.call);
                }
                if (c.end < 0) {
                    return Block.incomplete(calls, j);
                }
                i = c.end;
                continue;
            }
            if (j >= text.length() || prefixOf(text, j, CALL_OPEN) || prefixOf(text, j, SECTION_CLOSE)) {
                return partial ? Block.incomplete(calls, j) : (calls.isEmpty()
                        ? Block.malformed(text.length()) : Block.complete(calls, text.length()));
            }
            // text between calls is markup, not content
            int next = text.indexOf('<', j + 1);
            if (next < 0) {
                return partial ? Block.incomplete(calls, j) : Block.complete(calls, text.length());
            }
            i = next;
        }
    }

    /** One call read: the call (null while unread), its end (-1 cut off), or malformed. */
    private static final class Call {
        final ToolCall call;
        final int end;
        final boolean malformed;

        Call(ToolCall call, int end, boolean malformed) {
            this.call = call;
            this.end = end;
            this.malformed = malformed;
        }
    }

    private Call readCall(String text, int at, boolean partial) {
        int i = at + CALL_OPEN.length();
        if (text.startsWith(CALL_PREFIX, i)) {
            i += CALL_PREFIX.length();
        } else if (prefixOf(text, i, CALL_PREFIX) && partial) {
            return new Call(null, -1, false);
        }
        if (!text.startsWith(INVOKE_OPEN, i)) {
            if (partial && prefixOf(text, i, INVOKE_OPEN)) {
                return new Call(null, -1, false);
            }
            return new Call(null, -1, true);
        }
        int ne = text.indexOf(INVOKE_NAME_END, i + INVOKE_OPEN.length());
        if (ne < 0) {
            int stray = text.indexOf(CALL_CLOSE, i);
            if (stray >= 0) {
                return new Call(null, -1, true);
            }
            return partial ? new Call(null, -1, false) : new Call(null, -1, true);
        }
        String name = text.substring(i + INVOKE_OPEN.length(), ne).trim();
        if (name.isEmpty()) {
            return new Call(null, -1, true);
        }
        Params p = readParams(text, ne + INVOKE_NAME_END.length(), name, partial);
        if (p.end < 0) {
            return new Call(new ToolCall(name, p.args(false), null, false), -1, false);
        }
        ToolCall call = new ToolCall(name, p.args(true), null, true);
        int k = JsonScan.skipWs(text, p.end);
        if (text.startsWith(INVOKE_CLOSE, k)) {
            k = JsonScan.skipWs(text, k + INVOKE_CLOSE.length());
            if (text.startsWith(CALL_CLOSE, k)) {
                return new Call(call, k + CALL_CLOSE.length(), false);
            }
            if (partial && (k >= text.length() || prefixOf(text, k, CALL_CLOSE))) {
                return new Call(call, -1, false);
            }
            return new Call(call, k, false);
        }
        if (partial && (k >= text.length() || prefixOf(text, k, INVOKE_CLOSE))) {
            return new Call(call, -1, false);
        }
        return new Call(call, k >= text.length() ? text.length() : k, false);
    }

    /** The parameters read before the invoke's closer: their values in order and where the closer ends (-1 cut off). */
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
            if (text.startsWith(INVOKE_CLOSE, j)) {
                p.end = j + INVOKE_CLOSE.length();
                return p;
            }
            if (!text.startsWith(PARAM_OPEN, j)) {
                if (j >= text.length() || prefixOf(text, j, INVOKE_CLOSE) || prefixOf(text, j, PARAM_OPEN)) {
                    if (!partial) {
                        p.end = text.length();
                    }
                    return p;
                }
                int next = text.indexOf('<', j + 1);
                if (next < 0) {
                    if (!partial) {
                        p.end = text.length();
                    }
                    return p;
                }
                i = next;
                continue;
            }
            int ks = j + PARAM_OPEN.length();
            int ke = text.indexOf(PARAM_NAME_END, ks);
            if (ke < 0) {
                if (!partial) {
                    p.end = text.length();
                }
                return p;
            }
            String key = text.substring(ks, ke).trim();
            if (key.isEmpty()) {
                return new Params();
            }
            int vs = ke + PARAM_NAME_END.length();
            int ve = text.indexOf(PARAM_CLOSE, vs);
            if (ve < 0) {
                if (!partial) {
                    p.values.put(key, typed(function, key, text.substring(vs)));
                    p.end = text.length();
                }
                return p;
            }
            p.values.put(key, typed(function, key, text.substring(vs, ve)));
            i = ve + PARAM_CLOSE.length();
        }
    }

    /** The JSON value of the written text: the text stripped of surrounding whitespace, typed by the schema. */
    private Object typed(String function, String key, String raw) {
        return schemas.value(function, key, raw.trim());
    }
}
