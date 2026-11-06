package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Tool calls written as JSON objects — {@code {"name": ..., "arguments": {...}}}, with {@code parameters} for
 * {@code arguments} (Llama 3), {@code args} (Inkling), {@code tool_name}/{@code parameters}/{@code tool_call_id}
 * (Cohere Command) and the OpenAI {@code {"type": "function", "function": {...}}} nesting — and JSON arrays of
 * them. The first key of each role that the object carries wins. The arguments keep their source text, so a call
 * cut off inside its arguments reports a prefix of the text the finished call carries; arguments written as a JSON
 * string (a stringified object) are read once the string closes.
 */
final class JsonCalls {

    private static final String[] NAME_KEYS = {"name", "tool_name"};
    private static final String[] ARGUMENT_KEYS = {"arguments", "parameters", "args"};
    private static final String[] ID_KEYS = {"id", "tool_call_id"};

    private JsonCalls() {
    }

    /** One object read: the call (null when it names no function yet) and where the object ends (-1 cut off). */
    static final class ObjectCall {
        final ToolCall call;
        final int end;
        final boolean invalid;

        ObjectCall(ToolCall call, int end, boolean invalid) {
            this.call = call;
            this.end = end;
            this.invalid = invalid;
        }
    }

    /** The call of the object at the brace {@code s}; {@link ObjectCall#invalid} when it is no call object. */
    static ObjectCall object(String text, int s) {
        int e = JsonScan.valueEnd(text, s);
        if (e == JsonScan.INVALID) {
            return new ObjectCall(null, -1, true);
        }
        List<JsonScan.Member> members = JsonScan.members(text, s);
        if (members == null) {
            return new ObjectCall(null, e, true);
        }
        if (e >= 0) {
            Object parsed = JsonScan.parseOrNull(text.substring(s, e));
            if (!(parsed instanceof Map)) {
                return new ObjectCall(null, e, true);
            }
        }
        JsonScan.Member function = find(members, "function");
        if (function != null && function.valueStart < text.length() && text.charAt(function.valueStart) == '{') {
            ObjectCall inner = object(text, function.valueStart);
            if (inner.invalid) {
                return new ObjectCall(null, e, true);
            }
            ToolCall c = inner.call;
            String id = idOf(text, members);
            if (c != null && id != null) {
                c = new ToolCall(c.name(), c.arguments(), id, c.complete() && e >= 0);
            } else if (c != null && e < 0 && c.complete()) {
                c = new ToolCall(c.name(), c.arguments(), c.id(), false);
            }
            return new ObjectCall(c, e, false);
        }
        String name = null;
        for (String k : NAME_KEYS) {
            JsonScan.Member m = find(members, k);
            if (m != null && m.complete()) {
                Object v = JsonScan.parseOrNull(text.substring(m.valueStart, m.valueEnd));
                if (v instanceof String && !((String) v).isEmpty()) {
                    name = (String) v;
                }
            }
        }
        if (name == null) {
            return new ObjectCall(null, e, e >= 0);
        }
        String args = arguments(text, members, e >= 0);
        if (args == null) {
            return new ObjectCall(null, e, true);
        }
        return new ObjectCall(new ToolCall(name, args, idOf(text, members), e >= 0), e, false);
    }

    /**
     * The arguments text: the value's source for an object, the decoded text of a string, {@code {}} for a
     * complete object without arguments; while cut off, the source so far without trailing whitespace (a string
     * value is held). Null when a complete value is no object.
     */
    private static String arguments(String text, List<JsonScan.Member> members, boolean objectComplete) {
        JsonScan.Member a = null;
        for (String k : ARGUMENT_KEYS) {
            a = find(members, k);
            if (a != null) {
                break;
            }
        }
        if (a == null) {
            return objectComplete ? "{}" : "";
        }
        boolean isString = a.valueStart < text.length() && text.charAt(a.valueStart) == '"';
        if (!a.complete()) {
            return isString ? "" : BlockToolParser.rtrim(text.substring(a.valueStart));
        }
        String source = text.substring(a.valueStart, a.valueEnd);
        if (isString) {
            Object decoded = JsonScan.parseOrNull(source);
            if (!(decoded instanceof String)) {
                return null;
            }
            String inner = ((String) decoded).trim();
            return inner.isEmpty() ? "{}" : (JsonScan.parseOrNull(inner) instanceof Map ? inner : null);
        }
        return source.startsWith("{") ? source : null;
    }

    private static String idOf(String text, List<JsonScan.Member> members) {
        JsonScan.Member m = null;
        for (String k : ID_KEYS) {
            m = find(members, k);
            if (m != null) {
                break;
            }
        }
        if (m == null || !m.complete()) {
            return null;
        }
        Object v = JsonScan.parseOrNull(text.substring(m.valueStart, m.valueEnd));
        return v instanceof String && !((String) v).isEmpty() ? (String) v : null;
    }

    private static JsonScan.Member find(List<JsonScan.Member> members, String key) {
        for (JsonScan.Member m : members) {
            if (m.key.equals(key)) {
                return m;
            }
        }
        return null;
    }

    /**
     * A run of calls: the calls, where the run ends (-1 cut off), whether it is no call run, and where the object
     * the text ends inside starts (-1 when none).
     */
    static final class Run {
        final List<ToolCall> calls;
        final int end;
        final boolean invalid;
        final int rest;

        Run(List<ToolCall> calls, int end, boolean invalid) {
            this(calls, end, invalid, -1);
        }

        Run(List<ToolCall> calls, int end, boolean invalid, int rest) {
            this.calls = calls;
            this.end = end;
            this.invalid = invalid;
            this.rest = rest;
        }
    }

    /** The calls of the array whose bracket is at {@code i}. */
    static Run array(String text, int i) {
        List<ToolCall> calls = new ArrayList<>();
        int j = JsonScan.skipWs(text, i + 1);
        if (j < text.length() && text.charAt(j) == ']') {
            return new Run(calls, j + 1, false);
        }
        while (j < text.length()) {
            if (text.charAt(j) != '{') {
                return new Run(calls, -1, true);
            }
            ObjectCall o = object(text, j);
            if (o.invalid) {
                return new Run(calls, -1, true);
            }
            if (o.call != null) {
                calls.add(o.call);
            }
            if (o.end < 0) {
                return new Run(calls, -1, false, j);
            }
            j = JsonScan.skipWs(text, o.end);
            if (j >= text.length()) {
                return new Run(calls, -1, false);
            }
            char c = text.charAt(j);
            if (c == ']') {
                return new Run(calls, j + 1, false);
            }
            if (c != ',') {
                return new Run(calls, -1, true);
            }
            j = JsonScan.skipWs(text, j + 1);
        }
        return new Run(calls, -1, false);
    }

    /**
     * The calls of consecutive objects from the brace {@code i}, separated by whitespace, {@code ;} or {@code ,}
     * (Llama 3 writes {@code ;} between calls); the run ends before the first text that opens no object, and a
     * trailing separator belongs to it. While {@code partial}, a run the text ends after is cut off: another object
     * may still follow.
     */
    static Run objects(String text, int i, boolean several, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        int j = i;
        while (true) {
            ObjectCall o = object(text, j);
            if (o.invalid) {
                return new Run(calls, -1, true);
            }
            if (o.call != null) {
                calls.add(o.call);
            }
            if (o.end < 0) {
                return new Run(calls, -1, false, j);
            }
            if (!several) {
                return new Run(calls, o.end, false);
            }
            int k = JsonScan.skipWs(text, o.end);
            boolean separated = false;
            if (k < text.length() && (text.charAt(k) == ';' || text.charAt(k) == ',')) {
                k = JsonScan.skipWs(text, k + 1);
                separated = true;
            }
            if (k >= text.length()) {
                if (partial) {
                    return new Run(calls, -1, false);
                }
                return new Run(calls, separated ? k : o.end, false);
            }
            if (text.charAt(k) != '{') {
                return new Run(calls, separated ? k : o.end, false);
            }
            j = k;
        }
    }
}
