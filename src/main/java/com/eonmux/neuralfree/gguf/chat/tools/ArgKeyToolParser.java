package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Formats whose calls write the name, then each argument as a key tag, an optional type tag and a value tag, inside
 * a required section of calls:
 *
 * <ul>
 *   <li>HY v3: {@code <tool_calls{S}>}…{@code <tool_call{S}>NAME<tool_sep{S}><arg_key{S}>K</arg_key{S}>
 *       <arg_value{S}>V</arg_value{S}>…</tool_call{S}>…{@code </tool_calls{S}>}, the Hunyuan structural-token
 *       suffix {@code S} (for example {@code :6124c78e}) baked into every marker.</li>
 *   <li>HY v4: the same layout without {@code <tool_sep{S}>} — the name runs to the first key tag.</li>
 *   <li>K2 Horizon: {@code <ifm|tool_calls>}…{@code <ifm|tool_call>}NAME{@code <ifm|arg_key>K</ifm|arg_key>}
 *       [<ifm|arg_type>T</ifm|arg_type>]{@code <ifm|arg_value>V</ifm|arg_value>}…{@code </ifm|tool_call>}…
 *       {@code </ifm|tool_calls>}; a call whose body is a JSON object ({@code name}, {@code arguments}) is read as
 *       that call instead.</li>
 * </ul>
 *
 * <p>A value the type tag or the parameter schema types as string stays the text, any other is read as JSON and
 * keeps the text when it is no JSON value (K2's {@code _coerce_xml_value}). While a call streams, a value is
 * reported once its tag closes, so every partial argument text is a prefix of the final one.
 */
final class ArgKeyToolParser extends BlockToolParser {

    private final String sectionOpen;
    private final String sectionClose;
    private final String callOpen;
    private final String callClose;
    private final String nameSep;
    private final String keyOpen;
    private final String keyClose;
    private final String typeOpen;
    private final String typeClose;
    private final String valueOpen;
    private final String valueClose;
    private final ToolSchemas schemas;

    /**
     * @param sectionOpen  the section opener
     * @param sectionClose the section closer
     * @param callOpen     the call opener
     * @param callClose    the call closer
     * @param nameSep      the text ending the name, or null when the name runs to the first key tag
     * @param keyOpen      the argument key's opener
     * @param keyClose     the argument key's closer
     * @param typeOpen     the optional argument type's opener, or null
     * @param typeClose    the optional argument type's closer, or null
     * @param valueOpen    the argument value's opener
     * @param valueClose   the argument value's closer
     * @param schemas      the request's parameter types
     */
    ArgKeyToolParser(String sectionOpen, String sectionClose, String callOpen, String callClose, String nameSep,
                     String keyOpen, String keyClose, String typeOpen, String typeClose, String valueOpen,
                     String valueClose, ToolSchemas schemas) {
        super(Arrays.asList(sectionOpen, callOpen));
        this.sectionOpen = sectionOpen;
        this.sectionClose = sectionClose;
        this.callOpen = callOpen;
        this.callClose = callClose;
        this.nameSep = nameSep;
        this.keyOpen = keyOpen;
        this.keyClose = keyClose;
        this.typeOpen = typeOpen;
        this.typeClose = typeClose;
        this.valueOpen = valueOpen;
        this.valueClose = valueClose;
        this.schemas = schemas;
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        if (opener.equals(callOpen)) {
            Call c = readCall(text, at, partial);
            if (c.malformed) {
                return Block.malformed(at + callOpen.length());
            }
            if (c.call != null) {
                calls.add(c.call);
            }
            return c.end < 0 ? Block.incomplete(calls, at) : Block.complete(calls, c.end);
        }
        int i = at + sectionOpen.length();
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(sectionClose, j)) {
                return Block.complete(calls, j + sectionClose.length());
            }
            if (text.startsWith(callOpen, j)) {
                Call c = readCall(text, j, partial);
                if (c.malformed) {
                    return calls.isEmpty() ? Block.malformed(j + callOpen.length()) : Block.complete(calls, j);
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
            if (j >= text.length() || prefixOf(text, j, callOpen) || prefixOf(text, j, sectionClose)) {
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
        int i = at + callOpen.length();
        int body = JsonScan.skipWs(text, i);
        if (body < text.length() && text.charAt(body) == '{') {
            return jsonCall(text, body, partial);
        }
        int ne = nameEnd(text, i);
        if (ne < 0) {
            int close = text.indexOf(callClose, i);
            if (close < 0) {
                return partial ? new Call(null, -1, false) : new Call(null, -1, true);
            }
            if (nameSep != null) {
                // the separator ends every name; a call without one is no call
                return new Call(null, -1, true);
            }
            String name = text.substring(i, close).trim();
            if (name.isEmpty()) {
                return new Call(null, -1, true);
            }
            int after = close + callClose.length();
            return new Call(new ToolCall(name, "{}", null, true), after, false);
        }
        String name = text.substring(i, ne).trim();
        if (name.isEmpty()) {
            return new Call(null, -1, true);
        }
        Params p = readParams(text, ne, name, partial);
        if (p.end < 0) {
            return new Call(new ToolCall(name, p.args(false), null, false), -1, false);
        }
        return new Call(new ToolCall(name, p.args(true), null, true), p.end, false);
    }

    /** A K2 call written as a JSON object: the name, the arguments and the id come from the object. */
    private static Call jsonCall(String text, int at, boolean partial) {
        JsonCalls.ObjectCall o = JsonCalls.object(text, at);
        if (o.invalid) {
            return new Call(null, -1, true);
        }
        if (o.end < 0) {
            return o.call == null ? new Call(null, -1, false)
                    : new Call(new ToolCall(o.call.name(), o.call.arguments(), o.call.id(), false), -1, false);
        }
        int k = JsonScan.skipWs(text, o.end);
        return new Call(o.call == null ? null : o.call, k, o.call == null);
    }

    /** Where the name ends: the name separator, the first key tag, or -1. */
    private int nameEnd(String text, int from) {
        int key = text.indexOf(keyOpen, from);
        if (nameSep != null) {
            int sep = text.indexOf(nameSep, from);
            return minPositive(sep, key);
        }
        return key;
    }

    /** The parameters read before the call's closer: their values in order and where the closer ends (-1 cut off). */
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
            if (text.startsWith(callClose, j)) {
                p.end = j + callClose.length();
                return p;
            }
            if (!text.startsWith(keyOpen, j)) {
                if (j >= text.length() || prefixOf(text, j, callClose) || prefixOf(text, j, keyOpen)) {
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
            int ks = j + keyOpen.length();
            int ke = text.indexOf(keyClose, ks);
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
            int vs = JsonScan.skipWs(text, ke + keyClose.length());
            String type = null;
            if (typeOpen != null && text.startsWith(typeOpen, vs)) {
                int te = text.indexOf(typeClose, vs + typeOpen.length());
                if (te < 0) {
                    if (!partial) {
                        p.end = text.length();
                    }
                    return p;
                }
                type = text.substring(vs + typeOpen.length(), te).trim();
                vs = JsonScan.skipWs(text, te + typeClose.length());
            } else if (typeOpen != null && partial && prefixOf(text, vs, typeOpen)) {
                return p;
            }
            if (!text.startsWith(valueOpen, vs)) {
                if (partial && prefixOf(text, vs, valueOpen)) {
                    return p;
                }
                return new Params();
            }
            vs += valueOpen.length();
            int ve = text.indexOf(valueClose, vs);
            if (ve < 0) {
                if (!partial) {
                    p.values.put(key, typed(function, key, text.substring(vs), type));
                    p.end = text.length();
                }
                return p;
            }
            p.values.put(key, typed(function, key, text.substring(vs, ve), type));
            i = ve + valueClose.length();
        }
    }

    private static int minPositive(int a, int b) {
        if (a < 0) {
            return b;
        }
        return b < 0 ? a : Math.min(a, b);
    }

    /**
     * The JSON value of the written text: an explicit type tag decides (a string or {@code any} keeps the text,
     * another type reads JSON), else the parameter schema does.
     */
    private Object typed(String function, String key, String raw, String type) {
        if (type != null) {
            String base = type.split("\\[", 2)[0].trim().toLowerCase(Locale.ROOT);
            if (base.equals("string") || base.equals("any")) {
                return raw;
            }
            Object parsed = JsonScan.parseOrNull(raw.trim());
            return parsed == null ? raw : parsed;
        }
        return schemas.value(function, key, raw);
    }
}
