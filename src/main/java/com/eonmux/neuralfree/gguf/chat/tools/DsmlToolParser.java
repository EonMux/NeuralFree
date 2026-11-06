package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The DSML formats of DeepSeek V3.2, V4 and V4.1: invokes with typed parameter tags inside an optional section,
 * the section opener naming the variant — V3.2 {@code <｜DSML｜function_calls>}, V4 {@code <｜DSML｜tool_calls>}
 * (also written {@code <｜DSML｜toolcalls>} and {@code <｜DSML｜tool>}), V4.1 the spaced {@code <｜DSML｜ calls>}:
 *
 * <pre>{@code
 * <｜DSML｜invoke name="f">
 * <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
 * <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
 * </｜DSML｜invoke>
 * }</pre>
 *
 * <p>An invoke may also start the output without its section. The {@code string} attribute types the value: the
 * text stays verbatim when it reads {@code true}, and is otherwise read as JSON, falling back to the text
 * (vLLM's {@code _dsml_arg_converter}); a missing attribute behaves like {@code string="false"}. A parameter's
 * value ends at its closer or at the next parameter tag (vLLM's value lookahead). While a call streams, a value is
 * reported once its parameter closes, so every partial argument text is a prefix of the final one.
 */
final class DsmlToolParser extends BlockToolParser {

    private final String sectionClose;
    private final String invokeOpen;
    private final String invokeNameEnd;
    private final String invokeClose;
    private final String paramOpen;
    private final String paramClose;

    /**
     * @param sectionOpeners the section openers (V4's spellings included)
     * @param sectionClose   the section closer
     * @param invokeOpen     the invoke opener, up to the name attribute's opening quote
     * @param invokeNameEnd  the text ending the name
     * @param invokeClose    the invoke closer
     * @param paramOpen      the parameter opener, before the attributes
     * @param paramClose     the parameter closer
     */
    DsmlToolParser(List<String> sectionOpeners, String sectionClose, String invokeOpen, String invokeNameEnd,
                   String invokeClose, String paramOpen, String paramClose) {
        super(join(sectionOpeners, invokeOpen));
        this.sectionClose = sectionClose;
        this.invokeOpen = invokeOpen;
        this.invokeNameEnd = invokeNameEnd;
        this.invokeClose = invokeClose;
        this.paramOpen = paramOpen;
        this.paramClose = paramClose;
    }

    private static List<String> join(List<String> sections, String invokeOpen) {
        List<String> out = new ArrayList<>(sections);
        out.add(invokeOpen);
        return out;
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        if (opener.equals(invokeOpen)) {
            Func f = readInvoke(text, at, partial);
            if (f.malformed) {
                return Block.malformed(at + invokeOpen.length());
            }
            if (f.call != null) {
                calls.add(f.call);
            }
            return f.end < 0 ? Block.incomplete(calls, at) : Block.complete(calls, f.end);
        }
        int i = at + opener.length();
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(sectionClose, j)) {
                return Block.complete(calls, j + sectionClose.length());
            }
            if (text.startsWith(invokeOpen, j)) {
                Func f = readInvoke(text, j, partial);
                if (f.malformed) {
                    return calls.isEmpty() ? Block.malformed(j + invokeOpen.length()) : Block.complete(calls, j);
                }
                if (f.call != null) {
                    calls.add(f.call);
                }
                if (f.end < 0) {
                    return Block.incomplete(calls, j);
                }
                i = f.end;
                continue;
            }
            if (j >= text.length() || prefixOf(text, j, invokeOpen) || prefixOf(text, j, sectionClose)) {
                return partial ? Block.incomplete(calls, j) : (calls.isEmpty()
                        ? Block.malformed(text.length()) : Block.complete(calls, text.length()));
            }
            // text between invokes is markup, not content
            int next = text.indexOf('<', j + 1);
            if (next < 0) {
                return partial ? Block.incomplete(calls, j) : Block.complete(calls, text.length());
            }
            i = next;
        }
    }

    /** One invoke read: the call (null while unnamed), its end (-1 cut off), or malformed. */
    private static final class Func {
        final ToolCall call;
        final int end;
        final boolean malformed;

        Func(ToolCall call, int end, boolean malformed) {
            this.call = call;
            this.end = end;
            this.malformed = malformed;
        }
    }

    private Func readInvoke(String text, int at, boolean partial) {
        int i = at + invokeOpen.length();
        int ne = text.indexOf(invokeNameEnd, i);
        if (ne < 0) {
            return partial ? new Func(null, -1, false) : new Func(null, -1, true);
        }
        String name = text.substring(i, ne).trim();
        if (name.isEmpty()) {
            return new Func(null, -1, true);
        }
        Params p = readParams(text, ne + invokeNameEnd.length(), partial);
        if (p.end < 0) {
            return new Func(new ToolCall(name, p.args(false), null, false), -1, false);
        }
        return new Func(new ToolCall(name, p.args(true), null, true), p.end, false);
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

    private Params readParams(String text, int from, boolean partial) {
        Params p = new Params();
        int i = from;
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(invokeClose, j)) {
                p.end = j + invokeClose.length();
                return p;
            }
            if (!text.startsWith(paramOpen, j)) {
                if (j >= text.length() || prefixOf(text, j, invokeClose) || prefixOf(text, j, paramOpen)) {
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
            int head = paramTagEnd(text, j);
            if (head < 0) {
                if (!partial) {
                    p.end = text.length();
                }
                return p;
            }
            String key = attr(text, j, head, "name");
            if (key == null || key.isEmpty()) {
                return new Params();
            }
            String type = attr(text, j, head, "string");
            boolean stringValued = type != null && "true".equals(type.trim());
            int vs = head;
            int ve = valueEnd(text, vs);
            if (ve < 0) {
                if (!partial) {
                    // a value the output ends inside stands as its text
                    p.values.put(key, typed(text.substring(vs), stringValued));
                    p.end = text.length();
                }
                return p;
            }
            p.values.put(key, typed(text.substring(vs, ve), stringValued));
            i = text.startsWith(paramClose, ve) ? ve + paramClose.length() : ve;
        }
    }

    /** The index just past the {@code >} that ends the parameter tag opening at {@code at}, or -1. */
    private int paramTagEnd(String text, int at) {
        int end = text.indexOf('>', at + paramOpen.length());
        return end < 0 ? -1 : end + 1;
    }

    /** The value of attribute {@code name} in the tag spanning {@code [at, end)}, or null. */
    private static String attr(String text, int at, int end, String name) {
        String tag = text.substring(at, end);
        int k = tag.indexOf(name + "=\"");
        if (k < 0) {
            return null;
        }
        int vs = k + name.length() + 2;
        int ve = tag.indexOf('"', vs);
        return ve < 0 ? null : tag.substring(vs, ve);
    }

    /** Where the value starting at {@code vs} ends: its closer or the next parameter tag. */
    private int valueEnd(String text, int vs) {
        int close = text.indexOf(paramClose, vs);
        int next = text.indexOf(paramOpen, vs);
        if (close < 0) {
            return next < 0 ? -1 : next;
        }
        return next >= 0 && next < close ? next : close;
    }

    /** The JSON value of the written text: verbatim when the tag says {@code string="true"}, else its JSON or the text. */
    private static Object typed(String raw, boolean stringValued) {
        if (stringValued) {
            return raw;
        }
        Object parsed = JsonScan.parseOrNull(raw.trim());
        return parsed == null ? raw : parsed;
    }
}
