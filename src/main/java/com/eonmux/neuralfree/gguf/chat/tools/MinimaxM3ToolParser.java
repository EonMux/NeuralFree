package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * The MiniMax M3 format: every tool tag carries the namespace token {@code ]<]minimax[>[}, a call block holds the
 * invokes, and each argument is an element whose tag is the parameter name:
 *
 * <pre>{@code
 * ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="f">]<]minimax[>[<city>Paris</city>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>
 * }</pre>
 *
 * <p>An invoke may also start the output without its call block. The name is written with double quotes, single
 * quotes or none. An element's value is its text, typed by the parameter schema (an empty element under an
 * array or object schema is the empty container), or — the form the template writes for a nested object or array —
 * an element with children: an object of its child elements (repeated names collect into a list, vLLM's
 * {@code _convert_object}) or an array of its {@code <item>} children, each typed by the matching nested schema.
 * JSON text written under an array or object schema is read too. While a call streams, an element is reported once
 * its closer is read, so every partial argument text is a prefix of the final one.
 */
final class MinimaxM3ToolParser extends BlockToolParser {

    private static final String NS = "]<]minimax[>[";
    private static final String CALL_OPEN = NS + "<tool_call>";
    private static final String CALL_CLOSE = NS + "</tool_call>";
    private static final String INVOKE_CLOSE = NS + "</invoke>";
    private static final String INVOKE_DQ = NS + "<invoke name=\"";
    private static final String INVOKE_SQ = NS + "<invoke name='";
    private static final String INVOKE_UNQ = NS + "<invoke name=";

    private final ToolSchemas schemas;

    /** @param schemas the request's parameter types */
    MinimaxM3ToolParser(ToolSchemas schemas) {
        super(Arrays.asList(CALL_OPEN, INVOKE_DQ, INVOKE_SQ, INVOKE_UNQ));
        this.schemas = schemas;
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        if (!opener.equals(CALL_OPEN)) {
            Invoke f = readInvoke(text, at, opener, partial);
            if (f.malformed) {
                return Block.malformed(at + opener.length());
            }
            if (f.call != null) {
                calls.add(f.call);
            }
            return f.end < 0 ? Block.incomplete(calls, at) : Block.complete(calls, f.end);
        }
        int i = at + CALL_OPEN.length();
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(CALL_CLOSE, j)) {
                return Block.complete(calls, j + CALL_CLOSE.length());
            }
            String invoke = invokeOpen(text, j);
            if (invoke != null) {
                Invoke f = readInvoke(text, j, invoke, partial);
                if (f.malformed) {
                    return calls.isEmpty() ? Block.malformed(j + invoke.length()) : Block.complete(calls, j);
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
            if (j >= text.length() || prefixOfInvoke(text, j) || prefixOf(text, j, CALL_CLOSE)) {
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

    /** The invoke opener written at {@code at} (longest first), or null. */
    private static String invokeOpen(String text, int at) {
        if (text.startsWith(INVOKE_DQ, at)) {
            return INVOKE_DQ;
        }
        if (text.startsWith(INVOKE_SQ, at)) {
            return INVOKE_SQ;
        }
        if (text.startsWith(INVOKE_UNQ, at)) {
            return INVOKE_UNQ;
        }
        return null;
    }

    private static boolean prefixOfInvoke(String text, int at) {
        return prefixOf(text, at, INVOKE_DQ) || prefixOf(text, at, INVOKE_SQ) || prefixOf(text, at, INVOKE_UNQ);
    }

    /** One invoke read: the call (null while unnamed), its end (-1 cut off), or malformed. */
    private static final class Invoke {
        final ToolCall call;
        final int end;
        final boolean malformed;

        Invoke(ToolCall call, int end, boolean malformed) {
            this.call = call;
            this.end = end;
            this.malformed = malformed;
        }
    }

    private Invoke readInvoke(String text, int at, String opener, boolean partial) {
        int i = at + opener.length();
        String nameEnd = opener.equals(INVOKE_DQ) ? "\">" : opener.equals(INVOKE_SQ) ? "'>" : ">";
        int ne = text.indexOf(nameEnd, i);
        if (ne < 0) {
            return partial ? new Invoke(null, -1, false) : new Invoke(null, -1, true);
        }
        String name = text.substring(i, ne).trim();
        if (name.isEmpty()) {
            return new Invoke(null, -1, true);
        }
        Params p = new Params();
        int end = members(text, ne + nameEnd.length(), name, p, partial);
        if (end == JsonScan.INVALID) {
            return new Invoke(null, -1, true);
        }
        if (end < 0) {
            return new Invoke(new ToolCall(name, p.args(false), null, false), -1, false);
        }
        return new Invoke(new ToolCall(name, p.args(true), null, true), end, false);
    }

    /** The parameters read before the invoke's closer: their values in order and where the closer ends (-1 cut off). */
    private static final class Params {
        final Map<String, Object> values = new LinkedHashMap<>();
        int end = -1;

        String args(boolean complete) {
            String obj = ToolSchemas.arguments(values);
            return complete ? obj : obj.substring(0, obj.length() - 1);
        }

        /** Adds one element's value; a repeated name collects the values into a list (vLLM's {@code _convert_object}). */
        void add(String name, Object value) {
            Object prev = values.get(name);
            if (!values.containsKey(name)) {
                values.put(name, value);
                return;
            }
            List<Object> list = prev instanceof List ? new ArrayList<>((List<?>) prev)
                    : new ArrayList<>(Arrays.asList(prev));
            list.add(value);
            values.put(name, list);
        }
    }

    /**
     * The elements before the invoke's closer, read into {@code p}. Returns where the closer ends, -1 when the text
     * ends first, {@link JsonScan#INVALID} when an element is none.
     */
    private int members(String text, int from, String function, Params p, boolean partial) {
        int i = from;
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(INVOKE_CLOSE, j)) {
                p.end = j + INVOKE_CLOSE.length();
                return p.end;
            }
            if (j >= text.length()) {
                return -1;
            }
            if (!text.startsWith(NS, j) || !text.startsWith("<", j + NS.length())
                    || text.startsWith("</", j + NS.length())) {
                if (partial && prefixOf(text, j, INVOKE_CLOSE)) {
                    return -1;
                }
                if (!partial) {
                    p.end = text.length();
                    return p.end;
                }
                return -1;
            }
            int gt = text.indexOf('>', j + NS.length());
            if (gt < 0) {
                if (!partial) {
                    p.end = text.length();
                    return p.end;
                }
                return -1;
            }
            String name = text.substring(j + NS.length() + 1, gt).trim();
            if (name.isEmpty() || name.contains(" ")) {
                return JsonScan.INVALID;
            }
            int ve = elementEnd(text, gt + 1, name);
            if (ve < 0) {
                // an element the text ends inside: it stands as its text when the output is final
                if (!partial) {
                    p.add(name, value(function, name, text.substring(gt + 1),
                            schemas.schemaOf(function, name)));
                    p.end = text.length();
                    return p.end;
                }
                return -1;
            }
            p.add(name, value(function, name, text.substring(gt + 1,
                    ve - (NS.length() + name.length() + 3)), schemas.schemaOf(function, name)));
            i = ve;
        }
    }

    /**
     * Where the element named {@code name} that opens before {@code from} ends: one past its {@code </name>}
     * closer, matched over same-name nesting, or -1 when the text ends first.
     */
    private static int elementEnd(String text, int from, String name) {
        String open = NS + "<" + name + ">";
        String close = NS + "</" + name + ">";
        int depth = 1;
        int i = from;
        while (i < text.length()) {
            int nextOpen = text.indexOf(open, i);
            int nextClose = text.indexOf(close, i);
            if (nextClose < 0) {
                return -1;
            }
            if (nextOpen >= 0 && nextOpen < nextClose) {
                depth++;
                i = nextOpen + open.length();
                continue;
            }
            depth--;
            i = nextClose + close.length();
            if (depth == 0) {
                return i;
            }
        }
        return -1;
    }

    /**
     * The value of one element under its schema: its typed text when it has no children — an array or object
     * written as JSON text included — a list of its {@code <item>} children for an array, else an object of the
     * children, each typed by the matching property schema.
     */
    private Object value(String function, String name, String raw, Object schema) {
        int first = raw.indexOf(NS + "<");
        if (first < 0) {
            if (raw.trim().isEmpty() && schema instanceof Map) {
                Object type = ((Map<?, ?>) schema).get("type");
                if ("array".equals(type)) {
                    return new ArrayList<>();
                }
                if ("object".equals(type)) {
                    return new LinkedHashMap<>();
                }
            }
            return ToolSchemas.valueOfSchema(schema, raw);
        }
        if (ToolSchemas.arraySchema(schema)) {
            List<Object> items = new ArrayList<>();
            Object itemSchema = ToolSchemas.itemsSchema(schema);
            children(raw, (child, childRaw) -> items.add(value(function, child, childRaw, itemSchema)));
            return items;
        }
        Params p = new Params();
        children(raw, (child, childRaw) -> p.add(child,
                value(function, child, childRaw, ToolSchemas.propertySchema(schema, child))));
        return p.values;
    }

    /** Reports each namespace element written in {@code raw} to {@code sink}, in order. */
    private static void children(String raw, BiConsumer<String, String> sink) {
        int i = raw.indexOf(NS + "<");
        while (i >= 0 && i < raw.length()) {
            if (raw.startsWith(NS + "</", i)) {
                return;
            }
            int gt = raw.indexOf('>', i + NS.length());
            if (gt < 0) {
                return;
            }
            String child = raw.substring(i + NS.length() + 1, gt).trim();
            if (child.isEmpty()) {
                return;
            }
            int ve = elementEnd(raw, gt + 1, child);
            if (ve < 0) {
                return;
            }
            sink.accept(child, raw.substring(gt + 1, ve - (NS.length() + child.length() + 3)));
            i = raw.indexOf(NS + "<", ve);
        }
    }
}
