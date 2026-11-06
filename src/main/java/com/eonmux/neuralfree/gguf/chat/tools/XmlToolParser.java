package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Formats that write each argument as tagged text:
 *
 * <ul>
 *   <li>{@link Style#FUNCTION} — Qwen3-Coder, Qwen3.5/3.6/3.8, Nemotron 3:
 *       {@code <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>}, the
 *       value's wrapping newlines being template markup; Seed-OSS writes the same inside
 *       {@code <seed:tool_call>} without them. A bare {@code <function=NAME>} without the wrapper is read too, as
 *       llama.cpp and vLLM do for Qwen3-Coder.</li>
 *   <li>{@link Style#INVOKE} — MiniMax-M2: {@code <minimax:tool_call>\n<invoke name="NAME">\n<parameter
 *       name="KEY">VALUE</parameter>\n</invoke>\n</minimax:tool_call>}, several invokes per block.</li>
 *   <li>{@link Style#ATEM} — Muse Spark / Glimmer tool messages: the same invoke layout with {@code atem:} tags,
 *       {@code <atem:function_calls><atem:invoke name="NAME"><atem:parameter name="KEY">VALUE</atem:parameter>
 *       </atem:invoke></atem:function_calls>} (llama.cpp {@code common/parsers/muse-spark.cpp}).</li>
 *   <li>{@link Style#ARG_KEY} — GLM 4.5-4.7: {@code <tool_call>NAME\n<arg_key>KEY</arg_key>\n<arg_value>VALUE</arg_value>\n</tool_call>},
 *       one call per block.</li>
 * </ul>
 *
 * <p>Values are typed by the tool's parameter schema ({@link ToolSchemas}). While a call streams its arguments grow
 * one finished parameter at a time ({@code {"a":1} } is reported as {@code {"a":1}} only once the call closes), so
 * every partial argument text is a prefix of the final one. A parameter left open by the next opener or the
 * function's closer ends there, as vLLM's Qwen3 pattern allows.
 */
final class XmlToolParser extends BlockToolParser {

    /** The tag dialect. */
    enum Style { FUNCTION, INVOKE, ATEM, ARG_KEY }

    private final Style style;
    private final String wrapOpen;
    private final String wrapClose;
    private final boolean trimNewlines;
    private final boolean bareFunction;
    private final ToolSchemas schemas;

    private final String funcOpen;
    private final String funcNameEnd;
    private final String funcClose;
    private final String paramOpen;
    private final String paramNameEnd;
    private final String paramClose;

    /**
     * @param style        the tag dialect
     * @param wrapOpen     the block opener
     * @param wrapClose    the block closer
     * @param trimNewlines whether one leading and one trailing newline of a value are markup
     * @param bareFunction whether a {@code <function=} without the block opener opens a call
     * @param schemas      the request's parameter types
     */
    XmlToolParser(Style style, String wrapOpen, String wrapClose, boolean trimNewlines, boolean bareFunction,
                  ToolSchemas schemas) {
        super(bareFunction ? Arrays.asList(wrapOpen, "<function=") : Arrays.asList(wrapOpen));
        this.style = style;
        this.wrapOpen = wrapOpen;
        this.wrapClose = wrapClose;
        this.trimNewlines = trimNewlines;
        this.bareFunction = bareFunction;
        this.schemas = schemas;
        if (style == Style.INVOKE || style == Style.ATEM) {
            String ns = style == Style.ATEM ? "atem:" : "";
            funcOpen = "<" + ns + "invoke name=\"";
            funcNameEnd = "\">";
            funcClose = "</" + ns + "invoke>";
            paramOpen = "<" + ns + "parameter name=\"";
            paramNameEnd = "\">";
            paramClose = "</" + ns + "parameter>";
        } else if (style == Style.FUNCTION) {
            funcOpen = "<function=";
            funcNameEnd = ">";
            funcClose = "</function>";
            paramOpen = "<parameter=";
            paramNameEnd = ">";
            paramClose = "</parameter>";
        } else {
            funcOpen = null;
            funcNameEnd = null;
            funcClose = null;
            paramOpen = "<arg_key>";
            paramNameEnd = "</arg_key>";
            paramClose = "</arg_value>";
        }
    }

    @Override
    Block readBlock(String text, int at, String opener, boolean partial) {
        if (style == Style.ARG_KEY) {
            return argKeyBlock(text, at, partial);
        }
        List<ToolCall> calls = new ArrayList<>();
        if (!opener.equals(wrapOpen)) {
            Func f = readFunction(text, at, partial);
            if (f.malformed) {
                return Block.malformed(at + funcOpen.length());
            }
            if (f.call != null) {
                calls.add(f.call);
            }
            if (f.end < 0) {
                return Block.incomplete(calls, at);
            }
            int k = JsonScan.skipWs(text, f.end);
            if (text.startsWith(wrapClose, k)) {
                return Block.complete(calls, k + wrapClose.length());
            }
            return Block.complete(calls, f.end);
        }
        int i = at + wrapOpen.length();
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(wrapClose, j)) {
                return Block.complete(calls, j + wrapClose.length());
            }
            if (text.startsWith(funcOpen, j)) {
                Func f = readFunction(text, j, partial);
                if (f.malformed) {
                    return calls.isEmpty() ? Block.malformed(j + funcOpen.length()) : Block.complete(calls, j);
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
            if (j >= text.length() || prefixOf(text, j, funcOpen) || prefixOf(text, j, wrapClose)) {
                return partial ? Block.incomplete(calls, j) : (calls.isEmpty() ? Block.malformed(text.length())
                        : Block.complete(calls, text.length()));
            }
            return calls.isEmpty() ? Block.malformed(j) : Block.complete(calls, j);
        }
    }

    /** One function read: the call (null while unnamed), its end (-1 cut off), or malformed. */
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

    private Func readFunction(String text, int at, boolean partial) {
        int i = at + funcOpen.length();
        int ne = text.indexOf(funcNameEnd, i);
        int nl = text.indexOf('\n', i);
        if (ne < 0 || (nl >= 0 && nl < ne)) {
            return partial && nl < 0 ? new Func(null, -1, false) : new Func(null, -1, true);
        }
        String name = text.substring(i, ne).trim();
        if (name.isEmpty()) {
            return new Func(null, -1, true);
        }
        Params p = readParams(text, ne + funcNameEnd.length(), name, funcClose, partial);
        if (p.end < 0) {
            return new Func(new ToolCall(name, p.args(false), null, false), -1, false);
        }
        return new Func(new ToolCall(name, p.args(true), null, true), p.end, false);
    }

    /** GLM: one call per block, the name up to the first newline or {@code <arg_key>}. */
    private Block argKeyBlock(String text, int at, boolean partial) {
        int i = at + wrapOpen.length();
        int close = text.indexOf(wrapClose, i);
        int key = text.indexOf(paramOpen, i);
        int nl = text.indexOf('\n', i);
        int nameEnd = minPositive(minPositive(key, nl), close);
        List<ToolCall> calls = new ArrayList<>();
        if (nameEnd < 0) {
            return partial ? Block.incomplete(calls, at) : Block.malformed(text.length());
        }
        String name = text.substring(i, nameEnd).trim();
        if (name.isEmpty() || name.contains("<") || name.contains("{")) {
            return Block.malformed(nameEnd);
        }
        Params p = readParams(text, nameEnd, name, wrapClose, partial);
        if (p.end < 0) {
            calls.add(new ToolCall(name, p.args(false), null, false));
            return Block.incomplete(calls, at);
        }
        calls.add(new ToolCall(name, p.args(true), null, true));
        return Block.complete(calls, p.end);
    }

    private static int minPositive(int a, int b) {
        if (a < 0) {
            return b;
        }
        return b < 0 ? a : Math.min(a, b);
    }

    /** The parameters read before a closer: their values in order and where the closer ends (-1 cut off). */
    private static final class Params {
        final Map<String, Object> values = new LinkedHashMap<>();
        int end = -1;

        String args(boolean complete) {
            String obj = ToolSchemas.arguments(values);
            return complete ? obj : obj.substring(0, obj.length() - 1);
        }
    }

    private Params readParams(String text, int from, String function, String closer, boolean partial) {
        Params p = new Params();
        int i = from;
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (text.startsWith(closer, j)) {
                p.end = j + closer.length();
                return p;
            }
            if (!text.startsWith(paramOpen, j)) {
                if (j >= text.length() || prefixOf(text, j, closer) || prefixOf(text, j, paramOpen)) {
                    if (!partial) {
                        p.end = text.length();
                    }
                    return p;
                }
                // stray text between parameters: skip to the next tag
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
            int ks = j + paramOpen.length();
            int ke = text.indexOf(paramNameEnd, ks);
            if (ke < 0) {
                if (!partial) {
                    p.end = text.length();
                }
                return p;
            }
            String key = text.substring(ks, ke).trim();
            int vs = ke + paramNameEnd.length();
            if (style == Style.ARG_KEY) {
                vs = JsonScan.skipWs(text, vs);
                if (!text.startsWith("<arg_value>", vs)) {
                    if (prefixOf(text, vs, "<arg_value>") || vs >= text.length()) {
                        if (!partial) {
                            p.end = text.length();
                        }
                        return p;
                    }
                    i = vs;
                    continue;
                }
                vs += "<arg_value>".length();
            }
            int ve = valueEnd(text, vs, closer);
            if (ve < 0) {
                if (!partial) {
                    p.values.put(key, typed(function, key, text.substring(vs)));
                    p.end = text.length();
                }
                return p;
            }
            p.values.put(key, typed(function, key, text.substring(vs, ve)));
            i = text.startsWith(paramClose, ve) ? ve + paramClose.length() : ve;
        }
    }

    /** Where the value starting at {@code vs} ends: its closer, the next parameter opener or the call closer. */
    private int valueEnd(String text, int vs, String closer) {
        int end = -1;
        for (String t : new String[]{paramClose, paramOpen, closer}) {
            int k = text.indexOf(t, vs);
            if (k >= 0 && (end < 0 || k < end)) {
                end = k;
            }
        }
        return end;
    }

    private Object typed(String function, String key, String raw) {
        String v = raw;
        if (trimNewlines) {
            if (v.startsWith("\n")) {
                v = v.substring(1);
            }
            if (v.endsWith("\n")) {
                v = v.substring(0, v.length() - 1);
            }
        }
        return schemas.value(function, key, v);
    }
}
