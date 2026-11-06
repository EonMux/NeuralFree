package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * Formats that write a call as delimited name, then JSON arguments, inside an optional section:
 *
 * <ul>
 *   <li>DeepSeek V3 / R1: {@code <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\nARGS\n```<｜tool▁call▁end｜>...<｜tool▁calls▁end｜>};</li>
 *   <li>DeepSeek V3.1: {@code <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>ARGS<｜tool▁call▁end｜>...<｜tool▁calls▁end｜>};</li>
 *   <li>Kimi K2: {@code <|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>ARGS<|tool_call_end|>...<|tool_calls_section_end|>},
 *       the {@code functions.NAME:IDX} text being the call id.</li>
 *   <li>PLaMo 3: {@code <|plamo:begin_tool_requests:plamo|><|plamo:begin_tool_request:plamo|><|plamo:begin_tool_name:plamo|>NAME<|plamo:end_tool_name:plamo|><|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>ARGS<|plamo:end_tool_arguments:plamo|><|plamo:end_tool_request:plamo|>...<|plamo:end_tool_requests:plamo|>}.</li>
 * </ul>
 *
 * A call may also appear without its section (llama.cpp's DeepSeek parser accepts both).
 */
final class NameArgsToolParser extends BlockToolParser {

    /** A function name as written and the call id the format carries, or null. */
    static final class Named {
        final String name;
        final String id;

        Named(String name, String id) {
            this.name = name;
            this.id = id;
        }
    }

    private final String sectionOpen;
    private final String sectionClose;
    private final String callOpen;
    private final String typePrefix;
    private final String nameOpen;
    private final String nameEnd;
    private final String argsOpen;
    private final String callClose;
    private final Function<String, Named> naming;

    /**
     * @param sectionOpen  the section opener, or null
     * @param sectionClose the section closer, or null
     * @param callOpen     the call opener
     * @param typePrefix   text skipped after the call opener when present (DeepSeek V3's {@code function<｜tool▁sep｜>}), or null
     * @param nameOpen     the name's opener, skipped after the call opener (PLaMo 3's {@code <|plamo:begin_tool_name:plamo|>}), or null
     * @param nameEnd      the text ending the name
     * @param argsOpen     the arguments' opener, skipped after the name (PLaMo 3's arguments marker chain), or null
     * @param callClose    the call closer
     * @param naming       the name and id of the written name text
     */
    NameArgsToolParser(String sectionOpen, String sectionClose, String callOpen, String typePrefix, String nameOpen,
                       String nameEnd, String argsOpen, String callClose, Function<String, Named> naming) {
        super(sectionOpen == null ? Arrays.asList(callOpen) : Arrays.asList(sectionOpen, callOpen));
        this.sectionOpen = sectionOpen;
        this.sectionClose = sectionClose;
        this.callOpen = callOpen;
        this.typePrefix = typePrefix;
        this.nameOpen = nameOpen;
        this.nameEnd = nameEnd;
        this.argsOpen = argsOpen;
        this.callClose = callClose;
        this.naming = naming;
    }

    /** Names as written, without a format id. */
    static Named plain(String written) {
        return new Named(written, null);
    }

    /** Kimi K2: {@code functions.NAME:IDX} names {@code NAME}, the whole text is the call id. */
    static Named kimi(String written) {
        String name = written;
        if (name.startsWith("functions.")) {
            name = name.substring("functions.".length());
        }
        int colon = name.lastIndexOf(':');
        if (colon > 0 && name.substring(colon + 1).chars().allMatch(Character::isDigit)) {
            name = name.substring(0, colon);
        }
        return new Named(name, written);
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
            if (sectionClose != null && text.startsWith(sectionClose, j)) {
                return Block.complete(calls, j + sectionClose.length());
            }
            if (text.startsWith(callOpen, j)) {
                Call c = readCall(text, j, partial);
                if (c.malformed) {
                    return calls.isEmpty() ? Block.malformed(j + callOpen.length())
                            : Block.complete(calls, j);
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
            if (j >= text.length() || (partial && (prefixOf(text, j, callOpen)
                    || (sectionClose != null && prefixOf(text, j, sectionClose))))) {
                return partial ? Block.incomplete(calls, j) : Block.complete(calls, text.length());
            }
            return calls.isEmpty() ? Block.malformed(j) : Block.complete(calls, j);
        }
    }

    /** One call read: the call (null while its name is unread), its end (-1 cut off) or malformed. */
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
        if (typePrefix != null) {
            if (text.startsWith(typePrefix, i)) {
                i += typePrefix.length();
            } else if (prefixOf(text, i, typePrefix) && partial) {
                return new Call(null, -1, false);
            }
        }
        if (nameOpen != null) {
            if (text.startsWith(nameOpen, i)) {
                i = JsonScan.skipWs(text, i + nameOpen.length());
            } else if (prefixOf(text, i, nameOpen) && partial) {
                return new Call(null, -1, false);
            } else {
                return new Call(null, -1, true);
            }
        }
        int ne = text.indexOf(nameEnd, i);
        if (ne < 0) {
            int stray = text.indexOf(callClose, i);
            if (stray >= 0) {
                return new Call(null, -1, true);
            }
            return partial ? new Call(null, -1, false) : new Call(null, -1, true);
        }
        String written = text.substring(i, ne).trim();
        if (written.isEmpty()) {
            return new Call(null, -1, true);
        }
        Named named = naming.apply(written);
        int a = JsonScan.skipWs(text, ne + nameEnd.length());
        if (argsOpen != null) {
            if (text.startsWith(argsOpen, a)) {
                a = JsonScan.skipWs(text, a + argsOpen.length());
            } else if (partial && prefixOf(text, a, argsOpen)) {
                return new Call(new ToolCall(named.name, "", named.id, false), -1, false);
            } else {
                return new Call(null, -1, true);
            }
        }
        boolean fence = false;
        if (text.startsWith("```", a)) {
            fence = true;
            int f = a + 3;
            if (text.startsWith("json", f)) {
                f += 4;
            } else if (partial && prefixOf(text, f, "json")) {
                return new Call(new ToolCall(named.name, "", named.id, false), -1, false);
            }
            a = JsonScan.skipWs(text, f);
        } else if (partial && a < text.length() && prefixOf(text, a, "```")) {
            return new Call(new ToolCall(named.name, "", named.id, false), -1, false);
        }
        if (a >= text.length()) {
            return partial ? new Call(new ToolCall(named.name, "", named.id, false), -1, false)
                    : new Call(ToolCall.of(named.name, "{}"), text.length(), false);
        }
        if (text.charAt(a) != '{') {
            return new Call(null, -1, true);
        }
        int e = JsonScan.valueEnd(text, a);
        if (e == JsonScan.INVALID) {
            return new Call(null, -1, true);
        }
        if (e < 0) {
            return partial ? new Call(new ToolCall(named.name, rtrim(text.substring(a)), named.id, false), -1, false)
                    : new Call(null, -1, true);
        }
        String args = text.substring(a, e);
        if (JsonScan.parseOrNull(args) == null) {
            return new Call(null, -1, true);
        }
        int k = JsonScan.skipWs(text, e);
        if (fence) {
            if (text.startsWith("```", k)) {
                k = JsonScan.skipWs(text, k + 3);
            } else if (partial && (k >= text.length() || prefixOf(text, k, "```"))) {
                return new Call(new ToolCall(named.name, args, named.id, false), -1, false);
            }
        }
        if (text.startsWith(callClose, k)) {
            return new Call(new ToolCall(named.name, args, named.id, true), k + callClose.length(), false);
        }
        if (k >= text.length() || prefixOf(text, k, callClose)) {
            return partial ? new Call(new ToolCall(named.name, args, named.id, false), -1, false)
                    : new Call(new ToolCall(named.name, args, named.id, true), text.length(), false);
        }
        return new Call(new ToolCall(named.name, args, named.id, true), k, false);
    }
}
