package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Calls written as Python function calls with keyword arguments, {@code [get_weather(city="Paris", days=3)]}
 * (vLLM's {@code pythonic} and {@code llama4_pythonic} parsers): a bare list opening the output, or a list inside
 * an opener and closer — Llama 4's {@code <|python_start|>...<|python_end|>}, LFM2's
 * {@code <|tool_call_start|>[...]<|tool_call_end|>} — or, without brackets, one call per line inside OLMo 3's
 * {@code <function_calls>...</function_calls>}.
 *
 * <p>Argument values are Python literals (strings, numbers, {@code True}/{@code False}/{@code None}, lists, tuples
 * and dicts), converted to JSON as vLLM's {@code ast.literal_eval} route does. While streaming, arguments grow one
 * finished keyword at a time, so each partial argument text is a prefix of the final one.
 */
final class PythonicToolParser extends BlockToolParser {

    private final String closer;
    private final boolean bracketed;
    private final boolean bare;

    /**
     * @param opener    the opener, or null for the bare list only
     * @param closer    the closer, or null
     * @param bracketed whether the calls form a {@code [...]} list (else one call per line)
     * @param bare      whether a bare list may open the output without the opener
     */
    PythonicToolParser(String opener, String closer, boolean bracketed, boolean bare) {
        super(opener == null ? Collections.<String>emptyList() : Collections.singletonList(opener));
        if (opener == null && !bare) {
            throw new IllegalArgumentException("a pythonic format without an opener must accept the bare list");
        }
        this.closer = closer;
        this.bracketed = bracketed;
        this.bare = bare;
    }

    @Override
    int bareStart(String text, boolean partial) {
        if (!bare) {
            return -1;
        }
        int i = JsonScan.skipWs(text, 0);
        if (i >= text.length()) {
            return partial ? UNDECIDED : -1;
        }
        if (text.charAt(i) != '[') {
            return -1;
        }
        int j = JsonScan.skipWs(text, i + 1);
        int k = j;
        while (k < text.length() && isIdent(text.charAt(k), k == j)) {
            k++;
        }
        if (k >= text.length()) {
            return partial ? UNDECIDED : -1;
        }
        return k > j && text.charAt(k) == '(' ? i : -1;
    }

    @Override
    Block readBlock(String text, int at, String opener, boolean partial) {
        int i = JsonScan.skipWs(text, at + opener.length());
        List<ToolCall> calls = new ArrayList<>();
        if (bracketed) {
            if (i >= text.length()) {
                return partial ? Block.incomplete(calls, at) : Block.malformed(text.length());
            }
            if (text.charAt(i) != '[') {
                return Block.malformed(i);
            }
            i++;
        }
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (bracketed && j < text.length() && text.charAt(j) == ']') {
                return finish(text, j + 1, calls, partial, at);
            }
            if (!bracketed && closer != null && text.startsWith(closer, j)) {
                return Block.complete(calls, j + closer.length());
            }
            if (j >= text.length() || (!bracketed && closer != null && prefixOf(text, j, closer))) {
                return partial ? Block.incomplete(calls, at)
                        : (calls.isEmpty() ? Block.malformed(text.length()) : Block.complete(calls, text.length()));
            }
            Call c = readCall(text, j, partial);
            if (c.malformed) {
                return calls.isEmpty() ? Block.malformed(j + 1) : Block.complete(calls, j);
            }
            if (c.call != null) {
                calls.add(c.call);
            }
            if (c.end < 0) {
                return partial ? Block.incomplete(calls, at)
                        : (complete(calls) ? Block.incomplete(onlyComplete(calls), j) : Block.malformed(text.length()));
            }
            int k = JsonScan.skipWs(text, c.end);
            if (k < text.length() && text.charAt(k) == ',') {
                k++;
            }
            i = k;
        }
    }

    private static boolean complete(List<ToolCall> calls) {
        for (ToolCall c : calls) {
            if (c.complete()) {
                return true;
            }
        }
        return false;
    }

    private static List<ToolCall> onlyComplete(List<ToolCall> calls) {
        List<ToolCall> out = new ArrayList<>();
        for (ToolCall c : calls) {
            if (c.complete()) {
                out.add(c);
            }
        }
        return out;
    }

    /** After the list's {@code ]}: the closer when the format has one. */
    private Block finish(String text, int end, List<ToolCall> calls, boolean partial, int at) {
        if (closer == null) {
            return Block.complete(calls, end);
        }
        int k = JsonScan.skipWs(text, end);
        if (text.startsWith(closer, k)) {
            return Block.complete(calls, k + closer.length());
        }
        if (k >= text.length() || prefixOf(text, k, closer)) {
            return partial ? Block.incomplete(calls, at) : Block.complete(calls, text.length());
        }
        return Block.complete(calls, end);
    }

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

    private static Call readCall(String text, int at, boolean partial) {
        int k = at;
        while (k < text.length() && isIdent(text.charAt(k), k == at)) {
            k++;
        }
        if (k == at) {
            return new Call(null, -1, true);
        }
        if (k >= text.length()) {
            return partial ? new Call(null, -1, false) : new Call(null, -1, true);
        }
        if (text.charAt(k) != '(') {
            return new Call(null, -1, true);
        }
        String name = text.substring(at, k);
        Map<String, Object> args = new LinkedHashMap<>();
        int i = k + 1;
        while (true) {
            int j = JsonScan.skipWs(text, i);
            if (j >= text.length()) {
                return new Call(partialCall(name, args), -1, false);
            }
            if (text.charAt(j) == ')') {
                return new Call(new ToolCall(name, ToolSchemas.arguments(args), null, true), j + 1, false);
            }
            int ke = j;
            while (ke < text.length() && isIdent(text.charAt(ke), ke == j)) {
                ke++;
            }
            int eq = JsonScan.skipWs(text, ke);
            if (eq >= text.length()) {
                return new Call(partialCall(name, args), -1, false);
            }
            if (ke == j || text.charAt(eq) != '=') {
                return new Call(null, -1, true);
            }
            PyLiteral.Result v = PyLiteral.read(text, JsonScan.skipWs(text, eq + 1));
            if (v.invalid) {
                return new Call(null, -1, true);
            }
            if (v.end < 0) {
                return new Call(partialCall(name, args), -1, false);
            }
            args.put(text.substring(j, ke), v.value);
            int c = JsonScan.skipWs(text, v.end);
            if (c < text.length() && text.charAt(c) == ',') {
                c++;
            }
            i = c;
        }
    }

    private static ToolCall partialCall(String name, Map<String, Object> args) {
        String obj = ToolSchemas.arguments(args);
        return new ToolCall(name, obj.substring(0, obj.length() - 1), null, false);
    }

    private static boolean isIdent(char c, boolean first) {
        return Character.isLetter(c) || c == '_' || (!first && (Character.isDigit(c) || c == '.' || c == '-'));
    }
}
