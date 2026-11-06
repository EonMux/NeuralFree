package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Formats that write each call as a JSON object, or a JSON array of them, after an opener and before an optional
 * closer: Hermes and Qwen 2.5/3 ({@code <tool_call>{...}</tool_call>}), InternLM 2
 * ({@code <|action_start|><|plugin|>{...}<|action_end|>}), Jamba and Apriel ({@code <tool_calls>[...]</tool_calls>}),
 * Granite ({@code <|tool_call|>[...]}), Phi-4-mini and FireFunction ({@code functools[...]}), Llama 3.x
 * ({@code <|python_tag|>} or a bare object, {@code ;} between calls, {@code parameters} for the arguments) and xLAM
 * (a bare array, also inside a {@code ```json} fence).
 *
 * <p>A closer the final text never writes is not required: vLLM's Hermes pattern
 * ({@code <tool_call>(.*?)</tool_call>|<tool_call>(.*)}) accepts a call the output ends inside of.
 */
final class JsonToolParser extends BlockToolParser {

    /** What follows the opener. */
    enum Body { OBJECT, OBJECTS, ARRAY, ANY }

    private final String closer;
    private final Body body;
    private final boolean bare;

    /**
     * @param openers the openers; empty only with {@code bare}
     * @param closer  the closer, or null when a block ends with its JSON
     * @param body    what follows the opener
     * @param bare    whether a call may open the output without an opener
     */
    JsonToolParser(List<String> openers, String closer, Body body, boolean bare) {
        super(openers);
        this.closer = closer;
        this.body = Objects.requireNonNull(body, "body");
        this.bare = bare;
    }

    @Override
    int bareStart(String text, boolean partial) {
        if (!bare) {
            return -1;
        }
        int i = JsonScan.skipWs(text, 0);
        int s = skipFence(text, i);
        if (s == UNDECIDED) {
            return partial ? UNDECIDED : -1;
        }
        if (s >= text.length()) {
            return partial ? UNDECIDED : -1;
        }
        char c = text.charAt(s);
        boolean opens = (c == '{' && body != Body.ARRAY) || (c == '[' && body != Body.OBJECT
                && body != Body.OBJECTS);
        return opens ? i : -1;
    }

    @Override
    Block readBlock(String text, int at, String opener, boolean partial) {
        int i = JsonScan.skipWs(text, at + opener.length());
        int fenced = skipFence(text, i);
        if (fenced == UNDECIDED || fenced >= text.length()) {
            return partial ? Block.incomplete(new ArrayList<>(), at) : Block.malformed(text.length());
        }
        boolean hasFence = fenced != i;
        i = fenced;
        char c = text.charAt(i);
        JsonCalls.Run run;
        if (c == '[' && body != Body.OBJECT && body != Body.OBJECTS) {
            run = JsonCalls.array(text, i);
        } else if (c == '{' && body != Body.ARRAY) {
            run = JsonCalls.objects(text, i, body != Body.OBJECT, partial);
        } else {
            return Block.malformed(i + 1);
        }
        if (run.invalid) {
            return Block.malformed(Math.max(i + 1, run.end));
        }
        if (run.end < 0) {
            return partial ? Block.incomplete(run.calls, at) : finalCut(run, text);
        }
        int j = JsonScan.skipWs(text, run.end);
        if (hasFence) {
            if (text.startsWith("```", j)) {
                j = JsonScan.skipWs(text, j + 3);
            } else if (partial && prefixOf(text, j, "```")) {
                return Block.incomplete(run.calls, at);
            }
        }
        if (closer == null) {
            return Block.complete(run.calls, hasFence ? j : run.end);
        }
        if (text.startsWith(closer, j)) {
            return Block.complete(run.calls, j + closer.length());
        }
        if (j >= text.length() || prefixOf(text, j, closer)) {
            if (partial) {
                return Block.incomplete(run.calls, at);
            }
            return Block.complete(run.calls, text.length());
        }
        return Block.malformed(j);
    }

    /** A final text that ends inside the JSON: the finished calls stand, the unfinished object stays content. */
    private static Block finalCut(JsonCalls.Run run, String text) {
        List<ToolCall> done = new ArrayList<>();
        for (ToolCall c : run.calls) {
            if (c.complete()) {
                done.add(c);
            }
        }
        return done.isEmpty() || run.rest < 0 ? Block.malformed(text.length()) : Block.incomplete(done, run.rest);
    }

    /**
     * After a {@code ```json} (or bare {@code ```}) fence and the whitespace after it, or {@code i} when none
     * opens there; {@link #UNDECIDED} while the text ends inside a possible fence.
     */
    private static int skipFence(String text, int i) {
        if (text.startsWith("```", i)) {
            int j = i + 3;
            if (text.startsWith("json", j)) {
                j += 4;
            } else if (prefixOf(text, j, "json")) {
                return UNDECIDED;
            }
            return JsonScan.skipWs(text, j);
        }
        return prefixOf(text, i, "```") && i < text.length() ? UNDECIDED : i;
    }
}
