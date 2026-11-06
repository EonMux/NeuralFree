package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The output of a named {@code tool_choice} (vLLM's named function call): the grammar holds the whole content to
 * the function's parameter object, so the content is the arguments of one call to that function. Leading
 * and trailing whitespace is dropped; while streaming, the arguments grow with the text.
 */
final class NamedFunctionToolParser implements ToolParser {

    private final String name;

    NamedFunctionToolParser(String name) {
        this.name = Objects.requireNonNull(name, "name");
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        int i = JsonScan.skipWs(text, 0);
        if (i >= text.length()) {
            return partial ? ToolParse.empty() : new ToolParse("", Collections.singletonList(ToolCall.of(name, "{}")));
        }
        String args = text.substring(i);
        if (partial) {
            return new ToolParse("", Collections.singletonList(new ToolCall(name, BlockToolParser.rtrim(args), null,
                    false)));
        }
        String trimmed = args.trim();
        if (JsonScan.parseOrNull(trimmed) == null) {
            return new ToolParse(text, Collections.<ToolCall>emptyList());
        }
        return new ToolParse("", Collections.singletonList(ToolCall.of(name, trimmed)));
    }

    @Override
    public List<String> triggers() {
        return Collections.emptyList();
    }
}
