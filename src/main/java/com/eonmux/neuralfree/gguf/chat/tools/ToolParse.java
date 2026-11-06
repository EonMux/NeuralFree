package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * What a {@link ToolParser} read out of assistant text: the {@link #content()} outside the tool-call syntax and
 * the {@link #calls()} in order. Immutable.
 */
public final class ToolParse {

    private static final ToolParse EMPTY = new ToolParse("", Collections.<ToolCall>emptyList());

    private final String content;
    private final List<ToolCall> calls;

    public ToolParse(String content, List<ToolCall> calls) {
        this.content = Objects.requireNonNull(content, "content");
        this.calls = Collections.unmodifiableList(new ArrayList<>(calls));
    }

    /** No content and no calls. */
    public static ToolParse empty() {
        return EMPTY;
    }

    /** Text that is no tool call; whitespace around the calls is dropped. */
    public String content() {
        return content;
    }

    /** The calls in output order; unmodifiable. */
    public List<ToolCall> calls() {
        return calls;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ToolParse)) {
            return false;
        }
        ToolParse that = (ToolParse) o;
        return content.equals(that.content) && calls.equals(that.calls);
    }

    @Override
    public int hashCode() {
        return Objects.hash(content, calls);
    }

    @Override
    public String toString() {
        return "ToolParse(content=" + content + ", calls=" + calls + ")";
    }
}
