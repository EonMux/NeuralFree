package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.List;

/**
 * A format whose answer text sits inside {@code <response>...</response>} markup around its tool calls (ERNIE 4.5,
 * vLLM's {@code Ernie45ToolParser}): the calls come from the wrapped parser, and the content loses the tags and the
 * newlines right after {@code <response>}. Under {@code partial} a trailing prefix of a tag is held back, so every
 * partial content is a prefix of the final one.
 */
final class ResponseTagToolParser implements ToolParser {

    private static final String OPEN = "<response>";
    private static final String CLOSE = "</response>";

    private final ToolParser calls;

    ResponseTagToolParser(ToolParser calls) {
        this.calls = calls;
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        ToolParse parsed = calls.parse(text, partial);
        return new ToolParse(clean(parsed.content(), partial), parsed.calls());
    }

    @Override
    public List<String> triggers() {
        return calls.triggers();
    }

    /** The content without the response tags. */
    static String clean(String content, boolean partial) {
        StringBuilder out = new StringBuilder(content.length());
        int i = 0;
        boolean afterOpen = false;
        while (i < content.length()) {
            if (content.startsWith(OPEN, i)) {
                i += OPEN.length();
                afterOpen = true;
                continue;
            }
            if (content.startsWith(CLOSE, i)) {
                i += CLOSE.length();
                afterOpen = false;
                continue;
            }
            char c = content.charAt(i);
            if (partial && c == '<' && heldBack(content, i)) {
                break;
            }
            if (afterOpen && c == '\n') {
                i++;
                continue;
            }
            afterOpen = false;
            out.append(c);
            i++;
        }
        return out.toString();
    }

    /** Whether the text from {@code i} to the end could still grow into a tag. */
    private static boolean heldBack(String content, int i) {
        String rest = content.substring(i);
        return OPEN.startsWith(rest) || CLOSE.startsWith(rest);
    }
}
