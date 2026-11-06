package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.List;

/**
 * A format whose content ends at a marker the call syntax follows (GigaChat 3's {@code <|message_sep|>}): the calls
 * come from the wrapped parser and the content loses the marker and the text after it. Under {@code partial} a
 * trailing prefix of the marker is held back, so every partial content is a prefix of the final one.
 */
final class ContentEndToolParser implements ToolParser {

    private final ToolParser calls;
    private final String marker;

    /** @param marker the text at which the content ends */
    ContentEndToolParser(ToolParser calls, String marker) {
        this.calls = calls;
        this.marker = marker;
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        ToolParse parsed = calls.parse(text, partial);
        return new ToolParse(cut(parsed.content(), partial), parsed.calls());
    }

    @Override
    public List<String> triggers() {
        return calls.triggers();
    }

    /** The content up to the marker, a trailing proper prefix of the marker held back while streaming. */
    private String cut(String content, boolean partial) {
        int at = content.indexOf(marker);
        if (at >= 0) {
            return content.substring(0, at);
        }
        for (int k = Math.min(marker.length() - 1, content.length()); k > 0; k--) {
            if (partial && content.endsWith(marker.substring(0, k))) {
                return content.substring(0, content.length() - k);
            }
        }
        return content;
    }
}
