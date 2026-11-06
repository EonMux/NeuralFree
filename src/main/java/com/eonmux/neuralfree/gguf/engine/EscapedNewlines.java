package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Recovers the newlines a tokenizer decodes as escape text: {@code \n}, {@code \r} and {@code \r\n} (two or four
 * characters) become a newline, as vLLM's MiniCPM-V {@code EscapedNewlineNormalizer} recovers them
 * (vllm/parser/minicpmv.py). A doubled backslash, inline code, a code block, a {@code $...$}, {@code $$...$$},
 * {@code \(...\)} or {@code \[...\]} span, and every configured verbatim span (an open marker up to its close
 * marker) keep their text. The text streams in pieces: a trailing half of an escape, a backslash run or a marker
 * waits for the next piece, so the streamed result equals the whole-text one. Not thread-safe; one stream.
 */
final class EscapedNewlines {

    private final List<String> spans;
    /** The marker that closes the current kept span, or null outside one. */
    private String closing;
    private String pending = "";

    /** A normalizer keeping every {@code spans} open/close pair's text as written. */
    EscapedNewlines(List<String> spans) {
        this.spans = Collections.unmodifiableList(new ArrayList<>(spans));
    }

    /** Recovers the newlines of {@code text}; {@code last} releases everything held back. */
    String feed(String text, boolean last) {
        String s = pending + text;
        pending = "";
        StringBuilder out = new StringBuilder(s.length());
        int i = 0;
        while (i < s.length()) {
            int next = closing != null ? inSpan(s, i, last, out) : outsideSpan(s, i, last, out);
            if (next < 0) {
                pending = s.substring(i);
                break;
            }
            i = next;
        }
        if (last && !pending.isEmpty()) {
            out.append(pending);
            pending = "";
        }
        return out.toString();
    }

    /** One step inside a kept span; returns the next index, or -1 to hold the rest. */
    private int inSpan(String s, int i, boolean last, StringBuilder out) {
        if (s.startsWith(closing, i)) {
            out.append(closing);
            int next = i + closing.length();
            closing = null;
            return next;
        }
        if (!last && closing.startsWith(s.substring(i))) return -1;
        out.append(s.charAt(i));
        return i + 1;
    }

    /** One step outside a kept span; returns the next index, or -1 to hold the rest. */
    private int outsideSpan(String s, int i, boolean last, StringBuilder out) {
        String rest = s.substring(i);
        for (int k = 0; k < spans.size(); k += 2) {
            String open = spans.get(k);
            if (rest.startsWith(open)) {
                out.append(open);
                closing = spans.get(k + 1);
                return i + open.length();
            }
            if (!last && open.startsWith(rest)) return -1;
        }
        char c = s.charAt(i);
        if (c == '`') {
            if (!last && (rest.equals("`") || rest.equals("``"))) return -1;
            return opening(rest.startsWith("```") ? "```" : "`", i, out);
        }
        if (c == '$') {
            if (!last && rest.equals("$")) return -1;
            return opening(rest.startsWith("$$") ? "$$" : "$", i, out);
        }
        if (c != '\\') {
            out.append(c);
            return i + 1;
        }
        return escape(s, i, rest, last, out);
    }

    /** Opens a span closed by its own marker ({@code `}, {@code ```}, {@code $}, {@code $$}). */
    private int opening(String marker, int i, StringBuilder out) {
        out.append(marker);
        closing = marker;
        return i + marker.length();
    }

    /** A backslash at {@code i}: a run stays, an escaped newline becomes one, a LaTeX opener opens its span. */
    private int escape(String s, int i, String rest, boolean last, StringBuilder out) {
        int end = i + 1;
        while (end < s.length() && s.charAt(end) == '\\') end++;
        if (end - i > 1) {
            // the run may still grow, and its parity decides whether a following n stays literal
            if (!last && end == s.length()) return -1;
            out.append(s, i, end);
            return end;
        }
        if (!last && (rest.equals("\\") || rest.equals("\\r") || rest.equals("\\r\\"))) return -1;
        if (rest.startsWith("\\r\\n")) {
            out.append('\n');
            return i + 4;
        }
        if (rest.startsWith("\\n") || rest.startsWith("\\r")) {
            out.append('\n');
            return i + 2;
        }
        if (rest.startsWith("\\(") || rest.startsWith("\\[")) {
            out.append(rest, 0, 2);
            closing = rest.charAt(1) == '(' ? "\\)" : "\\]";
            return i + 2;
        }
        out.append('\\');
        return i + 1;
    }

    /**
     * A sink recovering newlines in the reasoning and in the content separately, so a half escape ending the
     * reasoning cannot swallow the start of the content: the reasoning releases its held text once content or a
     * tool call follows, both release at {@link #finish}. Tool-call text passes unchanged.
     */
    static final class RecoveringSink implements ReasoningFilter.Sink {

        private final ReasoningFilter.Sink out;
        private final EscapedNewlines reasoning;
        private final EscapedNewlines content;

        RecoveringSink(ReasoningFilter.Sink out, List<String> spans) {
            this.out = Objects.requireNonNull(out, "out");
            this.reasoning = new EscapedNewlines(spans);
            this.content = new EscapedNewlines(spans);
        }

        @Override
        public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
            if (channel == ReasoningFilter.Channel.REASONING) {
                send(ReasoningFilter.Channel.REASONING, tokenId, reasoning.feed(text, false));
                return;
            }
            send(ReasoningFilter.Channel.REASONING, tokenId, reasoning.feed("", true));
            if (channel == ReasoningFilter.Channel.CONTENT) {
                send(ReasoningFilter.Channel.CONTENT, tokenId, content.feed(text, false));
            } else {
                send(channel, tokenId, text);
            }
        }

        @Override
        public void toolCallStarted(String recipient) {
            send(ReasoningFilter.Channel.REASONING, -1, reasoning.feed("", true));
            out.toolCallStarted(recipient);
        }

        @Override
        public void stepped(int tokenId) {
            out.stepped(tokenId);
        }

        @Override
        public void contentFallback() {
            send(ReasoningFilter.Channel.REASONING, -1, reasoning.feed("", true));
            out.contentFallback();
        }

        /** Releases what both channels hold back. */
        void finish() {
            send(ReasoningFilter.Channel.REASONING, -1, reasoning.feed("", true));
            send(ReasoningFilter.Channel.CONTENT, -1, content.feed("", true));
        }

        private void send(ReasoningFilter.Channel channel, int tokenId, String text) {
            if (!text.isEmpty()) out.accept(channel, tokenId, text);
        }
    }
}
