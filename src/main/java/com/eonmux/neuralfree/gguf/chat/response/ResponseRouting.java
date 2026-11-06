package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Channel;

import java.util.Map;
import java.util.Objects;

/**
 * One generation's output routed by a serving {@link ResponseTemplate} (vLLM {@code _ResponseStream} of the
 * {@code hf} and {@code beam} parsers): the {@code thinking} region streams as reasoning — as content without a
 * reasoning parser ({@code parse_reasoning} false) — the {@code content} region as content, and a {@code tool_calls} region passes whole — opener, body and
 * closer — to the content, where the template's tool parser reads the call. Text of the prompt's message part never
 * routes. At the end, a closer the output stopped inside, whose missing rest is only the next message's start anchor,
 * is completed first ({@link ResponseParser#closerRest}). Not thread-safe.
 */
public final class ResponseRouting {

    /** Receives routed text. */
    @FunctionalInterface
    public interface Out {
        void route(Channel channel, String text);
    }

    private final ResponseParser parser;
    private final Out out;
    private final int prefixEnd;
    private final boolean startsInThinking;
    private final boolean parsesReasoning;
    private final Listener listener = new Listener();
    private boolean fed;
    private boolean finished;

    /**
     * Routing with the thinking region parsed as the reasoning.
     *
     * @param serving the serving template ({@link ResponseTemplate#forServing})
     * @param prompt  the rendered prompt, or null when unknown
     * @param out     receives the routed text
     */
    public ResponseRouting(ResponseTemplate serving, String prompt, Out out) {
        this(serving, prompt, out, true);
    }

    /**
     * @param serving         the serving template ({@link ResponseTemplate#forServing})
     * @param prompt          the rendered prompt, or null when unknown
     * @param out             receives the routed text
     * @param parsesReasoning whether the thinking region streams as reasoning, else as content
     */
    public ResponseRouting(ResponseTemplate serving, String prompt, Out out, boolean parsesReasoning) {
        Objects.requireNonNull(serving, "serving");
        this.parsesReasoning = parsesReasoning;
        this.out = Objects.requireNonNull(out, "out");
        String prefix = prompt == null ? "" : prompt;
        this.parser = new ResponseParser(serving, prefix, ToolArguments.NONE, null);
        this.prefixEnd = parser.inputLength();
        this.startsInThinking = startsInThinking(serving, prefix);
    }

    /** Routes the next piece of output. */
    public void feed(String text) {
        fed = true;
        parser.feed(text, listener);
    }

    /** Ends the output: a cut-off closer completed, held text released, the open region closed. */
    public void finish() {
        if (finished) return;
        finished = true;
        if (parser.inputLength() > prefixEnd) {
            parser.feed(parser.closerRest(), listener);
            parser.finish(listener);
        }
    }

    /**
     * Whether text goes to the thinking region now (before any output: whether the output starts there); never
     * without reasoning parsing.
     */
    public boolean inThinking() {
        if (!parsesReasoning) return false;
        return fed ? ResponseTemplate.THINKING.equals(parser.currentField()) : startsInThinking;
    }

    /** Whether output is held back as a possible delimiter. */
    public boolean holding() {
        return parser.holding();
    }

    /**
     * Whether a generation after {@code prompt} starts in the thinking region: the prompt leaves it open, or a
     * delimiter held at the prompt's end opens it, or it is the implicit region.
     */
    public static boolean startsInThinking(ResponseTemplate serving, String prompt) {
        ResponseParser p = new ResponseParser(serving, prompt == null ? "" : prompt, ToolArguments.NONE, null);
        if (ResponseTemplate.THINKING.equals(p.currentField())) return true;
        boolean[] opens = {false};
        p.feed("\0", new ResponseParser.Listener() {
            @Override public void opened(String field, int start, int end, Map<String, String> captures) {
                opens[0] |= ResponseTemplate.THINKING.equals(field);
            }
        });
        return opens[0] || ResponseTemplate.THINKING.equals(p.currentField());
    }

    /**
     * Whether a generation after {@code prompt} has its reasoning behind it (vLLM {@code _reasoning_has_ended}): the
     * template has no thinking field, the prompt closed the thinking region or opened the content or a tool region, a
     * delimiter held at the prompt's end does, or — with no region open — the request turned thinking off. A repeating
     * thinking field never ends.
     */
    public static boolean reasoningEnded(ResponseTemplate serving, String prompt, boolean thinkingDisabled) {
        ResponseTemplate.Field thinking = serving.field(ResponseTemplate.THINKING);
        if (thinking == null) return true;
        if (thinking.repeats()) return false;
        EndProbe initial = new EndProbe();
        ResponseParser p = new ResponseParser(serving, prompt == null ? "" : prompt, ToolArguments.NONE, initial);
        if (initial.thinkingClosed) return true;
        if (ResponseTemplate.CONTENT.equals(initial.active) || ResponseTemplate.TOOL_CALLS.equals(initial.active)) {
            return true;
        }
        EndProbe probe = new EndProbe();
        probe.probing = true;
        p.feed("\0", probe);
        if (probe.decided != null) return probe.decided;
        if (ResponseTemplate.THINKING.equals(initial.active)) return false;
        return thinkingDisabled;
    }

    /** Tracks the events of {@link #reasoningEnded}: the region left open, a closed thinking region, a probe's verdict. */
    private static final class EndProbe implements ResponseParser.Listener {
        String active;
        boolean thinkingClosed;
        boolean probing;
        Boolean decided;

        @Override public void opened(String field, int start, int end, Map<String, String> captures) {
            active = field;
            if (!probing || decided != null) return;
            if ((ResponseTemplate.CONTENT.equals(field) || ResponseTemplate.TOOL_CALLS.equals(field)) && end > start) {
                decided = true;
            } else if (ResponseTemplate.THINKING.equals(field)) {
                decided = false;
            }
        }

        @Override public void closed(String field, int start, int end, Object value) {
            ended(field);
        }

        @Override public void malformed(String field, int start, int end, String error) {
            ended(field);
        }

        private void ended(String field) {
            thinkingClosed |= ResponseTemplate.THINKING.equals(field);
            active = null;
            if (probing && decided == null && ResponseTemplate.THINKING.equals(field)) decided = true;
        }
    }

    /** Routes the output's events: thinking to reasoning, content to content, a tool region whole to content. */
    private final class Listener implements ResponseParser.Listener {
        @Override public void opened(String field, int start, int end, Map<String, String> captures) {
            delimiter(field, start, end);
        }

        @Override public void chunk(String field, String text, boolean raw) {
            boolean reasoning = parsesReasoning && ResponseTemplate.THINKING.equals(field);
            out.route(reasoning ? Channel.REASONING : Channel.CONTENT, text);
        }

        @Override public void closed(String field, int start, int end, Object value) {
            delimiter(field, start, end);
        }

        @Override public void malformed(String field, int start, int end, String error) {
            delimiter(field, start, end);
        }

        private void delimiter(String field, int start, int end) {
            if (!ResponseTemplate.TOOL_CALLS.equals(field)) return;
            int from = Math.max(start, prefixEnd);
            if (end > from) out.route(Channel.CONTENT, parser.input(from, end));
        }
    }
}
