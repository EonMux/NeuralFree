package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

/**
 * The streamed form of one generation's tool calls (llama.cpp's {@code common_chat_msg_diff}, vLLM's
 * {@code extract_tool_calls_streaming}): the content text is fed piece by piece, every feed re-parses the whole text
 * under {@code partial} and the difference to what was already sent goes to the {@link Listener} — content deltas,
 * the start of each call (index, id, name) and argument deltas. {@link #finish} parses the final text and sends the
 * rest. The parsers report partial text that the final text extends, so the deltas of each field concatenate to
 * the final parse; {@link #diverged()} reports a parse that broke that contract (a streamed call the final text
 * turned into content, say), which a stream cannot take back.
 *
 * <p>Call ids are the format's own when it writes them (Kimi, Mistral), else {@code call_} plus 24 random
 * alphanumerics, assigned when the call starts and kept in {@link #calls()}. Not thread-safe: one stream per
 * generation, fed in order.
 */
public final class ToolCallStream {

    /** Receives the streamed changes. */
    public interface Listener {

        /** More content text. */
        void content(String delta);

        /** Call {@code index} begins: its id and function name. */
        void callStarted(int index, String id, String name);

        /** More argument text of call {@code index}. */
        void arguments(int index, String delta);
    }

    private static final String ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    private final ToolParser parser;
    private final Supplier<String> idSource;
    private final StringBuilder text = new StringBuilder();
    private String sentContent = "";
    private final List<String> ids = new ArrayList<>();
    private final List<String> sentArguments = new ArrayList<>();
    private ToolParse last = ToolParse.empty();
    private boolean finished;
    private boolean diverged;

    /** A stream over {@code parser} with random {@code call_} ids. */
    public ToolCallStream(ToolParser parser) {
        this(parser, ToolCallStream::randomId);
    }

    /** A stream over {@code parser} whose generated ids come from {@code idSource}. */
    public ToolCallStream(ToolParser parser, Supplier<String> idSource) {
        this.parser = Objects.requireNonNull(parser, "parser");
        this.idSource = Objects.requireNonNull(idSource, "idSource");
    }

    /** An OpenAI-style call id: {@code call_} and 24 random alphanumerics. */
    public static String randomId() {
        ThreadLocalRandom r = ThreadLocalRandom.current();
        StringBuilder b = new StringBuilder("call_");
        for (int i = 0; i < 24; i++) {
            b.append(ALPHANUMERIC.charAt(r.nextInt(ALPHANUMERIC.length())));
        }
        return b.toString();
    }

    /**
     * Appends {@code piece} to the text and sends what the re-parse adds.
     *
     * @throws IllegalStateException after {@link #finish}
     */
    public void feed(String piece, Listener listener) {
        if (finished) {
            throw new IllegalStateException("tool call stream already finished");
        }
        if (piece == null || piece.isEmpty()) {
            return;
        }
        text.append(piece);
        send(parser.parse(text.toString(), true), listener);
    }

    /** Parses the final text, sends the rest and returns the final parse (with the stream's ids in {@link #calls()}). */
    public ToolParse finish(Listener listener) {
        if (!finished) {
            finished = true;
            ToolParse done = parser.parse(text.toString(), false);
            send(done, listener);
            if (!done.content().equals(sentContent) || done.calls().size() < ids.size()) {
                diverged = true;
            }
        }
        return last;
    }

    /** The whole text fed so far. */
    public String text() {
        return text.toString();
    }

    /** The latest parse: partial until {@link #finish}, final after it. */
    public ToolParse parse() {
        return last;
    }

    /** The calls of the latest parse, each with the id the stream sent for it. */
    public List<ToolCall> calls() {
        List<ToolCall> out = new ArrayList<>();
        List<ToolCall> calls = last.calls();
        for (int i = 0; i < calls.size(); i++) {
            ToolCall c = calls.get(i);
            String id = i < ids.size() ? ids.get(i) : (c.id() != null ? c.id() : null);
            out.add(new ToolCall(c.name(), c.arguments(), id, c.complete()));
        }
        return Collections.unmodifiableList(out);
    }

    /** Whether a parse contradicted text already sent (the final parse is authoritative then). */
    public boolean diverged() {
        return diverged;
    }

    private void send(ToolParse parse, Listener listener) {
        last = parse;
        String content = parse.content();
        if (content.startsWith(sentContent)) {
            if (content.length() > sentContent.length()) {
                listener.content(content.substring(sentContent.length()));
                sentContent = content;
            }
        } else {
            diverged = true;
        }
        List<ToolCall> calls = parse.calls();
        for (int i = 0; i < calls.size(); i++) {
            ToolCall c = calls.get(i);
            if (i == ids.size()) {
                String id = c.id() != null ? c.id() : idSource.get();
                ids.add(id);
                sentArguments.add("");
                listener.callStarted(i, id, c.name());
            }
            String args = c.arguments();
            String sent = sentArguments.get(i);
            if (args.startsWith(sent)) {
                if (args.length() > sent.length()) {
                    listener.arguments(i, args.substring(sent.length()));
                    sentArguments.set(i, args);
                }
            } else {
                diverged = true;
            }
        }
    }
}
