package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * The tool-call side of one generated choice: it takes the routed output — content pieces, and for harmony models
 * the tool recipients and {@code TOOL_CALL} bodies of the reasoning filter — and sends the client's view to a
 * {@link ToolCallStream.Listener}: content deltas, call starts and argument deltas under one index space.
 *
 * <ul>
 *   <li>Content runs through the format's {@link ToolParser} in a {@link ToolCallStream} when one is set, else it
 *       passes through unchanged.</li>
 *   <li>A harmony recipient {@code functions.NAME} starts a call to {@code NAME} whose arguments are the message
 *       body. Another recipient's body goes to the recipient parser when one is set ({@link RecipientBody}): a
 *       Muse tool message's ATEM block yields its calls, and any other body (gpt-oss built-ins such as
 *       {@code browser.search} or {@code python}, which are not client functions) is dropped, as llama.cpp's
 *       parsers consume it.</li>
 *   <li>Calls past {@code maxCalls} ({@code parallel_tool_calls: false} allows one) are dropped.</li>
 * </ul>
 *
 * Not thread-safe: one output per choice, fed in order.
 */
public final class ToolOutput {

    private static final String FUNCTIONS = "functions.";

    private final ToolCallStream stream;
    private final boolean channelCalls;
    private final int maxCalls;
    private final Supplier<String> ids;
    private final StringBuilder plainContent = new StringBuilder();
    private final List<Integer> streamIndex = new ArrayList<>();
    private final List<ToolCall> calls = new ArrayList<>();
    private final List<StringBuilder> channelArgs = new ArrayList<>();
    private final List<Integer> channelIndex = new ArrayList<>();
    private final ToolParser recipientParser;
    private final List<RecipientBody> bodies = new ArrayList<>();
    private RecipientBody body;
    private boolean inChannelCall;
    private boolean finished;

    /**
     * @param parser       the content's tool-call format, or null to pass content through
     * @param channelCalls whether harmony tool recipients start calls
     * @param maxCalls     the most calls kept (at least 1)
     * @param ids          the ids of calls whose format carries none
     */
    public ToolOutput(ToolParser parser, boolean channelCalls, int maxCalls, Supplier<String> ids) {
        this(parser, channelCalls, null, maxCalls, ids);
    }

    /**
     * @param parser          the content's tool-call format, or null to pass content through
     * @param channelCalls    whether harmony tool recipients start calls
     * @param recipientParser the format of a non-{@code functions.} recipient's body (the Muse ATEM block), or
     *                        null to drop such bodies
     * @param maxCalls        the most calls kept (at least 1)
     * @param ids             the ids of calls whose format carries none
     */
    public ToolOutput(ToolParser parser, boolean channelCalls, ToolParser recipientParser, int maxCalls,
                      Supplier<String> ids) {
        if (maxCalls < 1) {
            throw new IllegalArgumentException("maxCalls must be at least 1, got " + maxCalls);
        }
        this.ids = ids == null ? ToolCallStream::randomId : ids;
        this.stream = parser == null ? null : new ToolCallStream(parser, this.ids);
        this.channelCalls = channelCalls;
        this.recipientParser = recipientParser;
        this.maxCalls = maxCalls;
    }

    /** More content text. */
    public void content(String text, ToolCallStream.Listener listener) {
        requireOpen();
        endMessage(listener);
        if (text == null || text.isEmpty()) {
            return;
        }
        if (stream == null) {
            plainContent.append(text);
            listener.content(text);
            return;
        }
        stream.feed(text, remap(listener, streamIndex));
    }

    /** A harmony header named {@code recipient}; its body follows through {@link #toolCallText}. */
    public void toolCallStarted(String recipient, ToolCallStream.Listener listener) {
        requireOpen();
        endMessage(listener);
        if (!channelCalls || recipient == null) {
            return;
        }
        if (!recipient.startsWith(FUNCTIONS)) {
            if (recipientParser != null) {
                body = new RecipientBody(recipientParser, ToolParserRegistry.ATEM_CALLS, ids);
                bodies.add(body);
            }
            return;
        }
        if (recipient.length() == FUNCTIONS.length() || calls.size() >= maxCalls) {
            return;
        }
        String name = recipient.substring(FUNCTIONS.length());
        String id = ids.get();
        int index = calls.size();
        calls.add(new ToolCall(name, "", id, false));
        channelArgs.add(new StringBuilder());
        channelIndex.add(index);
        inChannelCall = true;
        listener.callStarted(index, id, name);
    }

    /** Body text of the current harmony tool message. */
    public void toolCallText(String text, ToolCallStream.Listener listener) {
        requireOpen();
        if (text == null || text.isEmpty()) {
            return;
        }
        if (body != null) {
            body.feed(text, remap(listener, body.indices()));
            return;
        }
        if (!inChannelCall) {
            return;
        }
        int local = channelArgs.size() - 1;
        channelArgs.get(local).append(text);
        int index = channelIndex.get(local);
        ToolCall c = calls.get(index);
        calls.set(index, new ToolCall(c.name(), channelArgs.get(local).toString(), c.id(), false));
        listener.arguments(index, text);
    }

    /** Ends the output: sends what the final parse adds and returns the content and calls with their ids. */
    public ToolParse finish(ToolCallStream.Listener listener) {
        if (finished) {
            return result();
        }
        endMessage(listener);
        finished = true;
        for (RecipientBody b : bodies) {
            complete(b.calls(), b.indices());
        }
        if (stream != null) {
            stream.finish(remap(listener, streamIndex));
            complete(stream.calls(), streamIndex);
        }
        for (int local = 0; local < channelIndex.size(); local++) {
            int index = channelIndex.get(local);
            ToolCall c = calls.get(index);
            calls.set(index, new ToolCall(c.name(), c.arguments().isEmpty() ? "{}" : c.arguments(), c.id(), true));
        }
        return result();
    }

    /** Ends the current harmony message: a recipient body's final parse lands, a functions call stops growing. */
    private void endMessage(ToolCallStream.Listener listener) {
        inChannelCall = false;
        if (body != null) {
            body.finish(remap(listener, body.indices()));
            body = null;
        }
    }

    /** Replaces the calls a parse reports with their final, complete form. */
    private void complete(List<ToolCall> parsed, List<Integer> indices) {
        for (int local = 0; local < parsed.size() && local < indices.size(); local++) {
            int index = indices.get(local);
            if (index >= 0) {
                ToolCall c = parsed.get(local);
                calls.set(index, new ToolCall(c.name(), c.arguments(), c.id(), true));
            }
        }
    }

    /** Whether any call started. */
    public boolean hasCalls() {
        return !calls.isEmpty();
    }

    /**
     * The content and calls. When the final parse contradicted what streamed ({@link ToolCallStream#diverged()}),
     * the final parse is authoritative: its calls, then the harmony recipients' calls, up to {@code maxCalls}.
     */
    private ToolParse result() {
        if (stream == null) {
            return new ToolParse(plainContent.toString(), calls);
        }
        if (!stream.diverged()) {
            return new ToolParse(stream.parse().content(), calls);
        }
        List<ToolCall> out = new ArrayList<>();
        for (ToolCall c : stream.calls()) {
            if (out.size() < maxCalls) {
                out.add(new ToolCall(c.name(), c.arguments(), c.id() != null ? c.id() : ids.get(), true));
            }
        }
        for (int index = 0; index < calls.size(); index++) {
            if (out.size() < maxCalls && !streamIndex.contains(index)) {
                out.add(calls.get(index));
            }
        }
        return new ToolParse(stream.parse().content(), out);
    }

    /** Whether the streamed deltas disagree with {@link #finish}'s result (the final parse broke prefix order). */
    public boolean diverged() {
        return stream != null && stream.diverged();
    }

    /**
     * The listener of one parse (the content stream or a recipient body), its call indices mapped into the
     * output's index space through {@code localToGlobal}, which grows as calls start.
     */
    private ToolCallStream.Listener remap(ToolCallStream.Listener listener, List<Integer> localToGlobal) {
        return new ToolCallStream.Listener() {
            @Override
            public void content(String delta) {
                listener.content(delta);
            }

            @Override
            public void callStarted(int local, String id, String name) {
                if (calls.size() >= maxCalls) {
                    localToGlobal.add(-1);
                    return;
                }
                int index = calls.size();
                localToGlobal.add(index);
                calls.add(new ToolCall(name, "", id, false));
                listener.callStarted(index, id, name);
            }

            @Override
            public void arguments(int local, String delta) {
                int index = local < localToGlobal.size() ? localToGlobal.get(local) : -1;
                if (index < 0) {
                    return;
                }
                ToolCall c = calls.get(index);
                calls.set(index, new ToolCall(c.name(), c.arguments() + delta, c.id(), false));
                listener.arguments(index, delta);
            }
        };
    }

    private void requireOpen() {
        if (finished) {
            throw new IllegalStateException("tool output already finished");
        }
    }
}
