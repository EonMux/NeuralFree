package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

/**
 * The body of one harmony-family message addressed to a recipient without the {@code functions.} prefix. A Muse
 * tool message ({@code to=get_weather}) carries its calls as an ATEM block, which streams through the body parser;
 * any other body — a gpt-oss built-in tool such as {@code browser.search} or {@code python} — is not a client
 * function and is dropped, as llama.cpp's parsers consume it. The body is held until its first non-blank text
 * decides which it is. Not thread-safe.
 */
final class RecipientBody {

    private enum Mode { UNDECIDED, PARSE, DROP }

    private final ToolCallStream stream;
    private final String opener;
    private final StringBuilder held = new StringBuilder();
    /** The output-wide index of each call this body started, -1 for a call past the output's limit. */
    private final List<Integer> indices = new ArrayList<>();
    private Mode mode = Mode.UNDECIDED;
    private boolean finished;

    /**
     * @param parser the body format's parser
     * @param opener the text a parsed body starts with (after blanks)
     * @param ids    the ids of calls the format carries none for
     */
    RecipientBody(ToolParser parser, String opener, Supplier<String> ids) {
        this.stream = new ToolCallStream(parser, ids);
        this.opener = opener;
    }

    /** More body text; the listener sees the calls of a parsed body, never its content. */
    void feed(String text, ToolCallStream.Listener listener) {
        if (mode == Mode.DROP || finished) {
            return;
        }
        if (mode == Mode.PARSE) {
            stream.feed(text, callsOnly(listener));
            return;
        }
        held.append(text);
        String lead = held.toString().stripLeading();
        if (lead.startsWith(opener)) {
            mode = Mode.PARSE;
            String buffered = held.toString();
            held.setLength(0);
            stream.feed(buffered, callsOnly(listener));
        } else if (!lead.isEmpty() && !opener.startsWith(lead)) {
            mode = Mode.DROP;
            held.setLength(0);
        }
    }

    /** Ends the body: the final parse's additions reach the listener. Idempotent. */
    void finish(ToolCallStream.Listener listener) {
        if (finished) {
            return;
        }
        finished = true;
        if (mode == Mode.PARSE) {
            stream.finish(callsOnly(listener));
        }
    }

    /** The calls the body's parse holds (empty for a dropped body). */
    List<ToolCall> calls() {
        return mode == Mode.PARSE ? stream.calls() : Collections.<ToolCall>emptyList();
    }

    /** The output-wide indices of the body's calls, in call order. */
    List<Integer> indices() {
        return indices;
    }

    private static ToolCallStream.Listener callsOnly(ToolCallStream.Listener listener) {
        return new ToolCallStream.Listener() {
            @Override
            public void content(String delta) {
                // text around the ATEM block is message markup, not content for the client
            }

            @Override
            public void callStarted(int callIndex, String id, String name) {
                listener.callStarted(callIndex, id, name);
            }

            @Override
            public void arguments(int callIndex, String delta) {
                listener.arguments(callIndex, delta);
            }
        };
    }
}
