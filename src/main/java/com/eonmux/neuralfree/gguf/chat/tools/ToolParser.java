package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;

import java.util.List;

/**
 * Reads the tool calls of one model family's output format out of the assistant's text (vLLM's
 * {@code ToolParser.extract_tool_calls}, llama.cpp's {@code common_chat_parse}). One parser serves one request: a
 * {@link Factory} creates it over the request's tools, whose parameter schemas type the values of formats that
 * write arguments as text ({@link ToolSchemas}).
 *
 * <p>{@link #parse} reads the whole text each time; under {@code partial} the text may still grow, so the parser
 * holds back what the next tokens could still turn into tool-call syntax, and reports an unfinished call with
 * argument text that the finished call's arguments extend. {@link ToolCallStream} diffs successive partial parses
 * into streamed deltas. Implementations are immutable after creation and safe to share.
 */
public interface ToolParser {

    /**
     * The content and calls of {@code text}.
     *
     * @param text    the assistant's text so far (reasoning already removed)
     * @param partial whether more text may follow
     */
    ToolParse parse(String text, boolean partial);

    /**
     * The literals that open a tool call in this format: the trigger words of a lazy tool-call grammar and the
     * prefixes a streamed content delta holds back.
     */
    List<String> triggers();

    /**
     * Creates the parsers of one format. A factory is the registry row ({@link ToolParserRegistry}) and the
     * factory type plugins contribute through {@code ToolParserProvider<ToolParser.Factory>}.
     */
    interface Factory {

        /** The parser name a request or the {@code neuralfree.tool.parser} property selects (vLLM's names). */
        String name();

        /** Further names that select the same format; empty when none. */
        List<String> aliases();

        /** A parser over the request's tools (their schemas type the argument values); never null. */
        ToolParser create(List<ChatTool> tools);
    }
}
