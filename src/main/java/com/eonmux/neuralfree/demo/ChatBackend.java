package com.eonmux.neuralfree.demo;

import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;

/**
 * Where a {@link ChatSession} gets its answers: a locally loaded model ({@link LocalChat}) or a running server
 * ({@link RemoteChat}). The session owns the conversation and the commands; the backend answers one turn at a
 * time over the whole message list.
 */
interface ChatBackend {

    /** Prints the backend's banner lines (model, modalities, settings) before the command list. */
    void printBanner(PrintStream out);

    /**
     * Why the backend cannot take a media file of {@code kind} ({@code image}, {@code audio} or {@code video}),
     * or null when it can.
     */
    String mediaRefusal(String kind);

    /**
     * Answers the conversation {@code messages} (OpenAI chat messages, the last a user message), streaming the
     * text to the console as it arrives and stopping early once {@code cancelled} turns true.
     */
    ChatAnswer answer(List<Map<String, Object>> messages, BooleanSupplier cancelled) throws IOException;

    /** Drops any cached state of the conversation ({@code /clear}). */
    void reset();
}
