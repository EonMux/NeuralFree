package com.eonmux.neuralfree.demo;

/**
 * The outcome of one assistant turn of a {@link ChatSession}: the content and reasoning text the backend
 * streamed, and the error text when the turn failed (the text streamed before the failure stays). Immutable.
 */
final class ChatAnswer {

    private final String content;
    private final String reasoning;
    private final String error;

    private ChatAnswer(String content, String reasoning, String error) {
        this.content = content == null ? "" : content;
        this.reasoning = reasoning == null ? "" : reasoning;
        this.error = error;
    }

    /** A completed (or interrupted) turn. */
    static ChatAnswer of(String content, String reasoning) {
        return new ChatAnswer(content, reasoning, null);
    }

    /** A failed turn with the text streamed before the failure. */
    static ChatAnswer failed(String content, String reasoning, String error) {
        if (error == null || error.isEmpty()) {
            throw new IllegalArgumentException("a failed turn needs its error text");
        }
        return new ChatAnswer(content, reasoning, error);
    }

    /** The content-channel text. */
    String content() {
        return content;
    }

    /** The reasoning-channel text, empty when there was none. */
    String reasoning() {
        return reasoning;
    }

    /** The error text of a failed turn, or null. */
    String error() {
        return error;
    }
}
