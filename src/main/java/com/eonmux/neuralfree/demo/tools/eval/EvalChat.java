package com.eonmux.neuralfree.demo.tools.eval;

/**
 * One chat completion answer: the generated text, the reasoning content some servers return beside
 * it, why generation stopped, and the usage and timings the server reported (null when it reported
 * none).
 */
public final class EvalChat {

    private final String content;
    private final String reasoningContent;
    private final String finishReason;
    private final Long tokens;
    private final Double tpsGen;
    private final Double tGenMs;

    public EvalChat(String content, String reasoningContent, String finishReason, Long tokens,
                    Double tpsGen, Double tGenMs) {
        this.content = content == null ? "" : content;
        this.reasoningContent = reasoningContent;
        this.finishReason = finishReason == null || finishReason.isEmpty() ? "stop" : finishReason;
        this.tokens = tokens;
        this.tpsGen = tpsGen;
        this.tGenMs = tGenMs;
    }

    /** The generated text (empty when the server returned none). */
    public String content() {
        return content;
    }

    /** The reasoning content, null when the server returned none. */
    public String reasoningContent() {
        return reasoningContent;
    }

    /** Why generation stopped, {@code stop} when the server omitted the field. */
    public String finishReason() {
        return finishReason;
    }

    /** The completion tokens, null when the server reported none. */
    public Long tokens() {
        return tokens;
    }

    /** The server's tokens per second, null when it reported none. */
    public Double tpsGen() {
        return tpsGen;
    }

    /** The server's generation time in milliseconds, null when it reported none. */
    public Double tGenMs() {
        return tGenMs;
    }
}
