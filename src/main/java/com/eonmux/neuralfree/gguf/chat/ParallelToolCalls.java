package com.eonmux.neuralfree.gguf.chat;

import java.util.Objects;

/**
 * A request's {@code parallel_tool_calls}: what the client asked for and what the request serves. vLLM refuses
 * a tool guarantee only on the client's explicit false ({@code request.parallel_tool_calls is False}); llama.cpp
 * serves an absent field as the chat template's {@code supports_parallel_tool_calls} (server-common.cpp:1447).
 * The served value limits the calls wherever a parser or grammar can enforce them; the explicit one alone refuses
 * a guarantee by name. Immutable.
 */
public final class ParallelToolCalls {

    /** The request sets {@code parallel_tool_calls: true}. */
    public static final ParallelToolCalls PARALLEL = new ParallelToolCalls(Boolean.TRUE, true);

    /** The request sets {@code parallel_tool_calls: false}. */
    public static final ParallelToolCalls SINGLE = new ParallelToolCalls(Boolean.FALSE, false);

    private final Boolean requested;
    private final boolean allowed;

    private ParallelToolCalls(Boolean requested, boolean allowed) {
        this.requested = requested;
        this.allowed = allowed;
    }

    /**
     * The request's {@code parallel_tool_calls}: {@code requested} when it sets one, else {@code byDefault}, the
     * chat template's ({@code EngineService.defaultParallelToolCalls}).
     */
    public static ParallelToolCalls of(Boolean requested, boolean byDefault) {
        if (requested == null) {
            return new ParallelToolCalls(null, byDefault);
        }
        return requested ? PARALLEL : SINGLE;
    }

    /** The client's value, null when the request does not set it. */
    public Boolean requested() {
        return requested;
    }

    /** The value the request serves: the client's when set, else the model template's default. */
    public boolean allowed() {
        return allowed;
    }

    /** Whether the client set {@code parallel_tool_calls: false}: the value a guarantee refusal answers. */
    public boolean singleAsked() {
        return requested == Boolean.FALSE;
    }

    @Override public boolean equals(Object o) {
        if (!(o instanceof ParallelToolCalls)) {
            return false;
        }
        ParallelToolCalls other = (ParallelToolCalls) o;
        return Objects.equals(requested, other.requested) && allowed == other.allowed;
    }

    @Override public int hashCode() {
        return Objects.hash(requested, allowed);
    }

    @Override public String toString() {
        return requested == null ? "parallel_tool_calls=default(" + allowed + ")"
                : "parallel_tool_calls=" + requested;
    }
}
