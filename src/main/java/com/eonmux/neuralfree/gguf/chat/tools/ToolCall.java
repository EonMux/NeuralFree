package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.Objects;

/**
 * One function call the model wrote, as a {@link ToolParser} reads it out of the assistant text: the function
 * {@link #name()}, its {@link #arguments()} as JSON object text, an optional {@link #id()} the format itself
 * carries (Kimi's {@code functions.name:0}, Mistral's {@code [CALL_ID]}) and whether the call is
 * {@link #complete()}.
 *
 * <p>While a generation streams, a call may be incomplete: its arguments are then a prefix of the arguments the
 * complete call will carry, so the streamed argument deltas concatenate to the final text
 * ({@link ToolCallStream}). Immutable.
 */
public final class ToolCall {

    private final String name;
    private final String arguments;
    private final String id;
    private final boolean complete;

    /**
     * @param name      the function name; never null or empty
     * @param arguments the arguments JSON text (a prefix of it while incomplete); never null
     * @param id        the call id the format carries, or null
     * @param complete  whether the call's closing syntax was read
     */
    public ToolCall(String name, String arguments, String id, boolean complete) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("tool call name is null or empty");
        }
        this.name = name;
        this.arguments = Objects.requireNonNull(arguments, "arguments");
        this.id = id;
        this.complete = complete;
    }

    /** A complete call without a format-carried id. */
    public static ToolCall of(String name, String arguments) {
        return new ToolCall(name, arguments, null, true);
    }

    public String name() {
        return name;
    }

    /** The arguments JSON object text; while {@link #complete()} is false, a prefix of the final text. */
    public String arguments() {
        return arguments;
    }

    /** The id the format wrote for the call, or null when the format carries none. */
    public String id() {
        return id;
    }

    public boolean complete() {
        return complete;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ToolCall)) {
            return false;
        }
        ToolCall that = (ToolCall) o;
        return complete == that.complete && name.equals(that.name) && arguments.equals(that.arguments)
                && Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, arguments, id, complete);
    }

    @Override
    public String toString() {
        return "ToolCall(" + name + ", " + arguments + (id == null ? "" : ", id=" + id)
                + (complete ? "" : ", partial") + ")";
    }
}
