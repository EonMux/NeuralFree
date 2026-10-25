package com.eonmux.neuralfree.utils.json;

/**
 * A JSON document that cannot be read or does not have the expected shape: a syntax error, an exceeded
 * {@link Json.Limits} cap, invalid UTF-8 in a file, or a {@link JsonView} type mismatch.
 *
 * <p>Errors found in the input text carry the offset where they were found ({@link #position()}), and their
 * message ends with {@code " at position <n>"}. Errors about an already parsed value (typed access) have no
 * position.</p>
 */
public final class JsonException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /** {@link #position()} of an error that is not tied to an input offset. */
    public static final long NO_POSITION = -1L;

    private final String detail;
    private final long position;

    /** An error without an input position; {@code message} is the whole message. */
    public JsonException(String message) {
        this(message, NO_POSITION, null);
    }

    /**
     * An error found at {@code position} of the input.
     *
     * @param detail   what is wrong, without the position
     * @param position character offset into the text (byte offset for encoding errors), or {@link #NO_POSITION}
     */
    public JsonException(String detail, long position) {
        this(detail, position, null);
    }

    /** As {@link #JsonException(String, long)}, caused by {@code cause}. */
    public JsonException(String detail, long position, Throwable cause) {
        super(position >= 0 ? detail + " at position " + position : detail, cause);
        this.detail = detail;
        this.position = position >= 0 ? position : NO_POSITION;
    }

    /** Offset into the input where the error was found, or {@link #NO_POSITION}. */
    public long position() {
        return position;
    }

    /** The message without the trailing position. */
    public String detail() {
        return detail;
    }

    /** This error with {@code source + ": "} in front of its message (same position), caused by this error. */
    public JsonException withSource(String source) {
        return new JsonException(source + ": " + detail, position, this);
    }
}
