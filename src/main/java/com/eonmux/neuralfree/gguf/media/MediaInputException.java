package com.eonmux.neuralfree.gguf.media;

/**
 * A media input the request cannot use: a malformed or refused URL, a fetch that failed, a payload over the size
 * limit, or more items than the per-prompt limit allows. The message names the offending field or URL. Serving
 * layers answer it as a client error.
 */
public final class MediaInputException extends IllegalArgumentException {

    private static final long serialVersionUID = 1L;

    /** An exception with {@code message}. */
    public MediaInputException(String message) {
        super(message);
    }

    /** An exception with {@code message} caused by {@code cause}. */
    public MediaInputException(String message, Throwable cause) {
        super(message, cause);
    }
}
