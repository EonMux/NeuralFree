package com.eonmux.neuralfree.server;

/**
 * A generation refused because every replica already holds {@link ServerOptions#maxQueuedRequests} requests waiting
 * to run (SGLang's {@code --max-queued-requests}): the request itself is fine and may be sent again shortly, so the
 * HTTP layer answers 429 with a {@code Retry-After} of {@link #retryAfterSeconds()} seconds.
 */
public final class EngineSaturatedException extends IllegalStateException {

    private static final long serialVersionUID = 1L;

    /** The seconds a refused client is told to wait before it sends the request again. */
    static final int RETRY_AFTER_SECONDS = 1;

    /** A refusal explained by {@code message}. */
    public EngineSaturatedException(String message) {
        super(message);
    }

    /** The {@code Retry-After} seconds of the refusal. */
    public int retryAfterSeconds() {
        return RETRY_AFTER_SECONDS;
    }
}
