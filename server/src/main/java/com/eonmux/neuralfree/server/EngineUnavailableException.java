package com.eonmux.neuralfree.server;

/**
 * A generation refused on the engine's serving state rather than on the request: no model loaded, an explicit
 * sleep, or a pause that rejects new work. The message names the route that clears the state; the HTTP layer
 * answers it with 503, where a refusal of the request itself (an oversize prompt, an unusable grammar) is a 400.
 */
public final class EngineUnavailableException extends IllegalStateException {

    private static final long serialVersionUID = 1L;

    /** A refusal explained by {@code message}. */
    public EngineUnavailableException(String message) {
        super(message);
    }
}
