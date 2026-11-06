package com.eonmux.neuralfree.server;

import io.vertx.core.http.HttpServerRequest;

/**
 * Cancels a request's generations when its client goes away before the response is written: the Vert.x response
 * close handler fires when the connection closes (or the HTTP/2 stream resets) while the response still uses it. A
 * stream also notices on its next failed write; the handler covers the non-streamed wait, which writes nothing until
 * the end.
 */
final class ClientLink {

    private ClientLink() {
    }

    /**
     * Runs {@code onClose} if the connection of {@code request} closes first; a null request (a resource built
     * outside the HTTP layer) links nothing.
     */
    static void onClose(HttpServerRequest request, Runnable onClose) {
        if (request == null) {
            return;
        }
        request.response().closeHandler(ignored -> onClose.run());
    }
}
