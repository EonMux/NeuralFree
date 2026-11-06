package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.util.NoSuchElementException;

/**
 * The gRPC status mapping of the serving adapter: the engine's refusals become the status that carries the same
 * meaning as their HTTP counterparts — a refused request is {@code INVALID_ARGUMENT}, an unloaded, sleeping or
 * paused engine {@code UNAVAILABLE}, a refusal of the serving state {@code FAILED_PRECONDITION}, an unknown model
 * or adapter {@code NOT_FOUND}, and an engine path the request kind does not serve {@code UNIMPLEMENTED}. Every
 * status carries the message the HTTP envelope would have carried, so a client sees the same name of the field
 * or the switch it has to change.
 */
final class GrpcFailures {

    private GrpcFailures() {
    }

    /** A refusal of the request by name, as {@code INVALID_ARGUMENT}. */
    static StatusRuntimeException invalid(String message) {
        return Status.INVALID_ARGUMENT.withDescription(message).asRuntimeException();
    }

    /** {@code field} is not supported by this server, as {@code UNIMPLEMENTED}. */
    static StatusRuntimeException unsupported(String message) {
        return Status.UNIMPLEMENTED.withDescription(message).asRuntimeException();
    }

    /** The status of {@code error}; a status already built is returned as it is. */
    static StatusRuntimeException of(Throwable error) {
        if (error instanceof StatusRuntimeException status) {
            return status;
        }
        if (error instanceof EngineUnavailableException unavailable) {
            return Status.UNAVAILABLE.withDescription(messageOf(unavailable)).asRuntimeException();
        }
        if (error instanceof WebApplicationException web) {
            Status status = statusOf(web.getResponse().getStatus());
            return status.withDescription(webMessage(web)).asRuntimeException();
        }
        if (error instanceof NoSuchFileException || error instanceof NoSuchElementException) {
            return Status.NOT_FOUND.withDescription(messageOf(error)).asRuntimeException();
        }
        if (error instanceof UnsupportedOperationException) {
            return Status.UNIMPLEMENTED.withDescription(messageOf(error)).asRuntimeException();
        }
        if (error instanceof IllegalArgumentException || error instanceof NumberFormatException) {
            return Status.INVALID_ARGUMENT.withDescription(messageOf(error)).asRuntimeException();
        }
        if (error instanceof IllegalStateException) {
            return Status.FAILED_PRECONDITION.withDescription(messageOf(error)).asRuntimeException();
        }
        if (error instanceof InterruptedException) {
            Thread.currentThread().interrupt();
            return Status.CANCELLED.withDescription(messageOf(error)).asRuntimeException();
        }
        if (error instanceof IOException) {
            return Status.INTERNAL.withDescription(messageOf(error)).asRuntimeException();
        }
        return Status.INTERNAL.withDescription(messageOf(error)).asRuntimeException();
    }

    private static Status statusOf(int http) {
        switch (http) {
            case 400:
                return Status.INVALID_ARGUMENT;
            case 401:
                return Status.UNAUTHENTICATED;
            case 403:
                return Status.PERMISSION_DENIED;
            case 404:
                return Status.NOT_FOUND;
            case 409:
                return Status.FAILED_PRECONDITION;
            case 429:
                return Status.RESOURCE_EXHAUSTED;
            case 501:
                return Status.UNIMPLEMENTED;
            case 503:
                return Status.UNAVAILABLE;
            default:
                return Status.INTERNAL;
        }
    }

    /** The message of a {@link ServerJson} error envelope, else the response entity as it stands. */
    private static String webMessage(WebApplicationException error) {
        Response response = error.getResponse();
        Object entity = response == null ? null : response.getEntity();
        if (entity instanceof String json) {
            try {
                JsonView view = JsonView.of(Json.parse(json, Json.Limits.UNTRUSTED), "error");
                String message = view.obj("error").str("message");
                if (message != null && !message.isEmpty()) {
                    return message;
                }
            } catch (RuntimeException ignored) {
                // not an envelope: report the entity itself
            }
            if (!json.isEmpty()) {
                return json;
            }
        }
        return messageOf(error);
    }

    private static String messageOf(Throwable error) {
        String message = error.getMessage();
        return message == null || message.isEmpty() ? error.getClass().getSimpleName() : message;
    }
}
