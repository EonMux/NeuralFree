package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The JSON and error plumbing every endpoint shares: bodies parse with the core JSON stack under the untrusted
 * limits, responses write in the compact API style, and errors carry the OpenAI envelope
 * {@code {"error": {"message", "type"}}} with the HTTP status.
 */
final class ServerJson {

    static final String NO_MODEL = "no model loaded: set -Dneuralfree.server.model=<gguf path>";

    private ServerJson() {
    }

    /** {@code value} as compact JSON text. */
    static String write(Object value) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(value, Json.Style.API, out);
        } catch (IOException e) {
            throw new IllegalStateException("a response body does not write as JSON", e);
        }
        return out.toString();
    }

    /** The request body, or a 400 naming the JSON error. */
    static JsonView parse(String bodyText) {
        if (bodyText == null || bodyText.isBlank()) {
            throw badRequest("the request body is empty; a JSON object is required");
        }
        try {
            return JsonView.of(Json.parse(bodyText, Json.Limits.UNTRUSTED), "body");
        } catch (RuntimeException e) {
            throw badRequest("invalid JSON body: " + e.getMessage());
        }
    }

    /** The error envelope. */
    static Map<String, Object> errorBody(String message, String type) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", message);
        error.put("type", type);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("error", error);
        return body;
    }

    /** An HTTP error carrying the envelope. */
    static WebApplicationException error(int status, String message, String type) {
        return new WebApplicationException(Response.status(status).type(MediaType.APPLICATION_JSON)
                .entity(write(errorBody(message, type))).build());
    }

    /** A 400 {@code invalid_request_error}. */
    static WebApplicationException badRequest(String message) {
        return error(400, message, "invalid_request_error");
    }

    /**
     * The probes' refusal ({@code /health}, {@code /ready}, {@code /ping}): 503 unavailable_error until a model is
     * open ({@link EngineService#ready}). A sleeping engine stays ready and the probe does not wake it.
     */
    static void requireReady(EngineService service) {
        if (!service.ready()) {
            throw error(503, NO_MODEL, "unavailable_error");
        }
    }

    /**
     * Refuses with 503 when the engine cannot serve, naming why: no model configured, an explicitly sleeping
     * engine, or a pause that rejects new work. An idle-sleeping engine wakes first — llama.cpp's server reloads
     * on the first task after sleeping.
     */
    static void requireLoaded(EngineService service) {
        if (!service.loaded() && service.wakesOnRequest()) {
            service.wakeAndWait();
        }
        if (!service.loaded() || service.servingRefusal() != null) {
            String refusal = service.servingRefusal();
            throw error(503, refusal != null ? refusal : NO_MODEL, "unavailable_error");
        }
    }

    /** One engine control operation the dev and RL routes run, and the JSON body a success answers. */
    interface Operation {
        Map<String, Object> run() throws InterruptedException;
    }

    /**
     * Runs one engine operation and writes its JSON body: a bad parameter or refusal names itself with 400, an
     * engine that cannot serve answers 503, and an operation the model's placement cannot take answers 409. An
     * interrupted wait answers 503 too and restores the flag.
     */
    static Response operate(Operation operation) {
        try {
            return Response.ok(write(operation.run())).build();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw error(503, "request interrupted", "unavailable_error");
        } catch (IllegalArgumentException e) {
            throw badRequest(e.getMessage());
        } catch (UnsupportedOperationException e) {
            throw error(409, e.getMessage(), "conflict_error");
        } catch (IllegalStateException e) {
            throw error(503, e.getMessage(), "unavailable_error");
        }
    }

    /** Writes one server-sent event: an optional {@code event:} line, then {@code data:} with the JSON body. */
    static void event(OutputStream out, String type, Object body) throws IOException {
        StringBuilder sb = new StringBuilder();
        if (type != null) {
            sb.append("event: ").append(type).append('\n');
        }
        sb.append("data: ").append(write(body)).append("\n\n");
        out.write(sb.toString().getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    /** Writes a raw {@code data:} line (the OpenAI stream's {@code [DONE]}). */
    static void data(OutputStream out, String data) throws IOException {
        out.write(("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }
}
