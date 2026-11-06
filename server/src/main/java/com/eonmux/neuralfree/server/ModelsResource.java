package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.DELETE;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * The router's management endpoints (llama.cpp's router routes, server.cpp's {@code ctx_http} custom routes):
 *
 * <ul>
 *   <li>{@code POST /models/load} with {@code {"model": name}} opens the model's engine and answers
 *       {@code {"success": true}}; an unknown model is a 404, an already running one a 400.</li>
 *   <li>{@code POST /models/unload} with {@code {"model": name}} closes it; an unknown model is a 404, one that
 *       holds no engine a 400.</li>
 *   <li>{@code POST /models} with {@code {"model": <gguf path>, "alias"?, "tags"?}} registers the file under its
 *       file name for routing and removal (llama.cpp's POST downloads a Hugging Face repo; the references' router
 *       serves local files, and so does this one).</li>
 *   <li>{@code DELETE /models?model=<name>} removes a registered model again, closing its engine first; a model a
 *       source directory or preset declares refuses (400) — the server configuration keeps it.</li>
 *   <li>{@code GET /models/sse} streams every entry's state change as
 *       {@code data: {"model", "event", "data"}} frames, a comment line between the events.</li>
 * </ul>
 *
 * <p>Without the router mode ({@code neuralfree.server.models.dir} and {@code .preset} unset) every route refuses
 * with 400 naming how to turn the mode on; {@code GET /models} then serves the single model's list, as always.
 */
@Path("/")
public class ModelsResource {

    /** The SSE events' idle gap, long enough to detect a gone client, short enough to feel live. */
    private static final int SSE_PING_SECONDS = 15;

    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit router. */
    public ModelsResource(ModelRouter router) {
        this.router = router;
    }

    public ModelsResource() {
    }

    @POST
    @Path("/models/load")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response load(String bodyText) {
        requireRouter();
        String name = modelName(ServerJson.parse(bodyText));
        return operate(() -> {
            router.load(name);
            return Map.of("success", true);
        });
    }

    @POST
    @Path("/models/unload")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response unload(String bodyText) {
        requireRouter();
        String name = modelName(ServerJson.parse(bodyText));
        return operate(() -> {
            router.unload(name);
            return Map.of("success", true);
        });
    }

    @POST
    @Path("/models")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response register(String bodyText) {
        requireRouter();
        JsonView body = ServerJson.parse(bodyText);
        java.nio.file.Path path = java.nio.file.Path.of(modelName(body));
        List<String> aliases = strings(body, "alias");
        List<String> tags = strings(body, "tag");
        return operate(() -> {
            ModelRouter.Entry entry = router.register(path, aliases, tags);
            return Map.of("success", true, "model", entry.name());
        });
    }

    @DELETE
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    public Response unregister(@QueryParam("model") String name) {
        requireRouter();
        if (name == null || name.isBlank()) {
            throw ServerJson.badRequest("model is required: the name of the model to remove");
        }
        return operate(() -> {
            router.unregister(name.trim());
            return Map.of("success", true, "model", name.trim());
        });
    }

    @GET
    @Path("/models/sse")
    @Produces("text/event-stream")
    public Response sse() {
        requireRouter();
        LinkedBlockingQueue<ModelRouter.Event> queue = router.subscribe();
        StreamingOutput stream = out -> {
            try {
                drain(out, queue);
            } finally {
                router.unsubscribe(queue);
            }
        };
        return Response.ok(stream).type("text/event-stream").build();
    }

    /** Streams the subscribed events until the client goes away: a frame per event, a comment between. */
    private static void drain(OutputStream out, LinkedBlockingQueue<ModelRouter.Event> queue) throws IOException {
        try {
            while (true) {
                ModelRouter.Event event = queue.poll(SSE_PING_SECONDS, TimeUnit.SECONDS);
                if (event != null) {
                    ServerJson.event(out, null, event.wire());
                } else {
                    out.write((": ping\n\n").getBytes(StandardCharsets.UTF_8));
                    out.flush();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * The router's model list ({@code GET /models} and {@code GET /v1/models} in router mode, llama.cpp's
     * {@code get_router_models}): one entry per model with its id, source, tags and status —
     * {@code unloaded} (with {@code failed} and {@code error} after a failed load), {@code loading}, {@code loaded}
     * or {@code sleeping} — and its context once the engine runs.
     */
    public static Map<String, Object> listing(ModelRouter router) {
        List<Object> data = new ArrayList<>();
        for (ModelRouter.Entry entry : router.list()) {
            Map<String, Object> model = new LinkedHashMap<>();
            model.put("id", entry.name());
            model.put("aliases", entry.aliases());
            model.put("tags", entry.tags());
            model.put("object", "model");
            model.put("owned_by", "neuralfree");
            model.put("created", 0L);
            model.put("status", status(entry));
            model.put("source", entry.source().label());
            model.put("can_remove", entry.source() == ModelRouter.Source.REGISTERED);
            int context = entry.contextLength();
            if (context > 0) {
                model.put("context_length", context);
            }
            data.add(model);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", "list");
        out.put("data", data);
        return out;
    }

    /** The {@code status} object of one entry: the value, and the failure of a load that did not open. */
    private static Map<String, Object> status(ModelRouter.Entry entry) {
        Map<String, Object> status = new LinkedHashMap<>();
        if (entry.ready() && entry.sleeping()) {
            status.put("value", "sleeping");
        } else {
            status.put("value", entry.status().name().toLowerCase(java.util.Locale.ROOT));
        }
        String error = entry.error();
        if (error != null) {
            status.put("failed", true);
            status.put("error", error);
        }
        return status;
    }

    // ------------------------------------------------------------------ plumbing

    /** 400 when the router mode is off, naming the properties that turn it on. */
    private void requireRouter() {
        if (router == null || !router.active()) {
            throw ServerJson.badRequest("router mode is off: set -D" + ModelRouter.DIR_PROPERTY + "=<dir> or -D "
                    + ModelRouter.PRESET_PROPERTY + "=<models.ini> to serve several models");
        }
    }

    /** Runs one management operation and writes its JSON body (the mapping {@code OpenAiResource.mapped} uses). */
    private interface Operation {
        Map<String, Object> run() throws IOException;
    }

    private static Response operate(Operation operation) {
        try {
            return Response.ok(ServerJson.write(operation.run())).build();
        } catch (IOException e) {
            throw ServerJson.error(503, String.valueOf(e.getMessage()), "unavailable_error");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        } catch (EngineUnavailableException e) {
            throw ServerJson.error(503, String.valueOf(e.getMessage()), "unavailable_error");
        }
    }

    /** The {@code model} value of a body: a non-blank string, else a 400 naming the field. */
    private static String modelName(JsonView body) {
        Object raw = body.raw("model");
        if (!(raw instanceof String name) || name.isBlank()) {
            throw ServerJson.badRequest("model must be a non-empty string");
        }
        return name.trim();
    }

    /** The strings of a body field: the value itself, or each entry of its array; null when absent. */
    private static List<String> strings(JsonView body, String field) {
        Object raw = body.raw(field);
        if (raw == null) {
            return List.of();
        }
        if (raw instanceof String value) {
            return value.isBlank() ? List.of() : List.of(value.trim());
        }
        if (!(raw instanceof List<?> values)) {
            throw ServerJson.badRequest(field + " must be a string or an array of strings");
        }
        List<String> out = new ArrayList<>(values.size());
        for (Object value : values) {
            if (!(value instanceof String text) || text.isBlank()) {
                throw ServerJson.badRequest(field + " must hold non-empty strings");
            }
            out.add(text.trim());
        }
        return out;
    }
}
