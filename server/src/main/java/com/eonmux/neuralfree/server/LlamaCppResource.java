package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The llama.cpp-native endpoints (PARITY plan P2 step 3): {@code GET /health} reports the load state and
 * {@code GET /props} the serving setup (model path, architecture, context length, sequence slots, the sampler
 * defaults a request takes when it leaves a setting unset). The Quarkus health extension serves the Kubernetes
 * probes separately under {@code /q/health}.
 */
@Path("/")
public class LlamaCppResource {

    @Inject
    EngineService service;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public LlamaCppResource(EngineService service) {
        this.service = service;
    }

    public LlamaCppResource() {
    }

    @GET
    @Path("/health")
    @Produces(MediaType.APPLICATION_JSON)
    public Response health() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", service.loaded() ? "ok" : "error");
        if (!service.loaded()) {
            body.put("error", "no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
        return Response.ok(json(body)).build();
    }

    @GET
    @Path("/props")
    @Produces(MediaType.APPLICATION_JSON)
    public Response props() {
        if (!service.loaded()) {
            throw new WebApplicationException(Response.status(503).type(MediaType.APPLICATION_JSON)
                    .entity("{\"error\":\"no model loaded: set -Dneuralfree.server.model=<gguf path>\"}").build());
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model_path", service.modelPath());
        body.put("model_arch", service.architecture());
        Map<String, Object> total = new LinkedHashMap<>();
        total.put("n_ctx", service.contextLength());
        total.put("n_slots", service.slots());
        body.put("total_slots", service.slots());
        body.put("default_generation_settings", total);
        return Response.ok(json(body)).build();
    }

    private String json(Map<String, Object> body) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(body, Json.Style.API, out);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return out.toString();
    }
}
