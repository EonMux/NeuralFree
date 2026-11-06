package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * The per-request LoRA adapter endpoints ({@link LoraAdmin}):
 *
 * <ul>
 *   <li>{@code POST /v1/load_lora_adapter} {@code {"lora_name", "lora_path", "load_inplace"}} and
 *       {@code POST /v1/unload_lora_adapter} {@code {"lora_name"}}: vLLM's runtime adapter loading, answering its
 *       plain-text success line. They serve only when runtime updating is on ({@link ServerLoras}); otherwise they
 *       answer 404 as vLLM's unregistered routes do, naming the switch. A missing path answers 404, an unusable
 *       adapter or a startup adapter's name 400, and an adapter running requests use (or the adapter limit filled
 *       by such adapters) 409.</li>
 *   <li>{@code GET /lora-adapters}: llama.cpp's adapter list, {@code [{"id", "path", "scale", "task_name",
 *       "prompt_prefix", "name"}]} in load order — the startup adapters first — the scale being the global one a
 *       request that selects no adapter runs at, the task name and prompt prefix the adapter's PEFT metadata (empty
 *       when it has none), {@code alora_invocation_string} and {@code alora_invocation_tokens} added for an
 *       activated LoRA, and the name the one a request's {@code model} selects it by; {@code POST /lora-adapters}
 *       {@code [{"id", "scale"}]} sets the global scales, every unlisted adapter to 0 ({@code {"success": true}}).
 *       </li>
 * </ul>
 *
 * <p>In router mode every route takes the model from {@code ?model=} (with {@code ?autoload=} to override the
 * server's), as llama.cpp's router proxies them; a missing name is a 400 and an unknown one a 404.</p>
 */
@Path("/")
public class LoraResource {

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public LoraResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public LoraResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    public LoraResource() {
    }

    /** The engine the {@code ?model=} query names in router mode, else the wired service. */
    private EngineService target(String model, String autoload) {
        return ModelRouter.targetQuery(router, service, model, autoload);
    }

    @POST
    @Path("/v1/load_lora_adapter")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces({MediaType.TEXT_PLAIN, MediaType.APPLICATION_JSON})
    public Response load(String bodyText, @QueryParam("model") String model,
                         @QueryParam("autoload") String autoload) {
        return loadOf(bodyText, target(model, autoload));
    }

    /** Test and manual wiring: the load on the wired service, without a model query. */
    Response load(String bodyText) {
        return loadOf(bodyText, service);
    }

    private Response loadOf(String bodyText, EngineService service) {
        service.loras().requireRuntimeUpdating();
        ServerJson.requireLoaded(service);
        JsonView body = ServerJson.parse(bodyText);
        String name = text(body, "lora_name");
        String path = text(body, "lora_path");
        if (name == null || path == null) {
            throw invalidInput("Both 'lora_name' and 'lora_path' must be provided.");
        }
        boolean inPlace = body.optBool("load_inplace", false);
        return changed(() -> service.loras().load(name, path, inPlace),
                "Success: LoRA adapter '" + name + "' added successfully.");
    }

    @POST
    @Path("/v1/unload_lora_adapter")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces({MediaType.TEXT_PLAIN, MediaType.APPLICATION_JSON})
    public Response unload(String bodyText, @QueryParam("model") String model,
                           @QueryParam("autoload") String autoload) {
        return unloadOf(bodyText, target(model, autoload));
    }

    /** Test and manual wiring: the unload on the wired service, without a model query. */
    Response unload(String bodyText) {
        return unloadOf(bodyText, service);
    }

    private Response unloadOf(String bodyText, EngineService service) {
        service.loras().requireRuntimeUpdating();
        ServerJson.requireLoaded(service);
        JsonView body = ServerJson.parse(bodyText);
        String name = text(body, "lora_name");
        if (name == null) {
            throw invalidInput("'lora_name' needs to be provided to unload a LoRA adapter.");
        }
        return changed(() -> service.loras().unload(name),
                "Success: LoRA adapter '" + name + "' removed successfully.");
    }

    @GET
    @Path("/lora-adapters")
    @Produces(MediaType.APPLICATION_JSON)
    public Response list(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        return listOf(target(model, autoload));
    }

    /** Test and manual wiring: the list of the wired service, without a model query. */
    Response list() {
        return listOf(service);
    }

    private Response listOf(EngineService service) {
        ServerJson.requireLoaded(service);
        List<Object> out = new ArrayList<>();
        List<LoraCatalog.Adapter> adapters = service.loras().list();
        for (int i = 0; i < adapters.size(); i++) {
            LoraCatalog.Adapter a = adapters.get(i);
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("id", i);
            entry.put("path", a.path());
            entry.put("scale", (double) a.scale());
            LoraCatalog.Metadata meta = a.metadata();
            entry.put("task_name", meta.taskName() == null ? "" : meta.taskName());
            entry.put("prompt_prefix", meta.promptPrefix() == null ? "" : meta.promptPrefix());
            if (!meta.invocationTokens().isEmpty()) {
                entry.put("alora_invocation_string", meta.invocationString());
                entry.put("alora_invocation_tokens", meta.invocationTokens());
            }
            entry.put("name", a.name());
            out.add(entry);
        }
        return Response.ok(ServerJson.write(out)).build();
    }

    @POST
    @Path("/lora-adapters")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response setScales(String bodyText, @QueryParam("model") String model,
                              @QueryParam("autoload") String autoload) {
        return setScalesOf(bodyText, target(model, autoload));
    }

    /** Test and manual wiring: the scale change on the wired service, without a model query. */
    Response setScales(String bodyText) {
        return setScalesOf(bodyText, service);
    }

    private Response setScalesOf(String bodyText, EngineService service) {
        ServerJson.requireLoaded(service);
        Object body;
        try {
            body = Json.parse(bodyText == null ? "" : bodyText, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw ServerJson.badRequest("invalid JSON body: " + e.getMessage());
        }
        try {
            service.loras().setScales(body);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        return Response.ok(ServerJson.write(Map.of("success", true))).build();
    }

    /** A change of the adapters; its refusals map to vLLM's statuses. */
    @FunctionalInterface
    private interface Change {
        void run() throws IOException, InterruptedException;
    }

    private static Response changed(Change change, String success) {
        try {
            change.run();
        } catch (NoSuchFileException | NoSuchElementException e) {
            throw ServerJson.error(404, messageOf(e), "NotFoundError");
        } catch (IOException | IllegalArgumentException | UnsupportedOperationException e) {
            throw invalidInput(String.valueOf(e.getMessage()));
        } catch (IllegalStateException e) {
            throw ServerJson.error(409, String.valueOf(e.getMessage()), "ConflictError");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "interrupted while changing the LoRA adapters", "unavailable_error");
        }
        return Response.ok(success, MediaType.TEXT_PLAIN).build();
    }

    private static String messageOf(Exception e) {
        return e instanceof NoSuchFileException missing && missing.getReason() != null ? missing.getReason()
                : String.valueOf(e.getMessage());
    }

    /** The non-blank string {@code key} of {@code body}, or null when absent or blank. */
    private static String text(JsonView body, String key) {
        Object raw = body.raw(key);
        if (raw != null && !(raw instanceof String)) {
            throw invalidInput("'" + key + "' must be a string");
        }
        String value = (String) raw;
        return value == null || value.isBlank() ? null : value;
    }

    private static jakarta.ws.rs.WebApplicationException invalidInput(String message) {
        return ServerJson.error(400, message, "InvalidUserInput");
    }
}
