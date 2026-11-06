package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The live weight transfer endpoints (vLLM's online RL routes, its in-process weight transfer engine): a trainer
 * drives the lifecycle {@code POST /init_weight_transfer_engine} → {@code POST /start_weight_update} →
 * {@code POST /update_weights} → {@code POST /finish_weight_update} → {@code POST
 * /shutdown_weight_transfer_engine} over every replica, and {@code GET /weight_info} reports the committed
 * {@code weight_version} ({@code default} until the first stamp). {@code POST /update_weight_version} stamps it
 * without writing weights. The tensors arrive in the request bodies as base64 ({@code data_b64}) beside their
 * stored name, ggml element type ({@code dtype}) and ggml shape ({@code ne}, {@code ne[0]} first) — one tensor
 * object, or a list of them for the parallel form, bare or under vLLM's {@code update_info} wrapper. The writes
 * land between the engine's steps and drop the idle engine's cached prefixes, so the next request serves the new
 * weights.
 *
 * <p>The only registered backend is the in-process one ({@code inprocess}); an initialization naming a
 * cross-process transport refuses by name. In router mode every route takes the model from {@code ?model=} (with
 * {@code ?autoload=} to override the server's), so a transfer touches one model's engine; a missing name is a 400
 * and an unknown one a 404.</p>
 *
 * <p>Every route is one of vLLM's development RL routes: it serves only with the {@link DevMode} switch on, and
 * answers 404 without it.</p>
 */
@Path("/")
public class WeightTransferResource {

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The development-route switch; every route here answers 404 while it is off. */
    @Inject
    DevMode devMode;

    /** Test and manual wiring: builds the resource over an explicit service and dev-mode switch. */
    public WeightTransferResource(EngineService service, DevMode devMode) {
        this(service, null, devMode);
    }

    /** Test and manual wiring over an explicit service, router and dev-mode switch. */
    public WeightTransferResource(EngineService service, ModelRouter router, DevMode devMode) {
        this.service = service;
        this.router = router;
        this.devMode = devMode;
    }

    public WeightTransferResource() {
    }

    /** The engine the {@code ?model=} query names in router mode, else the wired service. */
    private EngineService target(String model, String autoload) {
        return ModelRouter.targetQuery(router, service, model, autoload);
    }

    @POST
    @Path("/init_weight_transfer_engine")
    @Produces(MediaType.APPLICATION_JSON)
    public Response initWeightTransferEngine(String body, @QueryParam("model") String model,
                                             @QueryParam("autoload") String autoload) {
        devMode.require();
        return initWeightTransferEngineOf(body, target(model, autoload));
    }

    /** Test and manual wiring: the initialization on the wired service, without a model query. */
    Response initWeightTransferEngine(String body) {
        devMode.require();
        return initWeightTransferEngineOf(body, service);
    }

    private Response initWeightTransferEngineOf(String body, EngineService service) {
        ServerJson.requireLoaded(service);
        String backend = WeightTransferBodies.initBackend(body);
        return ServerJson.operate(() -> {
            String version = service.weightTransfer().init(backend);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("message", "Weight transfer initialized");
            out.put("backend", backend == null || backend.isBlank() ? WeightUpdater.INPROCESS_BACKEND
                    : backend.trim());
            out.put("weight_version", version);
            return out;
        });
    }

    @POST
    @Path("/shutdown_weight_transfer_engine")
    @Produces(MediaType.APPLICATION_JSON)
    public Response shutdownWeightTransferEngine(String body, @QueryParam("model") String model,
                                                 @QueryParam("autoload") String autoload) {
        devMode.require();
        return shutdownWeightTransferEngineOf(body, target(model, autoload));
    }

    /** Test and manual wiring: the shutdown on the wired service, without a model query. */
    Response shutdownWeightTransferEngine(String body) {
        devMode.require();
        return shutdownWeightTransferEngineOf(body, service);
    }

    private Response shutdownWeightTransferEngineOf(String body, EngineService service) {
        requireNoFields(body);
        return ServerJson.operate(() -> {
            service.weightTransfer().shutdown();
            return Map.of("message", "Weight transfer shut down");
        });
    }

    @POST
    @Path("/start_weight_update")
    @Produces(MediaType.APPLICATION_JSON)
    public Response startWeightUpdate(String body, @QueryParam("model") String model,
                                      @QueryParam("autoload") String autoload) {
        devMode.require();
        return startWeightUpdateOf(body, target(model, autoload));
    }

    /** Test and manual wiring: the round start on the wired service, without a model query. */
    Response startWeightUpdate(String body) {
        devMode.require();
        return startWeightUpdateOf(body, service);
    }

    private Response startWeightUpdateOf(String body, EngineService service) {
        ServerJson.requireLoaded(service);
        requireNoFields(body);
        return ServerJson.operate(() -> {
            service.weightTransfer().start();
            return Map.of("message", "Weight update started");
        });
    }

    @POST
    @Path("/update_weights")
    @Produces(MediaType.APPLICATION_JSON)
    public Response updateWeights(String body, @QueryParam("model") String model,
                                  @QueryParam("autoload") String autoload) {
        devMode.require();
        return updateWeightsOf(body, target(model, autoload));
    }

    /** Test and manual wiring: the update on the wired service, without a model query. */
    Response updateWeights(String body) {
        devMode.require();
        return updateWeightsOf(body, service);
    }

    private Response updateWeightsOf(String body, EngineService service) {
        ServerJson.requireLoaded(service);
        List<WeightUpdateTensor> tensors = WeightTransferBodies.updateTensors(body);
        return ServerJson.operate(() -> {
            service.weightTransfer().apply(tensors);
            return Map.of("message", "Weights updated", "tensors", tensors.size());
        });
    }

    @POST
    @Path("/finish_weight_update")
    @Produces(MediaType.APPLICATION_JSON)
    public Response finishWeightUpdate(String body, @QueryParam("model") String model,
                                       @QueryParam("autoload") String autoload) {
        devMode.require();
        return finishWeightUpdateOf(body, target(model, autoload));
    }

    /** Test and manual wiring: the finish on the wired service, without a model query. */
    Response finishWeightUpdate(String body) {
        devMode.require();
        return finishWeightUpdateOf(body, service);
    }

    private Response finishWeightUpdateOf(String body, EngineService service) {
        ServerJson.requireLoaded(service);
        String version = WeightTransferBodies.optionalVersion(body, "weight_version");
        return ServerJson.operate(() -> Map.of("message", "Weight update finished", "weight_version",
                service.weightTransfer().finish(version)));
    }

    @POST
    @Path("/update_weight_version")
    @Produces(MediaType.APPLICATION_JSON)
    public Response updateWeightVersion(String body, @QueryParam("model") String model,
                                        @QueryParam("autoload") String autoload) {
        devMode.require();
        return updateWeightVersionOf(body, target(model, autoload));
    }

    /** Test and manual wiring: the version stamp on the wired service, without a model query. */
    Response updateWeightVersion(String body) {
        devMode.require();
        return updateWeightVersionOf(body, service);
    }

    private Response updateWeightVersionOf(String body, EngineService service) {
        ServerJson.requireLoaded(service);
        JsonView view = ServerJson.parse(body);
        WeightTransferBodies.requireKnownKeys(view, Set.of("new_version"), "the body", "the field is new_version");
        String version = WeightTransferBodies.requiredStr(view, "new_version", "the body");
        return ServerJson.operate(() -> Map.of("success", true, "new_version",
                service.weightTransfer().setVersion(version)));
    }

    @GET
    @Path("/weight_info")
    @Produces(MediaType.APPLICATION_JSON)
    public Response weightInfo(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        return weightInfoOf(target(model, autoload));
    }

    /** Test and manual wiring: the weight info of the wired service, without a model query. */
    Response weightInfo() {
        devMode.require();
        return weightInfoOf(service);
    }

    private static Response weightInfoOf(EngineService service) {
        ServerJson.requireLoaded(service);
        EngineWeightTransfer transfer = service.weightTransfer();
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("weight_version", transfer.version());
        body.put("backend", WeightUpdater.INPROCESS_BACKEND);
        body.put("initialized", transfer.initialized());
        return Response.ok(ServerJson.write(body)).build();
    }

    // ------------------------------------------------------------------ body parsing

    /** Refuses a non-blank body: this route takes no fields. */
    private static void requireNoFields(String bodyText) {
        if (bodyText == null || bodyText.isBlank()) {
            return;
        }
        WeightTransferBodies.requireKnownKeys(ServerJson.parse(bodyText), Set.of(), "the body",
                "this route takes no fields");
    }

}
