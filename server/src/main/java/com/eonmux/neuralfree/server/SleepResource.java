package com.eonmux.neuralfree.server;

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

/**
 * The serving lifecycle endpoints (vLLM's sleep mode, its RL pause routes and its checkpoint routes, llama.cpp's
 * idle sleep):
 *
 * <ul>
 *   <li>{@code POST /sleep}?level=0..2&amp;mode=abort|wait|keep pauses the scheduling and releases memory at the
 *       level (0 scheduling only, 1 the KV cache contents, 2 the weights too); {@code POST /wake_up}?tags=
 *       restores it; {@code GET /is_sleeping} reports it. A sleep also pauses the engine, so generation refuses
 *       with 503 until the wake.</li>
 *   <li>{@code POST /pause}?mode=...&amp;clear_cache=... and {@code POST /resume} gate the scheduling on their own
 *       axis; {@code GET /is_paused} reports it. The abort and wait modes reject new requests with 503 until the
 *       resume; the keep mode keeps queueing them but schedules nothing.</li>
 *   <li>{@code POST /release_kv_cache_memory} drops the KV cache contents of an idle engine.</li>
 *   <li>{@code POST /checkpoint/prepare}?mode=..., {@code POST /checkpoint/restore} and {@code GET
 *       /checkpoint/status} run the capture window: prepared engines hold no transient state, and the status
 *       answers 200 once prepared, 503 before and 500 after a failure.</li>
 * </ul>
 *
 * <p>With {@code neuralfree.server.sleep.idle.seconds} set, the engine also sleeps by itself after so many
 * seconds without work (llama.cpp's {@code --sleep-idle-seconds}) and the next request wakes it; {@code /props}
 * reports {@code is_sleeping}.
 *
 * <p>In router mode every route takes the model from {@code ?model=} (with {@code ?autoload=} to override the
 * server's), so a sleep or wake touches one model's engine; a missing name is a 400 and an unknown one a 404.
 * The switches gate the request before the model query, so a gated route answers its refusal whether or not a
 * model is named.</p>
 *
 * <p>The sleep, release and pause routes are vLLM's development routes: they serve only with the {@link DevMode}
 * switch on, and answer 404 without it. The sleep operations themselves need sleep mode as well: without
 * {@link SleepModeSwitch}, {@code /sleep}, {@code /wake_up} and {@code /release_kv_cache_memory} answer the
 * sleep-mode refusal. The checkpoint window is vLLM's sleep-mode route: it is not a development route, and answers
 * 404 without sleep mode as an unregistered vLLM route does.</p>
 */
@Path("/")
public class SleepResource {

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The development-route switch; the sleep and pause routes answer 404 while it is off. */
    @Inject
    DevMode devMode;

    /** The sleep-mode switch; the sleep operations and the checkpoint window need it. */
    @Inject
    SleepModeSwitch sleepMode;

    /** Test and manual wiring: builds the resource over an explicit service and its switches. */
    public SleepResource(EngineService service, DevMode devMode, SleepModeSwitch sleepMode) {
        this(service, null, devMode, sleepMode);
    }

    /** Test and manual wiring over an explicit service, router and switches. */
    public SleepResource(EngineService service, ModelRouter router, DevMode devMode, SleepModeSwitch sleepMode) {
        this.service = service;
        this.router = router;
        this.devMode = devMode;
        this.sleepMode = sleepMode;
    }

    public SleepResource() {
    }

    /** The engine the {@code ?model=} query names in router mode, else the wired service. */
    private EngineService target(String model, String autoload) {
        return ModelRouter.targetQuery(router, service, model, autoload);
    }

    @POST
    @Path("/sleep")
    @Produces(MediaType.APPLICATION_JSON)
    public Response sleep(@QueryParam("level") String level, @QueryParam("mode") String mode,
                          @QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        sleepMode.requireSleep();
        return sleepOf(level, mode, target(model, autoload));
    }

    /** Test and manual wiring: the sleep on the wired service, without a model query. */
    Response sleep(String level, String mode) {
        devMode.require();
        sleepMode.requireSleep();
        return sleepOf(level, mode, service);
    }

    private Response sleepOf(String level, String mode, EngineService service) {
        requireModelOrSleep(service);
        int parsedLevel = wholeNumber(level, "level", 1);
        SleepControl.Mode parsedMode = mode(mode, "mode", SleepControl.Mode.ABORT);
        return operate(() -> {
            service.sleep(parsedLevel, parsedMode);
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("status", "sleeping");
            body.put("level", parsedLevel);
            return body;
        });
    }

    @POST
    @Path("/wake_up")
    @Produces(MediaType.APPLICATION_JSON)
    public Response wakeUp(@QueryParam("tags") List<String> tags, @QueryParam("model") String model,
                           @QueryParam("autoload") String autoload) {
        devMode.require();
        sleepMode.requireSleep();
        return wakeUpOf(tags, target(model, autoload));
    }

    /** Test and manual wiring: the wake on the wired service, without a model query. */
    Response wakeUp(List<String> tags) {
        devMode.require();
        sleepMode.requireSleep();
        return wakeUpOf(tags, service);
    }

    private Response wakeUpOf(List<String> tags, EngineService service) {
        requireModelOrSleep(service);
        List<String> given = tags == null ? List.of() : tags;
        return operate(() -> {
            String status = service.wakeUp(given);
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("status", status);
            body.put("tags", given);
            return body;
        });
    }

    @GET
    @Path("/is_sleeping")
    @Produces(MediaType.APPLICATION_JSON)
    public Response isSleeping(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        return isSleepingOf(target(model, autoload));
    }

    /** Test and manual wiring: the sleep state of the wired service, without a model query. */
    Response isSleeping() {
        devMode.require();
        return isSleepingOf(service);
    }

    private static Response isSleepingOf(EngineService service) {
        return Response.ok(ServerJson.write(Map.of("is_sleeping", service.isSleeping()))).build();
    }

    @POST
    @Path("/release_kv_cache_memory")
    @Produces(MediaType.APPLICATION_JSON)
    public Response releaseKvCacheMemory(@QueryParam("model") String model,
                                         @QueryParam("autoload") String autoload) {
        devMode.require();
        sleepMode.requireSleep();
        return releaseKvCacheMemoryOf(target(model, autoload));
    }

    /** Test and manual wiring: the release on the wired service, without a model query. */
    Response releaseKvCacheMemory() {
        devMode.require();
        sleepMode.requireSleep();
        return releaseKvCacheMemoryOf(service);
    }

    private static Response releaseKvCacheMemoryOf(EngineService service) {
        ServerJson.requireLoaded(service);
        return operate(() -> {
            service.releaseKvCacheMemory();
            return Map.of("status", "kv_cache_released");
        });
    }

    @POST
    @Path("/pause")
    @Produces(MediaType.APPLICATION_JSON)
    public Response pause(@QueryParam("mode") String mode, @QueryParam("clear_cache") String clearCache,
                          @QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        return pauseOf(mode, clearCache, target(model, autoload));
    }

    /** Test and manual wiring: the pause on the wired service, without a model query. */
    Response pause(String mode, String clearCache) {
        devMode.require();
        return pauseOf(mode, clearCache, service);
    }

    private static Response pauseOf(String mode, String clearCache, EngineService service) {
        ServerJson.requireLoaded(service);
        SleepControl.Mode parsedMode = mode(mode, "mode", SleepControl.Mode.ABORT);
        boolean parsedClearCache = clearCache == null || clearCache.isBlank()
                ? true : bool(clearCache, "clear_cache");
        return operate(() -> {
            service.pause(parsedMode, parsedClearCache);
            return Map.of("status", "paused");
        });
    }

    @POST
    @Path("/resume")
    @Produces(MediaType.APPLICATION_JSON)
    public Response resume(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        return resumeOf(target(model, autoload));
    }

    /** Test and manual wiring: the resume on the wired service, without a model query. */
    Response resume() {
        devMode.require();
        return resumeOf(service);
    }

    private static Response resumeOf(EngineService service) {
        return operate(() -> {
            service.resume();
            return Map.of("status", "resumed");
        });
    }

    @GET
    @Path("/is_paused")
    @Produces(MediaType.APPLICATION_JSON)
    public Response isPaused(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        return isPausedOf(target(model, autoload));
    }

    /** Test and manual wiring: the pause state of the wired service, without a model query. */
    Response isPaused() {
        devMode.require();
        return isPausedOf(service);
    }

    private static Response isPausedOf(EngineService service) {
        return Response.ok(ServerJson.write(Map.of("is_paused", service.isPaused()))).build();
    }

    @POST
    @Path("/checkpoint/prepare")
    @Produces(MediaType.APPLICATION_JSON)
    public Response checkpointPrepare(@QueryParam("mode") String mode, @QueryParam("model") String model,
                                      @QueryParam("autoload") String autoload) {
        sleepMode.requireCheckpoint();
        return checkpointPrepareOf(mode, target(model, autoload));
    }

    /** Test and manual wiring: the checkpoint preparation on the wired service, without a model query. */
    Response checkpointPrepare(String mode) {
        sleepMode.requireCheckpoint();
        return checkpointPrepareOf(mode, service);
    }

    private static Response checkpointPrepareOf(String mode, EngineService service) {
        requireModelOrSleep(service);
        SleepControl.Mode parsedMode = mode(mode, "mode", SleepControl.Mode.WAIT);
        return operate(() -> {
            String state = service.prepareCheckpoint(parsedMode);
            return stateBody(state, null);
        });
    }

    @POST
    @Path("/checkpoint/restore")
    @Produces(MediaType.APPLICATION_JSON)
    public Response checkpointRestore(@QueryParam("model") String model,
                                      @QueryParam("autoload") String autoload) {
        sleepMode.requireCheckpoint();
        return checkpointRestoreOf(target(model, autoload));
    }

    /** Test and manual wiring: the checkpoint restore on the wired service, without a model query. */
    Response checkpointRestore() {
        sleepMode.requireCheckpoint();
        return checkpointRestoreOf(service);
    }

    private static Response checkpointRestoreOf(EngineService service) {
        requireModelOrSleep(service);
        return operate(() -> {
            String state = service.restoreCheckpoint();
            return stateBody(state, null);
        });
    }

    @GET
    @Path("/checkpoint/status")
    @Produces(MediaType.APPLICATION_JSON)
    public Response checkpointStatus(@QueryParam("model") String model,
                                     @QueryParam("autoload") String autoload) {
        sleepMode.requireCheckpoint();
        return checkpointStatusOf(target(model, autoload));
    }

    /** Test and manual wiring: the checkpoint state of the wired service, without a model query. */
    Response checkpointStatus() {
        sleepMode.requireCheckpoint();
        return checkpointStatusOf(service);
    }

    private static Response checkpointStatusOf(EngineService service) {
        EngineService.CheckpointStatus status = service.checkpointStatus();
        boolean prepared = status.state().equals("prepared");
        boolean failed = status.state().equals("failed");
        String error = status.error() != null ? status.error()
                : !service.loaded() && !service.isSleeping() ? ServerJson.NO_MODEL : null;
        Map<String, Object> body = stateBody(status.state(), error);
        if (prepared) {
            return Response.ok(ServerJson.write(body)).build();
        }
        return Response.status(failed ? 500 : 503).type(MediaType.APPLICATION_JSON)
                .entity(ServerJson.write(body)).build();
    }

    // ------------------------------------------------------------------ plumbing

    /**
     * Runs one lifecycle operation and writes its JSON body: a bad parameter or refusal names itself with 400,
     * an engine that cannot serve answers 503, and a state conflict (a release beside running requests) answers
     * 409. An interrupted wait answers 503 too and restores the flag.
     */
    private interface Operation {
        Map<String, Object> run() throws InterruptedException;
    }

    private static Response operate(Operation operation) {
        try {
            return Response.ok(ServerJson.write(operation.run())).build();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(409, e.getMessage(), "conflict_error");
        } catch (IllegalStateException e) {
            throw ServerJson.error(503, e.getMessage(), "unavailable_error");
        }
    }

    /** The vLLM capture response: the state and, when present, the error. */
    private static Map<String, Object> stateBody(String state, String error) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("state", state);
        if (error != null) {
            body.put("error", error);
        }
        return body;
    }

    /** 503 when the service never had a model; a sleeping engine answers its lifecycle routes instead. */
    private static void requireModelOrSleep(EngineService service) {
        if (!service.loaded() && !service.isSleeping()) {
            ServerJson.requireLoaded(service);
        }
    }

    private static int wholeNumber(String raw, String field, int fallback) {
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw ServerJson.badRequest(field + " must be a whole number, got '" + raw + "'");
        }
    }

    private static boolean bool(String raw, String field) {
        if (raw.equalsIgnoreCase("true")) {
            return true;
        }
        if (raw.equalsIgnoreCase("false")) {
            return false;
        }
        throw ServerJson.badRequest(field + " must be true or false, got '" + raw + "'");
    }

    private static SleepControl.Mode mode(String raw, String field, SleepControl.Mode fallback) {
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return SleepControl.Mode.valueOf(raw.trim().toUpperCase(java.util.Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(field + " must be one of abort, wait, keep, got '" + raw + "'");
        }
    }
}
