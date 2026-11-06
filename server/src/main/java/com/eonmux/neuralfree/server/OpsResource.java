package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.SecretProperties;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.UriInfo;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The operational endpoints of the references: vLLM's instrumentator and dev routers and sglang's control
 * endpoints.
 *
 * <ul>
 *   <li>{@code GET /version}: the server version ({@link ServerVersion}), as vLLM's {@code {"version": …}}.</li>
 *   <li>{@code GET /server_info}: the server's configuration and state, vLLM's config-and-env report over the
 *       {@code neuralfree.*} settings, always as JSON ({@code config_format=json}; vLLM's {@code text} rendering
 *       refuses by name).</li>
 *   <li>{@code GET /load}: the requests in flight ({@code server_load}, vLLM's count over the generation and
 *       pooling routes; {@link EngineService#load}).</li>
 *   <li>{@code GET /ready} and {@code GET /readyz}: 200 with an empty body once the model is open (a sleeping
 *       engine stays ready, as {@code /health} reports it, and the probe does not wake it), 503 with the error
 *       envelope before; vLLM's supervisor probes.</li>
 *   <li>{@code GET} and {@code POST /ping}: vLLM's SageMaker probe, which answers as {@code /health} does: an
 *       empty 200 while a model is open or the engine sleeps, 503 with the error envelope before.</li>
 *   <li>{@code GET /tokenizer_info}: the model tokenizer's facts, vLLM's tokenizer report ({@code tokenizer_class},
 *       the Hugging Face {@code bos_token}-style special token texts, {@code model_max_length},
 *       {@code chat_template}) with the GGUF ids beside them.</li>
 *   <li>{@code POST /reset_prefix_cache}: clears the prefix cache of every replica on its pump thread,
 *       {@code {"success": …}} as vLLM answers; a sleeping engine holds no cached prefix and stays asleep. The
 *       query parameters vLLM adds beyond a plain reset refuse by name. It is vLLM's development route
 *       ({@link DevMode}).</li>
 *   <li>{@code POST /reset_mm_cache}: drops the projector's media cache (the {@code --mm-processor-cache-gb}
 *       layers the media parts' {@code uuid} keys: encodings and decoded, preprocessed media), and
 *       {@code POST /reset_encoder_cache} every encoder output (the projector's encodings and each replica's
 *       shared encoder cache, keeping the processed media); both answer an empty 200 as vLLM does, also without a
 *       projector or a model (nothing to drop). A request body or query parameter refuses by name. Both are
 *       vLLM's development routes ({@link DevMode}).</li>
 *   <li>{@code POST /abort_requests}: aborts every live generation ({@code {"aborted": n}}, sglang's
 *       {@code abort_all}); a body naming anything else refuses by name. It is vLLM's development RL route
 *       ({@link DevMode}).</li>
 *   <li>{@code POST /start_profile} and {@code POST /stop_profile}: the engine profile window
 *       ({@link EngineProfiler}), a Java Flight Recorder recording written to the directory named by
 *       {@code neuralfree.server.profile.dir} (or the {@link ProfilerConfig}'s {@code torch_profiler_dir}) with
 *       the engine's counter deltas in the {@code /stop_profile} report. The routes answer 404 until
 *       {@link ProfilerConfig} names a profiler, as vLLM registers them only then. The profiler overrides vLLM
 *       takes refuse by name.</li>
 * </ul>
 *
 * <p>In router mode the per-model routes ({@code /server_info}, {@code /tokenizer_info}, the cache resets and the
 * profile window) take the model from {@code ?model=}; the probes answer while the router runs, {@code /load}
 * sums the open models' requests, and {@code /abort_requests} aborts every open model's generations.
 * {@code /server_info} without a model reports the router's own configuration. The switches gate a request
 * before the model query, so a gated route answers its refusal whether or not a model is named.</p>
 *
 * <p>{@code /server_info}, the cache resets and {@code /abort_requests} are vLLM's development routes: they serve
 * only with the {@link DevMode} switch on, and answer 404 without it. {@code /version}, {@code /load}, the probes
 * and {@code /tokenizer_info} are served by vLLM outside the dev mode and stay on. The profile window is not a
 * development route; it is gated by its own {@link ProfilerConfig} switch.</p>
 */
@Path("/")
public class OpsResource {

    @Inject
    EngineService service;

    @Inject
    EngineProfiler profiler;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The development-route switch; the dev routes answer 404 while it is off. */
    @Inject
    DevMode devMode;

    /** The profiler switch; the profile routes answer 404 while it names no profiler. */
    @Inject
    ProfilerConfig profilerConfig;

    /** Test and manual wiring: builds the resource over an explicit service, profiler and switches. */
    public OpsResource(EngineService service, EngineProfiler profiler, DevMode devMode,
                       ProfilerConfig profilerConfig) {
        this(service, profiler, null, devMode, profilerConfig);
    }

    /** Test and manual wiring over an explicit service, profiler, router and switches. */
    public OpsResource(EngineService service, EngineProfiler profiler, ModelRouter router, DevMode devMode,
                       ProfilerConfig profilerConfig) {
        this.service = service;
        this.profiler = profiler;
        this.router = router;
        this.devMode = devMode;
        this.profilerConfig = profilerConfig;
    }

    public OpsResource() {
    }

    /** Whether the router mode serves the request. */
    private boolean routerMode() {
        return router != null && router.active();
    }

    /** The engine the {@code ?model=} query names in router mode, else the wired service. */
    private EngineService target(String model, String autoload) {
        return ModelRouter.targetQuery(router, service, model, autoload);
    }

    // ------------------------------------------------------------------ state probes

    /** The server version. */
    @GET
    @Path("/version")
    @Produces(MediaType.APPLICATION_JSON)
    public Response version() {
        return Response.ok(ServerJson.write(Map.of("version", ServerVersion.VALUE))).build();
    }

    /** Whether the server is ready ({@link EngineService#ready}): an empty 200, else a 503 with the envelope. */
    @GET
    @Path("/ready")
    public Response ready() {
        if (routerMode()) {
            // the router itself is up: llama.cpp's router answers its health once the HTTP server runs
            return Response.ok().build();
        }
        ServerJson.requireReady(service);
        return Response.ok().build();
    }

    /** {@link #ready} under the Kubernetes {@code readyz} spelling. */
    @GET
    @Path("/readyz")
    public Response readyz() {
        return ready();
    }

    /** The SageMaker probe: {@code /health}'s answer (vLLM's ping delegates to it) with an empty body. */
    @GET
    @Path("/ping")
    public Response ping() {
        return ready();
    }

    /** {@link #ping} for POST probes (vLLM's SageMaker route answers both verbs). */
    @POST
    @Path("/ping")
    public Response pingPost() {
        return ping();
    }

    // ------------------------------------------------------------------ reporting

    /** The requests in flight (0 on an idle or unloaded server). */
    @GET
    @Path("/load")
    @Produces(MediaType.APPLICATION_JSON)
    public Response load() {
        return Response.ok(ServerJson.write(Map.of("server_load", routerMode() ? router.load()
                : service.load()))).build();
    }

    /** The server's configuration and state; {@code configFormat} may be null or {@code json}. */
    @GET
    @Path("/server_info")
    @Produces(MediaType.APPLICATION_JSON)
    public Response serverInfo(@QueryParam("config_format") String configFormat,
                               @QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        if (configFormat != null && !configFormat.equals("json")) {
            throw ServerJson.badRequest("config_format " + configFormat + " is not supported: the server reports"
                    + " its configuration as JSON only (config_format=json)");
        }
        if (routerMode() && (model == null || model.isBlank())) {
            return Response.ok(ServerJson.write(routerInfo())).build();
        }
        return serverInfoOf(target(model, autoload));
    }

    /** Test and manual wiring: the wired service's report, without a model query. */
    Response serverInfo(String configFormat) {
        return serverInfo(configFormat, null, null);
    }

    /** The router's own configuration report, as {@code /props} answers its role. */
    private Map<String, Object> routerInfo() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("version", ServerVersion.VALUE);
        out.put("role", "router");
        out.put("loaded", false);
        out.put("is_sleeping", false);
        out.put("model_path", "none");
        out.put("model_alias", "neuralfree-server");
        out.put("max_instances", router.modelsMax());
        out.put("models_autoload", router.autoload());
        out.put("models", router.list().size());
        out.put("neuralfree_env", environment());
        return out;
    }

    private Response serverInfoOf(EngineService service) {
        // a released model leaves the setup captured while it was open (what /props reports while sleeping)
        boolean loaded = service.loaded();
        EngineService.Props captured = loaded ? null : service.capturedProps();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("version", ServerVersion.VALUE);
        out.put("loaded", loaded);
        out.put("is_sleeping", service.isSleeping());
        out.put("model_path", service.modelPath());
        out.put("model_alias", service.served().primary());
        out.put("served_names", service.served().names());
        out.put("model_arch", captured != null ? captured.architecture() : service.architecture());
        out.put("context_length", loaded ? service.contextLength()
                : captured != null ? captured.contextLength() : 0);
        out.put("slots", service.slots());
        out.putAll(LlamaCppResource.dataParallel(service));
        out.put("backend", loaded && service.backendName() != null ? service.backendName()
                : service.config().backend().toString());
        out.put("gpu_fallback_reason", loaded ? service.gpuFallbackReason()
                : captured != null ? captured.gpuFallbackReason() : null);
        out.put("prefix_reuse", service.config().memory().prefixReuse());
        out.put("request_timeout_seconds", service.options().requestTimeout().toSeconds());
        out.put("spm_infill", service.options().spmInfill());
        out.put("response_store", service.options().responseStore());
        out.put("neuralfree_env", environment());
        return Response.ok(ServerJson.write(out)).build();
    }

    /** The model tokenizer's facts. */
    @GET
    @Path("/tokenizer_info")
    @Produces(MediaType.APPLICATION_JSON)
    public Response tokenizerInfo(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        return tokenizerInfoOf(target(model, autoload));
    }

    /** Test and manual wiring: the wired service's tokenizer, without a model query. */
    Response tokenizerInfo() {
        return tokenizerInfoOf(service);
    }

    private static Response tokenizerInfoOf(EngineService service) {
        ServerJson.requireLoaded(service);
        GGUFTokenizer tokenizer = service.tokenizer();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("tokenizer_class", tokenizer.tokenizerModel());
        out.put("tokenizer_pre", tokenizer.tokenizerPre());
        out.put("vocab_size", tokenizer.vocabSize());
        out.put("add_bos_token", tokenizer.vocab().addBos());
        out.put("add_eos_token", tokenizer.vocab().addEos());
        out.put("bos_token", tokenText(tokenizer, tokenizer.bosId()));
        out.put("eos_token", tokenText(tokenizer, tokenizer.eosId()));
        out.put("pad_token", tokenText(tokenizer, tokenizer.padId()));
        out.put("unk_token", tokenText(tokenizer, tokenizer.unkId()));
        out.put("bos_token_id", tokenizer.bosId());
        out.put("eos_token_id", tokenizer.eosId());
        out.put("pad_token_id", tokenizer.padId());
        out.put("unk_token_id", tokenizer.unkId());
        out.put("stop_token_ids", new ArrayList<>(tokenizer.stopTokenIds()));
        out.put("model_max_length", service.maxModelLength());
        out.put("chat_template", tokenizer.chatTemplate());
        return Response.ok(ServerJson.write(out)).build();
    }

    /** The vocabulary text of special token {@code id}, or null when the model has no such token. */
    private static String tokenText(GGUFTokenizer tokenizer, int id) {
        return id >= 0 && id < tokenizer.vocabSize() ? tokenizer.getToken(id) : null;
    }

    /**
     * The {@code neuralfree.*} system settings (vLLM's {@code vllm_env}); a setting
     * {@link SecretProperties#isSecret(String)} marks secret stays out entirely, so its value never reaches a
     * client. That covers the API keys, the RPC entries with their inline keys, the watermark configuration and
     * the cloud credentials.
     */
    private Map<String, Object> environment() {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String name : System.getProperties().stringPropertyNames()) {
            if (!name.startsWith("neuralfree.") || SecretProperties.isSecret(name)) {
                continue;
            }
            out.put(name, System.getProperty(name));
        }
        return out;
    }

    // ------------------------------------------------------------------ control

    /**
     * Clears the prefix cache; {@code success} is false while sequences hold pages of the memory, as vLLM's reset
     * reports. vLLM's {@code reset_running_requests} and {@code reset_external} refuse by name.
     */
    @POST
    @Path("/reset_prefix_cache")
    @Produces(MediaType.APPLICATION_JSON)
    public Response resetPrefixCache(@QueryParam("reset_running_requests") boolean resetRunningRequests,
                                     @QueryParam("reset_external") boolean resetExternal,
                                     @QueryParam("model") String model,
                                     @QueryParam("autoload") String autoload) {
        devMode.require();
        return resetPrefixCacheOf(resetRunningRequests, resetExternal, target(model, autoload));
    }

    /** Test and manual wiring: the reset on the wired service, without a model query. */
    Response resetPrefixCache(boolean resetRunningRequests, boolean resetExternal) {
        devMode.require();
        return resetPrefixCacheOf(resetRunningRequests, resetExternal, service);
    }

    private static Response resetPrefixCacheOf(boolean resetRunningRequests, boolean resetExternal,
                                               EngineService service) {
        if (resetRunningRequests) {
            throw ServerJson.badRequest("reset_running_requests is not supported: running requests keep their"
                    + " pages until they finish (abort them through /abort_requests)");
        }
        if (resetExternal) {
            throw ServerJson.badRequest("reset_external is not supported: the server keeps no external KV cache");
        }
        ServerJson.requireReady(service);
        boolean success;
        try {
            success = service.resetPrefixCache();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        } catch (IllegalStateException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        }
        return Response.ok(ServerJson.write(Map.of("success", success))).build();
    }

    /**
     * Drops the projector's media cache (vLLM's dev route): the next request decodes and encodes its media anew.
     * Takes no body or query parameter.
     */
    @POST
    @Path("/reset_mm_cache")
    public Response resetMmCache(String bodyText, @Context UriInfo uri,
                                 @QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        devMode.require();
        return resetMmCacheOf(bodyText, uri, target(model, autoload));
    }

    /** Test and manual wiring: the media-cache reset on the wired service, without a model query. */
    Response resetMmCache(String bodyText, UriInfo uri) {
        devMode.require();
        return resetMmCacheOf(bodyText, uri, service);
    }

    private static Response resetMmCacheOf(String bodyText, UriInfo uri, EngineService service) {
        requireNoArguments("reset_mm_cache", bodyText, uri);
        service.resetMediaCache();
        return Response.ok().build();
    }

    /**
     * Drops every encoder output (vLLM's dev route): the next request of a medium runs its encoder again. Takes no
     * body or query parameter.
     */
    @POST
    @Path("/reset_encoder_cache")
    public Response resetEncoderCache(String bodyText, @Context UriInfo uri,
                                      @QueryParam("model") String model,
                                      @QueryParam("autoload") String autoload) {
        devMode.require();
        return resetEncoderCacheOf(bodyText, uri, target(model, autoload));
    }

    /** Test and manual wiring: the encoder-cache reset on the wired service, without a model query. */
    Response resetEncoderCache(String bodyText, UriInfo uri) {
        devMode.require();
        return resetEncoderCacheOf(bodyText, uri, service);
    }

    private static Response resetEncoderCacheOf(String bodyText, UriInfo uri, EngineService service) {
        requireNoArguments("reset_encoder_cache", bodyText, uri);
        try {
            service.resetEncoderCache();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        }
        return Response.ok().build();
    }

    /** Refuses a body or a query parameter on a route that takes neither, naming the first one. */
    private static void requireNoArguments(String route, String bodyText, UriInfo uri) {
        if (uri != null) {
            for (String name : uri.getQueryParameters().keySet()) {
                if (!name.equals("model") && !name.equals("autoload")) {
                    throw ServerJson.badRequest(route + " takes no " + name + ": it drops the whole cache");
                }
            }
        }
        if (bodyText != null && !bodyText.isBlank()) {
            JsonView body = ServerJson.parse(bodyText);
            if (!body.keys().isEmpty()) {
                throw ServerJson.badRequest(route + " takes no " + body.keys().iterator().next()
                        + ": it drops the whole cache");
            }
        }
    }

    /**
     * Aborts every live generation; each one's client sees {@code finish_reason: "abort"}. The body may be empty
     * or carry only {@code abort_all} (sglang's switch, true by default; false aborts nothing).
     */
    @POST
    @Path("/abort_requests")
    @Produces(MediaType.APPLICATION_JSON)
    public Response abortRequests(String bodyText) {
        devMode.require();
        boolean all = true;
        if (bodyText != null && !bodyText.isBlank()) {
            JsonView body = ServerJson.parse(bodyText);
            for (String field : body.keys()) {
                if (!field.equals("abort_all")) {
                    throw ServerJson.badRequest("abort_requests takes no " + field + ": requests carry no"
                            + " client-visible ids; the call aborts all of them");
                }
            }
            try {
                all = body.optBool("abort_all", true);
            } catch (JsonException e) {
                throw ServerJson.badRequest(e.getMessage());
            }
        }
        int aborted = !all ? 0 : routerMode() ? router.abortAll() : service.abortAll();
        return Response.ok(ServerJson.write(Map.of("aborted", aborted))).build();
    }

    /** Opens the profile window. The profiler overrides vLLM takes refuse by name. */
    @POST
    @Path("/start_profile")
    public Response startProfile(String bodyText, @QueryParam("model") String model,
                                 @QueryParam("autoload") String autoload) {
        profilerConfig.require();
        return startProfileOf(bodyText, target(model, autoload));
    }

    /** Test and manual wiring: the profile window on the wired service, without a model query. */
    Response startProfile(String bodyText) {
        profilerConfig.require();
        return startProfileOf(bodyText, service);
    }

    private Response startProfileOf(String bodyText, EngineService service) {
        if (bodyText != null && !bodyText.isBlank()) {
            JsonView body = ServerJson.parse(bodyText);
            if (!body.keys().isEmpty()) {
                throw ServerJson.badRequest("start_profile takes no " + body.keys().iterator().next()
                        + ": the server keeps no profiler overrides (vLLM's profile_prefix, delay_iterations,"
                        + " max_iterations, profiler_kwargs)");
            }
        }
        try {
            profiler.start(service.stats());
        } catch (IllegalStateException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        }
        return Response.ok().build();
    }

    /** Closes the profile window and reports the engine's deltas over it. */
    @POST
    @Path("/stop_profile")
    @Produces(MediaType.APPLICATION_JSON)
    public Response stopProfile(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        profilerConfig.require();
        return stopProfileOf(target(model, autoload));
    }

    /** Test and manual wiring: the profile report of the wired service, without a model query. */
    Response stopProfile() {
        profilerConfig.require();
        return stopProfileOf(service);
    }

    private Response stopProfileOf(EngineService service) {
        Map<String, Object> report;
        try {
            report = profiler.stop(service.stats());
        } catch (IllegalStateException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        }
        return Response.ok(ServerJson.write(report)).build();
    }
}
