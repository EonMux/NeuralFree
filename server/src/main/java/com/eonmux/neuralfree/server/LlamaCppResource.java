package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The llama.cpp-native endpoints, with the vLLM spellings of the shared ones:
 *
 * <ul>
 *   <li>{@code GET /health} (also {@code /v1/health}) reports the load state with llama.cpp's semantics: HTTP 200
 *       {@code {"status": "ok"}} while a model is open or the engine sleeps (with {@code is_sleeping}), HTTP 503
 *       unavailable_error before a model is open; {@code GET /props} the serving setup (model path, architecture,
 *       context, slots, chat template, the media {@code modalities} and {@code media_marker}, the web UI fields
 *       {@code ui} and {@code ui_settings}) plus
 *       {@code is_sleeping} (llama.cpp reports it there too, and serves the cached setup while sleeping). Both
 *       carry the data-parallel layout ({@link #dataParallel}): the global {@code data_parallel_size}, the local
 *       replica count, the global rank of the first local replica (vLLM's {@code --data-parallel-rank}, so a
 *       server behind an external balancer names its rank), the balancing mode and the load-balance method;
 *       {@code /props} also carries the per-replica {@code replicas} past one replica. The Quarkus health
 *       extension serves the Kubernetes probes separately under {@code /q/health}.</li>
 *   <li>{@code POST /tokenize}: llama.cpp's {@code content} (special tokens off by default,
 *       {@code with_pieces} adds each token's text) or vLLM's {@code prompt} (special tokens on by default) or
 *       chat {@code messages} rendered through the template, their image, audio and embedding parts as the media
 *       prompt with one {@code -1} (llama.cpp's {@code LLAMA_TOKEN_NULL}) per media row and a null piece; the
 *       answer carries llama.cpp's {@code tokens} and vLLM's {@code count} and {@code max_model_len}.</li>
 *   <li>{@code POST /detokenize}: {@code tokens} to text, as llama.cpp's {@code content} and vLLM's
 *       {@code prompt}.</li>
 *   <li>{@code POST /apply-template}: a chat body rendered through the template, as {@code prompt}; image and
 *       audio parts render as the {@code media_marker}, and the answer carries the rendering template's
 *       {@code chat_template_caps} (the request's {@code chat_template} override's when it carries one).</li>
 * </ul>
 */
@Path("/")
public class LlamaCppResource {

    /** The id a media row takes in a tokenized prompt, llama.cpp's {@code LLAMA_TOKEN_NULL}. */
    static final int MEDIA_ROW = -1;

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public LlamaCppResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public LlamaCppResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    public LlamaCppResource() {
    }

    @GET
    @Path("/health")
    @Produces(MediaType.APPLICATION_JSON)
    public Response health() {
        if (router != null && router.active()) {
            // the router itself is up: llama.cpp's router answers its health once the HTTP server runs
            return Response.ok(ServerJson.write(Map.of("status", "ok"))).build();
        }
        // llama.cpp's middleware: until a model is open every endpoint answers 503 unavailable_error; a sleeping
        // engine answers again the moment a request wakes it, so it stays healthy
        ServerJson.requireReady(service);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", "ok");
        body.put("is_sleeping", service.isSleeping());
        body.putAll(dataParallel(service));
        return Response.ok(ServerJson.write(body)).build();
    }

    @GET
    @Path("/v1/health")
    @Produces(MediaType.APPLICATION_JSON)
    public Response healthAlias() {
        return health();
    }

    @GET
    @Path("/props")
    @Produces(MediaType.APPLICATION_JSON)
    public Response props(@QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        if (router != null && router.active()) {
            if (model == null || model.isBlank()) {
                // llama.cpp's router props (get_router_props): the router has no model of its own
                Map<String, Object> routerBody = new LinkedHashMap<>();
                routerBody.put("role", "router");
                routerBody.put("max_instances", router.modelsMax());
                routerBody.put("models_autoload", router.autoload());
                routerBody.put("model_alias", "neuralfree-server");
                routerBody.put("model_path", "none");
                routerBody.put("default_generation_settings", Map.of("n_ctx", 0));
                return Response.ok(ServerJson.write(routerBody)).build();
            }
            EngineService target = ModelRouter.targetQuery(router, service, model, autoload);
            ServerJson.requireLoaded(target);
            return propsOf(target);
        }
        return propsOf(service);
    }

    /** The serving setup of one service ({@link #props} over the routed target). */
    private Response propsOf(EngineService service) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("is_sleeping", service.isSleeping());
        if (service.isSleeping()) {
            // llama.cpp serves its cached meta while the context is destroyed; the captured values answer here
            EngineService.Props captured = service.capturedProps();
            body.put("model_alias", service.served().primary());
            body.put("model_path", service.modelPath());
            body.put("model_arch", captured != null ? captured.architecture() : null);
            body.put("backend", captured != null ? captured.backend() : null);
            body.put("gpu_fallback_reason", captured != null ? captured.gpuFallbackReason() : null);
            body.put("total_slots", captured != null ? captured.slots() : 0);
            body.putAll(dataParallel(service));
            Map<String, Object> total = new LinkedHashMap<>();
            total.put("n_ctx", captured != null ? captured.maxModelLength() : 0);
            body.put("default_generation_settings", total);
            body.put("chat_template", captured != null ? captured.chatTemplate() : null);
            body.put("chat_template_caps", captured != null ? captured.chatTemplateCaps().toMap() : null);
            putMedia(body);
            putWebUi(body);
            return Response.ok(ServerJson.write(body)).build();
        }
        ServerJson.requireLoaded(service);
        int[] replicaSlots = service.replicaSlots();
        body.put("model_alias", service.served().primary());
        body.put("model_path", service.modelPath());
        body.put("model_arch", service.architecture());
        body.put("backend", service.backendName());
        body.put("gpu_fallback_reason", service.gpuFallbackReason());
        Map<String, Object> total = new LinkedHashMap<>();
        total.put("n_ctx", service.contextLength());
        total.put("n_slots", service.slots());
        body.put("total_slots", service.slots());
        body.put("default_generation_settings", total);
        body.put("chat_template", service.chatTemplate());
        body.put("chat_template_caps", service.pooling().dualEncoder() ? null : service.chatTemplateCaps().toMap());
        putMedia(body);
        putWebUi(body);
        appendReplicas(body, replicaSlots);
        return Response.ok(ServerJson.write(body)).build();
    }

    /**
     * The web UI fields of llama.cpp's {@code /props}: {@code ui} (whether the bundled UI serves, llama.cpp's
     * {@code params.ui}) and {@code ui_settings} (the {@code --ui-config} defaults, an empty object without one).
     */
    private void putWebUi(Map<String, Object> body) {
        body.put("ui", UiResource.enabled(System.getProperties()));
        body.put("ui_settings", UiResource.uiSettings(System.getProperties()));
    }

    /**
     * llama.cpp's media capabilities and the marker a raw multimodal prompt places its media at; a sleeping
     * engine reports them too (its released projector keeps its capabilities, as llama.cpp's cached meta does).
     */
    private void putMedia(Map<String, Object> body) {
        Map<String, Object> modalities = new LinkedHashMap<>();
        modalities.put("vision", service.media().vision());
        modalities.put("video", service.media().video());
        modalities.put("audio", service.media().audio());
        body.put("modalities", modalities);
        body.put("media_marker", ChatMessages.MEDIA_MARKER);
    }

    /**
     * The data-parallel layout fields: {@code data_parallel_size} (the global size, 0 while unloaded),
     * {@code data_parallel_size_local} (the replicas here) and, once loaded, {@code data_parallel_rank} (the global
     * rank of the first local replica), {@code data_parallel_external_lb} (whether other servers hold the other
     * ranks), {@code data_parallel_mode} ({@code internal}, {@code external} or {@code hybrid}) and
     * {@code load_balance_method}.
     */
    static Map<String, Object> dataParallel(EngineService service) {
        Map<String, Object> out = new LinkedHashMap<>();
        int local = service.replicaCount();
        out.put("data_parallel_size", service.dataParallelSize());
        out.put("data_parallel_size_local", local);
        if (local == 0) {
            return out;
        }
        DpLayout layout = service.dataParallelLayout();
        out.put("data_parallel_rank", service.replicaRank(0));
        out.put("data_parallel_external_lb", layout.externalLb(local));
        out.put("data_parallel_mode", layout.mode(local));
        out.put("load_balance_method", layout.balance().label());
        return out;
    }

    /**
     * The data-parallel layout ({@link #dataParallel}); with more than one replica, each one's global rank,
     * device, state, slots and load.
     */
    private void appendReplicas(Map<String, Object> body, int[] replicaSlots) {
        List<BatchStats> replicaStats = service.replicaStats();
        body.putAll(dataParallel(service));
        if (replicaStats.size() <= 1) {
            return;
        }
        List<Object> replicas = new ArrayList<>(replicaStats.size());
        for (int i = 0; i < replicaStats.size(); i++) {
            BatchStats stats = replicaStats.get(i);
            Map<String, Object> replica = new LinkedHashMap<>();
            replica.put("replica", i);
            replica.put("rank", service.replicaRank(i));
            replica.put("device", service.replicaDevice(i));
            replica.put("model_path", service.modelPath());
            replica.put("status", "ok");
            replica.put("n_ctx", service.contextLength());
            replica.put("n_slots", replicaSlots[i]);
            replica.put("queued", stats.waiting());
            replica.put("running", stats.running());
            replica.put("kv_cache_usage", stats.kvUsage());
            replica.put("prompt_tokens_in_flight", service.replicaPromptTokensInFlight(i));
            replicas.add(replica);
        }
        body.put("replicas", replicas);
    }

    @POST
    @Path("/tokenize")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response tokenize(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> tokenizeOf(target, body));
    }

    @POST
    @Path("/detokenize")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response detokenize(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> {
            if (!(body.raw("tokens") instanceof List<?> list)) {
                throw ServerJson.badRequest("tokens is required and must be an array of token ids");
            }
            String text = list.isEmpty() ? "" : target.decode(OpenAiRequests.tokens(list, target.vocabSize(),
                    "tokens"));
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("content", text);
            out.put("prompt", text);
            return Response.ok(ServerJson.write(out)).build();
        });
    }

    @POST
    @Path("/apply-template")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response applyTemplate(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> {
            ChatRequest chat = OpenAiRequests.chat(body, target.options());
            // media parts are checked and loaded like a generation's, then render as their markers
            boolean media = !ChatPlan.mediaItems(target, chat, mediaLorasOf(target, body)).isEmpty();
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("prompt", target.formatChat(chat, media));
            // the capabilities of the template that rendered: the request's chat_template override's when it
            // carries one, else the model template's
            out.put("chat_template_caps", target.chatTemplateCaps(chat).toMap());
            return Response.ok(ServerJson.write(out)).build();
        });
    }

    private Response tokenizeOf(EngineService service, JsonView body) {
        int[] tokens;
        if (body.has("messages")) {
            tokens = chatTokens(OpenAiRequests.chat(body, service.options()),
                    body.optBool("add_special_tokens", false), mediaLorasOf(service, body));
        } else if (body.has("content")) {
            tokens = service.encode(body.str("content"), body.optBool("add_special", false),
                    body.optBool("parse_special", true));
        } else if (body.has("prompt")) {
            tokens = service.encode(body.str("prompt"), body.optBool("add_special_tokens", true), true);
        } else {
            throw ServerJson.badRequest("tokenize takes content (llama.cpp), prompt or messages (vLLM)");
        }
        boolean pieces = body.optBool("with_pieces", false);
        List<Object> ids = new ArrayList<>(tokens.length);
        for (int id : tokens) {
            if (pieces) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("id", id);
                entry.put("piece", id == MEDIA_ROW ? null : service.tokenText(id));
                ids.add(entry);
            } else {
                ids.add(id);
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("tokens", ids);
        out.put("count", tokens.length);
        out.put("max_model_len", service.maxModelLength());
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * The tokens of a chat's rendered prompt. With image, audio or embedding parts the prompt is the media prompt
     * the generation runs: its text tokens, and one {@link #MEDIA_ROW} per media row, as llama.cpp's
     * {@code server_tokens} holds them, so the count is the KV rows the prompt fills.
     */
    private int[] chatTokens(ChatRequest chat, boolean addSpecial, Supplier<LoraSlotKey> mediaLoras) {
        List<ChatMedia.Item> items = ChatPlan.mediaItems(service, chat, mediaLoras);
        String rendered = service.formatChat(chat, !items.isEmpty());
        if (items.isEmpty() && !rendered.contains(ChatMessages.PROMPT_EMBEDS_MARKER)) {
            return service.encode(rendered, addSpecial, true);
        }
        MediaPrompt media = ChatPlan.mediaPrompt(service, chat, rendered, items, mediaLoras);
        int bos = addSpecial ? service.addedBos() : -1;
        int[] out = new int[media.nTokens() + (bos >= 0 ? 1 : 0)];
        int at = 0;
        if (bos >= 0) {
            out[at++] = bos;
        }
        for (MediaChunk chunk : media.chunks()) {
            if (chunk.isText()) {
                int[] text = chunk.tokens();
                System.arraycopy(text, 0, out, at, text.length);
                at += text.length;
            } else {
                Arrays.fill(out, at, at + chunk.nTokens(), MEDIA_ROW);
                at += chunk.nTokens();
            }
        }
        return out;
    }

    /**
     * The request's media-session adapters, resolved when its first media part asks; null when it names none, so
     * its media encodes under the startup adapters' keys.
     */
    private static Supplier<LoraSlotKey> mediaLorasOf(EngineService service, JsonView body) {
        return ChatPlan.mediaLoras(service, service.loras().selection(body, service.loras().model(body)));
    }
}
