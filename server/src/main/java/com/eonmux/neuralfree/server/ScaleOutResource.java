package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The vLLM scale-out serving surface:
 *
 * <ul>
 *   <li>Render: {@code POST /v1/chat/completions/render}, {@code /v1/completions/render},
 *       {@code /v1/messages/render} and {@code /v1/responses/render} answer the rendered prompt's token ids and the
 *       sampling parameters ({@link ScaleOutRender}), so a prompt-serving front end and a token-in, token-out
 *       engine can run on separate hosts.</li>
 *   <li>Derender: {@code POST /v1/chat/completions/derender} and {@code /v1/completions/derender} turn a generate
 *       response back into the OpenAI chat or completions object without a model forward
 *       ({@link ScaleOutDerender}); {@code stream: true} derenders one generate chunk per call with a client-held
 *       {@code stream_state} ({@link ScaleOutDerenderStream}).</li>
 *   <li>Generate: {@code POST /inference/v1/generate} runs a token prompt with sampling parameters and answers
 *       tokens or text, non-streamed or as server-sent events ({@link ScaleOutGenerate}); a client cancels a live
 *       request by its id through {@code POST /inference/v1/abort_requests}.</li>
 * </ul>
 *
 * The prompt token-count routes ({@code /v1/chat/completions/input_tokens}, {@code /v1/responses/input_tokens} and
 * their llama.cpp aliases) live on {@link OpenAiResource} and {@link ResponsesResource}; they count the same
 * rendered prompt rows the chat and responses endpoints generate.
 */
@Path("/")
public class ScaleOutResource {

    @Inject
    EngineService service;

    /** The current HTTP request, for disconnect cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The live generate batches by request id, for {@code /inference/v1/abort_requests}. */
    private final Map<String, GenerationBatch> active = new ConcurrentHashMap<>();

    /** Test and manual wiring: builds the resource over an explicit service. */
    public ScaleOutResource(EngineService service) {
        this(service, null, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public ScaleOutResource(EngineService service, ModelRouter router) {
        this(service, router, null);
    }

    /** A resource over an explicit service, router and request, so a stream cancels on disconnect. */
    public ScaleOutResource(EngineService service, ModelRouter router, HttpServerRequest http) {
        this.service = service;
        this.router = router;
        this.http = http;
    }

    public ScaleOutResource() {
    }

    @POST
    @Path("/v1/chat/completions/render")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response renderChat(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        return OpenAiResource.mapped(() -> ScaleOutRender.chat(target, body, "chatcmpl-" + baseId(body)));
    }

    @POST
    @Path("/v1/completions/render")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response renderCompletions(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        return OpenAiResource.mapped(() -> ScaleOutRender.completions(target, body, "cmpl-" + baseId(body)));
    }

    @POST
    @Path("/v1/messages/render")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response renderMessages(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        return OpenAiResource.mapped(() -> ScaleOutRender.messages(target, body, "chatcmpl-" + baseId(body)));
    }

    @POST
    @Path("/v1/responses/render")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response renderResponses(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        return OpenAiResource.mapped(() -> ScaleOutRender.responses(target, body, "resp_" + baseId(body)));
    }

    @POST
    @Path("/v1/chat/completions/derender")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response derenderChat(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        return OpenAiResource.mapped(() -> body.optBool("stream", false)
                ? ScaleOutDerenderStream.chat(target, body, target.loras().model(body))
                : ScaleOutDerender.chat(target, body, target.loras().model(body)));
    }

    @POST
    @Path("/v1/completions/derender")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response derenderCompletions(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        return OpenAiResource.mapped(() -> body.optBool("stream", false)
                ? ScaleOutDerenderStream.completions(target, body, target.loras().model(body))
                : ScaleOutDerender.completions(target, body, target.loras().model(body)));
    }

    @POST
    @Path("/inference/v1/generate")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response generate(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        return OpenAiResource.mapped(() -> ScaleOutGenerate.generate(target, body, http, active));
    }

    /**
     * Cancels the live generate requests named by {@code request_ids} and answers an empty 200, as vLLM's
     * {@code /inference/v1/abort_requests} does; ids with no live request are ignored.
     */
    @POST
    @Path("/inference/v1/abort_requests")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response abortRequests(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        Object raw = body.raw("request_ids");
        if (!(raw instanceof List<?> ids)) {
            throw ServerJson.badRequest("request_ids is required and must be an array of request ids");
        }
        for (Object id : ids) {
            if (!(id instanceof String name)) {
                throw ServerJson.badRequest("request_ids must hold request id strings");
            }
            GenerationBatch batch = active.get(name);
            if (batch != null) {
                batch.cancel();
            }
        }
        return Response.ok().build();
    }

    /** The engine a request serves on, loaded. */
    private EngineService target(JsonView body) {
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return target;
    }

    /** The render answer's id: the request's {@code request_id}, the {@code X-Request-Id} header, or a fresh one. */
    private String baseId(JsonView body) {
        String named = OpenAiRequests.requestId(http, body);
        return named != null && !named.isEmpty() ? named : String.valueOf(System.nanoTime());
    }
}
