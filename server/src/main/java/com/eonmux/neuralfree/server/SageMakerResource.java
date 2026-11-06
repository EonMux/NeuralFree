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

/**
 * SageMaker hosting's inference endpoint, {@code POST /invocations}, as vLLM serves it
 * (vllm/entrypoints/serve/sagemaker): the request body is one of the invocation request types — an OpenAI chat or
 * completion request, or a pooling embedding, classification, rerank, score or pooling request — and answers
 * exactly as that type's own route answers. The type is read from the body's fields ({@link CloudProtocols}); a
 * body naming none refuses with vLLM's message listing the expected request types. The {@code model} field selects
 * a routed model or a per-request LoRA adapter, as it does on every other route.
 *
 * <p>The SageMaker alarm probe is {@code GET} and {@code POST /ping}, served by {@link OpsResource} as vLLM
 * delegates to its health check: an empty 200 while a model is open, 503 before.</p>
 */
@Path("/")
public class SageMakerResource {

    @Inject
    EngineService service;

    /** The current HTTP request (a request-scoped proxy), for streaming cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public SageMakerResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public SageMakerResource(EngineService service, ModelRouter router) {
        this(service, router, null);
    }

    /** A resource over an explicit service, router and request, so a delegated stream cancels on disconnect. */
    public SageMakerResource(EngineService service, ModelRouter router, HttpServerRequest http) {
        this.service = service;
        this.router = router;
        this.http = http;
    }

    public SageMakerResource() {
    }

    @POST
    @Path("/invocations")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response invocations(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        CloudProtocols.Invocation invocation = CloudProtocols.requireInvocation(body, target);
        return OpenAiResource.mapped(() -> CloudProtocols.invoke(invocation.route(), service, router, http,
                bodyText));
    }
}
