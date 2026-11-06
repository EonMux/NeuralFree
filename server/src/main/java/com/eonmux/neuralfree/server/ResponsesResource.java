package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.DELETE;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The OpenAI Responses API (llama.cpp's {@code post_responses_oai}, vLLM's {@code OpenAIServingResponses}):
 * {@code POST /v1/responses} (and llama.cpp's {@code /responses}) converts the request to a chat-completions body
 * ({@link ResponsesRequests}) and runs it through the chat pipeline ({@link ChatPlan}), so sampling, structured
 * output, tools and reasoning behave as on {@code /v1/chat/completions}; the answer is the Response object or its
 * event stream ({@link ResponsesEvents}).
 *
 * <p>With the response store on ({@link ServerOptions#responseStore}) a response with {@code store} (the default)
 * is kept: {@code GET /v1/responses/{id}} returns it, {@code GET .../input_items} lists its input,
 * {@code DELETE} removes it, and {@code previous_response_id} continues its conversation. {@code background: true}
 * returns at once and runs the generation detached from the client; {@code GET ?stream=true&starting_after=N}
 * replays its events, and {@code POST .../cancel} stops it. {@code POST /v1/responses/input_tokens} counts the
 * prompt tokens of a request.</p>
 *
 * <p>In router mode a creation takes the model from the body's {@code "model"} and stores on that model's own
 * response store. The id routes take {@code ?model=}; without it they find the store that holds the id, so a
 * response created on one model is retrieved from that model.</p>
 */
@Path("/")
public class ResponsesResource {

    @Inject
    EngineService service;

    /** The current HTTP request, for disconnect cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public ResponsesResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public ResponsesResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    public ResponsesResource() {
    }

    /**
     * The engine a stored response of {@code id} lives on: the {@code ?model=} entry in router mode, else the
     * loaded entry whose store holds the id; without either, the wired service (its store answers the 404).
     */
    private EngineService storedOn(String id, String model, String autoload) {
        if (router == null || !router.active()) {
            return service;
        }
        if (model != null && !model.isBlank()) {
            return ModelRouter.targetQuery(router, service, model, autoload);
        }
        for (ModelRouter.Entry entry : router.list()) {
            EngineService open = entry.service();
            if (entry.ready() && open != null && open.responses().get(id) != null) {
                return open;
            }
        }
        return service;
    }

    /** The engine a created response serves on: the body's {@code model} in router mode, else the wired service. */
    private EngineService target(JsonView body) {
        return ModelRouter.target(router, service, body, null);
    }

    @POST
    @Path("/v1/responses")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response create(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> createOf(body, target));
    }

    /** llama.cpp's unversioned alias. */
    @POST
    @Path("/responses")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response createAlias(String bodyText) {
        return create(bodyText);
    }

    @POST
    @Path("/v1/responses/input_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response inputTokens(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = target(body);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> {
            JsonView chat = JsonView.of(ResponsesRequests.chatBody(body, history(body, target)), "body");
            return OpenAiResource.inputTokens(target, chat);
        });
    }

    /** llama.cpp's unversioned alias. */
    @POST
    @Path("/responses/input_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response inputTokensAlias(String bodyText) {
        return inputTokens(bodyText);
    }

    @GET
    @Path("/v1/responses/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response retrieve(@PathParam("id") String id, @QueryParam("stream") boolean stream,
                             @QueryParam("starting_after") Integer startingAfter,
                             @QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        return retrieveOf(id, stream, startingAfter, storedOn(id, model, autoload));
    }

    /** Test and manual wiring: the retrieval from the wired service, without a model query. */
    Response retrieve(String id, boolean stream, Integer startingAfter) {
        return retrieveOf(id, stream, startingAfter, service);
    }

    private static Response retrieveOf(String id, boolean stream, Integer startingAfter,
                                       EngineService service) {
        ResponseStore.Entry entry = stored(id, service);
        if (!stream) {
            return Response.ok(ServerJson.write(entry.response())).build();
        }
        if (!entry.background()) {
            throw ServerJson.badRequest("only a background response streams its events on retrieval");
        }
        int after = startingAfter == null ? -1 : startingAfter;
        StreamingOutput out = o -> replay(entry, after, o);
        return Response.ok(out).type("text/event-stream").build();
    }

    @GET
    @Path("/v1/responses/{id}/input_items")
    @Produces(MediaType.APPLICATION_JSON)
    public Response inputItems(@PathParam("id") String id, @QueryParam("limit") Integer limit,
                               @QueryParam("order") String order, @QueryParam("after") String after,
                               @QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        return inputItemsOf(id, limit, order, after, storedOn(id, model, autoload));
    }

    /** Test and manual wiring: the input items of the wired service, without a model query. */
    Response inputItems(String id, Integer limit, String order, String after) {
        return inputItemsOf(id, limit, order, after, service);
    }

    private static Response inputItemsOf(String id, Integer limit, String order, String after,
                                         EngineService service) {
        ResponseStore.Entry entry = stored(id, service);
        int size = limit == null ? 20 : limit;
        if (size < 1 || size > 100) {
            throw ServerJson.badRequest("limit must be between 1 and 100, got " + size);
        }
        String direction = order == null ? "desc" : order;
        if (!direction.equals("asc") && !direction.equals("desc")) {
            throw ServerJson.badRequest("order must be asc or desc, got '" + direction + "'");
        }
        return Response.ok(ServerJson.write(ResponsesPages.page(entry.inputItems(), size, direction, after)))
                .build();
    }

    @DELETE
    @Path("/v1/responses/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response delete(@PathParam("id") String id, @QueryParam("model") String model,
                           @QueryParam("autoload") String autoload) {
        return deleteOf(id, storedOn(id, model, autoload));
    }

    /** Test and manual wiring: the deletion from the wired service, without a model query. */
    Response delete(String id) {
        return deleteOf(id, service);
    }

    private static Response deleteOf(String id, EngineService service) {
        ResponseStore.Entry entry = stored(id, service);
        entry.cancel();
        service.responses().remove(id);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", id);
        out.put("object", "response");
        out.put("deleted", true);
        return Response.ok(ServerJson.write(out)).build();
    }

    @POST
    @Path("/v1/responses/{id}/cancel")
    @Produces(MediaType.APPLICATION_JSON)
    public Response cancel(@PathParam("id") String id, @QueryParam("model") String model,
                           @QueryParam("autoload") String autoload) {
        return cancelOf(id, storedOn(id, model, autoload));
    }

    /** Test and manual wiring: the cancellation on the wired service, without a model query. */
    Response cancel(String id) {
        return cancelOf(id, service);
    }

    private static Response cancelOf(String id, EngineService service) {
        ResponseStore.Entry entry = stored(id, service);
        if (!entry.background()) {
            throw ServerJson.badRequest("Cannot cancel a synchronous response.");
        }
        if (!entry.cancel()) {
            throw ServerJson.badRequest("Cannot cancel a response that is already "
                    + entry.response().get("status") + ".");
        }
        try {
            // cancellation lands between decode steps; the request budget bounds a long prefill
            entry.awaitFinished(service.options().requestTimeout().toMillis());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return Response.ok(ServerJson.write(entry.response())).build();
    }

    // ------------------------------------------------------------------ create

    private Response createOf(JsonView body, EngineService service) {
        if (body.has("conversation")) {
            throw ServerJson.badRequest("'conversation' is not supported; continue a response with"
                    + " previous_response_id");
        }
        if (body.has("prompt")) {
            throw ServerJson.badRequest("'prompt' (a stored prompt template) is not supported");
        }
        if (OpenAiRequests.beamSearch(body)) {
            throw ServerJson.badRequest("use_beam_search is not supported by the Responses API");
        }
        ResponseStore store = service.responses();
        boolean background = body.optBool("background", false);
        boolean stored = body.optBool("store", true) && store.enabled();
        if (background && !stored) {
            throw ServerJson.badRequest("background responses need the response store: start the server with -D"
                    + ServerOptions.RESPONSE_STORE_PROPERTY + "=<count> and keep store true");
        }
        Map<String, Object> chatBody = ResponsesRequests.chatBody(body, history(body, service));
        ChatPlan plan = ChatPlan.of(service, JsonView.of(chatBody, "body"));
        String id = ResponsesEvents.newId("resp_");
        Map<String, Object> head = ResponsesEvents.head(body, id, plan.model(), System.currentTimeMillis() / 1000,
                stored);
        ResponsesRun run = new ResponsesRun(plan, head, new OpenAiBodies(service, plan.top(), plan.model()),
                ResponsesRequests.wantsLogprobs(body), messages(chatBody));
        List<Object> inputItems = ResponsesPages.withIds(ResponsesRequests.inputItems(body));
        GenerationBatch batch = GenerationBatch.start(service, plan.jobs(), background ? null : http, id);
        if (background) {
            return background(batch, run, service, id, head, inputItems, body.optBool("stream", false));
        }
        if (body.optBool("stream", false)) {
            StreamingOutput out = o -> {
                Map<String, Object> done = run.stream(batch, (type, data) -> ServerJson.event(o, type, data));
                keep(stored, id, inputItems, done, run, service);
            };
            return Response.ok(out).type("text/event-stream").build();
        }
        batch.awaitAll();
        if (batch.failure(0) != null) {
            batch.result(0);
        }
        Map<String, Object> done = run.collect(batch);
        keep(stored, id, inputItems, done, run, service);
        return Response.ok(ServerJson.write(done)).build();
    }

    /** Starts the detached generation of a background response; answers it queued, or its event stream. */
    private Response background(GenerationBatch batch, ResponsesRun run, EngineService service, String id,
                                Map<String, Object> head, List<Object> inputItems, boolean streamed) {
        ResponseStore.Entry entry = new ResponseStore.Entry(id, inputItems, ResponsesEvents.pending(head, "queued"),
                batch);
        service.responses().put(entry);
        // the queued answer is rendered before the generation starts: the detached run flips the stored state to
        // in_progress as soon as it opens the response, and the create answer reports the queued state it started in
        String queued = ServerJson.write(entry.response());
        Thread.ofVirtual().name("response-" + id).start(() -> runDetached(batch, run, entry, head));
        if (streamed) {
            StreamingOutput out = o -> replay(entry, -1, o);
            return Response.ok(out).type("text/event-stream").build();
        }
        return Response.ok(queued).build();
    }

    /**
     * Runs a background response to its end. The entry always finishes, whatever is thrown: replays, cancels and
     * {@code previous_response_id} wait on it.
     */
    private static void runDetached(GenerationBatch batch, ResponsesRun run, ResponseStore.Entry entry,
                                    Map<String, Object> head) {
        Map<String, Object> done = null;
        List<Map<String, Object>> conversation = List.of();
        try {
            done = run.detached(batch, entry::event);
            conversation = run.conversation(done);
        } finally {
            if (done == null) {
                batch.cancel();
                done = ResponsesEvents.failed(head, "the background response ended abnormally");
            }
            entry.finish(done, conversation);
        }
    }

    /** Stores a finished response when the request asked for it. */
    private static void keep(boolean stored, String id, List<Object> inputItems, Map<String, Object> done,
                             ResponsesRun run, EngineService service) {
        if (!stored) {
            return;
        }
        ResponseStore.Entry entry = new ResponseStore.Entry(id, inputItems, done, null);
        entry.finish(done, run.conversation(done));
        service.responses().put(entry);
    }

    /** The conversation {@code previous_response_id} continues, or an empty list. */
    private static List<Map<String, Object>> history(JsonView body, EngineService service) {
        Object prev = body.raw("previous_response_id");
        if (prev == null) {
            return List.of();
        }
        if (!(prev instanceof String id) || id.isEmpty()) {
            throw ServerJson.badRequest("previous_response_id must be a response id");
        }
        if (!service.responses().enabled()) {
            throw ServerJson.badRequest("previous_response_id needs the response store: start the server with -D"
                    + ServerOptions.RESPONSE_STORE_PROPERTY + "=<count>");
        }
        ResponseStore.Entry entry = stored(id, service);
        List<Map<String, Object>> conversation = entry.conversation();
        if (conversation == null) {
            throw ServerJson.badRequest("Response '" + id + "' is still in progress");
        }
        return conversation;
    }

    private static ResponseStore.Entry stored(String id, EngineService service) {
        ResponseStore.Entry entry = service.responses().get(id);
        if (entry == null) {
            throw ServerJson.error(404, "Response with id '" + id + "' not found.", "invalid_request_error");
        }
        return entry;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> messages(Map<String, Object> chatBody) {
        return new ArrayList<>((List<Map<String, Object>>) chatBody.get("messages"));
    }

    /** Streams the stored events of {@code entry} after sequence number {@code after} until it finishes. */
    private static void replay(ResponseStore.Entry entry, int after, OutputStream out) throws IOException {
        int last = after;
        try {
            while (!entry.drained(last)) {
                for (Map<String, Object> data : entry.eventsAfter(last, 1_000)) {
                    ServerJson.event(out, (String) data.get("type"), data);
                    last = (Integer) data.get("sequence_number");
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("stream interrupted", e);
        }
    }

    /** The engine sequence's reused prompt prefix, the response's cached tokens. */
    static int cached(GenerationBatch batch) {
        BatchSequence sequence = batch.sequence(0);
        return sequence == null ? 0 : sequence.reusedPrefix();
    }
}
