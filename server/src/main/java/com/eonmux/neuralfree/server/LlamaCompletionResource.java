package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.InfillPrompt;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * llama.cpp's native completion, {@code POST /completion} and {@code /completions} (its OpenAI form is
 * {@code /v1/completions}): llama.cpp's prompt forms ({@link LlamaPrompts}), {@code n_predict} as the budget,
 * {@code n_probs}, {@code response_fields} and every sampler and grammar field the OpenAI routes take. One prompt
 * answers one object, several an array of them. A stream sends {@code data:} chunks of each released piece and a
 * final object per prompt, with no {@code [DONE]} marker.
 *
 * <p>{@code POST /infill} answers the same way over a fill-in-the-middle prompt ({@link InfillPrompt}):
 * {@code input_prefix}, {@code input_suffix}, optional {@code input_extra} chunks ({@code filename}, {@code text})
 * and an optional {@code prompt} appended to the prefix. A model without FIM tokens refuses with 501.</p>
 */
@Path("/")
public class LlamaCompletionResource {

    @Inject
    EngineService service;

    /** The current HTTP request, for disconnect cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public LlamaCompletionResource(EngineService service) {
        this.service = service;
    }

    public LlamaCompletionResource() {
    }

    @POST
    @Path("/completion")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response completion(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> completionOf(ServerJson.parse(bodyText)));
    }

    @POST
    @Path("/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response completions(String bodyText) {
        return completion(bodyText);
    }

    @POST
    @Path("/infill")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response infill(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> infillOf(ServerJson.parse(bodyText)));
    }

    private Response completionOf(JsonView body) {
        return respond(body, LlamaPrompts.of(body.raw("prompt"), service));
    }

    private Response infillOf(JsonView body) {
        String missing = service.infillMissingTokens();
        if (missing != null) {
            throw ServerJson.error(501, "Infill is not supported by this model: " + missing, "not_supported_error");
        }
        if (body.raw("prompt") != null && !(body.raw("prompt") instanceof String)) {
            throw ServerJson.badRequest("\"prompt\" must be a string");
        }
        if (!body.has("input_prefix")) {
            throw ServerJson.badRequest("\"input_prefix\" is required");
        }
        if (!body.has("input_suffix")) {
            throw ServerJson.badRequest("\"input_suffix\" is required");
        }
        int[] prefix = LlamaPrompts.text(body.raw("input_prefix"), service, false, false, "input_prefix");
        int[] suffix = LlamaPrompts.text(body.raw("input_suffix"), service, false, false, "input_suffix");
        String prompt = body.optStr("prompt", "");
        int[] promptTokens = prompt.isEmpty() ? new int[0] : service.encode(prompt, false, true);
        int nPredict = body.raw("n_predict") == null ? -1 : body.i32("n_predict");
        int[] tokens = service.infillPrompt(prefix, suffix, extraChunks(body), promptTokens, nPredict);
        return respond(body, List.of(tokens));
    }

    /** {@code input_extra}: an array of {@code {"filename": string, "text": string}}, filename optional. */
    private static List<InfillPrompt.Chunk> extraChunks(JsonView body) {
        Object raw = body.raw("input_extra");
        if (raw == null) {
            return List.of();
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}");
        }
        List<InfillPrompt.Chunk> out = new ArrayList<>();
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> chunk) || !(chunk.get("text") instanceof String text)) {
                throw ServerJson.badRequest("extra_context chunk must contain a \"text\" field with a string value");
            }
            Object filename = chunk.get("filename");
            if (filename != null && !(filename instanceof String)) {
                throw ServerJson.badRequest("extra_context chunk's \"filename\" field must be a string");
            }
            out.add(new InfillPrompt.Chunk((String) filename, text));
        }
        return out;
    }

    private Response respond(JsonView body, List<int[]> prompts) {
        String model = service.served().resolve(body);
        int nProbs = Math.max(0, body.optI32("n_probs", 0));
        if (Boolean.TRUE.equals(body.raw("post_sampling_probs"))) {
            throw ServerJson.badRequest("post_sampling_probs is not supported: the probabilities are the model's,"
                    + " before the sampler chain");
        }
        GenerationRequest request = request(body, nProbs);
        ConstraintSpec constraint = OpenAiRequests.constraint(body);
        OpenAiRequests.requireGenerationBudget(prompts.size(), 1);
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        for (int[] prompt : prompts) {
            jobs.add(new GenerationBatch.Job(prompt, request, null, constraint));
        }
        LlamaCompletionBodies bodies = new LlamaCompletionBodies(service, model, nProbs, fields(body));
        boolean streamed = body.optBool("stream", false);
        GenerationBatch batch = GenerationBatch.start(service, jobs, http);
        if (streamed) {
            StreamingOutput stream = out -> stream(out, batch, bodies, prompts, request);
            return Response.ok(stream).type("text/event-stream").build();
        }
        batch.awaitAll();
        List<Object> results = new ArrayList<>();
        for (int i = 0; i < batch.size(); i++) {
            results.add(bodies.finalBody(i, prompts.get(i), request, batch.result(i), batch.sequence(i), false));
        }
        return Response.ok(ServerJson.write(results.size() == 1 ? results.get(0) : results)).build();
    }

    /** The OpenAI request mapping with llama.cpp's {@code n_predict} as the budget (-1 unlimited). */
    private static GenerationRequest request(JsonView body, int nProbs) {
        GenerationRequest base = OpenAiRequests.sampling(body, nProbs > 0 ? nProbs : -1);
        if (body.raw("n_predict") == null) {
            return base;
        }
        int nPredict = body.i32("n_predict");
        if (nPredict < -1) {
            throw ServerJson.badRequest("n_predict must be -1 (unlimited) or at least 0, got " + nPredict);
        }
        return base.toBuilder().maxNewTokens(nPredict < 0 ? GenerationRequest.UNLIMITED : nPredict).build();
    }

    private static List<String> fields(JsonView body) {
        Object raw = body.raw("response_fields");
        if (raw == null) {
            return List.of();
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("response_fields must be an array of field paths");
        }
        List<String> out = new ArrayList<>();
        for (Object o : list) {
            if (!(o instanceof String path)) {
                throw ServerJson.badRequest("response_fields must be an array of field paths");
            }
            out.add(path);
        }
        return out;
    }

    private void stream(OutputStream out, GenerationBatch batch, LlamaCompletionBodies bodies, List<int[]> prompts,
                        GenerationRequest request) throws IOException {
        LlamaStreamChunks chunks = new LlamaStreamChunks(batch.size());
        try {
            int done = 0;
            while (done < batch.size()) {
                GenerationBatch.Event event = batch.next(50);
                if (event instanceof GenerationBatch.Text t) {
                    chunk(out, batch, bodies, prompts, chunks.offer(t));
                } else if (event instanceof GenerationBatch.Done d) {
                    done++;
                    chunk(out, batch, bodies, prompts, chunks.flush(d.choice()));
                    finish(out, batch, bodies, prompts, request, d.choice());
                }
            }
        } catch (IOException e) {
            batch.cancel();
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            batch.cancel();
            throw new IOException("stream interrupted", e);
        } catch (RuntimeException e) {
            batch.cancel();
            ServerJson.event(out, null, ServerJson.errorBody(String.valueOf(e.getMessage()), "server_error"));
        }
    }

    /** Writes one token's chunk when it released text or carries its logprobs. */
    private static void chunk(OutputStream out, GenerationBatch batch, LlamaCompletionBodies bodies,
                              List<int[]> prompts, GenerationBatch.Text piece) throws IOException {
        if (piece == null || piece.text().isEmpty() && piece.logprobs() == null) {
            return;
        }
        int i = piece.choice();
        ServerJson.event(out, null, bodies.partial(i, piece, prompts.get(i).length, batch.sequence(i)));
    }

    private static void finish(OutputStream out, GenerationBatch batch, LlamaCompletionBodies bodies,
                               List<int[]> prompts, GenerationRequest request, int choice) throws IOException {
        String failure = batch.failure(choice);
        if (failure != null) {
            ServerJson.event(out, null, ServerJson.errorBody(failure, "server_error"));
            return;
        }
        GenerationResult result = batch.result(choice);
        ServerJson.event(out, null, bodies.finalBody(choice, prompts.get(choice), request, result,
                batch.sequence(choice), true));
    }
}
