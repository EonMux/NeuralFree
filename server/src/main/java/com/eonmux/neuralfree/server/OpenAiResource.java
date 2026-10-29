package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * The OpenAI-compatible endpoints (PARITY plan P2): {@code GET /v1/models}, {@code POST /v1/completions} and
 * {@code POST /v1/chat/completions}, each with server-sent-event streaming ({@code stream: true}). Bodies are
 * read as text and parsed with the core JSON stack ({@code utils/json}). Sampling settings map onto
 * {@link SamplerConfig} (only present keys leave the llama.cpp defaults); the grammar settings map onto
 * {@link ConstraintSpec} kinds ({@code grammar}, {@code guided_json}, {@code guided_regex}, {@code guided_choice}).
 *
 * <p>Refusals carry the gap by name: {@code n} above 1, {@code logit_bias} (the sampler has no additive token
 * biases), a prompt array (one prompt per request; the batch engine runs requests, not prompt batches) and image
 * or audio message parts (the projector epic).
 *
 * <p>Finish reasons: {@code stop} for an end-of-generation token, a stop string or a cancellation; {@code length}
 * for the token budget or a full context. Reasoning output of the chat template's parser streams as
 * {@code reasoning_content} deltas (the llama.cpp and vLLM field); tool-call syntax reaches the client as content
 * until the tool-call parser epic lands.
 */
@Path("/")
public class OpenAiResource {

    /** One routed output piece on its way to a stream. */
    private record Piece(ReasoningFilter.Channel channel, String text) { }

    /** The longest wait on one streamed request before the stream gives up (the engine may be stuck). */
    private static final long STREAM_TIMEOUT_MINUTES = 30;

    @Inject
    EngineService service;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public OpenAiResource(EngineService service) {
        this.service = service;
    }

    public OpenAiResource() {
    }

    @GET
    @Path("/v1/models")
    @Produces(MediaType.APPLICATION_JSON)
    public Response models() {
        requireLoaded();
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("id", service.modelPath());
        model.put("object", "model");
        model.put("owned_by", "neuralfree");
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("object", "list");
        body.put("data", List.of(model));
        return Response.ok(json(body)).build();
    }

    @POST
    @Path("/v1/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response completions(String bodyText) {
        requireLoaded();
        JsonView body = parse(bodyText);
        Object rawPrompt = body.raw("prompt");
        if (!(rawPrompt instanceof String prompt)) {
            throw badRequest("prompt is required and must be a string;"
                    + " a prompt array is refused (one prompt per request)");
        }
        GenerationRequest request = samplingRequest(body);
        ConstraintSpec constraint = constraint(body);
        int[] tokens = service.encode(prompt, true, true);
        if (body.optBool("stream", false)) {
            return streamResponse(tokens, request, null, constraint, false);
        }
        List<String> reasoning = new ArrayList<>();
        GenerationResult result = run(tokens, request, constraint, reasoning);
        List<Map<String, Object>> choices = new ArrayList<>();
        Map<String, Object> choice = choice(0, result.text(), null, finishReason(result.stopReason()));
        if (!result.logprobs().isEmpty()) {
            choice.put("logprobs", Map.of("content", logprobsBody(result)));
        }
        choices.add(choice);
        return Response.ok(json(completionBody("text_completion", choices, result))).build();
    }

    @POST
    @Path("/v1/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatCompletions(String bodyText) {
        requireLoaded();
        JsonView body = parse(bodyText);
        com.eonmux.neuralfree.gguf.chat.ChatRequest chat = chatRequest(body);
        String rendered = service.formatChat(chat);
        GenerationRequest request = samplingRequest(body);
        ConstraintSpec constraint = constraint(body);
        int[] tokens = service.encode(rendered, false, true);
        if (body.optBool("stream", false)) {
            return streamResponse(tokens, request, rendered, constraint, true);
        }
        List<String> reasoning = new ArrayList<>();
        GenerationResult result = run(tokens, request, constraint, reasoning);
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", result.text());
        if (!reasoning.isEmpty()) {
            message.put("reasoning_content", String.join("", reasoning));
        }
        List<Map<String, Object>> choices = new ArrayList<>();
        Map<String, Object> choice = choice(0, null, message, finishReason(result.stopReason()));
        if (!result.logprobs().isEmpty()) {
            choice.put("logprobs", Map.of("content", logprobsBody(result)));
        }
        choices.add(choice);
        return Response.ok(json(completionBody("chat.completion", choices, result))).build();
    }

    @POST
    @Path("/v1/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response embeddings(String bodyText) throws IOException {
        requireLoaded();
        JsonView body = parse(bodyText);
        List<String> inputs = new ArrayList<>();
        Object raw = body.raw("input");
        if (raw instanceof String s) {
            inputs.add(s);
        } else if (raw instanceof List<?> list) {
            for (Object o : list) {
                if (o instanceof String s) {
                    inputs.add(s);
                } else {
                    throw badRequest("every input must be a string; token arrays are refused"
                            + " (the server tokenizes text)");
                }
            }
        } else {
            throw badRequest("input is required and must be a string or an array of strings");
        }
        List<Map<String, Object>> data = new ArrayList<>();
        long tokens = 0;
        for (int i = 0; i < inputs.size(); i++) {
            int[] ids = service.encode(inputs.get(i), false, false);
            tokens += ids.length;
            float[] embedding;
            try {
                embedding = service.embed(ids);
            } catch (UnsupportedOperationException e) {
                throw badRequest(e.getMessage());
            }
            List<Object> values = new ArrayList<>(embedding.length);
            for (float v : embedding) {
                values.add((double) v);
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("object", "embedding");
            entry.put("index", i);
            entry.put("embedding", values);
            data.add(entry);
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", tokens);
        usage.put("total_tokens", tokens);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", "list");
        out.put("data", data);
        out.put("model", service.modelPath());
        out.put("usage", usage);
        return Response.ok(json(out)).build();
    }

    @POST
    @Path("/v1/rerank")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerank(String bodyText) throws IOException {
        requireLoaded();
        JsonView body = parse(bodyText);
        String query = body.optStr("query", null);
        if (query == null) {
            throw badRequest("query is required");
        }
        List<String> documents = new ArrayList<>();
        Object raw = body.raw("documents");
        if (raw instanceof List<?> list) {
            for (Object o : list) {
                if (o instanceof String s) {
                    documents.add(s);
                } else if (o instanceof Map<?, ?> map && map.get("text") instanceof String s) {
                    documents.add(s);
                } else {
                    throw badRequest("every document must be a string or an object with a text field");
                }
            }
        } else {
            throw badRequest("documents is required and must be an array");
        }
        int topN = body.optI32("top_n", documents.size());
        List<float[]> scored = new ArrayList<>();
        long tokens = 0;
        for (int i = 0; i < documents.size(); i++) {
            int[] input = service.rerankInput(query, documents.get(i));
            tokens += input.length;
            float score;
            try {
                score = service.rerankScore(input);
            } catch (UnsupportedOperationException e) {
                throw badRequest(e.getMessage());
            } catch (IOException e) {
                throw new WebApplicationException("the rerank forward failed: " + e.getMessage(), 500);
            }
            scored.add(new float[]{score, i});
        }
        scored.sort((a, b) -> Float.compare(b[0], a[0]));
        List<Map<String, Object>> results = new ArrayList<>();
        for (int r = 0; r < Math.min(topN, scored.size()); r++) {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("index", (int) scored.get(r)[1]);
            result.put("relevance_score", (double) scored.get(r)[0]);
            results.add(result);
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("total_tokens", tokens);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "rerank-" + System.nanoTime());
        out.put("model", service.modelPath());
        out.put("results", results);
        out.put("usage", usage);
        return Response.ok(json(out)).build();
    }

    // ------------------------------------------------------------------ request mapping

    /** The sampling settings of one request body: only present keys leave their llama.cpp defaults. */
    private GenerationRequest samplingRequest(JsonView body) {
        int n = body.optI32("n", 1);
        if (n != 1) {
            throw badRequest("n = " + n + " is refused: one choice per request (the engine runs requests,"
                    + " not choice batches)");
        }
        if (body.has("logit_bias")) {
            throw badRequest("logit_bias is refused: the sampler has no additive token biases"
                    + " (BiasStage bans token ids only)");
        }
        GenerationRequest.Builder builder = GenerationRequest.builder();
        int maxTokens = body.has("max_completion_tokens") ? body.optI32("max_completion_tokens", -1)
                : body.optI32("max_tokens", -1);
        if (maxTokens >= 0) {
            builder.maxNewTokens(maxTokens);
        }
        int logprobs = 0;
        if (body.has("n_probs")) {
            logprobs = Math.max(0, body.optI32("n_probs", 0));
        }
        if (body.optBool("logprobs", false)) {
            logprobs = Math.max(logprobs, body.optI32("top_logprobs", 0) > 0
                    ? body.optI32("top_logprobs", 0) : 1);
        }
        if (logprobs > 0) {
            builder.logprobs(logprobs);
        }
        List<String> stop = new ArrayList<>();
        Object rawStop = body.raw("stop");
        if (rawStop instanceof String s) {
            stop.add(s);
        } else if (rawStop instanceof List<?> list) {
            for (Object o : list) {
                if (o instanceof String s) {
                    stop.add(s);
                }
            }
        }
        builder.stopStrings(stop);
        List<String> badWords = new ArrayList<>();
        Object rawBad = body.raw("bad_words");
        if (rawBad instanceof String s) {
            badWords.add(s);
        } else if (rawBad instanceof List<?> list) {
            for (Object o : list) {
                if (o instanceof String s) {
                    badWords.add(s);
                }
            }
        }
        builder.badWords(badWords);
        builder.minTokens(Math.max(0, body.optI32("min_tokens", 0)));
        builder.sampler(samplerConfig(body));
        return builder.build();
    }

    /** The sampler settings of the body over the llama.cpp defaults ({@code GenerationRequest.defaults()}). */
    private SamplerConfig samplerConfig(JsonView body) {
        SamplerConfig.Builder sampler = GenerationRequest.defaults().sampler().toBuilder();
        if (body.has("temperature")) {
            sampler.temperature((float) body.f64("temperature"));
        }
        if (body.has("top_p")) {
            sampler.topP((float) body.f64("top_p"));
        }
        if (body.has("top_k")) {
            sampler.topK(body.optI32("top_k", 0));
        }
        if (body.has("min_p")) {
            sampler.minP((float) body.f64("min_p"));
        }
        if (body.has("typical_p")) {
            sampler.typicalP((float) body.f64("typical_p"));
        }
        if (body.has("repeat_penalty")) {
            sampler.repeatPenalty((float) body.f64("repeat_penalty"));
        }
        if (body.has("presence_penalty")) {
            sampler.presencePenalty((float) body.f64("presence_penalty"));
        }
        if (body.has("frequency_penalty")) {
            sampler.frequencyPenalty((float) body.f64("frequency_penalty"));
        }
        if (body.has("seed")) {
            long seed = body.optI64("seed", Long.MIN_VALUE);
            if (seed != Long.MIN_VALUE && seed != -1) {
                sampler.seed(seed);
            }
        }
        return sampler.build();
    }

    /** The grammar constraint of the body: llama.cpp's {@code grammar} and the vLLM {@code guided_*} names. */
    private ConstraintSpec constraint(JsonView body) {
        if (body.has("grammar")) {
            return ConstraintSpec.gbnf(body.str("grammar"));
        }
        if (body.has("guided_json")) {
            return ConstraintSpec.jsonSchema(body.str("guided_json"), null);
        }
        if (body.has("guided_regex")) {
            return ConstraintSpec.regex(body.str("guided_regex"), false);
        }
        if (body.has("guided_choice")) {
            List<String> choices = new ArrayList<>();
            for (Object o : stringArray(body, "guided_choice")) {
                choices.add(String.valueOf(o));
            }
            return ConstraintSpec.choice(choices);
        }
        if (body.has("guided_structural_tag") || body.has("structural_tag")) {
            return ConstraintSpec.structuralTag(body.str(
                    body.has("guided_structural_tag") ? "guided_structural_tag" : "structural_tag"));
        }
        return null;
    }

    private List<?> stringArray(JsonView body, String key) {
        Object raw = body.raw(key);
        if (!(raw instanceof List<?> list)) {
            throw badRequest(key + " must be an array of strings");
        }
        return list;
    }

    /** The chat request of the body: role/content messages plus the template knobs the engine carries. */
    private com.eonmux.neuralfree.gguf.chat.ChatRequest chatRequest(JsonView body) {
        Object raw = body.raw("messages");
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw badRequest("messages is required and must be a non-empty array");
        }
        com.eonmux.neuralfree.gguf.chat.ChatRequest.Builder chat =
                com.eonmux.neuralfree.gguf.chat.ChatRequest.builder();
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> message)) {
                throw badRequest("every message must be an object with role and content");
            }
            Map<String, Object> typed = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : message.entrySet()) {
                typed.put(String.valueOf(e.getKey()), e.getValue());
            }
            chat.addMessage(typed);
        }
        if (body.has("add_generation_prompt")) {
            chat.addGenerationPrompt(body.optBool("add_generation_prompt", true));
        }
        if (body.has("enable_thinking")) {
            chat.enableThinking(body.optBool("enable_thinking", false));
        }
        Object kwargs = body.raw("chat_template_kwargs");
        if (kwargs instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> e : map.entrySet()) {
                chat.templateKwarg(String.valueOf(e.getKey()), e.getValue());
            }
        }
        return chat.build();
    }

    // ------------------------------------------------------------------ execution

    /** Runs the submission to its terminal state on the calling thread; reasoning pieces land in {@code out}. */
    private GenerationResult run(int[] prompt, GenerationRequest request, ConstraintSpec constraint,
                                 List<String> reasoning) {
        try {
            List<String> sink = reasoning;
            EngineService.Submission submission = service.submit(prompt, request, (channel, tokenId, text) -> {
                if (channel == ReasoningFilter.Channel.REASONING && sink != null) {
                    sink.add(text);
                }
            }, null, constraint);
            if (!submission.await(STREAM_TIMEOUT_MINUTES, TimeUnit.MINUTES)) {
                throw new WebApplicationException("the request outlived its " + STREAM_TIMEOUT_MINUTES
                        + "-minute budget", 503);
            }
            if (submission.refused() != null) {
                throw badRequest(String.valueOf(submission.refused().getMessage()));
            }
            BatchSequence sequence = submission.sequence();
            if (sequence.state() == BatchSequence.State.FAILED) {
                throw new WebApplicationException(
                        Response.status(500).entity(json(errorBody(String.valueOf(sequence.error())))).build());
            }
            return sequence.result();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new WebApplicationException("request interrupted", 503);
        }
    }

    /** Streams the submission as server-sent events; the pieces cross a queue from the pump thread. */
    private Response streamResponse(int[] prompt, GenerationRequest request, String rendered,
                                    ConstraintSpec constraint, boolean chat) {
        BlockingQueue<Piece> pieces = new ArrayBlockingQueue<>(1024);
        String id = "chatcmpl-" + System.nanoTime();
        long created = System.currentTimeMillis() / 1000;
        String model = service.modelPath();
        EngineService.Submission submission;
        try {
            submission = service.submit(prompt, request, (channel, tokenId, text) -> {
                try {
                    pieces.put(new Piece(channel, text));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }, rendered, constraint);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new WebApplicationException("the server is shutting down", 503);
        }
        StreamingOutput stream = out -> streamEvents(out, submission, pieces, id, created, model, chat);
        return Response.ok(stream).type("text/event-stream").build();
    }

    /** Writes SSE frames until the submission reaches its terminal state, then the final chunk and [DONE]. */
    private void streamEvents(OutputStream out, EngineService.Submission submission, BlockingQueue<Piece> pieces,
                              String id, long created, String model, boolean chat) throws IOException {
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(STREAM_TIMEOUT_MINUTES);
        try {
            while (true) {
                Piece piece = pieces.poll(20, TimeUnit.MILLISECONDS);
                if (piece != null) {
                    writeChunk(out, id, created, model, piece, chat);
                }
                if (submission.await(0, TimeUnit.MILLISECONDS)) {
                    break;
                }
                if (System.nanoTime() > deadline) {
                    throw new IOException("the request outlived its " + STREAM_TIMEOUT_MINUTES + "-minute budget");
                }
            }
            Piece piece;
            while ((piece = pieces.poll()) != null) {
                writeChunk(out, id, created, model, piece, chat);
            }
            if (submission.refused() != null) {
                writeEvent(out, json(errorBody(String.valueOf(submission.refused().getMessage()))));
                return;
            }
            BatchSequence sequence = submission.sequence();
            if (sequence == null || sequence.state() == BatchSequence.State.FAILED) {
                writeEvent(out, json(errorBody(String.valueOf(
                        sequence == null ? "the request never registered" : sequence.error()))));
                return;
            }
            String finish = finishReason(sequence.result().stopReason());
            StringBuilder last = new StringBuilder();
            last.append("data: ");
            Map<String, Object> choice = new LinkedHashMap<>();
            choice.put("index", 0);
            choice.put("delta", Map.of());
            choice.put("finish_reason", finish);
            Map<String, Object> chunk = chunkHead(id, created, model, chat);
            chunk.put("choices", List.of(choice));
            chunk.put("usage", usage(sequence.result()));
            Json.write(chunk, Json.Style.API, last);
            last.append("\n\n");
            out.write(last.toString().getBytes(StandardCharsets.UTF_8));
            out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
            out.flush();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("stream interrupted", e);
        }
    }

    private void writeChunk(OutputStream out, String id, long created, String model, Piece piece, boolean chat)
            throws IOException {
        Map<String, Object> delta = new LinkedHashMap<>();
        if (piece.channel() == ReasoningFilter.Channel.REASONING) {
            delta.put("reasoning_content", piece.text());
        } else if (chat) {
            delta.put("content", piece.text());
        } else {
            delta.put("text", piece.text());
        }
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", null);
        Map<String, Object> chunk = chunkHead(id, created, model, chat);
        chunk.put("choices", List.of(choice));
        writeEvent(out, json(chunk));
    }

    private Map<String, Object> chunkHead(String id, long created, String model, boolean chat) {
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", chat ? "chat.completion.chunk" : "text_completion.chunk");
        chunk.put("created", created);
        chunk.put("model", model);
        return chunk;
    }

    private void writeEvent(OutputStream out, String json) throws IOException {
        out.write(("data: " + json + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    // ------------------------------------------------------------------ response bodies

    private Map<String, Object> usage(GenerationResult result) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", result.timings().promptTokens());
        usage.put("completion_tokens", result.timings().generatedTokens());
        usage.put("total_tokens", result.timings().promptTokens() + result.timings().generatedTokens());
        return usage;
    }

    /**
     * The OpenAI logprobs shape of a result: one content entry per generated token with its log probability and
     * its top alternatives ({@code token} renders through the model's tokenizer).
     */
    private List<Map<String, Object>> logprobsBody(GenerationResult result) {
        List<Map<String, Object>> content = new ArrayList<>();
        for (TokenScore[] step : result.logprobs()) {
            Map<String, Object> top = new LinkedHashMap<>();
            for (int i = 1; i < step.length; i++) {
                top.put(service.tokenText(step[i].tokenId()), (double) step[i].logprob());
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("token", service.tokenText(step[0].tokenId()));
            entry.put("logprob", (double) step[0].logprob());
            entry.put("top_logprobs", List.of(top));
            content.add(entry);
        }
        return content;
    }

    private Map<String, Object> completionBody(String object, List<Map<String, Object>> choices,
                                               GenerationResult result) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", "chatcmpl-" + System.nanoTime());
        body.put("object", object);
        body.put("created", System.currentTimeMillis() / 1000);
        body.put("model", service.modelPath());
        body.put("choices", choices);
        body.put("usage", usage(result));
        return body;
    }

    private Map<String, Object> choice(int index, String text, Map<String, Object> message, String finish) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", index);
        if (text != null) {
            choice.put("text", text);
        }
        if (message != null) {
            choice.put("message", message);
        }
        choice.put("finish_reason", finish);
        return choice;
    }

    private String finishReason(StopReason reason) {
        return switch (reason) {
            case MAX_TOKENS, CONTEXT_FULL -> "length";
            default -> "stop";
        };
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

    private JsonView parse(String bodyText) {
        try {
            return JsonView.of(Json.parse(bodyText, Json.Limits.UNTRUSTED), "body");
        } catch (RuntimeException e) {
            throw badRequest("invalid JSON body: " + e.getMessage());
        }
    }

    private void requireLoaded() {
        if (!service.loaded()) {
            throw new WebApplicationException(
                    Response.status(503).entity(json(errorBody("no model loaded:"
                            + " set -Dneuralfree.server.model=<gguf path>"))).build());
        }
    }

    private static Map<String, Object> errorBody(String message) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", message);
        error.put("type", "invalid_request_error");
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("error", error);
        return body;
    }

    private static WebApplicationException badRequest(String message) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", message);
        error.put("type", "invalid_request_error");
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("error", error);
        StringBuilder out = new StringBuilder();
        try {
            Json.write(body, Json.Style.API, out);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return new WebApplicationException(
                Response.status(400).type(MediaType.APPLICATION_JSON).entity(out.toString()).build());
    }
}
