package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The OpenAI-compatible generation endpoints: {@code GET /v1/models}, {@code POST /v1/completions} and
 * {@code POST /v1/chat/completions}, each with server-sent-event streaming ({@code stream: true}).
 *
 * <ul>
 *   <li>Choices: {@code n} choices per prompt, decoded side by side; with a {@code seed}, choice {@code i}
 *       samples with {@code seed + i}. A completion takes every OpenAI prompt form (string, strings, token ids,
 *       token-id arrays), choice {@code p * n + i} answering prompt {@code p}; {@code echo} prepends the
 *       prompt.</li>
 *   <li>Constraints: {@code grammar}, {@code json_schema}, the vLLM {@code guided_*} fields and
 *       {@code structured_outputs}, and {@code response_format} ({@link OpenAiRequests#constraint}).</li>
 *   <li>Tools: {@code tools} render through the template; {@code tool_choice} {@code auto} reads calls with the
 *       configured parser, {@code required} and a named function add a forcing grammar ({@link ToolRouting}).
 *       Calls come back as {@code message.tool_calls} (streamed as {@code delta.tool_calls}) with finish reason
 *       {@code tool_calls}.</li>
 *   <li>Reasoning streams as {@code reasoning_content}. Logprobs follow OpenAI per endpoint: chat's
 *       {@code logprobs: true} with {@code top_logprobs} answers {@code logprobs.content} entries, the
 *       completions' integer {@code logprobs} the legacy {@code tokens}/{@code token_logprobs}/
 *       {@code top_logprobs}/{@code text_offset} arrays; llama.cpp's {@code n_probs} selects the count on
 *       either. {@code stream_options.include_usage}
 *       adds the usage chunk; without it the last finish chunk carries the usage.</li>
 * </ul>
 *
 * A client that disconnects mid-stream cancels its sequences.
 */
@Path("/")
public class OpenAiResource {

    @Inject
    EngineService service;

    /** The current HTTP request (a request-scoped proxy), for disconnect cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

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
        ServerJson.requireLoaded(service);
        List<Object> data = new ArrayList<>();
        for (String name : service.served().names()) {
            Map<String, Object> model = new LinkedHashMap<>();
            model.put("id", name);
            model.put("object", "model");
            model.put("created", 0L);
            model.put("owned_by", "neuralfree");
            model.put("root", service.modelPath());
            model.put("max_model_len", service.maxModelLength());
            data.add(model);
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("object", "list");
        body.put("data", data);
        return Response.ok(ServerJson.write(body)).build();
    }

    @POST
    @Path("/v1/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response completions(String bodyText) {
        ServerJson.requireLoaded(service);
        return mapped(() -> completionsOf(ServerJson.parse(bodyText)));
    }

    @POST
    @Path("/v1/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatCompletions(String bodyText) {
        ServerJson.requireLoaded(service);
        return mapped(() -> chatOf(ServerJson.parse(bodyText)));
    }

    /** llama.cpp's unversioned alias of {@code /v1/models}. */
    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    public Response modelsAlias() {
        return models();
    }

    /** llama.cpp's unversioned alias of {@code /v1/chat/completions}. */
    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatCompletionsAlias(String bodyText) {
        return chatCompletions(bodyText);
    }

    /** llama.cpp's prompt token count of a chat request ({@code handle_count_tokens}). */
    @POST
    @Path("/v1/chat/completions/input_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatInputTokens(String bodyText) {
        ServerJson.requireLoaded(service);
        return mapped(() -> inputTokens(service, ServerJson.parse(bodyText)));
    }

    /** llama.cpp's unversioned alias. */
    @POST
    @Path("/chat/completions/input_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatInputTokensAlias(String bodyText) {
        return chatInputTokens(bodyText);
    }

    /** {@code {"object": "response.input_tokens", "input_tokens": n}} of the rendered chat prompt. */
    static Response inputTokens(EngineService service, JsonView chatBody) {
        int[] prompt = service.encode(service.formatChat(OpenAiRequests.chat(chatBody)), false, true);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", "response.input_tokens");
        out.put("input_tokens", prompt.length);
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * Runs a request mapping; malformed fields the mapping did not catch and inputs the model cannot take (media
     * parts on a text-only path) are 400s.
     */
    static Response mapped(Supplier<Response> handler) {
        try {
            return handler.get();
        } catch (JsonException | IllegalArgumentException | UnsupportedOperationException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        }
    }

    // ------------------------------------------------------------------ completions

    private Response completionsOf(JsonView body) {
        String model = service.served().resolve(body);
        int n = OpenAiRequests.choices(body);
        int top = OpenAiRequests.topLogprobs(body, false);
        GenerationRequest base = OpenAiRequests.sampling(body, top);
        ConstraintSpec constraint = OpenAiRequests.constraint(body);
        List<OpenAiRequests.Prompt> prompts = OpenAiRequests.prompts(body, service.vocabSize());
        OpenAiRequests.requireGenerationBudget(prompts.size(), n);
        boolean echo = body.optBool("echo", false);
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        List<String> echoes = new ArrayList<>();
        long promptTokens = 0;
        for (OpenAiRequests.Prompt p : prompts) {
            int[] tokens = p.tokens() != null ? p.tokens() : service.encode(p.text(), true, true);
            if (tokens.length == 0) {
                throw ServerJson.badRequest("a prompt encodes to no tokens");
            }
            promptTokens += tokens.length;
            String echoed = !echo ? "" : p.text() != null ? p.text() : service.decode(tokens);
            for (int i = 0; i < n; i++) {
                jobs.add(new GenerationBatch.Job(tokens, OpenAiRequests.forChoice(base, i), null, constraint));
                echoes.add(echoed);
            }
        }
        OpenAiBodies bodies = new OpenAiBodies(service, top, model);
        String id = "cmpl-" + System.nanoTime();
        long created = System.currentTimeMillis() / 1000;
        boolean streamed = body.optBool("stream", false);
        boolean includeUsage = body.optObj("stream_options").optBool("include_usage", false);
        GenerationBatch batch = GenerationBatch.start(service, jobs, http);
        if (streamed) {
            long counted = promptTokens;
            StreamingOutput stream = out -> streamCompletions(out, batch, bodies, id, created, echoes, counted,
                    includeUsage);
            return Response.ok(stream).type("text/event-stream").build();
        }
        batch.awaitAll();
        List<Map<String, Object>> choices = new ArrayList<>();
        List<GenerationResult> results = new ArrayList<>();
        for (int i = 0; i < batch.size(); i++) {
            GenerationResult result = batch.result(i);
            results.add(result);
            Map<String, Object> choice = new LinkedHashMap<>();
            choice.put("index", i);
            choice.put("text", echoes.get(i) + result.text());
            choice.put("logprobs", result.logprobs().isEmpty() ? null
                    : bodies.completionLogprobs(result.logprobs(), echoes.get(i).length()));
            choice.put("finish_reason", OpenAiBodies.finishReason(result.stopReason()));
            choices.add(choice);
        }
        Map<String, Object> out = bodies.head(id, "text_completion", created);
        out.put("choices", choices);
        out.put("usage", OpenAiBodies.usage(promptTokens, results));
        return Response.ok(ServerJson.write(out)).build();
    }

    private void streamCompletions(OutputStream out, GenerationBatch batch, OpenAiBodies bodies, String id,
                                   long created, List<String> echoes, long promptTokens, boolean includeUsage)
            throws IOException {
        StreamWriter writer = new StreamWriter(out, batch, bodies, id, created, "text_completion");
        int[] offsets = new int[echoes.size()];
        for (int i = 0; i < echoes.size(); i++) {
            if (!echoes.get(i).isEmpty()) {
                writer.completionChunk(i, echoes.get(i), null, 0, null);
                offsets[i] = echoes.get(i).length();
            }
        }
        writer.run(event -> {
            if (event instanceof GenerationBatch.Text t && !(t.text().isEmpty() && t.logprobs() == null)) {
                writer.completionChunk(t.choice(), t.text(), t.logprobs(), offsets[t.choice()], null);
                offsets[t.choice()] += t.text().length();
            } else if (event instanceof GenerationBatch.Done d) {
                GenerationResult r = writer.finished(d.choice());
                if (r != null) {
                    writer.completionChunk(d.choice(), "", null, 0, OpenAiBodies.finishReason(r.stopReason()));
                }
            }
        }, promptTokens, includeUsage);
    }

    // ------------------------------------------------------------------ chat completions

    private Response chatOf(JsonView body) {
        int n = OpenAiRequests.choices(body);
        ChatPlan plan = ChatPlan.of(service, body);
        String model = plan.model();
        int top = plan.top();
        ToolRouting routing = plan.routing();
        boolean named = plan.named();
        int[] tokens = plan.tokens();
        List<GenerationBatch.Job> jobs = plan.jobs(n);
        List<ChatChoice> choices = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            choices.add(new ChatChoice(routing.newOutput()));
        }
        OpenAiBodies bodies = new OpenAiBodies(service, top, model);
        String id = "chatcmpl-" + System.nanoTime();
        long created = System.currentTimeMillis() / 1000;
        boolean streamed = body.optBool("stream", false);
        boolean includeUsage = body.optObj("stream_options").optBool("include_usage", false);
        GenerationBatch batch = GenerationBatch.start(service, jobs, http);
        if (streamed) {
            StreamingOutput stream = out -> streamChat(out, batch, bodies, id, created, choices, tokens.length,
                    includeUsage, named);
            return Response.ok(stream).type("text/event-stream").build();
        }
        batch.awaitAll();
        drainInto(batch, choices);
        List<Map<String, Object>> out = new ArrayList<>();
        List<GenerationResult> results = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            GenerationResult result = batch.result(i);
            results.add(result);
            out.add(chatChoiceBody(i, choices.get(i), result, bodies, named));
        }
        Map<String, Object> response = bodies.head(id, "chat.completion", created);
        response.put("choices", out);
        response.put("usage", OpenAiBodies.usage(tokens.length, results));
        return Response.ok(ServerJson.write(response)).build();
    }

    /** Routes every queued event (the generations finished) into its choice. */
    private static void drainInto(GenerationBatch batch, List<ChatChoice> choices) {
        try {
            GenerationBatch.Event e;
            while ((e = batch.next(0)) != null) {
                choices.get(e.choice()).accept(e);
            }
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        }
    }

    private static Map<String, Object> chatChoiceBody(int index, ChatChoice choice, GenerationResult result,
                                                      OpenAiBodies bodies, boolean named) {
        ToolParse parse = choice.finish();
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", parse.calls().isEmpty() || !parse.content().isEmpty() ? parse.content() : null);
        if (!choice.reasoning().isEmpty()) {
            message.put("reasoning_content", choice.reasoning());
        }
        if (!parse.calls().isEmpty()) {
            message.put("tool_calls", ChatChoice.toolCallsBody(parse.calls()));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("index", index);
        body.put("message", message);
        body.put("logprobs", choice.logprobs().isEmpty() ? null : bodies.chatLogprobs(choice.logprobs()));
        body.put("finish_reason", chatFinish(result.stopReason(), !parse.calls().isEmpty(), named));
        return body;
    }

    /** {@code tool_calls} when calls were read (a named choice answers {@code stop}, as OpenAI does). */
    static String chatFinish(StopReason reason, boolean calls, boolean named) {
        String base = OpenAiBodies.finishReason(reason);
        return calls && !named && base.equals("stop") ? "tool_calls" : base;
    }

    private void streamChat(OutputStream out, GenerationBatch batch, OpenAiBodies bodies, String id, long created,
                            List<ChatChoice> choices, long promptTokens, boolean includeUsage, boolean named)
            throws IOException {
        StreamWriter writer = new StreamWriter(out, batch, bodies, id, created, "chat.completion.chunk");
        for (int i = 0; i < choices.size(); i++) {
            Map<String, Object> role = new LinkedHashMap<>();
            role.put("role", "assistant");
            role.put("content", "");
            writer.chatChunk(i, role, null, null);
        }
        writer.run(event -> {
            ChatChoice choice = choices.get(event.choice());
            if (event instanceof GenerationBatch.Done d) {
                ToolParse parse = choice.finish();
                writeDeltas(writer, d.choice(), choice);
                GenerationResult r = writer.finished(d.choice());
                if (r != null) {
                    writer.chatChunk(d.choice(), Map.of(), null,
                            chatFinish(r.stopReason(), !parse.calls().isEmpty(), named));
                }
                return;
            }
            choice.accept(event);
            writeDeltas(writer, event.choice(), choice);
        }, promptTokens, includeUsage);
    }

    private static void writeDeltas(StreamWriter writer, int index, ChatChoice choice) throws IOException {
        for (ChatChoice.Delta d : choice.drain()) {
            writer.chatChunk(index, d.delta(), d.logprobs(), null);
        }
    }
}
