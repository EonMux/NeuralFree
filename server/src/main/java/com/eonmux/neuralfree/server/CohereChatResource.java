package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Cohere's Chat v2 API, {@code POST /cohere/v2/chat}, as vLLM serves it (vllm/entrypoints/cohere): the body maps
 * onto a chat request ({@link CohereChatRequest}) and generates through the same plan, tool routing and reasoning
 * filters as {@code /v1/chat/completions}. The answer is Cohere's shape: {@code finish_reason} ({@code COMPLETE},
 * {@code STOP_SEQUENCE}, {@code MAX_TOKENS}, {@code TOOL_CALL} or {@code ERROR}), a message of {@code thinking} and
 * {@code text} content blocks with any {@code tool_calls}, and the usage in billed units and tokens. With
 * {@code stream: true} the stream carries Cohere's event sequence ({@link CohereChatStream}) and ends with
 * {@code data: [DONE]}, as the Cohere clients expect.
 *
 * <p>Cohere's {@code logprobs} answers one item per generated token with its text, id and log probability, as
 * vLLM's conversion of the OpenAI capture does. The {@code id} echoes the request's {@code X-Request-Id} when one
 * names it, under the {@code chatcmpl-} prefix vLLM's delegating handler produces.</p>
 */
@Path("/")
public class CohereChatResource {

    @Inject
    EngineService service;

    /** The current HTTP request (a request-scoped proxy), for streaming cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public CohereChatResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public CohereChatResource(EngineService service, ModelRouter router) {
        this(service, router, null);
    }

    /** A resource over an explicit service, router and request, so a delegated stream cancels on disconnect. */
    public CohereChatResource(EngineService service, ModelRouter router, HttpServerRequest http) {
        this.service = service;
        this.router = router;
        this.http = http;
    }

    public CohereChatResource() {
    }

    @POST
    @Path("/cohere/v2/chat")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chat(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        CohereChatRequest request = CohereChatRequest.of(body);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> chatOf(target, request));
    }

    private Response chatOf(EngineService service, CohereChatRequest request) {
        JsonView openAi = JsonView.of(request.openAiBody(), "body");
        ChatPlan plan = ChatPlan.of(service, openAi);
        String id = "chatcmpl-" + requestId();
        GenerationBatch batch = GenerationBatch.start(service, plan.jobs(), http);
        if (request.streamed()) {
            StreamingOutput stream = out -> stream(out, batch, plan, request, id);
            return Response.ok(stream).type("text/event-stream").build();
        }
        batch.awaitAll();
        ChatChoice choice = plan.newChoice();
        OpenAiResource.drainInto(batch, List.of(choice));
        ToolParse parse = choice.finish();
        GenerationResult result = batch.result(0);
        Map<String, Object> out = body(service, request, choice, parse, result, plan, id,
                batch.cachedTokens(0));
        OpenAiResource.putKvTransfer(out, batch, 0);
        return Response.ok(ServerJson.write(out)).build();
    }

    /** The id's tail: the {@code X-Request-Id} header, else a fresh number. */
    private String requestId() {
        String named = OpenAiRequests.requestId(http, null);
        return named != null ? named : String.valueOf(System.nanoTime());
    }

    /** The non-streamed Cohere answer. */
    private static Map<String, Object> body(EngineService service, CohereChatRequest request, ChatChoice choice,
                                            ToolParse parse, GenerationResult result, ChatPlan plan, String id,
                                            long cached) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", id);
        out.put("finish_reason", finishReason(result, !parse.calls().isEmpty()));
        out.put("message", message(choice.messageReasoning(), parse));
        out.put("usage", usage(plan.promptRows(), cached, result.timings().generatedTokens()));
        if (request.logprobs()) {
            List<Object> logprobs = logprobs(service, choice, result);
            if (!logprobs.isEmpty()) {
                out.put("logprobs", logprobs);
            }
        }
        return out;
    }

    /** Cohere's assistant message: thinking and text content blocks, tool calls, and no empty fields. */
    private static Map<String, Object> message(String reasoning, ToolParse parse) {
        List<Object> content = new ArrayList<>();
        if (!reasoning.isEmpty()) {
            Map<String, Object> thinking = new LinkedHashMap<>();
            thinking.put("type", "thinking");
            thinking.put("thinking", reasoning);
            content.add(thinking);
        }
        if (!parse.content().isEmpty()) {
            Map<String, Object> text = new LinkedHashMap<>();
            text.put("type", "text");
            text.put("text", parse.content());
            content.add(text);
        }
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        if (!content.isEmpty()) {
            message.put("content", content);
        }
        if (!parse.calls().isEmpty()) {
            message.put("tool_calls", toolCalls(parse.calls()));
        }
        return message;
    }

    /** Cohere's tool calls: id, the {@code function} type and the name with its argument text. */
    private static List<Object> toolCalls(List<ToolCall> calls) {
        List<Object> out = new ArrayList<>(calls.size());
        for (ToolCall call : calls) {
            Map<String, Object> fn = new LinkedHashMap<>();
            fn.put("name", call.name());
            fn.put("arguments", call.arguments());
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("id", call.id());
            entry.put("type", "function");
            entry.put("function", fn);
            out.add(entry);
        }
        return out;
    }

    /**
     * Cohere's logprobs: one item per generated token, its text, its id and its log probability, as vLLM maps the
     * OpenAI content entries and token ids.
     */
    private static List<Object> logprobs(EngineService service, ChatChoice choice, GenerationResult result) {
        List<TokenScore[]> captures = choice.logprobs();
        int[] tokens = result.tokens();
        List<Object> out = new ArrayList<>(captures.size());
        for (int i = 0; i < captures.size() && i < tokens.length; i++) {
            TokenScore[] capture = captures.get(i);
            if (capture.length == 0) {
                continue;
            }
            double logprob = capture[0].logprob();
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("text", service.tokenText(tokens[i]));
            item.put("token_ids", List.of(tokens[i]));
            item.put("logprobs", List.of(Double.isFinite(logprob) ? logprob : OpenAiBodies.FLOOR_LOGPROB));
            out.add(item);
        }
        return out;
    }

    /**
     * Cohere's finish reason: a matched stop string is {@code STOP_SEQUENCE}, a token budget
     * {@code MAX_TOKENS}, a cancellation {@code ERROR}, a read tool call {@code TOOL_CALL}, otherwise
     * {@code COMPLETE} — vLLM's mapping of the OpenAI finish reason and {@code stop_reason}.
     */
    static String finishReason(GenerationResult result, boolean calls) {
        return switch (result.stopReason()) {
            case STOP_STRING -> "STOP_SEQUENCE";
            case MAX_TOKENS, CONTEXT_FULL -> "MAX_TOKENS";
            case CANCELLED -> "ERROR";
            case REPETITION -> "COMPLETE";
            default -> calls ? "TOOL_CALL" : "COMPLETE";
        };
    }

    /** Cohere's usage: the same counts as billed units and as tokens, with the cached prefix count beside them. */
    static Map<String, Object> usage(long prompt, long cached, int output) {
        Map<String, Object> billed = new LinkedHashMap<>();
        billed.put("input_tokens", prompt);
        billed.put("output_tokens", output);
        Map<String, Object> tokens = new LinkedHashMap<>();
        tokens.put("input_tokens", prompt);
        tokens.put("output_tokens", output);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("billed_units", billed);
        usage.put("tokens", tokens);
        usage.put("cached_tokens", cached);
        return usage;
    }

    /** Writes the answer's Cohere event stream; a failure becomes an {@code ERROR} message end. */
    private void stream(OutputStream out, GenerationBatch batch, ChatPlan plan, CohereChatRequest request, String id)
            throws IOException {
        CohereChatStream stream = new CohereChatStream(out, id);
        ChatChoice choice = plan.newChoice();
        try {
            while (true) {
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                stream.start();
                if (event instanceof GenerationBatch.Done) {
                    ToolParse parse = choice.finish();
                    drain(stream, choice);
                    String failure = batch.failure(0);
                    if (failure != null) {
                        stream.error(failure);
                        return;
                    }
                    GenerationResult result = batch.result(0);
                    stream.finish(finishReason(result, !parse.calls().isEmpty()),
                            usage(plan.promptRows(), batch.cachedTokens(0), result.timings().generatedTokens()),
                            batch.kvTransferParams(0));
                    return;
                }
                choice.accept(event);
                drain(stream, choice);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            batch.cancel();
            stream.error("stream interrupted");
        } catch (IOException e) {
            batch.cancel();
            throw e;
        } catch (RuntimeException e) {
            batch.cancel();
            stream.error(String.valueOf(e.getMessage()));
        }
    }

    private static void drain(CohereChatStream stream, ChatChoice choice) throws IOException {
        for (ChatChoice.Delta delta : choice.drain()) {
            stream.delta(delta.delta());
        }
    }
}
