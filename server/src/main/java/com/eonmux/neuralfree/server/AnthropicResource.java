package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolOutput;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.utils.json.Json;
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
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * The Anthropic Messages API ({@code POST /v1/messages}, llama.cpp's {@code post_anthropic_messages}): the body
 * maps onto the chat request ({@link AnthropicRequests}: system, text / thinking / tool_use / tool_result blocks,
 * tools and tool_choice, the thinking switch), {@code max_tokens} (required), {@code temperature},
 * {@code top_p}, {@code top_k} and {@code stop_sequences} onto the engine, and the answer is the Anthropic message:
 * {@code thinking}, {@code text} and {@code tool_use} content blocks, {@code stop_reason} ({@code end_turn},
 * {@code max_tokens}, {@code stop_sequence} with the matched sequence, {@code tool_use}) and the token counts.
 * With {@code stream: true} the stream carries Anthropic's event sequence ({@link AnthropicStream}).
 * {@code POST /v1/messages/count_tokens} returns the rendered prompt's token count.
 */
@Path("/")
public class AnthropicResource {

    private static final String ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    @Inject
    EngineService service;

    /** The current HTTP request (a request-scoped proxy), for disconnect cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public AnthropicResource(EngineService service) {
        this.service = service;
    }

    public AnthropicResource() {
    }

    @POST
    @Path("/v1/messages")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response messages(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> messagesOf(ServerJson.parse(bodyText)));
    }

    @POST
    @Path("/v1/messages/count_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response countTokens(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> {
            ChatRequest chat = AnthropicRequests.chat(ServerJson.parse(bodyText));
            int[] prompt = service.encode(service.formatChat(chat), false, true);
            return Response.ok(ServerJson.write(Map.of("input_tokens", prompt.length))).build();
        });
    }

    private Response messagesOf(JsonView body) {
        String model = service.served().resolve(body);
        ChatRequest chat = AnthropicRequests.chat(body);
        GenerationRequest request = AnthropicRequests.request(body);
        String rendered = service.formatChat(chat);
        ToolRouting routing = service.toolRouting(chat, rendered, AnthropicRequests.parallelToolCalls(body));
        int[] prompt = service.encode(rendered, false, true);
        ToolOutput tools = routing.newOutput(AnthropicResource::toolUseId);
        String id = "msg_" + System.nanoTime();
        boolean streamed = body.optBool("stream", false);
        GenerationBatch batch = GenerationBatch.start(service,
                List.of(new GenerationBatch.Job(prompt, request, rendered, routing.constraint())), http);
        if (streamed) {
            StreamingOutput stream = out -> stream(out, batch, tools, id, model, prompt.length);
            return Response.ok(stream).type("text/event-stream").build();
        }
        batch.awaitAll();
        ChatChoice choice = new ChatChoice(tools);
        try {
            GenerationBatch.Event e;
            while ((e = batch.next(0)) != null) {
                choice.accept(e);
            }
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        }
        GenerationResult result = batch.result(0);
        return Response.ok(ServerJson.write(messageBody(id, model, result, choice.reasoning(), choice.finish())))
                .build();
    }

    private Map<String, Object> messageBody(String id, String model, GenerationResult result, String reasoning,
                                            ToolParse parse) {
        List<Object> content = new ArrayList<>();
        if (!reasoning.isEmpty()) {
            Map<String, Object> thinking = new LinkedHashMap<>();
            thinking.put("type", "thinking");
            thinking.put("thinking", reasoning);
            thinking.put("signature", "");
            content.add(thinking);
        }
        if (!parse.content().isEmpty() || parse.calls().isEmpty()) {
            Map<String, Object> text = new LinkedHashMap<>();
            text.put("type", "text");
            text.put("text", parse.content());
            content.add(text);
        }
        for (ToolCall call : parse.calls()) {
            Map<String, Object> use = new LinkedHashMap<>();
            use.put("type", "tool_use");
            use.put("id", call.id());
            use.put("name", call.name());
            use.put("input", input(call.arguments()));
            content.add(use);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", id);
        out.put("type", "message");
        out.put("role", "assistant");
        out.put("content", content);
        out.put("model", model);
        String stop = stopReason(result, !parse.calls().isEmpty());
        out.put("stop_reason", stop);
        out.put("stop_sequence", "stop_sequence".equals(stop) ? result.stopString() : null);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", result.timings().promptTokens());
        usage.put("output_tokens", result.timings().generatedTokens());
        usage.put("cache_read_input_tokens", 0);
        out.put("usage", usage);
        return out;
    }

    private void stream(OutputStream out, GenerationBatch batch, ToolOutput tools, String id, String model,
                        int inputTokens) throws IOException {
        try {
            ServerJson.event(out, "message_start", Map.of("type", "message_start", "message",
                    startMessage(id, model, inputTokens)));
            AnthropicStream blocks = new AnthropicStream(out);
            while (true) {
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                if (event instanceof GenerationBatch.Done) {
                    break;
                }
                route(event, tools, blocks);
            }
            tools.finish(blocks);
            blocks.end();
            String failure = batch.failure(0);
            if (failure != null) {
                ServerJson.event(out, "error", Map.of("type", "error", "error",
                        Map.of("type", "invalid_request_error", "message", failure)));
                return;
            }
            GenerationResult result = batch.result(0);
            String stop = stopReason(result, blocks.calls());
            Map<String, Object> delta = new LinkedHashMap<>();
            delta.put("stop_reason", stop);
            delta.put("stop_sequence", "stop_sequence".equals(stop) ? result.stopString() : null);
            Map<String, Object> messageDelta = new LinkedHashMap<>();
            messageDelta.put("type", "message_delta");
            messageDelta.put("delta", delta);
            messageDelta.put("usage", Map.of("output_tokens", result.timings().generatedTokens()));
            ServerJson.event(out, "message_delta", messageDelta);
            ServerJson.event(out, "message_stop", Map.of("type", "message_stop"));
        } catch (UncheckedIOException e) {
            batch.cancel();
            throw e.getCause();
        } catch (IOException e) {
            batch.cancel();
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            batch.cancel();
            throw new IOException("stream interrupted", e);
        } catch (RuntimeException e) {
            // the request budget ran out or the output could not be mapped: stop the generation, tell the client
            batch.cancel();
            ServerJson.event(out, "error", Map.of("type", "error", "error",
                    Map.of("type", "api_error", "message", String.valueOf(e.getMessage()))));
        }
    }

    private static void route(GenerationBatch.Event event, ToolOutput tools, AnthropicStream blocks) {
        if (event instanceof GenerationBatch.ToolRecipient r) {
            tools.toolCallStarted(r.recipient(), blocks);
        } else if (event instanceof GenerationBatch.Text t) {
            if (t.channel() == ReasoningFilter.Channel.REASONING) {
                blocks.thinking(t.text());
            } else if (t.channel() == ReasoningFilter.Channel.TOOL_CALL) {
                tools.toolCallText(t.text(), blocks);
            } else {
                tools.content(t.text(), blocks);
            }
        }
    }

    private Map<String, Object> startMessage(String id, String model, int inputTokens) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("id", id);
        message.put("type", "message");
        message.put("role", "assistant");
        message.put("content", List.of());
        message.put("model", model);
        message.put("stop_reason", null);
        message.put("stop_sequence", null);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", inputTokens);
        usage.put("output_tokens", 0);
        message.put("usage", usage);
        return message;
    }

    /** Anthropic's stop reason: a tool call, a stop sequence, the budgets, or the end of the turn. */
    private static String stopReason(GenerationResult result, boolean calls) {
        return switch (result.stopReason()) {
            case STOP_STRING -> "stop_sequence";
            case MAX_TOKENS, CONTEXT_FULL -> "max_tokens";
            default -> calls ? "tool_use" : "end_turn";
        };
    }

    /** A call's arguments as the {@code input} object ({@code {}} when they are not a JSON object). */
    private static Object input(String arguments) {
        try {
            Object parsed = Json.parse(arguments, Json.Limits.UNTRUSTED);
            return parsed instanceof Map<?, ?> ? parsed : Map.of();
        } catch (RuntimeException e) {
            return Map.of();
        }
    }

    /** An Anthropic tool-use id: {@code toolu_} and 24 random alphanumerics. */
    static String toolUseId() {
        ThreadLocalRandom r = ThreadLocalRandom.current();
        StringBuilder b = new StringBuilder("toolu_");
        for (int i = 0; i < 24; i++) {
            b.append(ALPHANUMERIC.charAt(r.nextInt(ALPHANUMERIC.length())));
        }
        return b.toString();
    }
}
