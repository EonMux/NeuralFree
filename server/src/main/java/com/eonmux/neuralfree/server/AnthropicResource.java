package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.OutputGates;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.response.BeamGrammar;
import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolOutput;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
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
import java.util.function.Supplier;

/**
 * The Anthropic Messages API ({@code POST /v1/messages}, llama.cpp's {@code post_anthropic_messages}): the body
 * maps onto the chat request ({@link AnthropicRequests}: system, text / thinking / tool_use / tool_result blocks,
 * tools and tool_choice, the thinking switch), {@code max_tokens} (required), {@code temperature},
 * {@code top_p}, {@code top_k} and {@code stop_sequences} onto the engine, and the answer is the Anthropic message:
 * {@code thinking}, {@code text} and {@code tool_use} content blocks, {@code stop_reason} ({@code end_turn},
 * {@code max_tokens}, {@code stop_sequence} with the matched sequence, {@code tool_use}) and the token counts.
 * The usage splits the prompt as Anthropic does: {@code cache_read_input_tokens} are the prompt rows the prefix
 * cache served (the OpenAI routes' {@code prompt_tokens_details.cached_tokens}), {@code input_tokens} the rest, and
 * {@code cache_creation_input_tokens} is 0 — the cache stores every computed prefix itself, with no explicit
 * {@code cache_control} write. With {@code stream: true} the stream carries Anthropic's event sequence
 * ({@link AnthropicStream}), its {@code message_start} written once the prompt is evaluated so its usage holds the
 * cached count, and {@code message_delta} carrying the final usage.
 * {@code POST /v1/messages/count_tokens} returns the rendered prompt's token count, the rows of its images
 * included. Image blocks ({@code base64} and {@code url} sources) reach the projector as image parts, and a
 * repeated image's rows read back from the prefix cache like text.
 */
@Path("/")
public class AnthropicResource {

    private static final String ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    @Inject
    EngineService service;

    /** The current HTTP request (a request-scoped proxy), for disconnect cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public AnthropicResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public AnthropicResource(EngineService service, ModelRouter router) {
        this(service, router, null);
    }

    /** A resource over an explicit service, router and request, so a delegated stream cancels on disconnect. */
    public AnthropicResource(EngineService service, ModelRouter router, HttpServerRequest http) {
        this.service = service;
        this.router = router;
        this.http = http;
    }

    public AnthropicResource() {
    }

    @POST
    @Path("/v1/messages")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response messages(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> messagesOf(target, body));
    }

    @POST
    @Path("/v1/messages/count_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response countTokens(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> {
            ChatRequest chat = AnthropicRequests.chat(body, target.options());
            Supplier<LoraSlotKey> mediaLoras = ChatPlan.mediaLoras(target,
                    target.loras().selection(body, target.loras().model(body)));
            return Response.ok(ServerJson.write(Map.of("input_tokens",
                            OpenAiResource.promptRows(target, chat, mediaLoras))))
                    .build();
        });
    }

    private Response messagesOf(EngineService service, JsonView body) {
        String model = service.loras().model(body);
        ChatRequest chat = AnthropicRequests.chat(body, service.options());
        GenerationRequest request = service.loras().apply(body, model,
                AnthropicRequests.request(body, service.logitsProcessors(), service.options().watermark()));
        Supplier<LoraSlotKey> mediaLoras = ChatPlan.mediaLoras(service, request.lora());
        List<ChatMedia.Item> items = ChatPlan.mediaItems(service, chat, mediaLoras);
        String rendered = service.formatChat(chat, !items.isEmpty());
        ReasoningFilter reasoning = ReasoningFormats.forChat(service.reasoning(), chat);
        // the explicit disable_parallel_tool_use alone refuses a guarantee; the template's one-call default
        // (llama.cpp server-common.cpp:1447) still keeps the parsers to the first call
        ParallelToolCalls parallelToolCalls = ParallelToolCalls.of(AnthropicRequests.parallelToolCalls(body),
                service.defaultParallelToolCalls(chat));
        ToolRouting routing = service.toolRouting(chat, rendered, parallelToolCalls, reasoning);
        ConstraintSpec constraint = routing.constraint();
        // another tool format's forcing grammar holds the output itself
        BeamGrammar beam = constraint != null ? null : ChatPlan.beamGrammar(service, chat, rendered,
                parallelToolCalls, null, reasoning, true);
        if (beam != null) {
            constraint = beam.fromMessageStart() ? beam.constraint() : beam.constraint()
                    .withGate(OutputGates.forChat(reasoning, rendered, service.chatTemplate()));
        }
        int[] prompt = service.encode(rendered, false, true);
        MediaPrompt media = items.isEmpty() ? null
                : ChatPlan.mediaPrompt(service, chat, rendered, items, mediaLoras);
        int promptRows = media == null ? prompt.length : media.nTokens();
        GenerationBatch.Job job = media == null
                ? ChatPlan.textJob(service, chat, prompt, request, rendered, constraint, reasoning)
                : GenerationBatch.Job.ofMedia(media, request, rendered, constraint, reasoning);
        ToolOutput tools = routing.newOutput(AnthropicResource::toolUseId);
        String id = "msg_" + System.nanoTime();
        boolean streamed = body.optBool("stream", false);
        GenerationBatch batch = GenerationBatch.start(service,
                List.of(job), http, id);
        if (streamed) {
            StreamingOutput stream = out -> stream(out, batch, tools, id, model, promptRows);
            return Response.ok(stream).type("text/event-stream").build();
        }
        batch.awaitAll();
        ChatChoice choice = new ChatChoice(tools, true);
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
        Map<String, Object> usage = usage(promptRows, batch.cachedTokens(0), result.timings().generatedTokens());
        return Response.ok(ServerJson.write(messageBody(id, model, result, choice.messageReasoning(), choice.finish(),
                usage, batch.kvTransferParams(0)))).build();
    }

    /**
     * Anthropic's usage over a prompt of {@code promptTokens} rows (tokens and media rows), {@code cached} of them
     * served by the prefix cache: {@code input_tokens} the uncached rest, {@code cache_creation_input_tokens} 0 (no explicit cache
     * write), {@code cache_read_input_tokens} the cached ones, and the generated {@code output_tokens}.
     */
    static Map<String, Object> usage(int promptTokens, long cached, int outputTokens) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", Math.max(0, promptTokens - cached));
        usage.put("cache_creation_input_tokens", 0);
        usage.put("cache_read_input_tokens", cached);
        usage.put("output_tokens", outputTokens);
        return usage;
    }

    private Map<String, Object> messageBody(String id, String model, GenerationResult result, String reasoning,
                                            ToolParse parse, Map<String, Object> usage,
                                            Map<String, Object> kvTransfer) {
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
        out.put("usage", usage);
        if (kvTransfer != null) {
            out.put("kv_transfer_params", kvTransfer);
        }
        return out;
    }

    /**
     * Writes the message's event stream: {@code message_start} with the first generation event (the prompt is
     * evaluated by then, so its usage holds the cached count), the content blocks, then {@code message_delta}
     * with the stop reason, the final usage and a disaggregated transfer's {@code kv_transfer_params}, and
     * {@code message_stop}.
     */
    private void stream(OutputStream out, GenerationBatch batch, ToolOutput tools, String id, String model,
                        int promptTokens) throws IOException {
        EventDrain.HeldFlushes held = new EventDrain.HeldFlushes(out);
        try {
            AnthropicStream blocks = new AnthropicStream(held);
            boolean started = false;
            boolean ended = false;
            while (!ended) {
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                if (!started) {
                    ServerJson.event(out, "message_start", Map.of("type", "message_start", "message",
                            startMessage(id, model, usage(promptTokens, batch.cachedTokens(0), 0))));
                    started = true;
                }
                ended = EventDrain.drain(batch, event, held, e -> {
                    if (e instanceof GenerationBatch.Done) {
                        return true;
                    }
                    route(e, tools, blocks);
                    return false;
                });
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
            messageDelta.put("usage", usage(promptTokens, batch.cachedTokens(0), result.timings().generatedTokens()));
            Map<String, Object> transfer = batch.kvTransferParams(0);
            if (transfer != null) {
                messageDelta.put("kv_transfer_params", transfer);
            }
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

    private Map<String, Object> startMessage(String id, String model, Map<String, Object> usage) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("id", id);
        message.put("type", "message");
        message.put("role", "assistant");
        message.put("content", List.of());
        message.put("model", model);
        message.put("stop_reason", null);
        message.put("stop_sequence", null);
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
