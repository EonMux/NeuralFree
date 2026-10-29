package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
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
 * The Anthropic Messages API ({@code POST /v1/messages}, llama.cpp's {@code post_anthropic_messages}): the
 * {@code system} prompt and the user/assistant messages render through the model's chat template, {@code
 * max_tokens} (required, Anthropic semantics), {@code temperature}, {@code top_p}, {@code top_k} and {@code
 * stop_sequences} map onto the engine, and the response is the Anthropic message body — content blocks, {@code
 * stop_reason} ({@code end_turn}, {@code max_tokens}, {@code stop_sequence}) and the input/output token counts.
 * With {@code stream: true} the stream carries the Anthropic event sequence: {@code message_start}, one text
 * content block with its deltas, then {@code message_delta} and {@code message_stop}. {@code POST
 * /v1/messages/count_tokens} returns the rendered prompt's token count.
 *
 * <p>Refusals carry the gap by name: a message content block that is not text (the projector epic), and a body
 * without {@code max_tokens} (Anthropic requires it). Reasoning deltas of the template's parser stream as text
 * deltas of their own content block; tool use blocks are pending with the tool-call parser epic.
 */
@Path("/")
public class AnthropicResource {

    /** One routed piece on its way to a stream (the reasoning channel kept apart for its own block). */
    private record Piece(ReasoningFilter.Channel channel, String text) { }

    /** The longest wait on one streamed request before the stream gives up (the engine may be stuck). */
    private static final long STREAM_TIMEOUT_MINUTES = 30;

    @Inject
    EngineService service;

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
        requireLoaded();
        JsonView body = parse(bodyText);
        ChatRequest chat = chatRequest(body);
        String rendered = service.formatChat(chat);
        GenerationRequest request = request(body);
        int[] prompt = service.encode(rendered, false, true);
        if (body.optBool("stream", false)) {
            return streamResponse(prompt, request, rendered);
        }
        GenerationResult result = run(prompt, request);
        return Response.ok(json(messageBody(result, body))).build();
    }

    @POST
    @Path("/v1/messages/count_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response countTokens(String bodyText) {
        requireLoaded();
        JsonView body = parse(bodyText);
        ChatRequest chat = chatRequest(body);
        String rendered = service.formatChat(chat);
        int[] prompt = service.encode(rendered, false, true);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("input_tokens", prompt.length);
        return Response.ok(json(out)).build();
    }

    // ------------------------------------------------------------------ request mapping

    /** The chat request of the body: the system prompt first, then the user/assistant messages. */
    private ChatRequest chatRequest(JsonView body) {
        Object raw = body.raw("messages");
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw badRequest("messages is required and must be a non-empty array");
        }
        ChatRequest.Builder chat = ChatRequest.builder();
        Object system = body.raw("system");
        if (system instanceof String s) {
            if (!s.isEmpty()) {
                chat.addMessage(Map.of("role", "system", "content", s));
            }
        } else if (system instanceof List<?> blocks) {
            String text = textOfBlocks(blocks, "system");
            if (!text.isEmpty()) {
                chat.addMessage(Map.of("role", "system", "content", text));
            }
        } else if (system != null) {
            throw badRequest("system must be a string or an array of text blocks");
        }
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> message)) {
                throw badRequest("every message must be an object with role and content");
            }
            String role = String.valueOf(message.get("role"));
            if (!role.equals("user") && !role.equals("assistant")) {
                throw badRequest("message role must be user or assistant, got '" + role + "'");
            }
            Object content = message.get("content");
            String text = content instanceof String s ? s : content instanceof List<?> blocks
                    ? textOfBlocks(blocks, "a " + role + " message") : null;
            if (text == null) {
                throw badRequest("message content must be a string or an array of text blocks");
            }
            chat.addMessage(Map.of("role", role, "content", text));
        }
        return chat.build();
    }

    /** The text of every {@code {"type": "text", "text": ...}} block, concatenated; anything else refuses. */
    private String textOfBlocks(List<?> blocks, String what) {
        StringBuilder text = new StringBuilder();
        for (Object o : blocks) {
            if (!(o instanceof Map<?, ?> block)) {
                throw badRequest("every content block of " + what + " must be an object");
            }
            String type = String.valueOf(block.get("type"));
            if (!type.equals("text")) {
                throw badRequest("content block type '" + type + "' is refused: only text blocks are served"
                        + " (the projector epic owns image and audio blocks)");
            }
            if (block.get("text") instanceof String s) {
                text.append(s);
            }
        }
        return text.toString();
    }

    /** The sampling settings of the body; {@code max_tokens} is required, as Anthropic documents it. */
    private GenerationRequest request(JsonView body) {
        int maxTokens = body.optI32("max_tokens", -1);
        if (maxTokens < 1) {
            throw badRequest("max_tokens is required and must be a positive integer (Anthropic's contract)");
        }
        GenerationRequest.Builder builder = GenerationRequest.builder().maxNewTokens(maxTokens);
        if (body.has("temperature")) {
            builder.sampler(builder.build().sampler().toBuilder().temperature((float) body.f64("temperature")).build());
        }
        if (body.has("top_p")) {
            builder.sampler(builder.build().sampler().toBuilder().topP((float) body.f64("top_p")).build());
        }
        if (body.has("top_k")) {
            builder.sampler(builder.build().sampler().toBuilder().topK(body.optI32("top_k", 0)).build());
        }
        List<String> stops = new ArrayList<>();
        if (body.raw("stop_sequences") instanceof List<?> list) {
            for (Object o : list) {
                if (o instanceof String s) {
                    stops.add(s);
                }
            }
        }
        builder.stopStrings(stops);
        return builder.build();
    }

    // ------------------------------------------------------------------ execution

    /** Runs the submission to its terminal state on the calling thread. */
    private GenerationResult run(int[] prompt, GenerationRequest request) {
        try {
            EngineService.Submission submission = service.submit(prompt, request,
                    (channel, tokenId, text) -> { }, null, null);
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

    /** Streams the submission as the Anthropic event sequence; the pieces cross a queue from the pump thread. */
    private Response streamResponse(int[] prompt, GenerationRequest request, String rendered) {
        BlockingQueue<Piece> pieces = new ArrayBlockingQueue<>(1024);
        String id = "msg_" + System.nanoTime();
        EngineService.Submission submission;
        try {
            submission = service.submit(prompt, request, new ReasoningFilter.Sink() {
                @Override
                public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                    try {
                        pieces.put(new Piece(channel, text));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }, rendered, null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new WebApplicationException("the server is shutting down", 503);
        }
        StreamingOutput stream = out -> streamEvents(out, submission, pieces, id);
        return Response.ok(stream).type("text/event-stream").build();
    }

    private void streamEvents(OutputStream out, EngineService.Submission submission, BlockingQueue<Piece> pieces,
                              String id) throws IOException {
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(STREAM_TIMEOUT_MINUTES);
        try {
            Map<String, Object> start = new LinkedHashMap<>();
            start.put("type", "message_start");
            Map<String, Object> message = new LinkedHashMap<>();
            message.put("id", id);
            message.put("type", "message");
            message.put("role", "assistant");
            message.put("content", List.of());
            message.put("model", service.modelPath());
            message.put("stop_reason", null);
            message.put("stop_sequence", null);
            Map<String, Object> startUsage = new LinkedHashMap<>();
            startUsage.put("input_tokens", 0);
            startUsage.put("output_tokens", 0);
            message.put("usage", startUsage);
            start.put("message", message);
            event(out, "message_start", start);

            Map<String, Object> blockStart = new LinkedHashMap<>();
            blockStart.put("type", "content_block_start");
            blockStart.put("index", 0);
            blockStart.put("content_block", Map.of("type", "text", "text", ""));
            event(out, "content_block_start", blockStart);

            int block = 0;
            while (true) {
                Piece piece = pieces.poll(20, TimeUnit.MILLISECONDS);
                if (piece != null) {
                    if (piece.channel() == ReasoningFilter.Channel.REASONING && block == 0) {
                        // reasoning opens its own text block; the answer block follows at the next index
                        Map<String, Object> stop = new LinkedHashMap<>();
                        stop.put("type", "content_block_stop");
                        stop.put("index", 0);
                        event(out, "content_block_stop", stop);
                        Map<String, Object> start2 = new LinkedHashMap<>();
                        start2.put("type", "content_block_start");
                        start2.put("index", 1);
                        start2.put("content_block", Map.of("type", "text", "text", ""));
                        event(out, "content_block_start", start2);
                        block = 1;
                    }
                    Map<String, Object> delta = new LinkedHashMap<>();
                    delta.put("type", "content_block_delta");
                    delta.put("index", block);
                    delta.put("delta", Map.of("type", "text_delta", "text", piece.text()));
                    event(out, "content_block_delta", delta);
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
                Map<String, Object> delta = new LinkedHashMap<>();
                delta.put("type", "content_block_delta");
                delta.put("index", block);
                delta.put("delta", Map.of("type", "text_delta", "text", piece.text()));
                event(out, "content_block_delta", delta);
            }
            if (submission.refused() != null) {
                event(out, "error", errorBody(String.valueOf(submission.refused().getMessage())));
                return;
            }
            BatchSequence sequence = submission.sequence();
            if (sequence == null || sequence.state() == BatchSequence.State.FAILED) {
                event(out, "error", errorBody(String.valueOf(
                        sequence == null ? "the request never registered" : sequence.error())));
                return;
            }
            GenerationResult result = sequence.result();
            Map<String, Object> blockStop = new LinkedHashMap<>();
            blockStop.put("type", "content_block_stop");
            blockStop.put("index", block);
            event(out, "content_block_stop", blockStop);

            Map<String, Object> messageDelta = new LinkedHashMap<>();
            messageDelta.put("type", "message_delta");
            Map<String, Object> deltaBody = new LinkedHashMap<>();
            deltaBody.put("stop_reason", stopReason(result));
            deltaBody.put("stop_sequence", "stop_sequence".equals(stopReason(result))
                    ? result.stopString() : null);
            messageDelta.put("delta", deltaBody);
            Map<String, Object> deltaUsage = new LinkedHashMap<>();
            deltaUsage.put("output_tokens", result.timings().generatedTokens());
            messageDelta.put("usage", deltaUsage);
            event(out, "message_delta", messageDelta);
            event(out, "message_stop", Map.of("type", "message_stop"));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("stream interrupted", e);
        }
    }

    // ------------------------------------------------------------------ response bodies

    /** The Anthropic message body of a finished generation. */
    private Map<String, Object> messageBody(GenerationResult result, JsonView body) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "msg_" + System.nanoTime());
        out.put("type", "message");
        out.put("role", "assistant");
        List<Object> content = new ArrayList<>();
        Map<String, Object> textBlock = new LinkedHashMap<>();
        textBlock.put("type", "text");
        textBlock.put("text", result.text());
        content.add(textBlock);
        out.put("content", content);
        out.put("model", service.modelPath());
        String stop = stopReason(result);
        out.put("stop_reason", stop);
        out.put("stop_sequence", "stop_sequence".equals(stop) && result.stopString() != null
                ? result.stopString() : null);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", result.timings().promptTokens());
        usage.put("output_tokens", result.timings().generatedTokens());
        usage.put("cache_read_input_tokens", 0);
        out.put("usage", usage);
        return out;
    }

    /** Anthropic's stop reason of a result: end of generation, a stop sequence, or the budgets. */
    private String stopReason(GenerationResult result) {
        switch (result.stopReason()) {
            case STOP_STRING:
                return "stop_sequence";
            case MAX_TOKENS:
            case CONTEXT_FULL:
                return "max_tokens";
            default:
                return "end_turn";
        }
    }

    private void event(OutputStream out, String type, Map<String, Object> body) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("event: ").append(type).append("\ndata: ");
        Json.write(body, Json.Style.API, sb);
        sb.append("\n\n");
        out.write(sb.toString().getBytes(StandardCharsets.UTF_8));
        out.flush();
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
