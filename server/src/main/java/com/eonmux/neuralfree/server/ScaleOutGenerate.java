package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.OutputGates;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.grammar.ConstraintGate;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The token-in, token-out generate endpoint ({@code POST /inference/v1/generate}, vLLM's scale-out server): a
 * {@code token_ids} prompt and a {@code sampling_params} object run through the engine without the server rendering
 * a prompt; {@code output_mode} {@code tokens} answers token ids, {@code text} the detokenized output as well,
 * {@code derender} the parsed assistant message, and {@code stream} delivers the output as deltas. The sampling
 * parameters are vLLM's names over this engine's request, and any field the server does not run is refused by name.
 *
 * <p>A prompt rendered by another server may carry multimodal {@code features} (item hashes, placeholder spans and
 * serialized sources) or {@code content_parts} (raw media parts at {@code -1} placeholders); the items are spliced
 * back in through the media cache, keyed by their hashes, so the encoded rows travel between the two servers.</p>
 */
final class ScaleOutGenerate {

    /** The generate request fields this server runs; every other field is refused by name. */
    private static final Set<String> REQUEST_FIELDS = Set.of("request_id", "token_ids", "token_offsets", "features",
            "content_parts", "sampling_params", "model", "reasoning_ended", "reasoning_parser_kwargs",
            "return_token_ids", "output_mode", "parse_context", "stream", "stream_options", "cache_salt", "priority",
            "kv_transfer_params", "ec_transfer_params");

    private ScaleOutGenerate() {
    }

    /** One parsed generate request, ready to submit. */
    record Parsed(String requestId, int[] tokenIds, String model, int n, boolean stream, String outputMode,
                  boolean returnTokenIds, boolean includeUsage, boolean continuousUsage, long created,
                  int promptLogprobs, Map<String, Object> mmPlaceholders, JsonView parseContext,
                  List<GenerationBatch.Job> jobs) {

        /** Whether the answer carries decoded text. */
        boolean text() {
            return !outputMode.equals("tokens");
        }

        /** Whether the answer carries parsed assistant messages. */
        boolean derender() {
            return outputMode.equals("derender");
        }
    }

    /** Runs the request: the answer object, or its event stream. */
    static Response generate(EngineService service, JsonView body, HttpServerRequest http,
                             Map<String, GenerationBatch> active) {
        Parsed parsed = parse(service, body, http);
        GenerationBatch batch = GenerationBatch.start(service, parsed.jobs(), http);
        active.put(parsed.requestId(), batch);
        batch.onAllDone(() -> active.remove(parsed.requestId(), batch));
        if (parsed.stream()) {
            StreamingOutput out = o -> stream(o, batch, parsed, service);
            return Response.ok(out).type("text/event-stream").build();
        }
        try {
            batch.awaitAll();
            return Response.ok(ServerJson.write(full(batch, parsed, service))).build();
        } finally {
            active.remove(parsed.requestId(), batch);
        }
    }

    private static Parsed parse(EngineService service, JsonView body, HttpServerRequest http) {
        for (String key : body.keys()) {
            if (!REQUEST_FIELDS.contains(key)) {
                throw ServerJson.badRequest("'" + key + "' is not supported by the generate endpoint");
            }
        }
        Object raw = body.raw("token_ids");
        if (!(raw instanceof List<?> tokens) || tokens.isEmpty()) {
            throw ServerJson.badRequest("token_ids is required and must be a non-empty array of token ids");
        }
        if (body.raw("ec_transfer_params") != null) {
            throw ServerJson.badRequest("ec_transfer_params is not supported by this server");
        }
        if (body.raw("sampling_params") == null) {
            throw ServerJson.badRequest("sampling_params is required and must be an object");
        }
        JsonView sampling = body.obj("sampling_params");
        boolean stream = body.optBool("stream", false);
        String outputMode = body.optStr("output_mode", "tokens");
        if (!outputMode.equals("tokens") && !outputMode.equals("text") && !outputMode.equals("derender")) {
            throw ServerJson.badRequest("output_mode must be tokens, text or derender, got '" + outputMode + "'");
        }
        if (stream && outputMode.equals("derender")) {
            throw ServerJson.badRequest("output_mode derender does not support stream true");
        }
        ScaleOutRequests.requireOutputKind(sampling, stream);
        boolean detokenize = ScaleOutRequests.detokenize(sampling);
        if (!detokenize && (!outputMode.equals("tokens") || sampling.raw("stop") != null)) {
            throw ServerJson.badRequest("sampling_params.detokenize false needs output_mode tokens and no stop"
                    + " strings");
        }
        JsonView parseContext = body.raw("parse_context") == null ? null : body.obj("parse_context");
        if (outputMode.equals("derender") && parseContext == null && needsParseContext(service)) {
            throw ServerJson.badRequest("output_mode derender requires parse_context on a server with a tool or"
                    + " reasoning parser; send the one /render returned");
        }
        Input input = input(service, body, tokens);
        int[] tokenIds = input.tokenIds();
        tokenOffsets(body, tokenIds);
        GenerationRequest request = ScaleOutRequests.sampling(service, sampling);
        String model = service.loras().model(body);
        request = service.loras().apply(body, model, request);
        if (body.raw(com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferParams.FIELD) != null) {
            request = request.toBuilder().kvTransfer(OpenAiRequests.kvTransfer(body)).build();
        }
        if (body.raw("priority") != null) {
            request = request.toBuilder().priority(OpenAiRequests.priority(body)).build();
        }
        if (body.raw("cache_salt") != null) {
            request = request.toBuilder().cacheSalt(OpenAiRequests.cacheSalt(body)).build();
        }
        OpenAiRequests.requireServedPriority(request, service.schedulingPolicy());
        int n = OpenAiRequests.choices(sampling);
        OpenAiRequests.requireSingleTransfer(request, 1, n, 1);
        int promptLogprobs = OpenAiRequests.promptLogprobs(sampling);
        if (stream && promptLogprobs != 0) {
            throw ServerJson.badRequest("prompt_logprobs are not available when stream is true");
        }
        ConstraintSpec constraint = OpenAiRequests.constraint(sampling);
        if (constraint != null) {
            constraint = gate(constraint, body, service, service.decode(tokenIds));
        }
        List<GenerationBatch.Job> jobs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            GenerationRequest choice = OpenAiRequests.forChoice(request, i);
            jobs.add(input.media() != null
                    ? GenerationBatch.Job.ofMedia(input.media(), choice, null, constraint, ReasoningFilter.none())
                    : new GenerationBatch.Job(tokenIds, choice, null, constraint, ReasoningFilter.none()));
        }
        JsonView options = body.optObj("stream_options");
        boolean includeUsage = stream && options.optBool("include_usage", false);
        boolean continuousUsage = includeUsage && options.optBool("continuous_usage_stats", false);
        String requestId = requestId(http, body);
        Map<String, Object> placeholders = body.optBool("return_token_ids", false) ? input.mmPlaceholders() : null;
        return new Parsed(requestId, tokenIds, model, n, stream, outputMode,
                body.optBool("return_token_ids", false), includeUsage, continuousUsage,
                System.currentTimeMillis() / 1000, promptLogprobs, placeholders, parseContext, jobs);
    }

    /** One parsed prompt: its wire token ids, the media prompt when the request carried multimodal input. */
    private record Input(int[] tokenIds, MediaPrompt media, Map<String, Object> mmPlaceholders) { }

    /**
     * The prompt of a generate request: plain token ids, or the media prompt of its {@code features} (an expanded
     * token sequence whose placeholder spans name the items) or {@code content_parts} (a collapsed token sequence
     * whose {@code -1} tokens name where each part goes).
     */
    @SuppressWarnings("unchecked")
    private static Input input(EngineService service, JsonView body, List<?> tokens) {
        boolean features = body.raw("features") != null;
        boolean parts = body.raw("content_parts") != null;
        if (features && parts) {
            throw ServerJson.badRequest("features and content_parts are mutually exclusive");
        }
        if (features) {
            int[] ids = OpenAiRequests.tokens(tokens, service.vocabSize(), "token_ids");
            List<ChatMedia.Placed> placed = ScaleOutFeatures.placed(service, body.obj("features"), ids.length);
            MediaPrompt prompt = media(service, ids, placed);
            if (prompt.nTokens() != ids.length) {
                throw ServerJson.badRequest("the media prompt fills " + prompt.nTokens() + " rows for "
                        + ids.length + " token ids");
            }
            return new Input(ids, prompt, placeholders(body.obj("features")));
        }
        if (parts) {
            Object rawParts = body.raw("content_parts");
            if (!(rawParts instanceof List<?> list) || list.isEmpty()) {
                throw ServerJson.badRequest("content_parts must be a non-empty array of media parts");
            }
            int[] collapsed = collapsedTokens(tokens, service);
            List<ChatMedia.Item> items = ScaleOutFeatures.contentItems(service, list);
            ChatMedia.Rendered rendered = partsPrompt(service, collapsed, items);
            MediaPrompt prompt = rendered.prompt();
            return new Input(ScaleOutFeatures.expandedIds(prompt), prompt,
                    ScaleOutFeatures.placeholdersOf(rendered.placements()));
        }
        return new Input(OpenAiRequests.tokens(tokens, service.vocabSize(), "token_ids"), null, null);
    }

    /** The media prompt of {@code ids} with its items spliced in place, or a 400 naming what is malformed. */
    private static MediaPrompt media(EngineService service, int[] ids, List<ChatMedia.Placed> placed) {
        try {
            return service.media().promptAt(ids, placed, text -> service.encode(text, false, true));
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The media prompt of a collapsed prompt with one item per {@code -1} token. */
    private static ChatMedia.Rendered partsPrompt(EngineService service, int[] collapsed,
                                                  List<ChatMedia.Item> items) {
        try {
            return service.media().partsWithPlacements(collapsed, items, text -> service.encode(text, false, true));
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The collapsed token ids of a content-parts request: {@code -1} at each media placeholder. */
    private static int[] collapsedTokens(List<?> tokens, EngineService service) {
        int[] out = new int[tokens.size()];
        int media = 0;
        for (int i = 0; i < out.length; i++) {
            Object value = tokens.get(i);
            if (!(value instanceof Number number) || number.longValue() != number.doubleValue()
                    || number.longValue() > Integer.MAX_VALUE) {
                throw ServerJson.badRequest("token_ids[" + i + "] must be an integer token id");
            }
            long id = number.longValue();
            if (id == -1) {
                out[i] = -1;
                media++;
            } else if (id >= 0 && id < service.vocabSize()) {
                out[i] = (int) id;
            } else {
                throw ServerJson.badRequest("token_ids[" + i + "] token id " + id + " is outside the vocabulary;"
                        + " content_parts expects -1 at the media placeholders");
            }
        }
        if (media == 0) {
            throw ServerJson.badRequest("content_parts needs at least one -1 token in token_ids for its media"
                    + " placeholders");
        }
        return out;
    }

    /** The {@code mm_placeholders} of a features object, passed through to the answer. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> placeholders(JsonView features) {
        return (Map<String, Object>) features.raw("mm_placeholders");
    }

    /**
     * Checks the request's {@code token_offsets}: when present they must give one non-negative, ordered
     * {@code [start, end]} span per prompt token. They describe the prompt; generation does not read them.
     */
    private static void tokenOffsets(JsonView body, int[] tokenIds) {
        Object raw = body.raw("token_offsets");
        if (raw == null) {
            return;
        }
        if (!(raw instanceof List<?> offsets) || offsets.size() != tokenIds.length) {
            throw ServerJson.badRequest("token_offsets must be an array with one entry per token_id, matching "
                    + tokenIds.length + " token(s)");
        }
        int previous = 0;
        for (int i = 0; i < offsets.size(); i++) {
            if (!(offsets.get(i) instanceof List<?> span) || span.size() != 2
                    || !(span.get(0) instanceof Number start) || !(span.get(1) instanceof Number end)) {
                throw ServerJson.badRequest("token_offsets[" + i + "] must be a [start, end] pair");
            }
            long from = start.longValue();
            long to = end.longValue();
            if (from == 0 && to == 0) {
                continue; // a token with no source span (an inserted special token), as vLLM renders them
            }
            if (from < previous || to < from) {
                throw ServerJson.badRequest("token_offsets[" + i + "] must not overlap or precede the previous"
                        + " token, got [" + from + ", " + to + "] after " + previous);
            }
            previous = (int) to;
        }
    }

    /** Whether this server parses output with a tool or reasoning parser, so derender needs its parse context. */
    private static boolean needsParseContext(EngineService service) {
        if (service.reasoning().mode() != ReasoningFilter.Mode.NONE) {
            return true;
        }
        String parser = service.config().toolParser();
        return parser != null && !parser.equalsIgnoreCase("none") && !parser.equalsIgnoreCase("auto");
    }

    /**
     * The output constraint gated behind the reasoning end: a request that says the prompt already ended reasoning
     * ({@code reasoning_ended: true}) or whose decoded prompt does not open a reasoning block constrains from the
     * first token, otherwise the constraint starts after the model's reasoning end tag.
     */
    private static ConstraintSpec gate(ConstraintSpec constraint, JsonView body, EngineService service,
                                       String renderedPrompt) {
        boolean ended;
        if (body.raw("reasoning_ended") != null) {
            ended = body.bool("reasoning_ended");
        } else {
            ended = !service.reasoning().startsInReasoning(renderedPrompt);
        }
        if (ended) {
            return constraint;
        }
        ConstraintGate gate = OutputGates.forChat(service.reasoning(), renderedPrompt, service.chatTemplate());
        return gate == null ? constraint : constraint.withGate(gate);
    }

    /** The request id the answer carries: its {@code request_id}, the {@code X-Request-Id} header, or a fresh id. */
    private static String requestId(HttpServerRequest http, JsonView body) {
        String named = OpenAiRequests.requestId(http, body);
        return "generate-tokens-" + (named != null && !named.isEmpty() ? named : System.nanoTime());
    }

    // ------------------------------------------------------------------ non-streamed

    /** The complete answer: every choice's final tokens (and text or message), the prompt ids and the usage. */
    private static Map<String, Object> full(GenerationBatch batch, Parsed parsed, EngineService service) {
        List<Object> choices = new ArrayList<>(parsed.n());
        long completion = 0;
        JsonView chatBody = parsed.derender() && parsed.parseContext() != null
                ? chatBody(parsed.parseContext()) : null;
        boolean includeReasoning = chatBody == null || chatBody.optBool("include_reasoning", true);
        String renderedPrompt = parsed.derender() ? service.decode(parsed.tokenIds()) : null;
        for (int i = 0; i < parsed.n(); i++) {
            GenerationResult result = batch.result(i);
            completion += result.timings().generatedTokens();
            Map<String, Object> choice = new LinkedHashMap<>();
            choice.put("index", i);
            choice.put("stop_reason", OpenAiBodies.stopReason(result));
            Map<String, Object> message = parsed.derender()
                    ? parsed.parseContext() == null ? ScaleOutDerender.plainMessage(result.text())
                            : ScaleOutDerender.parsedMessage(service, chatBody, result.tokens(), renderedPrompt,
                                    result.text())
                    : null;
            String finish = OpenAiBodies.finishReason(result.stopReason());
            choice.put("finish_reason", message == null ? finish
                    : ScaleOutDerender.finishReason(finish, message.containsKey("tool_calls")));
            if (message != null) {
                choice.put("message", message);
            }
            if (!includeReasoning) {
                // a hidden reasoning keeps its text and logprobs out of the answer, as coupled chat does
                choice.put("logprobs", null);
            } else {
                choice.put("token_ids", ScaleOutRequests.ints(result.tokens()));
                if (parsed.text() && !parsed.derender() && result.text() != null) {
                    choice.put("text", result.text());
                }
                if (!result.logprobs().isEmpty()) {
                    choice.put("logprobs", logprobs(result.logprobs(), parsed.text(), service));
                }
            }
            choices.add(choice);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("request_id", parsed.requestId());
        out.put("created", parsed.created());
        out.put("model", parsed.model());
        out.put("output_mode", parsed.outputMode());
        if (parsed.returnTokenIds()) {
            out.put("prompt_token_ids", ScaleOutRequests.ints(parsed.tokenIds()));
            if (parsed.mmPlaceholders() != null) {
                out.put("mm_placeholders", parsed.mmPlaceholders());
            }
        }
        if (parsed.promptLogprobs() != 0) {
            out.put("prompt_logprobs", promptLogprobs(batch.result(0).promptLogprobs(), service));
        }
        out.put("choices", choices);
        out.put("usage", usage(parsed.tokenIds().length, completion));
        Map<String, Object> transfer = batch.kvTransferParams(0);
        if (transfer != null) {
            out.put("kv_transfer_params", transfer);
        }
        return out;
    }

    /**
     * The chat body a derender parses its output with: the {@code parse_context} fields of a render answer over an
     * empty user message, which the parsers only need for their tools and template knobs.
     */
    private static JsonView chatBody(JsonView parseContext) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("messages", List.of(Map.of("role", "user", "content", "")));
        for (String key : parseContext.keys()) {
            switch (key) {
                case "tools":
                case "tool_choice":
                case "parallel_tool_calls":
                case "include_reasoning":
                case "chat_template_kwargs":
                    body.put(key, parseContext.raw(key));
                    break;
                case "history_tool_call_cnt":
                    break;
                default:
                    throw ServerJson.badRequest("parse_context." + key + " is not supported by this server");
            }
        }
        return JsonView.of(body, "parse_context");
    }

    // ------------------------------------------------------------------ streamed

    /**
     * Writes the event stream: token (or text) deltas in generation order, each choice's finish chunk, the usage
     * chunk when requested, and the terminal {@code [DONE]}. Text held back as a stop-string prefix is released with
     * the tokens that complete it; a token none of whose text was released is reported with the tokens after it.
     */
    private static void stream(OutputStream out, GenerationBatch batch, Parsed parsed, EngineService service)
            throws IOException {
        int n = parsed.n();
        int[] emitted = new int[n];
        boolean[] buffering = new boolean[n];
        int[] generated = new int[n];
        long[] lastLogprob = new long[n];
        Arrays.fill(lastLogprob, -1);
        boolean[] promptIdsSent = new boolean[1];
        StringBuilder[] sent = new StringBuilder[n];
        for (int i = 0; i < n; i++) {
            sent[i] = new StringBuilder();
        }
        try {
            int done = 0;
            while (done < n) {
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                int choice = event.choice();
                if (event instanceof GenerationBatch.Text text) {
                    generated[choice] = Math.max(generated[choice], text.predicted());
                    int index = text.predicted() - 1;
                    if (parsed.text()) {
                        writeText(out, parsed, service, choice, index, text, emitted, sent, lastLogprob, generated,
                                promptIdsSent);
                    } else {
                        writeTokens(out, parsed, service, choice, index, text, emitted, buffering, generated,
                                promptIdsSent);
                    }
                } else if (event instanceof GenerationBatch.Done) {
                    done++;
                    finish(out, batch, parsed, service, choice, emitted, sent, generated, lastLogprob,
                            promptIdsSent);
                }
            }
            if (parsed.includeUsage()) {
                Map<String, Object> chunk = new LinkedHashMap<>();
                chunk.put("request_id", parsed.requestId());
                chunk.put("choices", List.of());
                chunk.put("usage", usage(parsed.tokenIds().length, sum(generated)));
                ServerJson.event(out, null, chunk);
            }
            ServerJson.data(out, "[DONE]");
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
            try {
                ServerJson.data(out, "[DONE]");
            } catch (IOException ignored) {
                // the client is gone; the error frame was the last write
            }
        }
    }

    /** Emits the tokens mode delta of one event, holding the tail back when a stop-string prefix delayed it. */
    private static void writeTokens(OutputStream out, Parsed parsed, EngineService service, int choice, int index,
                                    GenerationBatch.Text text, int[] emitted, boolean[] buffering, int[] generated,
                                    boolean[] promptIdsSent) throws IOException {
        if (buffering[choice] || index != emitted[choice] || text.tokenId() < 0) {
            if (index > emitted[choice]) {
                buffering[choice] = true;
            }
            return;
        }
        Map<String, Object> chunk = chunk(parsed, choice, null, null);
        Map<String, Object> bodyChoice = choiceOf(chunk);
        bodyChoice.put("token_ids", listOf(text.tokenId()));
        if (text.logprobs() != null) {
            bodyChoice.put("logprobs", logprobs(List.<TokenScore[]>of(text.logprobs()), false, service));
        }
        emitted[choice]++;
        if (parsed.continuousUsage()) {
            chunk.put("usage", usage(parsed.tokenIds().length, generated[choice]));
        }
        emit(out, chunk, parsed, promptIdsSent);
    }

    /** Emits the text mode delta of one event, its token id when the event released a token. */
    private static void writeText(OutputStream out, Parsed parsed, EngineService service, int choice, int index,
                                  GenerationBatch.Text text, int[] emitted, StringBuilder[] sent, long[] lastLogprob,
                                  int[] generated, boolean[] promptIdsSent) throws IOException {
        String piece = text.text() == null ? "" : text.text();
        boolean released = index == emitted[choice] && text.tokenId() >= 0;
        List<Integer> ids = released ? listOf(text.tokenId()) : null;
        if (released) {
            emitted[choice]++;
        }
        Map<String, Object> capture = null;
        if (text.logprobs() != null && index > lastLogprob[choice]) {
            capture = logprobs(List.<TokenScore[]>of(text.logprobs()), true, service);
            lastLogprob[choice] = index;
        }
        if (piece.isEmpty() && ids == null && capture == null) {
            return;
        }
        if (!piece.isEmpty()) {
            sent[choice].append(piece);
        }
        Map<String, Object> chunk = chunk(parsed, choice, null, null);
        Map<String, Object> bodyChoice = choiceOf(chunk);
        bodyChoice.put("text", piece);
        if (ids != null) {
            bodyChoice.put("token_ids", ids);
        }
        if (capture != null) {
            bodyChoice.put("logprobs", capture);
        }
        if (parsed.continuousUsage()) {
            chunk.put("usage", usage(parsed.tokenIds().length, generated[choice]));
        }
        emit(out, chunk, parsed, promptIdsSent);
    }

    /** Emits the tail the stop filter held back and the choice's finish chunk. */
    private static void finish(OutputStream out, GenerationBatch batch, Parsed parsed, EngineService service,
                               int choice, int[] emitted, StringBuilder[] sent, int[] generated,
                               long[] lastLogprob, boolean[] promptIdsSent) throws IOException {
        String failure = batch.failure(choice);
        if (failure != null) {
            ServerJson.event(out, null, ServerJson.errorBody(failure, "invalid_request_error"));
            return;
        }
        GenerationResult result = batch.result(choice);
        Map<String, Object> chunk = chunk(parsed, choice, null, null);
        Map<String, Object> bodyChoice = choiceOf(chunk);
        if (parsed.text()) {
            String full = result.text();
            String already = sent[choice].toString();
            String rest = full.length() > already.length() && full.startsWith(already)
                    ? full.substring(already.length()) : "";
            if (!rest.isEmpty()) {
                bodyChoice.put("text", rest);
                sent[choice].append(rest);
            }
            List<TokenScore[]> captures = result.logprobs();
            int from = (int) Math.min(captures.size(), lastLogprob[choice] + 1);
            if (captures.size() > from) {
                bodyChoice.put("logprobs", logprobs(captures.subList(from, captures.size()), true, service));
            }
        } else {
            List<Integer> tail = tail(result.tokens(), emitted[choice]);
            if (!tail.isEmpty()) {
                bodyChoice.put("token_ids", tail);
                List<TokenScore[]> captures = result.logprobs();
                if (captures.size() >= result.tokens().length && emitted[choice] < captures.size()) {
                    bodyChoice.put("logprobs", logprobs(captures.subList(emitted[choice], captures.size()), false,
                            service));
                }
            }
            emitted[choice] = result.tokens().length;
        }
        generated[choice] = result.tokens().length;
        bodyChoice.put("finish_reason", OpenAiBodies.finishReason(result.stopReason()));
        bodyChoice.put("stop_reason", OpenAiBodies.stopReason(result));
        if (parsed.continuousUsage()) {
            chunk.put("usage", usage(parsed.tokenIds().length, generated[choice]));
        }
        Map<String, Object> transfer = batch.kvTransferParams(choice);
        if (transfer != null) {
            chunk.put("kv_transfer_params", transfer);
        }
        emit(out, chunk, parsed, promptIdsSent);
    }

    /**
     * Writes one chunk, stamping the prompt token ids on the stream's first chunk when
     * {@code return_token_ids} asked for them (vLLM's first-chunk-only placement).
     */
    private static void emit(OutputStream out, Map<String, Object> chunk, Parsed parsed, boolean[] promptIdsSent)
            throws IOException {
        if (parsed.returnTokenIds() && !promptIdsSent[0]) {
            chunk.put("prompt_token_ids", ScaleOutRequests.ints(parsed.tokenIds()));
            if (parsed.mmPlaceholders() != null) {
                chunk.put("mm_placeholders", parsed.mmPlaceholders());
            }
            promptIdsSent[0] = true;
        }
        ServerJson.event(out, null, chunk);
    }

    /** The chunk head with one empty choice. */
    private static Map<String, Object> chunk(Parsed parsed, int choice, String finish, Object stopReason) {
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("request_id", parsed.requestId());
        chunk.put("output_mode", parsed.text() ? "text" : "tokens");
        Map<String, Object> bodyChoice = new LinkedHashMap<>();
        bodyChoice.put("index", choice);
        bodyChoice.put("finish_reason", finish);
        bodyChoice.put("stop_reason", stopReason);
        chunk.put("choices", List.of(bodyChoice));
        return chunk;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> choiceOf(Map<String, Object> chunk) {
        return (Map<String, Object>) ((List<?>) chunk.get("choices")).get(0);
    }

    /** The generated tokens from {@code from} on. */
    private static List<Integer> tail(int[] tokens, int from) {
        if (from >= tokens.length) {
            return List.of();
        }
        return ScaleOutRequests.ints(Arrays.copyOfRange(tokens, from, tokens.length));
    }

    private static List<Integer> listOf(int value) {
        return List.of(value);
    }

    private static int sum(int[] values) {
        int total = 0;
        for (int value : values) {
            total += value;
        }
        return total;
    }

    // ------------------------------------------------------------------ shared shapes

    /** vLLM's usage object over the prompt tokens and the generated count. */
    static Map<String, Object> usage(long promptTokens, long completionTokens) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", promptTokens);
        usage.put("completion_tokens", completionTokens);
        usage.put("total_tokens", promptTokens + completionTokens);
        return usage;
    }

    /**
     * vLLM's generate logprobs: {@code {"content": [{token_id, logprob, rank, top_logprobs}]}} in tokens mode,
     * the OpenAI chat form ({@code token}, {@code logprob}, {@code bytes}, {@code top_logprobs}) in text mode.
     */
    static Map<String, Object> logprobs(List<TokenScore[]> steps, boolean text, EngineService service) {
        List<Object> content = new ArrayList<>(steps.size());
        for (TokenScore[] step : steps) {
            content.add(logprobEntry(step, text, service));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("content", content);
        return out;
    }

    private static Object logprobEntry(TokenScore[] step, boolean text, EngineService service) {
        List<Object> tops = new ArrayList<>(step.length);
        for (TokenScore score : step) {
            tops.add(logprob(score, text, service));
        }
        Map<String, Object> entry = logprob(step[0], text, service);
        entry.put("top_logprobs", tops);
        return entry;
    }

    private static Map<String, Object> logprob(TokenScore score, boolean text, EngineService service) {
        Map<String, Object> entry = new LinkedHashMap<>();
        double value = Double.isFinite(score.logprob()) ? score.logprob() : OpenAiBodies.FLOOR_LOGPROB;
        if (text) {
            String token = service.tokenText(score.tokenId());
            entry.put("token", token);
            entry.put("logprob", value);
            List<Object> bytes = new ArrayList<>();
            for (byte b : token.getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
                bytes.add(b & 0xff);
            }
            entry.put("bytes", bytes);
        } else {
            entry.put("token_id", score.tokenId());
            entry.put("logprob", value);
            entry.put("rank", score.rank() > 0 ? score.rank() : null);
        }
        return entry;
    }

    /** vLLM's {@code prompt_logprobs}: null for the first prompt row, then {@code {token_id: {logprob, rank}}} per
     *  row, with the decoded token. */
    static List<Object> promptLogprobs(List<TokenScore[]> captures, EngineService service) {
        List<Object> out = new ArrayList<>(captures.size());
        for (TokenScore[] capture : captures) {
            if (capture == null) {
                out.add(null);
                continue;
            }
            Map<String, Object> row = new LinkedHashMap<>();
            for (TokenScore score : capture) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("logprob", Double.isFinite(score.logprob()) ? score.logprob()
                        : OpenAiBodies.FLOOR_LOGPROB);
                entry.put("rank", Math.max(1, score.rank()));
                entry.put("decoded_token", service.tokenText(score.tokenId()));
                row.put(String.valueOf(score.tokenId()), entry);
            }
            out.add(row);
        }
        return out;
    }
}
