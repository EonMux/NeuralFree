package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.StopCriteria;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.beam.BeamSearchEngine;
import com.eonmux.neuralfree.gguf.engine.beam.BeamSearchSequence;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechOptions;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * The OpenAI-compatible generation endpoints: {@code GET /v1/models}, {@code POST /v1/completions} and
 * {@code POST /v1/chat/completions}, each with server-sent-event streaming ({@code stream: true}).
 *
 * <ul>
 *   <li>Choices: {@code n} choices per prompt, decoded side by side; with a {@code seed}, choice {@code i}
 *       samples with {@code seed + i}. A completion takes every OpenAI prompt form (string, strings, token ids,
 *       token-id arrays), choice {@code p * n + i} answering prompt {@code p}; {@code echo} prepends the
 *       prompt. {@code prompt_embeds} (vLLM) feeds precomputed embedding rows instead of tokens, one prompt
 *       per entry.</li>
 *   <li>Constraints: {@code grammar}, {@code json_schema}, the vLLM {@code guided_*} fields and
 *       {@code structured_outputs}, and {@code response_format} ({@link OpenAiRequests#constraint}).</li>
 *   <li>Tools: {@code tools} render through the template; {@code tool_choice} {@code auto} reads calls with the
 *       configured parser, {@code required} and a named function add a forcing grammar ({@link ToolRouting}).
 *       Calls come back as {@code message.tool_calls} (streamed as {@code delta.tool_calls}) with finish reason
 *       {@code tool_calls}.</li>
 *   <li>Per-request template and reasoning controls: a {@code chat_template} override renders in place of the
 *       model's template (with vLLM's {@code documents} bound as the {@code documents} variable) on a server
 *       launched with {@code --trust-request-chat-template}, and is refused with 400 otherwise, {@code echo}
 *       prepends the last message's content to the answer, {@code include_reasoning} false drops the reasoning,
 *       and llama.cpp's {@code reasoning_format} selects the reasoning routing ({@link ReasoningFormats}). An
 *       assistant message's reasoning reaches the template under vLLM's {@code reasoning} and its deprecated
 *       {@code reasoning_content} alias ({@link com.eonmux.neuralfree.gguf.chat.ChatMessages#copyOf}).</li>
 *   <li>Reasoning streams and lands as vLLM's {@code reasoning} and llama.cpp's {@code reasoning_content}. Logprobs
 *       follow OpenAI per endpoint: chat's
 *       {@code logprobs: true} with {@code top_logprobs} answers {@code logprobs.content} entries, the
 *       completions' integer {@code logprobs} the legacy {@code tokens}/{@code token_logprobs}/
 *       {@code top_logprobs}/{@code text_offset} arrays; llama.cpp's {@code n_probs} selects the count on
 *       either. {@code stream_options.include_usage}
 *       adds the usage chunk; without it the last finish chunk carries the usage, and
 *       {@code stream_options.continuous_usage_stats} puts the running usage on every chunk.</li>
 *   <li>Control: {@code POST /v1/chat/completions/control} with {@code {"id", "action": "reasoning_end"}} ends the
 *       reasoning of the live completion {@code id} mid-stream (llama.cpp's control task): every still-running
 *       choice whose request armed a budget stage (a reasoning budget or {@code reasoning_control}) forces its
 *       reasoning end sequence from its next sampled token.</li>
 *   <li>Batched chat: {@code POST /v1/chat/completions/batch} takes {@code messages} as a list of conversations
 *       (vLLM's {@code BatchChatCompletionRequest}) and answers one choice per conversation over one shared
 *       generation batch. Streaming, tools and {@code prompt_logprobs} are refused; {@code n} must be 1.</li>
 *   <li>Beam search: {@code use_beam_search} answers one final, non-streamed body whose {@code n} choices are the
 *       ranked beams (the request's {@code n} is the beam width) under vLLM's {@code length_penalty}. Streaming,
 *       audio, server tool execution, grammars, media prompts, {@code best_of} and the request features the search
 *       cannot serve refuse by name ({@link BeamSearchRunner}). The offline batch runner and the Responses API
 *       refuse the field.</li>
 *   <li>vLLM request extras: {@code stop_token_ids} end a choice like an end-of-generation token,
 *       {@code include_stop_str_in_output} keeps the matched stop string, {@code allowed_token_ids} confines
 *       sampling, {@code skip_special_tokens} and {@code spaces_between_special_tokens} shape the rendered text,
 *       {@code repetition_detection} ends a choice whose output repeats a token pattern with
 *       {@code finish_reason: "repetition"}, {@code truncate_prompt_tokens} (with {@code truncation_side})
 *       bounds the prompt, {@code prompt_logprobs} answers each prompt token's log probabilities
 *       (non-streamed; the echoed prompt's count fills it under {@code echo}), {@code return_token_ids} adds {@code token_ids}/{@code prompt_token_ids},
 *       {@code return_tokens_as_token_ids} renders logprob tokens as {@code token_id:{id}}, {@code priority}
 *       orders admission, {@code cache_salt} partitions the prefix cache and an {@code X-Request-Id} header
 *       names the response id. Usage reports {@code prompt_tokens_details.cached_tokens}.
 *       Every choice carries vLLM's {@code stop_reason} (the matched stop string or stop token id, null for EOS
 *       and length), on the response and the stream's finish chunk. {@code echo} with {@code logprobs} on
 *       completions merges the prompt's entries into the legacy arrays (the first prompt token's logprob and
 *       alternatives null), in the response and on the stream's echo chunk. {@code best_of} generates that many
 *       candidates per prompt and returns the best {@code n} by cumulative log probability (a stream with
 *       {@code best_of > n} refuses), {@code logprob_token_ids} names the vocabulary ids whose log probabilities
 *       every generated token reports, {@code vllm_xargs} carries the custom logits processor arguments the
 *       engine's loaded processors read, and {@code kv_transfer_params} runs a disaggregated prefill/decode
 *       transfer whose answer rides the response (and the stream's finish chunk).</li>
 * </ul>
 *
 * <p>A client that disconnects mid-stream cancels its sequences. A streamed completion or chat completion whose
 * request carries {@code X-Conversation-Id} instead records its bytes into a resumable stream session
 * ({@link StreamSession}, served by {@link StreamResource}): a disconnect keeps the generation running and the
 * bytes recording, so the client can reattach with {@code GET /v1/stream?conv_id=}, stop it with
 * {@code DELETE /v1/stream} and ask about it on {@code POST /v1/streams/lookup}.</p>
 */
@Path("/")
public class OpenAiResource {

    @Inject
    EngineService service;

    /** The current HTTP request (a request-scoped proxy), for disconnect cancellation; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The resumable stream sessions a streamed request records into when it names a conversation id. */
    @Inject
    StreamSessions streams;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public OpenAiResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public OpenAiResource(EngineService service, ModelRouter router) {
        this(service, router, null);
    }

    /** A resource over an explicit service, router and request, so a delegated stream cancels on disconnect. */
    public OpenAiResource(EngineService service, ModelRouter router, HttpServerRequest http) {
        this(service, router, http, null);
    }

    /** A resource that also records streams naming a conversation id into {@code streams}. */
    public OpenAiResource(EngineService service, ModelRouter router, HttpServerRequest http,
                          StreamSessions streams) {
        this.service = service;
        this.router = router;
        this.http = http;
        this.streams = streams;
    }

    public OpenAiResource() {
    }

    /**
     * The served models: in router mode the router's list ({@link ModelsResource#listing}, llama.cpp's router
     * answers its model registry on {@code /v1/models} too, {@code reload=true} re-reads the sources); otherwise
     * every served name of the base model, then every per-request LoRA adapter as vLLM lists it ({@code root} its
     * path, {@code parent} the base model's first name).
     */
    @GET
    @Path("/v1/models")
    @Produces(MediaType.APPLICATION_JSON)
    public Response models(@QueryParam("reload") String reload) {
        if (router != null && router.active()) {
            if (reload != null && !reload.isBlank()) {
                try {
                    router.reload();
                } catch (IOException e) {
                    throw ServerJson.error(503, String.valueOf(e.getMessage()), "unavailable_error");
                }
            }
            return Response.ok(ServerJson.write(ModelsResource.listing(router))).build();
        }
        ServerJson.requireLoaded(service);
        List<Object> data = new ArrayList<>();
        for (String name : service.served().names()) {
            data.add(modelCard(name, service.modelPath(), null));
        }
        for (LoraCatalog.Adapter adapter : service.loras().list()) {
            data.add(modelCard(adapter.name(), adapter.path(), service.served().primary()));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("object", "list");
        body.put("data", data);
        return Response.ok(ServerJson.write(body)).build();
    }

    private Map<String, Object> modelCard(String id, String root, String parent) {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("id", id);
        model.put("object", "model");
        model.put("created", 0L);
        model.put("owned_by", "neuralfree");
        model.put("root", root);
        model.put("parent", parent);
        model.put("max_model_len", service.maxModelLength());
        return model;
    }

    @POST
    @Path("/v1/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response completions(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return mapped(() -> completionsOf(target, body));
    }

    @POST
    @Path("/v1/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatCompletions(String bodyText) {
        return chatCompletion(ServerJson.parse(bodyText));
    }

    /**
     * The chat mapping over an already parsed body: the speech-to-text routes carry their audio part in memory, so
     * a clip is not bounded by the request-body JSON limits.
     */
    Response chatCompletion(JsonView body) {
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return mapped(() -> chatOf(target, body));
    }

    /**
     * llama.cpp's mid-stream control, {@code POST /v1/chat/completions/control}: {@code {"id", "action"}} with the
     * {@code reasoning_end} action ends the reasoning of a live completion — its budget stages move into forcing
     * the end sequence (llama.cpp's {@code SERVER_TASK_TYPE_CONTROL}).
     */
    @POST
    @Path("/v1/chat/completions/control")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response control(String bodyText) {
        if (router == null || !router.active()) {
            ServerJson.requireLoaded(service);
        }
        return mapped(() -> controlOf(ServerJson.parse(bodyText)));
    }

    private Response controlOf(JsonView body) {
        String id = body.optStr("id", "");
        if (id.isEmpty()) {
            throw ServerJson.badRequest("id is required: the completion id of a live chat completion");
        }
        String action = body.optStr("action", "");
        if (!"reasoning_end".equals(action)) {
            throw ServerJson.badRequest("unknown control action '" + action + "'; the action is reasoning_end");
        }
        LiveCompletions.Outcome outcome = liveCompletions(id).endReasoning(id);
        switch (outcome.kind()) {
            case NO_COMPLETION:
                throw ServerJson.badRequest("no active completion for this id");
            case NOT_ARMED:
                throw ServerJson.badRequest("reasoning control not enabled for this completion; send"
                        + " reasoning_control: true or a reasoning budget to arm it");
            default: {
                Map<String, Object> out = new LinkedHashMap<>();
                out.put("success", true);
                out.put("forced", outcome.forced());
                return Response.ok(ServerJson.write(out)).build();
            }
        }
    }

    /** llama.cpp's unversioned alias of {@code /v1/models}. */
    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    public Response modelsAlias() {
        return models(null);
    }

    /** llama.cpp's unversioned alias of {@code /v1/chat/completions}. */
    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatCompletionsAlias(String bodyText) {
        return chatCompletions(bodyText);
    }

    /** vLLM's batched chat endpoint: many conversations, one response, one shared generation batch. */
    @POST
    @Path("/v1/chat/completions/batch")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response batchChatCompletions(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return mapped(() -> batchChatOf(target, body));
    }

    /** llama.cpp's prompt token count of a chat request ({@code handle_count_tokens}). */
    @POST
    @Path("/v1/chat/completions/input_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatInputTokens(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return mapped(() -> inputTokens(target, body));
    }

    /** llama.cpp's unversioned alias. */
    @POST
    @Path("/chat/completions/input_tokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response chatInputTokensAlias(String bodyText) {
        return chatInputTokens(bodyText);
    }

    /**
     * {@code {"object": "response.input_tokens", "input_tokens": n}} of the rendered chat prompt: its tokens, and
     * on the media path the rows its images, audio and embedding parts take ({@link #promptRows}).
     */
    static Response inputTokens(EngineService service, JsonView chatBody) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", "response.input_tokens");
        ChatRequest chat = OpenAiRequests.chat(chatBody, service.options());
        out.put("input_tokens", promptRows(service, chat, ChatPlan.mediaLoras(service,
                service.loras().selection(chatBody, service.loras().model(chatBody)))));
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * The KV rows the rendered prompt of {@code chat} fills: its token count, or with media or embedding parts the
     * rows of the media prompt (the media encoded once; the projector caches them for the generation that
     * follows), as llama.cpp counts {@code server_tokens}.
     */
    static int promptRows(EngineService service, ChatRequest chat) {
        return promptRows(service, chat, () -> null);
    }

    /**
     * {@link #promptRows(EngineService, ChatRequest)} for a request naming adapters {@code mediaLoras}: its media
     * encodes and caches under the adapters' identity, so the generation that follows the count hits the cache.
     */
    static int promptRows(EngineService service, ChatRequest chat, Supplier<LoraSlotKey> mediaLoras) {
        List<ChatMedia.Item> items = ChatPlan.mediaItems(service, chat, mediaLoras);
        String rendered = service.formatChat(chat, !items.isEmpty());
        if (items.isEmpty() && !rendered.contains(ChatMessages.PROMPT_EMBEDS_MARKER)) {
            return service.encode(rendered, false, true).length;
        }
        return ChatPlan.mediaPrompt(service, chat, rendered, items, mediaLoras).nTokens();
    }

    /**
     * Runs a request mapping; malformed fields the mapping did not catch and inputs the model cannot take (media
     * parts on a text-only path) are 400s, and an engine that went to sleep while the request was being read (its
     * model or media projector released) is the 503 of the sleeping engine.
     */
    static Response mapped(Supplier<Response> handler) {
        try {
            return handler.get();
        } catch (EngineUnavailableException e) {
            throw ServerJson.error(503, String.valueOf(e.getMessage()), "unavailable_error");
        } catch (JsonException | IllegalArgumentException | UnsupportedOperationException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        }
    }

    /** The live completions tracking {@code id}: the service's own, else a routed model's. */
    private LiveCompletions liveCompletions(String id) {
        if (service.completions().tracks(id)) {
            return service.completions();
        }
        if (router != null && router.active()) {
            for (ModelRouter.Entry entry : router.list()) {
                EngineService open = entry.service();
                if (entry.ready() && open != null && open.completions().tracks(id)) {
                    return open.completions();
                }
            }
        }
        return service.completions();
    }

    // ------------------------------------------------------------------ completions

    private Response completionsOf(EngineService service, JsonView body) {
        CompletionPlan plan = CompletionPlan.of(service, body);
        boolean streamed = body.optBool("stream", false);
        boolean beam = OpenAiRequests.beamSearch(body);
        if (beam && streamed) {
            throw ServerJson.badRequest("use_beam_search cannot serve a stream: the best beams are only known"
                    + " once the search ranked them");
        }
        if (streamed) {
            OpenAiRequests.requireStreamablePromptLogprobs(body);
        }
        if (beam) {
            return beamCompletionsOf(service, plan);
        }
        OpenAiBodies bodies = plan.bodies(service);
        String id = "cmpl-" + requestId();
        long created = System.currentTimeMillis() / 1000;
        boolean includeUsage = body.optObj("stream_options").optBool("include_usage", false);
        String conversationId = conversationId();
        boolean resumable = streamed && conversationId != null && streams != null;
        GenerationBatch batch = GenerationBatch.start(service, plan.jobs(), http, id, !resumable);
        if (streamed) {
            StreamWriter.Flags flags = new StreamWriter.Flags(includeUsage,
                    OpenAiRequests.continuousUsageStats(body), plan.returnTokenIds());
            StreamSession session = session(conversationId);
            BooleanSupplier stop = session == null ? null : session::isCancelled;
            StreamingOutput stream = out -> streamCompletions(out, batch, bodies, id, created, plan, flags,
                    stop);
            return streamed(session, stream);
        }
        batch.awaitAll();
        List<GenerationResult> results = resultsOf(batch);
        long cached = cachedTokens(batch, 0, batch.size());
        if (plan.bestOf() > plan.n()) {
            int[] picked = bestOfSelection(results, plan.n(), plan.bestOf());
            results = selected(results, picked);
            cached = cachedTokensOf(batch, picked);
        }
        Map<String, Object> response = completionBody(bodies, plan, id, created, results, cached);
        putKvTransfer(response, batch, 0);
        return Response.ok(ServerJson.write(response)).build();
    }

    /**
     * The completions body of a beam-search request ({@code use_beam_search}, vLLM's beam search over the
     * completion API): every prompt runs its own search of {@code n} beams ({@link BeamSearchRunner}) and the
     * ranked beams come back prompt-major as choices, the length-penalized score's order. {@code echo} beside
     * {@code logprobs} is refused (vLLM's rule: beam search computes no prompt logprobs to merge), and so are
     * media and embedding prompts, which the search's token resubmission cannot carry.
     */
    private Response beamCompletionsOf(EngineService service, CompletionPlan plan) {
        if (plan.top() >= 0 && !plan.echoes().get(0).isEmpty()) {
            throw ServerJson.badRequest("beam search does not compute prompt logprobs: drop echo or logprobs");
        }
        for (CompletionPlan.Input input : plan.prompts()) {
            if (input.media() != null) {
                throw ServerJson.badRequest("prompt_embeds and multimodal_data: beam search resubmits token"
                        + " sequences and serves token prompts");
            }
        }
        if (plan.constraint() != null) {
            throw ServerJson.badRequest("response_format and grammars (constraint): beam search samples the"
                    + " unconstrained top rows of every step, so the constraint cannot shape the beams;"
                    + " drop response_format or sample plainly");
        }
        BeamSearchEngine.requireSupported(plan.base());
        OpenAiBodies bodies = plan.bodies(service);
        String id = "cmpl-" + requestId();
        long created = System.currentTimeMillis() / 1000;
        List<GenerationResult> results = new ArrayList<>();
        long cached = 0;
        for (CompletionPlan.Input input : plan.prompts()) {
            BeamSearchRunner.Run run = BeamSearchRunner.run(service, plan.base(), input.tokens(), http, id);
            cached += run.cachedRows();
            for (BeamSearchSequence sequence : run.output().sequences()) {
                results.add(beamResult(sequence));
            }
        }
        Map<String, Object> response = completionBody(bodies, plan, id, created, results, cached);
        return Response.ok(ServerJson.write(response)).build();
    }

    /** The completions result of one beam: its output tokens (a terminating EOS dropped), decoded text, end
     *  reason and per-step captures. */
    private static GenerationResult beamResult(BeamSearchSequence sequence) {
        return new GenerationResult(sequence.outputTokens(false), sequence.text(), sequence.stopReason(), null,
                GenerationResult.Timings.total(0, sequence.stepCount(), 0))
                .withLogprobs(sequence.logprobs())
                .withStopTokenId(sequence.stopTokenId());
    }

    /**
     * Adds the choice's {@code kv_transfer_params} to a response body when the request ran a disaggregated
     * transfer: the prefill instance answers the transfer's address and id, which a router forwards to the decode
     * instance.
     */
    static void putKvTransfer(Map<String, Object> body, GenerationBatch batch, int choice) {
        Map<String, Object> params = batch.kvTransferParams(choice);
        if (params != null) {
            body.put("kv_transfer_params", params);
        }
    }

    /** The finished results of a batch in choice order. */
    static List<GenerationResult> resultsOf(GenerationBatch batch) {
        List<GenerationResult> results = new ArrayList<>();
        for (int i = 0; i < batch.size(); i++) {
            results.add(batch.result(i));
        }
        return results;
    }

    /**
     * The completions body: one choice per finished job, the echoes prepended, the token ids and
     * prompt logprobs the plan asks for, and usage over the prompts ({@code cachedTokens} of them from the prefix
     * cache) and the generated tokens.
     */
    static Map<String, Object> completionBody(OpenAiBodies bodies, CompletionPlan plan, String id, long created,
                                              List<GenerationResult> results, long cachedTokens) {
        List<Map<String, Object>> choices = new ArrayList<>();
        for (int i = 0; i < results.size(); i++) {
            GenerationResult result = results.get(i);
            Map<String, Object> choice = new LinkedHashMap<>();
            choice.put("index", i);
            choice.put("text", plan.echoes().get(i) + result.text());
            choice.put("logprobs", choiceLogprobs(bodies, plan, i, result));
            choice.put("finish_reason", OpenAiBodies.finishReason(result.stopReason()));
            choice.put("stop_reason", OpenAiBodies.stopReason(result));
            if (plan.returnTokenIds()) {
                choice.put("token_ids", idsOf(result.tokens()));
                int[] prompt = plan.promptOf(i);
                choice.put("prompt_token_ids", prompt == null ? null : idsOf(prompt));
            }
            if (plan.base().wantsPromptLogprobs()) {
                choice.put("prompt_logprobs", bodies.promptLogprobs(result.promptLogprobs()));
            }
            choices.add(choice);
        }
        Map<String, Object> out = bodies.head(id, "text_completion", created);
        out.put("choices", choices);
        out.put("usage", OpenAiBodies.usage(plan.promptTokens(), cachedTokens, results));
        return out;
    }

    /**
     * The logprobs body of one completion choice: null unless the request asked for {@code logprobs} (a
     * {@code best_of} candidate captures them for its ranking alone); with {@code echo}, the prompt's entries
     * merged before the generated ones (the first prompt token's entry null), otherwise the generated captures
     * at the echo text's length.
     */
    private static Object choiceLogprobs(OpenAiBodies bodies, CompletionPlan plan, int choice,
                                         GenerationResult result) {
        if (plan.top() < 0) {
            return null;
        }
        boolean echoMerge = !plan.echoes().get(choice).isEmpty() && plan.echoesLogprobs()
                && !result.promptLogprobs().isEmpty() && plan.promptOf(choice) != null;
        if (echoMerge) {
            return bodies.echoCompletionLogprobs(result.promptLogprobs(), plan.promptOf(choice), result.logprobs());
        }
        return result.logprobs().isEmpty() ? null
                : bodies.completionLogprobs(result.logprobs(), plan.echoes().get(choice).length());
    }

    /**
     * The job indices of the best {@code n} candidates of every {@code bestOf} group (vLLM's parallel sampling
     * selection): highest cumulative log probability first, ties in submission order, prompt-major with the
     * chosen {@code n} re-indexed from 0.
     */
    static int[] bestOfSelection(List<GenerationResult> results, int n, int bestOf) {
        int[] identity = new int[results.size()];
        for (int i = 0; i < identity.length; i++) {
            identity[i] = i;
        }
        if (bestOf <= n) {
            return identity;
        }
        int[] picked = new int[results.size() / bestOf * n];
        int at = 0;
        for (int from = 0; from < results.size(); from += bestOf) {
            final int group = from;
            Integer[] order = new Integer[bestOf];
            for (int i = 0; i < bestOf; i++) {
                order[i] = i;
            }
            java.util.Arrays.sort(order, (a, b) -> Double.compare(cumulativeLogprob(results.get(group + b)),
                    cumulativeLogprob(results.get(group + a))));
            for (int i = 0; i < n; i++) {
                picked[at++] = from + order[i];
            }
        }
        return picked;
    }

    /** A candidate's cumulative log probability: the sum of its sampled tokens' log probabilities (a missing
     *  capture, an infinite or NaN value ranks worst). */
    private static double cumulativeLogprob(GenerationResult result) {
        double sum = 0;
        for (TokenScore[] step : result.logprobs()) {
            float lp = step[0].logprob();
            sum += Double.isFinite(lp) ? lp : Double.NEGATIVE_INFINITY;
        }
        return sum;
    }

    /** The results at the {@code picked} job indices. */
    static List<GenerationResult> selected(List<GenerationResult> results, int[] picked) {
        List<GenerationResult> out = new ArrayList<>(picked.length);
        for (int i : picked) {
            out.add(results.get(i));
        }
        return out;
    }

    /** The prefix-cache tokens of the chosen job indices. */
    static long cachedTokensOf(GenerationBatch batch, int[] picked) {
        long cached = 0;
        for (int i : picked) {
            cached += batch.cachedTokens(i);
        }
        return cached;
    }


    /** The request's response id: the {@code X-Request-Id} header, else a fresh id. */
    private String requestId() {
        String named = OpenAiRequests.requestId(http, null);
        return named != null ? named : String.valueOf(System.nanoTime());
    }

    /** Token ids as a JSON array of integers. */
    private static List<Integer> idsOf(int[] tokens) {
        List<Integer> out = new ArrayList<>(tokens.length);
        for (int token : tokens) {
            out.add(token);
        }
        return out;
    }

    /** The prompt tokens the prefix cache served for the {@code size} choices of the batch from {@code from}. */
    static long cachedTokens(GenerationBatch batch, int from, int size) {
        long cached = 0;
        for (int i = from; i < from + size; i++) {
            cached += batch.cachedTokens(i);
        }
        return cached;
    }

    /**
     * Streams a completions batch: each choice's echoed prompt at its first event ({@link #emitEcho}), then every
     * token, then its finish chunk with the {@code stop_reason}. The writer stops at {@code stop} — null, or the
     * recorded stream's explicit stop.
     */
    private void streamCompletions(OutputStream out, GenerationBatch batch, OpenAiBodies bodies, String id,
                                   long created, CompletionPlan plan, StreamWriter.Flags flags,
                                   BooleanSupplier stop) throws IOException {
        StreamWriter writer = new StreamWriter(out, batch, bodies, id, created, "text_completion",
                plan.promptTokens(), flags);
        int choices = plan.echoes().size();
        int[] offsets = new int[choices];
        boolean[] first = new boolean[choices];
        boolean[] echoed = new boolean[choices];
        java.util.Arrays.fill(first, true);
        writer.run(event -> {
            if (event instanceof GenerationBatch.Text t && !(t.text().isEmpty() && t.logprobs() == null)) {
                emitEcho(writer, bodies, plan, batch, t.choice(), offsets, first, echoed);
                writer.completionChunk(t.choice(), t.text(), t.logprobs(), offsets[t.choice()], null, t.tokenId(),
                        first[t.choice()]);
                offsets[t.choice()] += t.text().length();
                first[t.choice()] = false;
            } else if (event instanceof GenerationBatch.Done d) {
                emitEcho(writer, bodies, plan, batch, d.choice(), offsets, first, echoed);
                GenerationResult r = writer.finished(d.choice());
                if (r != null) {
                    writer.completionFinish(d.choice(), OpenAiBodies.finishReason(r.stopReason()),
                            OpenAiBodies.stopReason(r), first[d.choice()]);
                }
            }
        }, stop);
    }

    /**
     * Writes the echo chunk of one choice once its first event arrives — the prefill has then read the prompt
     * rows back, so an {@code echo} with {@code logprobs} merges the prompt's entries into the chunk's legacy
     * arrays (vLLM's first chunk) and the generated tokens' offsets continue after them.
     */
    private static void emitEcho(StreamWriter writer, OpenAiBodies bodies, CompletionPlan plan,
                                 GenerationBatch batch, int choice, int[] offsets, boolean[] first,
                                 boolean[] echoed) throws IOException {
        if (echoed[choice]) {
            return;
        }
        echoed[choice] = true;
        String echo = plan.echoes().get(choice);
        if (echo.isEmpty()) {
            return;
        }
        List<TokenScore[]> promptSteps = plan.echoesLogprobs() ? batch.promptLogprobs(choice) : null;
        if (promptSteps != null && !promptSteps.isEmpty() && batch.promptTokenIds(choice) != null) {
            int[] promptTokens = batch.promptTokenIds(choice);
            writer.echoChunk(choice, echo,
                    bodies.echoCompletionLogprobs(promptSteps, promptTokens, List.<TokenScore[]>of()), first[choice]);
            offsets[choice] = bodies.echoLogprobsOffset(promptTokens);
        } else {
            writer.completionChunk(choice, echo, (TokenScore[]) null, 0, null, null, first[choice]);
            offsets[choice] = echo.length();
        }
        first[choice] = false;
    }

    // ------------------------------------------------------------------ chat completions

    private Response chatOf(EngineService service, JsonView body) {
        boolean beam = OpenAiRequests.beamSearch(body);
        List<String> modalities = OpenAiRequests.outputModalities(body);
        ToolsLoop.Prepared server = ToolsLoop.prepare(service, body);
        if (!server.names().isEmpty()) {
            if (beam) {
                throw ServerJson.badRequest("use_beam_search answers a plain chat completion: it cannot run"
                        + " through server tool execution");
            }
            if (modalities.contains("audio")) {
                throw ServerJson.badRequest("server tool execution cannot answer with audio: send no tools or drop"
                        + " the audio modality");
            }
            return serverToolChatOf(service, server);
        }
        if (beam && modalities.contains("audio")) {
            throw ServerJson.badRequest("use_beam_search answers text completions: drop the audio modality or"
                    + " beam search");
        }
        if (modalities.contains("audio")) {
            return audioChatOf(service, body, modalities);
        }
        int n = OpenAiRequests.choices(body);
        ChatPlan plan = ChatPlan.of(service, body);
        if (beam) {
            return beamChatOf(service, body, plan);
        }
        boolean streamed = body.optBool("stream", false);
        if (streamed) {
            OpenAiRequests.requireStreamablePromptLogprobs(body);
        }
        List<GenerationBatch.Job> jobs = plan.jobs();
        List<ChatChoice> choices = new ArrayList<>();
        for (int i = 0; i < jobs.size(); i++) {
            choices.add(plan.newChoice());
        }
        OpenAiBodies bodies = plan.bodies(service);
        String id = "chatcmpl-" + requestId();
        long created = System.currentTimeMillis() / 1000;
        boolean includeUsage = body.optObj("stream_options").optBool("include_usage", false);
        String conversationId = conversationId();
        boolean resumable = streamed && conversationId != null && streams != null;
        GenerationBatch batch = GenerationBatch.start(service, jobs, http, id, !resumable);
        service.completions().track(id, batch);
        List<ChatChoice> streamChoices = choices;
        if (streamed) {
            StreamWriter.Flags flags = new StreamWriter.Flags(includeUsage,
                    OpenAiRequests.continuousUsageStats(body), plan.returnTokenIds());
            StreamSession session = session(conversationId);
            BooleanSupplier stop = session == null ? null : session::isCancelled;
            StreamingOutput stream = out -> streamChat(out, batch, bodies, id, created, streamChoices, plan,
                    flags, stop);
            return streamed(session, stream);
        }
        batch.awaitAll();
        drainInto(batch, choices);
        List<GenerationResult> results = resultsOf(batch);
        long cached = cachedTokens(batch, 0, batch.size());
        if (plan.bestOf() > n) {
            int[] picked = bestOfSelection(results, n, plan.bestOf());
            results = selected(results, picked);
            choices = selectedChoices(choices, picked);
            cached = cachedTokensOf(batch, picked);
        }
        Map<String, Object> response = chatBody(bodies, plan, id, created, choices, results, cached);
        putKvTransfer(response, batch, 0);
        return Response.ok(ServerJson.write(response)).build();
    }

    /**
     * The chat completion of a beam-search request ({@code use_beam_search}, vLLM's chat beam search): the search
     * runs the request's beams over the pool ({@link BeamSearchRunner}) — one final body once every beam is
     * ranked, never a stream. Each beam's output feeds through the plan's reasoning and tool routing exactly as a
     * streamed choice's does ({@link #driveBeam}), so the response renders the ranked beams as messages.
     */
    private Response beamChatOf(EngineService service, JsonView body, ChatPlan plan) {
        if (body.optBool("stream", false)) {
            throw ServerJson.badRequest("stream must be false or omitted: beam search answers one final body once"
                    + " every beam is ranked");
        }
        if (plan.media() != null) {
            throw ServerJson.badRequest("media content parts: beam search resubmits token sequences and serves"
                    + " text-only prompts");
        }
        if (plan.bestOf() > plan.n()) {
            throw ServerJson.badRequest("best_of " + plan.bestOf() + " beside use_beam_search: the search ranks"
                    + " its own beams; drop best_of or sample plainly");
        }
        if (plan.constraint() != null) {
            throw ServerJson.badRequest("response_format and grammars (constraint): beam search samples the"
                    + " unconstrained top rows of every step, so the constraint cannot shape the beams;"
                    + " drop response_format or sample plainly");
        }
        BeamSearchEngine.requireSupported(plan.base());
        OpenAiBodies bodies = plan.bodies(service);
        String id = "chatcmpl-" + requestId();
        long created = System.currentTimeMillis() / 1000;
        BeamSearchRunner.Run run = BeamSearchRunner.run(service, plan.base(), plan.tokens(), http, id);
        List<ChatChoice> choices = new ArrayList<>();
        List<GenerationResult> results = new ArrayList<>();
        for (BeamSearchSequence sequence : run.output().sequences()) {
            BeamDriven driven = driveBeam(service, plan, sequence);
            choices.add(driven.choice());
            results.add(driven.result());
        }
        Map<String, Object> response = chatBody(bodies, plan, id, created, choices, results, run.cachedRows());
        return Response.ok(ServerJson.write(response)).build();
    }

    /** One beam's response parts: the routed choice and its result. */
    private record BeamDriven(ChatChoice choice, GenerationResult result) { }

    /**
     * Feeds one beam's output through the plan's reasoning and tool routing: every output token's text plays into
     * a fresh {@link ChatChoice} through a {@link StopCriteria} session (the streamed path's rendering — decoder
     * flushes, special-token and spacing settings, the reasoning parser and the tool channel), with the beam's own
     * capture row riding each token's text. The result is the session's over the beam's end reason, its stop token
     * id restored (the session never saw the end token, whose text the beam drops).
     */
    private static BeamDriven driveBeam(EngineService service, ChatPlan plan, BeamSearchSequence sequence) {
        ChatChoice choice = new ChatChoice(plan.routing().newOutput(), plan.includeReasoning());
        int[] textTokens = sequence.outputTokens(false);
        List<TokenScore[]> rows = sequence.logprobs();
        AtomicReference<TokenScore[]> row = new AtomicReference<>(null);
        int[] predicted = {0};
        StopCriteria.Session session = StopCriteria.of(plan.base(), service.tokenizer(), plan.reasoning())
                .start(new ReasoningFilter.Sink() {
                    @Override
                    public void stepped(int tokenId) {
                        predicted[0]++;
                    }

                    @Override
                    public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                        choice.accept(new GenerationBatch.Text(0, channel, text, row.get(),
                                tokenId, predicted[0]));
                    }

                    @Override
                    public void toolCallStarted(String recipient) {
                        choice.accept(new GenerationBatch.ToolRecipient(0, recipient));
                    }

                    @Override
                    public void contentFallback() {
                        choice.accept(new GenerationBatch.ContentFallback(0));
                    }
                }, plan.rendered());
        for (int i = 0; i < textTokens.length; i++) {
            row.set(i < rows.size() ? rows.get(i) : null);
            session.offer(textTokens[i]);
        }
        GenerationResult result = session.finish(sequence.stopReason(), GenerationResult.Timings.total(
                        plan.promptRows(), textTokens.length, 0))
                .withStopTokenId(sequence.stopTokenId());
        return new BeamDriven(choice, result);
    }

    /**
     * The chat completion of a request that listed server tools ({@link ToolsLoop#prepare}): the loop runs its
     * rounds, and the response is the final round's body with the executed calls of the rounds before it on the
     * choice, as {@code server_tool_calls}, and the usage summed over every round. One choice, no stream.
     */
    @SuppressWarnings("unchecked")
    private Response serverToolChatOf(EngineService service, ToolsLoop.Prepared server) {
        if (OpenAiRequests.choices(JsonView.of(server.body(), "body")) != 1) {
            throw ServerJson.badRequest("server tool execution answers one choice: set n to 1 or drop it");
        }
        if (Boolean.parseBoolean(String.valueOf(server.body().get("stream")))) {
            throw ServerJson.badRequest("server tool execution does not stream: send stream false or drop it");
        }
        ToolsLoop.Outcome outcome = ToolsLoop.run(service, server.body(), server.names(), http);
        ChatPlan plan = outcome.plan();
        OpenAiBodies bodies = plan.bodies(service);
        String id = "chatcmpl-" + requestId();
        long created = System.currentTimeMillis() / 1000;
        Map<String, Object> response = chatBody(bodies, plan, id, created, List.of(outcome.choice()),
                List.of(outcome.result()), outcome.cached());
        response.put("usage", OpenAiBodies.usage(outcome.promptRows(), outcome.cached(), outcome.rounds()));
        if (!outcome.executed().isEmpty()) {
            ((Map<String, Object>) ((List<?>) response.get("choices")).get(0))
                    .put("server_tool_calls", executedBody(outcome.executed()));
        }
        return Response.ok(ServerJson.write(response)).build();
    }

    /** The {@code server_tool_calls} entries: id, name, arguments and the result object of each execution. */
    private static List<Object> executedBody(List<ToolsLoop.Executed> executed) {
        List<Object> out = new ArrayList<>();
        for (ToolsLoop.Executed call : executed) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("id", call.id());
            entry.put("name", call.name());
            entry.put("arguments", call.arguments());
            entry.put("result", call.result());
            out.add(entry);
        }
        return out;
    }

    // ------------------------------------------------------------------ omni audio output

    /** The audio containers this server encodes into a chat answer. */
    private static final String AUDIO_WAV = "wav";
    private static final String AUDIO_PCM16 = "pcm16";

    /**
     * The chat completion that answers with speech ({@code modalities} or {@code output_modalities} naming
     * {@code audio}, OpenAI's audio output over an omni model). The conversation renders through the model's
     * conversation template with the request's {@code language} and {@code voice} bound; {@code ["audio"]} runs a
     * fresh AUDIO turn, which speaks the answer and stops at the end-of-audio frame, and
     * {@code ["text", "audio"]} a MIXED turn that interleaves text and speech blocks until {@code <|im_end|>} or
     * the step cap (mtmd-helper-gen.cpp:1392-1450). One choice; the answer's text carries
     * {@code message.content} (null without the {@code text} modality) and the waveform the OpenAI audio object
     * under {@code message.audio} ({@code id}, base64 {@code data}, {@code expires_at}, {@code transcript},
     * {@code format}). The stream sends the role chunk, the turn's text pieces as they sample, the whole waveform
     * as one {@code delta.audio} chunk, then stops with the usage when {@code stream_options.include_usage} asks
     * for it.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for more than one choice, a request the speech options
     *         refuse, an unsupported {@code audio.format}, or a model without the omni pipeline, its generator file
     *         or its conversation template (the message names the missing piece); a 500 when the generation fails
     */
    private Response audioChatOf(EngineService service, JsonView body, List<String> modalities) {
        if (OpenAiRequests.choices(body) != 1) {
            throw ServerJson.badRequest("audio output answers one choice: set n to 1 or drop it");
        }
        ChatRequest chat = OpenAiRequests.chat(body, service.options());
        OpenAiRequests.AudioOutput audio = OpenAiRequests.audioOutput(body);
        SpeechOptions options;
        try {
            options = audioOptions(body, audio);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        boolean withText = modalities.contains("text");
        Lfm3aSpeechGenerator.Mode mode = withText ? Lfm3aSpeechGenerator.Mode.MIXED : Lfm3aSpeechGenerator.Mode.AUDIO;
        // a MIXED turn continues the conversation, whose prompt the pipeline extends with mixed_start; an
        // audio-only turn is fresh and its prompt ends in audio_start
        boolean continuing = withText;
        String model = service.loras().model(body);
        String id = "chatcmpl-" + requestId();
        String audioId = "audio-" + requestId();
        long created = System.currentTimeMillis() / 1000;
        if (body.optBool("stream", false)) {
            boolean includeUsage = body.optObj("stream_options").optBool("include_usage", false);
            StreamingOutput stream = out -> audioStream(out, service, chat, mode, continuing, options, model, id,
                    audioId, created, withText, audio.format(), includeUsage);
            return Response.ok(stream).type("text/event-stream").build();
        }
        Lfm3aSpeechGenerator.OmniTurn turn = speak(service, chat, mode, continuing, options, null);
        Map<String, Object> response = audioBody(model, id, audioId, created, turn, withText, audio.format(),
                audioUsage(turn));
        return Response.ok(ServerJson.write(response)).build();
    }

    /**
     * The omni turn over the service; the service's refusals map to request errors that name the missing piece.
     */
    private static Lfm3aSpeechGenerator.OmniTurn speak(EngineService service, ChatRequest chat,
                                                     Lfm3aSpeechGenerator.Mode mode, boolean continuing,
                                                     SpeechOptions options, Lfm3aSpeechGenerator.OmniSink sink) {
        try {
            return service.speech().omniTurn(chat, mode, continuing, options, sink);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(400, e.getMessage(), "invalid_request_error");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "speech generation failed: " + e.getMessage(), "server_error");
        }
    }

    /**
     * The speech options of an audio-output chat request: the audio object's {@code voice} and {@code format}, the
     * request's {@code seed} and whole sampling configuration ({@link SamplingParams}), and the step cap of
     * {@code max_tokens} / {@code max_completion_tokens} (llama.cpp's TTS reads its {@code n_predict}).
     */
    private static SpeechOptions audioOptions(JsonView body, OpenAiRequests.AudioOutput audio) {
        if (Boolean.TRUE.equals(body.raw("watermarking"))) {
            // the speech sampler draws codec frames, not text; a text watermark has nothing to mark there
            throw ServerJson.badRequest("watermarking is not supported for audio output");
        }
        SpeechOptions options = SpeechOptions.defaults()
                .withLanguage(audio.language())
                .withVoice(audio.voice())
                .withSeed(body.optI64("seed", SpeechOptions.DEFAULT_SEED))
                .withSampler(SamplingParams.of(body));
        int steps = body.has("max_completion_tokens") ? body.optI32("max_completion_tokens", -1)
                : body.optI32("max_tokens", -1);
        if (steps > 0) {
            options = options.withMaxFrames(steps);
        }
        return options;
    }

    /** The non-streamed audio answer: the turn's text and its waveform under the OpenAI audio object. */
    private static Map<String, Object> audioBody(String model, String id, String audioId, long created,
                                                 Lfm3aSpeechGenerator.OmniTurn turn, boolean withText, String format,
                                                 Map<String, Object> usage) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", withText ? turn.text() : null);
        message.put("audio", audioObject(audioId, created, turn, format));
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("logprobs", null);
        choice.put("finish_reason", "stop");
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", id);
        response.put("object", "chat.completion");
        response.put("created", created);
        response.put("model", model);
        response.put("choices", List.of(choice));
        response.put("usage", usage);
        return response;
    }

    /** The OpenAI audio object: base64 data in the requested container, an hour of expiry and the turn's text. */
    private static Map<String, Object> audioObject(String audioId, long created,
                                                   Lfm3aSpeechGenerator.OmniTurn turn, String format) {
        AudioClip clip = turn.audio();
        Map<String, Object> audio = new LinkedHashMap<>();
        audio.put("id", audioId);
        audio.put("data", Base64.getEncoder().encodeToString(audioBytes(clip, format)));
        audio.put("expires_at", created + 3600);
        audio.put("transcript", turn.text());
        audio.put("format", format);
        return audio;
    }

    /** The waveform bytes in {@code format}: a RIFF WAVE file, or raw little-endian 16-bit PCM. */
    private static byte[] audioBytes(AudioClip clip, String format) {
        if (AUDIO_WAV.equals(format)) {
            return WavWriter.bytes(clip.samples(), clip.sampleRate());
        }
        return WavWriter.pcmChunk(clip.samples(), 0, clip.length());
    }

    /** The usage of a spoken answer: the prompt rows the turn prefilled and the steps it sampled. */
    private static Map<String, Object> audioUsage(Lfm3aSpeechGenerator.OmniTurn turn) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", turn.promptTokens());
        usage.put("completion_tokens", turn.steps());
        usage.put("total_tokens", turn.promptTokens() + turn.steps());
        return usage;
    }

    /**
     * The audio stream: the role chunk, the turn's text pieces as they sample, the whole waveform as one
     * {@code delta.audio} chunk, then the stop chunk with the usage. The generation runs on the engine thread while
     * this method waits, so the pieces reach the response in generation order.
     */
    private void audioStream(OutputStream out, EngineService service, ChatRequest chat,
                             Lfm3aSpeechGenerator.Mode mode, boolean continuing, SpeechOptions options, String model,
                             String id, String audioId, long created, boolean withText, String format,
                             boolean includeUsage) throws IOException {
        Map<String, Object> role = new LinkedHashMap<>();
        role.put("role", "assistant");
        if (withText) {
            role.put("content", "");
        }
        chunk(out, id, created, model, role, null, null);
        Lfm3aSpeechGenerator.OmniSink sink = new Lfm3aSpeechGenerator.OmniSink() {
            @Override
            public void text(String piece) {
                if (!withText) {
                    return;
                }
                try {
                    chunk(out, id, created, model, Map.of("content", piece), null, null);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }

            @Override
            public void audio(AudioClip chunk) {
                // the generator collects the turn's waveform; it goes out as one delta below
            }
        };
        Lfm3aSpeechGenerator.OmniTurn turn;
        try {
            turn = speak(service, chat, mode, continuing, options, sink);
        } catch (RuntimeException e) {
            if (e instanceof UncheckedIOException io) {
                throw io.getCause();
            }
            throw e;
        }
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("audio", audioObject(audioId, created, turn, format));
        chunk(out, id, created, model, delta, null, null);
        chunk(out, id, created, model, new LinkedHashMap<>(), "stop", includeUsage ? audioUsage(turn) : null);
        ServerJson.data(out, "[DONE]");
    }

    /** One chat stream chunk over a free-form delta: the choice's index, delta and finish reason. */
    private static void chunk(OutputStream out, String id, long created, String model, Map<String, Object> delta,
                              String finish, Map<String, Object> usage) throws IOException {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", finish);
        Map<String, Object> top = new LinkedHashMap<>();
        top.put("id", id);
        top.put("object", "chat.completion.chunk");
        top.put("created", created);
        top.put("model", model);
        top.put("choices", List.of(choice));
        if (usage != null) {
            top.put("usage", usage);
        }
        ServerJson.event(out, null, top);
    }

    /** The chat choices at the {@code picked} job indices. */
    static List<ChatChoice> selectedChoices(List<ChatChoice> choices, int[] picked) {
        List<ChatChoice> out = new ArrayList<>(picked.length);
        for (int i : picked) {
            out.add(choices.get(i));
        }
        return out;
    }

    /**
     * The non-streamed chat body: one choice per entry of {@code choices} over its finished result, the prompt
     * logprobs and prompt token ids the plan asks for, and usage over the prompt ({@code cachedTokens} of it from
     * the prefix cache) and the generated tokens.
     */
    static Map<String, Object> chatBody(OpenAiBodies bodies, ChatPlan plan, String id, long created,
                                        List<ChatChoice> choices, List<GenerationResult> results,
                                        long cachedTokens) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (int i = 0; i < choices.size(); i++) {
            out.add(chatChoiceBody(i, choices.get(i), results.get(i), bodies, plan));
        }
        Map<String, Object> response = bodies.head(id, "chat.completion", created);
        response.put("choices", out);
        if (plan.base().wantsPromptLogprobs()) {
            response.put("prompt_logprobs", bodies.promptLogprobs(results.get(0).promptLogprobs()));
        }
        if (plan.returnTokenIds()) {
            response.put("prompt_token_ids", plan.media() == null ? idsOf(plan.tokens()) : null);
        }
        response.put("usage", OpenAiBodies.usage(plan.promptRows(), cachedTokens, results));
        return response;
    }

    /** Routes every queued event (the generations finished) into its choice. */
    static void drainInto(GenerationBatch batch, List<ChatChoice> choices) {
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

    // ------------------------------------------------------------------ batched chat completions

    /**
     * The batched chat body: {@code messages} lists conversations, every other field is shared, and the response
     * holds one choice per conversation over one generation batch. Streaming, tools and {@code prompt_logprobs}
     * (the response has no per-conversation place for them) are refused; {@code n} must be 1, as vLLM's batch
     * endpoint requires.
     */
    private Response batchChatOf(EngineService service, JsonView body) {
        if (body.optBool("stream", false)) {
            throw ServerJson.badRequest("stream must be false or omitted: batch chat completions do not stream");
        }
        int n = OpenAiRequests.choices(body);
        if (n != 1) {
            throw ServerJson.badRequest("Batch chat completions do not support `n > 1`. Please set `n` to 1.");
        }
        if (OpenAiRequests.beamSearch(body)) {
            throw ServerJson.badRequest("use_beam_search is not supported by batch chat completions");
        }
        requireNoTools(body);
        if (body.raw("prompt_logprobs") != null) {
            throw ServerJson.badRequest("prompt_logprobs is not supported by batch chat completions");
        }
        List<?> conversations = conversations(body);
        OpenAiRequests.requireGenerationBudget(conversations.size(), 1);
        List<ChatPlan> plans = new ArrayList<>();
        for (int i = 0; i < conversations.size(); i++) {
            try {
                plans.add(ChatPlan.of(service, JsonView.of(singleConversation(body, conversations.get(i)),
                        "body")));
            } catch (WebApplicationException e) {
                throw ServerJson.badRequest("messages[" + i + "]: " + errorEnvelope(e).get("message"));
            }
        }
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        List<ChatChoice> choices = new ArrayList<>();
        for (ChatPlan plan : plans) {
            jobs.addAll(plan.jobs());
            for (int i = 0; i < plan.jobs().size(); i++) {
                choices.add(plan.newChoice());
            }
        }
        OpenAiBodies bodies = plans.get(0).bodies(service);
        String id = "chatcmpl-" + requestId();
        long created = System.currentTimeMillis() / 1000;
        GenerationBatch batch = GenerationBatch.start(service, jobs, http, id);
        batch.awaitAll();
        drainInto(batch, choices);
        List<GenerationResult> candidates = resultsOf(batch);
        List<Map<String, Object>> out = new ArrayList<>();
        List<GenerationResult> results = new ArrayList<>();
        List<ChatChoice> picked = new ArrayList<>();
        long promptTokens = 0;
        long cached = 0;
        int slot = 0;
        for (int i = 0; i < plans.size(); i++) {
            ChatPlan plan = plans.get(i);
            int candidatesOf = plan.bestOf();
            int[] order = bestOfSelection(candidates.subList(slot, slot + candidatesOf), 1, candidatesOf);
            int job = slot + order[0];
            slot += candidatesOf;
            picked.add(choices.get(job));
            results.add(candidates.get(job));
            cached += batch.cachedTokens(job);
            promptTokens += plan.promptRows();
            out.add(chatChoiceBody(i, picked.get(i), results.get(i), bodies, plan));
        }
        Map<String, Object> response = bodies.head(id, "chat.completion", created);
        response.put("choices", out);
        response.put("usage", OpenAiBodies.usage(promptTokens, cached, results));
        return Response.ok(ServerJson.write(response)).build();
    }

    /** The batch endpoint answers plain text: tools and a non-{@code none} tool choice are refused. */
    private static void requireNoTools(JsonView body) {
        if (body.raw("tools") != null) {
            throw ServerJson.badRequest("tools is not supported by batch chat completions");
        }
        Object choice = body.raw("tool_choice");
        if (choice != null && !"none".equals(choice)) {
            throw ServerJson.badRequest("tool_choice must be none in batch chat completions, got " + choice);
        }
    }

    /** The conversations: {@code messages} is a non-empty array of non-empty message arrays. */
    private static List<?> conversations(JsonView body) {
        Object raw = body.raw("messages");
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("messages is required and must be a non-empty array of conversations");
        }
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof List<?> conversation) || conversation.isEmpty()) {
                throw ServerJson.badRequest("messages[" + i + "] must be a non-empty array of messages");
            }
        }
        return list;
    }

    /** The single-conversation chat body of one conversation: every other field kept, {@code messages} replaced. */
    private static Map<String, Object> singleConversation(JsonView body, Object conversation) {
        Map<String, Object> single = new LinkedHashMap<>();
        for (String key : body.keys()) {
            if (!key.equals("messages")) {
                single.put(key, body.raw(key));
            }
        }
        single.put("messages", conversation);
        return single;
    }

    /** The error envelope {@code {message, type}} of a refused call. */
    static Map<String, Object> errorEnvelope(WebApplicationException e) {
        Map<String, Object> out = new LinkedHashMap<>();
        try {
            JsonView error = JsonView.of(Json.parse(String.valueOf(e.getResponse().getEntity()),
                    Json.Limits.UNTRUSTED), "error").obj("error");
            out.put("message", error.str("message"));
            out.put("type", error.str("type"));
        } catch (RuntimeException parse) {
            out.put("message", String.valueOf(e.getMessage()));
            out.put("type", "invalid_request_error");
        }
        return out;
    }

    /** One chat choice: the plan's echo before the content, its tool calls, logprobs and {@code token_ids}. */
    private static Map<String, Object> chatChoiceBody(int index, ChatChoice choice, GenerationResult result,
                                                      OpenAiBodies bodies, ChatPlan plan) {
        ToolParse parse = choice.finish();
        String content = plan.echo() + parse.content();
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", parse.calls().isEmpty() || !content.isEmpty() ? content : null);
        String reasoning = choice.messageReasoning();
        if (!reasoning.isEmpty()) {
            message.put("reasoning_content", reasoning);
            message.put("reasoning", reasoning);
        }
        if (!parse.calls().isEmpty()) {
            message.put("tool_calls", ChatChoice.toolCallsBody(parse.calls()));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("index", index);
        body.put("message", message);
        // a best_of candidate captures logprobs for its ranking; they answer only when the request asked
        body.put("logprobs", plan.top() < 0 || choice.logprobs().isEmpty() ? null
                : bodies.chatLogprobs(choice.logprobs()));
        body.put("finish_reason", chatFinish(result.stopReason(), !parse.calls().isEmpty(), plan.named()));
        body.put("stop_reason", OpenAiBodies.stopReason(result));
        if (plan.returnTokenIds()) {
            body.put("token_ids", idsOf(result.tokens()));
        }
        return body;
    }

    /** {@code tool_calls} when calls were read (a named choice answers {@code stop}, as OpenAI does). */
    static String chatFinish(StopReason reason, boolean calls, boolean named) {
        String base = OpenAiBodies.finishReason(reason);
        return calls && !named && base.equals("stop") ? "tool_calls" : base;
    }

    private void streamChat(OutputStream out, GenerationBatch batch, OpenAiBodies bodies, String id, long created,
                            List<ChatChoice> choices, ChatPlan plan, StreamWriter.Flags flags,
                            BooleanSupplier stop) throws IOException {
        StreamWriter writer = new StreamWriter(out, batch, bodies, id, created, "chat.completion.chunk",
                plan.promptRows(), flags);
        for (int i = 0; i < choices.size(); i++) {
            Map<String, Object> role = new LinkedHashMap<>();
            role.put("role", "assistant");
            role.put("content", "");
            writer.chatChunk(i, role, null, null, null, i == 0 && plan.media() == null ? plan.tokens() : null);
            if (!plan.echo().isEmpty()) {
                // vLLM's chat echo: the last message's content in the first delta
                writer.chatChunk(i, Map.of("content", plan.echo()), null, null);
            }
        }
        writer.run(event -> {
            ChatChoice choice = choices.get(event.choice());
            if (event instanceof GenerationBatch.Done d) {
                ToolParse parse = choice.finish();
                writeDeltas(writer, d.choice(), choice);
                GenerationResult r = writer.finished(d.choice());
                if (r != null) {
                    writer.chatChunk(d.choice(), Map.of(), null,
                            chatFinish(r.stopReason(), !parse.calls().isEmpty(), plan.named()), null, null,
                            OpenAiBodies.stopReason(r));
                }
                return;
            }
            choice.accept(event);
            writeDeltas(writer, event.choice(), choice);
        }, stop);
    }

    // ------------------------------------------------------------------ resumable streams

    /** The request's conversation id ({@code X-Conversation-Id}, llama.cpp's resumable stream key), or null. */
    private String conversationId() {
        if (http == null) {
            return null;
        }
        String named = http.headers().get("X-Conversation-Id");
        return named == null || named.isBlank() ? null : named.trim();
    }

    /**
     * The session a streamed request records into, replacing any previous one on the same conversation id.
     * Null without the header, or when the resource was wired without a session registry (manual wiring), in
     * which case the stream is not resumable.
     */
    private StreamSession session(String conversationId) {
        return conversationId == null || streams == null ? null : streams.createOrReplace(conversationId);
    }

    /**
     * The streamed response: without a session it streams as it does today; with one, {@code stream} writes
     * through the session's mirror ({@link StreamSession#mirror}), and the session always finishes when the
     * producer leaves, so its readers never wait on a dead stream.
     */
    private static Response streamed(StreamSession session, StreamingOutput stream) {
        if (session == null) {
            return Response.ok(stream).type("text/event-stream").build();
        }
        StreamingOutput producing = out -> {
            try {
                stream.write(session.mirror(out));
            } finally {
                session.finish();
            }
        };
        return Response.ok(producing).type("text/event-stream").build();
    }

    private static void writeDeltas(StreamWriter writer, int index, ChatChoice choice) throws IOException {
        for (ChatChoice.Delta d : choice.drain()) {
            writer.chatChunk(index, d.delta(), d.logprobs(), null, d.tokenId(), null);
        }
    }
}
