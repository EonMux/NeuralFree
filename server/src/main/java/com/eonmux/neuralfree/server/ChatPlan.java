package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.OutputGates;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserFormat;
import com.eonmux.neuralfree.gguf.chat.response.BeamGrammar;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * A chat-completions body made ready for the engine: the served model name, the sampling request, the output
 * constraint (gated behind the reasoning), the rendered prompt and its tokens, and the tool routing. The OpenAI chat
 * route and the Responses API (whose body converts to a chat body) share it. Immutable.
 *
 * <p>A request whose messages carry media content parts — images and audio ({@code image_url},
 * {@code input_audio}, {@code audio_url}) encoded by the projector, or {@code image_embeds} and
 * {@code audio_embeds} (vLLM's precomputed embeddings) — takes the media path: the rendered prompt holds one
 * {@link ChatMessages#MEDIA_MARKER} per image or audio part and one {@link ChatMessages#PROMPT_EMBEDS_MARKER} per
 * embedding part, and the prompt becomes a {@link MediaPrompt} of the encoded text segments with the media's
 * marker text and rows spliced between them ({@link ChatMedia#prompt}).</p>
 *
 * @param model            the served model name the response carries
 * @param top              the {@code top_logprobs} count, or -1 without logprobs
 * @param base             the sampling request of choice 0
 * @param constraint       the output constraint, or null
 * @param chat             the chat request
 * @param rendered         the rendered prompt, {@code truncate_prompt_tokens} applied
 * @param routing          the tool routing of the rendered prompt
 * @param tokens           the prompt tokens, {@code truncate_prompt_tokens} applied
 * @param media            the prompt with its media and embedding rows spliced in, or null on the token path
 * @param placements       the row spans and keys of the items {@code media} spliced in, in prompt order
 * @param named            whether {@code tool_choice} names one function
 * @param reasoning        the request's reasoning filter ({@code reasoning_format}), or the configured one
 * @param includeReasoning whether {@code reasoning_content} reaches the response ({@code include_reasoning})
 * @param echo             the text {@code echo} prepends to every answer, empty when it is off
 * @param n                the choices per prompt
 * @param bestOf           the candidates generated per prompt ({@code best_of}, at least {@code n})
 * @param returnTokenIds   whether the response carries {@code prompt_token_ids} and {@code token_ids}
 * @param tokensAsIds      whether logprob tokens render as {@code token_id:{id}}
 *                         ({@code return_tokens_as_token_ids})
 */
record ChatPlan(String model, int top, GenerationRequest base, ConstraintSpec constraint, ChatRequest chat,
                String rendered, ToolRouting routing, int[] tokens, MediaPrompt media,
                List<ChatMedia.Placement> placements, boolean named,
                ReasoningFilter reasoning, boolean includeReasoning, String echo, int n, int bestOf,
                boolean returnTokenIds, boolean tokensAsIds, int checkpointRow,
                List<int[]> tokenOffsets) {

    /** Plans {@code body}, a chat-completions body. */
    static ChatPlan of(EngineService service, JsonView body) {
        String model = service.loras().model(body);
        int n = OpenAiRequests.choices(body);
        int top = OpenAiRequests.topLogprobs(body, true);
        // vLLM's chat echo fills an absent prompt_logprobs with top_logprobs, whose default is 0
        int echoTop = body.optBool("echo", false) ? Math.max(0, top) : -1;
        GenerationRequest base = service.loras().apply(body, model,
                OpenAiRequests.sampling(body, top, service.vocabSize(), echoTop, service.options().logprobsMode(),
                        service.logitsProcessors(), service.options().watermark()));
        OpenAiRequests.requireServedPriority(base, service.schedulingPolicy());
        int bestOf = OpenAiRequests.bestOf(body, n, body.optBool("stream", false));
        base = OpenAiRequests.scoredCandidates(base, n, bestOf);
        base = OpenAiRequests.withBeamSearch(base, body, n);
        OpenAiRequests.requireSingleTransfer(base, 1, n, bestOf);
        ConstraintSpec constraint = OpenAiRequests.constraint(body);
        // the autoparser's template markers shape a structured-output request and may add literal stops
        AutoParserFormat autoparser = service.autoparser();
        base = withTemplateStops(base, autoparser);
        constraint = autoparserResponseFormat(constraint, autoparser);
        ChatRequest chat = OpenAiRequests.chat(body, service.options());
        Supplier<LoraSlotKey> mediaLoras = mediaLoras(service, base.lora());
        List<ChatMedia.Item> items = mediaItems(service, chat, mediaLoras);
        String fullRendered = service.formatChat(chat, !items.isEmpty());
        String rendered = fullRendered;
        boolean embedded = rendered.contains(ChatMessages.PROMPT_EMBEDS_MARKER) || !items.isEmpty();
        int[] encoded = service.encode(rendered, false, true);
        int[] tokens = OpenAiRequests.truncate(body, encoded);
        if (embedded && tokens != encoded) {
            throw ServerJson.badRequest("truncate_prompt_tokens cannot be combined with media or prompt embedding"
                    + " parts");
        }
        if (tokens != encoded) {
            // a truncated prompt prefills only its kept tokens; the rendered text follows them
            rendered = service.decode(tokens);
        }
        boolean streamed = body.optBool("stream", false);
        ReasoningFilter reasoning = ReasoningFormats.fromBody(body, ReasoningFormats.forChat(service.reasoning(), chat),
                streamed);
        boolean includeReasoning = body.optBool("include_reasoning", true);
        // vLLM refuses an explicitly false parallel_tool_calls alone; an absent one serves the template's default
        // (llama.cpp server-common.cpp:1447), which still keeps the parsers to one call
        ParallelToolCalls parallelToolCalls = ParallelToolCalls.of(OpenAiRequests.parallelToolCalls(body),
                service.defaultParallelToolCalls(chat));
        ToolRouting routing = service.toolRouting(chat, rendered, parallelToolCalls, reasoning);
        // another tool format's forcing grammar holds the output itself
        BeamGrammar beam = routing.constraint() != null ? null
                : beamGrammar(service, chat, rendered, parallelToolCalls, constraint, reasoning, includeReasoning);
        boolean fromStart = beam != null && beam.fromMessageStart();
        if (beam != null) {
            constraint = beam.constraint();
        }
        if (constraint != null && includeReasoning && !fromStart) {
            // the constraint holds the answer, not the reasoning before it; a hidden reasoning
            // (include_reasoning false) leaves vLLM's reasoning_ended true, so it constrains at once
            constraint = constraint.withGate(OutputGates.forChat(reasoning, rendered, service.chatTemplate()));
        }
        if (routing.constraint() != null) {
            if (constraint != null) {
                throw ServerJson.badRequest("tool_choice " + chat.toolChoice() + " constrains the output; it"
                        + " cannot be combined with another output constraint");
            }
            constraint = routing.constraint();
        }
        boolean named = chat.toolChoice() != null && chat.toolChoice().kind() == ChatRequest.ToolChoice.Kind.FUNCTION;
        ChatMedia.Rendered renderedMedia = embedded
                ? renderedMedia(service, chat, rendered, items, mediaLoras) : null;
        MediaPrompt media = renderedMedia == null ? null : renderedMedia.prompt();
        List<ChatMedia.Placement> placements = renderedMedia == null ? List.of() : renderedMedia.placements();
        String echo = body.optBool("echo", false) ? echoPrefix(chat) : "";
        int checkpointRow = embedded || tokens != encoded ? -1
                : lastMessageStart(service, chat, fullRendered, encoded);
        List<int[]> offsets = null;
        if (body.optBool("return_token_offsets", false) && !embedded) {
            offsets = ScaleOutOffsets.slice(ScaleOutOffsets.of(service, fullRendered, encoded), tokens, body);
        }
        return new ChatPlan(model, top, base, constraint, chat, rendered, routing, tokens, media, placements, named,
                reasoning, includeReasoning, echo, n, bestOf, body.optBool("return_token_ids", false),
                body.optBool("return_tokens_as_token_ids", false), checkpointRow, offsets);
    }

    /**
     * {@code request} with the autoparser template's literal stop strings appended (Laguna's
     * {@code </assistant>}), or {@code request} when the template adds none.
     */
    static GenerationRequest withTemplateStops(GenerationRequest request, AutoParserFormat autoparser) {
        if (autoparser == null || autoparser.additionalStops().isEmpty()) {
            return request;
        }
        GenerationRequest.Builder b = request.toBuilder();
        for (String stop : autoparser.additionalStops()) {
            if (!request.stopStrings().contains(stop)) {
                b.stopString(stop);
            }
        }
        return b.build();
    }

    /**
     * The structured-output constraint in the autoparsed template's own markers: a JSON schema or generic JSON
     * object becomes the template's fenced-or-bare value grammar (llama.cpp's autoparser
     * {@code has_response_format}); any other constraint kind, or no autoparser, is unchanged.
     */
    static ConstraintSpec autoparserResponseFormat(ConstraintSpec constraint, AutoParserFormat autoparser) {
        if (constraint == null || autoparser == null) {
            return constraint;
        }
        if (constraint.kind() == ConstraintSpec.Kind.JSON_SCHEMA) {
            ConstraintSpec wrapped = autoparser.responseFormat(constraint.text());
            return wrapped == null ? constraint : wrapped;
        }
        if (constraint.kind() == ConstraintSpec.Kind.JSON_OBJECT) {
            ConstraintSpec wrapped = autoparser.responseFormat("{\"type\": \"object\"}");
            return wrapped == null ? constraint : wrapped;
        }
        return constraint;
    }

    /**
     * The marker text a boundary probe renders the last message's content as: ordinary text (never a special
     * token), unique in any prompt, and absent from real conversations.
     */
    private static final String BOUNDARY_MARKER = "NFMSGBOUNDARY7X3Q";

    /**
     * The prompt row where the rendered prompt's last message starts (llama.cpp's message spans: a context
     * checkpoint there lets the next turn, or a branch of the conversation at the last message, restore the
     * shared prefix), or -1 when it cannot be found. The probe renders the same conversation with the last
     * message's content replaced by {@link #BOUNDARY_MARKER} and reads the boundary off the two token streams'
     * longest common prefix: the rendered text before the last message is identical, so the prefix ends where
     * the content starts, token for token the prompt's own rows. The probe's shared opening is not tokenized
     * again ({@link com.eonmux.neuralfree.gguf.GGUFTokenizer#commonPrefixLength}).
     */
    private static int lastMessageStart(EngineService service, ChatRequest chat, String rendered, int[] encoded) {
        List<Map<String, Object>> messages = chat.messages();
        if (messages.isEmpty()) {
            return -1;
        }
        List<Map<String, Object>> marked = new ArrayList<>(messages.size());
        for (int i = 0; i < messages.size(); i++) {
            Map<String, Object> copy = new LinkedHashMap<>(messages.get(i));
            if (i == messages.size() - 1) {
                copy.put(ChatMessages.CONTENT, BOUNDARY_MARKER);
            }
            marked.add(copy);
        }
        int at;
        try {
            String probe = service.formatChat(chat.toBuilder().messages(marked).build(), false);
            at = service.tokenizer().commonPrefixLength(rendered, encoded, probe);
        } catch (RuntimeException probeFailed) {
            return -1;
        }
        return at > 0 && at < encoded.length ? at : -1;
    }

    /**
     * The {@code beam} grammar of a request ({@link EngineService#beamGrammar}), a guarantee it cannot express refused
     * as a bad request naming the parameter; null without one.
     */
    static BeamGrammar beamGrammar(EngineService service, ChatRequest chat, String rendered,
                                   ParallelToolCalls parallelToolCalls,
                                   ConstraintSpec output, ReasoningFilter reasoning, boolean includeReasoning) {
        try {
            return service.beamGrammar(service.templateParsing(reasoning), chat, rendered, parallelToolCalls, output,
                    reasoning, includeReasoning);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The engine jobs of the request's candidates ({@code best_of}), each with its own seed offset and the
     *  request's reasoning filter. A text prompt with a message boundary submits as the media prompt of its
     *  tokens, which names the boundary the engine takes its checkpoint at. */
    List<GenerationBatch.Job> jobs() {
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        for (int i = 0; i < bestOf; i++) {
            GenerationRequest choice = OpenAiRequests.forChoice(base, i);
            jobs.add(media != null
                    ? GenerationBatch.Job.ofMedia(media, choice, rendered, constraint, reasoning)
                    : checkpointRow >= 0
                            ? GenerationBatch.Job.ofMedia(MediaPrompt.builder().addText(tokens)
                                            .checkpointRow(checkpointRow).build(), choice, rendered, constraint,
                                    reasoning)
                            : new GenerationBatch.Job(tokens, choice, rendered, constraint, reasoning));
        }
        return jobs;
    }

    /**
     * The job of a token prompt, the encoding of {@code rendered}, under {@code chat}'s rendering (the Anthropic
     * route): when the rendered prompt carries a message boundary ({@link #lastMessageStart}, the last message's
     * start), the job is the media prompt of its tokens that names the boundary, else the plain token prompt.
     */
    static GenerationBatch.Job textJob(EngineService service, ChatRequest chat, int[] tokens,
                                       GenerationRequest request, String rendered, ConstraintSpec constraint,
                                       ReasoningFilter reasoning) {
        int boundary = lastMessageStart(service, chat, rendered, tokens);
        return boundary >= 0
                ? GenerationBatch.Job.ofMedia(MediaPrompt.builder().addText(tokens).checkpointRow(boundary).build(),
                        request, rendered, constraint, reasoning)
                : new GenerationBatch.Job(tokens, request, rendered, constraint, reasoning);
    }

    /**
     * The response-body builder of the plan: its model, logprobs count, token rendering, and whether captures
     * hold the requested {@code logprob_token_ids}.
     */
    OpenAiBodies bodies(EngineService service) {
        return new OpenAiBodies(service, top, model, tokensAsIds, base.namesLogprobTokenIds());
    }

    /** A fresh output collector for one choice: the plan's tool routing and reasoning visibility. */
    ChatChoice newChoice() {
        return new ChatChoice(routing.newOutput(), includeReasoning);
    }

    /**
     * The text {@code echo} prepends to the answer (vLLM's chat {@code echo}): the last message's
     * content when its role is the one the generation continues — the assistant role under a
     * generation prompt, otherwise the last message's own role. Empty without such a match.
     */
    static String echoPrefix(ChatRequest chat) {
        List<Map<String, Object>> messages = chat.messages();
        if (messages.isEmpty()) return "";
        Map<String, Object> last = messages.get(messages.size() - 1);
        String role = chat.addGenerationPrompt() ? "assistant" : String.valueOf(last.get("role"));
        if (!role.equals(String.valueOf(last.get("role")))) return "";
        return textContent(last.get("content"));
    }

    /** A message's text: a string content, the {@code text} of its parts joined by newlines, or "" for null. */
    private static String textContent(Object content) {
        if (content instanceof String s) return s;
        if (content instanceof List<?> parts) {
            StringBuilder out = new StringBuilder();
            for (Object p : parts) {
                if (p instanceof Map<?, ?> part && part.get("text") instanceof String text) {
                    if (out.length() > 0) out.append('\n');
                    out.append(text);
                }
            }
            return out.toString();
        }
        return "";
    }

    /** The KV rows the prompt fills: its tokens and media rows on the media path, its tokens otherwise. */
    int promptRows() {
        return media != null ? media.nTokens() : tokens.length;
    }

    /**
     * The loaded image and audio parts of {@code chat}, in message and part order; empty when it has none.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for a part the server cannot take or load
     */
    static List<ChatMedia.Item> mediaItems(EngineService service, ChatRequest chat) {
        return mediaItems(service, chat, () -> null);
    }

    /**
     * {@link #mediaItems(EngineService, ChatRequest)} for a request naming adapters {@code mediaLoras}: a part's
     * uuid is looked up in the cache under the adapters' identity, so a repeated uuid is served by the entry the
     * same adapters encoded.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for a part the server cannot take or load
     */
    static List<ChatMedia.Item> mediaItems(EngineService service, ChatRequest chat,
                                           Supplier<LoraSlotKey> mediaLoras) {
        try {
            return service.media().items(chat, mediaLoras);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /**
     * The request's media-session adapter identity, resolved once per request on the primary replica's pump thread:
     * null when the request names no adapter, so its media encodes under the startup adapters' keys. The supplier
     * memoizes the resolution, so the request's media items resolve it at its first encode and its cache lookups
     * and prompt chunks all share it.
     */
    static Supplier<LoraSlotKey> mediaLoras(EngineService service, LoraSelection selection) {
        if (selection == null) {
            return () -> null;
        }
        return new Supplier<LoraSlotKey>() {
            private LoraSlotKey resolved;
            private boolean done;

            @Override
            public synchronized LoraSlotKey get() {
                if (!done) {
                    resolved = service.resolveMediaLoras(selection);
                    done = true;
                }
                return resolved;
            }
        };
    }

    /**
     * The {@link MediaPrompt} of a rendered prompt over media and embedding parts: the text segments the markers
     * split, each encoded separately, with each image's or audio's marker text and encoded rows, and each
     * embedding part's rows, where the parts sat (llama.cpp's {@code mtmd_tokenize}, vLLM's prompt embeds).
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for a malformed part, a marker count that does not match
     *         the parts, a row off the model's width, or media that do not decode
     */
    static MediaPrompt mediaPrompt(EngineService service, ChatRequest chat, String rendered,
                                   List<ChatMedia.Item> items) {
        return renderedMedia(service, chat, rendered, items).prompt();
    }

    /**
     * {@link #mediaPrompt(EngineService, ChatRequest, String, List)} for a request naming adapters
     * {@code mediaLoras}: each media part encodes in a LoRA session over the projector and its prompt chunks key
     * by the adapters' identity.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for a malformed part, a marker count that does not match
     *         the parts, a row off the model's width, or media that do not decode
     */
    static MediaPrompt mediaPrompt(EngineService service, ChatRequest chat, String rendered,
                                   List<ChatMedia.Item> items, Supplier<LoraSlotKey> mediaLoras) {
        return renderedMedia(service, chat, rendered, items, mediaLoras).prompt();
    }

    /**
     * {@link #mediaPrompt} with the placements of the items it spliced in, so a render endpoint can send the keys
     * and row spans of the media to a generate server.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for a malformed part, a marker count that does not match
     *         the parts, a row off the model's width, or media that do not decode
     */
    static ChatMedia.Rendered renderedMedia(EngineService service, ChatRequest chat, String rendered,
                                            List<ChatMedia.Item> items) {
        return renderedMedia(service, chat, rendered, items, () -> null);
    }

    /**
     * {@link #renderedMedia(EngineService, ChatRequest, String, List)} for a request naming adapters
     * {@code mediaLoras}: each media part encodes in a LoRA session over the projector and its prompt chunks and
     * placements key by the adapters' identity.
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for a malformed part, a marker count that does not match
     *         the parts, a row off the model's width, or media that do not decode
     */
    static ChatMedia.Rendered renderedMedia(EngineService service, ChatRequest chat, String rendered,
                                            List<ChatMedia.Item> items, Supplier<LoraSlotKey> mediaLoras) {
        List<float[][]> groups = rendered.contains(ChatMessages.PROMPT_EMBEDS_MARKER) ? embedGroups(chat)
                : List.of();
        int width = service.embeddingLength();
        for (int i = 0; i < groups.size(); i++) {
            for (int r = 0; r < groups.get(i).length; r++) {
                float[] row = groups.get(i)[r];
                if (row.length != width) {
                    throw ServerJson.badRequest("embedding part " + i + " row " + r + " width " + row.length
                            + " does not match the model's embedding width " + width);
                }
            }
        }
        try {
            return service.media().promptWithPlacements(rendered, items, groups, -1,
                    text -> service.encode(text, false, true), mediaLoras);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The rows of every {@code image_embeds} / {@code audio_embeds} part, in message and part order. */
    private static List<float[][]> embedGroups(ChatRequest chat) {
        List<float[][]> out = new ArrayList<>();
        for (int m = 0; m < chat.messages().size(); m++) {
            Object content = chat.messages().get(m).get(ChatMessages.CONTENT);
            if (!(content instanceof List<?> parts)) {
                continue;
            }
            for (int p = 0; p < parts.size(); p++) {
                if (!(parts.get(p) instanceof Map<?, ?> part)) {
                    continue; // forTemplate rejects non-object parts with its own error
                }
                Object type = part.get("type");
                if (!ChatMessages.PART_IMAGE_EMBEDS.equals(type) && !ChatMessages.PART_AUDIO_EMBEDS.equals(type)) {
                    continue;
                }
                if (part.get("uuid") != null) {
                    throw ServerJson.badRequest("message " + m + " content part " + p + " (" + type + "): 'uuid' is"
                            + " not supported on precomputed embeddings, which bypass the media cache; send the rows");
                }
                out.add(partRows(part.get(String.valueOf(type)), String.valueOf(type), m, p));
            }
        }
        if (out.isEmpty()) {
            throw ServerJson.badRequest("the rendered prompt carries " + ChatMessages.PROMPT_EMBEDS_MARKER
                    + " without image_embeds or audio_embeds content parts");
        }
        return out;
    }

    /** The rows of one part's payload: a flat row, or a list of rows; base64 strings and maps are refused. */
    private static float[][] partRows(Object payload, String type, int message, int part) {
        String at = "message " + message + " content part " + part + " (" + type + ")";
        if (payload instanceof List<?> flat && !flat.isEmpty() && flat.get(0) instanceof Number) {
            return new float[][] {OpenAiRequests.embedRow(flat, at, 0)};
        }
        if (payload instanceof List<?> rows && !rows.isEmpty() && rows.get(0) instanceof List<?>) {
            float[][] out = new float[rows.size()][];
            for (int r = 0; r < out.length; r++) {
                if (!(rows.get(r) instanceof List<?> row)) {
                    throw ServerJson.badRequest(at + " must hold arrays of numbers");
                }
                out[r] = OpenAiRequests.embedRow(row, at + " row " + r, 0);
            }
            return out;
        }
        throw ServerJson.badRequest(at + " must hold embedding rows: an array of numbers, or of arrays of"
                + " numbers; base64 tensors and maps are not supported in this build");
    }
}
